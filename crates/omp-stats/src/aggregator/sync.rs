//! TS `syncAllSessions` / `syncAllSessionsLocked` / `orderForIngest`.
//!
//! The calling thread owns every DB write and cursor; parsing (pure I/O + CPU)
//! fans out to scoped threads, one per changed transcript of a chunk of
//! `workers` files. Results are accepted in work-set order (the TS pool accepts
//! in completion order), which keeps fork ownership deterministic. Bounded
//! batches commit rows and cursors atomically; no SQLite transaction is held
//! across file I/O.

use std::fs::Metadata;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;

use super::lock::with_stats_sync_lock;
use crate::db::{
    apply_session_parse_results, complete_session_sync, get_file_offsets, init_db,
    mark_session_backfills_complete, prepare_session_sync, FileOffset, ParsedSession,
};
use crate::parser::{
    list_all_session_files, matches_session_file, parse_session_file, ParseSessionResult,
};

// Bound queued results by both file count and derived rows.
const SYNC_BATCH_FILES: usize = 512;
const SYNC_BATCH_ROWS: usize = 8192;
const SYNC_METADATA_FILES: usize = 128;
const SYNC_INLINE_READS: usize = 8;
/// Recency tiers for [`order_for_ingest`], newest first.
const INGEST_TIERS_MS: [f64; 2] = [
    24.0 * 60.0 * 60.0 * 1000.0,
    7.0 * 24.0 * 60.0 * 60.0 * 1000.0,
];

/// Progress event emitted after each session file is fully processed.
/// `current` = files completed (skipped + parsed), `total` = work-set size,
/// `processed` = running total of inserted rows (parsed files are reported only
/// after their batch commits), `changes` = running total of stored rows the
/// sync inserted, updated or deleted (0 while every transcript is unchanged).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SyncProgress {
    pub current: i64,
    pub total: i64,
    pub processed: i64,
    pub changes: i64,
    pub session_file: String,
}

/// TS `SyncOptions`.
#[derive(Default, Clone)]
pub struct SyncOptions<'a> {
    /// Called on the syncing thread after each file completes; keep it cheap.
    pub on_progress: Option<&'a dyn Fn(&SyncProgress)>,
    /// Parse thread count (default: host parallelism clamped to 2..=8; 1 on
    /// macOS like TS). `Some(1)` parses on the calling thread.
    pub workers: Option<usize>,
    /// Sync only these transcripts (e.g. files a watcher saw change) instead of
    /// listing every session. Missing files are skipped.
    pub files: Option<Vec<String>>,
}

/// TS `{ processed, files }`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SyncResult {
    /// Rows inserted.
    pub processed: i64,
    /// Transcripts whose rows were applied.
    pub files: i64,
}

/// Sync all session files to the database under the cross-process sync lock,
/// repeating while a pass asks for archive reconciliation.
///
/// # Errors
/// Lock, DB, or non-ENOENT transcript read failure (committed batches stay).
pub fn sync_all_sessions(opts: &SyncOptions<'_>) -> Result<SyncResult> {
    with_stats_sync_lock(&crate::paths::stats_db_path(), || {
        let mut total = SyncResult::default();
        loop {
            let (result, reconcile) = sync_all_sessions_locked(opts)?;
            total.processed += result.processed;
            total.files += result.files;
            if !reconcile {
                return Ok(total);
            }
        }
    })
}

fn default_worker_count() -> usize {
    // Bun can abort macOS processes re-entering the binary from workers; TS
    // keeps macOS serial and so do we.
    if cfg!(target_os = "macos") {
        return 1;
    }
    let hw = std::thread::available_parallelism().map_or(4, std::num::NonZero::get);
    // Parse is JSON-bound and SQLite writes serialize anyway.
    hw.clamp(2, 8)
}

fn now_ms() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64() * 1000.0)
}

/// Node `stat.mtimeMs`.
#[cfg(unix)]
fn mtime_ms(meta: &Metadata) -> f64 {
    use std::os::unix::fs::MetadataExt;
    (meta.mtime() as f64) * 1000.0 + (meta.mtime_nsec() as f64) / 1_000_000.0
}

/// Portable `mtime_ms` via `metadata.modified()` (same formula, same values).
#[cfg(not(unix))]
fn mtime_ms(meta: &Metadata) -> f64 {
    meta.modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map_or(0.0, |d| {
            (d.as_secs() as f64) * 1000.0 + f64::from(d.subsec_nanos()) / 1_000_000.0
        })
}

/// Node `stat.birthtimeMs || stat.ctimeMs`.
#[cfg(unix)]
fn born_ms(meta: &Metadata) -> f64 {
    use std::os::unix::fs::MetadataExt;
    let birth = metadata_birthtime_ms(meta);
    if birth != 0.0 {
        return birth;
    }
    (meta.ctime() as f64) * 1000.0 + (meta.ctime_nsec() as f64) / 1_000_000.0
}

/// Portable `born_ms`: `metadata.created()` only (no ctime on this platform).
#[cfg(not(unix))]
fn born_ms(meta: &Metadata) -> f64 {
    metadata_birthtime_ms(meta)
}

fn metadata_birthtime_ms(meta: &Metadata) -> f64 {
    meta.created()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map_or(0.0, |d| {
            (d.as_secs() as f64) * 1000.0 + f64::from(d.subsec_nanos()) / 1_000_000.0
        })
}

/// Order a full sync so recent activity lands first: transcripts modified in
/// the last day, then the last week, then the rest. Within a tier files go in
/// creation order, so a fork still follows its parent and the parent keeps
/// ownership of the entries the fork copied (first writer wins).
pub(crate) fn order_for_ingest(files: Vec<String>) -> Vec<String> {
    let now = now_ms();
    let mut ranked: Vec<(usize, f64, usize, String)> = files
        .into_iter()
        .enumerate()
        .map(|(index, file)| {
            let stat = std::fs::metadata(&file).ok();
            let age = stat.as_ref().map_or(f64::INFINITY, |m| now - mtime_ms(m));
            let tier = INGEST_TIERS_MS
                .iter()
                .position(|&limit| age <= limit)
                .unwrap_or(INGEST_TIERS_MS.len());
            let born = stat.as_ref().map_or(0.0, born_ms);
            (tier, born, index, file)
        })
        .collect();
    ranked.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)).then(a.2.cmp(&b.2)));
    ranked.into_iter().map(|entry| entry.3).collect()
}

/// A changed transcript to parse (TS `prepareFile` after the skip check).
struct Job {
    index: usize,
    from_offset: i64,
    stored: Option<FileOffset>,
}

/// One pass's accumulators (the TS closure state of `syncAllSessionsLocked`).
struct Pass<'o, 'a> {
    opts: &'o SyncOptions<'a>,
    total: i64,
    processed: i64,
    changes: i64,
    files_processed: i64,
    completed: i64,
    reconcile: bool,
    pending: Vec<ParsedSession>,
    pending_rows: usize,
}

impl Pass<'_, '_> {
    fn report(&mut self, session_file: &str) {
        self.completed += 1;
        if let Some(cb) = self.opts.on_progress {
            cb(&SyncProgress {
                current: self.completed,
                total: self.total,
                processed: self.processed,
                changes: self.changes,
                session_file: session_file.to_owned(),
            });
        }
    }

    fn flush(&mut self) -> Result<()> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let batch = std::mem::take(&mut self.pending);
        self.pending_rows = 0;
        let applied = apply_session_parse_results(&batch)?;
        self.processed += applied.processed;
        self.changes += applied.changes;
        self.files_processed += applied.files;
        self.reconcile |= applied.reconcile;
        // Report only durable progress: callbacks may interrupt the sync.
        for parsed in &batch {
            self.report(&parsed.session_file);
        }
        Ok(())
    }

    fn accept(&mut self, session_file: &str, parsed: Option<ParsedSession>) -> Result<()> {
        let Some(parsed) = parsed else {
            self.report(session_file);
            return Ok(());
        };
        let r = &parsed.result;
        self.pending_rows += r.stats.len()
            + r.user_stats.len()
            + r.user_links.len()
            + r.tool_calls.len()
            + r.tool_results.len();
        self.pending.push(parsed);
        if self.pending.len() >= SYNC_BATCH_FILES || self.pending_rows >= SYNC_BATCH_ROWS {
            self.flush()?;
        }
        Ok(())
    }

    fn finish(mut self) -> Result<(SyncResult, bool)> {
        self.flush()?;
        let result = SyncResult {
            processed: self.processed,
            files: self.files_processed,
        };
        // A targeted sync cannot settle reconciliation or backfills (they need
        // every transcript); leave their markers for the next full sync.
        if self.opts.files.is_some() {
            return Ok((result, false));
        }
        complete_session_sync(self.reconcile)?;
        mark_session_backfills_complete()?;
        Ok((result, self.reconcile))
    }
}

/// TS `prepareFile` skip check: unchanged size/mtime and the same file
/// identity mean the stored cursor is current. `last_modified` is Node's
/// fractional `mtimeMs`, compared exactly like TS (`===`).
#[allow(clippy::float_cmp)] // exact equality is the TS semantics
fn unchanged(stored: Option<&FileOffset>, file: &str, meta: &Metadata, replay: bool) -> bool {
    if replay {
        return false;
    }
    let Some(stored) = stored else { return false };
    let Some(state) = &stored.parser_state else {
        return false;
    };
    stored.last_modified == mtime_ms(meta)
        && i64::try_from(meta.len()).is_ok_and(|size| size == state.size)
        && matches_session_file(state, file, meta)
}

fn parse_job(file: &str, job: &Job, replay: bool) -> Result<ParsedSession> {
    let stored_state = job.stored.as_ref().and_then(|s| s.parser_state.as_ref());
    let unknown_identity = job.stored.is_some() && stored_state.is_none();
    let mut result: ParseSessionResult =
        parse_session_file(file, job.from_offset, stored_state, replay)?;
    if unknown_identity && result.parser_state.is_some() {
        result.reset = Some(true);
    }
    // During a full replay, a changed file can be a rewrite even if its old tail checkpoint matches.
    let rebuild = match stored_state {
        None => true,
        Some(old) => {
            replay
                && (result.parser_state.as_ref().map(|s| s.size) != Some(old.size)
                    || result.parser_state.as_ref().map(|s| s.mtime_ms) != Some(old.mtime_ms))
        }
    };
    Ok(ParsedSession {
        session_file: file.to_owned(),
        result,
        rebuild,
        replay,
    })
}

/// Parse `jobs` (≤ the pool size) concurrently; results in job order.
fn parse_chunk(
    files: &[String],
    jobs: &[Job],
    replay: bool,
    parallel: bool,
) -> Vec<Result<ParsedSession>> {
    if !parallel || jobs.len() <= 1 {
        return jobs
            .iter()
            .map(|job| parse_job(&files[job.index], job, replay))
            .collect();
    }
    std::thread::scope(|scope| {
        let handles: Vec<_> = jobs
            .iter()
            .map(|job| scope.spawn(move || parse_job(&files[job.index], job, replay)))
            .collect();
        handles
            .into_iter()
            .map(|h| {
                h.join()
                    .unwrap_or_else(|_| Err(anyhow::anyhow!("session parse thread panicked")))
            })
            .collect()
    })
}

fn sync_all_sessions_locked(opts: &SyncOptions<'_>) -> Result<(SyncResult, bool)> {
    init_db()?;
    let replay = prepare_session_sync()?;
    let files = match &opts.files {
        Some(files) => files.clone(),
        None => order_for_ingest(list_all_session_files()),
    };
    let mut pass = Pass {
        opts,
        total: i64::try_from(files.len()).unwrap_or(i64::MAX),
        processed: 0,
        changes: 0,
        files_processed: 0,
        completed: 0,
        reconcile: false,
        pending: Vec::new(),
        pending_rows: 0,
    };
    if files.is_empty() {
        return pass.finish();
    }

    let mut workers = opts.workers.unwrap_or_else(default_worker_count).max(1);
    // A few already-tracked transcripts (the watcher's usual work) need only
    // kilobyte tail reads: parsing them inline beats starting threads.
    if workers > 1 && opts.workers.is_none() && files.len() <= SYNC_INLINE_READS {
        let offsets = get_file_offsets(&files)?;
        if files
            .iter()
            .all(|f| offsets.get(f).is_some_and(|o| o.parser_state.is_some()))
        {
            workers = 1;
        }
    }
    let chunk = if workers == 1 {
        SYNC_INLINE_READS
    } else {
        workers
    };

    for meta_start in (0..files.len()).step_by(SYNC_METADATA_FILES) {
        let meta_end = (meta_start + SYNC_METADATA_FILES).min(files.len());
        let batch = &files[meta_start..meta_end];
        let mut offsets = get_file_offsets(batch)?;
        for chunk_start in (meta_start..meta_end).step_by(chunk) {
            let chunk_end = (chunk_start + chunk).min(meta_end);
            // `None` = skipped (missing or unchanged); `Some(i)` = jobs[i].
            let mut slots: Vec<Option<usize>> = Vec::with_capacity(chunk_end - chunk_start);
            let mut jobs = Vec::new();
            for (index, file) in files.iter().enumerate().take(chunk_end).skip(chunk_start) {
                let stored = offsets.remove(file);
                let Ok(meta) = std::fs::metadata(file) else {
                    slots.push(None);
                    continue;
                };
                if unchanged(stored.as_ref(), file, &meta, replay) {
                    slots.push(None);
                    continue;
                }
                let unknown_identity = stored.as_ref().is_some_and(|s| s.parser_state.is_none());
                let from_offset = if unknown_identity {
                    0
                } else {
                    stored.as_ref().map_or(0, |s| s.offset)
                };
                slots.push(Some(jobs.len()));
                jobs.push(Job {
                    index,
                    from_offset,
                    stored,
                });
            }
            let mut parsed = parse_chunk(&files, &jobs, replay, workers > 1).into_iter();
            // Preserve fork ownership order; the first failure aborts the pass
            // after the results before it are accepted (TS allSettled loop).
            for (offset, slot) in slots.into_iter().enumerate() {
                let file = &files[chunk_start + offset];
                match slot {
                    None => pass.accept(file, None)?,
                    Some(_) => {
                        let result = parsed
                            .next()
                            .ok_or_else(|| anyhow::anyhow!("missing parse result"))??;
                        pass.accept(file, Some(result))?;
                    }
                }
            }
        }
    }
    pass.finish()
}
