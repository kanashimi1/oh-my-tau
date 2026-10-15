//! TS `#watch` (recursive `fs.watch` on the sessions dir + debounce) and the
//! `FULL_SYNC_INTERVAL_MS` timer, on one thread.
//!
//! Normally a recursive `notify` watcher (inotify / FSEvents / ReadDirectoryChangesW,
//! like `fs.watch`) feeds `*.jsonl` paths into a debounce: [`WATCH_DEBOUNCE_MS`]
//! after the last write the batch is synced (only those files). When the watch
//! cannot be set up or fails (no sessions dir yet, inotify limits), the thread
//! falls back to stat-polling the tree every [`WATCH_POLL_MS`] and retries the
//! watch on each tick, so the costly poll only runs while no watcher works.

use std::collections::HashMap;
use std::path::Path;
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use notify::{EventKind, RecommendedWatcher, RecursiveMode, Watcher as _};

use super::{Inner, FULL_SYNC_INTERVAL_MS, WATCH_DEBOUNCE_MS};

/// Fallback polling cadence (also the watch-retry cadence) while no watcher works.
pub(super) const WATCH_POLL_MS: u64 = 1000;

/// Messages to the watcher thread.
pub(super) enum Msg {
    Fs(notify::Result<notify::Event>),
    Stop,
}

type Snapshot = HashMap<String, (u64, Option<SystemTime>)>;

fn is_transcript(path: &Path) -> bool {
    path.to_string_lossy().ends_with(".jsonl")
}

fn snapshot(root: &Path) -> Snapshot {
    walkdir::WalkDir::new(root)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().is_file() && is_transcript(entry.path()))
        .filter_map(|entry| {
            let meta = entry.metadata().ok()?;
            Some((entry.path().to_string_lossy().into_owned(), (meta.len(), meta.modified().ok())))
        })
        .collect()
}

fn try_watch(root: &Path, tx: &mpsc::Sender<Msg>) -> Option<RecommendedWatcher> {
    let events = tx.clone();
    let result = notify::recommended_watcher(move |event| {
        let _ = events.send(Msg::Fs(event));
    })
    .and_then(|mut watcher| watcher.watch(root, RecursiveMode::Recursive).map(|()| watcher));
    match result {
        Ok(watcher) => Some(watcher),
        Err(error) => {
            // No sessions dir yet, or recursive watch unsupported: poll instead.
            tracing::debug!(root = %root.display(), error = %error, "Stats session watcher unavailable");
            None
        }
    }
}

/// Spawn the watcher for `epoch`; it exits on [`Msg::Stop`] or once `stop`
/// moves the epoch on.
pub(super) fn spawn(inner: Arc<Inner>, epoch: u64) {
    let root = crate::paths::sessions_dir();
    let (tx, rx) = mpsc::channel();
    {
        let mut st = inner.lock();
        if st.epoch != epoch {
            return;
        }
        st.watch_tx = Some(tx.clone());
    }
    std::thread::spawn(move || run(&inner, epoch, &root, &tx, &rx));
}

/// Debounced work of the watcher loop.
struct Pending {
    files: Vec<String>,
    deadline: Option<Instant>,
}

impl Pending {
    fn add(&mut self, file: String, now: Instant) {
        if !self.files.contains(&file) {
            self.files.push(file);
        }
        // Each write restarts the debounce, like clearTimeout + setTimeout.
        self.deadline = Some(now + Duration::from_millis(WATCH_DEBOUNCE_MS));
    }
}

fn run(inner: &Arc<Inner>, epoch: u64, root: &Path, tx: &mpsc::Sender<Msg>, rx: &mpsc::Receiver<Msg>) {
    let poll = Duration::from_millis(WATCH_POLL_MS);
    let full_every = Duration::from_millis(FULL_SYNC_INTERVAL_MS);
    let mut watcher = try_watch(root, tx);
    // Baseline for polling; only maintained while no watcher works.
    let mut known: Option<Snapshot> = watcher.is_none().then(|| snapshot(root));
    let mut pending = Pending { files: Vec::new(), deadline: None };
    let mut next_poll = Instant::now() + poll;
    let mut next_full = Instant::now() + full_every;
    loop {
        if inner.lock().epoch != epoch {
            return;
        }
        let mut wake_at = pending.deadline.map_or(next_full, |d| d.min(next_full));
        if watcher.is_none() {
            wake_at = wake_at.min(next_poll);
        }
        let timeout = wake_at.saturating_duration_since(Instant::now());
        match rx.recv_timeout(timeout) {
            Ok(Msg::Stop) | Err(RecvTimeoutError::Disconnected) => return,
            Ok(Msg::Fs(Ok(event))) => {
                if !matches!(event.kind, EventKind::Access(_)) {
                    let now = Instant::now();
                    for path in event.paths.iter().filter(|p| is_transcript(p)) {
                        pending.add(path.to_string_lossy().into_owned(), now);
                    }
                }
            }
            Ok(Msg::Fs(Err(error))) => {
                // TS closes a failed watcher; we fall back to polling and retry it.
                tracing::warn!(error = %error, "Stats session watcher failed; polling until it recovers");
                watcher = None;
                known = Some(snapshot(root));
                next_poll = Instant::now() + poll;
            }
            Err(RecvTimeoutError::Timeout) => {}
        }
        let now = Instant::now();
        if watcher.is_none() && now >= next_poll {
            let current = snapshot(root);
            if let Some(known) = &known {
                for (file, meta) in &current {
                    if known.get(file) != Some(meta) {
                        pending.add(file.clone(), now);
                    }
                }
            }
            known = Some(current);
            watcher = try_watch(root, tx);
            if watcher.is_some() {
                known = None;
            }
            next_poll = now + poll;
        }
        if pending.deadline.is_some_and(|d| now >= d) {
            pending.deadline = None;
            inner.request_sync(Some(std::mem::take(&mut pending.files)));
        }
        if now >= next_full {
            next_full = now + full_every;
            inner.request_sync(None);
        }
    }
}
