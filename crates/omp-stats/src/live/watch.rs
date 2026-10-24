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
use std::path::{Path, PathBuf};
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

fn snapshot(roots: &[PathBuf]) -> Snapshot {
    roots
        .iter()
        .flat_map(|root| {
            walkdir::WalkDir::new(root)
                .into_iter()
                .filter_map(Result::ok)
        })
        .filter(|entry| entry.file_type().is_file() && is_transcript(entry.path()))
        .filter_map(|entry| {
            let meta = entry.metadata().ok()?;
            Some((
                entry.path().to_string_lossy().into_owned(),
                (meta.len(), meta.modified().ok()),
            ))
        })
        .collect()
}

/// One recursive `notify` watcher over every session root that exists.
/// Roots that appear later are added by [`Watches::refresh`].
#[derive(Default)]
struct Watches {
    watcher: Option<RecommendedWatcher>,
    watched: Vec<PathBuf>,
}

impl Watches {
    /// Whether any root is being watched (replaces polling).
    fn active(&self) -> bool {
        !self.watched.is_empty()
    }

    /// Whether some existing-or-future root is not watched yet.
    fn incomplete(&self, roots: &[PathBuf]) -> bool {
        roots.iter().any(|root| !self.watched.contains(root))
    }

    fn reset(&mut self) {
        self.watcher = None;
        self.watched.clear();
    }

    /// Watch every existing, not yet watched root; returns whether one was added.
    fn refresh(&mut self, roots: &[PathBuf], tx: &mpsc::Sender<Msg>) -> bool {
        let mut added = false;
        for root in roots {
            if self.watched.contains(root) || !root.is_dir() {
                continue;
            }
            if self.watcher.is_none() {
                let events = tx.clone();
                match notify::recommended_watcher(move |event| {
                    let _ = events.send(Msg::Fs(event));
                }) {
                    Ok(watcher) => self.watcher = Some(watcher),
                    Err(error) => {
                        tracing::debug!(error = %error, "Stats session watcher unavailable");
                        return added;
                    }
                }
            }
            let Some(watcher) = self.watcher.as_mut() else {
                continue;
            };
            match watcher.watch(root, RecursiveMode::Recursive) {
                Ok(()) => {
                    self.watched.push(root.clone());
                    added = true;
                }
                // No recursive watch (inotify limits): this root is polled instead.
                Err(error) => {
                    tracing::debug!(root = %root.display(), error = %error, "Stats session watcher unavailable");
                }
            }
        }
        added
    }
}

/// Spawn the watcher for `epoch`; it exits on [`Msg::Stop`] or once `stop`
/// moves the epoch on.
pub(super) fn spawn(inner: Arc<Inner>, epoch: u64) {
    let roots = crate::paths::session_roots();
    let (tx, rx) = mpsc::channel();
    {
        let mut st = inner.lock();
        if st.epoch != epoch {
            return;
        }
        st.watch_tx = Some(tx.clone());
    }
    std::thread::spawn(move || run(&inner, epoch, &roots, &tx, &rx));
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

fn run(
    inner: &Arc<Inner>,
    epoch: u64,
    roots: &[PathBuf],
    tx: &mpsc::Sender<Msg>,
    rx: &mpsc::Receiver<Msg>,
) {
    let poll = Duration::from_millis(WATCH_POLL_MS);
    let full_every = Duration::from_millis(FULL_SYNC_INTERVAL_MS);
    let mut watches = Watches::default();
    watches.refresh(roots, tx);
    // Baseline for polling; only maintained while no watcher works.
    let mut known: Option<Snapshot> = (!watches.active()).then(|| snapshot(roots));
    let mut pending = Pending {
        files: Vec::new(),
        deadline: None,
    };
    let mut next_poll = Instant::now() + poll;
    let mut next_full = Instant::now() + full_every;
    loop {
        if inner.lock().epoch != epoch {
            return;
        }
        let mut wake_at = pending.deadline.map_or(next_full, |d| d.min(next_full));
        if watches.incomplete(roots) {
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
                watches.reset();
                known = Some(snapshot(roots));
                next_poll = Instant::now() + poll;
            }
            Err(RecvTimeoutError::Timeout) => {}
        }
        let now = Instant::now();
        if now >= next_poll && watches.incomplete(roots) {
            // Poll exactly the roots no watcher covers (a root whose watch
            // failed while another root is watched must not wait for the
            // 5-minute full sync); watched roots are never re-walked.
            let unwatched: Vec<PathBuf> = roots
                .iter()
                .filter(|r| !watches.watched.contains(*r))
                .cloned()
                .collect();
            {
                let current = snapshot(&unwatched);
                if let Some(known) = &known {
                    for (file, meta) in &current {
                        if known.get(file) != Some(meta) {
                            pending.add(file.clone(), now);
                        }
                    }
                }
                known = Some(current);
            }
            if watches.refresh(roots, tx) {
                // A root appeared: pick up what it already holds.
                inner.request_sync(None);
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
