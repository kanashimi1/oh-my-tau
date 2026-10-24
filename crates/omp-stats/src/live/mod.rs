//! Rust port of `live.ts`: live ingest for a running dashboard — background
//! session sync, rollup refresh, and the status stream behind `/api/events`.
//!
//! The dashboard never waits for ingest. [`StatsLive::start`] kicks a full sync
//! in the background and watches the sessions directory; changed transcripts
//! are re-synced (only those files) shortly after they are written. Every batch
//! that changes stored rows bumps [`LiveStatus::version`] (throttled), so open
//! pages refetch while parsing is still running; syncs that change nothing
//! leave the version alone.
//!
//! Differences from TS (runtime, not behavior): `fs.watch` becomes a `notify`
//! watcher thread with a polling fallback ([`watch`]); timers are short-lived threads cancelled through an
//! epoch counter; listeners are a `tokio::sync::broadcast` channel of
//! [`LiveFrame`]s (sending never blocks and needs no runtime).

#[cfg(test)]
mod tests;
mod watch;

use std::sync::{mpsc, Arc, LazyLock, Mutex, MutexGuard};
use std::time::Duration;

use tokio::sync::broadcast;

use crate::aggregator::{sync_all_sessions, SyncOptions, SyncProgress};
use crate::db::{get_data_version, init_db};
use crate::rollup::{get_rollup_status, refresh_rollups_with, RefreshOptions};
use crate::types::{LiveStatus, LiveSyncPhase, LiveSyncStatus};

/// Wait this long after the last transcript write before syncing it.
const WATCH_DEBOUNCE_MS: u64 = 800;
/// Full resync cadence, catching anything the watcher missed.
const FULL_SYNC_INTERVAL_MS: u64 = 5 * 60 * 1000;
/// Minimum spacing of `version` bumps; each bump makes open pages refetch.
const VERSION_THROTTLE_MS: i64 = 1000;
/// Delay before retrying a failed sync (typically a lock held by another omp process).
const SYNC_RETRY_MS: u64 = 10_000;
/// Minimum spacing of progress-only status events.
const PROGRESS_THROTTLE_MS: u64 = 150;
/// Buffered frames per subscriber before it lags (a lagging SSE client just
/// skips to newer frames; every frame is a full status).
const CHANNEL_CAPACITY: usize = 64;

/// One distinct status and its JSON serialization (shared by every subscriber;
/// SSE writes `data: {frame}\n\n`).
#[derive(Debug, Clone)]
pub struct LiveFrame {
    pub status: LiveStatus,
    pub frame: Arc<str>,
}

/// Queued work: specific files (insertion order, deduplicated), or a full sync.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Queued {
    All,
    Files(Vec<String>),
}

#[derive(Default)]
struct State {
    version: i64,
    sync: LiveSyncStatus,
    indexing_hours: i64,
    /// Last frame sent; identical statuses are not re-sent.
    last_frame: Option<Arc<str>>,
    /// `PRAGMA data_version` at the last sync; `None` before the first.
    data_version: Option<i64>,
    started: bool,
    /// Bumped by `stop`: pending timers and the watcher of an older epoch do nothing.
    epoch: u64,
    queued: Option<Queued>,
    syncing: bool,
    refreshing: bool,
    refresh_again: bool,
    version_timer: bool,
    last_version_at: i64,
    progress_timer: bool,
    /// Only the newest scheduled retry runs (TS `clearTimeout(retryTimer)`).
    retry_seq: u64,
    /// Stops the running watcher thread.
    watch_tx: Option<mpsc::Sender<watch::Msg>>,
}

impl State {
    fn status(&self) -> LiveStatus {
        LiveStatus {
            version: self.version,
            sync: self.sync.clone(),
            indexing_hours: self.indexing_hours,
        }
    }
}

struct Inner {
    state: Mutex<State>,
    tx: broadcast::Sender<LiveFrame>,
}

/// One per process ([`stats_live`]); owned by the dashboard server.
pub struct StatsLive {
    inner: Arc<Inner>,
}

/// The process-wide live hub.
pub fn stats_live() -> &'static StatsLive {
    static INSTANCE: LazyLock<StatsLive> = LazyLock::new(StatsLive::new);
    &INSTANCE
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX))
}

impl Default for StatsLive {
    fn default() -> Self {
        Self::new()
    }
}

impl StatsLive {
    /// A fresh hub (version 1, idle). Use [`stats_live`] outside tests.
    #[must_use]
    pub fn new() -> Self {
        let (tx, _) = broadcast::channel(CHANNEL_CAPACITY);
        let state = State {
            version: 1,
            ..State::default()
        };
        Self {
            inner: Arc::new(Inner {
                state: Mutex::new(state),
                tx,
            }),
        }
    }

    /// Current status (also the first SSE frame of a new subscriber).
    #[must_use]
    pub fn status(&self) -> LiveStatus {
        self.inner.lock().status()
    }

    /// Receive every distinct status change from now on; drop to unsubscribe.
    #[must_use]
    pub fn subscribe(&self) -> broadcast::Receiver<LiveFrame> {
        self.inner.tx.subscribe()
    }

    /// Begin background ingest: an immediate full sync, the transcript watcher,
    /// and periodic resyncs. Idempotent.
    pub fn start(&self) {
        let epoch = {
            let mut st = self.inner.lock();
            if st.started {
                return;
            }
            st.started = true;
            st.epoch
        };
        watch::spawn(Arc::clone(&self.inner), epoch);
        self.inner.request_sync(None);
    }

    /// Stop watching and cancel pending timers; a running sync finishes.
    pub fn stop(&self) {
        let mut st = self.inner.lock();
        st.started = false;
        st.epoch += 1;
        st.version_timer = false;
        st.progress_timer = false;
        st.retry_seq += 1;
        st.queued = None;
        if let Some(tx) = st.watch_tx.take() {
            let _ = tx.send(watch::Msg::Stop);
        }
    }

    /// Queue a sync of `files`, or of every session when `None`. Requests made
    /// while a sync runs coalesce into one follow-up run.
    pub fn request_sync(&self, files: Option<Vec<String>>) {
        self.inner.request_sync(files);
    }
}

impl Inner {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn emit(&self, st: &mut State) {
        let status = st.status();
        let Ok(frame) = serde_json::to_string(&status) else {
            return;
        };
        if st.last_frame.as_deref() == Some(frame.as_str()) {
            return;
        }
        let frame: Arc<str> = frame.into();
        st.last_frame = Some(Arc::clone(&frame));
        // No subscribers is fine.
        let _ = self.tx.send(LiveFrame { status, frame });
    }

    fn request_sync(self: &Arc<Self>, files: Option<Vec<String>>) {
        let mut st = self.lock();
        match files {
            None => st.queued = Some(Queued::All),
            Some(files) => match &mut st.queued {
                Some(Queued::All) => {}
                Some(Queued::Files(queued)) => {
                    for file in files {
                        if !queued.contains(&file) {
                            queued.push(file);
                        }
                    }
                }
                None => {
                    let mut unique = Vec::with_capacity(files.len());
                    for file in files {
                        if !unique.contains(&file) {
                            unique.push(file);
                        }
                    }
                    st.queued = Some(Queued::Files(unique));
                }
            },
        }
        if st.syncing {
            return;
        }
        st.syncing = true;
        drop(st);
        let inner = Arc::clone(self);
        std::thread::spawn(move || inner.drain());
    }

    fn drain(self: &Arc<Self>) {
        loop {
            let work = {
                let mut st = self.lock();
                match st.queued.take() {
                    Some(work) => work,
                    None => {
                        st.syncing = false;
                        return;
                    }
                }
            };
            self.run_sync(match work {
                Queued::All => None,
                Queued::Files(files) => Some(files),
            });
        }
    }

    fn run_sync(self: &Arc<Self>, files: Option<Vec<String>>) {
        // Targeted syncs of a file or two stay quiet; only full syncs show progress.
        let visible = files.is_none();
        if visible {
            let mut st = self.lock();
            st.sync = LiveSyncStatus {
                phase: LiveSyncPhase::Syncing,
                current: 0,
                total: 0,
                processed: 0,
                error: None,
                ..st.sync.clone()
            };
            self.emit(&mut st);
        }
        let committed = std::cell::Cell::new(0_i64);
        let on_progress = |event: &SyncProgress| {
            let mut st = self.lock();
            if visible {
                st.sync.current = event.current;
                st.sync.total = event.total;
                st.sync.processed = event.processed;
                self.emit_progress(&mut st);
            }
            // Only committed row changes make pages refetch.
            if event.changes > committed.get() {
                committed.set(event.changes);
                self.changed(&mut st);
            }
        };
        let opts = SyncOptions {
            on_progress: Some(&on_progress),
            files: files.clone(),
            workers: None,
        };
        let result = init_db().and_then(|()| sync_all_sessions(&opts));
        let mut st = self.lock();
        match result {
            Ok(result) => {
                st.sync = LiveSyncStatus {
                    phase: LiveSyncPhase::Idle,
                    processed: if visible {
                        result.processed
                    } else {
                        st.sync.processed
                    },
                    last_synced_at: Some(now_ms()),
                    error: None,
                    ..st.sync.clone()
                };
            }
            Err(error) => {
                tracing::warn!(error = %format!("{error:#}"), "Stats live sync failed");
                st.sync.phase = LiveSyncPhase::Error;
                st.sync.error = Some(error.to_string());
                // Usually lock contention with another omp process; committed
                // batches are kept, so a retry resumes where it stopped.
                if st.started {
                    st.retry_seq += 1;
                    let (seq, epoch) = (st.retry_seq, st.epoch);
                    self.after(
                        Duration::from_millis(SYNC_RETRY_MS),
                        move |inner| {
                            let current = inner.lock().retry_seq == seq;
                            if current {
                                inner.request_sync(files);
                            }
                        },
                        epoch,
                    );
                }
            }
        }
        drop(st);
        // Another process may have ingested too; its commits move SQLite's data_version.
        let data_version = get_data_version().ok().flatten();
        let mut st = self.lock();
        if let Some(current) = data_version {
            let external = st.data_version != Some(current);
            st.data_version = Some(current);
            if external {
                self.changed(&mut st);
            }
        }
        self.emit(&mut st);
    }

    /// Run `f` after `delay` unless `stop` moved past `epoch` meanwhile.
    fn after(
        self: &Arc<Self>,
        delay: Duration,
        f: impl FnOnce(&Arc<Self>) + Send + 'static,
        epoch: u64,
    ) {
        let inner = Arc::clone(self);
        std::thread::spawn(move || {
            std::thread::sleep(delay);
            if inner.lock().epoch == epoch {
                f(&inner);
            }
        });
    }

    /// Stored data changed: refresh rollups and (throttled) bump the version.
    fn changed(self: &Arc<Self>, st: &mut State) {
        self.refresh(st);
        self.bump_version(st);
    }

    /// Bump the data version (throttled), making open pages refetch.
    fn bump_version(self: &Arc<Self>, st: &mut State) {
        if st.version_timer {
            return;
        }
        st.version_timer = true;
        let wait = (st.last_version_at + VERSION_THROTTLE_MS - now_ms()).max(0);
        let epoch = st.epoch;
        self.after(
            Duration::from_millis(wait.unsigned_abs()),
            move |inner| {
                let dirty = get_rollup_status().map(|s| s.dirty_hours);
                let mut st = inner.lock();
                if st.epoch != epoch {
                    return;
                }
                st.version_timer = false;
                st.last_version_at = now_ms();
                st.version += 1;
                if let Ok(dirty) = dirty {
                    st.indexing_hours = dirty;
                }
                inner.emit(&mut st);
            },
            epoch,
        );
    }

    fn refresh(self: &Arc<Self>, st: &mut State) {
        st.refresh_again = true;
        if st.refreshing {
            return;
        }
        st.refreshing = true;
        let inner = Arc::clone(self);
        std::thread::spawn(move || inner.refresh_loop());
    }

    fn refresh_loop(self: &Arc<Self>) {
        loop {
            {
                let mut st = self.lock();
                if !st.refresh_again {
                    st.refreshing = false;
                    break;
                }
                st.refresh_again = false;
            }
            let on_progress = |remaining: i64| {
                let mut st = self.lock();
                st.indexing_hours = remaining;
                // Each rolled batch fills in more history for open pages.
                if remaining > 0 {
                    self.bump_version(&mut st);
                }
            };
            let opts = RefreshOptions {
                on_progress: Some(&on_progress),
                abort: None,
            };
            if let Err(error) = refresh_rollups_with(&opts) {
                tracing::warn!(error = %format!("{error:#}"), "Stats rollup refresh failed");
                let mut st = self.lock();
                st.refresh_again = false;
                st.refreshing = false;
                break;
            }
        }
        let dirty = get_rollup_status().map(|s| s.dirty_hours);
        let mut st = self.lock();
        if let Ok(dirty) = dirty {
            st.indexing_hours = dirty;
        }
        self.emit(&mut st);
    }

    fn emit_progress(self: &Arc<Self>, st: &mut State) {
        if st.progress_timer {
            return;
        }
        st.progress_timer = true;
        let epoch = st.epoch;
        self.after(
            Duration::from_millis(PROGRESS_THROTTLE_MS),
            |inner| {
                let mut st = inner.lock();
                st.progress_timer = false;
                inner.emit(&mut st);
            },
            epoch,
        );
    }

    /// Test hook: no sync, refresh or timer still running.
    #[cfg(test)]
    fn quiescent(&self) -> bool {
        let st = self.lock();
        !st.syncing && !st.refreshing && !st.version_timer && !st.progress_timer
    }
}
