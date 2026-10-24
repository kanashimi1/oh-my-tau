//! Shadow-mode turn judge: at turn end the runtime hands the turn to a background worker that
//! scores it with Laya and appends one JSON line to `~/.omp/agent/laya_judge.jsonl`.
//!
//! Default OFF (`OMP_LAYA_JUDGE=1` enables it). Never blocks or fails a turn: the queue is
//! bounded (`try_send`, overflow is dropped and counted), the model is loaded lazily once on
//! the worker thread, and a missing model logs a single warning.

use std::fs::OpenOptions;
use std::io::Write as _;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{sync_channel, SyncSender, TrySendError};
use std::sync::{Arc, OnceLock};
use std::thread::JoinHandle;
use std::time::Instant;

use serde_json::{json, Map, Value};

use crate::episode::parse_entries;
use crate::scorer::{ModelPaths, Scorer, Scores};
use crate::view::episode_view;

/// Queue capacity of the global worker.
pub const QUEUE_CAP: usize = 8;

/// Worker configuration.
#[derive(Debug, Clone)]
pub struct Config {
    pub out: PathBuf,
    pub paths: ModelPaths,
    pub queue_cap: usize,
}

impl Config {
    /// `None` unless `OMP_LAYA_JUDGE=1`. Other knobs: `OMP_LAYA_JUDGE_OUT`, `OMP_LAYA_CKPT`,
    /// `OMP_LAYA_ONNX`, `OMP_LAYA_THREADS`.
    #[must_use]
    pub fn from_env(get: &dyn Fn(&str) -> Option<String>) -> Option<Self> {
        if get("OMP_LAYA_JUDGE").as_deref() != Some("1") {
            return None;
        }
        let mut paths = ModelPaths::default();
        if let Some(c) = get("OMP_LAYA_CKPT") {
            paths.ckpt = c.into();
        }
        if let Some(o) = get("OMP_LAYA_ONNX") {
            paths.onnx = o.into();
        }
        if let Some(t) = get("OMP_LAYA_THREADS").and_then(|t| t.parse().ok()) {
            paths.threads = t;
        }
        let out = get("OMP_LAYA_JUDGE_OUT")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(get("HOME").unwrap_or_else(|| ".".into()))
                    .join(".omp/agent/laya_judge.jsonl")
            });
        Some(Self {
            out,
            paths,
            queue_cap: QUEUE_CAP,
        })
    }
}

/// A finished turn (one prompt run). `entries` are session-shaped
/// (`{"type":"message","message":{...}}`) so the same episode builder serves live and offline.
#[derive(Debug)]
pub struct Job {
    pub session_id: String,
    pub turn_index: u64,
    pub ts: String,
    pub entries: Vec<Value>,
    pub is_sub: bool,
}

/// Wrap raw message values (`role`, `content`, ...) as session `message` entries.
#[must_use]
pub fn message_entries(messages: impl IntoIterator<Item = Value>) -> Vec<Value> {
    messages
        .into_iter()
        .map(|m| json!({"type": "message", "message": m}))
        .collect()
}

#[derive(Default)]
struct Counters {
    enqueued: AtomicU64,
    dropped: AtomicU64,
    scored: AtomicU64,
    skipped: AtomicU64,
    failed: AtomicU64,
    warnings: AtomicU64,
    done: AtomicU64,
}

/// Counter snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stats {
    pub enqueued: u64,
    /// Jobs dropped because the queue was full.
    pub dropped: u64,
    /// Rows written.
    pub scored: u64,
    /// Jobs without a human request, or arriving after the model failed to load.
    pub skipped: u64,
    /// Inference/write failures.
    pub failed: u64,
    /// Warnings logged (model load failure: exactly once).
    pub warnings: u64,
    /// Jobs the worker has finished handling.
    pub done: u64,
}

/// Handle to the background worker.
pub struct ShadowJudge {
    tx: Option<SyncSender<Job>>,
    counters: Arc<Counters>,
    handle: Option<JoinHandle<()>>,
}

enum Model {
    Unloaded,
    Ready(Box<Scorer>),
    Broken,
}

impl ShadowJudge {
    /// Spawn the worker thread (the model is not loaded until the first job).
    #[must_use]
    pub fn spawn(cfg: Config) -> Self {
        let (tx, rx) = sync_channel::<Job>(cfg.queue_cap.max(1));
        let counters = Arc::new(Counters::default());
        let c = counters.clone();
        let handle = std::thread::Builder::new()
            .name("omp-laya-judge".into())
            .spawn(move || {
                let mut model = Model::Unloaded;
                while let Ok(job) = rx.recv() {
                    let r =
                        catch_unwind(AssertUnwindSafe(|| handle_job(&cfg, &c, &mut model, &job)));
                    if r.is_err() {
                        c.failed.fetch_add(1, Ordering::Relaxed);
                        tracing::warn!("laya shadow judge: worker panicked on a job; continuing");
                    }
                    c.done.fetch_add(1, Ordering::Relaxed);
                }
            })
            .ok();
        Self {
            tx: Some(tx),
            counters,
            handle,
        }
    }

    /// Enqueue a turn without ever blocking. Returns `false` when it was dropped (queue full
    /// or worker gone).
    pub fn submit(&self, job: Job) -> bool {
        let Some(tx) = &self.tx else { return false };
        match tx.try_send(job) {
            Ok(()) => {
                self.counters.enqueued.fetch_add(1, Ordering::Relaxed);
                true
            }
            Err(TrySendError::Full(_) | TrySendError::Disconnected(_)) => {
                self.counters.dropped.fetch_add(1, Ordering::Relaxed);
                false
            }
        }
    }

    #[must_use]
    pub fn stats(&self) -> Stats {
        let c = &self.counters;
        let g = |a: &AtomicU64| a.load(Ordering::Relaxed);
        Stats {
            enqueued: g(&c.enqueued),
            dropped: g(&c.dropped),
            scored: g(&c.scored),
            skipped: g(&c.skipped),
            failed: g(&c.failed),
            warnings: g(&c.warnings),
            done: g(&c.done),
        }
    }

    /// Close the queue and wait for the worker to drain it.
    pub fn shutdown(mut self) -> Stats {
        self.tx = None;
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
        self.stats()
    }
}

fn handle_job(cfg: &Config, c: &Counters, model: &mut Model, job: &Job) {
    let eps = parse_entries(&job.entries, job.is_sub);
    if eps.is_empty() {
        c.skipped.fetch_add(1, Ordering::Relaxed);
        return;
    }
    if matches!(model, Model::Unloaded) {
        *model = match Scorer::load(&cfg.paths) {
            Ok(s) => Model::Ready(Box::new(s)),
            Err(e) => {
                c.warnings.fetch_add(1, Ordering::Relaxed);
                tracing::warn!("laya shadow judge disabled for this process: {e:#}");
                Model::Broken
            }
        };
    }
    let Model::Ready(scorer) = model else {
        c.skipped.fetch_add(1, Ordering::Relaxed);
        return;
    };
    for ep in &eps {
        let t0 = Instant::now();
        let view = episode_view(ep, job.is_sub);
        match scorer.score(&view) {
            Ok(s) => {
                let row = row(
                    &job.session_id,
                    job.turn_index,
                    &job.ts,
                    &s,
                    &cfg.paths.ckpt.to_string_lossy(),
                    t0.elapsed().as_millis(),
                );
                match append_row(&cfg.out, &row) {
                    Ok(()) => c.scored.fetch_add(1, Ordering::Relaxed),
                    Err(e) => {
                        tracing::debug!("laya shadow judge: write failed: {e}");
                        c.failed.fetch_add(1, Ordering::Relaxed)
                    }
                };
            }
            Err(e) => {
                tracing::debug!("laya shadow judge: inference failed: {e:#}");
                c.failed.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

/// The output row: `{session_id, turn_index, ts, success_p, rework_p, failure_mode_probs,
/// model_ckpt, latency_ms}`.
#[must_use]
pub fn row(
    session_id: &str,
    turn_index: u64,
    ts: &str,
    s: &Scores,
    ckpt: &str,
    latency_ms: u128,
) -> Map<String, Value> {
    let mut m = Map::new();
    m.insert("session_id".into(), json!(session_id));
    m.insert("turn_index".into(), json!(turn_index));
    m.insert("ts".into(), json!(ts));
    m.insert("success_p".into(), json!(s.success_p));
    m.insert("rework_p".into(), json!(s.rework_p));
    m.insert("failure_mode_probs".into(), s.failure_mode_json());
    m.insert("model_ckpt".into(), json!(ckpt));
    m.insert(
        "latency_ms".into(),
        json!(u64::try_from(latency_ms).unwrap_or(u64::MAX)),
    );
    m
}

fn append_row(path: &std::path::Path, row: &Map<String, Value>) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut f = OpenOptions::new().create(true).append(true).open(path)?;
    let mut line = serde_json::to_string(row).map_err(std::io::Error::other)?;
    line.push('\n');
    f.write_all(line.as_bytes())
}

static GLOBAL: OnceLock<Option<ShadowJudge>> = OnceLock::new();

/// Process-wide worker: `None` (and no thread, no file) unless `OMP_LAYA_JUDGE=1`.
pub fn global() -> Option<&'static ShadowJudge> {
    GLOBAL
        .get_or_init(|| Config::from_env(&|k| std::env::var(k).ok()).map(ShadowJudge::spawn))
        .as_ref()
}

/// Cheap check for the hook site, so disabled runs build nothing.
pub fn enabled() -> bool {
    global().is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turn() -> Vec<Value> {
        message_entries([
            json!({"role":"user","attribution":"user","content":"do the thing"}),
            json!({"role":"assistant","model":"m","stopReason":"stop","content":[{"type":"text","text":"done"}]}),
        ])
    }

    #[test]
    fn off_by_default() {
        assert!(Config::from_env(&|_| None).is_none());
        assert!(Config::from_env(&|k| (k == "OMP_LAYA_JUDGE").then(|| "0".to_owned())).is_none());
        assert!(Config::from_env(&|k| (k == "OMP_LAYA_JUDGE").then(|| "1".to_owned())).is_some());
    }

    #[test]
    fn missing_model_warns_once_and_worker_survives() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("laya_judge.jsonl");
        let cfg = Config {
            out: out.clone(),
            paths: ModelPaths {
                ckpt: dir.path().join("nope"),
                onnx: dir.path().join("nope.onnx"),
                threads: 1,
            },
            queue_cap: 8,
        };
        let w = ShadowJudge::spawn(cfg);
        for i in 0..3 {
            assert!(w.submit(Job {
                session_id: "s".into(),
                turn_index: i,
                ts: "t".into(),
                entries: turn(),
                is_sub: false
            }));
        }
        // a turn without a human request is skipped before the model is touched
        assert!(w.submit(Job {
            session_id: "s".into(),
            turn_index: 9,
            ts: "t".into(),
            entries: Vec::new(),
            is_sub: false
        }));
        let st = w.shutdown();
        assert_eq!(st.warnings, 1, "{st:?}");
        assert_eq!(st.done, 4);
        assert_eq!(st.scored, 0);
        assert_eq!(st.skipped, 4);
        assert!(!out.exists());
    }

    #[test]
    fn overflow_is_dropped_and_counted() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = Config {
            out: dir.path().join("o.jsonl"),
            paths: ModelPaths {
                ckpt: dir.path().join("nope"),
                onnx: dir.path().join("nope.onnx"),
                threads: 1,
            },
            queue_cap: 1,
        };
        let w = ShadowJudge::spawn(cfg);
        let mut accepted = 0;
        for i in 0..200 {
            if w.submit(Job {
                session_id: "s".into(),
                turn_index: i,
                ts: "t".into(),
                entries: turn(),
                is_sub: false,
            }) {
                accepted += 1;
            }
        }
        let st = w.shutdown();
        assert_eq!(st.enqueued, accepted);
        assert_eq!(st.enqueued + st.dropped, 200);
        assert_eq!(st.done, st.enqueued);
    }
}
