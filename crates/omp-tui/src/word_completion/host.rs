//! Engine host for ghost-text word completion: loads the n-gram engine off the
//! UI thread, keeps it current with `history.db`, learns submitted prompts and
//! persists the state.
//!
//! The TS client talks to a machine-global daemon (`predict/daemon.ts`) that
//! owns `~/.omp/agent/predict/ngram/`. omp-rs runs the same engine in-process,
//! so it keeps its **own** state dir (`predict/ngram-rs/`) in the identical
//! snapshot format instead of writing the daemon's: two writers racing on one
//! `ngram.snapshot` / `cursor.json` would lose each other's learning. A fresh
//! omp-rs state is seeded by a read-only copy of the TS snapshot + cursor, so
//! it starts as warm as the TS engine, then ingests `history.db` past that
//! cursor. The TS state is never written.

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;
use pi_predict::{Config, Method, Predictor, Query};

const SNAPSHOT_FILE: &str = "ngram.snapshot";
const CURSOR_FILE: &str = "cursor.json";
/// History rows per ingestion page (TS `INGEST_BATCH`).
const INGEST_BATCH: i64 = 1_000;

/// The `spelling.autocomplete` setting values (`off` disables ghost text;
/// `smollm`/`apple` need weights / AppKit that omp-rs does not ship and fall
/// back to the n-gram engine).
#[must_use]
pub fn method_for_setting(value: &str) -> Option<Method> {
    match value {
        "off" => None,
        _ => Some(Method::Ngram),
    }
}

enum Slot {
    /// Loading or ingesting on the background thread; prompts submitted
    /// meanwhile queue here and are observed once the engine is ready.
    Loading {
        pending: Vec<String>,
    },
    Ready(Box<dyn Predictor>),
    /// Unloadable (state dir unwritable, ...): ghost text stays off.
    Failed,
}

/// Answer to one completion request.
#[derive(Debug, PartialEq, Eq)]
pub enum Answer {
    /// The engine is not ready (or busy); ask again later, do not cache.
    Pending,
    /// The engine's suffix, or `None` to show nothing.
    Ready(Option<String>),
}

/// One in-process engine plus its persistence bookkeeping.
pub struct WordHost {
    slot: Mutex<Slot>,
    dirty: AtomicBool,
    cursor: AtomicI64,
    state_dir: Option<PathBuf>,
}

impl std::fmt::Debug for WordHost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WordHost")
            .field("state_dir", &self.state_dir)
            .finish_non_exhaustive()
    }
}

/// The agent directory (`PI_CODING_AGENT_DIR` / `OMP_AGENT_DIR` /
/// `~/.omp/agent`, TS `getAgentDir`).
#[must_use]
pub fn agent_dir() -> Option<PathBuf> {
    for var in ["PI_CODING_AGENT_DIR", "OMP_AGENT_DIR"] {
        if let Some(dir) = std::env::var_os(var).filter(|v| !v.is_empty()) {
            return Some(PathBuf::from(dir));
        }
    }
    std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".omp").join("agent"))
}

impl WordHost {
    /// Start loading the engine on a background thread; returns immediately.
    #[must_use]
    pub fn spawn(agent_dir: &Path, method: Method) -> Arc<Self> {
        Self::spawn_with(agent_dir, method, foreign_history_files())
    }

    /// [`Self::spawn`] with an explicit list of foreign prompt histories a
    /// fresh engine learns (tests pass none to stay hermetic).
    #[must_use]
    pub fn spawn_with(agent_dir: &Path, method: Method, foreign: Vec<PathBuf>) -> Arc<Self> {
        let state_dir = agent_dir.join("predict").join("ngram-rs");
        let host = Arc::new(Self {
            slot: Mutex::new(Slot::Loading {
                pending: Vec::new(),
            }),
            dirty: AtomicBool::new(false),
            cursor: AtomicI64::new(0),
            state_dir: Some(state_dir),
        });
        let loader = Arc::clone(&host);
        let agent_dir = agent_dir.to_path_buf();
        let spawned = std::thread::Builder::new()
            .name("word-completion-load".into())
            .spawn(move || {
                let loaded = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    loader.load(&agent_dir, method, &foreign)
                }));
                if loaded.is_err() || loaded.is_ok_and(|ok| !ok) {
                    *loader.slot.lock() = Slot::Failed;
                }
            });
        if spawned.is_err() {
            *host.slot.lock() = Slot::Failed;
        }
        host
    }

    /// A host over an already-built engine with no persistence (tests, embedders).
    #[must_use]
    pub fn with_engine(engine: Box<dyn Predictor>) -> Arc<Self> {
        Arc::new(Self {
            slot: Mutex::new(Slot::Ready(engine)),
            dirty: AtomicBool::new(false),
            cursor: AtomicI64::new(0),
            state_dir: None,
        })
    }

    /// Whether the engine has finished loading (successfully).
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.slot
            .try_lock()
            .is_some_and(|slot| matches!(*slot, Slot::Ready(_)))
    }

    /// Ghost suffix for `prefix` typed after `before`. Never blocks: a busy or
    /// still-loading engine answers [`Answer::Pending`].
    pub fn complete(&self, before: &str, prefix: &str) -> Answer {
        let Some(mut slot) = self.slot.try_lock() else {
            return Answer::Pending;
        };
        match &mut *slot {
            Slot::Loading { .. } => Answer::Pending,
            Slot::Failed => Answer::Ready(None),
            Slot::Ready(engine) => {
                let query = Query { before, prefix };
                Answer::Ready(
                    engine
                        .complete(&query)
                        .map(|s| s.suffix)
                        .filter(|s| !s.is_empty()),
                )
            }
        }
    }

    /// Report a shown suggestion the user accepted (Tab/→) or typed past.
    pub fn feedback(&self, before: &str, prefix: &str, suggestion: &str, accepted: bool) {
        if let Slot::Ready(engine) = &mut *self.slot.lock() {
            engine.feedback(&Query { before, prefix }, suggestion, accepted);
            self.dirty.store(true, Ordering::Relaxed);
        }
    }

    /// Learn one submitted prompt (the engine filters code, pastes and typos).
    pub fn observe(&self, prompt: &str) {
        match &mut *self.slot.lock() {
            Slot::Loading { pending } => pending.push(prompt.to_owned()),
            Slot::Ready(engine) => {
                engine.observe(prompt);
                self.dirty.store(true, Ordering::Relaxed);
            }
            Slot::Failed => {}
        }
    }

    /// Flush learned state when something changed since the last persist.
    pub fn persist(&self) {
        if !self.dirty.swap(false, Ordering::Relaxed) {
            return;
        }
        let Some(dir) = &self.state_dir else { return };
        let mut slot = self.slot.lock();
        let Slot::Ready(engine) = &mut *slot else {
            self.dirty.store(true, Ordering::Relaxed);
            return;
        };
        let cursor = self.cursor.load(Ordering::Relaxed);
        if engine.persist().is_err() || write_cursor(dir, cursor).is_err() {
            self.dirty.store(true, Ordering::Relaxed);
        }
    }

    /// Background load: restore, ingest history, persist, then go live.
    /// Returns `false` when the engine could not be opened.
    fn load(&self, agent_dir: &Path, method: Method, foreign: &[PathBuf]) -> bool {
        let Some(dir) = self.state_dir.clone() else {
            return false;
        };
        if std::fs::create_dir_all(&dir).is_err() {
            return false;
        }
        if !dir.join(SNAPSHOT_FILE).exists() {
            seed_from_daemon_state(&agent_dir.join("predict").join("ngram"), &dir);
        }
        let config = Config {
            state_dir: dir.clone(),
            model_dir: None,
            show_threshold: None,
        };
        let mut cursor = if dir.join(SNAPSHOT_FILE).exists() {
            read_cursor(&dir)
        } else {
            None
        };
        let mut engine = match pi_predict::open(method, &config) {
            Ok(engine) => engine,
            Err(_) => {
                // Unloadable state (corrupt, or an incompatible engine
                // version): rebuild it like a fresh install (TS `#open`).
                let _ = std::fs::remove_file(dir.join(SNAPSHOT_FILE));
                let _ = std::fs::remove_file(dir.join(CURSOR_FILE));
                cursor = None;
                match pi_predict::open(method, &config) {
                    Ok(engine) => engine,
                    Err(_) => return false,
                }
            }
        };
        let mut changed = false;
        if cursor.is_none() {
            let prompts = read_foreign_prompts(foreign);
            changed |= !prompts.is_empty();
            for prompt in &prompts {
                engine.observe(prompt);
            }
        }
        let history = agent_dir.join("history.db");
        let (after, ingested) = ingest_history(&mut *engine, &history, cursor.unwrap_or(0));
        changed |= ingested > 0 || cursor.is_none();
        self.cursor.store(after, Ordering::Relaxed);
        if changed {
            if engine.persist().is_err() || write_cursor(&dir, after).is_err() {
                self.dirty.store(true, Ordering::Relaxed);
            }
        }
        let mut slot = self.slot.lock();
        if let Slot::Loading { pending } = &mut *slot {
            for prompt in pending.drain(..) {
                engine.observe(&prompt);
                self.dirty.store(true, Ordering::Relaxed);
            }
        }
        *slot = Slot::Ready(engine);
        true
    }
}

fn read_cursor(dir: &Path) -> Option<i64> {
    let text = std::fs::read_to_string(dir.join(CURSOR_FILE)).ok()?;
    serde_json::from_str::<serde_json::Value>(&text)
        .ok()?
        .get("historyId")?
        .as_i64()
}

fn write_atomic(dir: &Path, name: &str, bytes: &[u8]) -> std::io::Result<()> {
    let tmp = dir.join(format!(".{name}.tmp{}", std::process::id()));
    let mut file = std::fs::File::create(&tmp)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    std::fs::rename(&tmp, dir.join(name))
}

fn write_cursor(dir: &Path, cursor: i64) -> std::io::Result<()> {
    write_atomic(
        dir,
        CURSOR_FILE,
        format!("{{\"historyId\":{cursor}}}").as_bytes(),
    )
}

/// Copy the TS daemon's snapshot + cursor (read-only on the source). The
/// cursor is read first: a stale cursor only re-ingests a few rows, a newer
/// one than the snapshot would skip them.
fn seed_from_daemon_state(ts_dir: &Path, own: &Path) {
    let Some(cursor) = read_cursor(ts_dir) else {
        return;
    };
    let Ok(snapshot) = std::fs::read(ts_dir.join(SNAPSHOT_FILE)) else {
        return;
    };
    if write_atomic(own, SNAPSHOT_FILE, &snapshot).is_ok() {
        let _ = write_cursor(own, cursor);
    }
}

/// Feed `history.db` rows past `cursor` through `observe`; returns the new
/// cursor and the number of rows ingested. A missing/unreadable db is not an
/// error (fresh install).
fn ingest_history(engine: &mut dyn Predictor, path: &Path, cursor: i64) -> (i64, usize) {
    use rusqlite::{Connection, OpenFlags};
    let Ok(conn) = Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    ) else {
        return (cursor, 0);
    };
    let _ = conn.busy_timeout(std::time::Duration::from_secs(2));
    let Ok(mut page) =
        conn.prepare("SELECT id, prompt FROM history WHERE id > ?1 ORDER BY id LIMIT ?2")
    else {
        return (cursor, 0);
    };
    let (mut cursor, mut ingested) = (cursor, 0usize);
    loop {
        let rows: Vec<(i64, String)> = match page.query_map([cursor, INGEST_BATCH], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
        }) {
            Ok(rows) => rows.filter_map(Result::ok).collect(),
            Err(_) => break,
        };
        let Some(&(last, _)) = rows.last() else { break };
        for (_, prompt) in &rows {
            engine.observe(prompt);
        }
        cursor = last;
        ingested += rows.len();
        if i64::try_from(rows.len()).is_ok_and(|n| n < INGEST_BATCH) {
            break;
        }
    }
    (cursor, ingested)
}

/// Claude Code's and Codex's `history.jsonl` (TS `readForeignPrompts`).
fn foreign_history_files() -> Vec<PathBuf> {
    let home = std::env::var_os("HOME").map(PathBuf::from);
    let claude = std::env::var_os("CLAUDE_CONFIG_DIR")
        .map(PathBuf::from)
        .or_else(|| home.as_ref().map(|h| h.join(".claude")));
    [
        claude.map(|d| d.join("history.jsonl")),
        home.map(|h| h.join(".codex").join("history.jsonl")),
    ]
    .into_iter()
    .flatten()
    .collect()
}

/// Prompts a fresh engine learns once from other coding agents' histories:
/// slash commands are not prose, Claude's paste and image placeholders are not
/// words the user typed.
fn read_foreign_prompts(files: &[PathBuf]) -> Vec<String> {
    let placeholder =
        regex::Regex::new(r"\[(?:Pasted text|Image) #\d+[^\]]*\]").expect("static regex");
    let mut prompts = Vec::new();
    for file in files {
        let Ok(text) = std::fs::read_to_string(file) else {
            continue;
        };
        for line in text.lines() {
            let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
                continue;
            };
            let Some(text) = value
                .get("display")
                .or_else(|| value.get("text"))
                .and_then(|v| v.as_str())
            else {
                continue;
            };
            if text.starts_with('/') {
                continue;
            }
            let prompt = placeholder.replace_all(text, " ");
            let prompt = prompt.trim();
            if !prompt.is_empty() {
                prompts.push(prompt.to_owned());
            }
        }
    }
    prompts
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_history(agent: &Path, rows: &[&str]) {
        let conn = rusqlite::Connection::open(agent.join("history.db")).unwrap();
        conn.execute_batch(
            "CREATE TABLE history(id INTEGER PRIMARY KEY AUTOINCREMENT, prompt TEXT NOT NULL)",
        )
        .unwrap();
        for row in rows {
            conn.execute("INSERT INTO history(prompt) VALUES (?1)", [row])
                .unwrap();
        }
    }

    fn wait_ready(host: &WordHost) {
        for _ in 0..600 {
            if host.is_ready() {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        panic!("engine never became ready");
    }

    #[test]
    fn setting_values_map_to_engines() {
        assert!(method_for_setting("off").is_none());
        for v in ["auto", "ngram", "smollm", "apple"] {
            assert_eq!(method_for_setting(v), Some(Method::Ngram), "{v}");
        }
    }

    #[test]
    fn ingests_history_learns_and_persists_without_touching_daemon_state() {
        let tmp = std::env::temp_dir().join(format!("omp-wc-host-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(tmp.join("predict/ngram")).unwrap();
        write_history(&tmp, &["please refactor the parser carefully"; 12]);
        // A daemon state that must stay byte-identical.
        std::fs::write(
            tmp.join("predict/ngram/ngram.snapshot"),
            b"junk-not-a-snapshot",
        )
        .unwrap();
        std::fs::write(
            tmp.join("predict/ngram/cursor.json"),
            b"{\"historyId\":999}",
        )
        .unwrap();

        let host = WordHost::spawn_with(&tmp, Method::Ngram, Vec::new());
        wait_ready(&host);
        let own = tmp.join("predict/ngram-rs");
        assert!(
            own.join(SNAPSHOT_FILE).exists(),
            "state persisted after ingest"
        );
        assert_eq!(
            read_cursor(&own),
            Some(12),
            "unloadable seed was rebuilt, history re-ingested"
        );
        assert_eq!(
            std::fs::read(tmp.join("predict/ngram/ngram.snapshot")).unwrap(),
            b"junk-not-a-snapshot"
        );
        assert_eq!(
            std::fs::read(tmp.join("predict/ngram/cursor.json")).unwrap(),
            b"{\"historyId\":999}"
        );

        let Answer::Ready(Some(suffix)) = host.complete("please ", "refac") else {
            panic!("expected a learned completion");
        };
        assert_eq!(suffix, "tor");

        // Live observation persists and survives a restart.
        host.observe(
            "zebrafish zebrafish zebrafish zebrafish zebrafish zebrafish zebrafish zebrafish",
        );
        host.persist();
        let again = WordHost::spawn_with(&tmp, Method::Ngram, Vec::new());
        wait_ready(&again);
        assert_eq!(read_cursor(&own), Some(12));
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// Byte-compat check against a snapshot the TS daemon wrote:
    /// `OMP_WC_TS_AGENT=~/.omp/agent cargo test -p omp-tui ts_snapshot -- --ignored`.
    /// The agent dir is only read; the engine state is copied to a temp dir.
    #[test]
    #[ignore = "needs a TS-written ~/.omp/agent/predict/ngram"]
    fn ts_snapshot_loads_and_completes() {
        let agent = PathBuf::from(std::env::var("OMP_WC_TS_AGENT").expect("OMP_WC_TS_AGENT"));
        let tmp = std::env::temp_dir().join(format!("omp-wc-ts-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(tmp.join("predict")).unwrap();
        let src = agent.join("predict/ngram");
        let dst = tmp.join("predict/ngram");
        std::fs::create_dir_all(&dst).unwrap();
        for f in [SNAPSHOT_FILE, CURSOR_FILE] {
            std::fs::copy(src.join(f), dst.join(f)).unwrap();
        }
        let host = WordHost::spawn_with(&tmp, Method::Ngram, Vec::new());
        wait_ready(&host);
        assert!(matches!(
            host.complete("please ", "refac"),
            Answer::Ready(_)
        ));
        assert_eq!(
            read_cursor(&tmp.join("predict/ngram-rs")),
            read_cursor(&src)
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn prompts_observed_while_loading_are_kept() {
        let tmp = std::env::temp_dir().join(format!("omp-wc-pending-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let host = WordHost::spawn_with(&tmp, Method::Ngram, Vec::new());
        host.observe("queued while loading, never lost during startup");
        wait_ready(&host);
        host.persist();
        assert!(tmp.join("predict/ngram-rs").join(SNAPSHOT_FILE).exists());
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
