//! Session persistence (omp session JSONL v3) via `omp-session`.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::Result;
use omp_session::{OmpMessage, SessionWriter};
use pa_agent::agent::Agent;
use pa_agent::types::AgentEvent;
use serde_json::{Map, Value};

use crate::agent::civil;

/// ISO-8601 UTC with milliseconds, like omp ("2026-10-11T14:08:39.790Z").
#[must_use]
pub fn iso_from_ms(ms: i64) -> String {
    let (y, mo, d) = civil(ms.div_euclid(86_400_000));
    let rem = ms.rem_euclid(86_400_000);
    format!(
        "{y:04}-{mo:02}-{d:02}T{:02}:{:02}:{:02}.{:03}Z",
        rem / 3_600_000,
        rem / 60_000 % 60,
        rem / 1000 % 60,
        rem % 1000
    )
}

#[must_use]
pub fn now_iso() -> String {
    iso_from_ms(pa_agent::now_ms())
}

/// UUID v7 string (48-bit ms timestamp + random).
#[must_use]
pub fn uuid_v7() -> String {
    use rand::RngCore;
    let mut b = [0u8; 16];
    rand::thread_rng().fill_bytes(&mut b);
    let ms = u64::try_from(pa_agent::now_ms()).unwrap_or(0);
    b[..6].copy_from_slice(&ms.to_be_bytes()[2..]);
    b[6] = 0x70 | (b[6] & 0x0f);
    b[8] = 0x80 | (b[8] & 0x3f);
    let mut h = String::with_capacity(b.len() * 2);
    for x in &b {
        use std::fmt::Write as _;
        write!(h, "{x:02x}").expect("writing to a String is infallible");
    }
    format!(
        "{}-{}-{}-{}-{}",
        &h[..8],
        &h[8..12],
        &h[12..16],
        &h[16..20],
        &h[20..]
    )
}

/// The default omp-rs session dir for `cwd` (`~/.omp-rs/sessions/<slug>`).
pub(crate) fn sessions_dir(cwd: &Path) -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    let slug = cwd.display().to_string().replace('/', "-");
    Some(PathBuf::from(home).join(".omp-rs/sessions").join(slug))
}

/// TS omp's session dir for `cwd` (`<agent dir>/sessions/<name>`, TS
/// `sessionDirForCwd`): read-only lookup, nothing is created or migrated.
/// The `/resume` picker and `--resume <id>` list it next to the omp-rs store.
#[must_use]
pub fn ts_sessions_dir(cwd: &Path) -> Option<PathBuf> {
    let home = PathBuf::from(std::env::var_os("HOME")?);
    let env_str = |k: &str| std::env::var(k).ok();
    let canon = |p: &Path| std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf());
    let tmp = omp_stats::paths::ts_tmpdir(env_str);
    let name =
        omp_stats::paths::ts_session_dir_name(&canon(cwd), &canon(&home), &canon(Path::new(&tmp)));
    Some(omp_stats::paths::sessions_dir().join(name))
}

/// Live session file: the agent's finished messages are appended to
/// `writer`, which `start_new` swaps for a fresh file (`new_session`).
///
/// An *ephemeral* session (`--no-session`) writes to a scratch directory that
/// is removed on drop and reports no `sessionFile`, like omp's in-memory
/// `SessionManager`.
pub struct Persistence {
    writer: Arc<Mutex<SessionWriter>>,
    session_id: Mutex<String>,
    /// Pending `/tree` leaf move: `Some(Some(id))` after `branch_to(Some(id))`,
    /// `Some(None)` after a root reset, `None` when the writer's append point is
    /// the last journaled entry (no navigation pending). `SessionWriter` keeps
    /// the same information in its private `leaf` and exposes no getter, so the
    /// compaction guard reads this mirror instead; every append (which advances
    /// the writer onto the new entry) clears it.
    nav_leaf: Arc<Mutex<Option<Option<String>>>>,
    ephemeral: bool,
    scratch: Option<PathBuf>,
    /// While set, an assistant error message is held back instead of written
    /// (omp persists a failed turn only once the retry saga gave up on it).
    defer_errors: Arc<AtomicBool>,
    deferred: Arc<Mutex<Option<(pa_types::session::AgentMessage, i64)>>>,
    /// While set, an empty aborted assistant turn is not journaled: a manual
    /// compaction's own abort of the live turn (TS detaches the session from the
    /// agent around that abort, so the stub never reaches the journal).
    drop_empty_aborted: Arc<AtomicBool>,
}

impl Drop for Persistence {
    fn drop(&mut self) {
        if let Some(dir) = &self.scratch {
            let _ = std::fs::remove_dir_all(dir);
        }
    }
}

impl Persistence {
    #[must_use]
    pub fn session_id(&self) -> String {
        self.session_id.lock().unwrap().clone()
    }

    /// Path of the live session file (a scratch path when ephemeral).
    #[must_use]
    pub fn path(&self) -> PathBuf {
        self.writer.lock().unwrap().path().to_path_buf()
    }

    /// The session file omp would report (`sessionFile`); `None` for `--no-session`.
    #[must_use]
    pub fn file(&self) -> Option<PathBuf> {
        (!self.ephemeral).then(|| self.path())
    }

    /// Append a non-message entry (`custom`, ...); omp kinds have typed helpers below.
    ///
    /// # Errors
    /// Fails on serialization or I/O errors.
    #[allow(dead_code)]
    pub fn append_entry(&self, kind: &str, payload: serde_json::Value) -> Result<()> {
        self.writer
            .lock()
            .unwrap()
            .append_entry(kind, payload, &now_iso())
            .map(|_| ())
            .inspect(|_| self.clear_nav())
    }

    /// A shareable sink appending non-message entries (`model_usage`, …) to
    /// this session, for subsystems that only see a process-global hook (RLM
    /// sub-call accounting). Writes are best-effort.
    #[must_use]
    pub fn entry_sink(&self) -> std::sync::Arc<dyn Fn(&str, serde_json::Value) + Send + Sync> {
        let writer = std::sync::Arc::clone(&self.writer);
        let nav_leaf = Arc::clone(&self.nav_leaf);
        std::sync::Arc::new(move |kind, payload| {
            let nav_leaf = Arc::clone(&nav_leaf);
            let _ = writer
                .lock()
                .unwrap()
                .append_entry(kind, payload, &now_iso());
            // An append advances the writer onto the new entry.
            nav_leaf.lock().unwrap().take();
        })
    }

    /// `model_change {model, role?, resolvedModelIsFallback}` (`SessionManager.appendModelChange`).
    ///
    /// # Errors
    /// Fails on serialization or I/O errors.
    pub fn append_model_change(
        &self,
        model: &str,
        role: Option<&str>,
        resolved_model_is_fallback: bool,
    ) -> Result<()> {
        self.writer
            .lock()
            .unwrap()
            .append_model_change(model, role, resolved_model_is_fallback, &now_iso())
            .map(|_| ())
            .inspect(|_| self.clear_nav())
    }

    /// `thinking_level_change {thinkingLevel, configured}` (`appendThinkingLevelChange`),
    /// plus the omp-rs `ultra` mode name while one is active. TS readers take
    /// `thinkingLevel` and ignore the extra field; without `ultra` the line is
    /// exactly TS's.
    ///
    /// # Errors
    /// Fails on serialization or I/O errors.
    pub fn append_thinking_level_change(
        &self,
        level: Option<&str>,
        configured: Option<&str>,
        ultra: Option<&str>,
    ) -> Result<()> {
        let mut w = self.writer.lock().unwrap();
        match ultra {
            None => w.append_thinking_level_change(level, configured, &now_iso()),
            Some(u) => w.append_entry(
                "thinking_level_change",
                serde_json::json!({"thinkingLevel": level, "configured": configured, "ultra": u}),
                &now_iso(),
            ),
        }
        .map(|_| ())
        .inspect(|_| self.clear_nav())
    }

    /// `title_change` audit entry.
    ///
    /// # Errors
    /// Fails on serialization or I/O errors.
    pub fn append_title_change(
        &self,
        title: &str,
        source: &str,
        previous: Option<&str>,
    ) -> Result<()> {
        self.writer
            .lock()
            .unwrap()
            .append_title_change(title, source, previous, None, &now_iso())
            .map(|_| ())
            .inspect(|_| self.clear_nav())
    }

    /// Append a finished conversation message.
    ///
    /// # Errors
    /// Fails on serialization or I/O errors.
    pub fn append_message(
        &self,
        message: &pa_types::session::AgentMessage,
        timestamp_ms: i64,
    ) -> Result<()> {
        self.writer
            .lock()
            .unwrap()
            .append_message(message, &iso_from_ms(timestamp_ms))
            .map(|_| ())
            .inspect(|_| self.clear_nav())
    }

    /// Append a `message` entry whose shape `pa_types` cannot model (e.g. a
    /// custom message with role `"developer"`), preserving the original JSON
    /// keys and order verbatim instead of dropping it.
    ///
    /// # Errors
    /// Fails on serialization or I/O errors.
    pub fn append_raw_message(&self, message: Map<String, Value>, timestamp_ms: i64) -> Result<()> {
        self.writer
            .lock()
            .unwrap()
            .append_omp_message(OmpMessage::Agent(message), &iso_from_ms(timestamp_ms))
            .map(|_| ())
            .inspect(|_| self.clear_nav())
    }

    /// Run `f` on the live session writer (compaction entries need the writer's id chain).
    ///
    /// Note: `f` sees the raw writer, so a leaf move or append inside `f` is
    /// invisible to the navigation mirror. Leaf moves go through
    /// [`Self::branch_to`]; the compaction/seed appends have wrappers below.
    pub fn with_writer<R>(&self, f: impl FnOnce(&mut SessionWriter) -> R) -> R {
        f(&mut self.writer.lock().unwrap())
    }

    /// Move the append point to an earlier entry (`/tree` navigation, omp
    /// `SessionManager.branch`): the next appended entry chains to `id` instead
    /// of the last journaled one; `None` makes it a new root. In-memory until
    /// the next append, like the writer's own leaf.
    ///
    /// # Errors
    /// `id` is not an entry of this session file.
    pub fn branch_to(&self, id: Option<String>) -> Result<()> {
        self.writer.lock().unwrap().branch_to(id.clone())?;
        *self.nav_leaf.lock().unwrap() = Some(id);
        Ok(())
    }

    /// The pending [`Self::branch_to`] move: `Some(target)` / `Some(None)` /
    /// `None` (no move pending — the append point is the last journaled entry).
    #[must_use]
    pub fn leaf_divergence(&self) -> Option<Option<String>> {
        self.nav_leaf.lock().unwrap().clone()
    }

    fn clear_nav(&self) {
        self.nav_leaf.lock().unwrap().take();
    }

    /// Mirror of an append done through the raw [`Self::with_writer`] escape
    /// hatch (the seed DOM fallback): the writer landed on the new entry.
    pub(crate) fn note_append(&self) {
        self.clear_nav();
    }

    /// Append the `compaction` entry on the writer's id chain. An append: the
    /// writer lands on the new entry, so a pending navigation move ends here.
    ///
    /// # Errors
    /// Writer serialization/I/O failure.
    pub fn append_compaction(
        &self,
        result: &omp_compact::summary::CompactionResult,
        timestamp_iso: &str,
        options: &omp_compact::apply::ApplyOptions,
    ) -> Result<String> {
        self.with_writer(|w| omp_compact::append_compaction(w, result, timestamp_iso, options))
            .inspect(|_| self.clear_nav())
    }

    /// Append one seeded branch line (fresh id chained to the writer's leaf,
    /// link fields remapped); falls back to the DOM path. An append.
    ///
    /// # Errors
    /// The line cannot be rewritten or written.
    pub fn append_seeded_line(
        &self,
        raw: &str,
        remap: &std::collections::HashMap<String, String>,
    ) -> Result<String> {
        self.with_writer(|w| w.append_seeded_line(raw, remap))
            .inspect(|_| self.clear_nav())
    }

    /// Append a `branch_summary` entry parented at `branch_from` and land the
    /// writer on it (`/tree` navigation with a summary; an append, which ends a
    /// pending leaf move).
    ///
    /// # Errors
    /// `branch_from` is not an entry of this session file, or the write failed.
    pub fn append_branch_summary(
        &self,
        branch_from: Option<&str>,
        summary: &str,
        details: Option<Value>,
        from_extension: Option<bool>,
        timestamp_iso: &str,
    ) -> Result<String> {
        self.with_writer(|w| {
            w.append_branch_summary(branch_from, summary, details, from_extension, timestamp_iso)
        })
        .inspect(|_| self.clear_nav())
    }

    /// Test-only persistence rooted at `dir` (no agent subscription).
    #[cfg(test)]
    pub(crate) fn for_test(dir: &Path) -> Result<Persistence> {
        let writer = SessionWriter::create(dir, "/test", &uuid_v7(), &now_iso())?;
        Ok(Persistence {
            writer: Arc::new(Mutex::new(writer)),
            session_id: Mutex::new(uuid_v7()),
            nav_leaf: Arc::new(Mutex::new(None)),
            ephemeral: false,
            scratch: None,
            defer_errors: Arc::new(AtomicBool::new(false)),
            deferred: Arc::new(Mutex::new(None)),
            drop_empty_aborted: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Skip (`true`) or journal (`false`) empty aborted assistant turns.
    pub fn drop_empty_aborted(&self, on: bool) {
        self.drop_empty_aborted.store(on, Ordering::SeqCst);
    }

    /// Hold back (`true`) or write straight through (`false`) assistant error turns.
    pub fn defer_errors(&self, on: bool) {
        self.defer_errors.store(on, Ordering::SeqCst);
    }

    /// The retry saga gave up: persist the held failed turn (`persistTerminalEmptyErrorTurn`),
    /// optionally with a rewritten `errorMessage`.
    pub fn flush_deferred_error(&self, error_message: Option<&str>) {
        let held = self.deferred.lock().unwrap().take();
        let Some((m, ts)) = held else { return };
        let m = match error_message {
            Some(e) => serde_json::to_value(&m)
                .ok()
                .map(|mut v| {
                    v["errorMessage"] = serde_json::json!(e);
                    v
                })
                .and_then(|v| serde_json::from_value(v).ok())
                .unwrap_or(m),
            None => m,
        };
        if let Err(e) = self.append_message(&m, ts) {
            eprintln!("omt: session write failed: {e:#}");
        }
    }

    /// The failed turn was retried: it never reaches the session file.
    pub fn drop_deferred_error(&self) {
        self.deferred.lock().unwrap().take();
    }

    /// Switch to a brand-new session file in `dir` (default: the cwd slot).
    ///
    /// # Errors
    /// Fails when the file cannot be created.
    pub fn start_new(&self, cwd: &Path, dir: Option<&Path>) -> Result<()> {
        let dir = match (&self.scratch, dir) {
            (Some(scratch), _) => scratch.clone(),
            (None, Some(d)) => d.to_path_buf(),
            (None, None) => sessions_dir(cwd).ok_or_else(|| anyhow::anyhow!("HOME is not set"))?,
        };
        let id = uuid_v7();
        let fresh = SessionWriter::create(&dir, &cwd.display().to_string(), &id, &now_iso())?;
        {
            let mut fresh = fresh;
            if let Some(e) = crate::context::session_entry() {
                let _ = fresh.append_entry("custom", e, &now_iso());
            }
            *self.writer.lock().unwrap() = fresh;
        }
        *self.session_id.lock().unwrap() = id;
        self.nav_leaf.lock().unwrap().take();
        self.deferred.lock().unwrap().take();
        Ok(())
    }

    /// Continue an existing session file in place (TS `SessionManager.open`): the file and its
    /// header stay untouched and new entries chain to `leaf` (the active branch tip; the last
    /// entry when `None`). The session id becomes the file header's. The previous file is
    /// removed when this process created it and nothing but its preamble was ever written
    /// (no orphan per launch).
    ///
    /// # Errors
    /// Fails when the file cannot be opened or `leaf` is not one of its entries.
    pub fn bind_existing(
        &self,
        path: &Path,
        session_id: Option<&str>,
        leaf: Option<&str>,
    ) -> Result<()> {
        let old = self.path();
        let opened = SessionWriter::open(path, leaf)?;
        *self.writer.lock().unwrap() = opened;
        if let Some(id) = session_id {
            *self.session_id.lock().unwrap() = id.to_owned();
        }
        self.nav_leaf.lock().unwrap().take();
        self.deferred.lock().unwrap().take();
        if old != path {
            discard_if_preamble_only(&old);
        }
        Ok(())
    }

    /// Drop the current file when it holds only the startup preamble (before a switch to a
    /// forked copy).
    pub fn discard_pristine(&self) {
        discard_if_preamble_only(&self.path());
    }
}

/// Remove `path` when every record is a session-start bookkeeping entry (no message ever landed).
fn discard_if_preamble_only(path: &Path) {
    let Ok(text) = std::fs::read_to_string(path) else {
        return;
    };
    let only_preamble = text.lines().filter(|l| !l.trim().is_empty()).all(|l| {
        serde_json::from_str::<Value>(l).ok().is_some_and(|v| {
            matches!(
                v.get("type").and_then(Value::as_str),
                Some(
                    "session"
                        | "custom"
                        | "model_change"
                        | "thinking_level_change"
                        | "service_tier_change"
                        | "session_init"
                )
            )
        })
    });
    if only_preamble {
        let _ = std::fs::remove_file(path);
    }
}

/// Create a session file for this run and append every completed message.
///
/// # Errors
/// Fails when the session file cannot be created.
#[allow(dead_code)]
pub async fn attach(agent: &Agent, cwd: &Path) -> Result<Persistence> {
    attach_with(agent, cwd, false).await
}

/// [`attach`] with `ephemeral = true` for `--no-session`: nothing is kept after exit
/// and `sessionFile` is not reported.
///
/// # Errors
/// Fails when the (scratch) session file cannot be created.
pub async fn attach_with(agent: &Agent, cwd: &Path, ephemeral: bool) -> Result<Persistence> {
    let session_id = uuid_v7();
    let (dir, scratch) = if ephemeral {
        let d = std::env::temp_dir().join(format!(
            "omp-rs-nosession-{}-{}",
            std::process::id(),
            &session_id[..8]
        ));
        (d.clone(), Some(d))
    } else {
        (
            sessions_dir(cwd).ok_or_else(|| anyhow::anyhow!("HOME is not set"))?,
            None,
        )
    };
    attach_in(agent, cwd, &dir, scratch, session_id).await
}

/// [`attach_with`] against an explicit session dir (`scratch` = the dir to remove on drop
/// for an ephemeral session).
///
/// # Errors
/// Fails when the session file cannot be created.
pub(crate) async fn attach_in(
    agent: &Agent,
    cwd: &Path,
    dir: &Path,
    scratch: Option<PathBuf>,
    session_id: String,
) -> Result<Persistence> {
    let ephemeral = scratch.is_some();
    let writer = SessionWriter::create(dir, &cwd.display().to_string(), &session_id, &now_iso())?;
    let mut writer = writer;
    if let Some(e) = crate::context::session_entry() {
        let _ = writer.append_entry("custom", e, &now_iso());
    }
    let writer = Arc::new(Mutex::new(writer));
    let defer_errors = Arc::new(AtomicBool::new(false));
    let deferred = Arc::new(Mutex::new(None));
    let drop_empty_aborted = Arc::new(AtomicBool::new(false));
    let nav_leaf: Arc<Mutex<Option<Option<String>>>> = Arc::new(Mutex::new(None));
    let sub = agent
        .subscribe({
            let writer = writer.clone();
            let defer_errors = defer_errors.clone();
            let drop_empty_aborted = drop_empty_aborted.clone();
            let deferred = deferred.clone();
            let nav_leaf = nav_leaf.clone();
            move |event, _| {
                if let AgentEvent::MessageEnd { message } = &event {
                    if drop_empty_aborted.load(Ordering::SeqCst) && is_empty_aborted(message) {
                        return Box::pin(async { Ok(()) });
                    }
                    let trace = std::env::var_os("OMP_RS_TURN_TRACE").is_some();
                    if trace {
                        crate::runtime::turn_trace!(
                            "[turn-trace] persist-begin role={} rss={}kB",
                            message.role(),
                            crate::runtime::rss_kb()
                        );
                    }
                    let converted = Ok::<_, anyhow::Error>(crate::events::message_value(message))
                        .and_then(|v| {
                            serde_json::from_value::<pa_types::session::AgentMessage>(v)
                                .map_err(anyhow::Error::from)
                        });
                    match converted {
                        Ok(m) => {
                            let held = defer_errors.load(Ordering::SeqCst)
                                && matches!(message, pa_agent::types::AgentMessage::Standard(pa_agent::types::Message::Assistant(a))
                                    if a.stop_reason == pa_agent::types::StopReason::Error);
                            if trace {
                                crate::runtime::turn_trace!(
                                    "[turn-trace] persist-converted rss={}kB",
                                    crate::runtime::rss_kb()
                                );
                            }
                            if held {
                                *deferred.lock().unwrap() = Some((m, message.timestamp()));
                            } else {
                                let ts = iso_from_ms(message.timestamp());
                                if let Err(e) = writer.lock().unwrap().append_message(&m, &ts) {
                                    eprintln!("omt: session write failed: {e:#}");
                                } else {
                                    // An append advances the writer onto the new entry.
                                    nav_leaf.lock().unwrap().take();
                                }
                            }
                        }
                        Err(e) => {
                            // Shapes pa_types cannot model (custom roles like
                            // "developer") still belong in the journal: write
                            // the message's JSON verbatim instead of dropping it.
                            match serde_json::to_value(message) {
                                Ok(serde_json::Value::Object(map)) => {
                                    let omp = OmpMessage::Agent(map);
                                    let ts = iso_from_ms(message.timestamp());
                                    if let Err(e2) =
                                        writer.lock().unwrap().append_omp_message(omp, &ts)
                                    {
                                        eprintln!("omt: session write failed: {e2:#}");
                                    } else {
                                        // An append advances the writer onto the new entry.
                                        nav_leaf.lock().unwrap().take();
                                    }
                                }
                                _ => eprintln!(
                                    "omt: session message conversion failed: {e:#}"
                                ),
                            }
                        }
                    }
                }
                Box::pin(async { Ok(()) })
            }
        })
        .await;
    // Dropping a Subscription does not unsubscribe; the listener lives with the agent.
    drop(sub);
    Ok(Persistence {
        writer,
        session_id: Mutex::new(session_id),
        nav_leaf,
        ephemeral,
        scratch,
        defer_errors,
        deferred,
        drop_empty_aborted,
    })
}

/// An aborted assistant turn that produced nothing (no tool call, thinking or text).
pub(crate) fn is_empty_aborted(message: &pa_agent::types::AgentMessage) -> bool {
    use pa_agent::types::{AgentMessage, AssistantContent, Message, StopReason};
    matches!(message, AgentMessage::Standard(Message::Assistant(a))
        if a.stop_reason == StopReason::Aborted
            && a.content.iter().all(|c| matches!(c, AssistantContent::Text(t) if t.text.trim().is_empty())))
}
