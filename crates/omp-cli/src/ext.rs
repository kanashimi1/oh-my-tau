//! Extension host wiring: spawns `ext-host/host.ts` (Bun), loads the user's
//! `~/.omp/agent/extensions` plus `--extension` paths, bridges the host onto
//! the agent (`transform_context`/`before_tool_call`/`after_tool_call`),
//! registers extension tools, serves host->core requests (`append_entry`,
//! `send_message`, ui frames), and emits omp lifecycle events.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{bail, Context, Result};
use pa_agent::abort::AbortSignal;
use pa_agent::agent::{Agent, Subscription};
use pa_agent::agent_loop::{AfterToolCallFn, BeforeToolCallFn, TransformContextFn};
use pa_agent::types::{
    AgentEvent, AgentMessage, AgentTool, AgentToolResult, CustomAgentMessage,
    GetContinuationMessagesContext, ToolResultContent,
};
use serde_json::{json, Value};

use omp_ext::hooks::ExtHooks;
use omp_ext::lifecycle::{self, InputSource, LifecycleEvent, SwitchReason};
use omp_ext::{ExtHost, HostConfig, HostMessage, ToolInfo};

use crate::events::{message_value, ExtUpdateEncoder};
use crate::runtime::Runtime;

/// Default budget for one extension tool call (host default matches).
fn tool_timeout() -> Duration {
    let ms = std::env::var("OMP_EXT_TIMEOUT_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(30_000);
    Duration::from_millis(ms)
}

/// TS `SESSION_STOP_CONTINUATION_CAP` (agent-session.ts): at most this many
/// `session_stop` continuation turns per run; `decision: "block"` is exempt.
const SESSION_STOP_CONTINUATION_CAP: usize = 8;

/// TS `prompts/system/session-stop-blocked.md` (rendered when a `block`
/// decision carries no reason/additionalContext).
const SESSION_STOP_BLOCKED_PROMPT: &str = "A session_stop handler blocked completion without a reason. Resolve the outstanding work before finishing.";

/// Per-run `session_stop` control-hook state (TS `#sessionStopContinuationCount`
/// / `#sessionStopHookActive`).
#[derive(Debug, Default, PartialEq, Eq)]
struct StopState {
    continuations: usize,
    hook_active: bool,
}

/// TS `#sessionStopContinuationContext`: the hidden next-turn message a
/// `session_stop` outcome schedules, if any.
fn stop_continuation_text(outcome: &omp_ext::lifecycle::SessionStopOutcome) -> Option<String> {
    let additional = outcome
        .additional_context
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty());
    let reason = outcome
        .reason
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty());
    if outcome.block {
        Some(
            reason
                .or(additional)
                .unwrap_or(SESSION_STOP_BLOCKED_PROMPT)
                .to_string(),
        )
    } else if outcome.continue_turn {
        additional.or(reason).map(str::to_string)
    } else {
        None
    }
}

/// Whether a session has any extension to load: explicit `--extension`/`--hook`
/// paths, or an entry the host's discovery would pick up in
/// `~/.omp/agent/extensions` (a `.ts`/`.js` file or a directory with an index).
#[must_use]
pub fn has_extensions(extra: &[PathBuf]) -> bool {
    if !extra.is_empty() {
        return true;
    }
    let Some(dir) = std::env::var_os("HOME")
        .map(PathBuf::from)
        .map(|h| h.join(".omp").join("agent").join("extensions"))
    else {
        return false;
    };
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return false;
    };
    entries.flatten().any(|entry| {
        let path = entry.path();
        if path.is_dir() {
            ["index.ts", "index.js"].iter().any(|i| path.join(i).is_file())
        } else {
            path.extension()
                .is_some_and(|e| e == "ts" || e == "js")
        }
    })
}

/// Locate `ext-host/host.ts`: `$OMP_RS_EXT_HOST`, else the workspace's copy
/// found by walking up from the binary's location.
#[must_use]
pub fn host_script() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("OMP_RS_EXT_HOST") {
        let p = PathBuf::from(p);
        if p.is_file() {
            return Some(p);
        }
    }
    let exe = std::env::current_exe().ok()?;
    for dir in exe.ancestors() {
        let cand = dir.join("ext-host").join("host.ts");
        if cand.is_file() {
            return Some(cand);
        }
    }
    None
}

/// FNV-1a: a hash that stays stable across Rust versions (socket names must
/// match between differently built omp-rs binaries sharing one daemon).
fn fnv1a(parts: &[&[u8]]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for part in parts {
        for b in *part {
            h ^= u64::from(*b);
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
        h ^= 0xff;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// Shared extension-host daemon socket (the default mode; one bun process per
/// user serves every omp-rs session, each in its own Worker). `None` selects
/// the per-process private host: `OMP_RS_EXT_HOST=private` (or the legacy
/// `OMP_RS_EXT_SHARED=0`).
///
/// Path: `$OMP_RS_EXT_SOCKET` verbatim, else
/// `$XDG_RUNTIME_DIR/omp-rs/ext-<key>.sock`, else
/// `~/.omp-rs/run/ext-<key>.sock`. `<key>` hashes the host version, the host
/// script path/size/mtime (a changed host never talks to an old daemon) and
/// the explicit extension paths, so different extension sets or builds get
/// their own daemon.
fn shared_socket_path(script: &Path, extra: &[PathBuf]) -> Option<PathBuf> {
    if std::env::var("OMP_RS_EXT_HOST").is_ok_and(|v| v == "private")
        || std::env::var_os("OMP_RS_EXT_SHARED").is_some_and(|v| v == "0")
    {
        return None;
    }
    if let Ok(p) = std::env::var("OMP_RS_EXT_SOCKET") {
        return Some(PathBuf::from(p));
    }
    let dir = match std::env::var_os("XDG_RUNTIME_DIR").filter(|d| !d.is_empty()) {
        Some(x) => PathBuf::from(x).join("omp-rs"),
        None => PathBuf::from(std::env::var_os("HOME")?)
            .join(".omp-rs")
            .join("run"),
    };
    let stamp = |p: &Path| {
        std::fs::metadata(p)
            .map(|m| {
                let t = m
                    .modified()
                    .ok()
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map_or(0, |d| d.as_nanos());
                format!("{}:{t}", m.len())
            })
            .unwrap_or_default()
    };
    let canon = |p: &Path| p.canonicalize().unwrap_or_else(|_| p.to_path_buf());
    let script = canon(script);
    let mut sets: Vec<String> = extra
        .iter()
        .map(|p| canon(p).to_string_lossy().into_owned())
        .collect();
    sets.sort();
    let mut parts: Vec<Vec<u8>> = vec![
        env!("CARGO_PKG_VERSION").into(),
        script.to_string_lossy().as_bytes().to_vec(),
        stamp(&script).into_bytes(),
    ];
    if let Some(parent) = script.parent() {
        parts.push(stamp(&parent.join("shard.ts")).into_bytes());
        parts.push(stamp(&parent.join("session.worker.ts")).into_bytes());
    }
    parts.extend(sets.into_iter().map(String::into_bytes));
    let refs: Vec<&[u8]> = parts.iter().map(Vec::as_slice).collect();
    Some(dir.join(format!("ext-{:016x}.sock", fnv1a(&refs))))
}

/// Gate decision from the `input` event before a prompt is admitted.
pub enum PromptGate {
    /// Run with (possibly rewritten) text.
    Run(String),
    /// An extension handled the input; skip the prompt.
    Handled,
}

/// Last session ctx sent to the host; kept so a shared-daemon reconnect (or a
/// title change) can re-send the full merged ctx (`host.ts set_ctx` merges
/// into state → `getSessionName()` etc.).
#[derive(Debug, Clone, Default)]
struct SessionCtx {
    session_id: String,
    session_file: Option<PathBuf>,
    #[allow(dead_code)] // ctx.model is the live Model object (see `Live`), not this ref
    model: String,
    session_name: Option<String>,
    /// TS `ctx.hasUI` (see [`ExtMode::has_ui`]).
    has_ui: bool,
    /// TS `ctx.mode`.
    mode: &'static str,
}

/// Which TS mode the host serves (`ctx.mode`, and `hasUI`: TS `hasUI()` is true exactly when a UI
/// context was provided — interactive and non-headless rpc — and false for print/json).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtMode {
    Print,
    Json,
    Rpc,
    Tui,
}

impl ExtMode {
    fn name(self) -> &'static str {
        match self {
            Self::Print => "print",
            Self::Json => "json",
            Self::Rpc => "rpc",
            Self::Tui => "tui",
        }
    }
    fn has_ui(self) -> bool {
        matches!(self, Self::Rpc | Self::Tui)
    }
}

/// `HostResponder` backed by the runtime: dialogs travel as `extension_ui_request` frames (the TUI
/// renders them; an rpc client answers them), `ctx.compact` / `navigateTree` run on the session.
struct RuntimeResponder {
    rt: std::sync::Weak<Runtime>,
    mode: ExtMode,
}

impl RuntimeResponder {
    /// Send one dialog and await the answer object (`None` = cancelled / no UI).
    async fn dialog(
        &self,
        method: &str,
        mut frame: serde_json::Map<String, Value>,
    ) -> Option<Value> {
        if !self.mode.has_ui() {
            return None;
        }
        let rt = self.rt.upgrade()?;
        let id = format!(
            "ext-ui-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos())
        );
        frame.insert("type".into(), json!("extension_ui_request"));
        frame.insert("id".into(), json!(id));
        frame.insert("method".into(), json!(method));
        let waiter = crate::rpc::register_pending_ui_request(&id);
        rt.send(Value::Object(frame));
        let answer = waiter.await.ok()?;
        (answer.get("cancelled").and_then(Value::as_bool) != Some(true)).then_some(answer)
    }

    fn text(answer: Option<Value>) -> Option<String> {
        answer?
            .get("value")
            .and_then(Value::as_str)
            .map(str::to_owned)
    }
}

impl omp_ext::lifecycle::HostResponder for RuntimeResponder {
    fn ui_confirm<'a>(
        &'a self,
        title: &'a str,
        message: &'a str,
    ) -> omp_ext::lifecycle::BoxFuture<'a, bool> {
        Box::pin(async move {
            let mut f = serde_json::Map::new();
            f.insert("title".into(), json!(title));
            f.insert("message".into(), json!(message));
            // TUI answers `{value: bool}`, an rpc client `{confirmed: bool}` (TS `RpcExtensionUIContext`).
            self.dialog("confirm", f).await.is_some_and(|a| {
                a.get("confirmed").and_then(Value::as_bool) == Some(true)
                    || a.get("value").and_then(Value::as_bool) == Some(true)
            })
        })
    }

    fn ui_select<'a>(
        &'a self,
        title: &'a str,
        options: Vec<String>,
    ) -> omp_ext::lifecycle::BoxFuture<'a, Option<String>> {
        Box::pin(async move {
            let mut f = serde_json::Map::new();
            f.insert("title".into(), json!(title));
            f.insert("options".into(), json!(options));
            Self::text(self.dialog("select", f).await)
        })
    }

    fn ui_input<'a>(
        &'a self,
        title: &'a str,
        placeholder: Option<&'a str>,
    ) -> omp_ext::lifecycle::BoxFuture<'a, Option<String>> {
        Box::pin(async move {
            let mut f = serde_json::Map::new();
            f.insert("title".into(), json!(title));
            f.insert("placeholder".into(), json!(placeholder));
            Self::text(self.dialog("input", f).await)
        })
    }

    fn ui_editor<'a>(
        &'a self,
        title: &'a str,
        prefill: Option<&'a str>,
    ) -> omp_ext::lifecycle::BoxFuture<'a, Option<String>> {
        Box::pin(async move {
            let mut f = serde_json::Map::new();
            f.insert("title".into(), json!(title));
            f.insert("prefill".into(), json!(prefill));
            Self::text(self.dialog("editor", f).await)
        })
    }

    fn compact(&self, options: Value) -> omp_ext::lifecycle::BoxFuture<'_, Result<(), String>> {
        Box::pin(async move {
            let Some(rt) = self.rt.upgrade() else {
                return Ok(());
            };
            let instructions = match &options {
                Value::String(s) => Some(s.clone()),
                Value::Object(o) => o
                    .get("customInstructions")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                _ => None,
            };
            rt.compact(instructions.as_deref())
                .await
                .map(|_| ())
                .map_err(|e| e.message)
        })
    }

    fn navigate_tree<'a>(
        &'a self,
        target: &'a str,
        options: Value,
    ) -> omp_ext::lifecycle::BoxFuture<'a, Result<Value, String>> {
        Box::pin(async move {
            // TS: only command contexts (interactive / rpc) carry a real handler.
            let (true, Some(rt)) = (
                matches!(self.mode, ExtMode::Tui | ExtMode::Rpc),
                self.rt.upgrade(),
            ) else {
                return Ok(json!({"cancelled": false}));
            };
            crate::tree::navigate_tree(&rt, target, &options, None)
                .await
                .map(|v| v.unwrap_or_else(|| json!({"cancelled": false})))
                .map_err(|e| e.message)
        })
    }

    fn model_complete(
        &self,
        request: Value,
    ) -> omp_ext::lifecycle::BoxFuture<'_, Result<Value, String>> {
        Box::pin(async move {
            let provider = request
                .get("provider")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let id = request
                .get("model")
                .and_then(Value::as_str)
                .unwrap_or_default();
            if provider.is_empty() || id.is_empty() {
                return Err("model_complete requires provider and model".into());
            }
            let selected = crate::model::select(Some(&format!("{provider}/{id}")))
                .await
                .map_err(|e| format!("model resolution failed: {e:#}"))?;
            let context: pa_types::ai::Context =
                serde_json::from_value(request.get("context").cloned().unwrap_or(Value::Null))
                    .map_err(|e| format!("invalid completion context: {e}"))?;
            let options_value = request.get("options").cloned().unwrap_or(Value::Null);
            let options = pa_ai::types::StreamOptions {
                api_key: selected.api_key,
                max_tokens: options_value.get("maxTokens").and_then(Value::as_u64),
                session_id: options_value
                    .get("sessionId")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                timeout_ms: options_value.get("timeoutMs").and_then(Value::as_u64),
                headers: selected.headers.map(|h| h.into_iter().collect()),
                ..Default::default()
            };
            let response = pa_ai::complete(&selected.model, &context, Some(options))
                .await
                .map_err(|e| format!("model completion failed: {e}"))?;
            serde_json::to_value(response).map_err(|e| format!("completion encoding failed: {e}"))
        })
    }
}

/// TS `normalizeCustomMessagePayload` + the `#promptAgent` wrapper: a `before_agent_start` message
/// payload becomes a `role: "custom"` message (attribution `user` unless the payload sets one).
fn injected_message(payload: &Value) -> Option<AgentMessage> {
    let (custom_type, content, display, details, attribution) = match payload {
        Value::String(text) => ("message".to_string(), json!(text), true, None, None),
        Value::Object(o) => (
            o.get("customType")
                .and_then(Value::as_str)
                .unwrap_or("message")
                .to_string(),
            o.get("content").cloned().unwrap_or_else(|| json!("")),
            o.get("display").and_then(Value::as_bool).unwrap_or(false),
            o.get("details").cloned(),
            o.get("attribution")
                .and_then(Value::as_str)
                .filter(|a| matches!(*a, "user" | "agent")),
        ),
        _ => return None,
    };
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(0));
    let mut payload = json!({
        "customType": custom_type,
        "content": content,
        "display": display,
        "attribution": attribution.unwrap_or("user"),
        "timestamp": timestamp,
    });
    if let Some(d) = details {
        payload["details"] = d;
    }
    Some(AgentMessage::Custom(pa_agent::types::CustomAgentMessage {
        role: "custom".to_string(),
        payload,
    }))
}

/// Live session facts the host ctx exposes synchronously (TS `ctx.isIdle()`,
/// `getContextUsage()`, `hasPendingMessages()`, `sessionManager.getBranch()`, ...). Refreshed by
/// [`Extensions::refresh_live`] at event boundaries; the host-side provider
/// ([`omp_ext::ExtHost::set_ctx_provider`]) rides it on every emit.
#[derive(Default)]
struct Live {
    snap: Mutex<serde_json::Map<String, Value>>,
    /// Branch entries (root -> leaf) not yet sent to the host.
    branch: Mutex<Option<Value>>,
    /// Session file len the cached branch was computed at.
    branch_len: Mutex<Option<(PathBuf, u64)>>,
    /// Some loaded extension reads the branch (`getBranch`/`getEntry`/`getLeafId`/`getEntries`).
    needs_branch: std::sync::atomic::AtomicBool,
}

/// Live extension host and its agent-facing hooks.
pub struct Extensions {
    pub host: Arc<ExtHost>,
    hooks: ExtHooks,
    /// Kept alive for the session's lifetime (drop unsubscribes).
    sub: Mutex<Option<Subscription>>,
    turn: Arc<AtomicU64>,
    /// `session_stop` continuation bookkeeping (TS agent-session fields).
    stop: Arc<Mutex<StopState>>,
    /// Session cwd (extension `load` param), for reconnect recovery.
    cwd: PathBuf,
    /// Explicit `--extension` paths, for reconnect recovery.
    extra: Vec<PathBuf>,
    /// Default extensions dir, for reconnect recovery.
    dir: Option<String>,
    /// Last session ctx sent to the host.
    ctx: Mutex<SessionCtx>,
    mode: Mutex<ExtMode>,
    /// `(base prompt, override applied)` of the last `before_agent_start` system-prompt override.
    prompt_override: Mutex<Option<(String, String)>>,
    /// Live per-event ctx facts (see [`Live`]).
    live: Arc<Live>,
    /// Runtime attached via [`Extensions::attach`]; re-served on reconnect.
    rt: Mutex<Option<Arc<Runtime>>>,
    /// Tool calls an extension `tool_call` handler blocked: their result events are delivered to
    /// extensions in TS's order (see `subscribe_lifecycle`).
    blocked_calls: Arc<Mutex<std::collections::HashSet<String>>>,
    /// Print/json runs: TS reaches extensions with `agent_end` only after `session_shutdown`
    /// (its settle path emits it detached), so the messages wait here for [`Self::shutdown`].
    deferred_agent_end: Arc<Mutex<Option<Vec<Value>>>>,
}

/// `session_before_compact` / `session_compact` emission around runtime compactions.
struct CompactBridge {
    host: Arc<ExtHost>,
}

impl crate::runtime::CompactionHooks for CompactBridge {
    fn before<'a>(
        &'a self,
        preparation: Value,
        branch_entries: Vec<Value>,
        custom_instructions: Option<&'a str>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = bool> + Send + 'a>> {
        Box::pin(async move {
            if !self.host.has_handlers("session_before_compact") {
                return false;
            }
            match self
                .host
                .emit_session_before_compact(
                    preparation,
                    branch_entries,
                    custom_instructions.map(str::to_owned),
                    None,
                )
                .await
            {
                Ok(outcome) => outcome.cancel,
                Err(e) => {
                    eprintln!("omt: extension session_before_compact: {e:#}");
                    false
                }
            }
        })
    }

    fn after<'a>(
        &'a self,
        compaction_entry: Value,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send + 'a>> {
        Box::pin(async move {
            emit_silent(
                self.host
                    .emit_session_compact(compaction_entry, false, None)
                    .await,
            );
        })
    }
}

impl Live {
    /// ctx merged into the host on each emit: the cheap snapshot, plus the branch once per change.
    fn provide(&self) -> Option<Value> {
        let mut out = self
            .snap
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        if let Some(b) = self
            .branch
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
        {
            out.insert(
                "leafId".into(),
                b.as_array()
                    .and_then(|a| a.last())
                    .and_then(|e| e.get("id"))
                    .cloned()
                    .unwrap_or(Value::Null),
            );
            out.insert("branch".into(), b);
        }
        (!out.is_empty()).then_some(Value::Object(out))
    }
}

/// Whether any extension source reads the session branch (so the core must feed it).
fn sources_read_branch(dir: Option<&str>, extra: &[PathBuf]) -> bool {
    fn scan(p: &Path, depth: usize) -> bool {
        if p.is_dir() {
            if depth == 0 {
                return false;
            }
            return std::fs::read_dir(p).is_ok_and(|rd| {
                rd.flatten()
                    .filter(|e| !e.file_name().to_string_lossy().starts_with('.'))
                    .any(|e| scan(&e.path(), depth - 1))
            });
        }
        p.extension().is_some_and(|e| e == "ts" || e == "js")
            && std::fs::read_to_string(p).is_ok_and(|t| {
                ["getBranch", "getEntries", "getEntry(", "getLeafId"]
                    .iter()
                    .any(|k| t.contains(k))
            })
    }
    dir.is_some_and(|d| scan(Path::new(d), 3)) || extra.iter().any(|p| scan(p, 3))
}

impl Extensions {
    /// Spawn the host and load extensions (`dir` = `~/.omp/agent/extensions`,
    /// `extra` = repeatable `--extension` paths).
    ///
    /// # Errors
    /// Host spawn or `load` transport failure (load errors on individual
    /// extensions are reported by the host and do not fail here).
    pub async fn start(cwd: &Path, extra: &[PathBuf]) -> Result<Arc<Self>> {
        let script = host_script().context("ext-host/host.ts not found (set OMP_RS_EXT_HOST)")?;
        let mut cfg = HostConfig::new(script);
        cfg.trace = std::env::var_os("OMP_EXT_TRACE").map(Into::into);
        // Shared host by default: one bun daemon per user serves all sessions;
        // each session still runs its extensions in an isolated bun Worker
        // (fresh module graph == per-process state). `OMP_RS_EXT_HOST=private`
        // keeps the per-process host.
        cfg.shared_socket = shared_socket_path(&cfg.host_script, extra);
        let host = ExtHost::spawn(&cfg).await?;
        let hooks = ExtHooks::new(host.clone());

        let dir = std::env::var_os("HOME")
            .map(PathBuf::from)
            .map(|h| h.join(".omp").join("agent").join("extensions"))
            .filter(|d| d.is_dir())
            .map(|d| d.to_string_lossy().into_owned());
        let paths: Vec<PathBuf> = extra.to_vec();
        let loaded = host
            .load(
                if paths.is_empty() { None } else { Some(&paths) },
                dir.as_deref(),
                &cwd.to_string_lossy(),
                Some(json!({"cwd": cwd.to_string_lossy(), "hasUI": false, "branch": []})),
            )
            .await?;
        for err in &loaded.errors {
            eprintln!(
                "omt: extension failed to load: {}: {}",
                err.path, err.error
            );
        }

        let ext = Arc::new(Self {
            host,
            hooks,
            sub: Mutex::new(None),
            turn: Arc::new(AtomicU64::new(0)),
            stop: Arc::new(Mutex::new(StopState::default())),
            cwd: cwd.to_path_buf(),
            extra: extra.to_vec(),
            dir,
            ctx: Mutex::new(SessionCtx::default()),
            mode: Mutex::new(ExtMode::Print),
            prompt_override: Mutex::new(None),
            live: Arc::new(Live::default()),
            rt: Mutex::new(None),
            blocked_calls: Arc::new(Mutex::new(std::collections::HashSet::new())),
            deferred_agent_end: Arc::new(Mutex::new(None)),
        });
        ext.live.needs_branch.store(
            sources_read_branch(ext.dir.as_deref(), &ext.extra),
            Ordering::Relaxed,
        );
        {
            let live = ext.live.clone();
            ext.host
                .set_ctx_provider(Some(Arc::new(move |_ty: &str| live.provide())));
        }
        if ext.host.may_reconnect() {
            // Shared-daemon supervisor: on daemon death the host transparently
            // falls back to a private host; re-load extensions, re-send the
            // session ctx, re-emit `session_start` (as TS does on extension
            // reload), and re-serve host->core messages.
            let ext = ext.clone();
            tokio::spawn(async move { ext.supervise_reconnect().await });
        }
        Ok(ext)
    }

    /// `pa-agent` hook points backed by the host.
    #[must_use]
    pub fn hook_fns(&self) -> (TransformContextFn, BeforeToolCallFn, AfterToolCallFn) {
        let inner = self.hooks.before_tool_call();
        let blocked = self.blocked_calls.clone();
        let before: BeforeToolCallFn = Arc::new(move |ctx, signal| {
            let id = ctx.tool_call.id.clone();
            let (fut, blocked) = (inner(ctx, signal), blocked.clone());
            Box::pin(async move {
                let result = fut.await?;
                if result.as_ref().is_some_and(|r| r.block) {
                    blocked
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .insert(id);
                }
                Ok(result)
            })
        });
        let inner_after = self.hooks.after_tool_call();
        let after: AfterToolCallFn = Arc::new(move |ctx, signal| {
            // A call the approval gate refused never ran, so (as in TS) `tool_result` is not emitted.
            if crate::approval::Gate::get().is_some_and(|g| g.take_refused(&ctx.tool_call.id)) {
                return Box::pin(async { Ok(None) });
            }
            inner_after(ctx, signal)
        });
        // While an ultra mode runs CLM, `context-slim` stands aside: the core
        // owns context maintenance then.
        let transform = self
            .hooks
            .transform_context_skipping(Arc::new(crate::context::skip_extensions));
        (transform, before, after)
    }

    /// Notify request-accounting hooks at the real provider boundary.
    /// Currently payload mutation remains on the native Rust stream path.
    pub async fn before_provider_request(&self) -> Result<()> {
        if self.host.has_handlers("before_provider_request") {
            self.host
                .emit(json!({"type":"before_provider_request","payload":{}}), None)
                .await?;
        }
        Ok(())
    }

    /// Extension tools as `AgentTool` wrappers (`list_tools` -> `call_tool`).
    ///
    /// # Errors
    /// Transport failure on `list_tools`.
    pub async fn tools(&self) -> Result<Vec<Arc<dyn AgentTool>>> {
        Ok(self
            .host
            .list_tools()
            .await?
            .into_iter()
            .map(|info| {
                Arc::new(ExtTool {
                    host: self.host.clone(),
                    info,
                }) as Arc<dyn AgentTool>
            })
            .collect())
    }

    /// Merge session info into the host's ctx state. Also sends the stored
    /// `sessionName` (see [`Self::set_session_name`]) so the merged state
    /// stays complete across reconnects.
    ///
    /// # Errors
    /// Transport failure.
    pub async fn set_session_ctx(
        &self,
        session_id: &str,
        session_file: Option<&Path>,
        model: &str,
    ) -> Result<()> {
        {
            let mut ctx = self
                .ctx
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            ctx.session_id = session_id.to_string();
            ctx.session_file = session_file.map(Path::to_path_buf);
            ctx.model = model.to_string();
        }
        self.send_ctx().await
    }

    /// Update the session title in the host's ctx state (`host.ts set_ctx`
    /// merges into state → `getSessionName()`), so extensions like
    /// herdr-pane-names see title changes without re-reading the session file.
    /// `None` clears the title. Safe to call on every title change: the value
    /// is stored and the full merged ctx is re-sent. Before the first
    /// `set_session_ctx` the update is stored only (nothing to merge into).
    ///
    /// # Errors
    /// Transport failure.
    pub async fn set_session_name(&self, name: Option<String>) -> Result<()> {
        let ready = {
            let mut ctx = self
                .ctx
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            ctx.session_name = name;
            !ctx.session_id.is_empty()
        };
        if ready {
            self.send_ctx().await?;
        }
        Ok(())
    }

    /// Send the stored session ctx (id/file/model/name) to the host.
    ///
    /// # Errors
    /// Transport failure.
    async fn send_ctx(&self) -> Result<()> {
        let body = {
            let ctx = self
                .ctx
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            json!({
                "sessionId": ctx.session_id,
                "sessionFile": ctx.session_file.as_ref().map(|p| p.to_string_lossy().into_owned()),
                "sessionName": ctx.session_name,
                "hasUI": ctx.has_ui,
                "mode": ctx.mode,
            })
        };
        self.host.set_ctx(body).await
    }

    /// `session_start` (once per session).
    pub async fn session_start(&self) {
        emit_silent(self.host.emit_session_start(None).await);
    }

    /// TS `before_agent_start` (agent-session `#promptAgent`, every mode): handlers may
    /// replace the system prompt and inject messages ahead of the prompt.
    ///
    /// # Errors
    /// Transport failure; handler outcomes are applied best-effort.
    pub async fn before_agent_start(
        &self,
        agent: &Agent,
        prompt: &str,
        images: Option<Value>,
    ) -> Result<Vec<AgentMessage>> {
        // TS keeps the override per turn (`setTurnSystemPromptOverride`, cleared on the next
        // prompt): start every prompt from the base prompt, never from the last override.
        let mut base = agent.system_prompt().await;
        let restore = {
            let mut applied = self
                .prompt_override
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            applied
                .take()
                .filter(|(_, set)| base == *set)
                .map(|(orig, _)| orig)
        };
        if let Some(orig) = restore {
            agent.set_system_prompt(orig.clone()).await;
            base = orig;
        }
        let outcome = self
            .host
            .emit_before_agent_start(prompt.to_string(), images, vec![base.clone()], None)
            .await?;
        if let Some(list) = outcome.system_prompt {
            let joined = list.join("\n\n");
            *self
                .prompt_override
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some((base, joined.clone()));
            agent.set_system_prompt(joined).await;
        }
        // Injected messages ride the prompt (TS `#promptAgent`: `[user, ...injected]`), so they get
        // message events and are persisted like any prompt message.
        Ok(outcome
            .messages
            .iter()
            .filter_map(injected_message)
            .collect())
    }

    /// TS `input` gate (interactive input-controller / rpc ordered-input dispatch; never print
    /// mode): handlers may rewrite the text or consume the input.
    ///
    /// # Errors
    /// Transport failure.
    pub async fn input_gate(&self, prompt: &str, source: InputSource) -> Result<PromptGate> {
        let input = self
            .host
            .emit_input(prompt.to_string(), None, source, None)
            .await?;
        if input.handled {
            return Ok(PromptGate::Handled);
        }
        Ok(PromptGate::Run(
            input.text.unwrap_or_else(|| prompt.to_string()),
        ))
    }

    /// Attach to a running runtime: serve host->core notifications
    /// (`append_entry` -> session writer, `send_message` -> queue,
    /// ui notifications -> `extension_ui_request` frames) and subscribe
    /// lifecycle forwarding (`agent_*`, `turn_*`, `message_*`,
    /// `tool_execution_*`, `session_stop`).
    pub fn attach(self: &Arc<Self>, rt: Arc<Runtime>) {
        *self
            .rt
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(rt.clone());
        self.serve_incoming(&rt);
        rt.set_compaction_hooks(Some(Arc::new(CompactBridge {
            host: self.host.clone(),
        })));
        // `before_agent_start` runs inside every prompt admission, like TS `#promptAgent`.
        let ext = self.clone();
        let weak = Arc::downgrade(&rt);
        rt.set_before_start_hook(Some(Arc::new(
            move |prompt: String, images: Option<Value>| {
                let (ext, weak) = (ext.clone(), weak.clone());
                Box::pin(async move {
                    let Some(rt) = weak.upgrade() else {
                        return Vec::new();
                    };
                    ext.before_agent_start(&rt.agent, &prompt, images)
                        .await
                        .unwrap_or_else(|e| {
                            eprintln!("omt: extension before_agent_start: {e:#}");
                            Vec::new()
                        })
                })
            },
        )));
        let ext = self.clone();
        tokio::spawn(async move {
            ext.refresh_live(&rt, true).await;
            let sub = ext.subscribe_lifecycle(rt).await;
            *ext.sub
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(sub);
            // Hold the subscription for the session's lifetime.
            std::future::pending::<()>().await;
        });
    }

    /// Serve host->core messages on the host's current incoming channel.
    /// Called by [`Self::attach`] and again on every reconnect (the fallback
    /// host has a fresh channel; the previous consumer's task ends when its
    /// sender drops).
    fn responder(&self, rt: &Arc<Runtime>) -> Arc<dyn omp_ext::lifecycle::HostResponder> {
        let mode = *self
            .mode
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        Arc::new(RuntimeResponder {
            rt: Arc::downgrade(rt),
            mode,
        })
    }

    /// Declare the serving mode before [`Self::attach`]: sets `ctx.mode` / `hasUI` like TS
    /// (`print`/`json`: no UI; `rpc` and `tui`: a UI context exists).
    fn mode(&self) -> ExtMode {
        *self
            .mode
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    pub fn set_mode(&self, mode: ExtMode) {
        *self
            .mode
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = mode;
        let mut ctx = self
            .ctx
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        ctx.has_ui = mode.has_ui();
        ctx.mode = mode.name();
    }

    fn serve_incoming(self: &Arc<Self>, rt: &Arc<Runtime>) {
        if let Some(incoming) = self.host.take_incoming() {
            let host = self.host.clone();
            let runtime = rt.clone();
            let mut notifications = lifecycle::serve(&host, incoming, self.responder(rt));
            tokio::spawn(async move {
                while let Some(msg) = notifications.recv().await {
                    handle_notification(&runtime, msg);
                }
            });
        }
    }

    /// Wait for shared-daemon-death reconnects and recover the session on
    /// each: re-load extensions, re-send the session ctx, re-emit
    /// `session_start` (as TS does on extension reload), resume serving
    /// host->core messages. Emits keep flowing afterwards; a failure is
    /// logged and the next prompt retries recovery on the following event.
    async fn supervise_reconnect(self: Arc<Self>) {
        let mut generations = self.host.reconnections();
        loop {
            if generations.changed().await.is_err() {
                return;
            }
            eprintln!("omt: extension host reconnected; reloading extensions and session ctx");
            if let Err(e) = self.recover().await {
                eprintln!("omt: extension host recovery: {e:#}");
            }
        }
    }

    /// Re-establish session state on a transparently reconnected host.
    ///
    /// # Errors
    /// Transport failure on reload or ctx re-send.
    async fn recover(self: &Arc<Self>) -> Result<()> {
        let loaded = self
            .host
            .load(
                if self.extra.is_empty() {
                    None
                } else {
                    Some(&self.extra)
                },
                self.dir.as_deref(),
                &self.cwd.to_string_lossy(),
                Some(json!({"cwd": self.cwd.to_string_lossy(), "hasUI": false, "branch": []})),
            )
            .await?;
        for err in &loaded.errors {
            eprintln!(
                "omt: extension failed to load: {}: {}",
                err.path, err.error
            );
        }
        self.send_ctx().await?;
        *self
            .live
            .branch_len
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
        emit_silent(self.host.emit_session_start(None).await);
        if let Some(rt) = self
            .rt
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
        {
            self.serve_incoming(&rt);
        }
        Ok(())
    }

    /// Full per-mode wiring on top of [`Self::attach`]: `ctx.mode`/`hasUI` (TS: UI present in tui
    /// and non-headless rpc), slash-command registration, a switch listener that refreshes the ext
    /// ctx (id/file/model/name) and emits `session_switch` for `/new`, `/resume` and `/fork`, the
    /// title listener, then the initial ctx push and `session_start`. Call it once the startup
    /// resume (if any) is loaded.
    pub async fn attach_session(self: &Arc<Self>, rt: Arc<Runtime>, mode: ExtMode) {
        self.set_mode(mode);
        rt.set_ext_host(Some(self.host.clone()));
        self.attach(rt.clone());
        crate::rpc::slash::set_extensions(self.clone());
        rt.set_name_listener(Some(self.name_listener()));
        let ext = self.clone();
        let weak = Arc::downgrade(&rt);
        rt.set_switch_listener(Some(Arc::new(move |reason, previous| {
            let (ext, weak) = (ext.clone(), weak.clone());
            tokio::spawn(async move {
                let Some(rt) = weak.upgrade() else { return };
                ext.push_session_ctx(&rt).await;
                // A branch/fork-into-file move emits `session_branch` from the caller (TS).
                if reason == "branch" {
                    return;
                }
                let reason = match reason {
                    "new" => SwitchReason::New,
                    "fork" => SwitchReason::Fork,
                    _ => SwitchReason::Resume,
                };
                let previous = previous.map(|p| p.to_string_lossy().into_owned());
                emit_silent(ext.host.emit_session_switch(reason, previous, None).await);
            });
        })));
        let host = self.host.clone();
        rt.set_switch_gate(Some(Arc::new(move |reason, target| {
            let host = host.clone();
            Box::pin(async move {
                if !host.has_handlers("session_before_switch") {
                    return false;
                }
                let reason = match reason {
                    "new" => SwitchReason::New,
                    "fork" => SwitchReason::Fork,
                    _ => SwitchReason::Resume,
                };
                let target = target.map(|p| p.to_string_lossy().into_owned());
                match host.emit_session_before_switch(reason, target, None).await {
                    Ok(o) => o.cancel,
                    Err(e) => {
                        eprintln!("omt: extension session_before_switch: {e:#}");
                        false
                    }
                }
            })
        })));
        self.push_session_ctx(&rt).await;
        self.session_start().await;
    }

    /// Refresh the live ctx facts from the runtime. `with_branch` also re-reads the session
    /// branch (only when an extension reads it, and only when the journal grew).
    async fn refresh_live(&self, rt: &Runtime, with_branch: bool) {
        let usage = rt.context_usage_value().await;
        let streaming = rt.agent.is_streaming().await;
        {
            let mut snap = self
                .live
                .snap
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            snap.insert("idle".into(), json!(!streaming));
            snap.insert("pending".into(), json!(rt.has_pending_messages()));
            snap.insert("contextUsage".into(), usage.unwrap_or(Value::Null));
            snap.insert("model".into(), rt.model_info());
            snap.insert(
                "asyncJobs".into(),
                crate::tools::session_jobs().map_or(Value::Null, |j| j.async_snapshot(5)),
            );
        }
        if with_branch && self.live.needs_branch.load(Ordering::Relaxed) {
            self.refresh_branch(rt);
        }
    }

    fn refresh_branch(&self, rt: &Runtime) {
        let path = rt.session_file();
        let len = std::fs::metadata(&path).map_or(0, |m| m.len());
        {
            let mut seen = self
                .live
                .branch_len
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if seen.as_ref().is_some_and(|(p, l)| *p == path && *l == len) {
                return;
            }
            *seen = Some((path, len));
        }
        let Ok(entries) = rt.entries() else { return };
        let branch = omp_session::tree::leaf_id(&entries)
            .map(|leaf| {
                omp_session::tree::branch_path(&entries, leaf)
                    .into_iter()
                    .cloned()
                    .collect::<Vec<Value>>()
            })
            .unwrap_or_default();
        *self
            .live
            .branch
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(Value::Array(branch));
    }

    /// Send the runtime's current session id/file/model/name to the host.
    async fn push_session_ctx(&self, rt: &Runtime) {
        // A new session invalidates the cached branch.
        *self
            .live
            .branch_len
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
        self.refresh_live(rt, true).await;
        let model = rt.agent.model().await;
        let model_ref = format!("{}/{}", model.provider, model.id);
        if let Err(e) = self
            .set_session_ctx(
                &rt.session_id(),
                rt.live_session_file().as_deref(),
                &model_ref,
            )
            .await
        {
            eprintln!("omt: extension ctx: {e:#}");
        }
        if let Err(e) = self.set_session_name(rt.session_name()).await {
            eprintln!("omt: extension ctx: {e:#}");
        }
    }

    /// Name-change sink: every title change re-sends the ext ctx with the new
    /// `sessionName` so hosts reading `getSessionName()` see it at once.
    pub fn name_listener(self: &Arc<Self>) -> crate::runtime::NameListener {
        let ext = self.clone();
        Arc::new(move |name: &str| {
            let ext = ext.clone();
            let name = name.to_string();
            tokio::spawn(async move {
                let _ = ext.set_session_name(Some(name)).await;
            });
        })
    }

    /// `rpc::run_with` `OnStart` hook form of [`Self::attach`].
    pub fn rpc_on_start(self: &Arc<Self>) -> crate::rpc::OnStart {
        let ext = self.clone();
        Box::new(move |rt: Arc<Runtime>| {
            Box::pin(async move { ext.attach_session(rt, ExtMode::Rpc).await })
        })
    }

    /// Emit `session_shutdown` and stop the host.
    pub async fn shutdown(&self) {
        emit_silent(self.host.emit_session_shutdown(None).await);
        let deferred = self
            .deferred_agent_end
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        if let Some(messages) = deferred {
            emit_silent(self.host.emit_agent_end(messages, None, None).await);
        }
        if let Err(e) = self.host.shutdown().await {
            eprintln!("omt: extension host shutdown: {e:#}");
        }
    }
}

fn emit_silent(r: Result<omp_ext::EmitResult>) {
    match r {
        Ok(r) => {
            for err in r.errors {
                eprintln!("omt: extension handler error: {err}");
            }
        }
        Err(e) => eprintln!("omt: extension emit: {e:#}"),
    }
}

fn handle_notification(rt: &Arc<Runtime>, msg: HostMessage) {
    let HostMessage::Notification { method, params } = msg else {
        return;
    };
    match method.as_str() {
        "append_entry" => {
            let entry = json!({
                "extension": params.get("extension").cloned().unwrap_or(Value::Null),
                "customType": params.get("customType").cloned().unwrap_or(Value::Null),
                "data": params.get("data").cloned().unwrap_or(Value::Null),
            });
            if let Err(e) = rt.append_entry("custom", entry) {
                eprintln!("omt: extension append_entry: {e:#}");
            }
        }
        "send_message" | "send_user_message" => {
            let text = params
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            if !text.is_empty() {
                rt.extension_message(&text);
            }
        }
        "log" => {
            if let Some(line) = params.get("message").and_then(Value::as_str) {
                eprintln!("ext: {line}");
            }
        }
        "ui_notify" | "tool_update" | "set_label" => {
            rt.send_extension_frame(
                params.get("extension").and_then(Value::as_str),
                &method,
                &params,
            );
        }
        _ => {}
    }
}

impl Extensions {
    /// Forward agent events to host lifecycle handlers. Handler errors are
    /// logged, never fatal to the run.
    async fn subscribe_lifecycle(self: &Arc<Self>, rt: Arc<Runtime>) -> Subscription {
        // `session_stop` runs from the loop's continuation hook: it fires when
        // the agent would otherwise stop, so `decision: "block"` /
        // `continue: true` can hand back a hidden next-turn message and the
        // run continues instead of settling (TS `#emitSessionStopEvent`).
        let hook_host = self.host.clone();
        let hook_stop = self.stop.clone();
        let hook_session_id = rt.session_id();
        let hook_session_file = rt.session_file().to_string_lossy().into_owned();
        let hook_turn = self.turn.clone();
        rt.agent.set_continuation_hook(Some(Arc::new(
            move |ctx: GetContinuationMessagesContext, _signal: AbortSignal| {
                let host = hook_host.clone();
                let stop = hook_stop.clone();
                let session_id = hook_session_id.clone();
                let session_file = hook_session_file.clone();
                let turn = hook_turn.clone();
                Box::pin(async move {
                    Ok(session_stop_continuation(
                        &host,
                        &stop,
                        &session_id,
                        &session_file,
                        &turn,
                        &ctx.new_messages,
                    )
                    .await)
                })
            },
        )));

        let host = self.host.clone();
        let turn = self.turn.clone();
        let stop = self.stop.clone();
        // One encoder per subscription: the agent delivers events to this
        // listener in order, and `forward_event` enqueues each emit while
        // holding the encoder, so encode order == wire order.
        let updates = Arc::new(Mutex::new(ExtUpdateEncoder::default()));
        // TS delivers `tool_execution_end` to extensions after the same call's tool-result
        // `message_start` (its session event path awaits before emitting the end), so the end is
        // held back one event. For a call an extension blocked the start is late too: the whole
        // `message_start` / `tool_execution_start` / `tool_execution_end` burst of an immediate
        // result reaches extensions as message_start, start, end.
        let held: Arc<Mutex<Vec<AgentEvent>>> = Arc::new(Mutex::new(Vec::new()));
        let blocked_calls = self.blocked_calls.clone();
        let deferred_end = self.deferred_agent_end.clone();
        let ext = self.clone();
        let weak = Arc::downgrade(&rt);
        rt.agent
            .subscribe(move |event, _| {
                let (ext, weak) = (ext.clone(), weak.clone());
                let host = host.clone();
                let turn = turn.clone();
                let stop = stop.clone();
                let updates = updates.clone();
                let held = held.clone();
                let blocked_calls = blocked_calls.clone();
                let deferred_end = deferred_end.clone();
                Box::pin(async move {
                    if let Some(rt) = weak.upgrade() {
                        let boundary = matches!(
                            event,
                            AgentEvent::AgentStart | AgentEvent::TurnStart | AgentEvent::AgentEnd { .. }
                        );
                        if boundary
                            || matches!(
                                event,
                                AgentEvent::MessageEnd { .. }
                                    | AgentEvent::TurnEnd { .. }
                                    | AgentEvent::ToolExecutionEnd { .. }
                            )
                        {
                            ext.refresh_live(&rt, boundary).await;
                        }
                    }
                    let mut pending: Vec<AgentEvent> = std::mem::take(
                        &mut *held.lock().unwrap_or_else(std::sync::PoisonError::into_inner),
                    );
                    if let AgentEvent::ToolExecutionEnd { tool_call_id: end_id, .. } = &event {
                        if pending.iter().any(|e| matches!(e, AgentEvent::ToolExecutionStart { tool_call_id, .. } if tool_call_id == end_id)) {
                            pending.push(event);
                            *held.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = pending;
                            return Ok(());
                        }
                    }
                    // A run's input messages (prompt, injected, queued) reach extensions as all
                    // starts first, then all ends: TS lets `message_end` lag `message_start`.
                    let input_role = |m: &AgentMessage| matches!(m.role(), "user" | "custom" | "developer");
                    if !pending.is_empty()
                        && pending.iter().all(|e| matches!(e, AgentEvent::MessageEnd { message } if input_role(message)))
                        && matches!(&event, AgentEvent::MessageStart { message } if input_role(message))
                    {
                        forward_event(&host, &turn, &stop, &updates, &event);
                        *held.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = pending;
                        return Ok(());
                    }
                    if !pending.is_empty() {
                        let owns_result = |id: &str| {
                            pending.iter().any(|e| {
                                matches!(e, AgentEvent::ToolExecutionStart { tool_call_id, .. }
                                    | AgentEvent::ToolExecutionEnd { tool_call_id, .. } if tool_call_id == id)
                            })
                        };
                        let result_start = matches!(
                            &event,
                            AgentEvent::MessageStart { message: AgentMessage::Standard(pa_agent::types::Message::ToolResult(r)) }
                                if owns_result(&r.tool_call_id)
                        );
                        if result_start {
                            forward_event(&host, &turn, &stop, &updates, &event);
                        }
                        for e in &pending {
                            forward_event(&host, &turn, &stop, &updates, e);
                        }
                        if result_start {
                            return Ok(());
                        }
                    }
                    if let AgentEvent::AgentEnd { messages } = &event {
                        if matches!(ext.mode(), ExtMode::Print | ExtMode::Json) {
                            *deferred_end.lock().unwrap_or_else(std::sync::PoisonError::into_inner) =
                                Some(messages.iter().map(message_value).collect());
                            return Ok(());
                        }
                    }
                    let delay = match &event {
                        AgentEvent::ToolExecutionEnd { .. } => true,
                        AgentEvent::MessageEnd { message } => input_role(message),
                        AgentEvent::ToolExecutionStart { tool_call_id, .. } => blocked_calls
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .contains(tool_call_id),
                        _ => false,
                    };
                    if delay {
                        held.lock().unwrap_or_else(std::sync::PoisonError::into_inner).push(event);
                        return Ok(());
                    }
                    forward_event(&host, &turn, &stop, &updates, &event);
                    Ok(())
                })
            })
            .await
    }
}

/// Emit `session_stop` from the would-stop point and translate its outcome
/// into a hidden continuation message (TS `#emitSessionStopEvent` +
/// `#queueHiddenNextTurnMessage`). `decision: "block"` stops the run from
/// settling with the blocked prompt; `continue: true` schedules one more turn
/// carrying `additionalContext`/`reason`.
async fn session_stop_continuation(
    host: &ExtHost,
    stop: &Mutex<StopState>,
    session_id: &str,
    session_file: &str,
    turn: &AtomicU64,
    new_messages: &[AgentMessage],
) -> Vec<AgentMessage> {
    if !host.has_handlers("session_stop") {
        // Same end state as an empty outcome below, without serializing the run's messages.
        *stop
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = StopState::default();
        return Vec::new();
    }
    let (hook_active, continuations) = {
        let s = stop
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        (s.hook_active, s.continuations)
    };
    let event = LifecycleEvent::SessionStop {
        messages: new_messages.iter().map(message_value).collect(),
        turn_id: turn.load(Ordering::SeqCst),
        last_assistant_message: new_messages
            .iter()
            .rev()
            .find(|m| m.role() == "assistant")
            .map(message_value),
        session_id: session_id.to_string(),
        session_file: Some(session_file.to_string()),
        stop_hook_active: hook_active,
    };
    let outcome = match host.emit_session_stop(event, None).await {
        Ok(outcome) => outcome,
        Err(e) => {
            eprintln!("omt: extension session_stop: {e:#}");
            return Vec::new();
        }
    };
    let Some(text) = stop_continuation_text(&outcome) else {
        *stop
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = StopState::default();
        return Vec::new();
    };
    {
        let mut s = stop
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !outcome.block {
            if continuations >= SESSION_STOP_CONTINUATION_CAP {
                eprintln!("omt: session_stop continuation cap reached");
                *s = StopState::default();
                return Vec::new();
            }
            s.continuations += 1;
        }
        s.hook_active = true;
    }
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(0));
    vec![AgentMessage::Custom(CustomAgentMessage {
        role: "developer".to_string(),
        payload: json!({
            "customType": "session-stop-continuation",
            "content": [{"type": "text", "text": text}],
            "display": false,
            "attribution": "agent",
            "timestamp": timestamp,
        }),
    })]
}

/// Queue one notification-only lifecycle event (fire-and-forget, ordered).
///
/// Every event forwarded here is observe-only in the host (`ext-host/host.ts`
/// `emit` falls through to its "everything else: observe" branch: no handler
/// return value is read), so the turn loop need not wait for the handlers.
/// The wire order is the call order and the host runs these emits strictly
/// serially, and ahead of any later awaited `emit`/`call_tool`. `build` runs
/// only when some extension subscribes to `ty`, so unsubscribed events cost no
/// serialization at all.
fn notify(host: &ExtHost, ty: &str, build: impl FnOnce() -> LifecycleEvent) {
    if !host.has_handlers(ty) {
        return;
    }
    if let Err(e) = host.emit_nowait(&build().to_value(), None, emit_silent) {
        eprintln!("omt: extension emit: {e:#}");
    }
}

/// Queue one `message_update` as a delta frame against the message the host
/// rebuilt from the previous frame (`ExtUpdateEncoder`; `ext-host/host.ts`
/// `rebuildMessageUpdate` restores the full event before any handler runs).
fn notify_update(
    host: &ExtHost,
    updates: &Mutex<ExtUpdateEncoder>,
    message: &AgentMessage,
    ame: &pa_agent::stream::AssistantMessageEvent,
) {
    let mut enc = updates
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if !host.has_handlers("message_update") {
        // Nothing reaches the host: its chain (if any) is stale from here on.
        enc.invalidate();
        return;
    }
    enc.set_generation(host.generation());
    let frame = enc.encode(message, ame);
    let chain = ExtUpdateEncoder::chain_id(&frame);
    let resync = enc.resync_flag();
    let sent = host.emit_nowait(&frame, None, move |r| {
        // `{resync: true}`: the host could not apply the frame (it skipped the
        // handlers); a failed emit leaves the host state unknown. Either way
        // the next update of this chain goes out full; a late answer about an
        // older chain leaves a newer one alone.
        let needs_full = match &r {
            Ok(out) => out.result.get("resync").and_then(Value::as_bool) == Some(true),
            Err(_) => true,
        };
        if let (true, Some(chain)) = (needs_full, chain) {
            resync.fetch_max(chain, Ordering::AcqRel);
        }
        emit_silent(r);
    });
    if let Err(e) = sent {
        enc.invalidate();
        eprintln!("omt: extension emit: {e:#}");
    }
}

#[allow(clippy::too_many_lines)] // one arm per `AgentEvent` variant; splitting adds indirection.
fn forward_event(
    host: &ExtHost,
    turn: &AtomicU64,
    stop: &Mutex<StopState>,
    updates: &Mutex<ExtUpdateEncoder>,
    event: &AgentEvent,
) {
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(0));
    match event {
        AgentEvent::AgentStart => {
            // A new run resets the session_stop continuation bookkeeping
            // (TS `#resetSessionStopContinuationState` on generation change).
            *stop
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = StopState::default();
            notify(host, "agent_start", || LifecycleEvent::AgentStart);
        }
        AgentEvent::AgentEnd { messages } => {
            // `session_stop` itself is emitted from the continuation hook
            // (see `subscribe_lifecycle`), where its block/continue outcome can
            // still keep the run alive.
            notify(host, "agent_end", || LifecycleEvent::AgentEnd {
                messages: messages.iter().map(message_value).collect(),
                will_continue: None,
            });
            if host.has_handlers("agent_settled") {
                if let Err(e) =
                    host.emit_nowait(&json!({"type":"agent_settled"}), None, emit_silent)
                {
                    eprintln!("omt: extension agent_settled: {e:#}");
                }
            }
        }
        AgentEvent::TurnStart => {
            let index = turn.fetch_add(1, Ordering::SeqCst) + 1;
            notify(host, "turn_start", || LifecycleEvent::TurnStart {
                turn_index: index,
                timestamp: now_ms,
            });
        }
        AgentEvent::TurnEnd {
            message,
            tool_results,
        } => {
            notify(host, "turn_end", || LifecycleEvent::TurnEnd {
                turn_index: turn.load(Ordering::SeqCst),
                message: message_value(message),
                tool_results: tool_results
                    .iter()
                    .map(serde_json::to_value)
                    .collect::<std::result::Result<Vec<Value>, _>>()
                    .unwrap_or_default(),
            });
        }
        AgentEvent::MessageStart { message } => {
            // Sync point: a new message always starts a fresh delta chain.
            updates
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .message_boundary();
            notify(host, "message_start", || LifecycleEvent::MessageStart {
                message: message_value(message),
            });
        }
        AgentEvent::MessageUpdate {
            message,
            assistant_message_event,
        } => notify_update(host, updates, message, assistant_message_event),
        AgentEvent::MessageEnd { message } => {
            updates
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .message_boundary();
            notify(host, "message_end", || LifecycleEvent::MessageEnd {
                message: message_value(message),
            });
        }
        AgentEvent::ToolExecutionStart {
            tool_call_id,
            tool_name,
            args,
        } => {
            notify(host, "tool_execution_start", || {
                LifecycleEvent::ToolExecutionStart {
                    tool_call_id: tool_call_id.clone(),
                    tool_name: tool_name.clone(),
                    args: args.clone(),
                    intent: None,
                }
            });
        }
        AgentEvent::ToolExecutionUpdate {
            tool_call_id,
            tool_name,
            args,
            partial_result,
        } => {
            notify(host, "tool_execution_update", || {
                LifecycleEvent::ToolExecutionUpdate {
                    tool_call_id: tool_call_id.clone(),
                    tool_name: tool_name.clone(),
                    args: args.clone(),
                    partial_result: serde_json::to_value(partial_result).unwrap_or(Value::Null),
                }
            });
        }
        AgentEvent::ToolExecutionEnd {
            tool_call_id,
            tool_name,
            result,
            is_error,
        } => {
            notify(host, "tool_execution_end", || {
                LifecycleEvent::ToolExecutionEnd {
                    tool_call_id: tool_call_id.clone(),
                    tool_name: tool_name.clone(),
                    result: serde_json::to_value(result).unwrap_or(Value::Null),
                    is_error: *is_error,
                }
            });
        }
    }
}

/// An extension-declared tool proxied to the host's `call_tool`.
struct ExtTool {
    host: Arc<ExtHost>,
    info: ToolInfo,
}

impl AgentTool for ExtTool {
    fn name(&self) -> &str {
        &self.info.name
    }
    fn description(&self) -> &str {
        self.info.description.as_deref().unwrap_or("")
    }
    fn parameters(&self) -> &Value {
        &self.info.parameters
    }
    fn label(&self) -> &str {
        self.info.label.as_deref().unwrap_or(&self.info.name)
    }
    fn execute(
        self: Arc<Self>,
        tool_call_id: String,
        params: Value,
        signal: AbortSignal,
        _on_update: pa_agent::types::AgentToolUpdateCallback,
    ) -> pa_agent::BoxFut<'static, Result<AgentToolResult>> {
        Box::pin(async move {
            let host = self.host.clone();
            let call = host.call_tool(&self.info.name, &tool_call_id, params, None);
            let value = tokio::select! {
                biased;
                // A caller abort must not leave the handler running in the
                // host: cancel it there too (`cancel_tool`), like an
                // AbortSignal reaching a TS extension tool.
                () = signal.aborted() => {
                    let _ = host.cancel_tool(&tool_call_id).await;
                    bail!("extension tool '{}' cancelled", self.info.name);
                }
                res = tokio::time::timeout(tool_timeout(), call) => res
                    .map_err(|_| anyhow::anyhow!("extension tool '{}' timed out", self.info.name))??,
            };
            let content: Vec<ToolResultContent> = value
                .get("content")
                .cloned()
                .map(serde_json::from_value)
                .transpose()?
                .unwrap_or_default();
            let is_error = value
                .get("isError")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let details = value.get("details").cloned().unwrap_or(Value::Null);
            if is_error {
                let text = content
                    .iter()
                    .filter_map(|c| match c {
                        ToolResultContent::Text(t) => Some(t.text.as_str()),
                        ToolResultContent::Image(_) => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                return Err(anyhow::anyhow!(if text.is_empty() {
                    format!("extension tool '{}' returned an error", self.info.name)
                } else {
                    text
                }));
            }
            Ok(AgentToolResult {
                content,
                details,
                terminate: None,
                is_error: None,
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pa_agent::abort::AbortController;

    async fn sh_host(
        dir: &std::path::Path,
        body: &str,
        env: Vec<(String, String)>,
    ) -> Arc<ExtHost> {
        let script = dir.join("fake-host.sh");
        std::fs::write(&script, body).unwrap();
        ExtHost::spawn(&HostConfig {
            bun: "/bin/sh".into(),
            host_script: script,
            trace: None,
            handler_timeout_ms: None,
            env,
            shared_socket: None,
        })
        .await
        .unwrap()
    }

    /// Answers every request with a `session_stop` outcome of
    /// `continue: true, additionalContext: "keep going"`.
    const CONTINUE_HOST: &str = concat!(
        "while IFS= read -r line; do\n",
        "  id=$(printf '%s' \"$line\" | sed -n 's/.*\"id\":\\([0-9][0-9]*\\).*/\\1/p')\n",
        "  if [ -n \"$id\" ]; then\n",
        "    printf '{\"jsonrpc\":\"2.0\",\"id\":%s,\"result\":{\"result\":{\"continue\":true,\"additionalContext\":\"keep going\"},\"handlers\":[],\"errors\":[]}}\n' \"$id\"\n",
        "  fi\n",
        "done\n",
    );

    #[test]
    fn stop_continuation_text_matches_ts_rules() {
        use omp_ext::lifecycle::SessionStopOutcome;
        let block_default = SessionStopOutcome {
            block: true,
            ..Default::default()
        };
        assert_eq!(
            stop_continuation_text(&block_default).as_deref(),
            Some(SESSION_STOP_BLOCKED_PROMPT)
        );
        let block_reason = SessionStopOutcome {
            block: true,
            continue_turn: false,
            reason: Some("not done".into()),
            additional_context: Some("ctx".into()),
        };
        assert_eq!(
            stop_continuation_text(&block_reason).as_deref(),
            Some("not done")
        );
        let cont = SessionStopOutcome {
            block: false,
            continue_turn: true,
            additional_context: Some("ctx".into()),
            reason: Some("why".into()),
        };
        assert_eq!(stop_continuation_text(&cont).as_deref(), Some("ctx"));
        // `continue: true` without context schedules nothing (runner.ts).
        let bare = SessionStopOutcome {
            block: false,
            continue_turn: true,
            additional_context: None,
            reason: None,
        };
        assert_eq!(stop_continuation_text(&bare), None);
        let none = SessionStopOutcome::default();
        assert_eq!(stop_continuation_text(&none), None);
    }

    /// `continue: true` schedules exactly one hidden continuation turn and
    /// flips `stop_hook_active` for the next emit (`ReviewWave3`).
    #[tokio::test]
    async fn session_stop_continue_schedules_hidden_turn() {
        let dir = tempfile::tempdir().unwrap();
        let host = sh_host(dir.path(), CONTINUE_HOST, Vec::new()).await;
        let stop = Mutex::new(StopState::default());
        let turn = AtomicU64::new(2);
        let messages = session_stop_continuation(&host, &stop, "s1", "/s.jsonl", &turn, &[]).await;
        assert_eq!(messages.len(), 1);
        let AgentMessage::Custom(custom) = &messages[0] else {
            panic!("continuation must be a custom message");
        };
        assert_eq!(custom.role, "developer", "convert_to_llm surfaces it");
        assert_eq!(custom.payload["customType"], "session-stop-continuation");
        assert_eq!(
            custom.payload["content"][0]["text"].as_str(),
            Some("keep going")
        );
        assert!(custom.payload["display"].as_bool() == Some(false));
        {
            let st = stop.lock().unwrap();
            assert!(st.hook_active && st.continuations == 1);
        }
        host.shutdown().await.unwrap();
    }

    /// The continuation cap (8) applies to `continue` outcomes; block is exempt.
    #[tokio::test]
    async fn session_stop_continuation_cap_resets_state() {
        let dir = tempfile::tempdir().unwrap();
        let host = sh_host(dir.path(), CONTINUE_HOST, Vec::new()).await;
        let stop = Mutex::new(StopState::default());
        let turn = AtomicU64::new(0);
        for i in 0..SESSION_STOP_CONTINUATION_CAP {
            let messages =
                session_stop_continuation(&host, &stop, "s1", "/s.jsonl", &turn, &[]).await;
            assert_eq!(messages.len(), 1, "continuation {i} scheduled");
        }
        let messages = session_stop_continuation(&host, &stop, "s1", "/s.jsonl", &turn, &[]).await;
        assert!(messages.is_empty(), "cap reached: no further continuation");
        assert_eq!(
            *stop.lock().unwrap(),
            StopState::default(),
            "cap resets the bookkeeping"
        );
        host.shutdown().await.unwrap();
    }

    /// An already-aborted signal cancels the host-side handler via
    /// `cancel_tool` instead of leaving it running (`ReviewWave3`).
    #[tokio::test]
    async fn aborted_signal_cancels_host_tool_call() {
        let dir = tempfile::tempdir().unwrap();
        let marker = dir.path().join("cancelled");
        let host = sh_host(
            dir.path(),
            concat!(
                "while IFS= read -r line; do\n",
                "  case \"$line\" in\n",
                "    *'\"method\":\"cancel_tool\"'*)\n",
                "      touch \"$MARKER\"\n",
                "      id=$(printf '%s' \"$line\" | sed -n 's/.*\"id\":\\([0-9][0-9]*\\).*/\\1/p')\n",
                "      printf '{\"jsonrpc\":\"2.0\",\"id\":%s,\"result\":true}\n' \"$id\";;\n",
                "  esac\n",
                "done\n",
            ),
            vec![("MARKER".into(), marker.display().to_string())],
        )
        .await;
        let tool = Arc::new(ExtTool {
            host: host.clone(),
            info: ToolInfo {
                name: "slow".into(),
                label: None,
                description: None,
                parameters: json!({}),
                extension: "test".into(),
            },
        });
        let controller = AbortController::new();
        let signal = controller.signal();
        controller.abort();
        let result = tool
            .execute(
                "call-1".into(),
                json!({}),
                signal,
                Arc::new(|_| {}) as pa_agent::types::AgentToolUpdateCallback,
            )
            .await;
        let err = result.expect_err("aborted call must fail");
        assert!(err.to_string().contains("cancelled"), "{err:#}");
        assert!(marker.exists(), "host never received cancel_tool");
        host.shutdown().await.unwrap();
    }

    #[test]
    fn injected_message_follows_ts_custom_message_normalization() {
        // Object payload: role custom, explicit fields kept, attribution defaults to "user".
        let m = injected_message(
            &json!({"customType": "memory-recall", "content": "hi", "display": false}),
        )
        .expect("object payload");
        let AgentMessage::Custom(c) = m else {
            panic!("custom message")
        };
        assert_eq!(c.role, "custom");
        assert_eq!(c.payload["customType"], "memory-recall");
        assert_eq!(c.payload["content"], "hi");
        assert_eq!(c.payload["display"], false);
        assert_eq!(c.payload["attribution"], "user");
        // Explicit attribution wins; array content stays an array.
        let m = injected_message(&json!({"customType": "x", "content": [{"type": "text", "text": "t"}], "attribution": "agent"}))
            .expect("explicit attribution");
        let AgentMessage::Custom(c) = m else {
            panic!("custom message")
        };
        assert_eq!(c.payload["attribution"], "agent");
        assert!(c.payload["content"].is_array());
        // A bare string becomes a displayed default-type message; other shapes are dropped.
        assert!(injected_message(&json!("note")).is_some());
        assert!(injected_message(&json!(7)).is_none());
    }

    #[test]
    fn extension_mode_matches_ts_has_ui() {
        // TS `hasUI()` is true exactly when a UI context was provided: tui and non-headless rpc.
        assert!(ExtMode::Tui.has_ui() && ExtMode::Rpc.has_ui());
        assert!(!ExtMode::Print.has_ui() && !ExtMode::Json.has_ui());
        assert_eq!(ExtMode::Tui.name(), "tui");
    }
}
