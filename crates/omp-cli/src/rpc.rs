//! `--mode rpc`: omp's JSON-lines RPC protocol (`modes/rpc`) on stdin/stdout.
//!
//! First frame `ready`; every command is answered with a `response` frame
//! (`{id?, type, command, success, data?|error?, code?}`); agent events,
//! `prompt_result` and `session_settled` frames are interleaved. Commands that
//! depend on subsystems omp-rs lacks answer `success: false` with an omp-style
//! error message (see `unsupported`).
//!
//! Commands are handled one at a time in arrival order on a worker task; the read loop keeps
//! routing `extension_ui_response` frames (and `abort`, while a command is in flight) so a
//! command awaiting a dialog answer cannot deadlock the transport.
//!
//! Interactive slash commands (`/branch`, `/copy`, `/agents`, `/modelpreset`, …) drive TS-shaped
//! `extension_ui_request` dialogs (`select` / `confirm` / `input` / `editor`, withdrawn by a
//! `cancelDialog` side-effect frame) unless the process runs with `--no-ui`; see
//! `slash_extra::dialog`. Host effects without an answer:
//! - `{"type":"extension_ui_request","id":…,"method":"set_editor_text","text":…}` — replace the
//!   client's prompt draft (TS rpc `set_editor_text`).
//! - `{"type":"clipboard_copy","text":…}` — `/copy` picked `text`; the client puts it on its
//!   clipboard (omp-rs extension; the TUI host copies itself).

use parking_lot::Mutex;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::sync::LazyLock;

use anyhow::{Context as _, Result};
use pa_agent::abort::AbortController;
use pa_agent::agent::{Agent, QueueMode};
use pa_agent::types::ThinkingLevel;
use pa_agent::BoxFut;
use serde_json::{json, Map, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixListener;
use tokio::sync::mpsc;

use crate::agent::Target;
use crate::paging;
use crate::persist::Persistence;
use crate::runtime::{
    available_models, level_name, model_info, newest_session, parse_level, unsupported, CmdError,
    CmdResult, Runtime,
};
use crate::tools::bash as shell;
use crate::tools::{AskUi, SelectOption};

#[path = "slash.rs"]
pub mod slash;

const MAX_FRAME_BYTES: u64 = 1_048_576;

fn lock<T>(m: &Mutex<T>) -> parking_lot::MutexGuard<'_, T> {
    m.lock()
}

/// Resolve a pending ask-UI dialog (`RpcAskUi` waiter) by frame id; no-op
/// when no waiter is registered. Also called from the interactive TUI's
/// bridge (`tui.rs`), which never runs the RPC command loop.
pub fn resolve_pending_ui_request(id: &str, frame: Value) {
    let tx = lock(&PENDING_UI_REQUESTS).remove(id);
    if let Some(tx) = tx {
        let _ = tx.send(frame);
    }
}

/// Register a waiter for the `extension_ui_response` that answers frame `id` (interactive
/// `/login` and `/logout` dialogs; resolved by [`resolve_pending_ui_request`]).
pub fn register_pending_ui_request(id: &str) -> tokio::sync::oneshot::Receiver<Value> {
    let (tx, rx) = tokio::sync::oneshot::channel();
    lock(&PENDING_UI_REQUESTS).insert(id.to_string(), tx);
    rx
}

/// Drop the waiter for frame `id` (its dialog was withdrawn unanswered).
pub fn forget_pending_ui_request(id: &str) {
    lock(&PENDING_UI_REQUESTS).remove(id);
}

/// Pending RPC ask-UI dialogs (`extension_ui_request` id → waiter).
static PENDING_UI_REQUESTS: LazyLock<Mutex<HashMap<String, tokio::sync::oneshot::Sender<Value>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Mirrors the `set_ask_dialog` RPC command (TS `askDialogEnabled`).
static ASK_DIALOG_ENABLED: AtomicBool = AtomicBool::new(false);

/// RPC transport for the `ask` tool dialog (TS `RpcExtensionUIContext`):
/// emits `extension_ui_request` frames and resolves from the matching
/// `extension_ui_response` (routed by id in the run loop).
struct RpcAskUi;

impl RpcAskUi {
    /// Send one dialog frame and await its response. `Ok(None)` = the dialog
    /// was cancelled/timed out (TS resolves `undefined`).
    async fn request(method: &str, mut frame: Map<String, Value>) -> Result<Option<Value>> {
        let (tx, rx) = tokio::sync::oneshot::channel::<Value>();
        let id = format!(
            "ui-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos())
        );
        let timeout_ms = frame.get("timeout").and_then(Value::as_u64);
        frame.insert("id".into(), json!(id));
        frame.insert("type".into(), json!("extension_ui_request"));
        frame.insert("method".into(), json!(method));
        lock(&PENDING_UI_REQUESTS).insert(id.clone(), tx);
        // Emit through the runtime's frame channel via a task-local runtime;
        // `RpcAskUi` is only constructed inside the RPC loop.
        RUNTIME_TX.with(|tx| {
            if let Some(tx) = &*tx.borrow() {
                tx.send(Value::Object(frame)).ok();
            }
        });
        let wait = async {
            match rx.await {
                Ok(response) => Ok(Some(response)),
                Err(_) => Err(anyhow::anyhow!("ask dialog dropped")),
            }
        };
        let result = match timeout_ms {
            Some(ms) => {
                match tokio::time::timeout(std::time::Duration::from_millis(ms), wait).await {
                    Ok(r) => r,
                    Err(_) => Ok(Some(json!({"cancelled": true, "timedOut": true}))),
                }
            }
            None => wait.await,
        };
        lock(&PENDING_UI_REQUESTS).remove(&id);
        result
    }
}

/// Internal end-of-stream frame: the stdout writer stops after the frames queued before it.
const FLUSH_MARKER: &str = "__omt_flush";

/// `OMP_RPC_SOCKET`, taken out of the environment at startup (see
/// [`take_detached_socket_env`]).
static DETACHED_SOCKET: std::sync::OnceLock<Option<std::ffi::OsString>> =
    std::sync::OnceLock::new();

/// Move `OMP_RPC_SOCKET` (set by `omt web` for the agent it spawns) out of the
/// environment before any thread starts. Left in place it is inherited by
/// every subprocess — a `task` child `omt --mode rpc` would rebind the
/// parent's control socket path and swallow the frames meant for the parent.
pub fn take_detached_socket_env() {
    let value = std::env::var_os("OMP_RPC_SOCKET");
    if value.is_some() {
        std::env::remove_var("OMP_RPC_SOCKET");
    }
    let _ = DETACHED_SOCKET.set(value);
}

thread_local! {
    /// Frame channel of the active RPC loop (set in `run_with`).
    static RUNTIME_TX: std::cell::RefCell<Option<mpsc::UnboundedSender<Value>>> =
        const { std::cell::RefCell::new(None) };
}

impl AskUi for RpcAskUi {
    fn has_ask_dialog(&self) -> bool {
        ASK_DIALOG_ENABLED.load(Ordering::SeqCst)
    }

    fn ask(
        &self,
        questions: Value,
        timeout_ms: Option<u64>,
    ) -> BoxFut<'static, anyhow::Result<Option<Value>>> {
        Box::pin(async move {
            let mut frame = Map::new();
            frame.insert("questions".into(), questions);
            if let Some(ms) = timeout_ms {
                frame.insert("timeout".into(), json!(ms));
            }
            // Cancelled responses are passed through: the tool interprets
            // `cancelled`/`timedOut` (TS `parseAskDialogResponse`).
            Self::request("ask", frame).await
        })
    }

    fn select(
        &self,
        prompt: String,
        options: Vec<SelectOption>,
        _multi: bool,
        _checked: Vec<usize>,
        _initial_index: Option<usize>,
    ) -> BoxFut<'static, anyhow::Result<Option<String>>> {
        Box::pin(async move {
            let labels: Vec<String> = options.iter().map(|o| o.label.clone()).collect();
            let mut frame = Map::new();
            frame.insert("title".into(), json!(prompt));
            frame.insert("options".into(), json!(labels));
            let details: Vec<Value> = options
                .iter()
                .map(|o| match &o.description {
                    Some(d) if !d.trim().is_empty() => json!({"description": d}),
                    _ => json!({}),
                })
                .collect();
            if details
                .iter()
                .any(|d| !d.as_object().is_none_or(Map::is_empty))
            {
                frame.insert("optionDetails".into(), json!(details));
            }
            Ok(Self::request("select", frame)
                .await?
                .and_then(|v| v.get("value").cloned())
                .and_then(|v| v.as_str().map(str::to_string)))
        })
    }
}

const MAX_REASSEMBLED_BYTES: u64 = 67_108_864;
const CACHE_WARMING_MODES: [&str; 3] = ["off", "streaming", "idle"];

/// Success/failure response frame in omp's key order.
fn response(id: Option<&Value>, command: &str, result: CmdResult) -> Value {
    let mut m = Map::new();
    if let Some(id) = id {
        m.insert("id".into(), id.clone());
    }
    m.insert("type".into(), json!("response"));
    m.insert("command".into(), json!(command));
    match result {
        Ok(data) => {
            m.insert("success".into(), json!(true));
            if let Some(d) = data {
                m.insert("data".into(), d);
            }
        }
        Err(e) => {
            m.insert("success".into(), json!(false));
            m.insert("error".into(), json!(e.message));
            if let Some(code) = e.code {
                m.insert("code".into(), json!(code));
            }
        }
    }
    Value::Object(m)
}

fn str_field<'a>(cmd: &'a Value, key: &str) -> Result<&'a str, CmdError> {
    cmd.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| CmdError::new(format!("{key} must be a string")))
}

fn bool_field(cmd: &Value, key: &str) -> Result<bool, CmdError> {
    cmd.get(key)
        .and_then(Value::as_bool)
        .ok_or_else(|| CmdError::new(format!("{key} must be a boolean")))
}

fn queue_mode_field(cmd: &Value) -> Result<QueueMode, CmdError> {
    match cmd.get("mode").and_then(Value::as_str) {
        Some("all") => Ok(QueueMode::All),
        Some("one-at-a-time") => Ok(QueueMode::OneAtATime),
        _ => Err(CmdError::new("mode must be \"all\" or \"one-at-a-time\"")),
    }
}

use omp_providers::login::providers::PROVIDERS as OAUTH_PROVIDERS;

/// `goal` with no goal runtime behind it: the answers omp gives while no goal exists.
fn goal(cmd: &Value) -> CmdResult {
    let empty = || ok(json!({"goal": null, "state": null}));
    match cmd.get("op").and_then(Value::as_str) {
        Some("get" | "pause" | "drop") => empty(),
        Some("create") => {
            if cmd
                .get("objective")
                .and_then(Value::as_str)
                .is_none_or(|o| o.trim().is_empty())
            {
                return Err(CmdError::new("objective is required when op=create"));
            }
            if let Some(b) = cmd.get("token_budget") {
                if !b.as_f64().is_some_and(|n| n.fract() == 0.0 && n > 0.0) {
                    return Err(CmdError::new(
                        "token_budget must be a positive integer when provided",
                    ));
                }
            }
            Err(unsupported("Goal mode"))
        }
        Some("resume") => Err(CmdError::new("No paused goal to resume.")),
        other => Err(CmdError::new(format!(
            "Unknown goal op: {}",
            other.map_or_else(|| "undefined".to_string(), str::to_string)
        ))),
    }
}

/// `set_host_tools`: omp's validation, then the no-bridge answer (an empty set is a plain success).
fn set_host_tools(cmd: &Value) -> CmdResult {
    let tools = cmd
        .get("tools")
        .and_then(Value::as_array)
        .ok_or_else(|| CmdError::new("tools must be an array"))?;
    for (index, tool) in tools.iter().enumerate() {
        let name = tool
            .get("name")
            .and_then(Value::as_str)
            .map_or("", str::trim);
        if name.is_empty() {
            return Err(CmdError::new(format!(
                "Host tool at index {index} must provide a non-empty name"
            )));
        }
        if tool
            .get("description")
            .and_then(Value::as_str)
            .is_none_or(|d| d.trim().is_empty())
        {
            return Err(CmdError::new(format!(
                "Host tool \"{name}\" must provide a non-empty description"
            )));
        }
        if !tool.get("parameters").is_some_and(Value::is_object) {
            return Err(CmdError::new(format!(
                "Host tool \"{name}\" must provide a JSON Schema object"
            )));
        }
    }
    if tools.is_empty() {
        return ok(json!({"toolNames": []}));
    }
    Err(unsupported("Host tools"))
}

/// `set_host_uri_schemes`: omp's scheme validation, then the no-bridge answer.
fn set_host_uri_schemes(cmd: &Value) -> CmdResult {
    let schemes = cmd
        .get("schemes")
        .and_then(Value::as_array)
        .ok_or_else(|| CmdError::new("schemes must be an array"))?;
    for raw in schemes {
        let scheme = raw
            .get("scheme")
            .and_then(Value::as_str)
            .map_or_else(String::new, |s| s.trim().to_lowercase());
        if scheme.is_empty() {
            return Err(CmdError::new("Host URI scheme must be a non-empty string"));
        }
        let mut chars = scheme.chars();
        let valid = chars.next().is_some_and(|c| c.is_ascii_lowercase())
            && chars.all(|c| {
                c.is_ascii_lowercase() || c.is_ascii_digit() || matches!(c, '+' | '.' | '-')
            });
        if !valid {
            let shown = raw.get("scheme").and_then(Value::as_str).unwrap_or("");
            return Err(CmdError::new(format!(
                "Host URI scheme contains invalid characters: {shown}"
            )));
        }
    }
    if schemes.is_empty() {
        return ok(json!({"schemes": []}));
    }
    Err(unsupported("Host URI schemes"))
}

fn subagent_id<'a>(cmd: &'a Value, _command: &str) -> Result<&'a str, CmdError> {
    cmd.get("subagentId")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| CmdError::new("`subagentId` must be a non-empty string."))
}

/// `get_subagent_messages` against an always-empty subagent roster.
fn subagent_messages(cmd: &Value) -> CmdResult {
    if let Some(from) = cmd.get("fromByte") {
        if !from.as_f64().is_some_and(f64::is_finite) {
            return Err(CmdError::new("fromByte must be a finite number"));
        }
    }
    if let Some(id) = cmd
        .get("subagentId")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
    {
        return Err(CmdError::new(format!(
            "Unknown subagent or session file unavailable: {id}"
        )));
    }
    if cmd
        .get("sessionFile")
        .and_then(Value::as_str)
        .is_some_and(|s| !s.is_empty())
    {
        return Err(CmdError::new("Unknown subagent session file"));
    }
    Err(CmdError::new(
        "get_subagent_messages requires subagentId or sessionFile",
    ))
}

/// `predict_word*` validation: `text` string, `cursor` an integer UTF-16 offset within it.
fn text_cursor(cmd: &Value) -> Result<(), CmdError> {
    let text = cmd.get("text").and_then(Value::as_str);
    let cursor = cmd.get("cursor").and_then(Value::as_f64);
    if let (Some(t), Some(c)) = (text, cursor) {
        let utf16_len = f64::from(u32::try_from(t.encode_utf16().count()).unwrap_or(u32::MAX));
        if c.fract() == 0.0 && c >= 0.0 && c <= utf16_len {
            return Ok(());
        }
    }
    Err(CmdError::new(
        "cursor must be an integer UTF-16 offset within text",
    ))
}

/// `get_login_providers`: omp's OAuth registry; `authenticated` = a credential source resolves.
async fn login_providers() -> Vec<Value> {
    let mut out = Vec::with_capacity(OAUTH_PROVIDERS.len());
    for (id, name) in OAUTH_PROVIDERS {
        // `!command` keys shell out: resolve on the blocking pool, not the executor.
        let from_models = match pa_ai::models_generated::get_models(id).first() {
            Some(m) => omp_providers::resolve_async(format!("{id}/{}", m.id))
                .await
                .is_ok_and(|r| r.api_key.is_some()),
            None => false,
        };
        let authenticated = from_models || pa_ai::env_api_keys::get_env_api_key(id).is_some();
        out.push(
            json!({"id": id, "name": name, "available": true, "authenticated": authenticated}),
        );
    }
    out
}

/// `images` of a prompt command: `[{type: "image", data, mimeType}]`.
fn images(cmd: &Value) -> Vec<pa_agent::types::ImageContent> {
    cmd.get("images")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|i| {
            Some(pa_agent::types::ImageContent {
                data: i.get("data")?.as_str()?.to_string(),
                mime_type: i.get("mimeType")?.as_str()?.to_string(),
            })
        })
        .collect()
}

// Result shape mirrors the TS command framing (`ok(...)` / `none()` feed `?`-chains).
#[allow(clippy::unnecessary_wraps)]
fn ok(data: Value) -> CmdResult {
    Ok(Some(data))
}

#[allow(clippy::unnecessary_wraps)]
fn none() -> CmdResult {
    Ok(None)
}

/// Conservative profile for agents that predate `get_capabilities` (they reject
/// it as an unknown command): the web frontend must assume these bounds
/// instead of requiring an agent restart. The legacy page limit is deliberately
/// below the current [`crate::paging::MAX_LIMIT`] so requests stay valid on
/// both old and new agents.
pub const LEGACY_MAX_GET_MESSAGES_PAGE_SIZE: usize = 256;

/// Versioned capability payload. `maxPageSize` is injected so tests (and the
/// legacy profile) can pin it independently of the live paging limit.
fn capabilities_payload(max_get_messages_page_size: usize) -> Value {
    json!({
        "protocolVersion": 1,
        "supportedProtocolVersions": [1, 2],
        "maxFrameBytes": MAX_FRAME_BYTES,
        "maxReassembledFrameBytes": MAX_REASSEMBLED_BYTES,
        "limits": {
            "getMessages": {
                "defaultPageSize": crate::paging::DEFAULT_LIMIT,
                "maxPageSize": max_get_messages_page_size,
                "maxPageBytes": crate::paging::MAX_PAGE_BYTES,
                "maxCursorChars": crate::paging::MAX_CURSOR_CHARS,
            },
        },
        "features": {
            "messagesBackwardPaging": true,
        }
    })
}

/// Payload a frontend should assume when `get_capabilities` fails (older
/// detached agent that answers "Unknown command").
pub fn legacy_capabilities_payload() -> Value {
    let mut capabilities = capabilities_payload(LEGACY_MAX_GET_MESSAGES_PAGE_SIZE);
    capabilities["supportedProtocolVersions"] = json!([1]);
    capabilities["features"]["messagesBackwardPaging"] = json!(false);
    capabilities
}

async fn cycle_model(rt: &Runtime) -> CmdResult {
    let info = rt.model_info();
    let models = available_models(None).await;
    if models.len() < 2 {
        return ok(Value::Null);
    }
    let idx = models
        .iter()
        .position(|m| {
            info.get("provider") == Some(&json!(m.provider.clone()))
                && info.get("id") == Some(&json!(m.id))
        })
        .unwrap_or(0);
    let next = &models[(idx + 1) % models.len()];
    let model = rt.cycle_to(&next.provider, &next.id).await?;
    let level = rt.agent.thinking_level().await;
    let ultra = rt.ultra().map(omp_tui::magic::Ultra::name);
    ok(
        json!({"model": model, "thinkingLevel": level_name(level), "ultra": ultra, "isScoped": false}),
    )
}

fn branch_messages(entries: &[Value]) -> Vec<Value> {
    entries
        .iter()
        .filter(|e| e.get("type").and_then(Value::as_str) == Some("message"))
        .filter(|e| e.pointer("/message/role").and_then(Value::as_str) == Some("user"))
        .filter_map(|e| {
            let id = e.get("id")?.as_str()?;
            let content = e.pointer("/message/content")?;
            let text = match content {
                Value::String(s) => s.clone(),
                Value::Array(parts) => parts
                    .iter()
                    .filter_map(|p| {
                        (p.get("type")? == "text")
                            .then(|| p.get("text")?.as_str().map(str::to_string))?
                    })
                    .collect::<Vec<_>>()
                    .join("\n"),
                _ => return None,
            };
            Some(json!({"entryId": id, "text": text}))
        })
        .collect()
}

fn tree(entries: &[Value]) -> Vec<Value> {
    fn build(entries: &[Value], parent: Option<&str>, known: &HashSet<&str>) -> Vec<Value> {
        entries
            .iter()
            .filter(|e| {
                let p = e.get("parentId").and_then(Value::as_str);
                match parent {
                    Some(id) => p == Some(id),
                    None => p.is_none_or(|p| !known.contains(p)),
                }
            })
            .map(|e| {
                // An id-less entry (e.g. the `title` header) has no children; recursing with
                // `parent: None` would rebuild the roots forever and overflow the stack.
                let children = e
                    .get("id")
                    .and_then(Value::as_str)
                    .map_or_else(Vec::new, |id| build(entries, Some(id), known));
                json!({"entry": e, "children": children})
            })
            .collect()
    }
    let known: HashSet<&str> = entries
        .iter()
        .filter_map(|e| e.get("id").and_then(Value::as_str))
        .collect();
    build(entries, None, &known)
}

async fn run_bash(rt: &Arc<Runtime>, command: &str) -> CmdResult {
    // TS `executeBash`: a `user_bash` handler's result replaces the execution.
    if let Some(host) = rt.ext_host().filter(|h| h.has_handlers("user_bash")) {
        match host
            .emit_user_bash(
                command.to_owned(),
                false,
                rt.cwd.display().to_string(),
                None,
            )
            .await
        {
            Ok(outcome) => {
                if let Some(result) = outcome.result {
                    return ok(result);
                }
            }
            Err(e) => eprintln!("omt: extension user_bash: {e:#}"),
        }
    }
    let controller = AbortController::new();
    let signal = controller.signal();
    rt.set(|s| s.running_bash(Some(controller)));
    let result = shell::execute(&rt.cwd, command, &signal).await;
    rt.set(|s| s.running_bash(None));
    let r = result.map_err(CmdError::from)?;
    let mut v = json!({
        "output": r.output,
        "cancelled": r.cancelled,
        "truncated": r.truncated,
        "totalLines": r.total_lines,
        "totalBytes": r.total_bytes,
        "outputLines": r.output_lines,
        "outputBytes": r.output_bytes,
        "workingDir": rt.cwd.display().to_string(),
    });
    if let Some(code) = r.exit_code {
        v["exitCode"] = json!(code);
    }
    if r.timed_out {
        v["timedOut"] = json!(true);
    }
    ok(v)
}

/// Handle one command (everything but `bash`, which runs in the background).
#[allow(clippy::too_many_lines)]
pub async fn handle(rt: &Arc<Runtime>, cmd: &Value) -> Value {
    let id = cmd.get("id");
    let kind = cmd.get("type").and_then(Value::as_str).unwrap_or("");
    let result = dispatch(rt, cmd, kind).await;
    response(id, kind, result)
}

#[allow(clippy::too_many_lines)]
// The `abort_bash` closure is required: the method-ref form `SettingsView::abort_bash`
// fails the HRTB lifetime check (`&mut SettingsView<'_>`), so clippy's replacement
// does not compile.
#[allow(clippy::redundant_closure, clippy::redundant_closure_for_method_calls)]
/// TS rpc `emitInput(..., "rpc")`: `None` when an extension consumed the input, else the
/// (possibly rewritten) text. No extension host / host failure passes the text through.
async fn input_gate(text: &str) -> Option<String> {
    let Some(ext) = slash::extensions() else {
        return Some(text.to_string());
    };
    match ext
        .input_gate(text, omp_ext::lifecycle::InputSource::Rpc)
        .await
    {
        Ok(crate::ext::PromptGate::Run(t)) => Some(t),
        Ok(crate::ext::PromptGate::Handled) => None,
        Err(e) => {
            eprintln!("omt: extension input: {e:#}");
            Some(text.to_string())
        }
    }
}

#[allow(clippy::too_many_lines)] // one arm per RPC command; a flat table keeps the protocol readable
async fn dispatch(rt: &Arc<Runtime>, cmd: &Value, kind: &str) -> CmdResult {
    match kind {
        "negotiate_protocol" => match cmd.get("protocolVersion").and_then(Value::as_i64) {
            Some(2) => ok(json!({"protocolVersion": 2})),
            other => Err(CmdError::new(format!(
                "Unsupported RPC protocol version: {}",
                other.map_or_else(|| "undefined".to_string(), |v| v.to_string())
            ))),
        },
        "prompt" => {
            let message = str_field(cmd, "message")?;
            // TS rpc ordered-input dispatch: `input` handlers run before the prompt/slash handling.
            let gated = input_gate(message).await;
            let outcome = match &gated {
                Some(text) => slash::try_dispatch(rt, text).await,
                None => slash::Outcome::ExtensionHandled,
            };
            let message = gated.as_deref().unwrap_or(message);
            match outcome {
                slash::Outcome::Handled { agent_invoked } => {
                    ok(json!({"agentInvoked": agent_invoked}))
                }
                slash::Outcome::ExtensionHandled => {
                    // TS: the extension consumed the prompt inside session.prompt;
                    // the response carries no data and a completed prompt_result
                    // follows the response frame.
                    let task_rt = Arc::clone(rt);
                    let id = cmd.get("id").cloned();
                    tokio::spawn(async move {
                        tokio::task::yield_now().await;
                        let mut f = json!({"type":"prompt_result","agentInvoked":false,
                            "status":"completed","sessionSettled":true});
                        if let Some(id) = id {
                            f["id"] = id;
                        }
                        task_rt.send(f);
                    });
                    none()
                }
                slash::Outcome::Prompt(rewritten) => {
                    rt.prompt(
                        cmd.get("id").cloned(),
                        &rewritten,
                        cmd.get("streamingBehavior").and_then(Value::as_str),
                        images(cmd),
                    )
                    .await?;
                    none()
                }
                slash::Outcome::NotSlash => {
                    rt.prompt(
                        cmd.get("id").cloned(),
                        message,
                        cmd.get("streamingBehavior").and_then(Value::as_str),
                        images(cmd),
                    )
                    .await?;
                    none()
                }
            }
        }
        "steer" | "follow_up" => {
            let message = str_field(cmd, "message")?;
            if let Some(message) = input_gate(message).await {
                if kind == "steer" {
                    rt.steer(&message);
                } else {
                    rt.follow_up(&message);
                }
            }
            none()
        }
        "remove_queued_message" => {
            let message = str_field(cmd, "message")?;
            let queue = cmd.get("queue").and_then(Value::as_str);
            if !matches!(queue, Some("steering" | "followUp")) {
                return Err(CmdError::new("queue must be \"steering\" or \"followUp\""));
            }
            ok(json!({"removed": rt.remove_queued(message, queue.unwrap_or("steering"))}))
        }
        "promote_queued_message" => {
            let message = str_field(cmd, "message")?;
            ok(json!({"promoted": rt.promote_queued(message)}))
        }
        "abort" => {
            rt.abort().await;
            none()
        }
        "abort_and_prompt" => {
            let message = str_field(cmd, "message")?.to_string();
            let images = images(cmd);
            let Some(message) = input_gate(&message).await else {
                return none();
            };
            rt.abort().await;
            let task_rt = rt.clone();
            let id = cmd.get("id").cloned();
            tokio::spawn(async move {
                if let Err(e) = task_rt.prompt(id.clone(), &message, None, images).await {
                    task_rt.send(response(
                        id.as_ref(),
                        "abort_and_prompt",
                        Err(CmdError::new(e.message.clone())),
                    ));
                    let mut f = json!({"type":"prompt_result","agentInvoked":false,"status":"error",
                        "error":{"message":e.message,"retryable":false},"sessionSettled":true});
                    if let Some(id) = id {
                        f["id"] = id;
                    }
                    task_rt.send(f);
                }
            });
            none()
        }
        "new_session" => match rt.new_session(None).await {
            Err(e) if crate::runtime::is_switch_cancelled(&e) => ok(json!({"cancelled": true})),
            other => {
                other?;
                ok(json!({"cancelled": false}))
            }
        },
        "switch_session" => {
            let path = PathBuf::from(str_field(cmd, "sessionPath")?);
            if !path.is_file() {
                return Err(CmdError::new(format!(
                    "Session file not found: {}",
                    path.display()
                )));
            }
            match rt.load_session(&path, None).await {
                Err(e) if crate::runtime::is_switch_cancelled(&e) => ok(json!({"cancelled": true})),
                other => {
                    other?;
                    ok(json!({"cancelled": false}))
                }
            }
        }
        "open_session" => {
            let dir = PathBuf::from(str_field(cmd, "sessionDir")?);
            let resumed = if let Some(path) = newest_session(&dir) {
                rt.load_session(&path, Some(&dir)).await?;
                true
            } else {
                rt.new_session(Some(&dir)).await?;
                false
            };
            ok(json!({
                "cancelled": false,
                "resumed": resumed,
                "sessionId": rt.session_id(),
                "sessionFile": rt.session_file().display().to_string(),
            }))
        }
        "branch" | "fork" => {
            let entry_id = if kind == "branch" {
                Some(str_field(cmd, "entryId")?)
            } else {
                cmd.get("entryId").and_then(Value::as_str)
            };
            if rt.agent.is_streaming().await {
                return Err(CmdError::coded(
                    format!("Cannot {kind} the session while session maintenance or user work is still running"),
                    "session_busy",
                ));
            }
            let result = if kind == "branch" {
                crate::tree::branch(&Arc::clone(rt), entry_id.unwrap_or_default()).await
            } else {
                crate::tree::fork(&Arc::clone(rt), entry_id).await
            };
            rt.publish_session_file();
            result
        }
        "get_state" => ok(rt.state().await),
        "set_fast_mode" => {
            if bool_field(cmd, "enabled")? {
                Err(CmdError::new(
                    "Fast mode is unavailable for the current model.",
                ))
            } else {
                ok(json!({"enabled": false, "active": false}))
            }
        }
        "goal" => goal(cmd),
        "set_ask_dialog" => {
            let enabled =
                rt.set(|s| s.ask_dialog(cmd.get("enabled").and_then(Value::as_bool) == Some(true)));
            ASK_DIALOG_ENABLED.store(
                cmd.get("enabled").and_then(Value::as_bool) == Some(true),
                Ordering::SeqCst,
            );
            ok(json!({"enabled": enabled}))
        }
        "get_available_commands" => ok(json!({"commands": slash::available_commands(rt).await})),
        "get_entries" => {
            let entries = rt.entries().map_err(CmdError::from)?;
            let leaf = entries
                .last()
                .and_then(|e| e.get("id").cloned())
                .unwrap_or(Value::Null);
            let selected = match cmd.get("since").and_then(Value::as_str) {
                None => entries,
                Some(since) => {
                    let idx = entries
                        .iter()
                        .position(|e| e.get("id").and_then(Value::as_str) == Some(since))
                        .ok_or_else(|| {
                            CmdError::coded(format!("Unknown entry id: {since}"), "unknown_since")
                        })?;
                    entries[idx + 1..].to_vec()
                }
            };
            ok(json!({"entries": selected, "leafId": leaf}))
        }
        "get_tree" => {
            let entries = rt.entries().map_err(CmdError::from)?;
            let leaf = entries
                .last()
                .and_then(|e| e.get("id").cloned())
                .unwrap_or(Value::Null);
            ok(json!({"tree": tree(&entries), "leafId": leaf}))
        }
        "set_todos" => {
            let phases = cmd
                .get("phases")
                .cloned()
                .filter(Value::is_array)
                .ok_or_else(|| CmdError::new("phases must be an array"))?;
            rt.set(|s| s.todo_phases(phases));
            ok(json!({"todoPhases": rt.set(|s| s.todo_phases_value())}))
        }
        "set_host_tools" => set_host_tools(cmd),
        "set_host_uri_schemes" => set_host_uri_schemes(cmd),
        "set_subagent_subscription" => {
            let level = str_field(cmd, "level")?;
            if !matches!(level, "off" | "progress" | "events") {
                return Err(CmdError::new(
                    "level must be \"off\", \"progress\" or \"events\"",
                ));
            }
            rt.set(|s| s.subagent_level(level));
            ok(json!({"level": level}))
        }
        "set_event_filter" => {
            let events = cmd.get("events");
            let filter = match events {
                Some(Value::Null) => None,
                Some(Value::Array(a)) => Some(
                    a.iter()
                        .filter_map(|v| v.as_str().map(str::to_string))
                        .collect::<HashSet<_>>(),
                ),
                _ => return Err(CmdError::new("events must be an array of strings or null")),
            };
            let delta = match cmd.get("messageUpdates").and_then(Value::as_str) {
                None | Some("full") => false,
                Some("delta") => true,
                Some(_) => {
                    return Err(CmdError::new(
                        "messageUpdates must be \"full\" or \"delta\"",
                    ))
                }
            };
            rt.set(|s| s.event_filter(filter, delta));
            let mut data = json!({"events": events.cloned().unwrap_or(Value::Null)});
            if let Some(m) = cmd.get("messageUpdates") {
                data["messageUpdates"] = m.clone();
            }
            ok(data)
        }
        "get_subagents" => ok(json!({"subagents": []})),
        "get_subagent_messages" => subagent_messages(cmd),
        "cancel_subagent" => {
            subagent_id(cmd, "cancel_subagent")?;
            ok(json!({"cancelled": false}))
        }
        "steer_subagent" => {
            let id = subagent_id(cmd, "steer_subagent")?;
            let message = str_field(cmd, "message")?;
            if message.trim().is_empty() {
                return Err(CmdError::new("`message` is required for steer_subagent."));
            }
            let receipt = crate::tools::eval_agents::send_to_agent(id, message)
                .await
                .map_err(CmdError::new)?;
            ok(json!({"delivered": true, "receipt": receipt}))
        }
        "live_start" => Err(unsupported("Live voice")),
        "live_stop" => none(),
        "live_mute" => Err(CmdError::new("No live session is active")),
        "set_model" => {
            let provider = str_field(cmd, "provider")?;
            let model_id = str_field(cmd, "modelId")?;
            ok(rt.set_model(provider, model_id).await?)
        }
        "cycle_model" => cycle_model(rt).await,
        "get_available_models" => {
            let current = {
                let info = rt.model_info();
                info.get("provider")
                    .zip(info.get("id"))
                    .map(|(p, i)| (p.clone(), i.clone()))
            };
            // Same catalogue as the TUI `/model` picker: every chat model of a
            // provider with credentials (bundled catalog, models.db, models.yml).
            let catalog = tokio::task::spawn_blocking(|| {
                crate::subcmd::models::browser_models(&omp_providers::AgentDir::default())
            })
            .await
            .unwrap_or_default();
            let mut models: Vec<Value> = catalog
                .iter()
                .filter(|b| b.available && b.kind.is_empty() && !b.model.id.is_empty())
                .map(|b| model_info(&b.model))
                .collect();
            if let Some((p, i)) = current {
                if !models
                    .iter()
                    .any(|m| m.get("provider") == Some(&p) && m.get("id") == Some(&i))
                {
                    models.insert(0, rt.model_info());
                }
            }
            ok(json!({"models": models}))
        }
        "set_thinking_level" => {
            let level = str_field(cmd, "level")?;
            if level == "auto" {
                rt.set_auto_thinking().await;
                return none();
            }
            if let Some(mode) = crate::runtime::parse_ultra_name(level) {
                rt.set_ultra(Some(mode)).await;
                return none();
            }
            let parsed = parse_level(level)
                .or_else(|| (level == "inherit").then_some(ThinkingLevel::Medium));
            let Some(parsed) = parsed else {
                return Err(CmdError::new(format!("Invalid thinking level: {level}")));
            };
            rt.set_thinking_level(parsed).await;
            none()
        }
        "cycle_thinking_level" => {
            let levels = rt.available_thinking_levels();
            if levels.len() < 2 {
                return ok(Value::Null);
            }
            let current = level_name(rt.agent.thinking_level().await);
            let idx = levels.iter().position(|l| *l == current).unwrap_or(0);
            let next = levels[(idx + 1) % levels.len()];
            if let Some(level) = parse_level(next) {
                rt.set_thinking_level(level).await;
            }
            ok(json!({"level": next}))
        }
        "get_available_thinking_levels" => ok(json!({"levels": rt.available_thinking_levels()})),
        "set_steering_mode" => {
            rt.set_queue_modes(Some(queue_mode_field(cmd)?), None);
            none()
        }
        "set_follow_up_mode" => {
            rt.set_queue_modes(None, Some(queue_mode_field(cmd)?));
            none()
        }
        "set_interrupt_mode" => {
            let mode = str_field(cmd, "mode")?;
            if !matches!(mode, "immediate" | "wait") {
                return Err(CmdError::new("mode must be \"immediate\" or \"wait\""));
            }
            rt.set(|s| s.interrupt_mode(mode));
            none()
        }
        "compact" => {
            let custom = cmd.get("customInstructions").and_then(Value::as_str);
            Ok(Some(rt.compact(custom).await?))
        }
        "set_auto_compaction" => {
            let enabled = bool_field(cmd, "enabled")?;
            rt.set(|s| s.auto_compaction(enabled));
            none()
        }
        "set_cache_warming" => {
            let mode = cmd.get("mode").and_then(Value::as_str).unwrap_or("");
            if !CACHE_WARMING_MODES.contains(&mode) {
                let shown = cmd.get("mode").map_or_else(
                    || "undefined".to_string(),
                    |m| m.as_str().map_or_else(|| m.to_string(), str::to_string),
                );
                return Err(CmdError::new(format!(
                    "Invalid cache warming mode: {shown}"
                )));
            }
            rt.set(|s| s.cache_warming(mode));
            ok(json!({"mode": mode}))
        }
        "set_auto_retry" => {
            let enabled = bool_field(cmd, "enabled")?;
            rt.set_auto_retry(enabled);
            none()
        }
        "abort_retry" => {
            rt.abort_retry();
            none()
        }
        // Attached-TUI parity (`omt attach`): the panels and actions the local
        // TUI port computes in-process, answered as response data so only the
        // asking client renders them.
        "get_session_info_panel" => ok(crate::tui::session_info_frame(rt).await),
        "get_usage_panel" => ok(slash::usage_panel_frame(rt).await),
        "navigate_tree" => {
            let entry = str_field(cmd, "entryId")?;
            crate::tree::navigate_tree(rt, entry, &json!({}), None).await
        }
        "enable_context_usage" => {
            rt.enable_context_usage_frames();
            rt.publish_context_usage();
            none()
        }
        "abort_bash" => {
            #[allow(clippy::redundant_closure_for_method_calls)]
            // fn-item form breaks: `set` passes `&mut SettingsView`
            rt.set(|s| s.abort_bash());
            none()
        }
        "get_session_stats" => ok(rt.session_stats().await),
        "export_html" => {
            if let Some(v) = cmd.get("outputPath") {
                if !v.is_string() {
                    return Err(CmdError::new("outputPath must be a string"));
                }
            }
            Err(unsupported("HTML export"))
        }
        "get_branch_messages" => {
            let entries = rt.entries().map_err(CmdError::from)?;
            ok(json!({"messages": branch_messages(&entries)}))
        }
        "get_last_assistant_text" => {
            ok(json!({"text": rt.last_assistant_text().await.map_or(Value::Null, Value::String)}))
        }
        "set_session_name" => {
            let name = str_field(cmd, "name")?.trim().to_string();
            if name.is_empty() {
                return Err(CmdError::new("Session name cannot be empty"));
            }
            rt.set_session_name(&name, "user").map_err(CmdError::from)?;
            none()
        }
        "handoff" => {
            if rt.agent.is_streaming().await {
                return Err(CmdError::new(
                    "Cannot hand off while a response is in progress",
                ));
            }
            Err(unsupported("Handoff"))
        }
        "get_messages" => ok(json!({"messages": rt.messages().await})),
        "get_capabilities" => ok(capabilities_payload(crate::paging::MAX_LIMIT)),
        "get_messages_page" => {
            let direction = cmd
                .get("direction")
                .and_then(Value::as_str)
                .unwrap_or("forward");
            if rt.agent.is_streaming().await && direction != "backward" {
                return Err(CmdError::coded(
                    "Cannot page messages while the session is changing",
                    "session_busy",
                ));
            }
            let messages = rt.messages().await;
            let entries = rt.entries().map_err(CmdError::from)?;
            let leaf = entries
                .last()
                .and_then(|e| e.get("id").and_then(Value::as_str))
                .map(str::to_string);
            let session_id = rt.session_id();
            let snapshot = paging::Snapshot {
                session_id: &session_id,
                leaf_id: leaf.as_deref(),
                message_count: messages.len(),
            };
            let limit = cmd.get("limit").and_then(Value::as_f64);
            let limit = match limit {
                None => None,
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                Some(n) if n.fract() == 0.0 && n >= 0.0 => Some(n as usize),
                Some(_) => Some(0),
            };
            let cursor = cmd.get("cursor").and_then(Value::as_str);
            match direction {
                "forward" => paging::page(&messages, &snapshot, cursor, limit),
                "backward" => paging::page_backward(&messages, &snapshot, cursor, limit),
                _ => {
                    return Err(CmdError::new(
                        "RPC message page direction must be forward or backward",
                    ));
                }
            }
            .map(Some)
            .map_err(|e| CmdError {
                message: e.message,
                code: e.code,
            })
        }
        "get_login_providers" => ok(json!({"providers": login_providers().await})),
        "login" => {
            let provider = str_field(cmd, "providerId")?;
            if !OAUTH_PROVIDERS.iter().any(|(id, _)| *id == provider) {
                return Err(CmdError::new(format!("Unknown OAuth provider: {provider}")));
            }
            Err(CmdError::new(format!(
                "Provider '{provider}' requires interactive prompts which are not supported in RPC mode. Use the terminal UI to log in."
            )))
        }
        "predict_word" => {
            text_cursor(cmd)?;
            ok(json!({"suffix": null}))
        }
        "predict_word_feedback" => {
            text_cursor(cmd)?;
            if cmd.get("suggestion").is_none_or(|v| !v.is_string())
                || cmd.get("accepted").is_none_or(|v| !v.is_boolean())
            {
                return Err(CmdError::new(
                    "suggestion must be a string and accepted a boolean",
                ));
            }
            none()
        }
        other => Err(CmdError::new(format!("Unknown command: {other}"))),
    }
}

const RPC_CHUNK_PAYLOAD_BYTES: usize = 256 * 1024;

/// Client-side reassembly of protocol-v2 `rpc_chunk` sequences (the web hub
/// and `omt attach` both read the same agent log).
#[derive(Default)]
pub struct ChunkAssembler {
    parts: HashMap<String, Vec<Option<Vec<u8>>>>,
}

impl ChunkAssembler {
    /// Store one `rpc_chunk`; returns the reassembled frame text once every
    /// part arrived. A malformed chunk drops its whole sequence.
    pub fn push(&mut self, chunk: &Value) -> Option<String> {
        use base64::Engine as _;
        let id = chunk.get("chunkId")?.as_str()?.to_owned();
        let parsed = (|| {
            let index = usize::try_from(chunk.get("index")?.as_u64()?).ok()?;
            let count = usize::try_from(chunk.get("count")?.as_u64()?).ok()?;
            let data = base64::engine::general_purpose::STANDARD
                .decode(chunk.get("data")?.as_str()?)
                .ok()?;
            (index < count).then_some((index, count, data))
        })();
        let Some((index, count, data)) = parsed else {
            self.parts.remove(&id);
            return None;
        };
        let parts = self
            .parts
            .entry(id.clone())
            .or_insert_with(|| vec![None; count]);
        if parts.len() != count {
            self.parts.remove(&id);
            return None;
        }
        parts[index] = Some(data);
        if parts.iter().any(Option::is_none) {
            return None;
        }
        let bytes: Vec<u8> = self
            .parts
            .remove(&id)?
            .into_iter()
            .flatten()
            .flatten()
            .collect();
        String::from_utf8(bytes).ok()
    }
}
const METADATA_STRING_CAP: usize = 1024;
const STRING_ELISION_RESERVE: usize = 80;
/// `(stringCap, arrayLimit, objectLimit)` per shrink pass, as omp's `SHRINK_PASSES`.
const SHRINK_PASSES: [(usize, usize, usize); 7] = [
    (256 * 1024, 512, 512),
    (64 * 1024, 256, 256),
    (16 * 1024, 128, 128),
    (4 * 1024, 64, 64),
    (1024, 32, 32),
    (256, 8, 16),
    (64, 1, 8),
];

fn base64(bytes: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for c in bytes.chunks(3) {
        let n = (u32::from(c[0]) << 16)
            | (u32::from(*c.get(1).unwrap_or(&0)) << 8)
            | u32::from(*c.get(2).unwrap_or(&0));
        out.push(T[(n >> 18) as usize & 63] as char);
        out.push(T[(n >> 12) as usize & 63] as char);
        out.push(if c.len() > 1 {
            T[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if c.len() > 2 {
            T[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

fn shrink_string(s: &str, cap: usize) -> String {
    let len = s.chars().count();
    if len <= cap {
        return s.to_string();
    }
    let head = cap.saturating_sub(STRING_ELISION_RESERVE);
    let kept: String = s.chars().take(head).collect();
    format!("{kept}\n…[{} chars elided for RPC frame]", len - head)
}

fn shrink_value(
    v: &Value,
    (string_cap, array_limit, object_limit): (usize, usize, usize),
) -> Value {
    let pass = (string_cap, array_limit, object_limit);
    match v {
        Value::String(s) => Value::String(shrink_string(s, string_cap)),
        Value::Array(a) => {
            let keep = a.len().min(array_limit);
            let mut out: Vec<Value> = a.iter().take(keep).map(|e| shrink_value(e, pass)).collect();
            if keep < a.len() {
                out.push(json!(format!(
                    "…[{} more items elided for RPC frame]",
                    a.len() - keep
                )));
            }
            Value::Array(out)
        }
        Value::Object(o) => {
            let keep = o.len().min(object_limit);
            let mut out = Map::new();
            for (k, e) in o.iter().take(keep) {
                out.insert(k.clone(), shrink_value(e, pass));
            }
            if keep < o.len() {
                out.insert("rpcFrameElidedKeys".into(), json!(o.len() - keep));
            }
            Value::Object(out)
        }
        other => other.clone(),
    }
}

/// `overflowFrame`: what replaces a frame that cannot fit the transport at all.
fn overflow_frame(frame: &Value) -> Value {
    let cap = |v: Option<&Value>| {
        v.and_then(Value::as_str)
            .map(|s| shrink_string(s, METADATA_STRING_CAP))
    };
    match frame.get("type").and_then(Value::as_str) {
        Some("response") => {
            let mut m = Map::new();
            if let Some(id) = cap(frame.get("id")) {
                m.insert("id".into(), json!(id));
            }
            m.insert("type".into(), json!("response"));
            m.insert(
                "command".into(),
                json!(cap(frame.get("command")).unwrap_or_else(|| "unknown".into())),
            );
            m.insert("success".into(), json!(false));
            m.insert(
                "error".into(),
                json!("RPC response exceeded the transport limit"),
            );
            Value::Object(m)
        }
        Some("agent_end") => json!({
            "type": "agent_end",
            "messages": [],
            "messageCount": frame.get("messageCount").and_then(Value::as_u64).unwrap_or(0),
        }),
        original => {
            let mut m = Map::new();
            m.insert("type".into(), json!("rpc_frame_error"));
            if let Some(t) = original {
                m.insert(
                    "originalType".into(),
                    json!(shrink_string(t, METADATA_STRING_CAP)),
                );
            }
            m.insert(
                "error".into(),
                json!("RPC frame exceeded the transport limit"),
            );
            Value::Object(m)
        }
    }
}

/// Port of omp's `RpcFrameEncoder`: keeps every physical line within `MAX_FRAME_BYTES`
/// (protocol v1: shrink / compact / replace; v2: `rpc_chunk` sequences).
struct FrameEncoder {
    version: u8,
    streamed: Vec<Value>,
    chunks: u64,
    /// `Some` for a detached child (stdout is the agent log): see [`UpdateCompactor`].
    compact: Option<UpdateCompactor>,
}

impl FrameEncoder {
    fn new() -> Self {
        Self {
            version: 1,
            streamed: Vec::new(),
            chunks: 0,
            compact: None,
        }
    }

    /// Write `message_update` frames as deltas (detached-log mode).
    fn compacting(mut self) -> Self {
        self.compact = Some(UpdateCompactor::default());
        self
    }

    fn fits(json: &str) -> bool {
        (json.len() as u64) < MAX_FRAME_BYTES
    }

    /// `agent_end` minus the messages the client already received as `message_end` frames.
    fn compact_terminal(&self, frame: &Value) -> Value {
        let Some(messages) = frame
            .get("messages")
            .and_then(Value::as_array)
            .filter(|_| frame.get("type") == Some(&json!("agent_end")))
        else {
            return frame.clone();
        };
        let mut streamed = 0;
        while streamed < self.streamed.len().min(messages.len())
            && self.streamed[streamed] == messages[streamed]
        {
            streamed += 1;
        }
        let mut out = frame.clone();
        out["messages"] = json!(messages[streamed..]);
        out["messageCount"] = json!(messages.len());
        out
    }

    fn encode_v1(&self, frame: &Value, json: &str) -> String {
        if Self::fits(json) {
            return json.to_string();
        }
        if frame.get("type") == Some(&json!("response")) {
            return overflow_frame(frame).to_string();
        }
        let compacted = self.compact_terminal(frame);
        let json = compacted.to_string();
        if Self::fits(&json) {
            return json;
        }
        for pass in SHRINK_PASSES {
            let shrunk = shrink_value(&compacted, pass).to_string();
            if Self::fits(&shrunk) {
                return shrunk;
            }
        }
        overflow_frame(&compacted).to_string()
    }

    /// Physical JSONL lines (without newlines) for one logical frame.
    fn encode(&mut self, frame: &Value) -> Vec<String> {
        let compacted = self.compact.as_mut().and_then(|c| c.compact(frame));
        let frame = compacted.as_ref().unwrap_or(frame);
        let kind = frame.get("type").and_then(Value::as_str);
        let trace = std::env::var_os("OMP_RS_TURN_TRACE").is_some();
        if trace {
            crate::runtime::turn_trace!(
                "[turn-trace] encode kind={kind:?} rss={}kB",
                crate::runtime::rss_kb()
            );
        }
        if kind == Some("agent_start") {
            self.streamed.clear();
        }
        let json = frame.to_string();
        if trace {
            crate::runtime::turn_trace!(
                "[turn-trace] encoded kind={kind:?} json={}B rss={}kB",
                json.len(),
                crate::runtime::rss_kb()
            );
        }
        let lines = if self.version == 2 && !Self::fits(&json) {
            let compacted = self.compact_terminal(frame);
            let cjson = compacted.to_string();
            if Self::fits(&cjson) {
                vec![cjson]
            } else if cjson.len() as u64 > MAX_REASSEMBLED_BYTES {
                vec![overflow_frame(&compacted).to_string()]
            } else {
                self.chunks += 1;
                let id = format!("rpc-{}", self.chunks);
                let bytes = cjson.as_bytes();
                let count = bytes.len().div_ceil(RPC_CHUNK_PAYLOAD_BYTES);
                bytes
                    .chunks(RPC_CHUNK_PAYLOAD_BYTES)
                    .enumerate()
                    .map(|(index, part)| {
                        json!({"type": "rpc_chunk", "chunkId": id, "index": index, "count": count,
                               "byteLength": bytes.len(), "data": base64(part)})
                        .to_string()
                    })
                    .collect()
            }
        } else {
            vec![self.encode_v1(frame, &json)]
        };
        match kind {
            Some("message_end") => {
                if let Some(m) = frame.get("message") {
                    self.streamed.push(m.clone());
                }
            }
            Some("agent_end") if frame.get("willContinue") != Some(&json!(true)) => {
                self.streamed.clear();
            }
            _ => {}
        }
        if kind == Some("response")
            && frame.get("command") == Some(&json!("negotiate_protocol"))
            && frame.get("success") == Some(&json!(true))
        {
            self.version = 2;
        }
        lines
    }
}

/// Byte length of the common prefix of `a` and `b`, at a char boundary of both.
fn common_prefix(a: &str, b: &str) -> usize {
    let mut p = a.bytes().zip(b.bytes()).take_while(|(x, y)| x == y).count();
    while !(a.is_char_boundary(p) && b.is_char_boundary(p)) {
        p -= 1;
    }
    p
}

/// Byte length of the common suffix of `a` and `b`, at a char boundary of both.
fn common_suffix(a: &str, b: &str) -> usize {
    let mut q = a
        .bytes()
        .rev()
        .zip(b.bytes().rev())
        .take_while(|(x, y)| x == y)
        .count();
    while !(a.is_char_boundary(a.len() - q) && b.is_char_boundary(b.len() - q)) {
        q -= 1;
    }
    q
}

/// Detached-log `message_update` compaction. Every `message_update` carries
/// the whole partial message — twice: `message` and
/// `assistantMessageEvent.partial` — so a log written verbatim grows
/// O(reply²) (one real log reached 1.3 GB). In the log, an update whose
/// serializations differ from the previous update's only by a small middle
/// part is written with `message` replaced (in place) by
/// `messageDelta: {n, p, q, s}` — the previous JSON (`n` bytes long) with the
/// bytes between its first `p` and its last `q` replaced by `s` — and
/// `assistantMessageEvent.partial` likewise by `partialDelta`. A full frame
/// (keyframe) goes out for a message's first update and whenever the message
/// has doubled since the last keyframe, so the log stays O(output) and a
/// reader joining mid-stream resyncs. Readers rebuild the full frames,
/// byte for byte, with [`UpdateExpander`].
#[derive(Default)]
struct UpdateCompactor {
    /// Serialized `message` of the previous update of the current message.
    prev_message: Option<String>,
    /// Serialized `assistantMessageEvent.partial` of that update.
    prev_partial: Option<String>,
    /// Length of the message JSON at the last keyframe.
    key_len: usize,
}

/// Field replacing `message` in a delta-encoded `message_update`.
const MESSAGE_DELTA_KEY: &str = "messageDelta";
/// Field replacing `assistantMessageEvent.partial` there.
const PARTIAL_DELTA_KEY: &str = "partialDelta";
const EVENT_KEY: &str = "assistantMessageEvent";

/// `{n, p, q, s}` turning `prev` into `json`, unless the change is most of `json`.
fn text_delta(prev: &str, json: &str) -> Option<Value> {
    let p = common_prefix(prev, json);
    let q = common_suffix(&prev[p..], &json[p..]);
    let s = &json[p..json.len() - q];
    (s.len() * 2 < json.len()).then(|| json!({"n": prev.len(), "p": p, "q": q, "s": s}))
}

/// Apply a [`text_delta`] to `base`.
fn apply_text_delta(base: &str, delta: &Value) -> Option<String> {
    let int = |k: &str| usize::try_from(delta.get(k)?.as_u64()?).ok();
    let (n, p, q) = (int("n")?, int("p")?, int("q")?);
    let s = delta.get("s")?.as_str()?;
    let tail = n.checked_sub(q)?;
    if base.len() != n || p > tail || !base.is_char_boundary(p) || !base.is_char_boundary(tail) {
        return None;
    }
    let mut out = String::with_capacity(p + s.len() + q);
    out.push_str(&base[..p]);
    out.push_str(s);
    out.push_str(&base[tail..]);
    Some(out)
}

/// `obj` with the value under `from` swapped for `value`, renamed `to`
/// (key order kept).
fn swap_field(obj: &Map<String, Value>, from: &str, to: &str, value: Value) -> Value {
    let mut value = Some(value);
    let mut m = Map::with_capacity(obj.len());
    for (k, v) in obj {
        match value.take_if(|_| k == from) {
            Some(value) => m.insert(to.to_owned(), value),
            None => m.insert(k.clone(), v.clone()),
        };
    }
    Value::Object(m)
}

impl UpdateCompactor {
    /// The delta frame replacing `frame`, or `None` to write `frame` as is.
    fn compact(&mut self, frame: &Value) -> Option<Value> {
        match frame.get("type").and_then(Value::as_str) {
            Some("message_update") => {
                let (Some(obj), Some(message)) = (frame.as_object(), frame.get("message")) else {
                    self.prev_message = None;
                    return None;
                };
                let event = frame.get(EVENT_KEY).and_then(Value::as_object);
                let message_json = message.to_string();
                let partial_json = event.and_then(|e| e.get("partial")).map(Value::to_string);
                // A keyframe on doubling lets a reader joining mid-stream resync.
                let keyframe = message_json.len() >= self.key_len.saturating_mul(2);
                let message_delta = self
                    .prev_message
                    .as_deref()
                    .filter(|_| !keyframe)
                    .and_then(|prev| text_delta(prev, &message_json));
                let partial_delta = match (&partial_json, &self.prev_partial) {
                    (None, _) => Some(None),
                    (Some(json), Some(prev)) => text_delta(prev, json).map(Some),
                    (Some(_), None) => None,
                };
                let out = message_delta.zip(partial_delta).map(|(md, pd)| {
                    let mut out = swap_field(obj, "message", MESSAGE_DELTA_KEY, md);
                    if let (Some(pd), Some(event)) = (pd, event) {
                        out[EVENT_KEY] = swap_field(event, "partial", PARTIAL_DELTA_KEY, pd);
                    }
                    out
                });
                if out.is_none() {
                    self.key_len = message_json.len();
                }
                self.prev_message = Some(message_json);
                self.prev_partial = partial_json;
                out
            }
            Some("message_start" | "message_end" | "agent_start" | "agent_end") => {
                self.prev_message = None;
                None
            }
            _ => None,
        }
    }
}

/// What the expander last saw of the current partial message.
enum UpdateBase {
    /// A full `message_update` line; its parts are extracted on demand.
    Line(String),
    /// The exact serialized `message` and `assistantMessageEvent.partial`.
    Parts {
        message: String,
        partial: Option<String>,
    },
}

/// What [`UpdateExpander::expand`] did with a log frame.
#[derive(Debug, PartialEq, Eq)]
pub enum Expansion {
    /// Pass the line on as is.
    Unchanged,
    /// A delta `message_update` rebuilt into the full frame line.
    Expanded(String),
    /// A delta whose base is unknown (reader joined mid-stream or lost
    /// sync): drop it; the next keyframe or `message_end` resyncs.
    Unsynced,
}

/// Reader side of [`UpdateCompactor`]: rebuilds full `message_update` lines
/// (byte-identical to what a non-compacting child writes) from a detached
/// agent log. Feed it every frame of the log in order.
#[derive(Default)]
pub struct UpdateExpander {
    base: Option<UpdateBase>,
}

/// Serialize `obj` compactly, writing raw JSON `raw` as `to` in place of `from`.
fn splice_object(out: &mut String, obj: &Map<String, Value>, from: &str, to: &str, raw: &str) {
    out.push('{');
    for (i, (k, v)) in obj.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        if k == from {
            out.push_str(&Value::String(to.to_owned()).to_string());
            out.push(':');
            out.push_str(raw);
        } else {
            out.push_str(&Value::String(k.clone()).to_string());
            out.push(':');
            out.push_str(&v.to_string());
        }
    }
    out.push('}');
}

impl UpdateExpander {
    /// `line` is the frame's JSON text (after `rpc_chunk` reassembly), `frame` its parse.
    pub fn expand(&mut self, line: &str, frame: &Value) -> Expansion {
        match frame.get("type").and_then(Value::as_str) {
            Some("message_update") => {
                let Some(delta) = frame.get(MESSAGE_DELTA_KEY) else {
                    self.base = frame
                        .get("message")
                        .map(|_| UpdateBase::Line(line.to_owned()));
                    return Expansion::Unchanged;
                };
                let partial_delta = frame.get(EVENT_KEY).and_then(|e| e.get(PARTIAL_DELTA_KEY));
                let rebuilt = self.base_parts().and_then(|(message, partial)| {
                    let message = apply_text_delta(message, delta)?;
                    let partial = match partial_delta {
                        Some(d) => Some(apply_text_delta(partial?, d)?),
                        None => None,
                    };
                    Some((message, partial))
                });
                let (Some((message, partial)), Some(obj)) = (rebuilt, frame.as_object()) else {
                    self.base = None;
                    return Expansion::Unsynced;
                };
                let event = match (&partial, obj.get(EVENT_KEY).and_then(Value::as_object)) {
                    (Some(partial), Some(event)) => {
                        let mut s = String::with_capacity(partial.len() + 128);
                        splice_object(&mut s, event, PARTIAL_DELTA_KEY, "partial", partial);
                        Some(s)
                    }
                    _ => None,
                };
                let mut out = String::with_capacity(
                    message.len() + event.as_ref().map_or(0, String::len) + 64,
                );
                out.push('{');
                for (i, (k, v)) in obj.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    if k == MESSAGE_DELTA_KEY {
                        out.push_str("\"message\":");
                        out.push_str(&message);
                        continue;
                    }
                    out.push_str(&Value::String(k.clone()).to_string());
                    out.push(':');
                    match (&event, k == EVENT_KEY) {
                        (Some(event), true) => out.push_str(event),
                        _ => out.push_str(&v.to_string()),
                    }
                }
                out.push('}');
                self.base = Some(UpdateBase::Parts { message, partial });
                Expansion::Expanded(out)
            }
            Some("message_start" | "message_end" | "agent_start" | "agent_end") => {
                self.base = None;
                Expansion::Unchanged
            }
            _ => Expansion::Unchanged,
        }
    }

    /// The exact serialized base parts, extracted from a full line on first use.
    fn base_parts(&mut self) -> Option<(&str, Option<&str>)> {
        #[derive(serde::Deserialize)]
        struct Event<'a> {
            #[serde(borrow, default)]
            partial: Option<&'a serde_json::value::RawValue>,
        }
        #[derive(serde::Deserialize)]
        struct Full<'a> {
            #[serde(borrow)]
            message: &'a serde_json::value::RawValue,
            #[serde(borrow, default, rename = "assistantMessageEvent")]
            event: Option<Event<'a>>,
        }
        if let Some(UpdateBase::Line(line)) = &self.base {
            let parts = serde_json::from_str::<Full<'_>>(line)
                .ok()
                .map(|f| UpdateBase::Parts {
                    message: f.message.get().to_owned(),
                    partial: f.event.and_then(|e| e.partial).map(|p| p.get().to_owned()),
                });
            self.base = parts;
        }
        match &self.base {
            Some(UpdateBase::Parts { message, partial }) => Some((message, partial.as_deref())),
            _ => None,
        }
    }
}

/// Test hook: the physical log lines a child writes for `frames` after the
/// hub negotiated protocol v2, with (`compact`, detached mode) or without
/// `message_update` compaction.
#[cfg(test)]
pub(crate) fn encode_log(frames: &[Value], compact: bool) -> Vec<String> {
    let mut encoder = FrameEncoder::new();
    encoder.version = 2;
    if compact {
        encoder = encoder.compacting();
    }
    frames.iter().flat_map(|f| encoder.encode(f)).collect()
}

/// Hook run once the runtime exists, before the `ready` frame (extension host start-up).
pub type OnStart = Box<
    dyn FnOnce(Arc<Runtime>) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>>
        + Send,
>;

/// Run the RPC loop until stdin closes — or, with `OMP_RPC_SOCKET` set, until
/// the process is signalled (detached mode; see [`serve_detached`]).
///
/// # Errors
/// Fails on stdin read errors or a failed control-socket bind.
pub async fn run(agent: Agent, target: Target, cwd: PathBuf, persist: Persistence) -> Result<()> {
    run_with(agent, target, cwd, persist, None).await
}

/// [`run`] with a start-up hook (see [`OnStart`]).
///
/// # Errors
/// Fails on stdin read errors or a failed control-socket bind.
#[allow(clippy::too_many_lines)] // the RPC dispatch loop; one branch per wire frame type, mirrors omp's handler table
pub async fn run_with(
    agent: Agent,
    target: Target,
    cwd: PathBuf,
    persist: Persistence,
    on_start: Option<OnStart>,
) -> Result<()> {
    // `OMP_RPC_SOCKET` (set by `omp-rs web`'s spawn): serve frames on a
    // per-agent Unix socket and outlive the hub. Binding before the runtime
    // warms up lets a restarting hub connect immediately; connections queue
    // in the listener backlog until `serve_detached` below starts.
    let detached = match DETACHED_SOCKET.get().cloned().flatten() {
        Some(path) => Some(bind_detached(Path::new(&path))?),
        None => None,
    };
    let (tx, mut rx) = mpsc::unbounded_channel::<Value>();
    let agent = Arc::new(agent);
    RUNTIME_TX.with(|slot| *slot.borrow_mut() = Some(tx.clone()));
    let rt = Runtime::start(agent.clone(), target, cwd, persist, tx.clone()).await;
    if let Some(gate) = crate::approval::Gate::get() {
        gate.bind(&rt, true);
    }

    // The `ask` tool gets the RPC dialog transport (TS: rich `askDialog` once
    // the client sends `set_ask_dialog`, `select`/`editor` frames otherwise).
    {
        let mut tools = rt.agent.tools().await;
        if let Some(slot) = tools.iter_mut().find(|t| t.name() == "ask") {
            *slot = crate::tools::tool_with_ui(Arc::new(RpcAskUi));
            rt.agent.set_tools(tools).await;
        }
    }
    // A detached child's stdout is the agent log the hub tails and replays.
    let compact_log = detached.is_some();
    let writer = tokio::spawn(async move {
        let mut out = tokio::io::stdout();
        let mut encoder = FrameEncoder::new();
        if compact_log {
            encoder = encoder.compacting();
        }
        while let Some(v) = rx.recv().await {
            if v.get(FLUSH_MARKER).is_some() {
                break;
            }
            let collect = matches!(
                v.get("type").and_then(Value::as_str),
                Some("session_settled")
            ) || (v.get("type").and_then(Value::as_str) == Some("agent_end")
                && v.get("willContinue") != Some(&json!(true)));
            let mut buf = String::new();
            for line in encoder.encode(&v) {
                buf.push_str(&line);
                buf.push('\n');
            }
            if out.write_all(buf.as_bytes()).await.is_err() || out.flush().await.is_err() {
                break;
            }
            drop(buf);
            if collect {
                crate::collect_retained_memory();
            }
        }
    });

    if let Some(hook) = on_start {
        hook(rt.clone()).await;
    }
    rt.send(json!({
        "type": "ready",
        "protocolVersion": 1,
        "supportedProtocolVersions": [1, 2],
        "maxFrameBytes": MAX_FRAME_BYTES,
        "maxReassembledFrameBytes": MAX_REASSEMBLED_BYTES,
    }));
    rt.announce_commands();
    let dynamic = slash::dynamic_commands(&rt).await;
    if !dynamic.is_empty() {
        rt.set_available_commands(dynamic);
        rt.announce_commands();
    }

    let mut lines = BufReader::new(tokio::io::stdin()).lines();
    // Drained on the stdio exit path; detached mode keeps it `Some` forever
    // (a detached child has no voluntary exit to drain for).
    let mut background: Option<tokio::task::JoinSet<()>> = Some(tokio::task::JoinSet::new());
    // Commands run one at a time, in arrival order, on a worker task, so the read loop keeps
    // routing `extension_ui_response` (and `abort`) while a command awaits a dialog answer
    // (interactive slash commands, extension `ui.select`): handling inline deadlocked.
    let (cmd_tx, mut cmd_rx) = mpsc::unbounded_channel::<Value>();
    let busy = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let worker = {
        let (rt, busy) = (rt.clone(), busy.clone());
        tokio::spawn(async move {
            while let Some(cmd) = cmd_rx.recv().await {
                busy.store(true, std::sync::atomic::Ordering::SeqCst);
                let reply = handle(&rt, &cmd).await;
                rt.send(reply);
                busy.store(false, std::sync::atomic::Ordering::SeqCst);
            }
        })
    };
    if let Some(listener) = detached {
        // Detached: stdin frames still work until EOF, but EOF no longer ends
        // the process — the control socket keeps it alive for the hub; only a
        // process-group SIGTERM (web DELETE) does.
        return serve_detached(&rt, listener, &cmd_tx, &busy).await;
    }
    while let Some(line) = lines.next_line().await? {
        if let Some(cmd) = parse_line(&rt, &line) {
            route_frame(&rt, cmd, &cmd_tx, &busy, &mut background).await;
        }
    }
    // stdin closed: let accepted work finish and flush its frames.
    drop(cmd_tx);
    let _ = worker.await;
    if let Some(mut background) = background.take() {
        while background.join_next().await.is_some() {}
    }
    agent.wait_for_idle().await;
    // `prompt_result` / `session_settled` land a tick after the final `agent_end`.
    rt.wait_settled().await;
    drop(rt);
    // Every frame queued so far precedes the marker (FIFO), so the writer
    // exits right after flushing them; the timeout only bounds a stalled stdout.
    let _ = tx.send(json!({ FLUSH_MARKER: true }));
    let mut writer = writer;
    if tokio::time::timeout(std::time::Duration::from_secs(2), &mut writer)
        .await
        .is_err()
    {
        writer.abort();
    }
    Ok(())
}

/// Bind the detached control socket, replacing a stale file left by an
/// earlier incarnation of the same agent id.
fn bind_detached(sock_path: &Path) -> Result<UnixListener> {
    if let Some(dir) = sock_path.parent() {
        std::fs::create_dir_all(dir).ok();
    }
    let _ = std::fs::remove_file(sock_path);
    UnixListener::bind(sock_path).with_context(|| format!("bind {}", sock_path.display()))
}

/// Serve frames on the control socket until the process is signalled. The
/// socket is inbound-only: replies flow to stdout (the per-agent log the hub
/// tails). A client's EOF just ends its reader; the process itself never
/// exits here, so nothing is drained or flushed — a detached child only dies
/// by signal.
async fn serve_detached(
    rt: &Arc<Runtime>,
    listener: UnixListener,
    cmd_tx: &mpsc::UnboundedSender<Value>,
    busy: &Arc<AtomicBool>,
) -> Result<()> {
    loop {
        let Ok((stream, _)) = listener.accept().await else {
            continue;
        };
        let (rt, cmd_tx, busy) = (rt.clone(), cmd_tx.clone(), busy.clone());
        tokio::spawn(async move {
            let mut lines = BufReader::new(stream).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                if let Some(cmd) = parse_line(&rt, &line) {
                    route_frame(&rt, cmd, &cmd_tx, &busy, &mut None).await;
                }
            }
        });
    }
}

/// Decode one input line for either transport: blank lines are skipped and
/// parse errors are answered, byte-for-byte the rules the stdio loop had.
fn parse_line(rt: &Runtime, line: &str) -> Option<Value> {
    if line.trim().is_empty() {
        return None;
    }
    match serde_json::from_str::<Value>(line) {
        Ok(v) if v.is_object() => Some(v),
        Ok(_) => {
            rt.send(response(
                None,
                "parse",
                Err(CmdError::new(
                    "Failed to parse command: expected a JSON object",
                )),
            ));
            None
        }
        Err(e) => {
            rt.send(response(
                None,
                "parse",
                Err(CmdError::new(format!("Failed to parse command: {e}"))),
            ));
            None
        }
    }
}

/// Route one decoded frame: dialog answers resolve their waiter, `bash` runs
/// concurrently with command dispatch, an `abort` interrupts an in-flight
/// command, everything else queues for the ordered worker. `background` is
/// the stdio-mode drain set; `None` (detached mode) detaches `bash` jobs.
async fn route_frame(
    rt: &Arc<Runtime>,
    cmd: Value,
    cmd_tx: &mpsc::UnboundedSender<Value>,
    busy: &Arc<AtomicBool>,
    background: &mut Option<tokio::task::JoinSet<()>>,
) {
    let kind = cmd.get("type").and_then(Value::as_str).unwrap_or("");
    if kind == "extension_ui_response" {
        // Resolve a pending ask-UI dialog (see `RpcAskUi`).
        if let Some(id) = cmd.get("id").and_then(Value::as_str) {
            resolve_pending_ui_request(id, cmd.clone());
        }
        return;
    }
    if matches!(
        kind,
        "host_tool_result" | "host_tool_update" | "host_uri_result"
    ) {
        return; // side-channel frames for sub-protocols this server never opens
    }
    if kind == "bash" {
        let rt = rt.clone();
        let job = async move {
            let result = match str_field(&cmd, "command") {
                Ok(c) => run_bash(&rt, c).await,
                Err(e) => Err(e),
            };
            rt.send(response(cmd.get("id"), "bash", result));
        };
        match background {
            Some(set) => {
                set.spawn(job);
            }
            None => {
                tokio::spawn(job);
            }
        }
        return;
    }
    if kind == "abort" && busy.load(Ordering::SeqCst) {
        // A command is in flight (possibly waiting on a dialog): abort the run now
        // instead of queueing behind it.
        let reply = handle(rt, &cmd).await;
        rt.send(reply);
        return;
    }
    let _ = cmd_tx.send(cmd);
}

#[cfg(test)]
#[allow(clippy::disallowed_methods)] // tests: throwaway runtimes drive async code
mod tests {
    use super::*;

    #[test]
    fn get_capabilities_reports_version_and_page_limits() {
        let caps = capabilities_payload(crate::paging::MAX_LIMIT);
        assert_eq!(caps["protocolVersion"], 1);
        assert_eq!(caps["supportedProtocolVersions"], json!([1, 2]));
        assert_eq!(caps["features"]["messagesBackwardPaging"], json!(true));
        assert_eq!(caps["maxFrameBytes"], MAX_FRAME_BYTES);
        assert_eq!(
            caps["limits"]["getMessages"]["maxPageSize"],
            crate::paging::MAX_LIMIT
        );
        assert_eq!(
            caps["limits"]["getMessages"]["defaultPageSize"],
            crate::paging::DEFAULT_LIMIT
        );
        assert_eq!(
            caps["limits"]["getMessages"]["maxPageBytes"],
            crate::paging::MAX_PAGE_BYTES
        );
        assert_eq!(
            caps["limits"]["getMessages"]["maxCursorChars"],
            crate::paging::MAX_CURSOR_CHARS
        );
    }

    #[test]
    fn legacy_profile_page_limit_is_within_both_agent_generations_bounds() {
        let caps = legacy_capabilities_payload();
        assert_eq!(caps["supportedProtocolVersions"], json!([1]));
        assert_eq!(caps["features"]["messagesBackwardPaging"], json!(false));
        // Same shape as the live payload (frontend needs one parser).
        assert_eq!(
            caps["limits"]["getMessages"]["defaultPageSize"],
            capabilities_payload(crate::paging::MAX_LIMIT)["limits"]["getMessages"]
                ["defaultPageSize"]
        );
        let legacy = caps["limits"]["getMessages"]["maxPageSize"]
            .as_u64()
            .unwrap();
        assert_eq!(legacy, LEGACY_MAX_GET_MESSAGES_PAGE_SIZE as u64);
        assert!(legacy <= crate::paging::MAX_LIMIT as u64);
        assert!(legacy >= 1);
    }

    #[test]
    fn page_accepts_legacy_limit_and_rejects_over_limit() {
        let m = vec![json!({"role": "user", "content": "hi"}); 300];
        let snap = paging::Snapshot {
            session_id: "s",
            leaf_id: None,
            message_count: 300,
        };
        // A page minted at the legacy limit must still be valid on a new agent.
        let paged = crate::paging::page(&m, &snap, None, Some(LEGACY_MAX_GET_MESSAGES_PAGE_SIZE))
            .expect("legacy limit accepted");
        assert_eq!(
            paged["messages"].as_array().unwrap().len(),
            LEGACY_MAX_GET_MESSAGES_PAGE_SIZE
        );
        assert!(paged["nextCursor"].is_string());
        let err =
            crate::paging::page(&m, &snap, None, Some(crate::paging::MAX_LIMIT + 1)).unwrap_err();
        assert_eq!(err.code, None);
        assert!(err.message.contains("between 1 and 512"));
    }

    #[test]
    fn tree_with_id_less_header_terminates() {
        // A real session journal starts with an id-less `title` entry; get_tree used to recurse
        // on it with `parent: None` until the agent died of a stack overflow.
        let entries = vec![
            json!({"type": "title"}),
            json!({"type": "session", "id": "s"}),
            json!({"type": "message", "id": "m1", "parentId": "s"}),
            json!({"type": "message", "id": "m2", "parentId": "m1"}),
        ];
        let t = tree(&entries);
        assert_eq!(t.len(), 2, "two roots: the title header and the session");
        assert_eq!(t[0]["children"], json!([]));
        assert_eq!(
            t[1]["children"][0]["children"][0]["entry"]["id"],
            json!("m2")
        );
    }

    fn big_response(id: &str) -> Value {
        json!({"id": id, "type": "response", "command": "get_available_models", "success": true,
               "data": {"models": [{"blob": "x".repeat(1_100_000)}]}})
    }

    #[test]
    fn protocol_v1_replaces_an_oversized_response_with_omp_error() {
        let mut enc = FrameEncoder::new();
        let lines = enc.encode(&big_response("g2"));
        assert_eq!(lines.len(), 1);
        let v: Value = serde_json::from_str(&lines[0]).unwrap();
        assert_eq!(
            v,
            json!({"id": "g2", "type": "response", "command": "get_available_models", "success": false,
                   "error": "RPC response exceeded the transport limit"})
        );
    }

    #[test]
    fn protocol_v1_shrinks_oversized_events_to_fit() {
        let mut enc = FrameEncoder::new();
        let frame = json!({"type": "tool_execution_end", "toolCallId": "t", "result": {"output": "y".repeat(2_000_000)}});
        let lines = enc.encode(&frame);
        assert!((lines[0].len() as u64) < MAX_FRAME_BYTES);
        let v: Value = serde_json::from_str(&lines[0]).unwrap();
        assert_eq!(v["type"], "tool_execution_end");
        assert!(v["result"]["output"]
            .as_str()
            .unwrap()
            .contains("chars elided for RPC frame"));
    }

    #[test]
    fn protocol_v2_chunks_reassemble_to_the_original_frame() {
        let mut enc = FrameEncoder::new();
        let negotiated = json!({"id": "n", "type": "response", "command": "negotiate_protocol", "success": true, "data": {"protocolVersion": 2}});
        assert_eq!(enc.encode(&negotiated).len(), 1);
        let frame = big_response("g2");
        let lines = enc.encode(&frame);
        assert!(lines.len() > 1);
        let mut bytes = Vec::new();
        for (i, line) in lines.iter().enumerate() {
            let c: Value = serde_json::from_str(line).unwrap();
            assert_eq!(
                (c["type"].clone(), c["index"].clone(), c["count"].clone()),
                (json!("rpc_chunk"), json!(i), json!(lines.len()))
            );
            assert!((line.len() as u64) < MAX_FRAME_BYTES);
            bytes.extend(unbase64(c["data"].as_str().unwrap()));
            assert_eq!(c["byteLength"], frame.to_string().len());
        }
        assert_eq!(serde_json::from_slice::<Value>(&bytes).unwrap(), frame);
    }

    #[test]
    fn terminal_agent_end_drops_messages_already_streamed() {
        let mut enc = FrameEncoder::new();
        let big = json!({"role": "assistant", "content": [{"type": "text", "text": "z".repeat(700_000)}]});
        let other = json!({"role": "assistant", "content": [{"type": "text", "text": "w".repeat(700_000)}]});
        enc.encode(&json!({"type": "agent_start"}));
        enc.encode(&json!({"type": "message_end", "message": big}));
        let lines =
            enc.encode(&json!({"type": "agent_end", "messages": [big, other], "isTerminal": true}));
        let v: Value = serde_json::from_str(&lines[0]).unwrap();
        assert_eq!(v["messageCount"], 2);
        assert_eq!(v["messages"], json!([other]));
    }

    // Bit-extraction casts are exact: the 24-bit accumulator is built from 6-bit sextets.
    #[allow(clippy::cast_possible_truncation)]
    fn unbase64(s: &str) -> Vec<u8> {
        const T: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let vals: Vec<u32> = s
            .bytes()
            .filter(|b| *b != b'=')
            .map(|b| T.find(b as char).unwrap() as u32)
            .collect();
        let mut out = Vec::new();
        for c in vals.chunks(4) {
            let n = c
                .iter()
                .enumerate()
                .fold(0u32, |a, (i, v)| a | (v << (18 - 6 * i)));
            out.push((n >> 16) as u8);
            if c.len() > 2 {
                out.push((n >> 8) as u8);
            }
            if c.len() > 3 {
                out.push(n as u8);
            }
        }
        out
    }

    #[test]
    fn unsupported_commands_follow_omp_validation_and_error_shapes() {
        assert_eq!(
            goal(&json!({"op": "get"})).unwrap(),
            Some(json!({"goal": null, "state": null}))
        );
        assert_eq!(
            goal(&json!({"op": "resume"})).unwrap_err().message,
            "No paused goal to resume."
        );
        assert_eq!(
            goal(&json!({"op": "create"})).unwrap_err().message,
            "objective is required when op=create"
        );
        assert_eq!(
            goal(&json!({"op": "create", "objective": "x", "token_budget": 0}))
                .unwrap_err()
                .message,
            "token_budget must be a positive integer when provided"
        );
        assert_eq!(
            goal(&json!({"op": "nope"})).unwrap_err().message,
            "Unknown goal op: nope"
        );
        assert_eq!(
            set_host_tools(
                &json!({"tools": [{"name": " ", "description": "d", "parameters": {}}]})
            )
            .unwrap_err()
            .message,
            "Host tool at index 0 must provide a non-empty name"
        );
        assert_eq!(
            set_host_tools(
                &json!({"tools": [{"name": "t", "description": "d", "parameters": []}]})
            )
            .unwrap_err()
            .message,
            "Host tool \"t\" must provide a JSON Schema object"
        );
        assert_eq!(
            set_host_tools(&json!({"tools": []})).unwrap(),
            Some(json!({"toolNames": []}))
        );
        assert_eq!(
            set_host_uri_schemes(&json!({"schemes": [{"scheme": "1bad"}]}))
                .unwrap_err()
                .message,
            "Host URI scheme contains invalid characters: 1bad"
        );
        assert_eq!(
            subagent_messages(&json!({})).unwrap_err().message,
            "get_subagent_messages requires subagentId or sessionFile"
        );
        assert_eq!(
            subagent_messages(&json!({"subagentId": "s1"}))
                .unwrap_err()
                .message,
            "Unknown subagent or session file unavailable: s1"
        );
        assert!(text_cursor(&json!({"text": "héllo", "cursor": 5})).is_ok());
        assert_eq!(
            text_cursor(&json!({"text": "hi", "cursor": 3}))
                .unwrap_err()
                .message,
            "cursor must be an integer UTF-16 offset within text"
        );
    }
}

#[cfg(test)]
#[allow(clippy::disallowed_methods)] // tests: throwaway runtimes drive async code
mod ask_ui_tests {
    use super::*;

    /// The pending-dialog registry is process-global; serialize the tests.
    static TEST_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

    #[tokio::test]
    async fn ask_ui_resolves_from_extension_ui_response() {
        let _guard = TEST_LOCK.lock().await;
        let questions = json!([{"id": "q1", "schema": {}}]);
        let ui = RpcAskUi;
        ASK_DIALOG_ENABLED.store(true, Ordering::SeqCst);
        // `RpcAskUi` is a zero-sized handle; build a fresh one inside the task
        // instead of moving the outer `ui` (used for assertions below).
        let ui_task = tokio::spawn(async { AskUi::ask(&RpcAskUi, questions, None).await });
        // Wait for the request frame, then answer it.
        let id = tokio::time::timeout(std::time::Duration::from_secs(2), async {
            loop {
                tokio::task::yield_now().await;
                let pending = lock(&PENDING_UI_REQUESTS);
                if let Some(key) = pending.keys().next().cloned() {
                    return key;
                }
            }
        })
        .await
        .expect("pending dialog registered");
        let response = json!({"type": "extension_ui_response", "id": id,
            "answers": [{"id": "q1", "value": "ok"}]});
        let tx = lock(&PENDING_UI_REQUESTS).remove(&id).expect("waiter");
        let _ = tx.send(response);
        let result = ui_task.await.unwrap().unwrap().unwrap();
        assert_eq!(result["answers"][0]["value"], json!("ok"));
        ASK_DIALOG_ENABLED.store(false, Ordering::SeqCst);
        assert!(!AskUi::has_ask_dialog(&ui));
    }

    #[tokio::test]
    async fn ask_ui_times_out_with_timed_out_frame() {
        let _guard = TEST_LOCK.lock().await;
        let mut frame = Map::new();
        frame.insert("questions".into(), json!([]));
        frame.insert("timeout".into(), json!(50));
        let result = RpcAskUi::request("ask", frame).await.unwrap().unwrap();
        assert_eq!(result["cancelled"], json!(true));
        assert_eq!(result["timedOut"], json!(true));
    }
}
