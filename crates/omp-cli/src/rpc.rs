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

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::sync::{LazyLock, Mutex};

use anyhow::Result;
use pa_agent::abort::AbortController;
use pa_agent::agent::{Agent, QueueMode};
use pa_agent::types::ThinkingLevel;
use pa_agent::BoxFut;
use serde_json::{json, Map, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
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

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
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
                let id = e.get("id").and_then(Value::as_str);
                json!({"entry": e, "children": build(entries, id, known)})
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
            if cmd
                .get("message")
                .and_then(Value::as_str)
                .is_none_or(|m| m.trim().is_empty())
            {
                return Err(CmdError::new("`message` is required for steer_subagent."));
            }
            Err(CmdError::new(format!("Subagent not running: {id}")))
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
        "abort_bash" => {
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
        "get_messages_page" => {
            if rt.agent.is_streaming().await {
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
            paging::page(
                &messages,
                &snapshot,
                cmd.get("cursor").and_then(Value::as_str),
                limit,
            )
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
}

impl FrameEncoder {
    fn new() -> Self {
        Self {
            version: 1,
            streamed: Vec::new(),
            chunks: 0,
        }
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

/// Hook run once the runtime exists, before the `ready` frame (extension host start-up).
pub type OnStart = Box<
    dyn FnOnce(Arc<Runtime>) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>>
        + Send,
>;

/// Run the RPC loop until stdin closes.
///
/// # Errors
/// Fails on stdin read errors.
pub async fn run(agent: Agent, target: Target, cwd: PathBuf, persist: Persistence) -> Result<()> {
    run_with(agent, target, cwd, persist, None).await
}

/// [`run`] with a start-up hook (see [`OnStart`]).
///
/// # Errors
/// Fails on stdin read errors.
#[allow(clippy::too_many_lines)] // the RPC dispatch loop; one branch per wire frame type, mirrors omp's handler table
pub async fn run_with(
    agent: Agent,
    target: Target,
    cwd: PathBuf,
    persist: Persistence,
    on_start: Option<OnStart>,
) -> Result<()> {
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
    let writer = tokio::spawn(async move {
        let mut out = tokio::io::stdout();
        let mut encoder = FrameEncoder::new();
        while let Some(v) = rx.recv().await {
            let mut buf = String::new();
            for line in encoder.encode(&v) {
                buf.push_str(&line);
                buf.push('\n');
            }
            if out.write_all(buf.as_bytes()).await.is_err() || out.flush().await.is_err() {
                break;
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
    let mut background = tokio::task::JoinSet::new();
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
    while let Some(line) = lines.next_line().await? {
        if line.trim().is_empty() {
            continue;
        }
        let cmd = match serde_json::from_str::<Value>(&line) {
            Ok(v) if v.is_object() => v,
            Ok(_) => {
                rt.send(response(
                    None,
                    "parse",
                    Err(CmdError::new(
                        "Failed to parse command: expected a JSON object",
                    )),
                ));
                continue;
            }
            Err(e) => {
                rt.send(response(
                    None,
                    "parse",
                    Err(CmdError::new(format!("Failed to parse command: {e}"))),
                ));
                continue;
            }
        };
        let kind = cmd.get("type").and_then(Value::as_str).unwrap_or("");
        if kind == "extension_ui_response" {
            // Resolve a pending ask-UI dialog (see `RpcAskUi`).
            if let Some(id) = cmd.get("id").and_then(Value::as_str) {
                resolve_pending_ui_request(id, cmd.clone());
            }
            continue;
        }
        if matches!(
            kind,
            "host_tool_result" | "host_tool_update" | "host_uri_result"
        ) {
            continue; // side-channel frames for sub-protocols this server never opens
        }
        if kind == "bash" {
            let rt = rt.clone();
            background.spawn(async move {
                let result = match str_field(&cmd, "command") {
                    Ok(c) => run_bash(&rt, c).await,
                    Err(e) => Err(e),
                };
                rt.send(response(cmd.get("id"), "bash", result));
            });
            continue;
        }
        if kind == "abort" && busy.load(std::sync::atomic::Ordering::SeqCst) {
            // A command is in flight (possibly waiting on a dialog): abort the run now
            // instead of queueing behind it.
            let reply = handle(&rt, &cmd).await;
            rt.send(reply);
            continue;
        }
        let _ = cmd_tx.send(cmd);
    }
    // stdin closed: let accepted work finish and flush its frames.
    drop(cmd_tx);
    let _ = worker.await;
    while background.join_next().await.is_some() {}
    agent.wait_for_idle().await;
    drop(rt);
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    writer.abort();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
