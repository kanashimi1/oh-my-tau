//! `--mode rpc`: omp's JSON-lines RPC protocol (`modes/rpc`) on stdin/stdout.
//!
//! First frame `ready`; every command is answered with a `response` frame
//! (`{id?, type, command, success, data?|error?, code?}`); agent events,
//! `prompt_result` and `session_settled` frames are interleaved. Commands that
//! depend on subsystems omp-rs lacks answer `success: false` with an omp-style
//! error message (see `unsupported`).

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use pa_agent::abort::AbortController;
use pa_agent::agent::{Agent, QueueMode};
use pa_agent::types::ThinkingLevel;
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

const MAX_FRAME_BYTES: u64 = 1_048_576;
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

fn ok(data: Value) -> CmdResult {
    Ok(Some(data))
}

fn none() -> CmdResult {
    Ok(None)
}

async fn cycle_model(rt: &Runtime) -> CmdResult {
    let info = rt.model_info().await;
    let models = available_models(None);
    if models.len() < 2 {
        return ok(Value::Null);
    }
    let idx = models
        .iter()
        .position(|m| info.get("provider") == Some(&json!(m.provider.to_string())) && info.get("id") == Some(&json!(m.id)))
        .unwrap_or(0);
    let next = &models[(idx + 1) % models.len()];
    let model = rt.set_model(&next.provider.to_string(), &next.id).await?;
    let level = rt.agent.state().await.thinking_level;
    ok(json!({"model": model, "thinkingLevel": level_name(level), "isScoped": false}))
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
                    .filter_map(|p| (p.get("type")? == "text").then(|| p.get("text")?.as_str().map(str::to_string))?)
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
    let known: HashSet<&str> = entries.iter().filter_map(|e| e.get("id").and_then(Value::as_str)).collect();
    build(entries, None, &known)
}

async fn run_bash(rt: &Arc<Runtime>, command: &str) -> CmdResult {
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
            rt.prompt(cmd.get("id").cloned(), message, cmd.get("streamingBehavior").and_then(Value::as_str))
                .await?;
            none()
        }
        "steer" | "follow_up" => {
            let message = str_field(cmd, "message")?;
            if kind == "steer" {
                rt.steer(message);
            } else {
                rt.follow_up(message);
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
            rt.abort().await;
            let task_rt = rt.clone();
            let id = cmd.get("id").cloned();
            tokio::spawn(async move {
                if let Err(e) = task_rt.prompt(id.clone(), &message, None).await {
                    task_rt.send(response(id.as_ref(), "abort_and_prompt", Err(CmdError::new(e.message.clone()))));
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
        "new_session" => {
            rt.new_session(None).await?;
            ok(json!({"cancelled": false}))
        }
        "switch_session" => {
            let path = PathBuf::from(str_field(cmd, "sessionPath")?);
            if !path.is_file() {
                return Err(CmdError::new(format!("Session file not found: {}", path.display())));
            }
            rt.load_session(&path, None).await?;
            ok(json!({"cancelled": false}))
        }
        "open_session" => {
            let dir = PathBuf::from(str_field(cmd, "sessionDir")?);
            let resumed = match newest_session(&dir) {
                Some(path) => {
                    rt.load_session(&path, Some(&dir)).await?;
                    true
                }
                None => {
                    rt.new_session(Some(&dir)).await?;
                    false
                }
            };
            ok(json!({
                "cancelled": false,
                "resumed": resumed,
                "sessionId": rt.session_id(),
                "sessionFile": rt.session_file().display().to_string(),
            }))
        }
        "branch" | "fork" => Err(unsupported("Session tree navigation")),
        "get_state" => ok(rt.state().await),
        "set_fast_mode" => {
            if bool_field(cmd, "enabled")? {
                Err(CmdError::new("Fast mode is unavailable for the current model."))
            } else {
                ok(json!({"enabled": false, "active": false}))
            }
        }
        "goal" => Err(unsupported("Goal mode")),
        "set_ask_dialog" => {
            let enabled = rt.set(|s| s.ask_dialog(cmd.get("enabled").and_then(Value::as_bool) == Some(true)));
            ok(json!({"enabled": enabled}))
        }
        "get_available_commands" => ok(json!({"commands": []})),
        "get_entries" => {
            let entries = rt.entries().map_err(CmdError::from)?;
            let leaf = entries.last().and_then(|e| e.get("id").cloned()).unwrap_or(Value::Null);
            let selected = match cmd.get("since").and_then(Value::as_str) {
                None => entries,
                Some(since) => {
                    let idx = entries
                        .iter()
                        .position(|e| e.get("id").and_then(Value::as_str) == Some(since))
                        .ok_or_else(|| CmdError::coded(format!("Unknown entry id: {since}"), "unknown_since"))?;
                    entries[idx + 1..].to_vec()
                }
            };
            ok(json!({"entries": selected, "leafId": leaf}))
        }
        "get_tree" => {
            let entries = rt.entries().map_err(CmdError::from)?;
            let leaf = entries.last().and_then(|e| e.get("id").cloned()).unwrap_or(Value::Null);
            ok(json!({"tree": tree(&entries), "leafId": leaf}))
        }
        "set_todos" => {
            let phases = cmd.get("phases").cloned().filter(Value::is_array).ok_or_else(|| CmdError::new("phases must be an array"))?;
            rt.set(|s| s.todo_phases(phases));
            ok(json!({"todoPhases": rt.set(|s| s.todo_phases_value())}))
        }
        "set_host_tools" => Err(unsupported("Host tools")),
        "set_host_uri_schemes" => Err(unsupported("Host URI schemes")),
        "set_subagent_subscription" => {
            let level = str_field(cmd, "level")?;
            if !matches!(level, "off" | "progress" | "events") {
                return Err(CmdError::new("level must be \"off\", \"progress\" or \"events\""));
            }
            rt.set(|s| s.subagent_level(level));
            ok(json!({"level": level}))
        }
        "set_event_filter" => {
            let events = cmd.get("events");
            let filter = match events {
                Some(Value::Null) => None,
                Some(Value::Array(a)) => Some(a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect::<HashSet<_>>()),
                _ => return Err(CmdError::new("events must be an array of strings or null")),
            };
            let delta = match cmd.get("messageUpdates").and_then(Value::as_str) {
                None | Some("full") => false,
                Some("delta") => true,
                Some(_) => return Err(CmdError::new("messageUpdates must be \"full\" or \"delta\"")),
            };
            rt.set(|s| s.event_filter(filter, delta));
            let mut data = json!({"events": events.cloned().unwrap_or(Value::Null)});
            if let Some(m) = cmd.get("messageUpdates") {
                data["messageUpdates"] = m.clone();
            }
            ok(data)
        }
        "get_subagents" => ok(json!({"subagents": []})),
        "get_subagent_messages" => Err(CmdError::new("Subagent not found")),
        "cancel_subagent" => ok(json!({"cancelled": false})),
        "steer_subagent" => Err(CmdError::new("Subagent not found")),
        "live_start" | "live_stop" | "live_mute" => Err(unsupported("Live voice")),
        "set_model" => {
            let provider = str_field(cmd, "provider")?;
            let model_id = str_field(cmd, "modelId")?;
            ok(rt.set_model(provider, model_id).await?)
        }
        "cycle_model" => cycle_model(rt).await,
        "get_available_models" => {
            let current = {
                let info = rt.model_info().await;
                info.get("provider").zip(info.get("id")).map(|(p, i)| (p.clone(), i.clone()))
            };
            let mut models: Vec<Value> = available_models(None).iter().map(model_info).collect();
            if let Some((p, i)) = current {
                if !models.iter().any(|m| m.get("provider") == Some(&p) && m.get("id") == Some(&i)) {
                    models.insert(0, rt.model_info().await);
                }
            }
            ok(json!({"models": models}))
        }
        "set_thinking_level" => {
            let level = str_field(cmd, "level")?;
            let parsed = parse_level(level).or_else(|| (level == "inherit").then_some(ThinkingLevel::Medium));
            let Some(parsed) = parsed else {
                return Err(CmdError::new(format!("Invalid thinking level: {level}")));
            };
            rt.set_thinking_level(parsed).await;
            none()
        }
        "cycle_thinking_level" => {
            let levels = rt.available_thinking_levels().await;
            if levels.len() < 2 {
                return ok(Value::Null);
            }
            let current = level_name(rt.agent.state().await.thinking_level);
            let idx = levels.iter().position(|l| *l == current).unwrap_or(0);
            let next = levels[(idx + 1) % levels.len()];
            if let Some(level) = parse_level(next) {
                rt.set_thinking_level(level).await;
            }
            ok(json!({"level": next}))
        }
        "get_available_thinking_levels" => ok(json!({"levels": rt.available_thinking_levels().await})),
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
        "compact" => Err(unsupported("Compaction")),
        "set_auto_compaction" => {
            let enabled = bool_field(cmd, "enabled")?;
            rt.set(|s| s.auto_compaction(enabled));
            none()
        }
        "set_cache_warming" => {
            let mode = cmd.get("mode").and_then(Value::as_str).unwrap_or("");
            if !CACHE_WARMING_MODES.contains(&mode) {
                let shown = cmd.get("mode").map_or_else(|| "undefined".to_string(), |m| m.as_str().map_or_else(|| m.to_string(), str::to_string));
                return Err(CmdError::new(format!("Invalid cache warming mode: {shown}")));
            }
            rt.set(|s| s.cache_warming(mode));
            ok(json!({"mode": mode}))
        }
        "set_auto_retry" => {
            let enabled = bool_field(cmd, "enabled")?;
            rt.set(|s| s.auto_retry(enabled));
            none()
        }
        "abort_retry" => none(),
        "abort_bash" => {
            rt.set(|s| s.abort_bash());
            none()
        }
        "get_session_stats" => ok(rt.session_stats().await),
        "export_html" => Err(unsupported("HTML export")),
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
            rt.set_session_name(&name).map_err(CmdError::from)?;
            none()
        }
        "handoff" => Err(unsupported("Handoff")),
        "get_messages" => ok(json!({"messages": rt.messages().await})),
        "get_messages_page" => {
            if rt.agent.state().await.is_streaming {
                return Err(CmdError::coded("Cannot page messages while the session is changing", "session_busy"));
            }
            let messages = rt.messages().await;
            let entries = rt.entries().map_err(CmdError::from)?;
            let leaf = entries.last().and_then(|e| e.get("id").and_then(Value::as_str)).map(str::to_string);
            let session_id = rt.session_id();
            let snapshot = paging::Snapshot { session_id: &session_id, leaf_id: leaf.as_deref(), message_count: messages.len() };
            let limit = cmd.get("limit").and_then(Value::as_f64);
            let limit = match limit {
                None => None,
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                Some(n) if n.fract() == 0.0 && n >= 0.0 => Some(n as usize),
                Some(_) => Some(0),
            };
            paging::page(&messages, &snapshot, cmd.get("cursor").and_then(Value::as_str), limit)
                .map(Some)
                .map_err(|e| CmdError { message: e.message, code: e.code })
        }
        "get_login_providers" => Err(unsupported("OAuth login")),
        "login" => Err(unsupported("OAuth login")),
        "predict_word" => ok(json!({"suffix": null})),
        "predict_word_feedback" => none(),
        other => Err(CmdError::new(format!("Unknown command: {other}"))),
    }
}

/// Run the RPC loop until stdin closes.
///
/// # Errors
/// Fails on stdin read errors.
pub async fn run(agent: Agent, target: Target, cwd: PathBuf, persist: Persistence) -> Result<()> {
    let (tx, mut rx) = mpsc::unbounded_channel::<Value>();
    let agent = Arc::new(agent);
    let rt = Runtime::start(agent.clone(), target, cwd, persist, tx.clone()).await;

    let writer = tokio::spawn(async move {
        let mut out = tokio::io::stdout();
        while let Some(v) = rx.recv().await {
            let mut line = v.to_string();
            line.push('\n');
            if out.write_all(line.as_bytes()).await.is_err() || out.flush().await.is_err() {
                break;
            }
        }
    });

    rt.send(json!({
        "type": "ready",
        "protocolVersion": 1,
        "supportedProtocolVersions": [1, 2],
        "maxFrameBytes": MAX_FRAME_BYTES,
        "maxReassembledFrameBytes": MAX_REASSEMBLED_BYTES,
    }));
    rt.send(json!({"type": "available_commands_update", "commands": []}));

    let mut lines = BufReader::new(tokio::io::stdin()).lines();
    let mut background = tokio::task::JoinSet::new();
    while let Some(line) = lines.next_line().await? {
        if line.trim().is_empty() {
            continue;
        }
        let cmd = match serde_json::from_str::<Value>(&line) {
            Ok(v) if v.is_object() => v,
            Ok(_) => {
                rt.send(response(None, "parse", Err(CmdError::new("Failed to parse command: expected a JSON object"))));
                continue;
            }
            Err(e) => {
                rt.send(response(None, "parse", Err(CmdError::new(format!("Failed to parse command: {e}")))));
                continue;
            }
        };
        let kind = cmd.get("type").and_then(Value::as_str).unwrap_or("");
        if matches!(
            kind,
            "extension_ui_response" | "host_tool_result" | "host_tool_update" | "host_uri_result"
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
        let reply = handle(&rt, &cmd).await;
        rt.send(reply);
    }
    // stdin closed: let accepted work finish and flush its frames.
    while background.join_next().await.is_some() {}
    agent.wait_for_idle().await;
    drop(rt);
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    writer.abort();
    Ok(())
}
