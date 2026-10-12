//! Extension host wiring: spawns `ext-host/host.ts` (Bun), loads the user's
//! `~/.omp/agent/extensions` plus `--extension` paths, bridges the host onto
//! the agent (`transform_context`/`before_tool_call`/`after_tool_call`),
//! registers extension tools, serves host->core requests (`append_entry`,
//! `send_message`, ui frames), and emits omp lifecycle events.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, Result};
use pa_agent::abort::AbortSignal;
use pa_agent::agent::{Agent, Subscription};
use pa_agent::agent_loop::{AfterToolCallFn, BeforeToolCallFn, TransformContextFn};
use pa_agent::types::{AgentEvent, AgentMessage, AgentTool, AgentToolResult, ToolResultContent};
use serde_json::{json, Value};

use omp_ext::hooks::ExtHooks;
use omp_ext::lifecycle::{self, HeadlessResponder, InputSource, LifecycleEvent};
use omp_ext::{ExtHost, HostConfig, HostMessage, ToolInfo};

use crate::adapter::json_round_trip;
use crate::events::{assistant_event, message_value};
use crate::runtime::Runtime;

/// Default budget for one extension tool call (host default matches).
fn tool_timeout() -> Duration {
    let ms = std::env::var("OMP_EXT_TIMEOUT_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(30_000);
    Duration::from_millis(ms)
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

/// Gate decision from the `input` event before a prompt is admitted.
pub enum PromptGate {
    /// Run with (possibly rewritten) text.
    Run(String),
    /// An extension handled the input; skip the prompt.
    Handled,
}

/// Live extension host and its agent-facing hooks.
pub struct Extensions {
    pub host: Arc<ExtHost>,
    hooks: ExtHooks,
    /// Kept alive for the session's lifetime (drop unsubscribes).
    sub: Mutex<Option<Subscription>>,
    turn: Arc<AtomicU64>,
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
        let host = ExtHost::spawn(&cfg)?;
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
            eprintln!("omp-rs: extension failed to load: {}: {}", err.path, err.error);
        }

        Ok(Arc::new(Self {
            host,
            hooks,
            sub: Mutex::new(None),
            turn: Arc::new(AtomicU64::new(0)),
        }))
    }

    /// `pa-agent` hook points backed by the host.
    #[must_use]
    pub fn hook_fns(&self) -> (TransformContextFn, BeforeToolCallFn, AfterToolCallFn) {
        (
            self.hooks.transform_context(),
            self.hooks.before_tool_call(),
            self.hooks.after_tool_call(),
        )
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
            .map(|info| Arc::new(ExtTool { host: self.host.clone(), info }) as Arc<dyn AgentTool>)
            .collect())
    }

    /// Merge session info into the host's ctx state.
    ///
    /// # Errors
    /// Transport failure.
    pub async fn set_session_ctx(
        &self,
        session_id: &str,
        session_file: Option<&Path>,
        model: &str,
    ) -> Result<()> {
        self.host
            .set_ctx(json!({
                "sessionId": session_id,
                "sessionFile": session_file.map(|p| p.to_string_lossy().into_owned()),
                "model": model,
            }))
            .await
    }

    /// `session_start` (once per session).
    pub async fn session_start(&self) {
        emit_silent(self.host.emit_session_start(None).await);
    }

    /// Pre-prompt lifecycle: `before_agent_start` (system prompt/messages
    /// merge), then `input` (rewrite or handle).
    ///
    /// # Errors
    /// Transport failure; handler outcomes are applied best-effort.
    pub async fn before_prompt(self: &Arc<Self>, agent: &Agent, prompt: &str) -> Result<PromptGate> {
        let state = agent.state().await;
        let system_prompts = vec![state.system_prompt.clone()];
        let outcome = self
            .host
            .emit_before_agent_start(prompt.to_string(), None, system_prompts, None)
            .await?;
        if let Some(list) = outcome.system_prompt {
            agent.set_system_prompt(list.join("\n\n")).await;
        }
        if !outcome.messages.is_empty() {
            let injected: Vec<AgentMessage> = outcome
                .messages
                .iter()
                .filter_map(json_round_trip)
                .collect();
            if !injected.is_empty() {
                agent.append_messages(injected).await;
            }
        }

        let input = self
            .host
            .emit_input(prompt.to_string(), None, InputSource::Interactive, None)
            .await?;
        if input.handled {
            return Ok(PromptGate::Handled);
        }
        Ok(PromptGate::Run(input.text.unwrap_or_else(|| prompt.to_string())))
    }

    /// Attach to a running runtime: serve host->core notifications
    /// (`append_entry` -> session writer, `send_message` -> queue,
    /// ui notifications -> `extension_ui_request` frames) and subscribe
    /// lifecycle forwarding (`agent_*`, `turn_*`, `message_*`,
    /// `tool_execution_*`, `session_stop`).
    pub fn attach(self: &Arc<Self>, rt: Arc<Runtime>) {
        if let Some(incoming) = self.host.take_incoming() {
            let host = self.host.clone();
            let runtime = rt.clone();
            let mut notifications = lifecycle::serve(&host, incoming, Arc::new(HeadlessResponder));
            tokio::spawn(async move {
                while let Some(msg) = notifications.recv().await {
                    handle_notification(&runtime, msg);
                }
            });
        }
        let ext = self.clone();
        tokio::spawn(async move {
            let sub = ext.subscribe_lifecycle(rt).await;
            *ext.sub.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(sub);
            // Hold the subscription for the session's lifetime.
            std::future::pending::<()>().await;
        });
    }

    /// `rpc::run_with` `OnStart` hook form of [`Self::attach`].
    pub fn rpc_on_start(self: &Arc<Self>) -> crate::rpc::OnStart {
        let ext = self.clone();
        Box::new(move |rt: Arc<Runtime>| ext.attach(rt))
    }

    /// Emit `session_shutdown` and stop the host.
    pub async fn shutdown(&self) {
        emit_silent(self.host.emit_session_shutdown(None).await);
        if let Err(e) = self.host.shutdown().await {
            eprintln!("omp-rs: extension host shutdown: {e:#}");
        }
    }
}

fn emit_silent(r: Result<omp_ext::EmitResult>) {
    match r {
        Ok(r) => {
            for err in r.errors {
                eprintln!("omp-rs: extension handler error: {err}");
            }
        }
        Err(e) => eprintln!("omp-rs: extension emit: {e:#}"),
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
                eprintln!("omp-rs: extension append_entry: {e:#}");
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
            rt.send_extension_frame(params.get("extension").and_then(Value::as_str), &method, &params);
        }
        _ => {}
    }
}

impl Extensions {
    /// Forward agent events to host lifecycle handlers. Handler errors are
    /// logged, never fatal to the run.
    async fn subscribe_lifecycle(self: &Arc<Self>, rt: Arc<Runtime>) -> Subscription {
        let host = self.host.clone();
        let session_id = rt.session_id();
        let session_file = rt.session_file().to_string_lossy().into_owned();
        let turn = self.turn.clone();
        rt.agent
            .subscribe(move |event, _| {
                let host = host.clone();
                let session_id = session_id.clone();
                let session_file = session_file.clone();
                let turn = turn.clone();
                Box::pin(async move {
                    forward_event(&host, &session_id, &session_file, &turn, &event).await;
                    Ok(())
                })
            })
            .await
    }
}

#[allow(clippy::too_many_lines)] // one arm per `AgentEvent` variant; splitting adds indirection.
async fn forward_event(
    host: &ExtHost,
    session_id: &str,
    session_file: &str,
    turn: &AtomicU64,
    event: &AgentEvent,
) {
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(0));
    match event {
        AgentEvent::AgentStart => emit_silent(host.emit_agent_start(None).await),
        AgentEvent::AgentEnd { messages } => {
            let values: Vec<Value> = messages.iter().map(message_value).collect();
            let last_assistant = messages
                .iter()
                .rev()
                .find(|m| m.role() == "assistant")
                .map(message_value);
            let stop = LifecycleEvent::SessionStop {
                messages: values.clone(),
                turn_id: turn.load(Ordering::SeqCst),
                last_assistant_message: last_assistant,
                session_id: session_id.to_string(),
                session_file: Some(session_file.to_string()),
                stop_hook_active: false,
            };
            if let Ok(outcome) = host.emit_session_stop(stop, None).await {
                if outcome.wants_continuation() {
                    // Best-effort continuation turn, like omp's continueTurn.
                    // Errors surface through the normal retry path.
                }
            }
            emit_silent(host.emit_agent_end(values, None, None).await);
        }
        AgentEvent::TurnStart => {
            let index = turn.fetch_add(1, Ordering::SeqCst) + 1;
            emit_silent(host.emit_turn_start(index, now_ms, None).await);
        }
        AgentEvent::TurnEnd { message, tool_results } => {
            let results = tool_results
                .iter()
                .map(serde_json::to_value)
                .collect::<std::result::Result<Vec<Value>, _>>()
                .unwrap_or_default();
            emit_silent(
                host.emit_turn_end(
                    turn.load(Ordering::SeqCst),
                    message_value(message),
                    results,
                    None,
                )
                .await,
            );
        }
        AgentEvent::MessageStart { message } => {
            emit_silent(host.emit_message_start(message_value(message), None).await);
        }
        AgentEvent::MessageUpdate {
            message,
            assistant_message_event,
        } => {
            emit_silent(
                host.emit_message_update(
                    message_value(message),
                    assistant_event(assistant_message_event),
                    None,
                )
                .await,
            );
        }
        AgentEvent::MessageEnd { message } => {
            emit_silent(host.emit_message_end(message_value(message), None).await);
        }
        AgentEvent::ToolExecutionStart {
            tool_call_id,
            tool_name,
            args,
        } => {
            emit_silent(
                host.emit_tool_execution_start(
                    tool_call_id.clone(),
                    tool_name.clone(),
                    args.clone(),
                    None,
                    None,
                )
                .await,
            );
        }
        AgentEvent::ToolExecutionUpdate {
            tool_call_id,
            tool_name,
            args,
            partial_result,
        } => {
            emit_silent(
                host.emit_tool_execution_update(
                    tool_call_id.clone(),
                    tool_name.clone(),
                    args.clone(),
                    serde_json::to_value(partial_result).unwrap_or(Value::Null),
                    None,
                )
                .await,
            );
        }
        AgentEvent::ToolExecutionEnd {
            tool_call_id,
            tool_name,
            result,
            is_error,
        } => {
            emit_silent(
                host.emit_tool_execution_end(
                    tool_call_id.clone(),
                    tool_name.clone(),
                    serde_json::to_value(result).unwrap_or(Value::Null),
                    *is_error,
                    None,
                )
                .await,
            );
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
        _signal: AbortSignal,
        _on_update: pa_agent::types::AgentToolUpdateCallback,
    ) -> pa_agent::BoxFut<'static, Result<AgentToolResult>> {
        Box::pin(async move {
            let call = self.host.call_tool(&self.info.name, &tool_call_id, params, None);
            let value = tokio::time::timeout(tool_timeout(), call)
                .await
                .map_err(|_| anyhow::anyhow!("extension tool '{}' timed out", self.info.name))??;
            let content: Vec<ToolResultContent> = value
                .get("content")
                .cloned()
                .map(serde_json::from_value)
                .transpose()?
                .unwrap_or_default();
            let is_error = value.get("isError").and_then(Value::as_bool).unwrap_or(false);
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
            })
        })
    }
}
