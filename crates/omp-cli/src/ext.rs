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
    /// `session_stop` continuation bookkeeping (TS agent-session fields).
    stop: Arc<Mutex<StopState>>,
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
            stop: Arc::new(Mutex::new(StopState::default())),
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
        rt.agent
            .subscribe(move |event, _| {
                let host = host.clone();
                let turn = turn.clone();
                let stop = stop.clone();
                Box::pin(async move {
                    forward_event(&host, &turn, &stop, &event).await;
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
    let (hook_active, continuations) = {
        let s = stop.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
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
            eprintln!("omp-rs: extension session_stop: {e:#}");
            return Vec::new();
        }
    };
    let Some(text) = stop_continuation_text(&outcome) else {
        *stop.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = StopState::default();
        return Vec::new();
    };
    {
        let mut s = stop.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if !outcome.block {
            if continuations >= SESSION_STOP_CONTINUATION_CAP {
                eprintln!("omp-rs: session_stop continuation cap reached");
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

#[allow(clippy::too_many_lines)] // one arm per `AgentEvent` variant; splitting adds indirection.
async fn forward_event(
    host: &ExtHost,
    turn: &AtomicU64,
    stop: &Mutex<StopState>,
    event: &AgentEvent,
) {
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(0));
    match event {
        AgentEvent::AgentStart => {
            // A new run resets the session_stop continuation bookkeeping
            // (TS `#resetSessionStopContinuationState` on generation change).
            *stop.lock().unwrap_or_else(std::sync::PoisonError::into_inner) =
                StopState::default();
            emit_silent(host.emit_agent_start(None).await);
        }
        AgentEvent::AgentEnd { messages } => {
            let values: Vec<Value> = messages.iter().map(message_value).collect();
            // `session_stop` itself is emitted from the continuation hook
            // (see `subscribe_lifecycle`), where its block/continue outcome can
            // still keep the run alive.
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

#[cfg(test)]
mod tests {
    use super::*;
    use pa_agent::abort::AbortController;

    fn sh_host(dir: &std::path::Path, body: &str, env: Vec<(String, String)>) -> Arc<ExtHost> {
        let script = dir.join("fake-host.sh");
        std::fs::write(&script, body).unwrap();
        ExtHost::spawn(&HostConfig {
            bun: "/bin/sh".into(),
            host_script: script,
            trace: None,
            handler_timeout_ms: None,
            env,
        })
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
        let host = sh_host(dir.path(), CONTINUE_HOST, Vec::new());
        let stop = Mutex::new(StopState::default());
        let turn = AtomicU64::new(2);
        let messages =
            session_stop_continuation(&host, &stop, "s1", "/s.jsonl", &turn, &[]).await;
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
        let host = sh_host(dir.path(), CONTINUE_HOST, Vec::new());
        let stop = Mutex::new(StopState::default());
        let turn = AtomicU64::new(0);
        for i in 0..SESSION_STOP_CONTINUATION_CAP {
            let messages =
                session_stop_continuation(&host, &stop, "s1", "/s.jsonl", &turn, &[]).await;
            assert_eq!(messages.len(), 1, "continuation {i} scheduled");
        }
        let messages =
            session_stop_continuation(&host, &stop, "s1", "/s.jsonl", &turn, &[]).await;
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
        );
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
}
