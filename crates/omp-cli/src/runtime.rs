//! Session runtime behind `--mode rpc`: the agent plus everything omp's
//! `AgentSession` adds around it (model switching, queues, session files,
//! stats, prompt-result bookkeeping, event forwarding).

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};

use anyhow::{Context, Result};
use pa_agent::abort::AbortController;
use pa_agent::agent::{Agent, QueueMode};
use pa_agent::types::{AgentEvent, AgentMessage, AssistantContent, Message, StopReason, ThinkingLevel, UserContent, UserPart};
use pi_edit::EditMode;
use serde_json::{json, Map, Value};
use tokio::sync::mpsc::UnboundedSender;

use crate::adapter::{json_round_trip, map_thinking_level, ProviderTarget};
use crate::agent::Target;
use crate::events::Framer;
use crate::persist::Persistence;
use crate::tools::{builtin_tools, edit_mode_for};

/// A command failure: model/host-facing message plus optional machine code.
#[derive(Debug)]
pub struct CmdError {
    pub message: String,
    pub code: Option<&'static str>,
}

impl CmdError {
    pub fn new(message: impl Into<String>) -> Self {
        Self { message: message.into(), code: None }
    }

    #[must_use]
    pub fn coded(message: impl Into<String>, code: &'static str) -> Self {
        Self { message: message.into(), code: Some(code) }
    }
}

impl From<anyhow::Error> for CmdError {
    fn from(e: anyhow::Error) -> Self {
        Self::new(format!("{e:#}"))
    }
}

/// `Ok(None)` = success without `data`; `Ok(Some(Value::Null))` = `data: null`.
pub type CmdResult = Result<Option<Value>, CmdError>;

#[derive(Debug, Clone, PartialEq)]
struct RunOutcome {
    status: &'static str,
    error: Option<Value>,
}

struct Ticket {
    id: Option<Value>,
    starts_at_begin: u64,
    /// Queued into a live run: reports at the first terminal end after the queue drains.
    waiting: bool,
}

#[derive(Default)]
struct PromptState {
    agent_starts: u64,
    open: Vec<Ticket>,
    /// Agent activity since the last `session_settled`.
    active: bool,
}

struct Settings {
    session_name: Option<String>,
    todo_phases: Value,
    auto_compaction: bool,
    auto_retry: bool,
    interrupt_mode: String,
    cache_warming: String,
    ask_dialog: bool,
    subagent_level: String,
    /// `None` forwards every event type.
    event_filter: Option<HashSet<String>>,
    delta_updates: bool,
    edit_mode: EditMode,
    last_queue: (Vec<String>, Vec<String>),
    running_bash: Option<AbortController>,
}

pub struct Runtime {
    pub agent: Arc<Agent>,
    target: Target,
    pub cwd: PathBuf,
    persist: Persistence,
    out: UnboundedSender<Value>,
    prompts: Mutex<PromptState>,
    settings: Mutex<Settings>,
    framer: Mutex<Framer>,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

pub fn level_name(level: ThinkingLevel) -> &'static str {
    match level {
        ThinkingLevel::Off => "off",
        ThinkingLevel::Minimal => "minimal",
        ThinkingLevel::Low => "low",
        ThinkingLevel::Medium => "medium",
        ThinkingLevel::High => "high",
        ThinkingLevel::Xhigh => "xhigh",
        ThinkingLevel::Max => "max",
    }
}

pub fn parse_level(s: &str) -> Option<ThinkingLevel> {
    Some(match s {
        "off" => ThinkingLevel::Off,
        "minimal" => ThinkingLevel::Minimal,
        "low" => ThinkingLevel::Low,
        "medium" => ThinkingLevel::Medium,
        "high" => ThinkingLevel::High,
        "xhigh" => ThinkingLevel::Xhigh,
        "max" => ThinkingLevel::Max,
        _ => return None,
    })
}

fn message_text(m: &AgentMessage) -> String {
    match m {
        AgentMessage::Standard(Message::User(u)) => match &u.content {
            UserContent::Text(t) => t.clone(),
            UserContent::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    UserPart::Text(t) => Some(t.text.as_str()),
                    UserPart::Image(_) => None,
                })
                .collect::<Vec<_>>()
                .join("\n"),
        },
        AgentMessage::Standard(Message::Assistant(a)) => a
            .content
            .iter()
            .filter_map(|c| match c {
                AssistantContent::Text(t) => Some(t.text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

impl Runtime {
    /// Wire the agent to the rpc output channel.
    pub async fn start(
        agent: Arc<Agent>,
        target: Target,
        cwd: PathBuf,
        persist: Persistence,
        out: UnboundedSender<Value>,
    ) -> Arc<Self> {
        let edit_mode = edit_mode_for(&agent.state().await.model.id);
        let rt = Arc::new(Self {
            agent,
            target,
            cwd,
            persist,
            out,
            prompts: Mutex::new(PromptState::default()),
            settings: Mutex::new(Settings {
                session_name: None,
                todo_phases: json!([]),
                auto_compaction: false,
                auto_retry: false,
                interrupt_mode: "immediate".into(),
                cache_warming: "off".into(),
                ask_dialog: false,
                subagent_level: "off".into(),
                event_filter: None,
                delta_updates: false,
                edit_mode,
                last_queue: (Vec::new(), Vec::new()),
                running_bash: None,
            }),
            framer: Mutex::new(Framer::default()),
        });
        rt.record_model_state().await;
        let listener = rt.clone();
        let sub = rt
            .agent
            .subscribe(move |event, _| {
                listener.on_event(&event);
                Box::pin(async { Ok(()) })
            })
            .await;
        drop(sub);
        rt
    }

    pub fn send(&self, frame: Value) {
        let _ = self.out.send(frame);
    }

    async fn record_model_state(&self) {
        let s = self.agent.state().await;
        let model = format!("{}/{}", s.model.provider, s.model.id);
        let level = level_name(s.thinking_level);
        let _ = self.persist.append_entry(
            "thinking_level_change",
            json!({"thinkingLevel": level, "configured": level}),
        );
        let _ = self
            .persist
            .append_entry("model_change", json!({"model": model, "resolvedModelIsFallback": false}));
    }

    // ------------------------------------------------------------- events

    fn queued(&self) -> (Vec<String>, Vec<String>) {
        (self.agent.steering_previews(), self.agent.follow_up_previews())
    }

    pub fn queued_count(&self) -> usize {
        let (a, b) = self.queued();
        a.len() + b.len()
    }

    fn emit_queue_update(&self) {
        let snapshot = self.queued();
        let mut st = lock(&self.settings);
        if st.last_queue == snapshot {
            return;
        }
        st.last_queue = snapshot.clone();
        drop(st);
        self.emit_event(json!({"type": "queue_update", "steering": snapshot.0, "followUp": snapshot.1}));
    }

    /// Forward a session event subject to `set_event_filter`.
    fn emit_event(&self, mut frame: Value) {
        let st = lock(&self.settings);
        let kind = frame.get("type").and_then(Value::as_str).unwrap_or("").to_string();
        if st.event_filter.as_ref().is_some_and(|f| !f.contains(&kind)) {
            return;
        }
        if st.delta_updates && kind == "message_update" {
            if let Some(obj) = frame.as_object_mut() {
                obj.remove("message");
                if let Some(ev) = obj.get_mut("assistantMessageEvent").and_then(Value::as_object_mut) {
                    ev.remove("partial");
                }
            }
        }
        drop(st);
        self.send(frame);
    }

    fn on_event(&self, event: &AgentEvent) {
        let frame = lock(&self.framer).frame(event);
        self.emit_event(frame);
        match event {
            AgentEvent::AgentStart => {
                let mut p = lock(&self.prompts);
                p.agent_starts += 1;
                p.active = true;
            }
            AgentEvent::MessageEnd { message } if message.role() == "user" => self.emit_queue_update(),
            AgentEvent::AgentEnd { messages } => {
                self.emit_queue_update();
                self.report_prompts(&run_outcome(messages));
            }
            _ => {}
        }
    }

    /// Emit `prompt_result` frames then `session_settled` after a terminal end.
    fn report_prompts(&self, outcome: &RunOutcome) {
        let drained = self.queued_count() == 0;
        let mut p = lock(&self.prompts);
        let starts = p.agent_starts;
        let mut remaining = Vec::new();
        let mut reports = Vec::new();
        for t in p.open.drain(..) {
            let ready = if t.waiting { drained } else { starts > t.starts_at_begin };
            if ready {
                reports.push(t);
            } else {
                remaining.push(t);
            }
        }
        p.open = remaining;
        let settle = p.active && drained && p.open.is_empty();
        if settle {
            p.active = false;
        }
        drop(p);
        for t in reports {
            let mut frame = json!({"type": "prompt_result", "agentInvoked": true, "status": outcome.status, "sessionSettled": drained});
            if let Some(id) = t.id {
                frame["id"] = id;
            }
            if let Some(e) = &outcome.error {
                frame["error"] = e.clone();
            }
            self.send(frame);
        }
        if settle {
            self.send(json!({"type": "session_settled"}));
        }
    }

    // ------------------------------------------------------------- prompts

    /// Admit a prompt (or queue it into a live run) and open its ticket.
    pub async fn prompt(&self, id: Option<Value>, message: &str, behavior: Option<&str>) -> Result<(), CmdError> {
        if self.agent.state().await.is_streaming {
            match behavior {
                Some("steer") => self.agent.steer(user_message(message)),
                Some("followUp") => self.agent.follow_up(user_message(message)),
                _ => {
                    return Err(CmdError::new(
                        "Agent is already processing a prompt. Specify streamingBehavior ('steer' or 'followUp') to queue the message.",
                    ))
                }
            }
            lock(&self.prompts).open.push(Ticket { id, starts_at_begin: 0, waiting: true });
            self.emit_queue_update();
            return Ok(());
        }
        let starts = {
            let mut p = lock(&self.prompts);
            let starts = p.agent_starts;
            p.open.push(Ticket { id: id.clone(), starts_at_begin: starts, waiting: false });
            starts
        };
        if let Err(e) = self.agent.prompt_until_accepted(message).await {
            lock(&self.prompts).open.retain(|t| !(t.starts_at_begin == starts && t.id == id && !t.waiting));
            return Err(CmdError::new(format!("{e:#}")));
        }
        Ok(())
    }

    /// Abort the run and wait until the agent is idle.
    pub async fn abort(&self) {
        self.agent.abort();
        self.agent.wait_for_idle().await;
    }

    pub fn steer(&self, message: &str) {
        self.agent.steer(user_message(message));
        self.emit_queue_update();
    }

    pub fn follow_up(&self, message: &str) {
        self.agent.follow_up(user_message(message));
        self.emit_queue_update();
    }

    pub fn remove_queued(&self, message: &str, queue: &str) -> bool {
        let (steering, follow) = self.queued();
        let in_steering = steering.iter().filter(|p| *p == message).count();
        let target = if queue == "steering" {
            steering.iter().position(|p| p == message)
        } else {
            follow.iter().position(|p| p == message).map(|i| i + in_steering)
        };
        let Some(target) = target else { return false };
        // Steering is scanned before follow-up, so `target` indexes the matches in that order.
        let seen = std::cell::Cell::new(0usize);
        let removed = self.agent.remove_queued_messages(|m| {
            if preview_matches(m, message) {
                let hit = seen.get() == target;
                seen.set(seen.get() + 1);
                hit
            } else {
                false
            }
        });
        let ok = !removed.is_empty();
        if ok {
            self.emit_queue_update();
        }
        ok
    }

    pub fn promote_queued(&self, message: &str) -> bool {
        let (_, follow) = self.queued();
        if !follow.iter().any(|p| p == message) {
            return false;
        }
        let removed = self.agent.remove_queued_messages(|m| preview_matches(m, message));
        let Some(first) = removed.into_iter().next() else { return false };
        self.agent.steer(first);
        self.emit_queue_update();
        true
    }

    pub fn set_queue_modes(&self, steering: Option<QueueMode>, follow_up: Option<QueueMode>) {
        if let Some(m) = steering {
            self.agent.set_steering_mode(m);
        }
        if let Some(m) = follow_up {
            self.agent.set_follow_up_mode(m);
        }
    }

    // --------------------------------------------------------------- state

    pub fn set<R>(&self, f: impl FnOnce(&mut SettingsView<'_>) -> R) -> R {
        let mut st = lock(&self.settings);
        f(&mut SettingsView { st: &mut st })
    }

    pub fn session_id(&self) -> String {
        self.persist.session_id()
    }

    pub fn session_file(&self) -> PathBuf {
        self.persist.path()
    }

    pub async fn model_info(&self) -> Value {
        let target = self.target.read().unwrap_or_else(std::sync::PoisonError::into_inner).clone();
        target.map_or(Value::Null, |t| model_info(&t.model))
    }

    pub async fn state(&self) -> Value {
        let s = self.agent.state().await;
        let st = lock(&self.settings);
        let (steering, follow) = self.queued();
        let queued = steering.len() + follow.len();
        let mut v = json!({
            "isStreaming": s.is_streaming,
            "isCompacting": false,
            "steeringMode": queue_mode(self.agent.steering_mode()),
            "followUpMode": queue_mode(self.agent.follow_up_mode()),
            "interruptMode": st.interrupt_mode,
            "sessionFile": self.persist.path().display().to_string(),
            "sessionId": self.persist.session_id(),
            "autoCompactionEnabled": st.auto_compaction,
            "queuedMessageCount": queued,
            "hasPendingAsyncWork": false,
            "isSettled": !s.is_streaming && queued == 0,
            "queuedMessages": {"steering": steering, "followUp": follow},
            "todoPhases": st.todo_phases,
            "fastModeEnabled": false,
            "tokensPerSecond": null,
            "fastModeActive": false,
            "messageCount": s.messages.len(),
            "systemPrompt": [s.system_prompt],
            "dumpTools": s.tools.iter().map(|t| json!({
                "name": t.name(), "description": t.description(), "parameters": t.parameters(),
            })).collect::<Vec<_>>(),
            "goal": null,
        });
        if let Some(name) = &st.session_name {
            v["sessionName"] = json!(name);
        }
        drop(st);
        let mut out = Map::new();
        out.insert("model".into(), self.model_info().await);
        out.insert("thinkingLevel".into(), json!(level_name(s.thinking_level)));
        if let Some(obj) = v.as_object() {
            out.extend(obj.clone());
        }
        if let Some(usage) = context_usage(&s.messages, s.model.context_window) {
            out.insert("contextUsage".into(), usage);
        }
        Value::Object(out)
    }

    pub async fn messages(&self) -> Vec<Value> {
        self.agent
            .state()
            .await
            .messages
            .iter()
            .map(|m| serde_json::to_value(m).unwrap_or(Value::Null))
            .collect()
    }

    pub async fn last_assistant_text(&self) -> Option<String> {
        let s = self.agent.state().await;
        s.messages.iter().rev().find_map(|m| match m {
            AgentMessage::Standard(Message::Assistant(a)) => {
                let t = message_text(&AgentMessage::Standard(Message::Assistant(a.clone())));
                (a.stop_reason != StopReason::Aborted || !t.is_empty()).then_some(t)
            }
            _ => None,
        })
    }

    pub async fn session_stats(&self) -> Value {
        let s = self.agent.state().await;
        let (mut user, mut assistant, mut calls, mut results) = (0u64, 0u64, 0u64, 0u64);
        let (mut input, mut output, mut cache_read, mut cache_write, mut cost) = (0u64, 0u64, 0u64, 0u64, 0.0f64);
        for m in &s.messages {
            match m {
                AgentMessage::Standard(Message::User(_)) => user += 1,
                AgentMessage::Standard(Message::Assistant(a)) => {
                    assistant += 1;
                    calls += a.tool_calls().len() as u64;
                    input += a.usage.input;
                    output += a.usage.output;
                    cache_read += a.usage.cache_read;
                    cache_write += a.usage.cache_write;
                    cost += a.usage.cost.total;
                }
                AgentMessage::Standard(Message::ToolResult(_)) => results += 1,
                AgentMessage::Custom(_) => {}
            }
        }
        let mut v = json!({
            "sessionFile": self.persist.path().display().to_string(),
            "sessionId": self.persist.session_id(),
            "userMessages": user,
            "assistantMessages": assistant,
            "toolCalls": calls,
            "toolResults": results,
            "totalMessages": s.messages.len(),
            "tokens": {
                "input": input, "output": output, "reasoning": 0,
                "cacheRead": cache_read, "cacheWrite": cache_write,
                "total": input + output + cache_read + cache_write,
            },
            "premiumRequests": 0,
            "cost": cost,
        });
        if let Some(u) = context_usage(&s.messages, s.model.context_window) {
            v["contextUsage"] = u;
        }
        v
    }

    // ------------------------------------------------------------ session

    /// Clear the conversation and start a fresh session file.
    pub async fn new_session(&self, dir: Option<&Path>) -> Result<()> {
        self.agent.abort();
        self.agent.wait_for_idle().await;
        self.agent.reset().await;
        self.persist.start_new(&self.cwd, dir)?;
        {
            let mut st = lock(&self.settings);
            st.session_name = None;
            st.last_queue = (Vec::new(), Vec::new());
        }
        lock(&self.prompts).open.clear();
        self.record_model_state().await;
        Ok(())
    }

    /// Load `path`'s messages into the agent. The continuation is written to a
    /// fresh session file seeded with the loaded history (the writer is
    /// append-only and cannot reopen an existing file).
    pub async fn load_session(&self, path: &Path, dir: Option<&Path>) -> Result<()> {
        let lines = omp_session::read(path).with_context(|| format!("cannot read session {}", path.display()))?;
        let mut messages = Vec::new();
        for l in &lines {
            let v: Value = serde_json::from_str(&l.raw)?;
            if v.get("type").and_then(Value::as_str) == Some("message") {
                if let Some(m) = v.get("message") {
                    if let Ok(m) = serde_json::from_value::<AgentMessage>(m.clone()) {
                        messages.push(m);
                    }
                }
            }
        }
        self.agent.abort();
        self.agent.wait_for_idle().await;
        self.agent.reset().await;
        self.persist.start_new(&self.cwd, dir)?;
        self.record_model_state().await;
        for m in &messages {
            let converted = serde_json::to_value(m)
                .map_err(anyhow::Error::from)
                .and_then(|v| serde_json::from_value::<pa_types::session::AgentMessage>(v).map_err(anyhow::Error::from));
            if let Ok(c) = converted {
                let _ = self.persist.append_message(&c, m.timestamp());
            }
        }
        self.agent.set_messages(messages).await;
        lock(&self.prompts).open.clear();
        Ok(())
    }

    pub fn set_session_name(&self, name: &str) -> Result<()> {
        self.persist
            .append_entry("title_change", json!({"title": name, "source": "user"}))?;
        lock(&self.settings).session_name = Some(name.to_string());
        Ok(())
    }

    /// Raw JSON entries of the live session file (header excluded), append order.
    pub fn entries(&self) -> Result<Vec<Value>> {
        read_entries(&self.persist.path())
    }

    // -------------------------------------------------------------- models

    pub async fn set_model(&self, provider: &str, model_id: &str) -> Result<Value, CmdError> {
        let reference = format!("{provider}/{model_id}");
        let resolved = omp_providers::resolve(&reference)
            .map_err(|_| CmdError::new(format!("Model not found: {provider}/{model_id}")))?;
        if resolved.provider != provider || resolved.id != model_id {
            return Err(CmdError::new(format!("Model not found: {provider}/{model_id}")));
        }
        let agent_model: pa_agent::types::Model = json_round_trip(&resolved.model)
            .ok_or_else(|| CmdError::new("model does not convert to the agent model shape"))?;
        let level = {
            let current = self.agent.state().await.thinking_level;
            if resolved.reasoning {
                resolved.thinking.map_or(current, map_thinking_level)
            } else {
                ThinkingLevel::Off
            }
        };
        let info = model_info(&resolved.model);
        *self.target.write().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(ProviderTarget {
            api_key: resolved.api_key.clone(),
            model: resolved.model.clone(),
            service_tier: None,
            headers: (!resolved.headers.is_empty()).then(|| resolved.headers.iter().cloned().collect()),
        });
        self.agent.set_model_and_thinking_level(agent_model, level).await;
        let mode = edit_mode_for(model_id);
        let rebuild = {
            let mut st = lock(&self.settings);
            let changed = st.edit_mode != mode;
            st.edit_mode = mode;
            changed
        };
        if rebuild {
            self.agent.set_tools(builtin_tools(&self.cwd, mode)).await;
        }
        let _ = self
            .persist
            .append_entry("model_change", json!({"model": reference, "resolvedModelIsFallback": false}));
        let name = level_name(level);
        let _ = self
            .persist
            .append_entry("thinking_level_change", json!({"thinkingLevel": name, "configured": name}));
        self.emit_event(json!({"type": "model_changed"}));
        Ok(info)
    }

    pub async fn set_thinking_level(&self, level: ThinkingLevel) {
        self.agent.set_thinking_level(level).await;
        let name = level_name(level);
        let _ = self
            .persist
            .append_entry("thinking_level_change", json!({"thinkingLevel": name, "configured": name}));
        self.emit_event(json!({"type": "thinking_level_changed", "thinkingLevel": name, "configured": name}));
    }

    /// Selectable levels for the live model, `off` first.
    pub async fn available_thinking_levels(&self) -> Vec<&'static str> {
        let target = self.target.read().unwrap_or_else(std::sync::PoisonError::into_inner).clone();
        let mut levels = vec!["off"];
        if let Some(t) = target {
            if t.model.reasoning {
                levels.extend(["minimal", "low", "medium", "high", "xhigh"]);
            }
        }
        levels
    }
}

/// Handle to the mutable per-session settings (no lock leaks out of `set`).
pub struct SettingsView<'a> {
    st: &'a mut Settings,
}

impl SettingsView<'_> {
    pub fn auto_compaction(&mut self, v: bool) {
        self.st.auto_compaction = v;
    }
    pub fn auto_retry(&mut self, v: bool) {
        self.st.auto_retry = v;
    }
    pub fn interrupt_mode(&mut self, v: &str) {
        self.st.interrupt_mode = v.to_string();
    }
    pub fn cache_warming(&mut self, v: &str) {
        self.st.cache_warming = v.to_string();
    }
    pub fn ask_dialog(&mut self, v: bool) -> bool {
        self.st.ask_dialog = v;
        v
    }
    pub fn subagent_level(&mut self, v: &str) {
        self.st.subagent_level = v.to_string();
    }
    pub fn todo_phases(&mut self, v: Value) {
        self.st.todo_phases = v;
    }
    pub fn event_filter(&mut self, filter: Option<HashSet<String>>, delta: bool) {
        self.st.event_filter = filter;
        self.st.delta_updates = delta;
    }
    pub fn running_bash(&mut self, v: Option<AbortController>) -> Option<AbortController> {
        std::mem::replace(&mut self.st.running_bash, v)
    }
    pub fn abort_bash(&mut self) {
        if let Some(c) = self.st.running_bash.take() {
            c.abort();
        }
    }
    #[must_use]
    pub fn todo_phases_value(&self) -> Value {
        self.st.todo_phases.clone()
    }
}

pub fn queue_mode(m: QueueMode) -> &'static str {
    match m {
        QueueMode::All => "all",
        QueueMode::OneAtATime => "one-at-a-time",
    }
}

fn user_message(text: &str) -> AgentMessage {
    AgentMessage::Standard(Message::User(pa_agent::types::UserMessage {
        content: UserContent::Text(text.to_string()),
        timestamp: pa_agent::now_ms(),
    }))
}

fn preview_matches(m: &AgentMessage, preview: &str) -> bool {
    let text = message_text(m);
    text == preview || text.starts_with(preview.trim_end_matches('…'))
}

fn run_outcome(messages: &[AgentMessage]) -> RunOutcome {
    for m in messages.iter().rev() {
        if let AgentMessage::Standard(Message::Assistant(a)) = m {
            return match a.stop_reason {
                StopReason::Error => RunOutcome {
                    status: "error",
                    error: Some(json!({
                        "message": a.error_message.clone().unwrap_or_else(|| "Provider request failed".into()),
                        "provider": a.provider,
                        "model": a.model,
                        "retryable": false,
                    })),
                },
                StopReason::Aborted => RunOutcome { status: "aborted", error: None },
                _ => RunOutcome { status: "completed", error: None },
            };
        }
    }
    RunOutcome { status: "completed", error: None }
}

/// Context size from the last assistant usage (omp `getContextUsage`).
fn context_usage(messages: &[AgentMessage], window: u64) -> Option<Value> {
    if window == 0 {
        return None;
    }
    let tokens = messages.iter().rev().find_map(|m| match m {
        AgentMessage::Standard(Message::Assistant(a)) if a.usage.total_tokens > 0 => Some(a.usage.total_tokens),
        _ => None,
    })?;
    #[allow(clippy::cast_precision_loss)]
    let percent = tokens as f64 / window as f64 * 100.0;
    Some(json!({"tokens": tokens, "contextWindow": window, "percent": percent}))
}

/// omp `ModelInfo` from a pa-ai model.
pub fn model_info(model: &pa_types::ai::Model) -> Value {
    let mut v = serde_json::to_value(model).unwrap_or(Value::Null);
    if let Some(o) = v.as_object_mut() {
        o.remove("featured");
        o.remove("maxTokensExplicit");
        o.remove("thinking_level_map");
        o.remove("thinkingLevelMap");
    }
    v
}

pub fn read_entries(path: &Path) -> Result<Vec<Value>> {
    let lines = omp_session::read(path)?;
    let mut out = Vec::new();
    for l in lines {
        let v: Value = serde_json::from_str(&l.raw)?;
        if v.get("type").and_then(Value::as_str) != Some("session") {
            out.push(v);
        }
    }
    Ok(out)
}

/// Newest non-empty `*.jsonl` session in `dir` (has at least one message entry).
pub fn newest_session(dir: &Path) -> Option<PathBuf> {
    let mut files: Vec<(std::time::SystemTime, PathBuf)> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(Result::ok)
        .filter(|e| e.path().extension().is_some_and(|x| x == "jsonl"))
        .filter_map(|e| Some((e.metadata().ok()?.modified().ok()?, e.path())))
        .collect();
    files.sort_by(|a, b| b.0.cmp(&a.0));
    files.into_iter().map(|(_, p)| p).find(|p| {
        std::fs::read_to_string(p).is_ok_and(|t| t.lines().any(|l| l.contains("\"type\":\"message\"")))
    })
}

/// Models whose provider has credentials (plus the live model), for
/// `get_available_models` / `cycle_model`.
pub fn available_models(current: Option<&pa_types::ai::Model>) -> Vec<pa_types::ai::Model> {
    let mut out: Vec<pa_types::ai::Model> = Vec::new();
    for provider in pa_ai::models_generated::get_providers() {
        let models = pa_ai::models_generated::get_models(provider);
        let Some(first) = models.first() else { continue };
        let has_key = omp_providers::resolve(&format!("{provider}/{}", first.id))
            .is_ok_and(|r| r.api_key.is_some());
        if has_key {
            out.extend(models.into_iter().cloned());
        }
    }
    if let Some(c) = current {
        if !out.iter().any(|m| m.provider == c.provider && m.id == c.id) {
            out.insert(0, c.clone());
        }
    }
    out
}

pub fn unsupported(what: &str) -> CmdError {
    CmdError::new(format!("{what} is not supported by omp-rs"))
}
