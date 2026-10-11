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

use crate::adapter::{json_round_trip, ProviderTarget};
use crate::agent::Target;
use crate::events::Framer;
use crate::persist::Persistence;
use self::recovery::{FailedTurn, RecoveryConfig, Resolver, TurnRecovery, Verdict, EPHEMERAL_ROLE};
use crate::tools::{builtin_tools, edit_mode_for};

// Declared here (not in main.rs, which another agent owns); same file `src/recovery.rs`.
#[path = "recovery.rs"]
pub mod recovery;

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
    recovery: Mutex<TurnRecovery>,
    resolver: Resolver,
    /// Slash-command catalog behind `get_available_commands` / `available_commands_update`.
    commands: Mutex<Vec<Value>>,
    announced: std::sync::atomic::AtomicBool,
    settled: tokio::sync::Notify,
}

/// Outcome of the retry pass over a failed turn.
enum Recovered {
    /// A backoff elapsed and the run is being continued.
    Retrying,
    /// The saga is over; the turn's error stands (optionally with a rewritten message).
    Final { error: Option<String> },
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

fn last_assistant_msg(messages: &[AgentMessage]) -> Option<&pa_agent::types::AssistantMessage> {
    messages.iter().rev().find_map(|m| match m {
        AgentMessage::Standard(Message::Assistant(a)) => Some(a),
        _ => None,
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
        Self::start_with(agent, target, cwd, persist, out, recovery::default_resolver(), None).await
    }

    /// [`Self::start`] with an explicit model resolver (fallback candidates, `set_model`) and,
    /// optionally, the retry/fallback configuration (default: derived from the live model).
    pub async fn start_with(
        agent: Arc<Agent>,
        target: Target,
        cwd: PathBuf,
        persist: Persistence,
        out: UnboundedSender<Value>,
        resolver: Resolver,
        recovery_cfg: Option<RecoveryConfig>,
    ) -> Arc<Self> {
        let live = agent.state().await.model;
        let edit_mode = edit_mode_for(&live.id);
        let recovery_cfg = recovery_cfg.unwrap_or_else(|| {
            resolver(&format!("{}/{}", live.provider, live.id))
                .map(|r| RecoveryConfig::from_resolved(&r))
                .unwrap_or_default()
        });
        let recovery = TurnRecovery::new(recovery_cfg);
        persist.defer_errors(true);
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
            recovery: Mutex::new(recovery),
            resolver,
            commands: Mutex::new(Vec::new()),
            announced: std::sync::atomic::AtomicBool::new(false),
            settled: tokio::sync::Notify::new(),
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

    /// Session preamble omp writes: `model_change` then `thinking_level_change`.
    async fn record_model_state(&self) {
        let s = self.agent.state().await;
        let model = format!("{}/{}", s.model.provider, s.model.id);
        let _ = self.persist.append_model_change(&model, None, false);
        let _ = self.persist.append_thinking_level_change(Some(level_name(s.thinking_level)), None);
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

    fn on_event(self: &Arc<Self>, event: &AgentEvent) {
        let frame = lock(&self.framer).frame(event);
        if let AgentEvent::AgentEnd { messages } = event {
            self.on_agent_end(frame, messages);
            return;
        }
        self.emit_event(frame);
        match event {
            AgentEvent::AgentStart => {
                let mut p = lock(&self.prompts);
                p.agent_starts += 1;
                p.active = true;
            }
            AgentEvent::MessageEnd { message } if message.role() == "user" => self.emit_queue_update(),
            AgentEvent::MessageEnd { message: AgentMessage::Standard(Message::Assistant(a)) } => {
                self.on_assistant_settled(a);
            }
            _ => {}
        }
    }

    /// `onAssistantSettledSuccessfully`: a turn that produced output settled on the live model.
    fn on_assistant_settled(&self, a: &pa_agent::types::AssistantMessage) {
        if matches!(a.stop_reason, StopReason::Error | StopReason::Aborted) || a.content.is_empty() {
            return;
        }
        let selector = self.live_selector_with_level(None);
        let frames = lock(&self.recovery).on_settled_ok(&selector);
        if !frames.is_empty() {
            // The failed turn was superseded: it never reaches the session file.
            self.persist.drop_deferred_error();
        }
        for f in frames {
            self.emit_event(f);
        }
    }

    /// Terminal `agent_end`, or the retry saga when the run failed transiently.
    fn on_agent_end(self: &Arc<Self>, frame: Value, messages: &[AgentMessage]) {
        self.emit_queue_update();
        let failed = last_assistant_msg(messages).and_then(FailedTurn::from_message);
        if let Some(f) = failed.filter(|f| lock(&self.recovery).is_retryable(f)) {
            let rt = self.clone();
            let messages = messages.to_vec();
            // The listener runs while the agent holds its state lock: recovery (model swap,
            // context surgery, continuing the run) happens after the run went idle.
            tokio::spawn(async move { rt.recover(frame, messages, f).await });
            return;
        }
        self.settle_terminal(frame, messages, None);
    }

    /// Final `agent_end` of a prompt: persist/close failure bookkeeping, then `prompt_result`s.
    fn settle_terminal(self: &Arc<Self>, mut frame: Value, messages: &[AgentMessage], error_override: Option<&str>) {
        let mut messages = messages.to_vec();
        if let Some(AgentMessage::Standard(Message::Assistant(a))) = messages.iter_mut().rev().find(|m| m.role() == "assistant") {
            if a.stop_reason == StopReason::Error {
                if let Some(e) = error_override {
                    a.error_message = Some(e.to_string());
                    if let Some(last) = frame
                        .get_mut("messages")
                        .and_then(Value::as_array_mut)
                        .and_then(|v| v.iter_mut().rev().find(|m| m.get("role") == Some(&json!("assistant"))))
                    {
                        last["errorMessage"] = json!(e);
                    }
                }
                self.persist.flush_deferred_error(error_override);
                let text = a.error_message.clone().unwrap_or_default();
                if let Some(end) = lock(&self.recovery).on_error_settled(&text) {
                    self.emit_event(end);
                }
            }
        }
        let outcome = run_outcome(&messages);
        self.emit_event(frame);
        self.report_prompts(&outcome);
    }

    /// Retry saga for a failed turn (`#handleRetryableError`), then continue the run.
    async fn recover(self: Arc<Self>, frame: Value, messages: Vec<AgentMessage>, failed: FailedTurn) {
        self.agent.wait_for_idle().await;
        match self.handle_retryable(&failed).await {
            Recovered::Final { error } => self.settle_terminal(frame, &messages, error.as_deref()),
            Recovered::Retrying => {
                let mut nonterminal = frame.clone();
                nonterminal["isTerminal"] = json!(false);
                nonterminal["yielded"] = json!(false);
                self.emit_event(nonterminal);
                let rt = self.clone();
                tokio::spawn(async move {
                    if let Err(e) = rt.continue_after_retry().await {
                        let local = format!("{e:#}");
                        let original = failed.message.clone();
                        let end = lock(&rt.recovery).end_continue_failed(&local, &original);
                        if let Some(end) = end {
                            rt.emit_event(end);
                        }
                        rt.settle_terminal(frame, &messages, None);
                    }
                });
            }
        }
    }

    /// `scheduleAgentContinue`: re-run from the context left after the failed turn was dropped.
    async fn continue_after_retry(&self) -> Result<()> {
        for _ in 0..200 {
            match self.agent.continue_run().await {
                Ok(()) => return Ok(()),
                Err(e) => {
                    let busy = e
                        .downcast_ref::<pa_agent::agent::AgentContinueError>()
                        .is_some_and(|c| c.code == pa_agent::agent::AgentContinueErrorCode::Busy);
                    if !busy {
                        return Err(e);
                    }
                    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                }
            }
        }
        anyhow::bail!("agent stayed busy")
    }

    fn live_provider_id(&self) -> Option<(String, String)> {
        let t = self.target.read().unwrap_or_else(std::sync::PoisonError::into_inner);
        t.as_ref().map(|t| (String::from(&*t.model.provider), t.model.id.clone()))
    }

    /// `provider/id[:level]` of the live model (`formatRetryFallbackSelector`).
    fn live_selector_with_level(&self, level: Option<ThinkingLevel>) -> String {
        let (p, i) = self.live_provider_id().unwrap_or_default();
        match level {
            Some(l) => format!("{p}/{i}:{}", level_name(l)),
            None => format!("{p}/{i}"),
        }
    }

    /// Port of `#handleRetryableError`: fallback consult, budget/fail-fast checks, backoff.
    async fn handle_retryable(self: &Arc<Self>, failed: &FailedTurn) -> Recovered {
        let level = self.agent.state().await.thinking_level;
        let current = self.live_selector_with_level(Some(level));
        let now = u64::try_from(pa_agent::now_ms()).unwrap_or(0);
        let (attempt, candidates) = {
            let mut rec = lock(&self.recovery);
            let attempt = rec.begin_attempt(failed, rand::random::<f64>());
            let candidates = rec.consult(&current, &attempt, now, rand::random::<f64>());
            (attempt, candidates)
        };
        let mut switched = false;
        for selector in candidates {
            let Ok(resolved) = (self.resolver)(&selector) else { continue };
            if resolved.api_key.is_none() {
                continue;
            }
            if self.apply_fallback(&resolved, &selector, &current, level).await {
                switched = true;
                break;
            }
        }
        let verdict = lock(&self.recovery).finish_attempt(&attempt, switched);
        let delay_ms = match verdict {
            Verdict::GiveUp { frame, error_message } => {
                self.emit_event(frame);
                return Recovered::Final { error: error_message };
            }
            Verdict::Retry { delay_ms } => delay_ms,
        };
        let announced = lock(&self.recovery).attempt();
        let max = attempt.max;
        self.emit_event(recovery::auto_retry_start_frame(announced, max, delay_ms, &attempt.error_message));
        // The failed turn is superseded by the retry: out of the file and out of the context.
        self.persist.drop_deferred_error();
        self.agent
            .mutate_messages(|m| {
                if matches!(m.last(), Some(AgentMessage::Standard(Message::Assistant(a))) if matches!(a.stop_reason, StopReason::Error | StopReason::Aborted)) {
                    m.pop();
                }
            })
            .await;
        let (generation, token) = lock(&self.recovery).arm_cancel();
        tokio::select! {
            () = tokio::time::sleep(std::time::Duration::from_millis(delay_ms)) => {
                lock(&self.recovery).disarm(generation);
                Recovered::Retrying
            }
            () = token.cancelled() => {
                let mut rec = lock(&self.recovery);
                if rec.superseded(generation) {
                    return Recovered::Final { error: None };
                }
                let end = rec.end_cancelled();
                drop(rec);
                self.emit_event(end);
                self.persist.flush_deferred_error(None);
                Recovered::Final { error: None }
            }
        }
    }

    /// `#applyRetryFallbackCandidate`: swap to `selector`, journal it, announce it.
    async fn apply_fallback(
        &self,
        resolved: &omp_providers::ResolvedModel,
        selector: &str,
        current: &str,
        current_level: ThinkingLevel,
    ) -> bool {
        let requested = selector
            .rsplit_once(':')
            .and_then(|(_, tail)| parse_level(tail))
            .unwrap_or(current_level);
        let from_thinking = level_name(current_level).to_string();
        self.swap_model(resolved, Some(EPHEMERAL_ROLE), true).await;
        let applied = self.apply_thinking(requested).await;
        let applied_name = level_name(applied).to_string();
        let role = {
            let mut rec = lock(&self.recovery);
            rec.commit_switch(current, Some(from_thinking), Some(applied_name));
            rec.role().to_string()
        };
        self.emit_event(recovery::fallback_applied_frame(current, selector, &role, None));
        true
    }

    /// Before a prompt: return to the primary model once its cooldown expired (`maybeRestoreRetryFallbackPrimary`).
    async fn maybe_restore_primary(&self) {
        let level = self.agent.state().await.thinking_level;
        let current = self.live_selector_with_level(Some(level));
        let now = u64::try_from(pa_agent::now_ms()).unwrap_or(0);
        let plan = lock(&self.recovery).state_mut().restore_plan(&current, Some(level_name(level)), now);
        let Some(plan) = plan else { return };
        let Ok(resolved) = (self.resolver)(&plan.selector) else { return };
        if resolved.api_key.is_none() {
            return;
        }
        let restored_level = plan.thinking.as_deref().and_then(parse_level);
        // Ownership ends before the swap so observers never see the primary tagged as a fallback.
        lock(&self.recovery).state_mut().clear_active();
        self.swap_model(&resolved, Some(EPHEMERAL_ROLE), false).await;
        if let Some(l) = restored_level {
            self.apply_thinking(l).await;
        }
    }

    /// `prompt_result` / `session_settled` land one tick after the terminal `agent_end` (omp's
    /// `setImmediate`): the response of the command that ended the run (e.g. `abort`) goes first and
    /// queue drains are visible when `sessionSettled` is read.
    fn report_prompts(self: &Arc<Self>, outcome: &RunOutcome) {
        let rt = self.clone();
        let outcome = outcome.clone();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(2)).await;
            rt.report_prompts_now(&outcome);
        });
    }

    /// Emit `prompt_result` frames then `session_settled` after a terminal end.
    fn report_prompts_now(&self, outcome: &RunOutcome) {
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
        self.settled.notify_waiters();
    }

    // ------------------------------------------------------------- prompts

    /// Admit a prompt (or queue it into a live run) and open its ticket.
    pub async fn prompt(
        &self,
        id: Option<Value>,
        message: &str,
        behavior: Option<&str>,
        images: Vec<pa_agent::types::ImageContent>,
    ) -> Result<(), CmdError> {
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
        // A new prompt supersedes a pending retry and may return to the primary model.
        self.abort_retry();
        self.maybe_restore_primary().await;
        let starts = {
            let mut p = lock(&self.prompts);
            let starts = p.agent_starts;
            p.open.push(Ticket { id: id.clone(), starts_at_begin: starts, waiting: false });
            starts
        };
        let input = pa_agent::agent::AgentPromptInput::Text { text: message.to_string(), images };
        if let Err(e) = self.agent.prompt_until_accepted(input).await {
            lock(&self.prompts).open.retain(|t| !(t.starts_at_begin == starts && t.id == id && !t.waiting));
            return Err(CmdError::new(format!("{e:#}")));
        }
        Ok(())
    }

    /// Abort the run and wait until the agent is idle.
    pub async fn abort(&self) {
        self.abort_retry();
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
        let retrying = lock(&self.recovery).saga_open();
        let mut v = json!({
            "isStreaming": s.is_streaming,
            "isCompacting": false,
            "steeringMode": queue_mode(self.agent.steering_mode()),
            "followUpMode": queue_mode(self.agent.follow_up_mode()),
            "interruptMode": st.interrupt_mode,
            "sessionId": self.persist.session_id(),
            "autoCompactionEnabled": st.auto_compaction,
            "queuedMessageCount": queued,
            "hasPendingAsyncWork": false,
            "isSettled": !s.is_streaming && queued == 0 && !retrying,
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
        if let Some(file) = self.persist.file() {
            v["sessionFile"] = json!(file.display().to_string());
        }
        drop(st);
        let mut out = Map::new();
        out.insert("model".into(), self.model_info().await);
        out.insert("thinkingLevel".into(), json!(level_name(s.thinking_level)));
        if let Some(obj) = v.as_object() {
            out.extend(obj.clone());
        }
        let base = prompt_overhead_chars(&s.system_prompt, &s.tools);
        if let Some(usage) = context_usage(&s.messages, s.model.context_window, base) {
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
            .map(crate::events::message_value)
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
        if let Some(file) = self.persist.file() {
            v["sessionFile"] = json!(file.display().to_string());
        }
        let base = prompt_overhead_chars(&s.system_prompt, &s.tools);
        if let Some(u) = context_usage(&s.messages, s.model.context_window, base) {
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
        let previous = lock(&self.settings).session_name.clone();
        self.persist.append_title_change(name, "user", previous.as_deref())?;
        lock(&self.settings).session_name = Some(name.to_string());
        Ok(())
    }

    /// Raw JSON entries of the live session file (header excluded), append order.
    pub fn entries(&self) -> Result<Vec<Value>> {
        read_entries(&self.persist.path())
    }

    // -------------------------------------------------------------- models

    /// `set_model`: `model_change` entry with role `default`.
    pub async fn set_model(&self, provider: &str, model_id: &str) -> Result<Value, CmdError> {
        self.select_model(provider, model_id, Some("default")).await
    }

    /// `cycle_model`: same switch, `model_change` without a role (`appendModelChange(model)`).
    pub async fn cycle_to(&self, provider: &str, model_id: &str) -> Result<Value, CmdError> {
        self.select_model(provider, model_id, None).await
    }

    async fn select_model(&self, provider: &str, model_id: &str, role: Option<&str>) -> Result<Value, CmdError> {
        let reference = format!("{provider}/{model_id}");
        let resolved = (self.resolver)(&reference)
            .map_err(|_| CmdError::new(format!("Model not found: {provider}/{model_id}")))?;
        if resolved.provider != provider || resolved.id != model_id {
            return Err(CmdError::new(format!("Model not found: {provider}/{model_id}")));
        }
        if resolved.api_key.is_none() {
            return Err(CmdError::new(format!("No API key for {provider}/{model_id}")));
        }
        let info = model_info(&resolved.model);
        // A user selection ends fallback ownership and clears the selector's cooldown.
        {
            let mut rec = lock(&self.recovery);
            rec.reconfigure(RecoveryConfig::from_resolved(&resolved));
            rec.state_mut().clear_suppressed(&reference);
        }
        self.swap_model(&resolved, role, false).await;
        // Re-apply thinking for the new model: the catalog's `thinking.defaultLevel`, else the current level.
        let catalog_default = crate::model::catalog_extras(provider, model_id)
            .into_iter()
            .flatten()
            .find(|(k, _)| *k == "thinking")
            .and_then(|(_, v)| v.get("defaultLevel").and_then(Value::as_str).and_then(parse_level));
        let preferred = match catalog_default {
            Some(l) => l,
            None => self.agent.state().await.thinking_level,
        };
        self.apply_thinking(preferred).await;
        Ok(info)
    }

    /// Swap the live model: provider target, agent model, edit-mode tools, `model_change` entry,
    /// `model_changed` event (only when the model actually changed). Thinking is left to the caller.
    async fn swap_model(&self, resolved: &omp_providers::ResolvedModel, role: Option<&str>, is_fallback: bool) {
        let agent_model: Option<pa_agent::types::Model> = json_round_trip(&resolved.model);
        let Some(agent_model) = agent_model else { return };
        let changing = self.live_provider_id().is_none_or(|(p, i)| p != resolved.provider || i != resolved.id);
        *self.target.write().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(ProviderTarget {
            api_key: resolved.api_key.clone(),
            model: resolved.model.clone(),
            service_tier: None,
            headers: (!resolved.headers.is_empty()).then(|| resolved.headers.iter().cloned().collect()),
        });
        self.agent.set_model(agent_model).await;
        let mode = edit_mode_for(&resolved.id);
        let rebuild = {
            let mut st = lock(&self.settings);
            let changed = st.edit_mode != mode;
            st.edit_mode = mode;
            changed
        };
        if rebuild {
            self.agent.set_tools(builtin_tools(&self.cwd, mode)).await;
        }
        if changing {
            self.emit_event(json!({"type": "model_changed"}));
        }
        let selector = format!("{}/{}", resolved.provider, resolved.id);
        let _ = self.persist.append_model_change(&selector, role, is_fallback);
    }

    /// `ModelControls.setThinkingLevel`: clamp to the live model; journal and announce only a change.
    /// Returns the effective level.
    async fn apply_thinking(&self, requested: ThinkingLevel) -> ThinkingLevel {
        let model = self.target.read().unwrap_or_else(std::sync::PoisonError::into_inner).as_ref().map(|t| t.model.clone());
        let effective = model.map_or(requested, |m| crate::model::clamp_level(&m, requested));
        let current = self.agent.state().await.thinking_level;
        if current != effective {
            self.agent.set_thinking_level(effective).await;
            let name = level_name(effective);
            let _ = self.persist.append_thinking_level_change(Some(name), Some(name));
            self.emit_event(json!({"type": "thinking_level_changed", "thinkingLevel": name}));
        }
        effective
    }

    pub async fn set_thinking_level(&self, level: ThinkingLevel) {
        self.apply_thinking(level).await;
    }

    /// Resolve once every admitted prompt reported and no retry saga is pending (print mode:
    /// `prompt`, then wait here, then read the transcript).
    pub async fn wait_settled(&self) {
        loop {
            let notified = self.settled.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            let idle = {
                let p = lock(&self.prompts);
                p.open.is_empty() && !p.active
            } && !lock(&self.recovery).saga_open();
            if idle {
                return;
            }
            notified.await;
        }
    }

    /// `set_auto_retry`: toggle the retry engine.
    pub fn set_auto_retry(&self, enabled: bool) {
        lock(&self.recovery).set_enabled(enabled);
    }

    /// `abort_retry`: cancel a pending backoff; the failed turn then stands.
    pub fn abort_retry(&self) {
        lock(&self.recovery).abort_retry();
    }

    /// Replace the extension/custom/file part of the slash-command catalog; announced (`available_commands_update`) once the
    /// session is live, stored silently before that (`rpc::run_with`'s start-up hook).
    #[allow(dead_code)]
    pub fn set_available_commands(&self, commands: Vec<Value>) {
        *lock(&self.commands) = commands;
        if self.announced.load(std::sync::atomic::Ordering::SeqCst) {
            self.announce_commands();
        }
    }

    /// Emit `available_commands_update` with the current catalog.
    pub fn announce_commands(&self) {
        self.announced.store(true, std::sync::atomic::Ordering::SeqCst);
        let commands = self.available_commands();
        self.send(json!({"type": "available_commands_update", "commands": commands}));
    }

    /// omp's builtin slash commands followed by the extension/custom/file commands supplied
    /// through [`Self::set_available_commands`] (`getAvailableCommands` order).
    #[must_use]
    pub fn available_commands(&self) -> Vec<Value> {
        let mut all = builtin_commands();
        all.extend(lock(&self.commands).iter().cloned());
        all
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
                        "retryable": recovery::is_transient(None, a.error_message.as_deref().unwrap_or("")),
                    })),
                },
                StopReason::Aborted => RunOutcome { status: "aborted", error: None },
                _ => RunOutcome { status: "completed", error: None },
            };
        }
    }
    RunOutcome { status: "completed", error: None }
}

/// Characters the system prompt and tool schemas add to every request.
fn prompt_overhead_chars(system_prompt: &str, tools: &[Arc<dyn pa_agent::types::AgentTool>]) -> usize {
    system_prompt.chars().count()
        + tools
            .iter()
            .map(|t| t.name().len() + t.description().len() + t.parameters().to_string().len())
            .sum::<usize>()
}

/// Rough token count of a message (omp `estimateTokens`: ~4 chars per token).
fn estimate_tokens(m: &AgentMessage) -> u64 {
    let chars = match m {
        AgentMessage::Standard(Message::Assistant(a)) => a
            .content
            .iter()
            .map(|c| match c {
                AssistantContent::Text(t) => t.text.chars().count(),
                AssistantContent::Thinking(t) => t.thinking.chars().count(),
                AssistantContent::ToolCall(c) => c.name.len() + c.arguments.to_string().len(),
            })
            .sum(),
        other => message_text(other).chars().count(),
    };
    u64::try_from(chars.div_ceil(4)).unwrap_or(u64::MAX)
}

/// omp `getContextUsage`: the last assistant usage plus an estimate of what followed it, or an
/// estimate of the whole request (`overhead_chars` = system prompt + tools) before any usage exists.
fn context_usage(messages: &[AgentMessage], window: u64, overhead_chars: usize) -> Option<Value> {
    if window == 0 {
        return None;
    }
    let last_usage = messages.iter().rposition(|m| {
        matches!(m, AgentMessage::Standard(Message::Assistant(a)) if a.usage.total_tokens > 0 && a.stop_reason != StopReason::Aborted)
    });
    let tokens = match last_usage {
        Some(i) => {
            let AgentMessage::Standard(Message::Assistant(a)) = &messages[i] else { return None };
            a.usage.total_tokens + messages[i + 1..].iter().map(estimate_tokens).sum::<u64>()
        }
        None => {
            u64::try_from(overhead_chars.div_ceil(4)).unwrap_or(u64::MAX) + messages.iter().map(estimate_tokens).sum::<u64>()
        }
    };
    #[allow(clippy::cast_precision_loss)]
    let percent = tokens as f64 / window as f64 * 100.0;
    Some(json!({"tokens": tokens, "contextWindow": window, "percent": percent}))
}

/// omp `ModelInfo` from a pa-ai model, plus the catalog metadata (`int`, `tps`, `thinking`,
/// `identity`, `tokenizer`, ...) pa-ai's descriptor does not carry.
pub fn model_info(model: &pa_types::ai::Model) -> Value {
    let mut v = serde_json::to_value(model).unwrap_or(Value::Null);
    if let Some(o) = v.as_object_mut() {
        o.remove("featured");
        o.remove("maxTokensExplicit");
        o.remove("thinking_level_map");
        o.remove("thinkingLevelMap");
        if let Some(extras) = crate::model::catalog_extras(&model.provider, &model.id) {
            let compat = o.remove("compat");
            for (k, val) in extras {
                o.insert(k.to_string(), val);
            }
            if let Some(c) = compat {
                o.insert("compat".into(), c);
            }
        }
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

/// omp's builtin slash-command descriptors (`source: "builtin"`), as `get_available_commands` lists them.
fn builtin_commands() -> Vec<Value> {
    static BUILTIN: std::sync::OnceLock<Vec<Value>> = std::sync::OnceLock::new();
    BUILTIN
        .get_or_init(|| serde_json::from_str(include_str!("builtin_commands.json")).unwrap_or_default())
        .clone()
}

/// Print mode (`-p`): run one prompt through the full runtime (retry, fallback chains) with
/// events discarded, and return the transcript.
///
/// # Errors
/// Fails when the prompt cannot be admitted.
pub async fn run_print(agent: Agent, target: Target, cwd: PathBuf, persist: Persistence, prompt: &str) -> Result<Vec<AgentMessage>> {
    let agent = Arc::new(agent);
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    drop(rx);
    let rt = Runtime::start(agent.clone(), target, cwd, persist, tx).await;
    rt.prompt(None, prompt, None, Vec::new()).await.map_err(|e| anyhow::anyhow!(e.message))?;
    rt.wait_settled().await;
    Ok(agent.state().await.messages)
}

pub fn unsupported(what: &str) -> CmdError {
    CmdError::new(format!("{what} is not supported by omp-rs"))
}
