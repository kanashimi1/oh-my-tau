//! Session runtime behind `--mode rpc`: the agent plus everything omp's
//! `AgentSession` adds around it (model switching, queues, session files,
//! stats, prompt-result bookkeeping, event forwarding).

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};

use anyhow::{Context, Result};
use pa_agent::abort::AbortController;
use pa_agent::agent::{Agent, QueueMode};
use pa_agent::types::{
    AgentEvent, AgentMessage, AssistantContent, Message, StopReason, ThinkingLevel, UserContent,
    UserPart,
};
use pi_edit::EditMode;
use serde_json::{json, Map, Value};
use tokio::sync::mpsc::UnboundedSender;

use self::recovery::{FailedTurn, RecoveryConfig, Resolver, TurnRecovery, Verdict, EPHEMERAL_ROLE};
use crate::adapter::{json_round_trip, ProviderTarget};
use crate::agent::Target;
use crate::events::Framer;
use crate::persist::Persistence;
use crate::tools::edit_mode_for;
use omp_tui::magic::Ultra;

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
        Self {
            message: message.into(),
            code: None,
        }
    }

    #[must_use]
    pub fn coded(message: impl Into<String>, code: &'static str) -> Self {
        Self {
            message: message.into(),
            code: Some(code),
        }
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
    /// Todo completion reminders sent for the current prompt (TS
    /// `TodoTracker.#reminderCount`, reset per prompt).
    todo_reminders: u32,
    /// A reminder is in flight and the agent has not touched the todo list
    /// since (TS `#reminderAwaitingProgress`): stay silent until it does.
    todo_awaiting_progress: bool,
}

#[allow(clippy::struct_excessive_bools)] // wire-format state bag; bools are independent flags.
struct Settings {
    session_name: Option<String>,
    /// `"user"` | `"auto"`: a user-set name is not replaced by auto titles (TS `titleSource`).
    todo_phases: Value,
    /// `todo.reminders` / `todo.remindersMax` (config.yml) behind the
    /// `TodoTracker.checkCompletion` port.
    todo_reminders_enabled: bool,
    todo_reminders_max: u32,
    auto_compaction: bool,
    interrupt_mode: String,
    cache_warming: String,
    ask_dialog: bool,
    subagent_level: String,
    /// `"user"` titles beat later auto titles.
    session_name_source: Option<String>,
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
    /// Prompt runs seen by the Laya shadow judge (`OMP_LAYA_JUDGE=1`): its `turn_index`.
    judge_turns: std::sync::atomic::AtomicU64,
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
    /// The task running the current retry saga (`recover`); awaited after cancelling it.
    retry_task: Mutex<Option<tokio::task::JoinHandle<()>>>,
    /// The spawned post-turn task while it may still touch the session: the
    /// auto-compaction pass (or the plain settle). A prompt, `/new` or `/resume`
    /// cancels + awaits it so nothing of the old turn lands after the switch.
    post_turn_task: Mutex<Option<tokio::task::JoinHandle<()>>>,
    /// `compaction.*` policy (config.yml) and the in-flight flag behind `isCompacting`.
    compaction: Mutex<omp_compact::CompactionSettings>,
    compacting: std::sync::atomic::AtomicBool,
    /// Guard of the in-flight compaction (auto or manual): Esc / a queued prompt
    /// cancel its token (TS `#compactionAbortController` / `#autoCompactionAbortController`).
    compaction_guard: Mutex<Option<Arc<crate::compact::CommitGuard>>>,
    /// Fired whenever a compaction pass ends; prompts park on it behind a
    /// manual compaction (TS `waitForManualMaintenanceCleanup`).
    compaction_done: tokio::sync::Notify,
    /// Bumped by every session switch (`new_session`, `load_session*`): part of
    /// the compaction guard's identity (TS `sessionId()` + generation).
    switch_generation: Arc<std::sync::atomic::AtomicU64>,
    /// Injected summarizer (default: the live model).
    summarizer: Mutex<Option<crate::compact::SummarizeFn>>,
    /// Set by `abort()` (TS `#advisors.autoResumeSuppressed`): a follow-up-only
    /// queued-message drain stays suppressed until the next explicit user prompt.
    user_interrupt: std::sync::atomic::AtomicBool,
    /// Set while a manual compaction aborts the live turn (TS
    /// `disconnectFromAgent` around that abort): the empty aborted stub is
    /// neither rendered nor journaled — the turn resumes after the summary.
    compaction_abort: std::sync::atomic::AtomicBool,
    /// A suppressed stub `message_start` whose `message_end` is suppressed too.
    compaction_stub_start: std::sync::atomic::AtomicBool,
    /// Interactive-TUI hosts only: emit `context_usage` frames (status-line
    /// model identity + context gauge). RPC consumers never see them.
    ui_context_frames: std::sync::atomic::AtomicBool,
    /// `auto` thinking selector active (TS `ModelControls.#autoThinking`): the
    /// session runs a provisional concrete level and the status line shows
    /// `auto` until a level is classified.
    auto_thinking: std::sync::atomic::AtomicBool,
    /// Active ultra mode (`/ultra`, Shift+Tab past `max`): thinking at the
    /// model's highest level plus the ultracode notice on every prompt, and
    /// for `ultralong`/`hypercode` the `clm-host` context strategy.
    ultra: Mutex<Option<Ultra>>,
    /// Report stashed by a successful `rewind` tool result, consumed by the
    /// `TurnEnd` hook that truncates the context (TS `#pendingRewindReport`).
    pending_rewind_report: Mutex<Option<String>>,
    /// Sinks every applied session-name change (ext bridge `set_session_name`).
    on_session_name: Mutex<Option<NameListener>>,
    /// `session_before_switch` gate.
    switch_gate: Mutex<Option<SwitchGate>>,
    ext_host: Mutex<Option<Arc<omp_ext::ExtHost>>>,
    /// Ordered feed of the maintenance frames extensions subscribe to (`auto_compaction_*`,
    /// `auto_retry_*`): these session events are the rpc frames themselves.
    ext_frames: Mutex<Option<tokio::sync::mpsc::UnboundedSender<Value>>>,
    /// Sinks every completed session switch (`new_session`/`load_session`/fork):
    /// the ext bridge's ctx refresh + `session_switch` event.
    on_session_switch: Mutex<Option<SwitchListener>>,
    /// Extension compaction bridge (`session_before_compact` / `session_compact`).
    compaction_hooks: Mutex<Option<Arc<dyn CompactionHooks>>>,
    /// Extension `before_agent_start` hook run at every prompt admission.
    before_start: Mutex<Option<BeforeStartHook>>,
    /// Native Herdr agent-state reporter (`HERDR_ENV=1` only; `None` elsewhere).
    herdr: Option<std::sync::Arc<crate::herdr::Herdr>>,
    /// `ask`-tool runs currently waiting on the user: >0 means Blocked.
    ask_blocked: std::sync::atomic::AtomicU64,
    /// In-flight automatic title generation (`--mode print` awaits it before exiting).
    title_task: Mutex<Option<tokio::task::JoinHandle<()>>>,
}

/// Sink for applied session-name changes (the ext bridge's `set_session_name`).
pub type NameListener = Arc<dyn Fn(&str) + Send + Sync>;

/// Extension bridge around one compaction (TS `session_before_compact` / `session_compact`).
pub trait CompactionHooks: Send + Sync {
    /// Before summarizing: `true` = an extension cancelled. `branch_entries` is the active branch.
    fn before<'a>(
        &'a self,
        preparation: Value,
        branch_entries: Vec<Value>,
        custom_instructions: Option<&'a str>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = bool> + Send + 'a>>;
    /// After the `compaction` entry is committed.
    fn after<'a>(
        &'a self,
        compaction_entry: Value,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send + 'a>>;
}

/// Adapts [`CompactionHooks`] to the per-compaction [`crate::compact::CompactHook`] (adds the branch).
struct HookWithBranch<'a> {
    hooks: &'a dyn CompactionHooks,
    branch: Vec<Value>,
}

impl crate::compact::CompactHook for HookWithBranch<'_> {
    fn before<'a>(
        &'a self,
        preparation: Value,
        custom_instructions: Option<&'a str>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = bool> + Send + 'a>> {
        self.hooks
            .before(preparation, self.branch.clone(), custom_instructions)
    }
}

/// Hook run right before a prompt starts a run: `(prompt text, images JSON)`; returns the
/// messages extensions inject after the user message.
pub type BeforeStartHook = Arc<
    dyn Fn(
            String,
            Option<Value>,
        )
            -> std::pin::Pin<Box<dyn std::future::Future<Output = Vec<AgentMessage>> + Send>>
        + Send
        + Sync,
>;

/// Gate run before a session switch (`session_before_switch`): `true` cancels it.
pub type SwitchGate = Arc<
    dyn Fn(
            &'static str,
            Option<PathBuf>,
        ) -> std::pin::Pin<Box<dyn std::future::Future<Output = bool> + Send>>
        + Send
        + Sync,
>;

/// A `session_before_switch` handler cancelled the switch (TS `{cancelled: true}`).
#[derive(Debug)]
pub struct SwitchCancelled;

impl std::fmt::Display for SwitchCancelled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Session switch cancelled")
    }
}

impl std::error::Error for SwitchCancelled {}

/// Whether `e` is a [`SwitchCancelled`].
pub fn is_switch_cancelled(e: &anyhow::Error) -> bool {
    e.downcast_ref::<SwitchCancelled>().is_some()
}

/// Sink for completed session switches: `(reason, previous session file)`,
/// reason being the TS `session_switch` reason (`new` | `resume` | `fork`).
pub type SwitchListener = Arc<dyn Fn(&'static str, Option<PathBuf>) + Send + Sync>;

/// One title-model candidate: model plus the credentials to call it with.
type TitleCandidate = (
    pa_types::ai::Model,
    Option<String>,
    Option<std::collections::BTreeMap<String, String>>,
);

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

/// RSS of this process (`VmRSS` kB) — used by the `OMP_RS_TURN_TRACE` diagnostic.
pub(crate) fn rss_kb() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find_map(|l| l.strip_prefix("VmRSS:"))
                .and_then(|l| l.trim().strip_suffix(" kB"))
                .and_then(|n| n.parse().ok())
        })
        .unwrap_or(0)
}

/// `OMP_RS_TURN_TRACE=1`: stderr line per traced step, for turn-phase
/// allocation attribution (peak-RSS tuning). Off by default, zero cost.
macro_rules! turn_trace {
    ($($arg:tt)*) => {
        if std::env::var_os("OMP_RS_TURN_TRACE").is_some() {
            eprintln!($($arg)*);
        }
    };
}
pub(crate) use turn_trace;

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

/// An ultra mode by its wire name (`ultracode`, `ultradeep`, `ultralong`,
/// `hypercode`); thinking-level selectors and `/ultra` short forms are not.
#[must_use]
pub fn parse_ultra_name(s: &str) -> Option<Ultra> {
    Ultra::ALL.into_iter().find(|u| u.name() == s)
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
        Self::start_with(
            agent,
            target,
            cwd,
            persist,
            out,
            recovery::default_resolver(),
            None,
        )
        .await
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
        let live = agent.model().await;
        let edit_mode = edit_mode_for(&live.id);
        let recovery_cfg = match recovery_cfg {
            Some(cfg) => cfg,
            None => match resolver(format!("{}/{}", live.provider, live.id)).await {
                Ok(mut r) => {
                    apply_launch_fallbacks(&mut r);
                    agent.set_credential_id(r.credential_id);
                    RecoveryConfig::from_resolved(&r)
                }
                Err(_) => RecoveryConfig::default(),
            },
        };
        let recovery = TurnRecovery::new(recovery_cfg);
        persist.defer_errors(true);
        // RLM sub-call accounting: every `llm_query` provider call lands as a
        // `model_usage` entry in this session (bench/stats count sub-call tokens).
        crate::tools::rlm::set_entry_sink(Some(persist.entry_sink()));
        let compaction = crate::compact::load_settings();
        let todo_cfg = omp_settings::Settings::load(&cwd);
        // Herdr pane integration (HERDR_ENV=1 only): adopt the current
        // session (resume command + session identity) and report idle so the
        // pane is held from the start. No-op outside Herdr.
        let herdr = crate::herdr::global().inspect(|h| {
            let model_ref = format!("{}/{}", live.provider, live.id);
            h.set_session(persist.file().as_deref(), Some(&model_ref), "startup");
            h.report(crate::herdr::State::Idle, None);
        });
        let rt = Arc::new(Self {
            agent,
            target,
            cwd,
            persist,
            judge_turns: std::sync::atomic::AtomicU64::new(0),
            out,
            prompts: Mutex::new(PromptState::default()),
            settings: Mutex::new(Settings {
                session_name: None,
                session_name_source: None,
                todo_phases: json!([]),
                todo_reminders_enabled: todo_cfg.get_bool("todo.enabled").unwrap_or(true)
                    && todo_cfg.get_bool("todo.reminders").unwrap_or(true),
                todo_reminders_max: reminders_max(&todo_cfg),
                auto_compaction: compaction.enabled,
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
            retry_task: Mutex::new(None),
            post_turn_task: Mutex::new(None),
            user_interrupt: std::sync::atomic::AtomicBool::new(false),
            compaction_abort: std::sync::atomic::AtomicBool::new(false),
            compaction_stub_start: std::sync::atomic::AtomicBool::new(false),
            ui_context_frames: std::sync::atomic::AtomicBool::new(false),
            auto_thinking: std::sync::atomic::AtomicBool::new(false),
            ultra: Mutex::new(None),
            pending_rewind_report: Mutex::new(None),
            on_session_name: Mutex::new(None),
            on_session_switch: Mutex::new(None),
            switch_gate: Mutex::new(None),
            ext_host: Mutex::new(None),
            ext_frames: Mutex::new(None),
            before_start: Mutex::new(None),
            compaction_hooks: Mutex::new(None),
            herdr,
            ask_blocked: std::sync::atomic::AtomicU64::new(0),
            title_task: Mutex::new(None),
            compaction: Mutex::new(compaction),
            compacting: std::sync::atomic::AtomicBool::new(false),
            compaction_guard: Mutex::new(None),
            compaction_done: tokio::sync::Notify::new(),
            switch_generation: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            summarizer: Mutex::new(None),
        });
        rt.publish_session_file();
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
        if matches!(
            frame["type"].as_str(),
            Some(
                "auto_compaction_start"
                    | "auto_compaction_end"
                    | "auto_retry_start"
                    | "auto_retry_end"
            )
        ) {
            if let Some(tx) = lock(&self.ext_frames).as_ref() {
                let _ = tx.send(frame.clone());
            }
        }
        let _ = self.out.send(frame);
    }

    /// Publish the live session file to `omp-urls` (URL resolution cache
    /// scoping); `None` under `--no-session`. Call at every session-file
    /// attach point (start, `new_session`, `load_session`, fork/branch).
    pub fn publish_session_file(&self) {
        omp_urls::set_global_session_file(self.persist.file());
    }

    /// Session preamble omp writes: `model_change` then `thinking_level_change`.
    async fn record_model_state(&self) {
        let m = self.agent.model().await;
        let model = format!("{}/{}", m.provider, m.id);
        let _ = self.persist.append_model_change(&model, None, false);
        let ultra = self.ultra().map(Ultra::name);
        let _ = self.persist.append_thinking_level_change(
            Some(level_name(self.agent.thinking_level().await)),
            None,
            ultra,
        );
    }

    // ------------------------------------------------------------- events

    fn queued(&self) -> (Vec<String>, Vec<String>) {
        (
            self.agent.steering_previews(),
            self.agent.follow_up_previews(),
        )
    }

    /// `(steering, followUp)` previews of the queued messages.
    #[cfg(test)]
    pub fn queued_previews(&self) -> (Vec<String>, Vec<String>) {
        self.queued()
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
        self.emit_event(
            json!({"type": "queue_update", "steering": snapshot.0, "followUp": snapshot.1}),
        );
    }

    /// Forward a session event subject to `set_event_filter`.
    fn emit_event(&self, mut frame: Value) {
        let st = lock(&self.settings);
        let kind = frame
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        if st.event_filter.as_ref().is_some_and(|f| !f.contains(&kind)) {
            return;
        }
        if st.delta_updates && kind == "message_update" {
            if let Some(obj) = frame.as_object_mut() {
                obj.remove("message");
                if let Some(ev) = obj
                    .get_mut("assistantMessageEvent")
                    .and_then(Value::as_object_mut)
                {
                    ev.remove("partial");
                }
            }
        }
        drop(st);
        self.send(frame);
    }

    fn on_event(self: &Arc<Self>, event: &AgentEvent) {
        let ev_kind = match event {
            AgentEvent::MessageUpdate { .. } => "message_update",
            AgentEvent::MessageEnd { message } => match message.role() {
                "user" => "message_end:user",
                "assistant" => "message_end:assistant",
                _ => "message_end:toolResult",
            },
            AgentEvent::AgentEnd { .. } => "agent_end",
            AgentEvent::TurnEnd { .. } => "turn_end",
            AgentEvent::AgentStart => "agent_start",
            AgentEvent::TurnStart => "turn_start",
            AgentEvent::MessageStart { .. } => "message_start",
            AgentEvent::ToolExecutionStart { .. } => "tool_execution_start",
            AgentEvent::ToolExecutionUpdate { .. } => "tool_execution_update",
            AgentEvent::ToolExecutionEnd { .. } => "tool_execution_end",
        };
        let trace = std::env::var_os("OMP_RS_TURN_TRACE").is_some();
        if trace {
            turn_trace!("[turn-trace] event={ev_kind} rss={}kB", rss_kb());
        }
        let frame = lock(&self.framer).frame(event);
        if trace {
            let n = serde_json::to_string(&frame).map_or(0, |s| s.len());
            turn_trace!(
                "[turn-trace] framed={ev_kind} frame_json={n}B rss={}kB",
                rss_kb()
            );
        }
        // context-usage frames are TUI chrome; rpc/print clients never read
        // them and each publish re-serializes the whole context. Skip in
        // non-interactive sessions (stdout is what rpc/print write to).
        let interactive = std::io::IsTerminal::is_terminal(&std::io::stdout());
        if interactive
            && matches!(
                event,
                AgentEvent::AgentEnd { .. }
                    | AgentEvent::MessageEnd {
                        message: AgentMessage::Standard(
                            Message::Assistant(_) | Message::ToolResult(_)
                        )
                    }
            )
        {
            self.publish_context_usage();
        }
        if let AgentEvent::AgentEnd { messages } = event {
            self.on_agent_end(frame, messages);
            return;
        }
        let mut frame = frame;
        if self
            .compaction_abort
            .load(std::sync::atomic::Ordering::SeqCst)
        {
            use std::sync::atomic::Ordering::SeqCst;
            match event {
                // Failure path: the stub's own start/end pair — render neither.
                AgentEvent::MessageStart { message }
                    if crate::persist::is_empty_aborted(message) =>
                {
                    self.compaction_stub_start.store(true, SeqCst);
                    return;
                }
                AgentEvent::MessageEnd { message } if crate::persist::is_empty_aborted(message) => {
                    if self.compaction_stub_start.swap(false, SeqCst) {
                        return;
                    }
                    // A streamed-but-empty turn: settle its open block without the error row.
                    if let Some(m) = frame.get_mut("message").and_then(Value::as_object_mut) {
                        m.remove("errorMessage");
                    }
                }
                _ => {}
            }
        }
        self.emit_event(frame);
        match event {
            AgentEvent::AgentStart => {
                let mut p = lock(&self.prompts);
                p.agent_starts += 1;
                p.active = true;
                drop(p);
                self.herdr_working();
            }
            AgentEvent::ToolExecutionStart {
                tool_name, args, ..
            } if tool_name == "ask" => {
                // Herdr: an ask dialog blocks on the user.
                self.ask_blocked
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let message = args
                    .get("questions")
                    .and_then(Value::as_array)
                    .and_then(|qs| {
                        qs.iter()
                            .find_map(|q| q.get("question").and_then(Value::as_str))
                    })
                    .map(str::to_string);
                if let (Some(h), Some(m)) = (&self.herdr, message) {
                    h.report(crate::herdr::State::Blocked, Some(&m));
                } else if let Some(h) = &self.herdr {
                    h.report(crate::herdr::State::Blocked, Some("waiting for user input"));
                }
            }
            AgentEvent::ToolExecutionEnd { tool_name, .. } if tool_name == "ask" => {
                let prev = self
                    .ask_blocked
                    .fetch_update(
                        std::sync::atomic::Ordering::SeqCst,
                        std::sync::atomic::Ordering::SeqCst,
                        |n| Some(n.saturating_sub(1)),
                    )
                    .unwrap_or(0);
                if prev == 1 {
                    self.herdr_working();
                }
            }
            AgentEvent::MessageEnd { message } if message.role() == "user" => {
                self.emit_queue_update();
            }
            AgentEvent::MessageEnd {
                message: AgentMessage::Standard(Message::Assistant(a)),
            } => {
                self.on_assistant_settled(a);
            }
            // Checkpoint runtime hooks (reports/Checkpoint.md): a successful
            // `checkpoint` tool result arms the shared checkpoint state with the
            // post-append message count; a successful `rewind` result stashes
            // its report for the TurnEnd truncation.
            AgentEvent::MessageEnd {
                message: AgentMessage::Standard(Message::ToolResult(tr)),
            } => {
                self.on_tool_result_settled(tr);
            }
            AgentEvent::TurnEnd { .. } => {
                self.maybe_apply_rewind();
            }
            _ => {}
        }
    }

    /// TS `AgentSession`'s checkpoint toolResult leg of `#checkpointActiveReminderFor`:
    /// arm the shared checkpoint state with the message count AFTER the
    /// checkpoint tool result is appended (async: the listener runs sync while
    /// the agent holds its state lock, so the count is read on a task).
    fn on_tool_result_settled(&self, tr: &pa_agent::types::ToolResultMessage) {
        if tr.is_error {
            return;
        }
        if tr.tool_name == "checkpoint" {
            let started_at = tr
                .details
                .as_ref()
                .and_then(|d| d.get("startedAt"))
                .and_then(Value::as_str)
                .map_or_else(crate::persist::now_iso, str::to_owned);
            let agent = self.agent.clone();
            tokio::spawn(async move {
                let count = agent.message_count().await;
                crate::tools::checkpoint::shared().set_checkpoint_state(
                    crate::tools::checkpoint::CheckpointState {
                        checkpoint_message_count: count,
                        checkpoint_entry_id: None,
                        started_at,
                    },
                );
            });
        } else if tr.tool_name == "rewind" {
            let report = tr
                .details
                .as_ref()
                .and_then(|d| d.get("report"))
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|r| !r.is_empty())
                .map(str::to_owned)
                .or_else(|| {
                    tr.content.iter().find_map(|part| match part {
                        pa_agent::types::ToolResultContent::Text(t) => {
                            let text = t.text.trim();
                            (!text.is_empty()).then(|| text.to_owned())
                        }
                        pa_agent::types::ToolResultContent::Image(_) => None,
                    })
                });
            if let Some(report) = report {
                *lock(&self.pending_rewind_report) = Some(report);
            }
        } else if tr.tool_name == "todo" {
            self.on_todo_settled(tr);
        }
    }

    /// TS `TodoTracker` state leg: a successful state-changing `todo` call is
    /// the canonical snapshot (`details.phases`). Track it for the settle-time
    /// completion reminder and emit `todo_auto_clear` when nothing is left open
    /// (the TUI drops the plan on that event).
    fn on_todo_settled(&self, tr: &pa_agent::types::ToolResultMessage) {
        let Some(phases) = tr
            .details
            .as_ref()
            .and_then(|d| d.get("phases"))
            .filter(|p| p.is_array())
        else {
            return;
        };
        lock(&self.settings).todo_phases = phases.clone();
        // Any progress clears the reminder-awaiting flag (TS todo touch).
        lock(&self.prompts).todo_awaiting_progress = false;
        // A read-only `view` echoes the current state: refresh the plan, but
        // only a state-changing call that leaves nothing open auto-clears it.
        let read_only = tr
            .details
            .as_ref()
            .and_then(|d| d.get("op"))
            .and_then(Value::as_str)
            .is_some_and(|op| op == "view");
        if !read_only && !has_incomplete_tasks(phases) {
            self.emit_event(crate::events::todo_auto_clear_frame());
        }
    }

    /// TS `setOnTurnEnd`'s rewind leg (`#applyRewind`): when this turn carried a
    /// successful `rewind` result, cut the context back to the checkpoint in a
    /// spawned task — the listener runs under the agent's state lock, so the
    /// `set_messages` surgery happens after the run went idle.
    fn maybe_apply_rewind(self: &Arc<Self>) {
        let Some(report) = lock(&self.pending_rewind_report).take() else {
            return;
        };
        if crate::tools::checkpoint::shared()
            .checkpoint_state()
            .is_none()
        {
            return;
        }
        let rt = self.clone();
        tokio::spawn(async move {
            rt.agent.wait_for_idle().await;
            let Some(state) = crate::tools::checkpoint::shared().checkpoint_state() else {
                return;
            };
            let count = state.checkpoint_message_count;
            // Only the kept prefix is copied (not the whole transcript, twice).
            let prefix = rt
                .agent
                .with_messages(|m| {
                    (m.len() > count).then(|| m.iter().take(count).cloned().collect::<Vec<_>>())
                })
                .await;
            if let Some(mut messages) = prefix {
                // Carry the report into the cut context like the TS branch view's
                // rewind-report custom message.
                messages.push(AgentMessage::Custom(pa_agent::types::CustomAgentMessage {
                    role: "agent".to_string(),
                    payload: serde_json::json!({
                        "customType": "rewind-report",
                        "content": report,
                        "display": false,
                        "details": {
                            "report": report,
                            "startedAt": state.started_at,
                            "rewoundAt": crate::persist::now_iso(),
                        },
                    }),
                }));
                rt.agent.set_messages(messages).await;
            }
            // Persist the `rewind-report` entry (TS `appendCustomMessageEntry`).
            let details = serde_json::json!({
                "report": report,
                "startedAt": state.started_at,
                "rewoundAt": crate::persist::now_iso(),
            });
            let _ = rt.persist.append_entry(
                "custom_message",
                serde_json::json!({
                    "customType": "rewind-report",
                    "content": report,
                    "display": false,
                    "details": details,
                    "attribution": "agent",
                }),
            );
            crate::tools::checkpoint::shared().set_last_completed_rewind(
                crate::tools::checkpoint::CompletedRewindState {
                    report,
                    started_at: state.started_at,
                    rewound_at: crate::persist::now_iso(),
                },
            );
            crate::tools::checkpoint::shared().clear_checkpoint_state();
        });
    }

    /// `onAssistantSettledSuccessfully`: a turn that produced output settled on the live model.
    fn on_assistant_settled(&self, a: &pa_agent::types::AssistantMessage) {
        if matches!(a.stop_reason, StopReason::Error | StopReason::Aborted) || a.content.is_empty()
        {
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

    /// Laya shadow judge (default OFF, `OMP_LAYA_JUDGE=1`): hand the finished run to the
    /// background worker. Never blocks or fails the turn: nothing is built when disabled, the
    /// queue is bounded and an overflowing turn is dropped (counted).
    fn shadow_judge_turn(&self, messages: &[AgentMessage]) {
        let Some(worker) = omp_judge::shadow::global() else {
            return;
        };
        let entries =
            omp_judge::shadow::message_entries(messages.iter().map(crate::events::message_value));
        let turn_index = self
            .judge_turns
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        worker.submit(omp_judge::shadow::Job {
            session_id: self.persist.session_id(),
            turn_index,
            ts: crate::persist::now_iso(),
            entries,
            is_sub: false,
        });
    }

    /// Terminal `agent_end`, or the retry saga when the run failed transiently.
    fn on_agent_end(self: &Arc<Self>, frame: Value, messages: &[AgentMessage]) {
        self.emit_queue_update();
        self.shadow_judge_turn(messages);
        let failed = last_assistant_msg(messages).and_then(FailedTurn::from_message);
        // Cheap pre-check only; the threshold decision itself (`checkCompaction`)
        // runs on the live typed context inside the task, off the agent's state
        // lock — no journal DOM is built for a turn that will not compact.
        let auto_maybe = failed.is_none()
            && !self.compacting.load(std::sync::atomic::Ordering::SeqCst)
            && lock(&self.settings).auto_compaction
            && last_assistant_msg(messages)
                .is_some_and(|a| !matches!(a.stop_reason, StopReason::Error | StopReason::Aborted));
        if let Some(f) = failed.filter(|f| lock(&self.recovery).is_retryable(f)) {
            // Herdr: the retry saga holds the pane Blocked until the backoff
            // elapses and the run continues (AgentStart -> working).
            if let Some(h) = &self.herdr {
                let msg = f.message.chars().take(200).collect::<String>();
                h.report(crate::herdr::State::Blocked, Some(&msg));
            }
            let rt = self.clone();
            let messages = messages.to_vec();
            // The listener runs while the agent holds its state lock: recovery (model swap,
            // context surgery, continuing the run) happens after the run went idle.
            let handle = tokio::spawn(async move { rt.recover(frame, messages, f).await });
            *lock(&self.retry_task) = Some(handle);
            return;
        }
        if auto_maybe {
            let rt = self.clone();
            let messages = messages.to_vec();
            // Like `recover`: the listener runs under the agent's state lock.
            let handle = tokio::spawn(async move {
                if rt.auto_compaction_due().await {
                    rt.emit_event(frame);
                    rt.run_auto_compaction().await;
                    rt.report_prompts(&run_outcome(&messages));
                    rt.drain_stranded_queue();
                } else {
                    rt.settle_terminal(frame, &messages, None);
                }
            });
            // A prompt / session switch behind this task cancels + awaits it
            // (bug-hunt #2: the pass used to run untracked, fire-and-forget).
            *lock(&self.post_turn_task) = Some(handle);
            return;
        }
        self.settle_terminal(frame, messages, None);
    }

    /// Final `agent_end` of a prompt: persist/close failure bookkeeping, then `prompt_result`s.
    fn settle_terminal(
        self: &Arc<Self>,
        mut frame: Value,
        messages: &[AgentMessage],
        error_override: Option<&str>,
    ) {
        // Herdr: the turn is over — the pane waits for input.
        self.herdr_idle();
        let mut messages = messages.to_vec();
        if let Some(AgentMessage::Standard(Message::Assistant(a))) =
            messages.iter_mut().rev().find(|m| m.role() == "assistant")
        {
            if a.stop_reason == StopReason::Error {
                if let Some(e) = error_override {
                    a.error_message = Some(e.to_string());
                    if let Some(last) = frame
                        .get_mut("messages")
                        .and_then(Value::as_array_mut)
                        .and_then(|v| {
                            v.iter_mut()
                                .rev()
                                .find(|m| m.get("role") == Some(&json!("assistant")))
                        })
                    {
                        last["errorMessage"] = json!(e);
                    }
                }
                self.persist.flush_deferred_error(error_override);
                let text = a.error_message.clone().unwrap_or_default();
                // Never emit while holding the recovery lock (`emit_event` takes `settings`,
                // `state()` takes `settings` then `recovery`).
                let end = lock(&self.recovery).on_error_settled(&text);
                if let Some(end) = end {
                    self.emit_event(end);
                }
            }
        }
        // TS `TodoTracker.checkCompletion`: a successful terminal turn with
        // incomplete todos nudges the agent to continue instead of settling.
        // Not for failed/aborted turns, and never over a user interrupt (the
        // reminder would resurrect the run the user just stopped).
        let failed_turn = messages
            .iter()
            .rev()
            .find(|m| m.role() == "assistant")
            .and_then(|m| match m {
                AgentMessage::Standard(Message::Assistant(a)) => Some(a),
                _ => None,
            })
            .is_some_and(|a| matches!(a.stop_reason, StopReason::Error | StopReason::Aborted));
        let interrupted = self
            .user_interrupt
            .load(std::sync::atomic::Ordering::SeqCst);
        if error_override.is_none()
            && !failed_turn
            && !interrupted
            && self.maybe_todo_reminder(&messages, frame.clone())
        {
            return;
        }
        let outcome = run_outcome(&messages);
        self.emit_event(frame);
        self.report_prompts(&outcome);
        self.drain_stranded_queue();
    }

    /// TS `TodoTracker.checkCompletion`: a successful terminal turn that stops
    /// with incomplete todos emits `todo_reminder`, appends the reminder as a
    /// hidden developer message, and continues the run — the `agent_end` goes
    /// out non-terminal (`isTerminal: false`, `yielded: false`, a plain
    /// `willContinue` settle, not `awaitingAsyncWork`). Returns `true` when the
    /// reminder took over the settle.
    fn maybe_todo_reminder(self: &Arc<Self>, messages: &[AgentMessage], mut frame: Value) -> bool {
        let (enabled, max_attempts, phases) = {
            let st = lock(&self.settings);
            (
                st.todo_reminders_enabled,
                st.todo_reminders_max,
                st.todo_phases.clone(),
            )
        };
        let mut p = lock(&self.prompts);
        if p.todo_awaiting_progress {
            return false;
        }
        if !enabled {
            p.todo_reminders = 0;
            return false;
        }
        if p.todo_reminders >= max_attempts {
            return false;
        }
        let incomplete = incomplete_todos(&phases);
        if incomplete.as_array().is_none_or(Vec::is_empty) {
            p.todo_reminders = 0;
            p.todo_awaiting_progress = false;
            return false;
        }
        let awaiting_user = messages
            .iter()
            .rev()
            .find(|m| m.role() == "assistant")
            .is_some_and(|m| is_awaiting_user_answer(&message_text(m)));
        if awaiting_user {
            return false;
        }
        p.todo_reminders += 1;
        p.todo_awaiting_progress = true;
        let attempt = p.todo_reminders;
        drop(p);
        self.emit_event(crate::events::todo_reminder_frame(
            &incomplete,
            attempt,
            max_attempts,
        ));
        let list = todo_reminder_list(&phases);
        let count = incomplete.as_array().map_or(0, Vec::len);
        let reminder = format!(
            "<system-reminder>\nYou stopped with {count} incomplete todo item(s):\n{list}\n\nPlease continue working on these tasks or mark them complete if finished.\n(Reminder {attempt}/{max_attempts})\n</system-reminder>"
        );
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(0));
        let reminder = AgentMessage::Custom(pa_agent::types::CustomAgentMessage {
            role: "developer".to_string(),
            payload: json!({
                "customType": "todo-reminder",
                "content": [{"type": "text", "text": reminder}],
                "display": false,
                "attribution": "agent",
                "timestamp": timestamp,
            }),
        });
        frame["isTerminal"] = json!(false);
        frame["yielded"] = json!(false);
        self.emit_event(frame.clone());
        let rt = self.clone();
        let messages = messages.to_vec();
        tokio::spawn(async move {
            // TS `agent.appendMessage` + `sessionManager.appendMessage`, before
            // the scheduled continue (`scheduleAgentContinue`).
            rt.agent.append_messages(vec![reminder.clone()]).await;
            // `json_round_trip` to pa_types returns None for custom roles like
            // "developer"; `seed_resumed_message` persists the verbatim shape.
            seed_resumed_message(&rt.persist, &reminder, None);
            if let Err(e) = rt.continue_after_retry().await {
                eprintln!("omt: todo reminder continuation failed: {e:#}");
                rt.settle_terminal(frame, &messages, None);
            }
        });
        true
    }

    /// Emits a transient `notice` frame (TS `AgentSession.emitNotice`): the TUI
    /// renders info/warning/error rows; RPC clients receive the same shape.
    // Not wired yet: session-persistence failure notices live in persist.rs
    // (another owner); the TUI asks for this exact emission point.
    #[allow(dead_code)]
    pub fn emit_notice(&self, level: &str, message: &str, source: Option<&str>) {
        self.emit_event(crate::events::notice_frame(level, message, source));
    }

    /// Emits a `tool_stream_update` frame for a streaming tool's incremental
    /// preview (TS `tool_stream_update`, e.g. edit-tool diff hunks). pa-agent
    /// tools do not stream yet; this is the emission point for when they do —
    /// the TUI already renders the shape.
    #[allow(dead_code)] // awaits a streaming pa-agent tool; the TUI consumes the shape already.
    pub fn emit_tool_stream_update(&self, tool_call_id: &str, tool_name: &str, update: &Value) {
        self.emit_event(crate::events::tool_stream_update_frame(
            tool_call_id,
            tool_name,
            update,
        ));
    }

    /// TS `#drainStrandedQueuedMessages` + `#canAutoContinueForFollowUp`: once the
    /// session settles, a steer/follow-up stranded after the loop's final queue poll
    /// (e.g. by an abort) starts a fresh turn via `Agent::continue_run`. A
    /// follow-up-only resume stays suppressed after a user interrupt until the next
    /// explicit prompt; a queued steer always resumes (it injects before the first
    /// provider call, so any tail is valid). The drain turn is a fresh run with its
    /// own abort signal (TS `#scheduleAgentContinue` -> `agent.continue()`), so an
    /// empty Enter that interrupts a turn delivers the queued steer as a real turn.
    fn drain_stranded_queue(self: &Arc<Self>) {
        let (steering, _follow) = self.queued();
        let interrupted = self
            .user_interrupt
            .load(std::sync::atomic::Ordering::SeqCst);
        if steering.is_empty() && interrupted {
            return;
        }
        let rt = self.clone();
        tokio::spawn(async move {
            rt.agent.wait_for_idle().await;
            if !rt.agent.has_queued_messages() {
                return;
            }
            let _ = rt.agent.continue_run_until_accepted().await;
        });
    }

    /// Retry saga for a failed turn (`#handleRetryableError`), then continue the run.
    async fn recover(
        self: Arc<Self>,
        frame: Value,
        messages: Vec<AgentMessage>,
        failed: FailedTurn,
    ) {
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
        let t = self
            .target
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        t.as_ref()
            .map(|t| (String::from(&*t.model.provider), t.model.id.clone()))
    }

    /// `provider/id[:level]` of the live model (`formatRetryFallbackSelector`).
    fn live_selector_with_level(&self, level: Option<ThinkingLevel>) -> String {
        let (p, i) = self.live_provider_id().unwrap_or_default();
        match level {
            Some(l) => format!("{p}/{i}:{}", level_name(l)),
            None => format!("{p}/{i}"),
        }
    }

    /// `#rotateAccountOnRateLimit`: the turn failed with a provider rate limit and the
    /// provider's credential store holds more than one account — hand the same model to the
    /// next account instead of only backing off.
    ///
    /// Returns `true` when an alternate credential took over (the caller treats the attempt
    /// as switched and skips the model fallback consult for this pass).
    async fn rotate_account_on_rate_limit(&self, failed: &FailedTurn) -> bool {
        if !lock(&self.recovery).is_rate_limit(failed) {
            return false;
        }
        let Some((provider, model_id)) = self.live_provider_id() else {
            return false;
        };
        let selector = recovery::model_selector(&provider, &model_id);
        let now = u64::try_from(pa_agent::now_ms()).unwrap_or(0);
        // The provider-level cooldown the fallback state just recorded decides whether a
        // recently-failed alternate may be retried already.
        let cooldown_active = lock(&self.recovery)
            .state_mut()
            .is_suppressed(&selector, now);
        let pool = match omp_providers::auth::credential_pool_async(
            omp_providers::AgentDir::default(),
            provider.clone(),
        )
        .await
        {
            Some(p) if p.has_alternates() => p,
            _ => return false,
        };
        let Some(next) =
            lock(&self.recovery)
                .accounts_mut()
                .next(&provider, &model_id, &pool, cooldown_active)
        else {
            return false;
        };
        // Same model, other account: swap the live provider target's key and pin the new row.
        let mut resolved = match (self.resolver)(selector.clone()).await {
            Ok(r) => r,
            Err(_) => return false,
        };
        resolved.api_key = Some(next.key.clone());
        resolved.credential_id = Some(next.id);
        let from_id = self
            .target
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_ref()
            .and_then(|t| t.credential_id);
        // Guard against swap_model's silent no-op (unconvertible model): only mutate the
        // rotation state when the swap actually applied.
        let convertible: Option<pa_agent::types::Model> = json_round_trip(&resolved.model);
        if convertible.is_none() {
            return false;
        }
        if !self
            .swap_model(&resolved, Some(recovery::EPHEMERAL_ROLE), true, true)
            .await
        {
            return false;
        }
        {
            let mut rec = lock(&self.recovery);
            rec.accounts_mut().promote(&provider, &model_id, next.id);
            if let Some(from) = from_id {
                rec.accounts_mut().burn(&provider, &model_id, from);
            }
        }
        let from_label = from_id
            .map(|id| id.to_string())
            .unwrap_or_else(|| "?".into());
        self.emit_event(recovery::account_rotated_frame(
            &selector,
            &selector,
            Some(&format!("rate limit: account {from_label} → {}", next.id)),
        ));
        true
    }

    // --- Herdr pane integration (all no-ops unless HERDR_ENV=1) ---

    /// Report `working`, unless an `ask` dialog still holds the pane Blocked.
    fn herdr_working(&self) {
        if let Some(h) = &self.herdr {
            if self.ask_blocked.load(std::sync::atomic::Ordering::SeqCst) == 0 {
                h.report(crate::herdr::State::Working, None);
            }
        }
    }

    fn herdr_idle(&self) {
        if let Some(h) = &self.herdr {
            h.report(crate::herdr::State::Idle, None);
        }
    }

    /// Session switch: rebuild the resume command and report the new identity.
    fn herdr_session(&self, start_source: &str) {
        if let Some(h) = &self.herdr {
            let model_ref = self.live_selector_with_level(None);
            h.set_session(
                self.persist.file().as_deref(),
                Some(&model_ref),
                start_source,
            );
        }
    }

    /// Port of `#handleRetryableError`: fallback consult, budget/fail-fast checks, backoff.
    async fn handle_retryable(self: &Arc<Self>, failed: &FailedTurn) -> Recovered {
        // A provider rate limit with a multi-account pool: same model, next account — before
        // any cross-model fallback consult.
        if self.rotate_account_on_rate_limit(failed).await {
            return self.finish_account_rotation(failed).await;
        }
        let level = self.agent.thinking_level().await;
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
            let Ok(resolved) = (self.resolver)(selector.clone()).await else {
                continue;
            };
            if resolved.api_key.is_none() {
                continue;
            }
            if self
                .apply_fallback(&resolved, &selector, &current, level)
                .await
            {
                switched = true;
                break;
            }
        }
        let verdict = lock(&self.recovery).finish_attempt(&attempt, switched);
        let delay_ms = match verdict {
            Verdict::GiveUp {
                frame,
                error_message,
            } => {
                self.emit_event(frame);
                return Recovered::Final {
                    error: error_message,
                };
            }
            Verdict::Retry { delay_ms } => delay_ms,
        };
        let announced = lock(&self.recovery).attempt();
        let max = attempt.max;
        self.emit_event(recovery::auto_retry_start_frame(
            announced,
            max,
            delay_ms,
            &attempt.error_message,
        ));
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

    /// Finalize a rate-limit retry pass whose switch was an account rotation: announce the
    /// backoff, drop the failed turn, and park the saga in the cancelable sleep — the same
    /// epilogue a model fallback has, without the fallback consult.
    async fn finish_account_rotation(&self, failed: &FailedTurn) -> Recovered {
        let (attempt, verdict) = {
            let mut rec = lock(&self.recovery);
            let attempt = rec.begin_attempt(failed, rand::random::<f64>());
            let verdict = rec.finish_attempt(&attempt, true);
            (attempt, verdict)
        };
        let delay_ms = match verdict {
            Verdict::GiveUp {
                frame,
                error_message,
            } => {
                self.emit_event(frame);
                return Recovered::Final {
                    error: error_message,
                };
            }
            Verdict::Retry { delay_ms } => delay_ms,
        };
        let announced = lock(&self.recovery).attempt();
        let max = attempt.max;
        self.emit_event(recovery::auto_retry_start_frame(
            announced,
            max,
            delay_ms,
            &attempt.error_message,
        ));
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
        self.swap_model(resolved, Some(EPHEMERAL_ROLE), true, true)
            .await;
        let applied = self.apply_thinking(requested).await;
        let applied_name = level_name(applied).to_string();
        let role = {
            let mut rec = lock(&self.recovery);
            rec.commit_switch(current, Some(from_thinking), Some(applied_name));
            rec.role().to_string()
        };
        self.emit_event(recovery::fallback_applied_frame(
            current, selector, &role, None,
        ));
        true
    }

    /// Before a prompt: return to the primary model once its cooldown expired (`maybeRestoreRetryFallbackPrimary`).
    async fn maybe_restore_primary(&self) {
        let level = self.agent.thinking_level().await;
        let current = self.live_selector_with_level(Some(level));
        let now = u64::try_from(pa_agent::now_ms()).unwrap_or(0);
        let plan =
            lock(&self.recovery)
                .state_mut()
                .restore_plan(&current, Some(level_name(level)), now);
        let Some(plan) = plan else { return };
        let Ok(resolved) = (self.resolver)(plan.selector.clone()).await else {
            return;
        };
        if resolved.api_key.is_none() {
            return;
        }
        let restored_level = plan.thinking.as_deref().and_then(parse_level);
        // Ownership ends before the swap so observers never see the primary tagged as a fallback.
        lock(&self.recovery).state_mut().clear_active();
        self.swap_model(&resolved, Some(EPHEMERAL_ROLE), false, true)
            .await;
        if let Some(l) = restored_level {
            self.apply_thinking(l).await;
        }
    }

    /// `prompt_result` / `session_settled` land one tick after the terminal `agent_end` (omp's
    /// `setImmediate`): the response of the command that ended the run (e.g. `abort`) goes first and
    /// queue drains are visible when `sessionSettled` is read.
    fn report_prompts(self: &Arc<Self>, outcome: &RunOutcome) {
        // Tickets of the runs that just ended are claimed now: a prompt admitted during the delay
        // below belongs to the next run and must not inherit this result.
        let claimed = {
            let mut p = lock(&self.prompts);
            let starts = p.agent_starts;
            let (ready, rest): (Vec<_>, Vec<_>) = p
                .open
                .drain(..)
                .partition(|t| !t.waiting && starts > t.starts_at_begin);
            p.open = rest;
            ready
        };
        let rt = self.clone();
        let outcome = outcome.clone();
        tokio::spawn(async move {
            // `prompt_result` must never precede run cleanup (`is_streaming =
            // false`, run slot taken): a client that fires its next prompt on
            // prompt_result otherwise races the still-registered run and gets
            // spurious "Agent is already processing" rejections — near-certain
            // with extensions, where the AgentEnd listener chain (ext-host RPC)
            // outlasts the 2 ms delay below. Wait for the run to actually
            // settle first, then keep omp's one-tick delay so the command
            // response still goes out before the result frame.
            rt.agent.wait_for_idle().await;
            tokio::time::sleep(std::time::Duration::from_millis(2)).await;
            rt.report_prompts_now(&outcome, claimed);
        });
    }

    /// Emit `prompt_result` frames then `session_settled` after a terminal end.
    fn report_prompts_now(&self, outcome: &RunOutcome, claimed: Vec<Ticket>) {
        let drained = self.queued_count() == 0;
        let mut p = lock(&self.prompts);
        let mut remaining = Vec::new();
        let mut reports = claimed;
        for t in p.open.drain(..) {
            if t.waiting && drained {
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
        self: &Arc<Self>,
        id: Option<Value>,
        message: &str,
        behavior: Option<&str>,
        images: Vec<pa_agent::types::ImageContent>,
    ) -> Result<(), CmdError> {
        // A prompt during in-flight compaction cancels it and waits for the
        // pass to settle first (TS prompt-side `abortCompaction` +
        // `waitForManualMaintenanceCleanup`): the turn must start on the
        // settled context, not race the compaction's `set_messages`. Runs
        // before the auto-title spawn so the title task cannot append its way
        // into a running pass.
        self.cancel_compaction_and_wait().await;
        self.spawn_auto_title(message);
        if self.agent.is_streaming().await {
            let steer = match behavior {
                Some("steer") => true,
                Some("followUp") => false,
                _ => {
                    return Err(CmdError::new(
                        "Agent is already processing a prompt. Specify streamingBehavior ('steer' or 'followUp') to queue the message.",
                    ))
                }
            };
            // Queued prompts carry their notices in the same batch, ahead of the message.
            let mut batch = self.prompt_notices(message, false).await;
            let queued = if batch.is_empty() {
                pa_agent::agent::AgentMessageBatch::Single(user_message(message))
            } else {
                batch.push(user_message(message));
                pa_agent::agent::AgentMessageBatch::Batch(batch)
            };
            if steer {
                self.agent.steer(queued);
            } else {
                self.agent.follow_up(queued);
            }
            lock(&self.prompts).open.push(Ticket {
                id,
                starts_at_begin: 0,
                waiting: true,
            });
            self.emit_queue_update();
            return Ok(());
        }
        self.user_interrupt
            .store(false, std::sync::atomic::Ordering::SeqCst);
        {
            // TS `TodoTracker.resetForPrompt`: reminder budget is per prompt.
            let mut p = lock(&self.prompts);
            p.todo_reminders = 0;
            p.todo_awaiting_progress = false;
        }
        // A new prompt supersedes a pending retry (and waits until the old turn is settled).
        self.cancel_retry_and_wait().await;
        self.maybe_restore_primary().await;
        let starts = {
            let mut p = lock(&self.prompts);
            let starts = p.agent_starts;
            p.open.push(Ticket {
                id: id.clone(),
                starts_at_begin: starts,
                waiting: false,
            });
            starts
        };
        let hook = lock(&self.before_start).clone();
        let mut injected = Vec::new();
        if let Some(hook) = hook {
            let images_json = (!images.is_empty())
                .then(|| serde_json::to_value(&images).ok())
                .flatten();
            injected = hook(message.to_string(), images_json).await;
        }
        let notices = self.prompt_notices(message, true).await;
        let input = prompt_input(message, images, notices, injected);
        let pua = self.agent.prompt_until_accepted(input).await;
        if let Err(e) = pua {
            lock(&self.prompts)
                .open
                .retain(|t| !(t.starts_at_begin == starts && t.id == id && !t.waiting));
            return Err(CmdError::new(format!("{e:#}")));
        }
        Ok(())
    }

    /// Hidden notices for a prompt (magic keywords, ultra mode). For a prompt
    /// that starts a turn (`starts_turn`), `ultrathink` under `auto` thinking
    /// also lifts the turn to the model's highest level (TS
    /// `applyAutoThinkingLevel`); queued messages leave the level alone.
    async fn prompt_notices(&self, message: &str, starts_turn: bool) -> Vec<AgentMessage> {
        let present = omp_tui::magic::present(message);
        let mode = self.ultra();
        if present.is_empty() && mode.is_none() {
            return Vec::new();
        }
        let settings = omp_settings::Settings::load(&self.cwd);
        if starts_turn
            && self
                .auto_thinking
                .load(std::sync::atomic::Ordering::Relaxed)
            && present.iter().any(|k| k.id == "ultrathink")
            && omp_tui::magic::enabled(&settings, "ultrathink")
        {
            self.ultrathink_auto().await;
        }
        let tools = self
            .agent
            .tools()
            .await
            .iter()
            .map(|t| t.name().to_owned())
            .collect();
        let ctx = crate::notices::Context::new(tools, &settings);
        crate::notices::build(message, &settings, &ctx, mode, pa_agent::now_ms())
    }

    /// `ultrathink` under `auto`: resolve this turn to the highest supported
    /// effort, journaled as an auto resolution.
    async fn ultrathink_auto(&self) {
        let Some(model) = self.live_model().filter(|m| m.reasoning) else {
            return;
        };
        let effective = crate::model::clamp_level(&model, ThinkingLevel::Max);
        let name = level_name(effective);
        if self.agent.thinking_level().await != effective {
            self.agent.set_thinking_level(effective).await;
            let _ = self
                .persist
                .append_thinking_level_change(Some(name), Some("auto"), None);
        }
        self.emit_event(json!({
            "type": "thinking_level_changed",
            "thinkingLevel": name,
            "configured": "auto",
            "resolved": name,
            "ultra": null,
        }));
    }

    /// Cancel a pending retry backoff and wait for its saga task to finish reporting, so nothing
    /// of the old turn can land after (or inside) whatever starts next.
    async fn cancel_retry_and_wait(&self) {
        self.abort_retry();
        let handle = lock(&self.retry_task).take();
        if let Some(h) = handle {
            let _ = h.await;
        }
    }

    /// Cancel in-flight compaction (TS `abortCompaction`): Esc on the
    /// "compacting… (esc to cancel)" loader, `abort`, and every prompt. The
    /// pass discards at its next checkpoint instead of writing the `compaction`
    /// entry or swapping the context.
    pub fn abort_compaction(&self) {
        if let Some(g) = lock(&self.compaction_guard).as_ref() {
            g.cancel();
        }
    }

    /// Cancel in-flight compaction and wait until it fully settled, so a prompt
    /// admitted behind it starts on the settled context instead of racing the
    /// compaction's context swap (TS `abortCompaction` + prompt-side
    /// `waitForManualMaintenanceCleanup`). Bug-hunt #2: without this, the
    /// concurrent turn was lost and the summary landed in the wrong file.
    async fn cancel_compaction_and_wait(&self) {
        self.abort_compaction();
        let handle = lock(&self.post_turn_task).take();
        if let Some(h) = handle {
            let _ = h.await;
        }
        // Manual compactions run inline in their rpc handler (no task handle):
        // park on the end notification like TS parks on the cleanup barrier.
        while self.compacting.load(std::sync::atomic::Ordering::SeqCst) {
            let done = self.compaction_done.notified();
            if !self.compacting.load(std::sync::atomic::Ordering::SeqCst) {
                break;
            }
            done.await;
        }
    }

    fn bump_generation(&self) {
        self.switch_generation
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }

    /// Abort the run and wait until the agent is idle.
    pub async fn abort(&self) {
        self.user_interrupt
            .store(true, std::sync::atomic::Ordering::SeqCst);
        self.abort_retry();
        self.abort_compaction();
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

    /// Remove the first queued user message whose text is exactly `message` from `queue`
    /// (`removeQueuedMessage`); `None` when there is no such message.
    fn take_queued_exact(&self, message: &str, queue: &str) -> Option<AgentMessage> {
        let (steering, follow) = self.queued();
        // The agent scans steering before follow-up and drops a whole batch per hit, so the
        // wanted message is the n-th exact match in that combined order.
        let target = if queue == "steering" {
            steering.iter().any(|p| p == message).then_some(0)?
        } else {
            follow.iter().any(|p| p == message).then_some(())?;
            steering.iter().filter(|p| *p == message).count()
        };
        let seen = std::cell::Cell::new(0usize);
        let removed = self.agent.remove_queued_messages(|m| {
            if is_user_text(m, message) {
                let hit = seen.get() == target;
                seen.set(seen.get() + 1);
                hit
            } else {
                false
            }
        });
        removed.into_iter().find(|m| is_user_text(m, message))
    }

    pub fn remove_queued(&self, message: &str, queue: &str) -> bool {
        let ok = self.take_queued_exact(message, queue).is_some();
        if ok {
            self.emit_queue_update();
        }
        ok
    }

    pub fn promote_queued(&self, message: &str) -> bool {
        let Some(first) = self.take_queued_exact(message, "followUp") else {
            return false;
        };
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

    /// The live session file; `None` under `--no-session`.
    pub fn live_session_file(&self) -> Option<PathBuf> {
        self.persist.file()
    }

    pub fn persist_file(&self) -> Option<PathBuf> {
        self.persist.file()
    }

    pub fn session_file(&self) -> PathBuf {
        self.persist.path()
    }

    pub fn model_info(&self) -> Value {
        let target = self
            .target
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        target.map_or(Value::Null, |t| model_info(&t.model))
    }

    pub async fn state(&self) -> Value {
        // Status without the transcript: this answers every `get_state`/slash busy check.
        let (s, message_count) = self.agent.status().await;
        let retrying = lock(&self.recovery).saga_open();
        let v = {
            let st = lock(&self.settings);
            let (steering, follow) = self.queued();
            let queued = steering.len() + follow.len();
            let mut v = json!({
                "isStreaming": s.is_streaming,
                "isCompacting": self.compacting.load(std::sync::atomic::Ordering::SeqCst),
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
                "messageCount": message_count,
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
            v
        };
        let mut out = Map::new();
        out.insert("model".into(), self.model_info());
        out.insert("thinkingLevel".into(), json!(level_name(s.thinking_level)));
        out.insert("ultra".into(), json!(self.ultra().map(Ultra::name)));
        if let Some(obj) = v.as_object() {
            out.extend(obj.clone());
        }
        let base = prompt_overhead_chars(&s.system_prompt, &s.tools);
        let window = s.model.context_window;
        if let Some(usage) = self
            .agent
            .with_messages(|m| context_usage(m.iter(), window, base))
            .await
        {
            out.insert("contextUsage".into(), usage);
        }
        Value::Object(out)
    }

    /// TS `ctx.getContextUsage()`: `{tokens, contextWindow, percent}` (None without a window).
    pub async fn context_usage_value(&self) -> Option<Value> {
        let (s, _) = self.agent.status().await;
        let base = prompt_overhead_chars(&s.system_prompt, &s.tools);
        let window = s.model.context_window;
        self.agent
            .with_messages(|m| context_usage(m.iter(), window, base))
            .await
    }

    /// Whether steering/follow-up messages are queued (TS `ctx.hasPendingMessages()`).
    pub fn has_pending_messages(&self) -> bool {
        let (steer, follow) = self.queued();
        !steer.is_empty() || !follow.is_empty()
    }

    /// The `context_usage` frame the TUI status line renders: active model
    /// (selector + display name), context tokens/percent/window and the
    /// auto-compaction gauge boundaries (TS `getContextUsage` +
    /// `getSessionCompactionBoundaries`).
    pub async fn context_usage_frame(&self) -> Value {
        let (s, _) = self.agent.status().await;
        let base = prompt_overhead_chars(&s.system_prompt, &s.tools);
        let window = s.model.context_window;
        let usage = self
            .agent
            .with_messages(|m| context_usage(m.iter(), window, base))
            .await;
        let mut frame = json!({
            "type": "context_usage",
            "model": format!("{}/{}", s.model.provider, s.model.id),
            "modelName": s.model.name,
            "contextWindow": window,
            "thinkingLevel": level_name(s.thinking_level),
            "thinkingLevels": self.supported_efforts(),
            "autoThinking": self.auto_thinking.load(std::sync::atomic::Ordering::Relaxed),
            "ultra": self.ultra().map(Ultra::name),
        });
        if let Some(Value::Object(u)) = usage {
            for (k, v) in u {
                frame[k.as_str()] = v;
            }
        }
        let compaction = crate::compact::load_settings();
        if window > 0 && compaction.enabled && compaction.strategy.as_deref() != Some("off") {
            let threshold = omp_compact::settings::resolve_threshold_tokens(window, &compaction);
            #[allow(clippy::cast_precision_loss)]
            if threshold > 0.0 && threshold <= window as f64 {
                let w = window as f64;
                frame["thresholdPercent"] = json!(threshold / w * 100.0);
                let async_enabled = omp_settings::Settings::load(&self.cwd)
                    .get_bool("compaction.asyncEnabled")
                    .unwrap_or(true);
                if async_enabled {
                    let lead = ((threshold * 0.125).floor()).clamp(8192.0, 32000.0);
                    frame["speculationPercent"] = json!((threshold - lead).max(0.0) / w * 100.0);
                }
            }
        }
        frame
    }

    /// Turn on `context_usage` frame emission (interactive TUI host).
    pub fn enable_context_usage_frames(&self) {
        self.ui_context_frames
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// Publish a fresh `context_usage` frame to the UI stream (off the
    /// caller's task: the agent's state lock must not be re-entered from an
    /// event callback).
    pub fn publish_context_usage(self: &Arc<Self>) {
        if !self
            .ui_context_frames
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            return;
        }
        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            return;
        };
        let rt = self.clone();
        handle.spawn(async move {
            let frame = rt.context_usage_frame().await;
            rt.send(frame);
        });
    }

    pub async fn messages(&self) -> Vec<Value> {
        // Shared snapshot: serialize outside the state lock, without a deep copy first.
        self.agent
            .messages_snapshot()
            .await
            .iter()
            .map(crate::events::message_value)
            .collect()
    }

    pub async fn last_assistant_text(&self) -> Option<String> {
        self.agent
            .with_messages(|messages| {
                messages.iter().rev().find_map(|m| match m {
                    AgentMessage::Standard(Message::Assistant(a)) => {
                        let t = message_text(m);
                        (a.stop_reason != StopReason::Aborted || !t.is_empty()).then_some(t)
                    }
                    _ => None,
                })
            })
            .await
    }

    pub async fn session_stats(&self) -> Value {
        let (s, total) = self.agent.status().await;
        let base = prompt_overhead_chars(&s.system_prompt, &s.tools);
        let window = s.model.context_window;
        // One in-place pass over the transcript under the state lock (no copy).
        let (user, assistant, calls, results, input, output, cache_read, cache_write, cost, usage) =
            self.agent
                .with_messages(|messages| {
                    let (mut user, mut assistant, mut calls, mut results) =
                        (0u64, 0u64, 0u64, 0u64);
                    let (mut input, mut output, mut cache_read, mut cache_write, mut cost) =
                        (0u64, 0u64, 0u64, 0u64, 0.0f64);
                    for m in messages.iter() {
                        match m {
                            AgentMessage::Standard(Message::User(_)) => user += 1,
                            AgentMessage::Standard(Message::Developer(_)) => {}
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
                    let usage = context_usage(messages.iter(), window, base);
                    (
                        user,
                        assistant,
                        calls,
                        results,
                        input,
                        output,
                        cache_read,
                        cache_write,
                        cost,
                        usage,
                    )
                })
                .await;
        let mut v = json!({
            "sessionId": self.persist.session_id(),
            "userMessages": user,
            "assistantMessages": assistant,
            "toolCalls": calls,
            "toolResults": results,
            "totalMessages": total,
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
        if let Some(u) = usage {
            v["contextUsage"] = u;
        }
        v
    }

    // ------------------------------------------------------------ session

    /// Rebuild the `todo` tool from session entries (`&[]` = fresh session),
    /// matching omp's resume rehydration.
    pub async fn rehydrate_todo_tool(&self, entries: &[Value]) {
        let session_file = self.persist.file().map(|p| p.display().to_string());
        self.swap_todo_tool(Arc::new(crate::tools::TodoTool::with_session(
            crate::tools::phases_from_entries(entries),
            session_file.as_deref(),
        )))
        .await;
    }

    /// Install `todo` as the agent's todo tool (no-op when the tool list has none).
    async fn swap_todo_tool(&self, todo: Arc<dyn pa_agent::types::AgentTool>) {
        let mut tools = self.agent.tools().await;
        if let Some(slot) = tools.iter().position(|t| t.name() == "todo") {
            tools[slot] = todo;
            self.agent.set_tools(tools).await;
        }
    }

    /// Clear the conversation and start a fresh session file.
    pub async fn new_session(&self, dir: Option<&Path>) -> Result<()> {
        self.check_switch("new", None).await?;
        let previous = self.persist.file();
        // An in-flight compaction must not write into the old file (or swap the
        // old context over the fresh one) after the switch (bug-hunt #2).
        self.cancel_compaction_and_wait().await;
        self.cancel_retry_and_wait().await;
        self.agent.abort();
        self.agent.wait_for_idle().await;
        self.agent.reset().await;
        self.persist.start_new(&self.cwd, dir)?;
        self.bump_generation();
        self.publish_session_file();
        self.herdr_session("new");
        self.rehydrate_todo_tool(&[]).await;
        {
            let mut st = lock(&self.settings);
            st.session_name = None;
            st.session_name_source = None;
            st.last_queue = (Vec::new(), Vec::new());
        }
        lock(&self.prompts).open.clear();
        self.record_model_state().await;
        // Interactive TUI: drop the rendered transcript and the terminal
        // scrollback (TS `#runNewSessionFlow`: `resetTranscript()` +
        // `requestRender(true, { clearScrollback: true })`). RPC clients never
        // see this frame.
        if self
            .ui_context_frames
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            self.send(json!({"type": "session_reset"}));
        }
        self.notify_session_switch("new", previous);
        Ok(())
    }

    /// Resume `path` in place (TS `SessionManager.open`): the agent context is rebuilt from the
    /// journal and new entries are appended to the SAME file, chained to the active branch tip.
    /// `dir` is unused (the file keeps its own directory).
    pub async fn load_session(&self, path: &Path, dir: Option<&Path>) -> Result<()> {
        self.load_session_as(path, dir, false, false).await
    }

    /// Bind a freshly written branch file (TS `#branchIntoNewSession`): no `session_before_switch`
    /// gate and no `session_switch` event — the caller emits `session_before_branch` /
    /// `session_branch` itself; the listener only refreshes the extension ctx (`"branch"`).
    pub async fn load_branched_session(&self, path: &Path) -> Result<()> {
        self.load_session_as(path, None, false, true).await
    }

    /// Fork `path` (TS `SessionManager.forkFrom`): the continuation is written to a fresh
    /// session file in `dir` seeded with the loaded history.
    pub async fn fork_session(&self, path: &Path, dir: Option<&Path>) -> Result<()> {
        self.load_session_as(path, dir, true, false).await
    }

    /// Shared body of [`Self::load_session`] / [`Self::fork_session`].
    ///
    /// Resume is lazy like omp's: the journal is scanned once into a link index
    /// ([`omp_session::index::SessionIndex`]) and only the active branch — TS
    /// `SessionEntryIndex.pathTo(leaf)`, root → leaf — is materialized, parsed
    /// on demand from the raw lines. Non-compacted sessions never build the
    /// entry DOM at all: messages deserialize straight from the line text.
    async fn load_session_as(
        &self,
        path: &Path,
        dir: Option<&Path>,
        fork: bool,
        branched: bool,
    ) -> Result<()> {
        if !branched {
            self.check_switch(
                if fork { "fork" } else { "resume" },
                Some(path.to_path_buf()),
            )
            .await?;
        }
        let previous = self.persist.file();
        // An in-flight compaction must not write into the old session or swap
        // its context over the resumed one after the switch (bug-hunt #2).
        self.cancel_compaction_and_wait().await;
        self.cancel_retry_and_wait().await;
        // `--no-session` must never write into the resumed file: seed a scratch copy instead.
        let fork = fork || self.persist.file().is_none();
        let trace_t0 = std::time::Instant::now();
        macro_rules! rss_trace {
            ($label:expr) => {
                if std::env::var_os("OMP_RS_RESUME_TRACE").is_some() {
                    if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
                        let rss = status
                            .lines()
                            .find_map(|l| l.strip_prefix("VmRSS:"))
                            .unwrap_or("?");
                        let hwm = status
                            .lines()
                            .find_map(|l| l.strip_prefix("VmHWM:"))
                            .unwrap_or("?");
                        eprintln!(
                            "[resume-trace] {} t={}ms rss={} hwm={}",
                            $label,
                            trace_t0.elapsed().as_millis(),
                            rss.trim(),
                            hwm.trim()
                        );
                    }
                }
            };
        }
        rss_trace!("start");
        let index = omp_session::index::SessionIndex::load(path)
            .with_context(|| format!("cannot read session {}", path.display()))?;
        rss_trace!("indexed");
        // A compacted session resumes with the rebuilt context (summary + kept tail),
        // not the raw log; checked over the whole journal like the old loader.
        let compacted = index.refs.iter().any(|r| {
            matches!(
                r.kind,
                omp_session::index::EntryKind::Compaction
                    | omp_session::index::EntryKind::ResetBoundary
                    | omp_session::index::EntryKind::BranchSummary
            )
        });
        let branch = index.branch();
        let brefs: Vec<&omp_session::index::EntryRef> =
            branch.iter().map(|&i| &index.refs[i as usize]).collect();
        let mut messages = Vec::new();
        // Original `message` JSON of custom-role messages (parallel to
        // `messages`); seeded verbatim when pa_types cannot model the shape.
        let mut raw_messages: Vec<Option<Map<String, Value>>> = Vec::new();
        // Reduced branch for the context rebuild: entries outside the emitted
        // ranges become link-only shims, so the DOM the rebuild works on stays
        // the size of the context, not of the journal.
        if compacted {
            let (full, notes_after, head_until) =
                reduced_branch_flags(&index, path, &branch, &brefs)?;
            let entries =
                materialize_reduced_branch(&index, path, &branch, &full, notes_after, head_until)?;
            rss_trace!("materialized");
            // Rebuild first, then release the entry DOM before converting:
            // the context messages and the typed messages never coexist with
            // the full reduced tree.
            // Owned rebuild: message payloads move out of the reduced tree into
            // the context, so the two never coexist (resume stays low-RSS).
            let ctx = omp_compact::build_session_context_owned(entries, omp_compact::Leaf::Last)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            rss_trace!("rebuilt");
            messages = crate::compact::context_messages(&ctx);
        } else {
            // No compaction on the journal: messages deserialize straight from
            // the raw lines, one pass, no entry DOM at all.
            index.for_each_line(path, &branch, |r, raw| {
                if r.kind == omp_session::index::EntryKind::Message {
                    if let Ok(envelope) = serde_json::from_str::<MessageEnvelope>(raw) {
                        let m = envelope.message;
                        // Custom roles pa_types cannot model (e.g. "developer")
                        // keep their original JSON shape so resume can re-persist
                        // them byte-identically instead of dropping them.
                        if matches!(m, AgentMessage::Custom(_)) {
                            let shape = serde_json::from_str::<Value>(raw)
                                .ok()
                                .and_then(|v| v.get("message").cloned())
                                .and_then(|v| v.as_object().cloned());
                            raw_messages.push(shape);
                        } else {
                            raw_messages.push(None);
                        }
                        messages.push(m);
                    }
                }
                Ok(())
            })?;
        }
        rss_trace!("messages");
        // Todo phases (TS `getLatestTodoPhasesFromEntries(getBranch())`).
        let phases = latest_todo_phases(&index, path, &branch)?;
        rss_trace!("loaded");
        self.agent.abort();
        self.agent.wait_for_idle().await;
        self.agent.reset().await;
        if fork {
            // TS fork: a NEW file seeded with the branch; the startup file never held anything.
            if self.persist.path() != path {
                self.persist.discard_pristine();
            }
            self.persist.start_new(&self.cwd, dir)?;
            self.publish_session_file();
            crate::tools::eval_journal::carry_over(path);
            self.record_model_state().await;
            if compacted {
                crate::compact::seed_branch(&self.persist, &index, path)?;
            } else {
                for (m, raw) in messages.iter().zip(&raw_messages) {
                    seed_resumed_message(&self.persist, m, raw.as_ref());
                }
            }
        } else {
            // TS `SessionManager.open`: keep appending to the same file (header untouched,
            // nothing re-seeded), chained to the active branch tip. A compacted journal is
            // only read to rebuild the context above, never rewritten.
            let header_id = index
                .refs
                .iter()
                .find(|r| r.kind == omp_session::index::EntryKind::Header)
                .map(|r| &*r.id);
            let tip = branch.last().map(|&i| &*index.refs[i as usize].id);
            self.persist
                .bind_existing(path, header_id, tip.filter(|t| !t.is_empty()))?;
            self.publish_session_file();
        }
        // New session identity: in-flight compaction guards captured before the
        // switch are stale from here on (bug-hunt #2).
        self.bump_generation();
        rss_trace!("seeded");
        rss_trace!("pre-set");
        if std::env::var_os("OMP_RS_RESUME_TRACE").is_some() {
            let json = serde_json::to_string(&messages).unwrap_or_default();
            eprintln!(
                "[resume-trace] messages: n={} json={}B sizeof={}B",
                messages.len(),
                json.len(),
                std::mem::size_of_val(&messages)
            );
        }
        rss_trace!("swapped");
        let session_file = self.persist.file().map(|p| p.display().to_string());
        self.swap_todo_tool(Arc::new(crate::tools::TodoTool::with_session(
            phases,
            session_file.as_deref(),
        )))
        .await;
        self.agent.set_messages(messages).await;
        rss_trace!("messages-set");
        // TS restores the session's model and thinking selector on resume and
        // on every session switch; explicit `--model` / `--thinking` win only
        // for the startup load.
        let saved = session_model_state(&index, path, &branch)?;
        if !MODEL_PINNED.swap(false, std::sync::atomic::Ordering::Relaxed) {
            self.restore_session_model(&saved).await;
        }
        if !THINKING_PINNED.swap(false, std::sync::atomic::Ordering::Relaxed) {
            self.restore_session_thinking(&saved).await;
        }
        self.herdr_session("resume");
        lock(&self.prompts).open.clear();
        self.restore_title_from(&index);
        self.notify_session_switch(
            if branched {
                "branch"
            } else if fork {
                "fork"
            } else {
                "resume"
            },
            previous,
        );
        Ok(())
    }

    /// TS `getRestorableSessionModels` + registry lookup: the last-used role's
    /// model, then the session default; the first one that resolves with
    /// credentials becomes live (no `model_change` is written, as in TS).
    async fn restore_session_model(&self, saved: &SessionModelState) {
        let mut candidates: Vec<&str> = Vec::new();
        let default = saved.models.get("default").map(String::as_str);
        match saved.last_role.as_deref() {
            None | Some("default" | EPHEMERAL_ROLE) => candidates.extend(default),
            Some(role) => match saved.models.get(role).map(String::as_str) {
                None => candidates.extend(default),
                Some(r) => {
                    candidates.push(r);
                    if let Some(d) = default.filter(|d| *d != r) {
                        candidates.push(d);
                    }
                }
            },
        }
        for reference in candidates {
            let Some((provider, id)) = reference.split_once('/') else {
                continue;
            };
            if self
                .live_provider_id()
                .is_some_and(|(p, i)| p == provider && i == id)
            {
                return;
            }
            let Ok(resolved) = (self.resolver)(reference.to_owned()).await else {
                continue;
            };
            if resolved.provider != provider || resolved.id != id || resolved.api_key.is_none() {
                continue;
            }
            lock(&self.recovery).reconfigure(RecoveryConfig::from_resolved(&resolved));
            self.swap_model(&resolved, None, false, false).await;
            return;
        }
    }

    /// TS `restoreThinkingLevel`: the branch's last `thinking_level_change`
    /// (its configured selector, so `auto` stays auto), clamped to the live
    /// model; not journaled.
    async fn restore_session_thinking(&self, saved: &SessionModelState) {
        let Some((level, configured)) = saved.thinking.as_ref() else {
            // No thinking record on this branch: an ultra mode does not leak in.
            self.leave_ultra().await;
            return;
        };
        if let Some(mode) = saved.ultra.as_deref().and_then(parse_ultra_name) {
            // The mode comes back with its effects (max thinking, CLM), not journaled.
            let prev = lock(&self.ultra).replace(mode);
            self.auto_thinking
                .store(false, std::sync::atomic::Ordering::Relaxed);
            let effective = self.live_model().map_or(ThinkingLevel::Max, |m| {
                crate::model::clamp_level(&m, ThinkingLevel::Max)
            });
            if self.agent.thinking_level().await != effective {
                self.agent.set_thinking_level(effective).await;
            }
            let name = level_name(effective);
            self.emit_event(json!({
                "type": "thinking_level_changed",
                "thinkingLevel": name,
                "configured": name,
                "ultra": mode.name(),
            }));
            self.sync_clm(prev, Some(mode)).await;
            return;
        }
        self.leave_ultra().await;
        let auto = configured.as_deref() == Some("auto");
        let Some(requested) = parse_level(level) else {
            return;
        };
        let model = self
            .target
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_ref()
            .map(|t| t.model.clone());
        let effective = model.map_or(requested, |m| crate::model::clamp_level(&m, requested));
        self.auto_thinking
            .store(auto, std::sync::atomic::Ordering::Relaxed);
        if self.agent.thinking_level().await != effective {
            self.agent.set_thinking_level(effective).await;
        }
        let name = level_name(effective);
        self.emit_event(json!({
            "type": "thinking_level_changed",
            "thinkingLevel": name,
            "configured": if auto { "auto" } else { name },
            "ultra": null,
        }));
    }

    async fn check_switch(&self, reason: &'static str, target: Option<PathBuf>) -> Result<()> {
        let gate = lock(&self.switch_gate).clone();
        if let Some(g) = gate {
            if g(reason, target).await {
                return Err(SwitchCancelled.into());
            }
        }
        Ok(())
    }

    /// The extension host, once attached: `session_before_branch` / `session_before_tree` and the
    /// other session events the runtime forwards itself.
    #[must_use]
    pub fn ext_host(&self) -> Option<Arc<omp_ext::ExtHost>> {
        lock(&self.ext_host).clone()
    }

    /// Attach the extension host (see [`Self::ext_host`]).
    pub fn set_ext_host(&self, host: Option<Arc<omp_ext::ExtHost>>) {
        let feed = host.clone().map(|host| {
            let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Value>();
            tokio::spawn(async move {
                while let Some(frame) = rx.recv().await {
                    let Some(ty) = frame["type"].as_str().map(str::to_owned) else {
                        continue;
                    };
                    if !host.has_handlers(&ty) {
                        continue;
                    }
                    if let Err(e) = host.emit(frame, None).await {
                        eprintln!("omt: extension {ty}: {e:#}");
                    }
                }
            });
            tx
        });
        *lock(&self.ext_frames) = feed;
        *lock(&self.ext_host) = host;
    }

    /// Install the `session_before_switch` gate.
    pub fn set_switch_gate(&self, g: Option<SwitchGate>) {
        *lock(&self.switch_gate) = g;
    }

    fn notify_session_switch(&self, reason: &'static str, previous: Option<PathBuf>) {
        if let Some(f) = lock(&self.on_session_switch).clone() {
            f(reason, previous);
        }
    }

    /// Install the extension compaction bridge (see [`CompactionHooks`]).
    pub fn set_compaction_hooks(&self, h: Option<Arc<dyn CompactionHooks>>) {
        *lock(&self.compaction_hooks) = h;
    }

    /// Install the extension `before_agent_start` hook (see [`BeforeStartHook`]).
    pub fn set_before_start_hook(&self, f: Option<BeforeStartHook>) {
        *lock(&self.before_start) = f;
    }

    /// Sink completed session switches (the ext bridge's `session_switch`).
    pub fn set_switch_listener(&self, f: Option<SwitchListener>) {
        *lock(&self.on_session_switch) = f;
    }

    /// A resumed session keeps its title: seed the name/source from the file's
    /// title slot (TS `SessionManager` restore) so auto-titling does not rename it.
    fn restore_title_from(&self, index: &omp_session::index::SessionIndex) {
        if let Some(slot) = &index.title_slot {
            if !slot.title.is_empty() {
                let mut st = lock(&self.settings);
                st.session_name = Some(slot.title.clone());
                st.session_name_source.clone_from(&slot.source);
            }
        }
    }
}

/// Re-persist one resumed context message into the fresh session file (TS
/// `SessionManager` keeps the same file; the port resumes into a new file, so
/// every message in the context is seeded). Messages pa_types cannot model
/// (custom roles like `"developer"`) are written with their original JSON
/// shape verbatim — the raw journal map when captured, otherwise the
/// re-serialized message — so nothing in the context is silently dropped.
fn seed_resumed_message(
    persist: &crate::persist::Persistence,
    m: &AgentMessage,
    raw: Option<&Map<String, Value>>,
) {
    let converted = serde_json::to_value(m)
        .and_then(|v| serde_json::from_value::<pa_types::session::AgentMessage>(v));
    if let Ok(c) = converted {
        let _ = persist.append_message(&c, m.timestamp());
        return;
    }
    let map = raw.cloned().or_else(|| {
        serde_json::to_value(m)
            .ok()
            .and_then(|v| v.as_object().cloned())
    });
    if let Some(map) = map {
        let ts = map
            .get("timestamp")
            .and_then(Value::as_i64)
            .unwrap_or_else(|| m.timestamp());
        let _ = persist.append_raw_message(map, ts);
    }
}

/// Compute the reduced-branch `full` flags for a compacted journal (TS
/// `buildReplanRebuildView`-style range selection): the rebuild reads the latest
/// compaction, the bookkeeping entries (state accumulation walks the whole
/// branch), and the emitted ranges (kept tail + post-compaction, or post-reset) —
/// but never the pre-compaction history again. Also returns the context-notes
/// start and the head-only message cut.
///
/// # Errors
/// The latest compaction entry cannot be read.
fn reduced_branch_flags(
    index: &omp_session::index::SessionIndex,
    path: &Path,
    branch: &[u32],
    brefs: &[&omp_session::index::EntryRef],
) -> Result<(Vec<bool>, usize, usize)> {
    let reset_pos = brefs
        .iter()
        .rposition(|r| r.kind == omp_session::index::EntryKind::ResetBoundary);
    let compaction_pos = brefs
        .iter()
        .rposition(|r| r.kind == omp_session::index::EntryKind::Compaction);
    let after_reset = match (reset_pos, compaction_pos) {
        (Some(r), Some(c)) => r > c,
        (Some(_), None) => true,
        _ => false,
    };
    let mut full = vec![false; branch.len()];
    if after_reset {
        for f in &mut full[reset_pos.unwrap() + 1..] {
            *f = true;
        }
    } else if let Some(cp) = compaction_pos {
        // The compaction entry alone decides the kept range.
        let cv = index
            .load_branch(path, &branch[cp..=cp], true)?
            .into_iter()
            .next()
            .and_then(|l| l.value)
            .ok_or_else(|| anyhow::anyhow!("compaction entry vanished"))?;
        let remote = omp_compact::openai_remote_payload(&cv);
        let anthropic = omp_compact::anthropic_compaction_payload(&cv);
        let has_replacement = remote
            .as_ref()
            .and_then(|p| p.get("items"))
            .is_some_and(Value::is_array);
        let id_pos = |id: &str| brefs.iter().position(|r| &*r.id == id);
        let first_kept = cv
            .get("firstKeptEntryId")
            .and_then(Value::as_str)
            .unwrap_or("");
        let kept_start = if has_replacement {
            None
        } else if anthropic.is_some() && first_kept.is_empty() {
            cv.get("providerReplayThroughEntryId")
                .and_then(Value::as_str)
                .and_then(|id| id_pos(id).map(|i| i + 1))
        } else {
            id_pos(first_kept)
        };
        if let Some(start) = kept_start.filter(|&s| s < cp) {
            for f in &mut full[start..cp] {
                *f = true;
            }
        }
        for f in &mut full[cp..] {
            *f = true;
        }
    } else {
        // The compaction marker lives on another branch: the rebuild replays the
        // whole branch through `context_message` — keep it verbatim rather than
        // shim what may be emitted.
        full.fill(true);
    }
    // Bookkeeping entries feed the state accumulation from anywhere on the
    // branch; `context_notes` scans the custom entries after the last reset
    // boundary.
    let notes_after = reset_pos.map_or(0, |r| r + 1);
    for (i, r) in brefs.iter().enumerate() {
        if is_bookkeeping(r.kind) {
            full[i] = true;
        }
    }
    // Message entries feed the default-model inference only until the first
    // model_change; earlier ones get a head-only shim.
    let head_until = brefs
        .iter()
        .position(|r| r.kind == omp_session::index::EntryKind::ModelChange)
        .unwrap_or(branch.len());
    Ok((full, notes_after, head_until))
}

/// Materialize the reduced branch for the context rebuild: `full` entries and
/// context-note custom entries parse for real, pre-head message entries become
/// head-only shims, everything else a link-only shim — so the DOM the rebuild
/// works on stays the size of the context, not of the journal.
fn materialize_reduced_branch(
    index: &omp_session::index::SessionIndex,
    path: &Path,
    branch: &[u32],
    full: &[bool],
    notes_after: usize,
    head_until: usize,
) -> Result<Vec<Value>> {
    let mut entries: Vec<Value> = Vec::with_capacity(branch.len());
    index.for_each_line(path, branch, |r, raw| {
        let pos = entries.len();
        let value = if full[pos] {
            serde_json::from_str::<Value>(raw)
                .with_context(|| format!("{}:{}: entry parse", path.display(), r.line_no))?
        } else if r.kind == omp_session::index::EntryKind::Message && pos < head_until {
            message_head_shim(r, raw)
        } else if r.kind == omp_session::index::EntryKind::Custom
            && pos >= notes_after
            && custom_type(raw).as_deref() == Some(omp_compact::context::CONTEXT_NOTES_ENTRY_TYPE)
        {
            serde_json::from_str::<Value>(raw)
                .with_context(|| format!("{}:{}: entry parse", path.display(), r.line_no))?
        } else {
            link_shim(r)
        };
        entries.push(value);
        Ok(())
    })?;
    Ok(entries)
}

/// Latest todo phases: scan the branch newest-first, parsing lines only until
/// the latest state-changing todo entry — typically the tail, so almost nothing.
fn latest_todo_phases(
    index: &omp_session::index::SessionIndex,
    path: &Path,
    branch: &[u32],
) -> Result<Vec<crate::tools::TodoPhase>> {
    let mut phases: Option<Vec<crate::tools::TodoPhase>> = None;
    // Only message/custom lines can carry todo state: other kinds are never read.
    let rev: Vec<u32> = branch
        .iter()
        .rev()
        .copied()
        .filter(|&i| {
            matches!(
                index.refs[i as usize].kind,
                omp_session::index::EntryKind::Message | omp_session::index::EntryKind::Custom
            )
        })
        .collect();
    index.scan_lines(path, &rev, |_r, raw| {
        // Cheap pre-filter: a state-changing todo entry mentions the todo tool
        // or the edit custom type; everything else is skipped without a DOM.
        if raw.contains("\"toolName\":\"todo\"")
            || raw.contains("\"customType\":\"user_todo_edit\"")
        {
            if let Ok(v) = serde_json::from_str::<Value>(raw) {
                let p = crate::tools::phases_from_entries(std::slice::from_ref(&v));
                if !p.is_empty() {
                    phases = Some(p);
                    // Newest state wins: nothing older can change the answer.
                    return Ok(std::ops::ControlFlow::Break(()));
                }
            }
        }
        Ok(std::ops::ControlFlow::Continue(()))
    })?;
    Ok(phases.unwrap_or_default())
}

impl Runtime {
    /// `SessionManager.setSessionName`: persist the `title_change` entry, patch the
    /// fixed-width `title` slot at line 1, update state, the terminal title, and the
    /// ext bridge. `source` is `"user"` (`/rename`, rpc `set_session_name`) or
    /// `"auto"` (title generator). A user-set name is never replaced by an auto
    /// title, and an auto title never replaces an existing one: both are no-ops.
    ///
    /// # Errors
    /// The cleaned name is empty, or a persistence write failed.
    pub fn set_session_name(&self, name: &str, source: &str) -> Result<()> {
        let title = crate::title::clean_title(name);
        anyhow::ensure!(!title.is_empty(), "Session name cannot be empty");
        let previous = {
            let mut st = lock(&self.settings);
            if source == "auto"
                && (st.session_name.is_some() || st.session_name_source.as_deref() == Some("user"))
            {
                return Ok(());
            }
            let previous = st.session_name.replace(title.clone());
            st.session_name_source = Some(source.to_string());
            previous
        };
        let ts = crate::persist::now_iso();
        self.persist
            .append_title_change(&title, source, previous.as_deref())?;
        let _ = self
            .persist
            .with_writer(|w| w.update_title_slot(&title, Some(source), &ts));
        crate::title::set_terminal_title(&crate::title::format_session_terminal_title(
            Some(&title),
            Some(&self.cwd),
        ));
        if let Some(f) = lock(&self.on_session_name).clone() {
            f(&title);
        }
        Ok(())
    }

    /// The current (restored or set) session name.
    pub fn session_name(&self) -> Option<String> {
        lock(&self.settings).session_name.clone()
    }

    /// Sink applied session-name changes (the ext bridge's `set_session_name`).
    pub fn set_name_listener(&self, f: Option<NameListener>) {
        *lock(&self.on_session_name) = f;
    }

    /// Start one background auto-title for the first non-low-signal user message
    /// of an unnamed session (TS `#startAutoTitle`); no-op when a name exists,
    /// titling is disabled, or the input is a greeting/filler.
    fn spawn_auto_title(self: &Arc<Self>, message: &str) {
        let trace = std::env::var_os("OMP_RS_TITLE_TRACE").is_some();
        if std::env::var_os("OMP_RS_NO_TITLE").is_some() {
            return;
        }
        // Print mode without a session file: the title has no reader (no
        // session to name, no UI to show it), so the side request is skipped.
        if NO_SESSION_PRINT.load(std::sync::atomic::Ordering::Relaxed) {
            return;
        }
        if lock(&self.settings).session_name.is_some() {
            return;
        }
        if crate::title::is_low_signal_title_input(message) {
            return;
        }
        if trace {
            eprintln!("[title] auto-title spawned");
        }
        let rt = self.clone();
        let message = message.to_string();
        let handle = tokio::spawn(async move {
            let Some(title) = rt.generate_auto_title(&message).await else {
                return;
            };
            let _ = rt.set_session_name(&title, "auto");
        });
        *lock(&self.title_task) = Some(handle);
    }

    /// Wait for a pending automatic title generation (print mode exits only
    /// after the turn pipeline drains; titling is part of that drain in TS).
    pub async fn wait_for_title(&self) {
        // Take the handle out first: the guard must not live across the await.
        let pending = lock(&self.title_task).take();
        if let Some(h) = pending {
            let _ = h.await;
        }
    }

    /// Title models: the configured `smol` role first (`config.yml` modelRoles),
    /// then the session's current model as fallback (TS `getTitleModels`).
    async fn generate_auto_title(&self, first_message: &str) -> Option<String> {
        let mut candidates: Vec<TitleCandidate> = Vec::new();
        if std::env::var_os("OMP_RS_BASE_URL").is_none() {
            match crate::model::select(Some("smol")).await {
                Ok(sel) => candidates.push((sel.model, sel.api_key, sel.headers)),
                Err(e) => {
                    if std::env::var_os("OMP_RS_TITLE_TRACE").is_some() {
                        eprintln!("[title] smol role resolve failed: {e:#}");
                    }
                }
            }
        }
        if std::env::var_os("OMP_RS_TITLE_TRACE").is_some() {
            eprintln!("[title] candidates: {}", candidates.len());
        }
        if let Ok(live) = crate::compact::LiveCompleter::from_target(&self.target) {
            let key = format!("{}/{}", live.model.provider, live.model.id);
            if candidates
                .iter()
                .all(|(m, _, _)| format!("{}/{}", m.provider, m.id) != key)
            {
                candidates.push((live.model, live.api_key, live.headers));
            }
        }
        let trace = std::env::var_os("OMP_RS_TITLE_TRACE").is_some();
        for (model, api_key, headers) in candidates {
            if trace {
                eprintln!("[title] candidate {}/{}", model.provider, model.id);
            }
            match crate::title::generate_title_with_model(first_message, &model, api_key, headers)
                .await
            {
                Some(title) => return Some(title),
                None if trace => eprintln!("[title] candidate produced no title"),
                None => {}
            }
        }
        if trace {
            eprintln!("[title] no candidate produced a title");
        }
        None
    }

    /// Raw JSON entries of the live session file (header excluded), append order.
    pub fn entries(&self) -> Result<Vec<Value>> {
        read_entries(&self.persist.path())
    }

    /// Root-to-leaf entries of the active branch (TS `sessionManager.getBranch()`): the writer's
    /// live append point after a `/tree` move, else the last journaled entry.
    pub fn branch_entries(&self) -> Vec<Value> {
        let Ok(entries) = self.entries() else {
            return Vec::new();
        };
        let leaf = match self.persist.leaf_divergence() {
            Some(Some(id)) => Some(id),
            Some(None) => None,
            None => omp_session::tree::leaf_id(&entries).map(str::to_owned),
        };
        leaf.map(|leaf| {
            omp_session::tree::branch_path(&entries, &leaf)
                .into_iter()
                .cloned()
                .collect()
        })
        .unwrap_or_default()
    }

    // ---------------------------------------------------------- compaction

    /// Replace the `compaction.*` policy (default: `config.yml`).
    #[allow(dead_code)]
    pub fn set_compaction_settings(&self, settings: omp_compact::CompactionSettings) {
        lock(&self.settings).auto_compaction = settings.enabled;
        *lock(&self.compaction) = settings;
    }

    /// Route compaction summaries through `f` instead of the live model.
    #[allow(dead_code)]
    pub fn set_summarizer(&self, f: Option<crate::compact::SummarizeFn>) {
        *lock(&self.summarizer) = f;
    }

    fn summarizer(&self) -> Result<crate::compact::Summarizer> {
        if let Some(f) = lock(&self.summarizer).clone() {
            return Ok(crate::compact::Summarizer::Injected(f));
        }
        Ok(crate::compact::Summarizer::Live(Arc::new(
            crate::compact::LiveCompleter::from_target(&self.target)?,
        )))
    }

    fn live_window(&self) -> u64 {
        self.target
            .read()
            .ok()
            .and_then(|t| t.as_ref().map(|t| t.model.context_window))
            .unwrap_or(0)
    }

    /// rpc `compact`: summarize the older history into a `compaction` entry and swap the agent's
    /// messages for the rebuilt context. Returns the `CompactionResult` the response carries.
    ///
    /// A live turn is aborted first and, once the summary commits, resumed (TS
    /// `SessionMaintenance.compact` → `#scheduleInterruptedTurnResume`): queued
    /// messages drive the resume, otherwise the hidden auto-continue prompt does.
    /// Without it an extension's mid-run `ctx.compact()` (compact-gate step-gate)
    /// ended the run as "Request was aborted" and left the agent idle.
    pub async fn compact(
        self: &Arc<Self>,
        custom_instructions: Option<&str>,
    ) -> Result<Value, CmdError> {
        use std::sync::atomic::Ordering::SeqCst;
        if self.compacting.swap(true, SeqCst) {
            return Err(CmdError::new("Compaction already in progress"));
        }
        let interrupted = self.agent.is_streaming().await;
        if interrupted {
            self.compaction_abort.store(true, SeqCst);
            self.persist.drop_empty_aborted(true);
        }
        self.agent.abort();
        self.agent.wait_for_idle().await;
        if interrupted {
            self.persist.drop_empty_aborted(false);
            self.compaction_abort.store(false, SeqCst);
            self.compaction_stub_start.store(false, SeqCst);
            // The stub never reached the journal; keep the live context in step.
            self.agent
                .mutate_messages(|m| {
                    if m.last().is_some_and(crate::persist::is_empty_aborted) {
                        m.pop();
                    }
                })
                .await;
        }
        let out = self.compact_inner(custom_instructions).await;
        self.compacting.store(false, SeqCst);
        // Release prompts parked in `cancel_compaction_and_wait`.
        self.compaction_done.notify_waiters();
        // No-op rejections (too small / already compacted) change no history, so
        // the cut turn resumes on them too; a hook cancel or a summarizer failure
        // does not (TS `rejectedAsNoOp`).
        let committed_or_noop = match &out {
            Ok(_) => true,
            Err(e) => {
                e.message.starts_with("Nothing to compact") || e.message == "Already compacted"
            }
        };
        if interrupted
            && committed_or_noop
            && !self.user_interrupt.load(SeqCst)
            && omp_settings::Settings::load(&self.cwd)
                .get("compaction.autoContinue")
                .as_bool()
                != Some(false)
        {
            self.resume_after_compaction();
        }
        out
    }

    /// TS `#scheduleCompactionContinuation` for a turn a manual compaction cut
    /// mid-run: a queued steer/follow-up continues the run, otherwise the
    /// hidden `auto-continue` developer prompt does.
    fn resume_after_compaction(self: &Arc<Self>) {
        let rt = self.clone();
        tokio::spawn(async move {
            rt.agent.wait_for_idle().await;
            if rt.agent.is_streaming().await {
                return; // a prompt parked on the compaction already took the session
            }
            if rt.agent.has_queued_messages() {
                let _ = rt.agent.continue_run_until_accepted().await;
                return;
            }
            let timestamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(0));
            let nudge = AgentMessage::Custom(pa_agent::types::CustomAgentMessage {
                role: "developer".to_string(),
                payload: json!({
                    "customType": "auto-continue",
                    "content": [{"type": "text", "text": AUTO_CONTINUE_PROMPT}],
                    "display": false,
                    "attribution": "agent",
                    "synthetic": true,
                    "timestamp": timestamp,
                }),
            });
            rt.agent.append_messages(vec![nudge.clone()]).await;
            seed_resumed_message(&rt.persist, &nudge, None);
            if let Err(e) = rt.continue_after_retry().await {
                eprintln!("omt: post-compaction resume failed: {e:#}");
            }
        });
    }

    /// One compaction pass with its session+generation guard: captured at
    /// start, the commit proceeds only while the session, the switch
    /// generation, the navigation state and the journal tip are unchanged;
    /// Esc / a queued prompt cancel it outright (bug-hunt #2).
    async fn compact_inner(&self, custom_instructions: Option<&str>) -> Result<Value, CmdError> {
        let guard = Arc::new(crate::compact::CommitGuard::capture(
            &self.persist,
            Arc::clone(&self.switch_generation),
        )?);
        *lock(&self.compaction_guard) = Some(Arc::clone(&guard));
        let out = self
            .compact_inner_guarded(custom_instructions, &guard)
            .await;
        *lock(&self.compaction_guard) = None;
        out
    }

    async fn compact_inner_guarded(
        &self,
        custom_instructions: Option<&str>,
        guard: &crate::compact::CommitGuard,
    ) -> Result<Value, CmdError> {
        let trace = std::env::var_os("OMP_RS_TURN_TRACE").is_some();
        turn_trace!("[turn-trace] compact-begin rss={}kB", rss_kb());
        let summarizer = self.summarizer()?;
        // Only the prompt overhead and the model id are needed from the state:
        // release the snapshot (a full copy of the live context) before the
        // summarization, which is the compaction's memory peak.
        let (non_message_tokens, model_id) = {
            let (st, _) = self.agent.status().await;
            (
                u64::try_from(prompt_overhead_chars(&st.system_prompt, &st.tools).div_ceil(4))
                    .unwrap_or(0),
                st.model.id,
            )
        };
        let settings = lock(&self.compaction).clone();
        let hooks = lock(&self.compaction_hooks).clone();
        // The branch extensions see matches what the pass will summarize: the
        // writer's live append point, not the file-order tail (bug-hunt #4).
        let nav_leaf = self.persist.leaf_divergence();
        let bridge = hooks.as_deref().map(|h| HookWithBranch {
            hooks: h,
            branch: self
                .entries()
                .ok()
                .map(|entries| {
                    let leaf = match nav_leaf.as_ref() {
                        Some(Some(id)) => Some(id.clone()),
                        Some(None) => None,
                        None => omp_session::tree::leaf_id(&entries).map(str::to_owned),
                    };
                    leaf.map(|leaf| {
                        omp_session::tree::branch_path(&entries, &leaf)
                            .into_iter()
                            .cloned()
                            .collect()
                    })
                    .unwrap_or_default()
                })
                .unwrap_or_default(),
        });
        let params = crate::compact::CompactParams {
            settings: &settings,
            model_id: &model_id,
            context_window: Some(self.live_window()).filter(|w| *w > 0),
            custom_instructions,
            non_message_tokens,
            hook: bridge
                .as_ref()
                .map(|b| b as &dyn crate::compact::CompactHook),
            guard: Some(guard),
        };
        if trace {
            turn_trace!("[turn-trace] compact-planned rss={}kB", rss_kb());
        }
        // The guard's token cancels the pass outright (Esc / a queued prompt):
        // dropping `compact_now` aborts the summarizer before anything is
        // written; a session that moved under the plan is caught by the
        // guard's own commit check inside `compact_now`.
        let session_path = self.persist.path();
        let (_result, data, view) = tokio::select! {
            out = crate::compact::compact_now(
                &self.persist,
                &session_path,
                &summarizer,
                &params,
            ) => out?,
            _ = guard.cancelled() => {
                return Err(CmdError::new("Compaction cancelled"));
            }
        };
        if trace {
            turn_trace!("[turn-trace] compact-summarized rss={}kB", rss_kb());
        }
        let rebuilt = crate::compact::rebuild_context(&view)?;
        if trace {
            turn_trace!("[turn-trace] compact-rebuilt rss={}kB", rss_kb());
        }
        // The commit landed (in the file it was planned for). Apply the rebuilt
        // context only while the session that started the pass is still live:
        // after `/new` or `/resume` the agent was reset, and a late
        // `set_messages` would clobber the fresh context (bug-hunt #2).
        if !guard.session_alive() {
            return Ok(data);
        }
        self.agent.set_messages(rebuilt).await;
        if trace {
            turn_trace!("[turn-trace] compact-set rss={}kB", rss_kb());
        }
        self.agent
            .set_compaction_epoch(self.agent.compaction_epoch() + 1);
        if let Some(h) = hooks.as_deref() {
            // TS `session_compact`: the committed `compaction` entry (last journal entry).
            if let Some(entry) = self.entries().ok().and_then(|e| {
                e.into_iter()
                    .rev()
                    .find(|e| e.get("type").and_then(Value::as_str) == Some("compaction"))
            }) {
                h.after(entry).await;
            }
        }
        Ok(data)
    }

    /// Post-turn threshold check (`checkCompaction`): auto-compaction is on, the turn succeeded and
    /// the provider-billed / stored context crossed `resolveThresholdTokens`. The provider side
    /// comes from the newest settled assistant message, the stored floor from a token estimate
    /// over the live typed context — no journal parse.
    async fn auto_compaction_due(&self) -> bool {
        let mut settings = lock(&self.compaction).clone();
        settings.enabled = true;
        let window = self.live_window();
        if !settings.enabled || window == 0 {
            return false;
        }
        // Read the live context in place (no snapshot copy of the messages).
        let mut usage = None;
        let mut stored = 0;
        self.agent
            .mutate_messages(|messages| {
                usage = messages.iter().rev().find_map(|m| match m {
                    AgentMessage::Standard(Message::Assistant(a))
                        if !matches!(a.stop_reason, StopReason::Error | StopReason::Aborted) =>
                    {
                        serde_json::to_value(&a.usage).ok()
                    }
                    _ => None,
                });
                stored = crate::compact::estimate_context_tokens(messages);
            })
            .await;
        crate::compact::auto_check(usage, stored, window, &settings)
    }

    /// The threshold compaction pass with its `auto_compaction_start` / `auto_compaction_end` events.
    async fn run_auto_compaction(&self) {
        const ACTION: &str = "context-full";
        // One pass at a time (the `on_agent_end` pre-check races two near-
        // simultaneous turn ends); the loser is a silent no-op — its guard
        // would discard it at the commit check anyway.
        if self
            .compacting
            .swap(true, std::sync::atomic::Ordering::SeqCst)
        {
            return;
        }
        self.emit_event(
            json!({"type": "auto_compaction_start", "reason": "threshold", "action": ACTION}),
        );
        let end = match self.compact_inner(None).await {
            Ok(result) => {
                json!({"type": "auto_compaction_end", "action": ACTION, "result": result, "aborted": false, "willRetry": false})
            }
            // An extension `session_before_compact` cancel is an abort, not a failure.
            Err(e) if e.message == "Compaction cancelled" => json!({
                "type": "auto_compaction_end", "action": ACTION, "aborted": true, "willRetry": false,
            }),
            Err(e) => json!({
                "type": "auto_compaction_end", "action": ACTION, "aborted": false, "willRetry": false,
                "errorMessage": format!("Auto-compaction failed: {}", e.message),
            }),
        };
        self.compacting
            .store(false, std::sync::atomic::Ordering::SeqCst);
        // Release prompts parked in `cancel_compaction_and_wait`.
        self.compaction_done.notify_waiters();
        self.emit_event(end);
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

    async fn select_model(
        &self,
        provider: &str,
        model_id: &str,
        role: Option<&str>,
    ) -> Result<Value, CmdError> {
        let reference = format!("{provider}/{model_id}");
        let resolved = (self.resolver)(reference.clone())
            .await
            .map_err(|_| CmdError::new(format!("Model not found: {provider}/{model_id}")))?;
        if resolved.provider != provider || resolved.id != model_id {
            return Err(CmdError::new(format!(
                "Model not found: {provider}/{model_id}"
            )));
        }
        if resolved.api_key.is_none() {
            return Err(CmdError::new(format!(
                "No API key for {provider}/{model_id}"
            )));
        }
        let info = model_info(&resolved.model);
        // A user selection ends fallback ownership and clears the selector's cooldown.
        {
            let mut rec = lock(&self.recovery);
            rec.reconfigure(RecoveryConfig::from_resolved(&resolved));
            rec.clear_cooldown(&reference);
        }
        self.swap_model(&resolved, role, false, true).await;
        // Re-apply thinking for the new model: the catalog's `thinking.defaultLevel`, else the current level.
        let catalog_default = crate::model::catalog_extras(provider, model_id)
            .into_iter()
            .flatten()
            .find(|(k, _)| *k == "thinking")
            .and_then(|(_, v)| {
                v.get("defaultLevel")
                    .and_then(Value::as_str)
                    .and_then(parse_level)
            });
        // An ultra mode keeps thinking at the highest level the new model supports.
        let preferred = match catalog_default {
            _ if self.ultra().is_some() => ThinkingLevel::Max,
            Some(l) => l,
            None => self.agent.thinking_level().await,
        };
        self.apply_thinking(preferred).await;
        Ok(info)
    }

    /// Swap the live model: provider target, agent model, edit-mode tools, `model_change` entry,
    /// `model_changed` event (only when the model actually changed). Thinking is left to the caller.
    async fn swap_model(
        &self,
        resolved: &omp_providers::ResolvedModel,
        role: Option<&str>,
        is_fallback: bool,
        journal: bool,
    ) -> bool {
        let agent_model: Option<pa_agent::types::Model> = json_round_trip(&resolved.model);
        let Some(agent_model) = agent_model else {
            return false;
        };
        let changing = self
            .live_provider_id()
            .is_none_or(|(p, i)| p != resolved.provider || i != resolved.id);
        *self
            .target
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(ProviderTarget {
            api_key: resolved.api_key.clone(),
            model: resolved.model.clone(),
            service_tier: None,
            headers: (!resolved.headers.is_empty())
                .then(|| resolved.headers.iter().cloned().collect()),
            credential_id: resolved.credential_id,
        });
        self.agent.set_model(agent_model).await;
        self.agent.set_credential_id(resolved.credential_id);
        let mode = edit_mode_for(&resolved.id);
        let rebuild = {
            let mut st = lock(&self.settings);
            let changed = st.edit_mode != mode;
            st.edit_mode = mode;
            changed
        };
        if rebuild {
            {
                let settings = omp_settings::Settings::load(&self.cwd);
                let mut tools = crate::tools::session_builtin_tools(&self.cwd, mode, &settings);
                tools.extend(crate::tools::mcp::mcp_tools(&self.cwd));
                // TS `reconcileThinkTool`: model swaps re-evaluate the hidden
                // think tool's eligibility.
                let hidden =
                    crate::tools::hidden_tools(&settings, Some(&resolved.model), false, false);
                tools.extend(hidden);
                // The live context strategy keeps its `context_edit` tool.
                if let Some(shared) = crate::context::live() {
                    let tool = crate::context::tool(shared);
                    tools.extend(match crate::approval::Gate::get() {
                        Some(gate) => gate.wrap_tools(vec![tool]),
                        None => vec![tool],
                    });
                }
                self.agent.set_tools(tools).await;
            }
        }
        if changing {
            self.emit_event(json!({"type": "model_changed"}));
            if self
                .ui_context_frames
                .load(std::sync::atomic::Ordering::Relaxed)
            {
                let usage = self.context_usage_frame().await;
                self.send(usage);
            }
        }
        if journal {
            let selector = format!("{}/{}", resolved.provider, resolved.id);
            let _ = self
                .persist
                .append_model_change(&selector, role, is_fallback);
        }
        true
    }

    /// `ModelControls.setThinkingLevel`: clamp to the live model; journal and announce only a change.
    /// Returns the effective level.
    async fn apply_thinking(&self, requested: ThinkingLevel) -> ThinkingLevel {
        let model = self
            .target
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_ref()
            .map(|t| t.model.clone());
        let effective = model.map_or(requested, |m| crate::model::clamp_level(&m, requested));
        let current = self.agent.thinking_level().await;
        if current != effective {
            self.agent.set_thinking_level(effective).await;
            let name = level_name(effective);
            let ultra = self.ultra().map(Ultra::name);
            let _ = self
                .persist
                .append_thinking_level_change(Some(name), Some(name), ultra);
            self.emit_event(
                json!({"type": "thinking_level_changed", "thinkingLevel": name, "ultra": ultra}),
            );
        }
        effective
    }

    pub async fn set_thinking_level(&self, level: ThinkingLevel) {
        // Pinning a concrete level leaves `auto` (TS `setThinkingLevel`) and any ultra mode.
        self.auto_thinking
            .store(false, std::sync::atomic::Ordering::Relaxed);
        let left_ultra = self.leave_ultra().await;
        let before = self.agent.thinking_level().await;
        let effective = self.apply_thinking(level).await;
        if left_ultra && effective == before {
            // Same level, but the mode is gone: journal and announce that.
            let name = level_name(effective);
            let _ = self
                .persist
                .append_thinking_level_change(Some(name), Some(name), None);
            self.emit_event(
                json!({"type": "thinking_level_changed", "thinkingLevel": name, "ultra": null}),
            );
        }
    }

    /// The active ultra mode.
    #[must_use]
    pub fn ultra(&self) -> Option<Ultra> {
        *lock(&self.ultra)
    }

    /// Enter (`Some`) or leave (`None`, `/ultra off`) an ultra mode.
    ///
    /// Entering pins thinking to the highest level the live model supports
    /// (leaving `auto`); leaving keeps the current level. Either way one
    /// `thinking_level_change` is journaled (`ultra` only while a mode is on)
    /// and `thinking_level_changed` announced; `ultralong`/`hypercode` switch
    /// the `clm-host` context strategy on, other modes / off switch it back.
    pub async fn set_ultra(&self, mode: Option<Ultra>) {
        let prev = std::mem::replace(&mut *lock(&self.ultra), mode);
        if prev.is_none() && mode.is_none() {
            return;
        }
        let name = match mode {
            Some(_) => {
                self.auto_thinking
                    .store(false, std::sync::atomic::Ordering::Relaxed);
                let effective = self.live_model().map_or(ThinkingLevel::Max, |m| {
                    crate::model::clamp_level(&m, ThinkingLevel::Max)
                });
                if self.agent.thinking_level().await != effective {
                    self.agent.set_thinking_level(effective).await;
                }
                level_name(effective)
            }
            None => level_name(self.agent.thinking_level().await),
        };
        let ultra = mode.map(Ultra::name);
        let _ = self
            .persist
            .append_thinking_level_change(Some(name), Some(name), ultra);
        self.emit_event(json!({
            "type": "thinking_level_changed",
            "thinkingLevel": name,
            "configured": name,
            "ultra": ultra,
        }));
        self.sync_clm(prev, mode).await;
    }

    /// Drop the ultra mode without journaling (the caller records the new
    /// level); `true` when one was active.
    async fn leave_ultra(&self) -> bool {
        let prev = lock(&self.ultra).take();
        self.sync_clm(prev, None).await;
        prev.is_some()
    }

    fn live_model(&self) -> Option<pa_types::ai::Model> {
        self.target
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_ref()
            .map(|t| t.model.clone())
    }

    /// Switch the `clm-host` strategy (and its `context_edit` tool) on or off
    /// when the ultra mode's CLM leg changes. A session started with
    /// `clm`/`clm-host` keeps its strategy: ultra never touches it.
    async fn sync_clm(&self, prev: Option<Ultra>, next: Option<Ultra>) {
        let (had, want) = (prev.is_some_and(Ultra::clm), next.is_some_and(Ultra::clm));
        if had == want || crate::context::active() != crate::context::Kind::Append {
            return;
        }
        let mut tools = self.agent.tools().await;
        tools.retain(|t| t.name() != crate::context::TOOL_NAME);
        if want {
            let settings = omp_settings::Settings::load(&self.cwd);
            let shared = crate::context::ultra_shared(&settings);
            let tool = crate::context::tool(shared.clone());
            tools.extend(match crate::approval::Gate::get() {
                Some(gate) => gate.wrap_tools(vec![tool]),
                None => vec![tool],
            });
            crate::context::set_live(Some(shared), true);
        } else {
            crate::context::set_live(None, false);
        }
        self.agent.set_tools(tools).await;
        let message = if want {
            format!(
                "Context strategy changed to clm-host ({}); the prompt cache resets once.",
                next.map_or("ultra", Ultra::name)
            )
        } else {
            "Context strategy changed back to append; the prompt cache resets once.".to_owned()
        };
        self.emit_notice("warning", &message, None);
    }

    /// `setThinkingLevel("auto")`: run a provisional concrete level and flag
    /// the session as auto (the status line shows `auto` until classified).
    pub async fn set_auto_thinking(&self) {
        self.leave_ultra().await;
        let provisional = self
            .target
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_ref()
            .map_or(ThinkingLevel::Medium, |t| {
                crate::model::clamp_level(&t.model, ThinkingLevel::Medium)
            });
        self.apply_thinking(provisional).await;
        let name = level_name(provisional);
        let _ = self
            .persist
            .append_thinking_level_change(Some(name), Some("auto"), None);
        self.auto_thinking
            .store(true, std::sync::atomic::Ordering::Relaxed);
        self.emit_event(json!({"type": "thinking_level_changed", "thinkingLevel": name, "configured": "auto", "ultra": null}));
    }

    /// Efforts the live model supports, least to most intensive, without
    /// `off` (TS `getSupportedEfforts`: catalog `thinking.efforts`; models
    /// without catalog metadata offer the standard ladder). Empty when the
    /// model cannot think.
    #[must_use]
    pub fn supported_efforts(&self) -> Vec<&'static str> {
        let target = self
            .target
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        let Some(t) = target.filter(|t| t.model.reasoning) else {
            return Vec::new();
        };
        match crate::model::catalog_efforts(&t.model.provider, &t.model.id) {
            Some(efforts) => efforts
                .into_iter()
                .map(level_name)
                .filter(|l| *l != "off")
                .collect(),
            // No catalog entry: offer the ladder levels the runtime would not
            // clamp away, so every selector value actually takes effect.
            None => ["minimal", "low", "medium", "high", "xhigh", "max"]
                .into_iter()
                .filter(|name| {
                    parse_level(name).is_some_and(|l| crate::model::clamp_level(&t.model, l) == l)
                })
                .collect(),
        }
    }

    /// Current service-tier override on the live provider target
    /// (TS `session.serviceTier`, driven by `/fast` and `/slow`).
    #[must_use]
    pub fn service_tier(&self) -> Option<pa_types::ai::ServiceTier> {
        self.target
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_ref()
            .and_then(|t| t.service_tier)
    }

    /// Set the service-tier override for subsequent requests (TS
    /// `session.setFastMode`/`setSlowMode`: `None` restores the default tier).
    pub fn set_service_tier(&self, tier: Option<pa_types::ai::ServiceTier>) {
        if let Some(t) = self
            .target
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_mut()
        {
            t.service_tier = tier;
        }
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
        self.announced
            .store(true, std::sync::atomic::Ordering::SeqCst);
        let commands = self.available_commands();
        self.send(json!({"type": "available_commands_update", "commands": commands}));
    }

    /// omp's builtin slash commands followed by the extension/custom/file commands supplied
    /// through [`Self::set_available_commands`] (`getAvailableCommands` order).
    #[must_use]
    pub fn available_commands(&self) -> Vec<Value> {
        let mut all = builtin_commands();
        all.extend(crate::slash_extra::extra_catalog());
        all.extend(lock(&self.commands).iter().cloned());
        all
    }

    /// Selectable levels for the live model, `off` first.
    pub fn available_thinking_levels(&self) -> Vec<&'static str> {
        let target = self
            .target
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
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

// ---------------------------------------------------------------------------
// Todo completion reminder helpers (TS `todo-tracker.ts`).
// ---------------------------------------------------------------------------

/// `todo.remindersMax` as a `u32` (TS default 3); a negative or non-finite
/// configured value falls back to the default.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // range-checked: `0.0 <= n < u32::MAX` before the cast.
fn reminders_max(cfg: &omp_settings::Settings) -> u32 {
    match cfg.get_number("todo.remindersMax") {
        Some(n) if n.is_finite() && n >= 0.0 => {
            let rounded = n.round();
            if rounded >= f64::from(u32::MAX) {
                u32::MAX
            } else {
                rounded as u32
            }
        }
        _ => 3,
    }
}

/// TS `prompts/system/auto-continue.md`: the hidden resume prompt after a
/// compaction cut the live turn.
const AUTO_CONTINUE_PROMPT: &str = include_str!("../prompts/system/auto-continue.md");

static MARKDOWN_PROMPT_PREFIX_RE: std::sync::LazyLock<regex::Regex> =
    std::sync::LazyLock::new(|| {
        regex::Regex::new(r"^(?:>\s*)?(?:(?:[-*+]|\d+[.)])\s+)*").expect("valid regex")
    });
static PROMPT_LABEL_RE: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
    regex::Regex::new(r"^(?i)(?:q(?:uestion)?|ask)\s*\d*\s*[:.)-]\s*").expect("valid regex")
});
static QUESTION_PROMPT_RE: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
    regex::Regex::new(
        r"(?i)^(what|which|when|where|why|how|who|whom|whose|do|does|did|can|could|would|will|should|is|are|am|may|shall)\b",
    )
    .expect("valid regex")
});
static USER_DIRECTED_PROMPT_RE: std::sync::LazyLock<regex::Regex> =
    std::sync::LazyLock::new(|| {
        regex::Regex::new(r"(?i)\b(?:you|your|we|our)\b").expect("valid regex")
    });
static NON_ASCII_TEXT_RE: std::sync::LazyLock<regex::Regex> =
    std::sync::LazyLock::new(|| regex::Regex::new(r"[^\x00-\x7F]").expect("valid regex"));
static USER_RESPONSE_CUE_RE: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
    regex::Regex::new(
        r"(?i)^(?:please\s+)?(?:confirm|reply|choose|pick|decide|advise)\b|^(?:please\s+)?answer\b|^(?:please\s+)?(?:let\s+me\s+know|tell\s+me)\b",
    )
    .expect("valid regex")
});
static TRAILING_PUNCT_RE: std::sync::LazyLock<regex::Regex> =
    std::sync::LazyLock::new(|| regex::Regex::new(r"[.!?。！？]+$").expect("valid regex"));

/// `promptLine` in TS: strip a markdown list/quote prefix, then a `q1:`-style
/// prompt label. Returns `(text, had_prompt_label)`.
fn prompt_line(line: &str) -> (String, bool) {
    let without_markdown_prefix = MARKDOWN_PROMPT_PREFIX_RE
        .replace(line.trim(), "")
        .trim()
        .to_string();
    let without_prompt_label = PROMPT_LABEL_RE
        .replace(&without_markdown_prefix, "")
        .trim()
        .to_string();
    let had_prompt_label = without_prompt_label != without_markdown_prefix;
    (without_prompt_label, had_prompt_label)
}

/// TS `isQuestionPromptLine`.
fn is_question_prompt_line(line: &str) -> bool {
    let (text, had_prompt_label) = prompt_line(line);
    if !text.ends_with('?') && !text.ends_with('\u{FF1F}') {
        return false;
    }
    had_prompt_label
        || QUESTION_PROMPT_RE.is_match(&text)
        || USER_DIRECTED_PROMPT_RE.is_match(&text)
        || NON_ASCII_TEXT_RE.is_match(&text)
}

/// TS `isResponseCueLine`.
fn is_response_cue_line(line: &str) -> bool {
    let (text, _) = prompt_line(line);
    let candidate = TRAILING_PUNCT_RE
        .replace(text.trim(), "")
        .trim()
        .to_string();
    USER_RESPONSE_CUE_RE.is_match(&candidate)
}

/// The `- Phase\n  - task` block inside the reminder text (TS `checkCompletion`'s
/// `todoList`): one heading per phase that still has open tasks.
fn todo_reminder_list(phases: &Value) -> String {
    phases
        .as_array()
        .map(|ps| {
            ps.iter()
                .filter_map(|phase| {
                    let name = phase.get("name").and_then(Value::as_str)?;
                    let open: Vec<String> = phase
                        .get("tasks")
                        .and_then(Value::as_array)
                        .map(|tasks| {
                            tasks
                                .iter()
                                .filter(|t| {
                                    t.get("status")
                                        .and_then(Value::as_str)
                                        .is_some_and(|s| s == "pending" || s == "in_progress")
                                })
                                .filter_map(|t| {
                                    t.get("content").and_then(Value::as_str).map(str::to_owned)
                                })
                                .collect()
                        })
                        .unwrap_or_default();
                    (!open.is_empty()).then(|| {
                        let tasks = open
                            .iter()
                            .map(|c| format!("  - {c}"))
                            .collect::<Vec<_>>()
                            .join("\n");
                        format!("- {name}\n{tasks}")
                    })
                })
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_default()
}

/// TS `isAwaitingUserAnswer`: the assistant's last text line is a direct
/// question or an explicit request for user input, so a completion reminder
/// would interrupt a genuine wait.
fn is_awaiting_user_answer(text: &str) -> bool {
    let Some(last_line) = text.lines().last().map(str::trim) else {
        return false;
    };
    if last_line.is_empty() {
        return false;
    }
    is_question_prompt_line(last_line) || is_response_cue_line(last_line)
}

/// Open tasks (`status` `pending`/`in_progress`) across all todo phases, as the
/// flat `[{content, status}]` list the `todo_reminder` frame carries (TS
/// `checkCompletion`'s `incomplete`).
fn incomplete_todos(phases: &Value) -> Value {
    let todos: Vec<Value> = phases
        .as_array()
        .map(|ps| {
            ps.iter()
                .filter_map(|p| p.get("tasks"))
                .filter_map(Value::as_array)
                .flatten()
                .filter(|t| {
                    t.get("status")
                        .and_then(Value::as_str)
                        .is_some_and(|s| s == "pending" || s == "in_progress")
                })
                .map(|t| {
                    json!({
                        "content": t.get("content").cloned().unwrap_or(Value::Null),
                        "status": t.get("status").cloned().unwrap_or(Value::Null),
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    json!(todos)
}

/// Whether any phase has an open task (`pending`/`in_progress`).
fn has_incomplete_tasks(phases: &Value) -> bool {
    phases.as_array().is_some_and(|ps| {
        ps.iter()
            .filter_map(|p| p.get("tasks"))
            .filter_map(Value::as_array)
            .flatten()
            .any(|t| {
                t.get("status")
                    .and_then(Value::as_str)
                    .is_some_and(|s| s == "pending" || s == "in_progress")
            })
    })
}

/// The prompt as the agent takes it: the user message, then any extension-injected messages.
/// The prompt's agent input: `prepend` (hidden notices, TS `prependMessages`)
/// ahead of the user message, extension-`injected` messages after it.
pub fn prompt_input(
    text: &str,
    images: Vec<pa_agent::types::ImageContent>,
    prepend: Vec<AgentMessage>,
    injected: Vec<AgentMessage>,
) -> pa_agent::agent::AgentPromptInput {
    if prepend.is_empty() && injected.is_empty() {
        return pa_agent::agent::AgentPromptInput::Text {
            text: text.to_string(),
            images,
        };
    }
    let mut content = vec![pa_agent::types::UserPart::Text(
        pa_agent::types::TextContent {
            text: text.to_string(),
            text_signature: None,
        },
    )];
    content.extend(images.into_iter().map(pa_agent::types::UserPart::Image));
    let user = AgentMessage::Standard(Message::User(pa_agent::types::UserMessage {
        content: UserContent::Parts(content),
        timestamp: pa_agent::now_ms(),
    }));
    let mut all = prepend;
    all.push(user);
    all.extend(injected);
    pa_agent::agent::AgentPromptInput::Messages(all)
}

fn user_message(text: &str) -> AgentMessage {
    AgentMessage::Standard(Message::User(pa_agent::types::UserMessage {
        content: UserContent::Text(text.to_string()),
        timestamp: pa_agent::now_ms(),
    }))
}

/// A user-authored message whose text is exactly `text` (queue edits match exactly).
fn is_user_text(m: &AgentMessage, text: &str) -> bool {
    matches!(m, AgentMessage::Standard(Message::User(_))) && message_text(m) == text
}

fn run_outcome(messages: &[AgentMessage]) -> RunOutcome {
    for m in messages.iter().rev() {
        if let AgentMessage::Standard(Message::Assistant(a)) = m {
            return match a.stop_reason {
                StopReason::Error => {
                    // TS `promptError`: strip the local raw-request dump lines, classify with the
                    // provider's HTTP status, and report that status as `httpStatus`.
                    let raw = a.error_message.as_deref().unwrap_or("");
                    let message = if raw.is_empty() {
                        "Provider request failed".to_string()
                    } else {
                        pa_ai::utils::http_inspector::strip_raw_http_request_diagnostics(raw)
                    };
                    let mut error = json!({
                        "message": message,
                        "provider": a.provider,
                        "model": a.model,
                        "retryable": recovery::is_transient(a.error_status, raw),
                    });
                    if let Some(status) = a.error_status {
                        error["httpStatus"] = json!(status);
                    }
                    RunOutcome {
                        status: "error",
                        error: Some(error),
                    }
                }
                StopReason::Aborted => RunOutcome {
                    status: "aborted",
                    error: None,
                },
                _ => RunOutcome {
                    status: "completed",
                    error: None,
                },
            };
        }
    }
    RunOutcome {
        status: "completed",
        error: None,
    }
}

/// Characters the system prompt and tool schemas add to every request.
fn prompt_overhead_chars(
    system_prompt: &str,
    tools: &[Arc<dyn pa_agent::types::AgentTool>],
) -> usize {
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
/// Last provider-reported usage plus an estimate of everything after it (or the whole
/// transcript + prompt overhead when no assistant reported usage). One reverse pass, so
/// it reads the transcript in place (`Agent::with_messages`).
fn context_usage<'a>(
    messages: impl DoubleEndedIterator<Item = &'a AgentMessage>,
    window: u64,
    overhead_chars: usize,
) -> Option<Value> {
    if window == 0 {
        return None;
    }
    let mut trailing = 0u64;
    let mut reported = None;
    for m in messages.rev() {
        match m {
            AgentMessage::Standard(Message::Assistant(a))
                if a.usage.total_tokens > 0 && a.stop_reason != StopReason::Aborted =>
            {
                reported = Some(a.usage.total_tokens);
                break;
            }
            _ => trailing += estimate_tokens(m),
        }
    }
    let tokens = match reported {
        Some(total) => total + trailing,
        None => u64::try_from(overhead_chars.div_ceil(4)).unwrap_or(u64::MAX) + trailing,
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

/// Borrowed view of a `message` entry line: deserializing the envelope straight
/// from the raw journal text yields the entry's `AgentMessage` without building
/// the entry's `Value` DOM (lazy resume).
#[derive(serde::Deserialize)]
struct MessageEnvelope {
    message: AgentMessage,
}

/// Head-only view of a `message` entry: enough for the context rebuild's
/// default-model inference (`role`/`provider`/`model`) without the payload.
/// Set by `main` when `--model`/`--provider` (resp. `--thinking`) was given:
/// the startup resume keeps the explicit choice instead of the session's
/// (TS `hasExplicitModel`); cleared by the first `load_session`.
pub static MODEL_PINNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub static THINKING_PINNED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Set by `main` for `-p` runs that persist no session: automatic titling
/// is skipped (see `spawn_auto_title`).
pub static NO_SESSION_PRINT: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Set by `main` when `--model` was an ordered chain: `(provider/id of the
/// selected entry, the entries after it)`. The startup recovery config of that
/// model walks this chain instead of its configured `retry.fallbackChains`
/// (TS `installSubagentRetryFallbackChain`: a non-empty selector chain wins).
pub static LAUNCH_FALLBACKS: std::sync::OnceLock<(String, Vec<String>)> =
    std::sync::OnceLock::new();

/// [`LAUNCH_FALLBACKS`] applied to the startup resolution of the live model.
fn apply_launch_fallbacks(resolved: &mut omp_providers::ResolvedModel) {
    if let Some((primary, chain)) = LAUNCH_FALLBACKS.get() {
        if *primary == format!("{}/{}", resolved.provider, resolved.id) && !chain.is_empty() {
            resolved.fallbacks.clone_from(chain);
        }
    }
}

/// Model/thinking bookkeeping of a session branch (TS `buildSessionContext`
/// `models` + `getLastModelChangeRole` + thinking fields).
#[derive(Debug, Default, PartialEq)]
struct SessionModelState {
    models: std::collections::BTreeMap<String, String>,
    last_role: Option<String>,
    /// `(thinkingLevel, configured)` of the last `thinking_level_change`.
    thinking: Option<(String, Option<String>)>,
    /// `ultra` of that same entry (omp-rs ultra mode; absent in TS files).
    ultra: Option<String>,
}

#[derive(serde::Deserialize)]
struct BookkeepingHead {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    role: Option<String>,
    #[serde(rename = "thinkingLevel", default)]
    thinking_level: Option<String>,
    #[serde(default)]
    configured: Option<String>,
    #[serde(default)]
    ultra: Option<String>,
}

/// Read only the branch's `model_change` / `thinking_level_change` entries
/// and, when no explicit default model exists, the last assistant message's
/// `provider/model` (TS infers `models.default` from assistant messages
/// until an explicit default is recorded).
fn session_model_state(
    index: &omp_session::index::SessionIndex,
    path: &Path,
    branch: &[u32],
) -> Result<SessionModelState> {
    use omp_session::index::EntryKind;
    let picks: Vec<u32> = branch
        .iter()
        .copied()
        .filter(|&i| {
            matches!(
                index.refs[i as usize].kind,
                EntryKind::ModelChange | EntryKind::ThinkingLevelChange
            )
        })
        .collect();
    let mut st = SessionModelState::default();
    index.for_each_line(path, &picks, |r, raw| {
        let Ok(h) = serde_json::from_str::<BookkeepingHead>(raw) else {
            return Ok(());
        };
        if r.kind == EntryKind::ModelChange {
            let role = h.role.unwrap_or_else(|| "default".to_owned());
            if let Some(m) = h.model.filter(|m| !m.is_empty()) {
                st.models.insert(role.clone(), m);
            }
            st.last_role = Some(role);
        } else {
            st.thinking = Some((
                h.thinking_level.unwrap_or_else(|| "off".to_owned()),
                h.configured,
            ));
            st.ultra = h.ultra;
        }
        Ok(())
    })?;
    if !st.models.contains_key("default") {
        for &i in branch.iter().rev() {
            if index.refs[i as usize].kind != EntryKind::Message {
                continue;
            }
            let mut found = None;
            index.for_each_line(path, &[i], |_, raw| {
                if let Ok(MessageHead { message: Some(m) }) =
                    serde_json::from_str::<MessageHead>(raw)
                {
                    if m.role.as_deref() == Some("assistant") {
                        found = Some(format!(
                            "{}/{}",
                            m.provider.as_deref().unwrap_or("undefined"),
                            m.model.as_deref().unwrap_or("undefined")
                        ));
                    }
                }
                Ok(())
            })?;
            if let Some(m) = found {
                st.models.insert("default".to_owned(), m);
                break;
            }
        }
    }
    Ok(st)
}

#[derive(serde::Deserialize)]
struct MessageHead {
    #[serde(default)]
    message: Option<MessageHeadFields>,
}

#[derive(serde::Deserialize)]
struct MessageHeadFields {
    #[serde(default)]
    role: Option<String>,
    #[serde(default)]
    provider: Option<String>,
    #[serde(default)]
    model: Option<String>,
}

#[derive(serde::Deserialize)]
struct CustomHead {
    #[serde(rename = "customType", default)]
    custom_type: Option<String>,
}

/// Bookkeeping entry kinds whose full payload the context-state accumulation
/// reads from anywhere on the branch (all small).
fn is_bookkeeping(kind: omp_session::index::EntryKind) -> bool {
    matches!(
        kind,
        omp_session::index::EntryKind::ThinkingLevelChange
            | omp_session::index::EntryKind::ModelChange
            | omp_session::index::EntryKind::ServiceTierChange
            | omp_session::index::EntryKind::TtsrInjection
            | omp_session::index::EntryKind::ModeChange
    )
}

/// Link-only stand-in for a branch entry the context rebuild never reads
/// beyond its `type`/`id`/`timestamp` (pre-history messages, old compactions,
/// labels, ...). Keeps the reduced branch layout-identical to the journal.
fn link_shim(r: &omp_session::index::EntryRef) -> Value {
    let mut o = serde_json::Map::with_capacity(3);
    o.insert("type".into(), r.type_name.to_string().into());
    if !r.id.is_empty() {
        o.insert("id".into(), r.id.to_string().into());
    }
    if !r.timestamp.is_empty() {
        o.insert("timestamp".into(), r.timestamp.to_string().into());
    }
    Value::Object(o)
}

/// A message shim carrying just the fields the rebuild's state accumulation
/// looks at (`message.role`/`provider`/`model` for the default-model inference).
fn message_head_shim(r: &omp_session::index::EntryRef, raw: &str) -> Value {
    let mut shim = link_shim(r);
    let Ok(head) = serde_json::from_str::<MessageHead>(raw) else {
        return shim;
    };
    if let Some(m) = head.message {
        let mut msg = serde_json::Map::with_capacity(3);
        if let Some(role) = m.role {
            msg.insert("role".into(), role.into());
        }
        if let Some(provider) = m.provider {
            msg.insert("provider".into(), provider.into());
        }
        if let Some(model) = m.model {
            msg.insert("model".into(), model.into());
        }
        shim["message"] = Value::Object(msg);
    }
    shim
}

/// The `customType` of a `custom` entry line, when it parses as one.
fn custom_type(raw: &str) -> Option<String> {
    serde_json::from_str::<CustomHead>(raw).ok()?.custom_type
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
    files.sort_by_key(|b| std::cmp::Reverse(b.0));
    files.into_iter().map(|(_, p)| p).find(|p| {
        std::fs::read_to_string(p)
            .is_ok_and(|t| t.lines().any(|l| l.contains("\"type\":\"message\"")))
    })
}

/// Models whose provider has credentials (plus the live model), for
/// `get_available_models` / `cycle_model`.
pub async fn available_models(current: Option<&pa_types::ai::Model>) -> Vec<pa_types::ai::Model> {
    let mut out: Vec<pa_types::ai::Model> = Vec::new();
    for provider in pa_ai::models_generated::get_providers() {
        let models = pa_ai::models_generated::get_models(provider);
        let Some(first) = models.first() else {
            continue;
        };
        let has_key = omp_providers::resolve_async(format!("{provider}/{}", first.id))
            .await
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
        .get_or_init(|| {
            serde_json::from_str(include_str!("builtin_commands.json")).unwrap_or_default()
        })
        .clone()
}

// -------------------------------------------------- extension hook points
// (CliExt2: new functions only; used by `ext.rs` for the extension host.)

impl Runtime {
    /// Run `f` against the live session writer (tree navigation: leaf moves,
    /// `branch_summary` appends).
    #[allow(dead_code)] // host-request API used by tree navigation, wired with the rpc surface
    pub fn with_writer<R>(&self, f: impl FnOnce(&mut omp_session::SessionWriter) -> R) -> R {
        self.persist.with_writer(f)
    }

    /// Move the append point to an earlier entry (`/tree` navigation, omp
    /// `SessionManager.branch`); tracked by the compaction guard's navigation
    /// mirror (bug-hunt #4: compact must target the selected branch).
    ///
    /// # Errors
    /// `id` is not an entry of the live session file.
    pub fn branch_to(&self, id: Option<String>) -> Result<()> {
        self.persist.branch_to(id)
    }

    /// Append a `branch_summary` entry parented at `branch_from` and land the
    /// writer on it (`/tree` navigation that abandons the branch durably).
    ///
    /// # Errors
    /// `branch_from` is not an entry of the live session file, or the write failed.
    pub fn append_branch_summary(
        &self,
        branch_from: Option<&str>,
        summary: &str,
        details: Option<Value>,
        from_extension: Option<bool>,
    ) -> Result<String> {
        self.persist.append_branch_summary(
            branch_from,
            summary,
            details,
            from_extension,
            &crate::persist::now_iso(),
        )
    }

    /// Append a non-message entry (`custom`, extension `append_entry`) to the
    /// live session file; a no-op write-wise for `--no-session` scratch files.
    ///
    /// # Errors
    /// Session-writer serialization/I/O failure.
    pub fn append_entry(&self, kind: &str, payload: Value) -> Result<()> {
        self.persist
            .append_entry(kind, payload)
            .map_err(|e| anyhow::anyhow!(e))
    }

    /// Queue a user message from an extension (`send_message` notification).
    pub fn extension_message(self: &Arc<Self>, text: &str) {
        // `prompt` steers into a live run and admits otherwise, matching omp's
        // queueing of extension-originated messages.
        let rt = self.clone();
        let text = text.to_string();
        tokio::spawn(async move {
            if let Err(e) = rt.prompt(None, &text, Some("steer"), Vec::new()).await {
                eprintln!("omt: extension send_message: {}", e.message);
            }
        });
    }

    /// Forward an extension ui notification as an `extension_ui_request` frame
    /// (rpc mode golden frames; print mode drops it — no consumer).
    pub fn send_extension_frame(&self, extension: Option<&str>, method: &str, params: &Value) {
        self.send(json!({
            "type": "extension_ui_request",
            "extension": extension,
            "method": method,
            "params": params,
        }));
    }
}

pub fn unsupported(what: &str) -> CmdError {
    CmdError::new(format!("{what} is not supported by omt"))
}

#[cfg(test)]
mod todo_reminder_tests {
    use super::*;
    use serde_json::json;

    fn phases(open: &[(&str, &str)], done: &[(&str, &str)]) -> Value {
        let task = |(content, status): &(&str, &str)| json!({"content": content, "status": status});
        json!([{
            "name": "Tasks",
            "tasks": open.iter().map(task).chain(done.iter().map(task)).collect::<Vec<_>>(),
        }])
    }

    #[test]
    fn incomplete_todos_flattens_open_tasks() {
        let v = phases(
            &[("a", "pending"), ("b", "in_progress")],
            &[("c", "completed")],
        );
        let inc = incomplete_todos(&v);
        assert_eq!(
            inc,
            json!([
                {"content": "a", "status": "pending"},
                {"content": "b", "status": "in_progress"},
            ])
        );
        assert!(has_incomplete_tasks(&v));
        let all_done = phases(&[], &[("a", "completed")]);
        assert_eq!(incomplete_todos(&all_done), json!([]));
        assert!(!has_incomplete_tasks(&all_done));
        assert!(!has_incomplete_tasks(&json!([])));
    }

    #[test]
    fn awaiting_user_answer_question_lines() {
        assert!(is_awaiting_user_answer("Which approach should I take?"));
        assert!(is_awaiting_user_answer(
            "Done so far.\nDo you want me to continue?"
        ));
        assert!(!is_awaiting_user_answer("All tasks are complete."));
        assert!(!is_awaiting_user_answer(
            "Refactored the parser and updated tests."
        ));
    }

    #[test]
    fn awaiting_user_answer_response_cues_and_labels() {
        assert!(is_awaiting_user_answer("Please confirm the approach."));
        assert!(is_awaiting_user_answer("Let me know if that works."));
        assert!(is_awaiting_user_answer("Q: what is your pick?"));
        // TS `isQuestionPromptLine`: any trailing question starting with an
        // interrogative counts, even a rhetorical one about the code.
        assert!(is_awaiting_user_answer(
            "Why does the parser reject this input?"
        ));
    }

    #[test]
    fn notice_and_stream_frames_match_ts_shape() {
        let n =
            crate::events::notice_frame("warning", "disk almost full", Some("session-persistence"));
        assert_eq!(n["type"], json!("notice"));
        assert_eq!(n["level"], json!("warning"));
        assert_eq!(n["source"], json!("session-persistence"));
        let n = crate::events::notice_frame("info", "ok", None);
        assert!(n.get("source").is_none());

        let t = crate::events::tool_stream_update_frame("call-1", "edit", &json!({"kind": "diff"}));
        assert_eq!(t["type"], json!("tool_stream_update"));
        assert_eq!(t["toolCallId"], json!("call-1"));
        assert_eq!(t["toolName"], json!("edit"));
        assert_eq!(t["update"]["kind"], json!("diff"));

        let r = crate::events::todo_reminder_frame(
            &json!([{"content": "a", "status": "pending"}]),
            1,
            3,
        );
        assert_eq!(r["type"], json!("todo_reminder"));
        assert_eq!(r["attempt"], json!(1));
        assert_eq!(r["maxAttempts"], json!(3));
        assert_eq!(r["todos"][0]["content"], json!("a"));
        assert_eq!(
            crate::events::todo_auto_clear_frame()["type"],
            json!("todo_auto_clear")
        );
    }
}

#[cfg(test)]
mod context_usage_tests {
    use super::*;
    use serde_json::json;

    /// The previous slice-based implementation, kept as the equivalence oracle.
    fn reference(messages: &[AgentMessage], window: u64, overhead_chars: usize) -> Option<Value> {
        if window == 0 {
            return None;
        }
        let last_usage = messages.iter().rposition(|m| {
            matches!(m, AgentMessage::Standard(Message::Assistant(a)) if a.usage.total_tokens > 0 && a.stop_reason != StopReason::Aborted)
        });
        let tokens = match last_usage {
            Some(i) => {
                let AgentMessage::Standard(Message::Assistant(a)) = &messages[i] else {
                    return None;
                };
                a.usage.total_tokens + messages[i + 1..].iter().map(estimate_tokens).sum::<u64>()
            }
            None => {
                u64::try_from(overhead_chars.div_ceil(4)).unwrap_or(u64::MAX)
                    + messages.iter().map(estimate_tokens).sum::<u64>()
            }
        };
        #[allow(clippy::cast_precision_loss)]
        let percent = tokens as f64 / window as f64 * 100.0;
        Some(json!({"tokens": tokens, "contextWindow": window, "percent": percent}))
    }

    fn assistant(text: &str, total: u64, stop: &str) -> AgentMessage {
        let a: pa_agent::types::AssistantMessage = serde_json::from_value(json!({
            "content": [{"type": "text", "text": text}], "api": "openai-completions", "provider": "p", "model": "a",
            "usage": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "totalTokens": total,
                      "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0}},
            "stopReason": stop, "timestamp": 1
        }))
        .expect("assistant json");
        AgentMessage::Standard(Message::Assistant(a))
    }

    #[test]
    fn reverse_pass_matches_slice_reference() {
        let u = |t: &str| AgentMessage::user(t);
        let cases: Vec<Vec<AgentMessage>> = vec![
            vec![],
            vec![u("hello there")],
            vec![
                u("q"),
                assistant("a1", 120, "stop"),
                u("follow up question"),
            ],
            vec![
                u("q"),
                assistant("a1", 120, "stop"),
                assistant("a2", 0, "stop"),
                u("x"),
            ],
            vec![
                u("q"),
                assistant("a1", 120, "stop"),
                assistant("aborted", 999, "aborted"),
                u("y"),
            ],
            vec![assistant("only", 50, "stop")],
        ];
        for (k, msgs) in cases.iter().enumerate() {
            for window in [0u64, 1000] {
                assert_eq!(
                    context_usage(msgs.iter(), window, 37),
                    reference(msgs, window, 37),
                    "case {k} window {window}"
                );
            }
        }
    }
}

#[cfg(test)]
mod resume_seed_tests {
    use super::*;
    use serde_json::json;

    /// Shape of a real `developer` custom message (see ~/.omp journals): keys
    /// beyond `role` are an arbitrary payload pa_types cannot model.
    fn developer_map() -> Map<String, Value> {
        serde_json::from_value(json!({
            "role": "developer",
            "customType": "todo-reminder",
            "content": [{"type": "text", "text": "<system-reminder>keep going</system-reminder>"}],
            "display": false,
            "attribution": "agent",
            "timestamp": 1757890000000i64,
            "synthetic": true,
        }))
        .expect("developer map")
    }

    /// The non-compacted `load_session` path: context messages plus the raw
    /// `message` JSON of custom-role messages, straight from the journal lines.
    fn load_like_resume(
        path: &std::path::Path,
    ) -> (Vec<AgentMessage>, Vec<Option<Map<String, Value>>>) {
        let index = omp_session::index::SessionIndex::load(path).expect("index");
        let branch = index.branch();
        let mut messages = Vec::new();
        let mut raws = Vec::new();
        index
            .for_each_line(path, &branch, |r, raw| {
                if r.kind == omp_session::index::EntryKind::Message {
                    if let Ok(envelope) = serde_json::from_str::<MessageEnvelope>(raw) {
                        let m = envelope.message;
                        if matches!(m, AgentMessage::Custom(_)) {
                            raws.push(
                                serde_json::from_str::<Value>(raw)
                                    .ok()
                                    .and_then(|v| v.get("message").cloned())
                                    .and_then(|v| v.as_object().cloned()),
                            );
                        } else {
                            raws.push(None);
                        }
                        messages.push(m);
                    }
                }
                Ok(())
            })
            .expect("for_each_line");
        (messages, raws)
    }

    /// `message` payloads of every `message` entry in a session file, in order.
    fn message_payloads(path: &std::path::Path) -> Vec<Value> {
        omp_session::read_values(path)
            .expect("read_values")
            .into_iter()
            .filter(|v| v.get("type").and_then(Value::as_str) == Some("message"))
            .map(|v| v["message"].clone())
            .collect()
    }

    #[test]
    fn developer_custom_message_survives_double_resume_verbatim() {
        let base = std::env::temp_dir().join(format!(
            "omp-rs-resume-seed-{}-{}",
            std::process::id(),
            crate::persist::uuid_v7()
        ));
        std::fs::create_dir_all(&base).unwrap();
        let dir_a = base.join("a");
        std::fs::create_dir_all(&dir_a).unwrap();

        // Original journal: a user turn and a developer custom message.
        let mut w = omp_session::SessionWriter::create(
            &dir_a,
            "/proj",
            "sess-a",
            "2026-10-10T00-00-00-000Z",
        )
        .unwrap();
        w.append_omp_message(
            omp_session::OmpMessage::Agent(developer_map()),
            "2026-10-10T00:00:01.000Z",
        )
        .unwrap();
        let original = w.path().to_path_buf();
        drop(w);
        let original_payloads = message_payloads(&original);
        assert_eq!(original_payloads.len(), 1);
        // The developer entry must be a context message (Custom), confirming
        // the resume-bug premise: it loads into memory...
        let (messages, raws) = load_like_resume(&original);
        assert_eq!(messages.len(), 1);
        assert!(
            matches!(messages[0], AgentMessage::Custom(_)),
            "developer message must load as a Custom context message"
        );
        // ...but never converts to pa_types, so the old seeding loop dropped it.
        assert!(
            serde_json::from_value::<pa_types::session::AgentMessage>(
                serde_json::to_value(&messages[0]).unwrap()
            )
            .is_err(),
            "pa_types cannot model role=developer"
        );

        // Resume #1: seed into a new session file.
        let dir_b = base.join("b");
        std::fs::create_dir_all(&dir_b).unwrap();
        let persist_b = crate::persist::Persistence::for_test(&dir_b).unwrap();
        for (m, raw) in messages.iter().zip(&raws) {
            seed_resumed_message(&persist_b, m, raw.as_ref());
        }
        let file_b = persist_b.file().unwrap();
        let payloads_b = message_payloads(&file_b);
        assert_eq!(payloads_b.len(), 1, "every context message is re-persisted");
        assert_eq!(
            payloads_b[0], original_payloads[0],
            "developer message keeps its exact original shape"
        );
        assert_eq!(
            payloads_b[0].to_string(),
            original_payloads[0].to_string(),
            "developer message keeps its original key order (byte-identical)"
        );

        // Resume #2 from the new file: same context, same shape again.
        let (messages2, raws2) = load_like_resume(&file_b);
        assert_eq!(messages2.len(), 1);
        assert!(matches!(messages2[0], AgentMessage::Custom(_)));
        let dir_c = base.join("c");
        std::fs::create_dir_all(&dir_c).unwrap();
        let persist_c = crate::persist::Persistence::for_test(&dir_c).unwrap();
        for (m, raw) in messages2.iter().zip(&raws2) {
            seed_resumed_message(&persist_c, m, raw.as_ref());
        }
        let payloads_c = message_payloads(&persist_c.file().unwrap());
        assert_eq!(payloads_c, payloads_b, "double resume is stable");

        let _ = std::fs::remove_dir_all(&base);
    }

    /// The compacted `load_session` path end to end (what `load_session` does
    /// when the journal has a `compaction` entry): reduced-branch rebuild →
    /// context messages, plus `seed_branch` copying the raw branch lines into
    /// the fresh session file. A custom-role (`developer`) message in the kept
    /// range must survive in both the context and the seeded file, byte-identical.
    fn compacted_resume_once(
        path: &std::path::Path,
        dir: &std::path::Path,
    ) -> (Vec<AgentMessage>, std::path::PathBuf) {
        let index = omp_session::index::SessionIndex::load(path).expect("index");
        let branch = index.branch();
        let brefs: Vec<&omp_session::index::EntryRef> =
            branch.iter().map(|&i| &index.refs[i as usize]).collect();
        let (full, notes_after, head_until) =
            reduced_branch_flags(&index, path, &branch, &brefs).expect("flags");
        let entries =
            materialize_reduced_branch(&index, path, &branch, &full, notes_after, head_until)
                .expect("materialize");
        let ctx = omp_compact::build_session_context_owned(entries, omp_compact::Leaf::Last)
            .expect("context");
        let messages = crate::compact::context_messages(&ctx);
        let persist = crate::persist::Persistence::for_test(dir).unwrap();
        crate::compact::seed_branch(&persist, &index, path).expect("seed");
        (messages, persist.file().unwrap())
    }

    /// `type` field counts over a session file.
    fn entry_type_counts(path: &std::path::Path) -> std::collections::BTreeMap<String, usize> {
        let mut counts = std::collections::BTreeMap::new();
        for v in omp_session::read_values(path).expect("read_values") {
            if let Some(t) = v.get("type").and_then(Value::as_str) {
                *counts.entry(t.to_owned()).or_default() += 1;
            }
        }
        counts
    }

    #[test]
    fn developer_custom_message_survives_compacted_double_resume_verbatim() {
        let base = std::env::temp_dir().join(format!(
            "omp-rs-compacted-resume-seed-{}-{}",
            std::process::id(),
            crate::persist::uuid_v7()
        ));
        std::fs::create_dir_all(&base).unwrap();
        let dir_a = base.join("a");
        std::fs::create_dir_all(&dir_a).unwrap();

        // Compacted journal: user turn, developer custom message (kept range),
        // a compaction entry, then a post-compaction user turn.
        let mut w = omp_session::SessionWriter::create(
            &dir_a,
            "/proj",
            "sess-a",
            "2026-10-10T00-00-00-000Z",
        )
        .unwrap();
        let first_kept = w
            .append_omp_message(
                omp_session::OmpMessage::Agent(
                    serde_json::from_value(json!({
                        "role": "user",
                        "content": [{"type": "text", "text": "before compaction"}],
                        "timestamp": 1757890000000i64,
                    }))
                    .expect("user map"),
                ),
                "2026-10-10T00:00:01.000Z",
            )
            .unwrap();
        w.append_omp_message(
            omp_session::OmpMessage::Agent(developer_map()),
            "2026-10-10T00:00:02.000Z",
        )
        .unwrap();
        let result = omp_compact::summary::CompactionResult {
            summary: "summary text".to_owned(),
            short_summary: "short".to_owned(),
            first_kept_entry_id: first_kept,
            tokens_before: 1000,
            details: json!({"readFiles": [], "modifiedFiles": []}),
            preserve_data: None,
        };
        omp_compact::apply::append_compaction(
            &mut w,
            &result,
            "2026-10-10T00:00:03.000Z",
            &omp_compact::apply::ApplyOptions::default(),
        )
        .unwrap();
        w.append_omp_message(
            omp_session::OmpMessage::Agent(
                serde_json::from_value(json!({
                    "role": "user",
                    "content": [{"type": "text", "text": "after compaction"}],
                    "timestamp": 1757890001000i64,
                }))
                .expect("user map"),
            ),
            "2026-10-10T00:00:04.000Z",
        )
        .unwrap();
        let original = w.path().to_path_buf();
        drop(w);
        let original_payloads = message_payloads(&original);
        assert_eq!(original_payloads.len(), 3);
        let developer_payload = original_payloads[1].clone();
        assert_eq!(developer_payload["role"], json!("developer"));

        // Resume #1 through the compacted path: the developer message is a
        // Custom context message...
        let dir_b = base.join("b");
        std::fs::create_dir_all(&dir_b).unwrap();
        let (messages_b, file_b) = compacted_resume_once(&original, &dir_b);
        assert!(
            messages_b
                .iter()
                .any(|m| matches!(m, AgentMessage::Custom(_))),
            "compacted context keeps the developer message (n={})",
            messages_b.len()
        );
        // ...and the seeded file carries it byte-identically.
        let counts_b = entry_type_counts(&file_b);
        assert_eq!(
            counts_b.get("compaction").copied(),
            Some(1),
            "compaction entry is re-seeded"
        );
        let payloads_b = message_payloads(&file_b);
        assert_eq!(payloads_b.len(), 3, "every branch message is re-seeded");
        let seeded_developer = payloads_b
            .iter()
            .find(|p| p.get("role").and_then(Value::as_str) == Some("developer"))
            .expect("developer message in seeded file");
        assert_eq!(
            seeded_developer.to_string(),
            developer_payload.to_string(),
            "developer message keeps its exact original shape and key order"
        );

        // Resume #2 from the seeded file: same context, same shape again.
        let dir_c = base.join("c");
        std::fs::create_dir_all(&dir_c).unwrap();
        let (messages_c, file_c) = compacted_resume_once(&file_b, &dir_c);
        assert_eq!(
            messages_c.len(),
            messages_b.len(),
            "double resume keeps the same context"
        );
        assert!(
            messages_c
                .iter()
                .any(|m| matches!(m, AgentMessage::Custom(_))),
            "second compacted context keeps the developer message"
        );
        assert_eq!(
            entry_type_counts(&file_c).get("compaction").copied(),
            Some(1),
            "compaction entry survives the second seed"
        );
        let payloads_c = message_payloads(&file_c);
        assert_eq!(payloads_c, payloads_b, "double compacted resume is stable");

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn standard_message_keeps_existing_conversion_path() {
        let dir = std::env::temp_dir().join(format!(
            "omp-rs-resume-seed-std-{}-{}",
            std::process::id(),
            crate::persist::uuid_v7()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let persist = crate::persist::Persistence::for_test(&dir).unwrap();
        let user = AgentMessage::user("hello there");
        seed_resumed_message(&persist, &user, None);
        let payloads = message_payloads(&persist.file().unwrap());
        assert_eq!(payloads.len(), 1);
        assert_eq!(payloads[0]["role"], json!("user"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}

/// Herdr teardown: the last Runtime drop (TUI/RPC/print run ended) is the
/// agent exiting its pane — release the pane so its agent name, state, and
/// resume command clear at once instead of via the idle-shell safety net.
/// Blocking is bounded by the reporter's per-report timeout and no-ops unless
/// `HERDR_ENV=1`.
impl Drop for Runtime {
    fn drop(&mut self) {
        if let Some(h) = &self.herdr {
            h.release_blocking();
        }
    }
}

#[cfg(test)]
mod turn_outcome_tests {
    use super::*;
    use serde_json::json;

    fn failed(error_message: &str, error_status: Option<u16>) -> AgentMessage {
        let mut value = json!({
            "content": [], "api": "openai-completions", "provider": "openai", "model": "gpt-5.5",
            "usage": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 0,
                      "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0}},
            "stopReason": "error", "errorMessage": error_message, "timestamp": 1
        });
        if let Some(status) = error_status {
            value["errorStatus"] = json!(status);
        }
        let a: pa_agent::types::AssistantMessage =
            serde_json::from_value(value).expect("assistant json");
        AgentMessage::Standard(Message::Assistant(a))
    }

    #[test]
    fn rate_limit_reports_http_status_and_is_retryable() {
        let message = failed(
            "429 Rate limit reached\nraw-http-request=/home/u/.omp/logs/http-400-requests/1.json",
            Some(429),
        );
        let outcome = run_outcome(&[message]);
        assert_eq!(outcome.status, "error");
        let error = outcome.error.expect("error payload");
        assert_eq!(error["httpStatus"], json!(429));
        assert_eq!(error["retryable"], json!(true));
        assert_eq!(error["message"], json!("429 Rate limit reached"));
        assert_eq!(error["provider"], json!("openai"));
    }

    #[test]
    fn bad_request_reports_http_status_and_is_not_retryable() {
        let outcome = run_outcome(&[failed("400 Invalid value: 'x'.", Some(400))]);
        let error = outcome.error.expect("error payload");
        assert_eq!(error["httpStatus"], json!(400));
        assert_eq!(error["retryable"], json!(false));
    }

    #[test]
    fn error_without_status_omits_http_status() {
        let outcome = run_outcome(&[failed("", None)]);
        let error = outcome.error.expect("error payload");
        assert!(error.get("httpStatus").is_none());
        assert_eq!(error["message"], json!("Provider request failed"));
    }
}

#[cfg(test)]
mod session_model_state_tests {
    use super::*;

    fn state_of(lines: &[&str]) -> SessionModelState {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.jsonl");
        std::fs::write(&path, lines.join("\n") + "\n").unwrap();
        let index = omp_session::index::SessionIndex::load(&path).expect("index");
        session_model_state(&index, &path, &index.branch()).expect("state")
    }

    const HEAD: &str = r#"{"type":"session","version":3,"id":"s","timestamp":"t","cwd":"/tmp"}"#;

    /// Regression: resume must see the model the session last switched to
    /// (here kimi after a glm start), not fall back to the config default.
    #[test]
    fn last_model_change_and_thinking_are_read() {
        let st = state_of(&[
            HEAD,
            r#"{"type":"model_change","id":"a","parentId":null,"timestamp":"t","model":"zai/glm-5.3-flash"}"#,
            r#"{"type":"thinking_level_change","id":"b","parentId":"a","timestamp":"t","thinkingLevel":"high","configured":"high"}"#,
            r#"{"type":"message","id":"c","parentId":"b","timestamp":"t","message":{"role":"assistant","provider":"zai","model":"glm-5.3-flash","content":[]}}"#,
            r#"{"type":"model_change","id":"d","parentId":"c","timestamp":"t","model":"kimi-code/kimi-for-coding","role":"default"}"#,
            r#"{"type":"thinking_level_change","id":"e","parentId":"d","timestamp":"t","thinkingLevel":"medium","configured":"max"}"#,
        ]);
        assert_eq!(
            st.models.get("default").map(String::as_str),
            Some("kimi-code/kimi-for-coding")
        );
        assert_eq!(st.last_role.as_deref(), Some("default"));
        assert_eq!(st.thinking, Some(("medium".into(), Some("max".into()))));
    }

    /// No model_change at all (old journals): the default is inferred from the
    /// last assistant message, as TS `buildSessionContext` does.
    #[test]
    fn default_inferred_from_last_assistant() {
        let st = state_of(&[
            HEAD,
            r#"{"type":"message","id":"a","parentId":null,"timestamp":"t","message":{"role":"assistant","provider":"zai","model":"glm-5.3-flash","content":[]}}"#,
            r#"{"type":"message","id":"b","parentId":"a","timestamp":"t","message":{"role":"assistant","provider":"anthropic","model":"claude-opus-5-5","content":[]}}"#,
            r#"{"type":"message","id":"c","parentId":"b","timestamp":"t","message":{"role":"user","content":"hi"}}"#,
        ]);
        assert_eq!(
            st.models.get("default").map(String::as_str),
            Some("anthropic/claude-opus-5-5")
        );
        assert_eq!(st.last_role, None);
        assert_eq!(st.thinking, None);
    }
}

#[cfg(test)]
mod ultra_tests {
    use std::time::Duration;

    use pa_agent::agent::{Agent, AgentInitialState, AgentOptions};
    use pa_agent::scripted::ScriptedProvider;
    use tokio::sync::mpsc::UnboundedReceiver;

    use super::*;
    use crate::adapter::ProviderTarget;

    /// Tests that switch the process-wide live context strategy run one at a time.
    static CLM_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

    struct Named(&'static str);

    impl pa_agent::types::AgentTool for Named {
        fn name(&self) -> &'static str {
            self.0
        }
        fn description(&self) -> &'static str {
            "test tool"
        }
        fn parameters(&self) -> &Value {
            static SCHEMA: std::sync::OnceLock<Value> = std::sync::OnceLock::new();
            SCHEMA.get_or_init(|| json!({"type": "object", "properties": {}}))
        }
        fn execute(
            self: Arc<Self>,
            _id: String,
            _params: Value,
            _signal: pa_agent::abort::AbortSignal,
            _on_update: pa_agent::types::AgentToolUpdateCallback,
        ) -> pa_agent::BoxFut<'static, anyhow::Result<pa_agent::types::AgentToolResult>> {
            Box::pin(async { Ok(pa_agent::types::AgentToolResult::text("ok")) })
        }
    }

    fn model() -> pa_types::ai::Model {
        serde_json::from_value(json!({
            "id": "a", "name": "a", "api": "openai-completions", "provider": "p",
            "baseUrl": "http://127.0.0.1:9", "reasoning": true, "input": ["text"],
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
            "contextWindow": 100_000, "maxTokens": 4096
        }))
        .expect("model json")
    }

    struct Harness {
        rt: Arc<Runtime>,
        rx: UnboundedReceiver<Value>,
        provider: Arc<ScriptedProvider>,
        _cwd: tempfile::TempDir,
    }

    /// A runtime over a scripted provider, the CLI's `convert_to_llm` (custom
    /// notices reach the wire as developer messages) and tools named `tools`.
    /// `config` is the project `.omp/config.yml`; `dir` a real session directory.
    async fn harness(tools: &[&'static str], config: &str, dir: Option<&Path>) -> Harness {
        let model = model();
        let provider = Arc::new(ScriptedProvider::new(
            json_round_trip(&model).expect("agent model"),
        ));
        for i in 0..4 {
            provider.push_text_turn(&format!("answer {i}"));
        }
        let agent = Agent::new(AgentOptions {
            initial_state: AgentInitialState {
                system_prompt: Some("test".into()),
                model: json_round_trip(&model),
                thinking_level: Some(ThinkingLevel::Medium),
                tools: Some(
                    tools
                        .iter()
                        .map(|n| Arc::new(Named(n)) as Arc<dyn pa_agent::types::AgentTool>)
                        .collect(),
                ),
                messages: None,
            },
            convert_to_llm: Some(crate::agent::convert_to_llm()),
            stream_fn: Some(provider.stream_fn()),
            ..AgentOptions::default()
        });
        let cwd = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(cwd.path().join(".omp")).unwrap();
        std::fs::write(cwd.path().join(".omp/config.yml"), config).unwrap();
        let persist = match dir {
            Some(d) => {
                crate::persist::attach_in(&agent, cwd.path(), d, None, crate::persist::uuid_v7())
                    .await
            }
            None => crate::persist::attach_with(&agent, cwd.path(), true).await,
        }
        .expect("session");
        let target = Arc::new(std::sync::RwLock::new(Some(ProviderTarget {
            api_key: Some("k".into()),
            model,
            service_tier: None,
            headers: None,
            credential_id: None,
        })));
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let rt = Runtime::start(
            Arc::new(agent),
            target,
            cwd.path().to_path_buf(),
            persist,
            tx,
        )
        .await;
        Harness {
            rt,
            rx,
            provider,
            _cwd: cwd,
        }
    }

    async fn until_settled(rx: &mut UnboundedReceiver<Value>) -> Vec<Value> {
        let mut out = Vec::new();
        loop {
            let f = tokio::time::timeout(Duration::from_secs(30), rx.recv())
                .await
                .expect("frame in time")
                .expect("open");
            let done = f["type"] == "session_settled";
            out.push(f);
            if done {
                return out;
            }
        }
    }

    fn drain(rx: &mut UnboundedReceiver<Value>) -> Vec<Value> {
        std::iter::from_fn(|| rx.try_recv().ok()).collect()
    }

    fn of_type<'a>(frames: &'a [Value], kind: &str) -> Vec<&'a Value> {
        frames.iter().filter(|f| f["type"] == kind).collect()
    }

    fn thinking_entries(path: &Path) -> Vec<Value> {
        std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str::<Value>(l).unwrap())
            .filter(|v| v["type"] == "thinking_level_change")
            .collect()
    }

    /// `(role, customType or text)` of the last request's messages.
    fn last_wire(provider: &ScriptedProvider) -> Vec<(String, String)> {
        let calls = provider.calls();
        let ctx = calls.last().expect("a provider call");
        ctx.messages
            .iter()
            .map(|m| {
                let v = serde_json::to_value(m).unwrap();
                let text = match &v["content"] {
                    Value::String(s) => s.clone(),
                    Value::Array(parts) => parts
                        .iter()
                        .filter_map(|p| p["text"].as_str())
                        .collect::<Vec<_>>()
                        .join(""),
                    _ => String::new(),
                };
                (v["role"].as_str().unwrap_or_default().to_owned(), text)
            })
            .collect()
    }

    async fn prompt(h: &mut Harness, text: &str) -> Vec<Value> {
        h.rt.prompt(None, text, None, Vec::new())
            .await
            .expect("prompt");
        until_settled(&mut h.rx).await
    }

    /// Journal + restore round trip: the session line carries `ultra` next to a
    /// TS-readable `thinkingLevel`, and loading the session brings the mode back.
    #[tokio::test]
    async fn set_ultra_journals_and_restores() {
        let dir = tempfile::tempdir().unwrap();
        let mut h = harness(&["task", "eval"], "", Some(dir.path())).await;
        prompt(&mut h, "first").await;
        h.rt.set_ultra(Some(Ultra::Code)).await;
        let frames = drain(&mut h.rx);
        let changed = of_type(&frames, "thinking_level_changed");
        assert_eq!(changed.len(), 1, "{frames:?}");
        assert_eq!(changed[0]["ultra"], "ultracode");
        let level = h.rt.agent.thinking_level().await;
        assert_eq!(
            level,
            crate::model::clamp_level(&model(), ThinkingLevel::Max)
        );
        assert_eq!(changed[0]["thinkingLevel"], level_name(level));
        assert_eq!(h.rt.state().await["ultra"], "ultracode");
        assert_eq!(h.rt.context_usage_frame().await["ultra"], "ultracode");
        prompt(&mut h, "second").await;

        let file = h.rt.session_file();
        let entries = thinking_entries(&file);
        let last = entries.last().expect("thinking entry");
        assert_eq!(last["ultra"], "ultracode", "{last}");
        assert!(
            parse_level(last["thinkingLevel"].as_str().unwrap()).is_some(),
            "TS-readable level: {last}"
        );
        assert_eq!(last["configured"], last["thinkingLevel"]);
        // Entries without a mode stay TS-shaped (no `ultra` key at all).
        assert!(entries[0].get("ultra").is_none(), "{}", entries[0]);

        let mut second = harness(&["task", "eval"], "", Some(dir.path())).await;
        assert_eq!(second.rt.ultra(), None);
        second.rt.load_session(&file, None).await.expect("resume");
        assert_eq!(second.rt.ultra(), Some(Ultra::Code));
        assert_eq!(second.rt.state().await["ultra"], "ultracode");
        let frames = drain(&mut second.rx);
        assert!(
            of_type(&frames, "thinking_level_changed")
                .iter()
                .any(|f| f["ultra"] == "ultracode"),
            "{frames:?}"
        );
    }

    #[tokio::test]
    async fn set_thinking_level_clears_ultra() {
        let mut h = harness(&[], "", None).await;
        h.rt.set_ultra(Some(Ultra::Deep)).await;
        let level = h.rt.agent.thinking_level().await;
        drain(&mut h.rx);
        // Same level as ultra pinned: the mode still ends, announced and journaled.
        h.rt.set_thinking_level(level).await;
        assert_eq!(h.rt.ultra(), None);
        let frames = drain(&mut h.rx);
        let changed = of_type(&frames, "thinking_level_changed");
        assert_eq!(changed.len(), 1, "{frames:?}");
        assert_eq!(changed[0]["ultra"], Value::Null);
        assert_eq!(h.rt.state().await["ultra"], Value::Null);
        // `auto` clears it too; `/ultra off` keeps the level.
        h.rt.set_ultra(Some(Ultra::Code)).await;
        h.rt.set_auto_thinking().await;
        assert_eq!(h.rt.ultra(), None);
        assert!(of_type(&drain(&mut h.rx), "thinking_level_changed")
            .iter()
            .all(|f| f.get("ultra").is_some()));
        h.rt.set_ultra(Some(Ultra::Code)).await;
        let level = h.rt.agent.thinking_level().await;
        h.rt.set_ultra(None).await;
        assert_eq!(h.rt.ultra(), None);
        assert_eq!(h.rt.agent.thinking_level().await, level);
        let frames = drain(&mut h.rx);
        assert_eq!(
            of_type(&frames, "thinking_level_changed").last().unwrap()["ultra"],
            Value::Null
        );
    }

    /// Bare `/modelpreset` used to index `tokens[1..]` of an empty arg list and kill the process.
    #[tokio::test]
    async fn bare_modelpreset_reports_instead_of_panicking() {
        let mut h = harness(&[], "", None).await;
        drain(&mut h.rx);
        let outcome = crate::rpc::slash::try_dispatch(&h.rt, "/modelpreset").await;
        assert!(matches!(outcome, crate::rpc::slash::Outcome::Handled { agent_invoked: false }));
        assert_eq!(of_type(&drain(&mut h.rx), "command_output").len(), 1);
    }

    #[tokio::test]
    async fn ultra_notice_prepended_on_every_prompt() {
        let mut h = harness(&["task", "eval"], "", None).await;
        prompt(&mut h, "plain question").await;
        assert!(
            last_wire(&h.provider)
                .iter()
                .all(|(role, _)| role != "developer"),
            "no notice without a mode"
        );
        h.rt.set_ultra(Some(Ultra::Code)).await;
        for text in ["fix the parser", "ultracode it"] {
            prompt(&mut h, text).await;
            let wire = last_wire(&h.provider);
            let n = wire.len();
            assert_eq!(wire[n - 1].0, "user", "{wire:?}");
            assert_eq!(
                wire[n - 2].0,
                "developer",
                "notice right before the user message: {wire:?}"
            );
            assert!(
                wire[n - 2].1.contains("Ultra mode **ultracode** is active"),
                "{}",
                wire[n - 2].1
            );
            assert!(
                !wire[n - 3].1.contains("ultracode"),
                "exactly one notice: {wire:?}"
            );
        }
    }

    #[tokio::test]
    async fn keyword_notices_prepended_skipped_when_disabled_or_tool_missing() {
        let mut h = harness(&["task", "eval"], "", None).await;
        prompt(&mut h, "please jevify these files").await;
        let wire = last_wire(&h.provider);
        let n = wire.len();
        assert_eq!(wire[n - 2].0, "developer", "{wire:?}");
        assert!(wire[n - 2].1.contains("**jevify**"), "{}", wire[n - 2].1);
        assert_eq!(
            wire[n - 1],
            ("user".to_owned(), "please jevify these files".to_owned())
        );
        // Persisted as a hidden custom message.
        let notice =
            h.rt.messages()
                .await
                .into_iter()
                .find(|m| m["customType"] == "jevify-notice")
                .expect("journaled");
        assert_eq!(notice["display"], false);
        assert_eq!(notice["attribution"], "user");

        // Required tool missing.
        let mut h = harness(&["task"], "", None).await;
        prompt(&mut h, "please jevify these files").await;
        assert!(last_wire(&h.provider)
            .iter()
            .all(|(role, _)| role != "developer"));

        // Disabled by setting.
        let mut h = harness(&["task", "eval"], "magicKeywords:\n  jevify: false\n", None).await;
        prompt(&mut h, "please jevify these files").await;
        assert!(last_wire(&h.provider)
            .iter()
            .all(|(role, _)| role != "developer"));
    }

    #[tokio::test]
    async fn clm_switch_adds_and_removes_context_edit() {
        let _g = CLM_LOCK.lock().await;
        let mut h = harness(&["task", "eval"], "", None).await;
        let names = |rt: &Arc<Runtime>| {
            let rt = rt.clone();
            async move {
                rt.agent
                    .tools()
                    .await
                    .iter()
                    .map(|t| t.name().to_owned())
                    .collect::<Vec<_>>()
            }
        };
        h.rt.set_ultra(Some(Ultra::Code)).await;
        assert!(!names(&h.rt)
            .await
            .iter()
            .any(|n| n == crate::context::TOOL_NAME));
        assert!(crate::context::live().is_none());
        drain(&mut h.rx);

        h.rt.set_ultra(Some(Ultra::Long)).await;
        assert!(names(&h.rt)
            .await
            .iter()
            .any(|n| n == crate::context::TOOL_NAME));
        let live = crate::context::live().expect("clm live");
        assert_eq!(live.kind, crate::context::Kind::ClmHost);
        assert_eq!(crate::context::skip_extensions(), ["context-slim"]);
        let frames = drain(&mut h.rx);
        let notices = of_type(&frames, "notice");
        assert_eq!(notices.len(), 1, "{frames:?}");
        assert_eq!(notices[0]["level"], "warning");
        assert!(notices[0]["message"]
            .as_str()
            .unwrap()
            .contains("prompt cache"));

        // Long -> Hyper keeps CLM: no second switch.
        h.rt.set_ultra(Some(Ultra::Hyper)).await;
        assert!(of_type(&drain(&mut h.rx), "notice").is_empty());
        assert_eq!(
            names(&h.rt)
                .await
                .iter()
                .filter(|n| *n == crate::context::TOOL_NAME)
                .count(),
            1
        );

        // Leaving CLM (here via a plain thinking level) removes the tool again.
        h.rt.set_thinking_level(ThinkingLevel::High).await;
        assert!(!names(&h.rt)
            .await
            .iter()
            .any(|n| n == crate::context::TOOL_NAME));
        assert!(crate::context::live().is_none());
        assert!(crate::context::skip_extensions().is_empty());
        assert_eq!(of_type(&drain(&mut h.rx), "notice").len(), 1);
        assert_eq!(names(&h.rt).await, ["task", "eval"]);
    }

    /// Ultra is a mode, not a thinking level: no default / CLI flag / model
    /// suffix carries it (subagents never inherit it).
    #[test]
    fn ultra_is_not_a_thinking_level() {
        for u in Ultra::ALL {
            assert_eq!(parse_level(u.name()), None);
            assert_eq!(parse_ultra_name(u.name()), Some(u));
        }
        assert_eq!(parse_ultra_name("code"), None);
        assert_eq!(parse_ultra_name("max"), None);
    }
}

/// Compaction concurrency guard (bug-hunt #2): a prompt, `/new` or Esc during
/// in-flight compaction must cancel/serialize the pass — the concurrent turn
/// is never silently replaced and the summary never lands in the wrong file.
#[cfg(test)]
mod compaction_guard_tests {
    use std::sync::Arc;
    use std::time::Duration;

    use pa_agent::agent::{Agent, AgentInitialState, AgentOptions};
    use pa_agent::scripted::ScriptedProvider;
    use serde_json::{json, Value};
    use tokio::sync::mpsc::UnboundedReceiver;

    use super::*;
    use crate::adapter::{json_round_trip, ProviderTarget};
    use crate::compact::SummarizeFn;
    use omp_compact::CompactionSettings;

    fn model() -> pa_types::ai::Model {
        serde_json::from_value(json!({
            "id": "a", "name": "a", "api": "openai-completions", "provider": "p",
            "baseUrl": "http://127.0.0.1:9", "reasoning": false, "input": ["text"],
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
            "contextWindow": 100_000, "maxTokens": 4096
        }))
        .expect("model json")
    }

    fn reply(text: &str) -> pa_types::ai::AssistantMessage {
        serde_json::from_value(json!({
            "content": [{"type": "text", "text": text}], "api": "openai-completions", "provider": "p", "model": "a",
            "usage": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 0,
                      "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0}},
            "stopReason": "stop", "timestamp": 1
        }))
        .expect("reply json")
    }

    /// A summarizer that parks until the test releases it, so the test can
    /// interleave a prompt / session switch / Esc while the pass runs.
    struct Gate {
        entered: tokio::sync::watch::Sender<bool>,
        /// Dropped (or set `true`) by the test to release the summarizer.
        proceed: std::sync::Mutex<tokio::sync::watch::Receiver<bool>>,
    }

    impl Gate {
        fn new() -> (
            Arc<Self>,
            tokio::sync::watch::Receiver<bool>,
            tokio::sync::watch::Sender<bool>,
        ) {
            let (entered_tx, entered_rx) = tokio::sync::watch::channel(false);
            let (proceed_tx, proceed_rx) = tokio::sync::watch::channel(false);
            (
                Arc::new(Self {
                    entered: entered_tx,
                    proceed: std::sync::Mutex::new(proceed_rx),
                }),
                entered_rx,
                proceed_tx,
            )
        }
    }

    fn gated_summarizer(text: &'static str, gate: &Arc<Gate>) -> SummarizeFn {
        let gate = Arc::clone(gate);
        Arc::new(move |_req| {
            let gate = Arc::clone(&gate);
            Box::pin(async move {
                gate.entered.send_replace(true);
                loop {
                    let mut rx = gate.proceed.lock().unwrap().clone();
                    if *rx.borrow_and_update() {
                        break;
                    }
                    if rx.changed().await.is_err() {
                        break;
                    }
                }
                Ok(reply(text))
            })
        })
    }

    struct Harness {
        rt: Arc<Runtime>,
        rx: UnboundedReceiver<Value>,
        _dir: tempfile::TempDir,
    }

    /// A runtime over a scripted provider with a REAL session file, so journal
    /// assertions see what a compaction would have written.
    async fn harness(turns: usize) -> Harness {
        harness_scripted(|p| {
            for i in 0..turns {
                p.push_text_turn(&format!("answer {i} {}", "y".repeat(400)));
            }
        })
        .await
    }

    async fn harness_scripted(script: impl FnOnce(&ScriptedProvider)) -> Harness {
        let dir = tempfile::tempdir().unwrap();
        let model = model();
        let provider = Arc::new(ScriptedProvider::new(
            json_round_trip(&model).expect("agent model"),
        ));
        script(&provider);
        let agent = Agent::new(AgentOptions {
            initial_state: AgentInitialState {
                system_prompt: Some("test".into()),
                model: json_round_trip(&model),
                thinking_level: Some(pa_agent::types::ThinkingLevel::Off),
                tools: Some(Vec::new()),
                messages: None,
            },
            stream_fn: Some(provider.stream_fn()),
            ..AgentOptions::default()
        });
        let persist = crate::persist::attach_in(
            &agent,
            dir.path(),
            dir.path(),
            None,
            crate::persist::uuid_v7(),
        )
        .await
        .expect("session");
        let target = Arc::new(std::sync::RwLock::new(Some(ProviderTarget {
            api_key: Some("k".into()),
            model,
            service_tier: None,
            headers: None,
            credential_id: None,
        })));
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let rt = Runtime::start(
            Arc::new(agent),
            target,
            dir.path().to_path_buf(),
            persist,
            tx,
        )
        .await;
        std::mem::forget(provider);
        Harness { rt, rx, _dir: dir }
    }

    async fn until_settled(rx: &mut UnboundedReceiver<Value>) {
        loop {
            let f = tokio::time::timeout(Duration::from_secs(30), rx.recv())
                .await
                .expect("frame in time")
                .expect("open");
            if f["type"] == "session_settled" {
                return;
            }
        }
    }

    fn journal(path: &std::path::Path) -> Vec<Value> {
        std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect()
    }

    fn small_keep() -> CompactionSettings {
        CompactionSettings {
            keep_recent_tokens: 100,
            ..CompactionSettings::default()
        }
    }

    async fn talk(h: &mut Harness, prompts: usize) {
        for i in 0..prompts {
            h.rt.prompt(
                None,
                &format!("question {i} {}", "x".repeat(400)),
                None,
                Vec::new(),
            )
            .await
            .expect("prompt");
            until_settled(&mut h.rx).await;
        }
    }

    /// A prompt admitted while a compaction is in flight cancels the pass and
    /// waits for it, then runs on the settled context: no compaction entry is
    /// written and the turn lands in the journal.
    #[tokio::test]
    async fn prompt_during_in_flight_compaction_cancels_it_and_runs() {
        let mut h = harness(6).await;
        talk(&mut h, 1).await;
        let file = h.rt.session_file();
        let before = journal(&file);

        // `_proceed` keeps the gate closed until the test ends: the pass is cancelled, never released.
        let (gate, mut entered, _proceed) = Gate::new();
        h.rt.set_summarizer(Some(gated_summarizer("TOO LATE SUMMARY", &gate)));
        h.rt.set_compaction_settings(small_keep());

        let rt2 = Arc::clone(&h.rt);
        let compact = tokio::spawn(async move { rt2.compact(None).await });
        tokio::time::timeout(Duration::from_secs(30), async {
            while !*entered.borrow_and_update() {
                entered.changed().await.expect("gate");
            }
        })
        .await
        .expect("summarizer entered");

        // The user submits a prompt while the summary runs.
        h.rt.prompt(
            Some(json!("p2")),
            &format!("question 2 {}", "x".repeat(400)),
            None,
            Vec::new(),
        )
        .await
        .expect("prompt admitted behind the compaction");
        until_settled(&mut h.rx).await;

        let out = compact
            .await
            .expect("compact task")
            .expect_err("compaction cancelled");
        assert!(
            out.message.contains("Compaction cancelled"),
            "{}",
            out.message
        );

        let after = journal(&file);
        assert_eq!(
            after.iter().filter(|e| e["type"] == "compaction").count(),
            0,
            "no summary lands behind the prompt: {after:?}"
        );
        assert!(
            after.len() > before.len(),
            "the concurrent turn still lands"
        );
        assert!(
            after.last().unwrap().to_string().contains("answer 1"),
            "the prompt turn completed: {:?}",
            after.last().unwrap()
        );
    }

    /// Regression ("Request was aborted" after compact-gate's step-gate): an
    /// extension's `ctx.compact()` lands mid-run. The live turn is aborted, the
    /// summary commits, and the run resumes on the auto-continue prompt — no
    /// aborted stub is rendered or journaled.
    #[tokio::test]
    async fn compaction_mid_run_resumes_the_interrupted_turn() {
        let mut h = harness_scripted(|p| {
            for i in 0..2 {
                p.push_text_turn(&format!("answer {i} {}", "y".repeat(400)));
            }
            p.push_stalled_turn("");
            p.push_text_turn("resumed after summary");
        })
        .await;
        talk(&mut h, 2).await;
        h.rt.set_summarizer(Some(Arc::new(|_req| {
            Box::pin(async { Ok(reply("THE SUMMARY")) })
        })));
        h.rt.set_compaction_settings(small_keep());

        h.rt.prompt(None, "third question", None, Vec::new())
            .await
            .expect("prompt");
        tokio::time::timeout(Duration::from_secs(30), async {
            while !h.rt.agent.is_streaming().await {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("turn in flight");

        h.rt.compact(None).await.expect("compaction commits");

        let mut frames = Vec::new();
        let resumed = tokio::time::timeout(Duration::from_secs(30), async {
            loop {
                let f = h.rx.recv().await.expect("open");
                let done =
                    f.to_string().contains("resumed after summary") && f["type"] == "message_end";
                frames.push(f);
                if done {
                    return;
                }
            }
        })
        .await;
        assert!(resumed.is_ok(), "the interrupted run resumes: {frames:?}");
        assert!(
            !frames
                .iter()
                .any(|f| f.to_string().contains("Request was aborted")
                    && matches!(f["type"].as_str(), Some("message_start" | "message_end"))),
            "no aborted stub is rendered"
        );
        let entries = journal(&h.rt.session_file());
        assert_eq!(
            entries.iter().filter(|e| e["type"] == "compaction").count(),
            1
        );
        assert!(
            !entries
                .iter()
                .any(|e| e.to_string().contains("Request was aborted")),
            "no aborted stub is journaled: {entries:?}"
        );
        let after = entries
            .iter()
            .position(|e| e["type"] == "compaction")
            .unwrap();
        assert!(
            entries[after..]
                .iter()
                .any(|e| e.to_string().contains("Resume the user's latest intent")),
            "auto-continue prompt journaled after the summary"
        );
    }

    /// A no-op rejection ("Nothing to compact") changes no history: the turn the
    /// pass cut still resumes (TS `rejectedAsNoOp`).
    #[tokio::test]
    async fn noop_compaction_mid_run_still_resumes() {
        let mut h = harness_scripted(|p| {
            p.push_stalled_turn("");
            p.push_text_turn("resumed after no-op");
        })
        .await;
        h.rt.prompt(None, "first question", None, Vec::new())
            .await
            .expect("prompt");
        tokio::time::timeout(Duration::from_secs(30), async {
            while !h.rt.agent.is_streaming().await {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("turn in flight");
        let err = h.rt.compact(None).await.expect_err("too small");
        assert!(
            err.message.starts_with("Nothing to compact"),
            "{}",
            err.message
        );
        let resumed = tokio::time::timeout(Duration::from_secs(30), async {
            loop {
                let f = h.rx.recv().await.expect("open");
                if f["type"] == "message_end" && f.to_string().contains("resumed after no-op") {
                    return;
                }
            }
        })
        .await;
        assert!(resumed.is_ok(), "the interrupted run resumes");
    }

    /// Esc (abort_compaction) cancels the in-flight auto-compaction: the end
    /// event reports `aborted` and nothing is committed.
    #[tokio::test]
    async fn esc_cancels_in_flight_auto_compaction() {
        let mut h = harness(6).await;
        talk(&mut h, 1).await;
        let file = h.rt.session_file();

        let (gate, mut entered, proceed) = Gate::new();
        h.rt.set_summarizer(Some(gated_summarizer("NEVER COMMITTED", &gate)));
        h.rt.set_compaction_settings(small_keep());

        let rt2 = Arc::clone(&h.rt);
        let pass = tokio::spawn(async move { rt2.run_auto_compaction().await });
        tokio::time::timeout(Duration::from_secs(30), async {
            while !*entered.borrow_and_update() {
                entered.changed().await.expect("gate");
            }
        })
        .await
        .expect("summarizer entered");

        h.rt.abort_compaction(); // the Esc ladder's `abort_compaction`
        drop(proceed);
        pass.await.expect("auto-compaction task");

        let mut end = None;
        while let Ok(f) = h.rx.try_recv() {
            if f["type"] == "auto_compaction_end" {
                end = Some(f);
            }
        }
        let end = end.expect("auto_compaction_end frame");
        assert_eq!(end["aborted"], json!(true), "{end}");
        assert!(
            journal(&file).iter().all(|e| e["type"] != "compaction"),
            "nothing committed after Esc"
        );
    }

    /// `/new` during in-flight compaction discards the pass: the summary never
    /// lands in the old file and the fresh session is untouched.
    #[tokio::test]
    async fn new_session_discards_in_flight_compaction() {
        let mut h = harness(6).await;
        talk(&mut h, 1).await;
        let file = h.rt.session_file();
        let before = journal(&file);

        // `_proceed` keeps the gate closed until the test ends: the pass is cancelled, never released.
        let (gate, mut entered, _proceed) = Gate::new();
        h.rt.set_summarizer(Some(gated_summarizer("WRONG FILE SUMMARY", &gate)));
        h.rt.set_compaction_settings(small_keep());

        let rt2 = Arc::clone(&h.rt);
        let compact = tokio::spawn(async move { rt2.compact(None).await });
        tokio::time::timeout(Duration::from_secs(30), async {
            while !*entered.borrow_and_update() {
                entered.changed().await.expect("gate");
            }
        })
        .await
        .expect("summarizer entered");

        h.rt.new_session(None).await.expect("new session");
        let out = compact
            .await
            .expect("compact task")
            .expect_err("compaction cancelled");
        assert!(
            out.message.contains("Compaction cancelled"),
            "{}",
            out.message
        );

        assert_eq!(
            journal(&file)
                .iter()
                .filter(|e| e["type"] == "compaction")
                .count(),
            0,
            "the old file is untouched"
        );
        assert_eq!(
            journal(&file).len(),
            before.len(),
            "no late entries in the old file"
        );
        assert_ne!(h.rt.session_file(), file, "the fresh session file is live");

        // The fresh session still accepts turns.
        h.rt.prompt(
            None,
            &format!("fresh {}", "x".repeat(400)),
            None,
            Vec::new(),
        )
        .await
        .expect("prompt after switch");
        until_settled(&mut h.rx).await;
        let fresh = journal(&h.rt.session_file());
        assert!(
            fresh.last().unwrap().to_string().contains("answer 1"),
            "the turn landed in the new session: {:?}",
            fresh.last().unwrap()
        );
    }
}
