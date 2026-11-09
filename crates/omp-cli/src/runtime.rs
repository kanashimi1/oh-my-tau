//! Session runtime behind `--mode rpc`: the agent plus everything omp's
//! `AgentSession` adds around it (model switching, queues, session files,
//! stats, prompt-result bookkeeping, event forwarding).

use parking_lot::{Mutex, MutexGuard};
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use pa_agent::abort::AbortController;
use pa_agent::agent::Agent;
use pa_agent::types::{
    AgentMessage, AssistantContent, Message, ThinkingLevel, UserContent, UserPart,
};
use pi_edit::EditMode;
use serde_json::{json, Value};
use tokio::sync::mpsc::UnboundedSender;

use self::recovery::{RecoveryConfig, Resolver, TurnRecovery};
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
    /// This session's owner token on the a2a delivery bus: children spawned
    /// while this runtime is the registered owner belong to it, and only its
    /// park/wake/wait legs consume their deliveries.
    a2a_owner: Arc<str>,
    /// Set once the first prompt has been admitted into a run
    /// (`prompt_until_accepted` accepted it). Until then the session has no
    /// turn boundary: the idle-wake delivery leg must not start a delivery
    /// turn ahead of the first user prompt (a hub message replayed during
    /// startup waits in the queue for that prompt's settle leg).
    first_prompt_admitted: std::sync::atomic::AtomicBool,
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

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock()
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

mod abort;
mod compact;
mod delivery;
mod events;
mod model;
mod notify;
mod prompt;
mod queue;
mod retry;
mod session;
mod session_state;
mod settings;
mod state;
mod title;
mod todo;
mod usage;

use self::model::apply_launch_fallbacks;
pub(crate) use self::model::{
    available_models, model_info, LAUNCH_FALLBACKS, MODEL_PINNED, NO_SESSION_PRINT, THINKING_PINNED,
};
pub(crate) use self::prompt::prompt_input;
pub(crate) use self::session_state::newest_session;
#[cfg(test)]
pub(crate) use self::session_state::read_entries;
use self::todo::reminders_max;

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
    #[allow(clippy::too_many_lines)] // one constructor: resolver, retry config, persistence, event wiring, in order
    pub async fn start_with(
        agent: Arc<Agent>,
        target: Target,
        cwd: PathBuf,
        persist: Persistence,
        out: UnboundedSender<Value>,
        resolver: Resolver,
        recovery_cfg: Option<RecoveryConfig>,
    ) -> Arc<Self> {
        // Tests: registering the a2a delivery owner is a process-global side
        // effect; serialize with the bus tests so their owner stays stable
        // for the whole test body (production runs one runtime per process).
        #[cfg(test)]
        #[allow(clippy::disallowed_methods)] // tests: throwaway runtimes drive async code
        let _a2a_serial = crate::tools::a2a::tests::LOCK.lock().await;
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
            a2a_owner: crate::persist::uuid_v7().into(),
            first_prompt_admitted: std::sync::atomic::AtomicBool::new(false),
        });
        rt.publish_session_file();
        rt.record_model_state().await;
        // A2A: a child message/result arriving while the session is idle starts
        // a delivery turn (best effort; weak ref so the bus never pins a
        // finished Runtime). This runtime is the delivery owner: spawns stamp
        // its token, and only its park/wake/wait legs consume the queue. The
        // process-global registration stays for Env-less paths (hub
        // endpoint, BusLease::acquire); the per-session token also goes on
        // the tools' Envs so concurrent runtimes never share an owner.
        crate::tools::a2a::set_owner(rt.a2a_owner.clone());
        crate::tools::stamp_a2a_owner(&rt.agent, &rt.a2a_owner).await;
        {
            let weak = std::sync::Arc::downgrade(&rt);
            crate::tools::a2a::set_delivery_wake(std::sync::Arc::new(move || {
                let Some(rt) = weak.upgrade() else { return };
                tokio::spawn(async move {
                    rt.on_delivery_wake().await;
                });
            }));
        }
        // Hub link (SPEC batch 2): register this session as a live endpoint
        // (`hello` + `deliver` pushes injected at turn boundaries) on a
        // background task. Opted out with OMP_HUB=0; when the hub cannot be
        // started the task logs once and the session continues without it.
        // Unit tests drive the endpoint explicitly against a temp hub dir.
        #[cfg(not(test))]
        crate::tools::a2a::start_hub_endpoint(&rt.a2a_owner);
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
}

pub fn unsupported(what: &str) -> CmdError {
    CmdError::new(format!("{what} is not supported by omt"))
}

/// Herdr teardown: the last Runtime drop (TUI/RPC/print run ended) is the
/// agent exiting its pane — release the pane so its agent name, state, and
/// resume command clear at once instead of via the idle-shell safety net.
/// Blocking is bounded by the reporter's per-report timeout and no-ops unless
/// `HERDR_ENV=1`.
impl Drop for Runtime {
    fn drop(&mut self) {
        // A2A: session end kills this session's running `task` children (the
        // runner's drop guard SIGKILLs the process groups even while the
        // runtime unwinds) and releases the delivery owner token.
        crate::tools::task::kill_async_children_for(&Some(self.a2a_owner.clone()));
        crate::tools::a2a::clear_owner(&self.a2a_owner);
        // Hub link: release the session's auto-claims and stop its endpoint
        // task (bounded; a dead hub never stalls teardown).
        crate::tools::a2a::shutdown_hub_endpoint(&self.a2a_owner);
        if let Some(h) = &self.herdr {
            h.release_blocking();
        }
    }
}
