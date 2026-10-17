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
    /// `compaction.*` policy (config.yml) and the in-flight flag behind `isCompacting`.
    compaction: Mutex<omp_compact::CompactionSettings>,
    compacting: std::sync::atomic::AtomicBool,
    /// Injected summarizer (default: the live model).
    summarizer: Mutex<Option<crate::compact::SummarizeFn>>,
    /// Set by `abort()` (TS `#advisors.autoResumeSuppressed`): a follow-up-only
    /// queued-message drain stays suppressed until the next explicit user prompt.
    user_interrupt: std::sync::atomic::AtomicBool,
    /// Interactive-TUI hosts only: emit `context_usage` frames (status-line
    /// model identity + context gauge). RPC consumers never see them.
    ui_context_frames: std::sync::atomic::AtomicBool,
    /// `auto` thinking selector active (TS `ModelControls.#autoThinking`): the
    /// session runs a provisional concrete level and the status line shows
    /// `auto` until a level is classified.
    auto_thinking: std::sync::atomic::AtomicBool,
    /// Report stashed by a successful `rewind` tool result, consumed by the
    /// `TurnEnd` hook that truncates the context (TS `#pendingRewindReport`).
    pending_rewind_report: Mutex<Option<String>>,
    /// Sinks every applied session-name change (ext bridge `set_session_name`).
    on_session_name: Mutex<Option<NameListener>>,
    /// Native Herdr agent-state reporter (`HERDR_ENV=1` only; `None` elsewhere).
    herdr: Option<std::sync::Arc<crate::herdr::Herdr>>,
    /// `ask`-tool runs currently waiting on the user: >0 means Blocked.
    ask_blocked: std::sync::atomic::AtomicU64,
    /// In-flight automatic title generation (`--mode print` awaits it before exiting).
    title_task: Mutex<Option<tokio::task::JoinHandle<()>>>,
}

/// Sink for applied session-name changes (the ext bridge's `set_session_name`).
pub type NameListener = Arc<dyn Fn(&str) + Send + Sync>;

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
                Ok(r) => {
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
            user_interrupt: std::sync::atomic::AtomicBool::new(false),
            ui_context_frames: std::sync::atomic::AtomicBool::new(false),
            auto_thinking: std::sync::atomic::AtomicBool::new(false),
            pending_rewind_report: Mutex::new(None),
            on_session_name: Mutex::new(None),
            herdr,
            ask_blocked: std::sync::atomic::AtomicU64::new(0),
            title_task: Mutex::new(None),
            compaction: Mutex::new(compaction),
            compacting: std::sync::atomic::AtomicBool::new(false),
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
        let _ = self
            .persist
            .append_thinking_level_change(Some(level_name(self.agent.thinking_level().await)), None);
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
            turn_trace!("[turn-trace] framed={ev_kind} frame_json={n}B rss={}kB", rss_kb());
        }
        if matches!(
            event,
            AgentEvent::AgentEnd { .. }
                | AgentEvent::MessageEnd {
                    message: AgentMessage::Standard(Message::Assistant(_) | Message::ToolResult(_))
                }
        ) {
            self.publish_context_usage();
        }
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
                drop(p);
                self.herdr_working();
            }
            AgentEvent::ToolExecutionStart { tool_name, args, .. } if tool_name == "ask" => {
                // Herdr: an ask dialog blocks on the user.
                self.ask_blocked.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let message = args
                    .get("questions")
                    .and_then(Value::as_array)
                    .and_then(|qs| qs.iter().find_map(|q| q.get("question").and_then(Value::as_str)))
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
                    .fetch_update(std::sync::atomic::Ordering::SeqCst, std::sync::atomic::Ordering::SeqCst, |n| Some(n.saturating_sub(1)))
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
        if crate::tools::checkpoint::shared().checkpoint_state().is_none() {
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
                .with_messages(|m| (m.len() > count).then(|| m.iter().take(count).cloned().collect::<Vec<_>>()))
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
        let Some(worker) = omp_judge::shadow::global() else { return };
        let entries = omp_judge::shadow::message_entries(messages.iter().map(crate::events::message_value));
        let turn_index = self.judge_turns.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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
            tokio::spawn(async move {
                if rt.auto_compaction_due().await {
                    rt.emit_event(frame);
                    rt.run_auto_compaction().await;
                    rt.report_prompts(&run_outcome(&messages));
                    rt.drain_stranded_queue();
                } else {
                    rt.settle_terminal(frame, &messages, None);
                }
            });
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
            .is_some_and(|a| {
                matches!(a.stop_reason, StopReason::Error | StopReason::Aborted)
            });
        let interrupted = self
            .user_interrupt
            .load(std::sync::atomic::Ordering::SeqCst);
        if error_override.is_none() && !failed_turn && !interrupted
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
                eprintln!("omp-rs: todo reminder continuation failed: {e:#}");
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
    /// provider call, so any tail is valid). A drain that settles a user interrupt
    /// inherits the still-aborted session signal (TS): the resumed turn stops
    /// immediately, so the queued follow-up is never polled by it.
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
            // The drain run inherits the user interrupt (TS: the queued turn
            // runs under the still-aborted session signal). Abort on its first
            // assistant message so the stream takes the graceful aborted-message
            // path (`content: []`) and the queued follow-up is never polled.
            // The listener is removed once that run settles: `Subscription` has
            // no `Drop`, and a leaked listener aborted every later turn.
            let sub = if interrupted {
                let agent = rt.agent.clone();
                Some(
                    agent
                        .clone()
                        .subscribe(move |event, _| {
                            if matches!(
                                &event,
                                AgentEvent::MessageStart { message }
                                    if message.role() == "assistant"
                            ) {
                                agent.abort();
                            }
                            Box::pin(async { Ok(()) })
                        })
                        .await,
                )
            } else {
                None
            };
            let _ = rt.agent.continue_run_until_accepted().await;
            if let Some(sub) = sub {
                rt.agent.wait_for_idle().await;
                sub.unsubscribe().await;
            }
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
        self.swap_model(&resolved, Some(EPHEMERAL_ROLE), false)
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
        self.spawn_auto_title(message);
        if self.agent.is_streaming().await {
            match behavior {
                Some("steer") => self.agent.steer(user_message(message)),
                Some("followUp") => self.agent.follow_up(user_message(message)),
                _ => {
                    return Err(CmdError::new(
                        "Agent is already processing a prompt. Specify streamingBehavior ('steer' or 'followUp') to queue the message.",
                    ))
                }
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
        let input = pa_agent::agent::AgentPromptInput::Text {
            text: message.to_string(),
            images,
        };
        let pua = self.agent.prompt_until_accepted(input).await;
        if let Err(e) = pua {
            lock(&self.prompts)
                .open
                .retain(|t| !(t.starts_at_begin == starts && t.id == id && !t.waiting));
            return Err(CmdError::new(format!("{e:#}")));
        }
        Ok(())
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

    /// Abort the run and wait until the agent is idle.
    pub async fn abort(&self) {
        self.user_interrupt
            .store(true, std::sync::atomic::Ordering::SeqCst);
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
        if let Some(obj) = v.as_object() {
            out.extend(obj.clone());
        }
        let base = prompt_overhead_chars(&s.system_prompt, &s.tools);
        let window = s.model.context_window;
        if let Some(usage) = self.agent.with_messages(|m| context_usage(m.iter(), window, base)).await {
            out.insert("contextUsage".into(), usage);
        }
        Value::Object(out)
    }

    /// The `context_usage` frame the TUI status line renders: active model
    /// (selector + display name), context tokens/percent/window and the
    /// auto-compaction gauge boundaries (TS `getContextUsage` +
    /// `getSessionCompactionBoundaries`).
    pub async fn context_usage_frame(&self) -> Value {
        let (s, _) = self.agent.status().await;
        let base = prompt_overhead_chars(&s.system_prompt, &s.tools);
        let window = s.model.context_window;
        let usage = self.agent.with_messages(|m| context_usage(m.iter(), window, base)).await;
        let mut frame = json!({
            "type": "context_usage",
            "model": format!("{}/{}", s.model.provider, s.model.id),
            "modelName": s.model.name,
            "contextWindow": window,
            "thinkingLevel": level_name(s.thinking_level),
            "thinkingLevels": self.supported_efforts(),
            "autoThinking": self.auto_thinking.load(std::sync::atomic::Ordering::Relaxed),
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
        if !self.ui_context_frames.load(std::sync::atomic::Ordering::Relaxed) {
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
        let (user, assistant, calls, results, input, output, cache_read, cache_write, cost, usage) = self
            .agent
            .with_messages(|messages| {
                let (mut user, mut assistant, mut calls, mut results) = (0u64, 0u64, 0u64, 0u64);
                let (mut input, mut output, mut cache_read, mut cache_write, mut cost) =
                    (0u64, 0u64, 0u64, 0u64, 0.0f64);
                for m in messages.iter() {
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
                let usage = context_usage(messages.iter(), window, base);
                (user, assistant, calls, results, input, output, cache_read, cache_write, cost, usage)
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
        self.cancel_retry_and_wait().await;
        self.agent.abort();
        self.agent.wait_for_idle().await;
        self.agent.reset().await;
        self.persist.start_new(&self.cwd, dir)?;
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
        Ok(())
    }

    /// Resume `path` in place (TS `SessionManager.open`): the agent context is rebuilt from the
    /// journal and new entries are appended to the SAME file, chained to the active branch tip.
    /// `dir` is unused (the file keeps its own directory).
    pub async fn load_session(&self, path: &Path, dir: Option<&Path>) -> Result<()> {
        self.load_session_as(path, dir, false).await
    }

    /// Fork `path` (TS `SessionManager.forkFrom`): the continuation is written to a fresh
    /// session file in `dir` seeded with the loaded history.
    pub async fn fork_session(&self, path: &Path, dir: Option<&Path>) -> Result<()> {
        self.load_session_as(path, dir, true).await
    }

    /// Shared body of [`Self::load_session`] / [`Self::fork_session`].
    ///
    /// Resume is lazy like omp's: the journal is scanned once into a link index
    /// ([`omp_session::index::SessionIndex`]) and only the active branch — TS
    /// `SessionEntryIndex.pathTo(leaf)`, root → leaf — is materialized, parsed
    /// on demand from the raw lines. Non-compacted sessions never build the
    /// entry DOM at all: messages deserialize straight from the line text.
    async fn load_session_as(&self, path: &Path, dir: Option<&Path>, fork: bool) -> Result<()> {
        self.cancel_retry_and_wait().await;
        // `--no-session` must never write into the resumed file: seed a scratch copy instead.
        let fork = fork || self.persist.file().is_none();
        let trace_t0 = std::time::Instant::now();
        macro_rules! rss_trace {
            ($label:expr) => {
                if std::env::var_os("OMP_RS_RESUME_TRACE").is_some() {
                    if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
                        let rss = status.lines().find_map(|l| l.strip_prefix("VmRSS:")).unwrap_or("?");
                        let hwm = status.lines().find_map(|l| l.strip_prefix("VmHWM:")).unwrap_or("?");
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
            let (full, notes_after, head_until) = reduced_branch_flags(&index, path, &branch, &brefs)?;
            let entries = materialize_reduced_branch(
                &index,
                path,
                &branch,
                &full,
                notes_after,
                head_until,
            )?;
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
            self.persist.bind_existing(path, header_id, tip.filter(|t| !t.is_empty()))?;
            self.publish_session_file();
        }
        rss_trace!("seeded");
        rss_trace!("pre-set");
        if std::env::var_os("OMP_RS_RESUME_TRACE").is_some() {
            let json = serde_json::to_string(&messages).unwrap_or_default();
            eprintln!("[resume-trace] messages: n={} json={}B sizeof={}B", messages.len(), json.len(), std::mem::size_of_val(&messages));
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
        self.herdr_session("resume");
        lock(&self.prompts).open.clear();
        self.restore_title_from(&index);
        Ok(())
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
    let map = raw
        .cloned()
        .or_else(|| serde_json::to_value(m).ok().and_then(|v| v.as_object().cloned()));
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
            && custom_type(raw).as_deref()
                == Some(omp_compact::context::CONTEXT_NOTES_ENTRY_TYPE)
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
            let Some(title) = rt.generate_auto_title(&message).await else { return };
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
    pub async fn compact(&self, custom_instructions: Option<&str>) -> Result<Value, CmdError> {
        if self
            .compacting
            .swap(true, std::sync::atomic::Ordering::SeqCst)
        {
            return Err(CmdError::new("Compaction already in progress"));
        }
        self.agent.abort();
        self.agent.wait_for_idle().await;
        let out = self.compact_inner(custom_instructions).await;
        self.compacting
            .store(false, std::sync::atomic::Ordering::SeqCst);
        out
    }

    async fn compact_inner(&self, custom_instructions: Option<&str>) -> Result<Value, CmdError> {
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
        let params = crate::compact::CompactParams {
            settings: &settings,
            model_id: &model_id,
            context_window: Some(self.live_window()).filter(|w| *w > 0),
            custom_instructions,
            non_message_tokens,
        };
        if trace {
            turn_trace!("[turn-trace] compact-planned rss={}kB", rss_kb());
        }
        let (_result, data, view) = crate::compact::compact_now(
            &self.persist,
            &self.persist.path(),
            &summarizer,
            &params,
        )
        .await?;
        if trace {
            turn_trace!("[turn-trace] compact-summarized rss={}kB", rss_kb());
        }
        let rebuilt = crate::compact::rebuild_context(&view)?;
        if trace {
            turn_trace!("[turn-trace] compact-rebuilt rss={}kB", rss_kb());
        }
        self.agent.set_messages(rebuilt).await;
        if trace {
            turn_trace!("[turn-trace] compact-set rss={}kB", rss_kb());
        }
        self.agent
            .set_compaction_epoch(self.agent.compaction_epoch() + 1);
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
        self.compacting
            .store(true, std::sync::atomic::Ordering::SeqCst);
        self.emit_event(
            json!({"type": "auto_compaction_start", "reason": "threshold", "action": ACTION}),
        );
        let end = match self.compact_inner(None).await {
            Ok(result) => {
                json!({"type": "auto_compaction_end", "action": ACTION, "result": result, "aborted": false, "willRetry": false})
            }
            Err(e) => json!({
                "type": "auto_compaction_end", "action": ACTION, "aborted": false, "willRetry": false,
                "errorMessage": format!("Auto-compaction failed: {}", e.message),
            }),
        };
        self.compacting
            .store(false, std::sync::atomic::Ordering::SeqCst);
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
        self.swap_model(&resolved, role, false).await;
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
        let preferred = match catalog_default {
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
    ) {
        let agent_model: Option<pa_agent::types::Model> = json_round_trip(&resolved.model);
        let Some(agent_model) = agent_model else {
            return;
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
                self.agent.set_tools(tools).await;
            }
        }
        if changing {
            self.emit_event(json!({"type": "model_changed"}));
            if self.ui_context_frames.load(std::sync::atomic::Ordering::Relaxed) {
                let usage = self.context_usage_frame().await;
                self.send(usage);
            }
        }
        let selector = format!("{}/{}", resolved.provider, resolved.id);
        let _ = self
            .persist
            .append_model_change(&selector, role, is_fallback);
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
            let _ = self
                .persist
                .append_thinking_level_change(Some(name), Some(name));
            self.emit_event(json!({"type": "thinking_level_changed", "thinkingLevel": name}));
        }
        effective
    }

    pub async fn set_thinking_level(&self, level: ThinkingLevel) {
        // Pinning a concrete level leaves `auto` (TS `setThinkingLevel`).
        self.auto_thinking.store(false, std::sync::atomic::Ordering::Relaxed);
        self.apply_thinking(level).await;
    }

    /// `setThinkingLevel("auto")`: run a provisional concrete level and flag
    /// the session as auto (the status line shows `auto` until classified).
    pub async fn set_auto_thinking(&self) {
        let provisional = self
            .target
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_ref()
            .map_or(ThinkingLevel::Medium, |t| crate::model::clamp_level(&t.model, ThinkingLevel::Medium));
        self.apply_thinking(provisional).await;
        let name = level_name(provisional);
        let _ = self.persist.append_thinking_level_change(Some(name), Some("auto"));
        self.auto_thinking.store(true, std::sync::atomic::Ordering::Relaxed);
        self.emit_event(json!({"type": "thinking_level_changed", "thinkingLevel": name, "configured": "auto"}));
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
            Some(efforts) => efforts.into_iter().map(level_name).filter(|l| *l != "off").collect(),
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
    std::sync::LazyLock::new(|| regex::Regex::new(r"(?i)\b(?:you|your|we|our)\b").expect("valid regex"));
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
    let candidate = TRAILING_PUNCT_RE.replace(text.trim(), "").trim().to_string();
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
                                    t.get("status").and_then(Value::as_str).is_some_and(|s| {
                                        s == "pending" || s == "in_progress"
                                    })
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
                eprintln!("omp-rs: extension send_message: {}", e.message);
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
    CmdError::new(format!("{what} is not supported by omp-rs"))
}

#[cfg(test)]
mod todo_reminder_tests {
    use super::*;
    use serde_json::json;

    fn phases(open: &[(&str, &str)], done: &[(&str, &str)]) -> Value {
        let task = |(content, status): &(&str, &str)| {
            json!({"content": content, "status": status})
        };
        json!([{
            "name": "Tasks",
            "tasks": open.iter().map(task).chain(done.iter().map(task)).collect::<Vec<_>>(),
        }])
    }

    #[test]
    fn incomplete_todos_flattens_open_tasks() {
        let v = phases(&[("a", "pending"), ("b", "in_progress")], &[("c", "completed")]);
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
        assert!(is_awaiting_user_answer("Done so far.\nDo you want me to continue?"));
        assert!(!is_awaiting_user_answer("All tasks are complete."));
        assert!(!is_awaiting_user_answer("Refactored the parser and updated tests."));
    }

    #[test]
    fn awaiting_user_answer_response_cues_and_labels() {
        assert!(is_awaiting_user_answer("Please confirm the approach."));
        assert!(is_awaiting_user_answer("Let me know if that works."));
        assert!(is_awaiting_user_answer("Q: what is your pick?"));
        // TS `isQuestionPromptLine`: any trailing question starting with an
        // interrogative counts, even a rhetorical one about the code.
        assert!(is_awaiting_user_answer("Why does the parser reject this input?"));
    }

    #[test]
    fn notice_and_stream_frames_match_ts_shape() {
        let n = crate::events::notice_frame("warning", "disk almost full", Some("session-persistence"));
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

        let r = crate::events::todo_reminder_frame(&json!([{"content": "a", "status": "pending"}]), 1, 3);
        assert_eq!(r["type"], json!("todo_reminder"));
        assert_eq!(r["attempt"], json!(1));
        assert_eq!(r["maxAttempts"], json!(3));
        assert_eq!(r["todos"][0]["content"], json!("a"));
        assert_eq!(crate::events::todo_auto_clear_frame()["type"], json!("todo_auto_clear"));
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
            vec![u("q"), assistant("a1", 120, "stop"), u("follow up question")],
            vec![u("q"), assistant("a1", 120, "stop"), assistant("a2", 0, "stop"), u("x")],
            vec![u("q"), assistant("a1", 120, "stop"), assistant("aborted", 999, "aborted"), u("y")],
            vec![assistant("only", 50, "stop")],
        ];
        for (k, msgs) in cases.iter().enumerate() {
            for window in [0u64, 1000] {
                assert_eq!(context_usage(msgs.iter(), window, 37), reference(msgs, window, 37), "case {k} window {window}");
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
    fn load_like_resume(path: &std::path::Path) -> (Vec<AgentMessage>, Vec<Option<Map<String, Value>>>) {
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
    fn compacted_resume_once(path: &std::path::Path, dir: &std::path::Path) -> (Vec<AgentMessage>, std::path::PathBuf) {
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
                omp_session::OmpMessage::Agent(serde_json::from_value(json!({
                    "role": "user",
                    "content": [{"type": "text", "text": "before compaction"}],
                    "timestamp": 1757890000000i64,
                }))
                .expect("user map")),
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
            omp_session::OmpMessage::Agent(serde_json::from_value(json!({
                "role": "user",
                "content": [{"type": "text", "text": "after compaction"}],
                "timestamp": 1757890001000i64,
            }))
            .expect("user map")),
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
            messages_b.iter().any(|m| matches!(m, AgentMessage::Custom(_))),
            "compacted context keeps the developer message (n={})",
            messages_b.len()
        );
        // ...and the seeded file carries it byte-identically.
        let counts_b = entry_type_counts(&file_b);
        assert_eq!(counts_b.get("compaction").copied(), Some(1), "compaction entry is re-seeded");
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
        assert_eq!(messages_c.len(), messages_b.len(), "double resume keeps the same context");
        assert!(
            messages_c.iter().any(|m| matches!(m, AgentMessage::Custom(_))),
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
        let a: pa_agent::types::AssistantMessage = serde_json::from_value(value).expect("assistant json");
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
