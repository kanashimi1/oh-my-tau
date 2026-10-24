//! Lifecycle events the core emits itself, with the payload shapes of omp's `ExtensionRunner`
//! (`pi-coding-agent/src/extensibility/{shared-events,extensions/types}.ts`), plus the answers to the
//! host's own requests (`ui_confirm`, `compact`, `navigate_tree`, `set_model`, ...).
//!
//! Optional TS fields that are `undefined` are omitted on the wire (`JSON.stringify`), nullable ones are
//! `null`; field order follows the TS object literals so the serialized event matches byte for byte.
//! `AbortSignal` members (`session_before_compact`, `session_before_tree`, `session_stop`) are not
//! serializable and are never sent.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use anyhow::Result;
use serde::Serialize;
use serde_json::{json, Value};
use tokio::sync::mpsc;

use crate::{EmitResult, ExtHost, HostMessage};

/// Why a session switch happens (`session_before_switch` / `session_switch`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum SwitchReason {
    New,
    Resume,
    Fork,
}

/// Branch transition kind (`session_before_branch` / `session_branch`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum BranchReason {
    Branch,
    Fork,
    Btw,
}

/// Origin of an `input` event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum InputSource {
    Interactive,
    Rpc,
    Extension,
}

/// Why auto-compaction started.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum CompactionReason {
    Threshold,
    Overflow,
    Idle,
    Incomplete,
}

/// Auto-compaction strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum CompactionAction {
    #[serde(rename = "context-full")]
    ContextFull,
    #[serde(rename = "remote")]
    Remote,
    #[serde(rename = "handoff")]
    Handoff,
    #[serde(rename = "shake")]
    Shake,
    #[serde(rename = "snapcompact")]
    Snapcompact,
}

/// Tool approval mode (`tools/approval.ts`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ApprovalMode {
    #[serde(rename = "always-ask")]
    AlwaysAsk,
    #[serde(rename = "write")]
    Write,
    #[serde(rename = "yolo")]
    Yolo,
}

/// An omp extension event. Message/entry payloads are opaque omp JSON (`AgentMessage`, `SessionEntry`, ...).
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type")]
pub enum LifecycleEvent {
    #[serde(rename = "session_start")]
    SessionStart,
    #[serde(rename = "before_agent_start", rename_all = "camelCase")]
    BeforeAgentStart {
        prompt: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        images: Option<Value>,
        system_prompt: Vec<String>,
    },
    #[serde(rename = "input")]
    Input {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        images: Option<Value>,
        source: InputSource,
    },
    #[serde(rename = "agent_start")]
    AgentStart,
    #[serde(rename = "agent_end", rename_all = "camelCase")]
    AgentEnd {
        messages: Vec<Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        will_continue: Option<bool>,
    },
    #[serde(rename = "turn_start", rename_all = "camelCase")]
    TurnStart { turn_index: u64, timestamp: u64 },
    #[serde(rename = "turn_end", rename_all = "camelCase")]
    TurnEnd {
        turn_index: u64,
        message: Value,
        tool_results: Vec<Value>,
    },
    #[serde(rename = "message_start")]
    MessageStart { message: Value },
    #[serde(rename = "message_update", rename_all = "camelCase")]
    MessageUpdate {
        message: Value,
        assistant_message_event: Value,
    },
    #[serde(rename = "message_end")]
    MessageEnd { message: Value },
    #[serde(rename = "tool_execution_start", rename_all = "camelCase")]
    ToolExecutionStart {
        tool_call_id: String,
        tool_name: String,
        args: Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        intent: Option<String>,
    },
    #[serde(rename = "tool_execution_update", rename_all = "camelCase")]
    ToolExecutionUpdate {
        tool_call_id: String,
        tool_name: String,
        args: Value,
        partial_result: Value,
    },
    #[serde(rename = "tool_execution_end", rename_all = "camelCase")]
    ToolExecutionEnd {
        tool_call_id: String,
        tool_name: String,
        result: Value,
        is_error: bool,
    },
    #[serde(rename = "session_before_compact", rename_all = "camelCase")]
    SessionBeforeCompact {
        preparation: Value,
        branch_entries: Vec<Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        custom_instructions: Option<String>,
    },
    #[serde(rename = "session_compact", rename_all = "camelCase")]
    SessionCompact {
        compaction_entry: Value,
        from_extension: bool,
    },
    #[serde(rename = "session_before_switch", rename_all = "camelCase")]
    SessionBeforeSwitch {
        reason: SwitchReason,
        #[serde(skip_serializing_if = "Option::is_none")]
        target_session_file: Option<String>,
    },
    #[serde(rename = "session_switch", rename_all = "camelCase")]
    SessionSwitch {
        reason: SwitchReason,
        #[serde(skip_serializing_if = "Option::is_none")]
        previous_session_file: Option<String>,
    },
    #[serde(rename = "session_before_branch", rename_all = "camelCase")]
    SessionBeforeBranch {
        reason: BranchReason,
        entry_id: String,
    },
    #[serde(rename = "session_branch", rename_all = "camelCase")]
    SessionBranch {
        reason: BranchReason,
        #[serde(skip_serializing_if = "Option::is_none")]
        previous_session_file: Option<String>,
    },
    #[serde(rename = "session_before_tree")]
    SessionBeforeTree { preparation: Value },
    #[serde(rename = "session_tree", rename_all = "camelCase")]
    SessionTree {
        new_leaf_id: Option<String>,
        old_leaf_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        summary_entry: Option<Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        from_extension: Option<bool>,
    },
    /// Main-agent turn about to settle. Fields are `snake_case` in omp.
    #[serde(rename = "session_stop")]
    SessionStop {
        messages: Vec<Value>,
        turn_id: u64,
        #[serde(skip_serializing_if = "Option::is_none")]
        last_assistant_message: Option<Value>,
        session_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        session_file: Option<String>,
        stop_hook_active: bool,
    },
    /// Finalized assistant message (rewritable: see [`AssistantRewriteOutcome`]).
    #[serde(rename = "assistant_message")]
    AssistantMessage { message: Value },
    #[serde(rename = "auto_compaction_start")]
    AutoCompactionStart {
        reason: CompactionReason,
        action: CompactionAction,
    },
    #[serde(rename = "auto_compaction_end", rename_all = "camelCase")]
    AutoCompactionEnd {
        action: CompactionAction,
        #[serde(skip_serializing_if = "Option::is_none")]
        result: Option<Value>,
        aborted: bool,
        will_retry: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        error_message: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        skipped: Option<bool>,
    },
    #[serde(rename = "auto_retry_start", rename_all = "camelCase")]
    AutoRetryStart {
        attempt: u64,
        max_attempts: u64,
        delay_ms: u64,
        error_message: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        error_id: Option<u64>,
    },
    #[serde(rename = "auto_retry_end", rename_all = "camelCase")]
    AutoRetryEnd {
        success: bool,
        attempt: u64,
        #[serde(skip_serializing_if = "Option::is_none")]
        final_error: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        retry_errors: Option<Vec<Value>>,
    },
    #[serde(rename = "retry_fallback_applied")]
    RetryFallbackApplied {
        from: String,
        to: String,
        role: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
    },
    #[serde(rename = "retry_fallback_succeeded")]
    RetryFallbackSucceeded { model: String, role: String },
    #[serde(rename = "ttsr_triggered")]
    TtsrTriggered { rules: Vec<Value> },
    #[serde(rename = "todo_reminder", rename_all = "camelCase")]
    TodoReminder {
        todos: Vec<Value>,
        attempt: u64,
        max_attempts: u64,
    },
    /// `goal` is `null` when the goal was cleared.
    #[serde(rename = "goal_updated")]
    GoalUpdated {
        goal: Option<Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        state: Option<Value>,
    },
    /// Field order follows `wrapper.ts` (sessionId, toolName, toolCallId, ...).
    #[serde(rename = "tool_approval_requested", rename_all = "camelCase")]
    ToolApprovalRequested {
        session_id: String,
        tool_name: String,
        tool_call_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
        approval_mode: ApprovalMode,
    },
    #[serde(rename = "tool_approval_resolved", rename_all = "camelCase")]
    ToolApprovalResolved {
        session_id: String,
        tool_name: String,
        tool_call_id: String,
        approved: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
    },
    /// `!`/`!!` shell command typed by the user.
    #[serde(rename = "user_bash", rename_all = "camelCase")]
    UserBash {
        command: String,
        exclude_from_context: bool,
        cwd: String,
    },
    /// `$`/`$$` Python typed by the user.
    #[serde(rename = "user_python", rename_all = "camelCase")]
    UserPython {
        code: String,
        exclude_from_context: bool,
        cwd: String,
    },
    /// Process exit (the session-ending event).
    #[serde(rename = "session_shutdown")]
    SessionShutdown,
}

impl LifecycleEvent {
    /// The wire form handed to [`ExtHost::emit`].
    ///
    /// # Panics
    /// Never in practice: every field serializes to JSON.
    #[must_use]
    pub fn to_value(&self) -> Value {
        serde_json::to_value(self).expect("lifecycle events are plain JSON")
    }
}

/// Outcome of `before_agent_start`: custom messages to inject and an optional replaced system prompt.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BeforeAgentStartOutcome {
    pub messages: Vec<Value>,
    pub system_prompt: Option<Vec<String>>,
}

/// Outcome of `input`: `handled` stops normal flow; `text`/`images` replace the submitted input.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct InputOutcome {
    pub handled: bool,
    pub text: Option<String>,
    pub images: Option<Value>,
}

/// Outcome of a cancellable `session_before_*` event; `result` is the last handler result
/// (e.g. `{compaction}` for `session_before_compact`, `{summary}` for `session_before_tree`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CancelOutcome {
    pub cancel: bool,
    pub result: Value,
}

/// Outcome of `session_stop`: a handler may request one continuation turn.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SessionStopOutcome {
    /// Claude/Codex-style `decision: "block"`.
    pub block: bool,
    /// omp-native `continue: true`.
    pub continue_turn: bool,
    pub additional_context: Option<String>,
    pub reason: Option<String>,
}

impl SessionStopOutcome {
    /// Whether the session must run one more turn.
    #[must_use]
    pub fn wants_continuation(&self) -> bool {
        self.block || self.continue_turn
    }
}

fn str_of(v: &Value, key: &str) -> Option<String> {
    v.get(key).and_then(Value::as_str).map(str::to_owned)
}

impl BeforeAgentStartOutcome {
    fn from_value(v: &Value) -> Self {
        Self {
            messages: v
                .get("messages")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default(),
            system_prompt: v.get("systemPrompt").and_then(Value::as_array).map(|a| {
                a.iter()
                    .filter_map(Value::as_str)
                    .map(str::to_owned)
                    .collect()
            }),
        }
    }
}

impl InputOutcome {
    fn from_value(v: &Value) -> Self {
        Self {
            handled: v.get("handled").and_then(Value::as_bool).unwrap_or(false),
            text: str_of(v, "text"),
            images: v.get("images").cloned(),
        }
    }
}

impl CancelOutcome {
    fn from_result(r: EmitResult) -> Self {
        Self {
            cancel: r.result.get("cancel").and_then(Value::as_bool) == Some(true),
            result: r.result,
        }
    }
}

impl SessionStopOutcome {
    fn from_value(v: &Value) -> Self {
        Self {
            block: v.get("decision").and_then(Value::as_str) == Some("block"),
            continue_turn: v.get("continue").and_then(Value::as_bool) == Some(true),
            additional_context: str_of(v, "additionalContext"),
            reason: str_of(v, "reason"),
        }
    }
}

/// Outcome of `assistant_message`: replacement `content` (text-only rewrites, chained) when a handler changed it.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AssistantRewriteOutcome {
    pub content: Option<Vec<Value>>,
}

/// Outcome of `user_bash` / `user_python`: the first handler result (`BashResult`/`PythonResult`), if any.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct UserShellOutcome {
    pub result: Option<Value>,
}

impl AssistantRewriteOutcome {
    fn from_value(v: &Value) -> Self {
        Self {
            content: v.get("content").and_then(Value::as_array).cloned(),
        }
    }
}

impl UserShellOutcome {
    fn from_value(v: &Value) -> Self {
        Self {
            result: v.get("result").cloned(),
        }
    }
}

macro_rules! notify_helpers {
    ($($(#[$m:meta])* $name:ident($($arg:ident: $ty:ty),*) => $ev:expr;)*) => {
        $(
            $(#[$m])*
            ///
            /// # Errors
            /// Transport failure; handler failures are reported in the returned [`EmitResult`].
            pub async fn $name(&self, $($arg: $ty,)* ctx: Option<Value>) -> Result<EmitResult> {
                self.emit_event(&$ev, ctx).await
            }
        )*
    };
}

impl ExtHost {
    /// Emit any [`LifecycleEvent`].
    ///
    /// # Errors
    /// Transport failure; handler failures are reported in the returned [`EmitResult`].
    pub async fn emit_event(
        &self,
        event: &LifecycleEvent,
        ctx: Option<Value>,
    ) -> Result<EmitResult> {
        self.emit(event.to_value(), ctx).await
    }

    notify_helpers! {
        /// `session_start`: initial session load.
        emit_session_start() => LifecycleEvent::SessionStart;
        /// `agent_start`: an agent loop starts (once per user prompt).
        emit_agent_start() => LifecycleEvent::AgentStart;
        /// `agent_end`: an agent loop ended; `will_continue` when a continuation is already scheduled.
        emit_agent_end(messages: Vec<Value>, will_continue: Option<bool>) => LifecycleEvent::AgentEnd { messages, will_continue };
        /// `turn_start`.
        emit_turn_start(turn_index: u64, timestamp: u64) => LifecycleEvent::TurnStart { turn_index, timestamp };
        /// `turn_end`.
        emit_turn_end(turn_index: u64, message: Value, tool_results: Vec<Value>) => LifecycleEvent::TurnEnd { turn_index, message, tool_results };
        /// `message_start`.
        emit_message_start(message: Value) => LifecycleEvent::MessageStart { message };
        /// `message_update`.
        emit_message_update(message: Value, assistant_message_event: Value) => LifecycleEvent::MessageUpdate { message, assistant_message_event };
        /// `message_end` (notification only).
        emit_message_end(message: Value) => LifecycleEvent::MessageEnd { message };
        /// `tool_execution_start`.
        emit_tool_execution_start(tool_call_id: String, tool_name: String, args: Value, intent: Option<String>) => LifecycleEvent::ToolExecutionStart { tool_call_id, tool_name, args, intent };
        /// `tool_execution_update`.
        emit_tool_execution_update(tool_call_id: String, tool_name: String, args: Value, partial_result: Value) => LifecycleEvent::ToolExecutionUpdate { tool_call_id, tool_name, args, partial_result };
        /// `tool_execution_end`.
        emit_tool_execution_end(tool_call_id: String, tool_name: String, result: Value, is_error: bool) => LifecycleEvent::ToolExecutionEnd { tool_call_id, tool_name, result, is_error };
        /// `session_compact`: compaction finished.
        emit_session_compact(compaction_entry: Value, from_extension: bool) => LifecycleEvent::SessionCompact { compaction_entry, from_extension };
        /// `session_switch`: after switching to another session.
        emit_session_switch(reason: SwitchReason, previous_session_file: Option<String>) => LifecycleEvent::SessionSwitch { reason, previous_session_file };
        /// `session_branch`: after branching.
        emit_session_branch(reason: BranchReason, previous_session_file: Option<String>) => LifecycleEvent::SessionBranch { reason, previous_session_file };
        /// `session_tree`: after tree navigation.
        emit_session_tree(new_leaf_id: Option<String>, old_leaf_id: Option<String>, summary_entry: Option<Value>, from_extension: Option<bool>) => LifecycleEvent::SessionTree { new_leaf_id, old_leaf_id, summary_entry, from_extension };
        /// `session_shutdown`: process exit (handlers run in parallel, 2 s cap in the host).
        emit_session_shutdown() => LifecycleEvent::SessionShutdown;
        /// `auto_compaction_start`.
        emit_auto_compaction_start(reason: CompactionReason, action: CompactionAction) => LifecycleEvent::AutoCompactionStart { reason, action };
        /// `auto_compaction_end`.
        emit_auto_compaction_end(action: CompactionAction, result: Option<Value>, aborted: bool, will_retry: bool, error_message: Option<String>, skipped: Option<bool>) => LifecycleEvent::AutoCompactionEnd { action, result, aborted, will_retry, error_message, skipped };
        /// `auto_retry_start`.
        emit_auto_retry_start(attempt: u64, max_attempts: u64, delay_ms: u64, error_message: String, error_id: Option<u64>) => LifecycleEvent::AutoRetryStart { attempt, max_attempts, delay_ms, error_message, error_id };
        /// `auto_retry_end`.
        emit_auto_retry_end(success: bool, attempt: u64, final_error: Option<String>, retry_errors: Option<Vec<Value>>) => LifecycleEvent::AutoRetryEnd { success, attempt, final_error, retry_errors };
        /// `retry_fallback_applied`.
        emit_retry_fallback_applied(from: String, to: String, role: String, reason: Option<String>) => LifecycleEvent::RetryFallbackApplied { from, to, role, reason };
        /// `retry_fallback_succeeded`.
        emit_retry_fallback_succeeded(model: String, role: String) => LifecycleEvent::RetryFallbackSucceeded { model, role };
        /// `ttsr_triggered`.
        emit_ttsr_triggered(rules: Vec<Value>) => LifecycleEvent::TtsrTriggered { rules };
        /// `todo_reminder`.
        emit_todo_reminder(todos: Vec<Value>, attempt: u64, max_attempts: u64) => LifecycleEvent::TodoReminder { todos, attempt, max_attempts };
        /// `goal_updated`.
        emit_goal_updated(goal: Option<Value>, state: Option<Value>) => LifecycleEvent::GoalUpdated { goal, state };
        /// `tool_approval_requested`.
        emit_tool_approval_requested(session_id: String, tool_name: String, tool_call_id: String, reason: Option<String>, approval_mode: ApprovalMode) => LifecycleEvent::ToolApprovalRequested { session_id, tool_name, tool_call_id, reason, approval_mode };
        /// `tool_approval_resolved`.
        emit_tool_approval_resolved(session_id: String, tool_name: String, tool_call_id: String, approved: bool, reason: Option<String>) => LifecycleEvent::ToolApprovalResolved { session_id, tool_name, tool_call_id, approved, reason };
    }

    /// `assistant_message`: a handler may rewrite the text of the finalized message (text blocks only).
    ///
    /// # Errors
    /// Transport failure.
    pub async fn emit_assistant_message(
        &self,
        message: Value,
        ctx: Option<Value>,
    ) -> Result<AssistantRewriteOutcome> {
        let ev = LifecycleEvent::AssistantMessage { message };
        Ok(AssistantRewriteOutcome::from_value(
            &self.emit_event(&ev, ctx).await?.result,
        ))
    }

    /// `user_bash`: the first handler result replaces execution.
    ///
    /// # Errors
    /// Transport failure.
    pub async fn emit_user_bash(
        &self,
        command: String,
        exclude_from_context: bool,
        cwd: String,
        ctx: Option<Value>,
    ) -> Result<UserShellOutcome> {
        let ev = LifecycleEvent::UserBash {
            command,
            exclude_from_context,
            cwd,
        };
        Ok(UserShellOutcome::from_value(
            &self.emit_event(&ev, ctx).await?.result,
        ))
    }

    /// `user_python`: the first handler result replaces execution.
    ///
    /// # Errors
    /// Transport failure.
    pub async fn emit_user_python(
        &self,
        code: String,
        exclude_from_context: bool,
        cwd: String,
        ctx: Option<Value>,
    ) -> Result<UserShellOutcome> {
        let ev = LifecycleEvent::UserPython {
            code,
            exclude_from_context,
            cwd,
        };
        Ok(UserShellOutcome::from_value(
            &self.emit_event(&ev, ctx).await?.result,
        ))
    }

    /// `before_agent_start`: returns the messages to inject and the (possibly replaced) system prompt.
    ///
    /// # Errors
    /// Transport failure.
    pub async fn emit_before_agent_start(
        &self,
        prompt: String,
        images: Option<Value>,
        system_prompt: Vec<String>,
        ctx: Option<Value>,
    ) -> Result<BeforeAgentStartOutcome> {
        let ev = LifecycleEvent::BeforeAgentStart {
            prompt,
            images,
            system_prompt,
        };
        Ok(BeforeAgentStartOutcome::from_value(
            &self.emit_event(&ev, ctx).await?.result,
        ))
    }

    /// `input`: returns the transformed text/images or `handled`.
    ///
    /// # Errors
    /// Transport failure.
    pub async fn emit_input(
        &self,
        text: String,
        images: Option<Value>,
        source: InputSource,
        ctx: Option<Value>,
    ) -> Result<InputOutcome> {
        let ev = LifecycleEvent::Input {
            text,
            images,
            source,
        };
        Ok(InputOutcome::from_value(
            &self.emit_event(&ev, ctx).await?.result,
        ))
    }

    /// `session_before_compact`: cancellable; `result["compaction"]` may carry an extension-made compaction.
    ///
    /// # Errors
    /// Transport failure.
    pub async fn emit_session_before_compact(
        &self,
        preparation: Value,
        branch_entries: Vec<Value>,
        custom_instructions: Option<String>,
        ctx: Option<Value>,
    ) -> Result<CancelOutcome> {
        let ev = LifecycleEvent::SessionBeforeCompact {
            preparation,
            branch_entries,
            custom_instructions,
        };
        Ok(CancelOutcome::from_result(self.emit_event(&ev, ctx).await?))
    }

    /// `session_before_switch`: cancellable.
    ///
    /// # Errors
    /// Transport failure.
    pub async fn emit_session_before_switch(
        &self,
        reason: SwitchReason,
        target_session_file: Option<String>,
        ctx: Option<Value>,
    ) -> Result<CancelOutcome> {
        let ev = LifecycleEvent::SessionBeforeSwitch {
            reason,
            target_session_file,
        };
        Ok(CancelOutcome::from_result(self.emit_event(&ev, ctx).await?))
    }

    /// `session_before_branch`: cancellable.
    ///
    /// # Errors
    /// Transport failure.
    pub async fn emit_session_before_branch(
        &self,
        reason: BranchReason,
        entry_id: String,
        ctx: Option<Value>,
    ) -> Result<CancelOutcome> {
        let ev = LifecycleEvent::SessionBeforeBranch { reason, entry_id };
        Ok(CancelOutcome::from_result(self.emit_event(&ev, ctx).await?))
    }

    /// `session_before_tree`: cancellable; `result["summary"]` may carry a custom summary.
    ///
    /// # Errors
    /// Transport failure.
    pub async fn emit_session_before_tree(
        &self,
        preparation: Value,
        ctx: Option<Value>,
    ) -> Result<CancelOutcome> {
        let ev = LifecycleEvent::SessionBeforeTree { preparation };
        Ok(CancelOutcome::from_result(self.emit_event(&ev, ctx).await?))
    }

    /// `session_stop`: a handler may request one continuation turn.
    ///
    /// # Errors
    /// Transport failure.
    pub async fn emit_session_stop(
        &self,
        event: LifecycleEvent,
        ctx: Option<Value>,
    ) -> Result<SessionStopOutcome> {
        anyhow::ensure!(
            matches!(event, LifecycleEvent::SessionStop { .. }),
            "emit_session_stop needs LifecycleEvent::SessionStop"
        );
        Ok(SessionStopOutcome::from_value(
            &self.emit_event(&event, ctx).await?.result,
        ))
    }
}

/// Boxed future used by [`HostResponder`] so it stays object safe.
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Answers the host's requests that need the core. Defaults are omp's `hasUI=false` behaviour
/// (no-op UI context and `ExtensionRunner` default actions).
pub trait HostResponder: Send + Sync {
    /// `ctx.ui.confirm(title, message)`; no UI declines.
    fn ui_confirm<'a>(&'a self, _title: &'a str, _message: &'a str) -> BoxFuture<'a, bool> {
        Box::pin(async { false })
    }
    /// `ctx.ui.select(title, options)`; no UI yields `None`.
    fn ui_select<'a>(
        &'a self,
        _title: &'a str,
        _options: Vec<String>,
    ) -> BoxFuture<'a, Option<String>> {
        Box::pin(async { None })
    }
    /// `ctx.ui.input(title, placeholder)`; no UI yields `None`.
    fn ui_input<'a>(
        &'a self,
        _title: &'a str,
        _placeholder: Option<&'a str>,
    ) -> BoxFuture<'a, Option<String>> {
        Box::pin(async { None })
    }
    /// `ctx.ui.editor(title, prefill)`; no UI yields `None`.
    fn ui_editor<'a>(
        &'a self,
        _title: &'a str,
        _prefill: Option<&'a str>,
    ) -> BoxFuture<'a, Option<String>> {
        Box::pin(async { None })
    }
    /// `ctx.compact(options)`; the default runner action is a no-op.
    fn compact(&self, _options: Value) -> BoxFuture<'_, Result<(), String>> {
        Box::pin(async { Ok(()) })
    }
    /// `ctx.navigateTree(target, options)` -> `{cancelled}`; the default runner action reports `{cancelled:false}`.
    fn navigate_tree<'a>(
        &'a self,
        _target: &'a str,
        _options: Value,
    ) -> BoxFuture<'a, Result<Value, String>> {
        Box::pin(async { Ok(json!({"cancelled": false})) })
    }
    /// Complete one auxiliary model request through the core's catalog and auth.
    fn model_complete(&self, _request: Value) -> BoxFuture<'_, Result<Value, String>> {
        Box::pin(async move { Err("model_complete is not supported by the core".into()) })
    }
    /// `pi.setModel(model)`; `false` = model not applied (no UI/credentials to switch with).
    fn set_model(&self, _model: Value) -> BoxFuture<'_, bool> {
        Box::pin(async { false })
    }
    /// `pi.setActiveTools(names)`.
    fn set_active_tools(&self, _names: Vec<String>) -> BoxFuture<'_, Result<(), String>> {
        Box::pin(async { Ok(()) })
    }
    /// `pi.setSessionName(name)`.
    fn set_session_name(&self, _name: String) -> BoxFuture<'_, Result<(), String>> {
        Box::pin(async { Ok(()) })
    }
}

/// `hasUI=false` answers only.
#[derive(Debug, Clone, Copy, Default)]
pub struct HeadlessResponder;

impl HostResponder for HeadlessResponder {}

/// Compute the reply for one host request.
///
/// # Errors
/// Unknown methods and responder failures become the JSON-RPC error message.
pub async fn answer(
    responder: &dyn HostResponder,
    method: &str,
    params: &Value,
) -> Result<Value, String> {
    let s = |k: &str| params.get(k).and_then(Value::as_str).unwrap_or_default();
    match method {
        "ui_confirm" => Ok(json!(responder.ui_confirm(s("title"), s("message")).await)),
        "ui_select" => {
            let options = params
                .get("options")
                .and_then(Value::as_array)
                .map(|a| {
                    a.iter()
                        .filter_map(Value::as_str)
                        .map(str::to_owned)
                        .collect()
                })
                .unwrap_or_default();
            Ok(responder
                .ui_select(s("title"), options)
                .await
                .map_or(Value::Null, Value::String))
        }
        "ui_input" => Ok(responder
            .ui_input(
                s("title"),
                params.get("placeholder").and_then(Value::as_str),
            )
            .await
            .map_or(Value::Null, Value::String)),
        "ui_editor" => Ok(responder
            .ui_editor(s("title"), params.get("prefill").and_then(Value::as_str))
            .await
            .map_or(Value::Null, Value::String)),
        "compact" => responder
            .compact(params.get("options").cloned().unwrap_or(Value::Null))
            .await
            .map(|()| Value::Null),
        "navigate_tree" => {
            responder
                .navigate_tree(
                    s("target"),
                    params.get("options").cloned().unwrap_or(Value::Null),
                )
                .await
        }
        "set_model" => Ok(json!(
            responder
                .set_model(params.get("model").cloned().unwrap_or(Value::Null))
                .await
        )),
        "set_active_tools" => {
            let names = params
                .get("names")
                .and_then(Value::as_array)
                .map(|a| {
                    a.iter()
                        .filter_map(Value::as_str)
                        .map(str::to_owned)
                        .collect()
                })
                .unwrap_or_default();
            responder
                .set_active_tools(names)
                .await
                .map(|()| Value::Null)
        }
        "set_session_name" => responder
            .set_session_name(s("name").to_owned())
            .await
            .map(|()| Value::Null),
        "model_complete" => responder.model_complete(params.clone()).await,
        other => Err(format!("{other} is not supported by the core")),
    }
}

/// Serve the host's requests with `responder` (each in its own task, so a slow `compact` does not stall
/// the rest) and hand every notification (`append_entry`, `send_message`, ...) to the returned channel.
/// Must run inside a tokio runtime; ends when the host closes.
///
/// `serve` only holds a weak handle: dropping the last [`ExtHost`] handle kills
/// the child process (`kill_on_drop`), after which the reader task closes the
/// message channel and this loop ends. A strong handle here would keep the host
/// (and its bun process) alive forever once the loop is running.
#[must_use]
pub fn serve(
    host: &Arc<ExtHost>,
    mut incoming: mpsc::UnboundedReceiver<HostMessage>,
    responder: Arc<dyn HostResponder>,
) -> mpsc::UnboundedReceiver<HostMessage> {
    let (tx, rx) = mpsc::unbounded_channel();
    let weak = Arc::downgrade(host);
    tokio::spawn(async move {
        while let Some(msg) = incoming.recv().await {
            match msg {
                HostMessage::Request { id, method, params } => {
                    let Some(host) = weak.upgrade() else {
                        break;
                    };
                    let responder = Arc::clone(&responder);
                    tokio::spawn(async move {
                        let reply = answer(responder.as_ref(), &method, &params).await;
                        let _ = host.respond(id, reply).await;
                    });
                }
                note @ HostMessage::Notification { .. } => {
                    if tx.send(note).is_err() {
                        break;
                    }
                }
            }
        }
    });
    rx
}
