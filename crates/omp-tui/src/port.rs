//! The boundary between this crate and the omp-cli `Runtime`.
//!
//! [`RuntimePort`] is the only thing `omp-tui` knows about the agent. The
//! embedding binary (omp-cli `src/tui.rs`) implements it with an
//! `Arc<Runtime>`: async `Runtime` methods are spawned onto the tokio
//! runtime from these synchronous hooks, so this crate stays executor-agnostic
//! and testable with a recording fake.
//!
//! TS reference: `pi-coding-agent/src/modes/interactive-mode.ts` — the
//! interactive controller calls the same `AgentSession` methods
//! (`prompt`, `steer`, `followUp`, `abort`, `setModel`, slash dispatch).

use serde_json::Value;

use crate::action::SubmitBehavior;

/// Synchronous bridge to the agent runtime.
pub trait RuntimePort: Send + Sync + 'static {
    /// Submit a prompt. `behavior` is `Some` when the agent is streaming
    /// (`Runtime::prompt(id, text, behavior, images)`).
    fn submit(&self, id: Option<String>, text: String, behavior: Option<SubmitBehavior>);

    /// Steer a message into the live run (`Runtime::steer`).
    fn steer(&self, text: String);

    /// Queue a follow-up (`Runtime::follow_up`).
    fn follow_up(&self, text: String);

    /// Remove a queued message (`Runtime::remove_queued`).
    fn remove_queued(&self, text: String, queue: crate::action::QueueKind);

    /// Promote a queued message (`Runtime::promote_queued`).
    fn promote_queued(&self, text: String);

    /// Abort the live run (`Runtime::abort`).
    fn abort(&self);

    /// Switch model (`Runtime::set_model`).
    fn set_model(&self, provider: String, id: String);

    /// Switch thinking level by name (`Runtime::set_thinking_level`).
    fn set_thinking_level(&self, level: String);

    /// Run a slash command by name with the raw argument string.
    fn run_slash(&self, name: String, args: String);

    /// Answer an extension dialog (`extension_ui_response` frame).
    fn extension_ui_response(&self, id: String, value: Value);
}

/// A recording fake for tests. Every call appends to [`Self::log`].
#[derive(Debug, Default)]
pub struct FakePort {
    /// Recorded calls, one debug string each.
    pub log: parking_lot::Mutex<Vec<String>>,
}

impl FakePort {
    /// Record one call.
    fn record(&self, call: String) {
        self.log.lock().push(call);
    }
}

impl RuntimePort for FakePort {
    fn submit(&self, id: Option<String>, text: String, behavior: Option<SubmitBehavior>) {
        self.record(format!("submit({id:?}, {text:?}, {behavior:?})"));
    }
    fn steer(&self, text: String) {
        self.record(format!("steer({text:?})"));
    }
    fn follow_up(&self, text: String) {
        self.record(format!("follow_up({text:?})"));
    }
    fn remove_queued(&self, text: String, queue: crate::action::QueueKind) {
        self.record(format!("remove_queued({text:?}, {queue:?})"));
    }
    fn promote_queued(&self, text: String) {
        self.record(format!("promote_queued({text:?})"));
    }
    fn abort(&self) {
        self.record("abort()".into());
    }
    fn set_model(&self, provider: String, id: String) {
        self.record(format!("set_model({provider}, {id})"));
    }
    fn set_thinking_level(&self, level: String) {
        self.record(format!("set_thinking_level({level})"));
    }
    fn run_slash(&self, name: String, args: String) {
        self.record(format!("run_slash({name}, {args:?})"));
    }
    fn extension_ui_response(&self, id: String, value: Value) {
        self.record(format!("extension_ui_response({id}, {value})"));
    }
}
