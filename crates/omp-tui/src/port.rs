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

/// The synthetic "continue" directive text, verbatim from TS
/// `prompts/system/manual-continue.md` (sent with `synthetic: true,
/// userInitiated: true` by the `.`/`c` shortcut).
pub const MANUAL_CONTINUE_PROMPT: &str = "\
Continue.

MUST resume most recent intent; complete unfinished work.
If interrupted mid-step: resume where stopped.
NEVER pause to summarize progress, re-confirm plan, or ask whether to proceed; continue.
";

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

    /// Resume a session journal (`Runtime::load_session`) or fork it
    /// (`tree::fork`) when `fork` is set.
    fn resume_session(&self, path: std::path::PathBuf, fork: bool);

    /// Send the synthetic continue directive ([`MANUAL_CONTINUE_PROMPT`]).
    fn continue_run(&self);

    /// Cancel a pending retry backoff (`Runtime::abort_retry`).
    fn abort_retry(&self);

    /// Rewind the session tree to just before `entry_id`
    /// (`tree::navigate_tree`); user-authored targets re-prompt the message.
    fn rewind_entry(&self, entry_id: String);

    /// Cancel a running auto-compaction. No-op in omp-rs until the runtime
    /// grows a compaction-abort RPC; the method exists so app wiring is
    /// stable.
    fn abort_compaction(&self);

    /// Refresh the online model catalog (TS `ModelRegistry.refresh`):
    /// per-provider `/models` discovery → `models.db` update. Errors are
    /// surfaced as system notices by the embedding port.
    fn refresh_models(&self);

    /// Effective values for `ids` (TS `cfg://` batch get: defaults < global
    /// config.yml < project config < env). Used by the `/settings` overlay
    /// to seed its rows; ids not found come back as `Null`.
    fn settings_get(&self, ids: &[String]) -> Vec<(String, Value)>;

    /// Persist `value` for `id` to the global config.yml (TS `cfg://` set →
    /// `saveConfigFile`). An error string is surfaced as a system notice.
    fn settings_set(&self, id: String, value: Value) -> Result<(), String>;

    /// Efforts the live model supports, least to most intensive and without
    /// `off` (TS `getAvailableThinkingLevels`); empty when the model cannot
    /// think. Drives the Shift+Tab cycle and the thinking selectors.
    fn available_thinking_levels(&self) -> Vec<String> {
        ["minimal", "low", "medium", "high", "xhigh"]
            .map(String::from)
            .to_vec()
    }

    /// Gather the `/session` info report; the runtime answers with a
    /// `{"type":"session_info","lines":[…]}` frame that opens the overlay.
    fn request_session_info(&self) {}

    /// Gather the `/usage` report (session tokens + provider rate limits);
    /// the runtime answers with a `session_info` frame titled "Usage".
    fn request_usage(&self) {}
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
    fn resume_session(&self, path: std::path::PathBuf, fork: bool) {
        self.record(format!("resume_session({path:?}, fork={fork})"));
    }
    fn continue_run(&self) {
        self.record("continue_run()".into());
    }
    fn abort_retry(&self) {
        self.record("abort_retry()".into());
    }
    fn abort_compaction(&self) {
        self.record("abort_compaction()".into());
    }
    fn refresh_models(&self) {
        self.record("refresh_models()".into());
    }
    fn rewind_entry(&self, entry_id: String) {
        self.record(format!("rewind_entry({entry_id})"));
    }
    fn settings_get(&self, ids: &[String]) -> Vec<(String, Value)> {
        self.record(format!("settings_get({ids:?})"));
        // Resolve the global config.yml from HOME like the real port (the
        // temp-HOME tests rely on this path).
        let settings = omp_settings::Settings::load(std::path::Path::new("/nonexistent"));
        ids.iter()
            .map(|id| (id.clone(), settings.get(id)))
            .collect()
    }
    fn settings_set(&self, id: String, value: Value) -> Result<(), String> {
        self.record(format!("settings_set({id}, {value})"));
        omp_settings::write_global(&id, &value)
            .map(|_| ())
            .map_err(|e| e.to_string())
    }
    fn request_session_info(&self) {
        self.record("request_session_info()".into());
    }
    fn request_usage(&self) {
        self.record("request_usage()".into());
    }
}
