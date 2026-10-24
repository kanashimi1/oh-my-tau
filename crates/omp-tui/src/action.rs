//! User intents produced by widgets and consumed by the embedding binary.
//!
//! The embedding crate (omp-cli `src/tui.rs`) implements [`crate::RuntimePort`]
//! and turns each [`Action`] into a `Runtime` call (`prompt`, `abort`,
//! `set_model`, ...). Actions are deliberately transport-shaped, not
//! UI-shaped: the chat editor, overlays and status line all emit the same
//! enum so keybindings can be remapped freely (see `keys.rs` / the
//! keybindings loader stub).

/// How a submitted prompt should behave when the agent is already streaming.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubmitBehavior {
    /// `Runtime::prompt(..., Some("steer"), ...)` — steer the live run.
    Steer,
    /// `Runtime::prompt(..., Some("followUp"), ...)` — queue for the next turn.
    FollowUp,
}

/// One user intent. Variants map 1:1 to `Runtime` methods in omp-cli.
#[derive(Debug, Clone, PartialEq)]
pub enum Action {
    /// Submit the editor content as a new prompt (or queue it per `behavior`).
    Submit {
        /// Optional caller-supplied id echoed back in `prompt_result`.
        id: Option<String>,
        /// The prompt text (may include newlines and `@file` references).
        text: String,
        /// `Some` when the agent is streaming.
        behavior: Option<SubmitBehavior>,
    },
    /// Steer a message into the live run (`Runtime::steer`).
    Steer { text: String },
    /// Queue a follow-up message (`Runtime::follow_up`).
    FollowUp { text: String },
    /// Remove a queued message (`Runtime::remove_queued`).
    RemoveQueued { text: String, queue: QueueKind },
    /// Promote a queued message to run next (`Runtime::promote_queued`).
    PromoteQueued { text: String },
    /// Abort the live run (`Runtime::abort`).
    Abort,
    /// Switch the active model (`Runtime::set_model`).
    SetModel { provider: String, id: String },
    /// Switch the thinking level (`Runtime::set_thinking_level`).
    SetThinkingLevel(String),
    /// Run a slash command (`Runtime`-side dispatch, TS `slash` module).
    Slash { name: String, args: String },
    /// Live preview of a highlighted `/settings` choice (not persisted).
    SettingPreview {
        /// Setting id (`theme.dark` / `theme.light`).
        id: String,
        /// Previewed value.
        value: String,
    },
    /// Answer an extension `ask`-dialog (`extension_ui_response` frame).
    ExtensionUiResponse {
        id: String,
        value: serde_json::Value,
    },
    /// Resume a session from the picker (`Runtime::load_session`), or fork it
    /// (`tree::fork`) when `fork` is set.
    ResumeSession {
        /// Session journal file to load.
        path: std::path::PathBuf,
        /// Fork into a new session id instead of resuming in place.
        fork: bool,
    },
    /// Synthetic "continue" directive (TS `prompts/system/manual-continue.md`,
    /// sent as a user-initiated synthetic prompt).
    Continue,
    /// Rewind the session tree to just before the given entry (double-Esc /
    /// tree navigator; user-authored targets re-prompt the message, TS
    /// `AgentSession.navigateTree`).
    Rewind {
        /// Target journal entry id.
        entry_id: String,
    },
    /// A `/settings` overlay edit (TS `cfg://` set): persist via the port
    /// and apply live where the TUI can (theme, thinking level).
    SettingChanged {
        /// Registered setting id (e.g. `theme.dark`).
        id: String,
        /// New value, already type-checked against the registry.
        value: serde_json::Value,
    },
    /// Scroll the chat backlog.
    Scroll(ScrollDir),
    /// Refresh the online model catalog (`F5` in the /model browser;
    /// TS `ModelRegistry.refresh` → `models.db` update).
    RefreshModels,
    /// Leave the TUI (Ctrl+C on the idle editor / `:q` future).
    Quit,
}

/// Which runtime queue a queued message lives in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueKind {
    /// Steering queue.
    Steering,
    /// Follow-up queue.
    FollowUp,
}

/// Backlog scroll direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScrollDir {
    /// One line up.
    Up,
    /// One line down.
    Down,
    /// One page up.
    PageUp,
    /// One page down.
    PageDown,
    /// Snap to the newest output.
    Bottom,
}

/// Convenience constructor for a plain submit.
#[must_use]
pub fn submit(text: String) -> Action {
    Action::Submit {
        id: None,
        text,
        behavior: None,
    }
}
