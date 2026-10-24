//! `omp-tui`: the interactive terminal UI for omp-rs — a Rust reimplementation
//! of omp's `pi-tui` TypeScript TUI (a port of the pi-tui modules; keep
//! each module's behavior in line with its TS counterpart).
//!
//! # Architecture in one paragraph
//!
//! [`run`] owns the terminal (raw mode + alternate screen via crossterm) and a
//! ratatui draw loop. The agent session talks to the UI through two channels:
//! the omp-cli `Runtime` forwards its JSON event frames into
//! [`events::AppEvent`] values (via `RunConfig::frames`), and the UI sends
//! [`action::Action`] values back through a [`port::RuntimePort`] implemented
//! by the embedding crate (omp-cli). Every on-screen widget implements
//! [`component::Component`]; all colors/spacing come from [`theme::Theme`],
//! all key handling goes through [`keys::Keymap`].
//!
//! # Module ownership
//!
//! Each top-level module (and each `mod.rs` + its children) is owned by one
//! implementer. Do not edit files outside your module; extend shared
//! contracts (`component`, `theme`, `keys`, `events`, `action`, `port`) only
//! by agreement between module owners.

pub mod action;
pub mod app;
pub mod autocomplete;
pub mod bracketed_paste;
pub mod chat;
pub mod chrome;
pub mod clipboard;
pub mod component;
pub mod composer;
pub mod controller;
pub mod editor;
pub mod events;
pub mod extension_ui;
pub mod images;
pub mod keybindings;
pub mod keys;
pub mod magic;
pub mod magic_paint;
pub mod mouse;
pub mod overlays;
pub mod port;
pub mod render;
pub mod status_line;
pub mod theme;
pub mod tools;
pub mod word_completion;

/// Process-wide guards for tests that touch global state.
#[cfg(test)]
pub(crate) mod test_support {
    /// `HOME` is process-global: every test that sets it must hold this for
    /// its whole duration (one mutex shared by all modules).
    pub(crate) static HOME_LOCK: parking_lot::Mutex<()> = parking_lot::Mutex::new(());
}

use std::path::PathBuf;

use tokio::sync::mpsc::UnboundedReceiver;

pub use action::{Action, SubmitBehavior};
pub use app::App;
pub use events::AppEvent;
pub use port::{RuntimePort, MANUAL_CONTINUE_PROMPT};
pub use theme::Theme;

/// Everything [`run`] needs from the embedding binary (omp-cli).
pub struct RunConfig {
    /// Frames from the omp-cli `Runtime` event stream (`Runtime::start`'s
    /// `out` channel). One `serde_json::Value` per line-shaped frame; parsed
    /// by [`events::parse_frame`].
    pub frames: UnboundedReceiver<serde_json::Value>,
    /// Model reference shown in the status line until the first
    /// `model_changed` frame arrives (`provider/id`); also preselects the
    /// `/model` picker's current row.
    pub model: String,
    /// Display name of `model` (models.yml `name`), when the catalogue has one.
    pub model_name: Option<String>,
    /// Version shown under the startup logo.
    pub version: String,
    /// Working directory the session runs in (status line / chrome).
    pub cwd: PathBuf,
    /// Model catalogue for the `/model` picker (providers with credentials,
    /// omp-cli `runtime::available_models`).
    pub models: Vec<overlays::models::ModelEntry>,
    /// Session stores backing the `/resume` picker: the native omp-rs
    /// folder dir first, then TS omp's dir for the same folder (tagged).
    /// Empty → session storage unavailable.
    pub session_sources: Vec<overlays::sessions::SessionSource>,
    /// Terminal background from the startup OSC 11 probe (`None` →
    /// COLORFGBG / dark default inside [`theme::detect_appearance`]).
    pub osc11_bg: Option<theme::Rgb>,
}

/// Run the interactive TUI until the user quits (Ctrl+C / Ctrl+Q / `:q` in
/// the future vim mode) or the terminal disappears.
///
/// This installs raw mode + the alternate screen, pumps crossterm input
/// events and `Runtime` frames, dispatches [`Action`]s to `port`, and draws
/// [`App`] each tick. Terminal state is restored on every exit path,
/// including panics.
///
/// # Errors
///
/// Returns an error if the terminal cannot be initialized or a draw fails.
pub async fn run(cfg: RunConfig, port: impl RuntimePort) -> anyhow::Result<()> {
    controller::run(cfg, port).await
}
