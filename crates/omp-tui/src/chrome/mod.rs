//! Terminal chrome: title states, notifications, progress HUD, header banner
//! and the startup prepaint cache (port of the pi-tui chrome).
//!
//! Everything here renders into plain [`ratatui`] buffers / writes terminal
//! control sequences directly and must not assume the alternate screen.

pub mod banner;
pub mod hud;
pub mod mascot;
pub mod notify;
pub mod osc11;
pub mod startup;
pub mod title;
pub mod welcome;

pub use banner::Banner;
pub use hud::{HudEntry, HudProgress};
pub use notify::{
    format_notification, notification_to_line, send_notification, NotifyKind, NotifyProtocol,
    TerminalNotification, Urgency,
};
pub use osc11::{parse_osc11_response, probe_osc11_background};
pub use startup::StartupCache;
pub use title::{TerminalTitle, TitleSpinner, TitleState};
