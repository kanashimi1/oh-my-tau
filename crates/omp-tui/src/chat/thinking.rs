//! Thinking/reasoning block rendering — the port of `pi-tui/src/thinking.ts`.
//!
//! Stub. Planned: collapsible reasoning blocks with the theme's muted
//! token, a thinking-level indicator (TS shows the level when non-default),
//! streaming-delta support. The state lives in the chat entry kinds once
//! those land.

/// Whether thinking blocks start expanded. Mirrors TS: expanded when the
/// session sets a non-minimal thinking level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ThinkingDisplay {
    /// Collapsed to a one-line header.
    #[default]
    Collapsed,
    /// Fully visible.
    Expanded,
}

impl ThinkingDisplay {
    /// Toggle.
    pub fn toggle(&mut self) {
        *self = match self {
            Self::Collapsed => Self::Expanded,
            Self::Expanded => Self::Collapsed,
        };
    }
}
