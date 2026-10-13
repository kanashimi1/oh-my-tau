//! Vim modal editing for the prompt editor — the port of `pi-tui/src/vim.ts`.
//!
//! Stub. Planned surface: a [`VimState`] machine (normal/insert/visual) that
//! wraps the base [`super::Editor`] buffer operations and translates key
//! events into motions/verbs; the `Editor` calls into this when vim mode is
//! enabled by the keybindings config.

/// Vim mode state. Placeholder until the vim implementer lands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VimState {
    /// Not yet implemented — behaves as plain insert mode.
    #[default]
    Disabled,
    /// Normal mode (future).
    Normal,
    /// Insert mode (future).
    Insert,
    /// Visual mode (future).
    Visual,
}
