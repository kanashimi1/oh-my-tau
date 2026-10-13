//! Key events, combos and the keymap — ports of `pi-tui/src/keys.ts`,
//! `keybindings.ts`, `keybinding-matchers.ts` and `app-keybindings.ts`.
//!
//! A [`KeyCombo`] is one physical chord (`ctrl+c`, `shift+tab`, `g` then
//! `g`-style sequences are future work). A [`Keymap`] binds combos to
//! [`Action`] names; the loader stub will read omp's
//! `~/.omp/agent/keybindings.yml` (same file the TS reads,
//! `keybindings.ts` `loadKeybindings`). Components match on
//! [`KeyEvent`] (re-exported crossterm type) and consult the keymap for
//! user-overridable chords; hardcoded chords (typing, Enter, Backspace) stay
//! in the editor.

use std::fmt;

pub use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use crate::action::Action;

/// One physical key chord.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeyCombo {
    /// The key.
    pub code: KeyCode,
    /// Active modifiers (CONTROL | ALT | SHIFT).
    pub modifiers: KeyModifiers,
}

impl KeyCombo {
    /// A plain unmodified key.
    #[must_use]
    pub fn plain(code: KeyCode) -> Self {
        Self {
            code,
            modifiers: KeyModifiers::NONE,
        }
    }

    /// A ctrl-modified key.
    #[must_use]
    pub fn ctrl(code: KeyCode) -> Self {
        Self {
            code,
            modifiers: KeyModifiers::CONTROL,
        }
    }

    /// Match a crossterm event against this combo, normalizing the
    /// shift-state of printable keys the way TS `matchesKey` does.
    #[must_use]
    pub fn matches(&self, event: &KeyEvent) -> bool {
        let mods = event.modifiers;
        if mods.contains(KeyModifiers::CONTROL) != self.modifiers.contains(KeyModifiers::CONTROL) {
            return false;
        }
        if mods.contains(KeyModifiers::ALT) != self.modifiers.contains(KeyModifiers::ALT) {
            return false;
        }
        // Shift is implied by uppercase char codes; compare char-invariant.
        match (self.code, event.code) {
            (KeyCode::Char(a), KeyCode::Char(b)) => {
                a.eq_ignore_ascii_case(&b)
                    && mods.contains(KeyModifiers::SHIFT)
                        == self.modifiers.contains(KeyModifiers::SHIFT)
            }
            (a, b) => {
                a == b
                    && mods.contains(KeyModifiers::SHIFT)
                        == self.modifiers.contains(KeyModifiers::SHIFT)
            }
        }
    }
}

impl fmt::Display for KeyCombo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.modifiers.contains(KeyModifiers::CONTROL) {
            write!(f, "ctrl+")?;
        }
        if self.modifiers.contains(KeyModifiers::ALT) {
            write!(f, "alt+")?;
        }
        if self.modifiers.contains(KeyModifiers::SHIFT) {
            write!(f, "shift+")?;
        }
        match self.code {
            KeyCode::Char(c) => write!(f, "{c}"),
            other => write!(f, "{other:?}"),
        }
    }
}

/// Bindings from chords to action names (TS `Keybinding` map). The action
/// name is the snake-case variant of [`Action`], e.g. `"abort"`, `"quit"`,
/// `"scroll_up"` — see `key-hint-format.ts` for how hints are rendered.
#[derive(Debug, Clone, Default)]
pub struct Keymap {
    bindings: Vec<(KeyCombo, &'static str)>,
}

impl Keymap {
    /// An empty keymap (all defaults are hardcoded per component for now).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Bind a chord to an action name (rebinding replaces).
    pub fn bind(&mut self, combo: KeyCombo, action: &'static str) {
        self.bindings.retain(|(c, _)| *c != combo);
        self.bindings.push((combo, action));
    }

    /// The action name bound to `combo`, if any.
    #[must_use]
    pub fn lookup(&self, combo: KeyCombo) -> Option<&'static str> {
        self.bindings
            .iter()
            .find(|(c, _)| *c == combo)
            .map(|(_, a)| *a)
    }

    /// Resolve a key event to an [`Action`] for the names this crate knows.
    /// Returns `None` for unbound chords; components then apply their
    /// built-in behavior.
    #[must_use]
    pub fn resolve(&self, event: KeyEvent) -> Option<Action> {
        let combo = KeyCombo {
            code: event.code,
            modifiers: event.modifiers,
        };
        match self.lookup(combo)? {
            "abort" => Some(Action::Abort),
            "quit" => Some(Action::Quit),
            "scroll_up" => Some(Action::Scroll(crate::action::ScrollDir::Up)),
            "scroll_down" => Some(Action::Scroll(crate::action::ScrollDir::Down)),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combo_display_round_trips_chords() {
        assert_eq!(KeyCombo::ctrl(KeyCode::Char('c')).to_string(), "ctrl+c");
        assert_eq!(KeyCombo::plain(KeyCode::Enter).to_string(), "Enter");
    }

    #[test]
    fn keymap_resolves_known_actions() {
        let mut km = Keymap::new();
        km.bind(KeyCombo::ctrl(KeyCode::Char('c')), "quit");
        let ev = KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL);
        assert_eq!(km.resolve(ev), Some(Action::Quit));
    }
}
