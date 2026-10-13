//! Theme tokens and resolution — the port of `pi-tui/src/theme/`.
//!
//! A [`Theme`] is a flat map from a *token* (e.g. `"accent"`, `"text"`,
//! `"error"`) to a concrete [`ThemeValue`]. Components never name ANSI colors
//! directly; they ask the theme for a token (mirroring TS `Theme.getColor`,
//! `theme-class.ts` `resolveColor`). Themes load from the user's
//! `~/.omp/agent/themes/*.json`-style files via [`loader`] (TS
//! `theme/loader.ts`); until that lands, [`Theme::default`] provides the
//! built-in dark theme (TS `defaultTheme`, `theme/theme.ts`).
//!
//! When adding tokens, keep the names aligned with the TS schema
//! (`theme/schema.ts`) so user themes port without translation.

use std::collections::BTreeMap;

use ratatui::style::Color;

mod loader;
pub use loader::{load, ThemeError};

/// One resolvable style value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThemeValue {
    /// A direct terminal color (rare; prefer named tokens).
    Color(Color),
    /// A named token resolved through the theme (may chain once).
    Token(&'static str),
}

/// A complete theme: token table plus glyph overrides.
#[derive(Debug, Clone)]
pub struct Theme {
    tokens: BTreeMap<&'static str, ThemeValue>,
    /// ASCII/Unicode glyph set in use (`pi-tui/src/theme/symbols.ts`):
    /// `"ascii"`, `"unicode"`, `"nerdfont"`, ...
    pub glyph_set: String,
}

impl Theme {
    /// Resolve a token to a concrete terminal color.
    ///
    /// Falls back to [`Color::Reset`] for unknown tokens; never panics —
    /// themes are user data.
    #[must_use]
    pub fn color(&self, token: &str) -> Color {
        match self.tokens.get(token) {
            Some(ThemeValue::Color(c)) => *c,
            Some(ThemeValue::Token(next)) => self.color(next),
            None => Color::Reset,
        }
    }

    /// Look up a glyph by logical name (`"prompt"`, `"success"`, ...).
    /// Unknown names return `"?"` rather than panicking.
    #[must_use]
    pub fn glyph(&self, name: &str) -> &'static str {
        glyphs(self.glyph_set.as_str())
            .get(name)
            .copied()
            .unwrap_or("?")
    }

    /// Insert/override a token (used by the loader and tests).
    pub fn set_token(&mut self, token: &'static str, value: ThemeValue) {
        self.tokens.insert(token, value);
    }
}

impl Default for Theme {
    /// The built-in dark theme (TS `defaultTheme`).
    fn default() -> Self {
        let mut t = Self {
            tokens: BTreeMap::new(),
            glyph_set: "unicode".into(),
        };
        t.set_token("accent", ThemeValue::Color(Color::Cyan));
        t.set_token("text", ThemeValue::Color(Color::White));
        t.set_token("text_muted", ThemeValue::Color(Color::DarkGray));
        t.set_token("background", ThemeValue::Color(Color::Black));
        t.set_token("success", ThemeValue::Color(Color::Green));
        t.set_token("warning", ThemeValue::Color(Color::Yellow));
        t.set_token("error", ThemeValue::Color(Color::Red));
        t.set_token("border", ThemeValue::Token("text_muted"));
        t.set_token("selected", ThemeValue::Token("accent"));
        t
    }
}

/// The glyph table for a glyph set (TS `theme/symbols.ts` rows).
fn glyphs(set: &str) -> BTreeMap<&'static str, &'static str> {
    let mut m = BTreeMap::new();
    if set == "ascii" {
        m.insert("prompt", ">");
        m.insert("success", "+");
        m.insert("failure", "x");
        m.insert("collapsed", "v");
        m.insert("expanded", ">");
    } else {
        // unicode / nerdfont defaults
        m.insert("prompt", "❯");
        m.insert("success", "✓");
        m.insert("failure", "✗");
        m.insert("collapsed", "▸");
        m.insert("expanded", "▾");
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_theme_resolves_core_tokens() {
        let t = Theme::default();
        assert_eq!(t.color("accent"), Color::Cyan);
        assert_eq!(t.color("border"), Color::DarkGray); // chained token
        assert_eq!(t.color("nope"), Color::Reset); // unknown never panics
        assert_eq!(t.glyph("prompt"), "❯");
    }

    #[test]
    fn ascii_glyph_set_swaps_glyphs() {
        let t = Theme {
            glyph_set: "ascii".into(),
            ..Default::default()
        };
        assert_eq!(t.glyph("prompt"), ">");
    }
}
