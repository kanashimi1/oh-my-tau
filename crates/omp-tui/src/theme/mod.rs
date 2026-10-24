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

pub mod loader;
pub use loader::{
    appearance_from_colorfgbg, appearance_of, available_themes, custom_themes_dir,
    detect_appearance, load, load_configured, theme_name_for, Appearance, LoadedTheme, Rgb,
    ThemeError, ThemeInfo,
};

/// One resolvable style value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThemeValue {
    /// A direct terminal color (rare; prefer named tokens).
    Color(Color),
    /// A named token resolved through the theme (may chain once).
    Token(&'static str),
}

/// Transcript/chrome presentation style.
///
/// `Classic` mirrors the TS `omp` layout (boxed tool cards, powerline status
/// line, icon settings tabs). `Clean` is the omp-rs look: rail tool rows,
/// dot-separated status line, text tabs. Resolved by [`UiStyle::resolve`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UiStyle {
    /// TS-parity layout (the renderer default; tests rely on it).
    #[default]
    Classic,
    /// Rail cards, dimmed chrome, text tabs.
    Clean,
}

impl UiStyle {
    /// `OMP_TUI_STYLE` env, else the `tui.style` setting, else `Clean`.
    #[must_use]
    pub fn resolve(setting: Option<&str>) -> Self {
        let env = std::env::var("OMP_TUI_STYLE").ok();
        match env.as_deref().or(setting) {
            Some("classic") => Self::Classic,
            _ => Self::Clean,
        }
    }
}

/// A complete theme: token table plus glyph overrides.
#[derive(Debug, Clone)]
pub struct Theme {
    tokens: BTreeMap<&'static str, ThemeValue>,
    /// ASCII/Unicode glyph set in use (`pi-tui/src/theme/symbols.ts`):
    /// `"ascii"`, `"unicode"`, `"nerdfont"`, ...
    pub glyph_set: String,
    /// Presentation style (rides on the theme so every renderer sees it).
    pub ui: UiStyle,
    /// Magic keywords switched off by settings (`magicKeywords.*`), as a
    /// [`crate::magic::disabled_mask`] bitmask; `0` paints every keyword.
    pub magic_off: u32,
}

impl Theme {
    /// Whether the omp-rs clean presentation is active.
    #[must_use]
    pub fn clean(&self) -> bool {
        self.ui == UiStyle::Clean
    }

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

    /// Look up a TS symbol-table glyph by key (`"status.enabled"`,
    /// `"icon.input"`, `"tree.last"`, …; `theme/symbols.ts`) in the preset
    /// matching [`Self::glyph_set`]. Unknown keys return `""`.
    #[must_use]
    pub fn symbol(&self, key: &str) -> &'static str {
        let preset = match self.glyph_set.as_str() {
            "ascii" => "ascii",
            "nerd" | "nerdfont" => "nerd",
            _ => "unicode",
        };
        loader::symbol_lookup(preset, key).unwrap_or("")
    }
}

impl Default for Theme {
    /// The built-in dark theme (TS `defaultTheme`).
    fn default() -> Self {
        let mut t = Self {
            tokens: BTreeMap::new(),
            glyph_set: "unicode".into(),
            ui: UiStyle::Classic,
            magic_off: 0,
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
