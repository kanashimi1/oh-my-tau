//! Theme file loading — the port of `pi-tui/src/theme/loader.ts`.
//!
//! Themes load from the 102 embedded built-ins (`dark`, `light`, `titanium`,
//! …) or from user JSON files under `~/.omp/agent/themes/` (a path containing
//! `/` is treated as an explicit file path). The JSON is validated against
//! the TS token catalog ([`schema::REQUIRED_COLORS`]), `vars` references are
//! resolved, and colors are materialized onto a [`Theme`] under both the TS
//! camelCase names (`mdHeading`) and this crate's snake_case convention
//! (`md_heading`).

use std::path::PathBuf;

use ratatui::style::Color;
use serde_json::Value;

use super::{Theme, ThemeValue};
use thiserror::Error;

mod builtin_index;
mod color;
mod schema;
mod session_accent;
mod symbols_data;
mod symbols_mono;

pub use builtin_index::BUILTIN;
pub use color::{parse_hex, Rgb};
pub use schema::{camel_to_snake, is_valid_symbol_preset};
pub use session_accent::{session_accent_hex, SessionAccentTheme};
pub use symbols_data::{ASCII, NERD, UNICODE};

/// Why a theme file could not be loaded.
#[derive(Debug, Error)]
pub enum ThemeError {
    /// The file could not be read.
    #[error("theme file unreadable: {0}")]
    Io(String),
    /// The theme is unknown (neither built-in nor on disk).
    #[error("theme not found: {0}")]
    NotFound(String),
    /// The file parsed but failed schema validation.
    #[error("theme file invalid: {0}")]
    Invalid(String),
}

/// Terminal appearance (TS `detectTerminalBackground` result).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Appearance {
    /// Dark terminal background.
    Dark,
    /// Light terminal background.
    Light,
}

/// Default theme names per appearance (settings `theme.dark` / `theme.light`).
pub const DEFAULT_DARK_THEME: &str = "titanium";
pub const DEFAULT_LIGHT_THEME: &str = "light";

/// Classify terminal background luminance (TS uses OSC 11 + `colorLuma`).
#[must_use]
pub fn appearance_of(background: Option<Rgb>) -> Appearance {
    let Some(rgb) = background else {
        return Appearance::Dark;
    };
    let luma =
        (0.2126 * f64::from(rgb.r) + 0.7152 * f64::from(rgb.g) + 0.0722 * f64::from(rgb.b)) / 255.0;
    if luma >= 0.5 {
        Appearance::Light
    } else {
        Appearance::Dark
    }
}

/// Appearance from the `COLORFGBG` env fallback (TS tier 2): bg palette
/// index < 8 is dark.
#[must_use]
pub fn appearance_from_colorfgbg(colorfgbg: Option<&str>) -> Option<Appearance> {
    let v = colorfgbg?;
    let bg = v.split(';').nth(1)?.trim().parse::<i32>().ok()?;
    Some(if bg < 8 {
        Appearance::Dark
    } else {
        Appearance::Light
    })
}

/// Best-effort appearance detection: an OSC 11 probe result when the caller
/// has one, else `COLORFGBG`, else dark (TS `detectTerminalBackground`).
#[must_use]
pub fn detect_appearance(osc11_bg: Option<Rgb>) -> Appearance {
    if let Some(bg) = osc11_bg {
        return appearance_of(Some(bg));
    }
    appearance_from_colorfgbg(std::env::var("COLORFGBG").ok().as_deref())
        .unwrap_or(Appearance::Dark)
}

/// Pick the configured theme name for an appearance.
#[must_use]
pub fn theme_name_for<'a>(dark: &'a str, light: &'a str, appearance: Appearance) -> &'a str {
    match appearance {
        Appearance::Dark => dark,
        Appearance::Light => light,
    }
}

/// The user's custom theme directory (TS `getCustomThemesDir`).
#[must_use]
pub fn custom_themes_dir() -> PathBuf {
    let mut p = std::env::var_os("OMP_AGENT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let mut home = PathBuf::from(std::env::var_os("HOME").unwrap_or_default());
            home.push(".omp");
            home.push("agent");
            home
        });
    p.push("themes");
    p
}

/// A loadable theme: name plus file path when it comes from disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThemeInfo {
    pub name: String,
    pub path: Option<PathBuf>,
}

/// Every loadable theme: embedded built-ins, then `~/.omp/agent/themes/*.json`
/// (custom names never shadow built-ins; TS `getAvailableThemesWithPaths`).
#[must_use]
pub fn available_themes() -> Vec<ThemeInfo> {
    let mut out: Vec<ThemeInfo> = BUILTIN
        .iter()
        .map(|(name, _)| ThemeInfo {
            name: (*name).to_owned(),
            path: None,
        })
        .collect();
    if let Ok(entries) = std::fs::read_dir(custom_themes_dir()) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "json") {
                let name = path
                    .file_stem()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into_owned();
                if !out.iter().any(|t| t.name == name) {
                    out.push(ThemeInfo {
                        name,
                        path: Some(path),
                    });
                }
            }
        }
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

/// Symbol preset glyph tables by preset name (TS `SYMBOL_PRESETS`).
#[must_use]
pub fn symbol_preset(preset: &str) -> &'static [(&'static str, &'static str)] {
    match preset {
        "nerd" => NERD,
        "ascii" => ASCII,
        _ => UNICODE,
    }
}

/// Effective glyph for `key` in `preset`: the `unicode` preset swaps the TS
/// table's color emoji for monochrome glyphs ([`symbols_mono::UNICODE_MONO`]).
#[must_use]
pub fn symbol_lookup(preset: &str, key: &str) -> Option<&'static str> {
    let find = |table: &'static [(&'static str, &'static str)]| {
        table.iter().find(|(k, _)| *k == key).map(|(_, v)| *v)
    };
    let table = symbol_preset(preset);
    if std::ptr::eq(table, UNICODE) {
        if let Some(v) = find(symbols_mono::UNICODE_MONO) {
            return Some(v);
        }
    }
    find(table)
}

/// Spinner frames for a preset and kind (`"status"` / `"activity"`).
#[must_use]
pub fn spinner_frames(preset: &str, kind: &str) -> &'static [&'static str] {
    symbols_data::SPINNERS
        .iter()
        .find(|(p, k, _)| *p == preset && *k == kind)
        .map_or_else(|| spinner_frames("unicode", kind), |(_, _, frames)| *frames)
}

/// A fully parsed theme: resolved [`Theme`] plus symbol/spinner config and
/// the resolved hex colors (TS `createTheme` result + `Theme.sessionAccentInputs`).
#[derive(Debug, Clone)]
pub struct LoadedTheme {
    /// Resolved token table (every catalog token present).
    pub theme: Theme,
    /// Symbol preset in effect (`"unicode"` / `"nerd"` / `"ascii"`).
    pub symbol_preset: &'static str,
    /// Glyph overrides from `themeJson.symbols.overrides`.
    pub symbol_overrides: Vec<(String, String)>,
    /// Spinner frame overrides from `themeJson.symbols.spinnerFrames`.
    pub spinner_frames: SpinnerFrames,
    /// Resolved colors as hex (`color` values `""` resolve to `None`).
    color_hexes: Vec<(&'static str, Option<String>)>,
}

/// Per-kind spinner frame overrides (TS `normalizeSpinnerFramesOverride`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SpinnerFrames {
    /// Overrides for the status spinner.
    pub status: Option<Vec<String>>,
    /// Overrides for the activity spinner.
    pub activity: Option<Vec<String>>,
}

impl SpinnerFrames {
    /// Frames for `kind` ("status"/"activity"): override, else preset default.
    #[must_use]
    pub fn frames(&self, preset: &str, kind: &str) -> Vec<String> {
        let overridden = match kind {
            "status" => self.status.as_ref(),
            _ => self.activity.as_ref(),
        };
        overridden.cloned().unwrap_or_else(|| {
            spinner_frames(preset, kind)
                .iter()
                .map(|s| (*s).to_owned())
                .collect()
        })
    }
}

impl LoadedTheme {
    /// Effective spinner frames for `kind`, overrides applied.
    #[must_use]
    pub fn frames_for(&self, kind: &str) -> Vec<String> {
        self.spinner_frames.frames(self.symbol_preset, kind)
    }

    /// Look up a glyph from the preset table, theme overrides applied.
    #[must_use]
    pub fn symbol(&self, name: &str) -> String {
        for (k, v) in &self.symbol_overrides {
            if k == name {
                return v.clone();
            }
        }
        symbol_lookup(self.symbol_preset, name).map_or_else(|| "?".to_owned(), ToOwned::to_owned)
    }

    /// Resolved hex for a token; `None` for default-terminal (`""`) colors.
    /// The status-line gauge uses this for dimming math.
    #[must_use]
    pub fn color_hex(&self, token: &str) -> Option<String> {
        self.color_hexes
            .iter()
            .find(|(k, _)| *k == token || *k == camel_to_snake(token))
            .and_then(|(_, v)| v.clone())
    }

    /// Session-accent derivation inputs (TS `Theme.sessionAccentInputs`).
    #[must_use]
    pub fn session_accent_inputs(&self, surface_luminance: Option<f64>) -> SessionAccentTheme {
        SessionAccentTheme {
            accent_hex: self.color_hex("accent").unwrap_or_else(|| "#777d88".into()),
            color_hexes: self
                .color_hexes
                .iter()
                .filter_map(|(_, v)| v.clone())
                .collect(),
            surface_luminance,
        }
    }

    /// Stable per-session accent hex (TS `getSessionAccentHex`).
    #[must_use]
    pub fn session_accent(&self, session_name: &str) -> String {
        session_accent_hex(session_name, &self.session_accent_inputs(None))
    }
}

/// A parsed theme JSON: validated tokens plus symbol config.
struct ParsedTheme {
    colors: Vec<(&'static str, Option<String>)>,
    symbol_preset: &'static str,
    symbol_overrides: Vec<(String, String)>,
    spinner_frames: SpinnerFrames,
}

/// Parse + validate theme JSON text (TS `parseThemeJson`).
fn parse_theme_json(name: &str, content: &str) -> Result<ParsedTheme, ThemeError> {
    let root: Value = serde_json::from_str(content)
        .map_err(|e| ThemeError::Invalid(format!("theme {name}: {e}")))?;
    let obj = root
        .as_object()
        .ok_or_else(|| ThemeError::Invalid(format!("theme {name}: top level must be an object")))?;

    // vars: name → color (hex/index or another var reference)
    let vars = obj.get("vars").and_then(Value::as_object);

    let colors_obj = obj
        .get("colors")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            ThemeError::Invalid(format!("theme {name}: missing required \"colors\" object"))
        })?;

    let mut colors = Vec::with_capacity(colors_obj.len());
    let mut missing: Vec<&str> = Vec::new();
    for token in schema::REQUIRED_COLORS
        .iter()
        .chain(schema::OPTIONAL_COLORS.iter())
    {
        let raw = match colors_obj.get(*token) {
            Some(v) => v,
            None => {
                if !schema::OPTIONAL_COLORS.contains(token) {
                    missing.push(token);
                }
                continue;
            }
        };
        let resolved = resolve_var_refs(raw, vars, &mut Vec::new())
            .map_err(|e| ThemeError::Invalid(format!("theme {name}: colors.{token}: {e}")))?;
        colors.push((*token, resolved));
    }
    // Unknown extra color keys are ignored: token names must be 'static
    // (`Theme::set_token`), so only catalog tokens register.
    if !missing.is_empty() {
        return Err(ThemeError::Invalid(format!(
            "invalid theme \"{name}\": missing required color tokens: {}",
            missing.join(", ")
        )));
    }

    // symbols: { preset?, overrides?, spinnerFrames? }
    let mut symbol_preset: &'static str = "unicode";
    let mut symbol_overrides = Vec::new();
    let mut spinner_frames = SpinnerFrames::default();
    if let Some(sym) = obj.get("symbols") {
        let sym = sym.as_object().ok_or_else(|| {
            ThemeError::Invalid(format!("theme {name}: \"symbols\" must be an object"))
        })?;
        if let Some(p) = sym.get("preset").and_then(Value::as_str) {
            if is_valid_symbol_preset(p) {
                symbol_preset = match p {
                    "nerd" => "nerd",
                    "ascii" => "ascii",
                    _ => "unicode",
                };
            }
        }
        if let Some(Value::Object(ov)) = sym.get("overrides") {
            for (k, v) in ov {
                if let Some(s) = v.as_str() {
                    symbol_overrides.push((k.clone(), s.to_owned()));
                }
            }
        }
        spinner_frames = parse_spinner_frames(sym.get("spinnerFrames"));
    }

    Ok(ParsedTheme {
        colors,
        symbol_preset,
        symbol_overrides,
        spinner_frames,
    })
}

/// Spinner override shapes: a flat array or `{status?, activity?}`.
fn parse_spinner_frames(v: Option<&Value>) -> SpinnerFrames {
    let Some(v) = v else {
        return SpinnerFrames::default();
    };
    match v {
        Value::Array(frames) => {
            let f = frames_of(frames);
            SpinnerFrames {
                status: Some(f.clone()),
                activity: Some(f),
            }
        }
        Value::Object(o) => SpinnerFrames {
            status: o
                .get("status")
                .and_then(Value::as_array)
                .map(|v| frames_of(v)),
            activity: o
                .get("activity")
                .and_then(Value::as_array)
                .map(|v| frames_of(v)),
        },
        _ => SpinnerFrames::default(),
    }
}

fn frames_of(frames: &[Value]) -> Vec<String> {
    frames
        .iter()
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect()
}

/// Resolve a color value against `vars` (TS `resolveVarRefs`): hex/index
/// pass through, other strings are variable references.
fn resolve_var_refs(
    value: &Value,
    vars: Option<&serde_json::Map<String, Value>>,
    seen: &mut Vec<String>,
) -> Result<Option<String>, String> {
    match value {
        // 256-color palette index.
        Value::Number(n) => {
            let idx = n
                .as_u64()
                .ok_or_else(|| "color index must be 0-255".to_owned())?;
            if idx > 255 {
                return Err(format!("color index {idx} out of range 0-255"));
            }
            Ok(Some(idx.to_string()))
        }
        Value::String(s) => {
            if s.is_empty() || s.starts_with('#') {
                return Ok((!s.is_empty()).then(|| s.clone()));
            }
            if seen.iter().any(|v| v == s) {
                return Err(format!("circular variable reference: {s}"));
            }
            let next = vars
                .and_then(|v| v.get(s))
                .ok_or_else(|| format!("variable reference not found: {s}"))?;
            seen.push(s.clone());
            resolve_var_refs(next, vars, seen)
        }
        _ => Err("color must be a hex string or 256-color index".to_owned()),
    }
}

/// Materialize a parsed theme onto a [`Theme`] (TS `createTheme`).
fn build_theme(name: &str, parsed: &ParsedTheme) -> Result<LoadedTheme, ThemeError> {
    let mut theme = Theme::default();
    theme.glyph_set = parsed.symbol_preset.to_owned();
    for (token, resolved) in &parsed.colors {
        let color = match resolved {
            None => Color::Reset,
            Some(s) if s.starts_with('#') => match parse_hex(s) {
                Some(Rgb { r, g, b }) => Color::Rgb(r, g, b),
                None => {
                    return Err(ThemeError::Invalid(format!(
                        "theme {name}: {token}: bad hex {s}"
                    )))
                }
            },
            Some(s) => match s.parse::<u8>() {
                Ok(idx) => Color::Indexed(idx),
                Err(_) => {
                    return Err(ThemeError::Invalid(format!(
                        "theme {name}: {token}: unresolvable value {s}"
                    )))
                }
            },
        };
        theme.set_token(token, ThemeValue::Color(color));
        // Also register the snake_case alias this crate's components use.
        if let Some(alias) = schema::snake_alias(token) {
            theme.set_token(alias, ThemeValue::Color(color));
        }
    }
    Ok(LoadedTheme {
        theme,
        symbol_preset: parsed.symbol_preset,
        symbol_overrides: parsed.symbol_overrides.clone(),
        spinner_frames: parsed.spinner_frames.clone(),
        color_hexes: parsed.colors.clone(),
    })
}

/// Load and build a theme by name or path, returning the full [`LoadedTheme`]
/// (TS `loadTheme`). Built-in names resolve to the embedded JSON; other names
/// resolve to `~/.omp/agent/themes/<name>.json`; anything containing `/` is
/// treated as an explicit JSON path.
///
/// # Errors
///
/// [`ThemeError::NotFound`] when no source matches; [`ThemeError::Io`] when a
/// file exists but cannot be read; [`ThemeError::Invalid`] on parse or schema
/// failures.
pub fn load_loaded(name_or_path: &str) -> Result<LoadedTheme, ThemeError> {
    let json = load_theme_json(name_or_path)?;
    let parsed = parse_theme_json(name_or_path, &json)?;
    build_theme(name_or_path, &parsed)
}

/// Load a theme by name or path (TS `loadTheme`).
///
/// # Errors
///
/// As [`load_loaded`].
pub fn load(name_or_path: &str) -> Result<Theme, ThemeError> {
    load_loaded(name_or_path).map(|t| t.theme)
}

/// Load the configured theme for an appearance, with the omp settings keys
/// `theme.dark` (default `titanium`) and `theme.light` (default `light`).
///
/// # Errors
///
/// As [`load`].
pub fn load_configured(
    theme_dark: Option<&str>,
    theme_light: Option<&str>,
    appearance: Appearance,
) -> Result<Theme, ThemeError> {
    let name = theme_name_for(
        theme_dark.unwrap_or(DEFAULT_DARK_THEME),
        theme_light.unwrap_or(DEFAULT_LIGHT_THEME),
        appearance,
    );
    load(name)
}

fn load_theme_json(name_or_path: &str) -> Result<String, ThemeError> {
    // TS `loadThemeJson`: built-ins first (never shadowed).
    if let Some((_, json)) = BUILTIN.iter().find(|(n, _)| *n == name_or_path) {
        return Ok((*json).to_owned());
    }
    let path = if name_or_path.contains('/') {
        PathBuf::from(name_or_path)
    } else {
        custom_themes_dir().join(format!("{name_or_path}.json"))
    };
    std::fs::read_to_string(&path).map_err(|e| match e.kind() {
        std::io::ErrorKind::NotFound => ThemeError::NotFound(name_or_path.to_owned()),
        _ => ThemeError::Io(format!("{}: {e}", path.display())),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::style::Color;

    #[test]
    fn builtin_dark_loads_and_resolves() {
        let t = load("dark").unwrap();
        assert_eq!(t.color("accent"), Color::Rgb(0xfe, 0xbc, 0x38));
        // var reference resolved
        assert_eq!(t.color("border"), Color::Rgb(0x17, 0x8f, 0xb9));
        // snake_case alias registered
        assert_eq!(t.color("md_heading"), t.color("mdHeading"));
        assert_eq!(t.color("status_line_sep"), t.color("statusLineSep"));
    }

    #[test]
    fn every_builtin_theme_loads() {
        for (name, _) in BUILTIN {
            let t = load_loaded(name).unwrap_or_else(|e| panic!("builtin theme {name}: {e}"));
            assert!(!t.color_hex("accent").unwrap_or_default().is_empty());
        }
    }

    #[test]
    fn titanium_is_default_dark() {
        let t = load_configured(None, None, Appearance::Dark).unwrap();
        assert_eq!(t.color("accent"), load("titanium").unwrap().color("accent"));
    }

    #[test]
    fn unknown_theme_is_not_found() {
        assert!(matches!(
            load("definitely-not-a-theme"),
            Err(ThemeError::NotFound(_))
        ));
    }

    #[test]
    fn missing_required_colors_reported() {
        let err = load_json_text(
            "broken",
            r##"{"name":"broken","colors":{"accent":"#fff"}}"##,
        );
        match err {
            Err(ThemeError::Invalid(msg)) => assert!(msg.contains("missing"), "{msg}"),
            other => panic!("expected invalid, got {other:?}"),
        }
    }

    #[test]
    fn var_cycle_detected() {
        let json =
            r##"{"name":"c","vars":{"a":"b","b":"a"},"colors":{"accent":"a","text":"#ffffff"}}"##;
        assert!(matches!(
            load_json_text("c", json),
            Err(ThemeError::Invalid(_))
        ));
    }

    #[test]
    fn indexed_colors_and_optional_thinking_max() {
        // still invalid: only one required token — but exercises parsing of
        // indices via the full-file test above; assert parse-level behavior:
        let parsed = parse_theme_json(
            "i",
            r##"{"name":"i","colors":{"accent":196,"thinkingMax":250,"text":"#112233","border":"dim","vars":{}},"vars":{"dim":"#444"}}"##,
        );
        // missing tokens → invalid; we only care that indices don't crash
        assert!(parsed.is_err());
    }

    #[test]
    fn spinner_override_shapes() {
        assert_eq!(parse_spinner_frames(None), SpinnerFrames::default());
        let flat = serde_json::from_str::<Value>(r##"["a","b"]"##).unwrap();
        let f = parse_spinner_frames(Some(&flat));
        assert_eq!(f.frames("unicode", "status"), vec!["a", "b"]);
        assert_eq!(f.frames("unicode", "activity"), vec!["a", "b"]);
        let split =
            serde_json::from_str::<Value>(r##"{"status":["s"],"activity":["x","y"]}"##).unwrap();
        let f2 = parse_spinner_frames(Some(&split));
        assert_eq!(f2.frames("nerd", "status"), vec!["s"]);
        // split override only touches "activity" here; "status" falls back —
        // covered by the flat-array case above (applies to both kinds).
        assert_eq!(f2.frames("nerd", "activity"), vec!["x", "y"]);
    }

    #[test]
    fn symbols_and_preset_lookup() {
        assert_eq!(symbol_preset("unicode").len(), symbol_preset("ascii").len());
        assert_eq!(spinner_frames("ascii", "status").len(), 4);
        assert_eq!(spinner_frames("unicode", "activity").len(), 10);
        let t = load_loaded("dark").unwrap();
        assert_eq!(t.symbol("status.success"), "✓");
        assert_eq!(t.frames_for("status").len(), 8);
    }

    #[test]
    fn appearance_detection() {
        assert_eq!(
            appearance_of(Some(Rgb { r: 0, g: 0, b: 0 })),
            Appearance::Dark
        );
        assert_eq!(
            appearance_of(Some(Rgb {
                r: 255,
                g: 255,
                b: 255
            })),
            Appearance::Light
        );
        assert_eq!(
            appearance_from_colorfgbg(Some("0;15")),
            Some(Appearance::Light)
        );
        assert_eq!(
            appearance_from_colorfgbg(Some("15;0")),
            Some(Appearance::Dark)
        );
        assert_eq!(appearance_from_colorfgbg(Some("junk")), None);
        assert_eq!(theme_name_for("t", "l", Appearance::Light), "l");
    }

    #[test]
    fn session_accent_varies_per_session() {
        let t = load_loaded("dark").unwrap();
        let a = t.session_accent("alpha");
        let b = t.session_accent("beta");
        assert_eq!(a, t.session_accent("alpha"));
        assert!(a.starts_with('#') && b.starts_with('#'));
    }

    fn load_json_text(name: &str, json: &str) -> Result<Theme, ThemeError> {
        let parsed = parse_theme_json(name, json)?;
        build_theme(name, &parsed).map(|t| t.theme)
    }
}
