//! Status line presets — the port of `status-line/presets.ts` and the
//! `statusLine.*` settings (`schema.ts`).

/// One renderable unit of the status line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(missing_docs)]
pub enum SegmentId {
    Pi,
    Status,
    Model,
    Mode,
    Path,
    Git,
    Pr,
    Subagents,
    TokenIn,
    TokenOut,
    TokenTotal,
    TokenRate,
    Cost,
    ContextPct,
    ContextTotal,
    TimeSpent,
    Time,
    Session,
    Hostname,
    CacheRead,
    CacheWrite,
    CacheHit,
    SessionName,
    Collab,
    Stream,
    Vim,
    Usage,
}

impl SegmentId {
    /// Parse a segment id from its settings/schema name.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "pi" => Self::Pi,
            "status" => Self::Status,
            "model" => Self::Model,
            "mode" => Self::Mode,
            "path" => Self::Path,
            "git" => Self::Git,
            "pr" => Self::Pr,
            "subagents" => Self::Subagents,
            "token_in" => Self::TokenIn,
            "token_out" => Self::TokenOut,
            "token_total" => Self::TokenTotal,
            "token_rate" => Self::TokenRate,
            "cost" => Self::Cost,
            "context_pct" => Self::ContextPct,
            "context_total" => Self::ContextTotal,
            "time_spent" => Self::TimeSpent,
            "time" => Self::Time,
            "session" => Self::Session,
            "hostname" => Self::Hostname,
            "cache_read" => Self::CacheRead,
            "cache_write" => Self::CacheWrite,
            "cache_hit" => Self::CacheHit,
            "session_name" => Self::SessionName,
            "collab" => Self::Collab,
            "stream" => Self::Stream,
            "vim" => Self::Vim,
            "usage" => Self::Usage,
            _ => return None,
        })
    }
}

/// A named arrangement of segments (`presets.ts:9-57`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(missing_docs)]
pub enum Preset {
    Default,
    Minimal,
    Compact,
    Full,
    Nerd,
    Ascii,
    Custom,
}

impl Preset {
    /// Resolve a preset from its settings name; unknown → `Default`
    /// (`presets.ts:55`).
    #[must_use]
    pub fn from_name(name: &str) -> Self {
        match name {
            "minimal" => Self::Minimal,
            "compact" => Self::Compact,
            "full" => Self::Full,
            "nerd" => Self::Nerd,
            "ascii" => Self::Ascii,
            "custom" => Self::Custom,
            _ => Self::Default,
        }
    }

    /// The separator a preset forces (`presets.ts:9-57`); `None` falls back
    /// to the `statusLine.separator` setting.
    #[must_use]
    pub const fn separator(self) -> Option<SeparatorStyle> {
        match self {
            Self::Minimal => Some(SeparatorStyle::Slash),
            Self::Full | Self::Nerd => Some(SeparatorStyle::Powerline),
            Self::Ascii => Some(SeparatorStyle::Ascii),
            _ => None,
        }
    }

    /// `(left, right)` segment lists (`presets.ts:9-57`).
    #[must_use]
    pub fn segments(self) -> (Vec<SegmentId>, Vec<SegmentId>) {
        use SegmentId as S;
        let l = |ids: &[S]| ids.to_vec();
        match self {
            Self::Default => (
                l(&[
                    S::Pi,
                    S::Vim,
                    S::Model,
                    S::Mode,
                    S::Collab,
                    S::Stream,
                    S::Path,
                    S::Git,
                    S::Pr,
                    S::ContextPct,
                    S::Cost,
                ]),
                l(&[S::SessionName]),
            ),
            Self::Minimal => (
                l(&[S::Vim, S::Path, S::Git]),
                l(&[S::SessionName, S::Mode, S::ContextPct]),
            ),
            Self::Compact => (
                l(&[S::Vim, S::Model, S::Mode, S::Git, S::Pr]),
                l(&[S::SessionName, S::Cost, S::ContextPct]),
            ),
            Self::Full => (
                l(&[
                    S::Pi,
                    S::Vim,
                    S::Hostname,
                    S::Model,
                    S::Mode,
                    S::Path,
                    S::Git,
                    S::Pr,
                    S::Subagents,
                ]),
                l(&[
                    S::SessionName,
                    S::TokenIn,
                    S::TokenOut,
                    S::CacheRead,
                    S::Cost,
                    S::Usage,
                    S::TimeSpent,
                    S::ContextPct,
                    S::Time,
                ]),
            ),
            // Nerd/Ascii reuse the full layout with the glyph set swapped
            // (resolved at render time via `Theme::glyph_set`).
            Self::Nerd | Self::Ascii => Self::Full.segments(),
            Self::Custom => (CUSTOM_LEFT.to_vec(), CUSTOM_RIGHT.to_vec()),
        }
    }
}

/// `CUSTOM_STATUS_LINE_DEFAULTS` (`schema.ts:25-31`).
pub const CUSTOM_LEFT: &[SegmentId] = &[
    SegmentId::Vim,
    SegmentId::Model,
    SegmentId::Mode,
    SegmentId::Path,
    SegmentId::Git,
    SegmentId::Pr,
];

/// See [`CUSTOM_LEFT`].
pub const CUSTOM_RIGHT: &[SegmentId] = &[
    SegmentId::SessionName,
    SegmentId::TokenTotal,
    SegmentId::Cost,
    SegmentId::ContextPct,
];

/// Clean-style segments for the `default` preset (omp-rs only).
pub const CLEAN_DEFAULT_LEFT: &[SegmentId] = &[
    SegmentId::Model,
    SegmentId::Mode,
    SegmentId::Path,
    SegmentId::Git,
    SegmentId::Pr,
    SegmentId::Subagents,
];

/// See [`CLEAN_DEFAULT_LEFT`].
pub const CLEAN_DEFAULT_RIGHT: &[SegmentId] = &[
    SegmentId::SessionName,
    SegmentId::Cost,
    SegmentId::ContextPct,
];

/// How segment groups are joined (`separators.ts`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(missing_docs)]
pub enum SeparatorStyle {
    Powerline,
    PowerlineThin,
    Slash,
    Pipe,
    Block,
    None,
    Ascii,
}

impl SeparatorStyle {
    /// Resolve from the settings name; unknown → `PowerlineThin`
    /// (`separators.ts:36`).
    #[must_use]
    pub fn from_name(name: &str) -> Self {
        match name {
            "powerline" => Self::Powerline,
            "slash" => Self::Slash,
            "pipe" => Self::Pipe,
            "block" => Self::Block,
            "none" => Self::None,
            "ascii" => Self::Ascii,
            _ => Self::PowerlineThin,
        }
    }

    /// Whether this style paints powerline end caps.
    #[must_use]
    pub fn is_powerline(self) -> bool {
        matches!(self, Self::Powerline | Self::PowerlineThin | Self::Ascii)
    }
}

/// Context-gauge mode (`statusLine.contextLine`, §5.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(missing_docs)]
pub enum ContextLineMode {
    Off,
    Percentage,
    Annotated,
    Embedded,
}

impl ContextLineMode {
    /// Resolve from the settings name.
    #[must_use]
    pub fn from_name(name: &str) -> Self {
        match name {
            "off" => Self::Off,
            "percentage" => Self::Percentage,
            "annotated" => Self::Annotated,
            _ => Self::Embedded,
        }
    }
}

/// Everything the renderer needs from settings (`statusLine.*` keys, §2).
#[allow(clippy::struct_excessive_bools)] // mirrors the statusLine.* settings keys
#[derive(Debug, Clone)]
pub struct StatusLineConfig {
    /// Selected preset.
    pub preset: Preset,
    /// Separator style (used when the preset doesn't force one).
    pub separator: SeparatorStyle,
    /// Gauge fill mode.
    pub context_line: ContextLineMode,
    /// Hash the session name to a stable accent color (§7.6).
    pub session_accent: bool,
    /// Transparent background (terminal default bg, no end caps).
    pub transparent: bool,
    /// Compact thinking level on the model segment.
    pub compact_thinking: bool,
    /// Print hook statuses under the status line (§5.5).
    pub show_hook_status: bool,
    /// `path` segment: max rendered length.
    pub path_max_length: usize,
    /// `path` segment: shorten parent dirs to initials.
    pub path_abbreviate: bool,
    /// `time` segment: 12h clock.
    pub time_12h: bool,
    /// `time` segment: show seconds.
    pub time_show_seconds: bool,
    /// `statusLine.leftSegments` (custom preset only; `None` →
    /// [`CUSTOM_LEFT`]).
    pub left_segments: Option<Vec<SegmentId>>,
    /// `statusLine.rightSegments` (custom preset only; `None` →
    /// [`CUSTOM_RIGHT`]).
    pub right_segments: Option<Vec<SegmentId>>,
}

impl StatusLineConfig {
    /// Read the `statusLine.*` keys from the layered settings (TS
    /// `StatusLineComponent.#resolveSettings`); unset keys keep the defaults.
    #[must_use]
    pub fn from_settings(settings: &omp_settings::Settings) -> Self {
        let mut cfg = Self::default();
        if let Some(name) = settings.get_string("statusLine.preset") {
            cfg.preset = Preset::from_name(&name);
        }
        if let Some(name) = settings.get_string("statusLine.separator") {
            cfg.separator = SeparatorStyle::from_name(&name);
        }
        if let Some(name) = settings.get_string("statusLine.contextLine") {
            cfg.context_line = ContextLineMode::from_name(&name);
        }
        if let Some(v) = settings.get_bool("statusLine.sessionAccent") {
            cfg.session_accent = v;
        }
        if let Some(v) = settings.get_bool("statusLine.transparent") {
            cfg.transparent = v;
        }
        if let Some(v) = settings.get_bool("statusLine.compactThinkingLevel") {
            cfg.compact_thinking = v;
        }
        if let Some(v) = settings.get_bool("statusLine.showHookStatus") {
            cfg.show_hook_status = v;
        }
        let segments = |id: &str| -> Option<Vec<SegmentId>> {
            let names = settings.get_string_array(id);
            (!names.is_empty()).then(|| {
                names
                    .iter()
                    .filter_map(|n| SegmentId::from_name(n))
                    .collect()
            })
        };
        cfg.left_segments = segments("statusLine.leftSegments");
        cfg.right_segments = segments("statusLine.rightSegments");
        cfg
    }
}

impl Default for StatusLineConfig {
    fn default() -> Self {
        Self {
            preset: Preset::Default,
            separator: SeparatorStyle::PowerlineThin,
            context_line: ContextLineMode::Embedded,
            session_accent: true,
            transparent: false,
            compact_thinking: true,
            show_hook_status: true,
            path_max_length: 40,
            path_abbreviate: true,
            time_12h: false,
            time_show_seconds: false,
            left_segments: None,
            right_segments: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_preset_falls_back_to_default() {
        assert_eq!(Preset::from_name("nope"), Preset::Default);
        let (l, r) = Preset::Default.segments();
        assert_eq!(l.first(), Some(&SegmentId::Pi));
        assert_eq!(r, vec![SegmentId::SessionName]);
    }

    #[test]
    fn unknown_separator_falls_back_to_thin() {
        assert_eq!(
            SeparatorStyle::from_name("nope"),
            SeparatorStyle::PowerlineThin
        );
        assert!(SeparatorStyle::Powerline.is_powerline());
        assert!(!SeparatorStyle::Slash.is_powerline());
    }

    #[test]
    fn custom_preset_uses_custom_defaults() {
        let (l, r) = Preset::Custom.segments();
        assert_eq!(l, CUSTOM_LEFT);
        assert_eq!(r, CUSTOM_RIGHT);
    }
}
