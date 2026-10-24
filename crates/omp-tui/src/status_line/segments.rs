//! The segment registry — the port of `status-line/segments.ts`.
//!
//! Every segment returns `Some(pieces)` or `None` (TS `visible=false`);
//! `None` omits the segment entirely. All colors are token names resolved
//! through [`super::style::color`]; icons come from [`super::style::icon`].

use ratatui::style::Color;

use super::metrics as fmt;
use super::presets::SegmentId;
use super::style;
use std::fmt::Write as _;

use super::{Mode, StatusLine, VimMode};

/// One styled run of text inside a segment.
#[derive(Debug, Clone)]
pub struct Piece {
    /// Text content (icons included).
    pub text: String,
    /// Theme token for the fg color (`"accent"`, `"warning"`,
    /// `"statusLineModel"`, ...).
    pub token: &'static str,
    /// Clean style: explicit multi-color runs `(text, token)` that replace
    /// the single-token paint (`text` holds their concatenation).
    pub runs: Vec<(String, &'static str)>,
    /// Clean style: literal-colour runs (gradients, the `τ` state mark);
    /// win over `runs` and `token` (`text` holds their concatenation).
    pub rgb: Vec<(String, Color)>,
}

impl Piece {
    pub(crate) fn new(token: &'static str, text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            token,
            runs: Vec::new(),
            rgb: Vec::new(),
        }
    }

    /// A piece painted with literal colours (clean gradients, `τ`).
    pub(crate) fn from_rgb(rgb: Vec<(String, Color)>) -> Self {
        Self {
            text: rgb.iter().map(|(t, _)| t.as_str()).collect(),
            token: "text",
            runs: Vec::new(),
            rgb,
        }
    }

    /// A piece painted as several colored runs (clean context bar).
    pub(crate) fn from_runs(runs: Vec<(String, &'static str)>) -> Self {
        let text = runs.iter().map(|(t, _)| t.as_str()).collect();
        Self {
            text,
            token: runs.first().map_or("dim", |(_, token)| *token),
            runs,
            rgb: Vec::new(),
        }
    }
}

/// Everything a segment needs for one frame (aggregated per
/// `status-line/types.ts` `SegmentContext`).
#[derive(Debug, Clone, Copy)]
pub struct Ctx<'a> {
    /// The status line state.
    pub st: &'a StatusLine,
    /// Session-accent color when enabled and the session is named (§7.6).
    pub accent: Option<Color>,
    /// Current `path.maxLength` (the overflow pass shrinks it).
    pub path_max: usize,
}

/// Tokens that swap to the session accent under `accentFg`
/// (`segments.ts:39-108`): model and subagents route through it; the
/// `pi`/`mode`/`pr`/`session_name`/`hostname`/`collab` segments call
/// [`Ctx::maybe_accent`] directly.
const ACCENTABLE: &[&str] = &["statusLineModel", "statusLineSubagents"];

impl Ctx<'_> {
    /// `accentFg(ctx, color, ...)`: the accent token when session-accent is
    /// active for an accentable token, else the token itself.
    fn tok(&self, token: &'static str) -> &'static str {
        if self.accent.is_some() && ACCENTABLE.contains(&token) {
            "accent"
        } else {
            token
        }
    }

    /// Direct accent for segments that are always accent-colored when
    /// session-accent is on (`pi`/`mode`/`pr`/`session_name`/`hostname`/`collab`).
    pub(crate) fn maybe_accent(&self, token: &'static str, text: impl Into<String>) -> Piece {
        if self.accent.is_some() {
            Piece::new("accent", text)
        } else {
            Piece::new(token, text)
        }
    }
}

/// Render one segment; `None` means hidden.
#[must_use]
pub fn render(id: SegmentId, ctx: &Ctx<'_>, theme: &crate::theme::Theme) -> Option<Vec<Piece>> {
    match id {
        SegmentId::Pi => Some(pi(ctx, theme)),
        SegmentId::Status => status(ctx),
        SegmentId::Model => Some(model(ctx, theme)),
        SegmentId::Mode => mode(ctx, theme),
        SegmentId::Path => Some(path(ctx, theme)),
        SegmentId::Git => git(ctx, theme),
        SegmentId::Pr => pr(ctx, theme),
        SegmentId::Subagents => subagents(ctx, theme),
        SegmentId::Vim => vim(ctx, theme),
        SegmentId::SessionName => session_name(ctx),
        SegmentId::Hostname => hostname(ctx, theme),
        SegmentId::Session => Some(session(ctx, theme)),
        SegmentId::Time => Some(time(ctx, theme)?),
        SegmentId::TimeSpent => time_spent(ctx, theme),
        SegmentId::Collab => collab(ctx),
        SegmentId::Stream => stream(ctx),
        _ => super::stats::render(id, ctx, theme),
    }
}

/// `pi` — brand segment (`segments.ts:203-246`).
fn pi(ctx: &Ctx<'_>, theme: &crate::theme::Theme) -> Vec<Piece> {
    if let Some(agent) = &ctx.st.focused_agent {
        return vec![Piece::new(
            "warning",
            format!("{} {agent}", style::icon(theme, "icon.ghost", "👻")),
        )];
    }
    let text = match ctx.st.turn_elapsed_ms {
        Some(elapsed) => format!(
            "{} {}",
            style::activity_frame(ctx.st.now_ms),
            fmt::format_brand_timer(elapsed)
        ),
        None => style::icon(theme, "icon.omp", "π").to_string(),
    };
    vec![ctx.maybe_accent("dim", text)]
}

/// `status` — hook statuses (`segments.ts:248-272`).
fn status(ctx: &Ctx<'_>) -> Option<Vec<Piece>> {
    if ctx.st.hook_statuses.is_empty() {
        return None;
    }
    let joined = ctx
        .st
        .hook_statuses
        .iter()
        .map(|s| fmt::sanitize_status_text(s))
        .collect::<Vec<_>>()
        .join(" · ");
    Some(vec![ctx.maybe_accent("accent", joined)])
}

/// `model` — name, thinking level, fast mode (`segments.ts:274-392`).
fn model(ctx: &Ctx<'_>, theme: &crate::theme::Theme) -> Vec<Piece> {
    let name = ctx
        .st
        .model_name
        .as_deref()
        .unwrap_or(&ctx.st.model)
        .trim_start_matches("Claude ")
        .to_string();
    if let (Some(ultra), true) = (ctx.st.ultra, theme.clean()) {
        return ultra_model(ctx, theme, &name, ultra);
    }
    let display = thinking_display(ctx.st, theme);
    // Compact mode swaps the model icon for the thinking-level glyph and
    // drops the ` · <level>` tail; an ultra mode always shows its name.
    let compact = ctx.st.compact_thinking && display.is_some() && ctx.st.ultra.is_none();
    let icon = match (&display, compact) {
        (Some(display), true) => display.split(' ').next().unwrap_or_default().to_owned(),
        _ => style::icon(theme, "icon.model", "⬢").to_owned(),
    };
    let mut text = format!("{icon} {name}");
    if ctx.st.fast_mode {
        text.push(' ');
        text.push_str(style::icon(theme, "icon.fast", "⚡"));
    }
    if let (Some(display), false) = (&display, compact) {
        text.push_str(theme.symbol("sep.dot"));
        text.push_str(display);
    }
    vec![Piece::new(ctx.tok("statusLineModel"), text)]
}

/// Characters of `text` painted along `stops` (first char = first stop,
/// last char = last stop).
#[allow(clippy::cast_precision_loss)]
pub(crate) fn gradient_runs(text: &str, stops: &[(u8, u8, u8)]) -> Vec<(String, Color)> {
    let last = text.chars().count().saturating_sub(1).max(1) as f64;
    text.chars()
        .enumerate()
        .map(|(i, c)| {
            let (red, green, blue) = crate::magic::gradient_at(stops, i as f64 / last);
            (c.to_string(), Color::Rgb(red, green, blue))
        })
        .collect()
}

/// Clean `model` with an ultra mode: `⬢ name τ∿` (the τ-wave in the mode's
/// gradient) and the thinking slot as `◉ <mode>` in the same gradient.
fn ultra_model(
    ctx: &Ctx<'_>,
    theme: &crate::theme::Theme,
    name: &str,
    ultra: crate::magic::Ultra,
) -> Vec<Piece> {
    let stops = ultra.gradient();
    let dim = style::dim(theme);
    let mut rgb = vec![
        (style::icon(theme, "icon.model", "⬢").to_owned(), dim),
        (
            format!(" {name}"),
            style::clean_color(theme, ctx.tok("statusLineModel")),
        ),
    ];
    if ctx.st.fast_mode {
        rgb.push((format!(" {}", style::icon(theme, "icon.fast", "⚡")), dim));
    }
    rgb.push((" ".to_owned(), dim));
    rgb.extend(gradient_runs("τ∿", stops));
    vec![
        Piece::from_rgb(rgb),
        Piece::from_rgb(gradient_runs(&format!("◉ {}", ultra.name()), stops)),
    ]
}

/// Thinking-level display (`modelThinkingDisplay`): `◒ high`, `⦸ off`,
/// `⟳ auto` while auto thinking has no classified level; `None` when the
/// model has no thinking. An ultra mode shows its name as the level label
/// behind the `max` glyph (`◉ ultracode`).
fn thinking_display(st: &super::StatusLine, theme: &crate::theme::Theme) -> Option<String> {
    let level = st.thinking.as_deref()?;
    if let Some(ultra) = st.ultra {
        return Some(match theme.symbol("thinking.max").split_once(' ') {
            Some((glyph, _)) => format!("{glyph} {}", ultra.name()),
            None => ultra.name().to_owned(),
        });
    }
    if st.auto_thinking {
        return Some(format!("{} auto", theme.symbol("thinking.autoPending")));
    }
    if level == "off" {
        return Some(format!("{} off", theme.symbol("status.disabled")));
    }
    let symbol = theme.symbol(&format!("thinking.{level}"));
    Some(if symbol.is_empty() {
        level.to_owned()
    } else {
        symbol.to_owned()
    })
}

/// `mode` — Plan/Prewalk/Goal/Vibe/Loop (`segments.ts:486-562`).
fn mode(ctx: &Ctx<'_>, theme: &crate::theme::Theme) -> Option<Vec<Piece>> {
    let (text, warning) = match &ctx.st.mode {
        Some(Mode::Plan { paused: true }) => (
            format!(
                "{} Plan {}",
                style::icon(theme, "icon.plan", "🗺"),
                style::icon(theme, "icon.pause", "⏸")
            ),
            true,
        ),
        Some(Mode::Plan { .. }) => (
            format!("{} Plan", style::icon(theme, "icon.plan", "🗺")),
            false,
        ),
        Some(Mode::Prewalk) => (
            format!("{} Prewalk", style::icon(theme, "icon.prewalk", "🏃")),
            false,
        ),
        Some(Mode::Goal) => {
            let mut t = format!("{} Goal", style::icon(theme, "icon.goal", "🎯"));
            if ctx.st.goal_status_in_footer {
                if let (Some(u), Some(b)) = (ctx.st.goal_used, ctx.st.goal_budget) {
                    t = format!("{t} {u}/{b}");
                }
            }
            (t, false)
        }
        Some(Mode::Vibe) => (
            format!("{} Vibe", style::icon(theme, "icon.agents", "👥")),
            false,
        ),
        Some(Mode::Loop { label }) => (
            format!("{} {label}", style::icon(theme, "icon.loop", "↻")),
            false,
        ),
        None => return None,
    };
    let piece = if warning {
        Piece::new("warning", text)
    } else {
        ctx.maybe_accent("accent", text)
    };
    Some(vec![piece])
}

/// `path` — project dir display, `~` shortening, clamp (`segments.ts:566-599`).
fn path(ctx: &Ctx<'_>, theme: &crate::theme::Theme) -> Vec<Piece> {
    if theme.clean() {
        // Clean: the last path component only, no folder icon.
        let trimmed = ctx.st.cwd.trim_end_matches(['/', '\\']);
        let base = trimmed.rsplit(['/', '\\']).next().unwrap_or_default();
        let base = if base.is_empty() {
            ctx.st.cwd.as_str()
        } else {
            base
        };
        let shown = fmt::clamp_path_length(base, ctx.path_max);
        if ctx.st.worktree {
            let icon = style::icon(theme, "icon.worktree", "🌳");
            return vec![Piece::new("statusLinePath", format!("{icon} {shown}"))];
        }
        return vec![Piece::new("statusLinePath", shown)];
    }
    if ctx.st.worktree {
        let shown = fmt::clamp_path_length(&ctx.st.cwd, ctx.path_max);
        let icon = style::icon(theme, "icon.worktree", "🌳");
        return vec![Piece::new("statusLinePath", format!("{icon} {shown}"))];
    }
    let raw = ctx.st.cwd.as_str();
    let (scratch, display) = fmt::project_dir_display(raw);
    let mut shown = if ctx.st.path_abbreviate {
        fmt::shorten_path(&display)
    } else {
        display
    };
    shown = fmt::clamp_path_length(&shown, ctx.path_max);
    let icon = if scratch {
        style::icon(theme, "icon.scratchFolder", "🗑")
    } else {
        style::icon(theme, "icon.folder", "📁")
    };
    vec![Piece::new("statusLinePath", format!("{icon} {shown}"))]
}

/// `git` — branch + dirty counters (`segments.ts:627-695`).
fn git(ctx: &Ctx<'_>, theme: &crate::theme::Theme) -> Option<Vec<Piece>> {
    let git = ctx.st.git.as_ref()?;
    if git.branch.is_none() && git.staged == 0 && git.unstaged == 0 && git.untracked == 0 {
        return None;
    }
    let mut text = String::new();
    if let Some(branch) = &git.branch {
        let _ = write!(text, "{} {branch}", style::icon(theme, "icon.branch", "⑂"));
    }
    if git.unstaged > 0 {
        let _ = write!(text, " *{}", git.unstaged);
    }
    if git.staged > 0 {
        let _ = write!(text, " +{}", git.staged);
    }
    if git.untracked > 0 {
        let _ = write!(text, " ?{}", git.untracked);
    }
    let dirty = git.staged + git.unstaged + git.untracked > 0;
    Some(vec![Piece::new(
        if dirty {
            "statusLineGitDirty"
        } else {
            "statusLineGitClean"
        },
        text,
    )])
}

/// `pr` — pull request link (`segments.ts:697-712`).
fn pr(ctx: &Ctx<'_>, theme: &crate::theme::Theme) -> Option<Vec<Piece>> {
    let pr = ctx.st.git.as_ref()?.pr.as_ref()?;
    Some(vec![ctx.maybe_accent(
        "accent",
        format!("#{} {}", pr.number, style::icon(theme, "icon.pr", "⤴")),
    )])
}

/// `subagents` — background agent count (`segments.ts:714-727`).
fn subagents(ctx: &Ctx<'_>, theme: &crate::theme::Theme) -> Option<Vec<Piece>> {
    if ctx.st.subagents == 0 {
        return None;
    }
    Some(vec![Piece::new(
        ctx.tok("statusLineSubagents"),
        format!(
            "{} {}",
            style::icon(theme, "icon.agents", "👥"),
            ctx.st.subagents
        ),
    )])
}

/// `vim` — modal editing indicator (`segments.ts:1078-1132`).
fn vim(ctx: &Ctx<'_>, theme: &crate::theme::Theme) -> Option<Vec<Piece>> {
    let mode = ctx.st.vim?;
    let (glyph, token, label) = match mode {
        VimMode::Normal => ("icon.vimNormal", "accent", "NORMAL"),
        VimMode::Insert => ("icon.vimInsert", "success", "INSERT"),
        VimMode::Visual => ("icon.vimVisual", "warning", "VISUAL"),
        VimMode::VisualLine => ("icon.vimVisualLine", "warning", "V-LINE"),
    };
    let mut text = if ctx.st.vim_text_display {
        label.to_string()
    } else {
        style::icon(theme, glyph, label).to_string()
    };
    if ctx.st.vim_selection_lines > 1 {
        let _ = write!(text, " {}L", ctx.st.vim_selection_lines);
    }
    if !ctx.st.vim_pending.is_empty() {
        text.push(' ');
        text.push_str(&ctx.st.vim_pending);
    }
    Some(vec![Piece::new(token, text)])
}

/// `session_name` (`segments.ts:1021-1037`).
fn session_name(ctx: &Ctx<'_>) -> Option<Vec<Piece>> {
    let name = ctx.st.session_name.as_ref()?;
    if name.is_empty() {
        return None;
    }
    Some(vec![
        ctx.maybe_accent("accent", fmt::sanitize_status_text(name))
    ])
}

/// `hostname` (`segments.ts:970-982`).
fn hostname(ctx: &Ctx<'_>, theme: &crate::theme::Theme) -> Option<Vec<Piece>> {
    let host = ctx.st.hostname.as_ref()?;
    let short = host.split('.').next().unwrap_or(host);
    Some(vec![ctx.maybe_accent(
        "accent",
        format!("{} {short}", style::icon(theme, "icon.host", "🖥")),
    )])
}

/// `session` (`segments.ts:955-968`).
fn session(ctx: &Ctx<'_>, theme: &crate::theme::Theme) -> Vec<Piece> {
    let id = ctx.st.session_id.as_deref().unwrap_or("new");
    let short = if id == "new" {
        id
    } else {
        &id[..id.len().min(8)]
    };
    vec![Piece::new(
        "muted",
        format!("{} {short}", style::icon(theme, "icon.session", "🆔")),
    )]
}

/// `time` — clock (`segments.ts:931-953`).
fn time(ctx: &Ctx<'_>, theme: &crate::theme::Theme) -> Option<Vec<Piece>> {
    if !ctx.st.time_enabled {
        return None;
    }
    let secs = ctx.st.now_ms / 1000;
    let (h, m, s) = ((secs / 3600) % 24, (secs % 3600) / 60, secs % 60);
    let clock = if ctx.st.time_12h {
        let suffix = if h < 12 { "am" } else { "pm" };
        let h12 = if h % 12 == 0 { 12 } else { h % 12 };
        if ctx.st.time_show_seconds {
            format!("{h12}:{m:02}:{s:02}{suffix}")
        } else {
            format!("{h12}:{m:02}{suffix}")
        }
    } else if ctx.st.time_show_seconds {
        format!("{h}:{m:02}:{s:02}")
    } else {
        format!("{h}:{m:02}")
    };
    Some(vec![Piece::new(
        "muted",
        format!("{} {clock}", style::icon(theme, "icon.time", "⏱")),
    )])
}

/// `time_spent` — accumulated agent windows (`segments.ts:913-929`).
fn time_spent(ctx: &Ctx<'_>, theme: &crate::theme::Theme) -> Option<Vec<Piece>> {
    if ctx.st.active_ms < 1000 {
        return None;
    }
    Some(vec![Piece::new(
        "muted",
        format!(
            "{} {}",
            style::icon(theme, "icon.time", "⏱"),
            fmt::format_duration(ctx.st.active_ms)
        ),
    )])
}

/// `collab` (`segments.ts:1039-1053`).
fn collab(ctx: &Ctx<'_>) -> Option<Vec<Piece>> {
    let (guest, n) = ctx.st.collab?;
    let label = if guest {
        format!("⇄ collab guest:{n}")
    } else {
        format!("⇄ collab:{n}")
    };
    Some(vec![ctx.maybe_accent("accent", label)])
}

/// `stream` — live viewers / recording (`segments.ts:1055-1076`).
fn stream(ctx: &Ctx<'_>) -> Option<Vec<Piece>> {
    let mut parts = Vec::new();
    if let Some(v) = ctx.st.stream_viewers {
        parts.push(format!("● LIVE {v}"));
    }
    if ctx.st.recording {
        parts.push("● REC".to_string());
    }
    if parts.is_empty() {
        None
    } else {
        Some(vec![Piece::new("thinkingHigh", parts.join(" "))])
    }
}
