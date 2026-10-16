//! The bottom status line — the port of `pi-tui/src/status-line/`
//! (`component.ts`, `segments.ts`, `status-line-host.ts`).
//!
//! The bar is a configurable pipeline (§3–§5 of `chrome.md`): a preset
//! picks left/right [`SegmentId`]s, each segment renders styled [`Piece`]s,
//! the groups are joined by powerline separators with end caps, the gap
//! between them is the context-reactive gauge (§5.4), and an ordered
//! overflow pass shrinks/drops segments until the bar fits (§5.3).
//!
//! Rendered as the last row of the root layout by [`crate::app::App`]; hook
//! statuses (§5.5) print on the rows below when the app grants the space.

use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::Frame;

use crate::component::Component;
use crate::theme::Theme;

mod gauge;
mod metrics;
mod presets;
mod segments;
mod separators;
mod stats;
mod style;

pub use presets::{ContextLineMode, Preset, SegmentId, SeparatorStyle, StatusLineConfig};
use segments::{Ctx, Piece};
pub use stats::{ContextLevel, QuotaWindow};

use metrics::str_width;

/// Pull-request info for the `pr` segment.
#[derive(Debug, Clone)]
pub struct PrInfo {
    /// PR number.
    pub number: u64,
    /// PR URL (OSC 8 hyperlink target in TS; not emitted in ratatui).
    pub url: String,
}

/// Git state for the `git`/`pr` segments.
#[derive(Debug, Clone, Default)]
pub struct GitInfo {
    /// Current branch (`None` in a detached/no-branch repo).
    pub branch: Option<String>,
    /// Staged file count.
    pub staged: usize,
    /// Unstaged file count.
    pub unstaged: usize,
    /// Untracked file count.
    pub untracked: usize,
    /// Associated pull request, when known.
    pub pr: Option<PrInfo>,
}

/// Token/usage accounting for the stats segments.
#[derive(Debug, Clone, Default)]
pub struct TokenStats {
    /// Input tokens.
    pub input: u64,
    /// Output tokens.
    pub output: u64,
    /// Cache-read tokens.
    pub cache_read: u64,
    /// Cache-write tokens.
    pub cache_write: u64,
    /// Orchestration input tokens.
    pub orchestration_input: u64,
    /// Orchestration output tokens.
    pub orchestration_output: u64,
    /// Session cost in the billing unit.
    pub cost: f64,
    /// Subagent tree cost.
    pub subagent_cost: f64,
    /// Premium requests used.
    pub premium_requests: u64,
}

/// The active special mode (mode segment, `segments.ts:486-562`).
#[derive(Debug, Clone)]
#[allow(missing_docs)]
pub enum Mode {
    /// Plan mode (with paused flag).
    Plan { paused: bool },
    /// Prewalk.
    Prewalk,
    /// Goal mode.
    Goal,
    /// Vibe (multi-agent) mode.
    Vibe,
    /// Loop mode with a prebuilt label.
    Loop { label: String },
}

/// Vim modal state (vim segment).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(missing_docs)]
pub enum VimMode {
    Normal,
    Insert,
    Visual,
    VisualLine,
}

/// Auto-compact speculation state (`context_pct` segment).
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(missing_docs)]
pub enum Speculation {
    /// Summarization can run.
    Armed,
    /// Speculative summarization is running.
    Running,
}

/// What the status line knows about the session right now.
#[allow(clippy::struct_excessive_bools)] // mirrors the TS SegmentContext aggregate
#[derive(Debug, Clone)]
pub struct StatusLine {
    /// `provider/id` of the active model.
    pub model: String,
    /// Model display name (models.yml `name`); falls back to `model`.
    pub model_name: Option<String>,
    /// Working directory (display form, e.g. project root path).
    pub cwd: String,
    /// Whether the agent is streaming.
    pub streaming: bool,
    /// Queued steer/follow-up counts.
    pub queued: usize,

    // Session identity
    /// Display name of the session (drives the session-accent hash).
    pub session_name: Option<String>,
    /// Session id (session segment shows the first 8 chars).
    pub session_id: Option<String>,
    /// Hostname (hostname segment).
    pub hostname: Option<String>,
    /// Focused subagent id (pi segment shows the ghost icon).
    pub focused_agent: Option<String>,

    // Git
    /// Git state; `None` disables the git/pr segments.
    pub git: Option<GitInfo>,

    // Usage
    /// Token/cost accounting.
    pub tokens: TokenStats,
    /// Subagent tree cost (cost segment `(+x.xx)` tail).
    pub subagent_tree_cost: f64,
    /// Subscription billing (`S<amount>`).
    pub using_subscription: bool,
    /// Estimated tokens per second (`token_rate` segment).
    pub tokens_per_second: Option<f64>,

    // Context
    /// Context used, percent of the window.
    pub context_percent: Option<f64>,
    /// Context window size in tokens.
    pub context_window: Option<u64>,
    /// Context tokens used.
    pub context_tokens: Option<u64>,
    /// Auto-compact speculation state.
    pub speculation: Option<Speculation>,
    /// Current blink phase (toggled by [`StatusLine::tick`]).
    pub speculation_blink_on: bool,
    /// Auto-compaction threshold percent (gauge tick).
    pub compaction_pct: Option<f64>,
    /// Speculation start percent (gauge tick).
    pub speculation_pct: Option<f64>,
    /// Whether the compaction boundaries are known at all.
    pub boundaries_known: bool,

    // Modes
    /// Active special mode, if any.
    pub mode: Option<Mode>,
    /// Goal `used` counter (`goals.statusInFooter`).
    pub goal_used: Option<u64>,
    /// Goal `budget` counter.
    pub goal_budget: Option<u64>,
    /// Append `used/budget` to the Goal badge.
    pub goal_status_in_footer: bool,

    // Vim
    /// Active vim mode (`None` = vim disabled).
    pub vim: Option<VimMode>,
    /// Pending vim command echo (showcmd).
    pub vim_pending: String,
    /// Number of selected lines (`>1` appends ` <n>L`).
    pub vim_selection_lines: usize,
    /// Text display (`NORMAL`) instead of glyphs.
    pub vim_text_display: bool,

    // Model extras
    /// Thinking level (`off|minimal|low|medium|high|xhigh|max`).
    pub thinking: Option<String>,
    /// Auto-thinking is on.
    pub auto_thinking: bool,
    /// Fast mode is active.
    pub fast_mode: bool,

    // Agents / jobs
    /// Running subagent count.
    pub subagents: usize,
    /// Running background jobs (`⚙ <n>` badge).
    pub jobs: usize,
    /// Collab role: `(is_guest, peer_count)`.
    pub collab: Option<(bool, u32)>,
    /// Live stream viewer count.
    pub stream_viewers: Option<u32>,
    /// A stream is being recorded.
    pub recording: bool,

    // Timers
    /// Current turn elapsed time (drives the pi spinner + brand timer).
    pub turn_elapsed_ms: Option<u64>,
    /// Sum of `agent_start`→`agent_end` windows (`time_spent` segment).
    pub active_ms: u64,
    /// Wall clock for this frame, milliseconds (set by [`StatusLine::tick`]).
    pub now_ms: u64,

    // Provider quotas (usage segment).
    /// Quota windows to display.
    pub usage_windows: Vec<QuotaWindow>,

    // Hook statuses (status segment + hook lines, §5.5).
    /// Raw hook status lines.
    pub hook_statuses: Vec<String>,

    // Segment options (mirrored from `config` for direct field access).
    /// Compact thinking level on the model segment.
    pub compact_thinking: bool,
    /// Abbreviate the path segment.
    pub path_abbreviate: bool,
    /// 12h clock.
    pub time_12h: bool,
    /// Show clock seconds.
    pub time_show_seconds: bool,
    /// Show the clock segment at all.
    pub time_enabled: bool,
    /// The cwd is a worktree (path segment icon).
    pub worktree: bool,

    /// Status line settings (preset, separators, gauge, ...).
    pub config: StatusLineConfig,
}

impl StatusLine {
    /// Build the initial status from [`crate::RunConfig`] data.
    #[must_use]
    pub fn new(model: String, cwd: impl Into<String>) -> Self {
        Self {
            model,
            model_name: None,
            cwd: cwd.into(),
            streaming: false,
            queued: 0,
            session_name: None,
            session_id: None,
            hostname: None,
            focused_agent: None,
            git: None,
            tokens: TokenStats::default(),
            subagent_tree_cost: 0.0,
            using_subscription: false,
            tokens_per_second: None,
            context_percent: None,
            context_window: None,
            context_tokens: None,
            speculation: None,
            speculation_blink_on: false,
            compaction_pct: None,
            speculation_pct: None,
            boundaries_known: false,
            mode: None,
            goal_used: None,
            goal_budget: None,
            goal_status_in_footer: false,
            vim: None,
            vim_pending: String::new(),
            vim_selection_lines: 0,
            vim_text_display: false,
            thinking: None,
            auto_thinking: false,
            fast_mode: false,
            subagents: 0,
            jobs: 0,
            collab: None,
            stream_viewers: None,
            recording: false,
            turn_elapsed_ms: None,
            active_ms: 0,
            now_ms: 0,
            usage_windows: Vec::new(),
            hook_statuses: Vec::new(),
            compact_thinking: true,
            path_abbreviate: true,
            time_12h: false,
            time_show_seconds: false,
            time_enabled: true,
            worktree: false,
            config: StatusLineConfig::default(),
        }
    }

    /// Advance live timers (§5.6): wall clock and the speculation blink.
    /// The app calls this once per draw tick.
    pub fn tick(&mut self, now_ms: u64) {
        self.now_ms = now_ms;
        // Blink interval ~500ms, phase-exact.
        self.speculation_blink_on = (now_ms / 500).is_multiple_of(2);
    }

    /// Session-accent color (§7.6): stable hash of the session name → hue.
    /// `None` when the session is unnamed or the feature is off.
    #[must_use]
    pub fn session_accent(&self) -> Option<Color> {
        if !self.config.session_accent {
            return None;
        }
        let name = self.session_name.as_ref()?;
        Some(session_accent_color(name))
    }
}

/// FNV-1a hash of the name → hue → saturated light color.
#[must_use]
pub fn session_accent_color(name: &str) -> Color {
    let mut hash: u32 = 0x811C_9DC5;
    for b in name.as_bytes() {
        hash ^= u32::from(*b);
        hash = hash.wrapping_mul(0x0100_0193);
    }
    let hue = f64::from(hash % 360);
    hsl_to_rgb(hue / 360.0, 0.65, 0.60)
}

/// HSL → RGB (saturated mid-light colors read well on dark terminals).
#[allow(clippy::many_single_char_names)] // channel math
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // clamped 0.0..=1.0 channels
fn hsl_to_rgb(h: f64, s: f64, l: f64) -> Color {
    fn hue_to_rgb(p: f64, q: f64, mut t: f64) -> f64 {
        if t < 0.0 {
            t += 1.0;
        }
        if t > 1.0 {
            t -= 1.0;
        }
        if t < 1.0 / 6.0 {
            return p + (q - p) * 6.0 * t;
        }
        if t < 1.0 / 2.0 {
            return q;
        }
        if t < 2.0 / 3.0 {
            return p + (q - p) * (2.0 / 3.0 - t) * 6.0;
        }
        p
    }
    let q = if l < 0.5 {
        l * (1.0 + s)
    } else {
        l + s - l * s
    };
    let p = 2.0 * l - q;
    let r = hue_to_rgb(p, q, h + 1.0 / 3.0);
    let g = hue_to_rgb(p, q, h);
    let b = hue_to_rgb(p, q, h - 1.0 / 3.0);
    Color::Rgb(
        (r * 255.0).round() as u8,
        (g * 255.0).round() as u8,
        (b * 255.0).round() as u8,
    )
}

/// A rendered group: styled spans plus its cell width.
struct Group {
    spans: Vec<Span<'static>>,
    width: usize,
}

/// Paint one side's pieces into a group (§5.2): bg + fg body, separators
/// between pieces, powerline end caps unless transparent.
fn paint_group(
    pieces: &[Piece],
    seps: &separators::Separators,
    right_side: bool,
    theme: &Theme,
    transparent: bool,
) -> Group {
    let bg = style::bg(theme);
    let bg_style = if transparent {
        Style::default()
    } else {
        Style::default().bg(bg)
    };
    let sep_color = style::color(theme, "statusLineSep");
    let sep = if right_side { seps.right } else { seps.left };
    let sep_span = Span::styled(sep.to_string(), bg_style.fg(sep_color));

    let mut spans: Vec<Span<'static>> = Vec::new();
    let mut width = 0usize;
    if !pieces.is_empty() {
        // Body: leading pad, pieces joined by " sep ", trailing pad.
        spans.push(Span::styled(" ".to_string(), bg_style));
        width += 1;
        for (i, piece) in pieces.iter().enumerate() {
            if i > 0 {
                spans.push(Span::styled(" ".to_string(), bg_style));
                spans.push(sep_span.clone());
                spans.push(Span::styled(" ".to_string(), bg_style));
                width += str_width(sep) + 2;
            }
            spans.push(Span::styled(
                piece.text.clone(),
                bg_style.fg(style::color(theme, piece.token)),
            ));
            width += str_width(&piece.text);
        }
        spans.push(Span::styled(" ".to_string(), bg_style));
        width += 1;
    }

    // End caps: left group closes right, right group opens left (§5.2).
    if seps.use_bg_as_fg && !transparent && !pieces.is_empty() {
        let cap = if right_side {
            seps.cap_left
        } else {
            seps.cap_right
        };
        if !cap.is_empty() {
            // Cap is the group bg recolored to fg on the terminal bg.
            spans.push(Span::styled(cap.to_string(), Style::default().fg(bg)));
            width += str_width(cap);
        }
    }
    Group { spans, width }
}

impl Component for StatusLine {
    fn render(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        let width = area.width as usize;
        if width == 0 || area.height == 0 {
            return;
        }
        let accent = self.session_accent();
        let ctx = Ctx {
            st: self,
            accent,
            path_max: self.config.path_max_length,
        };
        let gauge_input = gauge::GaugeInput {
            percent: self.context_percent,
            window: self.context_window,
            tokens: self.context_tokens,
            mode: self.config.context_line,
            compaction_pct: self.compaction_pct,
            speculation_pct: self.speculation_pct,
            accent,
            boundaries_known: self.boundaries_known,
            embed: false,
        };
        let line = layout(&ctx, theme, width, &gauge_input);
        frame.render_widget(Paragraph::new(line), Rect { height: 1, ..area });

        // §5.5 hook status lines on the rows below.
        if self.config.show_hook_status {
            for (row, status) in self.hook_statuses.iter().enumerate() {
                let Some(y) = u16::try_from(row + 1).ok() else {
                    break;
                };
                if y >= area.height {
                    break;
                }
                let text = metrics::sanitize_status_text(status);
                let truncated: String = text.chars().take(width).collect();
                let row_area = Rect {
                    x: area.x,
                    y: area.y + y,
                    width: area.width,
                    height: 1,
                };
                frame.render_widget(
                    Paragraph::new(Line::from(Span::styled(
                        truncated,
                        Style::default().fg(style::color(theme, "muted")),
                    ))),
                    row_area,
                );
            }
        }
    }
}

/// The full layout pass: resolve segments, run the overflow algorithm
/// (§5.3), paint the groups and join them with the gauge (§5.4). The right
/// group is flush right; the gauge (plus padding) fills the gap.
fn layout(
    ctx: &Ctx<'_>,
    theme: &Theme,
    width: usize,
    gauge_input: &gauge::GaugeInput,
) -> Line<'static> {
    let (mut left_ids, mut right_ids) = match ctx.st.config.preset {
        presets::Preset::Custom => (
            ctx.st
                .config
                .left_segments
                .clone()
                .unwrap_or_else(|| presets::CUSTOM_LEFT.to_vec()),
            ctx.st
                .config
                .right_segments
                .clone()
                .unwrap_or_else(|| presets::CUSTOM_RIGHT.to_vec()),
        ),
        p => p.segments(),
    };
    // `embedContext`: with a known window and at least one non-context
    // segment, the gauge absorbs the context segments (percent + window
    // labels ride inside it).
    let is_context = |id: &SegmentId| matches!(id, SegmentId::ContextPct | SegmentId::ContextTotal);
    let embed = ctx.st.config.context_line == presets::ContextLineMode::Embedded
        && ctx.st.context_window.unwrap_or(0) > 0
        && left_ids.iter().chain(right_ids.iter()).any(is_context)
        && left_ids.iter().chain(right_ids.iter()).any(|id| !is_context(id));
    if embed {
        left_ids.retain(|id| !is_context(id));
        right_ids.retain(|id| !is_context(id));
    }
    let mut gauge_input = *gauge_input;
    gauge_input.embed = embed;
    let gauge_input = &gauge_input;
    let sep_style = ctx
        .st
        .config
        .preset
        .separator()
        .unwrap_or(ctx.st.config.separator);
    let seps = separators::Separators::for_style(sep_style);
    let transparent = ctx.st.config.transparent;

    // 1. The right group also carries badges that are not registry
    //    segments: the `⚙ <n>` background-jobs badge (§5.3 step 1).
    let jobs_badge = ctx.st.jobs > 0;

    let mut path_max = ctx.path_max;
    let mut name_floor = usize::MAX; // session_name elastic budget
    let mut left: Vec<Vec<Piece>> = Vec::new();
    let mut right: Vec<Vec<Piece>> = Vec::new();
    for _attempt in 0..128 {
        left = render_side(&left_ids, ctx, theme, path_max);
        right = render_side(&right_ids, ctx, theme, path_max);
        if jobs_badge {
            right.insert(
                0,
                vec![Piece::new(
                    "muted",
                    format!("{} {}", style::icon(theme, "icon.job", "⚙"), ctx.st.jobs),
                )],
            );
        }
        if shrink_session_name(&mut right, ctx, &mut name_floor) {
            continue;
        }
        let left_w = group_width(&left, &seps, false, theme);
        let right_w = group_width(&right, &seps, true, theme);
        if left_w + right_w + gauge::embedded_min_width(gauge_input) <= width {
            break;
        }
        // 2. Pop right segments from the end (the badge is gone first).
        if right.len() > 1 {
            right.pop();
            continue;
        }
        // 3. Shrink `path` (min 8 cells, maxLength ≥ 4).
        let path_w = left
            .iter()
            .flatten()
            .find(|p| p.token == "statusLinePath")
            .map_or(0, |p| str_width(&p.text));
        if path_w > 8 && path_max > 4 {
            path_max = path_max.saturating_sub(4).max(4);
            continue;
        }
        // 4. Drop left segments right-to-left, never `path` first (§5.3).
        if !drop_left(&mut left_ids) {
            break;
        }
    }

    // Final paint: left group at x0, right group flush right, gauge in the
    // middle (widened by any leftover cells).
    let left_group = paint_group(&flatten(&left), &seps, false, theme, transparent);
    let right_group = paint_group(&flatten(&right), &seps, true, theme, transparent);
    let gauge_w = width
        .saturating_sub(left_group.width + right_group.width)
        .max(1);
    let pad_w = width.saturating_sub(left_group.width + gauge_w + right_group.width);

    let mut spans = left_group.spans;
    spans.extend(gauge::render(gauge_input, gauge_w, theme));
    if pad_w > 0 {
        let pad_style = if transparent {
            Style::default()
        } else {
            Style::default().bg(style::bg(theme))
        };
        spans.extend(
            std::iter::repeat_with(|| Span::styled(" ".to_string(), pad_style)).take(pad_w),
        );
    }
    spans.extend(right_group.spans);
    Line::from(spans)
}

/// Render each segment id to its pieces (`None` → omitted).
fn render_side(
    ids: &[SegmentId],
    ctx: &Ctx<'_>,
    theme: &Theme,
    path_max: usize,
) -> Vec<Vec<Piece>> {
    let mut side_ctx = *ctx;
    side_ctx.path_max = path_max;
    ids.iter()
        .filter_map(|id| segments::render(*id, &side_ctx, theme))
        .collect()
}

fn flatten(groups: &[Vec<Piece>]) -> Vec<Piece> {
    groups.iter().flatten().cloned().collect()
}

/// Width accounting (§5.2): pieces + separators + padding + caps.
fn group_width(
    groups: &[Vec<Piece>],
    seps: &separators::Separators,
    right: bool,
    theme: &Theme,
) -> usize {
    paint_group(&flatten(groups), seps, right, theme, true).width
}

/// Shrink the `session_name` piece toward its 8-cell floor (§5.3 step 2);
/// `true` while progress is still being made.
fn shrink_session_name(right: &mut [Vec<Piece>], ctx: &Ctx<'_>, floor: &mut usize) -> bool {
    let target = ctx.st.session_name.as_deref().unwrap_or("");
    if target.is_empty() {
        return false;
    }
    for pieces in right.iter_mut() {
        for piece in pieces.iter_mut() {
            if !piece.text.contains(target) {
                continue;
            }
            let w = str_width(&piece.text);
            let current = if *floor == usize::MAX { w } else { *floor };
            if current > 8 {
                *floor = current.saturating_sub(8).max(8);
                piece.text = clamp_cells(&piece.text, *floor);
                return true;
            }
        }
    }
    false
}

/// Truncate `s` to at most `max` display cells, `…`-marked.
fn clamp_cells(s: &str, max: usize) -> String {
    if str_width(s) <= max {
        return s.to_string();
    }
    let mut out = String::from("…");
    let mut w = 1usize;
    for c in s.chars() {
        let cw = metrics::char_width(c);
        if w + cw > max {
            break;
        }
        out.push(c);
        w += cw;
    }
    out
}

/// Drop the last removable left segment — never `path` first (§5.3 step 4);
/// `false` when nothing removable remains.
fn drop_left(left_ids: &mut Vec<SegmentId>) -> bool {
    if let Some(pos) = left_ids.iter().rposition(|id| *id != SegmentId::Path) {
        left_ids.remove(pos);
        return true;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    fn draw(status: &StatusLine, width: u16) -> String {
        let backend = TestBackend::new(width, 1);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|f| status.render(f, f.area(), &Theme::default()))
            .unwrap();
        let buf = terminal.backend().buffer();
        (0..width).map(|x| buf[(x, 0)].symbol()).collect()
    }

    #[test]
    fn renders_default_preset_segments() {
        let mut status = StatusLine::new("zai/glm".into(), "/work/omp-rs");
        status.git = Some(GitInfo {
            branch: Some("master".into()),
            unstaged: 3,
            staged: 1,
            untracked: 0,
            pr: None,
        });
        status.context_percent = Some(23.0);
        status.context_window = Some(200_000);
        let row = draw(&status, 120);
        assert!(row.contains('π'));
        assert!(row.contains("zai/glm"));
        assert!(row.contains("omp-rs"));
        assert!(row.contains("master"));
        assert!(row.contains("*3"));
        // Embedded gauge: percent and window labels ride inside the line.
        assert!(row.contains("23%"), "{row:?}");
        assert!(row.contains("200K"), "{row:?}");
    }

    #[test]
    fn box_layout_right_aligns_session_name() {
        let mut status = StatusLine::new("m".into(), "/repo");
        status.session_name = Some("my-session".into());
        let row = draw(&status, 80);
        assert!(row.contains("my-session"));
    }

    #[test]
    fn overflow_keeps_path_and_drops_others() {
        let mut status = StatusLine::new(
            "anthropic/claude-opus-5-5".into(),
            "/very/long/project/path",
        );
        status.git = Some(GitInfo {
            branch: Some("feature/very-long-branch-name".into()),
            unstaged: 42,
            staged: 7,
            untracked: 3,
            pr: Some(PrInfo {
                number: 123,
                url: "http://x".into(),
            }),
        });
        status.context_percent = Some(88.0);
        status.context_window = Some(1_000_000);
        let row = draw(&status, 60);
        // The path survives (clamped to its 8-cell floor) while looser
        // segments are dropped.
        assert!(row.contains("ath"), "row: {row:?}");
        // `draw` joins buffer cells, so a wide emoji adds its blank trailing
        // cell: the bar fills exactly 60 cells.
        assert_eq!(row.chars().count(), 60);
    }

    #[test]
    fn spinner_replaces_brand_icon_while_turn_runs() {
        let mut status = StatusLine::new("m".into(), "/repo");
        status.turn_elapsed_ms = Some(12_000);
        status.tick(0);
        let row = draw(&status, 60);
        assert!(row.contains("12s"));
        assert!(!row.contains('π'));
    }

    #[test]
    fn session_accent_is_stable() {
        let a = session_accent_color("Продолжить разработку");
        let b = session_accent_color("Продолжить разработку");
        assert_eq!(a, b);
        assert_ne!(session_accent_color("other"), a);
    }

    #[test]
    fn hook_statuses_render_below() {
        let mut status = StatusLine::new("m".into(), "/repo");
        status.hook_statuses = vec!["running tests…".into()];
        let backend = TestBackend::new(60, 2);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|f| status.render(f, f.area(), &Theme::default()))
            .unwrap();
        let buf = terminal.backend().buffer();
        let row: String = (0..60).map(|x| buf[(x, 1)].symbol()).collect();
        assert!(row.contains("running tests"));
    }
}
