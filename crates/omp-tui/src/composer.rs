//! Composer shapes — the port of `pi-tui/src/components/composer/*.ts`
//! (`ComposerStyle`) and `overlays/composer-shape-registry.ts`.
//!
//! A shape owns everything about how the prompt editor's frame looks:
//! top/bottom chrome, per-row side chrome, the default prompt gutter and
//! padding, plus where the status line attaches (embedded in the top border,
//! a flush band above, docked onto a top rule, or a standalone bottom bar).
//! The live editor ([`crate::editor::Editor`]) and the `/settings` shape
//! preview ([`preview_lines`]) render through the same functions, so the two
//! surfaces cannot drift apart.

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

use crate::editor::layout::display_width;
use crate::status_line::{StatusLayout, StatusLine};
use crate::theme::Theme;

/// A built-in composer shape (`BUILTIN_EDITOR_BORDER_STYLES`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ComposerShape {
    /// Flush soft-capped status band above a curved prompt, no frame (default).
    #[default]
    Band,
    /// Rounded frame, status embedded in the top border.
    Box,
    /// Full-width rules above/below a `❯ ` prompt, status at the bottom.
    Claude,
    /// Framed horizontal rules, status at the bottom.
    Pi,
    /// Bare `❯ ` prompt, status at the bottom.
    Borderless,
    /// Single top rule with the status docked onto it.
    Rule,
    /// Filled one-row field with accent end caps.
    Field,
    /// Filled one-row field anchored by a single accent rail.
    Rail,
}

/// Where the host attaches the status bar (`ComposerStyle.statusAttachment`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatusAttachment {
    /// Embedded in the top border.
    TopBorder,
    /// Flush soft-capped band above the input.
    TopBand,
    /// Docked onto the top rule (right group).
    TopRuleChip,
    /// No top attachment.
    None,
}

/// Which segment groups the standalone bottom bar shows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BottomBar {
    /// No standalone bar.
    None,
    /// Left group only.
    Left,
    /// Both groups.
    Full,
}

/// Pre-rendered status content injected into the top chrome.
#[derive(Debug, Clone, Default)]
pub struct TopBorder {
    /// Styled status spans.
    pub spans: Vec<Span<'static>>,
    /// Visible width of `spans`.
    pub width: usize,
}

/// Inputs shared by every chrome row (`ComposerChromeContext`).
#[derive(Debug, Clone)]
pub struct Chrome {
    /// Full width available to the composer.
    pub width: usize,
    /// Horizontal padding inside the side chrome.
    pub padding_x: usize,
    /// Border / rule style.
    pub border: Style,
    /// Stable accent (field caps, rails).
    pub accent: Style,
    /// Fill style of filled surfaces (background + foreground).
    pub surface: Style,
    /// Status content for the top chrome.
    pub top: Option<TopBorder>,
}

/// One content row (`ComposerRowContext`).
#[derive(Debug, Clone)]
pub struct Row {
    /// Fully decorated row text.
    pub text: Vec<Span<'static>>,
    /// Spaces padding the text out to the content width.
    pub pad: usize,
    /// Prompt gutter cell(s) for this row.
    pub gutter: Option<Span<'static>>,
    /// Whether this is the last visible row.
    pub is_last: bool,
}

const H: &str = "─";

fn spaces(n: usize) -> Span<'static> {
    Span::raw(" ".repeat(n))
}

fn rule(style: Style, n: usize) -> Span<'static> {
    Span::styled(H.repeat(n), style)
}

/// Truncate styled spans to `max` cells.
fn truncate_spans(spans: &[Span<'static>], max: usize) -> Vec<Span<'static>> {
    let mut out = Vec::new();
    let mut used = 0usize;
    for span in spans {
        if used >= max {
            break;
        }
        let w = display_width(&span.content);
        if used + w <= max {
            out.push(span.clone());
            used += w;
            continue;
        }
        let mut text = String::new();
        for ch in span.content.chars() {
            let cw = display_width(ch.encode_utf8(&mut [0; 4]));
            if used + cw > max {
                break;
            }
            text.push(ch);
            used += cw;
        }
        out.push(Span::styled(text, span.style));
        break;
    }
    out
}

impl ComposerShape {
    /// Every built-in shape in selector order (`BUILTIN_COMPOSER_SHAPES`).
    pub const ALL: &'static [Self] = &[
        Self::Band,
        Self::Box,
        Self::Claude,
        Self::Pi,
        Self::Borderless,
        Self::Rule,
        Self::Field,
        Self::Rail,
    ];

    /// Settings value (`composer.shape`).
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Band => "band",
            Self::Box => "box",
            Self::Claude => "claude",
            Self::Pi => "pi",
            Self::Borderless => "borderless",
            Self::Rule => "rule",
            Self::Field => "field",
            Self::Rail => "rail",
        }
    }

    /// Selector label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Band => "Status Band (Default)",
            Self::Box => "Rounded Box",
            Self::Claude => "Claude Code",
            Self::Pi => "Pi",
            Self::Borderless => "Borderless",
            Self::Rule => "Top Rule Dock",
            Self::Field => "Compact Field",
            Self::Rail => "Accent Rail",
        }
    }

    /// Selector description.
    #[must_use]
    pub const fn description(self) -> &'static str {
        match self {
            Self::Band => "Flush soft-capped status band above a curved prompt, no frame",
            Self::Box => "Status line embedded in top border, compact 2-line prompt",
            Self::Claude => "Full-width horizontal rules above and below, status line at bottom",
            Self::Pi => "Framed horizontal rules with status line at bottom",
            Self::Borderless => "Clean prompt glyph with status line at bottom, no box borders",
            Self::Rule => "Single top rule with status docked onto it and below",
            Self::Field => "Filled one-row field with accent end caps",
            Self::Rail => "Filled one-row field anchored by a single accent rail",
        }
    }

    /// Resolve a settings value; unknown ids fall back to `box` like
    /// `getComposerStyle`.
    #[must_use]
    pub fn from_name(name: &str) -> Self {
        Self::ALL
            .iter()
            .copied()
            .find(|shape| shape.name() == name)
            .unwrap_or(Self::Box)
    }

    /// Content rows carry left/right border glyphs.
    #[must_use]
    pub const fn side_borders(self) -> bool {
        matches!(self, Self::Box | Self::Field | Self::Rail)
    }

    /// Rows consumed by top + bottom chrome (`verticalChrome`).
    #[must_use]
    pub const fn vertical_chrome(self) -> u16 {
        match self {
            Self::Box | Self::Claude | Self::Pi => 2,
            Self::Band | Self::Rule => 1,
            Self::Borderless | Self::Field | Self::Rail => 0,
        }
    }

    /// Whether the shape renders a top chrome row.
    #[must_use]
    pub const fn has_top(self) -> bool {
        matches!(
            self,
            Self::Band | Self::Box | Self::Claude | Self::Pi | Self::Rule
        )
    }

    /// Whether the shape renders a separate bottom chrome row (box merges
    /// its bottom border into the last content row).
    #[must_use]
    pub const fn has_bottom(self) -> bool {
        matches!(self, Self::Claude | Self::Pi)
    }

    /// Where the status bar attaches.
    #[must_use]
    pub const fn status_attachment(self) -> StatusAttachment {
        match self {
            Self::Box => StatusAttachment::TopBorder,
            Self::Band => StatusAttachment::TopBand,
            Self::Claude | Self::Rule => StatusAttachment::TopRuleChip,
            Self::Pi | Self::Borderless | Self::Field | Self::Rail => StatusAttachment::None,
        }
    }

    /// Which groups the standalone bottom bar shows.
    #[must_use]
    pub const fn bottom_bar(self) -> BottomBar {
        match self {
            Self::Band | Self::Box => BottomBar::None,
            Self::Claude | Self::Rule => BottomBar::Left,
            Self::Pi | Self::Borderless | Self::Field | Self::Rail => BottomBar::Full,
        }
    }

    /// Blank spacer row between the editor and the standalone bar.
    #[must_use]
    pub const fn bottom_bar_gap(self) -> bool {
        matches!(self, Self::Rule | Self::Field | Self::Rail)
    }

    /// Default prompt gutter.
    #[must_use]
    pub const fn default_prompt_gutter(self) -> Option<&'static str> {
        match self {
            Self::Band => Some("╰─ "),
            Self::Claude | Self::Borderless | Self::Rule => Some("❯ "),
            Self::Box | Self::Pi | Self::Field | Self::Rail => None,
        }
    }

    /// Default horizontal padding (theme `editorPaddingX` unset).
    #[must_use]
    pub const fn default_padding_x(self) -> usize {
        match self {
            Self::Box => 2,
            Self::Pi | Self::Field | Self::Rail => 1,
            Self::Band | Self::Claude | Self::Borderless | Self::Rule => 0,
        }
    }

    /// Cells consumed per side on content rows (border glyph + padding).
    #[must_use]
    pub const fn side_chrome_width(self, padding_x: usize) -> usize {
        match self {
            Self::Band | Self::Borderless => 0,
            Self::Box | Self::Field | Self::Rail => padding_x + 1,
            Self::Claude | Self::Pi | Self::Rule => padding_x,
        }
    }

    /// Whether rows paint their own filled surface.
    #[must_use]
    pub const fn filled_surface(self) -> bool {
        matches!(self, Self::Field | Self::Rail)
    }

    /// Top chrome row; `None` renders none.
    #[must_use]
    pub fn render_top(self, ctx: &Chrome) -> Option<Line<'static>> {
        match self {
            Self::Band => {
                // The band always owns one chrome row; keep it reserved while
                // the status line has nothing to show.
                let Some(top) = ctx.top.as_ref().filter(|t| t.width > 0) else {
                    return Some(Line::default());
                };
                let spans = if top.width > ctx.width {
                    truncate_spans(&top.spans, ctx.width)
                } else {
                    top.spans.clone()
                };
                Some(Line::from(spans))
            }
            Self::Box => {
                let pad = ctx.padding_x;
                let fill_width = ctx.width.saturating_sub(self.side_chrome_width(pad) * 2);
                let mut spans = vec![Span::styled(format!("╭{}", H.repeat(pad)), ctx.border)];
                match &ctx.top {
                    None => spans.push(rule(ctx.border, fill_width)),
                    Some(top) if top.width <= fill_width => {
                        spans.extend(top.spans.clone());
                        spans.push(rule(ctx.border, fill_width - top.width));
                    }
                    Some(top) => {
                        let truncated = truncate_spans(&top.spans, fill_width.saturating_sub(1));
                        let used: usize = truncated.iter().map(|s| display_width(&s.content)).sum();
                        spans.extend(truncated);
                        spans.push(rule(ctx.border, fill_width.saturating_sub(used)));
                    }
                }
                spans.push(Span::styled(format!("{}╮", H.repeat(pad)), ctx.border));
                Some(Line::from(spans))
            }
            Self::Claude | Self::Rule => Some(render_top_rule(ctx)),
            Self::Pi => Some(Line::from(rule(ctx.border, ctx.width))),
            Self::Borderless | Self::Field | Self::Rail => None,
        }
    }

    /// Chrome-wrapped content row.
    #[must_use]
    pub fn render_row(self, ctx: &Chrome, row: &Row) -> Line<'static> {
        let pad = ctx.padding_x;
        let mut spans: Vec<Span<'static>> = Vec::new();
        match self {
            Self::Band | Self::Claude | Self::Borderless | Self::Rule => {
                spans.extend(row.gutter.clone());
                spans.extend(row.text.clone());
                spans.push(spaces(row.pad));
            }
            Self::Pi => {
                spans.push(spaces(self.side_chrome_width(pad)));
                spans.extend(row.gutter.clone());
                spans.extend(row.text.clone());
                spans.push(spaces(row.pad));
            }
            Self::Box => {
                let right_cells = (pad + 1).max(1);
                if row.is_last {
                    spans.push(Span::styled(
                        format!("╰{H}{}", " ".repeat(pad.saturating_sub(1))),
                        ctx.border,
                    ));
                    spans.extend(row.text.clone());
                    spans.push(spaces(row.pad));
                    let right_pad = right_cells.saturating_sub(2);
                    let horizontal = if right_cells >= 2 { H } else { "" };
                    spans.push(Span::styled(
                        format!("{}{horizontal}╯", " ".repeat(right_pad)),
                        ctx.border,
                    ));
                } else {
                    spans.push(Span::styled(format!("│{}", " ".repeat(pad)), ctx.border));
                    spans.extend(row.text.clone());
                    spans.push(spaces(row.pad));
                    spans.push(Span::styled(
                        format!("{}│", " ".repeat(right_cells.saturating_sub(1))),
                        ctx.border,
                    ));
                }
            }
            Self::Field => {
                spans.push(Span::styled("▐", ctx.accent));
                let mut interior = vec![spaces(pad)];
                interior.extend(row.gutter.clone());
                interior.extend(row.text.clone());
                interior.push(spaces(row.pad + pad));
                spans.extend(interior.into_iter().map(|s| restyle(s, ctx.surface)));
                spans.push(Span::styled("▌", ctx.accent));
            }
            Self::Rail => {
                spans.push(Span::styled("▎", ctx.accent));
                let mut interior = vec![spaces(pad)];
                interior.extend(row.gutter.clone());
                interior.extend(row.text.clone());
                interior.push(spaces(row.pad + pad + 1));
                spans.extend(interior.into_iter().map(|s| restyle(s, ctx.surface)));
            }
        }
        Line::from(spans)
    }

    /// Bottom chrome row; `None` renders none.
    #[must_use]
    pub fn render_bottom(self, ctx: &Chrome) -> Option<Line<'static>> {
        match self {
            Self::Claude | Self::Pi => Some(Line::from(rule(ctx.border, ctx.width))),
            _ => None,
        }
    }
}

/// Lay the surface style under a span (the span's own foreground wins).
fn restyle(span: Span<'static>, surface: Style) -> Span<'static> {
    let style = surface.patch(span.style);
    Span::styled(span.content, style)
}

/// Full-width rule with the status chip docked at its right edge
/// (`renderTopRule`).
fn render_top_rule(ctx: &Chrome) -> Line<'static> {
    let width = ctx.width;
    if let Some(top) = ctx.top.as_ref().filter(|t| t.width > 0) {
        if width > 2 {
            let (spans, chip_w) = if top.width > width - 2 {
                let spans = truncate_spans(&top.spans, width - 2);
                let w = spans.iter().map(|s| display_width(&s.content)).sum();
                (spans, w)
            } else {
                (top.spans.clone(), top.width)
            };
            let left_fill = width.saturating_sub(chip_w + 1);
            let mut out = vec![rule(ctx.border, left_fill)];
            out.extend(spans);
            out.push(rule(ctx.border, 1));
            return Line::from(out);
        }
    }
    Line::from(rule(ctx.border, width))
}

/// The theme styles composer chrome uses.
#[must_use]
pub fn chrome_styles(theme: &Theme, border: Style) -> (Style, Style) {
    let accent = Style::default().fg(theme.color("accent"));
    let surface = {
        let bg = theme.color("userMessageBg");
        let fg = theme.color("userMessageText");
        let mut style = Style::default();
        if bg != Color::Reset {
            style = style.bg(bg);
        }
        if fg != Color::Reset {
            style = style.fg(fg);
        }
        style
    };
    let _ = border;
    (accent, surface)
}

/// The shape's status top border for `width` columns (`attachToEditor`
/// providers): `None` when the status line does not attach at the top.
#[must_use]
pub fn top_border(
    shape: ComposerShape,
    status: &StatusLine,
    theme: &Theme,
    width: usize,
) -> Option<TopBorder> {
    let chrome = shape.side_chrome_width(shape.default_padding_x());
    let (layout, avail) = match shape.status_attachment() {
        StatusAttachment::TopBorder => (StatusLayout::Box, width.saturating_sub(chrome * 2).max(1)),
        StatusAttachment::TopBand => (StatusLayout::Band, width),
        StatusAttachment::TopRuleChip => (StatusLayout::PlainRight, width),
        StatusAttachment::None => return None,
    };
    let (spans, w) = status.render_layout(theme, avail, layout)?;
    Some(TopBorder { spans, width: w })
}

/// The standalone bottom bar for `shape`, if it has one.
#[must_use]
pub fn bottom_bar(
    shape: ComposerShape,
    status: &StatusLine,
    theme: &Theme,
    width: usize,
) -> Option<Line<'static>> {
    let layout = match shape.bottom_bar() {
        BottomBar::None => return None,
        BottomBar::Left => StatusLayout::PlainLeft,
        BottomBar::Full => StatusLayout::PlainFull,
    };
    let (spans, _) = status.render_layout(theme, width, layout)?;
    Some(Line::from(spans))
}

/// Stand-in prompt of the shape preview.
const PREVIEW_PROMPT: &str = "Ask anything, edit files, run tools";

/// The composer as `shape` renders it around a stand-in prompt, with the live
/// status rows (`renderComposerShapePreview`).
#[must_use]
pub fn preview_lines(
    shape: ComposerShape,
    width: usize,
    theme: &Theme,
    status: &StatusLine,
) -> Vec<Line<'static>> {
    let width = width.max(24);
    let padding_x = shape.default_padding_x();
    let chrome_w = shape.side_chrome_width(padding_x);
    let border = Style::default().fg(theme.color("borderAccent"));
    let (accent, surface) = chrome_styles(theme, border);
    let ctx = Chrome {
        width,
        padding_x,
        border,
        accent,
        surface,
        top: top_border(shape, status, theme, width),
    };
    let gutter = shape.default_prompt_gutter().unwrap_or("");
    let content_width = width
        .saturating_sub(chrome_w * 2 + display_width(gutter))
        .max(1);
    let prompt: String = PREVIEW_PROMPT
        .chars()
        .take(content_width.saturating_sub(1).max(1))
        .collect();
    let used = display_width(&prompt) + 1;
    let text_style = if shape.filled_surface() {
        Style::default()
    } else {
        Style::default().fg(theme.color("text"))
    };
    let row = Row {
        text: vec![
            Span::styled(prompt, text_style),
            Span::styled(" ", Style::default().add_modifier(Modifier::REVERSED)),
        ],
        pad: content_width.saturating_sub(used),
        gutter: (!gutter.is_empty()).then(|| Span::styled(gutter.to_owned(), accent)),
        is_last: true,
    };
    let mut lines = Vec::new();
    lines.extend(shape.render_top(&ctx));
    lines.push(shape.render_row(&ctx, &row));
    lines.extend(shape.render_bottom(&ctx));
    if let Some(bar) = bottom_bar(shape, status, theme, width) {
        if shape.bottom_bar_gap() {
            lines.push(Line::default());
        }
        lines.push(bar);
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plain(line: &Line<'_>) -> String {
        line.spans.iter().map(|s| s.content.as_ref()).collect()
    }

    fn status() -> StatusLine {
        let mut st = StatusLine::new("model".to_owned(), "~/project");
        st.context_percent = Some(5.0);
        st.context_window = Some(200_000);
        st
    }

    #[test]
    fn names_round_trip_and_unknown_falls_back_to_box() {
        for shape in ComposerShape::ALL {
            assert_eq!(ComposerShape::from_name(shape.name()), *shape);
        }
        assert_eq!(ComposerShape::from_name("nope"), ComposerShape::Box);
        assert_eq!(ComposerShape::default(), ComposerShape::Band);
    }

    #[test]
    fn every_shape_previews_a_prompt_row_of_exact_width() {
        let theme = Theme::default();
        let st = status();
        for shape in ComposerShape::ALL {
            let lines = preview_lines(*shape, 90, &theme, &st);
            assert!(!lines.is_empty(), "{shape:?}");
            let prompt = lines
                .iter()
                .find(|l| plain(l).contains("Ask anything"))
                .unwrap_or_else(|| panic!("{shape:?}: no prompt row in {lines:?}"));
            // `pi` draws no right padding (TS `piComposerStyle.renderRow`).
            let want = if *shape == ComposerShape::Pi { 89 } else { 90 };
            assert_eq!(
                display_width(&plain(prompt)),
                want,
                "{shape:?}: {}",
                plain(prompt)
            );
        }
    }

    #[test]
    fn box_frames_the_prompt_in_two_rows() {
        let theme = Theme::default();
        let lines = preview_lines(ComposerShape::Box, 80, &theme, &status());
        assert_eq!(lines.len(), 2, "{lines:?}");
        let top = plain(&lines[0]);
        assert!(top.starts_with("╭──") && top.ends_with("──╮"), "{top}");
        let row = plain(&lines[1]);
        assert!(
            row.starts_with("╰─ Ask anything") && row.ends_with(" ─╯"),
            "{row}"
        );
    }

    #[test]
    fn rule_shapes_draw_rules_and_standalone_bars() {
        let theme = Theme::default();
        let st = status();
        let claude = preview_lines(ComposerShape::Claude, 60, &theme, &st);
        assert_eq!(claude.len(), 4, "{claude:?}");
        assert!(plain(&claude[0]).starts_with('─'));
        assert!(plain(&claude[1]).starts_with("❯ "));
        assert_eq!(plain(&claude[2]), "─".repeat(60));
        let rule = preview_lines(ComposerShape::Rule, 60, &theme, &st);
        // top rule, prompt, spacer, bar
        assert_eq!(rule.len(), 4, "{rule:?}");
        assert!(plain(&rule[2]).is_empty());
        let borderless = preview_lines(ComposerShape::Borderless, 60, &theme, &st);
        assert_eq!(borderless.len(), 2, "{borderless:?}");
    }

    #[test]
    fn filled_shapes_wear_their_caps() {
        let theme = Theme::default();
        let field = preview_lines(ComposerShape::Field, 60, &theme, &status());
        let row = plain(&field[0]);
        assert!(row.starts_with('▐') && row.ends_with('▌'), "{row}");
        let rail = preview_lines(ComposerShape::Rail, 60, &theme, &status());
        assert!(plain(&rail[0]).starts_with('▎'));
    }
}
