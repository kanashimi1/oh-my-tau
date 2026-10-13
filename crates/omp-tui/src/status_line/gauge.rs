//! The context gauge — the fill between the left and right groups in the
//! box layout (`#buildContextGaugeFill`, `component.ts:2994-3130`).

use ratatui::style::Color;
use ratatui::text::Span;

use super::metrics as fmt;
use super::presets::ContextLineMode;
use super::stats::{context_level, ContextLevel};
use super::style;

/// Gauge input for one frame.
#[derive(Debug, Clone, Copy)]
pub struct GaugeInput {
    /// Used percent (`None` = fresh session).
    pub percent: Option<f64>,
    /// Context window in tokens (`None` = unknown).
    pub window: Option<u64>,
    /// Context tokens used.
    pub tokens: Option<u64>,
    /// Gauge mode (`statusLine.contextLine`).
    pub mode: ContextLineMode,
    /// Auto-compaction threshold percent (tick position).
    pub compaction_pct: Option<f64>,
    /// Speculation start percent (tick position; hidden when equal to the
    /// threshold or when speculation can't run).
    pub speculation_pct: Option<f64>,
    /// Session-accent color for the used portion.
    pub accent: Option<Color>,
    /// Compaction boundaries unknown → no ticks.
    pub boundaries_known: bool,
}

/// How wide the gauge must be for its embedded labels to stay visible
/// (§5.3 step 5); falls back to 1 cell when only one group fits.
#[must_use]
pub fn embedded_min_width(input: &GaugeInput) -> usize {
    if input.mode != ContextLineMode::Embedded {
        return 1;
    }
    // `<pct>%` + `<window>` labels plus breathing room.
    let pct_w = input.percent.map_or(0, |p| format!("{p:.0}%").len());
    let win_w = input.window.map_or(0, |w| fmt::format_number(w).len());
    pct_w + win_w + 3
}

/// Percent → cell count (gauge geometry; widths are small, pct ≥ 0).
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_precision_loss)]
fn pct_cells(pct: f64, width: usize) -> usize {
    (pct / 100.0 * width as f64).round() as usize
}

/// Render the gauge as styled spans totalling exactly `width` cells.
#[must_use]
pub fn render(input: &GaugeInput, width: usize, theme: &crate::theme::Theme) -> Vec<Span<'static>> {
    if width == 0 {
        return Vec::new();
    }
    let glyph = style::icon(theme, "boxRound.horizontal", "─");
    let used_color = input
        .accent
        .unwrap_or_else(|| style::color(theme, "accent"));
    let border = style::color(theme, "border");
    let level = context_level(input.percent, input.tokens);
    let level_color = match level {
        ContextLevel::Normal => style::color(theme, "statusLineContext"),
        ContextLevel::Warning => style::color(theme, "warning"),
        ContextLevel::Purple => style::color(theme, "thinkingHigh"),
        ContextLevel::Error => style::color(theme, "error"),
    };
    let used_color = if input.mode == ContextLineMode::Off {
        used_color
    } else {
        level_color
    };

    let used_count = match input.percent {
        Some(p) => pct_cells(p, width),
        // Fresh session: exactly one lit cell at the left (§5.4).
        None => 1,
    };

    // Boundary ticks for annotated/embedded when auto-compact is on and the
    // gap is at least 8 cells. Threshold tick wins a contested cell.
    let ticks = if input.boundaries_known
        && matches!(
            input.mode,
            ContextLineMode::Annotated | ContextLineMode::Embedded
        )
        && width >= 8
    {
        let mut v: Vec<(usize, &'static str)> = Vec::new();
        if let Some(c) = input.compaction_pct {
            v.push((pct_cells(c, width), "context.compaction"));
        }
        if let Some(s) = input.speculation_pct {
            if input.compaction_pct != Some(s) {
                v.push((pct_cells(s, width), "context.speculation"));
            }
        }
        v
    } else {
        Vec::new()
    };

    let mut spans: Vec<Span<'static>> = Vec::with_capacity(width);
    for i in 0..width {
        let is_used = i < used_count;
        let tick = ticks
            .iter()
            .find(|(pos, _)| *pos == i)
            .map(|(_, name)| style::icon(theme, name, "┃"));
        // Compaction threshold wins a contested cell: prefer the last tick
        // (threshold sorts after speculation here via ordering above).
        let text = tick.map_or(glyph, |t| t).to_string();
        let color = match (is_used, tick) {
            (_, Some(_)) if input.boundaries_known => style::color(theme, "warning"),
            (true, _) => used_color,
            (false, _) => border,
        };
        spans.push(Span::styled(
            text,
            ratatui::style::Style::default().fg(color),
        ));
    }

    // Embedded labels (`embedded`, default): `<pct>%` and `<window>` at the
    // right end; pct label nudged to avoid boundary ticks.
    if input.mode == ContextLineMode::Embedded {
        embed_labels(&mut spans, input, used_count, theme);
    }
    spans
}

fn embed_labels(
    spans: &mut [Span<'static>],
    input: &GaugeInput,
    used_count: usize,
    theme: &crate::theme::Theme,
) {
    let width = spans.len();
    let level = context_level(input.percent, input.tokens);
    let level_color = match level {
        ContextLevel::Normal => style::color(theme, "statusLineContext"),
        ContextLevel::Warning => style::color(theme, "warning"),
        ContextLevel::Purple => style::color(theme, "thinkingHigh"),
        ContextLevel::Error => style::color(theme, "error"),
    };
    let error = style::color(theme, "error");

    // Window label anchored right.
    if let Some(w) = input.window {
        let win = fmt::format_number(w);
        if width >= win.len() {
            let start = width - win.len();
            for (i, c) in win.chars().enumerate() {
                spans[start + i] = Span::styled(
                    c.to_string(),
                    ratatui::style::Style::default().fg(level_color),
                );
            }
        }
    }
    // Percent label placed at the used boundary, nudged off boundary ticks.
    if let Some(p) = input.percent {
        let label = format!("{p:.0}%");
        if label.len() <= width {
            let mut start = used_count.min(width - label.len());
            // Nudge left while we'd cover a boundary tick or exceed the row.
            while start > 0 && start + label.len() > width {
                start -= 1;
            }
            // Pct > 100 renders the percent in error anchored past the window
            // label (§5.4).
            let color = if p > 100.0 { error } else { level_color };
            for (i, c) in label.chars().enumerate() {
                if start + i < width {
                    spans[start + i] =
                        Span::styled(c.to_string(), ratatui::style::Style::default().fg(color));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(pct: Option<f64>, window: Option<u64>) -> GaugeInput {
        GaugeInput {
            percent: pct,
            window,
            tokens: None,
            mode: ContextLineMode::Embedded,
            compaction_pct: None,
            speculation_pct: None,
            accent: None,
            boundaries_known: false,
        }
    }

    #[test]
    fn gauge_always_fills_exact_width() {
        let theme = crate::theme::Theme::default();
        for w in [1, 8, 40] {
            let spans = render(&input(Some(23.0), Some(200_000)), w, &theme);
            assert_eq!(spans.len(), w);
        }
    }

    #[test]
    fn fresh_session_has_one_lit_cell() {
        let theme = crate::theme::Theme::default();
        let spans = render(&input(None, Some(200_000)), 20, &theme);
        // First cell is the used color (not the border); compare mid-row
        // (the right end may be recolored by the embedded window label).
        assert_ne!(spans[0].style.fg, spans[spans.len() / 2].style.fg);
    }

    #[test]
    fn embedded_min_width_tracks_labels() {
        let mut i = input(Some(23.0), Some(200_000));
        assert!(embedded_min_width(&i) > 1);
        i.mode = ContextLineMode::Percentage;
        assert_eq!(embedded_min_width(&i), 1);
    }
}
