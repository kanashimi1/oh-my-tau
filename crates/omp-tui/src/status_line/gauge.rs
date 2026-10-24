//! The context gauge — the fill between the left and right groups in the
//! box layout (`#buildContextGaugeFill`, `component.ts:2994-3130`).

use ratatui::style::Color;
use ratatui::text::Span;

use super::metrics as fmt;
use super::presets::ContextLineMode;
use super::style;

/// Gauge input for one frame.
#[derive(Debug, Clone, Copy)]
pub struct GaugeInput {
    /// Used percent (`None` = fresh session).
    pub percent: Option<f64>,
    /// Context window in tokens (`None` = unknown).
    pub window: Option<u64>,
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
    /// The standalone context segments were absorbed into the gauge
    /// (`embedContext`): it carries the percent + window labels.
    pub embed: bool,
}

/// `formatEmbeddedContextPercent`: one decimal below 1%, else whole.
fn percent_label(percent: f64) -> String {
    if percent > 0.0 && percent < 1.0 {
        format!("{percent:.1}%")
    } else {
        format!("{}%", percent.round())
    }
}

/// How wide the gauge must be for its embedded labels to stay visible
/// (`embeddedContextGaugeMinWidth`); falls back to 1 cell when the gauge
/// does not embed the context labels.
#[must_use]
pub fn embedded_min_width(input: &GaugeInput) -> usize {
    if !input.embed {
        return 1;
    }
    let pct_w = input
        .percent
        .map_or(0, |p| percent_label(p).chars().count() + 2);
    let win_w = fmt::format_number(input.window.unwrap_or(0))
        .chars()
        .count();
    pct_w + win_w + 2
}

/// Percent → cell index on a `scale`-cell gauge (`Math.round` semantics).
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn cell_for(percent: f64, scale: usize) -> usize {
    ((percent / 100.0 * scale as f64).round().max(0.0) as usize).min(scale.saturating_sub(1))
}

/// `adjustHsv(hex, {s: 0.7, v: 0.75})` for the threshold marker color.
#[allow(
    clippy::many_single_char_names,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn dimmed(color: Color) -> Color {
    let Color::Rgb(r, g, b) = color else {
        return color;
    };
    let (r, g, b) = (
        f64::from(r) / 255.0,
        f64::from(g) / 255.0,
        f64::from(b) / 255.0,
    );
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let d = max - min;
    let h = if d == 0.0 {
        0.0
    } else if max == r {
        ((g - b) / d).rem_euclid(6.0)
    } else if max == g {
        (b - r) / d + 2.0
    } else {
        (r - g) / d + 4.0
    };
    let s = if max == 0.0 { 0.0 } else { d / max } * 0.7;
    let v = max * 0.75;
    let c = v * s;
    let x = c * (1.0 - ((h % 2.0) - 1.0).abs());
    let m = v - c;
    let (r1, g1, b1) = match h as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let ch = |v: f64| ((v + m) * 255.0).round().clamp(0.0, 255.0) as u8;
    Color::Rgb(ch(r1), ch(g1), ch(b1))
}

/// Render the gauge as styled spans totalling exactly `width` cells —
/// a port of `#buildContextGaugeFill` (`component.ts`).
#[must_use]
#[allow(
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
pub fn render(input: &GaugeInput, width: usize, theme: &crate::theme::Theme) -> Vec<Span<'static>> {
    use ratatui::style::Style;
    if width == 0 {
        return Vec::new();
    }
    let horizontal = style::icon(theme, "boxRound.horizontal", "─");
    let used_color = input
        .accent
        .unwrap_or_else(|| style::color(theme, "borderAccent"));
    let used_color = if used_color == Color::Reset {
        style::color(theme, "accent")
    } else {
        used_color
    };
    let unused_color = style::color(theme, "border");
    let window = input.window.unwrap_or(0);
    if input.mode == ContextLineMode::Off || (input.percent.is_none() && window == 0) {
        return (0..width)
            .map(|_| Span::styled(horizontal.to_string(), Style::default().fg(used_color)))
            .collect();
    }

    let pct = input.percent;
    let clamped = pct.map_or(0.0, |p| p.clamp(0.0, 100.0));
    let overflow = pct.is_some_and(|p| p > 100.0);
    let mut percent_label_text = String::new();
    let mut window_label = String::new();
    let mut percent_start: Option<usize> = None;
    let mut window_start: Option<usize> = None;
    let mut scale = width;
    if input.embed {
        let cand_pct = pct.map_or(String::new(), |p| {
            percent_label(if overflow { p } else { clamped })
        });
        let cand_win = fmt::format_number(window);
        if width >= embedded_min_width(input) {
            percent_label_text = cand_pct;
            window_label = cand_win;
            let plen = percent_label_text.chars().count();
            let wlen = window_label.chars().count();
            if overflow {
                let ps = width.saturating_sub(plen);
                percent_start = Some(ps);
                window_start = Some(ps.saturating_sub(1 + wlen));
            } else {
                window_start = Some(width.saturating_sub(wlen + 1));
            }
            scale = window_start.unwrap_or(width);
        }
    }

    let used_count = scale.min(((clamped / 100.0 * scale as f64).round() as usize).max(1));

    let mut spec_idx: Option<usize> = None;
    let mut thr_idx: Option<usize> = None;
    if matches!(
        input.mode,
        ContextLineMode::Annotated | ContextLineMode::Embedded
    ) && input.boundaries_known
        && width >= 8
        && scale > 0
    {
        thr_idx = input.compaction_pct.map(|c| cell_for(c, scale));
        spec_idx = input.speculation_pct.map(|c| cell_for(c, scale));
        if spec_idx == thr_idx {
            spec_idx = None;
        }
    }

    let plen = percent_label_text.chars().count();
    if !percent_label_text.is_empty() && percent_start.is_none() && scale >= plen + 2 {
        let max_start = scale - plen - 1;
        let preferred = max_start.min(used_count.max(1));
        let overlaps = |start: usize| {
            let end = start + plen;
            spec_idx.is_some_and(|i| i >= start && i < end)
                || thr_idx.is_some_and(|i| i >= start && i < end)
        };
        for distance in 0..=max_start {
            if preferred >= distance {
                let left = preferred - distance;
                if left >= 1 && !overlaps(left) {
                    percent_start = Some(left);
                    break;
                }
            }
            if distance == 0 {
                continue;
            }
            let right = preferred + distance;
            if right <= max_start && !overlaps(right) {
                percent_start = Some(right);
                break;
            }
        }
    }

    let spec_glyph = style::icon(theme, "context.speculation", "╎");
    let thr_glyph = style::icon(theme, "context.compaction", "┃");
    let spec_color = style::color(theme, "muted");
    let overflow_color = style::color(theme, "error");
    let thr_color = dimmed(used_color);

    let percent_chars: Vec<char> = percent_label_text.chars().collect();
    let window_chars: Vec<char> = window_label.chars().collect();
    let percent_end = percent_start.map(|s| s + percent_chars.len());
    let window_end = window_start.map(|s| s + window_chars.len());
    let mut spans: Vec<Span<'static>> = Vec::with_capacity(width);
    for i in 0..width {
        let (text, color) = if let Some((ps, _)) = percent_start
            .zip(percent_end)
            .filter(|(s, e)| i >= *s && i < *e)
        {
            (
                percent_chars[i - ps].to_string(),
                if overflow { overflow_color } else { used_color },
            )
        } else if thr_idx == Some(i) {
            (thr_glyph.to_string(), thr_color)
        } else if spec_idx == Some(i) {
            (spec_glyph.to_string(), spec_color)
        } else if let Some((ws, _)) = window_start
            .zip(window_end)
            .filter(|(s, e)| i >= *s && i < *e)
        {
            (window_chars[i - ws].to_string(), thr_color)
        } else {
            (
                horizontal.to_string(),
                if i < used_count {
                    used_color
                } else {
                    unused_color
                },
            )
        };
        spans.push(Span::styled(text, Style::default().fg(color)));
    }
    spans
}

/// Cells in the clean-style context bar.
pub const CLEAN_BAR_CELLS: usize = 10;

/// Clean-style fill token: `accent` below 70%, `warning` up to 90%,
/// `error` above.
#[must_use]
pub fn clean_fill_token(percent: f64) -> &'static str {
    if percent < 70.0 {
        "accent"
    } else if percent <= 90.0 {
        "warning"
    } else {
        "error"
    }
}

/// The clean-style context segment: a fixed 10-cell `━`/`─` bar plus
/// ` N%`; `None` when neither the percent nor the window is known.
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
pub fn clean_bar(percent: Option<f64>, window: Option<u64>) -> Option<super::segments::Piece> {
    if percent.is_none() && window.unwrap_or(0) == 0 {
        return None;
    }
    let pct = percent.unwrap_or(0.0).max(0.0);
    let filled = ((pct / 100.0 * CLEAN_BAR_CELLS as f64).round() as usize).min(CLEAN_BAR_CELLS);
    let fill = clean_fill_token(pct);
    let label_token = if fill == "accent" { "dim" } else { fill };
    let mut runs = Vec::new();
    if filled > 0 {
        runs.push(("━".repeat(filled), fill));
    }
    if filled < CLEAN_BAR_CELLS {
        runs.push(("─".repeat(CLEAN_BAR_CELLS - filled), "dim"));
    }
    runs.push((format!(" {}", percent_label(pct)), label_token));
    Some(super::segments::Piece::from_runs(runs))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(pct: Option<f64>, window: Option<u64>) -> GaugeInput {
        GaugeInput {
            percent: pct,
            window,
            mode: ContextLineMode::Embedded,
            compaction_pct: None,
            speculation_pct: None,
            accent: None,
            boundaries_known: false,
            embed: true,
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
        i.embed = false;
        assert_eq!(embedded_min_width(&i), 1);
    }

    #[test]
    fn clean_bar_thresholds_and_cells() {
        assert_eq!(clean_fill_token(0.0), "accent");
        assert_eq!(clean_fill_token(69.9), "accent");
        assert_eq!(clean_fill_token(70.0), "warning");
        assert_eq!(clean_fill_token(90.0), "warning");
        assert_eq!(clean_fill_token(90.5), "error");
        let bar = clean_bar(Some(95.0), Some(200_000)).unwrap();
        assert_eq!(bar.text, "━━━━━━━━━━ 95%");
        assert_eq!(bar.runs[0].1, "error");
        let bar = clean_bar(Some(30.0), None).unwrap();
        assert_eq!(bar.text, "━━━─────── 30%");
        assert_eq!(bar.runs[1].1, "dim");
        assert_eq!(
            clean_bar(None, Some(128_000)).unwrap().text,
            "────────── 0%"
        );
        assert!(clean_bar(None, None).is_none());
    }
}
