//! Progress HUD — port of `pi-coding-agent/src/modes/progress-hud.ts` §11.
//!
//! Right-aligned rows shown above the working line: judge batches
//! (`<done>/<total> · $<cost>` + bar) and download activity
//! (`<pct>% · <bytes/s>` + bar, indeterminate → no bar). Each row is
//! `label ━━━ tail` right-aligned within the available width, truncated to a
//! fallback when it cannot fit (spec §11.1).

use ratatui::layout::Rect;
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::Frame;

use crate::component::Component;
use crate::theme::Theme;

/// One HUD row. `progress.id` keys updates/deletes (TS `progress.id`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HudEntry {
    /// Stable id (batch id / download item id).
    pub id: String,
    /// Left label (`progress.intent` / item name).
    pub label: String,
    /// Done units (bytes for downloads).
    pub done: u64,
    /// Total units; `None` → indeterminate (no bar).
    pub total: Option<u64>,
    /// Free-form tail after the counter (cost / rate), already formatted.
    pub tail: Option<String>,
    /// Failed count; rendered ` · <n> failed` in warning when > 0.
    pub failed: u64,
}

impl HudEntry {
    /// Build a row with just id + label.
    #[must_use]
    pub fn new(id: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            done: 0,
            total: None,
            tail: None,
            failed: 0,
        }
    }
}

/// Map of HUD rows, insertion-ordered; feeds the [`HudProgress`] component.
#[derive(Debug, Default, Clone)]
pub struct HudProgress {
    rows: Vec<HudEntry>,
}

impl HudProgress {
    /// Insert or replace a row by id; preserves first-seen order.
    pub fn update(&mut self, entry: HudEntry) {
        match self.rows.iter_mut().find(|r| r.id == entry.id) {
            Some(row) => *row = entry,
            None => self.rows.push(entry),
        }
    }

    /// Remove a row by id.
    pub fn delete(&mut self, id: &str) {
        self.rows.retain(|r| r.id != id);
    }

    /// Remove all rows.
    pub fn clear(&mut self) {
        self.rows.clear();
    }

    /// Current rows (render order).
    #[must_use]
    pub fn rows(&self) -> &[HudEntry] {
        &self.rows
    }

    /// Whether any row is visible.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
}

/// Visible width of a rendered row body (label + bar + tail), ignoring ANSI.
fn row_parts(entry: &HudEntry, width: usize) -> (String, Option<(char, usize)>, String) {
    // Tail first (spec §11.1): counter, cost/rate tail, failed note.
    let mut tail = String::new();
    match entry.total {
        Some(total) => tail.push_str(&format!("{}/{}", entry.done, total)),
        None => tail.push_str(&format!("{}", entry.done)),
    }
    if let Some(t) = &entry.tail {
        if !t.is_empty() {
            tail.push_str(&format!(" · {t}"));
        }
    }
    if entry.failed > 0 {
        tail.push_str(&format!(" · {} failed", entry.failed));
    }
    // Bar: filled `━` / empty `─`, width min(18, remaining-1), floor 4.
    let remaining = width.saturating_sub(tail.chars().count() + 1);
    let bar_width = remaining.min(18).max(4);
    let bar = entry.total.map(|total| {
        let filled = if total == 0 {
            0
        } else {
            ((bar_width as u64 * entry.done) / total).min(bar_width as u64)
        };
        (if filled > 0 { '━' } else { '─' }, filled as usize)
    });
    (entry.label.clone(), bar, tail)
}

/// Render one row right-aligned into `width` cells; falls back to the tail
/// alone when the whole row overflows (spec §11.1).
#[must_use]
pub fn render_hud_row(entry: &HudEntry, width: usize, theme: &Theme) -> Line<'static> {
    let (label, bar, tail) = row_parts(entry, width);
    let tail_len = tail.chars().count();
    if width <= tail_len {
        // Whole row overflows: the fallback is the tail alone.
        return Line::from(Span::styled(tail, Style::default().fg(theme.color("text"))));
    }
    // Bar width: min(18, remaining-1), floor 4 (progress-bar min).
    let remaining = (width - tail_len).saturating_sub(1);
    let bar_len = if bar.is_some() {
        remaining.min(18).max(4)
    } else {
        0
    };
    let label_len = label.chars().count();
    let show_label = !label.is_empty() && width >= label_len + 1 + bar_len + tail_len;

    let mut spans: Vec<Span<'static>> = Vec::new();
    if show_label {
        spans.push(Span::styled(
            label.clone(),
            Style::default().fg(theme.color("text_muted")),
        ));
        spans.push(Span::raw(" "));
    }
    if let Some((glyph, filled)) = bar {
        let filled = filled.min(bar_len);
        let filled_str: String = std::iter::repeat(glyph).take(filled).collect();
        let empty_str: String = std::iter::repeat('─').take(bar_len - filled).collect();
        spans.push(Span::styled(
            filled_str,
            Style::default().fg(theme.color("accent")),
        ));
        spans.push(Span::styled(
            empty_str,
            Style::default().fg(theme.color("text_muted")),
        ));
    }
    spans.push(Span::styled(tail, Style::default().fg(theme.color("text"))));
    let used: usize = spans.iter().map(|s| s.content.chars().count()).sum();
    let pad = width.saturating_sub(used);
    let mut out = vec![Span::raw(" ".repeat(pad))];
    out.extend(spans);
    Line::from(out)
}

impl Component for HudProgress {
    fn render(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        if area.height == 0 || self.rows.is_empty() {
            return;
        }
        let shown = self.rows.len().min(area.height as usize);
        let start = self.rows.len() - shown;
        for (i, entry) in self.rows[start..].iter().enumerate() {
            let row = render_hud_row(entry, area.width as usize, theme);
            let area = Rect::new(area.x, area.y + i as u16, area.width, 1);
            frame.render_widget(ratatui::widgets::Paragraph::new(row), area);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render_plain(line: &Line<'_>, width: usize) -> String {
        let mut out = String::new();
        for span in &line.spans {
            out.push_str(&span.content);
        }
        let used = out.chars().count();
        if used < width {
            out.push_str(&" ".repeat(width - used));
        }
        out
    }

    #[test]
    fn batch_row_shape() {
        let theme = Theme::default();
        let entry = HudEntry {
            done: 3,
            total: Some(5),
            tail: Some("$0.42".into()),
            failed: 1,
            ..HudEntry::new("b1", "judge batch")
        };
        // Wide enough for label + bar + tail.
        let row = render_hud_row(&entry, 60, &theme);
        let text = render_plain(&row, 60);
        assert!(text.ends_with("3/5 \u{b7} $0.42 \u{b7} 1 failed"));
        assert!(text.contains("judge batch"));
        assert!(text.starts_with(' '));
        assert_eq!(text.chars().count(), 60);
        assert!(text.contains('\u{2501}'));
        assert!(text.contains('\u{2500}'));
        // Narrower: tail+bar leave <2 cells, label drops but tail stays.
        let row = render_hud_row(&entry, 40, &theme);
        let text = render_plain(&row, 40);
        assert!(text.ends_with("3/5 \u{b7} $0.42 \u{b7} 1 failed"));
        assert!(!text.contains("judge batch"));
        assert_eq!(text.chars().count(), 40);
    }

    #[test]
    fn indeterminate_row_has_no_bar() {
        let theme = Theme::default();
        let entry = HudEntry {
            done: 1024,
            ..HudEntry::new("d1", "gguf-9b")
        };
        let row = render_hud_row(&entry, 30, &theme);
        let text = render_plain(&row, 30);
        assert!(text.ends_with("1024"));
        assert!(!text.contains('\u{2501}'));
        assert!(text.contains("gguf-9b"));
    }

    #[test]
    fn overflow_falls_back_to_tail() {
        let theme = Theme::default();
        let entry = HudEntry {
            done: 9,
            total: Some(10),
            ..HudEntry::new("b", "a-very-long-label-that-will-not-fit")
        };
        let row = render_hud_row(&entry, 4, &theme);
        let text = render_plain(&row, 4);
        assert_eq!(text.trim(), "9/10");
    }

    #[test]
    fn hud_update_delete_clear() {
        let mut hud = HudProgress::default();
        hud.update(HudEntry::new("a", "one"));
        hud.update(HudEntry::new("b", "two"));
        hud.update(HudEntry {
            done: 1,
            ..HudEntry::new("a", "one!")
        });
        assert_eq!(hud.rows().len(), 2);
        assert_eq!(hud.rows()[0].label, "one!");
        hud.delete("a");
        assert_eq!(hud.rows().len(), 1);
        hud.clear();
        assert!(hud.is_empty());
    }
}
