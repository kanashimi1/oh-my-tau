//! Display layout for the editor: wide-char cell widths, soft wrapping,
//! and atomic-marker scanning — the port of `TextChunk`/`#wrapLine`
//! (`components/editor.ts:136,2533`) plus `#atomicTokenAt` (`:3363`).
//!
//! Wrap is greedy per grapheme cluster, matching pi-tui's
//! `wrapLine(line, width)`; wide (CJK/emoji) clusters occupy two cells
//! (`visibleWidth`).

use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

/// Display width of a string in terminal cells.
#[must_use]
pub fn display_width(s: &str) -> usize {
    UnicodeWidthStr::width(s)
}

/// One visual line produced by wrapping: `(line_index, byte_start, text)`.
/// `text` is the wrapped slice (no trailing newline).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayoutLine {
    /// Logical line this visual row belongs to.
    pub line: usize,
    /// Byte offset of `text` within the logical line.
    pub start: usize,
    /// The visible text of the row.
    pub text: String,
}

/// Wrap one logical line to `width` cells, greedily, grapheme-aware.
/// An empty line yields one empty [`LayoutLine`]. `width` 0 yields the
/// whole line as one row.
#[must_use]
pub fn wrap_line(line_index: usize, line: &str, width: usize) -> Vec<LayoutLine> {
    let width = width.max(1);
    let mut rows = Vec::new();
    let mut start = 0usize;
    let mut cur_width = 0usize;
    for (i, cluster) in line.grapheme_indices(true) {
        let w = UnicodeWidthStr::width(cluster).max(1);
        if cur_width + w > width && i > start {
            rows.push(LayoutLine {
                line: line_index,
                start,
                text: line[start..i].to_owned(),
            });
            start = i;
            cur_width = 0;
        }
        cur_width += w;
    }
    rows.push(LayoutLine {
        line: line_index,
        start,
        text: line[start..].to_owned(),
    });
    rows
}

/// Wrap a whole buffer to `width` cells.
#[must_use]
pub fn layout(lines: &[String], width: usize) -> Vec<LayoutLine> {
    lines
        .iter()
        .enumerate()
        .flat_map(|(i, l)| wrap_line(i, l, width))
        .collect()
}

/// An atomic marker occurrence in a line (TS "atom"): paste/image/video
/// placeholders the caret jumps over and backspace deletes whole.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MarkerSpan {
    /// Byte start (inclusive).
    pub start: usize,
    /// Byte end (exclusive).
    pub end: usize,
}

/// Marker kinds, matching TS `PLACEHOLDER_REGEX`
/// (`composer-attachments.ts:161`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarkerKind {
    /// `[Paste #N, ...]`
    Paste,
    /// `[Image #N, ...]`
    Image,
    /// `[Video #N]`
    Video,
}

/// Find the marker starting exactly at byte `col` in `line`, if any.
#[must_use]
pub fn marker_at(line: &str, col: usize) -> Option<(MarkerSpan, MarkerKind)> {
    let rest = line.get(col..)?;
    let kind = if rest.starts_with("[Paste #") {
        MarkerKind::Paste
    } else if rest.starts_with("[Image #") {
        MarkerKind::Image
    } else if rest.starts_with("[Video #") {
        MarkerKind::Video
    } else {
        return None;
    };
    let end_rel = rest.find(']')?;
    let digits = match kind {
        MarkerKind::Paste => "[Paste #".len(),
        MarkerKind::Image => "[Image #".len(),
        MarkerKind::Video => "[Video #".len(),
    };
    let num = &rest[digits..end_rel];
    let valid = num
        .split(',')
        .next()
        .unwrap_or("")
        .chars()
        .all(|c| c.is_ascii_digit())
        && !num.split(',').next().unwrap_or("").is_empty();
    if !valid {
        return None;
    }
    Some((
        MarkerSpan {
            start: col,
            end: col + end_rel + 1,
        },
        kind,
    ))
}

/// The marker whose span covers or precedes `col` (for caret-jump and
/// backspace-widening): if `col` is inside a marker or exactly at its end,
/// return the span.
#[must_use]
pub fn marker_covering(line: &str, col: usize) -> Option<MarkerSpan> {
    // Markers are short (<40 bytes); scan a window before col.
    let from = col.saturating_sub(64);
    let window = line.get(from..)?;
    let mut offset = 0usize;
    while offset < window.len() {
        let abs = from + offset;
        if let Some((span, _)) = marker_at(line, abs) {
            if col > span.start && col <= span.end {
                return Some(span);
            }
            offset = span.end - from;
        } else {
            // advance one char
            let ch = window[offset..].chars().next()?;
            offset += ch.len_utf8();
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn width_counts_wide_chars_as_two() {
        assert_eq!(display_width("abc"), 3);
        assert_eq!(display_width("汉a"), 3);
        assert_eq!(display_width("🙂x"), 3);
    }

    #[test]
    fn wrap_splits_on_width() {
        let rows = wrap_line(0, "hello world", 5);
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].text, "hello");
        assert_eq!(rows[1].text, " worl");
        assert_eq!(rows[2].text, "d");
    }

    #[test]
    fn wrap_empty_line_yields_one_row() {
        let rows = wrap_line(0, "", 10);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].text, "");
    }

    #[test]
    fn marker_detection() {
        let line = "see [Paste #1, +12 lines] ok";
        let pos = line.find("[Paste").unwrap();
        let (span, kind) = marker_at(line, pos).unwrap();
        assert_eq!(kind, MarkerKind::Paste);
        assert_eq!(&line[span.start..span.end], "[Paste #1, +12 lines]");
        assert!(marker_at(line, pos + 1).is_none());
        assert!(marker_at("Paste #1", 0).is_none());
    }

    #[test]
    fn marker_covering_finds_span_at_end() {
        let line = "x [Image #2, 640x480]";
        let end = line.len();
        let span = marker_covering(line, end).unwrap();
        assert_eq!(&line[span.start..span.end], "[Image #2, 640x480]");
    }
}
