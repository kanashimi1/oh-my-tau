//! Magic-keyword gradient painting for ratatui spans — the port of
//! `pi-tui/src/prompt/gradient-highlight.ts` on top of [`crate::magic`]:
//! the live editor (shimmering while a keyword is visible) and sent user
//! bubbles (static palette).

use std::ops::Range;

use ratatui::style::{Color, Style};
use ratatui::text::Span;

use crate::magic::{self, Keyword};

/// Editor shimmer repaint cadence (TS `SHIMMER_FRAME_MS`).
pub const SHIMMER_FRAME_MS: u64 = 70;
/// One full gradient sweep across a keyword (TS `SHIMMER_PERIOD_MS`).
pub const SHIMMER_PERIOD_MS: u64 = 1800;

/// Editor shimmer phase in `[0, 1)` at wall-clock `now_ms`.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn shimmer_phase(now_ms: u64) -> f64 {
    (now_ms % SHIMMER_PERIOD_MS) as f64 / SHIMMER_PERIOD_MS as f64
}

/// Whether any line holds an enabled keyword standing as prose. The
/// substring probe keeps the common no-keyword path to a few `contains`.
#[must_use]
pub fn any_keyword(lines: &[String], disabled: u32) -> bool {
    let probe = |l: &String| magic::KEYWORDS.iter().any(|k| l.contains(k.word));
    lines.iter().any(probe) && !magic::enabled_ranges(&lines.join("\n"), disabled).is_empty()
}

/// Split `seg` (which starts at byte `seg_off` of the text the `hits` were
/// computed on) into spans: keyword characters painted with their gradient
/// over `base`, everything else in `base`. A hit cut by the segment keeps
/// its per-character colours (character index counted from the hit start).
#[must_use]
pub fn paint_segment(
    seg: &str,
    seg_off: usize,
    hits: &[(Range<usize>, &'static Keyword)],
    base: Style,
    phase: f64,
) -> Vec<Span<'static>> {
    let seg_end = seg_off + seg.len();
    let mut spans = Vec::new();
    let mut cursor = seg_off;
    for (range, k) in hits {
        if range.end <= seg_off || range.start >= seg_end {
            continue;
        }
        let from = range.start.max(seg_off);
        let to = range.end.min(seg_end);
        if from > cursor {
            spans.push(Span::styled(
                seg[cursor - seg_off..from - seg_off].to_owned(),
                base,
            ));
        }
        // The keyword words are ASCII: byte offset == char index.
        let len = range.len();
        for (pos, ch) in seg[from - seg_off..to - seg_off].char_indices() {
            let (red, green, blue) = magic::keyword_rgb(k, from + pos - range.start, len, phase);
            spans.push(Span::styled(
                ch.to_string(),
                base.fg(Color::Rgb(red, green, blue)),
            ));
        }
        cursor = to;
    }
    if cursor < seg_end || spans.is_empty() {
        spans.push(Span::styled(seg[cursor - seg_off..].to_owned(), base));
    }
    spans
}

/// Paint every enabled keyword of `text` (static bubble use).
#[must_use]
pub fn paint(text: &str, base: Style, disabled: u32, phase: f64) -> Vec<Span<'static>> {
    let hits = magic::enabled_ranges(text, disabled);
    if hits.is_empty() {
        return vec![Span::styled(text.to_owned(), base)];
    }
    paint_segment(text, 0, &hits, base, phase)
}

/// Keyword hits of a markdown source, replayed onto its rendered spans
/// (user bubbles): every raw occurrence of a keyword word in the source is
/// flagged prose-hit or not (code span, fence, XML, glued identifier);
/// rendering keeps words in order, so the n-th occurrence met in the
/// rendered spans takes the n-th flag.
#[derive(Debug, Default)]
pub struct RenderedHits {
    /// Per [`magic::KEYWORDS`] index: occurrence flags and the next one.
    flags: Vec<(Vec<bool>, usize)>,
}

impl RenderedHits {
    /// Flags for `source`; `None` when it holds no enabled prose hit.
    #[must_use]
    pub fn new(source: &str, disabled: u32) -> Option<Self> {
        let hits = magic::enabled_ranges(source, disabled);
        if hits.is_empty() {
            return None;
        }
        let flags = magic::KEYWORDS
            .iter()
            .map(|k| {
                let occ = source
                    .match_indices(k.word)
                    .map(|(at, _)| hits.iter().any(|(r, hk)| r.start == at && hk.id == k.id))
                    .collect();
                (occ, 0)
            })
            .collect();
        Some(Self { flags })
    }

    /// Paint the next occurrences found in `content` (one rendered span);
    /// `None` when none of them is a prose hit (keep the span as is).
    pub fn paint_span(
        &mut self,
        content: &str,
        base: Style,
        phase: f64,
    ) -> Option<Vec<Span<'static>>> {
        let mut found: Vec<(usize, usize)> = magic::KEYWORDS
            .iter()
            .enumerate()
            .flat_map(|(ki, k)| content.match_indices(k.word).map(move |(at, _)| (at, ki)))
            .collect();
        if found.is_empty() {
            return None;
        }
        found.sort_unstable();
        let mut hits = Vec::new();
        for (at, ki) in found {
            let (flags, next) = &mut self.flags[ki];
            let hit = flags.get(*next).copied().unwrap_or(false);
            *next += 1;
            if hit {
                let k = &magic::KEYWORDS[ki];
                hits.push((at..at + k.word.len(), k));
            }
        }
        (!hits.is_empty()).then(|| paint_segment(content, 0, &hits, base, phase))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(spans: &[Span<'_>]) -> String {
        spans.iter().map(|s| s.content.as_ref()).collect()
    }

    #[test]
    fn paints_keyword_chars_and_keeps_the_rest() {
        let base = Style::default().fg(Color::White);
        let spans = paint("go ultrathink now", base, 0, 0.0);
        assert_eq!(text(&spans), "go ultrathink now");
        let k = &magic::KEYWORDS[0];
        let painted: Vec<_> = spans
            .iter()
            .filter(|s| s.style.fg != Some(Color::White))
            .collect();
        assert_eq!(painted.len(), 10);
        let (r, g, b) = magic::keyword_rgb(k, 0, 10, 0.0);
        assert_eq!(painted[0].content, "u");
        assert_eq!(painted[0].style.fg, Some(Color::Rgb(r, g, b)));
        // Code spans stay plain; disabled keywords stay plain.
        assert_eq!(paint("`ultrathink`", base, 0, 0.0).len(), 1);
        assert_eq!(paint("go ultrathink", base, 1, 0.0).len(), 1);
    }

    #[test]
    fn segment_split_keeps_char_indices() {
        let full = "a ultracode";
        let hits = magic::enabled_ranges(full, 0);
        // Wrapped after "a ultra": the second row starts at byte 7.
        let second = paint_segment("code", 7, &hits, Style::default(), 0.0);
        let k = hits[0].1;
        let (r, g, b) = magic::keyword_rgb(k, 5, 9, 0.0);
        assert_eq!(second[0].content, "c");
        assert_eq!(second[0].style.fg, Some(Color::Rgb(r, g, b)));
        assert!(any_keyword(&["x ultracode".to_owned()], 0));
        assert!(!any_keyword(&["x ultracodes".to_owned()], 0));
    }

    #[test]
    fn rendered_hits_skip_code_occurrences_in_order() {
        // Source: a code span first, then the prose hit; rendered without backticks.
        let mut hits = RenderedHits::new("`ultrathink` then ultrathink", 0).unwrap();
        assert!(
            hits.paint_span("ultrathink", Style::default(), 0.0)
                .is_none(),
            "code occurrence"
        );
        let painted = hits
            .paint_span(" then ultrathink", Style::default(), 0.0)
            .unwrap();
        assert_eq!(text(&painted), " then ultrathink");
        assert!(RenderedHits::new("```\nultrathink\n```", 0).is_none());
    }

    #[test]
    fn shimmer_phase_wraps_each_period() {
        assert!((shimmer_phase(900) - 0.5).abs() < 1e-9);
        assert!(shimmer_phase(SHIMMER_PERIOD_MS).abs() < 1e-9);
    }
}
