//! Thinking/reasoning block rendering — the port of `pi-tui/src/thinking.ts`
//! plus the display half of `pi-tui/src/chat/thinking-display.ts` (spec
//! chat.md §3.1–§3.5, §5).
//!
//! Pure presentation: text formatting, the collapsed/expanded/hidden
//! decision, the starburst pulse line and the finished-thought labels. The
//! streaming reveal lives in [`crate::chat::reveal`]; the assistant message
//! integration lives with the chat model.

use std::time::Duration;

use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span, Text};
use unicode_segmentation::UnicodeSegmentation;

use crate::theme::Theme;

/// The 8 starburst pulse frames (TS `#thinkingDotsLabel`, chat.md §3.1).
pub const STARBURST_FRAMES: [&str; 8] = ["✻", "✼", "❉", "❊", "✺", "✹", "✸", "✶"];

/// Frame dwell eases between 70 ms and 230 ms over the 8-frame cycle.
pub const PULSE_MIN_MS: u64 = 70;
pub const PULSE_MAX_MS: u64 = 230;

/// User-visible thinking settings (TS `hideThinkingBlock` +
/// `proseOnlyThinking`, chat.md §3.1/§5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ThinkingDisplay {
    /// Fold fenced code blocks away and show prose only (default on).
    pub prose_only: bool,
    /// Hide thinking bodies entirely (only the live pulse line shows).
    pub hidden: bool,
}

impl Default for ThinkingDisplay {
    fn default() -> Self {
        Self {
            prose_only: true,
            hidden: false,
        }
    }
}

impl ThinkingDisplay {
    /// Toggle the `hideThinkingBlock` setting (TS `app.thinking.toggle`).
    pub fn toggle_hidden(&mut self) {
        self.hidden = !self.hidden;
    }
}

/// Legacy one-bit toggle kept for callers that only need collapsed/expanded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ThinkingExpand {
    /// Collapsed to a one-line header.
    #[default]
    Collapsed,
    /// Fully visible.
    Expanded,
}

impl ThinkingExpand {
    /// Toggle.
    pub fn toggle(&mut self) {
        *self = match self {
            Self::Collapsed => Self::Expanded,
            Self::Expanded => Self::Collapsed,
        };
    }
}

/// Drop gpt-5-style empty-comment noise lines (`<!-- -->` and a trailing
/// unterminated `<!--`) outside code fences (chat.md §5).
fn strip_comment_noise(text: &str) -> String {
    let mut out: Vec<&str> = Vec::new();
    let mut in_fence = false;
    for line in text.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            in_fence = !in_fence;
            out.push(line);
            continue;
        }
        if !in_fence {
            let t = trimmed.trim_end();
            if t == "<!--" {
                continue; // drop both the marker line and its (blank) body
            }
            if t.starts_with("<!--") && t.ends_with("-->") && t.len() <= 8 {
                continue;
            }
        }
        out.push(line);
    }
    // A trailing unterminated `<!--` opener is noise.
    while out
        .last()
        .is_some_and(|l| l.trim().starts_with("<!--") && !l.trim().ends_with("-->"))
    {
        out.pop();
    }
    let mut s = out.join("\n");
    if text.ends_with('\n') {
        s.push('\n');
    }
    s
}

/// Rewrite the preceding non-blank line with a trailing `...`
/// (TS `appendFoldEllipsis`: `…` → `...`, sentences keep their period).
fn fold_ellipsis(line: &mut String) {
    let trimmed = line.trim_end();
    if let Some(stripped) = trimmed.strip_suffix('…') {
        *line = format!("{stripped}...");
    } else if trimmed.ends_with('.') {
        *line = format!("{trimmed}..");
    } else {
        *line = format!("{trimmed}...");
    }
}

/// `formatThinkingForDisplay` (chat.md §5): both modes drop comment-noise
/// lines; prose-only additionally elides each fenced code block entirely and
/// rewrites the preceding non-blank line with a trailing `...`.
#[must_use]
pub fn format_thinking_for_display(text: &str, prose_only: bool) -> String {
    let stripped = strip_comment_noise(text);
    if !prose_only {
        return stripped;
    }
    let mut out: Vec<String> = Vec::new();
    let mut in_fence = false;
    for line in stripped.lines() {
        if line.trim_start().starts_with("```") || line.trim_start().starts_with("~~~") {
            if !in_fence {
                // Opening fence: elide the whole block, rewrite the previous
                // non-blank line.
                for prev in out.iter_mut().rev() {
                    if !prev.trim().is_empty() {
                        fold_ellipsis(prev);
                        break;
                    }
                }
            }
            in_fence = !in_fence;
            continue;
        }
        if !in_fence {
            out.push(line.to_owned());
        }
    }
    format!("{}\n", out.join("\n"))
}

/// `hasDisplayableThinking` (chat.md §3.3): formatted text is non-blank and
/// not just placeholder punctuation (`.`, `…`, whitespace).
#[must_use]
pub fn has_displayable_thinking(text: &str, prose_only: bool) -> bool {
    let formatted = format_thinking_for_display(text, prose_only);
    if formatted.trim().is_empty() {
        return false;
    }
    !formatted
        .chars()
        .all(|c| c == '.' || c == '…' || c.is_whitespace())
}

/// `thoughtLabel` (chat.md §3.2): `"Thought"` when the block was never seen
/// streaming, else `"Thought for <n>s"` (seconds under a minute, then a
/// compact duration).
#[must_use]
pub fn thought_label(streamed: bool, elapsed: Duration) -> String {
    if !streamed {
        return "Thought".to_owned();
    }
    let secs = elapsed.as_secs();
    if secs < 60 {
        format!("Thought for {secs}s")
    } else {
        format!("Thought for {}m{}s", secs / 60, secs % 60)
    }
}

/// Raised-cosine pulse dwell between [`PULSE_MIN_MS`] and [`PULSE_MAX_MS`]
/// over the 8-frame cycle (chat.md §3.1).
#[must_use]
pub fn pulse_dwell_ms(frame: usize) -> u64 {
    let t = (frame % STARBURST_FRAMES.len()) as f64 / STARBURST_FRAMES.len() as f64;
    let mid = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * t).cos();
    (PULSE_MIN_MS as f64 + mid * (PULSE_MAX_MS - PULSE_MIN_MS) as f64).round() as u64
}

/// The live thinking head: starburst glyph + `Thinking` (chat.md §3.1).
#[must_use]
pub fn pulse_line(frame: usize, theme: &Theme) -> Line<'static> {
    Line::from(vec![
        Span::styled(
            STARBURST_FRAMES[frame % STARBURST_FRAMES.len()].to_owned(),
            Style::default().fg(theme.color("thinkingText")),
        ),
        Span::styled(" Thinking", Style::default().fg(theme.color("text_muted"))),
    ])
}

/// Render state for one thinking block.
#[derive(Debug, Clone, Default)]
pub struct ThinkingBlockView {
    /// Raw (or already-formatted, see `preformatted`) reasoning text.
    pub text: String,
    /// Block is currently receiving deltas.
    pub live: bool,
    /// Total tokens reported by the provider, if known.
    pub tokens: Option<u64>,
    /// Elapsed time while live / final duration when finished.
    pub elapsed: Duration,
    /// Whether this block was seen streaming (affects the finished label).
    pub streamed: bool,
    /// User toggle overriding the default collapse state.
    pub expanded: Option<bool>,
    /// Text has already been through `format_thinking_for_display`.
    pub preformatted: bool,
    /// Pulse frame for the live spinner.
    pub pulse_frame: usize,
}

impl ThinkingBlockView {
    /// Display text per §3.3.
    #[must_use]
    pub fn display_text(&self, settings: &ThinkingDisplay) -> String {
        if self.preformatted {
            self.text.trim().to_owned()
        } else {
            format_thinking_for_display(&self.text, settings.prose_only)
                .trim()
                .to_owned()
        }
    }

    /// Whether the block has a body worth showing.
    #[must_use]
    pub fn displayable(&self, settings: &ThinkingDisplay) -> bool {
        if self.preformatted {
            has_displayable_thinking(&self.text, false)
        } else {
            has_displayable_thinking(&self.text, settings.prose_only)
        }
    }

    /// Default collapse: collapsed unless live and visible (chat.md §3.2:
    /// `collapsed: userToggle ?? (hideThinkingBlock || !live)`).
    #[must_use]
    pub fn collapsed(&self, settings: &ThinkingDisplay) -> bool {
        self.expanded.map_or(settings.hidden || !self.live, |e| !e)
    }

    /// Head line: live pulse (`Thinking`) or finished label with token
    /// count; the block is collapsible around it.
    #[must_use]
    pub fn head_line(&self, theme: &Theme) -> Line<'static> {
        if self.live {
            return pulse_line(self.pulse_frame, theme);
        }
        let mut spans = vec![Span::styled(
            thought_label(self.streamed, self.elapsed),
            Style::default()
                .fg(theme.color("text_muted"))
                .add_modifier(Modifier::ITALIC),
        )];
        if let Some(tokens) = self.tokens {
            spans.push(Span::styled(
                format!(" · {tokens} tokens"),
                Style::default().fg(theme.color("text_muted")),
            ));
        }
        Line::from(spans)
    }

    /// Full render: hidden live → pulse only; hidden finished → nothing;
    /// collapsed → head only; expanded → head + prose body.
    #[must_use]
    pub fn render(&self, settings: &ThinkingDisplay, theme: &Theme) -> Text<'static> {
        if settings.hidden {
            if self.live {
                return Text::from(self.head_line(theme));
            }
            return Text::default();
        }
        if self.collapsed(settings) {
            return Text::from(self.head_line(theme));
        }
        let mut lines = vec![self.head_line(theme)];
        if self.displayable(settings) {
            let body = self.display_text(settings);
            for l in body.lines() {
                lines.push(Line::styled(
                    l.to_owned(),
                    Style::default().fg(theme.color("text_muted")),
                ));
            }
        }
        Text::from(lines)
    }
}

/// Grapheme count of `s` (the reveal unit; TS `countGraphemes`).
#[must_use]
pub fn count_graphemes(s: &str) -> usize {
    s.graphemes(true).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prose_only_elides_code_blocks() {
        let src = "Let me check.\n```rust\nfn main() {}\n```\nDone.";
        let out = format_thinking_for_display(src, true);
        assert!(out.starts_with("Let me check...\n"));
        assert!(!out.contains("fn main"));
        assert!(out.contains("Done."));
        // raw mode is the identity modulo noise-stripping
        assert_eq!(format_thinking_for_display(src, false), src);
    }

    #[test]
    fn comment_noise_dropped() {
        let src = "<!-- -->\nreal thought\n<!--";
        let out = format_thinking_for_display(src, false);
        assert_eq!(out, "real thought");
    }

    #[test]
    fn placeholder_text_is_not_displayable() {
        assert!(!has_displayable_thinking(". .. … ", true));
        assert!(has_displayable_thinking("considering options", true));
    }

    #[test]
    fn labels_and_dwell() {
        assert_eq!(thought_label(false, Duration::from_secs(30)), "Thought");
        assert_eq!(
            thought_label(true, Duration::from_secs(9)),
            "Thought for 9s"
        );
        assert_eq!(
            thought_label(true, Duration::from_secs(95)),
            "Thought for 1m35s"
        );
        let dwell: Vec<u64> = (0..8).map(pulse_dwell_ms).collect();
        assert_eq!(*dwell.iter().min().unwrap(), PULSE_MIN_MS);
        assert_eq!(*dwell.iter().max().unwrap(), PULSE_MAX_MS);
        assert_eq!(pulse_dwell_ms(0), PULSE_MIN_MS);
        assert_eq!(pulse_dwell_ms(4), PULSE_MAX_MS);
    }

    #[test]
    fn render_hidden_collapsed_expanded() {
        let theme = Theme::default();
        let settings = ThinkingDisplay::default();
        let mut view = ThinkingBlockView {
            text: "step 1: read file".into(),
            live: true,
            ..ThinkingBlockView::default()
        };
        // live + visible + collapsed default → pulse head
        assert!(view.render(&settings, &theme).lines[0]
            .spans
            .iter()
            .any(|s| s.content.contains("Thinking")));
        view.live = false;
        view.streamed = true;
        view.elapsed = Duration::from_secs(4);
        // finished + collapsed → "Thought for 4s"
        let t = view.render(&settings, &theme);
        assert_eq!(t.lines.len(), 1);
        assert!(format!("{t:?}").contains("Thought for 4s"));
        // expanded → body shown
        view.expanded = Some(true);
        let t = view.render(&settings, &theme);
        assert!(t.lines.len() >= 2);
        // hidden → nothing when finished
        let hidden = ThinkingDisplay {
            hidden: true,
            ..ThinkingDisplay::default()
        };
        assert_eq!(view.render(&hidden, &theme).lines.len(), 0);
    }

    #[test]
    fn grapheme_units() {
        assert_eq!(count_graphemes("a\u{e9}\u{306}\u{1f1fa}\u{1f1f8}"), 3);
    }
}
