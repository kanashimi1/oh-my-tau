//! Streaming reveal — the port of
//! `pi-coding-agent/src/modes/controllers/streaming-reveal.ts` and
//! `tool-args-reveal.ts` (spec chat.md §8, interactive.md §2.3).
//!
//! The assistant stream is revealed at a 30 fps cadence in **grapheme**
//! units: each tick advances the revealed prefix by [`next_step`] graphemes
//! so a backlog catches up over ~8 frames. Tool-call arguments are revealed
//! by the same cadence in [`ToolArgsRevealController`].

use std::collections::HashMap;

use unicode_segmentation::UnicodeSegmentation;

use crate::chat::thinking::{count_graphemes, format_thinking_for_display, ThinkingDisplay};

/// 30 fps frame interval (TS `STREAMING_REVEAL_FRAME_MS`).
pub const FRAME_MS: u64 = 1000 / 30;
/// Minimum graphemes revealed per tick (`MIN_STEP`).
pub const MIN_STEP: usize = 3;
/// Backlog catch-up horizon in frames (`CATCHUP_FRAMES`).
pub const CATCHUP_FRAMES: usize = 8;

/// `nextStep(backlog)`: catch up over ~8 frames, never slower than
/// [`MIN_STEP`] per tick.
#[must_use]
pub fn next_step(backlog: usize) -> usize {
    usize::max(MIN_STEP, backlog.div_ceil(CATCHUP_FRAMES))
}

/// One revealable assistant content block. Thinking text is kept in
/// **display form** (already prose-folded) so the component renders it
/// verbatim, mirroring TS `rawThinking`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RevealBlock {
    /// Assistant markdown text.
    Text(String),
    /// Reasoning; holds the display-formatted text plus the raw source.
    Thinking {
        /// Display text (per [`ThinkingDisplay`]).
        display: String,
        /// Raw model output (kept for expand-after-finalize).
        raw: String,
    },
}

/// `visibleUnits` (chat.md §8): turn raw `(text, thinking)` block pairs into
/// revealable blocks, honoring `hideThinkingBlock` and `proseOnlyThinking`.
#[must_use]
pub fn visible_blocks(
    blocks: &[(Option<String>, Option<String>)],
    settings: &ThinkingDisplay,
) -> Vec<RevealBlock> {
    blocks
        .iter()
        .filter_map(|pair| match pair {
            (Some(t), _) => Some(RevealBlock::Text(t.clone())),
            (_, Some(t)) => {
                if settings.hidden {
                    None
                } else {
                    Some(RevealBlock::Thinking {
                        display: format_thinking_for_display(t, settings.prose_only),
                        raw: t.clone(),
                    })
                }
            }
            _ => None,
        })
        .collect()
}

/// Total visible graphemes across blocks.
#[must_use]
pub fn visible_units(blocks: &[RevealBlock]) -> usize {
    blocks
        .iter()
        .map(|b| match b {
            RevealBlock::Text(t) | RevealBlock::Thinking { display: t, .. } => count_graphemes(t),
        })
        .sum()
}

/// Slice blocks in order to a revealed grapheme prefix
/// (TS `buildDisplayMessage`): the last partially revealed block is cut at a
/// grapheme boundary; later blocks render empty.
#[must_use]
pub fn reveal_prefix(blocks: &[RevealBlock], revealed: usize) -> Vec<RevealBlock> {
    let mut out = Vec::with_capacity(blocks.len());
    let mut left = revealed;
    for b in blocks {
        if left == 0 {
            out.push(match b {
                RevealBlock::Text(_) => RevealBlock::Text(String::new()),
                RevealBlock::Thinking { raw, .. } => RevealBlock::Thinking {
                    display: String::new(),
                    raw: raw.clone(),
                },
            });
            continue;
        }
        let text = match b {
            RevealBlock::Text(t) | RevealBlock::Thinking { display: t, .. } => t,
        };
        let total = count_graphemes(text);
        if left >= total {
            out.push(b.clone());
            left -= total;
        } else {
            let cut: String = text.graphemes(true).take(left).collect();
            out.push(match b {
                RevealBlock::Text(_) => RevealBlock::Text(cut),
                RevealBlock::Thinking { raw, .. } => RevealBlock::Thinking {
                    display: cut,
                    raw: raw.clone(),
                },
            });
            left = 0;
        }
    }
    out
}

/// Incremental per-block grapheme counter (TS `BlockUnitCounter`): only the
/// suffix after the last safely-counted boundary is re-segmented on append.
#[derive(Debug, Default)]
pub struct BlockUnitCounter {
    counted: String,
    count: usize,
    /// Bytes at the tail not yet counted (recounted on append).
    tail: String,
}

impl BlockUnitCounter {
    /// Append stream text; recounts only the uncounted tail.
    pub fn push_str(&mut self, s: &str) {
        self.tail.push_str(s);
        // A grapheme cluster never spans more than a handful of chars; keep a
        // small safety tail so combining marks / emoji ZWJ sequences at the
        // boundary are counted exactly once, in order.
        const SAFETY: usize = 16;
        let char_count = self.tail.chars().count();
        if char_count > SAFETY {
            let boundary = self
                .tail
                .char_indices()
                .nth(char_count - SAFETY)
                .map_or(self.tail.len(), |(i, _)| i);
            let (done, rest) = self.tail.split_at(boundary);
            self.count += count_graphemes(done);
            self.counted.push_str(done);
            self.tail = rest.to_owned();
        }
    }

    /// Total graphemes seen so far.
    #[must_use]
    pub fn count(&self) -> usize {
        self.count + count_graphemes(&self.tail)
    }
}

/// Top-level (depth-1) string args keys revealed progressively per tool
/// (TS `STREAMING_STRING_KEYS_BY_TOOL`, chat.md §8).
#[must_use]
pub fn streaming_string_keys(tool: &str) -> &'static [&'static str] {
    match tool {
        "write" => &["content"],
        "edit" => &["input", "_input"],
        "eval" => &["code"],
        _ => &[],
    }
}

/// Smooth streaming reveal of assistant blocks
/// (TS `StreamingRevealController`).
///
/// Snapshots grow by appending, so grapheme bookkeeping is incremental: a
/// new snapshot segments only the text after each block's last stable
/// boundary, and a tick resumes cutting from the previous cut. A grapheme
/// boundary between two characters depends only on what precedes it (UAX
/// #29), so boundaries inside the old text survive an append; only the old
/// end may merge with what follows.
#[derive(Debug, Clone)]
pub struct StreamingRevealController {
    target: Vec<RevealBlock>,
    /// Grapheme bookkeeping per `target` block.
    units: Vec<BlockUnits>,
    total: usize,
    revealed: usize,
    smooth: bool,
    /// Set when the target changed while caught up; the next tick coalesces
    /// delta bursts (TS `#targetDirty`).
    target_dirty: bool,
    /// The last cut: a grapheme boundary inside one `target` block.
    cursor: RevealCursor,
}

/// Grapheme count of one target block.
#[derive(Debug, Clone, Copy, Default)]
struct BlockUnits {
    /// Graphemes in the block.
    total: usize,
    /// Byte offset where the last grapheme starts (0 when empty): every
    /// boundary up to here survives appends; the block's end may not.
    last_start: usize,
}

/// `units` graphemes of `target[block]` end at byte `byte`.
#[derive(Debug, Clone, Copy, Default)]
struct RevealCursor {
    block: usize,
    units: usize,
    byte: usize,
}

/// The text a block reveals (thinking reveals its display form).
fn unit_text(block: &RevealBlock) -> &str {
    match block {
        RevealBlock::Text(t) | RevealBlock::Thinking { display: t, .. } => t,
    }
}

/// Count `text`'s graphemes given that `base_units` of them end at the
/// boundary `base_byte`.
fn units_from(text: &str, base_units: usize, base_byte: usize) -> BlockUnits {
    let mut out = BlockUnits {
        total: base_units,
        last_start: base_byte,
    };
    for (i, _) in text[base_byte..].grapheme_indices(true) {
        out.total += 1;
        out.last_start = base_byte + i;
    }
    out
}

impl StreamingRevealController {
    /// `begin`: with smooth streaming off, everything is revealed at once
    /// (still transient until `message_end` seals it).
    #[must_use]
    pub fn new(smooth: bool) -> Self {
        Self {
            target: Vec::new(),
            units: Vec::new(),
            total: 0,
            revealed: 0,
            smooth,
            target_dirty: false,
            cursor: RevealCursor::default(),
        }
    }

    /// Replace the cumulative stream snapshot (chat.md §2.3: snapshots, not
    /// raw deltas).
    pub fn set_target(&mut self, blocks: Vec<RevealBlock>) {
        let mut units = Vec::with_capacity(blocks.len());
        for (idx, block) in blocks.iter().enumerate() {
            let text = unit_text(block);
            // The old count carries over when the block only grew.
            let grown = self
                .target
                .get(idx)
                .filter(|old| {
                    std::mem::discriminant(*old) == std::mem::discriminant(block)
                        && text.starts_with(unit_text(old))
                })
                .and(self.units.get(idx).copied());
            units.push(match grown {
                Some(old) if old.total > 0 => units_from(text, old.total - 1, old.last_start),
                _ => units_from(text, 0, 0),
            });
            if self.cursor.block == idx
                && !grown.is_some_and(|old| self.cursor.byte <= old.last_start)
            {
                self.cursor = RevealCursor {
                    block: idx,
                    ..RevealCursor::default()
                };
            }
        }
        self.total = units.iter().map(|u| u.total).sum();
        if !self.smooth {
            self.revealed = self.total;
        }
        self.target = blocks;
        self.units = units;
        self.target_dirty = true;
    }

    /// The stream includes tool calls: the leading segment finishes
    /// immediately (full text before the tool card, TS `unchangedSinceSnap`).
    pub fn snap_to_tool_call(&mut self) {
        self.revealed = self.total;
    }

    /// Advance one frame; returns the revealed snapshot when anything
    /// changed. Returns `None` when caught up and the target is unchanged.
    pub fn tick(&mut self) -> Option<Vec<RevealBlock>> {
        let backlog = self.total.saturating_sub(self.revealed);
        if backlog == 0 {
            if self.target_dirty {
                self.target_dirty = false;
                return Some(self.target.clone());
            }
            return None;
        }
        self.revealed += next_step(backlog);
        self.target_dirty = false;
        Some(self.prefix(self.revealed))
    }

    /// [`reveal_prefix`]`(target, revealed)`, cutting from the last cut.
    fn prefix(&mut self, revealed: usize) -> Vec<RevealBlock> {
        let mut out = Vec::with_capacity(self.target.len());
        let mut left = revealed;
        for (idx, block) in self.target.iter().enumerate() {
            if left == 0 {
                out.push(match block {
                    RevealBlock::Text(_) => RevealBlock::Text(String::new()),
                    RevealBlock::Thinking { raw, .. } => RevealBlock::Thinking {
                        display: String::new(),
                        raw: raw.clone(),
                    },
                });
                continue;
            }
            let total = self.units[idx].total;
            if left >= total {
                out.push(block.clone());
                left -= total;
                continue;
            }
            let text = unit_text(block);
            let mut from = self.cursor;
            if from.block != idx || from.units > left {
                from = RevealCursor {
                    block: idx,
                    ..RevealCursor::default()
                };
            }
            let byte = text[from.byte..]
                .grapheme_indices(true)
                .nth(left - from.units)
                .map_or(text.len(), |(i, _)| from.byte + i);
            self.cursor = RevealCursor {
                block: idx,
                units: left,
                byte,
            };
            let cut = text[..byte].to_owned();
            out.push(match block {
                RevealBlock::Text(_) => RevealBlock::Text(cut),
                RevealBlock::Thinking { raw, .. } => RevealBlock::Thinking {
                    display: cut,
                    raw: raw.clone(),
                },
            });
            left = 0;
        }
        out
    }

    /// Fully revealed snapshot (used at `message_end`).
    #[must_use]
    pub fn finish(&mut self) -> Vec<RevealBlock> {
        self.revealed = self.total;
        self.target.clone()
    }

    /// Graphemes revealed so far.
    #[must_use]
    pub fn revealed(&self) -> usize {
        self.revealed
    }

    /// True while a [`Self::tick`] would still change the snapshot (a
    /// reveal backlog, or a coalesced target not yet published) — the
    /// controller's frame timer runs exactly while this holds.
    #[must_use]
    pub fn is_animating(&self) -> bool {
        self.total > self.revealed || self.target_dirty
    }
}

/// One per-tool-call reveal entry (TS `ToolArgsRevealController`).
#[derive(Debug, Default)]
struct ArgsEntry {
    target: String,
    /// Revealed byte offset (args JSON is revealed by raw bytes).
    revealed: usize,
    last_display: String,
}

/// Progressive reveal of streaming tool-call arguments, one shared 30 fps
/// cadence across calls.
#[derive(Debug, Default)]
pub struct ToolArgsRevealController {
    entries: HashMap<String, ArgsEntry>,
}

impl ToolArgsRevealController {
    /// An empty controller.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// `updateArgs`: the cumulative raw args JSON snapshot for one call.
    pub fn update_args(&mut self, call_id: &str, args: &str) {
        let e = self.entries.entry(call_id.to_owned()).or_default();
        e.target = args.to_owned();
        // Keep the offset monotone within bounds.
        if e.revealed > e.target.len() {
            e.revealed = e.target.len();
        }
    }

    /// `tool_execution_start` finishes the reveal with the authoritative
    /// args.
    pub fn finish(&mut self, call_id: &str) {
        if let Some(e) = self.entries.get_mut(call_id) {
            e.revealed = e.target.len();
        }
    }

    /// Drop a call's entry (card finalized and removed).
    pub fn remove(&mut self, call_id: &str) {
        self.entries.remove(call_id);
    }

    /// Advance one frame; returns `(call_id, display_args)` for every entry
    /// whose revealed prefix changed. The prefix is cut on a UTF-8 char
    /// boundary.
    pub fn tick(&mut self) -> Vec<(String, String)> {
        let mut changed = Vec::new();
        for (id, e) in &mut self.entries {
            let backlog = e.target.len().saturating_sub(e.revealed);
            if backlog == 0 {
                continue;
            }
            e.revealed = (e.revealed + next_step(backlog)).min(e.target.len());
            let mut end = e.revealed;
            while end > 0 && !e.target.is_char_boundary(end) {
                end -= 1;
            }
            let display = e.target[..end].to_owned();
            if display != e.last_display {
                e.last_display = display.clone();
                changed.push((id.clone(), display));
            }
        }
        changed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn next_step_catches_up_over_eight_frames() {
        assert_eq!(next_step(0), MIN_STEP);
        assert_eq!(next_step(10), MIN_STEP);
        assert_eq!(next_step(80), 10);
        assert_eq!(next_step(1600), 200);
        // Geometric decay: after 8 ticks most of an 800-unit backlog is gone,
        // and it fully drains well under a second and a half of frames.
        let mut revealed = 0;
        let total = 800;
        let mut ticks = 0;
        while revealed < total && ticks < 100 {
            let backlog = total - revealed;
            revealed = (revealed + next_step(backlog)).min(total);
            ticks += 1;
        }
        assert!(revealed >= 500, "8 ticks revealed only {revealed}");
        assert!(ticks < 60, "drain took {ticks} ticks");
    }

    #[test]
    fn reveal_prefix_slices_by_graphemes() {
        let blocks = vec![
            RevealBlock::Text("héllo".to_owned()),
            RevealBlock::Thinking {
                display: "deep".to_owned(),
                raw: "deep".to_owned(),
            },
        ];
        assert_eq!(visible_units(&blocks), 9);
        let r = reveal_prefix(&blocks, 3);
        assert_eq!(r[0], RevealBlock::Text("hél".to_owned()));
        assert_eq!(
            r[1],
            RevealBlock::Thinking {
                display: String::new(),
                raw: "deep".to_owned()
            }
        );
        let r = reveal_prefix(&blocks, 7);
        assert_eq!(
            r[1],
            RevealBlock::Thinking {
                display: "de".to_owned(),
                raw: "deep".to_owned()
            }
        );
        // fully revealed is the identity
        assert_eq!(reveal_prefix(&blocks, 100), blocks);
    }

    #[test]
    fn controller_reveals_progressively_then_finishes() {
        let mut c = StreamingRevealController::new(true);
        c.set_target(vec![RevealBlock::Text("a".repeat(30))]);
        let mut last = 0;
        for _ in 0..12 {
            if let Some(snapshot) = c.tick() {
                let n = visible_units(&snapshot);
                assert!(n >= last);
                last = n;
                if n == 30 {
                    break;
                }
            }
        }
        assert_eq!(last, 30);
        assert_eq!(c.finish().len(), 1);
        assert_eq!(c.revealed(), 30);
    }

    #[test]
    fn incremental_counts_and_cuts_match_full_recount() {
        // Appends that split grapheme clusters: combining marks, ZWJ emoji,
        // skin tones, flags (regional-indicator pairs), CRLF, Hangul jamo.
        const PIECES: &[&str] = &[
            "a",
            "e\u{301}",
            "\u{301}",
            "\u{200d}",
            "👩",
            "\u{200d}💻",
            "🏽",
            "🇯",
            "🇵",
            "\r",
            "\n",
            "\r\n",
            "日本",
            "ᄀ",
            "ᅡ",
            "ᆨ",
            " ",
            "naïve ",
            "x\u{20e3}",
        ];
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        let mut rand = |n: usize| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % n as u64) as usize
        };
        for _ in 0..300 {
            let mut c = StreamingRevealController::new(true);
            let mut texts = vec![String::new(); 1 + rand(3)];
            let kinds: Vec<bool> = texts.iter().map(|_| rand(3) == 0).collect();
            for _ in 0..40 {
                // Grow a random block (earlier ones too, as thinking can).
                let k = rand(texts.len());
                for _ in 0..1 + rand(4) {
                    texts[k].push_str(PIECES[rand(PIECES.len())]);
                }
                let blocks: Vec<RevealBlock> = texts
                    .iter()
                    .zip(&kinds)
                    .map(|(t, &thinking)| {
                        if thinking {
                            RevealBlock::Thinking {
                                display: t.clone(),
                                raw: t.clone(),
                            }
                        } else {
                            RevealBlock::Text(t.clone())
                        }
                    })
                    .collect();
                c.set_target(blocks.clone());
                assert_eq!(c.total, visible_units(&blocks), "{texts:?}");
                for _ in 0..rand(3) {
                    if let Some(snapshot) = c.tick() {
                        let expected = if c.revealed >= c.total {
                            blocks.clone()
                        } else {
                            reveal_prefix(&blocks, c.revealed)
                        };
                        assert_eq!(snapshot, expected, "{texts:?} at {}", c.revealed);
                    }
                }
                if rand(8) == 0 {
                    // A rewritten snapshot (not an append) recounts.
                    texts[k] = texts[k].chars().rev().collect();
                }
            }
        }
    }

    #[test]
    fn unsmooth_reveals_everything_at_once() {
        let mut c = StreamingRevealController::new(false);
        c.set_target(vec![RevealBlock::Text("hello world".to_owned())]);
        let s = c.tick().expect("first tick publishes target");
        assert_eq!(visible_units(&s), 11);
    }

    #[test]
    fn block_counter_counts_incrementally() {
        let mut ctr = BlockUnitCounter::default();
        let mut full = BlockUnitCounter::default();
        for g in ["a", "b", "🇺🇸", "é", "̆", "c", "d", "e", "f", "g", "h"] {
            ctr.push_str(g);
            full.push_str(g);
            assert_eq!(ctr.count(), full.count(), "after appending {g}");
        }
        let mut whole = BlockUnitCounter::default();
        whole.push_str("ab🇺🇸é̆cdefgh");
        assert_eq!(ctr.count(), whole.count());
    }

    #[test]
    fn tool_args_reveal_progressive_and_finishable() {
        let mut c = ToolArgsRevealController::new();
        let args = r#"{"path":"/tmp/x.rs","content":"fn main() {}"}"#;
        c.update_args("call1", args);
        let mut saw_progress = false;
        for _ in 0..40 {
            for (id, d) in c.tick() {
                if id == "call1" && !d.is_empty() && d.len() < args.len() {
                    saw_progress = true;
                }
            }
        }
        assert!(saw_progress);
        c.finish("call1");
        c.tick();
        assert_eq!(c.entries["call1"].revealed, args.len());
        assert_eq!(streaming_string_keys("write"), &["content"]);
        assert_eq!(streaming_string_keys("edit"), &["input", "_input"]);
        assert_eq!(streaming_string_keys("eval"), &["code"]);
        assert!(streaming_string_keys("bash").is_empty());
    }

    #[test]
    fn visible_blocks_honors_settings() {
        let raw = Some("check\n```rust\ncode\n```\ndone".to_owned());
        let vis = visible_blocks(&[(None, raw.clone())], &ThinkingDisplay::default());
        assert!(matches!(&vis[0], RevealBlock::Thinking { display, .. }
            if display.contains("...") && !display.contains("code")));
        let hidden = ThinkingDisplay {
            hidden: true,
            ..ThinkingDisplay::default()
        };
        assert!(visible_blocks(&[(None, raw)], &hidden).is_empty());
    }
}
