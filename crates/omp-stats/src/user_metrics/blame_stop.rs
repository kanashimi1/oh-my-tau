//! `/BLAME_STOP_RE/g` and `/SAD_EMOTICON_RE/g` - hand-rolled because both use
//! look-behind assertions the `regex` crate cannot express.

use super::scanners::{is_js_space, is_js_word};

/// `/SAD_EMOTICON_RE/g`: `(?<=^|[\s.!?])[:;]-?\(+`. Requires a leading
/// boundary so pasted code like `foo:(bar)` doesn't fire; `\(+` folds `:(((` into
/// one hit. Dot runs (`..` / `...`) are deliberately not counted: on a real
/// corpus they are dominated by neutral trail-offs and template placeholders.
pub(super) fn count_sad_emoticons(text: &str) -> usize {
    let chars: Vec<char> = text.chars().collect();
    let mut count = 0usize;
    let mut i = 0usize;
    while i < chars.len() {
        let lead_ok =
            i == 0 || is_js_space(chars[i - 1]) || matches!(chars[i - 1], '.' | '!' | '?');
        if lead_ok && matches!(chars[i], ':' | ';') {
            let mut j = i + 1;
            if chars.get(j) == Some(&'-') {
                j += 1;
            }
            if chars.get(j) == Some(&'(') {
                while chars.get(j) == Some(&'(') {
                    j += 1;
                }
                count += 1;
                i = j;
                continue;
            }
        }
        i += 1;
    }
    count
}

/// `stop` (case-insensitive) plus the trailing `\s+\w+ing\b`, returning the
/// index just past the match. Greedy `\w+` backtracks so that `ing` can follow.
fn stop_ing_at(chars: &[char], start: usize) -> Option<usize> {
    let n = chars.len();
    let mut i = start;
    while i < n && is_js_space(chars[i]) {
        i += 1;
    }
    let word: String = chars.get(i..i + 4)?.iter().collect();
    if !word.eq_ignore_ascii_case("stop") {
        return None;
    }
    i += 4;
    let gap = i;
    while i < n && is_js_space(chars[i]) {
        i += 1;
    }
    if i == gap {
        return None; // `\s+` needs at least one space
    }
    let word_start = i;
    while i < n && is_js_word(chars[i]) {
        i += 1;
    }
    for end in (word_start + 4..=i).rev() {
        if chars[end - 3..end].iter().copied().eq("ing".chars())
            && !(end < n && is_js_word(chars[end]))
        {
            return Some(end);
        }
    }
    None
}

/// `/BLAME_STOP_RE/g`: `(?:^|(?<=[.!?\n]))\s*stop\s+\w+ing\b` with the `i`/`m`
/// flags. `stop <verb>ing` is only frustration as an imperative, so it must open
/// a line or follow a sentence terminator. Returns the hit count.
pub(super) fn count_blame_stop(text: &str) -> usize {
    let chars: Vec<char> = text.chars().collect();
    let mut count = 0usize;
    let mut i = 0usize;
    while i < chars.len() {
        // `^` with the `m` flag: index 0 or just after a `\n`.
        let start_ok = i == 0 || chars[i - 1] == '\n' || matches!(chars[i - 1], '.' | '!' | '?');
        if start_ok {
            if let Some(end) = stop_ing_at(&chars, i) {
                count += 1;
                i = end;
                continue;
            }
        }
        i += 1;
    }
    count
}
