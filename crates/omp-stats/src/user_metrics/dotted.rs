//! Dotted-token stripping (`DOTTED_TOKEN_RE`) - hand-rolled because the JS
//! pattern uses two look-around assertions the `regex` crate cannot express.

use super::scanners::{is_js_space, is_js_word, is_token_char};

/// The `(?=$|[\s)"'\],:;!?]|\.(?!\w))` tail: is the candidate terminated?
fn has_terminator(chars: &[char], e: usize) -> bool {
    if e >= chars.len() {
        return true;
    }
    let c = chars[e];
    is_js_space(c)
        || matches!(c, ')' | '"' | '\'' | ']' | ',' | ':' | ';' | '!' | '?')
        || (c == '.' && !chars.get(e + 1).is_some_and(|n| is_js_word(*n)))
}

/// `(?<=^|[\s("'[])` - the leading-boundary look-behind.
fn has_lead_boundary(chars: &[char], i: usize) -> bool {
    i == 0 || is_js_space(chars[i - 1]) || matches!(chars[i - 1], '(' | '"' | '\'' | '[')
}

/// `[\w-]+(?:\.[\w-]+)+` at `start`: the longest candidate end that also
/// satisfies the trailing look-ahead (greedy match first, then backtracking).
fn token_end(chars: &[char], start: usize) -> Option<usize> {
    if !chars.get(start).is_some_and(|c| is_token_char(*c)) {
        return None;
    }
    let n = chars.len();
    let mut ends = Vec::new();
    let mut p = start;
    while p < n && is_token_char(chars[p]) {
        p += 1;
    }
    ends.push(p);
    while p < n && chars[p] == '.' && chars.get(p + 1).is_some_and(|c| is_token_char(*c)) {
        p += 1;
        while p < n && is_token_char(chars[p]) {
            p += 1;
        }
        ends.push(p);
    }
    if ends.len() < 2 {
        return None;
    }
    ends.iter()
        .rev()
        .copied()
        .find(|&e| has_terminator(chars, e))
}

/// `/DOTTED_TOKEN_RE/g` replaced with `" "`: filenames (`AGENTS.md`), dotted
/// identifiers (`Bun.file`) and versions (`1.2.3`). Stripped so the sentence
/// scan doesn't split them into all-caps fragments that register as yelling.
pub(super) fn replace_dotted_tokens(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut copied = 0usize;
    let mut i = 0usize;
    while i < chars.len() {
        if has_lead_boundary(&chars, i) {
            if let Some(end) = token_end(&chars, i) {
                out.extend(chars[copied..i].iter());
                out.push(' ');
                copied = end;
                i = end;
                continue;
            }
        }
        i += 1;
    }
    out.extend(chars[copied..].iter());
    out
}
