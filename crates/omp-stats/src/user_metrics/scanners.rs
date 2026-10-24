//! Hand-rolled matchers for the TS patterns the `regex` crate cannot express
//! (look-behind and back-references), plus the JS string primitives those
//! patterns depend on.
//!
//! `regex` supports look-*ahead* but not look-behind, so `SAD_EMOTICON_RE`,
//! `DOTTED_TOKEN_RE` and `BLAME_STOP_RE` are scanned here; `XML_TAG_PAIR_RE`
//! needs a back-reference, which `regex` also lacks. Leftmost-first /
//! backtracking semantics are reproduced by hand so counts match the JS engine.

/// JS `\s`: White_Space minus U+0085 (which JS omits), plus the BOM (which JS
/// `\s` and `trim` both include).
#[inline]
pub(super) fn is_js_space(c: char) -> bool {
    matches!(
        c,
        '\u{9}'..='\u{d}'
            | ' '
            | '\u{a0}'
            | '\u{1680}'
            | '\u{2000}'..='\u{200a}'
            | '\u{2028}'
            | '\u{2029}'
            | '\u{202f}'
            | '\u{205f}'
            | '\u{3000}'
            | '\u{feff}'
    )
}

/// JS `\w`: ASCII word characters only, even with the `u` flag.
#[inline]
pub(super) fn is_js_word(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// JS `[\w-]`.
#[inline]
pub(super) fn is_token_char(c: char) -> bool {
    is_js_word(c) || c == '-'
}

/// `String.prototype.trim` (same char set as JS `\s`).
pub(super) fn js_trim(text: &str) -> String {
    text.trim_matches(is_js_space).to_string()
}

/// `/(\p{Lu})\1\1/u` - a tripled uppercase letter (no back-references in
/// `regex`, and it is a plain char-by-char test anyway).
pub(super) fn has_tripled_upper(text: &str) -> bool {
    let chars: Vec<char> = text.chars().collect();
    chars
        .windows(3)
        .any(|w| w[0] == w[1] && w[1] == w[2] && w[0].is_uppercase() && w[0].is_alphabetic())
}

fn find_at(haystack: &[char], from: usize, needle: &[char]) -> Option<usize> {
    if needle.is_empty() || from > haystack.len() || needle.len() > haystack.len() - from {
        return None;
    }
    (from..=haystack.len() - needle.len()).find(|&j| &haystack[j..j + needle.len()] == needle)
}

/// `<([A-Za-z][\w-]*)\b[^>]*>` at `start`: the tag name plus the index just
/// past the open tag's `>`.
fn open_tag(chars: &[char], start: usize) -> Option<(String, usize)> {
    if chars.get(start) != Some(&'<') || !chars.get(start + 1)?.is_ascii_alphabetic() {
        return None;
    }
    let name_start = start + 1;
    let mut name_end = name_start + 1;
    while name_end < chars.len() && is_token_char(chars[name_end]) {
        name_end += 1;
    }
    // `\b` after the name only holds when the name ends on a word character, so
    // the greedy `[\w-]*` backtracks over trailing hyphens.
    while name_end > name_start + 1 && chars[name_end - 1] == '-' {
        name_end -= 1;
    }
    let name: String = chars[name_start..name_end].iter().collect();
    // `[^>]*>` cannot cross a `>`.
    let close = (name_end..chars.len()).find(|&j| chars[j] == '>')?;
    Some((name, close + 1))
}

/// `/<([A-Za-z][\w-]*)\b[^>]*>[\s\S]*?<\/\1>/g` replaced with `"\n"`.
///
/// The lazily-shortest `</name>` wins, exactly like the JS engine; a failed
/// attempt at one start position advances by a single character.
pub(super) fn replace_xml_tag_pairs(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut copied = 0usize;
    let mut i = 0usize;
    while i < chars.len() {
        if let Some((name, after_open)) = open_tag(&chars, i) {
            let closing: Vec<char> = format!("</{name}>").chars().collect();
            if let Some(close_at) = find_at(&chars, after_open, &closing) {
                let end = close_at + closing.len();
                out.extend(chars[copied..i].iter());
                out.push('\n');
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
