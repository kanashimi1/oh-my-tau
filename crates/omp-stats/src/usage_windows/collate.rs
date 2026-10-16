//! `localeCompare` approximation for the TS sorts in `usage-windows.ts`.

use std::cmp::Ordering;

/// Punctuation and symbols in the order ICU root collation sorts them (ASCII).
const PUNCT_ORDER: &str = " _-,;:!?.'\"()[]{}@*/\\&#%`^+<=>|~$";

/// Primary collation weight of one char; `None` for ignorable controls.
fn primary_weight(c: char) -> Option<u32> {
    if let Some(i) = PUNCT_ORDER.find(c) {
        return Some(i as u32 + 1);
    }
    if c.is_ascii_digit() {
        return Some(100 + (c as u32 - '0' as u32));
    }
    if c.is_ascii_alphabetic() {
        return Some(200 + (c.to_ascii_lowercase() as u32 - 'a' as u32));
    }
    if c.is_ascii_control() {
        return None;
    }
    Some(1000 + c.to_lowercase().next().unwrap_or(c) as u32)
}

/// Approximation of JS `a.localeCompare(b)` (ICU root collation) used by the TS
/// sorts. Exact for ASCII text as the TS sees it (provider ids, limit ids,
/// emails): punctuation < digits < letters, letters compare case-insensitively,
/// and on a full tie the lowercase char sorts first. Non-ASCII chars sort after
/// ASCII by lowercased code point, which is not ICU order.
#[must_use]
pub fn locale_compare(a: &str, b: &str) -> Ordering {
    let primary = |s: &str| s.chars().filter_map(primary_weight).collect::<Vec<u32>>();
    let ord = primary(a).cmp(&primary(b));
    if ord != Ordering::Equal {
        return ord;
    }
    for (x, y) in a.chars().zip(b.chars()) {
        if x != y {
            // Same primary weight, different char: case decides, lowercase first.
            return (x.is_uppercase(), x).cmp(&(y.is_uppercase(), y));
        }
    }
    a.len().cmp(&b.len())
}
