//! Revision triples (`major.minor.patch`) extracted from model identifiers.
//!
//! Port of `catalog/src/compat/revision.ts`. The compat engine compares
//! revisions as three unsigned 8-bit components; omitted components are zero
//! (`4.6` ≡ `4.6.0`). Shared by the compile-time rule compiler and the runtime
//! taxonomy/cascade.

/// A parsed `major.minor.patch` revision.
pub type Revision = (u8, u8, u8);

fn parse_component(value: &str) -> Option<u8> {
    if value.is_empty() {
        return None;
    }
    let mut out: u16 = 0;
    for b in value.bytes() {
        if !(48..=57).contains(&b) {
            return None;
        }
        out = out * 10 + u16::from(b - 48);
        if out > 255 {
            return None;
        }
    }
    Some(out as u8)
}

/// Parses an explicit revision literal: one to three components separated by
/// `.` or `-` (`"4.6"`, `"4-6-1"`). Returns `None` on malformed input.
pub fn parse_revision(value: &str) -> Option<Revision> {
    let mut out = [0u8, 0u8, 0u8];
    let mut count = 0;
    for part in value.split(['.', '-']) {
        if count == 3 {
            return None;
        }
        let component = parse_component(part)?;
        out[count] = component;
        count += 1;
    }
    if count > 0 {
        Some((out[0], out[1], out[2]))
    } else {
        None
    }
}

/// Extracts a leading revision from an identifier tail that begins with a
/// digit: up to three components separated by `.` or by `-` followed by a
/// digit (`"4-6-turbo"` → `[4, 6, 0]`). Mirrors o2 `parse_revision_prefix`.
/// A component whose digits run directly into a letter is a parameter-count
/// or size token (`qwen3-32b`, `llama-3.3-70b`), never a revision component.
pub fn parse_revision_prefix(value: &str) -> Option<Revision> {
    let bytes = value.as_bytes();
    let mut out = [0u8, 0u8, 0u8];
    let mut count = 0usize;
    let mut index = 0usize;
    while count < 3 {
        let start = index;
        while index < bytes.len() && (48..=57).contains(&bytes[index]) {
            index += 1;
        }
        let trailing = if index < bytes.len() { bytes[index] } else { 0 };
        let is_size_token = (97..=122).contains(&trailing) || (65..=90).contains(&trailing);
        let component = if is_size_token {
            None
        } else {
            parse_component(&value[start..index])
        };
        if component.is_none() {
            return if count > 0 {
                Some((out[0], out[1], out[2]))
            } else {
                None
            };
        }
        out[count] = component.unwrap_or(0);
        count += 1;
        if index >= bytes.len() {
            break;
        }
        let separator = bytes[index];
        let next = *bytes.get(index + 1).unwrap_or(&0);
        if (separator != b'.' && separator != b'-') || !(48..=57).contains(&next) {
            break;
        }
        index += 1;
    }
    Some((out[0], out[1], out[2]))
}

/// Lexicographic triple comparison: negative, zero, or positive.
pub fn compare_revision(a: Revision, b: Revision) -> i32 {
    let ordering = a.0.cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2));
    ordering as i32
}

/// Renders a revision as its canonical `major.minor.patch` string.
pub fn format_revision(revision: Revision) -> String {
    format!("{}.{}.{}", revision.0, revision.1, revision.2)
}
