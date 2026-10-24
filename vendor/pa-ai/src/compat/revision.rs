//! Revision triples (`major.minor.patch`) extracted from model identifiers (TS `compat/revision.ts`).
//! Components are unsigned 8-bit; omitted components are zero.

pub type Revision = [u32; 3];

fn parse_component(value: &str) -> Option<u32> {
    if value.is_empty() {
        return None;
    }
    let mut out: u32 = 0;
    for b in value.bytes() {
        if !b.is_ascii_digit() {
            return None;
        }
        out = out * 10 + u32::from(b - b'0');
        if out > 255 {
            return None;
        }
    }
    Some(out)
}

/// Explicit revision literal: one to three components separated by `.` or `-`.
pub fn parse_revision(value: &str) -> Option<Revision> {
    let mut out = [0u32; 3];
    let mut count = 0;
    for part in value.split(['.', '-']) {
        if count == 3 {
            return None;
        }
        out[count] = parse_component(part)?;
        count += 1;
    }
    (count > 0).then_some(out)
}

/// Leading revision of an identifier tail that begins with a digit (`"4-6-turbo"` → `4.6.0`).
pub fn parse_revision_prefix(value: &str) -> Option<Revision> {
    let b = value.as_bytes();
    let mut out = [0u32; 3];
    let mut count = 0;
    let mut index = 0;
    while count < 3 {
        let start = index;
        while index < b.len() && b[index].is_ascii_digit() {
            index += 1;
        }
        let is_size_token = index < b.len() && b[index].is_ascii_alphabetic();
        let component = if is_size_token {
            None
        } else {
            parse_component(&value[start..index])
        };
        let Some(component) = component else {
            return if count > 0 { Some(out) } else { None };
        };
        out[count] = component;
        count += 1;
        let Some(&separator) = b.get(index) else {
            break;
        };
        let next_is_digit = b.get(index + 1).is_some_and(u8::is_ascii_digit);
        if (separator != b'.' && separator != b'-') || !next_is_digit {
            break;
        }
        index += 1;
    }
    Some(out)
}

pub fn compare_revision(a: &Revision, b: &Revision) -> std::cmp::Ordering {
    a.cmp(b)
}

pub fn format_revision(r: &Revision) -> String {
    format!("{}.{}.{}", r[0], r[1], r[2])
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RevisionOp {
    Gte,
    Gt,
    Lte,
    Lt,
    Eq,
}

#[derive(Debug, Clone)]
pub struct RevisionTerm {
    pub op: RevisionOp,
    pub revision: Revision,
}

impl RevisionOp {
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            ">=" => Self::Gte,
            ">" => Self::Gt,
            "<=" => Self::Lte,
            "<" => Self::Lt,
            "=" => Self::Eq,
            _ => return None,
        })
    }
}

pub fn revision_satisfies(revision: &Revision, terms: &[RevisionTerm]) -> bool {
    use std::cmp::Ordering::{Equal, Greater, Less};
    terms.iter().all(|t| {
        let cmp = compare_revision(revision, &t.revision);
        match t.op {
            RevisionOp::Gte => cmp != Less,
            RevisionOp::Gt => cmp == Greater,
            RevisionOp::Lte => cmp != Greater,
            RevisionOp::Lt => cmp == Less,
            RevisionOp::Eq => cmp == Equal,
        }
    })
}
