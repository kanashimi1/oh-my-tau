//! Source-location handling, split into two explicitly separated
//! classes:
//!
//! - **Observable locations** — `file!()`, `line!()`, `column!()`.
//!   These macros produce *values* the compiled program can read, so
//!   their source positions are semantic. They are represented
//!   symbolically ([`SymbolicSpan`]: stable item id + span id +
//!   content anchor), resolved to line/column only late
//!   ([`resolve`]), and their dependency set is hashed into cache
//!   keys ([`loc_dep_hash`]). Relocating an observable use changes
//!   the dependency hash and therefore invalidates dependent caches;
//!   an observable use is never relocated merely to keep a
//!   diagnostic cache warm.
//! - **Diagnostic-only cold spans** — positions used purely to render
//!   diagnostics. They carry no semantic weight and may be relocated
//!   ([`relocate`]) without any semantic invalidation.
//!
//! Detection is token-level over the whole file (macro bodies
//! included, since their tokens are in the stream), attributed to the
//! enclosing top-level item. This is syntax-derived, not full rustc
//! HIR: a user macro named `line` invoked as `line!(..)` is
//! conservatively treated as observable (the safe direction — it can
//! only over-invalidate, never serve a stale location).

use serde::{Deserialize, Serialize};

use crate::cst::Cst;
use crate::hash::{hash_bytes, ContentHash};
use crate::item::ItemId;
use crate::lex::{Span, Tk};

/// Stable identifier of a location within one scan of one source.
pub type SpanId = u32;

/// Which observable location macro a use refers to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LocMacro {
    /// `file!()`
    File,
    /// `line!()`
    Line,
    /// `column!()`
    Column,
}

/// Whether a span's position is observable by compiled code.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LocClass {
    /// `file!()`/`line!()`/`column!()`: position is a program value.
    Observable,
    /// Diagnostic rendering only; freely relocatable.
    DiagnosticOnly,
}

/// A location represented symbolically: the enclosing item (by
/// stable id within its index), the scan-local span id, and a content
/// anchor (hash of the enclosing item's source text) that is stable
/// across pure relocation of the item but changes when the item's
/// text — including the relative position of the use — changes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SymbolicSpan {
    /// Enclosing top-level item, or `None` for file scope.
    pub item: Option<ItemId>,
    /// Scan-local span id.
    pub span_id: SpanId,
    /// Hash anchoring the use to its enclosing item's content.
    pub anchor: ContentHash,
}

/// One detected `file!()`/`line!()`/`column!()` use.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LocUse {
    /// The symbolic identity of the use.
    pub symbolic: SymbolicSpan,
    /// Which macro is used.
    pub which: LocMacro,
    /// Current byte span of the `ident!` tokens (recomputed on every
    /// scan; never used as identity).
    pub span: Span,
}

fn loc_macro_of(text: &str) -> Option<LocMacro> {
    Some(match text {
        "file" => LocMacro::File,
        "line" => LocMacro::Line,
        "column" => LocMacro::Column,
        _ => return None,
    })
}

fn next_nonblank(cst: &Cst, mut i: usize) -> Option<usize> {
    while let Some(t) = cst.tokens.get(i) {
        if !t.kind.is_blank() {
            return Some(i);
        }
        i += 1;
    }
    None
}

/// Anchor for a use: hashes the macro kind, the enclosing item's full
/// text, and the use's byte offset *relative to the item start* — so
/// moving a use within or across items changes the anchor, while
/// unrelated edits elsewhere in the file do not.
fn anchor(cst: &Cst, which: LocMacro, item: Option<usize>, rel: usize) -> ContentHash {
    let mut s = String::new();
    s.push(match which {
        LocMacro::File => 'f',
        LocMacro::Line => 'l',
        LocMacro::Column => 'c',
    });
    s.push('\u{1f}');
    if let Some(idx) = item {
        if let Some(node) = cst.nodes.get(idx) {
            s.push_str(cst.text(node.span));
        }
    } else {
        s.push_str("<file-scope>");
        s.push_str(cst.text(Span {
            start: 0,
            end: cst.source_len,
        }));
    }
    s.push('\u{1e}');
    s.push_str(&rel.to_string());
    crate::hash::hash_str(&s)
}

/// Scan a CST for observable location-macro uses. Every token is
/// inspected, including tokens inside item bodies and macro
/// invocation bodies; a use is attributed to the enclosing top-level
/// item by span containment (or file scope if none contains it).
#[must_use]
pub fn scan_loc_uses(cst: &Cst) -> Vec<LocUse> {
    let mut uses = Vec::new();
    for (i, t) in cst.tokens.iter().enumerate() {
        if !matches!(t.kind, Tk::Ident | Tk::RawIdent) {
            continue;
        }
        let Some(which) = loc_macro_of(t.text(&cst.source)) else {
            continue;
        };
        // Require `ident ! <delim>`; the delimiter requirement also
        // rejects `x != y` (where `!` is followed by `=`).
        let Some(bang) = next_nonblank(cst, i + 1) else { continue };
        if cst.tokens.get(bang).map(|b| b.kind) != Some(Tk::Exclaim) {
            continue;
        }
        let Some(delim) = next_nonblank(cst, bang + 1) else {
            continue;
        };
        if !matches!(
            cst.tokens.get(delim).map(|d| d.kind),
            Some(Tk::OpenParen | Tk::OpenBracket | Tk::OpenBrace)
        ) {
            continue;
        }
        let item = cst
            .nodes
            .iter()
            .position(|n| n.span.start <= t.span.start && t.span.end <= n.span.end);
        let rel = item.map_or(t.span.start, |idx| {
            t.span.start.saturating_sub(cst.nodes[idx].span.start)
        });
        uses.push(LocUse {
            symbolic: SymbolicSpan {
                item,
                span_id: u32::try_from(uses.len()).unwrap_or(u32::MAX),
                anchor: anchor(cst, which, item, rel),
            },
            which,
            span: Span {
                start: t.span.start,
                end: cst.tokens[delim].span.start,
            },
        });
    }
    uses
}

/// Hash of the observable-location dependency set. Cache entries
/// whose values could depend on resolved `file!`/`line!`/`column!`
/// positions must include this hash in their keys: adding, removing,
/// or relocating any observable use changes it.
#[must_use]
pub fn loc_dep_hash(uses: &[LocUse]) -> ContentHash {
    let mut buf = Vec::with_capacity(uses.len() * 24);
    for u in uses {
        buf.extend_from_slice(&u.symbolic.anchor.0.to_le_bytes());
        // Identity is symbolic, but line!/column!() values are not: even a
        // whole-item shift changes their observable result. Keep the stable
        // anchor for tracking and add the current absolute span to the
        // semantic dependency, so every relocation invalidates.
        buf.extend_from_slice(&u.span.start.to_le_bytes());
        buf.extend_from_slice(&u.span.end.to_le_bytes());
    }
    hash_bytes(&buf)
}

/// A location resolved late from the current source.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedLoc {
    /// File name, supplied by the caller (the parser does not know
    /// paths); `None` when unknown.
    pub file: Option<String>,
    /// 1-based line.
    pub line: u32,
    /// 1-based column (in bytes, matching rustc's `column!`).
    pub col: u32,
}

/// Resolve a byte offset to a 1-based (line, column) pair against the
/// current source. This is the *late* half of symbolic spans: callers
/// cache symbolic identities and call this only when a concrete
/// position is actually needed.
#[must_use]
pub fn resolve_offset(src: &str, offset: usize) -> (u32, u32) {
    let off = offset.min(src.len());
    let prefix = src.get(..off).unwrap_or("");
    let line = u32::try_from(prefix.matches('\n').count()).unwrap_or(u32::MAX) + 1;
    let col_start = prefix.rfind('\n').map_or(0, |p| p + 1);
    let col = u32::try_from(off - col_start).unwrap_or(u32::MAX) + 1;
    (line, col)
}

/// Resolve an observable use to a concrete location against the
/// current CST and a caller-supplied file name.
#[must_use]
pub fn resolve_use(cst: &Cst, use_: &LocUse, file: Option<&str>) -> ResolvedLoc {
    let (line, col) = resolve_offset(&cst.source, use_.span.start);
    ResolvedLoc {
        file: file.map(str::to_owned),
        line,
        col,
    }
}

/// Why a relocation was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RelocateError {
    /// The span is [`LocClass::Observable`]; relocating it would
    /// change an observable program value and is refused. Update the
    /// source and re-scan instead — the changed [`loc_dep_hash`]
    /// performs the semantic invalidation.
    ObservablePinned,
}

/// A tracked span with its observability class. Diagnostic-only
/// spans may be relocated freely; observable spans are pinned to
/// their symbolic identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrackedSpan {
    /// Symbolic identity.
    pub symbolic: SymbolicSpan,
    /// Observability class.
    pub class: LocClass,
    /// Current byte span.
    pub span: Span,
}

/// Relocate a tracked span to a new byte position (e.g. after
/// inserting text earlier in the file). Diagnostic-only spans move
/// without any semantic invalidation; observable spans refuse.
///
/// # Errors
/// [`RelocateError::ObservablePinned`] for observable spans.
pub fn relocate(tracked: &mut TrackedSpan, new_span: Span) -> Result<(), RelocateError> {
    if tracked.class == LocClass::Observable {
        return Err(RelocateError::ObservablePinned);
    }
    tracked.span = new_span;
    Ok(())
}
