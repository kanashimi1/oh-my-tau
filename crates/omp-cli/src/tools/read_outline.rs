//! Rust item map for `read`, built on `omp-ladder` (the item-level parsing crate).
//!
//! A read of a `.rs` file that does not show the whole file ends with the file's
//! item map: top-level items with exact line ranges plus the members of
//! `impl`/`trait`/`mod` bodies, so the next range is chosen without paging
//! through the file or spending a separate `grep` turn. `path.rs:outline`
//! returns the map alone. The map is syntax-derived (no name resolution, no
//! type information); `macro_rules!` bodies and macro invocations are one row
//! and are never descended into.
//!
//! Automatic attachment is opt-in (`OMP_READ_OUTLINE=1`) and happens once per
//! file content; the explicit `:outline` selector works regardless.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::fmt::Write as _;
use std::sync::{Arc, LazyLock};

use omp_ladder::cache::CstCache;
use omp_ladder::cst::{Cst, TopKind};
use omp_ladder::hash::{ContentHash, hash_str};
use parking_lot::Mutex;
use pa_agent::types::{AgentToolResult, ToolResultContent};

/// Largest source a map is built for.
const MAX_FILE_BYTES: usize = 8_000_000;
/// Rows of an attached map; the rest stays behind `path.rs:outline`.
const ATTACH_MAX_ROWS: usize = 48;
/// Rows of an explicit `path.rs:outline` map.
const FULL_MAX_ROWS: usize = 400;
/// A signature head is cut to this many characters.
const MAX_WIDTH: usize = 96;
/// Lines joined into one signature head at most.
const MAX_HEAD_LINES: usize = 4;
/// Parse cache budget, in bytes of source.
const CST_CACHE_BYTES: usize = 8 * 1024 * 1024;
/// Nesting listed by an attached map: members of a top-level block.
const ATTACH_MAX_DEPTH: usize = 1;
/// Nesting listed by an explicit map: one block level more.
const FULL_MAX_DEPTH: usize = 2;

static ENABLED: LazyLock<bool> = LazyLock::new(|| flag("OMP_READ_OUTLINE"));
static SHOWN: LazyLock<Mutex<HashMap<PathBuf, ContentHash>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
static CST: LazyLock<Mutex<CstCache>> =
    LazyLock::new(|| Mutex::new(CstCache::new(CST_CACHE_BYTES)));

fn flag(name: &str) -> bool {
    std::env::var_os(name).is_some_and(|value| value == "1")
}

/// Whether `path` is a Rust source file (what `omp-ladder` parses).
#[must_use]
pub(crate) fn is_rust(path: &Path) -> bool {
    path.extension()
        .and_then(std::ffi::OsStr::to_str)
        .is_some_and(|ext| ext.eq_ignore_ascii_case("rs"))
}

struct Row {
    depth: usize,
    start: usize,
    end: usize,
    head: String,
}

/// Byte offset of every line start (`0` first).
fn line_starts(text: &str) -> Vec<usize> {
    let mut starts = vec![0];
    starts.extend(
        text.bytes()
            .enumerate()
            .filter_map(|(i, b)| (b == b'\n').then_some(i + 1)),
    );
    starts
}

/// 1-based line containing `offset`.
fn line_of(starts: &[usize], offset: usize) -> usize {
    starts.partition_point(|&s| s <= offset)
}

/// First `cfg`/`test` attribute of a line, short enough for a row.
fn tag(line: &str) -> Option<String> {
    if line.contains("[test]") {
        return Some("[test]".to_owned());
    }
    let at = line.find("[cfg(")?;
    let rest = &line[at..];
    let end = rest.find(")]").map_or(rest.len(), |i| i + 2);
    Some(clip(&rest[..end], 28))
}

/// Head of an item: byte offset of its declaration inside the item text (doc
/// comments and outer attributes are skipped), the declaration up to its body,
/// and the first `cfg`/`test` attribute when there is one.
fn head(item: &str) -> (usize, String, Option<String>) {
    let mut offset = 0usize;
    let mut start = 0usize;
    let mut brackets = 0usize;
    let mut attr = None;
    let mut head = String::new();
    let mut taken = 0usize;
    for line in item.split_inclusive('\n') {
        let trimmed = line.trim();
        if taken == 0 {
            let in_attr = brackets > 0 || trimmed.starts_with("#[");
            if trimmed.is_empty() || trimmed.starts_with("//") || in_attr {
                if in_attr {
                    if attr.is_none() {
                        attr = tag(trimmed);
                    }
                    for ch in trimmed.chars() {
                        match ch {
                            '[' => brackets += 1,
                            ']' => brackets = brackets.saturating_sub(1),
                            _ => {}
                        }
                    }
                }
                offset += line.len();
                continue;
            }
            start = offset + (line.len() - line.trim_start().len());
        }
        if taken > 0 {
            head.push(' ');
        }
        head.push_str(trimmed);
        taken += 1;
        if trimmed.contains('{') || trimmed.ends_with(';') || taken >= MAX_HEAD_LINES {
            break;
        }
        offset += line.len();
    }
    let mut cut = head.len();
    if let Some(i) = head.find(" where ") {
        cut = i;
    }
    if let Some(i) = head.find('{').filter(|&i| i < cut) {
        cut = i;
    }
    let declaration = head[..cut].trim_end().trim_end_matches(';').trim_end();
    (start, clip(declaration, MAX_WIDTH), attr)
}

/// Cut to at most `max` characters, marking the cut.
fn clip(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_owned();
    }
    let end = text
        .char_indices()
        .nth(max.saturating_sub(1))
        .map_or(text.len(), |(i, _)| i);
    format!("{}…", &text[..end])
}

fn rows_of(
    cst: &Cst,
    starts: &[usize],
    base: usize,
    depth: usize,
    max_depth: usize,
    rows: &mut Vec<Row>,
) {
    let mut uses: Option<(usize, usize, usize)> = None;
    for node in &cst.nodes {
        if node.kind == TopKind::Use {
            match &mut uses {
                None => uses = Some((node.span.start, node.span.end, 1)),
                Some((_, end, count)) => {
                    *end = node.span.end;
                    *count += 1;
                }
            }
            continue;
        }
        if let Some((from, to, count)) = uses.take() {
            rows.push(Row {
                depth,
                start: line_of(starts, base + from),
                end: line_of(starts, base + to.saturating_sub(1)),
                head: format!("use ×{count}"),
            });
        }
        if node.kind == TopKind::InnerAttr {
            continue;
        }
        let (decl, declaration, attr) = head(cst.text(node.span));
        let head = match attr {
            Some(attr) => format!("{attr} {declaration}"),
            None => declaration,
        };
        let head = if node.ambiguous {
            format!("{head}  [span not trusted]")
        } else {
            head
        };
        rows.push(Row {
            depth,
            start: line_of(starts, base + node.span.start + decl),
            end: line_of(starts, base + node.span.end.saturating_sub(1)),
            head,
        });
        if let (Some(body), true) = (
            node.body_span.filter(|_| depth < max_depth),
            matches!(node.kind, TopKind::Impl | TopKind::Trait | TopKind::Mod),
        ) {
            let inner = &cst.source[body.start + 1..body.end.saturating_sub(1)];
            let nested = omp_ladder::cst::parse_top_level(inner);
            rows_of(&nested, starts, base + body.start + 1, depth + 1, max_depth, rows);
        }
    }
    if let Some((from, to, count)) = uses.take() {
        rows.push(Row {
            depth,
            start: line_of(starts, base + from),
            end: line_of(starts, base + to.saturating_sub(1)),
            head: format!("use ×{count}"),
        });
    }
}

/// The item map of a Rust source, or `None` when it has no items.
#[must_use]
pub(crate) fn outline(
    text: &str,
    display: &str,
    selector_base: &str,
    max_rows: usize,
    max_depth: usize,
) -> Option<String> {
    let starts = line_starts(text);
    let total_lines = text.lines().count();
    let cst: Arc<Cst> = CST.lock().get_or_parse(text);
    let mut rows = Vec::new();
    rows_of(&cst, &starts, 0, 0, max_depth, &mut rows);
    if rows.is_empty() {
        return None;
    }
    let hidden = rows.len().saturating_sub(max_rows);
    let hint = if hidden > 0 {
        format!(", {selector_base}:outline for the full map")
    } else {
        String::new()
    };
    let mut out = format!(
        "[outline of {display}: {} items, {total_lines} lines; read {selector_base}:N-M for a range{hint}]",
        rows.len()
    );
    for row in &rows[..rows.len() - hidden] {
        let indent = "  ".repeat(row.depth);
        out.push('\n');
        let _ = write!(out, "{indent}L{:>5}-{:>5} {}", row.start, row.end, row.head);
    }
    if hidden > 0 {
        out.push('\n');
        let _ = write!(
            out,
            "  ... {hidden} more items not listed; {selector_base}:outline lists all"
        );
    }
    Some(out)
}

/// `path.rs:outline`: the full map of an already-loaded Rust source.
///
/// # Errors
/// The file is not Rust source, is too large, or has no items.
pub(crate) fn full(
    path: &Path,
    text: &str,
    display: &str,
    selector_base: &str,
) -> anyhow::Result<String> {
    if !is_rust(path) {
        anyhow::bail!("Selector ':outline' is supported for Rust source files (.rs) only.");
    }
    if text.len() > MAX_FILE_BYTES {
        anyhow::bail!("Selector ':outline': file is too large ({} bytes).", text.len());
    }
    outline(text, display, selector_base, FULL_MAX_ROWS, FULL_MAX_DEPTH)
        .ok_or_else(|| anyhow::anyhow!("No Rust items found in '{display}'."))
}

/// Append the map to a partial read of a Rust file, once per file content.
pub(crate) fn append(
    result: AgentToolResult,
    path: &Path,
    text: &str,
    display: &str,
    selector_base: &str,
    partial: bool,
) -> AgentToolResult {
    append_with(result, path, text, display, selector_base, partial, *ENABLED)
}

fn append_with(
    mut result: AgentToolResult,
    path: &Path,
    text: &str,
    display: &str,
    selector_base: &str,
    partial: bool,
    enabled: bool,
) -> AgentToolResult {
    if !enabled || !partial || result.is_error == Some(true) || !is_rust(path) {
        return result;
    }
    let version = hash_str(text);
    if SHOWN.lock().get(path) == Some(&version) {
        return result;
    }
    let Some(map) = outline(text, display, selector_base, ATTACH_MAX_ROWS, ATTACH_MAX_DEPTH)
    else {
        return result;
    };
    SHOWN.lock().insert(path.to_path_buf(), version);
    if let Some(ToolResultContent::Text(t)) = result.content.first_mut() {
        t.text.push_str("\n\n");
        t.text.push_str(&map);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::{append_with, is_rust, outline};
    use pa_agent::types::{AgentToolResult, ToolResultContent};

    const SOURCE: &str = "//! Module docs.\n\
\n\
use std::collections::HashMap;\n\
use std::path::PathBuf;\n\
use std::sync::Arc;\n\
\n\
/// A span.\n\
#[derive(Clone, Copy)]\n\
pub struct Span {\n\
    pub start: usize,\n\
}\n\
\n\
/// Builds.\n\
#[cfg_attr(\n\
    feature = \"x\",\n\
    derive(Clone)\n\
)]\n\
pub fn build(\n\
    src: &str,\n\
    depth: usize,\n\
) -> Span {\n\
    Span { start: depth }\n\
}\n\
\n\
impl Span {\n\
    fn start_of(&self) -> usize {\n\
        self.start\n\
    }\n\
\n\
    pub fn end_of(&self) -> usize {\n\
        self.start + 1\n\
    }\n\
}\n\
\n\
#[cfg(test)]\n\
mod tests {\n\
    #[test]\n\
    fn t() {\n\
        assert_eq!(1, 1);\n\
    }\n\
}\n";

    #[test]
    fn outline_lists_items_with_ranges_and_block_members() {
        let map = outline(SOURCE, "span.rs", "span.rs", 48, 2).expect("map");
        let lines: Vec<&str> = map.lines().collect();
        assert_eq!(
            lines[0],
            "[outline of span.rs: 8 items, 41 lines; read span.rs:N-M for a range]"
        );
        let expected = [
            "L    3-    5 use ×3",
            "L    9-   11 pub struct Span",
            "L   18-   23 pub fn build( src: &str, depth: usize, ) -> Span",
            "L   25-   33 impl Span",
            "  L   26-   28 fn start_of(&self) -> usize",
            "  L   30-   32 pub fn end_of(&self) -> usize",
            "L   36-   41 [cfg(test)] mod tests",
            "  L   38-   40 [test] fn t()",
        ];
        assert_eq!(&lines[1..], &expected, "{map}");
    }

    #[test]
    fn long_maps_are_cut_and_point_at_the_full_one() {
        let mut text = String::new();
        for i in 0..60 {
            text.push_str(&format!("fn f{i}() -> usize {{ {i} }}\n"));
        }
        let map = outline(&text, "big.rs", "big.rs", 48, 1).expect("map");
        assert!(map.starts_with("[outline of big.rs: 60 items, 60 lines"), "{map}");
        assert!(map.contains("read big.rs:N-M for a range, big.rs:outline for the full map]"));
        assert!(map.contains("fn f47()") && !map.contains("fn f48("), "{map}");
        assert!(map.ends_with("... 12 more items not listed; big.rs:outline lists all"));
        let full = outline(&text, "big.rs", "big.rs", 400, 1).expect("map");
        assert!(full.contains("fn f59()"));
    }

    #[test]
    fn heads_skip_docs_and_multiline_attributes_and_are_cut() {
        let map = outline(SOURCE, "span.rs", "span.rs", 48, 1).expect("map");
        assert!(map.contains("L   18-   23 pub fn build("), "{map}");
        assert!(!map.contains("cfg_attr") && !map.contains("///"), "{map}");
        let long = "pub fn very_long_signature_name(a: usize, b: usize, c: usize, d: usize, e: usize, f: usize) -> usize {\n    1\n}\n";
        let map = outline(long, "l.rs", "l.rs", 48, 1).expect("map");
        let row = map.lines().nth(1).expect("row");
        assert!(row.contains("pub fn very_long_signature_name(") && row.ends_with('…'), "{row}");
        // "L" + start(5) + "-" + end(5) + " " precede the head.
        assert_eq!(row[13..].chars().count(), 96, "{row}");
    }

    #[test]
    fn attach_is_once_per_content_and_only_for_partial_reads() {
        let path = std::path::Path::new("/ws/span.rs");
        let fresh = || AgentToolResult::text("1: //! Module docs.");
        let text = |r: &AgentToolResult| match r.content.first() {
            Some(ToolResultContent::Text(t)) => t.text.clone(),
            _ => String::new(),
        };
        assert!(
            text(&append_with(fresh(), path, SOURCE, "span.rs", "span.rs", true, true))
                .contains("\n\n[outline of span.rs: ")
        );
        assert_eq!(
            text(&append_with(fresh(), path, SOURCE, "span.rs", "span.rs", true, true)),
            "1: //! Module docs."
        );
        assert_eq!(
            text(&append_with(fresh(), path, SOURCE, "span.rs", "span.rs", false, true)),
            "1: //! Module docs."
        );
        let c_path = std::path::Path::new("/ws/a.c");
        assert_eq!(
            text(&append_with(fresh(), c_path, SOURCE, "a.c", "a.c", true, true)),
            "1: //! Module docs."
        );
        assert!(!is_rust(c_path) && is_rust(path));
        let changed = format!("{SOURCE}pub fn later() {{}}\n");
        assert!(
            text(&append_with(fresh(), path, &changed, "span.rs", "span.rs", true, true))
                .contains("pub fn later()")
        );
        let mut error = fresh();
        error.is_error = Some(true);
        assert_eq!(
            text(&append_with(error, path, SOURCE, "span.rs", "span.rs", true, true)),
            "1: //! Module docs."
        );
    }
}
