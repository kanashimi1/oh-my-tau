//! Resolve the syntactic block that begins on a given source line.
//!
//! Powers the hashline `replace block N:` operator: given a 1-indexed line,
//! parse the source with tree-sitter and return the line span of the outermost
//! named node that *begins* on that line (excluding the whole-file root). Brace
//! languages anchor a construct's block to its opening line, so pointing at the
//! line that opens an `if` / `function` / `struct` resolves to that construct's
//! full span; pointing at a continuation line or a lone closing delimiter
//! resolves to nothing.

use std::collections::BTreeSet;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tree_sitter::{Node, Point, TreeCursor};

use crate::{
    parse_cache::{MAX_ENTRY_SOURCE_BYTES, parse_cached},
    summary::{node_content_end_line, node_start_line, resolve_language},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockRangeOptions {
    /// Source code to inspect.
    pub code: String,
    /// Language alias (e.g. "rust", "typescript") used before path inference.
    pub lang: Option<String>,
    /// File path used to infer language by extension when `lang` is omitted.
    pub path: Option<String>,
    /// 1-indexed source line the block must begin on.
    pub line: u32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct BlockRange {
    /// 1-indexed inclusive first line of the resolved block.
    pub start_line: u32,
    /// 1-indexed inclusive last line of the resolved block.
    pub end_line: u32,
}

/// Count of leading space/tab bytes on `row` (0-indexed), i.e. the byte column
/// of the first content character. Returns `None` when `row` is out of range
/// or the line is blank / whitespace-only — there is no block to resolve there.
fn first_content_column(code: &str, row: usize) -> Option<usize> {
    let line = code.split('\n').nth(row)?;
    for (col, byte) in line.bytes().enumerate() {
        if byte != b' ' && byte != b'\t' {
            return Some(col);
        }
    }
    None
}

/// Resolve the block beginning on `options.line`.
///
/// Returns `None` (a soft "no block here", surfaced as a hard error one layer
/// up) when the language is unrecognized, the line is out of range / blank, no
/// node begins on that line, or the resolved subtree contains a syntax error.
pub fn block_range_at(options: BlockRangeOptions) -> Result<Option<BlockRange>> {
    let BlockRangeOptions {
        code,
        lang,
        path,
        line,
    } = options;
    if line == 0 || code.is_empty() {
        return Ok(None);
    }
    let Some(language) = resolve_language(lang.as_deref(), path.as_deref()) else {
        return Ok(None);
    };
    let row = (line - 1) as usize;
    let Some(col) = first_content_column(&code, row) else {
        return Ok(None);
    };

    let Some(tree) = parse_cached(&code, language)? else {
        return Ok(None);
    };
    let root = tree.root_node();

    // Query a one-column-wide range over the first content character rather
    // than a zero-width point. Some grammars (e.g. tree-sitter-swift) insert a
    // zero-width separator node at the start of a statement that follows a
    // blank line. An empty point range at that node's start gets absorbed into
    // the invisible node, which has no children and is not "relevant", so
    // `named_descendant_for_point_range` bubbles back up to the last visible
    // ancestor (the enclosing body, or the file root). That made `replace
    // block` on a line like `var body: some View {` preceded by a blank line
    // resolve to the whole enclosing type body and then fail. Spanning the
    // first character skips the zero-width node (its end is < the range end)
    // and forces the descent into the node that begins on `row`.
    let point = Point::new(row, col);
    let point_end = Point::new(row, col + 1);
    let Some(leaf) = root.named_descendant_for_point_range(point, point_end) else {
        return Ok(None);
    };
    // A leaf whose own start row is earlier than `row` means `point` landed on
    // a continuation line or a closing delimiter of a block that opened earlier
    // — there is no block *beginning* on line N.
    if leaf.start_position().row != row {
        return Ok(None);
    }
    // Climb to the outermost named ancestor that still begins on `row`,
    // excluding the whole-file root. The first parent that starts before `row`
    // stops the climb, and so does a statement-sequence container: it begins
    // exactly where its first statement does, so adopting it would swallow
    // every following sibling statement.
    let mut node = leaf;
    while let Some(parent) = node.parent() {
        if parent.id() == root.id() {
            break;
        }
        if parent.start_position().row != row {
            break;
        }
        if is_statement_sequence(parent, node) {
            break;
        }
        node = parent;
    }
    // Refuse degenerate error-recovery spans: a missing brace can make
    // tree-sitter wrap a huge region in an ERROR node. Checking only the
    // resolved node's subtree (not the whole file) keeps an unrelated syntax
    // error elsewhere from disabling the feature.
    if node.has_error() {
        return Ok(None);
    }
    Ok(Some(BlockRange {
        start_line: node_start_line(node),
        end_line: node_content_end_line(node),
    }))
}

/// Is `parent` a statement-sequence container that `node` merely opens? These
/// nodes wrap the sibling statements or entries of a body without a token of
/// their own, so they begin exactly where their first child does: Go and
/// PowerShell `statement_list`, Python/Starlark/Lua `block`, Ruby
/// `body_statement`, Swift/Kotlin `statements`, HCL/CMake `body`, Scala
/// `indented_block`, YAML `block_mapping`/`block_sequence`, Kotlin
/// `import_list`, PowerShell `hash_literal_body`.
///
/// The container is matched by kind because grammars expose no uniform
/// structural marker: field-less prefix runs such as Java/Kotlin/Swift
/// `modifiers` (one annotation per line) look identical but must be climbed
/// through to reach the declaration. Braced `block`s never match because they
/// begin at `{`; requiring the next sibling to start a later row at `node`'s
/// column also rules out a leading label (`'a: {` in Rust). Extras (comments)
/// are skipped when looking for that next sibling, so `if … {} // note` and
/// comment-only lines — even dedented ones — still stop the climb. When only
/// extras follow `node`, the container merely adds those trailing comments to
/// `node`'s span, so the climb stops there too rather than swallowing them.
fn is_statement_sequence(parent: Node<'_>, node: Node<'_>) -> bool {
    if !matches!(
        parent.kind(),
        "statement_list"
            | "block"
            | "body_statement"
            | "statements"
            | "body"
            | "indented_block"
            | "block_mapping"
            | "block_sequence"
            | "import_list"
            | "hash_literal_body"
    ) || parent.start_byte() != node.start_byte()
    {
        return false;
    }
    let end_row = node.end_position().row;
    let mut next = node.next_named_sibling();
    let mut skipped_extra = false;
    while let Some(sibling) = next.filter(|s| s.is_extra()) {
        skipped_extra = true;
        next = sibling.next_named_sibling();
    }
    match next {
        Some(next) => {
            let next_start = next.start_position();
            next_start.row > end_row && next_start.column == node.start_position().column
        }
        None => skipped_extra,
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct LineRange {
    /// 1-indexed inclusive first visible line.
    pub start_line: u32,
    /// 1-indexed inclusive last visible line.
    pub end_line: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnclosingBoundaryOptions {
    /// Source code to inspect.
    pub code: String,
    /// Language alias (e.g. "rust", "typescript") used before path inference.
    pub lang: Option<String>,
    /// File path used to infer language by extension when `lang` is omitted.
    pub path: Option<String>,
    /// 1-indexed inclusive visible line ranges (the lines actually shown).
    pub ranges: Vec<LineRange>,
}

/// Sort, drop invalid, and merge adjacent/overlapping ranges so visibility
/// tests can binary-search a non-overlapping list.
fn normalize_ranges(mut ranges: Vec<LineRange>) -> Vec<LineRange> {
    ranges.retain(|range| range.start_line > 0 && range.end_line >= range.start_line);
    ranges.sort_by(|a, b| {
        a.start_line
            .cmp(&b.start_line)
            .then(a.end_line.cmp(&b.end_line))
    });
    let mut merged: Vec<LineRange> = Vec::with_capacity(ranges.len());
    for range in ranges {
        if let Some(last) = merged.last_mut()
            && range.start_line <= last.end_line.saturating_add(1)
        {
            last.end_line = last.end_line.max(range.end_line);
            continue;
        }
        merged.push(range);
    }
    merged
}

fn is_visible(merged: &[LineRange], line: u32) -> bool {
    merged
        .binary_search_by(|range| {
            if line < range.start_line {
                std::cmp::Ordering::Greater
            } else if line > range.end_line {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .is_ok()
}

/// Does any visible line fall inside the inclusive line span `[start, end]`?
fn intersects_visible(merged: &[LineRange], start: u32, end: u32) -> bool {
    // `merged` is sorted and non-overlapping, so the first range that can
    // possibly overlap is the first one whose `end_line` reaches `start`.
    let idx = merged.partition_point(|range| range.end_line < start);
    merged.get(idx).is_some_and(|range| range.start_line <= end)
}

/// Depth-first walk collecting boundary lines from every multi-line named node
/// that straddles a visible-range edge. A single reused [`TreeCursor`] keeps
/// the traversal allocation-free.
fn collect_boundaries(cursor: &mut TreeCursor<'_>, merged: &[LineRange], out: &mut BTreeSet<u32>) {
    let node = cursor.node();
    // Prune whole subtrees that cannot contribute. A node contributes only when
    // one of its own endpoint lines is visible, and both of those lines lie
    // inside its raw row span; every descendant's span is contained in this
    // one, so a span holding no visible line rules out this node *and*
    // everything beneath it. Without the prune the walk is O(nodes in file)
    // even though the answer is bounded by the window size — which made the
    // traversal cost roughly twice the parse on a large file.
    let raw_start = node.start_position().row.saturating_add(1) as u32;
    let raw_end = node.end_position().row.saturating_add(1) as u32;
    if !intersects_visible(merged, raw_start, raw_end) {
        return;
    }
    // Skip the whole-file root: its only "boundary" is EOF, never a useful
    // matching line (mirrors `block_range_at` excluding the root).
    if node.is_named() && node.parent().is_some() {
        let start = node_start_line(node);
        let end = node_content_end_line(node);
        if end > start {
            let start_visible = is_visible(merged, start);
            let end_visible = is_visible(merged, end);
            // Opener shown, closer off-window → surface the closer (and vice
            // versa). A node fully inside or fully outside the window adds
            // nothing.
            if start_visible && !end_visible {
                out.insert(end);
            } else if end_visible && !start_visible {
                out.insert(start);
            }
        }
    }
    if cursor.goto_first_child() {
        loop {
            collect_boundaries(cursor, merged, out);
            if !cursor.goto_next_sibling() {
                break;
            }
        }
        cursor.goto_parent();
    }
}

/// Generalize "show the matching bracket" to every tree-sitter block: for each
/// multi-line named node whose span crosses the visible window, return the
/// boundary line sitting *outside* that window.
///
/// - node opens on a visible line but closes past the window → its closing line
/// - node closes on a visible line but opens before the window → its opening
///   line
///
/// Because the trigger is an endpoint *inside* the window, the result is
/// bounded by the window size (not nesting depth), exactly like a bracket scan
/// — but it also covers indentation languages (Python) and uses real syntactic
/// spans.
///
/// Returns `None` when the language is unrecognized or the source fails to
/// parse / carries a syntax error (caller falls back to a lexical bracket
/// scan); `Some(sorted unique boundary lines)` otherwise (possibly empty).
pub fn enclosing_block_boundaries(options: EnclosingBoundaryOptions) -> Result<Option<Vec<u32>>> {
    let EnclosingBoundaryOptions {
        code,
        lang,
        path,
        ranges,
    } = options;
    let merged = normalize_ranges(ranges);
    if code.is_empty() || merged.is_empty() {
        return Ok(Some(Vec::new()));
    }
    let Some(language) = resolve_language(lang.as_deref(), path.as_deref()) else {
        return Ok(None);
    };
    let Some(tree) = parse_cached(&code, language)? else {
        return Ok(None);
    };
    let root = tree.root_node();
    // A file-level syntax error makes error-recovery spans unreliable; defer to
    // the lexical scanner rather than emit boundaries off a broken tree.
    if root.has_error() {
        return Ok(None);
    }

    let mut boundaries = BTreeSet::new();
    let mut cursor = root.walk();
    collect_boundaries(&mut cursor, &merged, &mut boundaries);
    Ok(Some(boundaries.into_iter().collect()))
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct NodeSpan {
    /// 1-indexed inclusive first line of the node.
    pub start_line: u32,
    /// 1-indexed inclusive last content line of the node.
    pub end_line: u32,
    /// Tree-sitter grammar node kind (e.g. `attribute_item`, `function_item`).
    pub kind: String,
}

/// Parse `code` into the shared tree cache.
///
/// The next block query over the same source and language is then a cache
/// hit. Does nothing for empty source or an unrecognized language, where the
/// queries never parse either, nor for a source the cache will not retain
/// (the tree would be discarded and parsed again by the query).
pub fn warm_parse(code: &str, lang: Option<&str>, path: Option<&str>) -> Result<()> {
    if !code.is_empty()
        && code.len() <= MAX_ENTRY_SOURCE_BYTES
        && let Some(language) = resolve_language(lang, path)
    {
        parse_cached(code, language)?;
    }
    Ok(())
}

/// Named-node chain containing `options.line`, innermost-first, excluding the
/// whole-file root.
///
/// The chain descends through the line's first content character, so
/// single-line nodes beginning on the line (attributes, decorators, one-line
/// statements) come first, followed by every enclosing construct up to — but
/// not including — the file root. Callers use it to classify a line by grammar
/// node kind and to enumerate enclosing construct end lines.
///
/// ERROR and MISSING recovery nodes are skipped rather than failing the whole
/// chain: an unrelated syntax error elsewhere in the file leaves healthy
/// ancestors' spans meaningful.
///
/// Returns `None` when the language is unrecognized, the line is out of
/// range / blank, or the source fails to parse entirely.
pub fn node_chain_at(options: BlockRangeOptions) -> Result<Option<Vec<NodeSpan>>> {
    let BlockRangeOptions {
        code,
        lang,
        path,
        line,
    } = options;
    if line == 0 || code.is_empty() {
        return Ok(None);
    }
    let Some(language) = resolve_language(lang.as_deref(), path.as_deref()) else {
        return Ok(None);
    };
    let row = (line - 1) as usize;
    let Some(col) = first_content_column(&code, row) else {
        return Ok(None);
    };
    let Some(tree) = parse_cached(&code, language)? else {
        return Ok(None);
    };
    let root = tree.root_node();
    // One-column-wide range for the same zero-width-node reason as
    // `block_range_at` above.
    let point = Point::new(row, col);
    let point_end = Point::new(row, col + 1);
    let Some(leaf) = root.named_descendant_for_point_range(point, point_end) else {
        return Ok(None);
    };
    let mut chain = Vec::new();
    let mut node = Some(leaf);
    while let Some(current) = node {
        if current.id() == root.id() {
            break;
        }
        if current.is_named() && !current.is_error() && !current.is_missing() {
            chain.push(NodeSpan {
                start_line: node_start_line(current),
                end_line: node_content_end_line(current),
                kind: current.kind().to_string(),
            });
        }
        node = current.parent();
    }
    Ok(Some(chain))
}
