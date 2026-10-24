//! Structural source summaries powered by tree-sitter.

use std::{collections::BTreeSet, path::Path};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tree_sitter::Node;

use crate::{language::SupportLang, parse_cache::parse_cached};

const DEFAULT_MIN_BODY_LINES: u32 = 4;
const DEFAULT_MIN_COMMENT_LINES: u32 = 6;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummaryOptions {
    /// Source code to summarize.
    pub code: String,
    /// Language alias (e.g. "rust", "typescript") used before path inference.
    pub lang: Option<String>,
    /// File path used to infer language by extension when `lang` is omitted.
    pub path: Option<String>,
    /// Minimum total node lines before eliding a body/literal node.
    pub min_body_lines: Option<u32>,
    /// Minimum total comment lines before eliding a multiline block comment.
    pub min_comment_lines: Option<u32>,
    /// Target visible-line count for BFS unfold. Starting from every elidable
    /// span folded, this progressively reveals outer-then-inner spans until
    /// the visible line count meets the target. `None` or `0` disables BFS
    /// and keeps only the outermost elisions (every nested span stays hidden
    /// behind its parent).
    pub unfold_until_lines: Option<u32>,
    /// Hard ceiling for BFS unfold. If a candidate unfold would push the
    /// visible count past this value, revert that step and stop. Defaults
    /// to `unfold_until_lines * 2` when omitted (with `unfold_until_lines`
    /// itself as the floor so a single threshold also works).
    pub unfold_limit_lines: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SummarySegment {
    /// "kept" or "elided".
    pub kind: String,
    /// 1-based inclusive start line.
    pub start_line: u32,
    /// 1-based inclusive end line.
    pub end_line: u32,
    /// Verbatim text for kept segments; absent for elided segments.
    pub text: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SummaryResult {
    /// Canonical language name when parsing succeeded.
    pub language: Option<String>,
    /// True when tree-sitter parsed the source without syntax errors.
    pub parsed: bool,
    /// True when at least one elision span was emitted.
    pub elided: bool,
    /// Total source lines.
    pub total_lines: u32,
    /// Kept/elided segments in source order.
    pub segments: Vec<SummarySegment>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct LineSpan {
    start: u32,
    end: u32,
}

impl LineSpan {
    const fn lines(self) -> u32 {
        self.end.saturating_sub(self.start).saturating_add(1)
    }
}

/// One elidable region plus its directly-nested elidable descendants. The
/// forest is built by AST traversal in source order, so `children` reflects
/// the structural hierarchy: a child's span is always strictly contained
/// within its parent's.
#[derive(Debug)]
struct SpanNode {
    span: LineSpan,
    children: Vec<usize>,
}

/// Flat arena of elidable spans organized as a forest. `roots` lists the
/// topmost spans — anything whose AST ancestors held no elidable container.
#[derive(Debug, Default)]
struct ElidableForest {
    nodes: Vec<SpanNode>,
    roots: Vec<usize>,
}

impl ElidableForest {
    fn push(&mut self, parent: Option<usize>, span: LineSpan) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(SpanNode {
            span,
            children: Vec::new(),
        });
        match parent {
            Some(p) => self.nodes[p].children.push(idx),
            None => self.roots.push(idx),
        }
        idx
    }
}

/// BFS unfold. Start with every root span folded (matches the legacy
/// outermost-only behavior) and progressively replace folded spans with
/// their elidable children, breadth-first, until the visible line count
/// reaches `unfold_until`. A candidate unfold whose revealed lines would
/// push the visible count past `unfold_limit` is skipped — that span stays
/// folded and its subtree is not explored — while the BFS keeps unfolding
/// the remaining queued siblings. A single oversized, un-unfoldable leaf
/// (e.g. a `<style>` raw-text block) therefore no longer aborts the whole
/// pass and starve its siblings. `unfold_until == 0` short-circuits to the
/// legacy behavior.
fn select_folded_spans(
    forest: &ElidableForest,
    total_lines: u32,
    unfold_until: u32,
    unfold_limit: u32,
) -> Vec<LineSpan> {
    use std::collections::{HashSet, VecDeque};

    let nodes = &forest.nodes;
    let mut folded: HashSet<usize> = forest.roots.iter().copied().collect();
    if unfold_until == 0 || folded.is_empty() {
        return folded.into_iter().map(|i| nodes[i].span).collect();
    }

    let folded_line_total: u32 = folded.iter().map(|&i| nodes[i].span.lines()).sum();
    let mut visible = total_lines.saturating_sub(folded_line_total);
    let mut queue: VecDeque<usize> = forest.roots.iter().copied().collect();

    while let Some(idx) = queue.pop_front() {
        if visible >= unfold_until {
            break;
        }
        if !folded.contains(&idx) {
            continue;
        }
        let node = &nodes[idx];
        let child_line_total: u32 = node.children.iter().map(|&c| nodes[c].span.lines()).sum();
        // Unfolding swaps the parent's span for its children's, so the visible
        // gain is the difference between them. `saturating_sub` keeps the math
        // honest if children somehow over-cover (they shouldn't by construction).
        let revealed = node.span.lines().saturating_sub(child_line_total);
        let new_visible = visible.saturating_add(revealed);
        if new_visible > unfold_limit {
            continue;
        }
        folded.remove(&idx);
        for &c in &node.children {
            folded.insert(c);
            queue.push_back(c);
        }
        visible = new_visible;
    }

    folded.into_iter().map(|i| nodes[i].span).collect()
}

pub fn summarize_code(options: SummaryOptions) -> Result<SummaryResult> {
    let source = options.code;
    let total_lines = count_lines(&source);
    if source.is_empty() {
        return Ok(unparsed_result(source, total_lines));
    }

    let Some(language) = resolve_language(options.lang.as_deref(), options.path.as_deref()) else {
        return Ok(unparsed_result(source, total_lines));
    };

    let Some(tree) = parse_cached(&source, language)? else {
        return Ok(unparsed_result(source, total_lines));
    };
    let root = tree.root_node();
    if root.has_error() {
        return Ok(unparsed_result(source, total_lines));
    }

    let min_body_lines = options
        .min_body_lines
        .unwrap_or(DEFAULT_MIN_BODY_LINES)
        .max(2);
    let min_comment_lines = options
        .min_comment_lines
        .unwrap_or(DEFAULT_MIN_COMMENT_LINES)
        .max(4);
    let unfold_until = options.unfold_until_lines.unwrap_or(0);
    let unfold_limit = options
        .unfold_limit_lines
        .unwrap_or_else(|| unfold_until.saturating_mul(2));
    let mut forest = ElidableForest::default();
    collect_elidable_tree(
        root,
        None,
        language,
        min_body_lines,
        min_comment_lines,
        &mut forest,
    );
    let spans = select_folded_spans(&forest, total_lines, unfold_until, unfold_limit);
    let spans = normalize_spans(spans, total_lines);
    let segments = build_segments(&source, total_lines, &spans);

    Ok(SummaryResult {
        language: Some(language.canonical_name().to_string()),
        parsed: true,
        elided: !spans.is_empty(),
        total_lines,
        segments,
    })
}

pub(crate) fn resolve_language(lang: Option<&str>, path: Option<&str>) -> Option<SupportLang> {
    if let Some(lang) = lang.map(str::trim).filter(|lang| !lang.is_empty()) {
        return SupportLang::from_alias(lang);
    }
    let path = path?.trim();
    if path.is_empty() {
        return None;
    }
    SupportLang::from_path(Path::new(path))
}

fn unparsed_result(source: String, total_lines: u32) -> SummaryResult {
    let segments = if source.is_empty() {
        Vec::new()
    } else {
        vec![SummarySegment {
            kind: "kept".to_string(),
            start_line: 1,
            end_line: total_lines,
            text: Some(source),
        }]
    };
    SummaryResult {
        language: None,
        parsed: false,
        elided: false,
        total_lines,
        segments,
    }
}

fn count_lines(source: &str) -> u32 {
    if source.is_empty() {
        0
    } else {
        source.lines().count().max(1).min(u32::MAX as usize) as u32
    }
}

fn collect_elidable_tree(
    node: Node<'_>,
    elidable_parent: Option<usize>,
    language: SupportLang,
    min_body_lines: u32,
    min_comment_lines: u32,
    forest: &mut ElidableForest,
) {
    let total_lines = node_line_count(node);
    if is_comment_kind(language, node.kind()) {
        if total_lines >= min_comment_lines {
            let start_line = node_start_line(node) + 2;
            let end_line = node_end_line(node).saturating_sub(1);
            if start_line <= end_line {
                forest.push(
                    elidable_parent,
                    LineSpan {
                        start: start_line,
                        end: end_line,
                    },
                );
            }
        }
        return;
    }

    let mut current_parent = elidable_parent;
    if is_elidable_kind(language, node.kind()) && total_lines >= min_body_lines {
        let start_line = node_start_line(node) + 1;
        let end_line = node_end_line(node).saturating_sub(1);
        if start_line <= end_line {
            // Unlike the legacy outermost-only collector, we DO recurse into
            // the elided node so nested elisions are recorded as children.
            // The BFS unfold pass decides which level actually fires.
            current_parent = Some(forest.push(
                elidable_parent,
                LineSpan {
                    start: start_line,
                    end: end_line,
                },
            ));
        }
    }

    // Detect consecutive runs of groupable siblings (e.g. import statements).
    // When the run's total line span meets `min_body_lines`, elide the lines
    // strictly between the first and last sibling's content, leaving the
    // boundary statements visible.
    let child_count = node.child_count();
    let mut run_first: Option<Node<'_>> = None;
    let mut run_last: Option<Node<'_>> = None;
    let mut run_count: u32 = 0;
    for index in 0..child_count {
        let Some(child) = node.child(index) else {
            continue;
        };
        if is_groupable_kind(language, child.kind()) {
            if run_first.is_none() {
                run_first = Some(child);
            }
            run_last = Some(child);
            run_count += 1;
        } else {
            flush_groupable_run(
                run_first,
                run_last,
                run_count,
                min_body_lines,
                forest,
                current_parent,
            );
            run_first = None;
            run_last = None;
            run_count = 0;
        }
    }
    flush_groupable_run(
        run_first,
        run_last,
        run_count,
        min_body_lines,
        forest,
        current_parent,
    );

    for index in 0..child_count {
        if let Some(child) = node.child(index) {
            collect_elidable_tree(
                child,
                current_parent,
                language,
                min_body_lines,
                min_comment_lines,
                forest,
            );
        }
    }
}

fn flush_groupable_run(
    first: Option<Node<'_>>,
    last: Option<Node<'_>>,
    count: u32,
    min_body_lines: u32,
    forest: &mut ElidableForest,
    parent: Option<usize>,
) {
    if count < 2 {
        return;
    }
    let (Some(first), Some(last)) = (first, last) else {
        return;
    };
    let first_start = node_start_line(first);
    let last_start = node_start_line(last);
    let last_end = node_end_line(last);
    let span_lines = last_end.saturating_sub(first_start).saturating_add(1);
    if span_lines < min_body_lines {
        return;
    }
    // Use the line of the first node's last visible content as the lower bound
    // (some grammars include trailing newlines in the node range, which would
    // otherwise place `end_line` on the next sibling's first line).
    let first_content_end = node_content_end_line(first).min(last_start.saturating_sub(1));
    let start = first_content_end.saturating_add(1);
    let end = last_start.saturating_sub(1);
    if start <= end {
        forest.push(parent, LineSpan { start, end });
    }
}

pub(crate) fn node_start_line(node: Node<'_>) -> u32 {
    node.start_position()
        .row
        .saturating_add(1)
        .min(u32::MAX as usize) as u32
}

fn node_end_line(node: Node<'_>) -> u32 {
    node.end_position()
        .row
        .saturating_add(1)
        .min(u32::MAX as usize) as u32
}

/// Last source line containing a content byte from `node`.
///
/// Tree-sitter reports `end_position` as the position one past the last byte.
/// When that byte is a newline, the resulting position lands at column 0 of
/// the next row, which makes the naive `row + 1` answer one greater than the
/// row of the last visible content. This helper subtracts that off.
pub(crate) fn node_content_end_line(node: Node<'_>) -> u32 {
    let pos = node.end_position();
    let row = if pos.column == 0 && pos.row > 0 {
        pos.row - 1
    } else {
        pos.row
    };
    row.saturating_add(1).min(u32::MAX as usize) as u32
}

fn node_line_count(node: Node<'_>) -> u32 {
    node_end_line(node)
        .saturating_sub(node_start_line(node))
        .saturating_add(1)
}

fn is_comment_kind(language: SupportLang, kind: &str) -> bool {
    match language {
        SupportLang::TypeScript | SupportLang::Tsx | SupportLang::JavaScript => kind == "comment",
        SupportLang::Rust => kind == "block_comment",
        SupportLang::Python => kind == "comment",
        SupportLang::Go => kind == "comment",
        SupportLang::Fortran => kind == "comment",
        SupportLang::Java => kind == "block_comment",
        SupportLang::C | SupportLang::Cpp | SupportLang::ObjC => kind == "comment",
        SupportLang::CSharp => kind == "comment",
        SupportLang::Ruby => kind == "comment",
        SupportLang::Php => kind == "comment",
        SupportLang::Swift => kind == "comment",
        SupportLang::Kotlin => kind == "block_comment",
        SupportLang::Scala => kind == "block_comment",
        SupportLang::Lua => kind == "comment",
        SupportLang::EmacsLisp => kind == "comment",
        _ => false,
    }
}

fn is_elidable_kind(language: SupportLang, kind: &str) -> bool {
    match language {
        SupportLang::TypeScript | SupportLang::Tsx | SupportLang::JavaScript => matches!(
            kind,
            "statement_block"
                | "function_body"
                | "object"
                | "array"
                | "template_string"
                | "class_body"
                | "interface_body"
                | "enum_body"
                | "object_type"
                | "switch_body"
                | "jsx_element"
                | "jsx_self_closing_element"
        ),
        SupportLang::Rust => matches!(
            kind,
            "block"
                | "array_expression"
                | "tuple_expression"
                | "struct_expression"
                | "match_block"
                | "raw_string_literal"
                | "declaration_list"
                | "field_declaration_list"
                | "ordered_field_declaration_list"
                | "enum_variant_list"
                | "where_clause"
                | "use_list"
                | "macro_definition"
                | "token_tree"
        ),
        SupportLang::Python => matches!(
            kind,
            "block"
                | "dictionary"
                | "list"
                | "set"
                | "string"
                | "tuple"
                | "argument_list"
                | "parameters"
                | "parenthesized_expression"
                | "list_comprehension"
                | "set_comprehension"
                | "dictionary_comprehension"
                | "generator_expression"
                | "import_from_statement"
                | "subscript"
        ),
        SupportLang::Go => matches!(
            kind,
            "block"
                | "composite_literal"
                | "interpreted_string_literal"
                | "raw_string_literal"
                | "import_spec_list"
                | "const_declaration"
                | "var_declaration"
                | "field_declaration_list"
                | "interface_type"
                | "expression_switch_statement"
                | "type_switch_statement"
                | "select_statement"
        ),
        SupportLang::Java => matches!(
            kind,
            "block"
                | "array_initializer"
                | "class_body"
                | "interface_body"
                | "enum_body"
                | "annotation_type_body"
                | "constructor_body"
                | "switch_block"
                | "string_literal"
        ),
        SupportLang::C => matches!(
            kind,
            "compound_statement"
                | "initializer_list"
                | "string_literal"
                | "field_declaration_list"
                | "enumerator_list"
                | "concatenated_string"
        ),
        SupportLang::Cpp => matches!(
            kind,
            "compound_statement"
                | "initializer_list"
                | "string_literal"
                | "field_declaration_list"
                | "enumerator_list"
                | "concatenated_string"
                | "declaration_list"
                | "raw_string_literal"
                | "requires_clause"
        ),
        SupportLang::ObjC => matches!(
            kind,
            "compound_statement"
                | "initializer_list"
                | "string_literal"
                | "protocol_declaration"
                | "class_interface"
                | "class_implementation"
                | "instance_variables"
                | "array_literal"
                | "dictionary_literal"
        ),
        SupportLang::CSharp => matches!(
            kind,
            "block"
                | "initializer_expression"
                | "array_initializer_expression"
                | "declaration_list"
                | "enum_member_declaration_list"
                | "switch_expression"
                | "raw_string_literal"
                | "interpolated_string_expression"
        ),
        SupportLang::Ruby => matches!(
            kind,
            "body_statement"
                | "method"
                | "do_block"
                | "array"
                | "hash"
                | "block"
                | "case"
                | "heredoc_body"
        ),
        SupportLang::Php => matches!(
            kind,
            "compound_statement"
                | "array_creation_expression"
                | "declaration_list"
                | "enum_declaration_list"
                | "match_block"
                | "heredoc"
                | "nowdoc"
        ),
        SupportLang::Swift => matches!(
            kind,
            "function_body"
                | "array_literal"
                | "dictionary_literal"
                | "multi_line_string_literal"
                | "class_body"
                | "protocol_body"
                | "enum_class_body"
                | "computed_property"
                | "lambda_literal"
        ),
        SupportLang::Kotlin => matches!(
            kind,
            "function_body"
                | "collection_literal"
                | "multi_line_string_literal"
                | "class_body"
                | "enum_class_body"
                | "when_expression"
                | "import_list"
        ),
        SupportLang::Scala => matches!(
            kind,
            "block"
                | "collection_literal"
                | "template_body"
                | "enum_body"
                | "match_expression"
                | "for_expression"
                | "string"
        ),
        SupportLang::Lua => matches!(kind, "block" | "table_constructor" | "string"),
        SupportLang::Dart => matches!(
            kind,
            "block"
                | "function_expression_body"
                | "class_body"
                | "enum_body"
                | "extension_body"
                | "mixin_body"
                | "list_literal"
                | "set_or_map_literal"
                | "string_literal"
        ),
        SupportLang::Bash => matches!(
            kind,
            "compound_statement"
                | "if_statement"
                | "case_statement"
                | "do_group"
                | "subshell"
                | "array"
                | "heredoc_body"
        ),
        SupportLang::Powershell => matches!(
            kind,
            "script_block"
                | "statement_block"
                | "class_statement"
                | "param_block"
                | "hash_literal_expression"
                | "array_expression"
                | "expandable_here_string_literal"
                | "verbatim_here_string_characters"
        ),
        SupportLang::Haskell => matches!(
            kind,
            "imports"
                | "data_type"
                | "class"
                | "instance"
                | "function"
                | "do"
                | "case"
                | "let"
                | "local_binds"
                | "list"
                | "tuple"
        ),
        SupportLang::Ocaml => matches!(
            kind,
            "structure"
                | "signature"
                | "variant_declaration"
                | "record_declaration"
                | "match_expression"
                | "match_case"
                | "let_expression"
                | "value_definition"
                | "list_expression"
        ),
        SupportLang::Elixir => matches!(kind, "do_block" | "list" | "map" | "string" | "sigil"),
        SupportLang::Erlang => matches!(
            kind,
            "fun_decl"
                | "case_expr"
                | "if_expr"
                | "receive_expr"
                | "record_decl"
                | "list"
                | "map_expr"
                | "tuple"
        ),
        SupportLang::EmacsLisp => matches!(
            kind,
            "function_definition"
                | "macro_definition"
                | "special_form"
                | "list"
                | "vector"
                | "hash_table"
                | "bytecode"
                | "string_text_properties"
                | "string"
        ),
        SupportLang::Clojure => {
            matches!(
                kind,
                "list_lit" | "map_lit" | "vec_lit" | "set_lit" | "str_lit"
            )
        }
        SupportLang::Solidity => {
            matches!(
                kind,
                "contract_body" | "function_body" | "struct_body" | "enum_body"
            )
        }
        SupportLang::Sql => matches!(kind, "column_definitions" | "case"),
        SupportLang::Zig => matches!(kind, "Block" | "ContainerDecl" | "InitList"),
        SupportLang::Odin => matches!(
            kind,
            "block" | "struct_declaration" | "enum_declaration" | "union_declaration" | "struct"
        ),
        SupportLang::Verilog => matches!(
            kind,
            "module_declaration"
                | "seq_block"
                | "case_statement"
                | "function_declaration"
                | "task_declaration"
                | "list_of_port_declarations"
        ),
        SupportLang::Tlaplus => matches!(kind, "module" | "theorem" | "let_in"),
        SupportLang::Nix => matches!(
            kind,
            "attrset_expression"
                | "list_expression"
                | "let_expression"
                | "indented_string_expression"
        ),
        SupportLang::Proto => matches!(kind, "message_body" | "enum_body" | "oneof" | "service"),
        SupportLang::Julia => matches!(
            kind,
            "function_definition"
                | "struct_definition"
                | "module_definition"
                | "do_clause"
                | "vector_expression"
                | "string_literal"
        ),
        SupportLang::R => matches!(kind, "braced_expression" | "call" | "string"),
        SupportLang::Starlark => matches!(kind, "block" | "list" | "dictionary" | "string"),
        SupportLang::Astro => {
            matches!(
                kind,
                "frontmatter_js_block" | "script_element" | "style_element" | "element"
            )
        }
        SupportLang::Vue => {
            matches!(
                kind,
                "template_element" | "script_element" | "style_element" | "element"
            )
        }
        SupportLang::Svelte => matches!(kind, "script_element" | "style_element" | "element"),
        SupportLang::Html => matches!(kind, "element" | "script_element" | "style_element"),
        SupportLang::Css => matches!(kind, "block" | "keyframe_block_list"),
        SupportLang::Json => matches!(kind, "object" | "array"),
        SupportLang::Xml => kind == "element",
        SupportLang::Markdown => matches!(kind, "fenced_code_block" | "pipe_table" | "list"),
        SupportLang::Graphql => matches!(
            kind,
            "fields_definition"
                | "enum_values_definition"
                | "input_fields_definition"
                | "schema_definition"
        ),
        SupportLang::Hcl => matches!(kind, "body" | "object"),
        SupportLang::Dockerfile => kind == "shell_command",
        SupportLang::Cmake => matches!(kind, "argument_list" | "body"),
        SupportLang::Make => kind == "recipe",
        SupportLang::Just => kind == "recipe_body",
        SupportLang::Fortran => false,
        // Skip: data formats with no closing-token anchor (Yaml mappings,
        // Toml tables, Ini sections), the diff format whose informational
        // content IS the lines inside hunks, and the leaf-token-only Regex
        // grammar. Eliding any of these deletes the only content worth
        // reading.
        SupportLang::Yaml
        | SupportLang::Toml
        | SupportLang::Ini
        | SupportLang::Diff
        | SupportLang::Regex => false,
    }
}

fn is_groupable_kind(language: SupportLang, kind: &str) -> bool {
    match language {
        SupportLang::TypeScript | SupportLang::Tsx | SupportLang::JavaScript => {
            kind == "import_statement"
        }
        SupportLang::Rust => matches!(kind, "use_declaration" | "extern_crate_declaration"),
        SupportLang::Python => {
            matches!(
                kind,
                "import_statement" | "import_from_statement" | "future_import_statement"
            )
        }
        SupportLang::Go => kind == "import_declaration",
        SupportLang::Java => kind == "import_declaration",
        SupportLang::C | SupportLang::Cpp => kind == "preproc_include",
        SupportLang::ObjC => matches!(kind, "preproc_include" | "import_declaration"),
        SupportLang::CSharp => kind == "using_directive",
        SupportLang::Php => kind == "namespace_use_declaration",
        SupportLang::Swift => kind == "import_declaration",
        SupportLang::Scala => matches!(kind, "import_declaration" | "import"),
        SupportLang::Dart => kind == "import_or_export",
        SupportLang::Ocaml => kind == "open_module",
        SupportLang::Solidity => kind == "import_directive",
        SupportLang::Julia => matches!(kind, "import_statement" | "using_statement"),
        SupportLang::Proto => kind == "import",
        SupportLang::Fortran => kind == "use_statement",
        // Languages where imports either have no run pattern, are wrapped in a
        // single AST node already covered by `is_elidable_kind` (Kotlin's
        // `import_list`, Haskell's `imports`), or live inside a too-generic
        // container (Powershell `statement_list`).
        SupportLang::Kotlin
        | SupportLang::Haskell
        | SupportLang::Powershell
        | SupportLang::Ruby
        | SupportLang::Lua
        | SupportLang::Elixir
        | SupportLang::Erlang
        | SupportLang::EmacsLisp
        | SupportLang::Clojure
        | SupportLang::Sql
        | SupportLang::Zig
        | SupportLang::Odin
        | SupportLang::Verilog
        | SupportLang::Tlaplus
        | SupportLang::Nix
        | SupportLang::R
        | SupportLang::Starlark
        | SupportLang::Bash
        | SupportLang::Astro
        | SupportLang::Vue
        | SupportLang::Svelte
        | SupportLang::Html
        | SupportLang::Css
        | SupportLang::Json
        | SupportLang::Xml
        | SupportLang::Markdown
        | SupportLang::Graphql
        | SupportLang::Hcl
        | SupportLang::Dockerfile
        | SupportLang::Cmake
        | SupportLang::Make
        | SupportLang::Just
        | SupportLang::Yaml
        | SupportLang::Toml
        | SupportLang::Ini
        | SupportLang::Diff
        | SupportLang::Regex => false,
    }
}

fn normalize_spans(mut spans: Vec<LineSpan>, total_lines: u32) -> Vec<LineSpan> {
    if total_lines == 0 {
        return Vec::new();
    }
    spans.retain(|span| span.start <= span.end && span.start <= total_lines);
    for span in &mut spans {
        span.end = span.end.min(total_lines);
    }
    spans.sort_by_key(|span| (span.start, span.end));
    let mut merged: Vec<LineSpan> = Vec::new();
    for span in spans {
        if let Some(last) = merged.last_mut()
            && span.start <= last.end.saturating_add(1)
        {
            last.end = last.end.max(span.end);
            continue;
        }
        merged.push(span);
    }
    merged
}

fn build_segments(source: &str, total_lines: u32, spans: &[LineSpan]) -> Vec<SummarySegment> {
    if total_lines == 0 {
        return Vec::new();
    }
    let source_lines: Vec<&str> = source.lines().collect();
    let elided_lines = spans
        .iter()
        .flat_map(|span| span.start..=span.end)
        .collect::<BTreeSet<_>>();
    let mut segments = Vec::new();
    let mut current_kind: Option<&str> = None;
    let mut current_start = 1;
    let mut current_lines: Vec<&str> = Vec::new();

    for line_number in 1..=total_lines {
        let is_elided = elided_lines.contains(&line_number);
        let kind = if is_elided { "elided" } else { "kept" };
        if current_kind.is_some_and(|existing| existing != kind) {
            push_segment(
                &mut segments,
                current_kind.expect("kind set"),
                current_start,
                line_number - 1,
                &current_lines,
            );
            current_start = line_number;
            current_lines.clear();
        }
        current_kind = Some(kind);
        if !is_elided {
            let index = line_number.saturating_sub(1) as usize;
            current_lines.push(source_lines.get(index).copied().unwrap_or_default());
        }
    }

    if let Some(kind) = current_kind {
        push_segment(
            &mut segments,
            kind,
            current_start,
            total_lines,
            &current_lines,
        );
    }
    segments
}

fn push_segment(
    segments: &mut Vec<SummarySegment>,
    kind: &str,
    start_line: u32,
    end_line: u32,
    lines: &[&str],
) {
    segments.push(SummarySegment {
        kind: kind.to_string(),
        start_line,
        end_line,
        text: (kind == "kept").then(|| lines.join("\n")),
    });
}
