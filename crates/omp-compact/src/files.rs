//! File-operation tracking for compaction summaries (pi-agent-core `compaction/utils.ts`,
//! pi-utils `path-tree.ts`): the `<files>` block appended to every summary.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;

use regex::Regex;
use serde_json::Value;

use crate::json::{message_role, str_field};

/// Maximum files listed in the `<files>` block.
const FILE_OPERATION_SUMMARY_LIMIT: usize = 20;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FileOperations {
    pub read: BTreeSet<String>,
    pub written: BTreeSet<String>,
    pub edited: BTreeSet<String>,
}

fn range_chunk() -> &'static str {
    r"L?\d+(?:(?:[-+]|\.\.)L?\d+|-|\.\.)?"
}

fn selector_regexes() -> &'static (Regex, Regex, Regex) {
    static CELL: OnceLock<(Regex, Regex, Regex)> = OnceLock::new();
    CELL.get_or_init(|| {
        let list = format!("{c}(?:,{c})*", c = range_chunk());
        (
            Regex::new(&format!("(?i)^(?:{list}|raw|conflicts)$")).expect("selector regex"),
            Regex::new(&format!("(?i)^{list}$")).expect("range regex"),
            Regex::new("(?i)^raw$").expect("raw regex"),
        )
    })
}

/// `splitReadSelector`: base path and trailing read-tool selector (`:50-200`, `:raw`, ...).
#[must_use]
pub fn split_read_selector(path: &str) -> (String, Option<String>) {
    let (selector, range_only, raw_only) = selector_regexes();
    let Some(colon) = path.rfind(':').filter(|c| *c > 0) else {
        return (path.to_owned(), None);
    };
    let candidate = &path[colon + 1..];
    if !selector.is_match(candidate) {
        return (path.to_owned(), None);
    }
    let mut base = &path[..colon];
    let mut sel = candidate.to_owned();
    if let Some(inner) = base.rfind(':').filter(|i| *i > 0) {
        let inner_candidate = &base[inner + 1..];
        let inner_raw = raw_only.is_match(inner_candidate);
        let outer_raw = raw_only.is_match(candidate);
        let inner_range = range_only.is_match(inner_candidate);
        let outer_range = range_only.is_match(candidate);
        if (inner_raw && outer_range) || (inner_range && outer_raw) {
            sel = format!("{inner_candidate}:{candidate}");
            base = &base[..inner];
        }
    }
    (base.to_owned(), Some(sel))
}

/// `stripReadSelector`.
#[must_use]
pub fn strip_read_selector(path: &str) -> String {
    split_read_selector(path).0
}

/// `isUrlSchemePath`: `scheme://` URLs are not re-groundable files.
#[must_use]
pub fn is_url_scheme_path(path: &str) -> bool {
    static CELL: OnceLock<Regex> = OnceLock::new();
    CELL.get_or_init(|| Regex::new("(?i)[a-z][a-z0-9+.-]*://").expect("url regex"))
        .is_match(path)
}

/// `extractFileOpsFromMessage`: record `read` / `write` / `edit` tool-call paths of an assistant
/// message.
pub fn extract_file_ops_from_message(message: &Value, ops: &mut FileOperations) {
    if message_role(message) != "assistant" {
        return;
    }
    for block in message
        .get("content")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        if str_field(block, "type") != Some("toolCall") {
            continue;
        }
        let Some(name) = block.get("name") else {
            continue;
        };
        let Some(path) = block
            .get("arguments")
            .and_then(|a| str_field(a, "path"))
            .filter(|p| !p.is_empty())
        else {
            continue;
        };
        if is_url_scheme_path(path) {
            continue;
        }
        match name.as_str() {
            Some("read") => {
                ops.read.insert(strip_read_selector(path));
            }
            Some("write") => {
                ops.written.insert(path.to_owned());
            }
            Some("edit") => {
                ops.edited.insert(path.to_owned());
            }
            _ => {}
        }
    }
}

/// `computeFileLists`: `(readFiles, modifiedFiles)`, each sorted; files only read vs modified.
#[must_use]
pub fn compute_file_lists(ops: &FileOperations) -> (Vec<String>, Vec<String>) {
    let modified: BTreeSet<&String> = ops
        .edited
        .iter()
        .chain(&ops.written)
        .filter(|f| !is_url_scheme_path(f))
        .collect();
    let read_only = ops
        .read
        .iter()
        .filter(|f| !is_url_scheme_path(f) && !modified.contains(f))
        .cloned()
        .collect();
    (read_only, modified.into_iter().cloned().collect())
}

/// One directory of the path tree: file leaves `(name, original path)` and child directories,
/// both in first-seen order.
#[derive(Default)]
struct Node {
    files: Vec<(String, String)>,
    subdirs: Vec<(String, Node)>,
}

impl Node {
    fn subdir(&mut self, name: &str) -> &mut Node {
        let idx = if let Some(i) = self.subdirs.iter().position(|(n, _)| n == name) {
            i
        } else {
            self.subdirs.push((name.to_owned(), Node::default()));
            self.subdirs.len() - 1
        };
        &mut self.subdirs[idx].1
    }

    fn add_file(&mut self, name: &str, key: &str) {
        if !self.files.iter().any(|(n, _)| n == name) {
            self.files.push((name.to_owned(), key.to_owned()));
        }
    }
}

fn walk(node: &Node, depth: usize, annotate: &dyn Fn(&str) -> String, out: &mut Vec<String>) {
    for (name, key) in &node.files {
        out.push(format!("{name}{}", annotate(key)));
    }
    for (name, child) in &node.subdirs {
        let mut parts = vec![name.as_str()];
        let mut cur = child;
        while cur.files.is_empty() && cur.subdirs.len() == 1 {
            parts.push(cur.subdirs[0].0.as_str());
            cur = &cur.subdirs[0].1;
        }
        out.push(format!("{} {}/", "#".repeat(depth + 1), parts.join("/")));
        walk(cur, depth + 1, annotate, out);
    }
}

/// `formatGroupedPaths`: a flat path list as a prefix-folded directory tree (`# dir/` headers,
/// bare basenames); `annotate` receives each file's original path.
#[must_use]
pub fn format_grouped_paths(paths: &[String], annotate: &dyn Fn(&str) -> String) -> String {
    let mut root = Node::default();
    for raw in paths {
        let normalized = raw.replace('\\', "/");
        let is_dir = normalized.ends_with('/');
        if is_url_scheme_path(&normalized) {
            root.add_file(&normalized, raw);
            continue;
        }
        let trimmed = normalized.strip_suffix('/').unwrap_or(&normalized);
        if trimmed.is_empty() {
            continue;
        }
        let segments: Vec<&str> = trimmed.split('/').collect();
        let dir_count = if is_dir {
            segments.len()
        } else {
            segments.len() - 1
        };
        let mut node = &mut root;
        for seg in &segments[..dir_count] {
            node = node.subdir(seg);
        }
        if !is_dir {
            node.add_file(segments[segments.len() - 1], raw);
        }
    }
    let mut lines = Vec::new();
    walk(&root, 0, annotate, &mut lines);
    lines.join("\n")
}

/// `formatFileOperations`: one `<files>` block with a `(Read)` / `(Write)` / `(RW)` marker per
/// file; `""` when there are none.
#[must_use]
pub fn format_file_operations(
    read_files: &[String],
    modified_files: &[String],
    read_set: Option<&BTreeSet<String>>,
) -> String {
    if read_files.is_empty() && modified_files.is_empty() {
        return String::new();
    }
    let mut mode: BTreeMap<&str, &str> = BTreeMap::new();
    for f in read_files {
        mode.insert(f, "Read");
    }
    for f in modified_files {
        mode.insert(
            f,
            if read_set.is_some_and(|s| s.contains(f)) {
                "RW"
            } else {
                "Write"
            },
        );
    }
    let all: Vec<String> = mode.keys().map(|k| (*k).to_owned()).collect();
    let shown = &all[..all.len().min(FILE_OPERATION_SUMMARY_LIMIT)];
    let mut files = format_grouped_paths(shown, &|p| {
        format!(" ({})", mode.get(p).copied().unwrap_or(""))
    });
    if all.len() > FILE_OPERATION_SUMMARY_LIMIT {
        files.push_str(&format!(
            "\n[…{} files elided…]",
            all.len() - FILE_OPERATION_SUMMARY_LIMIT
        ));
    }
    crate::prompts::format(&format!("<files>\n{files}\n</files>"))
}

fn strip_file_operation_tags(summary: &str) -> String {
    static CELL: OnceLock<[Regex; 3]> = OnceLock::new();
    let res = CELL.get_or_init(|| {
        [
            Regex::new(r"<files>[\s\S]*?</files>\s*").expect("files re"),
            Regex::new(r"<read-files>[\s\S]*?</read-files>\s*").expect("read re"),
            Regex::new(r"<modified-files>[\s\S]*?</modified-files>\s*").expect("mod re"),
        ]
    });
    let mut text = summary.to_owned();
    for re in res {
        text = re.replace_all(&text, "").into_owned();
    }
    text.trim_end().to_owned()
}

/// `upsertFileOperations`: replace any `<files>` block in `summary` with a fresh one.
#[must_use]
pub fn upsert_file_operations(
    summary: &str,
    read_files: &[String],
    modified_files: &[String],
    read_set: Option<&BTreeSet<String>>,
) -> String {
    let base = strip_file_operation_tags(summary);
    let ops = format_file_operations(read_files, modified_files, read_set);
    if ops.is_empty() {
        base
    } else if base.is_empty() {
        ops
    } else {
        format!("{base}\n\n{ops}")
    }
}
