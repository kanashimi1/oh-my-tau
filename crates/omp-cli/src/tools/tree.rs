//! omp `pi-utils/path-tree`: prefix-folded directory grouping shared by the
//! `glob` and `grep` result renderers.

#[derive(Debug, PartialEq, Eq)]
pub enum Kind {
    Dir,
    File,
}

/// One node emitted while walking the tree.
#[derive(Debug, PartialEq, Eq)]
pub struct Event {
    pub kind: Kind,
    /// 0-based nesting depth.
    pub depth: usize,
    /// Folded chain for dirs (no trailing slash); basename for files.
    pub name: String,
    /// Index into the input list for files; `usize::MAX` for dirs.
    pub key: usize,
}

#[derive(Default)]
struct Node {
    files: Vec<(String, usize)>,
    subdirs: Vec<(String, usize)>, // (name, arena index)
}

fn is_url_like(p: &str) -> bool {
    let Some((scheme, rest)) = p.split_once("://") else {
        return false;
    };
    let _ = rest;
    let mut chars = scheme.chars();
    chars.next().is_some_and(|c| c.is_ascii_alphabetic())
        && chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '.' | '-'))
}

/// Walk `entries` (`(path, is_dir)`) as omp's `buildPathTree` + `walkPathTree`.
#[must_use]
pub fn events(entries: &[(String, bool)]) -> Vec<Event> {
    let mut arena = vec![Node::default()];
    for (key, (raw, is_dir)) in entries.iter().enumerate() {
        let normalized = raw.replace('\\', "/");
        if is_url_like(&normalized) {
            add_file(&mut arena[0], normalized, key);
            continue;
        }
        let trimmed = normalized.strip_suffix('/').unwrap_or(&normalized);
        if trimmed.is_empty() {
            continue;
        }
        let segments: Vec<&str> = trimmed.split('/').collect();
        let dir_count = if *is_dir {
            segments.len()
        } else {
            segments.len() - 1
        };
        let mut node = 0;
        for seg in &segments[..dir_count] {
            let existing = arena[node]
                .subdirs
                .iter()
                .find(|(n, _)| n == seg)
                .map(|(_, i)| *i);
            node = if let Some(i) = existing {
                i
            } else {
                arena.push(Node::default());
                let idx = arena.len() - 1;
                arena[node].subdirs.push(((*seg).to_string(), idx));
                idx
            };
        }
        if !*is_dir {
            add_file(
                &mut arena[node],
                segments[segments.len() - 1].to_string(),
                key,
            );
        }
    }
    let mut out = Vec::new();
    walk(&arena, 0, 0, &mut out);
    out
}

fn add_file(node: &mut Node, name: String, key: usize) {
    if node.files.iter().any(|(n, _)| *n == name) {
        return;
    }
    node.files.push((name, key));
}

fn walk(arena: &[Node], node: usize, depth: usize, out: &mut Vec<Event>) {
    for (name, key) in &arena[node].files {
        out.push(Event {
            kind: Kind::File,
            depth,
            name: name.clone(),
            key: *key,
        });
    }
    for (name, idx) in &arena[node].subdirs {
        let mut dir = *idx;
        let mut parts = vec![name.clone()];
        while arena[dir].files.is_empty() && arena[dir].subdirs.len() == 1 {
            let (only_name, only_idx) = &arena[dir].subdirs[0];
            parts.push(only_name.clone());
            dir = *only_idx;
        }
        out.push(Event {
            kind: Kind::Dir,
            depth,
            name: parts.join("/"),
            key: usize::MAX,
        });
        walk(arena, dir, depth + 1, out);
    }
}

/// omp `formatGroupedPaths`: dir headers `# a/b/` and bare file names.
#[must_use]
pub fn format_grouped_paths(paths: &[String]) -> String {
    let entries: Vec<(String, bool)> = paths
        .iter()
        .map(|p| (p.clone(), p.ends_with('/')))
        .collect();
    let mut lines = Vec::new();
    for e in events(&entries) {
        match e.kind {
            Kind::Dir => lines.push(format!("{} {}/", "#".repeat(e.depth + 1), e.name)),
            Kind::File => lines.push(e.name),
        }
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn folds_single_child_chains_and_lists_files_before_subdirs() {
        let paths = ["src/a/b/x.rs", "src/a/b/y.rs", "README.md", "src/lib.rs"]
            .map(String::from)
            .to_vec();
        assert_eq!(
            format_grouped_paths(&paths),
            "README.md\n# src/\nlib.rs\n## a/b/\nx.rs\ny.rs"
        );
    }

    #[test]
    fn directory_leaves_keep_trailing_slash_headers() {
        let paths = ["docs/".to_string(), "docs/a.md".to_string()];
        assert_eq!(format_grouped_paths(&paths), "# docs/\na.md");
    }
}
