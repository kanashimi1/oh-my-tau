//! Session tree semantics: entries form a tree via `parentId`, the live branch is a leaf
//! selection, and abandoned branches are marked with `branch_summary` entries.
//!
//! Ports of coding-agent `session/session-manager.ts` (`SessionEntryIndex.tree`,
//! `branch`/`branchWithSummary` entry shapes, leaf reconstruction) and
//! `session/session-entries.ts` (`SessionTreeNode`). Everything operates on raw entry
//! values so it works for both modeled and unmodeled entry kinds.

use std::collections::HashMap;

use serde_json::{Map, Value};

/// One node of the session tree (TS `SessionTreeNode`): the entry, its children and the
/// resolved label for the entry, if any.
#[derive(Debug, Clone, PartialEq)]
pub struct TreeNode {
    pub entry: Value,
    pub children: Vec<TreeNode>,
    /// Resolved label for this entry, if any (TS omits the key when unset).
    pub label: Option<String>,
}

impl TreeNode {
    /// The entry's id.
    #[must_use]
    pub fn id(&self) -> Option<&str> {
        self.entry.get("id").and_then(Value::as_str)
    }

    /// `{entry, children, label?}` — the shape `getTree` reports over RPC.
    #[must_use]
    pub fn to_value(&self) -> Value {
        let mut node = Map::new();
        node.insert("entry".into(), self.entry.clone());
        node.insert(
            "children".into(),
            Value::Array(self.children.iter().map(TreeNode::to_value).collect()),
        );
        if let Some(label) = &self.label {
            node.insert("label".into(), Value::String(label.clone()));
        }
        Value::Object(node)
    }
}

/// Entry id and `parentId` (`None` for null / missing / non-string).
fn links(entry: &Value) -> (Option<&str>, Option<&str>) {
    let id = entry.get("id").and_then(Value::as_str);
    let parent = entry.get("parentId").and_then(Value::as_str);
    (id, parent)
}

/// `SessionEntryIndex` label map, in file order: a `label` entry with a truthy `label` sets the
/// label for `targetId`; one with a falsy `label` clears it.
#[must_use]
pub fn labels_in_effect(entries: &[Value]) -> HashMap<String, String> {
    let mut labels = HashMap::new();
    for entry in entries {
        if entry.get("type").and_then(Value::as_str) != Some("label") {
            continue;
        }
        let Some(target) = entry.get("targetId").and_then(Value::as_str) else {
            continue;
        };
        match entry.get("label").and_then(Value::as_str) {
            Some(label) if !label.is_empty() => labels.insert(target.to_owned(), label.to_owned()),
            _ => labels.remove(target),
        };
    }
    labels
}

/// Port of `SessionEntryIndex.tree`: every entry becomes a node; an entry whose `parentId` is
/// null, its own id, or unknown (orphaned by a broken parent chain) is a root, otherwise it
/// hangs under its parent. Children are sorted by entry timestamp (stable, so equal timestamps
/// keep file order) at every level.
#[must_use]
pub fn build_tree(entries: &[Value]) -> Vec<TreeNode> {
    let labels = labels_in_effect(entries);
    let mut nodes: HashMap<&str, TreeNode> = HashMap::with_capacity(entries.len());
    for entry in entries {
        let (Some(id), _) = links(entry) else {
            continue;
        };
        nodes.insert(
            id,
            TreeNode {
                entry: entry.clone(),
                children: Vec::new(),
                label: labels.get(id).cloned(),
            },
        );
    }

    // Index the nodes that are somebody's child; everything else is a root.
    let mut children: HashMap<&str, Vec<&str>> = HashMap::new();
    let mut roots: Vec<&str> = Vec::new();
    for entry in entries {
        let (Some(id), parent) = links(entry) else {
            continue;
        };
        match parent {
            None => roots.push(id),
            Some(p) if p == id => roots.push(id),
            Some(p) if nodes.contains_key(p) => children.entry(p).or_default().push(id),
            Some(_) => roots.push(id),
        }
    }

    fn sort_level(
        nodes: &mut HashMap<&str, TreeNode>,
        children: &HashMap<&str, Vec<&str>>,
        id: &str,
    ) -> TreeNode {
        let mut node = nodes.remove(id).expect("node indexed before building");
        let mut kids: Vec<TreeNode> = children
            .get(id)
            .map(|ids| ids.iter().map(|c| sort_level(nodes, children, c)).collect())
            .unwrap_or_default();
        kids.sort_by(|a, b| {
            let ta = a
                .entry
                .get("timestamp")
                .and_then(Value::as_str)
                .unwrap_or("");
            let tb = b
                .entry
                .get("timestamp")
                .and_then(Value::as_str)
                .unwrap_or("");
            ta.cmp(tb)
        });
        node.children = kids;
        node
    }

    roots
        .drain(..)
        .map(|id| sort_level(&mut nodes, &children, id))
        .collect()
}

/// The leaf the loader reconstructs: the id of the last physical journal entry
/// (`SessionEntryIndex.insert` advances the leaf on every entry), `None` for an empty session.
#[must_use]
pub fn leaf_id(entries: &[Value]) -> Option<&str> {
    entries
        .iter()
        .rev()
        .find_map(|e| e.get("id").and_then(Value::as_str))
}

/// Root-to-leaf branch path for `leaf` (TS `SessionEntryIndex.pathTo` / `getBranch`):
/// walk `parentId` links from the leaf, stop at a missing parent or the first repeat
/// (pre-fix files can contain parent cycles), then reverse.
#[must_use]
pub fn branch_path<'a>(entries: &'a [Value], leaf: &str) -> Vec<&'a Value> {
    let by_id: HashMap<&str, &Value> = entries
        .iter()
        .filter_map(|e| e.get("id").and_then(Value::as_str).map(|id| (id, e)))
        .collect();
    let mut seen = std::collections::HashSet::new();
    let mut path = Vec::new();
    let mut current = by_id.get(leaf).copied();
    while let Some(entry) = current {
        let (Some(id), _) = links(entry) else { break };
        if !seen.insert(id) {
            break;
        }
        path.push(entry);
        current = links(entry).1.and_then(|p| by_id.get(p).copied());
    }
    path.reverse();
    path
}

/// The `branch_summary` entry `branchWithSummary` records when the leaf moves to `branch_from`
/// (TS key order: `type, id, parentId, timestamp, fromId, summary, details?, fromExtension?`;
/// `fromId` is `"root"` when branching from nothing).
#[must_use]
pub fn branch_summary_entry(
    id: &str,
    branch_from: Option<&str>,
    timestamp: &str,
    summary: &str,
    details: Option<Value>,
    from_extension: Option<bool>,
) -> Value {
    let mut entry = Map::new();
    entry.insert("type".into(), Value::String("branch_summary".into()));
    entry.insert("id".into(), Value::String(id.into()));
    entry.insert(
        "parentId".into(),
        branch_from.map_or(Value::Null, |p| Value::String(p.into())),
    );
    entry.insert("timestamp".into(), Value::String(timestamp.into()));
    entry.insert(
        "fromId".into(),
        branch_from.map_or_else(|| Value::String("root".into()), |p| Value::String(p.into())),
    );
    entry.insert("summary".into(), Value::String(summary.into()));
    if let Some(details) = details {
        entry.insert("details".into(), details);
    }
    if let Some(from_extension) = from_extension {
        entry.insert("fromExtension".into(), Value::Bool(from_extension));
    }
    Value::Object(entry)
}

/// Ids of the entries a navigation from `old_leaf` to `target` abandons, in chronological order
/// (TS `collectEntriesForBranchSummary`): everything on the old branch from the leaf back to
/// (excluding) the deepest common ancestor of the two branch paths.
#[must_use]
pub fn entries_for_branch_summary<'a>(
    entries: &'a [Value],
    old_leaf: &str,
    target: &str,
) -> (Vec<&'a Value>, Option<String>) {
    let old_path: std::collections::HashSet<&str> = branch_path(entries, old_leaf)
        .iter()
        .filter_map(|e| e.get("id").and_then(Value::as_str))
        .collect();
    let target_path = branch_path(entries, target);
    let common = target_path.iter().rev().find_map(|e| {
        let id = e.get("id").and_then(Value::as_str)?;
        old_path.contains(id).then(|| id.to_owned())
    });

    let mut abandoned = Vec::new();
    let mut current = Some(old_leaf);
    while let Some(id) = current {
        if Some(id) == common.as_deref() {
            break;
        }
        let Some(entry) = entries
            .iter()
            .find(|e| e.get("id").and_then(Value::as_str) == Some(id))
        else {
            break;
        };
        abandoned.push(entry);
        current = entry.get("parentId").and_then(Value::as_str);
    }
    abandoned.reverse();
    (abandoned, common)
}

/// True for a user `message` entry — the only valid rpc `branch` target
/// (TS `AgentSession.branch` validates `type === "message" && message.role === "user"`).
#[must_use]
pub fn is_user_message_entry(entry: &Value) -> bool {
    entry.get("type").and_then(Value::as_str) == Some("message")
        && entry.pointer("/message/role").and_then(Value::as_str) == Some("user")
}

/// The text of a user message entry (rpc `branch` returns it so the host can re-prompt):
/// plain string content, or the `text` parts joined by newlines (TS `extractUserMessageText`).
#[must_use]
pub fn user_message_text(entry: &Value) -> Option<String> {
    if !is_user_message_entry(entry) {
        return None;
    }
    match entry.pointer("/message/content")? {
        Value::String(s) => Some(s.clone()),
        Value::Array(parts) => Some(
            parts
                .iter()
                .filter_map(|p| {
                    (p.get("type")? == "text")
                        .then(|| p.get("text")?.as_str().map(str::to_owned))?
                })
                .collect::<Vec<_>>()
                .join("\n"),
        ),
        _ => None,
    }
}

#[cfg(test)]
use serde_json::json;

/// Convenience for tests and callers building fixtures: a minimal entry with the given links.
#[cfg(test)]
pub(crate) fn entry(id: &str, parent: Option<&str>, timestamp: &str, kind: &str) -> Value {
    json!({
        "type": kind,
        "id": id,
        "parentId": parent,
        "timestamp": timestamp,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Linear chain a←b←c: one root, sorted children.
    #[test]
    fn linear_chain_single_root() {
        let entries = vec![
            entry("a", None, "2026-10-01T00:00:00.000Z", "message"),
            entry("b", Some("a"), "2026-10-01T00:01:00.000Z", "message"),
            entry("c", Some("b"), "2026-10-01T00:02:00.000Z", "message"),
        ];
        let tree = build_tree(&entries);
        assert_eq!(tree.len(), 1);
        assert_eq!(tree[0].id(), Some("a"));
        assert_eq!(tree[0].children[0].id(), Some("b"));
        assert_eq!(tree[0].children[0].children[0].id(), Some("c"));
        assert_eq!(leaf_id(&entries), Some("c"));
        assert_eq!(branch_path(&entries, "c").len(), 3);
    }

    /// Rewind to `a`, then append `d`: `d` parents to `a`, making a sibling branch of b.
    #[test]
    fn sibling_branches_share_parent() {
        let entries = vec![
            entry("a", None, "2026-10-01T00:00:00.000Z", "message"),
            entry("b", Some("a"), "2026-10-01T00:01:00.000Z", "message"),
            entry("c", Some("b"), "2026-10-01T00:02:00.000Z", "message"),
            entry("d", Some("a"), "2026-10-01T00:03:00.000Z", "message"),
        ];
        let tree = build_tree(&entries);
        assert_eq!(tree.len(), 1);
        assert_eq!(tree[0].children.len(), 2);
        assert_eq!(tree[0].children[0].id(), Some("b")); // timestamp order
        assert_eq!(tree[0].children[1].id(), Some("d"));
        // Abandoned-path collection from leaf c navigating to d: common ancestor a, so [b, c].
        let (abandoned, common) = entries_for_branch_summary(&entries, "c", "d");
        assert_eq!(common.as_deref(), Some("a"));
        assert_eq!(
            abandoned
                .iter()
                .filter_map(|e| e.get("id").and_then(Value::as_str))
                .collect::<Vec<_>>(),
            vec!["b", "c"]
        );
    }

    /// Orphan (broken parent chain) and self-parent become roots, like TS.
    #[test]
    fn orphans_and_self_parents_are_roots() {
        let entries = vec![
            entry("a", None, "2026-10-01T00:00:00.000Z", "message"),
            entry("x", Some("missing"), "2026-10-01T00:01:00.000Z", "message"),
            entry("y", Some("y"), "2026-10-01T00:02:00.000Z", "message"),
        ];
        let tree = build_tree(&entries);
        assert_eq!(tree.len(), 3);
    }

    /// Label entries resolve onto their target; a later empty label clears it.
    #[test]
    fn labels_resolve_and_clear() {
        let mut labeled = entry("a", None, "2026-10-01T00:00:00.000Z", "message");
        let entries = vec![
            labeled.clone(),
            json!({"type": "label", "id": "l1", "parentId": "a", "timestamp": "2026-10-01T00:01:00.000Z", "targetId": "a", "label": "checkpoint"}),
        ];
        let tree = build_tree(&entries);
        assert_eq!(tree[0].label.as_deref(), Some("checkpoint"));

        let cleared = vec![
            labeled.clone(),
            json!({"type": "label", "id": "l1", "parentId": "a", "timestamp": "2026-10-01T00:01:00.000Z", "targetId": "a", "label": "checkpoint"}),
            json!({"type": "label", "id": "l2", "parentId": "l1", "timestamp": "2026-10-01T00:02:00.000Z", "targetId": "a"}),
        ];
        assert_eq!(build_tree(&cleared)[0].label, None);
        labeled["timestamp"] = json!("x");
    }

    /// branch_summary entry shape matches TS `branchWithSummary` (key order and fromId="root").
    #[test]
    fn branch_summary_entry_shape() {
        let v = branch_summary_entry(
            "s1",
            Some("a"),
            "2026-10-01T00:00:00.000Z",
            "done",
            None,
            None,
        );
        let keys: Vec<&str> = v.as_object().unwrap().keys().map(String::as_str).collect();
        assert_eq!(
            keys,
            vec!["type", "id", "parentId", "timestamp", "fromId", "summary"]
        );
        let root = branch_summary_entry(
            "s2",
            None,
            "t",
            "summary",
            Some(json!({"readFiles": []})),
            Some(true),
        );
        assert_eq!(root["fromId"], json!("root"));
        assert_eq!(root["parentId"], Value::Null);
        assert_eq!(root["details"], json!({"readFiles": []}));
        assert_eq!(root["fromExtension"], json!(true));
    }

    /// Parent cycles are bounded, mirroring `buildSessionContext`'s seen-set.
    #[test]
    fn parent_cycles_stop() {
        let entries = vec![
            json!({"type": "message", "id": "a", "parentId": "b", "timestamp": "2026-10-01T00:00:00.000Z"}),
            json!({"type": "message", "id": "b", "parentId": "a", "timestamp": "2026-10-01T00:01:00.000Z"}),
        ];
        assert_eq!(branch_path(&entries, "a").len(), 2);
    }

    #[test]
    fn user_message_text_extraction() {
        let plain = json!({"type": "message", "id": "a", "parentId": null, "timestamp": "t",
            "message": {"role": "user", "content": "hello", "timestamp": 1}});
        assert_eq!(user_message_text(&plain).as_deref(), Some("hello"));
        let parts = json!({"type": "message", "id": "a", "parentId": null, "timestamp": "t",
            "message": {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "image", "data": "x", "mimeType": "image/png"}, {"type": "text", "text": "b"}], "timestamp": 1}});
        assert_eq!(user_message_text(&parts).as_deref(), Some("a\nb"));
        let assistant = json!({"type": "message", "id": "a", "parentId": null, "timestamp": "t",
            "message": {"role": "assistant", "content": [], "timestamp": 1}});
        assert_eq!(user_message_text(&assistant), None);
    }
}
