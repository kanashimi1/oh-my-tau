//! Parity with the TypeScript agent: the session tree computed from a real (redacted) branched
//! session file must match omp's `SessionManager.getTree()` exactly. The golden digest in
//! `branched-session.expectation.json` was produced by running the TS sources under bun (see
//! `reports/SessionTree.md`); the companion `buildSessionContext` check lives in omp-cli
//! (which depends on omp-compact; a dev-dependency here would cycle).

use serde_json::{json, Value};

/// Fixture entries: every JSONL record except the fixed-width title slot and the v3 header.
fn fixture_entries() -> Vec<Value> {
    let raw = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/branched-session.jsonl"
    ))
    .unwrap();
    raw.lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
        .filter(|e| {
            !matches!(
                e.get("type").and_then(Value::as_str),
                Some("title" | "session")
            )
        })
        .collect()
}

fn expectation() -> Value {
    serde_json::from_str(include_str!("fixtures/branched-session.expectation.json")).unwrap()
}

/// The TS digest is flat: one `{id, parentId, children (ids), label?}` record per node, in
/// pre-order. `build_tree` roots arrive in the same order the TS `getTree` returns them.
fn node_digest(n: &omp_session::tree::TreeNode, out: &mut Vec<Value>) {
    let mut node = json!({
        "id": n.entry.get("id").and_then(Value::as_str).unwrap(),
        "parentId": n.entry.get("parentId").and_then(Value::as_str).map_or(Value::Null, |p| Value::String(p.to_owned())),
        "children": n.children.iter().filter_map(|c| c.id()).collect::<Vec<_>>(),
    });
    if let Some(label) = &n.label {
        node["label"] = Value::String(label.clone());
    }
    out.push(node);
    for child in &n.children {
        node_digest(child, out);
    }
}

#[test]
fn tree_matches_ts_get_tree() {
    let entries = fixture_entries();
    let golden = expectation();
    assert_eq!(
        omp_session::tree::leaf_id(&entries),
        golden["leafId"].as_str(),
        "leaf id must be the last physical journal entry"
    );
    let tree = omp_session::tree::build_tree(&entries);
    let mut digest = Vec::new();
    for root in &tree {
        node_digest(root, &mut digest);
    }
    assert_eq!(
        json!(digest),
        golden["tree"],
        "tree differs from TS getTree()"
    );
}
