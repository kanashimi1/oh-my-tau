//! Unit tests for path normalization/dedupe (cases from TS behavior) and
//! golden parity for the six `/api/stats/gain` requests.

use std::collections::HashSet;

use super::{canonical_project_path, dedupe_projects, is_same_or_sub_path, normalize_project_path};

fn n(p: &str) -> Option<String> {
    normalize_project_path(p)
}

#[test]
fn canonical_strips_trailing_slashes_and_backslashes() {
    assert_eq!(canonical_project_path("/repo/sub/"), "/repo/sub");
    assert_eq!(canonical_project_path("C:\\repo\\sub"), "C:/repo/sub");
    assert_eq!(canonical_project_path("/"), "/");
    assert_eq!(canonical_project_path("///"), "/");
    assert_eq!(canonical_project_path(""), "/");
}

#[test]
fn temp_paths_are_dropped() {
    assert_eq!(n("/tmp/foo"), None);
    assert_eq!(n("/var/folders/x/y"), None);
    assert_eq!(n("/repo/T/inner"), None);
    assert_eq!(n("/repo/pi-bash-exec/x"), None);
    assert_eq!(n("/repo/omp-bash-exec/"), None);
    assert_eq!(n("/repo/pi-bash-detach/a/b"), None);
    // But these are only segments, not prefixes elsewhere:
    assert_eq!(n("/tmpfoo/repo"), Some("/tmpfoo/repo".into()));
}

#[test]
fn omp_internal_wt_dropped() {
    assert_eq!(n("/repo/.omp/wt/lane"), None);
    assert_eq!(n("/repo/.omp/wt/lane/src"), None);
}

#[test]
fn worktree_suffixes_collapse_to_root() {
    assert_eq!(n("/repo/.wt/lane"), Some("/repo".into()));
    assert_eq!(n("/repo/.wt/lane/src/deep"), Some("/repo".into()));
    assert_eq!(n("/repo/.worktrees/lane/src"), Some("/repo".into()));
    assert_eq!(n("/repo-wt/lane"), Some("/repo".into()));
    assert_eq!(n("/repo-worktrees/lane/src"), Some("/repo".into()));
    assert_eq!(n("/repo.wt/lane"), Some("/repo".into()));
    // Nested root keeps outer prefix:
    assert_eq!(n("/a/repo/.wt/lane"), Some("/a/repo".into()));
    // Greedy `.+` like TS regex: last `.wt/` wins.
    assert_eq!(n("/a.wt/b.wt/c"), Some("/a.wt/b".into()));
    // No worktree marker → unchanged.
    assert_eq!(n("/repo/src"), Some("/repo/src".into()));
}

#[test]
fn sub_path_comparison() {
    assert!(is_same_or_sub_path("/repo", "/repo"));
    assert!(is_same_or_sub_path("/repo/sub", "/repo"));
    assert!(!is_same_or_sub_path("/repository", "/repo"));
    assert!(!is_same_or_sub_path("/repo/sub", "/other"));
    assert!(is_same_or_sub_path("/repo/sub/", "/repo/"));
}

#[test]
fn dedupe_drops_deep_sub_paths_of_normalized_roots() {
    let mut raw: HashSet<String> = [
        "/repo/.wt/lane",
        "/repo",
        "/home/u/proj/x/y/z",
        "/home/u/proj/x",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    raw.insert("/tmp/x".into());
    // "/home/u/proj/x" has 4 segments, so it absorbs the deeper child;
    // "/repo" (2 segments) absorbs the worktree path.
    assert_eq!(dedupe_projects(&raw), vec!["/home/u/proj/x", "/repo"]);
}

#[test]
fn dedupe_keeps_shallow_similar_prefixes() {
    let raw: HashSet<String> = ["/repo", "/repository"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    assert_eq!(dedupe_projects(&raw), vec!["/repo", "/repository"]);
}

#[test]
fn dedupe_only_collapses_at_depth_4() {
    // Parent shallower than 4 segments does NOT absorb the child.
    let raw: HashSet<String> = ["/a/b/c", "/a/b/c/d"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    assert_eq!(dedupe_projects(&raw), vec!["/a/b/c", "/a/b/c/d"]);
    // At depth >= 4 the parent absorbs the child.
    let raw: HashSet<String> = ["/a/b/c/d", "/a/b/c/d/e"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    assert_eq!(dedupe_projects(&raw), vec!["/a/b/c/d"]);
}

#[test]
fn dedupe_output_is_sorted() {
    let raw: HashSet<String> = ["/z/repo", "/a/repo", "/m/repo"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    assert_eq!(dedupe_projects(&raw), vec!["/a/repo", "/m/repo", "/z/repo"]);
}

// --- Golden parity ---------------------------------------------------------

#[cfg(all(test, feature = "db_ingest"))]
mod golden {
    use crate::testutil::golden::{assert_golden, fixture_home, golden, golden_requests};

    #[test]
    fn gain_matches_golden() {
        let Some(home) = fixture_home() else { return };
        for request in golden_requests("/api/stats/gain") {
            let stats =
                super::super::get_gain_dashboard_stats(request.split("range=").nth(1), None)
                    .expect("gain stats");
            let actual = serde_json::to_value(&stats).expect("serialize");
            let (status, _) = golden(&request);
            assert_eq!(status, 200, "{request}");
            assert_golden(&request, &actual, &home);
        }
    }
}

/// Synthetic snapcompact file vs. the TS oracle (`runs/omp-rust/webui-port/gain-synth.ts`).
/// The TS run used HOME=/tmp/gain-synth-home, so session paths in the jsonl carry that
/// prefix; they are rewritten to this test's temp home before the file is copied in.
#[cfg(all(test, feature = "db_ingest"))]
mod synthetic {
    use std::path::Path;

    use serde_json::Value;

    use crate::testutil::golden::fixture_home;

    const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests-data/gain-synthetic");
    const TS_HOME: &str = "/tmp/gain-synth-home";

    fn read_expected(name: &str) -> Value {
        let text = std::fs::read_to_string(Path::new(DIR).join(name)).expect("read expected json");
        serde_json::from_str(&text).expect("parse expected json")
    }

    /// Same rules as `testutil::golden` (numbers within 1e-9 relative, key order ignored).
    fn assert_same(actual: &Value, expected: &Value, path: &str) {
        match (actual, expected) {
            (Value::Number(a), Value::Number(e)) => {
                let (a, e) = (a.as_f64().expect("f64"), e.as_f64().expect("f64"));
                let tol = 1e-9 * 1f64.max(a.abs()).max(e.abs());
                assert!((a - e).abs() <= tol, "{path}: rust {a} != ts {e}");
            }
            (Value::Object(a), Value::Object(e)) => {
                let keys_a: Vec<&String> = a.keys().collect();
                let mut keys_e: Vec<&String> = e.keys().collect();
                let mut keys_a_sorted = keys_a.clone();
                keys_a_sorted.sort();
                keys_e.sort();
                assert_eq!(keys_a_sorted, keys_e, "{path}: key sets differ");
                for (k, ev) in e {
                    assert_same(&a[k], ev, &format!("{path}.{k}"));
                }
            }
            (Value::Array(a), Value::Array(e)) => {
                assert_eq!(a.len(), e.len(), "{path}: array length differs");
                for (i, (av, ev)) in a.iter().zip(e).enumerate() {
                    assert_same(av, ev, &format!("{path}[{i}]"));
                }
            }
            _ => assert_eq!(actual, expected, "{path}"),
        }
    }

    fn run(project: Option<&str>, expected_name: &str) {
        let Some(home) = fixture_home() else { return };
        let jsonl = std::fs::read_to_string(Path::new(DIR).join("snapcompact-savings.jsonl"))
            .expect("read jsonl");
        let jsonl = jsonl.replace(TS_HOME, &home.path().to_string_lossy());
        std::fs::write(home.path().join(".omp/snapcompact-savings.jsonl"), jsonl)
            .expect("copy jsonl");
        let stats =
            crate::gain::get_gain_dashboard_stats(Some("all"), project).expect("gain stats");
        let actual = serde_json::to_value(&stats).expect("serialize");
        assert_same(&actual, &read_expected(expected_name), "$");
    }

    #[test]
    fn snapcompact_all_matches_ts() {
        run(None, "expected-all.json");
    }

    #[test]
    fn snapcompact_project_filter_matches_ts() {
        run(Some("-home-user-omrs"), "expected-project.json");
    }
}
