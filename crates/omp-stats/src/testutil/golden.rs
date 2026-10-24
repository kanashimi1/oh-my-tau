//! Parity oracle shared by every read-model test: a temp `HOME` holding the
//! 25-session fixture (ingested exactly like a sync), plus the TS responses
//! recorded by `runs/omp-rust/webui-port/golden.ts` for 85 `/api/*` requests.
//!
//! The goldens embed the fixture's absolute path; [`assert_golden`] rewrites
//! the temp home prefix in the Rust value to that path before comparing.
//! Comparison: numbers within 1e-9 relative, object key order ignored, and
//! numeric `id` keys skipped — they are sqlite rowids whose values depend on
//! the ingest order, which TS derives from `Date.now()` and file birth times
//! (`orderForIngest`), so they are not reproducible (`/api/request/<id>` tests
//! must map the golden row's `sessionFile` + `entryId` to the Rust id). For
//! relative ranges (`range=1h|24h|7d|30d|90d`, "now"-dependent) only the shape
//! (key sets and JSON types) is compared.

use std::path::Path;

use serde_json::Value;

use super::TestHome;

/// The fixture `HOME` the goldens were recorded against.
pub(crate) const FIXTURE_HOME: &str =
    "/home/user/omrs-fixtures/stats/fixture/home";
/// `golden/_index.json` + `NNN.json` (`{request, status, body}`).
pub(crate) const GOLDEN_DIR: &str = "/home/user/omrs-fixtures/stats/golden";

const RELATIVE_RANGES: [&str; 5] = [
    "range=1h",
    "range=24h",
    "range=7d",
    "range=30d",
    "range=90d",
];

/// A fresh temp `HOME` whose sessions dir links to the fixture, with every
/// fixture session parsed and applied into its stats db. `None` (test should
/// return early) when the fixture is not on this machine.
pub(crate) fn fixture_home() -> Option<TestHome> {
    let sessions = Path::new(FIXTURE_HOME).join(".omp/agent/sessions");
    if !sessions.is_dir() {
        eprintln!("fixture sessions not found; skipping golden test");
        return None;
    }
    let home = TestHome::new();
    let agent = home.path().join(".omp/agent");
    std::fs::create_dir_all(&agent).expect("create agent dir");
    std::os::unix::fs::symlink(&sessions, agent.join("sessions")).expect("link fixture sessions");
    let batch: Vec<crate::db::ParsedSession> = crate::parser::list_all_session_files()
        .into_iter()
        .map(|file| {
            let result = crate::parser::parse_session_file(&file, 0, None, false)
                .expect("parse fixture session");
            crate::db::ParsedSession {
                session_file: file,
                result,
                rebuild: false,
                replay: false,
            }
        })
        .collect();
    crate::db::apply_session_parse_results(&batch).expect("apply fixture sessions");
    Some(home)
}

/// The recorded TS `(status, body)` for `request` (path + query exactly as in
/// `_index.json`, e.g. `/api/stats?range=all`).
///
/// # Panics
/// Unknown request or unreadable golden file.
pub(crate) fn golden(request: &str) -> (u16, Value) {
    let index: Value = read_json(&Path::new(GOLDEN_DIR).join("_index.json"));
    let entry = index
        .get(request)
        .unwrap_or_else(|| panic!("no golden for {request}"));
    let file = entry["file"].as_str().expect("golden file name");
    let recorded = read_json(&Path::new(GOLDEN_DIR).join(file));
    let status =
        u16::try_from(recorded["status"].as_u64().expect("status")).expect("status fits u16");
    (status, recorded["body"].clone())
}

/// Every recorded request whose path (before `?`) equals `path`.
pub(crate) fn golden_requests(path: &str) -> Vec<String> {
    let index: Value = read_json(&Path::new(GOLDEN_DIR).join("_index.json"));
    index
        .as_object()
        .map(|o| {
            o.keys()
                .filter(|k| k.split('?').next() == Some(path))
                .cloned()
                .collect()
        })
        .unwrap_or_default()
}

/// Assert `actual` (a serialized Rust response body) matches the golden body
/// of `request`, after mapping `home`'s path back to [`FIXTURE_HOME`].
///
/// # Panics
/// On mismatch, listing up to 40 differing JSON paths.
pub(crate) fn assert_golden(request: &str, actual: &Value, home: &TestHome) {
    let (_, expected) = golden(request);
    let actual = remap(actual, &home.path().to_string_lossy());
    let shape_only = RELATIVE_RANGES.iter().any(|r| request.contains(r));
    let mut diffs = Vec::new();
    diff(&actual, &expected, "$", shape_only, &mut diffs);
    assert!(
        diffs.is_empty(),
        "{request} differs from TS golden:\n{}",
        diffs.join("\n")
    );
}

fn read_json(path: &Path) -> Value {
    let text =
        std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
}

fn remap(value: &Value, from: &str) -> Value {
    match value {
        Value::String(s) => Value::String(s.replace(from, FIXTURE_HOME)),
        Value::Array(items) => Value::Array(items.iter().map(|v| remap(v, from)).collect()),
        Value::Object(map) => Value::Object(
            map.iter()
                .map(|(k, v)| (k.replace(from, FIXTURE_HOME), remap(v, from)))
                .collect(),
        ),
        other => other.clone(),
    }
}

fn same_num(a: f64, b: f64) -> bool {
    a == b || (a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-9 * 1f64.max(a.abs()).max(b.abs())
}

fn kind(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn diff(actual: &Value, expected: &Value, path: &str, shape_only: bool, out: &mut Vec<String>) {
    if out.len() >= 40 {
        return;
    }
    match (actual, expected) {
        (Value::Number(a), Value::Number(b)) => {
            let (a, b) = (
                a.as_f64().unwrap_or(f64::NAN),
                b.as_f64().unwrap_or(f64::NAN),
            );
            if !shape_only && !same_num(a, b) {
                out.push(format!("{path}: rust {a} != ts {b}"));
            }
        }
        (Value::Array(a), Value::Array(b)) => {
            if !shape_only && a.len() != b.len() {
                out.push(format!(
                    "{path}: rust len {} != ts len {}",
                    a.len(),
                    b.len()
                ));
                return;
            }
            for (i, (x, y)) in a.iter().zip(b).enumerate() {
                diff(x, y, &format!("{path}[{i}]"), shape_only, out);
            }
        }
        (Value::Object(a), Value::Object(b)) => {
            for key in b.keys().filter(|k| !a.contains_key(*k)) {
                out.push(format!("{path}.{key}: missing in rust"));
            }
            for key in a.keys().filter(|k| !b.contains_key(*k)) {
                out.push(format!("{path}.{key}: extra in rust"));
            }
            for (key, x) in a {
                if key == "id" && x.is_number() {
                    continue;
                }
                if let Some(y) = b.get(key) {
                    diff(x, y, &format!("{path}.{key}"), shape_only, out);
                }
            }
        }
        // Relative ranges may be empty on one side, so null vs value is not a
        // shape difference there.
        (a, b) if kind(a) != kind(b) => {
            if !(shape_only && (a.is_null() || b.is_null())) {
                out.push(format!("{path}: rust {} != ts {}", kind(a), kind(b)));
            }
        }
        (a, b) => {
            if !shape_only && a != b {
                out.push(format!("{path}: rust {a} != ts {b}"));
            }
        }
    }
}

#[cfg(all(test, feature = "db_query"))]
mod tests {
    use super::*;

    /// The oracle itself: ingest + a pure db read must reproduce the TS body.
    #[test]
    fn recent_requests_match_golden() {
        let Some(home) = fixture_home() else { return };
        let rows = crate::db_query::get_recent_requests(50).expect("recent requests");
        let actual = serde_json::to_value(rows).expect("serialize");
        assert_golden("/api/stats/recent?limit=50", &actual, &home);
    }
}
