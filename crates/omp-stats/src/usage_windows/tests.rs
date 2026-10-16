//! Behaviour checks against the TS `usage-windows.ts`. Expected values were
//! recorded by `runs/omp-rust/webui-port/usage-windows-oracle.ts` (bun running
//! the real TS module on the same SQL fixture as `testdata/usage_history.sql`).

use std::cmp::Ordering;
use std::collections::HashMap;

use serde_json::Value;

use super::*;

const FIXTURE_SQL: &str = include_str!("testdata/usage_history.sql");
const EXPECTED_WINDOWS: &str = include_str!("testdata/expected_windows.json");
const LOCALE_PAIRS: &str = include_str!("testdata/locale_pairs.json");

/// Build a temp agent.db from the shared SQL fixture.
fn fixture_db(dir: &tempfile::TempDir) -> std::path::PathBuf {
    let path = dir.path().join("agent.db");
    let conn = rusqlite::Connection::open(&path).expect("create db");
    conn.execute_batch(FIXTURE_SQL).expect("apply fixture");
    path
}

/// Compare JSON with numbers compared as f64 (JS has one number type).
fn assert_json_eq(actual: &Value, expected: &Value, path: &str) {
    match (actual, expected) {
        (Value::Number(a), Value::Number(e)) => {
            let (a, e) = (a.as_f64().expect("f64"), e.as_f64().expect("f64"));
            assert!(a == e, "{path}: {a} != {e}");
        }
        (Value::Array(a), Value::Array(e)) => {
            assert_eq!(a.len(), e.len(), "{path}: length");
            for (i, (a, e)) in a.iter().zip(e).enumerate() {
                assert_json_eq(a, e, &format!("{path}[{i}]"));
            }
        }
        (Value::Object(a), Value::Object(e)) => {
            assert_eq!(a.len(), e.len(), "{path}: keys");
            for (k, ev) in e {
                assert_json_eq(a.get(k).unwrap_or(&Value::Null), ev, &format!("{path}.{k}"));
            }
        }
        _ => assert_eq!(actual, expected, "{path}"),
    }
}

#[test]
fn window_stats_match_ts_oracle_on_same_db() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = fixture_db(&dir);
    let expected: Value = serde_json::from_str(EXPECTED_WINDOWS).expect("expected json");
    let since = expected["sinceMs"].as_i64().expect("sinceMs");
    let tokens: HashMap<String, f64> = expected["tokensByProvider"]
        .as_object()
        .expect("tokens")
        .iter()
        .map(|(k, v)| (k.clone(), v.as_f64().expect("tokens")))
        .collect();

    let rows = read_usage_snapshots(since, &db);
    // The row before the cutoff is excluded: 1026 fixture rows, 1025 since.
    assert_eq!(rows.len(), 1025);
    let stats = compute_usage_window_stats(&rows, &tokens);
    let actual = serde_json::json!({
        "usageSeries": stats.usage_series,
        "windowInsights": stats.window_insights,
    });
    assert_json_eq(&actual, &expected["stats"], "stats");
}

#[test]
fn missing_db_and_missing_table_yield_no_rows() {
    let dir = tempfile::tempdir().expect("tempdir");
    assert!(read_usage_snapshots(0, &dir.path().join("absent.db")).is_empty());
    let bare = dir.path().join("bare.db");
    rusqlite::Connection::open(&bare)
        .expect("create")
        .execute_batch("CREATE TABLE other (x INTEGER);")
        .expect("ddl");
    assert!(read_usage_snapshots(0, &bare).is_empty());
}

#[test]
fn empty_input_yields_empty_stats() {
    let stats = compute_usage_window_stats(&[], &HashMap::new());
    assert!(stats.usage_series.is_empty());
    assert!(stats.window_insights.is_empty());
}

#[test]
fn locale_compare_matches_js_local_compare() {
    let pairs: Vec<(String, String, i64)> = serde_json::from_str::<Vec<Value>>(LOCALE_PAIRS)
        .expect("pairs")
        .iter()
        .map(|p| {
            (
                p[0].as_str().expect("a").to_owned(),
                p[1].as_str().expect("b").to_owned(),
                p[2].as_i64().expect("sign"),
            )
        })
        .collect();
    assert!(pairs.len() > 1000);
    for (a, b, sign) in pairs {
        let got = match locale_compare(&a, &b) {
            Ordering::Less => -1,
            Ordering::Equal => 0,
            Ordering::Greater => 1,
        };
        assert_eq!(got, sign, "localeCompare({a:?}, {b:?})");
    }
}

#[test]
fn window_label_suffix_rules() {
    let row = |label: &str, window: Option<&str>| UsageSnapshotRow {
        recorded_at: 0,
        provider: "p".into(),
        account_key: "k".into(),
        email: None,
        account_id: None,
        limit_id: "l".into(),
        label: label.into(),
        window_label: window.map(str::to_owned),
        used_fraction: None,
        status: None,
    };
    assert_eq!(
        window_display_label(&row("Claude 7 Day", Some("7 Day"))),
        "Claude 7 Day"
    );
    assert_eq!(
        window_display_label(&row("Usage (Google)", Some("Daily"))),
        "Usage (Google) · Daily"
    );
    assert_eq!(window_display_label(&row("Usage", Some(""))), "Usage");
    assert_eq!(window_display_label(&row("Usage", None)), "Usage");
}
