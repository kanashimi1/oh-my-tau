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

// ----------------------------------------------------------- broker mode

use crate::testutil::TestHome;

use super::broker::{
    set_usage_broker, sum_fleet_tokens, ClientProviderUsage, ClientUsageClientSummary, UsageBroker,
};

fn client(install: &str, providers: &[(&str, f64, f64, f64, f64)]) -> ClientUsageClientSummary {
    ClientUsageClientSummary {
        install_id: install.to_string(),
        providers: providers
            .iter()
            .map(
                |(provider, input, output, cache_read, cache_write)| ClientProviderUsage {
                    provider: provider.to_string(),
                    input_tokens: *input,
                    output_tokens: *output,
                    cache_read_tokens: *cache_read,
                    cache_write_tokens: *cache_write,
                },
            )
            .collect(),
    }
}

#[test]
fn sum_fleet_tokens_folds_all_token_kinds_per_provider() {
    let clients = vec![
        client(
            "a",
            &[
                ("google-antigravity", 10.0, 5.0, 2.0, 1.0),
                ("kimi", 3.0, 1.0, 0.0, 0.0),
            ],
        ),
        client("b", &[("google-antigravity", 1.0, 1.0, 1.0, 1.0)]),
        client("c", &[]),
    ];
    let tokens = sum_fleet_tokens(&clients).expect("fleet tokens");
    assert_eq!(tokens["google-antigravity"], 22.0);
    assert_eq!(tokens["kimi"], 4.0);
}

#[test]
fn sum_fleet_tokens_empty_is_none_not_zero() {
    assert_eq!(sum_fleet_tokens(&[]), None);
    // A client that reported nothing contributes no entry.
    assert_eq!(sum_fleet_tokens(&[client("a", &[])]), None);
}

/// Scripted broker: `history`/`summary` are Results consumed per call.
struct FakeBroker {
    history: std::sync::Mutex<Vec<anyhow::Result<Vec<UsageSnapshotRow>>>>,
    summary: std::sync::Mutex<Vec<anyhow::Result<Vec<ClientUsageClientSummary>>>>,
}

impl FakeBroker {
    fn ok(rows: Vec<UsageSnapshotRow>, clients: Vec<ClientUsageClientSummary>) -> Self {
        Self {
            history: std::sync::Mutex::new(vec![Ok(rows)]),
            summary: std::sync::Mutex::new(vec![Ok(clients)]),
        }
    }
}

impl UsageBroker for FakeBroker {
    fn usage_history(&self, _since_ms: i64) -> anyhow::Result<Vec<UsageSnapshotRow>> {
        self.history
            .lock()
            .unwrap()
            .pop()
            .unwrap_or_else(|| Err(anyhow::anyhow!("no scripted history")))
    }

    fn client_usage_summary(
        &self,
        _since_ms: i64,
    ) -> anyhow::Result<Vec<ClientUsageClientSummary>> {
        self.summary
            .lock()
            .unwrap()
            .pop()
            .unwrap_or_else(|| Err(anyhow::anyhow!("no scripted summary")))
    }
}

fn broker_row(provider: &str, fraction: f64) -> UsageSnapshotRow {
    UsageSnapshotRow {
        recorded_at: 1_728_000_000_000,
        provider: provider.to_string(),
        account_key: "broker-key".to_string(),
        email: None,
        account_id: None,
        limit_id: "broker-limit".to_string(),
        label: "Broker".to_string(),
        window_label: None,
        used_fraction: Some(fraction),
        status: None,
    }
}

/// Temp HOME with one local `usage_history` row under `.omp/agent/agent.db`.
struct LocalHome {
    _home: TestHome,
}

impl LocalHome {
    fn with_local_row(provider: &str) -> Self {
        let home = TestHome::new();
        let agent = home.path().join(".omp").join("agent");
        std::fs::create_dir_all(&agent).unwrap();
        let conn = rusqlite::Connection::open(agent.join("agent.db")).unwrap();
        conn.execute_batch(
            "CREATE TABLE usage_history (recorded_at INTEGER, provider TEXT, account_key TEXT, email TEXT, account_id TEXT, limit_id TEXT, label TEXT, window_label TEXT, used_fraction REAL, status TEXT);",
        )
        .unwrap();
        conn.execute(
            "INSERT INTO usage_history VALUES (1728000000000, ?1, 'local-key', NULL, NULL, 'local-limit', 'Local', NULL, 0.25, NULL)",
            [provider],
        )
        .unwrap();
        Self { _home: home }
    }
}

#[test]
fn fetch_usage_data_uses_broker_rows_and_fleet_tokens() {
    let _local = LocalHome::with_local_row("local-provider");
    set_usage_broker(Some(std::sync::Arc::new(FakeBroker::ok(
        vec![broker_row("google-antigravity", 0.5)],
        vec![client("a", &[("google-antigravity", 10.0, 5.0, 2.0, 1.0)])],
    ))));
    let data = fetch_usage_data(0);
    set_usage_broker(None);
    assert_eq!(data.rows.len(), 1);
    assert_eq!(data.rows[0].provider, "google-antigravity");
    let fleet = data.fleet_tokens_by_provider.expect("fleet tokens");
    assert_eq!(fleet["google-antigravity"], 18.0);
}

#[test]
fn fetch_usage_data_history_error_falls_back_to_local() {
    let _local = LocalHome::with_local_row("local-provider");
    let broker = FakeBroker {
        history: std::sync::Mutex::new(vec![Err(anyhow::anyhow!("broker down"))]),
        summary: std::sync::Mutex::new(vec![]),
    };
    set_usage_broker(Some(std::sync::Arc::new(broker)));
    let data = fetch_usage_data(0);
    set_usage_broker(None);
    assert_eq!(data.rows.len(), 1);
    assert_eq!(data.rows[0].provider, "local-provider");
    assert_eq!(data.fleet_tokens_by_provider, None);
}

#[test]
fn fetch_usage_data_summary_error_leaves_fleet_none_but_keeps_broker_rows() {
    let _local = LocalHome::with_local_row("local-provider");
    let broker = FakeBroker {
        history: std::sync::Mutex::new(vec![Ok(vec![broker_row("google-antigravity", 0.5)])]),
        summary: std::sync::Mutex::new(vec![Err(anyhow::anyhow!("summary unavailable"))]),
    };
    set_usage_broker(Some(std::sync::Arc::new(broker)));
    let data = fetch_usage_data(0);
    set_usage_broker(None);
    assert_eq!(data.rows.len(), 1);
    assert_eq!(data.rows[0].provider, "google-antigravity");
    assert_eq!(data.fleet_tokens_by_provider, None);
}

#[test]
fn fetch_usage_data_without_broker_reads_local_db() {
    let _local = LocalHome::with_local_row("local-provider");
    set_usage_broker(None);
    let data = fetch_usage_data(0);
    assert_eq!(data.rows.len(), 1);
    assert_eq!(data.rows[0].provider, "local-provider");
    assert_eq!(data.fleet_tokens_by_provider, None);
}
