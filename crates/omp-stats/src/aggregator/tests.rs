//! Aggregator parity: every dashboard read against the recorded TS goldens,
//! plus a from-scratch `sync_all_sessions` over the fixture.

use serde_json::Value;

use super::*;
use crate::testutil::golden::{assert_golden, fixture_home, golden, golden_requests, FIXTURE_HOME};
use crate::testutil::TestHome;

/// `fixture_home` with rollups drained, as the TS server had them when recording.
fn home() -> Option<TestHome> {
    let home = fixture_home()?;
    crate::rollup::refresh_rollups().expect("refresh rollups");
    Some(home)
}

fn query_param(request: &str, key: &str) -> Option<String> {
    let query = request.split_once('?')?.1;
    query.split('&').find_map(|pair| {
        let (k, v) = pair.split_once('=')?;
        (k == key).then(|| v.to_owned())
    })
}

/// Compare `read(range)` with every recorded `path?range=…` golden.
fn check_ranges<T: serde::Serialize>(path: &str, read: impl Fn(Option<&str>) -> Result<T>) {
    let Some(home) = home() else { return };
    let requests = golden_requests(path);
    assert!(
        requests.iter().any(|r| r.contains("range=all")),
        "{path}: no range=all golden"
    );
    assert!(
        requests.len() >= 2,
        "{path}: expected all + relative ranges"
    );
    for request in requests {
        let range = query_param(&request, "range");
        let actual =
            serde_json::to_value(read(range.as_deref()).expect("read")).expect("serialize");
        assert_golden(&request, &actual, &home);
    }
}

#[test]
fn dashboard_stats_match_golden() {
    check_ranges("/api/stats", get_dashboard_stats);
}

#[test]
fn overview_stats_match_golden() {
    check_ranges("/api/stats/overview", get_overview_stats);
}

#[test]
fn model_dashboard_stats_match_golden() {
    check_ranges("/api/stats/model-dashboard", get_model_dashboard_stats);
}

#[test]
fn cost_dashboard_stats_match_golden() {
    check_ranges("/api/stats/costs", get_cost_dashboard_stats);
}

#[test]
fn tool_dashboard_stats_match_golden() {
    check_ranges("/api/stats/tools", get_tool_dashboard_stats);
}

#[test]
fn provider_dashboard_stats_match_golden() {
    check_ranges("/api/stats/providers", get_provider_dashboard_stats);
}

#[test]
fn folder_stats_match_golden() {
    check_ranges("/api/stats/folders", get_folder_stats);
}

#[test]
fn recent_errors_match_golden() {
    let Some(home) = home() else { return };
    let requests = golden_requests("/api/stats/errors");
    assert!(requests.len() >= 6, "errors goldens: {requests:?}");
    for request in requests {
        let range = query_param(&request, "range");
        let limit = query_param(&request, "limit").map(|l| l.parse::<i64>().expect("limit"));
        let actual =
            serde_json::to_value(get_recent_errors(range.as_deref(), limit).expect("errors"))
                .expect("json");
        if range.is_none() {
            // Default range is "24h" relative to now, which `assert_golden` only
            // shape-checks when the query says `range=`; compare shape here too.
            let (_, expected) = golden(&request);
            assert_eq!(actual.is_array(), expected.is_array(), "{request}");
            continue;
        }
        assert_golden(&request, &actual, &home);
    }
}

#[test]
fn recent_requests_match_golden() {
    let Some(home) = home() else { return };
    let actual =
        serde_json::to_value(get_recent_requests(Some(50)).expect("recent")).expect("json");
    assert_golden("/api/stats/recent?limit=50", &actual, &home);
    assert_eq!(get_recent_requests(None).expect("default limit").len(), 100);
}

/// The Rust rowid of the golden row (`sessionFile` + `entryId`): rowids depend on ingest order.
fn rust_id(home: &TestHome, expected: &Value) -> i64 {
    let file = expected["sessionFile"].as_str().expect("sessionFile");
    let file = file.replace(FIXTURE_HOME, &home.path().to_string_lossy());
    let entry = expected["entryId"].as_str().expect("entryId").to_owned();
    crate::db::with_db(|conn| {
        Ok(conn.query_row(
            "SELECT id FROM messages WHERE session_file = ?1 AND entry_id = ?2",
            rusqlite::params![file, entry],
            |row| row.get(0),
        )?)
    })
    .expect("golden row ingested")
}

#[test]
fn request_details_match_golden() {
    let Some(home) = home() else { return };
    for request in ["/api/request/1", "/api/request/2"] {
        let (status, expected) = golden(request);
        assert_eq!(status, 200, "{request}");
        let id = rust_id(&home, &expected);
        let details = get_request_details(id).expect("details").expect("known id");
        assert_golden(
            request,
            &serde_json::to_value(details).expect("json"),
            &home,
        );
    }
    let (status, _) = golden("/api/request/999999");
    assert_eq!(status, 404);
    assert!(get_request_details(999_999).expect("details").is_none());
}

#[test]
fn provider_window_stats_without_usage_snapshots_are_empty() {
    let Some(_home) = home() else { return };
    let stats = get_provider_window_stats(Some("all"), Some("anthropic")).expect("windows");
    assert_eq!(
        serde_json::to_value(stats).expect("json"),
        serde_json::json!({"windowInsights": [], "usageSeries": []})
    );
}

fn count(table: &str) -> i64 {
    crate::db::with_db(|conn| {
        Ok(conn.query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |r| r.get(0))?)
    })
    .expect("count")
}

/// A temp HOME whose sessions dir links to the fixture, nothing ingested.
pub(crate) fn linked_fixture_home() -> Option<TestHome> {
    let sessions = std::path::Path::new(FIXTURE_HOME).join(".omp/agent/sessions");
    if !sessions.is_dir() {
        return None;
    }
    let home = TestHome::new();
    let agent = home.path().join(".omp/agent");
    std::fs::create_dir_all(&agent).expect("agent dir");
    std::os::unix::fs::symlink(&sessions, agent.join("sessions")).expect("link sessions");
    Some(home)
}

#[test]
fn sync_all_sessions_ingests_fixture_then_is_incremental() {
    let Some(_home) = linked_fixture_home() else {
        return;
    };
    let events = std::cell::RefCell::new(Vec::<SyncProgress>::new());
    let on_progress = |e: &SyncProgress| events.borrow_mut().push(e.clone());
    let first = sync_all_sessions(&SyncOptions {
        on_progress: Some(&on_progress),
        ..SyncOptions::default()
    })
    .expect("first sync");
    // TS `processed` counts message + user-message rows.
    assert_eq!(first.processed, 900 + 41);
    assert_eq!(count("messages"), 900);
    assert_eq!(count("tool_calls"), 1113);
    assert_eq!(count("user_messages"), 41);
    let events = events.into_inner();
    let total = crate::parser::list_all_session_files().len() as i64;
    assert_eq!(events.len() as i64, total, "one progress event per file");
    assert!(events
        .iter()
        .enumerate()
        .all(|(i, e)| e.current == i as i64 + 1 && e.total == total));
    assert!(events
        .last()
        .is_some_and(|e| e.processed == 941 && e.changes > 0));

    let second = sync_all_sessions(&SyncOptions::default()).expect("second sync");
    assert_eq!(
        second,
        SyncResult {
            processed: 0,
            files: 0
        }
    );
    assert_eq!(count("messages"), 900);
}

#[test]
fn serial_sync_matches_parallel_totals() {
    let Some(_home) = linked_fixture_home() else {
        return;
    };
    let result = sync_all_sessions(&SyncOptions {
        workers: Some(1),
        ..SyncOptions::default()
    })
    .expect("serial sync");
    assert_eq!(result.processed, 941);
    assert_eq!((count("tool_calls"), count("user_messages")), (1113, 41));
    // Targeted sync of an unchanged + a missing file: nothing to do, no error.
    let file = crate::parser::list_all_session_files()
        .into_iter()
        .next()
        .expect("a session");
    let targeted = SyncOptions {
        files: Some(vec![file, "/nonexistent/x.jsonl".into()]),
        ..SyncOptions::default()
    };
    assert_eq!(
        sync_all_sessions(&targeted).expect("targeted"),
        SyncResult::default()
    );
}

#[test]
fn total_message_count_matches_fixture() {
    let Some(_home) = home() else { return };
    assert_eq!(get_total_message_count().expect("count"), 900);
}

/// The TS-written stats db the goldens were recorded from (fixture fully ingested).
const TS_DB: &str = "/home/user/omrs-fixtures/stats/golden/_stats.full.db";

/// Node/Bun `stat.mtimeMs` / `birthtimeMs` formula (`sec * 1e3 + nsec / 1e6`).
fn node_ms(t: std::time::SystemTime) -> f64 {
    let d = t.duration_since(std::time::UNIX_EPOCH).expect("post-epoch");
    (d.as_secs() as f64) * 1000.0 + f64::from(d.subsec_nanos()) / 1_000_000.0
}

#[test]
fn ts_written_fractional_cursor_is_read_and_skipped() {
    let Some(_home) = linked_fixture_home() else {
        return;
    };
    let file = crate::parser::list_all_session_files()
        .into_iter()
        .next()
        .expect("a session");
    let meta = std::fs::metadata(&file).expect("stat");
    let mtime = node_ms(meta.modified().expect("mtime"));
    let mut state = crate::parser::parse_session_file(&file, 0, None, false)
        .expect("parse")
        .parser_state
        .expect("parser state");
    // Exactly what TS writes: fractional mtime/birthtime, REAL last_modified.
    state.mtime_ms = mtime;
    state.birthtime_ms = node_ms(meta.created().expect("birthtime"));
    // Probe: a stale checkpoint would make any re-parse replay the whole file,
    // so zero rows below proves the skip check (not the tail resume) fired.
    state.checkpoint = "0000000000000000".into();
    let json = serde_json::to_string(&state).expect("state json");
    crate::db::with_db(|conn| {
        conn.execute(
            "INSERT INTO file_offsets (session_file, offset, last_modified, parser_state) VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![file, state.offset, mtime, json],
        )?;
        let kind: String =
            conn.query_row("SELECT typeof(last_modified) FROM file_offsets", [], |r| r.get(0))?;
        assert_eq!(kind, if mtime.fract() == 0.0 { "integer" } else { "real" });
        Ok(())
    })
    .expect("seed TS cursor");
    let stored = crate::db::get_file_offset(&file)
        .expect("read REAL cursor")
        .expect("cursor");
    assert!(
        (stored.last_modified - mtime).abs() < f64::EPSILON,
        "fractional mtime survives the round trip"
    );
    let targeted = SyncOptions {
        files: Some(vec![file]),
        ..SyncOptions::default()
    };
    assert_eq!(
        sync_all_sessions(&targeted).expect("sync over a TS cursor"),
        SyncResult::default()
    );
    assert_eq!(count("messages"), 0, "the file was skipped, not re-parsed");
}

#[test]
fn sync_over_ts_written_db_processes_nothing() {
    if !std::path::Path::new(TS_DB).is_file() {
        return;
    }
    let Some(home) = linked_fixture_home() else {
        return;
    };
    let db = crate::paths::stats_db_path();
    std::fs::create_dir_all(db.parent().expect("db dir")).expect("mkdir");
    std::fs::copy(TS_DB, &db).expect("copy TS db");
    let here = home.path().to_string_lossy().into_owned();
    crate::db::with_db(|conn| {
        // The TS db names the fixture's own path; point it at this HOME's link.
        for table in ["messages", "user_messages", "tool_calls", "file_offsets"] {
            conn.execute(
                &format!("UPDATE {table} SET session_file = replace(session_file, ?1, ?2)"),
                rusqlite::params![FIXTURE_HOME, here],
            )?;
        }
        Ok(())
    })
    .expect("remap session paths");
    let cursors = || {
        crate::db::with_db(|conn| {
            let mut stmt = conn.prepare(
                "SELECT session_file, last_modified, parser_state FROM file_offsets ORDER BY 1",
            )?;
            let rows = stmt.query_map([], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, f64>(1)?,
                    r.get::<_, String>(2)?,
                ))
            })?;
            Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
        })
        .expect("cursors")
    };
    let before = cursors();
    let result = sync_all_sessions(&SyncOptions::default()).expect("sync over TS db");
    assert_eq!(
        result,
        SyncResult::default(),
        "TS already ingested every transcript"
    );
    assert_eq!(
        (
            count("messages"),
            count("tool_calls"),
            count("user_messages")
        ),
        (900, 1113, 41)
    );
    assert_eq!(
        cursors(),
        before,
        "every TS cursor matched: no file re-parsed or rewritten"
    );
}
