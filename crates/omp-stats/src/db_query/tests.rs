//! Tests run against a fresh `~/.omp/stats.db` in a temp HOME. The schema is
//! the verbatim `CREATE TABLE` text from `db.ts` `initDb`; rows are seeded here.

use rusqlite::{params, Connection};

use super::*;
use crate::db::with_db;
use crate::testutil::TestHome;
use crate::types::{StopReason, ToolCallStats, ToolResultLink, UserMessageLink, UserMessageStats};

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS messages (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	session_file TEXT NOT NULL,
	entry_id TEXT NOT NULL,
	folder TEXT NOT NULL,
	model TEXT NOT NULL,
	provider TEXT NOT NULL,
	api TEXT NOT NULL,
	timestamp INTEGER NOT NULL,
	duration INTEGER,
	ttft INTEGER,
	stop_reason TEXT NOT NULL,
	error_message TEXT,
	input_tokens INTEGER NOT NULL,
	output_tokens INTEGER NOT NULL,
	cache_read_tokens INTEGER NOT NULL,
	cache_write_tokens INTEGER NOT NULL,
	total_tokens INTEGER NOT NULL,
	premium_requests REAL NOT NULL,
	cost_input REAL NOT NULL,
	cost_output REAL NOT NULL,
	cost_cache_read REAL NOT NULL,
	cost_cache_write REAL NOT NULL,
	cost_total REAL NOT NULL,
	cost_no_cache_input REAL,
	cost_unpriced INTEGER NOT NULL DEFAULT 0,
	agent_type TEXT NOT NULL DEFAULT 'main',
	service_tier TEXT,
	UNIQUE(session_file, entry_id)
);
CREATE TABLE IF NOT EXISTS user_messages (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	session_file TEXT NOT NULL,
	entry_id TEXT NOT NULL,
	folder TEXT NOT NULL,
	timestamp INTEGER NOT NULL,
	model TEXT,
	provider TEXT,
	chars INTEGER NOT NULL,
	words INTEGER NOT NULL,
	yelling INTEGER NOT NULL,
	profanity INTEGER NOT NULL,
	anguish INTEGER NOT NULL,
	negation INTEGER NOT NULL DEFAULT 0,
	repetition INTEGER NOT NULL DEFAULT 0,
	blame INTEGER NOT NULL DEFAULT 0,
	prose TEXT NOT NULL DEFAULT '',
	prose_hash TEXT NOT NULL DEFAULT '',
	UNIQUE(session_file, entry_id)
);
CREATE TABLE IF NOT EXISTS frustration_verdicts (
	prose_hash TEXT PRIMARY KEY,
	p_annoyed REAL NOT NULL,
	p_angry REAL NOT NULL,
	target TEXT NOT NULL,
	judge TEXT NOT NULL,
	judged_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS tool_calls (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	session_file TEXT NOT NULL,
	entry_id TEXT NOT NULL,
	tool_call_id TEXT NOT NULL,
	folder TEXT NOT NULL,
	tool_name TEXT NOT NULL,
	model TEXT NOT NULL,
	provider TEXT NOT NULL,
	timestamp INTEGER NOT NULL,
	agent_type TEXT NOT NULL DEFAULT 'main',
	calls_in_turn INTEGER NOT NULL DEFAULT 1,
	args_chars INTEGER NOT NULL DEFAULT 0,
	result_chars INTEGER,
	is_error INTEGER,
	UNIQUE(session_file, tool_call_id)
);
";

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| i64::try_from(d.as_millis()).unwrap_or(0))
        .unwrap_or(0)
}

/// A TestHome whose `.omp/stats.db` has the schema and no rows yet.
fn fresh_home() -> TestHome {
    let home = TestHome::new();
    let omp = home.path().join(".omp");
    std::fs::create_dir_all(&omp).expect("omp dir");
    let conn = Connection::open(omp.join("stats.db")).expect("open fixture db");
    conn.execute_batch(SCHEMA).expect("schema");
    home
}

/// Seed `messages` rows: `(entry_id, timestamp, stop_reason)`.
fn seed_messages(rows: &[(String, i64, &str)]) {
    with_db(|conn| {
        for (entry, ts, stop) in rows {
            conn.execute(
                "INSERT INTO messages (session_file, entry_id, folder, model, provider, api, timestamp,
                    stop_reason, input_tokens, output_tokens, cache_read_tokens, cache_write_tokens,
                    total_tokens, premium_requests, cost_input, cost_output, cost_cache_read,
                    cost_cache_write, cost_total)
                 VALUES ('s.jsonl', ?, '/w', 'm1', 'anthropic', 'anthropic-messages', ?, ?,
                    10, 5, 2, 1, 18, 0.33, 0.1, 0.2, 0.03, 0.04, 0.37)",
                params![entry, ts, stop],
            )?;
        }
        Ok(())
    })
    .expect("seed messages");
}

#[test]
fn recent_requests_newest_first_and_limited() {
    let _home = fresh_home();
    let base = now_ms() - 10_000_000;
    let rows: Vec<(String, i64, &str)> = (0..8)
        .map(|i| {
            (
                format!("e{i}"),
                base + i * 1000,
                if i == 3 { "error" } else { "stop" },
            )
        })
        .collect();
    seed_messages(&rows);

    let recent = get_recent_requests(5).expect("recent");
    assert_eq!(recent.len(), 5);
    let stamps: Vec<i64> = recent.iter().map(|m| m.timestamp).collect();
    assert_eq!(
        stamps,
        vec![
            base + 7000,
            base + 6000,
            base + 5000,
            base + 4000,
            base + 3000
        ]
    );
    assert_eq!(recent[0].entry_id, "e7");
    assert_eq!(recent[0].usage.total_tokens, 18);
    assert_eq!(recent[0].cost_unpriced, Some(false));
}

#[test]
fn unknown_stop_reason_is_preserved_not_collapsed_to_stop() {
    let _home = fresh_home();
    let base = now_ms() - 1_000_000;
    seed_messages(&[
        ("odd".into(), base, "contentFilter"),
        ("known".into(), base + 1000, "tool_calls"),
    ]);
    let recent = get_recent_requests(2).expect("recent");
    assert_eq!(recent[0].entry_id, "known");
    assert_eq!(
        recent[0].stop_reason,
        StopReason::ToolUse,
        "legacy alias maps to toolUse"
    );
    assert_eq!(
        recent[1].stop_reason,
        StopReason::Other("contentFilter".into())
    );
    // The wire form is the stored string, so the API and a re-read agree.
    let json = serde_json::to_value(&recent[1]).expect("serialize");
    assert!(json.to_string().contains("\"contentFilter\""), "{json}");
    let back: StopReason =
        serde_json::from_value(serde_json::json!("contentFilter")).expect("deserialize");
    assert_eq!(back, StopReason::Other("contentFilter".into()));
}

#[test]
fn message_by_id_round_trips() {
    let _home = fresh_home();
    let base = now_ms() - 1_000_000;
    seed_messages(&[("only".into(), base, "stop")]);

    let id = get_recent_requests(1).expect("recent")[0].id.expect("id");
    let found = get_message_by_id(id).expect("by id").expect("row");
    assert_eq!(found.entry_id, "only");
    assert_eq!(found.id, Some(id));
    assert!((found.usage.cost.total - 0.37).abs() < 1e-9);
    assert!(get_message_by_id(id + 999).expect("missing").is_none());
}

#[test]
fn recent_errors_filters_stop_reason_and_cutoff() {
    let _home = fresh_home();
    let base = now_ms() - 1_000_000;
    seed_messages(&[
        ("a".into(), base, "error"),
        ("b".into(), base + 1000, "stop"),
        ("c".into(), base + 2000, "error"),
    ]);
    let all = get_recent_errors(10, None).expect("errors");
    assert_eq!(
        all.iter().map(|m| m.entry_id.as_str()).collect::<Vec<_>>(),
        vec!["c", "a"]
    );
    let late = get_recent_errors(10, Some(base + 1500)).expect("errors cutoff");
    assert_eq!(late.len(), 1);
    assert_eq!(late[0].entry_id, "c");
}

#[cfg(not(feature = "rollup"))]
#[test]
fn daily_activity_from_raw_rows_is_non_empty() {
    let _home = fresh_home();
    let now = now_ms();
    let rows: Vec<(String, i64, &str)> = (0..48)
        .map(|i| (format!("h{i}"), now - i * 3_600_000, "stop"))
        .collect();
    seed_messages(&rows);

    let days = get_daily_activity(371).expect("daily");
    assert!(!days.is_empty());
    assert_eq!(days.iter().map(|d| d.requests).sum::<i64>(), 48);
    assert!(days.iter().all(|d| d.total_tokens == d.requests * 18));
}

#[test]
fn tool_calls_insert_then_update_results() {
    let _home = fresh_home();
    let call = |session: &str, id: &str| ToolCallStats {
        session_file: session.into(),
        entry_id: "e1".into(),
        tool_call_id: id.into(),
        folder: "/w".into(),
        tool_name: "read".into(),
        model: "m1".into(),
        provider: "anthropic".into(),
        timestamp: 1,
        calls_in_turn: 2,
        args_chars: 9,
        ..Default::default()
    };
    let calls = || vec![call("s1", "c1"), call("s1", "c2"), call("s1", "c3")];

    assert_eq!(insert_tool_calls(calls()).expect("insert"), 3);
    // Same assistant entry copied into a forked session: first write wins.
    assert_eq!(
        insert_tool_calls(vec![call("s2", "c1"), call("s2", "c2")]).expect("fork"),
        0
    );

    let link = |id: &str, chars: i64, err: bool| ToolResultLink {
        session_file: "s1".into(),
        tool_call_id: id.into(),
        result_chars: chars,
        is_error: err,
    };
    let links = || {
        vec![
            link("c1", 100, false),
            link("c2", 0, true),
            link("missing", 5, false),
        ]
    };
    assert_eq!(update_tool_results(links()).expect("update"), 2);
    // `result_chars IS NULL` guard: a re-sync changes nothing.
    assert_eq!(update_tool_results(links()).expect("resync"), 0);

    let (with_result, errors): (i64, i64) = with_db(|conn| {
        Ok(conn.query_row(
            "SELECT COUNT(result_chars), SUM(is_error) FROM tool_calls",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )?)
    })
    .expect("counts");
    assert_eq!((with_result, errors), (2, 1));
}

fn user_msg(session: &str, entry: &str, ts: i64, prose: &str, hash: &str) -> UserMessageStats {
    UserMessageStats {
        session_file: session.into(),
        entry_id: entry.into(),
        folder: "/w".into(),
        timestamp: ts,
        chars: prose.len() as i64,
        words: 2,
        prose: prose.into(),
        prose_hash: hash.into(),
        ..Default::default()
    }
}

#[test]
fn user_messages_insert_link_and_frustration_cache() {
    let _home = fresh_home();
    let base = now_ms() - 100_000;
    let msgs = vec![
        user_msg("s1", "u1", base, "why is this broken", "h1"),
        user_msg("s1", "u2", base + 1, "thanks a lot", "h2"),
    ];
    assert_eq!(insert_user_message_stats(msgs).expect("insert"), 2);
    // Same (entry_id, timestamp) owned by another session: skipped.
    assert_eq!(
        insert_user_message_stats(vec![user_msg("s2", "u1", base, "x", "h1")]).expect("dup"),
        0
    );

    let link = |entry: &str| UserMessageLink {
        session_file: "s1".into(),
        entry_id: entry.into(),
        model: "m1".into(),
        provider: "anthropic".into(),
    };
    assert_eq!(
        update_user_message_links(vec![link("u1")]).expect("link"),
        1
    );
    assert_eq!(
        update_user_message_links(vec![link("u1")]).expect("relink"),
        0
    );

    let by_model = get_frustration_by_model(None).expect("by model");
    assert_eq!(by_model.len(), 1);
    assert_eq!(by_model[0].counts.messages, 1);
    assert_eq!(by_model[0].first_seen, base);

    let overall = get_frustration_overall(None).expect("overall");
    assert_eq!(overall.messages, 2);
    assert_eq!(overall.judged, 0);

    let pending = get_pending_frustration_totals(None).expect("totals");
    assert_eq!(pending.messages, 2);
    assert_eq!(
        pending.chars,
        "why is this broken".len() as i64 + "thanks a lot".len() as i64
    );
    assert_eq!(
        get_pending_frustration_prose(Some(base + 1))
            .expect("prose")
            .len(),
        1
    );

    upsert_frustration_verdicts(&[FrustrationVerdict {
        prose_hash: "h1".into(),
        p_annoyed: 0.9,
        p_angry: 0.1,
        target: "assistant".into(),
        judge: "p/m".into(),
        judged_at: base,
    }])
    .expect("upsert");
    let judged = get_frustration_overall(None).expect("overall after verdict");
    assert_eq!(
        (judged.judged, judged.annoyed, judged.at_assistant),
        (1, 1, 1)
    );
    assert_eq!(
        get_pending_frustration_totals(None)
            .expect("totals")
            .messages,
        1
    );
}
