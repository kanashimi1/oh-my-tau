use super::*;
use crate::testutil::TestHome;
use rusqlite::params;

/// Raw tables as created by TS `initDb` (db.ts), so the triggers and the
/// rollup fill read the same columns the parser writes.
const RAW_SCHEMA: &str = "
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
CREATE TABLE IF NOT EXISTS meta (
	key TEXT PRIMARY KEY,
	value TEXT NOT NULL
);
";

const HOUR_A: i64 = (1_700_000_000_000 / HOUR_MS) * HOUR_MS;
const HOUR_B: i64 = HOUR_A + HOUR_MS;

/// (session, entry, timestamp, input, output, total)
fn insert_message(
    conn: &Connection,
    session: &str,
    entry: &str,
    ts: i64,
    input: i64,
    output: i64,
    total: i64,
) {
    conn.execute(
        "INSERT INTO messages (session_file, entry_id, folder, model, provider, api, timestamp,
			stop_reason, input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, total_tokens,
			premium_requests, cost_input, cost_output, cost_cache_read, cost_cache_write, cost_total)
		VALUES (?1, ?2, '/work', 'glm-5.3-flash', 'zai', 'openai-completions', ?3,
			'stop', ?4, ?5, 0, 0, ?6, 0, 0.1, 0.2, 0, 0, 0.3)",
        params![session, entry, ts, input, output, total],
    )
    .expect("insert message");
}

fn insert_tool_call(
    conn: &Connection,
    session: &str,
    entry: &str,
    call_id: &str,
    tool: &str,
    ts: i64,
    is_error: i64,
) {
    conn.execute(
		"INSERT INTO tool_calls (session_file, entry_id, tool_call_id, folder, tool_name, model, provider,
			timestamp, args_chars, result_chars, is_error)
		VALUES (?1, ?2, ?3, '/work', ?4, 'glm-5.3-flash', 'zai', ?5, 10, 20, ?6)",
		params![session, entry, call_id, tool, ts, is_error],
	)
	.expect("insert tool call");
}

fn scalar(conn: &Connection, sql: &str) -> i64 {
    conn.query_row(sql, [], |row| row.get(0))
        .expect("scalar query")
}

#[test]
fn refresh_builds_hourly_rollups_and_range_query_reads_them() {
    let _home = TestHome::new();
    crate::db::with_db(|conn| {
        conn.execute_batch(RAW_SCHEMA)?;
        ensure_rollup_schema(conn)?;
        // Two messages in hour A (one per session), one in hour B.
        insert_message(conn, "a.jsonl", "e1", HOUR_A + 1_000, 10, 5, 18);
        insert_message(conn, "a.jsonl", "e2", HOUR_A + 2_000, 20, 6, 26);
        insert_message(conn, "b.jsonl", "e3", HOUR_B + 500, 5, 1, 6);
        insert_tool_call(conn, "a.jsonl", "e1", "c1", "bash", HOUR_A + 1_000, 0);
        insert_tool_call(conn, "b.jsonl", "e3", "c2", "read", HOUR_B + 500, 1);
        Ok(())
    })
    .expect("seed raw rows");

    refresh_rollups().expect("refresh");

    crate::db::with_db(|conn| {
        // Refresh drains the backlog.
        assert_eq!(scalar(conn, "SELECT COUNT(*) FROM rollup_dirty"), 0);
        assert_eq!(scalar(conn, "SELECT COUNT(*) FROM session_dirty"), 0);

        // One message_rollup row per hour (same dimensions), sums match raw rows.
        assert_eq!(scalar(conn, "SELECT COUNT(*) FROM message_rollup"), 2);
        let (hour_a_requests, hour_a_tokens): (i64, i64) = conn.query_row(
            "SELECT requests, total_tokens FROM message_rollup WHERE bucket = ?",
            [HOUR_A],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;
        assert_eq!((hour_a_requests, hour_a_tokens), (2, 44));
        assert_eq!(
            scalar(conn, "SELECT SUM(total_tokens) FROM message_rollup"),
            scalar(conn, "SELECT SUM(total_tokens) FROM messages"),
        );

        // Tool rollup: one row per hour per tool, error counted once.
        assert_eq!(scalar(conn, "SELECT COUNT(*) FROM tool_rollup"), 2);
        assert_eq!(scalar(conn, "SELECT SUM(calls) FROM tool_rollup"), 2);
        assert_eq!(scalar(conn, "SELECT SUM(errors) FROM tool_rollup"), 1);

        // Session rollup: one row per transcript.
        assert_eq!(scalar(conn, "SELECT COUNT(*) FROM session_rollup"), 2);
        assert_eq!(
            scalar(
                conn,
                "SELECT requests FROM session_rollup WHERE session_file = 'a.jsonl'"
            ),
            2
        );
        Ok(())
    })
    .expect("assert rollup tables");

    // Range query starting at hour B reads only that hour's rollup.
    let overall = get_overall_stats(Some(HOUR_B)).expect("overall stats");
    assert_eq!(overall.total_requests, 1);
    assert_eq!(overall.total_input_tokens, 5);

    // A later write re-marks its hour dirty; the next refresh re-rolls it.
    crate::db::with_db(|conn| {
        insert_message(conn, "a.jsonl", "e4", HOUR_A + 3_000, 1, 1, 2);
        Ok(())
    })
    .expect("insert late message");
    crate::db::with_db(|conn| Ok(scalar(conn, "SELECT COUNT(*) FROM rollup_dirty")))
        .map(|dirty| assert_eq!(dirty, 1))
        .expect("dirty mark");
    refresh_rollups().expect("second refresh");
    let overall = get_overall_stats(None).expect("overall after refresh");
    assert_eq!(overall.total_requests, 4);
}
