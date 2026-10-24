//! `ensureRollupSchema`: create (or rebuild after a version bump) the rollup
//! tables and the triggers that mark hours dirty.

use anyhow::Result;
use rusqlite::params;
use rusqlite::Connection;

use super::{
    hour_of, ImmediateTx, ROLLUP_VERSION, ROLLUP_VERSION_KEY, TRIGGER_VERSION, TRIGGER_VERSION_KEY,
};

/// Trigger statements marking a row's hour and transcript dirty (`row` is
/// `NEW` or `OLD`), without any conflict clause.
fn mark_dirty(row: &str) -> String {
    format!(
        "INSERT INTO rollup_dirty (bucket)
		SELECT h FROM (SELECT {h} AS h) WHERE h NOT IN (SELECT bucket FROM rollup_dirty);
INSERT INTO session_dirty (session_file)
		SELECT {row}.session_file WHERE {row}.session_file NOT IN (SELECT session_file FROM session_dirty);",
        h = hour_of(&format!("{row}.timestamp"))
    )
}

const ROLLUP_TABLES: &str = r#"
CREATE TABLE IF NOT EXISTS message_rollup (
	bucket INTEGER NOT NULL,
	model TEXT NOT NULL,
	provider TEXT NOT NULL,
	folder TEXT NOT NULL,
	agent_type TEXT NOT NULL,
	requests INTEGER NOT NULL,
	failed INTEGER NOT NULL,
	input_tokens INTEGER NOT NULL,
	output_tokens INTEGER NOT NULL,
	cache_read_tokens INTEGER NOT NULL,
	cache_write_tokens INTEGER NOT NULL,
	total_tokens INTEGER NOT NULL,
	premium_requests REAL NOT NULL,
	cost_total REAL NOT NULL,
	cost_input REAL NOT NULL,
	cost_output REAL NOT NULL,
	cost_cache_read REAL NOT NULL,
	cost_cache_write REAL NOT NULL,
	unpriced INTEGER NOT NULL,
	cached_prompt_cost REAL NOT NULL,
	no_cache_input_cost REAL NOT NULL,
	duration_sum REAL NOT NULL,
	duration_n INTEGER NOT NULL,
	ttft_sum REAL NOT NULL,
	ttft_n INTEGER NOT NULL,
	tps_sum REAL NOT NULL,
	tps_n INTEGER NOT NULL,
	first_ts INTEGER NOT NULL,
	last_ts INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_message_rollup_bucket ON message_rollup(bucket);
CREATE TABLE IF NOT EXISTS tool_rollup (
	bucket INTEGER NOT NULL,
	tool_name TEXT NOT NULL,
	model TEXT NOT NULL,
	provider TEXT NOT NULL,
	calls INTEGER NOT NULL,
	errors INTEGER NOT NULL,
	args_chars INTEGER NOT NULL,
	result_chars INTEGER NOT NULL,
	total_tokens_share REAL NOT NULL,
	output_tokens_share REAL NOT NULL,
	cost_share REAL NOT NULL,
	unpriced_share REAL NOT NULL,
	last_used INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tool_rollup_bucket ON tool_rollup(bucket);
CREATE TABLE IF NOT EXISTS rollup_dirty (bucket INTEGER PRIMARY KEY) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS session_rollup (
	session_file TEXT PRIMARY KEY,
	requests INTEGER NOT NULL,
	started_at INTEGER NOT NULL,
	ended_at INTEGER NOT NULL,
	total_tokens INTEGER NOT NULL,
	cost_total REAL NOT NULL,
	unpriced INTEGER NOT NULL,
	models TEXT,
	tool_calls INTEGER NOT NULL
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS session_dirty (session_file TEXT PRIMARY KEY) WITHOUT ROWID;
"#;

/// TS `ensureRollupSchema`: version-checked and idempotent.
///
/// Steady state touches no schema: DDL on every open would contend with
/// other omp processes writing the same database.
///
/// # Errors
/// SQL failure (also when the raw `meta` table does not exist yet).
pub fn ensure_rollup_schema(conn: &Connection) -> Result<()> {
    let fresh = read_meta(conn, ROLLUP_VERSION_KEY)?.as_deref() != Some(ROLLUP_VERSION);
    if !fresh && read_meta(conn, TRIGGER_VERSION_KEY)?.as_deref() == Some(TRIGGER_VERSION) {
        return Ok(());
    }
    let tx = ImmediateTx::begin(conn)?;
    if fresh {
        for table in [
            "message_rollup",
            "tool_rollup",
            "rollup_dirty",
            "session_rollup",
            "session_dirty",
        ] {
            conn.execute_batch(&format!("DROP TABLE IF EXISTS {table}"))?;
        }
    }
    conn.execute_batch(ROLLUP_TABLES)?;
    // A tool share reads its assistant message, so message changes dirty the
    // hour for both rollups (tool calls share their message's timestamp).
    // Triggers must not use `OR IGNORE`: a trigger statement inherits the
    // outer statement's conflict policy, and the UPSERTs in
    // `insertMessageStats` would turn a duplicate dirty mark into a failed ingest.
    for table in ["messages", "tool_calls"] {
        for (event, body) in [
            ("insert", mark_dirty("NEW")),
            ("delete", mark_dirty("OLD")),
            (
                "update",
                format!("{} {}", mark_dirty("OLD"), mark_dirty("NEW")),
            ),
        ] {
            let name = format!("rollup_dirty_{table}_{event}");
            conn.execute_batch(&format!("DROP TRIGGER IF EXISTS {name}"))?;
            conn.execute_batch(&format!(
                "CREATE TRIGGER {name} AFTER {event_upper} ON {table} BEGIN {body} END",
                event_upper = event.to_uppercase()
            ))?;
        }
    }
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
        params![TRIGGER_VERSION_KEY, TRIGGER_VERSION],
    )?;
    if fresh {
        let hour = hour_of("timestamp");
        conn.execute_batch(&format!(
            "INSERT OR IGNORE INTO rollup_dirty (bucket)
SELECT DISTINCT {hour} FROM messages
UNION SELECT DISTINCT {hour} FROM tool_calls"
        ))?;
        conn.execute_batch(
            "INSERT OR IGNORE INTO session_dirty (session_file)
SELECT DISTINCT session_file FROM messages UNION SELECT DISTINCT session_file FROM tool_calls",
        )?;
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            params![ROLLUP_VERSION_KEY, ROLLUP_VERSION],
        )?;
    }
    tx.commit()
}

fn read_meta(conn: &Connection, key: &str) -> Result<Option<String>> {
    Ok(conn
        .query_row("SELECT value FROM meta WHERE key = ?", [key], |row| {
            row.get::<_, String>(0)
        })
        .ok())
}
