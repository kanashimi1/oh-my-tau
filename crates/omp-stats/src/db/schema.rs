//! Port of the `initDb` schema/migration/backfill body of `db.ts` (everything
//! after opening the DB and setting pragmas — those live in `db/mod.rs`).
//! Rollup tables/triggers are NOT created here: feature `db` may not depend on
//! `rollup`, so `rollup::ensure_rollup_schema` installs them on its own entry
//! points (same objects TS `ensureRollupSchema` creates, at the same point in
//! the init order: after every backfill below).

use anyhow::Result;
use rusqlite::Connection;

const BACKFILL_COMPLETE: &str = "complete";
const BACKFILL_PENDING: &str = "pending";
pub(crate) const USER_MESSAGES_BACKFILL_KEY: &str = "user_messages_v9";
pub(crate) const USER_MESSAGE_LINKS_REPAIR_KEY: &str = "user_message_links_v1";
pub(crate) const PRIORITY_PREMIUM_REQUESTS_BACKFILL_KEY: &str = "premium_requests_priority_v2";
const AGENT_TYPE_BACKFILL_KEY: &str = "agent_type_v1";
const FORK_DEDUPE_KEY: &str = "fork_dedupe_v1";
pub(crate) const TOOL_CALLS_BACKFILL_KEY: &str = "tool_calls_v2";
pub(crate) const COST_REINGEST_BACKFILL_KEY: &str = "messages_cost_reingest_v1";
pub(crate) const COST_UNPRICED_BACKFILL_KEY: &str = "messages_cost_unpriced_v1";
const NO_CACHE_INPUT_BACKFILL_KEY: &str = "cost_no_cache_input_v1";
const CATALOG_COST_BACKFILL_KEY: &str = "catalog_cost_backfill";
/// TS `VERSION` (`pi-utils`, from package.json) — the value the
/// `catalog_cost_backfill` sentinel records so zero-cost rows reprice once per
/// release. Must match the TS build byte-for-byte for shared DBs.
pub(crate) const CATALOG_BACKFILL_VERSION: &str = "18.9.1";

pub(crate) fn should_reset_backfill(value: Option<&str>) -> bool {
    value != Some(BACKFILL_COMPLETE) && value != Some(BACKFILL_PENDING)
}

/// TS `initDb` body: create/migrate every table and run the one-time
/// backfills, in the same order as the TS. Idempotent.
///
/// # Errors
/// SQL failure.
pub fn ensure_schema(conn: &Connection) -> Result<()> {
    // Sampled BEFORE CREATE TABLE adds the table: drives the one-time
    // agent_type backfill enrollment below.
    let messages_table_existed: bool = conn
        .query_row(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'messages'",
            [],
            |_| Ok(()),
        )
        .is_ok();

    conn.execute_batch(
        r#"
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

		CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp);
		CREATE INDEX IF NOT EXISTS idx_messages_entry_timestamp ON messages(entry_id, timestamp);
		CREATE INDEX IF NOT EXISTS idx_messages_model ON messages(model);
		CREATE INDEX IF NOT EXISTS idx_messages_folder ON messages(folder);
		CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_file);
		CREATE INDEX IF NOT EXISTS idx_messages_timestamp_model_provider ON messages(timestamp, model, provider);
		CREATE INDEX IF NOT EXISTS idx_messages_timestamp_folder ON messages(timestamp, folder);
		CREATE INDEX IF NOT EXISTS idx_messages_stop_reason_timestamp ON messages(stop_reason, timestamp);

		CREATE TABLE IF NOT EXISTS file_offsets (
			session_file TEXT PRIMARY KEY,
			offset INTEGER NOT NULL,
			last_modified INTEGER NOT NULL,
			parser_state TEXT
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

		CREATE INDEX IF NOT EXISTS idx_user_messages_timestamp ON user_messages(timestamp);
		CREATE INDEX IF NOT EXISTS idx_user_messages_entry_timestamp ON user_messages(entry_id, timestamp);
		CREATE INDEX IF NOT EXISTS idx_user_messages_timestamp_model ON user_messages(timestamp, model, provider);

		-- Judge verdicts keyed by prose hash: identical messages share one verdict,
		-- and verdicts outlive user_messages re-parses. No backfill wipes this table.
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

		CREATE INDEX IF NOT EXISTS idx_tool_calls_timestamp ON tool_calls(timestamp);
		CREATE INDEX IF NOT EXISTS idx_tool_calls_entry_timestamp ON tool_calls(entry_id, timestamp);
		CREATE INDEX IF NOT EXISTS idx_tool_calls_tool_timestamp ON tool_calls(tool_name, timestamp);

		CREATE TABLE IF NOT EXISTS meta (
			key TEXT PRIMARY KEY,
			value TEXT NOT NULL
		);
		"#,
    )?;

    // ---- Column migrations, each guarded by pragma table_info ----
    if !column_exists(conn, "file_offsets", "parser_state")? {
        conn.execute("ALTER TABLE file_offsets ADD COLUMN parser_state TEXT", [])?;
    }
    let message_columns = table_columns(conn, "messages")?;
    if !message_columns.iter().any(|c| c == "premium_requests") {
        conn.execute(
            "ALTER TABLE messages ADD COLUMN premium_requests REAL NOT NULL DEFAULT 0",
            [],
        )?;
    }
    if !message_columns.iter().any(|c| c == "cost_no_cache_input") {
        conn.execute(
            "ALTER TABLE messages ADD COLUMN cost_no_cache_input REAL",
            [],
        )?;
    }
    // Rows ingested before this column existed carry no served tier; a re-parse
    // fills them from the session's assistant messages.
    if !message_columns.iter().any(|c| c == "service_tier") {
        conn.execute("ALTER TABLE messages ADD COLUMN service_tier TEXT", [])?;
    }
    // Rows ingested before this column existed default to 0 (not unpriced), so
    // their epoch-sentinel zeros read as free until a re-parse rewrites them.
    if !message_columns.iter().any(|c| c == "cost_unpriced") {
        conn.execute(
            "ALTER TABLE messages ADD COLUMN cost_unpriced INTEGER NOT NULL DEFAULT 0",
            [],
        )?;
    }
    let has_agent_type_column = message_columns.iter().any(|c| c == "agent_type");
    if !has_agent_type_column {
        conn.execute(
            "ALTER TABLE messages ADD COLUMN agent_type TEXT NOT NULL DEFAULT 'main'",
            [],
        )?;
    }
    // Enroll the backfill PENDING for any pre-existing table unless a prior run
    // settled the sentinel — OR IGNORE leaves an existing COMPLETE/PENDING
    // intact, so an ALTER that committed before its sentinel write still
    // reclassifies on the next init. A brand-new table settles COMPLETE.
    conn.execute(
        "INSERT OR IGNORE INTO meta (key, value) VALUES (?1, ?2)",
        rusqlite::params![
            AGENT_TYPE_BACKFILL_KEY,
            if messages_table_existed {
                BACKFILL_PENDING
            } else {
                BACKFILL_COMPLETE
            }
        ],
    )?;
    conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_timestamp_agent_type ON messages(timestamp, agent_type)", [])?;

    // Each behavior-metric bump invalidates previously-ingested rows: detect
    // the stale schema by column name and wipe + re-ingest (v1..v9 history in
    // the TS comment on `backfillUserMessages`).
    let user_message_columns = table_columns(conn, "user_messages")?;
    let has_stale_column = user_message_columns
        .iter()
        .any(|c| c == "caps_words" || c == "drama_runs" || c == "yelling_sentences");
    let has_v4_columns = user_message_columns.iter().any(|c| c == "negation");
    let has_old_user_messages = !user_message_columns.is_empty();
    if has_stale_column || (has_old_user_messages && !has_v4_columns) {
        // IF NOT EXISTS above already produced the new schema; drop for a
        // clean wipe + re-ingest.
        conn.execute("DROP TABLE user_messages", [])?;
        conn.execute_batch(
            r#"
			CREATE TABLE user_messages (
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
			CREATE INDEX IF NOT EXISTS idx_user_messages_timestamp ON user_messages(timestamp);
			CREATE INDEX IF NOT EXISTS idx_user_messages_timestamp_model ON user_messages(timestamp, model, provider);
			"#,
        )?;
    } else if !user_message_columns.iter().any(|c| c == "prose") {
        // v9 judge prose: the backfill below clears the rows, so defaults are
        // only placeholders until the re-parse repopulates them.
        conn.execute_batch(
            "ALTER TABLE user_messages ADD COLUMN prose TEXT NOT NULL DEFAULT '';
             ALTER TABLE user_messages ADD COLUMN prose_hash TEXT NOT NULL DEFAULT '';",
        )?;
    }
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_user_messages_prose_hash ON user_messages(prose_hash)",
        [],
    )?;

    // ---- One-time backfills, TS initDb order ----
    backfills::backfill_user_messages(conn)?;
    backfills::backfill_tool_calls(conn)?;
    backfills::backfill_reingest_costs(conn)?;
    backfills::backfill_unpriced_costs(conn)?;
    backfills::repair_user_message_links(conn)?;
    backfills::backfill_priority_premium_requests(conn)?;
    backfills::backfill_agent_type(conn)?;
    backfills::backfill_missing_catalog_costs(conn)?;
    backfills::backfill_no_cache_input_costs(conn)?;
    backfills::backfill_fork_duplicates(conn)?;
    // TS ends with ensureRollupSchema(db); rollup owns that and self-installs.
    Ok(())
}

pub(crate) fn table_columns(conn: &Connection, table: &str) -> Result<Vec<String>> {
    let mut stmt = conn.prepare(&format!("PRAGMA table_info({table})"))?;
    let cols = stmt
        .query_map([], |row| row.get::<_, String>(1))?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(cols)
}

pub(crate) fn column_exists(conn: &Connection, table: &str, column: &str) -> Result<bool> {
    Ok(table_columns(conn, table)?.iter().any(|c| c == column))
}

pub(crate) fn meta_value(conn: &Connection, key: &str) -> Result<Option<String>> {
    let mut stmt = conn.prepare("SELECT value FROM meta WHERE key = ?1")?;
    let mut rows = stmt.query(rusqlite::params![key])?;
    if let Some(row) = rows.next()? {
        return Ok(Some(row.get(0)?));
    }
    Ok(None)
}

pub(crate) fn set_meta(conn: &Connection, key: &str, value: &str) -> Result<()> {
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)",
        rusqlite::params![key, value],
    )?;
    Ok(())
}

pub(crate) fn reset_offsets(conn: &Connection, key: &str) -> Result<()> {
    if !should_reset_backfill(meta_value(conn, key)?.as_deref()) {
        return Ok(());
    }
    conn.execute("DELETE FROM file_offsets", [])?;
    set_meta(conn, key, BACKFILL_PENDING)?;
    Ok(())
}

/// One-time backfill bodies and the db.ts pricing helpers shared with
/// `ingest`; split out only to keep files under ~400 lines.
#[path = "schema_backfills.rs"]
pub(crate) mod backfills;

#[cfg(any(feature = "db_ingest", feature = "db_query"))]
pub(crate) use backfills::agent_type_str;
// Pricing helpers are only consumed by the ingest write path; gate them so
// `db_query`-alone builds stay warning-free.
#[cfg(feature = "db_ingest")]
pub(crate) use backfills::{calculate_no_cache_input_cost, resolve_stored_cost, usage_from_input};

#[cfg(test)]
mod tests {
    use super::*;

    /// TS-built oracle DB from the fixture sync. Override with
    /// `GOLDEN_STATS_DB=/path/_stats.full.db` if it moves.
    fn golden_path() -> Option<std::path::PathBuf> {
        if let Ok(p) = std::env::var("GOLDEN_STATS_DB") {
            return Some(std::path::PathBuf::from(p));
        }
        let default = std::path::Path::new(
            "/home/user/omrs-fixtures/stats/golden/_stats.full.db",
        );
        default.exists().then(|| default.to_path_buf())
    }

    /// Objects rollup::ensure_rollup_schema owns (created only with the
    /// `rollup` feature; TS `ensureRollupSchema` creates them after the
    /// backfills above).
    const ROLLUP_OWNED: &[&str] = &[
        "message_rollup",
        "tool_rollup",
        "session_rollup",
        "rollup_dirty",
        "session_dirty",
        "idx_message_rollup_bucket",
        "idx_tool_rollup_bucket",
        "rollup_dirty_messages_insert",
        "rollup_dirty_messages_update",
        "rollup_dirty_messages_delete",
        "rollup_dirty_tool_calls_insert",
        "rollup_dirty_tool_calls_update",
        "rollup_dirty_tool_calls_delete",
    ];

    fn is_rollup_owned(name: &str) -> bool {
        ROLLUP_OWNED.contains(&name) || name.starts_with("rollup_")
    }

    /// name -> whitespace-normalized SQL for every object both DBs share.
    fn schema_map(conn: &Connection) -> Result<std::collections::BTreeMap<String, String>> {
        let mut out = std::collections::BTreeMap::new();
        let mut stmt =
            conn.prepare("SELECT type, name, sql FROM sqlite_master WHERE sql IS NOT NULL")?;
        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
            ))
        })?;
        for row in rows {
            let (ty, name, sql) = row?;
            if ty == "index" && name.starts_with("sqlite_autoindex") {
                continue;
            }
            if is_rollup_owned(&name) {
                continue;
            }
            let norm: String = sql.split_whitespace().collect::<Vec<_>>().join(" ");
            out.insert(name, norm);
        }
        Ok(out)
    }

    fn object_names(conn: &Connection) -> Result<Vec<String>> {
        let mut stmt = conn.prepare("SELECT name FROM sqlite_master ORDER BY name")?;
        let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
        Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
    }

    #[test]
    fn fresh_schema_matches_golden_minus_rollup() -> Result<()> {
        let Some(golden) = golden_path() else {
            eprintln!("golden DB not found; skipping");
            return Ok(());
        };
        let golden =
            Connection::open_with_flags(golden, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)?;
        let golden_map = schema_map(&golden)?;

        let fresh = Connection::open_in_memory()?;
        ensure_schema(&fresh)?;
        let fresh_map = schema_map(&fresh)?;

        // Every non-rollup object the TS created must exist with identical SQL.
        let mut missing = Vec::new();
        let mut different = Vec::new();
        for (name, sql) in &golden_map {
            match fresh_map.get(name) {
                None => missing.push(name.clone()),
                Some(fresh_sql) if fresh_sql != sql => different.push(name.clone()),
                _ => {}
            }
        }
        assert!(missing.is_empty(), "missing objects: {missing:?}");
        assert!(different.is_empty(), "sql mismatch: {different:?}");
        // And nothing extra beyond rollup objects sqlite owns implicitly.
        let extra: Vec<_> = fresh_map
            .keys()
            .filter(|k| !golden_map.contains_key(*k))
            .cloned()
            .collect();
        assert!(extra.is_empty(), "extra objects: {extra:?}");
        Ok(())
    }

    #[test]
    fn ensure_schema_is_idempotent() -> Result<()> {
        let conn = Connection::open_in_memory()?;
        ensure_schema(&conn)?;
        let first = object_names(&conn)?;
        ensure_schema(&conn)?;
        ensure_schema(&conn)?;
        let third = object_names(&conn)?;
        assert_eq!(first, third);
        // The full backfill sentinel set must be settled exactly once.
        let count: i64 = conn.query_row("SELECT COUNT(*) FROM meta", [], |r| r.get(0))?;
        assert!(count >= 8, "expected backfill sentinels, got {count}");
        Ok(())
    }
}
