//! Row-writing statements shared by the standalone ingest exports and the
//! `apply_session_parse_results` batch transaction (part of `ingest`).

use anyhow::Result;
use rusqlite::Connection;

use crate::parser::SessionParserState;
use crate::types::{MessageStatsInput, ServiceTier};

use crate::db::schema::{
    agent_type_str, calculate_no_cache_input_cost, resolve_stored_cost, usage_from_input,
};

fn service_tier_str(tier: Option<ServiceTier>) -> Option<&'static str> {
    tier.map(|t| match t {
        ServiceTier::Auto => "auto",
        ServiceTier::Default => "default",
        ServiceTier::Flex => "flex",
        ServiceTier::Scale => "scale",
        ServiceTier::Priority => "priority",
        ServiceTier::Ultrafast => "ultrafast",
    })
}

/// TS `setFileOffset` (conn-bound so the batch can write cursors inside its
/// single transaction).
pub(super) fn set_file_offset(
    conn: &Connection,
    session_file: &str,
    offset: i64,
    last_modified: f64,
    parser_state: Option<&SessionParserState>,
) -> Result<()> {
    let json = parser_state.map(serde_json::to_string).transpose()?;
    conn.execute(
        "INSERT OR REPLACE INTO file_offsets (session_file, offset, last_modified, parser_state)
         VALUES (?1, ?2, ?3, ?4)",
        rusqlite::params![session_file, offset, last_modified, json],
    )?;
    Ok(())
}

// ---- Stored-row snapshots for the reconcile/replay branches ----

pub(super) fn load_message_keys(
    conn: &Connection,
    session_file: &str,
) -> Result<Vec<(String, i64)>> {
    let mut stmt =
        conn.prepare("SELECT entry_id, timestamp FROM messages WHERE session_file = ?1")?;
    let rows = stmt.query_map(rusqlite::params![session_file], |row| {
        Ok((row.get(0)?, row.get(1)?))
    })?;
    Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
}

pub(super) fn load_user_keys(
    conn: &Connection,
    session_file: &str,
) -> Result<Vec<(String, i64, bool)>> {
    let mut stmt = conn.prepare(
        "SELECT entry_id, timestamp, model IS NOT NULL FROM user_messages WHERE session_file = ?1",
    )?;
    let rows = stmt.query_map(rusqlite::params![session_file], |row| {
        Ok((row.get(0)?, row.get(1)?, row.get::<_, i64>(2)? != 0))
    })?;
    Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
}

pub(super) fn load_tool_keys(
    conn: &Connection,
    session_file: &str,
) -> Result<Vec<(String, i64, String, bool)>> {
    let mut stmt = conn.prepare(
        "SELECT entry_id, timestamp, tool_call_id, result_chars IS NOT NULL FROM tool_calls WHERE session_file = ?1",
    )?;
    let rows = stmt.query_map(rusqlite::params![session_file], |row| {
        Ok((
            row.get(0)?,
            row.get(1)?,
            row.get(2)?,
            row.get::<_, i64>(3)? != 0,
        ))
    })?;
    Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
}

pub(super) fn delete_all_messages(
    conn: &Connection,
    session_file: &str,
    had_rows: bool,
) -> Result<i64> {
    if !had_rows {
        return Ok(0);
    }
    Ok(conn.execute(
        "DELETE FROM messages WHERE session_file = ?1",
        rusqlite::params![session_file],
    )? as i64)
}

pub(super) fn delete_all_users(
    conn: &Connection,
    session_file: &str,
    had_rows: bool,
) -> Result<i64> {
    if !had_rows {
        return Ok(0);
    }
    Ok(conn.execute(
        "DELETE FROM user_messages WHERE session_file = ?1",
        rusqlite::params![session_file],
    )? as i64)
}

pub(super) fn delete_all_tools(
    conn: &Connection,
    session_file: &str,
    had_rows: bool,
) -> Result<i64> {
    if !had_rows {
        return Ok(0);
    }
    Ok(conn.execute(
        "DELETE FROM tool_calls WHERE session_file = ?1",
        rusqlite::params![session_file],
    )? as i64)
}

pub(super) fn delete_message(conn: &Connection, session_file: &str, entry_id: &str) -> Result<i64> {
    Ok(conn.execute(
        "DELETE FROM messages WHERE session_file = ?1 AND entry_id = ?2",
        rusqlite::params![session_file, entry_id],
    )? as i64)
}

pub(super) fn delete_user(conn: &Connection, session_file: &str, entry_id: &str) -> Result<i64> {
    Ok(conn.execute(
        "DELETE FROM user_messages WHERE session_file = ?1 AND entry_id = ?2",
        rusqlite::params![session_file, entry_id],
    )? as i64)
}

pub(super) fn delete_tool(
    conn: &Connection,
    session_file: &str,
    tool_call_id: &str,
) -> Result<i64> {
    Ok(conn.execute(
        "DELETE FROM tool_calls WHERE session_file = ?1 AND tool_call_id = ?2",
        rusqlite::params![session_file, tool_call_id],
    )? as i64)
}

/// TS `insertMessageStats` statement: the `WHERE NOT EXISTS` clause skips
/// inserts whose `(entry_id, timestamp)` already exists under a different
/// `session_file` (fork copies); same-file re-syncs hit the upsert, which
/// re-derives cost and keeps `premium_requests` monotonic. Returns rows
/// inserted/updated.
pub(super) fn insert_message_stats_conn(
    conn: &Connection,
    stats: impl IntoIterator<Item = MessageStatsInput>,
) -> Result<i64> {
    let mut stmt = conn.prepare(
        r#"
		INSERT INTO messages (
			session_file, entry_id, folder, model, provider, api, timestamp,
			duration, ttft, stop_reason, error_message,
			input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, total_tokens, premium_requests,
			cost_input, cost_output, cost_cache_read, cost_cache_write, cost_total, cost_no_cache_input,
			cost_unpriced, agent_type, service_tier
		)
		SELECT ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21, ?22, ?23, ?24, ?25, ?26
		WHERE NOT EXISTS (
			SELECT 1 FROM messages
			WHERE entry_id = ?27 AND timestamp = ?28 AND session_file <> ?29
		)
		ON CONFLICT(session_file, entry_id) DO UPDATE SET
			premium_requests = MAX(messages.premium_requests, excluded.premium_requests),
			cost_input = excluded.cost_input,
			cost_output = excluded.cost_output,
			cost_cache_read = excluded.cost_cache_read,
			cost_cache_write = excluded.cost_cache_write,
			cost_total = excluded.cost_total,
			cost_no_cache_input = excluded.cost_no_cache_input,
			cost_unpriced = excluded.cost_unpriced,
			service_tier = excluded.service_tier
		"#,
    )?;
    let mut inserted = 0i64;
    for s in stats {
        let resolved = resolve_stored_cost(&s);
        let tokens = usage_from_input(&s.usage);
        let no_cache_input =
            calculate_no_cache_input_cost(&s.provider, &s.model, &tokens, s.timestamp)
                .unwrap_or(0.0);
        let changes = stmt.execute(rusqlite::params![
            s.session_file,
            s.entry_id,
            s.folder,
            s.model,
            s.provider,
            s.api,
            s.timestamp,
            s.duration,
            s.ttft,
            s.stop_reason.as_str(),
            s.error_message,
            s.usage.input,
            s.usage.output,
            s.usage.cache_read,
            s.usage.cache_write,
            s.usage.total_tokens,
            s.usage.premium_requests.unwrap_or(0.0),
            resolved.cost.input,
            resolved.cost.output,
            resolved.cost.cache_read,
            resolved.cost.cache_write,
            resolved.cost.total,
            no_cache_input,
            if resolved.unpriced { 1 } else { 0 },
            agent_type_str(s.agent_type),
            service_tier_str(s.service_tier),
            s.entry_id,
            s.timestamp,
            s.session_file,
        ])?;
        if changes > 0 {
            inserted += 1;
        }
    }
    Ok(inserted)
}
