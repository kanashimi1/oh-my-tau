//! Conn-bound writers for `user_messages` and `tool_calls`, shared by the
//! `db_query` exports (own transaction) and `ingest`'s batch transaction
//! (which already holds the global connection, so it cannot call `with_db`).

use anyhow::Result;
use rusqlite::Connection;

use crate::db::schema::agent_type_str;
use crate::types::{ToolCallStats, ToolResultLink, UserMessageLink, UserMessageStats};

/// TS `insertUserMessageStats` (conn-bound; `db_query` wraps it in a transaction,
/// `ingest` runs it inside `apply_session_parse_results`). Idempotent via
/// UNIQUE + the fork-aware WHERE NOT EXISTS.
pub(crate) fn insert_user_message_stats_conn(
    conn: &Connection,
    stats: impl IntoIterator<Item = UserMessageStats>,
) -> Result<usize> {
    let mut stmt = conn.prepare(
        r#"
		INSERT OR IGNORE INTO user_messages (
			session_file, entry_id, folder, timestamp, model, provider,
			chars, words, yelling, profanity, anguish,
			negation, repetition, blame, prose, prose_hash
		)
		SELECT ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16
		WHERE NOT EXISTS (
			SELECT 1 FROM user_messages
			WHERE entry_id = ?17 AND timestamp = ?18 AND session_file <> ?19
		)
		"#,
    )?;
    let mut inserted = 0usize;
    for s in stats {
        let changes = stmt.execute(rusqlite::params![
            s.session_file,
            s.entry_id,
            s.folder,
            s.timestamp,
            s.model,
            s.provider,
            s.chars,
            s.words,
            s.yelling,
            s.profanity,
            s.anguish,
            s.negation,
            s.repetition,
            s.blame,
            s.prose,
            s.prose_hash,
            s.entry_id,
            s.timestamp,
            s.session_file,
        ])?;
        inserted += changes;
    }
    Ok(inserted)
}

/// TS `updateUserMessageLinks` (conn-bound):
/// backfill `model`/`provider` on rows persisted before their reply was
/// parsed; the `model IS NULL` guard makes each row update at most once.
pub(crate) fn update_user_message_links_conn(
    conn: &Connection,
    links: impl IntoIterator<Item = UserMessageLink>,
) -> Result<usize> {
    let mut stmt = conn.prepare(
        "UPDATE user_messages SET model = ?1, provider = ?2
         WHERE session_file = ?3 AND entry_id = ?4 AND model IS NULL",
    )?;
    let mut updated = 0usize;
    for link in links {
        updated += stmt.execute(rusqlite::params![
            link.model,
            link.provider,
            link.session_file,
            link.entry_id
        ])?;
    }
    Ok(updated)
}

/// TS `insertToolCalls` (conn-bound).
/// Idempotent via UNIQUE(session_file, tool_call_id) plus the fork-aware
/// WHERE NOT EXISTS.
pub(crate) fn insert_tool_calls_conn(
    conn: &Connection,
    calls: impl IntoIterator<Item = ToolCallStats>,
) -> Result<usize> {
    let mut stmt = conn.prepare(
        r#"
		INSERT OR IGNORE INTO tool_calls (
			session_file, entry_id, tool_call_id, folder, tool_name, model, provider,
			timestamp, agent_type, calls_in_turn, args_chars
		)
		SELECT ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11
		WHERE NOT EXISTS (
			SELECT 1 FROM tool_calls
			WHERE entry_id = ?12 AND timestamp = ?13 AND tool_call_id = ?14 AND session_file <> ?15
		)
		"#,
    )?;
    let mut inserted = 0usize;
    for call in calls {
        let changes = stmt.execute(rusqlite::params![
            call.session_file,
            call.entry_id,
            call.tool_call_id,
            call.folder,
            call.tool_name,
            call.model,
            call.provider,
            call.timestamp,
            agent_type_str(call.agent_type),
            call.calls_in_turn,
            call.args_chars,
            call.entry_id,
            call.timestamp,
            call.tool_call_id,
            call.session_file,
        ])?;
        inserted += changes;
    }
    Ok(inserted)
}

/// TS `updateToolResults` (conn-bound):
/// attach result size / error flag; results can arrive before their call, so
/// only rows already present are updated (0 changes otherwise).
pub(crate) fn update_tool_results_conn(
    conn: &Connection,
    links: impl IntoIterator<Item = ToolResultLink>,
) -> Result<usize> {
    let mut stmt = conn.prepare(
        "UPDATE tool_calls SET result_chars = ?1, is_error = ?2
         WHERE session_file = ?3 AND tool_call_id = ?4 AND result_chars IS NULL",
    )?;
    let mut updated = 0usize;
    for link in links {
        updated += stmt.execute(rusqlite::params![
            link.result_chars,
            if link.is_error { 1 } else { 0 },
            link.session_file,
            link.tool_call_id,
        ])?;
    }
    Ok(updated)
}
