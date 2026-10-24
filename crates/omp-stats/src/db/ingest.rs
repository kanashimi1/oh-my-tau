//! Port of the db.ts write path: file offsets, `applySessionParseResults`,
//! `insertMessageStats`, and the sync bookkeeping around them. Everything goes
//! through `crate::db::with_db` (the shared process-global connection).
//!
//! The row-writing statements live in `ingest_write.rs` so both the standalone
//! public exports and the batch transaction in
//! [`apply_session_parse_results`] share one implementation.

#[path = "ingest_write.rs"]
mod write;

use std::collections::HashMap;

use anyhow::{Context as _, Result};
use serde::{Deserialize, Serialize};

use crate::parser::{ParseSessionResult, SessionParserState};
use crate::types::MessageStatsInput;

use super::with_db;

/// TS `FileOffset`: persisted transcript identity and parser cursor.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FileOffset {
    pub offset: i64,
    /// Node `stat.mtimeMs` (fractional): TS stores the float in the INTEGER-affinity
    /// column, so fractional values come back as REAL; whole ones as INTEGER.
    pub last_modified: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parser_state: Option<SessionParserState>,
}

/// TS `decodeFileOffset`: a stored cursor is only trusted when it parses and
/// its embedded offset matches the row (a missing cursor is reconstructed
/// from the transcript).
fn decode_file_offset(offset: i64, last_modified: f64, parser_state: Option<String>) -> FileOffset {
    let parser_state = parser_state.and_then(|json| {
        let state: SessionParserState = serde_json::from_str(&json).ok()?;
        (state.version == 1 && state.offset == offset).then_some(state)
    });
    FileOffset {
        offset,
        last_modified,
        parser_state,
    }
}

/// TS `getFileOffset`: one persisted cursor for an individual transcript.
///
/// # Errors
/// DB failure.
pub fn get_file_offset(session_file: &str) -> Result<Option<FileOffset>> {
    with_db(|conn| {
        let mut stmt = conn.prepare(
            "SELECT session_file, offset, last_modified, parser_state FROM file_offsets WHERE session_file = ?1",
        )?;
        let mut rows = stmt.query(rusqlite::params![session_file])?;
        if let Some(row) = rows.next()? {
            return Ok(Some(decode_file_offset(
                row.get(1)?,
                row.get(2)?,
                row.get(3)?,
            )));
        }
        Ok(None)
    })
}

/// TS `getFileOffsets`: a bounded sync work set's cursors in one snapshot.
///
/// # Errors
/// DB failure.
pub fn get_file_offsets(session_files: &[String]) -> Result<HashMap<String, FileOffset>> {
    let mut out = HashMap::new();
    if session_files.is_empty() {
        return Ok(out);
    }
    with_db(|conn| {
        let placeholders = session_files
            .iter()
            .map(|_| "?")
            .collect::<Vec<_>>()
            .join(",");
        let sql = format!(
            "SELECT session_file, offset, last_modified, parser_state FROM file_offsets WHERE session_file IN ({placeholders})"
        );
        let mut stmt = conn.prepare(&sql)?;
        let values: Vec<&dyn rusqlite::ToSql> = session_files
            .iter()
            .map(|f| f as &dyn rusqlite::ToSql)
            .collect();
        let mut rows = stmt.query(&values[..])?;
        while let Some(row) = rows.next()? {
            let file: String = row.get(0)?;
            let offset = decode_file_offset(row.get(1)?, row.get(2)?, row.get(3)?);
            out.insert(file, offset);
        }
        Ok(out)
    })
}

/// TS `setFileOffset`: update the stored offset for a session file.
///
/// # Errors
/// DB failure.
pub fn set_file_offset(
    session_file: &str,
    offset: i64,
    last_modified: f64,
    parser_state: Option<&SessionParserState>,
) -> Result<()> {
    with_db(|conn| write::set_file_offset(conn, session_file, offset, last_modified, parser_state))
}

/// TS `ParsedSession`: a parsed transcript queued by the sync loop for an
/// atomic database batch.
#[derive(Debug, Clone)]
pub struct ParsedSession {
    pub session_file: String,
    pub result: ParseSessionResult,
    pub rebuild: bool,
    /// Recover missing rows from a full-history scan without rewriting unchanged records.
    pub replay: bool,
}

/// TS return shape of `applySessionParseResults`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ApplySessionOutcome {
    pub processed: i64,
    pub files: i64,
    pub reconcile: bool,
    pub changes: i64,
}

/// TS `applySessionParseResults`: commit distinct transcripts' rows,
/// reconciliation state, and cursors together; failure rolls back the batch.
/// `changes` counts stored rows inserted, updated or deleted (cursor
/// bookkeeping excluded).
///
/// # Errors
/// DB failure (the batch rolls back).
pub fn apply_session_parse_results(sessions: &[ParsedSession]) -> Result<ApplySessionOutcome> {
    with_db(|conn| {
        let tx = conn.unchecked_transaction()?;
        let outcome = apply_in_transaction(&tx, sessions).context("apply session parse results")?;
        tx.commit()?;
        Ok(outcome)
    })
}

fn apply_in_transaction(
    conn: &rusqlite::Connection,
    sessions: &[ParsedSession],
) -> Result<ApplySessionOutcome> {
    let mut processed: i64 = 0;
    let mut files: i64 = 0;
    let mut reconcile = false;
    let mut changes: i64 = 0;
    let mut writes: Vec<ParseSessionResult> = Vec::new();

    for session in sessions {
        let ParsedSession {
            session_file,
            result,
            rebuild,
            replay,
        } = session;
        if result.parser_state.is_none() {
            continue;
        }
        let mut rows = result.clone();
        if result.reset.unwrap_or(false) || *rebuild || *replay {
            let stored_messages = write::load_message_keys(conn, session_file)?;
            let stored_users = write::load_user_keys(conn, session_file)?;
            let stored_tools = write::load_tool_keys(conn, session_file)?;
            let replace = result.reset.unwrap_or(false) || *rebuild;
            if replace {
                // Only removed owners require another full replay, not an
                // identity-only file replacement.
                if !reconcile
                    && (!stored_messages.is_empty()
                        || !stored_users.is_empty()
                        || !stored_tools.is_empty())
                {
                    let retained_messages: std::collections::HashSet<String> = result
                        .stats
                        .iter()
                        .map(|r| format!("[{:?},{}]", r.entry_id, r.timestamp))
                        .collect();
                    let retained_users: std::collections::HashSet<String> = result
                        .user_stats
                        .iter()
                        .map(|r| format!("[{:?},{}]", r.entry_id, r.timestamp))
                        .collect();
                    let retained_tools: std::collections::HashSet<String> = result
                        .tool_calls
                        .iter()
                        .map(|r| format!("[{:?},{},{:?}]", r.entry_id, r.timestamp, r.tool_call_id))
                        .collect();
                    let removed_messages = stored_messages
                        .iter()
                        .any(|(id, ts)| !retained_messages.contains(&format!("[{:?},{}]", id, ts)));
                    let removed_users = stored_users
                        .iter()
                        .any(|(id, ts, _)| !retained_users.contains(&format!("[{:?},{}]", id, ts)));
                    let removed_tools = stored_tools.iter().any(|(id, ts, tc, _)| {
                        !retained_tools.contains(&format!("[{:?},{},{:?}]", id, ts, tc))
                    });
                    reconcile = removed_messages || removed_users || removed_tools;
                }
                changes +=
                    write::delete_all_messages(conn, session_file, !stored_messages.is_empty())?;
                changes += write::delete_all_users(conn, session_file, !stored_users.is_empty())?;
                changes += write::delete_all_tools(conn, session_file, !stored_tools.is_empty())?;
            } else {
                // A reconciliation replay only needs missing rows and
                // unfinished links; repair stale owners without invalidating
                // IDs of records still present in the transcript.
                let message_ts: HashMap<&str, i64> = stored_messages
                    .iter()
                    .map(|(id, ts)| (id.as_str(), *ts))
                    .collect();
                let user_by_id: HashMap<&str, (i64, bool)> = stored_users
                    .iter()
                    .map(|(id, ts, has_model)| (id.as_str(), (*ts, *has_model)))
                    .collect();
                let tool_by_id: HashMap<&str, (String, i64, bool)> = stored_tools
                    .iter()
                    .map(|(entry_id, ts, tc, has_result)| {
                        (tc.as_str(), (entry_id.clone(), *ts, *has_result))
                    })
                    .collect();
                let mut absent_messages: std::collections::HashSet<&str> =
                    message_ts.keys().copied().collect();
                let mut absent_users: std::collections::HashSet<&str> =
                    user_by_id.keys().copied().collect();
                let mut absent_tools: std::collections::HashSet<&str> =
                    tool_by_id.keys().copied().collect();
                rows.stats.retain(|row| {
                    if message_ts.get(row.entry_id.as_str()) != Some(&row.timestamp) {
                        return true;
                    }
                    absent_messages.remove(row.entry_id.as_str());
                    false
                });
                rows.user_stats.retain(|row| {
                    if user_by_id.get(row.entry_id.as_str()).map(|(ts, _)| *ts)
                        != Some(row.timestamp)
                    {
                        return true;
                    }
                    absent_users.remove(row.entry_id.as_str());
                    false
                });
                rows.tool_calls
                    .retain(|row| match tool_by_id.get(row.tool_call_id.as_str()) {
                        Some((entry_id, ts, _))
                            if *entry_id == row.entry_id && *ts == row.timestamp =>
                        {
                            absent_tools.remove(row.tool_call_id.as_str());
                            false
                        }
                        _ => true,
                    });
                // A reused ID needs fresh linkage after its old identity is removed.
                rows.user_links.retain(|row| {
                    absent_users.contains(row.entry_id.as_str())
                        || !user_by_id
                            .get(row.entry_id.as_str())
                            .map(|(_, m)| *m)
                            .unwrap_or(false)
                });
                rows.tool_results.retain(|row| {
                    absent_tools.contains(row.tool_call_id.as_str())
                        || !tool_by_id
                            .get(row.tool_call_id.as_str())
                            .map(|(_, _, r)| *r)
                            .unwrap_or(false)
                });
                if !absent_messages.is_empty()
                    || !absent_users.is_empty()
                    || !absent_tools.is_empty()
                {
                    reconcile = true;
                    for id in &absent_messages {
                        changes += write::delete_message(conn, session_file, id)?;
                    }
                    for id in &absent_users {
                        changes += write::delete_user(conn, session_file, id)?;
                    }
                    for id in &absent_tools {
                        changes += write::delete_tool(conn, session_file, id)?;
                    }
                }
            }
        }
        let count = (rows.stats.len() + rows.user_stats.len()) as i64;
        processed += count;
        if count > 0 {
            files += 1;
        }
        writes.push(rows);
    }

    // Remove replaced rows before choosing new fork owners, then write each
    // table once per batch.
    changes += write::insert_message_stats_conn(
        conn,
        writes.iter().flat_map(|w| w.stats.iter().cloned()),
    )?;
    changes += super::rows::insert_user_message_stats_conn(
        conn,
        writes.iter().flat_map(|w| w.user_stats.iter().cloned()),
    )? as i64;
    changes += super::rows::update_user_message_links_conn(
        conn,
        writes.iter().flat_map(|w| w.user_links.iter().cloned()),
    )? as i64;
    changes += super::rows::insert_tool_calls_conn(
        conn,
        writes.iter().flat_map(|w| w.tool_calls.iter().cloned()),
    )? as i64;
    changes += super::rows::update_tool_results_conn(
        conn,
        writes.iter().flat_map(|w| w.tool_results.iter().cloned()),
    )? as i64;

    for session in sessions {
        if let Some(state) = &session.result.parser_state {
            write::set_file_offset(
                conn,
                &session.session_file,
                session.result.new_offset,
                state.mtime_ms,
                Some(state),
            )?;
        }
    }
    if reconcile {
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('session_reconciliation', 'pending')",
            [],
        )?;
    }
    Ok(ApplySessionOutcome {
        processed,
        files,
        reconcile,
        changes,
    })
}

/// TS `getDataVersion`: `PRAGMA data_version` moves only when another
/// connection (another omp process) commits.
///
/// # Errors
/// DB failure.
pub fn get_data_version() -> Result<Option<i64>> {
    with_db(|conn| {
        let mut stmt = conn.prepare("PRAGMA data_version")?;
        let mut rows = stmt.query([])?;
        if let Some(row) = rows.next()? {
            return Ok(Some(row.get(0)?));
        }
        Ok(None)
    })
}

/// TS `prepareSessionSync`: whether a pending `session_reconciliation` needs a
/// full replay pass before the sync.
///
/// # Errors
/// DB failure.
pub fn prepare_session_sync() -> Result<bool> {
    with_db(|conn| {
        let mut stmt = conn.prepare("SELECT 1 FROM meta WHERE key = 'session_reconciliation'")?;
        let mut rows = stmt.query([])?;
        Ok(rows.next()?.is_some())
    })
}

/// TS `completeSessionSync`: clear the reconciliation sentinel when the sync
/// did not find (or already handled) removals.
///
/// # Errors
/// DB failure.
pub fn complete_session_sync(reconcile: bool) -> Result<()> {
    if reconcile {
        return Ok(());
    }
    with_db(|conn| {
        conn.execute("DELETE FROM meta WHERE key = 'session_reconciliation'", [])?;
        Ok(())
    })
}

/// TS `insertMessageStats`: fork-aware insert/upsert of request rows, priced
/// through the bundled catalog. Returns the number of rows inserted/updated.
///
/// # Errors
/// DB failure.
pub fn insert_message_stats(stats: impl IntoIterator<Item = MessageStatsInput>) -> Result<i64> {
    // One transaction like TS `db.transaction()`: all rows or none.
    with_db(|conn| {
        let tx = conn.unchecked_transaction()?;
        let changed = write::insert_message_stats_conn(&tx, stats)?;
        tx.commit()?;
        Ok(changed)
    })
}

/// TS `getMessageCount`.
///
/// # Errors
/// DB failure.
pub fn get_message_count() -> Result<i64> {
    with_db(|conn| {
        let count = conn.query_row("SELECT COUNT(*) FROM messages", [], |row| row.get(0))?;
        Ok(count)
    })
}

/// TS `markSessionBackfillsComplete`: settle every full-session backfill
/// sentinel after a successful sync pass.
///
/// # Errors
/// DB failure.
pub fn mark_session_backfills_complete() -> Result<()> {
    const KEYS: &[&str] = &[
        super::USER_MESSAGES_BACKFILL_KEY,
        super::TOOL_CALLS_BACKFILL_KEY,
        super::USER_MESSAGE_LINKS_REPAIR_KEY,
        super::PRIORITY_PREMIUM_REQUESTS_BACKFILL_KEY,
        super::COST_REINGEST_BACKFILL_KEY,
        super::COST_UNPRICED_BACKFILL_KEY,
    ];
    with_db(|conn| {
        let tx = conn.unchecked_transaction()?;
        {
            let mut mark =
                tx.prepare("INSERT OR REPLACE INTO meta (key, value) VALUES (?1, 'complete')")?;
            for key in KEYS {
                mark.execute(rusqlite::params![key])?;
            }
        }
        tx.commit()?;
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::TestHome;
    use crate::types::{UsageInput, UserMessageStats};

    fn parser_state(offset: i64) -> SessionParserState {
        SessionParserState {
            version: 1,
            offset,
            dev: 0.0,
            ino: 0.0,
            birthtime_ms: 0.0,
            size: offset,
            mtime_ms: 1_700_000_000_000.0,
            checkpoint: "abc123".to_string(),
            service_tier: None,
        }
    }

    fn message(entry_id: &str, timestamp: i64) -> MessageStatsInput {
        MessageStatsInput {
            id: None,
            session_file: "proj/session.jsonl".to_string(),
            entry_id: entry_id.to_string(),
            folder: "proj".to_string(),
            model: "test-model".to_string(),
            provider: "test-provider".to_string(),
            api: "test".to_string(),
            timestamp,
            duration: Some(12.0),
            ttft: Some(3.0),
            stop_reason: crate::types::StopReason::Stop,
            error_message: None,
            usage: UsageInput {
                input: 100,
                output: 50,
                cache_read: 10,
                cache_write: 5,
                total_tokens: 165,
                premium_requests: Some(1.0),
                cost: None,
                ..UsageInput::default()
            },
            agent_type: crate::types::AgentType::Main,
            service_tier: None,
        }
    }

    fn user_message(entry_id: &str, timestamp: i64) -> UserMessageStats {
        UserMessageStats {
            id: None,
            session_file: "proj/session.jsonl".to_string(),
            entry_id: entry_id.to_string(),
            folder: "proj".to_string(),
            timestamp,
            model: None,
            provider: None,
            chars: 42,
            words: 8,
            yelling: 0,
            profanity: 0,
            anguish: 0,
            negation: 0,
            repetition: 0,
            blame: 0,
            prose: String::new(),
            prose_hash: String::new(),
        }
    }

    fn parsed() -> ParsedSession {
        ParsedSession {
            session_file: "proj/session.jsonl".to_string(),
            result: ParseSessionResult {
                stats: vec![message("e1", 1000), message("e2", 2000)],
                user_stats: vec![user_message("u1", 900)],
                user_links: vec![],
                tool_calls: vec![],
                tool_results: vec![],
                new_offset: 4096,
                parser_state: Some(parser_state(4096)),
                reset: None,
            },
            rebuild: false,
            replay: false,
        }
    }

    #[test]
    fn apply_inserts_rows_and_second_apply_is_noop() -> Result<()> {
        let _home = TestHome::new();
        let outcome = apply_session_parse_results(&[parsed()])?;
        assert_eq!(outcome.processed, 3);
        assert_eq!(outcome.files, 1);
        assert!(!outcome.reconcile);
        assert_eq!(outcome.changes, 3, "2 messages + 1 user message inserted");

        assert_eq!(get_message_count()?, 2);

        // Cursor bookkeeping landed outside `changes`.
        let offset = get_file_offset("proj/session.jsonl")?.expect("offset persisted");
        assert_eq!(offset.offset, 4096);
        assert_eq!(offset.last_modified, 1_700_000_000_000.0);
        assert_eq!(offset.parser_state.as_ref().map(|s| s.offset), Some(4096));

        // Identical batch via the reconciliation-replay path: stored rows
        // match, so nothing is rewritten and the batch reports 0 changes.
        // (A bare upsert of identical values still counts as a change in
        // SQLite — verified against bun:sqlite — so the no-op guarantee comes
        // from the replay filter, exactly as in TS.)
        let mut replayed = parsed();
        replayed.replay = true;
        let again = apply_session_parse_results(&[replayed])?;
        assert_eq!(again.changes, 0);
        assert_eq!(again.processed, 0);
        assert_eq!(get_message_count()?, 2);

        mark_session_backfills_complete()?;
        let pending = prepare_session_sync()?;
        assert!(!pending);
        Ok(())
    }

    #[test]
    fn rebuild_replaces_rows_and_sets_reconcile_on_removal() -> Result<()> {
        let _home = TestHome::new();
        apply_session_parse_results(&[parsed()])?;

        // Same session re-parsed without e2: the replace removes it, which
        // requires another full replay (reconcile = true).
        let mut rebuilt = parsed();
        rebuilt.rebuild = true;
        rebuilt.result.stats.retain(|m| m.entry_id != "e2");
        let outcome = apply_session_parse_results(&[rebuilt])?;
        assert!(
            outcome.reconcile,
            "removed owner row must trigger reconcile"
        );
        assert_eq!(get_message_count()?, 1);
        assert!(prepare_session_sync()?);
        // TS completeSessionSync: reconcile=true keeps the sentinel.
        complete_session_sync(outcome.reconcile)?;
        assert!(prepare_session_sync()?);
        complete_session_sync(false)?;
        assert!(!prepare_session_sync()?);
        Ok(())
    }

    /// End-to-end: parse all fixture sessions and ingest them into a temp
    /// HOME; row counts must equal the TS-built golden DB
    /// (messages 900, tool_calls 1113, user_messages 41, file_offsets 26).
    /// Skips (not fails) when the fixture is absent.
    #[test]
    fn fixture_ingest_matches_golden_counts() -> Result<()> {
        const FIXTURE_SESSIONS: &str =
            "/home/user/omrs-fixtures/stats/fixture/home/.omp/agent/sessions";
        if !std::path::Path::new(FIXTURE_SESSIONS).is_dir() {
            eprintln!("fixture sessions not found; skipping");
            return Ok(());
        }
        let _home = crate::testutil::TestHome::new();
        let agent = _home.path().join(".omp/agent");
        std::fs::create_dir_all(&agent)?;
        std::os::unix::fs::symlink(FIXTURE_SESSIONS, agent.join("sessions"))?;

        let files = crate::parser::list_all_session_files();
        assert_eq!(files.len(), 26, "fixture file count");
        let mut batch = Vec::new();
        for file in &files {
            let result = crate::parser::parse_session_file(file, 0, None, false)?;
            batch.push(ParsedSession {
                session_file: file.clone(),
                result,
                rebuild: false,
                replay: false,
            });
        }
        let outcome = apply_session_parse_results(&batch)?;
        assert!(outcome.changes > 0);

        let messages = get_message_count()?;
        let tool_calls: i64 = super::with_db(|c| -> Result<i64> {
            c.query_row("SELECT COUNT(*) FROM tool_calls", [], |r| r.get(0))
                .map_err(Into::into)
        })?;
        let user_messages: i64 = super::with_db(|c| -> Result<i64> {
            c.query_row("SELECT COUNT(*) FROM user_messages", [], |r| r.get(0))
                .map_err(Into::into)
        })?;
        let offsets: i64 = super::with_db(|c| -> Result<i64> {
            c.query_row("SELECT COUNT(*) FROM file_offsets", [], |r| r.get(0))
                .map_err(Into::into)
        })?;
        assert_eq!(messages, 900, "messages vs golden/_stats.full.db");
        assert_eq!(tool_calls, 1113, "tool_calls vs golden/_stats.full.db");
        assert_eq!(user_messages, 41, "user_messages vs golden/_stats.full.db");
        assert_eq!(offsets, 26, "file_offsets vs golden/_stats.full.db");

        // Second pass resumes from the persisted cursors like the real sync:
        // nothing new to read, so the batch is a no-op.
        let cursors = get_file_offsets(&files)?;
        assert_eq!(cursors.len(), 26);
        let mut second = Vec::new();
        for file in &files {
            let offset = cursors.get(file).expect("cursor");
            let result = crate::parser::parse_session_file(
                file,
                offset.offset,
                offset.parser_state.as_ref(),
                false,
            )?;
            assert!(
                result.stats.is_empty() && result.tool_calls.is_empty(),
                "tail is drained"
            );
            second.push(ParsedSession {
                session_file: file.clone(),
                result,
                rebuild: false,
                replay: false,
            });
        }
        let again = apply_session_parse_results(&second)?;
        assert_eq!(get_message_count()?, 900);
        assert_eq!(again.changes, 0, "resume of unchanged files is a no-op");
        Ok(())
    }
}
