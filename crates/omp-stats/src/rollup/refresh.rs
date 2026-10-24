//! `getRollupStatus` / `refreshRollupBatch` / `refreshRollups`.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use anyhow::Result;
use rusqlite::params;

use super::{
    message_facts, session_facts, tool_facts, with_rollup_db, ImmediateTx, RollupStatus, HOUR_MS,
    MESSAGE_DIMENSIONS, MESSAGE_ROLLUP_COLUMNS, REFRESH_MAX_BATCH, REFRESH_TARGET_MS,
    SESSIONS_PER_HOUR, SESSION_ROLLUP_COLUMNS, TOOL_JOIN, TOOL_ROLLUP_COLUMNS,
};

/// Backlog of hours and transcripts whose rollups are stale.
///
/// # Errors
/// DB failure.
pub fn get_rollup_status() -> Result<RollupStatus> {
    with_rollup_db(|conn| {
        Ok(conn.query_row(
            "SELECT (SELECT COUNT(*) FROM rollup_dirty) AS dirtyHours,
                (SELECT COUNT(*) FROM session_dirty) AS dirtySessions",
            [],
            |row| {
                Ok(RollupStatus {
                    dirty_hours: row.get(0)?,
                    dirty_sessions: row.get(1)?,
                })
            },
        )?)
    })
}

/// Recompute up to `limit` dirty hours, newest first, in one immediate
/// transaction (so no writer can dirty an hour between its read and its
/// dirty-mark removal). Returns the number of hours and transcripts refreshed.
///
/// # Errors
/// DB failure.
pub fn refresh_rollup_batch(limit: i64) -> Result<i64> {
    with_rollup_db(|conn| {
        let tx = ImmediateTx::begin(conn)?;

        let buckets = {
            let mut stmt =
                conn.prepare("SELECT bucket FROM rollup_dirty ORDER BY bucket DESC LIMIT ?")?;
            let rows = stmt.query_map([limit], |row| row.get::<_, i64>(0))?;
            let mut buckets = Vec::new();
            for bucket in rows {
                buckets.push(bucket?);
            }
            buckets
        };

        let fill_messages = format!(
            "INSERT INTO message_rollup ({MESSAGE_ROLLUP_COLUMNS})
SELECT {facts} FROM messages WHERE timestamp >= ?1 AND timestamp < ?1 + {HOUR_MS}
GROUP BY {MESSAGE_DIMENSIONS}",
            facts = message_facts("?1", "")
        );
        let fill_tools = format!(
            "INSERT INTO tool_rollup ({TOOL_ROLLUP_COLUMNS})
SELECT {facts} FROM tool_calls t {TOOL_JOIN}
WHERE t.timestamp >= ?1 AND t.timestamp < ?1 + {HOUR_MS}
GROUP BY t.tool_name, t.model, t.provider",
            facts = tool_facts("?1")
        );
        let mut clear_messages = conn.prepare("DELETE FROM message_rollup WHERE bucket = ?")?;
        let mut clear_tools = conn.prepare("DELETE FROM tool_rollup WHERE bucket = ?")?;
        let mut insert_messages = conn.prepare(&fill_messages)?;
        let mut insert_tools = conn.prepare(&fill_tools)?;
        let mut clean = conn.prepare("DELETE FROM rollup_dirty WHERE bucket = ?")?;
        for &bucket in &buckets {
            clear_messages.execute([bucket])?;
            clear_tools.execute([bucket])?;
            insert_messages.execute(params![bucket])?;
            insert_tools.execute(params![bucket])?;
            clean.execute([bucket])?;
        }

        // Transcripts are cheap (indexed by session_file); roll many per hour-batch.
        let sessions = {
            let mut stmt = conn.prepare("SELECT session_file FROM session_dirty LIMIT ?")?;
            let rows =
                stmt.query_map([limit * SESSIONS_PER_HOUR], |row| row.get::<_, String>(0))?;
            let mut sessions = Vec::new();
            for session in rows {
                sessions.push(session?);
            }
            sessions
        };
        let fill_session = format!(
            "INSERT INTO session_rollup ({SESSION_ROLLUP_COLUMNS})
SELECT {facts} FROM messages m WHERE m.session_file = ?1 GROUP BY m.session_file",
            facts = session_facts("m.")
        );
        let mut clear_session =
            conn.prepare("DELETE FROM session_rollup WHERE session_file = ?")?;
        let mut insert_session = conn.prepare(&fill_session)?;
        let mut clean_session = conn.prepare("DELETE FROM session_dirty WHERE session_file = ?")?;
        for session_file in &sessions {
            clear_session.execute([session_file])?;
            insert_session.execute(params![session_file])?;
            clean_session.execute([session_file])?;
        }

        tx.commit()?;
        Ok(i64::try_from(buckets.len() + sessions.len()).unwrap_or(i64::MAX))
    })
}

/// TS `RefreshOptions`.
#[derive(Default, Clone, Copy)]
pub struct RefreshOptions<'a> {
    /// Called after each committed batch with the remaining backlog.
    pub on_progress: Option<&'a (dyn Fn(i64) + Sync)>,
    /// Stop early (between batches).
    pub abort: Option<&'a AtomicBool>,
}

/// Drain the dirty-hour backlog until it is empty.
///
/// # Errors
/// DB failure.
pub fn refresh_rollups() -> Result<()> {
    refresh_rollups_with(&RefreshOptions::default())
}

/// Drain the dirty-hour backlog in small transactions, yielding between
/// batches so a host process (TUI, dashboard server) stays responsive during
/// the initial build. Batch size adapts so each transaction takes about
/// `REFRESH_TARGET_MS`; busy hours shrink it to one.
///
/// # Errors
/// DB failure.
pub fn refresh_rollups_with(opts: &RefreshOptions<'_>) -> Result<()> {
    let mut batch: usize = 4;
    loop {
        if opts
            .abort
            .is_some_and(|signal| signal.load(Ordering::Relaxed))
        {
            return Ok(());
        }
        let started = Instant::now();
        let done = refresh_rollup_batch(batch as i64)?;
        if done == 0 {
            return Ok(());
        }
        let elapsed = started.elapsed().as_millis();
        if elapsed < REFRESH_TARGET_MS / 2 {
            batch = (batch * 2).min(REFRESH_MAX_BATCH);
        } else if elapsed > REFRESH_TARGET_MS * 2 {
            batch = ((batch as f64 / 2.0).floor() as usize).max(1);
        }
        if let Some(on_progress) = opts.on_progress {
            on_progress(get_rollup_status()?.dirty_hours);
        }
        std::thread::yield_now();
    }
}
