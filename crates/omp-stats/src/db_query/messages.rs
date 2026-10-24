//! Request/error/activity reads over `messages` and the user-message writes
//! (port of `db.ts` `rowToMessageStats` .. `getDailyActivity`, plus
//! `insertUserMessageStats` / `updateUserMessageLinks`).

use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use rusqlite::{params, params_from_iter, OptionalExtension, Row};
use serde_json::Value;

use crate::db::{rows, with_db};
use crate::types::{
    AgentType, DailyActivityPoint, MessageStats, StopReason, Usage, UsageCost, UserMessageLink,
    UserMessageStats,
};

/// TS `rowToMessageStats`. Column names are read by name, not position.
fn row_to_message_stats(row: &Row<'_>) -> rusqlite::Result<MessageStats> {
    let stop_reason: String = row.get("stop_reason")?;
    let agent_type: Option<String> = row.get("agent_type")?;
    let service_tier: Option<String> = row.get("service_tier")?;
    let cost_unpriced: Option<i64> = row.get("cost_unpriced")?;
    // TS `row.premium_requests ?? 0`; REAL because Copilot multipliers are fractional.
    let premium: Option<f64> = row.get("premium_requests")?;
    Ok(MessageStats {
        id: Some(row.get("id")?),
        session_file: row.get("session_file")?,
        entry_id: row.get("entry_id")?,
        folder: row.get("folder")?,
        model: row.get("model")?,
        provider: row.get("provider")?,
        api: row.get("api")?,
        timestamp: row.get("timestamp")?,
        duration: row.get("duration")?,
        ttft: row.get("ttft")?,
        stop_reason: StopReason::from_wire(&stop_reason),
        error_message: row.get("error_message")?,
        usage: Usage {
            input: row.get("input_tokens")?,
            output: row.get("output_tokens")?,
            cache_read: row.get("cache_read_tokens")?,
            cache_write: row.get("cache_write_tokens")?,
            total_tokens: row.get("total_tokens")?,
            premium_requests: Some(premium.unwrap_or(0.0)),
            reasoning_tokens: None,
            cost: UsageCost {
                input: row.get("cost_input")?,
                output: row.get("cost_output")?,
                cache_read: row.get("cost_cache_read")?,
                cache_write: row.get("cost_cache_write")?,
                total: row.get("cost_total")?,
            },
        },
        agent_type: agent_type
            .and_then(|s| serde_json::from_value::<AgentType>(Value::String(s)).ok())
            .unwrap_or_default(),
        service_tier: service_tier.map(Value::String),
        cost_unpriced: Some(cost_unpriced == Some(1)),
    })
}

/// TS `getRecentRequests(limit)`: newest requests first.
///
/// # Errors
/// Db open or query failure.
pub fn get_recent_requests(limit: i64) -> Result<Vec<MessageStats>> {
    with_db(|conn| {
        let mut stmt = conn.prepare(
            "SELECT * FROM messages \
             ORDER BY timestamp DESC \
             LIMIT ?",
        )?;
        let rows = stmt.query_map(params![limit], row_to_message_stats)?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    })
}

/// TS `getRecentErrors(limit, cutoff)`: newest `stop_reason = 'error'` rows,
/// optionally only those at or after `cutoff` (ms).
///
/// # Errors
/// Db open or query failure.
pub fn get_recent_errors(limit: i64, cutoff: Option<i64>) -> Result<Vec<MessageStats>> {
    with_db(|conn| {
        let has_cutoff = cutoff.is_some();
        let sql = format!(
            "SELECT * FROM messages \
             WHERE stop_reason = 'error' \
             {} \
             ORDER BY timestamp DESC \
             LIMIT ?",
            if has_cutoff { "AND timestamp >= ?" } else { "" }
        );
        let mut stmt = conn.prepare(&sql)?;
        let binds: Vec<i64> = cutoff.into_iter().chain(std::iter::once(limit)).collect();
        let rows = stmt.query_map(params_from_iter(binds.iter()), row_to_message_stats)?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    })
}

/// TS `getMessageById(id)`.
///
/// # Errors
/// Db open or query failure.
pub fn get_message_by_id(id: i64) -> Result<Option<MessageStats>> {
    with_db(|conn| {
        let mut stmt = conn.prepare("SELECT * FROM messages WHERE id = ?")?;
        Ok(stmt
            .query_row(params![id], row_to_message_stats)
            .optional()?)
    })
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|d| i64::try_from(d.as_millis()).ok())
        .unwrap_or(0)
}

/// TS `getDailyActivity(days)`: per-local-day aggregates, oldest first.
/// Reads the hourly rollup first when the `rollup` feature is compiled in
/// (TS `getDailyActivityFromRollup`), falling back to raw `messages` rows.
///
/// # Errors
/// Db open or query failure.
pub fn get_daily_activity(days: i64) -> Result<Vec<DailyActivityPoint>> {
    let cutoff = now_ms() - days * 24 * 60 * 60 * 1000;
    // Called outside `with_db`: the rollup reader takes the same connection lock.
    #[cfg(feature = "rollup")]
    if let Some(rolled) = crate::rollup::get_daily_activity_from_rollup(cutoff)? {
        return Ok(rolled);
    }
    with_db(|conn| {
        let mut stmt = conn.prepare(
            "SELECT
                date(timestamp / 1000, 'unixepoch', 'localtime') as day,
                SUM(cost_total) as cost,
                COUNT(*) as requests,
                SUM(total_tokens) as total_tokens
             FROM messages
             WHERE timestamp >= ?
             GROUP BY day
             ORDER BY day ASC",
        )?;
        let rows = stmt.query_map(params![cutoff], |row| {
            let cost: Option<f64> = row.get("cost")?;
            let total: Option<i64> = row.get("total_tokens")?;
            Ok(DailyActivityPoint {
                day: row.get("day")?,
                cost: cost.unwrap_or(0.0),
                requests: row.get("requests")?,
                total_tokens: total.unwrap_or(0),
            })
        })?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    })
}

/// TS `insertUserMessageStats`: one transaction, `INSERT OR IGNORE` plus a
/// guard that skips rows another session_file already owns. Returns rows inserted.
///
/// # Errors
/// Db open, transaction or statement failure.
pub fn insert_user_message_stats(stats: impl IntoIterator<Item = UserMessageStats>) -> Result<i64> {
    with_db(|conn| {
        let tx = conn.unchecked_transaction()?;
        let inserted = rows::insert_user_message_stats_conn(&tx, stats)?;
        tx.commit()?;
        Ok(inserted as i64)
    })
}

/// TS `updateUserMessageLinks`: fill the responding model/provider on rows
/// still missing it (`model IS NULL` guard makes each row update once).
/// Returns rows updated.
///
/// # Errors
/// Db open, transaction or statement failure.
pub fn update_user_message_links(links: impl IntoIterator<Item = UserMessageLink>) -> Result<i64> {
    with_db(|conn| {
        let tx = conn.unchecked_transaction()?;
        let updated = rows::update_user_message_links_conn(&tx, links)?;
        tx.commit()?;
        Ok(updated as i64)
    })
}
