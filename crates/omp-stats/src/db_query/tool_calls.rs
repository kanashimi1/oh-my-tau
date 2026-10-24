//! Tool-call row writes (port of `insertToolCalls` / `updateToolResults`).
//! The statements live in `db::rows`, shared with the ingest batch.

use anyhow::Result;

use crate::db::{rows, with_db};
use crate::types::{ToolCallStats, ToolResultLink};

/// TS `insertToolCalls`. Idempotent via `UNIQUE(session_file, tool_call_id)`.
/// The `WHERE NOT EXISTS` guard mirrors `insertUserMessageStats`: forked
/// sessions deep-copy assistant entries, so first-write-wins across the
/// lineage keeps aggregates from double counting. Returns rows inserted.
///
/// # Errors
/// Db open, transaction or statement failure.
pub fn insert_tool_calls(calls: impl IntoIterator<Item = ToolCallStats>) -> Result<i64> {
    with_db(|conn| {
        let tx = conn.unchecked_transaction()?;
        let inserted = rows::insert_tool_calls_conn(&tx, calls)?;
        tx.commit()?;
        Ok(inserted as i64)
    })
}

/// TS `updateToolResults`: attach result size and error flag to persisted
/// call rows. Results can land in a later sync pass than their call, so this
/// is an UPDATE keyed by (session_file, tool_call_id); the `IS NULL` guard
/// keeps re-syncs idempotent. Returns rows updated.
///
/// # Errors
/// Db open, transaction or statement failure.
pub fn update_tool_results(links: impl IntoIterator<Item = ToolResultLink>) -> Result<i64> {
    with_db(|conn| {
        let tx = conn.unchecked_transaction()?;
        let updated = rows::update_tool_results_conn(&tx, links)?;
        tx.commit()?;
        Ok(updated as i64)
    })
}
