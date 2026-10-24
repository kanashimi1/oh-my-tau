//! Read queries and the remaining write helpers of `db.ts`, outside `initDb`
//! and the ingest path. Every function goes through [`crate::db::with_db`]
//! and copies the TS SQL text. Cutoffs are `Option<i64>` ms; TS `null`/`undefined`
//! map to `None`. Inserts return the number of changed rows.
//!
//! - `messages`: request/error/by-id reads, daily activity, user-message writes.
//! - `frustration`: frustration tallies, pending prose, verdict cache.
//! - `tool_calls`: tool-call inserts and result updates.

mod frustration;
mod messages;
#[cfg(test)]
mod tests;
mod tool_calls;

pub use frustration::{
    get_frustration_by_model, get_frustration_overall, get_pending_frustration_prose,
    get_pending_frustration_totals, upsert_frustration_verdicts, FrustrationModelRow,
    FrustrationVerdict, PendingProse, PendingProseTotals,
};
pub use messages::{
    get_daily_activity, get_message_by_id, get_recent_errors, get_recent_requests,
    insert_user_message_stats, update_user_message_links,
};
pub use tool_calls::{insert_tool_calls, update_tool_results};
