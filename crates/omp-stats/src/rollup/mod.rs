//! Rust port of `rollup.ts`: hourly rollups of `messages` and `tool_calls`,
//! and every range query the dashboard runs over them.
//!
//! The raw tables hold millions of rows; grouping them per request made every
//! range switch cost seconds. Instead, `message_rollup` / `tool_rollup` keep one
//! row per (hour, dimensions) with additive sums, so a 90-day query reads a few
//! thousand rows.
//!
//! Freshness is trigger-driven: any INSERT/UPDATE/DELETE on the raw tables — from
//! this process, a sync worker, or another omp process sharing the database —
//! records the touched hour in `rollup_dirty`. [`refresh_rollups`] recomputes
//! dirty hours newest-first in short immediate transactions.
//!
//! Reads never wait for a refresh and are exact: each query unions
//! - clean rollup hours inside the range,
//! - raw rows for the partial hour at the range start, and
//! - raw rows for dirty hours (while few are dirty),
//!
//! so live ingest shows up before its hour is re-rolled. While the initial build
//! (or a mass re-ingest) leaves many hours dirty, reads use the rollup rows as
//! they stand (stale-but-present; not-yet-built hours are missing) and
//! [`get_rollup_status`] reports the backlog so the UI can show progress.
//!
//! The 1h range needs 5-minute buckets, so it aggregates raw rows directly
//! (an hour of data is small and indexed by timestamp).
//!
//! `db::with_db` only creates the raw schema; the rollup tables and the
//! dirty-marking triggers are owned here ([`ensure_rollup_schema`]), because
//! cargo features are acyclic: feature `db` may not depend on `rollup`. Every
//! entry point below therefore ensures the schema first.

mod refresh;
mod schema;
mod series;
mod source;
mod stats;
#[cfg(test)]
mod tests;
mod tools;

use anyhow::Result;
use rusqlite::{Connection, Row};
use serde::{Deserialize, Serialize};

pub use refresh::{
    get_rollup_status, refresh_rollup_batch, refresh_rollups, refresh_rollups_with, RefreshOptions,
};
pub use schema::ensure_rollup_schema;
pub use series::{
    get_cost_time_series, get_model_performance_series, get_model_time_series,
    get_provider_time_series, get_time_series,
};
pub use stats::{
    get_daily_activity_from_rollup, get_overall_stats, get_provider_hourly_burn,
    get_session_rollups, get_stats_by_agent_type, get_stats_by_folder, get_stats_by_model,
    get_stats_by_provider,
};
pub use tools::{get_tool_stats, get_tool_stats_by_model, get_tool_time_series};

pub(crate) const HOUR_MS: i64 = 60 * 60 * 1000;
pub(crate) const DAY_MS: i64 = 24 * HOUR_MS;

/// Bump when the rollup schema or fact definitions change: the tables are then
/// dropped and every hour is rebuilt from the raw tables.
const ROLLUP_VERSION: &str = "2";
const ROLLUP_VERSION_KEY: &str = "rollup_version";
/// Bump when the dirty-marking triggers change; they are then recreated in place.
const TRIGGER_VERSION: &str = "3";
const TRIGGER_VERSION_KEY: &str = "rollup_triggers_version";

/// Above this many dirty hours, reads use stale rollup rows instead of scanning
/// dirty hours raw.
pub(crate) const EXACT_DIRTY_LIMIT: i64 = 96;
/// Target wall time of one refresh transaction; batch size adapts to hit it.
const REFRESH_TARGET_MS: u128 = 40;
const REFRESH_MAX_BATCH: usize = 64;
/// Dirty transcripts rolled per dirty hour in one refresh batch.
pub(crate) const SESSIONS_PER_HOUR: i64 = 16;
/// Above this many dirty transcripts, session reads use stale rows instead of
/// scanning raw.
pub(crate) const EXACT_DIRTY_SESSIONS: i64 = 512;

pub(crate) const MESSAGE_DIMENSIONS: &str = "model, provider, folder, agent_type";

pub(crate) const MESSAGE_ROLLUP_COLUMNS: &str = r#"bucket, model, provider, folder, agent_type, requests, failed, input_tokens, output_tokens,
	cache_read_tokens, cache_write_tokens, total_tokens, premium_requests, cost_total, cost_input, cost_output,
	cost_cache_read, cost_cache_write, unpriced, cached_prompt_cost, no_cache_input_cost, duration_sum, duration_n,
	ttft_sum, ttft_n, tps_sum, tps_n, first_ts, last_ts"#;

pub(crate) const TOOL_ROLLUP_COLUMNS: &str = r#"bucket, tool_name, model, provider, calls, errors, args_chars, result_chars,
	total_tokens_share, output_tokens_share, cost_share, unpriced_share, last_used"#;

pub(crate) const SESSION_ROLLUP_COLUMNS: &str =
    "session_file, requests, started_at, ended_at, total_tokens, cost_total, unpriced, models, tool_calls";

pub(crate) const TOOL_JOIN: &str =
    "LEFT JOIN messages m ON m.session_file = t.session_file AND m.entry_id = t.entry_id";

/// A range query's time window. `cutoff` `None` = all time; `bucket_ms`
/// `None` = not a time series (hour granularity, rollup-backed).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RangeWindow {
    pub cutoff: Option<i64>,
    /// Bucket size for time series (5 minutes, an hour, or a day).
    pub bucket_ms: Option<i64>,
}

impl RangeWindow {
    /// All time.
    #[must_use]
    pub fn all() -> Self {
        Self {
            cutoff: None,
            bucket_ms: None,
        }
    }

    /// A plain range query over `[cutoff, now]` (hour granularity).
    #[must_use]
    pub fn since(cutoff: Option<i64>) -> Self {
        Self {
            cutoff,
            bucket_ms: None,
        }
    }

    /// A time-series query over `[cutoff, now]` bucketed by `bucket_ms`.
    #[must_use]
    pub fn series(cutoff: Option<i64>, bucket_ms: i64) -> Self {
        Self {
            cutoff,
            bucket_ms: Some(bucket_ms),
        }
    }

    /// Sub-hour buckets aggregate raw rows in 5-minute buckets.
    pub(crate) fn fine(&self) -> bool {
        self.bucket_ms.is_some_and(|ms| ms < HOUR_MS)
    }
}

/// TS aggregator `getTimeRangeConfig`: resolve a `?range=` value (default
/// `24h`, unknown values too) to its cutoff and series bucket size. Lives here
/// beside [`RangeWindow`] so gain/frustration need not depend on `aggregator`.
#[must_use]
pub fn get_time_range_config(range: Option<&str>) -> RangeWindow {
    let (span_ms, bucket_ms) = match range.map(|r| r.trim().to_lowercase()).as_deref() {
        Some("1h") => (Some(HOUR_MS), 5 * 60 * 1000),
        Some("7d") => (Some(7 * DAY_MS), DAY_MS),
        Some("30d") => (Some(30 * DAY_MS), DAY_MS),
        Some("90d") => (Some(90 * DAY_MS), DAY_MS),
        Some("all") => (None, DAY_MS),
        _ => (Some(DAY_MS), HOUR_MS),
    };
    RangeWindow::series(span_ms.map(|span| now_ms() - span), bucket_ms)
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX))
}

/// Backlog of hours and transcripts whose rollups are stale.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RollupStatus {
    pub dirty_hours: i64,
    pub dirty_sessions: i64,
}

/// Per-transcript-file rollup for the Traces session list.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionRollupRow {
    pub session_file: String,
    pub requests: i64,
    pub started_at: i64,
    pub ended_at: i64,
    pub total_tokens: i64,
    pub cost_total: f64,
    pub unpriced_requests: i64,
    /// Comma-joined DISTINCT models.
    pub models: Option<String>,
    pub tool_calls: i64,
}

/// Additive per-bucket facts over `messages` (columns qualified by `p`).
pub(crate) fn message_facts(bucket_expr: &str, p: &str) -> String {
    format!(
        r#"
		{bucket_expr} AS bucket,
		{p}model AS model, {p}provider AS provider, {p}folder AS folder, {p}agent_type AS agent_type,
		COUNT(*) AS requests,
		SUM(CASE WHEN {p}stop_reason = 'error' THEN 1 ELSE 0 END) AS failed,
		SUM({p}input_tokens) AS input_tokens,
		SUM({p}output_tokens) AS output_tokens,
		SUM({p}cache_read_tokens) AS cache_read_tokens,
		SUM({p}cache_write_tokens) AS cache_write_tokens,
		SUM({p}total_tokens) AS total_tokens,
		TOTAL({p}premium_requests) AS premium_requests,
		TOTAL({p}cost_total) AS cost_total,
		TOTAL({p}cost_input) AS cost_input,
		TOTAL({p}cost_output) AS cost_output,
		TOTAL({p}cost_cache_read) AS cost_cache_read,
		TOTAL({p}cost_cache_write) AS cost_cache_write,
		SUM({unpriced}) AS unpriced,
		TOTAL(CASE WHEN {p}cost_no_cache_input > 0
			THEN {p}cost_input + {p}cost_cache_read + {p}cost_cache_write ELSE 0 END) AS cached_prompt_cost,
		TOTAL({p}cost_no_cache_input) AS no_cache_input_cost,
		TOTAL({p}duration) AS duration_sum,
		COUNT({p}duration) AS duration_n,
		TOTAL({p}ttft) AS ttft_sum,
		COUNT({p}ttft) AS ttft_n,
		TOTAL(CASE WHEN {p}duration > 0 THEN {p}output_tokens * 1000.0 / {p}duration END) AS tps_sum,
		COUNT(CASE WHEN {p}duration > 0 THEN 1 END) AS tps_n,
		MIN({p}timestamp) AS first_ts,
		MAX({p}timestamp) AS last_ts"#,
        unpriced = crate::catalog::unpriced_request_sql(p)
    )
}

/// Additive per-bucket facts over `tool_calls t LEFT JOIN messages m`. Provider
/// usage comes from the invoking assistant turn divided by `calls_in_turn`, so
/// per-tool token/cost shares stay additive across tools.
pub(crate) fn tool_facts(bucket_expr: &str) -> String {
    format!(
        r#"
		{bucket_expr} AS bucket,
		t.tool_name AS tool_name, t.model AS model, t.provider AS provider,
		COUNT(*) AS calls,
		SUM(CASE WHEN t.is_error = 1 THEN 1 ELSE 0 END) AS errors,
		SUM(t.args_chars) AS args_chars,
		SUM(COALESCE(t.result_chars, 0)) AS result_chars,
		TOTAL(COALESCE(m.total_tokens, 0) * 1.0 / t.calls_in_turn) AS total_tokens_share,
		TOTAL(COALESCE(m.output_tokens, 0) * 1.0 / t.calls_in_turn) AS output_tokens_share,
		TOTAL(COALESCE(m.cost_total, 0) / t.calls_in_turn) AS cost_share,
		TOTAL({unpriced} * 1.0 / t.calls_in_turn) AS unpriced_share,
		MAX(t.timestamp) AS last_used"#,
        unpriced = crate::catalog::unpriced_request_sql("m.")
    )
}

/// Per-transcript facts over `messages` (qualified by `p`); tool calls counted
/// by subquery.
pub(crate) fn session_facts(p: &str) -> String {
    format!(
        r#"
		{p}session_file AS session_file,
		COUNT(*) AS requests,
		MIN({p}timestamp) AS started_at,
		MAX({p}timestamp + COALESCE({p}duration, 0)) AS ended_at,
		SUM({p}total_tokens) AS total_tokens,
		TOTAL({p}cost_total) AS cost_total,
		SUM({unpriced}) AS unpriced,
		GROUP_CONCAT(DISTINCT {p}model) AS models,
		(SELECT COUNT(*) FROM tool_calls tc WHERE tc.session_file = {p}session_file) AS tool_calls"#,
        unpriced = crate::catalog::unpriced_request_sql(p)
    )
}

pub(crate) fn hour_of(column: &str) -> String {
    format!("((({column}) / {HOUR_MS}) * {HOUR_MS})")
}

/// TS `Math.ceil(cutoff / HOUR_MS) * HOUR_MS`: the hour at or after `cutoff`.
pub(crate) fn hour_ceil(cutoff: i64) -> i64 {
    if cutoff.rem_euclid(HOUR_MS) == 0 {
        cutoff
    } else {
        cutoff.div_euclid(HOUR_MS) * HOUR_MS + HOUR_MS
    }
}

/// `crate::db::with_db` with the rollup schema ensured first.
pub(crate) fn with_rollup_db<T>(f: impl FnOnce(&Connection) -> Result<T>) -> Result<T> {
    crate::db::with_db(|conn| {
        schema::ensure_rollup_schema(conn)?;
        f(conn)
    })
}

/// TS `BEGIN IMMEDIATE` on the shared `&Connection`; rolls back on drop.
pub(crate) struct ImmediateTx<'c> {
    conn: &'c Connection,
    committed: bool,
}

impl<'c> ImmediateTx<'c> {
    /// # Errors
    /// SQL failure (e.g. a transaction already open on the connection).
    pub(crate) fn begin(conn: &'c Connection) -> Result<Self> {
        conn.execute_batch("BEGIN IMMEDIATE")?;
        Ok(Self {
            conn,
            committed: false,
        })
    }

    /// # Errors
    /// SQL failure on COMMIT.
    pub(crate) fn commit(mut self) -> Result<()> {
        self.conn.execute_batch("COMMIT")?;
        self.committed = true;
        Ok(())
    }
}

impl Drop for ImmediateTx<'_> {
    fn drop(&mut self) {
        if !self.committed {
            let _ = self.conn.execute_batch("ROLLBACK");
        }
    }
}

// --- row helpers: SQLite NULL maps to the TS `?? 0` / `?? null` defaults ---

pub(crate) fn i64_col(row: &Row, name: &str) -> i64 {
    row.get::<_, Option<i64>>(name).ok().flatten().unwrap_or(0)
}

pub(crate) fn f64_col(row: &Row, name: &str) -> f64 {
    row.get::<_, Option<f64>>(name)
        .ok()
        .flatten()
        .unwrap_or(0.0)
}

pub(crate) fn opt_f64_col(row: &Row, name: &str) -> Option<f64> {
    row.get::<_, Option<f64>>(name).ok().flatten()
}

pub(crate) fn text_col(row: &Row, name: &str) -> String {
    row.get::<_, Option<String>>(name)
        .ok()
        .flatten()
        .unwrap_or_default()
}

pub(crate) fn opt_text_col(row: &Row, name: &str) -> Option<String> {
    row.get::<_, Option<String>>(name).ok().flatten()
}
