//! Stats page: overall/model/folder/agent-type/provider aggregates, the daily
//! activity shortcut and the per-transcript session rollups.

use anyhow::Result;
use rusqlite::Row;

use super::source::{dirty_is_small, query_messages};
use super::{
    f64_col, i64_col, opt_f64_col, opt_text_col, text_col, with_rollup_db, RangeWindow,
    SessionRollupRow, SESSION_ROLLUP_COLUMNS,
};
use crate::types::{
    AgentType, AgentTypeStats, AggregatedStats, DailyActivityPoint, FolderStats, ModelStats,
    ProviderAggregate, ProviderHourlyPoint,
};

const AGGREGATE_COLUMNS: &str = r#"
	SUM(f.requests) AS requests,
	SUM(f.failed) AS failed,
	SUM(f.input_tokens) AS input_tokens,
	SUM(f.output_tokens) AS output_tokens,
	SUM(f.cache_read_tokens) AS cache_read_tokens,
	SUM(f.cache_write_tokens) AS cache_write_tokens,
	TOTAL(f.premium_requests) AS premium_requests,
	TOTAL(f.cost_total) AS cost_total,
	SUM(f.unpriced) AS unpriced,
	TOTAL(f.cached_prompt_cost) AS cached_prompt_cost,
	TOTAL(f.no_cache_input_cost) AS no_cache_input_cost,
	TOTAL(f.duration_sum) / NULLIF(SUM(f.duration_n), 0) AS avg_duration,
	TOTAL(f.ttft_sum) / NULLIF(SUM(f.ttft_n), 0) AS avg_ttft,
	TOTAL(f.tps_sum) / NULLIF(SUM(f.tps_n), 0) AS avg_tps,
	MIN(f.first_ts) AS first_ts,
	MAX(f.last_ts) AS last_ts"#;

fn to_aggregated_stats(row: &Row) -> AggregatedStats {
    let total_requests = i64_col(row, "requests");
    let failed_requests = i64_col(row, "failed");
    let total_input_tokens = i64_col(row, "input_tokens");
    let total_cache_read_tokens = i64_col(row, "cache_read_tokens");
    let no_cache_input_cost = f64_col(row, "no_cache_input_cost");
    let cached_prompt_cost = f64_col(row, "cached_prompt_cost");
    AggregatedStats {
        total_requests,
        successful_requests: total_requests - failed_requests,
        failed_requests,
        error_rate: if total_requests > 0 {
            failed_requests as f64 / total_requests as f64
        } else {
            0.0
        },
        total_input_tokens,
        total_output_tokens: i64_col(row, "output_tokens"),
        total_cache_read_tokens,
        total_cache_write_tokens: i64_col(row, "cache_write_tokens"),
        cache_rate: if total_input_tokens + total_cache_read_tokens > 0 {
            total_cache_read_tokens as f64 / (total_input_tokens + total_cache_read_tokens) as f64
        } else {
            0.0
        },
        cache_savings: if no_cache_input_cost > 0.0 {
            (no_cache_input_cost - cached_prompt_cost) / no_cache_input_cost
        } else {
            0.0
        },
        total_cost: f64_col(row, "cost_total"),
        unpriced_requests: i64_col(row, "unpriced"),
        total_premium_requests: f64_col(row, "premium_requests"),
        avg_duration: opt_f64_col(row, "avg_duration"),
        avg_ttft: opt_f64_col(row, "avg_ttft"),
        avg_tokens_per_second: opt_f64_col(row, "avg_tps"),
        first_timestamp: i64_col(row, "first_ts"),
        last_timestamp: i64_col(row, "last_ts"),
    }
}

/// Overall request/token/cost aggregate since `cutoff` (all time when `None`).
///
/// # Errors
/// DB failure.
pub fn get_overall_stats(cutoff: Option<i64>) -> Result<AggregatedStats> {
    Ok(
        query_messages(&RangeWindow::since(cutoff), AGGREGATE_COLUMNS, "", |row| {
            Ok(to_aggregated_stats(row))
        })?
        .into_iter()
        .next()
        .unwrap_or_default(),
    )
}

/// Aggregates per (model, provider), busiest first.
///
/// # Errors
/// DB failure.
pub fn get_stats_by_model(cutoff: Option<i64>) -> Result<Vec<ModelStats>> {
    query_messages(
        &RangeWindow::since(cutoff),
        &format!("f.model AS model, f.provider AS provider, {AGGREGATE_COLUMNS}"),
        "GROUP BY f.model, f.provider ORDER BY requests DESC",
        |row| {
            Ok(ModelStats {
                model: text_col(row, "model"),
                provider: text_col(row, "provider"),
                base: to_aggregated_stats(row),
            })
        },
    )
}

/// Aggregates per project folder, busiest first; at most `limit` folders
/// (`None` = unlimited, TS `Number.MAX_SAFE_INTEGER`).
///
/// # Errors
/// DB failure.
pub fn get_stats_by_folder(cutoff: Option<i64>, limit: Option<i64>) -> Result<Vec<FolderStats>> {
    query_messages(
        &RangeWindow::since(cutoff),
        &format!("f.folder AS folder, {AGGREGATE_COLUMNS}"),
        &format!(
            "GROUP BY f.folder ORDER BY requests DESC LIMIT {}",
            limit.unwrap_or(i64::MAX).max(0)
        ),
        |row| {
            Ok(FolderStats {
                folder: text_col(row, "folder"),
                base: to_aggregated_stats(row),
            })
        },
    )
}

/// Token usage per agent type (main, subagent, advisor).
///
/// # Errors
/// DB failure.
pub fn get_stats_by_agent_type(cutoff: Option<i64>) -> Result<Vec<AgentTypeStats>> {
    query_messages(
        &RangeWindow::since(cutoff),
        r#"f.agent_type AS agent_type, SUM(f.requests) AS requests, SUM(f.input_tokens) AS input_tokens,
		 SUM(f.output_tokens) AS output_tokens, SUM(f.cache_read_tokens) AS cache_read_tokens,
		 SUM(f.cache_write_tokens) AS cache_write_tokens, TOTAL(f.cost_total) AS cost_total"#,
        "GROUP BY f.agent_type",
        |row| {
            Ok(AgentTypeStats {
                agent_type: agent_type_from(&text_col(row, "agent_type")),
                total_requests: i64_col(row, "requests"),
                total_input_tokens: i64_col(row, "input_tokens"),
                total_output_tokens: i64_col(row, "output_tokens"),
                total_cache_read_tokens: i64_col(row, "cache_read_tokens"),
                total_cache_write_tokens: i64_col(row, "cache_write_tokens"),
                total_cost: f64_col(row, "cost_total"),
            })
        },
    )
}

fn agent_type_from(value: &str) -> AgentType {
    match value {
        "subagent" => AgentType::Subagent,
        "advisor" => AgentType::Advisor,
        _ => AgentType::Main,
    }
}

/// Request/token/cost totals per provider, biggest token burn first.
///
/// # Errors
/// DB failure.
pub fn get_stats_by_provider(cutoff: Option<i64>) -> Result<Vec<ProviderAggregate>> {
    let mut rows = query_messages(
        &RangeWindow::since(cutoff),
        r#"f.provider AS provider, SUM(f.requests) AS requests, SUM(f.failed) AS failed,
		 COUNT(DISTINCT f.model) AS models, SUM(f.input_tokens) AS input_tokens, SUM(f.output_tokens) AS output_tokens,
		 SUM(f.cache_read_tokens) AS cache_read_tokens, SUM(f.cache_write_tokens) AS cache_write_tokens,
		 TOTAL(f.cost_total) AS cost_total, SUM(f.unpriced) AS unpriced, TOTAL(f.premium_requests) AS premium_requests,
		 TOTAL(f.tps_sum) / NULLIF(SUM(f.tps_n), 0) AS avg_tps"#,
        "GROUP BY f.provider",
        |row| {
            let input = i64_col(row, "input_tokens");
            let output = i64_col(row, "output_tokens");
            let cache_read = i64_col(row, "cache_read_tokens");
            let cache_write = i64_col(row, "cache_write_tokens");
            Ok(ProviderAggregate {
                provider: text_col(row, "provider"),
                total_requests: i64_col(row, "requests"),
                failed_requests: i64_col(row, "failed"),
                models: i64_col(row, "models"),
                total_input_tokens: input,
                total_output_tokens: output,
                total_cache_read_tokens: cache_read,
                total_cache_write_tokens: cache_write,
                total_tokens: input + output + cache_read + cache_write,
                total_cost: f64_col(row, "cost_total"),
                unpriced_requests: i64_col(row, "unpriced"),
                total_premium_requests: f64_col(row, "premium_requests"),
                avg_tokens_per_second: opt_f64_col(row, "avg_tps"),
            })
        },
    )?;
    rows.sort_by_key(|r| std::cmp::Reverse(r.total_tokens));
    Ok(rows)
}

/// Token burn per provider per local hour of day (0-23). Hours use the
/// server's timezone — a localhost tool, so server and viewer coincide. Rolled
/// hours are attributed by their start, so half-hour UTC offsets shift burn by
/// up to 30 minutes.
///
/// # Errors
/// DB failure.
pub fn get_provider_hourly_burn(cutoff: Option<i64>) -> Result<Vec<ProviderHourlyPoint>> {
    query_messages(
        &RangeWindow::since(cutoff),
        r#"f.provider AS provider, CAST(strftime('%H', f.bucket / 1000, 'unixepoch', 'localtime') AS INTEGER) AS hour,
		 SUM(f.input_tokens + f.output_tokens + f.cache_read_tokens + f.cache_write_tokens) AS total_tokens,
		 SUM(f.output_tokens) AS output_tokens, SUM(f.requests) AS requests"#,
        "GROUP BY f.provider, hour ORDER BY f.provider, hour",
        |row| {
            Ok(ProviderHourlyPoint {
                provider: text_col(row, "provider"),
                hour: i64_col(row, "hour"),
                total_tokens: i64_col(row, "total_tokens"),
                output_tokens: i64_col(row, "output_tokens"),
                requests: i64_col(row, "requests"),
            })
        },
    )
}

/// Per-local-day request/token/cost totals since `cutoff`, oldest first, read
/// from the hourly rollup. Returns `None` when the local UTC offset is not a
/// whole hour (now or at `cutoff`), since hourly buckets then straddle local
/// midnight, or while the dirty backlog is too large for exact reads; callers
/// then aggregate the raw rows instead.
///
/// # Errors
/// DB failure.
pub fn get_daily_activity_from_rollup(cutoff: i64) -> Result<Option<Vec<DailyActivityPoint>>> {
    let now_offset = local_offset_minutes(None)?;
    let cutoff_offset = local_offset_minutes(Some(cutoff))?;
    if now_offset % 60 != 0 || cutoff_offset % 60 != 0 {
        return Ok(None);
    }
    if !with_rollup_db(dirty_is_small)? {
        return Ok(None);
    }
    let points = query_messages(
        &RangeWindow::since(Some(cutoff)),
        r#"date(f.bucket / 1000, 'unixepoch', 'localtime') AS day, TOTAL(f.cost_total) AS cost,
		 SUM(f.requests) AS requests, SUM(f.total_tokens) AS total_tokens"#,
        "GROUP BY day ORDER BY day",
        |row| {
            Ok(DailyActivityPoint {
                day: text_col(row, "day"),
                cost: f64_col(row, "cost"),
                requests: i64_col(row, "requests"),
                total_tokens: i64_col(row, "total_tokens"),
            })
        },
    )?;
    Ok(Some(points))
}

/// Local UTC offset in minutes, JS `getTimezoneOffset()` sign (minutes to add
/// to local time to reach UTC). `None` = now. SQLite's `localtime` modifier and
/// the C library behind it are the same source the TS `Date` uses.
pub(crate) fn local_offset_minutes(at_ms: Option<i64>) -> Result<i64> {
    let sql = match at_ms {
        None => concat!(
            "SELECT CAST(strftime('%s', 'now', 'localtime') AS INTEGER)",
            " - CAST(strftime('%s', 'now') AS INTEGER)"
        )
        .to_string(),
        Some(ms) => format!(
            r#"SELECT CAST(strftime('%s', {ms} / 1000, 'localtime') AS INTEGER) - CAST(strftime('%s', {ms} / 1000) AS INTEGER)"#
        ),
    };
    with_rollup_db(|conn| {
        let seconds: i64 = conn.query_row(&sql, [], |row| row.get(0))?;
        Ok(-seconds / 60)
    })
}

/// One row per synced transcript file (subagents unfolded). Same exactness
/// rules as the range queries: dirty transcripts are aggregated raw while few.
///
/// # Errors
/// DB failure.
pub fn get_session_rollups() -> Result<Vec<SessionRollupRow>> {
    with_rollup_db(|conn| {
        let n: i64 = conn.query_row(
            "SELECT COUNT(*) AS n FROM (SELECT 1 FROM session_dirty LIMIT ?)",
            [super::EXACT_DIRTY_SESSIONS + 1],
            |row| row.get(0),
        )?;
        let exact = n <= super::EXACT_DIRTY_SESSIONS;
        let mut parts = vec![format!(
            "SELECT {SESSION_ROLLUP_COLUMNS} FROM session_rollup {clean}",
            clean = if exact {
                "WHERE session_file NOT IN (SELECT session_file FROM session_dirty)"
            } else {
                ""
            }
        )];
        if exact {
            parts.push(format!(
                "SELECT {facts} FROM session_dirty d JOIN messages m ON m.session_file = d.session_file GROUP BY m.session_file",
                facts = super::session_facts("m.")
            ));
        }
        let sql = format!(
            r#"SELECT session_file, requests, started_at, ended_at, total_tokens, cost_total,
			 unpriced, models, tool_calls FROM ({})"#,
            parts.join(" UNION ALL ")
        );
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map([], |row| {
            Ok(SessionRollupRow {
                session_file: text_col(row, "session_file"),
                requests: i64_col(row, "requests"),
                started_at: i64_col(row, "started_at"),
                ended_at: i64_col(row, "ended_at"),
                total_tokens: i64_col(row, "total_tokens"),
                cost_total: f64_col(row, "cost_total"),
                unpriced_requests: i64_col(row, "unpriced"),
                models: opt_text_col(row, "models"),
                tool_calls: i64_col(row, "tool_calls"),
            })
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    })
}
