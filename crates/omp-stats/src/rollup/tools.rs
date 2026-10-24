//! Tools page: per-tool usage, per (tool, model, provider) and per bucket.

use anyhow::Result;
use rusqlite::Row;

use super::source::{query_tools, series_bucket};
use super::{f64_col, i64_col, opt_text_col, text_col, RangeWindow, HOUR_MS};
use crate::types::{ToolModelStats, ToolTimeSeriesPoint, ToolUsageStats};

const TOOL_AGGREGATE_COLUMNS: &str = r#"
	SUM(f.calls) AS calls,
	SUM(f.errors) AS errors,
	SUM(f.args_chars) AS args_chars,
	SUM(f.result_chars) AS result_chars,
	TOTAL(f.total_tokens_share) AS total_tokens_share,
	TOTAL(f.output_tokens_share) AS output_tokens_share,
	TOTAL(f.cost_share) AS cost_share,
	TOTAL(f.unpriced_share) AS unpriced_share,
	MAX(f.last_used) AS last_used"#;

fn to_tool_usage(row: &Row) -> ToolUsageStats {
    ToolUsageStats {
        tool: text_col(row, "tool_name"),
        calls: i64_col(row, "calls"),
        errors: i64_col(row, "errors"),
        args_chars: i64_col(row, "args_chars"),
        result_chars: i64_col(row, "result_chars"),
        total_tokens_share: f64_col(row, "total_tokens_share"),
        output_tokens_share: f64_col(row, "output_tokens_share"),
        cost_share: f64_col(row, "cost_share"),
        unpriced_requests_share: f64_col(row, "unpriced_share"),
        last_used: i64_col(row, "last_used"),
    }
}

/// Tool usage per tool name, most-called first.
///
/// # Errors
/// DB failure.
pub fn get_tool_stats(cutoff: Option<i64>) -> Result<Vec<ToolUsageStats>> {
    query_tools(
        &RangeWindow::since(cutoff),
        &format!("f.tool_name AS tool_name, {TOOL_AGGREGATE_COLUMNS}"),
        "GROUP BY f.tool_name ORDER BY calls DESC, f.tool_name",
        |row| Ok(to_tool_usage(row)),
    )
}

/// Tool usage per (tool, model, provider), most-called first.
///
/// # Errors
/// DB failure.
pub fn get_tool_stats_by_model(cutoff: Option<i64>) -> Result<Vec<ToolModelStats>> {
    query_tools(
        &RangeWindow::since(cutoff),
        &format!(
            "f.tool_name AS tool_name, f.model AS model, f.provider AS provider, {TOOL_AGGREGATE_COLUMNS}"
        ),
        "GROUP BY f.tool_name, f.model, f.provider ORDER BY calls DESC, f.tool_name, f.model, f.provider",
        |row| {
            Ok(ToolModelStats {
                model: opt_text_col(row, "model").unwrap_or_default(),
                provider: opt_text_col(row, "provider").unwrap_or_default(),
                base: to_tool_usage(row),
            })
        },
    )
}

/// Calls and errors per bucket per tool.
///
/// # Errors
/// DB failure.
pub fn get_tool_time_series(window: RangeWindow) -> Result<Vec<ToolTimeSeriesPoint>> {
    let bucket = series_bucket(window.bucket_ms.unwrap_or(HOUR_MS));
    query_tools(
        &window,
        &format!(
            r#"{bucket} AS bucket, f.tool_name AS tool_name, SUM(f.calls) AS calls, SUM(f.errors) AS errors"#
        ),
        "GROUP BY 1, f.tool_name ORDER BY 1",
        |row| {
            Ok(ToolTimeSeriesPoint {
                timestamp: i64_col(row, "bucket"),
                tool: text_col(row, "tool_name"),
                calls: i64_col(row, "calls"),
                errors: i64_col(row, "errors"),
            })
        },
    )
}
