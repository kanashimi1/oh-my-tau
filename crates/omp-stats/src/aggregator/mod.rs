//! Rust port of `aggregator.ts`: session sync into the stats db and the
//! dashboard read API (`get_*`), which serves rollup-backed range queries.
//!
//! The TS fans parsing out to `sync-worker.ts` Bun workers; here parsing runs
//! on scoped threads inside [`sync_all_sessions`], so there is no worker to
//! smoke-test (`smokeTestSyncWorker` is intentionally not ported).
//!
//! Subscription windows (`usage-windows.ts`) are served by
//! [`get_provider_window_stats`] from the broker when one is installed, else
//! the agent.db `usage_history` snapshots (see [`crate::usage_windows`]).

mod lock;
mod sync;
#[cfg(test)]
mod tests;

use crate::usage_windows;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::db::init_db;
use crate::db_query;
use crate::parser::get_session_entry;
pub use crate::rollup::get_time_range_config;
use crate::rollup::{
    get_cost_time_series, get_model_performance_series, get_model_time_series, get_overall_stats,
    get_provider_hourly_burn, get_provider_time_series, get_stats_by_agent_type,
    get_stats_by_folder, get_stats_by_model, get_stats_by_provider, get_time_series,
    get_tool_stats, get_tool_stats_by_model, get_tool_time_series,
};
use crate::types::{
    AgentTypeStats, AggregatedStats, CostTimeSeriesPoint, DashboardStats, FolderStats,
    MessageStats, ModelPerformancePoint, ModelStats, ModelTimeSeriesPoint, ProviderDashboardStats,
    ProviderWindowStats, RequestDetails, TimeSeriesPoint, ToolDashboardStats,
};

pub use lock::with_stats_sync_lock;
pub use sync::{sync_all_sessions, SyncOptions, SyncProgress, SyncResult};

/// Most folders the projects payload carries (busiest first); the tail is thousands of temp dirs.
const FOLDER_LIMIT: i64 = 2000;
/// TS `db.getRecentRequests` / `getRecentErrors` default `limit`.
const DEFAULT_RECENT_LIMIT: i64 = 100;

/// TS `Pick<DashboardStats, "overall" | "byAgentType" | "timeSeries">`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OverviewStats {
    pub overall: AggregatedStats,
    pub by_agent_type: Vec<AgentTypeStats>,
    pub time_series: Vec<TimeSeriesPoint>,
}

/// TS `Pick<DashboardStats, "byModel" | "modelSeries" | "modelPerformanceSeries">`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelDashboardStats {
    pub by_model: Vec<ModelStats>,
    pub model_series: Vec<ModelTimeSeriesPoint>,
    pub model_performance_series: Vec<ModelPerformancePoint>,
}

/// TS `Pick<DashboardStats, "costSeries">`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CostDashboardStats {
    pub cost_series: Vec<CostTimeSeriesPoint>,
}

/// Get all dashboard stats.
///
/// # Errors
/// DB failure.
pub fn get_dashboard_stats(range: Option<&str>) -> Result<DashboardStats> {
    init_db()?;
    let window = get_time_range_config(range);
    Ok(DashboardStats {
        overall: get_overall_stats(window.cutoff)?,
        by_model: get_stats_by_model(window.cutoff)?,
        by_folder: get_stats_by_folder(window.cutoff, Some(FOLDER_LIMIT))?,
        by_agent_type: get_stats_by_agent_type(window.cutoff)?,
        time_series: get_time_series(window)?,
        model_series: get_model_time_series(window)?,
        model_performance_series: get_model_performance_series(window)?,
        cost_series: get_cost_time_series(window.cutoff)?,
    })
}

/// Overview page payload.
///
/// # Errors
/// DB failure.
pub fn get_overview_stats(range: Option<&str>) -> Result<OverviewStats> {
    init_db()?;
    let window = get_time_range_config(range);
    Ok(OverviewStats {
        overall: get_overall_stats(window.cutoff)?,
        by_agent_type: get_stats_by_agent_type(window.cutoff)?,
        time_series: get_time_series(window)?,
    })
}

/// Models page payload.
///
/// # Errors
/// DB failure.
pub fn get_model_dashboard_stats(range: Option<&str>) -> Result<ModelDashboardStats> {
    init_db()?;
    let window = get_time_range_config(range);
    Ok(ModelDashboardStats {
        by_model: get_stats_by_model(window.cutoff)?,
        model_series: get_model_time_series(window)?,
        model_performance_series: get_model_performance_series(window)?,
    })
}

/// Costs page payload.
///
/// # Errors
/// DB failure.
pub fn get_cost_dashboard_stats(range: Option<&str>) -> Result<CostDashboardStats> {
    init_db()?;
    Ok(CostDashboardStats {
        cost_series: get_cost_time_series(get_time_range_config(range).cutoff)?,
    })
}

/// Projects page payload (busiest [`FOLDER_LIMIT`] folders).
///
/// # Errors
/// DB failure.
pub fn get_folder_stats(range: Option<&str>) -> Result<Vec<FolderStats>> {
    init_db()?;
    get_stats_by_folder(get_time_range_config(range).cutoff, Some(FOLDER_LIMIT))
}

/// Newest requests first (`limit` defaults to 100, like TS).
///
/// # Errors
/// DB failure.
pub fn get_recent_requests(limit: Option<i64>) -> Result<Vec<MessageStats>> {
    init_db()?;
    db_query::get_recent_requests(limit.unwrap_or(DEFAULT_RECENT_LIMIT))
}

/// Newest failed requests within `range` (`limit` defaults to 100).
///
/// # Errors
/// DB failure.
pub fn get_recent_errors(range: Option<&str>, limit: Option<i64>) -> Result<Vec<MessageStats>> {
    init_db()?;
    db_query::get_recent_errors(
        limit.unwrap_or(DEFAULT_RECENT_LIMIT),
        get_time_range_config(range).cutoff,
    )
}

/// One request with its journaled entry; `None` = TS `null` (unknown id, or
/// the entry is gone / not a message).
///
/// # Errors
/// DB failure or a non-ENOENT read error on the transcript.
pub fn get_request_details(id: i64) -> Result<Option<RequestDetails>> {
    init_db()?;
    let Some(msg) = db_query::get_message_by_id(id)? else {
        return Ok(None);
    };
    let Some(entry) = get_session_entry(&msg.session_file, &msg.entry_id)? else {
        return Ok(None);
    };
    let entry = serde_json::to_value(entry)?;
    // Role-model attempts (judge, auto-thinking, …) journal only a `model_usage`
    // record: usage and outcome, no request or response payload.
    match entry.get("type").and_then(serde_json::Value::as_str) {
        Some("model_usage") => Ok(Some(RequestDetails {
            base: msg,
            messages: vec![entry],
            output: serde_json::Value::Null,
        })),
        Some("message") => {
            let Some(output) = entry.get("message").cloned() else {
                return Ok(None);
            };
            Ok(Some(RequestDetails {
                base: msg,
                messages: vec![entry],
                output,
            }))
        }
        _ => Ok(None),
    }
}

/// Get the current message count in the database.
///
/// # Errors
/// DB failure.
pub fn get_total_message_count() -> Result<i64> {
    init_db()?;
    crate::db::get_message_count()
}

/// Tools page payload: per-tool totals, per-(tool, model) breakdown, and the
/// call time series.
///
/// # Errors
/// DB failure.
pub fn get_tool_dashboard_stats(range: Option<&str>) -> Result<ToolDashboardStats> {
    init_db()?;
    let window = get_time_range_config(range);
    Ok(ToolDashboardStats {
        by_tool: get_tool_stats(window.cutoff)?,
        by_tool_model: get_tool_stats_by_model(window.cutoff)?,
        series: get_tool_time_series(window)?,
    })
}

/// Providers page payload: per-provider totals, peak-burn-hours histogram and
/// provider token time series. Subscription windows are served separately by
/// [`get_provider_window_stats`].
///
/// # Errors
/// DB failure.
pub fn get_provider_dashboard_stats(range: Option<&str>) -> Result<ProviderDashboardStats> {
    init_db()?;
    let window = get_time_range_config(range);
    Ok(ProviderDashboardStats {
        providers: crate::rollup::get_stats_by_provider(window.cutoff)?,
        hourly: get_provider_hourly_burn(window.cutoff)?,
        series: get_provider_time_series(window)?,
    })
}

/// Subscription-window analytics (TS `getProviderWindowStats`): insights for
/// every provider window in range, plus utilization series for `provider` only
/// (all series can number in the thousands).
///
/// Token estimates use the broker's fleet tokens when a broker is installed
/// (the window fractions cover every install sharing the broker's
/// credentials, so dividing them into local-only tokens would undercount);
/// otherwise they come from local message stats.
///
/// # Errors
/// DB failure.
pub fn get_provider_window_stats(
    range: Option<&str>,
    provider: Option<&str>,
) -> Result<ProviderWindowStats> {
    init_db()?;
    let cutoff = get_time_range_config(range).cutoff;
    let usage = usage_windows::cached_usage_data(cutoff.unwrap_or(0));
    let tokens_by_provider = match usage.fleet_tokens_by_provider {
        Some(fleet) => fleet,
        None => get_stats_by_provider(cutoff)?
            .into_iter()
            .map(|p| (p.provider, p.total_tokens as f64))
            .collect(),
    };
    let stats = usage_windows::compute_usage_window_stats(&usage.rows, &tokens_by_provider);
    Ok(ProviderWindowStats {
        window_insights: stats.window_insights,
        usage_series: match provider {
            Some(provider) if !provider.is_empty() => stats
                .usage_series
                .into_iter()
                .filter(|series| series.provider == provider)
                .collect(),
            _ => Vec::new(),
        },
    })
}
