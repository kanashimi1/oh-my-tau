//! Series endpoints: request/token/cost over time, per model and per provider.

use anyhow::Result;

use super::source::{query_messages, series_bucket};
use super::{f64_col, i64_col, opt_f64_col, text_col, RangeWindow, DAY_MS, HOUR_MS};
use crate::types::{
    CostTimeSeriesPoint, ModelPerformancePoint, ModelTimeSeriesPoint, ProviderTimeSeriesPoint,
    TimeSeriesPoint,
};

fn bucket_ms(window: &RangeWindow) -> i64 {
    window.bucket_ms.unwrap_or(HOUR_MS)
}

/// Requests, errors, tokens and cost per bucket.
///
/// # Errors
/// DB failure.
pub fn get_time_series(window: RangeWindow) -> Result<Vec<TimeSeriesPoint>> {
    let bucket = series_bucket(bucket_ms(&window));
    query_messages(
        &window,
        &format!(
            r#"{bucket} AS bucket, SUM(f.requests) AS requests, SUM(f.failed) AS errors,
		 SUM(f.total_tokens) AS tokens, TOTAL(f.cost_total) AS cost"#
        ),
        "GROUP BY 1 ORDER BY 1",
        |row| {
            Ok(TimeSeriesPoint {
                timestamp: i64_col(row, "bucket"),
                requests: i64_col(row, "requests"),
                errors: i64_col(row, "errors"),
                tokens: i64_col(row, "tokens"),
                cost: f64_col(row, "cost"),
            })
        },
    )
}

/// Requests per bucket per (model, provider).
///
/// # Errors
/// DB failure.
pub fn get_model_time_series(window: RangeWindow) -> Result<Vec<ModelTimeSeriesPoint>> {
    let bucket = series_bucket(bucket_ms(&window));
    query_messages(
        &window,
        &format!(
            r#"{bucket} AS bucket, f.model AS model, f.provider AS provider, SUM(f.requests) AS requests"#
        ),
        "GROUP BY 1, f.model, f.provider ORDER BY 1",
        |row| {
            Ok(ModelTimeSeriesPoint {
                timestamp: i64_col(row, "bucket"),
                model: text_col(row, "model"),
                provider: text_col(row, "provider"),
                requests: i64_col(row, "requests"),
            })
        },
    )
}

/// Average TTFT and output tokens/s per bucket per (model, provider).
///
/// # Errors
/// DB failure.
pub fn get_model_performance_series(window: RangeWindow) -> Result<Vec<ModelPerformancePoint>> {
    let bucket = series_bucket(bucket_ms(&window));
    query_messages(
        &window,
        &format!(
            r#"{bucket} AS bucket, f.model AS model, f.provider AS provider, SUM(f.requests) AS requests,
		 TOTAL(f.ttft_sum) / NULLIF(SUM(f.ttft_n), 0) AS avg_ttft,
		 TOTAL(f.tps_sum) / NULLIF(SUM(f.tps_n), 0) AS avg_tps"#
        ),
        "GROUP BY 1, f.model, f.provider ORDER BY 1",
        |row| {
            Ok(ModelPerformancePoint {
                timestamp: i64_col(row, "bucket"),
                model: text_col(row, "model"),
                provider: text_col(row, "provider"),
                requests: i64_col(row, "requests"),
                avg_ttft: opt_f64_col(row, "avg_ttft"),
                avg_tokens_per_second: opt_f64_col(row, "avg_tps"),
            })
        },
    )
}

/// Daily cost per (model, provider) with the per-component split.
///
/// # Errors
/// DB failure.
pub fn get_cost_time_series(cutoff: Option<i64>) -> Result<Vec<CostTimeSeriesPoint>> {
    let bucket = series_bucket(DAY_MS);
    query_messages(
        &RangeWindow::since(cutoff),
        &format!(
            r#"{bucket} AS bucket, f.model AS model, f.provider AS provider, TOTAL(f.cost_total) AS cost,
		 SUM(f.unpriced) AS unpriced, TOTAL(f.cost_input) AS cost_input, TOTAL(f.cost_output) AS cost_output,
		 TOTAL(f.cost_cache_read) AS cost_cache_read, TOTAL(f.cost_cache_write) AS cost_cache_write,
		 SUM(f.requests) AS requests"#
        ),
        "GROUP BY 1, f.model, f.provider ORDER BY 1",
        |row| {
            Ok(CostTimeSeriesPoint {
                timestamp: i64_col(row, "bucket"),
                model: text_col(row, "model"),
                provider: text_col(row, "provider"),
                cost: f64_col(row, "cost"),
                unpriced_requests: i64_col(row, "unpriced"),
                cost_input: f64_col(row, "cost_input"),
                cost_output: f64_col(row, "cost_output"),
                cost_cache_read: f64_col(row, "cost_cache_read"),
                cost_cache_write: f64_col(row, "cost_cache_write"),
                requests: i64_col(row, "requests"),
            })
        },
    )
}

/// Tokens/cost per bucket per provider.
///
/// # Errors
/// DB failure.
pub fn get_provider_time_series(window: RangeWindow) -> Result<Vec<ProviderTimeSeriesPoint>> {
    let bucket = series_bucket(bucket_ms(&window));
    query_messages(
        &window,
        &format!(
            r#"{bucket} AS bucket, f.provider AS provider,
		 SUM(f.input_tokens + f.output_tokens + f.cache_read_tokens + f.cache_write_tokens) AS total_tokens,
		 TOTAL(f.cost_total) AS cost, SUM(f.unpriced) AS unpriced, SUM(f.requests) AS requests"#
        ),
        "GROUP BY 1, f.provider ORDER BY 1",
        |row| {
            Ok(ProviderTimeSeriesPoint {
                timestamp: i64_col(row, "bucket"),
                provider: text_col(row, "provider"),
                total_tokens: i64_col(row, "total_tokens"),
                cost: f64_col(row, "cost"),
                unpriced_requests: i64_col(row, "unpriced"),
                requests: i64_col(row, "requests"),
            })
        },
    )
}
