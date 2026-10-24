//! `handleApi`: the `/api/*` router. Handlers are synchronous (sqlite + file
//! reads); the hyper adapter runs them on the blocking pool.

use hyper::HeaderMap;
use serde::Serialize;
use serde_json::{json, Value};

use super::query::{parse_int, Target};
use crate::{aggregator, frustration, gain, live, rollup};

/// Required on every spending/mutating POST. A custom header forces a CORS
/// preflight, which this server never approves, so a hostile page cannot
/// trigger a paid judge run through a cross-site form or `fetch`.
pub const STATS_ACTION_HEADER: &str = "X-Omp-Stats-Action";

/// A fully materialised HTTP response (the TS `Response`).
#[derive(Debug, Clone)]
pub struct ApiResponse {
    pub status: u16,
    pub content_type: String,
    /// Extra headers beyond `Content-Type` (e.g. `ETag`).
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl ApiResponse {
    /// `Response.json(value, { status })`.
    pub fn json<T: Serialize>(status: u16, value: &T) -> anyhow::Result<Self> {
        Ok(Self {
            status,
            content_type: "application/json;charset=utf-8".to_string(),
            headers: Vec::new(),
            body: serde_json::to_vec(value)?,
        })
    }

    /// `new Response(text, { status })`.
    #[must_use]
    pub fn text(status: u16, text: &str) -> Self {
        Self {
            status,
            content_type: "text/plain;charset=utf-8".to_string(),
            headers: Vec::new(),
            body: text.as_bytes().to_vec(),
        }
    }

    /// `new Response(null, { status })`.
    #[must_use]
    pub fn empty(status: u16) -> Self {
        Self {
            status,
            content_type: String::new(),
            headers: Vec::new(),
            body: Vec::new(),
        }
    }

    /// `{ error: message }` with `status`.
    #[must_use]
    pub fn error(status: u16, message: &str) -> Self {
        Self::json(status, &json!({ "error": message })).unwrap_or_else(|_| Self::empty(status))
    }

    /// Parse the JSON body (test helper and in-process callers).
    #[must_use]
    pub fn body_json(&self) -> Option<Value> {
        serde_json::from_slice(&self.body).ok()
    }
}

fn ok<T: Serialize>(value: &T) -> anyhow::Result<ApiResponse> {
    ApiResponse::json(200, value)
}

/// `limit ? parseInt(limit, 10) : undefined`.
fn limit_param(target: &Target) -> Option<i64> {
    target
        .get("limit")
        .filter(|s| !s.is_empty())
        .and_then(parse_int)
}

/// Handle one API request. `path_and_query` is the raw request target
/// (`/api/stats?range=7d`). Errors are the TS "unhandled → 500" case; the
/// adapter turns them into `{ error }`.
///
/// # Errors
/// Any handler failure (sqlite, I/O) other than the mapped 4xx responses.
pub fn handle_api(
    method: &str,
    path_and_query: &str,
    headers: &HeaderMap,
) -> anyhow::Result<ApiResponse> {
    let target = Target::parse(path_and_query);
    let path = target.path.as_str();
    // Stats reads are DB-only; ingest runs in the background (see `live`).
    let range = target.get("range");

    match path {
        "/api/status" => return ok(&live::stats_live().status()),
        "/api/stats" => return ok(&aggregator::get_dashboard_stats(range)?),
        "/api/stats/overview" => return ok(&aggregator::get_overview_stats(range)?),
        "/api/stats/model-dashboard" => return ok(&aggregator::get_model_dashboard_stats(range)?),
        "/api/stats/costs" => return ok(&aggregator::get_cost_dashboard_stats(range)?),
        "/api/stats/frustration" => {
            return ok(&frustration::get_frustration_dashboard_stats(range)?)
        }
        "/api/frustration/estimate" => return ok(&frustration::estimate_frustration_run(range)?),
        "/api/frustration/judge" | "/api/frustration/cancel" => {
            return frustration_action(method, path, range, headers)
        }
        "/api/stats/tools" => return ok(&aggregator::get_tool_dashboard_stats(range)?),
        "/api/stats/provider-windows" => {
            return ok(&aggregator::get_provider_window_stats(
                range,
                target.get("provider"),
            )?);
        }
        "/api/stats/providers" => return ok(&aggregator::get_provider_dashboard_stats(range)?),
        "/api/stats/recent" => return ok(&aggregator::get_recent_requests(limit_param(&target))?),
        "/api/stats/errors" => {
            return ok(&aggregator::get_recent_errors(range, limit_param(&target))?)
        }
        "/api/stats/models" => {
            // Also the port-conflict identity probe: keep it to the one query it serves.
            return ok(&rollup::get_stats_by_model(
                rollup::get_time_range_config(range).cutoff,
            )?);
        }
        "/api/stats/folders" => return ok(&aggregator::get_folder_stats(range)?),
        "/api/stats/timeseries" => {
            return ok(&rollup::get_time_series(rollup::get_time_range_config(
                range,
            ))?)
        }
        "/api/sync" => {
            if method != "POST" {
                return ApiResponse::json(405, &json!({ "error": "POST required" }));
            }
            let live = live::stats_live();
            live.request_sync(None);
            return ApiResponse::json(202, &live.status());
        }
        "/api/stats/gain" => {
            return ok(&gain::get_gain_dashboard_stats(
                range,
                target.get("project"),
            )?)
        }
        "/api/sessions" => {
            // Number(limit ?? "100"), finite and > 0, floored; else 100.
            let limit = target
                .get("limit")
                .map_or(Some(100.0), |s| {
                    if s.trim().is_empty() {
                        Some(0.0)
                    } else {
                        s.trim().parse::<f64>().ok()
                    }
                })
                .filter(|n| n.is_finite() && *n > 0.0)
                .map_or(100, |n| n.floor() as i64);
            return ok(&crate::trace::list_session_summaries(
                limit,
                target.get("q"),
            )?);
        }
        "/api/session/trace" => return super::session::session_trace(&target, headers),
        "/api/session/entry" => return super::session::session_entry(&target),
        _ => {}
    }

    if let Some(rest) = path.strip_prefix("/api/request/") {
        let id = rest.rsplit('/').next().unwrap_or("");
        if id.is_empty() {
            return Ok(ApiResponse::text(400, "Bad Request"));
        }
        return match parse_int(id)
            .map(aggregator::get_request_details)
            .transpose()?
            .flatten()
        {
            Some(details) => ok(&details),
            None => Ok(ApiResponse::text(404, "Not Found")),
        };
    }
    Ok(ApiResponse::text(404, "Not Found"))
}

/// `/api/frustration/judge` and `/api/frustration/cancel`: spending POSTs.
fn frustration_action(
    method: &str,
    path: &str,
    range: Option<&str>,
    headers: &HeaderMap,
) -> anyhow::Result<ApiResponse> {
    if method != "POST" {
        return ApiResponse::json(405, &json!({ "error": "POST required" }));
    }
    let has_header = headers
        .get(STATS_ACTION_HEADER)
        .and_then(|v| v.to_str().ok())
        == Some("1");
    if !has_header {
        return ApiResponse::json(
            403,
            &json!({ "error": format!("{STATS_ACTION_HEADER}: 1 header required") }),
        );
    }
    if path == "/api/frustration/cancel" {
        return ok(&frustration::cancel_frustration_run());
    }
    match frustration::start_frustration_run(range)? {
        frustration::StartFrustrationRunResult::Started { job } => ApiResponse::json(202, &job),
        frustration::StartFrustrationRunResult::Refused { status, error } => {
            ApiResponse::json(status, &json!({ "error": error }))
        }
    }
}
