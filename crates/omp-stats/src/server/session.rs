//! `/api/session/trace` and `/api/session/entry`.

use hyper::HeaderMap;
use serde_json::json;

use super::api::ApiResponse;
use super::query::Target;
use crate::trace::{self, TracePathError, TRACE_ETAG_VERSION};

/// JS `String(number)` for the millisecond mtimes used in ETags.
fn js_number(n: f64) -> String {
    if n.is_finite() && n.fract() == 0.0 && n.abs() < 1e21 {
        format!("{n:.0}")
    } else {
        n.to_string()
    }
}

fn not_modified() -> ApiResponse {
    ApiResponse::empty(304)
}

/// Map trace errors like the TS catch blocks: path errors 400, ENOENT 404.
fn map_error(err: anyhow::Error, enoent: Option<&str>) -> anyhow::Result<ApiResponse> {
    if err.downcast_ref::<TracePathError>().is_some() {
        return Ok(ApiResponse::error(400, &err.to_string()));
    }
    if let Some(message) = enoent {
        if err
            .downcast_ref::<std::io::Error>()
            .is_some_and(|e| e.kind() == std::io::ErrorKind::NotFound)
        {
            return Ok(ApiResponse::error(404, message));
        }
    }
    Err(err)
}

pub(super) fn session_trace(target: &Target, headers: &HeaderMap) -> anyhow::Result<ApiResponse> {
    let Some(file) = target.get("file").filter(|f| !f.is_empty()) else {
        return Ok(ApiResponse::error(400, "file required"));
    };
    // ETag-first: compare the client's etag against the root+child fingerprint
    // WITHOUT building the trace. The fingerprint covers child transcripts, so
    // a subagent-only append changes the ETag and never 304s stale.
    let client_etag = headers
        .get("if-none-match")
        .and_then(|v| v.to_str().ok())
        .filter(|s| !s.is_empty());
    let mut fingerprint = None;
    if let Some(client) = client_etag {
        fingerprint = trace::trace_fingerprint_for_etag(file);
        if let Some(fp) = &fingerprint {
            let etag = format!(
                "\"{TRACE_ETAG_VERSION}:{}:{}\"",
                js_number(fp.root_mtime_ms),
                fp.child_fingerprint
            );
            if client == etag {
                return Ok(not_modified());
            }
        }
    }
    match trace::build_session_trace(file, fingerprint.as_ref()) {
        Ok(built) => {
            let etag = format!("\"{TRACE_ETAG_VERSION}:{}\"", built.etag);
            if client_etag == Some(etag.as_str()) {
                return Ok(not_modified());
            }
            let mut response = ApiResponse::json(200, &built)?;
            response.headers.push(("ETag".to_string(), etag));
            Ok(response)
        }
        Err(err) => map_error(err, Some("session not found")),
    }
}

pub(super) fn session_entry(target: &Target) -> anyhow::Result<ApiResponse> {
    let (Some(file), Some(id)) = (
        target.get("file").filter(|f| !f.is_empty()),
        target.get("id").filter(|f| !f.is_empty()),
    ) else {
        return Ok(ApiResponse::error(400, "file and id required"));
    };
    match trace::get_trace_entry(file, id) {
        Ok(Some(entry)) => ApiResponse::json(200, &json!({ "entry": entry })),
        Ok(None) => Ok(ApiResponse::error(404, "entry not found")),
        Err(err) => map_error(err, None),
    }
}
