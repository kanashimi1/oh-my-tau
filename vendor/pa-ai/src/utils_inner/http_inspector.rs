//! Port of `packages/ai/src/utils/http-inspector.ts`: the error-message
//! finalization every TS provider applies to a failed HTTP request — the
//! captured error body's `message (type=… param=… code=…)` detail line, and
//! for 400/413 rejections a sanitized dump of the raw request under
//! `<logs>/http-400-requests/` referenced by a `raw-http-request=<path>` line.

use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use std::time::{Duration, SystemTime};

use regex::Regex;
use serde::Serialize;
use serde_json::Value;

use crate::utils_inner::stream_failure::ProviderError;

const RAW_HTTP_REQUEST_LINE: &str = "raw-http-request=";
const RAW_HTTP_REQUEST_SAVE_FAILED_LINE: &str = "raw-http-request-save-failed=";
const HTTP_DUMP_DIR_NAME: &str = "http-400-requests";
const HTTP_DUMP_MAX_AGE: Duration = Duration::from_secs(7 * 24 * 60 * 60);
const HTTP_DUMP_MAX_TOTAL_BYTES: u64 = 64 * 1024 * 1024;

static SENSITIVE_HEADER: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)key|token|secret|auth|credential|cookie").expect("valid regex")
});
static NO_BODY: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\bstatus code\s*\(no body\)").expect("valid regex"));

/// The request a provider sent, kept so a rejected request can be dumped.
#[derive(Debug, Clone, Default)]
pub struct RawHttpRequestDump {
    /// Provider id (`model.provider`).
    pub provider: String,
    /// Wire API (`openai-completions`, …).
    pub api: String,
    /// Model id.
    pub model: String,
    /// HTTP method.
    pub method: String,
    /// Request URL.
    pub url: String,
    /// Request headers in send order.
    pub headers: Vec<(String, String)>,
    /// Serialized request body (JSON for every provider API).
    pub body: String,
}

/// TS `finalizeErrorMessage` for an HTTP error response: append the
/// captured body's detail line (providers whose TS transport captures the
/// error response pass `captured_body`), or replace a bare
/// `status code (no body)` message with it; then the raw-request dump
/// reference for 400/413.
#[must_use]
pub fn finalize_http_error_message(
    message: &str,
    status: u16,
    captured_body: Option<&str>,
    dump: Option<&RawHttpRequestDump>,
) -> String {
    let mut message = message.to_owned();
    if let Some(captured) = captured_body.and_then(format_captured_http_error) {
        if NO_BODY.is_match(&message) {
            message = format!("{status} status code: {captured}");
        } else if !message.contains(&captured) {
            message = format!("{message}\n{captured}");
        }
    }
    append_raw_http_request_dump(message, status, dump)
}

/// [`finalize_http_error_message`] applied to a provider HTTP error in place
/// (other variants pass through). `capture_body`: the TS transport captures
/// the error response (openai SDK family), so its detail line is appended.
#[must_use]
pub fn finalize_http_error(
    mut error: ProviderError,
    capture_body: bool,
    dump: Option<&RawHttpRequestDump>,
) -> ProviderError {
    if let ProviderError::Http(http) = &mut error {
        let status = http.status.unwrap_or(0);
        let captured = if capture_body {
            http.body.as_deref()
        } else {
            None
        };
        http.message = finalize_http_error_message(&http.message, status, captured, dump);
    }
    error
}

/// TS `formatCapturedHttpError`: `message (type=… param=… code=…)` from an
/// error body (`{"error": {...}}` or flat), the raw text when it is not a
/// JSON object. The labels follow TS exactly, including its positional
/// quirk: present extras are labelled `type=`, `param=`, `code=` in order of
/// appearance, so a body without `type`/`param` labels its `code` `type=`.
#[must_use]
pub fn format_captured_http_error(body: &str) -> Option<String> {
    let body_text = body.trim();
    if body_text.is_empty() {
        return None;
    }
    let Ok(Value::Object(payload)) = serde_json::from_str::<Value>(body_text) else {
        return Some(body_text.to_owned());
    };
    let nested = payload.get("error").and_then(Value::as_object);
    let error = nested.unwrap_or(&payload);
    let string = |map: &serde_json::Map<String, Value>, key: &str| {
        map.get(key)
            .and_then(Value::as_str)
            .filter(|s| !s.trim().is_empty())
            .map(str::to_owned)
    };
    let string_error = if nested.is_none() {
        string(&payload, "error")
    } else {
        None
    };
    let message = string(error, "message")
        .or_else(|| string(&payload, "message"))
        .or(string_error)
        .unwrap_or_else(|| body_text.to_owned());
    let extras: Vec<String> = ["type", "param", "code"]
        .iter()
        .filter_map(|key| string(error, key).or_else(|| string(&payload, key)))
        .enumerate()
        .map(|(index, value)| match index {
            0 => format!("type={value}"),
            1 => format!("param={value}"),
            _ => format!("code={value}"),
        })
        .collect();
    Some(if extras.is_empty() {
        message
    } else {
        format!("{message} ({})", extras.join(" "))
    })
}

/// TS `stripRawHttpRequestDiagnostics`: drop the trailing local request-dump lines
/// ([`append_raw_http_request_dump`]) so relayed provider errors carry no OMP-local paths.
#[must_use]
pub fn strip_raw_http_request_diagnostics(message: &str) -> String {
    let lines: Vec<&str> = message.split('\n').collect();
    let mut end = lines.len();
    while end > 0
        && (lines[end - 1].starts_with(RAW_HTTP_REQUEST_LINE)
            || lines[end - 1].starts_with(RAW_HTTP_REQUEST_SAVE_FAILED_LINE))
    {
        end -= 1;
    }
    if end == lines.len() {
        message.to_owned()
    } else {
        lines[..end].join("\n")
    }
}

/// TS `appendRawHttpRequestDumpFor400`: write the sanitized request (plus
/// the error) for a 400/413 rejection and reference it from the message.
fn append_raw_http_request_dump(
    message: String,
    status: u16,
    dump: Option<&RawHttpRequestDump>,
) -> String {
    let Some(dump) = dump else {
        return message;
    };
    // TS skips dumps under the bun test runtime; unit tests here likewise.
    if cfg!(test) || !(status == 400 || status == 413) {
        return message;
    }
    let payload = dump_payload(dump, status, &message);
    let Ok(text) = serde_json::to_string_pretty(&payload) else {
        return message;
    };
    let now_ms = crate::utils_inner::diagnostics::now_ms();
    let file_name = format!(
        "{now_ms}-{}.json",
        crate::utils_inner::hash::short_hash(&text)
    );
    let dir = logs_dir().join(HTTP_DUMP_DIR_NAME);
    let path = dir.join(&file_name);
    let written =
        std::fs::create_dir_all(&dir).and_then(|()| std::fs::write(&path, format!("{text}\n")));
    match written {
        Ok(()) => {
            prune_http_request_dumps(&dir, &file_name);
            format!("{message}\n{RAW_HTTP_REQUEST_LINE}{}", path.display())
        }
        Err(error) => format!("{message}\n{RAW_HTTP_REQUEST_SAVE_FAILED_LINE}{error}"),
    }
}

#[derive(Serialize)]
struct DumpPayload<'a> {
    provider: &'a str,
    api: &'a str,
    model: &'a str,
    method: &'a str,
    url: String,
    headers: serde_json::Map<String, Value>,
    body: Value,
    #[serde(rename = "errorResponse")]
    error_response: ErrorResponse<'a>,
}

#[derive(Serialize)]
struct ErrorResponse<'a> {
    status: u16,
    message: &'a str,
}

/// TS `buildHttp400DumpPayload`: query string and sensitive headers redacted.
fn dump_payload<'a>(
    dump: &'a RawHttpRequestDump,
    status: u16,
    message: &'a str,
) -> DumpPayload<'a> {
    let headers = dump
        .headers
        .iter()
        .map(|(key, value)| {
            let value = if SENSITIVE_HEADER.is_match(key) {
                "[redacted]".to_owned()
            } else {
                value.clone()
            };
            (key.clone(), Value::String(value))
        })
        .collect();
    DumpPayload {
        provider: &dump.provider,
        api: &dump.api,
        model: &dump.model,
        method: &dump.method,
        url: redact_url_query(&dump.url),
        headers,
        body: serde_json::from_str(&dump.body).unwrap_or_else(|_| Value::String(dump.body.clone())),
        error_response: ErrorResponse { status, message },
    }
}

fn redact_url_query(url: &str) -> String {
    match url::Url::parse(url) {
        Ok(mut parsed) if parsed.query().is_some() => {
            parsed.set_query(None);
            format!("{parsed}[redacted-query]")
        }
        _ => url.to_owned(),
    }
}

/// pi-utils `getLogsDir()`: `$XDG_STATE_HOME/omp/logs` when that app root
/// exists, else `~/.omp/logs` (`PI_CONFIG_DIR` renames `.omp`).
fn logs_dir() -> PathBuf {
    if let Some(state) = std::env::var_os("XDG_STATE_HOME").filter(|v| !v.is_empty()) {
        let root = PathBuf::from(state).join("omp");
        if root.is_dir() {
            return root.join("logs");
        }
    }
    let home = std::env::var_os("HOME").map_or_else(|| PathBuf::from("/"), PathBuf::from);
    let name = std::env::var("PI_CONFIG_DIR")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| ".omp".to_owned());
    home.join(name).join("logs")
}

/// TS `pruneHttpRequestDumps`: newest first, keep `keep` always, drop dumps
/// older than 7 days or past 64 MiB of retained total. Best-effort.
fn prune_http_request_dumps(dir: &Path, keep: &str) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let now = SystemTime::now();
    let mut dumps: Vec<(String, SystemTime, u64)> = entries
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let name = entry.file_name().into_string().ok()?;
            let meta = entry.metadata().ok()?;
            (name.ends_with(".json") && meta.is_file())
                .then(|| (name, meta.modified().unwrap_or(now), meta.len()))
        })
        .collect();
    dumps.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| b.0.cmp(&a.0)));
    let mut retained = 0u64;
    for (name, modified, size) in dumps {
        let age = now.duration_since(modified).unwrap_or_default();
        if name == keep
            || (age <= HTTP_DUMP_MAX_AGE && retained + size <= HTTP_DUMP_MAX_TOTAL_BYTES)
        {
            retained += size;
            continue;
        }
        let _ = std::fs::remove_file(dir.join(name));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn captured_detail_follows_ts_labels() {
        let body = r#"{"error":{"message":"E2E-PROVIDER-FAILURE","type":"invalid_request_error","code":"e2e_failure"}}"#;
        assert_eq!(
            format_captured_http_error(body).as_deref(),
            Some("E2E-PROVIDER-FAILURE (type=invalid_request_error param=e2e_failure)")
        );
        assert_eq!(
            format_captured_http_error(r#"{"error":{"message":"bad"}}"#).as_deref(),
            Some("bad")
        );
        assert_eq!(
            format_captured_http_error("<html>502</html>").as_deref(),
            Some("<html>502</html>")
        );
        assert_eq!(format_captured_http_error("  "), None);
        assert_eq!(
            format_captured_http_error(r#"{"error":"flat","code":"x"}"#).as_deref(),
            Some("flat (type=x)")
        );
    }

    #[test]
    fn finalize_appends_detail_once() {
        let body = r#"{"error":{"message":"E2E-PROVIDER-FAILURE","type":"invalid_request_error","code":"e2e_failure"}}"#;
        assert_eq!(
            finalize_http_error_message("400 E2E-PROVIDER-FAILURE", 400, Some(body), None),
            "400 E2E-PROVIDER-FAILURE\nE2E-PROVIDER-FAILURE (type=invalid_request_error param=e2e_failure)"
        );
        // Already contained: unchanged.
        assert_eq!(
            finalize_http_error_message(
                "400 bad",
                400,
                Some(r#"{"error":{"message":"bad"}}"#),
                None
            ),
            "400 bad"
        );
        // A bare no-body message takes the captured text.
        assert_eq!(
            finalize_http_error_message(
                "400 status code (no body)",
                400,
                Some(r#"{"detail":"x","message":"m"}"#),
                None
            ),
            "400 status code: m"
        );
    }

    #[test]
    fn dump_payload_redacts_secrets() {
        let dump = RawHttpRequestDump {
            provider: "mock".into(),
            api: "openai-completions".into(),
            model: "fail".into(),
            method: "POST".into(),
            url: "http://h/v1/chat/completions?key=1".into(),
            headers: vec![
                ("Authorization".into(), "Bearer s".into()),
                ("content-type".into(), "application/json".into()),
            ],
            body: r#"{"model":"fail"}"#.into(),
        };
        let payload = serde_json::to_value(dump_payload(&dump, 400, "400 x")).unwrap();
        assert_eq!(payload["headers"]["Authorization"], "[redacted]");
        assert_eq!(payload["headers"]["content-type"], "application/json");
        assert_eq!(
            payload["url"],
            "http://h/v1/chat/completions[redacted-query]"
        );
        assert_eq!(payload["body"]["model"], "fail");
        assert_eq!(payload["errorResponse"]["status"], 400);
    }
}
