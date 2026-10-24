//! Parity: every recorded TS request replayed through `handle_api`, plus a
//! real-TCP end-to-end smoke test of the hyper server.

use std::io::{Read, Write};
use std::net::TcpStream;

use hyper::HeaderMap;
use serde_json::Value;

use super::{handle_api, start_server, ApiResponse, StartServerOptions};
use crate::testutil::golden::{self, assert_golden, FIXTURE_HOME, GOLDEN_DIR};
use crate::testutil::TestHome;

fn encode_slashes(path: &str) -> String {
    path.replace('/', "%2F")
}

/// The recorded request pointed at the fixture path; retarget it at the temp home.
fn localize_request(request: &str, home: &TestHome) -> String {
    request.replace(
        &encode_slashes(FIXTURE_HOME),
        &encode_slashes(&home.path().to_string_lossy()),
    )
}

/// Rust rowid for the golden `/api/request/<n>` row, matched by session file + entry id.
fn rust_request_id(request: &str, home: &TestHome) -> i64 {
    let (_, body) = golden::golden(request);
    let session_file = body["sessionFile"]
        .as_str()
        .expect("sessionFile")
        .replace(FIXTURE_HOME, &home.path().to_string_lossy());
    let entry_id = body["entryId"].as_str().expect("entryId").to_string();
    crate::db::with_db(|db| {
        Ok(db.query_row(
            "SELECT id FROM messages WHERE session_file = ?1 AND entry_id = ?2",
            rusqlite::params![session_file, entry_id],
            |row| row.get::<_, i64>(0),
        )?)
    })
    .expect("row id for golden request")
}

fn body_value(response: &ApiResponse) -> Value {
    response
        .body_json()
        .unwrap_or_else(|| Value::String(String::from_utf8_lossy(&response.body).into_owned()))
}

fn all_requests() -> Vec<String> {
    let text = std::fs::read_to_string(std::path::Path::new(GOLDEN_DIR).join("_index.json"))
        .expect("golden index");
    let index: Value = serde_json::from_str(&text).expect("golden index json");
    index
        .as_object()
        .expect("index object")
        .keys()
        .cloned()
        .collect()
}

#[test]
fn every_golden_request_matches_through_handle_api() {
    let Some(home) = golden::fixture_home() else {
        return;
    };
    let requests = all_requests();
    assert_eq!(requests.len(), 85, "golden index size");
    let headers = HeaderMap::new();
    let mut failures = Vec::new();
    for request in &requests {
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            check_request(request, &home, &headers)
        }));
        if let Err(panic) = outcome {
            let message = panic
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| panic.downcast_ref::<&str>().map(|s| (*s).to_string()));
            failures.push(message.unwrap_or_else(|| format!("{request}: panicked")));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} golden requests failed:\n{}",
        failures.len(),
        requests.len(),
        failures.join("\n")
    );
}

fn check_request(request: &str, home: &TestHome, headers: &HeaderMap) {
    {
        let (status, _) = golden::golden(request);
        let actual_request = match request.strip_prefix("/api/request/") {
            Some(n) if n != "999999" => format!("/api/request/{}", rust_request_id(request, home)),
            _ => localize_request(request, home),
        };
        let response = handle_api("GET", &actual_request, headers)
            .unwrap_or_else(|e| panic!("{request}: {e:#}"));
        assert_eq!(response.status, status, "{request}: status");
        assert_golden(request, &body_value(&response), home);
    }
}

#[test]
fn router_rejections() {
    let Some(_home) = golden::fixture_home() else {
        return;
    };
    let headers = HeaderMap::new();
    let get = |m: &str, p: &str| handle_api(m, p, &headers).expect("response");
    assert_eq!(get("GET", "/api/nope").status, 404);
    assert_eq!(get("GET", "/api/request/").status, 400);
    assert_eq!(get("GET", "/api/sync").status, 405);
    assert_eq!(get("GET", "/api/frustration/judge").status, 405);
    assert_eq!(get("POST", "/api/frustration/judge").status, 403);
    assert_eq!(get("GET", "/api/session/trace").status, 400);
    assert_eq!(get("GET", "/api/session/entry?file=x").status, 400);
}

fn http_get(port: u16, path: &str) -> (u16, String, Vec<u8>) {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).expect("connect");
    write!(
        stream,
        "GET {path} HTTP/1.1\r\nHost: localhost:{port}\r\nConnection: close\r\n\r\n"
    )
    .expect("send");
    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).expect("read");
    let split = raw
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .expect("header end");
    let head = String::from_utf8_lossy(&raw[..split]).into_owned();
    let status: u16 = head
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .expect("status");
    (status, head, raw[split + 4..].to_vec())
}

#[test]
fn serves_api_and_client_over_tcp() {
    let Some(home) = golden::fixture_home() else {
        return;
    };
    let handle = start_server(0, "127.0.0.1", StartServerOptions::default()).expect("start server");
    assert_ne!(handle.port, 0);

    let (status, head, body) = http_get(handle.port, "/api/stats?range=all");
    assert_eq!(status, 200);
    assert!(
        head.to_ascii_lowercase()
            .contains("x-omp-stats-dashboard: 3"),
        "identity header: {head}"
    );
    let actual: Value = serde_json::from_slice(&body).expect("json body");
    assert_golden("/api/stats?range=all", &actual, &home);

    let (status, head, body) = http_get(handle.port, "/");
    assert_eq!(status, 200);
    assert!(head.to_ascii_lowercase().contains("text/html"), "{head}");
    assert!(String::from_utf8_lossy(&body).contains("<html"));
    // SPA fallback and exact bundle files.
    assert_eq!(http_get(handle.port, "/some/route").0, 200);
    let (status, head, _) = http_get(handle.port, "/index.js");
    assert_eq!(status, 200);
    assert!(head.to_ascii_lowercase().contains("javascript"), "{head}");

    // SSE: the first frame is the current status.
    let mut stream = TcpStream::connect(("127.0.0.1", handle.port)).expect("connect");
    write!(
        stream,
        "GET /api/events HTTP/1.1\r\nHost: localhost:{}\r\n\r\n",
        handle.port
    )
    .expect("send");
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(5)))
        .expect("timeout");
    let mut seen = String::new();
    let mut buf = [0u8; 2048];
    while !seen.contains("data: ") {
        let n = stream.read(&mut buf).expect("sse read");
        assert!(n > 0, "sse closed early: {seen}");
        seen.push_str(&String::from_utf8_lossy(&buf[..n]));
    }
    assert!(
        seen.to_ascii_lowercase().contains("text/event-stream"),
        "{seen}"
    );
    drop(stream);
    handle.stop();
}

/// `--host [::1]` binds `::1`, identifies itself as `::1` (not `[::1]`), and
/// its guard accepts the bracketed `Host` the browser sends.
#[test]
fn bracketed_ipv6_host_binds_bare_and_passes_its_own_guard() {
    let handle =
        start_server(0, "[::1]", StartServerOptions::default()).expect("start server on [::1]");
    assert_eq!(handle.hostname, "::1");
    assert_eq!(
        super::format_stats_dashboard_url(&handle.hostname, handle.port),
        format!("http://[::1]:{}", handle.port)
    );
    let mut stream = TcpStream::connect(("::1", handle.port)).expect("connect");
    write!(
        stream,
        "GET /some/route HTTP/1.1\r\nHost: [::1]:{}\r\nConnection: close\r\n\r\n",
        handle.port
    )
    .expect("send");
    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).expect("read");
    handle.stop();
    let head = String::from_utf8_lossy(&raw).to_ascii_lowercase();
    assert!(head.starts_with("http/1.1 200"), "{head}");
    assert!(head.contains("x-omp-stats-hostname: ::1\r\n"), "{head}");
}
