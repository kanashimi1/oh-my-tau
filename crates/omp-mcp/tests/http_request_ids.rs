#![allow(clippy::disallowed_methods)] // tests/entry points: throwaway runtimes drive async code
//! Streamable-HTTP JSON-RPC ids: unique per request within a session, never
//! reused after a failed request and distinct for concurrent requests.
#![allow(clippy::unwrap_used, clippy::expect_used)] // test code: a failed step should fail the test loudly

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use omp_mcp::transport::http::HttpTransport;
use omp_mcp::transport::McpTransport;
use omp_mcp::{McpServerConfig, ServerKind};
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

async fn read_request(stream: &mut TcpStream) -> Value {
    let mut bytes = Vec::new();
    let (header_end, content_len) = loop {
        let mut chunk = [0_u8; 4096];
        let n = stream.read(&mut chunk).await.expect("read HTTP request");
        assert!(n > 0, "request ended before headers");
        bytes.extend_from_slice(&chunk[..n]);
        if let Some(end) = bytes.windows(4).position(|w| w == b"\r\n\r\n") {
            let headers = String::from_utf8_lossy(&bytes[..end]);
            let len = headers
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().expect("content length"))
                })
                .unwrap_or(0);
            break (end + 4, len);
        }
    };
    while bytes.len() < header_end + content_len {
        let mut chunk = [0_u8; 4096];
        let n = stream.read(&mut chunk).await.expect("read HTTP body");
        assert!(n > 0, "request ended before body");
        bytes.extend_from_slice(&chunk[..n]);
    }
    serde_json::from_slice(&bytes[header_end..header_end + content_len]).expect("JSON body")
}

async fn respond(stream: &mut TcpStream, status: &str, body: &str) {
    let response = format!(
        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream
        .write_all(response.as_bytes())
        .await
        .expect("write response");
    stream.shutdown().await.expect("shutdown response");
}

/// Fake server: answers `n` requests, the first with HTTP 500, the rest with
/// a JSON-RPC result; returns the request ids in arrival order.
/// Notifications (the `notifications/cancelled` a failed request owes) are
/// acknowledged and not counted.
async fn serve(listener: TcpListener, n: usize) -> Vec<Value> {
    let mut ids = Vec::new();
    while ids.len() < n {
        let (mut stream, _) = listener.accept().await.expect("accept");
        let body = read_request(&mut stream).await;
        if body.get("id").is_none() {
            respond(&mut stream, "202 Accepted", "").await;
            continue;
        }
        let id = body["id"].clone();
        ids.push(id.clone());
        if ids.len() == 1 {
            respond(&mut stream, "500 Internal Server Error", "boom").await;
        } else {
            let ok = json!({"jsonrpc": "2.0", "id": id, "result": {}}).to_string();
            respond(&mut stream, "200 OK", &ok).await;
        }
    }
    ids
}

fn transport(url: String) -> HttpTransport {
    let config = McpServerConfig {
        enabled: Some(true),
        timeout: Some(5_000),
        kind: ServerKind::Http,
        command: None,
        args: None,
        env: None,
        cwd: None,
        url: Some(url),
        headers: None,
        extra: BTreeMap::new(),
    };
    HttpTransport::new("ids", config, Arc::new(PathBuf::from(".")))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn failed_request_id_is_not_reused() {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let url = format!("http://{}/mcp", listener.local_addr().unwrap());
    let server = tokio::spawn(serve(listener, 2));
    let mut transport = transport(url);

    transport
        .request("tools/list", json!({}))
        .await
        .expect_err("HTTP 500 fails the request");
    transport
        .request("tools/list", json!({}))
        .await
        .expect("second request succeeds");

    let ids = server.await.expect("server");
    assert_ne!(ids[0], ids[1], "an id must not be reused after a failure");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn concurrent_requests_get_distinct_ids() {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let url = format!("http://{}/mcp", listener.local_addr().unwrap());
    let server = tokio::spawn(serve(listener, 3));
    let mut transport = transport(url);

    let first = transport.request("tools/list", json!({}));
    let second = transport.request("tools/list", json!({}));
    let third = transport.request("tools/list", json!({}));
    let _ = futures::join!(first, second, third);

    let mut ids = server.await.expect("server");
    ids.sort_by_key(Value::as_u64);
    ids.dedup();
    assert_eq!(ids.len(), 3, "concurrent requests share no id");
}
