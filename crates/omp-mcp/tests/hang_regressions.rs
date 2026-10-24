//! Regression tests for the confirmed MCP hang bugs (vega report F1): a
//! request whose frame exceeds the OS pipe buffer must fail bounded by the
//! request timeout even when the server never drains stdin — the old code
//! awaited `write_all` before arming any timeout and hung forever.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use omp_mcp::errors::{Failure, McpError};
use omp_mcp::transport::stdio::StdioTransport;
use omp_mcp::transport::McpTransport;
use omp_mcp::McpServerConfig;

/// A server process that never reads stdin and never writes stdout: the
/// write of a frame larger than the ~64KB pipe buffer parks forever.
fn wedged_server_config(timeout_ms: u64) -> McpServerConfig {
    McpServerConfig {
        enabled: Some(true),
        timeout: Some(timeout_ms),
        kind: omp_mcp::ServerKind::Stdio,
        // `sleep` never touches stdin and outlives the harness bound; the
        // transport's close() reaps it at the end of the test.
        command: Some("sleep".into()),
        args: Some(vec!["60".into()]),
        env: None,
        cwd: None,
        url: None,
        headers: None,
        extra: Default::default(),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn request_to_server_that_never_reads_stdin_times_out() {
    let mut transport = StdioTransport::new(
        "wedged",
        wedged_server_config(300),
        Arc::new(PathBuf::from(".")),
    );
    transport
        .connect()
        .await
        .expect("connect must spawn the subprocess");

    // ~30x the 64KB OS pipe buffer: write_all cannot complete, and `sleep`
    // will never drain it.
    let payload = "x".repeat(2_000_000);
    let started = Instant::now();
    let outcome = tokio::time::timeout(
        Duration::from_secs(10),
        transport.request("tools/list", serde_json::json!({ "pad": payload })),
    )
    .await;
    let elapsed = started.elapsed();

    assert!(
        outcome.is_ok(),
        "request must return within the harness bound, not hang"
    );
    match outcome.unwrap() {
        Err(McpError::Transport(err)) => {
            assert_eq!(
                err.failure,
                Failure::Timeout,
                "expected the request timeout, got {err}"
            )
        }
        other => panic!("expected a Timeout transport error, got {other:?}"),
    }
    // Bounded by the 300ms request timeout (with slack), not forever.
    assert!(
        elapsed < Duration::from_secs(5),
        "a wedged stdin write must be bounded by the request timeout, took {elapsed:?}"
    );
    transport.close().await;
}

/// The same wedge on the notification path: `notify` has no pending entry to
/// time out, so its write is bounded directly.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn notify_to_server_that_never_reads_stdin_times_out() {
    let mut transport = StdioTransport::new(
        "wedged",
        wedged_server_config(300),
        Arc::new(PathBuf::from(".")),
    );
    transport
        .connect()
        .await
        .expect("connect must spawn the subprocess");

    let payload = "x".repeat(2_000_000);
    let started = Instant::now();
    let outcome = tokio::time::timeout(
        Duration::from_secs(10),
        transport.notify("notifications/pad", serde_json::json!({ "pad": payload })),
    )
    .await;
    assert!(
        outcome.is_ok(),
        "notify must return within the harness bound, not hang"
    );
    assert!(
        outcome.unwrap().is_err(),
        "the wedged notification write must surface an error"
    );
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "a wedged notification write must be bounded, took {:?}",
        started.elapsed()
    );
    transport.close().await;
}
