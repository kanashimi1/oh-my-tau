//! One accepted connection: hyper http1 with a header-read timeout and an
//! idle timeout for plain keep-alive connections. SSE streams are exempt from
//! the idle timeout (they are long-lived by design and carry heartbeats).

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper_util::rt::{TokioIo, TokioTimer};

use super::http::{serve, Ctx};

/// Per-connection timeouts; see [`super::StartServerOptions`].
#[derive(Debug, Clone, Copy)]
pub(super) struct Timeouts {
    /// Max time to receive a request's headers (slow-loris guard).
    pub header_read: Duration,
    /// Close a non-SSE connection after this long without an active request.
    pub idle: Duration,
}

impl Default for Timeouts {
    fn default() -> Self {
        Self { header_read: Duration::from_secs(30), idle: Duration::from_secs(60) }
    }
}

struct ConnState {
    last_activity: Mutex<Instant>,
    inflight: AtomicUsize,
    sse: AtomicBool,
}

impl ConnState {
    fn touch(&self) {
        *self.last_activity.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Instant::now();
    }

    fn idle_for(&self) -> Option<Duration> {
        if self.sse.load(Ordering::SeqCst) || self.inflight.load(Ordering::SeqCst) > 0 {
            return None;
        }
        Some(self.last_activity.lock().unwrap_or_else(std::sync::PoisonError::into_inner).elapsed())
    }
}

/// Serve `stream` until the peer closes, an error occurs or the idle timeout fires.
pub(super) async fn serve_connection(stream: tokio::net::TcpStream, ctx: Arc<Ctx>, timeouts: Timeouts) {
    let state = Arc::new(ConnState { last_activity: Mutex::new(Instant::now()), inflight: AtomicUsize::new(0), sse: AtomicBool::new(false) });
    let svc_state = Arc::clone(&state);
    let service = service_fn(move |req| {
        let (ctx, state) = (Arc::clone(&ctx), Arc::clone(&svc_state));
        async move {
            if req.uri().path() == "/api/events" {
                state.sse.store(true, Ordering::SeqCst);
            }
            state.inflight.fetch_add(1, Ordering::SeqCst);
            let response = serve(req, ctx).await;
            state.inflight.fetch_sub(1, Ordering::SeqCst);
            state.touch();
            response
        }
    });
    let conn = http1::Builder::new()
        .timer(TokioTimer::new())
        .header_read_timeout(timeouts.header_read)
        .serve_connection(TokioIo::new(stream), service);
    tokio::pin!(conn);
    let tick = (timeouts.idle / 4).clamp(Duration::from_millis(10), Duration::from_secs(5));
    loop {
        tokio::select! {
            result = conn.as_mut() => {
                if let Err(error) = result {
                    tracing::debug!(error = %error, "Stats dashboard connection ended");
                }
                return;
            }
            () = tokio::time::sleep(tick) => {
                if state.idle_for().is_some_and(|idle| idle >= timeouts.idle) {
                    conn.as_mut().graceful_shutdown();
                    let _ = conn.await;
                    return;
                }
            }
        }
    }
}
