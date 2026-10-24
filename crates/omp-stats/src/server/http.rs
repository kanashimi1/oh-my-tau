//! hyper 1 adapter: dashboard identity headers, OPTIONS, SSE, 500 mapping.

use std::convert::Infallible;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;

use http_body_util::{combinators::BoxBody, BodyExt, Full};
use hyper::body::{Bytes, Frame, Incoming};
use hyper::header::{HeaderName, HeaderValue, CACHE_CONTROL, CONTENT_TYPE};
use hyper::{HeaderMap, Method, Request, Response};
use tokio::sync::{broadcast, mpsc};

use super::api::{handle_api, ApiResponse};
use super::guard;
use super::statics::handle_static;
use crate::live;
use crate::port_conflict::{
    STATS_DASHBOARD_HEADER, STATS_DASHBOARD_HOSTNAME_HEADER, STATS_DASHBOARD_SECURITY_VERSION,
};

pub(super) type Body = BoxBody<Bytes, Infallible>;

/// Keep-alive comment cadence so proxies and the browser keep the stream open.
const EVENT_HEARTBEAT: Duration = Duration::from_secs(15);

fn full(bytes: Vec<u8>) -> Body {
    Full::new(Bytes::from(bytes)).boxed()
}

/// The identity header lets another omp session's reuse probe positively
/// recognize this dashboard without allowing cross-origin API reads.
fn stamp(headers: &mut HeaderMap, hostname: &str) {
    headers.insert(
        HeaderName::from_static(STATS_DASHBOARD_HEADER),
        HeaderValue::from_static(STATS_DASHBOARD_SECURITY_VERSION),
    );
    if let Ok(value) = HeaderValue::from_str(hostname) {
        headers.insert(
            HeaderName::from_static(STATS_DASHBOARD_HOSTNAME_HEADER),
            value,
        );
    }
}

fn to_response(api: ApiResponse, hostname: &str) -> Response<Body> {
    let mut response = Response::new(full(api.body));
    *response.status_mut() =
        hyper::StatusCode::from_u16(api.status).unwrap_or(hyper::StatusCode::INTERNAL_SERVER_ERROR);
    let headers = response.headers_mut();
    if !api.content_type.is_empty() {
        if let Ok(value) = HeaderValue::from_str(&api.content_type) {
            headers.insert(CONTENT_TYPE, value);
        }
    }
    for (name, value) in api.headers {
        if let (Ok(name), Ok(value)) = (
            HeaderName::try_from(name.as_str()),
            HeaderValue::from_str(&value),
        ) {
            headers.insert(name, value);
        }
    }
    stamp(headers, hostname);
    response
}

/// Per-server request context.
pub(super) struct Ctx {
    pub hostname: String,
    pub port: u16,
}

/// Serve one request.
pub(super) async fn serve(
    req: Request<Incoming>,
    ctx: Arc<Ctx>,
) -> Result<Response<Body>, Infallible> {
    let hostname = ctx.hostname.as_str();
    if guard::enforced(hostname) {
        if let Err(reason) = guard::check(req.headers(), hostname, ctx.port) {
            return Ok(to_response(ApiResponse::text(403, reason), hostname));
        }
    }
    if req.method() == Method::OPTIONS {
        return Ok(to_response(ApiResponse::empty(200), hostname));
    }
    let path = req.uri().path().to_string();
    if path == "/api/events" {
        return Ok(event_stream(hostname));
    }
    let method = req.method().as_str().to_string();
    let target = req
        .uri()
        .path_and_query()
        .map_or_else(|| path.clone(), |pq| pq.as_str().to_string());
    let headers = req.headers().clone();
    let joined = tokio::task::spawn_blocking(move || {
        if path.starts_with("/api/") {
            handle_api(&method, &target, &headers)
        } else {
            Ok(handle_static(&path))
        }
    })
    .await;
    let api = match joined {
        Ok(Ok(api)) => api,
        Ok(Err(error)) => failure(&error.to_string()),
        Err(join) => failure(&format!("handler panicked: {join}")),
    };
    Ok(to_response(api, hostname))
}

fn failure(message: &str) -> ApiResponse {
    ApiResponse::error(500, message)
}

/// SSE body fed by a task that owns the live-status subscription; dropping the
/// body (client disconnect) aborts the task, which drops the subscription.
struct SseBody {
    rx: mpsc::Receiver<Bytes>,
    task: tokio::task::JoinHandle<()>,
}

impl hyper::body::Body for SseBody {
    type Data = Bytes;
    type Error = Infallible;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, Infallible>>> {
        self.rx
            .poll_recv(cx)
            .map(|chunk| chunk.map(|bytes| Ok(Frame::data(bytes))))
    }
}

impl Drop for SseBody {
    fn drop(&mut self) {
        self.task.abort();
    }
}

/// Server-sent events of `LiveStatus`: the current status immediately, then
/// every change. Ingest starts when the first page connects, not when the
/// server binds.
fn event_stream(hostname: &str) -> Response<Body> {
    let live = live::stats_live();
    live.start();
    let mut frames = live.subscribe();
    let first = format!(
        "data: {}\n\n",
        serde_json::to_string(&live.status()).unwrap_or_else(|_| "{}".to_string())
    );
    let (tx, rx) = mpsc::channel::<Bytes>(64);
    let task = tokio::spawn(async move {
        if tx.send(Bytes::from(first)).await.is_err() {
            return;
        }
        let mut heartbeat = tokio::time::interval(EVENT_HEARTBEAT);
        heartbeat.tick().await;
        loop {
            let chunk = tokio::select! {
                received = frames.recv() => match received {
                    Ok(frame) => format!("data: {}\n\n", frame.frame),
                    Err(broadcast::error::RecvError::Lagged(_)) => {
                        let status = serde_json::to_string(&live::stats_live().status()).unwrap_or_default();
                        format!("data: {status}\n\n")
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                },
                _ = heartbeat.tick() => ": keep-alive\n\n".to_string(),
                () = tx.closed() => break,
            };
            if tx.send(Bytes::from(chunk)).await.is_err() {
                break;
            }
        }
    });
    let mut response = Response::new(SseBody { rx, task }.boxed());
    let headers = response.headers_mut();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
    headers.insert(CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    headers.insert(
        hyper::header::CONNECTION,
        HeaderValue::from_static("keep-alive"),
    );
    stamp(headers, hostname);
    response
}
