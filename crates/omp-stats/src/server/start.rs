//! `startServer`: bind (or reuse / reclaim) the dashboard port.

use std::collections::HashMap;
use std::io::ErrorKind;
use std::net::TcpListener as StdListener;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex};

use anyhow::{anyhow, Context as _, Result};
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper_util::rt::TokioIo;

use super::http::serve;
use crate::live;
use crate::port_conflict::{prepare_stats_port, recover_stats_port, PortAction};

/// Default dashboard port (TS `startServer(port = 3847)`).
pub const DEFAULT_PORT: u16 = 3847;

/// Format a dashboard origin, including brackets required by IPv6 literals.
#[must_use]
pub fn format_stats_dashboard_url(hostname: &str, port: u16) -> String {
    let host = if hostname.contains(':') && !hostname.starts_with('[') { format!("[{hostname}]") } else { hostname.to_string() };
    format!("http://{host}:{port}")
}

/// Options for [`start_server`]. The TS `judge` provider (paid LLM frustration
/// judge) is out of scope for this port, so there is nothing to configure yet.
#[derive(Debug, Clone, Default)]
pub struct StartServerOptions {}

struct Inner {
    key: String,
    /// The server's own runtime: dropping it (via `shutdown_background`) closes
    /// the listener and every open connection, like `server.stop(true)`.
    runtime: Mutex<Option<tokio::runtime::Runtime>>,
}

/// A running (or reused) dashboard. Cloning shares the same server.
#[derive(Clone)]
pub struct StatsServerHandle {
    pub hostname: String,
    pub port: u16,
    inner: Option<Arc<Inner>>,
}

// Dashboards this process already bound, keyed by requested `hostname:port`.
// A second in-process start (e.g. `/trace` twice in one session) must return
// the live handle: probing our own port can time out under load.
static ACTIVE_SERVERS: LazyLock<Mutex<HashMap<String, StatsServerHandle>>> = LazyLock::new(Mutex::default);
/// Dashboards bound by this process (any port); background ingest stops when the last one does.
static LIVE_SERVERS: AtomicUsize = AtomicUsize::new(0);

impl StatsServerHandle {
    fn reused(hostname: &str, port: u16) -> Self {
        Self { hostname: hostname.to_string(), port, inner: None }
    }

    /// Stop the server. Idempotent; a reused (foreign) dashboard is left alone.
    pub fn stop(&self) {
        let Some(inner) = &self.inner else { return };
        if let Ok(mut active) = ACTIVE_SERVERS.lock() {
            active.remove(&inner.key);
        }
        let runtime = inner.runtime.lock().ok().and_then(|mut slot| slot.take());
        let Some(runtime) = runtime else { return };
        runtime.shutdown_background();
        // The last dashboard in this process takes background ingest down with it.
        if LIVE_SERVERS.fetch_sub(1, Ordering::SeqCst) == 1 {
            live::stats_live().stop();
        }
    }
}

/// Bind `hostname:port` and serve on a dedicated runtime. Blocking (port probe
/// up to ~0.5s): call from `spawn_blocking` inside async code.
fn create_dashboard_server(port: u16, hostname: &str) -> std::io::Result<(tokio::runtime::Runtime, u16)> {
    let bare = hostname.trim_start_matches('[').trim_end_matches(']');
    let listener = StdListener::bind((bare, port))?;
    listener.set_nonblocking(true)?;
    let actual = listener.local_addr()?.port();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .thread_name("omp-stats-http")
        .enable_all()
        .build()?;
    let host: Arc<str> = Arc::from(hostname);
    runtime.spawn(async move {
        let Ok(listener) = tokio::net::TcpListener::from_std(listener) else { return };
        loop {
            let Ok((stream, _)) = listener.accept().await else { continue };
            let host = Arc::clone(&host);
            tokio::spawn(async move {
                let service = service_fn(move |req| serve(req, Arc::clone(&host)));
                let _ = http1::Builder::new().serve_connection(TokioIo::new(stream), service).await;
            });
        }
    });
    Ok((runtime, actual))
}

/// Start the HTTP server, reusing a live dashboard or reclaiming a stale omp listener.
///
/// # Errors
/// Port preflight/reclaim failure, or the bind failing even after reclaiming.
pub fn start_server(port: u16, hostname: &str, _options: StartServerOptions) -> Result<StatsServerHandle> {
    let key = format!("{hostname}:{port}");
    if port != 0 {
        if let Some(active) = ACTIVE_SERVERS.lock().ok().and_then(|m| m.get(&key).cloned()) {
            return Ok(active);
        }
    }
    if prepare_stats_port(port, hostname)? == PortAction::Reuse {
        return Ok(StatsServerHandle::reused(hostname, port));
    }
    let register = |(runtime, actual): (tokio::runtime::Runtime, u16)| {
        LIVE_SERVERS.fetch_add(1, Ordering::SeqCst);
        let handle = StatsServerHandle {
            hostname: hostname.to_string(),
            port: actual,
            inner: Some(Arc::new(Inner { key: key.clone(), runtime: Mutex::new(Some(runtime)) })),
        };
        if port != 0 {
            if let Ok(mut active) = ACTIVE_SERVERS.lock() {
                active.insert(key.clone(), handle.clone());
            }
        }
        handle
    };
    match create_dashboard_server(port, hostname) {
        Ok(server) => Ok(register(server)),
        Err(error) if error.kind() == ErrorKind::AddrInUse => {
            if recover_stats_port(port, hostname)? == PortAction::Reuse {
                return Ok(StatsServerHandle::reused(hostname, port));
            }
            create_dashboard_server(port, hostname).map(register).map_err(|retry| {
                anyhow!("Failed to start stats dashboard on {hostname}:{port} after reclaiming it.").context(retry)
            })
        }
        Err(error) => Err(error).with_context(|| format!("bind stats dashboard on {hostname}:{port}")),
    }
}
