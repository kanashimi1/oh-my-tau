//! `startServer`: bind (or reuse / reclaim) the dashboard port.

use std::collections::HashMap;
use std::io::ErrorKind;
use std::net::TcpListener as StdListener;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex};

use super::conn::{serve_connection, Timeouts};
use super::http::Ctx;
use crate::live;
use crate::port_conflict::{normalize_host, prepare_stats_port, recover_stats_port, PortAction};
use anyhow::{anyhow, Context as _, Result};

/// Default dashboard port (TS `startServer(port = 3847)`).
pub const DEFAULT_PORT: u16 = 3847;

/// Format a dashboard origin, including brackets required by IPv6 literals.
#[must_use]
pub fn format_stats_dashboard_url(hostname: &str, port: u16) -> String {
    let host = normalize_host(hostname);
    let host = if host.contains(':') {
        format!("[{host}]")
    } else {
        host.to_string()
    };
    format!("http://{host}:{port}")
}

/// Options for [`start_server`]. The TS `judge` provider (paid LLM frustration
/// judge) is out of scope for this port.
#[derive(Debug, Clone, Default)]
pub struct StartServerOptions {
    /// Max time to receive request headers (default 30s).
    pub header_read_timeout: Option<std::time::Duration>,
    /// Close non-SSE connections idle this long (default 60s).
    pub idle_timeout: Option<std::time::Duration>,
}

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
static ACTIVE_SERVERS: LazyLock<Mutex<HashMap<String, StatsServerHandle>>> =
    LazyLock::new(Mutex::default);
/// Dashboards bound by this process (any port); background ingest stops when the last one does.
static LIVE_SERVERS: AtomicUsize = AtomicUsize::new(0);

impl StatsServerHandle {
    fn reused(hostname: &str, port: u16) -> Self {
        Self {
            hostname: hostname.to_string(),
            port,
            inner: None,
        }
    }

    /// Whether this handle points at a dashboard owned by another process.
    #[must_use]
    pub fn is_reused(&self) -> bool {
        self.inner.is_none()
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
fn create_dashboard_server(
    port: u16,
    hostname: &str,
    timeouts: Timeouts,
) -> std::io::Result<(tokio::runtime::Runtime, u16)> {
    let listener = StdListener::bind((hostname, port))?;
    listener.set_nonblocking(true)?;
    let actual = listener.local_addr()?.port();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .thread_name("omp-stats-http")
        .enable_all()
        .build()?;
    let ctx = Arc::new(Ctx {
        hostname: hostname.to_string(),
        port: actual,
    });
    runtime.spawn(async move {
        let Ok(listener) = tokio::net::TcpListener::from_std(listener) else {
            return;
        };
        loop {
            match listener.accept().await {
                Ok((stream, _)) => {
                    tokio::spawn(serve_connection(stream, Arc::clone(&ctx), timeouts));
                }
                Err(error) => {
                    // EMFILE and friends: back off instead of spinning on the error.
                    tracing::warn!(error = %error, "Stats dashboard accept failed");
                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                }
            }
        }
    });
    Ok((runtime, actual))
}

/// Start the HTTP server, reusing a live dashboard or reclaiming a stale omp listener.
///
/// # Errors
/// Port preflight/reclaim failure, or the bind failing even after reclaiming.
pub fn start_server(
    port: u16,
    hostname: &str,
    options: StartServerOptions,
) -> Result<StatsServerHandle> {
    // One form everywhere: the bind, the reuse key, the identity header and the probe.
    let hostname = normalize_host(hostname);
    let key = format!("{hostname}:{port}");
    let defaults = Timeouts::default();
    let timeouts = Timeouts {
        header_read: options.header_read_timeout.unwrap_or(defaults.header_read),
        idle: options.idle_timeout.unwrap_or(defaults.idle),
    };
    if port != 0 {
        if let Some(active) = ACTIVE_SERVERS
            .lock()
            .ok()
            .and_then(|m| m.get(&key).cloned())
        {
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
            inner: Some(Arc::new(Inner {
                key: key.clone(),
                runtime: Mutex::new(Some(runtime)),
            })),
        };
        if port != 0 {
            if let Ok(mut active) = ACTIVE_SERVERS.lock() {
                active.insert(key.clone(), handle.clone());
            }
        }
        handle
    };
    match create_dashboard_server(port, hostname, timeouts) {
        Ok(server) => Ok(register(server)),
        Err(error) if error.kind() == ErrorKind::AddrInUse => {
            if recover_stats_port(port, hostname)? == PortAction::Reuse {
                return Ok(StatsServerHandle::reused(hostname, port));
            }
            create_dashboard_server(port, hostname, timeouts)
                .map(register)
                .map_err(|retry| {
                    anyhow!(
                        "Failed to start stats dashboard on {hostname}:{port} after reclaiming it."
                    )
                    .context(retry)
                })
        }
        Err(error) => {
            Err(error).with_context(|| format!("bind stats dashboard on {hostname}:{port}"))
        }
    }
}
