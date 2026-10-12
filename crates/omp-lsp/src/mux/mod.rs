//! Broker-shared language servers: one mux daemon per project keeps one
//! language-server process per (command, args, cwd, env) key, multiplexing
//! every omp-rs session of the same user over a local socket
//! (omp `src/lsp/mux`, v18.6.1).
//!
//! - [`protocol`]: shared wire contract (byte-compatible with the TS daemon).
//! - [`server`]: the mux daemon itself (also the `omp-lsp-mux` binary).
//! - [`daemon`]: session side — ensure the daemon, dial, `omp/muxConnect`
//!   handshake, then treat the socket like a stdio transport.
//!
//! Sharing is opt-in exactly like TS: `sharedLspEnabled` starts false and the
//! host sets it from the `lsp.shared` setting (default on) at session
//! creation; any shared-path failure falls back to a private spawn.

use std::sync::atomic::{AtomicBool, Ordering};

pub mod protocol;

#[cfg(unix)]
pub mod daemon;
#[cfg(unix)]
pub mod server;

#[cfg(unix)]
pub use daemon::connect_shared_lsp_transport;
#[cfg(unix)]
pub use daemon::MuxTransport;

static SHARED_LSP_ENABLED: AtomicBool = AtomicBool::new(false);

/// Enable or disable attaching to broker-shared language servers
/// (omp `setSharedLspEnabled`). Hosts call this from the `lsp.shared`
/// setting at session creation; defaults to false like the TS module flag.
pub fn set_shared_lsp_enabled(enabled: bool) {
    SHARED_LSP_ENABLED.store(enabled, Ordering::SeqCst);
}

/// Whether broker-shared servers are currently enabled.
#[must_use]
pub fn shared_lsp_enabled() -> bool {
    SHARED_LSP_ENABLED.load(Ordering::SeqCst)
}
