//! Durable agent-to-agent hub for omt (`runs/a2a-hub-20261026/SPEC.md`).
//!
//! One typed [`Envelope`] for every channel; a SQLite-WAL [`store`] that
//! survives restarts and dedups on `message_id`; a tokio UDS JSON-RPC
//! [`server`] pushing `deliver` notifications with urgent priority first over
//! bounded per-endpoint channels; [`pane`] delivery into herdr panes
//! (`w4:pXX`) with the existing 4-line `[FROM]/[ID]/[POINTER]/[EXPECTED]`
//! head; and a [`client`] library with connect(auto-start) used by
//! `omt hub` and the task tool.

pub mod client;
pub mod envelope;
pub mod pane;
pub mod schema;
pub mod server;
pub mod store;

pub use client::{hub_dir_from, DeliverStream, HubClient, SendReceipt};
pub use envelope::{
    answer_envelope, looks_like_pane, new_message_id, normalize_address, pane_text, Envelope, Kind,
    PayloadRef, Priority, State, DEFAULT_REPLY_DEADLINE_MS, KIND_ANSWER, PANE_WAKE_TOOL,
    SENDER_HUB,
};
pub use schema::{validate_reply, ReplyValidation};
pub use server::{
    idle_exit_from_env, serve, serve_with, HubDir, ServeExit, DEFAULT_IDLE_EXIT, IDLE_EXIT_ENV,
};
pub use store::{
    ClaimOutcome, ClaimRow, EnvelopeRow, InsertOutcome, PendingReply, StatusReport, Store,
};

/// Default hub directory (`~/.omp/hub`), overridable with `OMP_HUB_DIR`.
#[must_use]
pub fn default_dir() -> std::path::PathBuf {
    if let Some(d) = std::env::var_os("OMP_HUB_DIR") {
        return std::path::PathBuf::from(d);
    }
    if let Some(home) = std::env::var_os("HOME") {
        return std::path::PathBuf::from(home).join(".omp").join("hub");
    }
    std::env::temp_dir().join("omp-hub")
}

/// Socket path: `OMP_HUB_SOCK` wins, else `<dir>/hub.sock`.
#[must_use]
pub fn socket_path(dir: &std::path::Path) -> std::path::PathBuf {
    if let Some(s) = std::env::var_os("OMP_HUB_SOCK") {
        return std::path::PathBuf::from(s);
    }
    dir.join("hub.sock")
}

/// Current unix time in milliseconds.
#[must_use]
pub fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
}
