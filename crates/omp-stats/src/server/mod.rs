//! Port of `server.ts`: the dashboard HTTP server.
//!
//! * [`handle_api`] — pure(ish) router mirroring TS `handleApi`, route by route.
//! * [`statics`] — the checked-in React bundle (`client/`) served byte-for-byte
//!   with the TS SPA fallback.
//! * [`http`] — hyper 1 adapter: dashboard identity headers, OPTIONS, SSE
//!   `/api/events`, 500 mapping.
//! * [`start`] — `start_server` / `StatsServerHandle` with port-conflict
//!   reuse/reclaim and the in-process active-server registry.

mod api;
mod conn;
mod guard;
mod http;
mod query;
mod session;
mod start;
mod statics;

#[cfg(test)]
mod conn_tests;
#[cfg(test)]
mod roots_tests;
#[cfg(test)]
mod tests;

pub use api::{handle_api, ApiResponse, STATS_ACTION_HEADER};
pub use start::{
    format_stats_dashboard_url, start_server, StartServerOptions, StatsServerHandle, DEFAULT_PORT,
};
pub use statics::handle_static;
