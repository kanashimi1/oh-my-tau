//! Rust port of `@oh-my-pi/omp-stats` (vendor-src/oh-my-pi/packages/stats/src).
//!
//! One module per TS file, same exported names in snake_case. The React
//! dashboard client is not ported: its bun-built bundle (`client/`) is served
//! as-is by `server`. Modules are feature-gated along the TS import graph so
//! each can be built alone; `default` enables everything.

#[cfg(feature = "aggregator")]
pub mod aggregator;
#[cfg(feature = "catalog")]
pub mod catalog;
#[cfg(feature = "db")]
pub mod db;
#[cfg(feature = "db_query")]
pub mod db_query;
#[cfg(feature = "frustration")]
pub mod frustration;
#[cfg(feature = "gain")]
pub mod gain;
#[cfg(feature = "live")]
pub mod live;
#[cfg(feature = "parser")]
pub mod parser;
pub mod paths;
#[cfg(feature = "port_conflict")]
pub mod port_conflict;
#[cfg(feature = "rollup")]
pub mod rollup;
#[cfg(feature = "server")]
pub mod server;
#[cfg(feature = "taxonomy")]
pub mod taxonomy;
#[cfg(all(
    test,
    any(
        feature = "parser",
        feature = "db_ingest",
        feature = "db_query",
        feature = "rollup"
    )
))]
pub(crate) mod testutil;
#[cfg(feature = "trace")]
pub mod trace;
#[cfg(feature = "types")]
pub mod types;
#[cfg(feature = "aggregator")]
pub mod usage_windows;
#[cfg(feature = "user_metrics")]
pub mod user_metrics;
