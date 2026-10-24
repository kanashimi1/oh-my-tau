//! TEMPORARY module list — T1-protocol local verification only.
//! T6-integration owns the final `lib.rs` (module decls + re-exports) and
//! will overwrite this file when wiring the crate together.

pub mod crypto;
pub mod link;
pub mod protocol;

// T2-relay: transport ports (relay-client.ts, collab-web/scripts/local-relay.ts).
pub mod relay_client;
pub mod relay_server;

// T4-guest: runtime local host registry (registry.ts port).
pub mod registry;

// T5-shrink: host→guest payload bounding (replication-shrink.ts port) plus
// the host.ts snapshot-chunk packing primitive.
pub mod replication;
// T3-host: host peer (host.ts port) + controller (controller.ts port).
pub mod controller;
pub mod host;

// Re-export the local relay + static client server for omp-cli's
// `serve-collab` subcommand (relay_server::serve_collab_client).
pub use relay_server::serve_collab_client;

// T4-guest: the guest side of a collab live session (guest.ts port).
pub mod guest;
