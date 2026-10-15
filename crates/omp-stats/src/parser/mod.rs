//! Port of `stats/src/parser.ts` (+ `sync-worker.ts`, which is synchronous
//! in-process work here — no worker thread). One module per concern:
//!
//! - [`state`] — the parser ↔ ingest contract (integrator-owned).
//! - [`wyhash`] — `Bun.hash` (wyhash final4) for parser checkpoints.
//! - [`tier`] — the pi-ai service-tier helpers `parser.ts` uses.
//! - [`util`] — JS `Date.parse` / `String.length` semantics.
//! - [`entries`] — entry guards and stat extractors.
//! - [`session`] — file listing, byte-level scanning, `parse_session_file`.

pub mod entries;
pub mod session;
pub mod state;
pub mod tier;
pub mod util;
pub mod wyhash;

pub use session::{
    classify_agent_type, extract_folder_from_path, get_session_entry, list_all_session_files,
    list_session_files, list_session_folders, matches_session_file, parse_all_session_entries,
    parse_session_file,
};
pub use state::{ParseSessionResult, SessionParserState};

pub use entries::{resolve_usage_total, UsageBucketView};
pub use tier::{
    coerce_service_tier_by_family, get_premium_service_tier_requests, parse_service_tier,
    realizes_priority_service_tier, resolve_model_service_tier, service_tier_family,
};
pub use wyhash::{wyhash, wyhash_hex};

#[cfg(test)]
mod tests;
