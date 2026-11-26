//! Re-exports and crate docs.
//!
//! # omp-cas
//!
//! Content-addressed store for omt with explicit object lifetimes:
//!
//! * [`Store::put_scratch`] / [`Store::put_durable`] — the tier is explicit
//!   at the call site; nothing promotes scratch data implicitly.
//! * [`Store::promote`] — the one deliberate scratch → durable move.
//! * [`Store::get`] / [`Store::read_into`] — verified reads (every byte is
//!   re-hashed against its digest before it leaves the store).
//! * [`Store::build_pack`] / [`Store::import_pack`] — transitive dependency
//!   packs with cycle-safe traversal and bounded import.
//! * [`Store::set_scratch_budget`] — byte-budget LRU eviction for scratch.

pub mod cache;
pub mod depgraph;
pub mod digest;
pub mod error;
pub mod object;
pub mod pack;
pub mod store;

pub use cache::{CachePolicy, ScratchStats};
pub use depgraph::DepGraph;
pub use digest::Digest;
pub use error::{Error, Result};
pub use object::{ObjectId, Tier};
pub use store::Store;
