//! Stable, deterministic content hashing for cache keys.
//!
//! Uses `FxHasher` with no random seed so hashes are reproducible
//! across processes and runs. A 64-bit hash is **not** collision-safe
//! and is never used as proof of equality: the caches in
//! [`crate::cache`] verify every hit against the full stored identity
//! (exact source bytes, full [`crate::cfg::CfgContext`]), so a hash
//! collision degrades to a miss-and-replace, never to wrong data.

use std::hash::Hasher;

use rustc_hash::FxHasher;
use serde::{Deserialize, Serialize};

/// A deterministic 64-bit content hash.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ContentHash(pub u64);

/// Hash an arbitrary byte slice.
#[must_use]
pub fn hash_bytes(bytes: &[u8]) -> ContentHash {
    let mut h = FxHasher::default();
    h.write_usize(bytes.len());
    h.write(bytes);
    ContentHash(h.finish())
}

/// Hash a string.
#[must_use]
pub fn hash_str(s: &str) -> ContentHash {
    hash_bytes(s.as_bytes())
}
