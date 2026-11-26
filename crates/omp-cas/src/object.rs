//! Object layout and on-disk names shared by both tiers.

use std::path::PathBuf;

use crate::digest::Digest;

/// One content-addressed object.
///
/// Layout: 2-char hex prefix directory, then the 64-char lowercase-hex
/// digest as file name — `objects/ab/abcd…`. Git/CAS layout; the prefix
/// split keeps per-directory entry counts bounded. Every name component is
/// derived from a strictly validated [`Digest`], so the layout cannot be
/// coerced into a path traversal (`..`, absolute paths, separators are all
/// outside the hex alphabet).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ObjectId {
    digest: Digest,
    /// Content size in bytes, as stored/verified. Kept beside the object so
    /// pack manifests and byte budgets never re-read payloads to size them.
    size: u64,
}

impl ObjectId {
    /// Identify `bytes`.
    #[must_use]
    pub fn of(bytes: &[u8]) -> Self {
        Self {
            digest: Digest::of(bytes),
            size: bytes.len() as u64,
        }
    }

    /// Wrap a known digest and its content size.
    #[must_use]
    pub const fn from_digest(digest: Digest, size: u64) -> Self {
        Self { digest, size }
    }

    /// The validated digest.
    #[must_use]
    pub const fn digest(&self) -> Digest {
        self.digest
    }

    /// Content size in bytes.
    #[must_use]
    pub const fn size(&self) -> u64 {
        self.size
    }

    /// Canonical lowercase-hex form.
    #[must_use]
    pub fn hex(&self) -> String {
        self.digest.hex()
    }

    /// Store-relative path for the object file: `objects/ab/<64 hex>`.
    #[must_use]
    pub fn store_path(&self) -> PathBuf {
        let hex = self.digest.hex();
        PathBuf::from("objects").join(&hex[..2]).join(hex)
    }
}

/// Lifecycle tier of an object. The distinction is explicit at the API
/// surface: nothing in the crate promotes scratch data on its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Tier {
    /// Ephemeral: subject to LRU eviction when the scratch budget is hit.
    /// Every write to the scratch tier touches the recency clock.
    Scratch,
    /// Durable: never evicted; removed only by an explicit `remove`.
    Durable,
}

impl Tier {
    /// Directory under the store root holding this tier's `objects/` tree.
    #[must_use]
    pub const fn dir(self) -> &'static str {
        match self {
            Self::Scratch => "scratch",
            Self::Durable => "durable",
        }
    }
}
