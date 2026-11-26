//! Recency accounting and policy for the scratch tier.
//!
//! `ScratchClock` is a monotone counter bumped on every scratch touch; each
//! object stores the tick it was last touched at. Eviction sorts by that
//! tick — true LRU over touch time, no heap maintained on the hot path.

use std::collections::HashMap;

use parking_lot::Mutex;

use crate::digest::Digest;

/// Interior-mutable recency state for the scratch tier.
#[derive(Debug, Default)]
pub(crate) struct ScratchClock {
    inner: Mutex<ClockInner>,
}

#[derive(Debug, Default)]
struct ClockInner {
    tick: u64,
    /// digest -> last-touch tick.
    stamps: HashMap<Digest, u64>,
    /// Total content bytes currently in the scratch tier (verified writes
    /// only, so numbers match what is on disk).
    bytes: u64,
    /// Objects currently in the scratch tier.
    count: u64,
}

impl ScratchClock {
    /// Seed from a pre-existing scratch tier (reopen path): one stamp per
    /// object in scan order, after the current tick. Sizes are `stat`
    /// bookkeeping; reads still verify.
    pub(crate) fn seed(&self, entries: &[(Digest, u64)]) {
        let mut inner = self.inner.lock();
        for (digest, size) in entries {
            inner.tick = inner.tick.checked_add(1).unwrap_or(1);
            let tick = inner.tick;
            inner.stamps.insert(*digest, tick);
            inner.bytes += size;
            inner.count += 1;
        }
    }

    pub(crate) fn stamp_now(&self, digest: &Digest) -> u64 {
        let mut inner = self.inner.lock();
        inner.tick = inner.tick.checked_add(1).unwrap_or(1);
        let tick = inner.tick;
        inner.stamps.insert(*digest, tick);
        tick
    }

    pub(crate) fn touch_if_present(&self, digest: &Digest) {
        let mut inner = self.inner.lock();
        if inner.stamps.contains_key(digest) {
            inner.tick = inner.tick.checked_add(1).unwrap_or(1);
            let tick = inner.tick;
            inner.stamps.insert(*digest, tick);
        }
    }

    pub(crate) fn forget(&self, digest: &Digest) {
        let mut inner = self.inner.lock();
        inner.stamps.remove(digest);
    }

    /// All `(digest, tick)` pairs, ascending by tick (LRU order).
    pub(crate) fn lru_order(&self) -> Vec<(Digest, u64)> {
        let inner = self.inner.lock();
        let mut pairs: Vec<(Digest, u64)> = inner
            .stamps
            .iter()
            .map(|(d, t)| (*d, *t))
            .collect();
        pairs.sort_by_key(|(_, t)| *t);
        pairs
    }

    /// Current tracked scratch footprint (objects, bytes).
    pub(crate) fn tracked(&self) -> (u64, u64) {
        let inner = self.inner.lock();
        (inner.count, inner.bytes)
    }

    pub(crate) fn record_put(&self, size: u64) {
        let mut inner = self.inner.lock();
        inner.count += 1;
        inner.bytes += size;
    }

    pub(crate) fn record_remove(&self, size: u64) {
        let mut inner = self.inner.lock();
        inner.count = inner.count.saturating_sub(1);
        inner.bytes = inner.bytes.saturating_sub(size);
    }
}

/// Snapshot of scratch-tier accounting for tests and callers that display
/// store status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ScratchStats {
    /// Content bytes currently in the scratch tier.
    pub bytes: u64,
    /// Object count in the scratch tier.
    pub objects: u64,
    /// How many objects were evicted by the byte budget since store open.
    pub evictions: u64,
}

/// Scratch-tier policy knobs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CachePolicy {
    /// Evict least-recently-touched scratch objects until tracked bytes fit
    /// this budget. `None` disables budget eviction.
    pub max_bytes: Option<u64>,
    /// Reject (don't store) a single scratch object larger than this.
    /// `None` allows any size.
    pub max_object_bytes: Option<u64>,
}

impl CachePolicy {
    /// Policy with all bounds disabled (tests / explicit management).
    #[must_use]
    pub const fn unbounded() -> Self {
        Self {
            max_bytes: None,
            max_object_bytes: None,
        }
    }
}

impl Default for CachePolicy {
    fn default() -> Self {
        Self::unbounded()
    }
}
