//! Bounded, content-addressed caches with explicit separation between
//! the syntax layer ([`CstCache`], keyed by source content) and the
//! semantic layer ([`SemanticCache`], keyed by source content *and*
//! cfg context *and* macro environment).
//!
//! The two layers never share keys: a CST cache hit therefore cannot
//! suppress semantic re-evaluation when the cfg context or the macro
//! environment changed, because the semantic key includes both.
//!
//! The 64-bit content hashes used for keys are **not** collision-safe
//! on their own. Every hit is therefore verified against the full
//! stored identity — exact source bytes (never merely a length or
//! hash check) and, for the semantic and interface layers, the full
//! [`CfgContext`] — and a mismatch is treated as a miss-and-replace,
//! never as a hit.
//!
//! All caches are size-bounded with least-recently-used eviction, so
//! invalidation is bounded: evicted entries are simply re-derived on
//! next access.

use std::collections::HashMap;
use std::sync::Arc;

use crate::cfg::CfgContext;
use crate::cst::{parse_top_level, Cst};
use crate::hash::{hash_str, ContentHash};
use crate::iface::{project_interface, InterfaceProjection};
use crate::item::{cfg_hash, macro_env_hash, project, SemanticIndex};
use crate::loc::{loc_dep_hash, scan_loc_uses};

/// Cache hit/miss/eviction counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
}

struct Entry<V> {
    value: Arc<V>,
    /// Approximate retained size in bytes, for bounding.
    bytes: usize,
    /// Monotonic access tick for LRU eviction.
    tick: u64,
}

/// Content-addressed CST cache, bounded by total retained bytes.
pub struct CstCache {
    entries: HashMap<ContentHash, Entry<Cst>>,
    bytes: usize,
    max_bytes: usize,
    tick: u64,
    stats: CacheStats,
}

impl CstCache {
    /// A cache retaining at most `max_bytes` of source text.
    #[must_use]
    pub fn new(max_bytes: usize) -> Self {
        Self {
            entries: HashMap::new(),
            bytes: 0,
            max_bytes,
            tick: 0,
            stats: CacheStats::default(),
        }
    }

    /// Current counters.
    #[must_use]
    pub fn stats(&self) -> CacheStats {
        self.stats
    }

    /// Number of cached entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Drop the entry for a source content hash, if present.
    pub fn invalidate(&mut self, source: ContentHash) {
        if let Some(e) = self.entries.remove(&source) {
            self.bytes = self.bytes.saturating_sub(e.bytes);
        }
    }

    /// Look up or parse a source file. On a hit the parse is skipped
    /// entirely. Hits are verified by exact source-byte equality (the
    /// 64-bit key hash alone is not collision-safe): a hash collision
    /// behaves as a miss-and-replace rather than returning wrong data.
    pub fn get_or_parse(&mut self, src: &str) -> Arc<Cst> {
        let key = hash_str(src);
        self.tick += 1;
        let tick = self.tick;
        if let Some(e) = self.entries.get_mut(&key) {
            if e.value.source.as_ref() == src {
                e.tick = tick;
                self.stats.hits += 1;
                return Arc::clone(&e.value);
            }
        }
        self.stats.misses += 1;
        let cst = Arc::new(parse_top_level(src));
        let bytes = cst.source_len + cst.tokens.len() * std::mem::size_of::<crate::lex::Token>();
        if let Some(old) = self.entries.insert(
            key,
            Entry {
                value: cst.clone(),
                bytes,
                tick,
            },
        ) {
            self.bytes = self.bytes.saturating_sub(old.bytes);
        }
        self.bytes += bytes;
        self.evict_if_needed();
        cst
    }

    fn evict_if_needed(&mut self) {
        while self.bytes > self.max_bytes && !self.entries.is_empty() {
            let Some((&victim, _)) = self.entries.iter().min_by_key(|(_, e)| e.tick) else {
                break;
            };
            if let Some(e) = self.entries.remove(&victim) {
                self.bytes = self.bytes.saturating_sub(e.bytes);
                self.stats.evictions += 1;
            }
        }
    }

    /// Insert `cst` under an arbitrary key, bypassing hashing. Exists
    /// so tests can simulate a hash collision (equal keys, different
    /// source bytes) without finding a real 64-bit collision.
    #[doc(hidden)]
    pub fn force_insert_for_test(&mut self, key: ContentHash, cst: Cst) {
        let bytes = cst.source_len + cst.tokens.len() * std::mem::size_of::<crate::lex::Token>();
        self.tick += 1;
        if let Some(old) = self.entries.insert(
            key,
            Entry {
                value: Arc::new(cst),
                bytes,
                tick: self.tick,
            },
        ) {
            self.bytes = self.bytes.saturating_sub(old.bytes);
        }
        self.bytes += bytes;
        self.evict_if_needed();
    }
}

/// Semantic-layer cache key: source content, cfg context, and macro
/// environment are all part of the identity.
type SemanticKey = (ContentHash, ContentHash, ContentHash);

/// A cache entry that carries its full verification identity: the
/// exact source bytes and the full cfg context the value was derived
/// from. Hits compare both (the key hashes alone are not
/// collision-safe); a mismatch is a miss-and-replace.
struct VerifiedEntry<V> {
    value: Arc<V>,
    source: Arc<str>,
    ctx: CfgContext,
    /// Monotonic access tick for LRU eviction.
    tick: u64,
}

impl<V> VerifiedEntry<V> {
    /// Whether this entry may answer for `cst` + `ctx`.
    fn verifies(&self, cst: &Cst, ctx: &CfgContext) -> bool {
        self.source.as_ref() == cst.source.as_ref() && self.ctx == *ctx
    }
}

/// Content-addressed semantic index cache, bounded by entry count.
pub struct SemanticCache {
    entries: HashMap<SemanticKey, VerifiedEntry<SemanticIndex>>,
    max_entries: usize,
    tick: u64,
    stats: CacheStats,
}

impl SemanticCache {
    /// A cache retaining at most `max_entries` projections.
    #[must_use]
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: HashMap::new(),
            max_entries,
            tick: 0,
            stats: CacheStats::default(),
        }
    }

    /// Current counters.
    #[must_use]
    pub fn stats(&self) -> CacheStats {
        self.stats
    }

    /// Number of cached entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Drop every projection derived from the given source content.
    pub fn invalidate_source(&mut self, source: ContentHash) {
        self.entries.retain(|(s, _, _), _| *s != source);
    }

    /// The key under which `cst` + `ctx` would be cached.
    #[must_use]
    pub fn key_for(cst: &Cst, ctx: &CfgContext) -> (ContentHash, ContentHash, ContentHash) {
        (cst.source_hash, cfg_hash(ctx), macro_env_hash(cst))
    }

    /// Look up or compute the semantic projection of a CST under a
    /// cfg context. A CST cache hit with a changed cfg context is a
    /// *miss* here by construction. Hits are verified against the
    /// exact source bytes and the full [`CfgContext`] (not just the
    /// key hashes): a hash collision is a miss-and-replace.
    pub fn get_or_project(&mut self, cst: &Cst, ctx: &CfgContext) -> Arc<SemanticIndex> {
        let key = Self::key_for(cst, ctx);
        self.tick += 1;
        let tick = self.tick;
        if let Some(e) = self.entries.get_mut(&key) {
            if e.verifies(cst, ctx) {
                e.tick = tick;
                self.stats.hits += 1;
                return Arc::clone(&e.value);
            }
        }
        self.stats.misses += 1;
        let index = Arc::new(project(cst, ctx));
        self.insert_verified(key, Arc::clone(&cst.source), ctx.clone(), index.clone(), tick);
        index
    }

    fn insert_verified(
        &mut self,
        key: SemanticKey,
        source: Arc<str>,
        ctx: CfgContext,
        value: Arc<SemanticIndex>,
        tick: u64,
    ) {
        self.entries.insert(
            key,
            VerifiedEntry {
                value,
                source,
                ctx,
                tick,
            },
        );
        while self.entries.len() > self.max_entries && !self.entries.is_empty() {
            let Some((&victim, _)) = self.entries.iter().min_by_key(|(_, e)| e.tick) else {
                break;
            };
            self.entries.remove(&victim);
            self.stats.evictions += 1;
        }
    }

    /// Insert an entry under the key of `key_cst` + `key_ctx` while
    /// storing a *different* verification identity, simulating a hash
    /// collision for regression tests.
    #[doc(hidden)]
    pub fn force_insert_for_test(
        &mut self,
        key_cst: &Cst,
        key_ctx: &CfgContext,
        stored_source: Arc<str>,
        stored_ctx: CfgContext,
        index: SemanticIndex,
    ) {
        let key = Self::key_for(key_cst, key_ctx);
        self.tick += 1;
        let tick = self.tick;
        self.insert_verified(key, stored_source, stored_ctx, Arc::new(index), tick);
    }
}

/// Interface-layer cache key: source content, cfg context, macro
/// environment, workspace digest, and the observable-location
/// dependency hash are all part of the identity. The `loc_dep`
/// component means adding, removing, or relocating a
/// `file!`/`line!`/`column!` use is a cache miss by construction —
/// an observable use is never silently relocated under a warm key.
type InterfaceKey = (
    ContentHash,
    ContentHash,
    ContentHash,
    ContentHash,
    ContentHash,
);

/// Content-addressed interface-projection cache, bounded by entry
/// count with LRU eviction. Hits are verified against the exact
/// source bytes and the full [`CfgContext`]; the workspace digest is
/// caller-supplied and part of the key.
pub struct InterfaceCache {
    entries: HashMap<InterfaceKey, VerifiedEntry<InterfaceProjection>>,
    max_entries: usize,
    tick: u64,
    stats: CacheStats,
}

impl InterfaceCache {
    /// A cache retaining at most `max_entries` projections.
    #[must_use]
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: HashMap::new(),
            max_entries,
            tick: 0,
            stats: CacheStats::default(),
        }
    }

    /// Current counters.
    #[must_use]
    pub fn stats(&self) -> CacheStats {
        self.stats
    }

    /// Number of cached entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Drop every projection derived from the given source content.
    pub fn invalidate_source(&mut self, source: ContentHash) {
        self.entries.retain(|(s, _, _, _, _), _| *s != source);
    }

    /// The key under which `cst` + `ctx` + `workspace` would be
    /// cached. The observable-location dependency is recomputed from
    /// the current token stream (a cheap scan) so a stale `loc_dep`
    /// can never mask a relocation.
    #[must_use]
    pub fn key_for(cst: &Cst, ctx: &CfgContext, workspace: ContentHash) -> InterfaceKey {
        (
            cst.source_hash,
            cfg_hash(ctx),
            macro_env_hash(cst),
            workspace,
            loc_dep_hash(&scan_loc_uses(cst)),
        )
    }

    /// Look up or compute the interface projection. Any change to
    /// source, cfg context, macro environment, workspace, or the
    /// observable-location dependency set is a miss. Hits are
    /// verified against the exact source bytes and the full
    /// [`CfgContext`] (not just the key hashes): a hash collision is
    /// a miss-and-replace.
    pub fn get_or_project(
        &mut self,
        cst: &Cst,
        ctx: &CfgContext,
        workspace: ContentHash,
    ) -> Arc<InterfaceProjection> {
        let key = Self::key_for(cst, ctx, workspace);
        self.tick += 1;
        let tick = self.tick;
        if let Some(e) = self.entries.get_mut(&key) {
            if e.verifies(cst, ctx) {
                e.tick = tick;
                self.stats.hits += 1;
                return Arc::clone(&e.value);
            }
        }
        self.stats.misses += 1;
        let index = project(cst, ctx);
        let proj = Arc::new(project_interface(&index, cst, workspace));
        self.entries.insert(
            key,
            VerifiedEntry {
                value: proj.clone(),
                source: Arc::clone(&cst.source),
                ctx: ctx.clone(),
                tick,
            },
        );
        while self.entries.len() > self.max_entries && !self.entries.is_empty() {
            let Some((&victim, _)) = self.entries.iter().min_by_key(|(_, e)| e.tick) else {
                break;
            };
            self.entries.remove(&victim);
            self.stats.evictions += 1;
        }
        proj
    }
}
