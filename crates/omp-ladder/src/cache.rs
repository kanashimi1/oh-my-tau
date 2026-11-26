//! Bounded, content-addressed caches with explicit separation between
//! the syntax layer ([`CstCache`], keyed by source content) and the
//! semantic layer ([`SemanticCache`], keyed by source content *and*
//! cfg context *and* macro environment).
//!
//! The two layers never share keys: a CST cache hit therefore cannot
//! suppress semantic re-evaluation when the cfg context or the macro
//! environment changed, because the semantic key includes both.
//!
//! Both caches are size-bounded with least-recently-used eviction, so
//! invalidation is bounded: evicted entries are simply re-derived on
//! next access.

use std::collections::HashMap;
use std::sync::Arc;

use crate::cfg::CfgContext;
use crate::cst::{parse_top_level, Cst};
use crate::hash::{hash_str, ContentHash};
use crate::item::{cfg_hash, macro_env_hash, project, SemanticIndex};

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
    /// entirely; the length guard makes a hash collision behave as a
    /// miss-and-replace rather than returning wrong data.
    pub fn get_or_parse(&mut self, src: &str) -> Arc<Cst> {
        let key = hash_str(src);
        self.tick += 1;
        let tick = self.tick;
        if let Some(e) = self.entries.get_mut(&key) {
            if e.value.source_len == src.len() {
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
}

/// Semantic-layer cache key: source content, cfg context, and macro
/// environment are all part of the identity.
type SemanticKey = (ContentHash, ContentHash, ContentHash);

/// Content-addressed semantic index cache, bounded by entry count.
pub struct SemanticCache {
    entries: HashMap<SemanticKey, Entry<SemanticIndex>>,
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
    /// *miss* here by construction.
    pub fn get_or_project(&mut self, cst: &Cst, ctx: &CfgContext) -> Arc<SemanticIndex> {
        let key = Self::key_for(cst, ctx);
        self.tick += 1;
        let tick = self.tick;
        if let Some(e) = self.entries.get_mut(&key) {
            e.tick = tick;
            self.stats.hits += 1;
            return Arc::clone(&e.value);
        }
        self.stats.misses += 1;
        let index = Arc::new(project(cst, ctx));
        self.entries.insert(
            key,
            Entry {
                value: index.clone(),
                bytes: 0,
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
        index
    }
}
