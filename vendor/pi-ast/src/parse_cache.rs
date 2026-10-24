//! Bounded, content-addressed tree-sitter parse cache.
//!
//! Every structural entry point in this crate ([`crate::block`],
//! [`crate::summary`]) is dominated by one cost: `Parser::parse` over the whole
//! file. Measured on an M4 Max (release, warm page cache) that is ~13.5 ms for
//! an 81 KB / 2057-line TypeScript file and ~187 ms for a 1.05 MB file, while
//! everything else those functions do totals well under a millisecond.
//!
//! The results themselves are not cacheable: `enclosing_block_boundaries`
//! depends on the caller's visible `ranges`, which differ on every call. The
//! reusable artifact is the [`Tree`], so the cache lives here and hands out
//! cheap clones of it.
//!
//! `Tree` is `Send` but not `Sync`, so entries live behind a [`Mutex`] and the
//! lock is only ever held for a map probe, a byte comparison, and a
//! `ts_tree_copy` refcount bump — never across a parse or a tree walk.

use std::{
    collections::HashMap,
    sync::{LazyLock, Mutex, MutexGuard, PoisonError},
};

use anyhow::{Result, anyhow};
use tree_sitter::Tree;

use crate::language::{
    SupportLang,
    grammar::{self, GrammarError, LanguageGrammar},
};

/// Arbitrary fixed seed (golden-ratio constant). Fixed, not random, so a key is
/// reproducible across calls within a process; it never leaves the process, so
/// there is nothing to harden against `HashDoS` here.
const HASH_SEED: u64 = 0x9e37_79b9_7f4a_7c15;

/// Largest source that may occupy a slot.
///
/// A tree-sitter tree runs roughly an order of magnitude larger than its
/// source, so admitting an arbitrarily large file would let one `read` of a
/// multi-megabyte blob dominate process RSS. Files above this are still parsed,
/// just never retained.
pub const MAX_ENTRY_SOURCE_BYTES: usize = 4 << 20;

/// Ceiling on retained source bytes across all slots.
///
/// Equal to [`MAX_ENTRY_SOURCE_BYTES`] so a single hot large file can still be
/// cached (it evicts everything else, which is what LRU should do when that
/// file *is* the working set).
pub const MAX_TOTAL_SOURCE_BYTES: usize = 4 << 20;

/// Slot ceiling, independent of byte size.
///
/// Bounds the tree footprint against a burst of small files. Twelve covers the
/// realistic hot set for a coding agent (the handful of files being read and
/// edited) while keeping the LRU scan trivially cheap.
pub const MAX_ENTRIES: usize = 12;

/// Cache key. The 64-bit hash is a *bucket selector only*: a hit additionally
/// verifies [`Entry::source`] against the request byte-for-byte before the tree
/// is handed back, so a hash collision can only ever cost a re-parse (the
/// colliding slot is overwritten) and can never return a tree built from
/// different content. `len` is folded in because it is free and makes
/// accidental bucket sharing rarer; `lang` is in the key because the same bytes
/// parsed as TypeScript and as Python are different trees and must not share a
/// slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Key {
    hash: u64,
    len: usize,
    lang: SupportLang,
}

fn key_for(code: &str, lang: SupportLang) -> Key {
    Key {
        hash: xxhash_rust::xxh64::xxh64(code.as_bytes(), HASH_SEED),
        len: code.len(),
        lang,
    }
}

struct Entry {
    /// Retained verbatim so a hit is verified by comparison, not by trusting
    /// the hash.
    source: Box<str>,
    tree: Tree,
    /// Value of [`Cache::clock`] at last use; smallest wins eviction.
    stamp: u64,
}

struct Cache {
    entries: HashMap<Key, Entry>,
    source_bytes: usize,
    clock: u64,
    hits: u64,
    misses: u64,
    evictions: u64,
    max_entries: usize,
    max_total_bytes: usize,
    max_entry_bytes: usize,
}

impl Cache {
    fn new(max_entries: usize, max_total_bytes: usize, max_entry_bytes: usize) -> Self {
        Self {
            entries: HashMap::new(),
            source_bytes: 0,
            clock: 0,
            hits: 0,
            misses: 0,
            evictions: 0,
            max_entries,
            max_total_bytes,
            max_entry_bytes,
        }
    }

    fn get(&mut self, key: &Key, code: &str) -> Option<Tree> {
        self.clock += 1;
        let stamp = self.clock;
        let tree = match self.entries.get_mut(key) {
            Some(entry) if &*entry.source == code => {
                entry.stamp = stamp;
                // `ts_tree_copy`: an atomic refcount bump on immutable subtree
                // data, so the clone can be walked off-lock on any thread.
                entry.tree.clone()
            }
            _ => {
                self.misses += 1;
                return None;
            }
        };
        self.hits += 1;
        Some(tree)
    }

    fn insert(&mut self, key: Key, code: &str, tree: &Tree) {
        if code.len() > self.max_entry_bytes {
            return;
        }
        if let Some(previous) = self.entries.remove(&key) {
            self.source_bytes -= previous.source.len();
        }
        while self.entries.len() >= self.max_entries
            || self.source_bytes + code.len() > self.max_total_bytes
        {
            if !self.evict_oldest() {
                break;
            }
        }
        self.clock += 1;
        self.source_bytes += code.len();
        self.entries.insert(
            key,
            Entry {
                source: Box::from(code),
                tree: tree.clone(),
                stamp: self.clock,
            },
        );
    }

    /// Drop the least-recently-used slot. `false` when there was nothing left
    /// to drop, which is what terminates [`Self::insert`]'s eviction loop.
    fn evict_oldest(&mut self) -> bool {
        // Linear over at most `max_entries` slots: cheaper than maintaining an
        // intrusive LRU list at this size.
        let Some(&oldest) = self
            .entries
            .iter()
            .min_by_key(|(_, entry)| entry.stamp)
            .map(|(key, _)| key)
        else {
            return false;
        };
        if let Some(entry) = self.entries.remove(&oldest) {
            self.source_bytes -= entry.source.len();
            self.evictions += 1;
        }
        true
    }

    /// Drop every entry and zero the counters, preserving the configured bounds.
    fn clear(&mut self) {
        self.entries.clear();
        self.source_bytes = 0;
        self.clock = 0;
        self.hits = 0;
        self.misses = 0;
        self.evictions = 0;
    }
}

static CACHE: LazyLock<Mutex<Cache>> = LazyLock::new(|| {
    Mutex::new(Cache::new(
        MAX_ENTRIES,
        MAX_TOTAL_SOURCE_BYTES,
        MAX_ENTRY_SOURCE_BYTES,
    ))
});

/// Every critical section is a handful of infallible map operations plus a
/// refcount bump, so panicking while holding the lock is not reachable.
/// Recovering the guard anyway means a hypothetical panic could never escalate
/// into every later parse panicking.
fn lock() -> MutexGuard<'static, Cache> {
    CACHE.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Parse `code` as `lang`, reusing a cached [`Tree`] when the exact same bytes
/// were last parsed as the same language and have not been evicted.
///
/// Semantics match a bare `Parser::new()` / `set_language` / `parse` sequence
/// exactly: `Err` when the grammar fails to load, `Ok(None)` when `parse`
/// yields nothing or `lang`'s wasm grammar is not installed (callers treat it
/// like an unsupported language), `Ok(Some(tree))` otherwise. Trees carrying
/// syntax errors are
/// cached like any other — `has_error()` is a property of the tree, so callers
/// that reject on it reach the identical verdict from a cached tree, and
/// repeated "does this parse" probes over the same broken file get the speedup
/// too.
pub fn parse_cached(code: &str, lang: SupportLang) -> Result<Option<Tree>> {
    let key = key_for(code, lang);
    // Bound the guard to a `let` so it drops at the end of this statement: an
    // `if let` scrutinee would hold the lock across the early return.
    let cached = lock().get(&key, code);
    if let Some(tree) = cached {
        return Ok(Some(tree));
    }
    let language = match lang.grammar().load() {
        Ok(language) => language,
        Err(GrammarError::NotInstalled { .. }) => return Ok(None),
        Err(err) => return Err(err.into()),
    };
    let Some(tree) = grammar::parse(&language, code)
        .map_err(|err| anyhow!("Failed to load tree-sitter language: {err}"))?
    else {
        return Ok(None);
    };
    lock().insert(key, code, &tree);
    Ok(Some(tree))
}

/// Occupancy and counters, for diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ParseCacheStats {
    pub entries: usize,
    pub source_bytes: usize,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
}

pub fn parse_cache_stats() -> ParseCacheStats {
    let cache = lock();
    ParseCacheStats {
        entries: cache.entries.len(),
        source_bytes: cache.source_bytes,
        hits: cache.hits,
        misses: cache.misses,
        evictions: cache.evictions,
    }
}

/// Drop every cached tree and zero the counters.
pub fn clear_parse_cache() {
    lock().clear();
}
