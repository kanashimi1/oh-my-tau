//! Composite cycle keys and content-anchored negative constraints for
//! fix-cycle caching.
//!
//! A cycle entry is identified by exactly
//! `(workspace_digest, active_subgoal_id, normalized_error_sig)`:
//! identical error text in different workspaces or under different
//! subgoals can never collide. The error signature is normalized
//! (paths reduced to basenames, line/column suffixes folded, whitespace
//! collapsed) so cosmetic differences do not fragment the cache. Error
//! codes and numeric Rust semantics are preserved.
//!
//! Negative constraints ("no fix is known for this error in this
//! context") are content-anchored: each records the anchor (e.g. an
//! [`crate::iface::InterfaceProjection`] content hash) it was
//! derived from, and any change to that anchor invalidates it — an
//! added matching item therefore revives the cycle instead of
//! serving a stale "unfixable".

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::hash::{hash_str, ContentHash};

/// Normalize a raw compiler error message into a stable signature:
/// absolute or relative path tokens are reduced to their basename,
/// numeric runs immediately following `:` are folded to `0` for
/// line/column suffixes, and whitespace is collapsed to single spaces.
/// Error-code suffixes, integer types and other semantic digits are kept.
#[must_use]
pub fn normalize_error_sig(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut first = true;
    for tok in raw.split_whitespace() {
        if !first {
            out.push(' ');
        }
        first = false;
        let base = tok.rsplit(['/', '\\']).next().unwrap_or(tok);
        let mut chars = base.chars().peekable();
        while let Some(c) = chars.next() {
            out.push(c);
            if c == ':' && chars.peek().is_some_and(char::is_ascii_digit) {
                out.push('0');
                while chars.peek().is_some_and(char::is_ascii_digit) {
                    chars.next();
                }
            }
        }
    }
    out
}

/// The composite identity of a fix cycle. All three components are
/// part of every cache key derived from it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CycleKey {
    /// Digest of the workspace the cycle runs in.
    pub workspace_digest: ContentHash,
    /// The active subgoal within the workspace.
    pub active_subgoal_id: u64,
    /// Hash of the normalized error signature.
    pub normalized_error_sig: ContentHash,
}

/// Build a cycle key from its raw parts.
#[must_use]
pub fn cycle_key(workspace: &str, active_subgoal_id: u64, raw_error: &str) -> CycleKey {
    CycleKey {
        workspace_digest: hash_str(workspace),
        active_subgoal_id,
        normalized_error_sig: hash_str(&normalize_error_sig(raw_error)),
    }
}

/// A content-anchored negative constraint: "no entry exists for
/// `key` with `missing` absent, as of `anchor`".
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NegativeConstraint {
    /// The cycle key the constraint was recorded under.
    pub key: CycleKey,
    /// Hash of the missing name/fact.
    pub missing: ContentHash,
    /// Content anchor at record time; a change invalidates.
    pub anchor: ContentHash,
}

/// Cycle cache with positive entries and content-anchored negative
/// constraints, keyed by the full [`CycleKey`].
pub struct CycleStore<V> {
    positive: HashMap<CycleKey, V>,
    negative: Vec<NegativeConstraint>,
}

impl<V> Default for CycleStore<V> {
    fn default() -> Self {
        Self {
            positive: HashMap::new(),
            negative: Vec::new(),
        }
    }
}

impl<V> CycleStore<V> {
    /// An empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Look up a positive entry by its full composite key.
    #[must_use]
    pub fn get(&self, key: &CycleKey) -> Option<&V> {
        self.positive.get(key)
    }

    /// Insert a positive entry; any negative constraint for the same
    /// key and the entry's identity is dropped (the negative answer
    /// is disproven).
    pub fn insert(&mut self, key: CycleKey, value: V) {
        self.negative.retain(|n| n.key != key);
        self.positive.insert(key, value);
    }

    /// Record a content-anchored negative constraint. `missing` is
    /// the name/fact whose absence was verified; `anchor` is the
    /// content hash of the projection it was verified against.
    pub fn record_negative(&mut self, key: CycleKey, missing: &str, anchor: ContentHash) {
        self.negative.push(NegativeConstraint {
            key,
            missing: hash_str(missing),
            anchor,
        });
    }

    /// Whether a negative constraint still vouches for `missing`
    /// under `key` at the current `anchor`. Only a constraint with
    /// the *same* key and the *same* anchor answers `true`; a changed
    /// anchor (e.g. an item was added) invalidates by construction.
    #[must_use]
    pub fn negative_holds(&self, key: &CycleKey, missing: &str, anchor: ContentHash) -> bool {
        let m = hash_str(missing);
        self.negative
            .iter()
            .any(|n| n.key == *key && n.missing == m && n.anchor == anchor)
    }

    /// Drop every entry and constraint for a workspace (e.g. on
    /// workspace reload).
    pub fn invalidate_workspace(&mut self, workspace_digest: ContentHash) {
        self.positive.retain(|k, _| k.workspace_digest != workspace_digest);
        self.negative
            .retain(|n| n.key.workspace_digest != workspace_digest);
    }

    /// Number of positive entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.positive.len()
    }

    /// Whether the store is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.positive.is_empty() && self.negative.is_empty()
    }
}
