//! Interface projection: the public surface of a semantic index,
//! organised into explicit coherence buckets, with content-anchored
//! negative guards.
//!
//! Coherence buckets ([`BucketKind`]) separate the namespaces that
//! matter for coherence reasoning: types, traits, impls, values,
//! macros. Only *unconditionally public* (`pub`, cfg-active, parsed)
//! items enter the name-keyed buckets; private items, `pub(...)`
//! restricted items, cfg-disabled items, and opaque macro calls never
//! leak into the global projection. `impl` blocks are unnamed but
//! coherence-relevant, so they enter the `Impls` bucket keyed by
//! content hash rather than by name.
//!
//! Fail-closed rule: a `pub use` can re-export arbitrary names that
//! this syntax-derived layer cannot enumerate (no name resolution),
//! so its presence marks the projection [`InterfaceProjection::incomplete`]
//! and all negative guards for it refuse to answer — an unknown is
//! never reported as an absent name.
//!
//! Negative guards ([`NegativeGuard`]) record "bucket B has no item
//! named N" anchored to the bucket's content hash at record time.
//! Adding any matching item changes that hash, so the guard is
//! invalidated by construction; guards also carry the workspace
//! digest so identical text in different workspaces cannot collide.

use std::fmt::Write as _;

use serde::{Deserialize, Serialize};

use crate::cst::{Cst, Visibility};
use crate::hash::{hash_str, ContentHash};
use crate::item::{Item, ItemId, ItemKind, ItemStatus, SemanticIndex};
use crate::loc::{loc_dep_hash, scan_loc_uses, LocUse};

/// A coherence bucket: one namespace of the public interface.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum BucketKind {
    /// `struct` / `enum` / `union` / `type` items (the type namespace).
    Types,
    /// `trait` items (coherence-relevant in their own right).
    Traits,
    /// `impl` blocks, keyed by content (unnamed but coherence-defining).
    Impls,
    /// `fn` / `const` / `static` items (the value namespace).
    Values,
    /// `pub macro` definitions.
    Macros,
    /// `pub mod` declarations.
    Modules,
}

/// One item projected into a bucket.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BucketEntry {
    /// Item id in the source semantic index.
    pub item: ItemId,
    /// Item name (`None` only for impl blocks).
    pub name: Option<String>,
    /// Item kind.
    pub kind: ItemKind,
    /// Hash of the item's full source text.
    pub content: ContentHash,
}

/// One coherence bucket with its content anchor.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Bucket {
    /// Which namespace this bucket holds.
    pub kind: BucketKind,
    /// Entries in source order.
    pub entries: Vec<BucketEntry>,
    /// Content anchor: hash over the bucket's entries. Any addition,
    /// removal, or edit of a member item changes it.
    pub content_hash: ContentHash,
}

/// The projected public interface of one source under one cfg
/// context, for one workspace.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct InterfaceProjection {
    /// Workspace digest the projection belongs to.
    pub workspace: ContentHash,
    /// Coherence buckets, sorted by [`BucketKind`].
    pub buckets: Vec<Bucket>,
    /// Observable `file!`/`line!`/`column!` uses found in the source.
    pub observable_locs: Vec<LocUse>,
    /// Hash of the observable-location dependency set; part of every
    /// cache key derived from this projection.
    pub loc_dep: ContentHash,
    /// True when the source contains a `pub use` whose re-exported
    /// names cannot be enumerated syntactically. Negative guards must
    /// fail closed (no answer) against an incomplete projection.
    pub incomplete: bool,
    /// Hash over the whole projection.
    pub content_hash: ContentHash,
}

impl InterfaceProjection {
    /// Look up a bucket by kind.
    #[must_use]
    pub fn bucket(&self, kind: BucketKind) -> Option<&Bucket> {
        self.buckets.iter().find(|b| b.kind == kind)
    }

    /// Whether a *named* public item with this name exists in the
    /// bucket.
    #[must_use]
    pub fn has_name(&self, kind: BucketKind, name: &str) -> bool {
        self.bucket(kind).is_some_and(|b| {
            b.entries.iter().any(|e| e.name.as_deref() == Some(name))
        })
    }

    /// Serialize into a write-once wire arena (metadata-free,
    /// lossless JSON).
    ///
    /// # Errors
    /// Propagates [`crate::wire::WireError`] from the arena.
    pub fn write_wire(&self, arena: &mut crate::wire::WireArena) -> Result<(), crate::wire::WireError> {
        let v = serde_json::to_value(self).unwrap_or(serde_json::Value::Null);
        arena.write_value(&v)
    }
}

fn bucket_of(item: &Item) -> Option<BucketKind> {
    Some(match item.kind {
        ItemKind::Struct | ItemKind::Enum | ItemKind::Union | ItemKind::TypeAlias => {
            BucketKind::Types
        }
        ItemKind::Trait => BucketKind::Traits,
        ItemKind::Impl => BucketKind::Impls,
        ItemKind::Fn | ItemKind::Const | ItemKind::Static => BucketKind::Values,
        ItemKind::MacroDef => BucketKind::Macros,
        ItemKind::Mod => BucketKind::Modules,
        // `use` (handled via `incomplete`), extern crates/blocks,
        // macro calls and unknowns have no place in a name-keyed
        // coherence projection.
        ItemKind::Use
        | ItemKind::ExternCrate
        | ItemKind::ExternBlock
        | ItemKind::MacroCall => return None,
    })
}

fn bucket_content_hash(kind: BucketKind, entries: &[BucketEntry]) -> ContentHash {
    let mut s = format!("{kind:?}\u{1f}");
    for e in entries {
        s.push_str(e.name.as_deref().unwrap_or("<impl>"));
        s.push('\u{1e}');
        let _ = write!(s, "{:?}\u{1e}", e.kind);
        let _ = write!(s, "{}\u{1f}", e.content.0);
    }
    hash_str(&s)
}

/// Project a semantic index into coherence buckets. Only `pub`,
/// cfg-active, syntax-parsed items enter name buckets; impl blocks
/// (which carry no visibility) enter the `Impls` bucket when active.
/// Opaque macro invocations are never projected — their expansion is
/// unknown, so they can only make the projection *incomplete*, never
/// add a name.
#[must_use]
pub fn project_interface(
    index: &SemanticIndex,
    cst: &Cst,
    workspace: ContentHash,
) -> InterfaceProjection {
    let mut incomplete = false;
    let mut per_kind: Vec<(BucketKind, Vec<BucketEntry>)> = Vec::new();
    for item in &index.items {
        if item.kind == ItemKind::Use && item.visibility == Visibility::Pub {
            // A public re-export may introduce arbitrary names this
            // syntax layer cannot see: fail closed.
            incomplete = true;
            continue;
        }
        if item.status == ItemStatus::Opaque {
            // An opaque macro expansion or ambiguous item can introduce
            // public names. Omitting it from positive buckets is safe only
            // if negative guards also refuse to answer.
            incomplete = true;
            continue;
        }
        if item.status != ItemStatus::Active {
            // cfg-disabled items never leak into the global projection;
            // the cfg guard is propagated through the cache key.
            continue;
        }
        if (item.kind == ItemKind::MacroDef && item.visibility != Visibility::Pub)
            || item.kind == ItemKind::ExternBlock
            || (item.kind == ItemKind::ExternCrate && item.visibility == Visibility::Pub)
        {
            // macro_export and foreign/re-exported declarations need more
            // semantic information than this syntax layer has.
            incomplete = true;
            continue;
        }
        let Some(kind) = bucket_of(item) else { continue };
        let is_public = item.visibility == Visibility::Pub;
        if kind == BucketKind::Impls {
            // impl blocks: no visibility, coherence-relevant.
        } else if !is_public {
            // Private, restricted, and anonymous items must not leak.
            continue;
        }
        let entry = BucketEntry {
            item: item.id,
            name: item.name.clone(),
            kind: item.kind,
            content: hash_str(cst.text(item.span)),
        };
        if let Some((_, entries)) = per_kind.iter_mut().find(|(k, _)| *k == kind) {
            entries.push(entry);
        } else {
            per_kind.push((kind, vec![entry]));
        }
    }
    per_kind.sort_by_key(|(k, _)| *k);
    let buckets: Vec<Bucket> = per_kind
        .into_iter()
        .map(|(kind, entries)| Bucket {
            kind,
            content_hash: bucket_content_hash(kind, &entries),
            entries,
        })
        .collect();
    let observable_locs = scan_loc_uses(cst);
    let loc_dep = loc_dep_hash(&observable_locs);
    let mut ch = format!("{workspace:?}\u{1f}{loc_dep:?}\u{1f}{incomplete}\u{1e}");
    for b in &buckets {
        ch.push_str(&b.content_hash.0.to_string());
        ch.push('\u{1f}');
    }
    InterfaceProjection {
        workspace,
        buckets,
        observable_locs,
        loc_dep,
        incomplete,
        content_hash: hash_str(&ch),
    }
}

/// Why a negative-guard lookup did not produce a cached answer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GuardMiss {
    /// No guard was ever recorded for this (bucket, name).
    NoGuard,
    /// The name is now present: the guard is invalidated by the
    /// added item.
    NameNowPresent,
    /// The bucket's content changed since the guard was recorded;
    /// the guard is stale and must be re-derived.
    AnchorChanged,
    /// The projection is incomplete (`pub use` present); absence can
    /// never be asserted.
    Incomplete,
    /// The projection belongs to a different workspace.
    WrongWorkspace,
}

/// Outcome of checking a negative guard.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GuardVerdict {
    /// The guarded absence still holds.
    Hit,
    /// No valid cached absence; the reason says why.
    Miss(GuardMiss),
}

/// A content-anchored negative guard: "bucket `bucket` contained no
/// item named `name` when anchored at `bucket_anchor`".
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NegativeGuard {
    /// Workspace digest the guard belongs to.
    pub workspace: ContentHash,
    /// Bucket whose absence is asserted.
    pub bucket: BucketKind,
    /// The absent name.
    pub name: String,
    /// Bucket content hash at record time.
    pub bucket_anchor: ContentHash,
}

/// Store of negative guards.
#[derive(Default)]
pub struct GuardStore {
    guards: Vec<NegativeGuard>,
}

impl GuardStore {
    /// An empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record that `name` is absent from `bucket` in `proj`. Returns
    /// `None` (records nothing) when the name is actually present or
    /// the projection is incomplete — a guard is only ever created
    /// for a *verified* absence.
    pub fn record(
        &mut self,
        proj: &InterfaceProjection,
        bucket: BucketKind,
        name: &str,
    ) -> Option<NegativeGuard> {
        if proj.incomplete || proj.has_name(bucket, name) {
            return None;
        }
        let anchor = proj
            .bucket(bucket)
            .map_or_else(|| bucket_content_hash(bucket, &[]), |b| b.content_hash);
        let guard = NegativeGuard {
            workspace: proj.workspace,
            bucket,
            name: name.to_owned(),
            bucket_anchor: anchor,
        };
        self.guards.push(guard.clone());
        Some(guard)
    }

    /// Check whether a recorded guard still vouches for `name`'s
    /// absence in `bucket` under `proj`. Adding any matching item
    /// changes the bucket's content hash, so the guard reports
    /// [`GuardMiss::AnchorChanged`] or [`GuardMiss::NameNowPresent`]
    /// rather than a stale hit.
    #[must_use]
    pub fn check(
        &self,
        proj: &InterfaceProjection,
        bucket: BucketKind,
        name: &str,
    ) -> GuardVerdict {
        if proj.incomplete {
            return GuardVerdict::Miss(GuardMiss::Incomplete);
        }
        let Some(guard) = self
            .guards
            .iter()
            .find(|g| g.workspace == proj.workspace && g.bucket == bucket && g.name == name)
        else {
            // A guard for this (bucket, name) in another workspace is
            // never a hit here: report it distinctly for diagnostics.
            if self.guards.iter().any(|g| g.bucket == bucket && g.name == name) {
                return GuardVerdict::Miss(GuardMiss::WrongWorkspace);
            }
            return GuardVerdict::Miss(GuardMiss::NoGuard);
        };
        if proj.has_name(bucket, name) {
            return GuardVerdict::Miss(GuardMiss::NameNowPresent);
        }
        let anchor = proj
            .bucket(bucket)
            .map_or_else(|| bucket_content_hash(bucket, &[]), |b| b.content_hash);
        if anchor != guard.bucket_anchor {
            return GuardVerdict::Miss(GuardMiss::AnchorChanged);
        }
        GuardVerdict::Hit
    }

    /// Number of recorded guards.
    #[must_use]
    pub fn len(&self) -> usize {
        self.guards.len()
    }

    /// Whether the store is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.guards.is_empty()
    }
}
