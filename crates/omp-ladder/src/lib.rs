//! Item-level parsing phase for Rust sources, built as a reusable
//! production library.
//!
//! Layers, explicitly separated:
//!
//! - [`lex`]: a real Rust lexer built on `rustc_lexer` (the rustc
//!   frontend's tokeniser). Strings, raw strings, char literals and
//!   comments (including nested block comments) are single tokens, so
//!   downstream phases never re-scan token interiors.
//! - [`cst`]: a concrete-syntax top-level pass over the token stream.
//!   It records byte spans for top-level items, balanced-delimiter
//!   bodies, outer attributes (with `cfg` / `cfg_attr` extracted), and
//!   treats macro invocations and `macro_rules!` bodies as opaque token
//!   regions — inner "items" inside a macro body are never surfaced.
//! - [`cfg`]: `cfg(...)` / `cfg_attr(...)` expression parsing and
//!   evaluation against an explicit [`cfg::CfgContext`].
//! - [`item`]: the semantic projection — a *syntax-derived semantic
//!   index* of items with statuses (`Active`, `CfgDisabled`, `Opaque`)
//!   and conservative dependency edges for macro invocations. This is
//!   deliberately **not** full rustc HIR: no name resolution, no type
//!   information, no expansion of macro bodies.
//! - [`cache`]: bounded, content-addressed caches. The CST cache is
//!   keyed by source content hash; the semantic cache is keyed by
//!   `(source hash, cfg-context hash, macro-environment hash)` so a
//!   CST cache hit can never suppress semantic re-evaluation when the
//!   cfg context or macro environment changes; the interface cache
//!   additionally keys on the workspace digest and the
//!   observable-location dependency hash.
//! - [`iface`]: interface projection into explicit coherence buckets
//!   (public, cfg-active, parsed items only) with content-anchored
//!   negative guards. Syntax-derived, **not** full rustc HIR: `pub
//!   use` re-exports mark the projection incomplete and negative
//!   guards fail closed.
//! - [`loc`]: separation of cold diagnostic spans (freely relocatable)
//!   from compile-time observable `file!()`/`line!()`/`column!()`
//!   values (symbolic spans, late resolution, invalidation-carrying
//!   dependency hash).
//! - [`wire`]: a bounded, write-once escaped wire arena for
//!   serialized projected data — no append after freeze, lossless
//!   arbitrary JSON primitives.
//! - [`cycle`]: composite fix-cycle keys
//!   `(workspace_digest, active_subgoal_id, normalized_error_sig)`
//!   with content-anchored negative-constraint invalidation.

pub mod cache;
pub mod cfg;
pub mod cst;
pub mod cycle;
pub mod hash;
pub mod iface;
pub mod item;
pub mod lex;
pub mod loc;
pub mod wire;

pub use cache::{CacheStats, CstCache, InterfaceCache, SemanticCache};
pub use cfg::{CfgContext, CfgExpr, Meta};
pub use cst::{parse_top_level, Attr, Cst, TopKind, TopNode, Visibility};
pub use cycle::{cycle_key, normalize_error_sig, CycleKey, CycleStore, NegativeConstraint};
pub use hash::{hash_bytes, hash_str, ContentHash};
pub use iface::{
    project_interface, Bucket, BucketEntry, BucketKind, GuardMiss, GuardStore, GuardVerdict,
    InterfaceProjection, NegativeGuard,
};
pub use item::{
    cfg_hash, macro_env_hash, project, Dep, Item, ItemId, ItemKind, ItemStatus, SemanticIndex,
};
pub use lex::{lex, Span, Tk, Token};
pub use loc::{
    loc_dep_hash, relocate, resolve_offset, resolve_use, scan_loc_uses, LocClass, LocMacro,
    LocUse, RelocateError, ResolvedLoc, SpanId, SymbolicSpan, TrackedSpan,
};
pub use wire::{WireArena, WireError};
