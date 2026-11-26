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
//!   cfg context or macro environment changes.

pub mod cache;
pub mod cfg;
pub mod cst;
pub mod hash;
pub mod item;
pub mod lex;

pub use cache::{CacheStats, CstCache, SemanticCache};
pub use cfg::{CfgContext, CfgExpr, Meta};
pub use cst::{parse_top_level, Attr, Cst, TopKind, TopNode};
pub use hash::{hash_bytes, hash_str, ContentHash};
pub use item::{
    cfg_hash, macro_env_hash, project, Dep, Item, ItemId, ItemKind, ItemStatus, SemanticIndex,
};
pub use lex::{lex, Span, Tk, Token};
