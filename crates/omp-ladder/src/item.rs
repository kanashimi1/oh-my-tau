//! Semantic projection of the CST: a *syntax-derived semantic item
//! index*. This is explicitly **not** full rustc HIR — there is no
//! name resolution, no type checking, and no macro expansion. What it
//! provides is a stable, content-addressed, cfg-aware index of
//! top-level items suitable for downstream interface projections.
//!
//! Safety rules enforced here:
//!
//! - Items whose effective `cfg` evaluates false are marked
//!   [`ItemStatus::CfgDisabled`] and must not be treated as compiled
//!   code.
//! - Macro invocations are [`ItemStatus::Opaque`]: their expansion is
//!   unknown, so they are never reported as ordinary parsed items.
//!   Their dependency list carries the invoked path plus the current
//!   macro-environment hash, so any change to a `macro_rules!` /
//!   `macro` definition in the file conservatively invalidates every
//!   macro-call item.

use serde::{Deserialize, Serialize};

use crate::cfg::{cfg_attr_from_meta, cfg_from_meta, CfgContext, CfgExpr};
use crate::cst::{Cst, TopKind};
use crate::hash::{hash_str, ContentHash};
use crate::lex::Span;

/// Stable identifier of an item within one index.
pub type ItemId = usize;

/// Semantic item kind (mirrors [`TopKind`] minus non-item nodes).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ItemKind {
    Fn,
    Struct,
    Enum,
    Union,
    Impl,
    Trait,
    Mod,
    Use,
    TypeAlias,
    Const,
    Static,
    ExternCrate,
    ExternBlock,
    MacroDef,
    MacroCall,
}

/// Whether an item may be treated as live, ordinary parsed code.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ItemStatus {
    /// cfg-active, syntax-parsed ordinary item.
    Active,
    /// cfg-disabled under the projection's [`CfgContext`].
    CfgDisabled,
    /// Macro invocation: expansion unknown, body opaque. Never treat
    /// as safely parsed.
    Opaque,
}

/// A conservative dependency edge.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Dep {
    /// This item invokes the named macro.
    MacroInvoke(String),
    /// This item's validity depends on the file's macro environment
    /// (the set of macro definitions); the hash captures it.
    MacroEnv(ContentHash),
}

/// One indexed item.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Item {
    pub id: ItemId,
    pub kind: ItemKind,
    pub name: Option<String>,
    pub span: Span,
    pub body_span: Option<Span>,
    /// Effective cfg constraint conjoined from all `cfg`/`cfg_attr`
    /// attributes, if any.
    pub cfg: Option<CfgExpr>,
    /// Whether `cfg` evaluates true under the projection context.
    pub cfg_enabled: bool,
    pub status: ItemStatus,
    /// Invoked macro path for [`ItemKind::MacroCall`].
    pub macro_path: Option<String>,
    pub deps: Vec<Dep>,
    /// Arbitrary caller JSON attached to the item. Serialized
    /// verbatim; `false`, `0` and `null` payloads are preserved.
    pub metadata: serde_json::Map<String, serde_json::Value>,
}

/// The semantic index of one source file under one cfg context.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SemanticIndex {
    pub source_hash: ContentHash,
    pub source_len: usize,
    pub cfg_hash: ContentHash,
    pub macro_env_hash: ContentHash,
    pub items: Vec<Item>,
}

impl SemanticIndex {
    /// Serialize to JSON (metadata preserved verbatim).
    #[must_use]
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }

    /// Deserialize from JSON produced by [`SemanticIndex::to_json`].
    #[must_use]
    pub fn from_json(v: &serde_json::Value) -> Option<Self> {
        serde_json::from_value(v.clone()).ok()
    }
}

/// Hash of a cfg context (flags and key/value pairs, sorted).
#[must_use]
pub fn cfg_hash(ctx: &CfgContext) -> ContentHash {
    let mut s = String::new();
    for f in &ctx.flags {
        s.push_str(f);
        s.push('\u{1f}');
    }
    s.push('\u{1e}');
    for (k, v) in &ctx.kv {
        s.push_str(k);
        s.push('=');
        s.push_str(v);
        s.push('\u{1f}');
    }
    hash_str(&s)
}

/// Hash of the file's macro environment: every macro definition's
/// name and full source text, in order. Any edit to any macro
/// definition changes this hash.
#[must_use]
pub fn macro_env_hash(cst: &Cst) -> ContentHash {
    let mut s = String::new();
    for n in &cst.nodes {
        if n.kind == TopKind::MacroDef {
            s.push_str(n.name.as_deref().unwrap_or("?"));
            s.push('\u{1f}');
            s.push_str(cst.text(n.span));
            s.push('\u{1e}');
        }
    }
    hash_str(&s)
}

fn kind_of(k: TopKind) -> Option<ItemKind> {
    Some(match k {
        TopKind::Fn => ItemKind::Fn,
        TopKind::Struct => ItemKind::Struct,
        TopKind::Enum => ItemKind::Enum,
        TopKind::Union => ItemKind::Union,
        TopKind::Impl => ItemKind::Impl,
        TopKind::Trait => ItemKind::Trait,
        TopKind::Mod => ItemKind::Mod,
        TopKind::Use => ItemKind::Use,
        TopKind::TypeAlias => ItemKind::TypeAlias,
        TopKind::Const => ItemKind::Const,
        TopKind::Static => ItemKind::Static,
        TopKind::ExternCrate => ItemKind::ExternCrate,
        TopKind::ExternBlock => ItemKind::ExternBlock,
        TopKind::MacroDef => ItemKind::MacroDef,
        TopKind::MacroCall => ItemKind::MacroCall,
        TopKind::InnerAttr | TopKind::Unknown => return None,
    })
}

/// Project a CST into a semantic item index under a cfg context.
#[must_use]
pub fn project(cst: &Cst, ctx: &CfgContext) -> SemanticIndex {
    let env_hash = macro_env_hash(cst);
    let mut items = Vec::new();
    for node in &cst.nodes {
        let Some(kind) = kind_of(node.kind) else {
            continue;
        };
        // Conjoin all cfg constraints from the item's attributes.
        let mut cfg: Option<CfgExpr> = None;
        for attr in &node.attrs {
            let Some(meta) = &attr.meta else { continue };
            let constraint = if let Some(e) = cfg_from_meta(meta) {
                Some(e)
            } else {
                cfg_attr_from_meta(meta, ctx).unwrap_or(None)
            };
            if let Some(c) = constraint {
                cfg = Some(match cfg {
                    None => c,
                    Some(prev) => CfgExpr::All(vec![prev, c]),
                });
            }
        }
        let cfg_enabled = cfg.as_ref().is_none_or(|c| c.eval(ctx));
        let (status, deps) = if kind == ItemKind::MacroCall {
            // Conservative: opaque, depends on the macro env.
            let mut deps = vec![Dep::MacroEnv(env_hash)];
            if let Some(p) = &node.macro_path {
                deps.push(Dep::MacroInvoke(p.clone()));
            }
            (ItemStatus::Opaque, deps)
        } else if cfg_enabled {
            (ItemStatus::Active, Vec::new())
        } else {
            (ItemStatus::CfgDisabled, Vec::new())
        };
        items.push(Item {
            id: items.len(),
            kind,
            name: node.name.clone(),
            span: node.span,
            body_span: node.body_span,
            cfg,
            cfg_enabled,
            status,
            macro_path: node.macro_path.clone(),
            deps,
            metadata: serde_json::Map::new(),
        });
    }
    SemanticIndex {
        source_hash: cst.source_hash,
        source_len: cst.source_len,
        cfg_hash: cfg_hash(ctx),
        macro_env_hash: env_hash,
        items,
    }
}
