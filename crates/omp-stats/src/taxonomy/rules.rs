//! Deserialized shape of the compiled compat rules (`data/rules.json`).
//!
//! Only the `taxonomy` subtree is used at runtime by `classify_model` and the
//! collapse helpers; the other subtrees are deserialized as opaque JSON so
//! the single embedded file stays the source of truth.

use serde::{Deserialize, Serialize};

/// User-facing thinking levels (least to most intensive), plus `"off"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Effort {
    Minimal,
    Low,
    Medium,
    High,
    XHigh,
    Max,
    /// Thinking disabled (effort-suffix rules and overrides).
    #[serde(rename = "off")]
    Off,
}

/// Class-membership matcher kinds, most to least specific.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MatcherKind {
    Exact,
    Bounded,
    Namespace,
    Prefix,
    Glob,
}

/// One compiled class-membership matcher (token pre-lowercased).
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompiledMatcher {
    pub kind: MatcherKind,
    pub token: String,
    /// Namespace matchers only: accept dot/colon segments with token boundaries.
    #[serde(default)]
    pub bounded: bool,
}

/// One compiled product-family rule (glob pre-lowercased).
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompiledFamily {
    pub id: String,
    pub glob: String,
    pub priority: i32,
}

/// One compiled revision-extraction prefix (pre-lowercased).
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompiledRevisionPrefix {
    pub prefix: String,
    #[serde(default)]
    pub anywhere: bool,
}

/// One compiled reviewed identity correction with exactly one bare-model
/// selector (`model` exact or `glob` pattern, never both).
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompiledIdentityOverride {
    pub id: String,
    #[serde(default)]
    pub provider: Option<String>,
    #[serde(default)]
    pub logical: Option<String>,
    #[serde(default)]
    pub class: Option<String>,
    #[serde(default)]
    pub family: Option<String>,
    /// Canonical `major.minor.patch`.
    #[serde(default)]
    pub revision: Option<String>,
    #[serde(default)]
    pub effort: Option<Effort>,
    #[serde(default)]
    pub thinking_variant: bool,
    #[serde(default)]
    pub expires_at_ms: Option<i64>,
    /// Exact bare-model selector (mutually exclusive with `glob`).
    #[serde(default)]
    pub model: Option<String>,
    /// Anchored, case-insensitive bare-model glob (mutually exclusive with `model`).
    #[serde(default)]
    pub glob: Option<String>,
}

/// One compiled model class: matchers, families, revision rules, overrides.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompiledClass {
    pub id: String,
    pub matchers: Vec<CompiledMatcher>,
    pub families: Vec<CompiledFamily>,
    pub revision_prefixes: Vec<CompiledRevisionPrefix>,
    pub skip_bare: Vec<String>,
    pub overrides: Vec<CompiledIdentityOverride>,
}

/// One collapse suffix rule (thinking or effort variant).
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompiledCollapseSuffix {
    pub suffix: String,
    /// Present on `effort-suffix` rules.
    #[serde(default)]
    pub effort: Option<Effort>,
    /// True on `thinking-suffix` rules.
    #[serde(default)]
    pub thinking: bool,
    #[serde(default)]
    pub except_bare_prefix: Option<String>,
}

/// One provider-scoped effort-lane suffix rule.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompiledEffortLane {
    pub suffix: String,
    pub providers: Vec<String>,
    #[serde(default)]
    pub bare_prefix: Option<String>,
}

/// One provider-scoped routing-variant suffix rule.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
#[allow(dead_code)]
pub struct CompiledRoutingVariant {
    pub suffix: String,
    pub providers: Vec<String>,
}

/// One reviewed provider-scoped effort-sibling family seed.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompiledEffortFamily {
    pub provider: String,
    pub logical: String,
    pub aliases: Vec<String>,
}

/// The declared collapse vocabulary.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Collapse {
    pub suffixes: Vec<CompiledCollapseSuffix>,
    pub pair_tokens: Vec<String>,
    pub lanes: Vec<CompiledEffortLane>,
    #[allow(dead_code)]
    pub routing_variants: Vec<CompiledRoutingVariant>,
    pub effort_families: Vec<CompiledEffortFamily>,
}

/// The declared discovery vocabulary (only fields used by the taxonomy port).
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Discovery {
    pub canonical_recovery: Vec<String>,
    pub billing_variant_suffixes: Vec<String>,
}

/// The `taxonomy` subtree of `rules.json`.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TaxonomyRules {
    pub classes: Vec<CompiledClass>,
    pub collapse: Collapse,
    pub discovery: Discovery,
}

/// Full `rules.json`; non-taxonomy subtrees stay opaque.
#[derive(Debug, Clone, Deserialize)]
pub struct Rules {
    pub taxonomy: TaxonomyRules,
    #[serde(default, rename = "cascade")]
    #[allow(dead_code)]
    pub cascade: Option<serde_json::Value>,
    #[serde(default, rename = "behavior")]
    #[allow(dead_code)]
    pub behavior: Option<serde_json::Value>,
    #[serde(default, rename = "auth")]
    #[allow(dead_code)]
    pub auth: Option<serde_json::Value>,
    #[serde(default, rename = "providers")]
    #[allow(dead_code)]
    pub providers: Option<serde_json::Value>,
    #[serde(default, rename = "version")]
    #[allow(dead_code)]
    pub version: Option<serde_json::Value>,
    #[serde(default, rename = "files")]
    #[allow(dead_code)]
    pub files: Option<serde_json::Value>,
}

/// The embedded compiled rule tree (parsed once per process).
pub static RULES: std::sync::LazyLock<Rules> = std::sync::LazyLock::new(|| {
    let raw = include_str!("../../data/rules.json");
    serde_json::from_str(raw).expect("rules.json must deserialize")
});

/// The taxonomy subtree (read-only view for collapse/classify logic).
pub fn taxonomy_rules() -> &'static TaxonomyRules {
    &RULES.taxonomy
}
