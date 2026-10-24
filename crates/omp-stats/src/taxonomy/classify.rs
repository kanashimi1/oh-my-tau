//! Runtime model-identity classification: `classify_model`.
//!
//! Faithful port of the o2 reference (`taxonomy.rs` / `classify.rs`):
//! matchers rank `exact(4) > bounded(3) > namespace(2) > prefix(1) > glob(0)`
//! with token byte length as tiebreak; equal cross-class or cross-family ranks
//! error unless classification is `lenient` (discovery normalization).

use std::collections::HashMap;
use std::sync::OnceLock;

use anyhow::{bail, Result};
use serde::Serialize;

use super::collapse::{collapse_variant_id, CollapsedVariant};
use super::revision::{format_revision, parse_revision_prefix};
use super::rules::{
    taxonomy_rules, CompiledClass, CompiledIdentityOverride, CompiledMatcher, Effort, MatcherKind,
};

/// Structured identity of one classified model (port of `ModelIdentity`).
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelIdentity {
    /// Vendor lineage id, `"unknown"` when unclassified.
    pub class: String,
    /// Product family within the class, when classified.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub family: Option<String>,
    /// Canonical `major.minor.patch`, when extracted.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
    /// Effort tier collapsed out of the id, when the id was an effort variant.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<Effort>,
    /// Whether the id carried a thinking-variant suffix.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_variant: Option<bool>,
    /// Canonical logical id when it differs from the wire id.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logical_id: Option<String>,
}

/// Options for [`classify_model`].
#[derive(Debug, Clone, Copy, Default)]
pub struct ClassifyOptions {
    /// Observation time for override expiry; absent keeps expiring overrides active.
    pub observed_at_ms: Option<i64>,
    /// Swallow ambiguity instead of erroring: an ambiguous class resolves to
    /// `unknown`, an ambiguous family to no family. Discovery normalization
    /// uses this; catalog compilation stays strict.
    pub lenient: bool,
}

fn matcher_rank(kind: MatcherKind) -> i32 {
    match kind {
        MatcherKind::Exact => 4,
        MatcherKind::Bounded => 3,
        MatcherKind::Namespace => 2,
        MatcherKind::Prefix => 1,
        MatcherKind::Glob => 0,
    }
}

fn bare_of(id: &str) -> &str {
    match id.rfind('/') {
        Some(slash) => &id[slash + 1..],
        None => id,
    }
}

fn bounded_match(value: &str, token: &str) -> bool {
    if value == token {
        return true;
    }
    if !value.starts_with(token) {
        return false;
    }
    match value.as_bytes().get(token.len()) {
        None => false,
        Some(&next) => {
            next == b'-'
                || next == b'_'
                || next == b'.'
                || next == b':'
                || (48..=57).contains(&next)
        }
    }
}

fn glob_match(pattern: &str, value: &str) -> bool {
    // Anchored `*`-wildcard match; both sides pre-lowercased by callers.
    let segments: Vec<&str> = pattern.split('*').collect();
    if segments.len() == 1 {
        return value == pattern;
    }
    let head = segments[0];
    if !value.starts_with(head) {
        return false;
    }
    let mut remainder = &value[head.len()..];
    for segment in &segments[1..segments.len() - 1] {
        if segment.is_empty() {
            continue;
        }
        let Some(found) = remainder.find(segment) else {
            return false;
        };
        remainder = &remainder[found + segment.len()..];
    }
    let last = segments[segments.len() - 1];
    last.is_empty() || remainder.ends_with(last)
}

fn matcher_matches(matcher: &CompiledMatcher, lower: &str, bare: &str) -> bool {
    match matcher.kind {
        MatcherKind::Exact => bare == matcher.token,
        MatcherKind::Bounded => bounded_match(bare, &matcher.token),
        MatcherKind::Namespace => {
            let parts: Vec<&str> = if matcher.bounded {
                lower.split(['/', '.', ':']).collect()
            } else {
                lower.split('/').collect()
            };
            for part in parts {
                if part.is_empty() {
                    continue;
                }
                let hit = if matcher.bounded {
                    bounded_match(part, &matcher.token)
                } else {
                    part == matcher.token
                };
                if hit {
                    return true;
                }
            }
            false
        }
        MatcherKind::Prefix => bare.starts_with(&matcher.token),
        MatcherKind::Glob => glob_match(&matcher.token, bare),
    }
}

fn non_wildcard_bytes(glob: &str) -> usize {
    glob.bytes().filter(|b| *b != b'*').count()
}

struct FamilyRanking {
    winner: Option<(i32, usize, String)>,
    tied: Option<(String, String)>,
}

fn rank_families(cls: &CompiledClass, subject: &str) -> FamilyRanking {
    let mut winner: Option<(i32, usize, String)> = None;
    let mut tied: Option<(String, String)> = None;
    for family in &cls.families {
        if !glob_match(&family.glob, subject) {
            continue;
        }
        let rank = (family.priority, non_wildcard_bytes(&family.glob));
        if let Some((wp, wb, ref wid)) = winner {
            if wp == rank.0 && wb == rank.1 && *wid != family.id {
                tied = Some((wid.clone(), family.id.clone()));
            } else if wp < rank.0 || (wp == rank.0 && wb < rank.1) {
                winner = Some((rank.0, rank.1, family.id.clone()));
                tied = None;
            }
        } else {
            winner = Some((rank.0, rank.1, family.id.clone()));
            tied = None;
        }
    }
    FamilyRanking { winner, tied }
}

fn strip_class_namespace<'a>(cls: &CompiledClass, bare: &'a str) -> Option<&'a str> {
    let separator = bare
        .find(['.', ':'])
        .filter(|&i| i > 0 && i < bare.len() - 1)?;
    let namespace = &bare[..separator];
    if !cls.matchers.iter().any(|m| m.token == namespace) {
        return None;
    }
    Some(&bare[separator + 1..])
}

fn classify_family(
    cls: &CompiledClass,
    bare: &str,
    model: &str,
    lenient: bool,
) -> Result<Option<String>> {
    let initial = rank_families(cls, bare);
    if let Some(tied) = initial.tied {
        // A Bedrock-style `vendor.model` id names the product after the class
        // namespace. Rescore only ambiguities so existing classifications stay put.
        if let Some(scoped) = strip_class_namespace(cls, bare) {
            let rescored = rank_families(cls, scoped);
            if let Some((.., id)) = rescored.winner {
                if rescored.tied.is_none() {
                    return Ok(Some(id));
                }
            }
        }
        if lenient {
            return Ok(None);
        }
        bail!(
            "ambiguous family for `{model}`: `{}` and `{}` tie",
            tied.0,
            tied.1
        );
    }
    Ok(initial.winner.map(|(.., id)| id))
}

fn extract_revision(cls: &CompiledClass, bare: &str) -> Option<String> {
    if cls.skip_bare.iter().any(|b| b == bare) {
        return None;
    }
    for rule in &cls.revision_prefixes {
        let tail = if rule.anywhere {
            let start = bare.find(&rule.prefix)?;
            if start == usize::MAX {
                continue;
            }
            &bare[start + rule.prefix.len()..]
        } else if bare.starts_with(&rule.prefix) {
            &bare[rule.prefix.len()..]
        } else {
            continue;
        };
        let digit = tail.find(|c: char| c.is_ascii_digit())?;
        let revision = parse_revision_prefix(&tail[digit..])?;
        return Some(format_revision(revision));
    }
    None
}

#[derive(Debug, Default)]
struct ClassRanks {
    class: Option<String>,
    family: Option<String>,
    revision: Option<String>,
}

fn classify_ranks(model: &str, lenient: bool) -> Result<ClassRanks> {
    let lower = model.trim().to_lowercase();
    let bare = bare_of(&lower);
    let mut winner: Option<(i32, usize, String)> = None;
    let mut tied: Option<(String, String)> = None;
    for cls in &taxonomy_rules().classes {
        for matcher in &cls.matchers {
            if !matcher_matches(matcher, &lower, bare) {
                continue;
            }
            let rank = (matcher_rank(matcher.kind), matcher.token.len());
            if let Some((wp, wt, ref wid)) = winner {
                if wp == rank.0 && wt == rank.1 && *wid != cls.id {
                    tied = Some((wid.clone(), cls.id.clone()));
                } else if wp < rank.0 || (wp == rank.0 && wt < rank.1) {
                    winner = Some((rank.0, rank.1, cls.id.clone()));
                    tied = None;
                }
            } else {
                winner = Some((rank.0, rank.1, cls.id.clone()));
                tied = None;
            }
        }
    }
    if let Some((first, second)) = tied {
        if lenient {
            return Ok(ClassRanks {
                class: Some("unknown".into()),
                ..Default::default()
            });
        }
        bail!("ambiguous class for `{lower}`: `{first}` and `{second}` tie");
    }
    let Some((.., class_id)) = winner else {
        return Ok(ClassRanks {
            class: Some("unknown".into()),
            ..Default::default()
        });
    };
    let cls = taxonomy_rules()
        .classes
        .iter()
        .find(|c| c.id == class_id)
        .expect("winning class exists");
    let mut ranks = ClassRanks {
        class: Some(class_id),
        ..Default::default()
    };
    ranks.family = classify_family(cls, bare, &lower, lenient)?;
    ranks.revision = extract_revision(cls, bare);
    Ok(ranks)
}

fn ranks_in_class(class_id: &str, model: &str, lenient: bool) -> Result<ClassRanks> {
    let Some(cls) = taxonomy_rules().classes.iter().find(|c| c.id == class_id) else {
        return Ok(ClassRanks::default());
    };
    let lower = model.trim().to_lowercase();
    let bare = bare_of(&lower);
    Ok(ClassRanks {
        class: None,
        family: classify_family(cls, bare, &lower, lenient)?,
        revision: extract_revision(cls, bare),
    })
}

// Exact overrides retain a direct lowercase index. Pattern overrides are few
// and remain in one static list, ranked only after exact lookup misses.
#[derive(Default)]
struct OverrideBucket {
    by_provider: HashMap<String, &'static CompiledIdentityOverride>,
    agnostic: Option<&'static CompiledIdentityOverride>,
}

struct IndexedPatternOverride {
    override_ref: &'static CompiledIdentityOverride,
    glob: &'static str,
    provider: Option<String>,
    specificity: usize,
}

#[derive(Default)]
struct OverrideIndexes {
    exact: HashMap<String, OverrideBucket>,
    patterns: Vec<IndexedPatternOverride>,
}

fn get_override_indexes() -> &'static OverrideIndexes {
    static INDEXES: OnceLock<OverrideIndexes> = OnceLock::new();
    INDEXES.get_or_init(|| {
        let mut exact: HashMap<String, OverrideBucket> = HashMap::new();
        let mut patterns: Vec<IndexedPatternOverride> = Vec::new();
        for cls in &taxonomy_rules().classes {
            for ovr in &cls.overrides {
                let (Some(model), Some(glob)) = (&ovr.model, ovr.glob.as_ref()) else {
                    if ovr.model.is_some() {
                        let key = ovr.model.as_deref().unwrap_or_default().to_lowercase();
                        let bucket = exact.entry(key).or_default();
                        match &ovr.provider {
                            Some(p) => {
                                let pk = p.to_lowercase();
                                bucket.by_provider.entry(pk).or_insert(ovr);
                            }
                            None => {
                                if bucket.agnostic.is_none() {
                                    bucket.agnostic = Some(ovr);
                                }
                            }
                        }
                    } else if let Some(glob) = &ovr.glob {
                        patterns.push(IndexedPatternOverride {
                            override_ref: ovr,
                            glob,
                            provider: ovr.provider.as_ref().map(|p| p.to_lowercase()),
                            specificity: non_wildcard_bytes(glob),
                        });
                    }
                    continue;
                };
                let _ = (model, glob);
            }
        }
        OverrideIndexes { exact, patterns }
    })
}

fn override_is_active(ovr: &CompiledIdentityOverride, observed_at_ms: Option<i64>) -> bool {
    match (ovr.expires_at_ms, observed_at_ms) {
        (None, _) => true,
        (Some(_), None) => true,
        (Some(exp), Some(obs)) => obs < exp,
    }
}

struct PatternOverrideRanking {
    winner: Option<(usize, &'static CompiledIdentityOverride)>,
    tied: Option<(
        &'static CompiledIdentityOverride,
        &'static CompiledIdentityOverride,
    )>,
}

fn rank_pattern_overrides(
    patterns: &[IndexedPatternOverride],
    provider: Option<&str>,
    bare_model: &str,
    observed_at_ms: Option<i64>,
) -> PatternOverrideRanking {
    let mut winner: Option<(usize, &'static CompiledIdentityOverride)> = None;
    let mut tied: Option<(
        &'static CompiledIdentityOverride,
        &'static CompiledIdentityOverride,
    )> = None;
    for pattern in patterns {
        if pattern.provider.as_deref() != provider
            || !override_is_active(pattern.override_ref, observed_at_ms)
        {
            continue;
        }
        if !glob_match(pattern.glob, bare_model) {
            continue;
        }
        if let Some((specificity, _)) = winner {
            if specificity == pattern.specificity {
                tied = Some((winner.unwrap().1, pattern.override_ref));
            } else if specificity < pattern.specificity {
                winner = Some((pattern.specificity, pattern.override_ref));
                tied = None;
            }
        } else {
            winner = Some((pattern.specificity, pattern.override_ref));
            tied = None;
        }
    }
    PatternOverrideRanking { winner, tied }
}

fn find_identity_override(
    provider: &str,
    bare_model: &str,
    observed_at_ms: Option<i64>,
    lenient: bool,
) -> Result<Option<&'static CompiledIdentityOverride>> {
    let lower_provider = provider.to_lowercase();
    let lower_bare_model = bare_model.to_lowercase();
    let indexes = get_override_indexes();
    if let Some(bucket) = indexes.exact.get(&lower_bare_model) {
        if let Some(scoped) = bucket.by_provider.get(&lower_provider) {
            if override_is_active(scoped, observed_at_ms) {
                return Ok(Some(scoped));
            }
        }
        if let Some(agnostic) = bucket.agnostic {
            if override_is_active(agnostic, observed_at_ms) {
                return Ok(Some(agnostic));
            }
        }
    }
    let scoped = rank_pattern_overrides(
        &indexes.patterns,
        Some(&lower_provider),
        &lower_bare_model,
        observed_at_ms,
    );
    let ranked = if scoped.winner.is_none() && scoped.tied.is_none() {
        rank_pattern_overrides(&indexes.patterns, None, &lower_bare_model, observed_at_ms)
    } else {
        scoped
    };
    if let Some((first, second)) = ranked.tied {
        if lenient {
            return Ok(None);
        }
        bail!(
            "ambiguous override for `{lower_bare_model}`: `{}` and `{}` tie",
            first.id,
            second.id
        );
    }
    Ok(ranked.winner.map(|(_, o)| o))
}

fn classify_model_uncached(
    provider: &str,
    model_id: &str,
    opts: ClassifyOptions,
) -> Result<ModelIdentity> {
    let lenient = opts.lenient;
    let trimmed = model_id.trim();
    let bare = bare_of(trimmed);
    if let Some(ovr) = find_identity_override(provider, bare, opts.observed_at_ms, lenient)? {
        let logical = ovr.logical.clone().unwrap_or_else(|| trimmed.to_string());
        let cls = match &ovr.class {
            Some(c) => c.clone(),
            None => classify_ranks(&logical, lenient)?.class.unwrap_or_default(),
        };
        let inferred = ranks_in_class(&cls, &logical, lenient)?;
        let mut identity = ModelIdentity {
            class: cls,
            ..Default::default()
        };
        identity.family = ovr.family.clone().or(inferred.family);
        identity.revision = ovr.revision.clone().or(inferred.revision);
        identity.effort = ovr.effort;
        if ovr.thinking_variant {
            identity.thinking_variant = Some(true);
        }
        if logical != trimmed {
            identity.logical_id = Some(logical);
        }
        return Ok(identity);
    }
    let collapsed: CollapsedVariant = if trimmed.len() == model_id.len() {
        collapse_variant_id(provider, trimmed)
    } else {
        CollapsedVariant {
            logical_id: trimmed.to_string(),
            effort: None,
            thinking_variant: false,
        }
    };
    let ranks = classify_ranks(&collapsed.logical_id, lenient)?;
    let mut identity = ModelIdentity {
        class: ranks.class.unwrap_or_else(|| "unknown".into()),
        family: ranks.family,
        revision: ranks.revision,
        effort: collapsed.effort,
        thinking_variant: if collapsed.thinking_variant {
            Some(true)
        } else {
            None
        },
        ..Default::default()
    };
    if collapsed.logical_id != trimmed {
        identity.logical_id = Some(collapsed.logical_id);
    }
    Ok(identity)
}

fn classify_memo_key(provider: &str, model_id: &str, lenient: bool) -> String {
    format!(
        "{}:{}{}:{}{}",
        provider.len(),
        provider,
        model_id.len(),
        model_id,
        if lenient { 1 } else { 0 }
    )
}

/// Classifies a model into its structured identity: reviewed override first,
/// then suffix collapse, then class/family/revision ranks over the logical id.
///
/// Results are memoized per (provider, model, lenient) unless `observed_at_ms`
/// is set, so override expiry stays exact. Callers receive a fresh copy.
///
/// # Errors
/// Ambiguous equal-rank cross-class, cross-family, or reviewed-pattern
/// matches unless `opts.lenient`.
pub fn classify_model(
    provider: &str,
    model_id: &str,
    opts: Option<ClassifyOptions>,
) -> Result<ModelIdentity> {
    let opts = opts.unwrap_or_default();
    if opts.observed_at_ms.is_none() {
        let key = classify_memo_key(provider, model_id, opts.lenient);
        static MEMO: std::sync::Mutex<Option<HashMap<String, ModelIdentity>>> =
            std::sync::Mutex::new(None);
        let mut memo = MEMO.lock().expect("memo lock poisoned");
        let memo = memo.get_or_insert_with(HashMap::new);
        if let Some(cached) = memo.get(&key) {
            return Ok(cached.clone());
        }
        let identity = classify_model_uncached(provider, model_id, opts)?;
        memo.insert(key, identity.clone());
        return Ok(identity);
    }
    classify_model_uncached(provider, model_id, opts)
}
