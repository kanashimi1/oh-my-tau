//! Model-identity taxonomy (TS `compat/taxonomy.ts`): classifies wire model ids into
//! `(class, family, revision)`, applies reviewed identity overrides and suffix collapse.

use serde::Deserialize;
use serde_json::{Map, Value};

use super::revision::{format_revision, parse_revision_prefix};
use super::Rules;

#[derive(Debug, thiserror::Error)]
pub enum AmbiguousIdentity {
    #[error("ambiguous {kind} for `{model}`: `{first}` and `{second}` tie")]
    Tie {
        model: String,
        first: String,
        second: String,
        kind: &'static str,
    },
}

#[derive(Debug, Deserialize)]
pub struct Matcher {
    pub kind: String,
    pub token: String,
    #[serde(default)]
    pub bounded: bool,
}

#[derive(Debug, Deserialize)]
pub struct Family {
    pub id: String,
    pub glob: String,
    #[serde(default)]
    pub priority: i64,
}

#[derive(Debug, Deserialize)]
pub struct RevisionPrefix {
    pub prefix: String,
    #[serde(default)]
    pub anywhere: bool,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Override {
    pub id: String,
    pub model: Option<String>,
    pub glob: Option<String>,
    pub provider: Option<String>,
    pub logical: Option<String>,
    pub class: Option<String>,
    pub family: Option<String>,
    pub revision: Option<String>,
    pub effort: Option<String>,
    #[serde(default)]
    pub thinking_variant: bool,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Class {
    pub id: String,
    pub matchers: Vec<Matcher>,
    pub families: Vec<Family>,
    #[serde(default)]
    pub revision_prefixes: Vec<RevisionPrefix>,
    #[serde(default)]
    pub skip_bare: Vec<String>,
    #[serde(default)]
    pub overrides: Vec<Override>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Suffix {
    pub suffix: String,
    #[serde(default)]
    pub thinking: bool,
    pub effort: Option<String>,
    pub except_bare_prefix: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Lane {
    pub suffix: String,
    pub providers: Vec<String>,
    pub bare_prefix: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct EffortFamily {
    pub provider: String,
    pub logical: String,
    pub aliases: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Collapse {
    pub suffixes: Vec<Suffix>,
    pub lanes: Vec<Lane>,
    pub effort_families: Vec<EffortFamily>,
    pub pair_tokens: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct Taxonomy {
    pub classes: Vec<Class>,
    pub collapse: Collapse,
}

/// Structured identity of one classified model (TS `ModelIdentity`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Identity {
    pub class: String,
    pub family: Option<String>,
    pub revision: Option<String>,
    pub effort: Option<String>,
    pub thinking_variant: bool,
    pub logical_id: Option<String>,
}

impl Identity {
    pub fn to_json(&self) -> Value {
        let mut m = Map::new();
        m.insert("class".into(), self.class.clone().into());
        if let Some(v) = &self.family {
            m.insert("family".into(), v.clone().into());
        }
        if let Some(v) = &self.revision {
            m.insert("revision".into(), v.clone().into());
        }
        if let Some(v) = &self.effort {
            m.insert("effort".into(), v.clone().into());
        }
        if self.thinking_variant {
            m.insert("thinkingVariant".into(), true.into());
        }
        if let Some(v) = &self.logical_id {
            m.insert("logicalId".into(), v.clone().into());
        }
        Value::Object(m)
    }
}

/// Anchored `*`-wildcard match; both sides pre-lowercased.
pub fn glob_match(pattern: &str, value: &str) -> bool {
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

fn matcher_rank(kind: &str) -> i32 {
    match kind {
        "exact" => 4,
        "bounded" => 3,
        "namespace" => 2,
        "prefix" => 1,
        _ => 0,
    }
}

fn bare_of(id: &str) -> &str {
    id.rfind('/').map_or(id, |i| &id[i + 1..])
}

fn bounded_match(value: &str, token: &str) -> bool {
    if value == token {
        return true;
    }
    if !value.starts_with(token) {
        return false;
    }
    matches!(
        value.as_bytes().get(token.len()),
        Some(b'-' | b'_' | b'.' | b':' | b'0'..=b'9')
    )
}

fn matcher_matches(m: &Matcher, lower: &str, bare: &str) -> bool {
    match m.kind.as_str() {
        "exact" => bare == m.token,
        "bounded" => bounded_match(bare, &m.token),
        "namespace" => {
            if m.bounded {
                lower
                    .split(['/', '.', ':'])
                    .any(|p| !p.is_empty() && bounded_match(p, &m.token))
            } else {
                lower.split('/').any(|p| !p.is_empty() && p == m.token)
            }
        }
        "prefix" => bare.starts_with(&m.token),
        _ => glob_match(&m.token, bare),
    }
}

fn non_wildcard_bytes(glob: &str) -> usize {
    glob.bytes().filter(|b| *b != b'*').count()
}

struct FamilyRanking<'a> {
    winner: Option<((i64, usize), &'a str)>,
    tied: Option<(&'a str, &'a str)>,
}

fn rank_families<'a>(cls: &'a Class, subject: &str) -> FamilyRanking<'a> {
    let mut winner: Option<((i64, usize), &str)> = None;
    let mut tied = None;
    for family in &cls.families {
        if !glob_match(&family.glob, subject) {
            continue;
        }
        let rank = (family.priority, non_wildcard_bytes(&family.glob));
        match winner {
            Some((w, id)) if w == rank && id != family.id => tied = Some((id, family.id.as_str())),
            Some((w, _)) if w >= rank => {}
            _ => {
                winner = Some((rank, family.id.as_str()));
                tied = None;
            }
        }
    }
    FamilyRanking { winner, tied }
}

fn find_dot_or_colon(s: &str) -> Option<usize> {
    s.find(['.', ':'])
}

fn strip_class_namespace<'a>(cls: &Class, bare: &'a str) -> Option<&'a str> {
    let sep = find_dot_or_colon(bare)?;
    if sep == 0 || sep == bare.len() - 1 {
        return None;
    }
    let namespace = &bare[..sep];
    if !cls.matchers.iter().any(|m| m.token == namespace) {
        return None;
    }
    Some(&bare[sep + 1..])
}

fn classify_family(
    cls: &Class,
    bare: &str,
    model: &str,
    lenient: bool,
) -> Result<Option<String>, AmbiguousIdentity> {
    let initial = rank_families(cls, bare);
    let Some((a, b)) = initial.tied else {
        return Ok(initial.winner.map(|(_, id)| id.to_string()));
    };
    if let Some(scoped) = strip_class_namespace(cls, bare) {
        let rescored = rank_families(cls, scoped);
        if let (Some((_, id)), None) = (rescored.winner, rescored.tied) {
            return Ok(Some(id.to_string()));
        }
    }
    if lenient {
        return Ok(None);
    }
    Err(AmbiguousIdentity::Tie {
        model: model.into(),
        first: a.into(),
        second: b.into(),
        kind: "family",
    })
}

fn extract_revision(cls: &Class, bare: &str) -> Option<String> {
    if cls.skip_bare.iter().any(|s| s == bare) {
        return None;
    }
    for rule in &cls.revision_prefixes {
        let tail = if rule.anywhere {
            bare.find(&rule.prefix)
                .map(|start| &bare[start + rule.prefix.len()..])
        } else {
            bare.strip_prefix(rule.prefix.as_str())
        };
        let Some(tail) = tail else { continue };
        let digit = tail.find(|c: char| c.is_ascii_digit())?;
        return parse_revision_prefix(&tail[digit..]).map(|r| format_revision(&r));
    }
    None
}

struct ClassRanks {
    class: String,
    family: Option<String>,
    revision: Option<String>,
}

fn classify_ranks(
    rules: &Rules,
    model: &str,
    lenient: bool,
) -> Result<ClassRanks, AmbiguousIdentity> {
    let lower = model.trim().to_lowercase();
    let bare = bare_of(&lower);
    let mut winner: Option<((i32, usize), &Class)> = None;
    let mut tied: Option<(&str, &str)> = None;
    for cls in &rules.taxonomy.classes {
        for matcher in &cls.matchers {
            if !matcher_matches(matcher, &lower, bare) {
                continue;
            }
            let rank = (matcher_rank(&matcher.kind), matcher.token.len());
            match winner {
                Some((w, c)) if w == rank && c.id != cls.id => {
                    tied = Some((c.id.as_str(), cls.id.as_str()))
                }
                Some((w, _)) if w >= rank => {}
                _ => {
                    winner = Some((rank, cls));
                    tied = None;
                }
            }
        }
    }
    let unknown = || ClassRanks {
        class: "unknown".into(),
        family: None,
        revision: None,
    };
    if let Some((a, b)) = tied {
        if lenient {
            return Ok(unknown());
        }
        return Err(AmbiguousIdentity::Tie {
            model: lower.clone(),
            first: a.into(),
            second: b.into(),
            kind: "class",
        });
    }
    let Some((_, cls)) = winner else {
        return Ok(unknown());
    };
    Ok(ClassRanks {
        class: cls.id.clone(),
        family: classify_family(cls, bare, &lower, lenient)?,
        revision: extract_revision(cls, bare),
    })
}

fn ranks_in_class(
    rules: &Rules,
    class_id: &str,
    model: &str,
    lenient: bool,
) -> Result<(Option<String>, Option<String>), AmbiguousIdentity> {
    let Some(cls) = rules.taxonomy.classes.iter().find(|c| c.id == class_id) else {
        return Ok((None, None));
    };
    let lower = model.trim().to_lowercase();
    let bare = bare_of(&lower);
    Ok((
        classify_family(cls, bare, &lower, lenient)?,
        extract_revision(cls, bare),
    ))
}

fn find_identity_override<'a>(
    rules: &'a Rules,
    provider: &str,
    bare_model: &str,
    lenient: bool,
) -> Result<Option<&'a Override>, AmbiguousIdentity> {
    let lower_provider = provider.to_lowercase();
    let lower_bare = bare_model.to_lowercase();
    let all = || {
        rules
            .taxonomy
            .classes
            .iter()
            .flat_map(|c| c.overrides.iter())
    };
    // Exact index: first provider-scoped hit, else first agnostic hit.
    let exact: Vec<&Override> = all()
        .filter(|o| {
            o.model
                .as_ref()
                .is_some_and(|m| m.to_lowercase() == lower_bare)
        })
        .collect();
    if let Some(o) = exact.iter().find(|o| {
        o.provider
            .as_ref()
            .is_some_and(|p| p.to_lowercase() == lower_provider)
    }) {
        return Ok(Some(o));
    }
    if let Some(o) = exact.iter().find(|o| o.provider.is_none()) {
        return Ok(Some(o));
    }
    let patterns: Vec<&Override> = all()
        .filter(|o| o.model.is_none() && o.glob.is_some())
        .collect();
    let rank = |provider: Option<&str>| {
        let mut winner: Option<(usize, &Override)> = None;
        let mut tied: Option<(&Override, &Override)> = None;
        for o in &patterns {
            let op = o.provider.as_ref().map(|p| p.to_lowercase());
            if op.as_deref() != provider {
                continue;
            }
            let glob = o.glob.as_deref().unwrap_or_default();
            if !glob_match(glob, &lower_bare) {
                continue;
            }
            let specificity = non_wildcard_bytes(glob);
            match winner {
                Some((s, w)) if s == specificity => tied = Some((w, o)),
                Some((s, _)) if s > specificity => {}
                _ => {
                    winner = Some((specificity, o));
                    tied = None;
                }
            }
        }
        (winner.map(|(_, o)| o), tied)
    };
    let (mut winner, mut tied) = rank(Some(&lower_provider));
    if winner.is_none() && tied.is_none() {
        (winner, tied) = rank(None);
    }
    if let Some((a, b)) = tied {
        if lenient {
            return Ok(None);
        }
        return Err(AmbiguousIdentity::Tie {
            model: lower_bare,
            first: a.id.clone(),
            second: b.id.clone(),
            kind: "override",
        });
    }
    Ok(winner)
}

pub struct Collapsed {
    pub logical_id: String,
    pub effort: Option<String>,
    pub thinking_variant: bool,
}

pub fn collapse_variant_id(rules: &Rules, provider: &str, model: &str) -> Collapsed {
    let collapse = &rules.taxonomy.collapse;
    let lower = model.to_lowercase();
    let lower_provider = provider.to_lowercase();
    for family in &collapse.effort_families {
        if family.provider == lower_provider && family.aliases.contains(&lower) {
            return Collapsed {
                logical_id: family.logical.clone(),
                effort: None,
                thinking_variant: false,
            };
        }
    }
    let bare = bare_of(&lower);
    let mut winner: Option<&Suffix> = None;
    for rule in &collapse.suffixes {
        if !lower.ends_with(&rule.suffix) {
            continue;
        }
        if rule
            .except_bare_prefix
            .as_ref()
            .is_some_and(|p| bare.starts_with(p.as_str()))
        {
            continue;
        }
        if winner.is_none_or(|w| rule.suffix.len() > w.suffix.len()) {
            winner = Some(rule);
        }
    }
    if let Some(w) = winner {
        return Collapsed {
            logical_id: model[..model.len() - w.suffix.len()].to_string(),
            effort: w.effort.clone(),
            thinking_variant: w.thinking,
        };
    }
    for lane in &collapse.lanes {
        if !lane.providers.contains(&lower_provider) || !lower.ends_with(&lane.suffix) {
            continue;
        }
        let trimmed = &lower[..lower.len() - lane.suffix.len()];
        let trimmed_bare = bare_of(trimmed);
        if lane
            .bare_prefix
            .as_ref()
            .is_some_and(|p| !trimmed_bare.starts_with(p.as_str()))
        {
            continue;
        }
        let mut effort_rule: Option<&Suffix> = None;
        for rule in &collapse.suffixes {
            if rule.effort.is_none() || !trimmed.ends_with(&rule.suffix) {
                continue;
            }
            if rule
                .except_bare_prefix
                .as_ref()
                .is_some_and(|p| trimmed_bare.starts_with(p.as_str()))
            {
                continue;
            }
            if effort_rule.is_none_or(|w| rule.suffix.len() > w.suffix.len()) {
                effort_rule = Some(rule);
            }
        }
        let Some(effort_rule) = effort_rule else {
            continue;
        };
        let base = &model[..trimmed.len() - effort_rule.suffix.len()];
        if base.is_empty() || base.ends_with('/') {
            continue;
        }
        return Collapsed {
            logical_id: format!("{base}{}", &model[trimmed.len()..]),
            effort: effort_rule.effort.clone(),
            thinking_variant: false,
        };
    }
    Collapsed {
        logical_id: model.to_string(),
        effort: None,
        thinking_variant: false,
    }
}

/// First declared thinking-variant suffix token removed from a model id (`None` when absent).
pub fn strip_thinking_variant_suffix(rules: &Rules, model: &str) -> Option<String> {
    let lower = model.to_lowercase();
    let lb = lower.as_bytes();
    for token in &rules.taxonomy.collapse.pair_tokens {
        let needle = format!("-{token}");
        let mut search_from = 0usize;
        while search_from < lower.len() {
            let Some(rel) = lower[search_from..].find(&needle) else {
                break;
            };
            let index = search_from + rel;
            let end = index + needle.len();
            let followed = lb
                .get(end)
                .is_some_and(|c| c.is_ascii_digit() || c.is_ascii_lowercase());
            let mut word_start = index;
            while word_start > 0 {
                let c = lb[word_start - 1];
                if !(c.is_ascii_digit() || c.is_ascii_lowercase()) {
                    break;
                }
                word_start -= 1;
            }
            let preceding = &lower[word_start..index];
            if !followed && preceding != "non" && preceding != "no" {
                let stripped = format!("{}{}", &model[..index], &model[end..]);
                return (!stripped.is_empty()).then_some(stripped);
            }
            search_from = index + 1;
        }
    }
    None
}

/// Reviewed override first, then suffix collapse, then class/family/revision ranks.
pub fn classify_model(
    rules: &Rules,
    provider: &str,
    model_id: &str,
    lenient: bool,
) -> Result<Identity, AmbiguousIdentity> {
    let trimmed = model_id.trim();
    let bare = bare_of(trimmed);
    if let Some(ov) = find_identity_override(rules, provider, bare, lenient)? {
        let logical = ov.logical.clone().unwrap_or_else(|| trimmed.to_string());
        let class = match &ov.class {
            Some(c) => c.clone(),
            None => classify_ranks(rules, &logical, lenient)?.class,
        };
        let (inf_family, inf_rev) = ranks_in_class(rules, &class, &logical, lenient)?;
        return Ok(Identity {
            class,
            family: ov.family.clone().or(inf_family),
            revision: ov.revision.clone().or(inf_rev),
            effort: ov.effort.clone(),
            thinking_variant: ov.thinking_variant,
            logical_id: (logical != trimmed).then_some(logical),
        });
    }
    let collapsed = if trimmed.len() == model_id.len() {
        collapse_variant_id(rules, provider, trimmed)
    } else {
        Collapsed {
            logical_id: trimmed.to_string(),
            effort: None,
            thinking_variant: false,
        }
    };
    let ranks = classify_ranks(rules, &collapsed.logical_id, lenient)?;
    Ok(Identity {
        class: ranks.class,
        family: ranks.family,
        revision: ranks.revision,
        effort: collapsed.effort,
        thinking_variant: collapsed.thinking_variant,
        logical_id: (collapsed.logical_id != trimmed).then_some(collapsed.logical_id),
    })
}
