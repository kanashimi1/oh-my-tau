//! Runtime compat cascade (TS `compat/cascade.ts`): per-axis wire/thinking/catalog assignments for
//! one structured model target from the compiled rule tree. Rules are conjunctions over
//! `(class, provider, api, upstream, family, revision, models)`; per axis the matching rule with
//! the greatest `(model-selector exactness, constrained-dimension count, priority)` tuple wins and
//! an equal-tuple same-axis contest is an error.

use std::collections::BTreeMap;

use serde::Deserialize;
use serde_json::{Map, Value};

use super::revision::{parse_revision, revision_satisfies, Revision, RevisionOp, RevisionTerm};
use super::taxonomy::glob_match;
use super::Rules;

#[derive(Debug, thiserror::Error)]
#[error("ambiguous overlap for `{provider}/{model}` on axis `{axis}`: rules `{first}` and `{second}` tie; add an explicit priority")]
pub struct AmbiguousOverlap {
    pub provider: String,
    pub model: String,
    pub axis: String,
    pub first: String,
    pub second: String,
}

#[derive(Debug, Deserialize)]
pub struct Selector {
    pub kind: String,
    pub value: String,
}

#[derive(Debug, Deserialize)]
struct RawTerm {
    op: String,
    revision: String,
}

#[derive(Debug, Deserialize)]
struct RawRule {
    source: String,
    class: Option<String>,
    providers: Option<Vec<String>>,
    apis: Option<Vec<String>>,
    upstreams: Option<Vec<String>>,
    family: Option<String>,
    revision: Option<Vec<RawTerm>>,
    models: Option<Vec<Selector>>,
    priority: Option<i64>,
    wire: Option<Map<String, Value>>,
    thinking: Option<Map<String, Value>>,
    catalog: Option<Map<String, Value>>,
}

#[derive(Debug)]
pub struct Rule {
    pub source: String,
    pub class: Option<String>,
    pub providers: Option<Vec<String>>,
    pub apis: Option<Vec<String>>,
    pub upstreams: Option<Vec<String>>,
    pub family: Option<String>,
    revision: Option<Vec<RevisionTerm>>,
    pub models: Option<Vec<Selector>>,
    priority: i64,
    dimensions: i64,
    has_exact_efforts_rule: bool,
    pub wire: Option<Map<String, Value>>,
    pub thinking: Option<Map<String, Value>>,
    pub catalog: Option<Map<String, Value>>,
}

#[derive(Debug, Deserialize)]
struct RawCascade {
    rules: Vec<RawRule>,
}

#[derive(Debug, Deserialize)]
#[serde(try_from = "RawCascade")]
pub struct Cascade {
    pub rules: Vec<Rule>,
}

impl TryFrom<RawCascade> for Cascade {
    type Error = String;
    fn try_from(raw: RawCascade) -> Result<Self, String> {
        let mut rules = Vec::with_capacity(raw.rules.len());
        for r in raw.rules {
            let revision = match &r.revision {
                Some(terms) => {
                    let mut out = Vec::new();
                    for t in terms {
                        let revision = parse_revision(&t.revision).ok_or_else(|| {
                            format!("invalid compiled revision term in {}", r.source)
                        })?;
                        let op = RevisionOp::parse(&t.op)
                            .ok_or_else(|| format!("bad revision op in {}", r.source))?;
                        out.push(RevisionTerm { op, revision });
                    }
                    Some(out)
                }
                None => None,
            };
            let dimensions = i64::from(r.class.is_some())
                + i64::from(r.providers.is_some())
                + i64::from(r.apis.is_some())
                + i64::from(r.upstreams.is_some())
                + i64::from(r.family.is_some())
                + i64::from(r.revision.is_some())
                + i64::from(r.models.is_some());
            let has_exact_efforts_rule = r
                .thinking
                .as_ref()
                .is_some_and(|t| t.contains_key("efforts"));
            rules.push(Rule {
                source: r.source,
                class: r.class,
                providers: r.providers,
                apis: r.apis,
                upstreams: r.upstreams,
                family: r.family,
                revision,
                models: r.models,
                priority: r.priority.unwrap_or(0),
                dimensions,
                has_exact_efforts_rule,
                wire: r.wire,
                thinking: r.thinking,
                catalog: r.catalog,
            });
        }
        Ok(Self { rules })
    }
}

/// Structured identity and capability input to the cascade.
#[derive(Debug, Clone)]
pub struct Target {
    pub provider: String,
    pub api: String,
    pub upstream: Option<String>,
    pub class: String,
    pub family: Option<String>,
    pub revision: Option<String>,
    pub model: String,
    pub reasoning: bool,
}

#[derive(Debug, Default)]
pub struct ResolvedAxes {
    pub wire: Map<String, Value>,
    pub thinking: Map<String, Value>,
    pub catalog: Map<String, Value>,
    pub reasoning: bool,
}

type Rank = (i64, i64, i64);

struct Prepared<'a> {
    target: &'a Target,
    revision: Option<Revision>,
    model_lower: String,
    model_tokens: Vec<String>,
}

fn prepare(target: &Target) -> Prepared<'_> {
    let model_lower = target.model.to_lowercase();
    let model_tokens = model_lower
        .split(|c: char| !(c.is_ascii_lowercase() || c.is_ascii_digit()))
        .map(str::to_string)
        .collect();
    Prepared {
        target,
        revision: target.revision.as_deref().and_then(parse_revision),
        model_lower,
        model_tokens,
    }
}

fn selector_matches(s: &Selector, p: &Prepared<'_>) -> bool {
    match s.kind.as_str() {
        "exact" => s.value == p.target.model,
        "glob" => glob_match(&s.value, &p.model_lower),
        "token" => p.model_tokens.iter().any(|t| *t == s.value),
        _ => false,
    }
}

fn rank_rule(rule: &Rule, p: &Prepared<'_>) -> Option<Rank> {
    let t = p.target;
    if rule.class.as_ref().is_some_and(|c| *c != t.class) {
        return None;
    }
    if rule
        .providers
        .as_ref()
        .is_some_and(|ps| !ps.contains(&t.provider))
    {
        return None;
    }
    if rule.apis.as_ref().is_some_and(|a| !a.contains(&t.api)) {
        return None;
    }
    if let Some(ups) = &rule.upstreams {
        match &t.upstream {
            Some(u) if ups.contains(u) => {}
            _ => return None,
        }
    }
    if rule
        .family
        .as_ref()
        .is_some_and(|f| t.family.as_ref() != Some(f))
    {
        return None;
    }
    if let Some(terms) = &rule.revision {
        match &p.revision {
            Some(r) if revision_satisfies(r, terms) => {}
            _ => return None,
        }
    }
    let mut exactness = 0;
    if let Some(models) = &rule.models {
        let mut best = -1;
        for s in models {
            if !selector_matches(s, p) {
                continue;
            }
            let v = if s.kind == "exact" { 2 } else { 1 };
            if v > best {
                best = v;
            }
        }
        if best < 0 {
            return None;
        }
        exactness = best;
    }
    Some((exactness, rule.dimensions, rule.priority))
}

fn ranked<'a>(cascade: &'a Cascade, p: &Prepared<'_>) -> Vec<(&'a Rule, Rank)> {
    cascade
        .rules
        .iter()
        .filter_map(|r| rank_rule(r, p).map(|rank| (r, rank)))
        .collect()
}

type Winners<'a> = BTreeMap<String, (Rank, &'a Rule)>;

fn contest_axis<'a>(
    winners: &mut Winners<'a>,
    axis: &str,
    rank: Rank,
    rule: &'a Rule,
    target: &Target,
) -> Result<(), AmbiguousOverlap> {
    if let Some((held_rank, held_rule)) = winners.get(axis) {
        match held_rank.cmp(&rank) {
            std::cmp::Ordering::Equal => {
                return Err(AmbiguousOverlap {
                    provider: target.provider.clone(),
                    model: target.model.clone(),
                    axis: axis.into(),
                    first: held_rule.source.clone(),
                    second: rule.source.clone(),
                })
            }
            std::cmp::Ordering::Greater => return Ok(()),
            std::cmp::Ordering::Less => {}
        }
    }
    winners.insert(axis.to_string(), (rank, rule));
    Ok(())
}

fn contest<'a>(
    winners: &mut Winners<'a>,
    axes: Option<&Map<String, Value>>,
    rank: Rank,
    rule: &'a Rule,
    target: &Target,
) -> Result<(), AmbiguousOverlap> {
    if let Some(axes) = axes {
        for axis in axes.keys() {
            contest_axis(winners, axis, rank, rule, target)?;
        }
    }
    Ok(())
}

fn collect(
    winners: &Winners<'_>,
    pick: impl Fn(&Rule) -> Option<&Map<String, Value>>,
) -> Map<String, Value> {
    let mut out = Map::new();
    for (axis, (_, rule)) in winners {
        if let Some(v) = pick(rule).and_then(|m| m.get(axis)) {
            out.insert(axis.clone(), v.clone());
        }
    }
    out
}

fn identity_scoped(rule: &Rule) -> bool {
    rule.class.as_ref().is_some_and(|c| c != "unknown")
        || rule.family.is_some()
        || rule.revision.is_some()
        || rule.models.is_some()
}

/// Resolve wire, thinking, and catalog assignments for one structured target.
pub fn resolve_cascade(rules: &Rules, target: &Target) -> Result<ResolvedAxes, AmbiguousOverlap> {
    let cascade = &rules.cascade;
    let prepared = prepare(target);
    let ranked = ranked(cascade, &prepared);
    let mut reasoning = target.reasoning;
    if !reasoning {
        let mut upgrade: Winners<'_> = BTreeMap::new();
        let mut has_efforts = false;
        for (rule, rank) in &ranked {
            let Some(thinking) = &rule.thinking else {
                continue;
            };
            if thinking.contains_key("upgradeNeutral") {
                contest_axis(&mut upgrade, "upgradeNeutral", *rank, rule, target)?;
            }
            if thinking.contains_key("efforts") {
                has_efforts = true;
            }
            if rule.has_exact_efforts_rule && rank.0 == 2 {
                reasoning = true;
            }
        }
        if let Some((_, upgrade_rule)) = upgrade.get("upgradeNeutral") {
            let neutral = upgrade_rule
                .thinking
                .as_ref()
                .and_then(|t| t.get("upgradeNeutral"))
                .and_then(Value::as_bool)
                == Some(true);
            if identity_scoped(upgrade_rule) && neutral && has_efforts {
                reasoning = true;
            }
        }
    }
    let mut wire: Winners<'_> = BTreeMap::new();
    let mut thinking: Winners<'_> = BTreeMap::new();
    let mut catalog: Winners<'_> = BTreeMap::new();
    for (rule, rank) in &ranked {
        contest(&mut wire, rule.wire.as_ref(), *rank, rule, target)?;
        contest(&mut catalog, rule.catalog.as_ref(), *rank, rule, target)?;
        if reasoning {
            contest(&mut thinking, rule.thinking.as_ref(), *rank, rule, target)?;
        }
    }
    Ok(ResolvedAxes {
        wire: collect(&wire, |r| r.wire.as_ref()),
        thinking: collect(&thinking, |r| r.thinking.as_ref()),
        catalog: collect(&catalog, |r| r.catalog.as_ref()),
        reasoning,
    })
}

/// Whether the effort ladder a target resolves to comes from a rule scoped to the model's identity.
pub fn has_model_scoped_efforts_rule(
    rules: &Rules,
    target: &Target,
) -> Result<bool, AmbiguousOverlap> {
    let prepared = prepare(target);
    let mut winners: Winners<'_> = BTreeMap::new();
    for (rule, rank) in ranked(&rules.cascade, &prepared) {
        contest(&mut winners, rule.thinking.as_ref(), rank, rule, target)?;
    }
    Ok(winners
        .get("efforts")
        .is_some_and(|(_, r)| identity_scoped(r)))
}
