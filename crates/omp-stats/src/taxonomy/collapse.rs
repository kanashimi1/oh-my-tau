//! Suffix-vocabulary collapse helpers shared by `classify_model`.
//!
//! Port of `collapseVariantId`, `stripThinkingVariantSuffix` and the
//! discovery-plain helpers from `catalog/src/compat/taxonomy.ts`.

use super::rules::{taxonomy_rules, Effort};

/// Result of collapsing a wire id through the suffix vocabulary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollapsedVariant {
    /// Logical id after suffix collapse (original bytes preserved where possible).
    pub logical_id: String,
    /// Effort tier collapsed out of the id, when it was an effort variant.
    pub effort: Option<Effort>,
    /// Whether the id carried a thinking-variant suffix.
    pub thinking_variant: bool,
}

fn bare_of(id: &str) -> &str {
    match id.rfind('/') {
        Some(slash) => &id[slash + 1..],
        None => id,
    }
}

/// Collapses a declared thinking or effort suffix from a model identifier.
///
/// Exact `effort-family` aliases collapse to the family's logical model
/// without assigning an effort; provider-scoped effort lanes additionally
/// collapse an effort suffix wedged before the lane token.
pub fn collapse_variant_id(provider: &str, model: &str) -> CollapsedVariant {
    let collapse = &taxonomy_rules().collapse;
    let lower = model.to_lowercase();
    let lower_provider = provider.to_lowercase();
    for family in &collapse.effort_families {
        if family.provider == lower_provider && family.aliases.contains(&lower) {
            return CollapsedVariant {
                logical_id: family.logical.clone(),
                effort: None,
                thinking_variant: false,
            };
        }
    }
    let bare = bare_of(&lower);
    let mut winner: Option<&super::rules::CompiledCollapseSuffix> = None;
    for rule in &collapse.suffixes {
        if !lower.ends_with(&rule.suffix) {
            continue;
        }
        if let Some(prefix) = &rule.except_bare_prefix {
            if bare.starts_with(prefix) {
                continue;
            }
        }
        match winner {
            Some(w) if w.suffix.len() >= rule.suffix.len() => {}
            _ => winner = Some(rule),
        }
    }
    if let Some(rule) = winner {
        return CollapsedVariant {
            logical_id: model[..model.len() - rule.suffix.len()].to_string(),
            effort: rule.effort,
            thinking_variant: rule.thinking,
        };
    }
    for lane in &collapse.lanes {
        if !lane.providers.iter().any(|c| c == &lower_provider) || !lower.ends_with(&lane.suffix) {
            continue;
        }
        let trimmed = &model[..model.len() - lane.suffix.len()];
        let trimmed_lower = &lower[..lower.len() - lane.suffix.len()];
        let trimmed_bare = bare_of(trimmed_lower);
        if let Some(prefix) = &lane.bare_prefix {
            if !trimmed_bare.starts_with(prefix) {
                continue;
            }
        }
        // The lane wraps effort tiers only; thinking variants never lane.
        let mut effort_rule: Option<&super::rules::CompiledCollapseSuffix> = None;
        for rule in &collapse.suffixes {
            if rule.effort.is_none() || !trimmed_lower.ends_with(&rule.suffix) {
                continue;
            }
            if let Some(prefix) = &rule.except_bare_prefix {
                if trimmed_bare.starts_with(prefix) {
                    continue;
                }
            }
            match effort_rule {
                Some(w) if w.suffix.len() >= rule.suffix.len() => {}
                _ => effort_rule = Some(rule),
            }
        }
        let Some(effort_rule) = effort_rule else {
            continue;
        };
        let base = &trimmed[..trimmed.len() - effort_rule.suffix.len()];
        if base.is_empty() || base.ends_with('/') {
            continue;
        }
        // Preserve the caller's original lane bytes on the logical id.
        return CollapsedVariant {
            logical_id: format!("{}{}", base, &model[trimmed.len()..]),
            effort: effort_rule.effort,
            thinking_variant: false,
        };
    }
    CollapsedVariant {
        logical_id: model.to_string(),
        effort: None,
        thinking_variant: false,
    }
}

/// Removes the first declared thinking-variant suffix token from a model id.
///
/// Unlike [`collapse_variant_id`], this vocabulary-only helper also handles an
/// infix token used to pair live discovery siblings. Negated forms such as
/// `non-thinking` and `no-thinking` are not variants.
pub fn strip_thinking_variant_suffix(model: &str) -> Option<String> {
    // ASCII-only lowercasing keeps byte offsets aligned with `model` (full
    // Unicode lowercasing can change byte lengths and misplace the slices
    // below); tokens and word characters are ASCII, so matching is unchanged.
    let lower = model.to_ascii_lowercase();
    for token in &taxonomy_rules().collapse.pair_tokens {
        let needle = format!("-{}", token);
        let mut search_from = 0usize;
        while search_from < lower.len() {
            let Some(found) = lower[search_from..].find(&needle) else {
                break;
            };
            let index = search_from + found;
            let end = index + needle.len();
            let next = lower.as_bytes().get(end).copied().unwrap_or(0);
            let followed_by_token_character =
                (48..=57).contains(&next) || (97..=122).contains(&next);
            let mut word_start = index;
            while word_start > 0 {
                let code = lower.as_bytes()[word_start - 1];
                if !((48..=57).contains(&code) || (97..=122).contains(&code)) {
                    break;
                }
                word_start -= 1;
            }
            let preceding = &lower[word_start..index];
            if !followed_by_token_character && preceding != "non" && preceding != "no" {
                let stripped = format!("{}{}", &model[..index], &model[end..]);
                return if stripped.is_empty() {
                    None
                } else {
                    Some(stripped)
                };
            }
            search_from = index + 1;
        }
    }
    None
}

/// Strips a declared billing-variant suffix (`-free`, `-contributor`) from a
/// wire identifier, returning the base id it shares a transport with.
pub fn billing_variant_plain(wire_model: &str) -> Option<String> {
    for suffix in &taxonomy_rules().discovery.billing_variant_suffixes {
        // TS `if (split <= 0) continue`: a suffix longer than the id skips to
        // the next one instead of ending the search.
        let Some(split) = wire_model.len().checked_sub(suffix.len()) else {
            continue;
        };
        if split == 0 || !wire_model.is_char_boundary(split) {
            continue;
        }
        if wire_model[split..].to_lowercase() == *suffix {
            return Some(wire_model[..split].to_string());
        }
    }
    None
}

/// The standard-lane id when `model` ends in a declared effort lane for `provider`.
pub fn strip_effort_lane(provider: &str, model: &str) -> String {
    let lower_provider = provider.to_lowercase();
    for lane in &taxonomy_rules().collapse.lanes {
        if !lane.providers.contains(&lower_provider) {
            continue;
        }
        if model.len() < lane.suffix.len() {
            continue;
        }
        let split = model.len() - lane.suffix.len();
        if model[split..].to_lowercase() == lane.suffix {
            return model[..split].to_string();
        }
    }
    model.to_string()
}

/// Whether `provider`'s discovery recovers canonical intrinsic parameters.
pub fn recovers_canonical_params(provider: &str) -> bool {
    let lower = provider.to_lowercase();
    taxonomy_rules()
        .discovery
        .canonical_recovery
        .iter()
        .any(|p| p == &lower)
}
