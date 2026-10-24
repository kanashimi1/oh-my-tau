//! Port of the `@oh-my-pi/pi-ai` service-tier helpers `parser.ts` uses
//! (`ai/src/types.ts`): `parseServiceTier`, `coerceServiceTierByFamily`,
//! `serviceTierFamily`, `resolveModelServiceTier`, `shouldSendServiceTier`,
//! `realizesPriorityServiceTier`, `getPremiumServiceTierRequests`.
//!
//! The stats parser rebuilds slim models from historical session JSONL, so a
//! model here carries only `provider`, `api` and the classified identity
//! class — no discovery metadata (`serviceTiers` is always absent).

use serde_json::Value;

use crate::types::{ServiceTier, ServiceTierByFamily};

/// Slim model view the tier matrix branches on.
#[derive(Debug, Clone, Copy)]
pub struct TierModel<'a> {
    pub provider: &'a str,
    pub api: &'a str,
    /// `classify_model(...).class` (`"openai"`, `"anthropic"`, `"gemini"`, ...).
    pub identity_class: &'a str,
}

/// Parse a provider-reported `service_tier` echo into a known tier.
#[must_use]
pub fn parse_service_tier(value: &Value) -> Option<ServiceTier> {
    match value.as_str()? {
        "auto" => Some(ServiceTier::Auto),
        "default" => Some(ServiceTier::Default),
        "flex" => Some(ServiceTier::Flex),
        "scale" => Some(ServiceTier::Scale),
        "priority" => Some(ServiceTier::Priority),
        "ultrafast" => Some(ServiceTier::Ultrafast),
        _ => None,
    }
}

/// Coerce a persisted service-tier value to a [`ServiceTierByFamily`]. Newer
/// sessions store the family map directly; legacy sessions stored a single
/// scalar (`"priority"` everywhere, `"openai-only"`/`"claude-only"` scoped,
/// remaining values OpenAI-only semantics).
#[must_use]
pub fn coerce_service_tier_by_family(value: &Value) -> Option<ServiceTierByFamily> {
    if value.is_null() {
        return None;
    }
    if let Some(src) = value.as_object() {
        let mut out = ServiceTierByFamily::default();
        for (family, slot) in [
            ("openai", &mut out.openai),
            ("anthropic", &mut out.anthropic),
            ("google", &mut out.google),
        ] {
            if let Some(tier) = src.get(family).and_then(parse_service_tier) {
                *slot = Some(tier);
            }
        }
        return if out.openai.is_some() || out.anthropic.is_some() || out.google.is_some() {
            Some(out)
        } else {
            None
        };
    }
    match value.as_str()? {
        "priority" => Some(ServiceTierByFamily {
            openai: Some(ServiceTier::Priority),
            anthropic: Some(ServiceTier::Priority),
            google: Some(ServiceTier::Priority),
        }),
        "openai-only" => Some(ServiceTierByFamily {
            openai: Some(ServiceTier::Priority),
            ..ServiceTierByFamily::default()
        }),
        "claude-only" => Some(ServiceTierByFamily {
            anthropic: Some(ServiceTier::Priority),
            ..ServiceTierByFamily::default()
        }),
        "auto" => Some(ServiceTierByFamily {
            openai: Some(ServiceTier::Auto),
            ..ServiceTierByFamily::default()
        }),
        "default" => Some(ServiceTierByFamily {
            openai: Some(ServiceTier::Default),
            ..ServiceTierByFamily::default()
        }),
        "flex" => Some(ServiceTierByFamily {
            openai: Some(ServiceTier::Flex),
            ..ServiceTierByFamily::default()
        }),
        "scale" => Some(ServiceTierByFamily {
            openai: Some(ServiceTier::Scale),
            ..ServiceTierByFamily::default()
        }),
        _ => None,
    }
}

fn is_openai_service_tier_api(api: &str) -> bool {
    api == "openai-completions" || api == "openai-responses" || api == "openai-codex-responses"
}

fn excludes_inferred_openai_service_tier(provider: &str) -> bool {
    provider == "fireworks" || provider == "github-copilot"
}

fn is_openai_service_tier_model(model: &TierModel<'_>) -> bool {
    !excludes_inferred_openai_service_tier(model.provider)
        && is_openai_service_tier_api(model.api)
        && model.identity_class == "openai"
}

/// Classify a model into the service-tier family whose knob governs it.
#[must_use]
pub fn service_tier_family(model: &TierModel<'_>) -> Option<&'static str> {
    if model.provider == "openrouter" {
        return match model.identity_class {
            "anthropic" => Some("anthropic"),
            "gemini" => Some("google"),
            "openai" => Some("openai"),
            _ => None,
        };
    }
    if model.provider == "openai" || model.provider == "openai-codex" {
        return Some("openai");
    }
    if model.api == "anthropic-messages" {
        return Some("anthropic");
    }
    if model.provider == "google" || model.provider == "google-vertex" {
        return Some("google");
    }
    if is_openai_service_tier_model(model) {
        return Some("openai");
    }
    None
}

fn family_tier(tiers: &ServiceTierByFamily, family: &str) -> Option<ServiceTier> {
    match family {
        "openai" => tiers.openai,
        "anthropic" => tiers.anthropic,
        "google" => tiers.google,
        _ => None,
    }
}

/// Reduce a per-family tier map to the single wire tier for `model`.
#[must_use]
pub fn resolve_model_service_tier(
    tiers: Option<&ServiceTierByFamily>,
    model: &TierModel<'_>,
) -> Option<ServiceTier> {
    let tiers = tiers?;
    let family = service_tier_family(model)?;
    family_tier(tiers, family)
}

/// True when the tier should be sent on the wire as the provider's
/// service-tier request field. Bare-provider stats models never carry
/// discovery metadata, so `serviceTiers` gating reduces to "not reported".
#[must_use]
pub fn should_send_service_tier(service_tier: Option<ServiceTier>, model: &TierModel<'_>) -> bool {
    let Some(tier) = service_tier else {
        return false;
    };
    if tier == ServiceTier::Auto {
        return false;
    }
    if tier == ServiceTier::Ultrafast {
        return model.provider == "openai" || model.provider == "openai-codex";
    }
    if model.provider == "openai" || model.provider == "openai-codex" {
        return true;
    }
    if model.provider == "openrouter" {
        return matches!(
            tier,
            ServiceTier::Flex | ServiceTier::Scale | ServiceTier::Priority
        );
    }
    if is_openai_service_tier_model(model) {
        return true;
    }
    if model.provider == "google" {
        return tier == ServiceTier::Flex || tier == ServiceTier::Priority;
    }
    if model.provider == "google-vertex" || model.provider == "fireworks" {
        return tier == ServiceTier::Priority;
    }
    false
}

/// True when `priority` will actually be realized on the wire for `model`.
#[must_use]
pub fn realizes_priority_service_tier(
    service_tier: Option<ServiceTier>,
    model: &TierModel<'_>,
) -> bool {
    if service_tier != Some(ServiceTier::Priority) {
        return false;
    }
    if model.provider == "anthropic" {
        return true;
    }
    if model.provider == "openrouter" {
        return matches!(service_tier_family(model), Some("openai") | Some("google"));
    }
    if model.api == "anthropic-messages" {
        return false;
    }
    should_send_service_tier(service_tier, model)
}

/// Premium-request weight contributed by a request a provider bills above
/// standard. Pass `served = true` when the tier is the one the provider
/// reported serving (realization then already proven).
#[must_use]
pub fn get_premium_service_tier_requests(
    service_tier: Option<ServiceTier>,
    model: &TierModel<'_>,
    served: bool,
) -> f64 {
    let Some(tier) = service_tier else { return 0.0 };
    let provider = model.provider;
    if tier == ServiceTier::Ultrafast {
        if provider != "openai" && provider != "openai-codex" {
            return 0.0;
        }
        return if served || should_send_service_tier(service_tier, model) {
            1.0
        } else {
            0.0
        };
    }
    if tier != ServiceTier::Priority {
        return 0.0;
    }
    if !served && !realizes_priority_service_tier(service_tier, model) {
        return 0.0;
    }
    if matches!(
        provider,
        "openai" | "openai-codex" | "anthropic" | "google" | "google-vertex"
    ) {
        1.0
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_wire_tiers() {
        assert_eq!(
            parse_service_tier(&Value::from("priority")),
            Some(ServiceTier::Priority)
        );
        assert_eq!(parse_service_tier(&Value::from("bogus")), None);
        assert_eq!(parse_service_tier(&Value::Null), None);
    }

    #[test]
    fn coerces_legacy_scalars() {
        let v = Value::from("priority");
        let f = coerce_service_tier_by_family(&v).expect("map");
        assert_eq!(f.openai, Some(ServiceTier::Priority));
        assert_eq!(f.anthropic, Some(ServiceTier::Priority));
        let v = Value::from("claude-only");
        let f = coerce_service_tier_by_family(&v).expect("map");
        assert_eq!(f.openai, None);
        assert_eq!(f.anthropic, Some(ServiceTier::Priority));
    }

    #[test]
    fn family_routing() {
        let m = TierModel {
            provider: "zai",
            api: "openai-completions",
            identity_class: "openai",
        };
        assert_eq!(service_tier_family(&m), Some("openai"));
        let m = TierModel {
            provider: "amazon-bedrock",
            api: "anthropic-messages",
            identity_class: "anthropic",
        };
        assert_eq!(service_tier_family(&m), Some("anthropic"));
        assert_eq!(
            get_premium_service_tier_requests(Some(ServiceTier::Priority), &m, false),
            0.0
        );
        // Served tier only skips the realization gate; the provider must
        // still bill premium (TS billing list excludes Bedrock).
        assert_eq!(
            get_premium_service_tier_requests(Some(ServiceTier::Priority), &m, true),
            0.0
        );
        // Vertex Claude drops priority on request but bills it when served.
        let m = TierModel {
            provider: "google-vertex",
            api: "anthropic-messages",
            identity_class: "anthropic",
        };
        assert_eq!(
            get_premium_service_tier_requests(Some(ServiceTier::Priority), &m, false),
            0.0
        );
        assert_eq!(
            get_premium_service_tier_requests(Some(ServiceTier::Priority), &m, true),
            1.0
        );
    }
}
