//! Compaction settings and the trigger arithmetic (pi-agent-core `compaction.ts`).

use serde_json::Value;

/// Tokens reserved below the context window when `reserveTokens` is unset.
pub const DEFAULT_RESERVE_TOKENS: u64 = 16_384;
/// Hard ceiling on a generated compaction summary.
pub const MAX_SUMMARY_TOKENS: u64 = DEFAULT_RESERVE_TOKENS;

/// The settings the local (`soft` / `context-full`) compaction path reads.
#[derive(Debug, Clone, PartialEq)]
pub struct CompactionSettings {
    pub enabled: bool,
    /// `context-full`, `handoff`, `shake`, `snapcompact` or `off`.
    pub strategy: Option<String>,
    /// `<= 0` / non-finite = unset.
    pub threshold_percent: Option<f64>,
    /// `<= 0` / non-finite = unset.
    pub threshold_tokens: Option<f64>,
    /// `None` = defaulted (provenance matters, see [`resolve_budget_reserve_tokens`]).
    pub reserve_tokens: Option<u64>,
    pub keep_recent_tokens: u64,
}

impl Default for CompactionSettings {
    /// `DEFAULT_COMPACTION_SETTINGS`.
    fn default() -> Self {
        Self {
            enabled: true,
            strategy: Some("context-full".to_owned()),
            threshold_percent: Some(-1.0),
            threshold_tokens: Some(-1.0),
            reserve_tokens: None,
            keep_recent_tokens: 20_000,
        }
    }
}

fn usage_u64(usage: &Value, key: &str) -> u64 {
    usage
        .get(key)
        .and_then(Value::as_f64)
        .map_or(0, |f| if f > 0.0 { f as u64 } else { 0 })
}

/// `calculateContextTokens`: provider-reported context occupancy, else total tokens minus
/// provider-side orchestration tokens.
#[must_use]
pub fn calculate_context_tokens(usage: &Value) -> u64 {
    if let Some(ctx) = usage.get("contextTokens").and_then(Value::as_f64) {
        return ctx.max(0.0) as u64;
    }
    let orchestration = usage.get("orchestration").map_or(0, |o| {
        usage_u64(o, "input") + usage_u64(o, "output") + usage_u64(o, "cacheRead")
    });
    let total = usage_u64(usage, "totalTokens");
    let raw = if total == 0 {
        usage_u64(usage, "input")
            + usage_u64(usage, "output")
            + usage_u64(usage, "cacheRead")
            + usage_u64(usage, "cacheWrite")
    } else {
        total
    };
    raw.saturating_sub(orchestration)
}

/// `calculatePromptTokens`.
#[must_use]
pub fn calculate_prompt_tokens(usage: &Value) -> u64 {
    if let Some(ctx) = usage.get("contextTokens").and_then(Value::as_f64) {
        return ctx.max(0.0) as u64;
    }
    let prompt =
        usage_u64(usage, "input") + usage_u64(usage, "cacheRead") + usage_u64(usage, "cacheWrite");
    if prompt > 0 {
        prompt
    } else {
        calculate_context_tokens(usage)
    }
}

/// `getLastAssistantUsage`: usage of the newest assistant message that did not abort or fail.
#[must_use]
pub fn last_assistant_usage<'a>(entries: &[&'a Value]) -> Option<&'a Value> {
    entries.iter().rev().find_map(|entry| {
        if crate::json::entry_type(entry) != "message" {
            return None;
        }
        let message = entry.get("message")?;
        if crate::json::message_role(message) != "assistant" {
            return None;
        }
        if matches!(
            crate::json::str_field(message, "stopReason"),
            Some("aborted" | "error")
        ) {
            return None;
        }
        message
            .get("usage")
            .filter(|u| crate::json::truthy(Some(u)))
    })
}

/// `Math.floor(window * 0.15)`, in floating point exactly as JS evaluates it.
fn fifteen_percent(context_window: u64) -> u64 {
    (context_window as f64 * 0.15).floor() as u64
}

/// `effectiveReserveTokens`: at least 15% of the window, or the configured floor.
#[must_use]
pub fn effective_reserve_tokens(context_window: u64, settings: &CompactionSettings) -> u64 {
    fifteen_percent(context_window).max(settings.reserve_tokens.unwrap_or(DEFAULT_RESERVE_TOKENS))
}

/// `resolveBudgetReserveTokens`: a *defaulted* reserve that cannot fit the window is replaced by
/// the proportional 15%; an explicit one is honoured unless it exceeds the window.
#[must_use]
pub fn resolve_budget_reserve_tokens(context_window: u64, settings: &CompactionSettings) -> u64 {
    let reserve = effective_reserve_tokens(context_window, settings);
    let proportional = fifteen_percent(context_window).max(1);
    let defaulted = settings.reserve_tokens.is_none();
    let impossible = defaulted && reserve >= context_window.saturating_sub(proportional);
    if impossible || reserve >= context_window {
        proportional
    } else {
        reserve
    }
}

/// `resolveThresholdTokens`: fixed token threshold, else percentage, else window minus reserve.
#[must_use]
pub fn resolve_threshold_tokens(context_window: u64, settings: &CompactionSettings) -> f64 {
    let window = context_window as f64;
    if let Some(t) = settings
        .threshold_tokens
        .filter(|t| t.is_finite() && *t > 0.0)
    {
        return (window - 1.0).min(t.max(1.0));
    }
    match settings
        .threshold_percent
        .filter(|p| p.is_finite() && *p > 0.0)
    {
        None => {
            let budget = resolve_budget_reserve_tokens(context_window, settings) as f64;
            (window - 1.0).min(window - budget).max(0.0)
        }
        Some(p) => (window * (p.clamp(1.0, 99.0) / 100.0)).floor(),
    }
}

/// `shouldCompact`: whether `context_tokens` exceeds the threshold.
#[must_use]
pub fn should_compact(
    context_tokens: u64,
    context_window: u64,
    settings: &CompactionSettings,
) -> bool {
    if !settings.enabled || settings.strategy.as_deref() == Some("off") || context_window == 0 {
        return false;
    }
    context_tokens as f64 > resolve_threshold_tokens(context_window, settings)
}

/// `compactionContextTokens`: the provider figure floored by the local estimate of the stored
/// conversation, so on-wire compression cannot hide real growth.
#[must_use]
pub fn compaction_context_tokens(
    provider_context_tokens: u64,
    stored_conversation_estimate: u64,
) -> u64 {
    provider_context_tokens.max(stored_conversation_estimate)
}
