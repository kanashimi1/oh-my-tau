//! Pricing port of `catalog/src/models.ts` cost math plus the pricing
//! helpers from `stats/src/db.ts`. The price table is the extracted
//! `{provider: {modelId: ModelCost}}` JSON from catalog models.json,
//! embedded at compile time and parsed once.
//!
//! TS `number` maps to `f64`; token counts and Unix-ms timestamps to `i64`.
//! `calculate_usage_cost` writes the amounts into `usage.cost`, mirroring the
//! TS side effect on `usage.cost.*`.

use serde::Deserialize;
use std::collections::HashMap;
use std::sync::LazyLock;

/// Bundled price rows: provider -> modelId -> ModelCost.
static PRICING: LazyLock<HashMap<String, HashMap<String, ModelCost>>> = LazyLock::new(|| {
    serde_json::from_str(include_str!("../../data/pricing.json"))
        .expect("pricing.json holds valid ModelCost rows")
});

/// Per-million-token rates for one model pricing tier.
#[derive(Debug, Clone, Copy, PartialEq, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TokenCost {
    pub input: f64,
    pub output: f64,
    pub cache_read: f64,
    pub cache_write: f64,
}

/// Rates applied to the full request when its prompt exceeds
/// `input_threshold` (or reaches it when `input_threshold_inclusive`).
#[derive(Debug, Clone, Copy, PartialEq, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LongContextTokenCost {
    #[serde(flatten)]
    pub base: TokenCost,
    pub input_threshold: i64,
    /// Whether the long-context tier starts exactly at `input_threshold`.
    #[serde(default)]
    pub input_threshold_inclusive: Option<bool>,
}

/// Recurring UTC peak interval; weekdays use Sunday = 0, end exclusive.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PeakPricingWindow {
    pub weekdays: Vec<u8>,
    pub start_minute: i64,
    pub end_minute: i64,
}

/// Complete replacement rate card effective from a Unix-ms timestamp.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EffectiveTokenCost {
    #[serde(flatten)]
    pub rates: TokenCost,
    pub effective_from: i64,
    #[serde(default)]
    pub long_context: Option<LongContextTokenCost>,
}

/// Scheduled discounts applied after selecting the rate card and context tier.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TimeBasedCost {
    pub off_peak_multiplier: f64,
    pub peak_windows: Vec<PeakPricingWindow>,
    #[serde(default)]
    pub effective_rates: Option<Vec<EffectiveTokenCost>>,
}

/// Base token rates plus optional long-context and time-based pricing.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelCost {
    #[serde(flatten)]
    pub base: TokenCost,
    #[serde(default)]
    pub long_context: Option<LongContextTokenCost>,
    #[serde(default)]
    pub time_based: Option<TimeBasedCost>,
}

impl ModelCost {
    pub fn input(&self) -> f64 {
        self.base.input
    }
    pub fn output(&self) -> f64 {
        self.base.output
    }
    pub fn cache_read(&self) -> f64 {
        self.base.cache_read
    }
    pub fn cache_write(&self) -> f64 {
        self.base.cache_write
    }
}

/// Usage buckets + cost breakdown the pricing functions read/write.
/// (The full TS `Usage` carries more optional metadata; only these matter here.)
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Usage {
    pub input: i64,
    pub output: i64,
    pub cache_read: i64,
    pub cache_write: i64,
    /// Provider-side orchestration tokens (billed, not conversation buckets).
    pub orchestration: Option<OrchestrationUsage>,
    /// Cache-write TTL breakdown (Anthropic only); components sum to `cache_write`.
    pub cttl: Option<CttlUsage>,
    pub cost: UsageCost,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct OrchestrationUsage {
    pub input: Option<i64>,
    pub cache_read: Option<i64>,
    pub output: Option<i64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct CttlUsage {
    pub ephemeral5m: Option<i64>,
    pub ephemeral1h: Option<i64>,
}

/// Monetary breakdown for a priced usage record.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct UsageCost {
    pub input: f64,
    pub output: f64,
    pub cache_read: f64,
    pub cache_write: f64,
    pub total: f64,
}

/// Rates after selecting the effective card and the context tier.
enum EffectiveRates {
    Base(TokenCost),
    Long(LongContextTokenCost),
}

impl EffectiveRates {
    fn input(&self) -> f64 {
        match self {
            EffectiveRates::Base(t) => t.input,
            EffectiveRates::Long(l) => l.base.input,
        }
    }
    fn output(&self) -> f64 {
        match self {
            EffectiveRates::Base(t) => t.output,
            EffectiveRates::Long(l) => l.base.output,
        }
    }
    fn cache_read(&self) -> f64 {
        match self {
            EffectiveRates::Base(t) => t.cache_read,
            EffectiveRates::Long(l) => l.base.cache_read,
        }
    }
    fn cache_write(&self) -> f64 {
        match self {
            EffectiveRates::Base(t) => t.cache_write,
            EffectiveRates::Long(l) => l.base.cache_write,
        }
    }
    fn as_token_cost(&self) -> TokenCost {
        TokenCost {
            input: self.input(),
            output: self.output(),
            cache_read: self.cache_read(),
            cache_write: self.cache_write(),
        }
    }
}

/// Intermediate view of a (possibly time-adjusted) rate card.
struct ModelCostEffective {
    base: TokenCost,
    long_context: Option<LongContextTokenCost>,
}

fn resolve_token_cost(
    cost: &ModelCost,
    prompt_input_tokens: i64,
    timestamp: Option<i64>,
) -> EffectiveRates {
    let mut rates = ModelCostEffective {
        base: cost.base,
        long_context: cost.long_context,
    };
    let mut effective_from = i64::MIN;
    if let (Some(ts), Some(sched)) = (timestamp, cost.time_based.as_ref()) {
        if let Some(candidates) = sched.effective_rates.as_ref() {
            for candidate in candidates {
                if candidate.effective_from <= ts && candidate.effective_from > effective_from {
                    rates = ModelCostEffective {
                        base: candidate.rates,
                        long_context: candidate.long_context,
                    };
                    effective_from = candidate.effective_from;
                }
            }
        }
    }
    let Some(long_context) = rates.long_context else {
        return EffectiveRates::Base(rates.base);
    };
    let threshold = long_context.input_threshold;
    let reaches = prompt_input_tokens > threshold
        || (long_context.input_threshold_inclusive.unwrap_or(false)
            && prompt_input_tokens == threshold);
    if reaches {
        EffectiveRates::Long(long_context)
    } else {
        EffectiveRates::Base(rates.base)
    }
}

fn is_peak_pricing_period(schedule: &TimeBasedCost, timestamp: i64) -> bool {
    // Unix epoch was Thursday. Arithmetic keeps this UTC-only without a Date.
    let day = timestamp.div_euclid(86_400_000);
    let weekday = ((day + 4).rem_euclid(7)) as u8;
    let minute = (timestamp - day * 86_400_000).div_euclid(60_000);
    schedule
        .peak_windows
        .iter()
        .any(|w| minute >= w.start_minute && minute < w.end_minute && w.weekdays.contains(&weekday))
}

fn time_based_multiplier(schedule: Option<&TimeBasedCost>, timestamp: Option<i64>) -> f64 {
    match (schedule, timestamp) {
        (Some(sched), Some(ts)) => {
            if is_peak_pricing_period(sched, ts) {
                1.0
            } else {
                sched.off_peak_multiplier
            }
        }
        _ => 1.0,
    }
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// Price a fully uncached prompt at its request timestamp (Unix ms);
/// scheduled prices default to now.
pub fn calculate_uncached_input_cost(
    cost: &ModelCost,
    prompt_input_tokens: i64,
    timestamp: Option<i64>,
) -> f64 {
    let pricing_timestamp = if cost.time_based.is_some() {
        timestamp.or_else(|| Some(now_ms()))
    } else {
        None
    };
    let rates = resolve_token_cost(cost, prompt_input_tokens, pricing_timestamp);
    (rates.input() / 1_000_000.0)
        * prompt_input_tokens as f64
        * time_based_multiplier(cost.time_based.as_ref(), pricing_timestamp)
}

/// Price usage at its request timestamp (Unix ms); only scheduled prices
/// default to now. Writes the amounts into `usage.cost` and returns a copy.
pub fn calculate_usage_cost(
    cost: &ModelCost,
    usage: &mut Usage,
    timestamp: Option<i64>,
) -> UsageCost {
    let orch = usage.orchestration.unwrap_or_default();
    let orch_input = orch.input.unwrap_or(0);
    let orch_cache_read = orch.cache_read.unwrap_or(0);
    let orch_output = orch.output.unwrap_or(0);
    let prompt_input_tokens =
        usage.input + usage.cache_read + usage.cache_write + orch_input + orch_cache_read;
    let pricing_timestamp = if cost.time_based.is_some() {
        timestamp.or_else(|| Some(now_ms()))
    } else {
        None
    };
    let rates = resolve_token_cost(cost, prompt_input_tokens, pricing_timestamp);
    let multiplier = time_based_multiplier(cost.time_based.as_ref(), pricing_timestamp);
    let token_rates = rates.as_token_cost();
    usage.cost.input =
        (token_rates.input / 1_000_000.0) * (usage.input + orch_input) as f64 * multiplier;
    usage.cost.output =
        (token_rates.output / 1_000_000.0) * (usage.output + orch_output) as f64 * multiplier;
    usage.cost.cache_read = (token_rates.cache_read / 1_000_000.0)
        * (usage.cache_read + orch_cache_read) as f64
        * multiplier;
    usage.cost.cache_write = cache_write_cost(&token_rates, usage) * multiplier;
    usage.cost.total =
        usage.cost.input + usage.cost.output + usage.cost.cache_read + usage.cost.cache_write;
    usage.cost
}

/// Price cache-write tokens, honoring the TTL breakdown when reported.
///
/// `rates.cache_write` is the 5-minute write rate (Anthropic bills 5m writes
/// at 1.25x base input). With `usage.cttl` the write can mix 5m and 1h
/// breakpoints, and 1h writes bill at 2x base input, so each component is
/// priced at its own rate. An unattributed remainder is priced at the flat
/// rate so a stale breakdown never makes write tokens free.
fn cache_write_cost(rates: &TokenCost, usage: &Usage) -> f64 {
    let rate5m = rates.cache_write / 1_000_000.0;
    let Some(cttl) = usage.cttl else {
        return rate5m * usage.cache_write as f64;
    };
    let five_minute = cttl.ephemeral5m.unwrap_or(0);
    let one_hour = cttl.ephemeral1h.unwrap_or(0);
    let residual = (usage.cache_write - five_minute - one_hour).max(0);
    rate5m * (five_minute + residual) as f64 + ((rates.input * 2.0) / 1_000_000.0) * one_hour as f64
}

/// Look up one bundled model's cost, or `None` when provider/model is absent.
pub fn bundled_model_cost(provider: &str, model_id: &str) -> Option<ModelCost> {
    PRICING.get(provider)?.get(model_id).cloned()
}

/// Whether any per-token rate on the card is nonzero (a real price exists).
pub fn has_billable_cost(cost: &ModelCost) -> bool {
    cost.base.input != 0.0
        || cost.base.output != 0.0
        || cost.base.cache_read != 0.0
        || cost.base.cache_write != 0.0
}

fn get_bundled_model_cost(provider: &str, model_id: &str) -> Option<ModelCost> {
    bundled_model_cost(provider, model_id)
}

/// Catalog cost for (provider, model), with the openai-codex/xai-oauth fallbacks.
pub fn get_catalog_cost(provider: &str, model_id: &str) -> Option<ModelCost> {
    let primary = get_bundled_model_cost(provider, model_id);
    if let Some(cost) = primary.as_ref() {
        if has_billable_cost(cost) {
            return primary;
        }
    }
    let fallback_provider = if provider == "openai-codex" {
        Some("openai")
    } else if provider == "xai-oauth" {
        Some("xai")
    } else {
        None
    };
    if let Some(fp) = fallback_provider {
        let fallback = get_bundled_model_cost(fp, model_id);
        if let Some(cost) = fallback.as_ref() {
            if has_billable_cost(cost) {
                return fallback;
            }
        }
    }
    None
}

/// Whether the catalog prices this model on a time-based (scheduled) card.
pub fn is_scheduled_catalog_model(provider: &str, model_id: &str) -> bool {
    get_catalog_cost(provider, model_id)
        .map(|c| c.time_based.is_some())
        .unwrap_or(false)
}

/// SQL fragment marking token-bearing rows that recorded no price
/// (subscription billing or a scheduled card with no recoverable timestamp).
pub fn unpriced_request_sql(prefix: &str) -> String {
    format!(
        "CASE WHEN {p}total_tokens > 0 AND {p}cost_total = 0
\t\tAND ({p}provider = 'xai-oauth' OR {p}cost_unpriced = 1) THEN 1 ELSE 0 END",
        p = prefix
    )
}

#[cfg(test)]
mod tests;
