//! Turn recovery: auto-retry with backoff plus `retry.fallbackChains` model
//! switching (port of omp `session/turn-recovery.ts` `#handleRetryableError`).
//!
//! This module holds the decision state and the wire frames; `runtime.rs`
//! performs the effects (model swap, sleeping, continuing the run) in the
//! TS-faithful order:
//!
//! 1. `model_changed` (swap) -> `model_change` entry (`role: "fallback"`,
//!    `resolvedModelIsFallback: true`) -> `thinking_level_changed` ->
//!    `retry_fallback_applied` -> `auto_retry_start` -> backoff sleep ->
//!    `agent_end{isTerminal:false}` -> continued run;
//! 2. on the first settled turn: `retry_fallback_succeeded` (once per fallback)
//!    then `auto_retry_end{success:true}`;
//! 3. every dead end closes the saga with `auto_retry_end{success:false}`.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use anyhow::Result;
use omp_providers::{
    is_retryable_error, parse_rate_limit_reason, rate_limit_backoff_ms, AgentDir, CandidateOptions,
    FallbackConfig, FallbackState, RateLimitReason, ResolvedModel, RetryConfig, RevertPolicy,
};
use pa_agent::types::AssistantMessage;
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

/// `EPHEMERAL_MODEL_CHANGE_ROLE`: role of `model_change` entries written by a fallback switch.
pub const EPHEMERAL_ROLE: &str = "fallback";

/// Resolves a `provider/id[:level]` selector to a callable model (credentials included).
/// Async: `!command` api keys shell out, so resolution must stay off the executor threads.
pub type Resolver = Arc<
    dyn Fn(String) -> Pin<Box<dyn Future<Output = Result<ResolvedModel>> + Send>> + Send + Sync,
>;

/// The resolver over omp's `config.yml` / `models.yml` / `agent.db` in `dir`.
#[must_use]
pub fn resolver_in(dir: AgentDir) -> Resolver {
    Arc::new(
        move |selector: String| -> Pin<Box<dyn Future<Output = Result<ResolvedModel>> + Send>> {
            Box::pin(omp_providers::resolve_in_async(dir.clone(), selector))
        },
    )
}

/// The default resolver: omp's `config.yml` / `models.yml` / `agent.db` in the default agent dir.
#[must_use]
pub fn default_resolver() -> Resolver {
    resolver_in(AgentDir::default())
}

/// `retry.*` settings plus the fallback chain of the live model.
#[derive(Clone, Default)]
pub struct RecoveryConfig {
    pub retry: RetryConfig,
    pub fallback: FallbackConfig,
    /// Chain key the walk is pinned to (`provider/id` of the primary).
    pub chain_key: String,
    /// Role name reported in `retry_fallback_*` events (e.g. `default`); the chain key when `None`.
    pub role_label: Option<String>,
}

impl RecoveryConfig {
    /// Chain of `primary`: its `ResolvedModel::fallbacks` after it, as a model-selector chain.
    #[must_use]
    pub fn from_resolved(primary: &ResolvedModel) -> Self {
        let chain_key = format!("{}/{}", primary.provider, primary.id);
        let mut fallback = FallbackConfig::default();
        if !primary.fallbacks.is_empty() {
            fallback
                .chains
                .push((chain_key.clone(), primary.fallbacks.clone()));
        }
        fallback.known_providers = primary
            .fallbacks
            .iter()
            .filter_map(|s| s.split('/').next())
            .chain([primary.provider.as_str()])
            .map(String::from)
            .collect();
        Self {
            retry: primary.retry.clone(),
            fallback,
            chain_key,
            role_label: primary.role.clone(),
        }
    }

    fn role(&self) -> &str {
        self.role_label.as_deref().unwrap_or(&self.chain_key)
    }
}

/// What the recovery logic needs to know about a failed assistant turn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FailedTurn {
    pub message: String,
    pub status: Option<u16>,
    pub provider: String,
    pub model: String,
    /// Provider-stated wait from the response headers (`Retry-After`,
    /// `retry-after-ms`), when the stream-failure diagnostic carries one.
    pub retry_after_ms: Option<u64>,
}

impl FailedTurn {
    /// `None` unless `m` stopped with `error`.
    #[must_use]
    pub fn from_message(m: &AssistantMessage) -> Option<Self> {
        if m.stop_reason != pa_agent::types::StopReason::Error {
            return None;
        }
        let status = m.error_status.or_else(|| {
            m.diagnostics
                .iter()
                .flatten()
                .filter_map(|d| d.details.as_ref())
                .find_map(|d| d.get("status").and_then(Value::as_u64))
                .and_then(|s| u16::try_from(s).ok())
        });
        let retry_after_ms = m
            .diagnostics
            .iter()
            .flatten()
            .filter_map(|d| d.details.as_ref())
            .find_map(|d| d.get("retryAfterMs").and_then(Value::as_u64));
        Some(Self {
            message: m
                .error_message
                .clone()
                .unwrap_or_else(|| "Unknown error".to_string()),
            status,
            retry_after_ms,
            provider: m.provider.clone(),
            model: m.model.clone(),
        })
    }
}

/// `AIError.Flag.Transient` approximation: omp-providers' classifier plus transport failures
/// the SDKs word differently ("Connection error.", "fetch failed", ECONNREFUSED, ...).
#[must_use]
pub fn is_transient(status: Option<u16>, message: &str) -> bool {
    if is_retryable_error(status, message) {
        return true;
    }
    let m = message.to_ascii_lowercase();
    status.is_none()
        && [
            "connection error",
            "fetch failed",
            "econnrefused",
            "econnreset",
            "enotfound",
            "etimedout",
            "dns",
            "tls handshake",
        ]
        .iter()
        .any(|p| m.contains(p))
}

/// Longest provider-stated wait found in an error text (`retry-after-ms`, `retry-after`,
/// "try again in 2.5s", "resets in 3 minutes"), in ms.
#[must_use]
pub fn parse_retry_after_ms(message: &str) -> Option<u64> {
    let lower = message.to_ascii_lowercase();
    let mut best: Option<f64> = None;
    let mut take = |ms: f64| {
        if ms.is_finite() && ms > 0.0 {
            best = Some(best.map_or(ms, |b| b.max(ms)));
        }
    };
    let number_after = |at: usize| -> Option<(f64, &str)> {
        let rest = lower[at..].trim_start_matches([' ', ':', '=', '"']);
        let end = rest
            .find(|c: char| !(c.is_ascii_digit() || c == '.'))
            .unwrap_or(rest.len());
        let n: f64 = rest[..end].parse().ok()?;
        Some((n, rest[end..].trim_start()))
    };
    let unit_ms = |unit: &str| -> f64 {
        let word: String = unit.chars().take_while(char::is_ascii_alphabetic).collect();
        match word.as_str() {
            "ms" | "msec" | "millisecond" | "milliseconds" => 1.0,
            "m" | "min" | "mins" | "minute" | "minutes" => 60_000.0,
            "h" | "hr" | "hrs" | "hour" | "hours" => 3_600_000.0,
            _ => 1000.0,
        }
    };
    for (needle, ms_unit) in [("retry-after-ms", true), ("retry_after_ms", true)] {
        if let Some(i) = lower.find(needle) {
            if let Some((n, _)) = number_after(i + needle.len()) {
                take(if ms_unit { n } else { n * 1000.0 });
            }
        }
    }
    for needle in ["retry-after", "retry after", "retry_after"] {
        let mut from = 0;
        while let Some(i) = lower[from..].find(needle) {
            let at = from + i + needle.len();
            if !lower[at..].starts_with("-ms") && !lower[at..].starts_with("_ms") {
                if let Some((n, rest)) = number_after(at) {
                    take(n * unit_ms(rest));
                }
            }
            from = at;
        }
    }
    for needle in [
        "try again in",
        "retry in",
        "resets in",
        "reset in",
        "available in",
        "wait ",
    ] {
        let mut from = 0;
        while let Some(i) = lower[from..].find(needle) {
            let at = from + i + needle.len();
            if let Some((n, rest)) = number_after(at) {
                take(n * unit_ms(rest));
            }
            from = at;
        }
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    best.map(|ms| ms.ceil() as u64)
}

// ---------------------------------------------------------------- frames

#[must_use]
pub fn auto_retry_start_frame(
    attempt: u32,
    max_attempts: u32,
    delay_ms: u64,
    error_message: &str,
) -> Value {
    json!({"type": "auto_retry_start", "attempt": attempt, "maxAttempts": max_attempts, "delayMs": delay_ms, "errorMessage": error_message})
}

#[must_use]
pub fn auto_retry_end_frame(success: bool, attempt: u32, final_error: Option<&str>) -> Value {
    let mut v = json!({"type": "auto_retry_end", "success": success, "attempt": attempt});
    if let Some(e) = final_error {
        v["finalError"] = json!(e);
    }
    v
}

#[must_use]
pub fn fallback_applied_frame(from: &str, to: &str, role: &str, reason: Option<&str>) -> Value {
    let mut v = json!({"type": "retry_fallback_applied", "from": from, "to": to, "role": role});
    if let Some(r) = reason {
        v["reason"] = json!(r);
    }
    v
}

/// The account-rotation counterpart of [`fallback_applied_frame`]: same model, other account.
#[must_use]
pub fn account_rotated_frame(from: &str, to: &str, reason: Option<&str>) -> Value {
    let mut v = json!({"type": "retry_account_rotated", "from": from, "to": to});
    if let Some(r) = reason {
        v["reason"] = json!(r);
    }
    v
}

#[must_use]
pub fn fallback_succeeded_frame(model: &str, role: &str) -> Value {
    json!({"type": "retry_fallback_succeeded", "model": model, "role": role})
}

// ----------------------------------------------------------------- state

/// The user-facing `provider/id` selector for a model (no thinking level suffix).
pub fn model_selector(provider: &str, model_id: &str) -> String {
    format!("{provider}/{model_id}")
}

/// One `handle_retryable_error` pass, before the fallback consult.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Attempt {
    pub number: u32,
    pub max: u32,
    pub budget_exhausted: bool,
    pub error_message: String,
    pub retry_after_ms: Option<u64>,
    /// Backoff before any fallback switch / retry-after adjustment.
    pub delay_ms: u64,
}

/// How an attempt ends once the fallback consult is known.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// Announce `auto_retry_start` with `delay_ms`, sleep, continue.
    Retry { delay_ms: u64 },
    /// Give up; `frame` is the closing `auto_retry_end`, `error_message` replaces the turn's error text if set.
    GiveUp {
        frame: Value,
        error_message: Option<String>,
    },
}

/// Account-rotation bookkeeping (TS: multi-account rate-limit rotation). One slot per
/// `(provider, model)`: the account pool is per provider, but cooldowns and the failing
/// row belong to the specific model that hit the limit (parallel panes share the pool
/// state of a process).
#[derive(Debug, Default)]
pub struct AccountRotation {
    /// Pool alternates of `(provider, model)` that recently 429'd — skipped while the
    /// provider's cooldown runs.
    cooling: HashMap<String, Vec<i64>>,
    /// Every alternate that took a turn for `(provider, model)` this session: never again,
    /// a rotation cycle must not loop back to an account that already failed.
    burned: HashMap<String, Vec<i64>>,
    /// The account each `(provider, model)` currently streams through (the promoted
    /// alternate's row id, `None` until a rotation swaps the key). `next` starts after it,
    /// so the serving account is not re-picked immediately and cycles back only after
    /// every other account had its turn.
    serving: HashMap<String, i64>,
}

impl AccountRotation {
    fn slot(provider: &str, model: &str) -> String {
        format!("{provider}/{model}")
    }

    /// The alternates `next` hands out, in store order. Pool factories exclude the store's
    /// preferred row from `alternates`, so this is just the pool order; the serving account
    /// is additionally filtered out in `next` (it is still present here after a promotion).
    #[must_use]
    fn window(pool: &omp_providers::auth::CredentialPool) -> Vec<omp_providers::auth::Credential> {
        pool.alternates.clone()
    }

    /// The next untried account of `pool` for `model`, honouring the rotation cooldown.
    #[must_use]
    pub fn next(
        &mut self,
        provider: &str,
        model: &str,
        pool: &omp_providers::auth::CredentialPool,
        cooldown_active: bool,
    ) -> Option<omp_providers::auth::Credential> {
        let slot = Self::slot(provider, model);
        let burned = self.burned.entry(slot.clone()).or_default();
        let cooling = self.cooling.entry(slot).or_default();
        let mut window = Self::window(pool);
        if let Some(serving) = self.serving.get(&Self::slot(provider, model)) {
            // The serving account never re-enters the rotation window: it cycles back only
            // after every other account burned out and a fresh pool re-arms the cycle.
            window.retain(|c| c.id != *serving);
        }
        window
            .into_iter()
            .find(|c| !burned.contains(&c.id) && !(cooldown_active && cooling.contains(&c.id)))
    }

    /// `alternate` just took over `model` (switch applied): it becomes the serving account
    /// and leaves the cooling/burned sets.
    pub fn promote(&mut self, provider: &str, model: &str, alternate: i64) {
        let slot = Self::slot(provider, model);
        if let Some(v) = self.cooling.get_mut(&slot) {
            v.retain(|id| *id != alternate);
        }
        if let Some(v) = self.burned.get_mut(&slot) {
            v.retain(|id| *id != alternate);
        }
        self.serving.insert(slot, alternate);
    }

    /// `alternate` was tried for `model` and rate-limited: remember it in both sets.
    pub fn burn(&mut self, provider: &str, model: &str, alternate: i64) {
        let slot = Self::slot(provider, model);
        for set in [&mut self.burned, &mut self.cooling] {
            let v = set.entry(slot.clone()).or_default();
            if !v.contains(&alternate) {
                v.push(alternate);
            }
        }
    }

    /// The provider's cooldown expired: rotation may retry the cooled alternates once —
    /// clear the cooling set (the burned set keeps them out of the next cycle).
    #[cfg(test)]
    pub fn clear_cooldown(&mut self, provider: &str, model: &str) {
        if let Some(v) = self.cooling.get_mut(&Self::slot(provider, model)) {
            v.clear();
        }
    }

    #[cfg(test)]
    fn burned_of(&self, provider: &str, model: &str) -> Vec<i64> {
        self.burned
            .get(&Self::slot(provider, model))
            .cloned()
            .unwrap_or_default()
    }
}

pub struct TurnRecovery {
    cfg: RecoveryConfig,
    state: FallbackState,
    /// Multi-account rotation state (per `(provider, model)` cooldown/burned sets).
    accounts: AccountRotation,
    enabled: bool,
    attempt: u32,
    generation: u64,
    cancel: Option<(u64, CancellationToken)>,
    /// Between `begin_attempt` and `arm_cancel` (fallback consult in flight): an `abort_retry`
    /// arriving now is remembered and cancels the backoff as soon as it is armed.
    handling: bool,
    abort_pending: bool,
}

impl TurnRecovery {
    #[must_use]
    pub fn new(cfg: RecoveryConfig) -> Self {
        let state = FallbackState::new(RevertPolicy::from_setting(Some(&cfg.retry.revert_policy)));
        let enabled = cfg.retry.enabled;
        Self {
            cfg,
            state,
            accounts: AccountRotation::default(),
            enabled,
            attempt: 0,
            generation: 0,
            cancel: None,
            handling: false,
            abort_pending: false,
        }
    }

    /// A user-selected model replaces the chain and ends fallback ownership; cooldowns of other
    /// selectors survive (the caller clears the selected model's own).
    pub fn reconfigure(&mut self, cfg: RecoveryConfig) {
        self.cfg = cfg;
        self.state.clear_active();
    }

    /// Clear the cooldown of a selector the user just picked.
    pub fn clear_cooldown(&mut self, selector: &str) {
        self.state.clear_suppressed(selector);
    }

    /// Whether `selector` is on a cooldown at `now_ms`.
    #[cfg(test)]
    #[must_use]
    pub fn is_cooling(&self, selector: &str, now_ms: u64) -> bool {
        self.state.is_suppressed(selector, now_ms)
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    #[must_use]
    pub fn attempt(&self) -> u32 {
        self.attempt
    }

    /// True while a saga is open (an `auto_retry_start` was announced and not yet closed).
    #[must_use]
    pub fn saga_open(&self) -> bool {
        self.attempt > 0
    }

    /// Mut access to the multi-account rotation bookkeeping (runtime owns the swap logic).
    pub fn accounts_mut(&mut self) -> &mut AccountRotation {
        &mut self.accounts
    }

    #[must_use]
    pub fn role(&self) -> &str {
        self.cfg.role()
    }

    pub fn state_mut(&mut self) -> &mut FallbackState {
        &mut self.state
    }

    /// `isRetryableError`: enabled and the turn failed with a transient error.
    #[must_use]
    pub fn is_retryable(&self, failed: &FailedTurn) -> bool {
        self.enabled && is_transient(failed.status, &failed.message)
    }

    /// This failure is a provider *rate limit* (not a generic transient): candidates for an
    /// account rotation (same provider, same model, next account's credential).
    #[must_use]
    pub fn is_rate_limit(&self, failed: &FailedTurn) -> bool {
        if !self.enabled || failed.status.is_some_and(|s| s != 429) {
            return false;
        }
        let m = failed.message.to_ascii_lowercase();
        m.contains("rate limit")
            || m.contains("rate_limit")
            || m.contains("too many requests")
            || m.contains("quota")
    }

    /// Open (or continue) a saga for `failed`: bump the attempt counter, compute the backoff.
    pub fn begin_attempt(&mut self, failed: &FailedTurn, rand01: f64) -> Attempt {
        self.attempt += 1;
        self.handling = true;
        self.abort_pending = false;
        let max = self.cfg.retry.max_retries;
        // An explicit provider wait is authoritative in both directions: a
        // header `Retry-After: 0` means "retry now", so the rate-limit
        // heuristic window below does not apply.
        let retry_after_ms = failed
            .retry_after_ms
            .or_else(|| parse_retry_after_ms(&failed.message));
        let mut delay_ms = self.cfg.retry.backoff_ms(self.attempt, rand01);
        // HTTP 429 with a fallback chain configured: the primary model's quota
        // is gone — stop backing off on it and switch to the fallback
        // immediately. Without a fallback the Retry-After handling stands.
        let has_fallback = self
            .cfg
            .fallback
            .chains
            .iter()
            .any(|(k, v)| !v.is_empty() && (k == &self.cfg.chain_key || k == "default"));
        if failed.status.is_some_and(|s| s == 429) && has_fallback {
            delay_ms = 0;
        } else if retry_after_ms.is_none() {
            let reason = parse_rate_limit_reason(&failed.message);
            if matches!(
                reason,
                RateLimitReason::ConcurrentLimit | RateLimitReason::RateLimitExceeded
            ) {
                delay_ms = delay_ms.max(rate_limit_backoff_ms(reason, rand01));
            }
        }
        Attempt {
            number: self.attempt,
            max,
            budget_exhausted: self.attempt > max,
            error_message: failed.message.clone(),
            retry_after_ms,
            delay_ms,
        }
    }

    /// Cool `current` down and list the chain candidates still worth trying, in walk order.
    pub fn consult(
        &mut self,
        current: &str,
        attempt: &Attempt,
        now_ms: u64,
        rand01: f64,
    ) -> Vec<String> {
        if self.cfg.fallback.chains.is_empty() {
            return Vec::new();
        }
        self.state.note_cooldown(
            current,
            attempt.retry_after_ms,
            &attempt.error_message,
            now_ms,
            rand01,
        );
        self.state.candidates(
            &self.cfg.fallback,
            current,
            None,
            CandidateOptions::default(),
            now_ms,
        )
    }

    /// Record a committed switch onto a fallback (`from` is the selector left behind).
    pub fn commit_switch(
        &mut self,
        from: &str,
        from_thinking: Option<String>,
        applied_thinking: Option<String>,
    ) {
        let key = self.cfg.chain_key.clone();
        self.state
            .apply(&key, from, from_thinking, applied_thinking, false);
    }

    /// Decide how the attempt ends (`switched` = a fallback model took over).
    pub fn finish_attempt(&mut self, a: &Attempt, switched: bool) -> Verdict {
        let verdict = self.finish_attempt_inner(a, switched);
        if matches!(verdict, Verdict::GiveUp { .. }) {
            self.handling = false;
            self.abort_pending = false;
        }
        verdict
    }

    fn finish_attempt_inner(&mut self, a: &Attempt, switched: bool) -> Verdict {
        let mut delay = a.delay_ms;
        if switched {
            delay = 0;
        } else if let Some(ra) = a.retry_after_ms.filter(|r| *r > delay) {
            delay = ra;
        }
        if a.budget_exhausted {
            if !switched {
                let done = a.number - 1;
                let message = format!(
                    "Retry budget exhausted after {done} {}: {}",
                    if done == 1 { "retry" } else { "retries" },
                    a.error_message
                );
                self.attempt = 0;
                return Verdict::GiveUp {
                    frame: auto_retry_end_frame(false, done, Some(&a.error_message)),
                    error_message: Some(message),
                };
            }
            self.attempt = 1;
        }
        let cap = self.cfg.retry.max_delay_ms;
        if cap > 0 && delay > cap && !switched {
            let attempt = self.attempt;
            self.attempt = 0;
            return Verdict::GiveUp {
                frame: auto_retry_end_frame(
                    false,
                    attempt,
                    Some(&format!(
                        "Provider requested {delay}ms wait, exceeds retry.maxDelayMs ({cap}ms). Original error: {}",
                        a.error_message
                    )),
                ),
                error_message: None,
            };
        }
        Verdict::Retry { delay_ms: delay }
    }

    /// Frames for a settled successful turn served by `selector`: the one-shot
    /// `retry_fallback_succeeded`, then `auto_retry_end{success:true}` closing an open saga.
    pub fn on_settled_ok(&mut self, selector: &str) -> Vec<Value> {
        let mut out = Vec::new();
        if self.state.mark_served() {
            out.push(fallback_succeeded_frame(selector, self.cfg.role()));
        }
        if self.attempt > 0 {
            out.push(auto_retry_end_frame(true, self.attempt, None));
            self.attempt = 0;
        }
        out
    }

    /// A terminal error settled with no retry owning it: close an open saga.
    pub fn on_error_settled(&mut self, error_message: &str) -> Option<Value> {
        if self.attempt == 0 {
            return None;
        }
        let attempt = self.attempt;
        self.attempt = 0;
        Some(auto_retry_end_frame(false, attempt, Some(error_message)))
    }

    /// Arm the backoff-sleep cancel token (a newer one supersedes the older).
    pub fn arm_cancel(&mut self) -> (u64, CancellationToken) {
        self.generation += 1;
        let token = CancellationToken::new();
        if let Some((_, old)) = self.cancel.replace((self.generation, token.clone())) {
            old.cancel();
        }
        self.handling = false;
        if std::mem::take(&mut self.abort_pending) {
            token.cancel();
        }
        (self.generation, token)
    }

    /// Backoff finished or was cancelled; `true` when `generation` is still the armed one.
    pub fn disarm(&mut self, generation: u64) -> bool {
        match &self.cancel {
            Some((g, _)) if *g == generation => {
                self.cancel = None;
                true
            }
            _ => false,
        }
    }

    /// A newer backoff replaced `generation`'s (its cancellation is not ours to report).
    #[must_use]
    pub fn superseded(&self, generation: u64) -> bool {
        matches!(&self.cancel, Some((g, _)) if *g != generation)
    }

    /// `abortRetry`: cancel the backoff sleep. `true` when a sleep was pending.
    pub fn abort_retry(&mut self) -> bool {
        if let Some((_, token)) = self.cancel.take() {
            token.cancel();
            true
        } else {
            self.abort_pending = self.handling;
            false
        }
    }

    /// `#endCancelledRetry`: close the saga of a cancelled backoff.
    pub fn end_cancelled(&mut self) -> Value {
        let attempt = self.attempt;
        self.attempt = 0;
        auto_retry_end_frame(false, attempt, Some("Retry cancelled"))
    }

    /// The scheduled continuation failed locally (no provider request was made).
    pub fn end_continue_failed(&mut self, local_error: &str, original: &str) -> Option<Value> {
        if self.attempt == 0 {
            return None;
        }
        let attempt = self.attempt;
        self.attempt = 0;
        Some(auto_retry_end_frame(
            false,
            attempt,
            Some(&format!(
                "Retry continuation failed locally: {local_error}. Original error: {original}"
            )),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn failed(msg: &str) -> FailedTurn {
        FailedTurn {
            message: msg.into(),
            status: None,
            provider: "p".into(),
            model: "m".into(),
            retry_after_ms: None,
        }
    }

    fn cfg(chain: &[&str]) -> RecoveryConfig {
        let mut fallback = FallbackConfig::default();
        fallback.chains.push((
            "p/a".into(),
            chain.iter().map(|s| (*s).to_string()).collect(),
        ));
        fallback.known_providers = ["p", "q", "r"].iter().map(|s| (*s).to_string()).collect();
        RecoveryConfig {
            retry: RetryConfig {
                enabled: true,
                max_retries: 2,
                base_delay_ms: 100,
                max_delay_ms: 60_000,
                revert_policy: "cooldown-expiry".into(),
            },
            fallback,
            chain_key: "p/a".into(),
            role_label: Some("default".into()),
        }
    }

    #[test]
    fn retry_after_takes_the_longest_hint() {
        assert_eq!(
            parse_retry_after_ms("429 rate limited. Retry-After: 3"),
            Some(3000)
        );
        assert_eq!(parse_retry_after_ms("please try again in 2.5s"), Some(2500));
        assert_eq!(
            parse_retry_after_ms("retry-after-ms: 1500; resets in 2 minutes"),
            Some(120_000)
        );
        assert_eq!(parse_retry_after_ms("boom"), None);
    }

    /// HTTP 429 with a fallback chain configured: immediate switch — zero
    /// same-model backoff before the fallback consult.
    #[test]
    fn http_429_with_fallback_switches_immediately() {
        let mut r = TurnRecovery::new(cfg(&["q/b"]));
        let mut f = failed("429 rate limit exceeded");
        f.status = Some(429);
        let a = r.begin_attempt(&f, 0.0);
        assert_eq!(a.delay_ms, 0);
        assert!(!a.budget_exhausted);
    }

    /// HTTP 429 without a fallback chain: Retry-After handling stands.
    #[test]
    fn http_429_without_fallback_keeps_retry_after() {
        let fallback = FallbackConfig::default();
        let c = RecoveryConfig {
            retry: RetryConfig {
                enabled: true,
                max_retries: 2,
                base_delay_ms: 100,
                max_delay_ms: 60_000,
                revert_policy: "cooldown-expiry".into(),
            },
            fallback,
            chain_key: "p/a".into(),
            role_label: None,
        };
        let mut r = TurnRecovery::new(c);
        let mut f = failed("429 rate limit exceeded. Retry-After: 5");
        f.status = Some(429);
        f.retry_after_ms = Some(5_000);
        let a = r.begin_attempt(&f, 0.0);
        match r.finish_attempt(&a, false) {
            Verdict::Retry { delay_ms } => assert!(delay_ms >= 5_000),
            Verdict::GiveUp { .. } => panic!("429 without fallback must retry, not give up"),
        }
    }

    /// Buffon `http-429-exhausted`: a 429 with `Retry-After: 0` retries on the
    /// exponential backoff, not the 30 s rate-limit window, so an exhausted
    /// budget ends in well under a minute instead of ten 30 s waits.
    #[test]
    fn header_retry_after_zero_overrides_the_rate_limit_window() {
        let mut r = TurnRecovery::new(cfg(&[]));
        let mut f = failed("429 buffon injected 429 (type=rate_limit_error)");
        f.status = Some(429);
        let heuristic = r.begin_attempt(&f, 0.0);
        assert!(heuristic.delay_ms >= 10_000, "{heuristic:?}");
        r.finish_attempt(&heuristic, false);
        let mut r = TurnRecovery::new(cfg(&[]));
        f.retry_after_ms = Some(0);
        let a = r.begin_attempt(&f, 0.0);
        assert_eq!(a.retry_after_ms, Some(0));
        assert_eq!(a.delay_ms, 100);
        assert!(matches!(
            r.finish_attempt(&a, false),
            Verdict::Retry { delay_ms: 100 }
        ));
    }

    #[test]
    fn is_rate_limit_only_matches_real_429s() {
        let r = TurnRecovery::new(cfg(&[]));
        assert!(r.is_rate_limit(&FailedTurn {
            message: "Provider rate limit exceeded (rate_limit_error, 429)".into(),
            status: Some(429),
            provider: "p".into(),
            model: "m".into(),
            retry_after_ms: None,
        }));
        assert!(!r.is_rate_limit(&FailedTurn {
            message: "503 overloaded".into(),
            status: Some(503),
            provider: "p".into(),
            model: "m".into(),
            retry_after_ms: None,
        }));
        // A transient without an explicit 429 status is not a rotation trigger.
        assert!(!r.is_rate_limit(&failed("please try again later")));
    }

    #[test]
    fn account_rotation_walks_alternates_and_burns_failed() {
        // The runtime keeps the pool's `current` in sync with the promoted account.
        let mut pool = omp_providers::auth::CredentialPool {
            current: omp_providers::auth::Credential {
                key: "k0".into(),
                id: 10,
            },
            alternates: vec![
                omp_providers::auth::Credential {
                    key: "k1".into(),
                    id: 11,
                },
                omp_providers::auth::Credential {
                    key: "k2".into(),
                    id: 12,
                },
            ],
        };
        let mut rot = AccountRotation::default();

        let first = rot.next("p", "m", &pool, false).expect("first alternate");
        assert_eq!(first.id, 11);
        rot.promote("p", "m", first.id);
        rot.burn("p", "m", 10);
        pool.current = first.clone();
        assert_eq!(rot.burned_of("p", "m"), vec![10]);

        // 11 serves now (serving): 12 is the only unburned alternate.
        let second = rot.next("p", "m", &pool, false).expect("second alternate");
        assert_eq!(second.id, 12);
        rot.promote("p", "m", second.id);
        rot.burn("p", "m", 11);
        pool.current = second;
        assert_eq!(rot.burned_of("p", "m"), vec![10, 11]);

        // 12 serves now; 11 is burned and serving filters 12 out — nothing left.
        assert!(rot.next("p", "m", &pool, false).is_none());
        assert!(rot.next("p", "m", &pool, true).is_none());
    }

    #[test]
    fn account_rotation_never_cycles_promoted_back_immediately() {
        let mut rot = AccountRotation::default();
        let pool = omp_providers::auth::CredentialPool {
            current: omp_providers::auth::Credential {
                key: "k0".into(),
                id: 10,
            },
            alternates: vec![
                omp_providers::auth::Credential {
                    key: "k1".into(),
                    id: 11,
                },
                omp_providers::auth::Credential {
                    key: "k2".into(),
                    id: 12,
                },
            ],
        };
        // Window after current=10: 11, 12.
        let first = rot.next("p", "m", &pool, false).expect("first");
        assert_eq!(first.id, 11);
        rot.promote("p", "m", 11);
        rot.burn("p", "m", 10);
        // Static pool (current still 10): next must not re-pick the promoted 11.
        let second = rot.next("p", "m", &pool, false).expect("second");
        assert_eq!(second.id, 12);
    }

    #[test]
    fn account_rotation_cooldown_skips_recently_failed() {
        let pool = omp_providers::auth::CredentialPool {
            current: omp_providers::auth::Credential {
                key: "k0".into(),
                id: 20,
            },
            alternates: vec![
                omp_providers::auth::Credential {
                    key: "k1".into(),
                    id: 21,
                },
                omp_providers::auth::Credential {
                    key: "k2".into(),
                    id: 22,
                },
            ],
        };
        let mut rot = AccountRotation::default();
        rot.burn("p", "m", 21);
        // While the provider cooldown runs, the burned alternate is skipped: 22 goes first.
        assert_eq!(rot.next("p", "m", &pool, true).unwrap().id, 22);
        // Cooldown cleared: the account is eligible again (burned list still skips 21 for a
        // fresh cycle only after clear_cooldown — cooling is the short-term gate).
        rot.clear_cooldown("p", "m");
        assert_eq!(rot.next("p", "m", &pool, false).unwrap().id, 22);
    }

    #[test]
    fn budget_exhaustion_without_switch_gives_up_and_closes_saga() {
        let mut r = TurnRecovery::new(cfg(&[]));
        let f = failed("503 overloaded");
        for n in 1..=2 {
            let a = r.begin_attempt(&f, 0.0);
            assert_eq!((a.number, a.budget_exhausted), (n, false));
            assert!(matches!(r.finish_attempt(&a, false), Verdict::Retry { .. }));
        }
        let a = r.begin_attempt(&f, 0.0);
        assert!(a.budget_exhausted);
        match r.finish_attempt(&a, false) {
            Verdict::GiveUp {
                frame,
                error_message,
            } => {
                assert_eq!(frame["type"], "auto_retry_end");
                assert_eq!(frame["success"], false);
                assert_eq!(frame["attempt"], 2);
                assert_eq!(
                    error_message.as_deref(),
                    Some("Retry budget exhausted after 2 retries: 503 overloaded")
                );
            }
            Verdict::Retry { .. } => panic!("expected give-up"),
        }
        assert!(!r.saga_open());
    }

    #[test]
    fn a_switch_resets_the_budget_and_skips_the_backoff() {
        let mut r = TurnRecovery::new(cfg(&["q/b"]));
        let f = failed("503 overloaded");
        for _ in 0..2 {
            let a = r.begin_attempt(&f, 0.0);
            r.finish_attempt(&a, false);
        }
        let a = r.begin_attempt(&f, 0.0);
        assert!(a.budget_exhausted);
        assert_eq!(r.consult("p/a", &a, 1_000, 0.0), vec!["q/b".to_string()]);
        r.commit_switch("p/a", Some("high".into()), Some("high".into()));
        assert_eq!(r.finish_attempt(&a, true), Verdict::Retry { delay_ms: 0 });
        assert_eq!(r.attempt(), 1);
        // the chain keeps walking from the fallback, never back to the cooled-down primary
        let a = r.begin_attempt(&f, 0.0);
        assert!(r.consult("q/b", &a, 2_000, 0.0).is_empty());
    }

    #[test]
    fn long_provider_wait_fails_fast() {
        let mut r = TurnRecovery::new(cfg(&[]));
        let a = r.begin_attempt(&failed("429 retry-after: 7200"), 0.0);
        match r.finish_attempt(&a, false) {
            Verdict::GiveUp { frame, .. } => {
                let e = frame["finalError"].as_str().unwrap();
                assert!(
                    e.starts_with(
                        "Provider requested 7200000ms wait, exceeds retry.maxDelayMs (60000ms)"
                    ),
                    "{e}"
                );
            }
            Verdict::Retry { .. } => panic!("expected give-up"),
        }
    }

    #[test]
    fn settled_ok_orders_fallback_succeeded_before_retry_end() {
        let mut r = TurnRecovery::new(cfg(&["q/b"]));
        let f = failed("503");
        let a = r.begin_attempt(&f, 0.0);
        r.consult("p/a", &a, 0, 0.0);
        r.commit_switch("p/a", None, None);
        r.finish_attempt(&a, true);
        let frames = r.on_settled_ok("q/b");
        let kinds: Vec<_> = frames
            .iter()
            .map(|f| f["type"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(kinds, ["retry_fallback_succeeded", "auto_retry_end"]);
        assert_eq!(frames[0]["role"], "default");
        assert_eq!(frames[1]["success"], true);
        assert!(r.on_settled_ok("q/b").is_empty(), "one-shot");
    }

    #[test]
    fn abort_retry_cancels_only_the_armed_sleep() {
        let mut r = TurnRecovery::new(cfg(&[]));
        assert!(!r.abort_retry());
        let (g, token) = r.arm_cancel();
        assert!(r.abort_retry());
        assert!(token.is_cancelled());
        assert!(!r.disarm(g), "already taken by abort");
        let (g2, t2) = r.arm_cancel();
        let (_g3, t3) = r.arm_cancel();
        assert!(t2.is_cancelled(), "superseded");
        assert!(!r.disarm(g2));
        assert!(!t3.is_cancelled());
    }

    // ------------------------------------------------------- saga through the runtime

    use std::sync::Mutex;
    use std::time::Duration;

    use omp_providers::ThinkingSource;
    use pa_agent::agent::{Agent, AgentInitialState, AgentOptions};
    use pa_agent::scripted::ScriptedProvider;
    use pa_agent::stream::StreamFn;
    use tokio::sync::mpsc::UnboundedReceiver;

    use crate::adapter::{json_round_trip, ProviderTarget};
    use crate::runtime::Runtime;

    fn pa_model(provider: &str, id: &str) -> pa_types::ai::Model {
        serde_json::from_value(json!({
            "id": id, "name": id, "api": "openai-completions", "provider": provider,
            "baseUrl": "http://127.0.0.1:9", "reasoning": true, "input": ["text"],
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
            "contextWindow": 100_000, "maxTokens": 4096
        }))
        .expect("model json")
    }

    fn resolved(provider: &str, id: &str) -> ResolvedModel {
        ResolvedModel {
            provider: provider.into(),
            id: id.into(),
            api: "openai-completions".into(),
            base_url: "http://127.0.0.1:9".into(),
            api_key: Some("test-key".into()),
            credential_id: None,
            headers: Vec::new(),
            context_window: 100_000,
            max_tokens: 4096,
            reasoning: true,
            thinking: None,
            thinking_auto: false,
            thinking_source: ThinkingSource::None,
            selector: format!("{provider}/{id}"),
            role: None,
            fallbacks: Vec::new(),
            retry: RetryConfig::default(),
            model: pa_model(provider, id),
        }
    }

    struct Harness {
        rt: std::sync::Arc<Runtime>,
        rx: UnboundedReceiver<Value>,
        models_seen: std::sync::Arc<Mutex<Vec<String>>>,
        session_file: std::path::PathBuf,
        _provider: std::sync::Arc<ScriptedProvider>,
    }

    /// `turns`: scripted provider turns, `Err(msg)` = stream failure, `Ok(text)` = a normal answer.
    async fn harness(turns: &[Result<&str, &str>], cfg: RecoveryConfig) -> Harness {
        harness_with(
            |provider| {
                for t in turns {
                    match t {
                        Ok(text) => provider.push_text_turn(text),
                        Err(msg) => provider.push_stream_failure_turn("", msg),
                    }
                }
            },
            cfg,
        )
        .await
    }

    async fn harness_with(script: impl FnOnce(&ScriptedProvider), cfg: RecoveryConfig) -> Harness {
        let primary = pa_model("p", "a");
        let provider = std::sync::Arc::new(ScriptedProvider::new(
            json_round_trip(&primary).expect("agent model"),
        ));
        script(&provider);
        let inner = provider.stream_fn();
        let models_seen = std::sync::Arc::new(Mutex::new(Vec::new()));
        let seen = models_seen.clone();
        let stream_fn: StreamFn = std::sync::Arc::new(move |m, c, o| {
            seen.lock().unwrap().push(m.id.clone());
            inner(m, c, o)
        });
        let agent = Agent::new(AgentOptions {
            initial_state: AgentInitialState {
                system_prompt: Some("test".into()),
                model: json_round_trip(&primary),
                thinking_level: Some(pa_agent::types::ThinkingLevel::Off),
                tools: Some(Vec::new()),
                messages: None,
            },
            stream_fn: Some(stream_fn),
            ..AgentOptions::default()
        });
        let cwd = std::env::temp_dir();
        let persist = crate::persist::attach_with(&agent, &cwd, true)
            .await
            .expect("session");
        let session_file = persist.path();
        let target = std::sync::Arc::new(std::sync::RwLock::new(Some(ProviderTarget {
            api_key: Some("test-key".into()),
            model: primary,
            service_tier: None,
            headers: None,
            credential_id: None,
        })));
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let resolver: Resolver = std::sync::Arc::new(|sel: String| {
            let out: Result<ResolvedModel> = (|| {
                let base = sel.split(':').next().unwrap_or(&sel);
                let (p, i) = base
                    .split_once('/')
                    .ok_or_else(|| anyhow::anyhow!("bad selector"))?;
                Ok(resolved(p, i))
            })();
            Box::pin(async move { out })
        });
        let rt = Runtime::start_with(
            std::sync::Arc::new(agent),
            target,
            cwd,
            persist,
            tx,
            resolver,
            Some(cfg),
        )
        .await;
        Harness {
            rt,
            rx,
            models_seen,
            session_file,
            _provider: provider,
        }
    }

    fn fast_cfg(chain: &[&str], max_retries: u32, base_delay_ms: u64) -> RecoveryConfig {
        let mut c = cfg(chain);
        c.retry.max_retries = max_retries;
        c.retry.base_delay_ms = base_delay_ms;
        c
    }

    /// Frames up to and including `session_settled` (30 s guard).
    async fn until_settled(rx: &mut UnboundedReceiver<Value>) -> Vec<Value> {
        let mut out = Vec::new();
        loop {
            let f = tokio::time::timeout(Duration::from_secs(30), rx.recv())
                .await
                .expect("frame in time")
                .expect("open");
            let done = f["type"] == "session_settled";
            out.push(f);
            if done {
                return out;
            }
        }
    }

    fn kinds(frames: &[Value]) -> Vec<String> {
        frames
            .iter()
            .map(|f| f["type"].as_str().unwrap_or("?").to_string())
            .filter(|k| {
                !matches!(
                    k.as_str(),
                    "message_start"
                        | "message_update"
                        | "message_end"
                        | "turn_start"
                        | "turn_end"
                        | "queue_update"
                )
            })
            .collect()
    }

    fn session_entries(path: &std::path::Path) -> Vec<Value> {
        crate::runtime::read_entries(path).expect("entries")
    }

    #[tokio::test]
    async fn retry_saga_with_fallback_switch_orders_frames_and_entries() {
        let mut h = harness(
            &[
                Err("503 Service Unavailable: overloaded"),
                Ok("hello from b"),
            ],
            fast_cfg(&["q/b"], 3, 1),
        )
        .await;
        h.rt.prompt(Some(json!("p1")), "hi", None, Vec::new())
            .await
            .expect("prompt");
        let frames = until_settled(&mut h.rx).await;
        assert_eq!(
            kinds(&frames),
            [
                "agent_start",
                "model_changed",
                "retry_fallback_applied",
                "auto_retry_start",
                "agent_end",
                "agent_start",
                "retry_fallback_succeeded",
                "auto_retry_end",
                "agent_end",
                "prompt_result",
                "session_settled",
            ]
        );
        let get = |k: &str| frames.iter().find(|f| f["type"] == k).unwrap().clone();
        assert_eq!(get("retry_fallback_applied")["from"], "p/a:off");
        assert_eq!(get("retry_fallback_applied")["to"], "q/b");
        assert_eq!(get("retry_fallback_applied")["role"], "default");
        let start = get("auto_retry_start");
        assert_eq!(
            (
                start["attempt"].clone(),
                start["maxAttempts"].clone(),
                start["delayMs"].clone()
            ),
            (json!(1), json!(3), json!(0))
        );
        assert_eq!(get("retry_fallback_succeeded")["model"], "q/b");
        let end = get("auto_retry_end");
        assert_eq!(
            (end["success"].clone(), end["attempt"].clone()),
            (json!(true), json!(1))
        );
        let ends: Vec<_> = frames.iter().filter(|f| f["type"] == "agent_end").collect();
        assert_eq!(
            (ends[0]["isTerminal"].clone(), ends[0]["yielded"].clone()),
            (json!(false), json!(false))
        );
        assert_eq!(ends[1]["isTerminal"], true);
        assert_eq!(get("prompt_result")["status"], "completed");
        assert_eq!(
            *h.models_seen.lock().unwrap(),
            ["a", "b"],
            "second attempt ran on the fallback model"
        );

        let entries = session_entries(&h.session_file);
        let changes: Vec<_> = entries
            .iter()
            .filter(|e| e["type"] == "model_change")
            .collect();
        let last = changes.last().unwrap();
        assert_eq!(last["model"], "q/b");
        assert_eq!(last["role"], "fallback");
        assert_eq!(last["resolvedModelIsFallback"], true);
        let assistants: Vec<_> = entries
            .iter()
            .filter(|e| e["type"] == "message" && e["message"]["role"] == "assistant")
            .collect();
        assert_eq!(
            assistants.len(),
            1,
            "the failed turn is not persisted: {assistants:?}"
        );
        assert_eq!(assistants[0]["message"]["stopReason"], "stop");
    }

    #[tokio::test]
    async fn retry_on_same_model_backs_off_then_succeeds() {
        let mut h = harness(
            &[Err("503 overloaded"), Err("503 overloaded"), Ok("finally")],
            fast_cfg(&[], 5, 5),
        )
        .await;
        h.rt.prompt(Some(json!("p1")), "hi", None, Vec::new())
            .await
            .expect("prompt");
        let frames = until_settled(&mut h.rx).await;
        let k = kinds(&frames);
        assert_eq!(
            k,
            [
                "agent_start",
                "auto_retry_start",
                "agent_end",
                "agent_start",
                "auto_retry_start",
                "agent_end",
                "agent_start",
                "auto_retry_end",
                "agent_end",
                "prompt_result",
                "session_settled"
            ]
        );
        let attempts: Vec<_> = frames
            .iter()
            .filter(|f| f["type"] == "auto_retry_start")
            .map(|f| f["attempt"].clone())
            .collect();
        assert_eq!(attempts, [json!(1), json!(2)]);
        assert_eq!(
            frames
                .iter()
                .find(|f| f["type"] == "auto_retry_end")
                .unwrap()["attempt"],
            2
        );
        assert_eq!(*h.models_seen.lock().unwrap(), ["a", "a", "a"]);
        assert!(!frames.iter().any(|f| f["type"] == "model_changed"));
        let users: Vec<_> = frames
            .iter()
            .filter(|f| f["type"] == "message_end" && f["message"]["role"] == "user")
            .collect();
        assert_eq!(
            users[0]["message"]["attribution"], "user",
            "frame: {}",
            users[0]
        );
        let entries = session_entries(&h.session_file);
        let user = entries
            .iter()
            .find(|e| e["message"]["role"] == "user")
            .expect("user entry");
        assert_eq!(user["message"]["attribution"], "user", "entry: {user}");
        let raw = std::fs::read_to_string(&h.session_file).unwrap();
        let line = raw
            .lines()
            .find(|l| l.contains("\"role\":\"user\""))
            .unwrap();
        assert!(
            line.find("\"content\"").unwrap() < line.find("\"attribution\"").unwrap(),
            "{line}"
        );
    }

    #[tokio::test]
    async fn exhausted_budget_closes_saga_and_persists_the_failure() {
        let mut h = harness(
            &[Err("503 overloaded"), Err("503 overloaded")],
            fast_cfg(&[], 1, 1),
        )
        .await;
        h.rt.prompt(Some(json!("p1")), "hi", None, Vec::new())
            .await
            .expect("prompt");
        let frames = until_settled(&mut h.rx).await;
        assert_eq!(
            kinds(&frames),
            [
                "agent_start",
                "auto_retry_start",
                "agent_end",
                "agent_start",
                "auto_retry_end",
                "agent_end",
                "prompt_result",
                "session_settled"
            ]
        );
        let end = frames
            .iter()
            .find(|f| f["type"] == "auto_retry_end")
            .unwrap();
        assert_eq!(
            (
                end["success"].clone(),
                end["attempt"].clone(),
                end["finalError"].clone()
            ),
            (json!(false), json!(1), json!("503 overloaded"))
        );
        let result = frames
            .iter()
            .find(|f| f["type"] == "prompt_result")
            .unwrap();
        assert_eq!(result["status"], "error");
        assert_eq!(
            result["error"]["message"],
            "Retry budget exhausted after 1 retry: 503 overloaded"
        );
        tokio::time::timeout(Duration::from_secs(5), h.rt.wait_settled())
            .await
            .expect("settled");
        let entries = session_entries(&h.session_file);
        let failed: Vec<_> = entries
            .iter()
            .filter(|e| e["message"]["stopReason"] == "error")
            .collect();
        assert_eq!(failed.len(), 1, "only the terminal failure is persisted");
        assert_eq!(
            failed[0]["message"]["errorMessage"],
            "Retry budget exhausted after 1 retry: 503 overloaded"
        );
    }

    #[tokio::test]
    async fn abort_retry_during_backoff_ends_cancelled() {
        let mut h = harness(&[Err("503 overloaded")], fast_cfg(&[], 5, 60_000)).await;
        h.rt.prompt(Some(json!("p1")), "hi", None, Vec::new())
            .await
            .expect("prompt");
        let mut frames = Vec::new();
        loop {
            let f = tokio::time::timeout(Duration::from_secs(30), h.rx.recv())
                .await
                .expect("frame")
                .expect("open");
            let start = f["type"] == "auto_retry_start";
            frames.push(f);
            if start {
                break;
            }
        }
        h.rt.abort_retry();
        frames.extend(until_settled(&mut h.rx).await);
        let k = kinds(&frames);
        assert_eq!(
            k,
            [
                "agent_start",
                "auto_retry_start",
                "auto_retry_end",
                "agent_end",
                "prompt_result",
                "session_settled"
            ]
        );
        let end = frames
            .iter()
            .find(|f| f["type"] == "auto_retry_end")
            .unwrap();
        assert_eq!(
            (end["success"].clone(), end["finalError"].clone()),
            (json!(false), json!("Retry cancelled"))
        );
        assert_eq!(
            frames
                .iter()
                .find(|f| f["type"] == "prompt_result")
                .unwrap()["status"],
            "error"
        );
    }

    #[tokio::test]
    async fn non_retryable_error_settles_without_a_saga() {
        let mut h = harness(&[Err("401 invalid api key")], fast_cfg(&["q/b"], 3, 1)).await;
        h.rt.prompt(Some(json!("p1")), "hi", None, Vec::new())
            .await
            .expect("prompt");
        let frames = until_settled(&mut h.rx).await;
        assert_eq!(
            kinds(&frames),
            [
                "agent_start",
                "agent_end",
                "prompt_result",
                "session_settled"
            ]
        );
        assert_eq!(
            frames.iter().find(|f| f["type"] == "agent_end").unwrap()["isTerminal"],
            true
        );
        let entries = session_entries(&h.session_file);
        assert_eq!(
            entries
                .iter()
                .filter(|e| e["message"]["stopReason"] == "error")
                .count(),
            1
        );
    }

    #[tokio::test]
    async fn abort_reports_the_command_response_before_prompt_result() {
        let mut h = harness_with(|p| p.push_stalled_turn("partial"), fast_cfg(&[], 3, 1)).await;
        h.rt.prompt(Some(json!("p1")), "hi", None, Vec::new())
            .await
            .expect("prompt");
        loop {
            let f = tokio::time::timeout(Duration::from_secs(30), h.rx.recv())
                .await
                .expect("frame")
                .expect("open");
            if f["type"] == "message_start" && f["message"]["role"] == "assistant" {
                break;
            }
        }
        let reply = crate::rpc::handle(&h.rt, &json!({"id": "a1", "type": "abort"})).await;
        h.rt.send(reply);
        let frames = until_settled(&mut h.rx).await;
        let k = kinds(&frames);
        assert_eq!(
            k,
            ["agent_end", "response", "prompt_result", "session_settled"],
            "{k:?}"
        );
        let result = frames
            .iter()
            .find(|f| f["type"] == "prompt_result")
            .unwrap();
        assert_eq!(
            (
                result["status"].clone(),
                result["id"].clone(),
                result["sessionSettled"].clone()
            ),
            (json!("aborted"), json!("p1"), json!(true))
        );
        let end = frames.iter().find(|f| f["type"] == "agent_end").unwrap();
        assert_eq!(end["isTerminal"], true);
    }

    #[tokio::test]
    async fn user_model_switch_and_thinking_changes_are_journaled_with_omp_shapes() {
        let mut h = harness(&[], fast_cfg(&[], 3, 1)).await;
        h.rt.set_model("q", "b").await.expect("switch");
        h.rt.set_thinking_level(pa_agent::types::ThinkingLevel::High)
            .await;
        h.rt.set_thinking_level(pa_agent::types::ThinkingLevel::High)
            .await; // unchanged: silent
        h.rt.cycle_to("r", "c").await.expect("cycle");
        let mut frames = Vec::new();
        while let Ok(f) = h.rx.try_recv() {
            frames.push(f);
        }
        let k: Vec<_> = frames
            .iter()
            .map(|f| f["type"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(
            k,
            ["model_changed", "thinking_level_changed", "model_changed"],
            "{frames:?}"
        );
        assert_eq!(
            frames[1],
            json!({"type": "thinking_level_changed", "thinkingLevel": "high", "ultra": null})
        );
        let entries = session_entries(&h.session_file);
        let shapes: Vec<_> = entries
            .iter()
            .filter(|e| {
                matches!(
                    e["type"].as_str(),
                    Some("model_change" | "thinking_level_change")
                )
            })
            .map(|e| {
                let mut m = e.as_object().unwrap().clone();
                for k in ["id", "parentId", "timestamp"] {
                    m.remove(k);
                }
                Value::Object(m)
            })
            .collect();
        assert_eq!(
            shapes,
            [
                json!({"type": "model_change", "model": "p/a", "resolvedModelIsFallback": false}),
                json!({"type": "thinking_level_change", "thinkingLevel": "off", "configured": null}),
                json!({"type": "model_change", "model": "q/b", "role": "default", "resolvedModelIsFallback": false}),
                json!({"type": "thinking_level_change", "thinkingLevel": "high", "configured": "high"}),
                json!({"type": "model_change", "model": "r/c", "resolvedModelIsFallback": false}),
            ]
        );
    }

    #[test]
    fn abort_before_the_backoff_is_armed_still_cancels_it() {
        let mut r = TurnRecovery::new(cfg(&[]));
        r.begin_attempt(&failed("503"), 0.0);
        assert!(!r.abort_retry(), "nothing armed yet");
        let (_g, token) = r.arm_cancel();
        assert!(token.is_cancelled());
        // an abort with no attempt in flight must not poison the next backoff
        let mut r = TurnRecovery::new(cfg(&[]));
        assert!(!r.abort_retry());
        r.begin_attempt(&failed("503"), 0.0);
        assert!(!r.arm_cancel().1.is_cancelled());
    }

    #[test]
    fn reconfigure_keeps_other_selectors_cooldowns_and_drops_fallback_ownership() {
        let mut r = TurnRecovery::new(cfg(&["q/b"]));
        let a = r.begin_attempt(&failed("503"), 0.0);
        r.consult("p/a", &a, 1_000, 0.0);
        r.commit_switch("p/a", None, None);
        assert!(r.is_cooling("p/a", 1_001));
        r.reconfigure(cfg(&["r/c"]));
        r.clear_cooldown("q/b");
        assert!(
            r.is_cooling("p/a", 1_001),
            "unrelated cooldown survives a model switch"
        );
        assert!(
            r.on_settled_ok("q/b")
                .iter()
                .all(|f| f["type"] != "retry_fallback_succeeded"),
            "ownership ended"
        );
    }

    // ------------------------------------------------ review round 3 regressions

    /// Frames until `auto_retry_start` (backoff pending).
    async fn until_backoff(h: &mut Harness) -> Vec<Value> {
        let mut frames = Vec::new();
        loop {
            let f = tokio::time::timeout(Duration::from_secs(30), h.rx.recv())
                .await
                .expect("frame")
                .expect("open");
            let start = f["type"] == "auto_retry_start";
            frames.push(f);
            if start {
                return frames;
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn terminal_error_settle_and_get_state_do_not_deadlock() {
        for _ in 0..30 {
            let mut h = harness(&[Err("401 invalid api key")], fast_cfg(&[], 3, 1)).await;
            // open a saga so `on_error_settled` has something to close
            h.rt.prompt(Some(json!("p")), "hi", None, Vec::new())
                .await
                .unwrap();
            let rt = h.rt.clone();
            let poller = tokio::spawn(async move {
                for _ in 0..200 {
                    rt.state().await;
                    tokio::task::yield_now().await;
                }
            });
            tokio::time::timeout(Duration::from_secs(10), until_settled(&mut h.rx))
                .await
                .expect("no deadlock");
            tokio::time::timeout(Duration::from_secs(10), poller)
                .await
                .expect("state() finished")
                .unwrap();
        }
    }

    #[tokio::test]
    async fn new_session_during_backoff_cancels_it_and_leaves_the_new_file_clean() {
        let mut h = harness(&[Err("503 overloaded")], fast_cfg(&[], 5, 60_000)).await;
        h.rt.prompt(Some(json!("p1")), "hi", None, Vec::new())
            .await
            .unwrap();
        until_backoff(&mut h).await;
        h.rt.new_session(None).await.expect("new session");
        let rest = tokio::time::timeout(Duration::from_secs(5), async {
            let mut v = Vec::new();
            while let Some(f) = h.rx.recv().await {
                let done = f["type"] == "session_settled";
                v.push(f);
                if done {
                    break;
                }
            }
            v
        })
        .await
        .expect("saga ended promptly");
        assert!(
            rest.iter()
                .any(|f| f["type"] == "auto_retry_end" && f["finalError"] == "Retry cancelled"),
            "{rest:?}"
        );
        let entries = session_entries(&h.rt.session_file());
        assert!(
            entries
                .iter()
                .all(|e| e["message"]["stopReason"] != "error"),
            "old failure leaked into the new session"
        );
        assert!(!h.rt.state().await["isSettled"].is_null());
        assert_eq!(h.rt.state().await["isSettled"], true);
    }

    #[tokio::test]
    async fn prompt_during_backoff_does_not_inherit_the_old_turns_result() {
        let mut h = harness(
            &[Err("503 overloaded"), Ok("fresh")],
            fast_cfg(&[], 5, 60_000),
        )
        .await;
        h.rt.prompt(Some(json!("old")), "first", None, Vec::new())
            .await
            .unwrap();
        until_backoff(&mut h).await;
        h.rt.prompt(Some(json!("new")), "second", None, Vec::new())
            .await
            .unwrap();
        let mut frames = Vec::new();
        while frames
            .iter()
            .filter(|f: &&Value| f["type"] == "prompt_result")
            .count()
            < 2
        {
            frames.push(
                tokio::time::timeout(Duration::from_secs(30), h.rx.recv())
                    .await
                    .expect("frame")
                    .expect("open"),
            );
        }
        let detail: Vec<_> = frames
            .iter()
            .filter(|f| f["type"] == "prompt_result")
            .map(ToString::to_string)
            .collect();
        let results: Vec<_> = frames
            .iter()
            .filter(|f| f["type"] == "prompt_result")
            .map(|f| (f["id"].clone(), f["status"].clone()))
            .collect();
        assert_eq!(
            results,
            [
                (json!("old"), json!("error")),
                (json!("new"), json!("completed"))
            ],
            "{detail:?}"
        );
    }

    #[tokio::test]
    async fn cancelled_retry_never_persists_the_superseded_turn() {
        let mut h = harness(&[Err("503 overloaded")], fast_cfg(&[], 5, 60_000)).await;
        h.rt.prompt(Some(json!("p1")), "hi", None, Vec::new())
            .await
            .unwrap();
        until_backoff(&mut h).await;
        h.rt.abort().await;
        until_settled(&mut h.rx).await;
        let entries = session_entries(&h.session_file);
        // omp: a failed attempt reaches the file only when the saga gives up; a cancel is not that.
        assert!(entries
            .iter()
            .all(|e| e["message"]["stopReason"] != "error"));
    }

    #[tokio::test]
    async fn queue_edits_match_exactly_and_touch_one_message() {
        let mut h = harness_with(|p| p.push_stalled_turn("busy"), fast_cfg(&[], 3, 1)).await;
        h.rt.prompt(Some(json!("p")), "run", None, Vec::new())
            .await
            .unwrap();
        loop {
            let f = tokio::time::timeout(Duration::from_secs(30), h.rx.recv())
                .await
                .unwrap()
                .unwrap();
            if f["type"] == "message_start" && f["message"]["role"] == "assistant" {
                break;
            }
        }
        for t in ["fix the bug", "fix the bug", "fix"] {
            h.rt.prompt(None, t, Some("followUp"), Vec::new())
                .await
                .unwrap();
        }
        let q = |rt: &Runtime| rt.queued_previews().1;
        assert_eq!(q(&h.rt), ["fix the bug", "fix the bug", "fix"]);
        assert!(
            !h.rt.remove_queued("fix the", "followUp"),
            "prefix must not match"
        );
        assert!(!h.rt.remove_queued("fix", "steering"), "wrong queue");
        assert!(h.rt.remove_queued("fix", "followUp"));
        assert_eq!(
            q(&h.rt),
            ["fix the bug", "fix the bug"],
            "exactly one, and the right one"
        );
        assert!(h.rt.promote_queued("fix the bug"));
        assert_eq!(
            (h.rt.queued_previews().0, q(&h.rt)),
            (
                vec!["fix the bug".to_string()],
                vec!["fix the bug".to_string()]
            )
        );
        assert!(!h.rt.promote_queued("fix"));
        h.rt.abort().await;
    }

    /// A slow `!command` api key must run on the blocking pool: a single-threaded
    /// executor keeps ticking while the 2 s key resolves.
    #[tokio::test]
    async fn slow_command_key_does_not_block_the_executor() {
        let t = tempfile::tempdir().unwrap();
        std::fs::write(
            t.path().join("models.yml"),
            "providers:\n  zz:\n    baseUrl: http://127.0.0.1:1/v1\n    apiKey: \"!sleep 2; echo k\"\n    api: openai-completions\n    models:\n      - id: m1\n",
        )
        .unwrap();
        let resolver = resolver_in(omp_providers::AgentDir(t.path().to_path_buf()));
        let ticks = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let ticker = {
            let ticks = ticks.clone();
            tokio::spawn(async move {
                loop {
                    tokio::time::sleep(Duration::from_millis(20)).await;
                    ticks.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                }
            })
        };
        let model = resolver("zz/m1".to_string()).await.unwrap();
        let ticks_during = ticks.load(std::sync::atomic::Ordering::SeqCst);
        ticker.abort();
        assert_eq!(model.api_key.as_deref(), Some("k"));
        assert!(
            ticks_during >= 10,
            "executor blocked during the slow key: {ticks_during} ticks"
        );
    }

    /// TS `#drainStrandedQueuedMessages`: an abort leaves a queued steer/follow-up
    /// stranded with no loop left to poll it, so the idle session starts a fresh turn
    /// for the steer. A follow-up-only remainder stays suppressed after the user
    /// interrupt (`#canAutoContinueForFollowUp` + `autoResumeSuppressed`).
    #[tokio::test]
    async fn abort_drains_queued_steer_into_a_fresh_turn() {
        let mut h = harness_with(
            |provider| {
                provider.push_stalled_turn("partial");
                // The drain turn is a fresh run (TS `agent.continue()`): it answers.
                provider.push_text_turn("drained answer");
            },
            cfg(&["p/a"]),
        )
        .await;
        h.rt.prompt(None, "hello", None, Vec::new())
            .await
            .expect("prompt");
        // Wait until the stalled turn is actually streaming.
        wait_for(&mut h.rx, |f| {
            f["type"] == "message_update" && f["assistantMessageEvent"]["type"] == "text_delta"
        })
        .await;
        h.rt.steer("more detail");
        h.rt.follow_up("and more");
        h.rt.abort().await;

        let mut agent_starts = 1; // the first run already started
        let mut second_turn_user = false;
        let mut second_turn_ended = false;
        let deadline = tokio::time::sleep(Duration::from_secs(20));
        tokio::pin!(deadline);
        loop {
            tokio::select! {
                () = &mut deadline => break,
                frame = h.rx.recv() => {
                    let f = frame.expect("open");
                    match f["type"].as_str() {
                        Some("agent_start") => agent_starts += 1,
                        Some("message_start") => {
                            if f["message"]["role"] == json!("user")
                                && json!(f["message"]["content"]) == json!("more detail")
                            {
                                second_turn_user = true;
                            }
                        }
                        Some("agent_end") if second_turn_user => second_turn_ended = true,
                        _ => (),
                    }
                }
            }
        }
        assert_eq!(
            agent_starts, 2,
            "queued steer must start exactly one fresh turn"
        );
        assert!(
            second_turn_user,
            "the drained turn carries the steer message"
        );
        assert!(second_turn_ended, "the drained turn settles");
    }

    /// Regression ("Request was aborted" after an empty Enter): interrupting a run
    /// with a queued steer delivers that steer as a real turn (TS
    /// `#scheduleAgentContinue` -> `agent.continue()` with a fresh signal), and the
    /// next prompt is not aborted either.
    #[tokio::test]
    async fn interrupted_run_delivers_queued_steer_as_a_real_turn() {
        let mut h = harness_with(
            |provider| {
                provider.push_stalled_turn("partial");
                provider.push_text_turn("drained answer");
                provider.push_text_turn("fresh answer");
            },
            cfg(&["p/a"]),
        )
        .await;
        h.rt.prompt(None, "hello", None, Vec::new())
            .await
            .expect("prompt");
        wait_for(&mut h.rx, |f| {
            f["type"] == "message_update" && f["assistantMessageEvent"]["type"] == "text_delta"
        })
        .await;
        h.rt.steer("как успехи");
        h.rt.abort().await;
        wait_for(&mut h.rx, |f| f["type"] == "agent_start").await;
        wait_for(&mut h.rx, |f| f["type"] == "agent_end").await;
        h.rt.agent.wait_for_idle().await;
        let last_stop = |msgs: &[pa_agent::types::AgentMessage]| {
            let last = msgs
                .iter()
                .rev()
                .find(|m| m.role() == "assistant")
                .expect("assistant");
            let pa_agent::types::AgentMessage::Standard(pa_agent::types::Message::Assistant(a)) =
                last
            else {
                panic!("assistant message")
            };
            a.stop_reason.clone()
        };
        let msgs = h.rt.agent.state().await.messages;
        assert_ne!(
            last_stop(&msgs),
            pa_agent::types::StopReason::Aborted,
            "the drained steer turn must run, not end as 'Request was aborted'"
        );

        h.rt.prompt(None, "next", None, Vec::new())
            .await
            .expect("prompt");
        wait_for(&mut h.rx, |f| {
            f["type"] == "message_end" && f["message"]["role"] == json!("assistant")
        })
        .await;
        h.rt.agent.wait_for_idle().await;
        let msgs = h.rt.agent.state().await.messages;
        assert_ne!(
            last_stop(&msgs),
            pa_agent::types::StopReason::Aborted,
            "next prompt not aborted"
        );
    }

    /// Read frames until `pred` matches (30 s guard).
    async fn wait_for(rx: &mut UnboundedReceiver<Value>, pred: impl Fn(&Value) -> bool) {
        loop {
            let f = tokio::time::timeout(Duration::from_secs(30), rx.recv())
                .await
                .expect("frame in time")
                .expect("open");
            if pred(&f) {
                return;
            }
        }
    }
}
