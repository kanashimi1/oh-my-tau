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

use std::sync::Arc;

use anyhow::Result;
use omp_providers::{
    is_retryable_error, parse_rate_limit_reason, rate_limit_backoff_ms, CandidateOptions, FallbackConfig,
    FallbackState, RateLimitReason, ResolvedModel, RetryConfig, RevertPolicy,
};
use pa_agent::types::AssistantMessage;
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

/// `EPHEMERAL_MODEL_CHANGE_ROLE`: role of `model_change` entries written by a fallback switch.
pub const EPHEMERAL_ROLE: &str = "fallback";

/// Resolves a `provider/id[:level]` selector to a callable model (credentials included).
pub type Resolver = Arc<dyn Fn(&str) -> Result<ResolvedModel> + Send + Sync>;

/// The default resolver: omp's `config.yml` / `models.yml` / `agent.db`.
#[must_use]
pub fn default_resolver() -> Resolver {
    Arc::new(omp_providers::resolve)
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
            fallback.chains.push((chain_key.clone(), primary.fallbacks.clone()));
        }
        fallback.known_providers = primary
            .fallbacks
            .iter()
            .filter_map(|s| s.split('/').next())
            .chain([primary.provider.as_str()])
            .map(String::from)
            .collect();
        Self { retry: primary.retry.clone(), fallback, chain_key, role_label: primary.role.clone() }
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
}

impl FailedTurn {
    /// `None` unless `m` stopped with `error`.
    #[must_use]
    pub fn from_message(m: &AssistantMessage) -> Option<Self> {
        if m.stop_reason != pa_agent::types::StopReason::Error {
            return None;
        }
        let status = m
            .diagnostics
            .iter()
            .flatten()
            .filter_map(|d| d.details.as_ref())
            .find_map(|d| d.get("status").and_then(Value::as_u64))
            .and_then(|s| u16::try_from(s).ok());
        Some(Self {
            message: m.error_message.clone().unwrap_or_else(|| "Unknown error".to_string()),
            status,
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
        && ["connection error", "fetch failed", "econnrefused", "econnreset", "enotfound", "etimedout", "dns", "tls handshake"]
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
        let end = rest.find(|c: char| !(c.is_ascii_digit() || c == '.')).unwrap_or(rest.len());
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
    for needle in ["try again in", "retry in", "resets in", "reset in", "available in", "wait "] {
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
pub fn auto_retry_start_frame(attempt: u32, max_attempts: u32, delay_ms: u64, error_message: &str) -> Value {
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

#[must_use]
pub fn fallback_succeeded_frame(model: &str, role: &str) -> Value {
    json!({"type": "retry_fallback_succeeded", "model": model, "role": role})
}

// ----------------------------------------------------------------- state

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
    GiveUp { frame: Value, error_message: Option<String> },
}

pub struct TurnRecovery {
    cfg: RecoveryConfig,
    state: FallbackState,
    enabled: bool,
    attempt: u32,
    generation: u64,
    cancel: Option<(u64, CancellationToken)>,
}

impl TurnRecovery {
    #[must_use]
    pub fn new(cfg: RecoveryConfig) -> Self {
        let state = FallbackState::new(RevertPolicy::from_setting(Some(&cfg.retry.revert_policy)));
        let enabled = cfg.retry.enabled;
        Self { cfg, state, enabled, attempt: 0, generation: 0, cancel: None }
    }

    /// A user-selected model replaces the chain and ends fallback ownership.
    pub fn reconfigure(&mut self, cfg: RecoveryConfig) {
        let enabled = self.enabled;
        let (attempt, generation, cancel) = (self.attempt, self.generation, self.cancel.take());
        *self = Self::new(cfg);
        self.enabled = enabled;
        (self.attempt, self.generation, self.cancel) = (attempt, generation, cancel);
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

    /// Open (or continue) a saga for `failed`: bump the attempt counter, compute the backoff.
    pub fn begin_attempt(&mut self, failed: &FailedTurn, rand01: f64) -> Attempt {
        self.attempt += 1;
        let max = self.cfg.retry.max_retries;
        let retry_after_ms = parse_retry_after_ms(&failed.message);
        let mut delay_ms = self.cfg.retry.backoff_ms(self.attempt, rand01);
        if retry_after_ms.is_none() {
            let reason = parse_rate_limit_reason(&failed.message);
            if matches!(reason, RateLimitReason::ConcurrentLimit | RateLimitReason::RateLimitExceeded) {
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
    pub fn consult(&mut self, current: &str, attempt: &Attempt, now_ms: u64, rand01: f64) -> Vec<String> {
        if self.cfg.fallback.chains.is_empty() {
            return Vec::new();
        }
        self.state.note_cooldown(current, attempt.retry_after_ms, &attempt.error_message, now_ms, rand01);
        self.state.candidates(&self.cfg.fallback, current, None, CandidateOptions::default(), now_ms)
    }

    /// Record a committed switch onto a fallback (`from` is the selector left behind).
    pub fn commit_switch(&mut self, from: &str, from_thinking: Option<String>, applied_thinking: Option<String>) {
        let key = self.cfg.chain_key.clone();
        self.state.apply(&key, from, from_thinking, applied_thinking, false);
    }

    /// Decide how the attempt ends (`switched` = a fallback model took over).
    pub fn finish_attempt(&mut self, a: &Attempt, switched: bool) -> Verdict {
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
        match self.cancel.take() {
            Some((_, token)) => {
                token.cancel();
                true
            }
            None => false,
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
            Some(&format!("Retry continuation failed locally: {local_error}. Original error: {original}")),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn failed(msg: &str) -> FailedTurn {
        FailedTurn { message: msg.into(), status: None, provider: "p".into(), model: "m".into() }
    }

    fn cfg(chain: &[&str]) -> RecoveryConfig {
        let mut fallback = FallbackConfig::default();
        fallback.chains.push(("p/a".into(), chain.iter().map(|s| (*s).to_string()).collect()));
        fallback.known_providers = ["p", "q", "r"].iter().map(|s| (*s).to_string()).collect();
        RecoveryConfig {
            retry: RetryConfig { enabled: true, max_retries: 2, base_delay_ms: 100, max_delay_ms: 60_000, revert_policy: "cooldown-expiry".into() },
            fallback,
            chain_key: "p/a".into(),
            role_label: Some("default".into()),
        }
    }

    #[test]
    fn retry_after_takes_the_longest_hint() {
        assert_eq!(parse_retry_after_ms("429 rate limited. Retry-After: 3"), Some(3000));
        assert_eq!(parse_retry_after_ms("please try again in 2.5s"), Some(2500));
        assert_eq!(parse_retry_after_ms("retry-after-ms: 1500; resets in 2 minutes"), Some(120_000));
        assert_eq!(parse_retry_after_ms("boom"), None);
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
            Verdict::GiveUp { frame, error_message } => {
                assert_eq!(frame["type"], "auto_retry_end");
                assert_eq!(frame["success"], false);
                assert_eq!(frame["attempt"], 2);
                assert_eq!(error_message.as_deref(), Some("Retry budget exhausted after 2 retries: 503 overloaded"));
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
                assert!(e.starts_with("Provider requested 7200000ms wait, exceeds retry.maxDelayMs (60000ms)"), "{e}");
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
        let kinds: Vec<_> = frames.iter().map(|f| f["type"].as_str().unwrap().to_string()).collect();
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
        let provider = std::sync::Arc::new(ScriptedProvider::new(json_round_trip(&primary).expect("agent model")));
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
        let persist = crate::persist::attach_with(&agent, &cwd, true).await.expect("session");
        let session_file = persist.path();
        let target = std::sync::Arc::new(std::sync::RwLock::new(Some(ProviderTarget {
            api_key: Some("test-key".into()),
            model: primary,
            service_tier: None,
            headers: None,
        })));
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let resolver: Resolver = std::sync::Arc::new(|sel| {
            let base = sel.split(':').next().unwrap_or(sel);
            let (p, i) = base.split_once('/').ok_or_else(|| anyhow::anyhow!("bad selector"))?;
            Ok(resolved(p, i))
        });
        let rt = Runtime::start_with(std::sync::Arc::new(agent), target, cwd, persist, tx, resolver, Some(cfg)).await;
        Harness { rt, rx, models_seen, session_file, _provider: provider }
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
            let f = tokio::time::timeout(Duration::from_secs(30), rx.recv()).await.expect("frame in time").expect("open");
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
            .filter(|k| !matches!(k.as_str(), "message_start" | "message_update" | "message_end" | "turn_start" | "turn_end" | "queue_update"))
            .collect()
    }

    fn session_entries(path: &std::path::Path) -> Vec<Value> {
        crate::runtime::read_entries(path).expect("entries")
    }

    #[tokio::test]
    async fn retry_saga_with_fallback_switch_orders_frames_and_entries() {
        let mut h = harness(&[Err("503 Service Unavailable: overloaded"), Ok("hello from b")], fast_cfg(&["q/b"], 3, 1)).await;
        h.rt.prompt(Some(json!("p1")), "hi", None, Vec::new()).await.expect("prompt");
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
        assert_eq!((start["attempt"].clone(), start["maxAttempts"].clone(), start["delayMs"].clone()), (json!(1), json!(3), json!(0)));
        assert_eq!(get("retry_fallback_succeeded")["model"], "q/b");
        let end = get("auto_retry_end");
        assert_eq!((end["success"].clone(), end["attempt"].clone()), (json!(true), json!(1)));
        let ends: Vec<_> = frames.iter().filter(|f| f["type"] == "agent_end").collect();
        assert_eq!((ends[0]["isTerminal"].clone(), ends[0]["yielded"].clone()), (json!(false), json!(false)));
        assert_eq!(ends[1]["isTerminal"], true);
        assert_eq!(get("prompt_result")["status"], "completed");
        assert_eq!(*h.models_seen.lock().unwrap(), ["a", "b"], "second attempt ran on the fallback model");

        let entries = session_entries(&h.session_file);
        let changes: Vec<_> = entries.iter().filter(|e| e["type"] == "model_change").collect();
        let last = changes.last().unwrap();
        assert_eq!(last["model"], "q/b");
        assert_eq!(last["role"], "fallback");
        assert_eq!(last["resolvedModelIsFallback"], true);
        let assistants: Vec<_> = entries.iter().filter(|e| e["type"] == "message" && e["message"]["role"] == "assistant").collect();
        assert_eq!(assistants.len(), 1, "the failed turn is not persisted: {assistants:?}");
        assert_eq!(assistants[0]["message"]["stopReason"], "stop");
    }

    #[tokio::test]
    async fn retry_on_same_model_backs_off_then_succeeds() {
        let mut h = harness(&[Err("503 overloaded"), Err("503 overloaded"), Ok("finally")], fast_cfg(&[], 5, 5)).await;
        h.rt.prompt(Some(json!("p1")), "hi", None, Vec::new()).await.expect("prompt");
        let frames = until_settled(&mut h.rx).await;
        let k = kinds(&frames);
        assert_eq!(
            k,
            [
                "agent_start", "auto_retry_start", "agent_end", "agent_start", "auto_retry_start", "agent_end",
                "agent_start", "auto_retry_end", "agent_end", "prompt_result", "session_settled"
            ]
        );
        let attempts: Vec<_> = frames.iter().filter(|f| f["type"] == "auto_retry_start").map(|f| f["attempt"].clone()).collect();
        assert_eq!(attempts, [json!(1), json!(2)]);
        assert_eq!(frames.iter().find(|f| f["type"] == "auto_retry_end").unwrap()["attempt"], 2);
        assert_eq!(*h.models_seen.lock().unwrap(), ["a", "a", "a"]);
        assert!(!frames.iter().any(|f| f["type"] == "model_changed"));
        let users: Vec<_> = frames
            .iter()
            .filter(|f| f["type"] == "message_end" && f["message"]["role"] == "user")
            .collect();
        assert_eq!(users[0]["message"]["attribution"], "user", "frame: {}", users[0]);
        let entries = session_entries(&h.session_file);
        let user = entries.iter().find(|e| e["message"]["role"] == "user").expect("user entry");
        assert_eq!(user["message"]["attribution"], "user", "entry: {user}");
        let raw = std::fs::read_to_string(&h.session_file).unwrap();
        let line = raw.lines().find(|l| l.contains("\"role\":\"user\"")).unwrap();
        assert!(line.find("\"content\"").unwrap() < line.find("\"attribution\"").unwrap(), "{line}");
    }

    #[tokio::test]
    async fn exhausted_budget_closes_saga_and_persists_the_failure() {
        let mut h = harness(&[Err("503 overloaded"), Err("503 overloaded")], fast_cfg(&[], 1, 1)).await;
        h.rt.prompt(Some(json!("p1")), "hi", None, Vec::new()).await.expect("prompt");
        let frames = until_settled(&mut h.rx).await;
        assert_eq!(
            kinds(&frames),
            ["agent_start", "auto_retry_start", "agent_end", "agent_start", "auto_retry_end", "agent_end", "prompt_result", "session_settled"]
        );
        let end = frames.iter().find(|f| f["type"] == "auto_retry_end").unwrap();
        assert_eq!((end["success"].clone(), end["attempt"].clone(), end["finalError"].clone()), (json!(false), json!(1), json!("503 overloaded")));
        let result = frames.iter().find(|f| f["type"] == "prompt_result").unwrap();
        assert_eq!(result["status"], "error");
        assert_eq!(result["error"]["message"], "Retry budget exhausted after 1 retry: 503 overloaded");
        tokio::time::timeout(Duration::from_secs(5), h.rt.wait_settled()).await.expect("settled");
        let entries = session_entries(&h.session_file);
        let failed: Vec<_> = entries.iter().filter(|e| e["message"]["stopReason"] == "error").collect();
        assert_eq!(failed.len(), 1, "only the terminal failure is persisted");
        assert_eq!(failed[0]["message"]["errorMessage"], "Retry budget exhausted after 1 retry: 503 overloaded");
    }

    #[tokio::test]
    async fn abort_retry_during_backoff_ends_cancelled() {
        let mut h = harness(&[Err("503 overloaded")], fast_cfg(&[], 5, 60_000)).await;
        h.rt.prompt(Some(json!("p1")), "hi", None, Vec::new()).await.expect("prompt");
        let mut frames = Vec::new();
        loop {
            let f = tokio::time::timeout(Duration::from_secs(30), h.rx.recv()).await.expect("frame").expect("open");
            let start = f["type"] == "auto_retry_start";
            frames.push(f);
            if start {
                break;
            }
        }
        h.rt.abort_retry();
        frames.extend(until_settled(&mut h.rx).await);
        let k = kinds(&frames);
        assert_eq!(k, ["agent_start", "auto_retry_start", "auto_retry_end", "agent_end", "prompt_result", "session_settled"]);
        let end = frames.iter().find(|f| f["type"] == "auto_retry_end").unwrap();
        assert_eq!((end["success"].clone(), end["finalError"].clone()), (json!(false), json!("Retry cancelled")));
        assert_eq!(frames.iter().find(|f| f["type"] == "prompt_result").unwrap()["status"], "error");
    }

    #[tokio::test]
    async fn non_retryable_error_settles_without_a_saga() {
        let mut h = harness(&[Err("401 invalid api key")], fast_cfg(&["q/b"], 3, 1)).await;
        h.rt.prompt(Some(json!("p1")), "hi", None, Vec::new()).await.expect("prompt");
        let frames = until_settled(&mut h.rx).await;
        assert_eq!(kinds(&frames), ["agent_start", "agent_end", "prompt_result", "session_settled"]);
        assert_eq!(frames.iter().find(|f| f["type"] == "agent_end").unwrap()["isTerminal"], true);
        let entries = session_entries(&h.session_file);
        assert_eq!(entries.iter().filter(|e| e["message"]["stopReason"] == "error").count(), 1);
    }

    #[tokio::test]
    async fn abort_reports_the_command_response_before_prompt_result() {
        let mut h = harness_with(|p| p.push_stalled_turn("partial"), fast_cfg(&[], 3, 1)).await;
        h.rt.prompt(Some(json!("p1")), "hi", None, Vec::new()).await.expect("prompt");
        loop {
            let f = tokio::time::timeout(Duration::from_secs(30), h.rx.recv()).await.expect("frame").expect("open");
            if f["type"] == "message_start" && f["message"]["role"] == "assistant" {
                break;
            }
        }
        let reply = crate::rpc::handle(&h.rt, &json!({"id": "a1", "type": "abort"})).await;
        h.rt.send(reply);
        let frames = until_settled(&mut h.rx).await;
        let k = kinds(&frames);
        assert_eq!(k, ["agent_end", "response", "prompt_result", "session_settled"], "{k:?}");
        let result = frames.iter().find(|f| f["type"] == "prompt_result").unwrap();
        assert_eq!((result["status"].clone(), result["id"].clone(), result["sessionSettled"].clone()), (json!("aborted"), json!("p1"), json!(true)));
        let end = frames.iter().find(|f| f["type"] == "agent_end").unwrap();
        assert_eq!(end["isTerminal"], true);
    }

    #[tokio::test]
    async fn user_model_switch_and_thinking_changes_are_journaled_with_omp_shapes() {
        let mut h = harness(&[], fast_cfg(&[], 3, 1)).await;
        h.rt.set_model("q", "b").await.expect("switch");
        h.rt.set_thinking_level(pa_agent::types::ThinkingLevel::High).await;
        h.rt.set_thinking_level(pa_agent::types::ThinkingLevel::High).await; // unchanged: silent
        h.rt.cycle_to("r", "c").await.expect("cycle");
        let mut frames = Vec::new();
        while let Ok(f) = h.rx.try_recv() {
            frames.push(f);
        }
        let k: Vec<_> = frames.iter().map(|f| f["type"].as_str().unwrap().to_string()).collect();
        assert_eq!(k, ["model_changed", "thinking_level_changed", "model_changed"], "{frames:?}");
        assert_eq!(frames[1], json!({"type": "thinking_level_changed", "thinkingLevel": "high"}));
        let entries = session_entries(&h.session_file);
        let shapes: Vec<_> = entries
            .iter()
            .filter(|e| matches!(e["type"].as_str(), Some("model_change" | "thinking_level_change")))
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
}
