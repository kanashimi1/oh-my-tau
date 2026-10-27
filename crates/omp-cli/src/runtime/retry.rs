//! Turn recovery: retry backoff, account rotation on rate limits, and model fallback.

use super::recovery;
use super::recovery::{FailedTurn, Verdict, EPHEMERAL_ROLE};
use crate::adapter::json_round_trip;
use anyhow::Result;
use pa_agent::types::{AgentMessage, Message, StopReason, ThinkingLevel};
use serde_json::{json, Value};
use std::sync::Arc;

use super::{level_name, lock, parse_level, Runtime};

/// Outcome of the retry pass over a failed turn.
enum Recovered {
    /// A backoff elapsed and the run is being continued.
    Retrying,
    /// The saga is over; the turn's error stands (optionally with a rewritten message).
    Final { error: Option<String> },
}

impl Runtime {
    /// Retry saga for a failed turn (`#handleRetryableError`), then continue the run.
    pub(super) async fn recover(
        self: Arc<Self>,
        frame: Value,
        messages: Vec<AgentMessage>,
        failed: FailedTurn,
    ) {
        self.agent.wait_for_idle().await;
        match self.handle_retryable(&failed).await {
            Recovered::Final { error } => self.settle_terminal(frame, &messages, error.as_deref()),
            Recovered::Retrying => {
                let mut nonterminal = frame.clone();
                nonterminal["isTerminal"] = json!(false);
                nonterminal["yielded"] = json!(false);
                self.emit_event(nonterminal);
                let rt = self.clone();
                tokio::spawn(async move {
                    if let Err(e) = rt.continue_after_retry().await {
                        let local = format!("{e:#}");
                        let original = failed.message.clone();
                        let end = lock(&rt.recovery).end_continue_failed(&local, &original);
                        if let Some(end) = end {
                            rt.emit_event(end);
                        }
                        rt.settle_terminal(frame, &messages, None);
                    }
                });
            }
        }
    }

    /// `scheduleAgentContinue`: re-run from the context left after the failed turn was dropped.
    pub(super) async fn continue_after_retry(&self) -> Result<()> {
        for _ in 0..200 {
            match self.agent.continue_run().await {
                Ok(()) => return Ok(()),
                Err(e) => {
                    let busy = e
                        .downcast_ref::<pa_agent::agent::AgentContinueError>()
                        .is_some_and(|c| c.code == pa_agent::agent::AgentContinueErrorCode::Busy);
                    if !busy {
                        return Err(e);
                    }
                    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                }
            }
        }
        anyhow::bail!("agent stayed busy")
    }

    /// `#rotateAccountOnRateLimit`: the turn failed with a provider rate limit and the
    /// provider's credential store holds more than one account — hand the same model to the
    /// next account instead of only backing off.
    ///
    /// Returns `true` when an alternate credential took over (the caller treats the attempt
    /// as switched and skips the model fallback consult for this pass).
    async fn rotate_account_on_rate_limit(&self, failed: &FailedTurn) -> bool {
        if !lock(&self.recovery).is_rate_limit(failed) {
            return false;
        }
        let Some((provider, model_id)) = self.live_provider_id() else {
            return false;
        };
        let selector = recovery::model_selector(&provider, &model_id);
        let now = u64::try_from(pa_agent::now_ms()).unwrap_or(0);
        // The provider-level cooldown the fallback state just recorded decides whether a
        // recently-failed alternate may be retried already.
        let cooldown_active = lock(&self.recovery)
            .state_mut()
            .is_suppressed(&selector, now);
        let pool = match omp_providers::auth::credential_pool_async(
            omp_providers::AgentDir::default(),
            provider.clone(),
        )
        .await
        {
            Some(p) if p.has_alternates() => p,
            _ => return false,
        };
        let Some(next) =
            lock(&self.recovery)
                .accounts_mut()
                .next(&provider, &model_id, &pool, cooldown_active)
        else {
            return false;
        };
        // Same model, other account: swap the live provider target's key and pin the new row.
        let Ok(mut resolved) = (self.resolver)(selector.clone()).await else {
            return false;
        };
        resolved.api_key = Some(next.key.clone());
        resolved.credential_id = Some(next.id);
        let from_id = self.target.read().as_ref().and_then(|t| t.credential_id);
        // Guard against swap_model's silent no-op (unconvertible model): only mutate the
        // rotation state when the swap actually applied.
        let convertible: Option<pa_agent::types::Model> = json_round_trip(&resolved.model);
        if convertible.is_none() {
            return false;
        }
        if !self
            .swap_model(&resolved, Some(recovery::EPHEMERAL_ROLE), true, true)
            .await
        {
            return false;
        }
        {
            let mut rec = lock(&self.recovery);
            rec.accounts_mut().promote(&provider, &model_id, next.id);
            if let Some(from) = from_id {
                rec.accounts_mut().burn(&provider, &model_id, from);
            }
        }
        let from_label = from_id.map_or_else(|| "?".into(), |id| id.to_string());
        self.emit_event(recovery::account_rotated_frame(
            &selector,
            &selector,
            Some(&format!("rate limit: account {from_label} → {}", next.id)),
        ));
        true
    }

    /// Port of `#handleRetryableError`: fallback consult, budget/fail-fast checks, backoff.
    async fn handle_retryable(self: &Arc<Self>, failed: &FailedTurn) -> Recovered {
        // A provider rate limit with a multi-account pool: same model, next account — before
        // any cross-model fallback consult.
        if self.rotate_account_on_rate_limit(failed).await {
            return self.finish_account_rotation(failed).await;
        }
        let level = self.agent.thinking_level().await;
        let current = self.live_selector_with_level(Some(level));
        let now = u64::try_from(pa_agent::now_ms()).unwrap_or(0);
        let (attempt, candidates) = {
            let mut rec = lock(&self.recovery);
            let attempt = rec.begin_attempt(failed, rand::random::<f64>());
            let candidates = rec.consult(&current, &attempt, now, rand::random::<f64>());
            (attempt, candidates)
        };
        let mut switched = false;
        for selector in candidates {
            let Ok(resolved) = (self.resolver)(selector.clone()).await else {
                continue;
            };
            if resolved.api_key.is_none() {
                continue;
            }
            if self
                .apply_fallback(&resolved, &selector, &current, level)
                .await
            {
                switched = true;
                break;
            }
        }
        let verdict = lock(&self.recovery).finish_attempt(&attempt, switched);
        let delay_ms = match verdict {
            Verdict::GiveUp {
                frame,
                error_message,
            } => {
                self.emit_event(frame);
                return Recovered::Final {
                    error: error_message,
                };
            }
            Verdict::Retry { delay_ms } => delay_ms,
        };
        let announced = lock(&self.recovery).attempt();
        let max = attempt.max;
        self.emit_event(recovery::auto_retry_start_frame(
            announced,
            max,
            delay_ms,
            &attempt.error_message,
        ));
        // The failed turn is superseded by the retry: out of the file and out of the context.
        self.persist.drop_deferred_error();
        self.agent
            .mutate_messages(|m| {
                if matches!(m.last(), Some(AgentMessage::Standard(Message::Assistant(a))) if matches!(a.stop_reason, StopReason::Error | StopReason::Aborted)) {
                    m.pop();
                }
            })
            .await;
        let (generation, token) = lock(&self.recovery).arm_cancel();
        tokio::select! {
            () = tokio::time::sleep(std::time::Duration::from_millis(delay_ms)) => {
                lock(&self.recovery).disarm(generation);
                Recovered::Retrying
            }
            () = token.cancelled() => {
                let mut rec = lock(&self.recovery);
                if rec.superseded(generation) {
                    return Recovered::Final { error: None };
                }
                let end = rec.end_cancelled();
                drop(rec);
                self.emit_event(end);
                self.persist.flush_deferred_error(None);
                Recovered::Final { error: None }
            }
        }
    }

    /// Finalize a rate-limit retry pass whose switch was an account rotation: announce the
    /// backoff, drop the failed turn, and park the saga in the cancelable sleep — the same
    /// epilogue a model fallback has, without the fallback consult.
    async fn finish_account_rotation(&self, failed: &FailedTurn) -> Recovered {
        let (attempt, verdict) = {
            let mut rec = lock(&self.recovery);
            let attempt = rec.begin_attempt(failed, rand::random::<f64>());
            let verdict = rec.finish_attempt(&attempt, true);
            (attempt, verdict)
        };
        let delay_ms = match verdict {
            Verdict::GiveUp {
                frame,
                error_message,
            } => {
                self.emit_event(frame);
                return Recovered::Final {
                    error: error_message,
                };
            }
            Verdict::Retry { delay_ms } => delay_ms,
        };
        let announced = lock(&self.recovery).attempt();
        let max = attempt.max;
        self.emit_event(recovery::auto_retry_start_frame(
            announced,
            max,
            delay_ms,
            &attempt.error_message,
        ));
        self.persist.drop_deferred_error();
        self.agent
            .mutate_messages(|m| {
                if matches!(m.last(), Some(AgentMessage::Standard(Message::Assistant(a))) if matches!(a.stop_reason, StopReason::Error | StopReason::Aborted)) {
                    m.pop();
                }
            })
            .await;
        let (generation, token) = lock(&self.recovery).arm_cancel();
        tokio::select! {
            () = tokio::time::sleep(std::time::Duration::from_millis(delay_ms)) => {
                lock(&self.recovery).disarm(generation);
                Recovered::Retrying
            }
            () = token.cancelled() => {
                let mut rec = lock(&self.recovery);
                if rec.superseded(generation) {
                    return Recovered::Final { error: None };
                }
                let end = rec.end_cancelled();
                drop(rec);
                self.emit_event(end);
                self.persist.flush_deferred_error(None);
                Recovered::Final { error: None }
            }
        }
    }

    /// `#applyRetryFallbackCandidate`: swap to `selector`, journal it, announce it.
    async fn apply_fallback(
        &self,
        resolved: &omp_providers::ResolvedModel,
        selector: &str,
        current: &str,
        current_level: ThinkingLevel,
    ) -> bool {
        let requested = selector
            .rsplit_once(':')
            .and_then(|(_, tail)| parse_level(tail))
            .unwrap_or(current_level);
        let from_thinking = level_name(current_level).to_string();
        self.swap_model(resolved, Some(EPHEMERAL_ROLE), true, true)
            .await;
        let applied = self.apply_thinking(requested).await;
        let applied_name = level_name(applied).to_string();
        let role = {
            let mut rec = lock(&self.recovery);
            rec.commit_switch(current, Some(from_thinking), Some(applied_name));
            rec.role().to_string()
        };
        self.emit_event(recovery::fallback_applied_frame(
            current, selector, &role, None,
        ));
        true
    }

    /// Before a prompt: return to the primary model once its cooldown expired (`maybeRestoreRetryFallbackPrimary`).
    pub(super) async fn maybe_restore_primary(&self) {
        let level = self.agent.thinking_level().await;
        let current = self.live_selector_with_level(Some(level));
        let now = u64::try_from(pa_agent::now_ms()).unwrap_or(0);
        let plan =
            lock(&self.recovery)
                .state_mut()
                .restore_plan(&current, Some(level_name(level)), now);
        let Some(plan) = plan else { return };
        let Ok(resolved) = (self.resolver)(plan.selector.clone()).await else {
            return;
        };
        if resolved.api_key.is_none() {
            return;
        }
        let restored_level = plan.thinking.as_deref().and_then(parse_level);
        // Ownership ends before the swap so observers never see the primary tagged as a fallback.
        lock(&self.recovery).state_mut().clear_active();
        self.swap_model(&resolved, Some(EPHEMERAL_ROLE), false, true)
            .await;
        if let Some(l) = restored_level {
            self.apply_thinking(l).await;
        }
    }

    /// Cancel a pending retry backoff and wait for its saga task to finish reporting, so nothing
    /// of the old turn can land after (or inside) whatever starts next.
    pub(super) async fn cancel_retry_and_wait(&self) {
        self.abort_retry();
        let handle = lock(&self.retry_task).take();
        if let Some(h) = handle {
            let _ = h.await;
        }
    }

    /// `set_auto_retry`: toggle the retry engine.
    pub fn set_auto_retry(&self, enabled: bool) {
        lock(&self.recovery).set_enabled(enabled);
    }

    /// `abort_retry`: cancel a pending backoff; the failed turn then stands.
    pub fn abort_retry(&self) {
        lock(&self.recovery).abort_retry();
    }
}

#[cfg(test)]
#[allow(clippy::disallowed_methods)] // tests: throwaway runtimes drive async code
mod turn_outcome_tests {
    use super::*;
    use crate::runtime::prompt::run_outcome;
    use serde_json::json;

    fn failed(error_message: &str, error_status: Option<u16>) -> AgentMessage {
        let mut value = json!({
            "content": [], "api": "openai-completions", "provider": "openai", "model": "gpt-5.5",
            "usage": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 0,
                      "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0}},
            "stopReason": "error", "errorMessage": error_message, "timestamp": 1
        });
        if let Some(status) = error_status {
            value["errorStatus"] = json!(status);
        }
        let a: pa_agent::types::AssistantMessage =
            serde_json::from_value(value).expect("assistant json");
        AgentMessage::Standard(Message::Assistant(a))
    }

    #[test]
    fn rate_limit_reports_http_status_and_is_retryable() {
        let message = failed(
            "429 Rate limit reached\nraw-http-request=/home/u/.omp/logs/http-400-requests/1.json",
            Some(429),
        );
        let outcome = run_outcome(&[message]);
        assert_eq!(outcome.status, "error");
        let error = outcome.error.expect("error payload");
        assert_eq!(error["httpStatus"], json!(429));
        assert_eq!(error["retryable"], json!(true));
        assert_eq!(error["message"], json!("429 Rate limit reached"));
        assert_eq!(error["provider"], json!("openai"));
    }

    #[test]
    fn bad_request_reports_http_status_and_is_not_retryable() {
        let outcome = run_outcome(&[failed("400 Invalid value: 'x'.", Some(400))]);
        let error = outcome.error.expect("error payload");
        assert_eq!(error["httpStatus"], json!(400));
        assert_eq!(error["retryable"], json!(false));
    }

    #[test]
    fn error_without_status_omits_http_status() {
        let outcome = run_outcome(&[failed("", None)]);
        let error = outcome.error.expect("error payload");
        assert!(error.get("httpStatus").is_none());
        assert_eq!(error["message"], json!("Provider request failed"));
    }
}
