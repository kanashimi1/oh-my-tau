// Derived from prime-agent (MIT) crates/pa-core/src/session_engine/provider_adapter.rs.
#![allow(dead_code, clippy::pedantic)]
//! Real-provider stream adapter: pa-ai completion streaming bridged into
//! the pa-agent loop's `StreamFn`/`ModelStream`, crossing the crate
//! boundary by wire-shape (JSON) round-trip; shared by pa-cli and
//! pa-daemon.

use std::sync::Arc;

use pa_agent::stream::{LlmContext, ModelStream, StreamFn, StreamRequestOptions};
use pa_agent::types::{Model as AgentModel, ThinkingLevel};
use pa_types::ai::Model;

/// Wire-shape conversion at the pa-agent/pa-ai boundary: both sides
/// serialize to the same camelCase wire shapes.
pub fn json_round_trip<T, U>(value: &T) -> Option<U>
where
    T: serde::Serialize,
    U: serde::de::DeserializeOwned,
{
    serde_json::to_value(value)
        .ok()
        .and_then(|value| serde_json::from_value(value).ok())
}

/// Thinking-level mapping across the two crates.
#[must_use]
pub fn map_thinking_level(level: pa_types::ai::ModelThinkingLevel) -> ThinkingLevel {
    match level {
        pa_types::ai::ModelThinkingLevel::Off => ThinkingLevel::Off,
        pa_types::ai::ModelThinkingLevel::Minimal => ThinkingLevel::Minimal,
        pa_types::ai::ModelThinkingLevel::Low => ThinkingLevel::Low,
        pa_types::ai::ModelThinkingLevel::Medium => ThinkingLevel::Medium,
        pa_types::ai::ModelThinkingLevel::High => ThinkingLevel::High,
        pa_types::ai::ModelThinkingLevel::Xhigh => ThinkingLevel::Xhigh,
        pa_types::ai::ModelThinkingLevel::Max => ThinkingLevel::Max,
    }
}

/// The inverse of [`map_thinking_level`]: the pa-types view of the agent
/// state's thinking level.
#[must_use]
pub fn model_thinking_level(level: ThinkingLevel) -> pa_types::ai::ModelThinkingLevel {
    match level {
        ThinkingLevel::Off => pa_types::ai::ModelThinkingLevel::Off,
        ThinkingLevel::Minimal => pa_types::ai::ModelThinkingLevel::Minimal,
        ThinkingLevel::Low => pa_types::ai::ModelThinkingLevel::Low,
        ThinkingLevel::Medium => pa_types::ai::ModelThinkingLevel::Medium,
        ThinkingLevel::High => pa_types::ai::ModelThinkingLevel::High,
        ThinkingLevel::Xhigh => pa_types::ai::ModelThinkingLevel::Xhigh,
        ThinkingLevel::Max => pa_types::ai::ModelThinkingLevel::Max,
    }
}

/// Adapt the loop-level payload hook to the pa-ai hook shape. A failed
/// round-trip keeps the payload unchanged — hooks never fail the request.
fn agent_payload_hook_to_ai(hook: pa_agent::stream::OnPayloadHook) -> pa_ai::types::OnPayloadHook {
    std::sync::Arc::new(move |payload: serde_json::Value, model: &Model| {
        match json_round_trip::<_, pa_agent::types::Model>(model) {
            Some(agent_model) => hook(payload, &agent_model),
            None => Some(payload),
        }
    })
}

/// Adapt the loop-level response hook to the pa-ai hook shape; a failed
/// round-trip drops the hook call (advisory, never fatal).
fn agent_response_hook_to_ai(
    hook: pa_agent::stream::OnResponseHook,
) -> pa_ai::types::OnResponseHook {
    std::sync::Arc::new(
        move |response: pa_ai::types::ProviderResponse, model: &Model| {
            let agent_response = pa_agent::stream::ProviderResponse {
                status: response.status,
                headers: response.headers,
            };
            if let Some(agent_model) = json_round_trip::<_, pa_agent::types::Model>(model) {
                hook(agent_response, &agent_model);
            }
        },
    )
}

/// The mutable provider target a live session's stream reads per call
/// (daemon `set_model` and the failover switch swap it).
#[derive(Debug, Clone)]
pub struct ProviderTarget {
    pub api_key: Option<String>,
    pub model: Model,
    pub service_tier: Option<pa_types::ai::ServiceTier>,
    /// The auth-resolved provider-request headers: a live `set_model` carries
    /// them through to the stream like the build-time resolution does.
    pub headers: Option<std::collections::BTreeMap<String, String>>,
    /// The stored `auth_credentials` row `api_key` came from (`None` for config/env/`--api-key`
    /// keys). Such a key is re-resolved through the credential store on every request (so an
    /// expired OAuth token is refreshed) and refreshed + retried once after a provider 401.
    pub credential_id: Option<i64>,
}

/// The current key for a stored-credential target (see [`ProviderTarget::credential_id`]): the
/// credential store's bearer, refreshed when expired; the last-known key when the lookup fails.
pub async fn fresh_stored_key(
    provider: &str,
    credential_id: Option<i64>,
    api_key: Option<String>,
) -> (Option<String>, Option<i64>) {
    if credential_id.is_none() {
        return (api_key, None);
    }
    match omp_providers::auth::credential_async(
        omp_providers::AgentDir::default(),
        provider.to_string(),
    )
    .await
    {
        Some(c) => (Some(request_key(provider, c.key, c.id)), Some(c.id)),
        None => (api_key, credential_id),
    }
}

/// The request key for a stored credential: the Cloud Code Assist providers (`google-gemini-cli`,
/// `google-antigravity`) need the structured `{token, projectId, ...}` payload (TS
/// `getOAuthApiKey`), not the bare bearer — with a bare token the envelope carries `project: ""`.
fn request_key(provider: &str, bearer: String, id: i64) -> String {
    omp_providers::auth::structured_key_by_id(&omp_providers::AgentDir::default(), provider, id)
        .unwrap_or(bearer)
}

/// Whether a failed assistant message is a provider authentication failure (HTTP 401).
fn is_auth_failure(error: &pa_agent::types::AssistantMessage) -> bool {
    error.error_status == Some(401)
        || error
            .error_message
            .as_deref()
            .is_some_and(|m| m.contains("401") && m.to_ascii_lowercase().contains("authentication"))
}

type Reopen = Box<dyn FnOnce() -> pa_agent::BoxFut<'static, Option<Box<dyn ModelStream>>> + Send>;

/// TS `auth-retry`: a stream whose first failure is a provider 401, before any content, is
/// re-opened once with a force-refreshed credential (`reopen` yields `None` when there is no
/// different bearer to try); the failed attempt's events are replaced, not shown.
struct AuthRetryStream {
    inner: Box<dyn ModelStream>,
    reopen: Option<Reopen>,
    start_forwarded: bool,
}

impl ModelStream for AuthRetryStream {
    fn next_event(
        &mut self,
    ) -> pa_agent::BoxFut<'_, Option<pa_agent::stream::AssistantMessageEvent>> {
        use pa_agent::stream::AssistantMessageEvent as E;
        Box::pin(async move {
            loop {
                let event = self.inner.next_event().await;
                match &event {
                    Some(E::Start { .. }) if !self.start_forwarded => self.start_forwarded = true,
                    // The replacement attempt re-emits `start`; the caller already saw one.
                    Some(E::Start { .. }) => continue,
                    Some(E::Error { error, .. })
                        if self.reopen.is_some() && is_auth_failure(error) =>
                    {
                        if let Some(reopen) = self.reopen.take() {
                            if let Some(next) = reopen().await {
                                self.inner = next;
                                continue;
                            }
                        }
                    }
                    // Content arrived: the attempt is committed, drop the retry inputs.
                    Some(_) => self.reopen = None,
                    None => {}
                }
                return event;
            }
        })
    }

    fn result(
        &mut self,
    ) -> pa_agent::BoxFut<'_, anyhow::Result<pa_agent::types::AssistantMessage>> {
        self.inner.result()
    }

    fn close(&mut self) {
        self.inner.close();
    }
}

/// A real pa-ai stream adapter reading its target from a shared slot the host can swap live
/// (`set_model`, failover); `None` only before the host sets the build-time target.
///
/// # Panics
///
/// Panics at stream time if the provider target lock is poisoned, or if the
/// target slot was never set before the first stream.
pub fn switchable_stream_fn(target: Arc<std::sync::RwLock<Option<ProviderTarget>>>) -> StreamFn {
    Arc::new(
        move |_requested: AgentModel, context: LlmContext, options: StreamRequestOptions| {
            let ProviderTarget {
                api_key,
                model,
                service_tier,
                headers,
                credential_id,
            } = target
                .read()
                .expect("provider target lock")
                .clone()
                .expect("provider target set before the first stream");
            Box::pin(async move {
                if credential_id.is_none() {
                    return stream_once(&model, api_key, service_tier, headers, context, options);
                }
                let (api_key, credential_id) =
                    fresh_stored_key(&model.provider, credential_id, api_key).await;
                let (Some(key), Some(id)) = (api_key.clone(), credential_id) else {
                    return stream_once(&model, api_key, service_tier, headers, context, options);
                };
                let retry_inputs = (context.clone(), options.clone());
                let stream = stream_once(
                    &model,
                    api_key,
                    service_tier,
                    headers.clone(),
                    context,
                    options,
                )?;
                let reopen: Reopen = Box::new(move || {
                    Box::pin(async move {
                        let failed = omp_providers::auth::Credential { key, id };
                        let next = omp_providers::auth::recover_async(
                            omp_providers::AgentDir::default(),
                            model.provider.clone(),
                            failed,
                        )
                        .await?;
                        let (context, options) = retry_inputs;
                        stream_once(
                            &model,
                            Some(request_key(&model.provider, next.key, next.id)),
                            service_tier,
                            headers,
                            context,
                            options,
                        )
                        .ok()
                    })
                });
                Ok(Box::new(AuthRetryStream {
                    inner: stream,
                    reopen: Some(reopen),
                    start_forwarded: false,
                }) as Box<dyn ModelStream>)
            })
        },
    )
}

/// The reasoning level sent with a request. `off` on a model that always thinks (catalog
/// `thinking.requiresEffort`, e.g. zai glm) goes out as that model's lowest effort instead of a
/// disable toggle the provider rejects.
fn reasoning_for_request(
    model: &Model,
    level: ThinkingLevel,
) -> Option<pa_types::ai::ModelThinkingLevel> {
    if level == ThinkingLevel::Off {
        if let Some(lowest) = crate::model::always_thinking_floor(&model.provider, &model.id) {
            return Some(model_thinking_level(lowest));
        }
    }
    Some(model_thinking_level(level))
}

/// GLM-5.x on a z.ai host (`thinkingFormat: "zai"` + `zaiReasoningEffortDialect`) wants
/// `thinking: {type: "enabled"}` and `reasoning_effort` on the wire; pa-ai's zai arm only sends
/// `enable_thinking`, which z.ai answers without reasoning. Patch the payload here.
fn zai_effort_hook(
    model: &Model,
    effort: Option<pa_types::ai::ModelThinkingLevel>,
) -> Option<pa_ai::types::OnPayloadHook> {
    let effort = effort.filter(|e| *e != pa_types::ai::ModelThinkingLevel::Off)?;
    let compat = serde_json::to_value(model).ok()?.get("compat")?.clone();
    let flag = |k: &str| compat.get(k) == Some(&serde_json::Value::Bool(true));
    if compat
        .get("thinkingFormat")
        .and_then(serde_json::Value::as_str)
        != Some("zai")
        || !flag("zaiReasoningEffortDialect")
        || !flag("supportsReasoningEffort")
    {
        return None;
    }
    let wire = effort.wire_name().to_string();
    Some(Arc::new(
        move |mut payload: serde_json::Value, _m: &Model| {
            if let Some(o) = payload.as_object_mut() {
                o.insert("thinking".into(), serde_json::json!({"type": "enabled"}));
                o.insert("reasoning_effort".into(), serde_json::json!(wire));
            }
            Some(payload)
        },
    ))
}

fn chain_payload_hooks(
    first: Option<pa_ai::types::OnPayloadHook>,
    second: Option<pa_ai::types::OnPayloadHook>,
) -> Option<pa_ai::types::OnPayloadHook> {
    match (first, second) {
        (Some(a), Some(b)) => Some(Arc::new(move |p: serde_json::Value, m: &Model| {
            b(a(p, m)?, m)
        })),
        (a, b) => a.or(b),
    }
}

/// Stream one completion against `model` with `api_key` and the auth-resolved
/// request `headers`: the per-request tail the switchable seams and the CLI's
/// route-authoritative variant share.
///
/// # Errors
///
/// Returns the provider stream's error when the request fails (the
/// per-attempt failures the retry driver classifies).
pub fn stream_once(
    model: &Model,
    api_key: Option<String>,
    service_tier: Option<pa_types::ai::ServiceTier>,
    headers: Option<std::collections::BTreeMap<String, String>>,
    mut context: LlmContext,
    options: StreamRequestOptions,
) -> anyhow::Result<Box<dyn ModelStream>> {
    // Messages cross the crate boundary one at a time, consuming the owned
    // context as they go: only one message's `Value` tree is alive at once
    // and each source message is freed once converted, so a large resumed
    // context never holds the whole-vector `Value` DOM next to both typed
    // copies. Same result as a whole-vector round trip with a per-item
    // fallback: an item that cannot cross is skipped either way.
    let mut messages: Vec<pa_types::ai::Message> = Vec::with_capacity(context.messages.len());
    messages.extend(
        std::mem::take(&mut context.messages)
            .into_iter()
            .filter_map(|m| json_round_trip(&m)),
    );
    let tools: Vec<pa_types::ai::Tool> = json_round_trip(&context.tools)
        .unwrap_or_else(|| context.tools.iter().filter_map(json_round_trip).collect());
    let ai_context = pa_types::ai::Context {
        system_prompt: context.system_prompt,
        messages,
        tools: Some(tools),
    };
    // The turn's abort signal reaches the transport: the in-flight request races this
    // token; the loop's abort paths fire it through [`ModelStream::close`] or the drop.
    let cancel = tokio_util::sync::CancellationToken::new();
    let stream_options = pa_ai::types::SimpleStreamOptions {
        base: pa_ai::types::StreamOptions {
            temperature: options.temperature,
            max_tokens: options.max_tokens,
            signal: Some(cancel.clone()),
            api_key,
            transport: None,
            service_tier,
            cache_retention: None,
            session_id: options.session_id.clone(),
            // The loop-level request hooks cross the crate boundary here: the payload
            // hook may replace the wire payload, the response hook observes headers.
            on_payload: chain_payload_hooks(
                zai_effort_hook(model, reasoning_for_request(model, options.reasoning)),
                options.on_payload.map(agent_payload_hook_to_ai),
            ),
            on_response: options.on_response.map(agent_response_hook_to_ai),
            // StreamOptions carries a plain map; the target's ordered
            // (BTreeMap) resolution converts here, with the request's own
            // headers (the semantic request id) merged over it — TS
            // providers' `mergeHeaders(..., optionsHeaders)` order, the
            // options win.
            headers: {
                let mut merged = headers;
                if let Some(request_headers) = options.headers.clone() {
                    merged
                        .get_or_insert_with(std::collections::BTreeMap::new)
                        .extend(request_headers);
                }
                merged.map(|headers| headers.into_iter().collect())
            },
            metadata: None,
            timeout_ms: None,
        },
        reasoning: reasoning_for_request(model, options.reasoning),
        thinking_budgets: None,
        disable_reasoning: false,
        tool_choice: None,
    };
    let stream = pa_ai::stream_simple(model, &ai_context, Some(stream_options))
        .map_err(|error| anyhow::anyhow!("{error:?}"))?;
    // Pump pa-ai events into the loop's event stream, verbatim.
    let (handle, consumer) = pa_agent::stream::event_stream();
    let forwarder = tokio::spawn(async move {
        let mut stream = stream;
        while let Some(event) = stream.next_event().await {
            if let Some(converted) = convert_stream_event(&event) {
                handle.push(converted);
            }
        }
        let result = stream.result().await;
        if let Some(converted) = json_round_trip::<_, pa_agent::types::AssistantMessage>(&result) {
            handle.end(Some(converted));
        } else {
            handle.end(None);
        }
    });
    // Keep the pump task alive as long as the stream lives.
    let (forwarder, consumer) = (forwarder, consumer);
    Ok(consumer_pump(forwarder, consumer, cancel))
}

/// A stream adapter pinned to one target (headless runtimes resolve
/// their model once, so the slot never changes).
#[must_use]
pub fn real_stream_fn(api_key: Option<String>, model: Model) -> StreamFn {
    switchable_stream_fn(Arc::new(std::sync::RwLock::new(Some(ProviderTarget {
        api_key,
        model,
        service_tier: None,
        headers: None,
        credential_id: None,
    }))))
}

/// Convert one pa-ai stream event into the pa-agent loop's event enum.
/// Payloads cross the boundary by wire-shape (JSON) round-trip.
///
/// # Panics
///
/// Panics when an assistant message cannot round-trip across the two
/// crates' wire shapes (a structural shape-mismatch bug).
#[must_use]
pub fn convert_stream_event(
    event: &pa_types::ai::AssistantMessageEvent,
) -> Option<pa_agent::stream::AssistantMessageEvent> {
    use pa_agent::stream::AssistantMessageEvent as Out;
    use pa_types::ai::AssistantMessageEvent as In;
    fn convert_partial(
        message: &pa_types::ai::AssistantMessage,
    ) -> pa_agent::types::AssistantMessage {
        json_round_trip(message).expect("assistant wire shapes match")
    }
    Some(match event {
        In::Start { partial } => Out::Start {
            partial: convert_partial(partial),
        },
        In::TextStart {
            content_index,
            partial,
        } => Out::TextStart {
            content_index: *content_index as usize,
            partial: convert_partial(partial),
        },
        In::TextDelta {
            content_index,
            delta,
            partial,
        } => Out::TextDelta {
            content_index: *content_index as usize,
            delta: delta.clone(),
            partial: convert_partial(partial),
        },
        In::TextEnd {
            content_index,
            content,
            partial,
        } => Out::TextEnd {
            content_index: *content_index as usize,
            content: content.clone(),
            partial: convert_partial(partial),
        },
        In::ThinkingStart {
            content_index,
            partial,
        } => Out::ThinkingStart {
            content_index: *content_index as usize,
            partial: convert_partial(partial),
        },
        In::ThinkingDelta {
            content_index,
            delta,
            partial,
        } => Out::ThinkingDelta {
            content_index: *content_index as usize,
            delta: delta.clone(),
            partial: convert_partial(partial),
        },
        In::ThinkingEnd {
            content_index,
            partial,
            ..
        } => Out::ThinkingEnd {
            content_index: *content_index as usize,
            partial: convert_partial(partial),
        },
        In::ToolcallStart {
            content_index,
            partial,
        } => Out::ToolCallStart {
            content_index: *content_index as usize,
            partial: convert_partial(partial),
        },
        In::ToolcallDelta {
            content_index,
            delta,
            partial,
        } => Out::ToolCallDelta {
            content_index: *content_index as usize,
            delta: delta.clone(),
            partial: convert_partial(partial),
        },
        In::ToolcallEnd {
            content_index,
            tool_call,
            partial,
        } => Out::ToolCallEnd {
            content_index: *content_index as usize,
            tool_call: json_round_trip(tool_call).expect("tool call wire shapes match"),
            partial: convert_partial(partial),
        },
        In::Done { reason, message } => Out::Done {
            reason: json_round_trip(reason).expect("stop reason wire shapes match"),
            message: convert_partial(message),
        },
        In::Error { reason, error } => Out::Error {
            reason: json_round_trip(reason).expect("stop reason wire shapes match"),
            error: convert_partial(error),
        },
    })
}

/// Wrap the consumer so the pump task is aborted when the stream drops.
fn consumer_pump(
    forwarder: tokio::task::JoinHandle<()>,
    consumer: pa_agent::stream::AssistantMessageEventStream,
    cancel: tokio_util::sync::CancellationToken,
) -> Box<dyn ModelStream> {
    Box::new(PumpedStream {
        _forwarder: forwarder,
        stream: consumer,
        cancel,
    })
}

/// A `ModelStream` whose lifetime keeps the pa-ai pump task alive and owns
/// the fetch's cancellation token (the transport half of the turn-abort).
struct PumpedStream {
    _forwarder: tokio::task::JoinHandle<()>,
    stream: pa_agent::stream::AssistantMessageEventStream,
    cancel: tokio_util::sync::CancellationToken,
}

impl Drop for PumpedStream {
    fn drop(&mut self) {
        // The in-flight fetch cancels instead of running to completion
        // detached (TS: the fetch dies with its iterator).
        self.cancel.cancel();
    }
}

impl ModelStream for PumpedStream {
    fn next_event(
        &mut self,
    ) -> pa_agent::BoxFut<'_, Option<pa_agent::stream::AssistantMessageEvent>> {
        self.stream.next_event()
    }

    fn result(
        &mut self,
    ) -> pa_agent::BoxFut<'_, anyhow::Result<pa_agent::types::AssistantMessage>> {
        self.stream.result()
    }

    /// Close/cancel: the in-flight fetch cancels immediately. Idempotent
    /// — the token's cancelled state is sticky.
    fn close(&mut self) {
        self.cancel.cancel();
    }
}

#[cfg(test)]
mod auth_retry_tests {
    use super::*;
    use pa_agent::stream::{event_stream, AssistantMessageEvent as E};
    use pa_agent::types::{AssistantMessage as Msg, StopReason};

    fn message(text: &str, stop: &str, status: Option<u16>) -> Msg {
        serde_json::from_value(serde_json::json!({
            "content": [{"type": "text", "text": text}], "api": "openai-completions", "provider": "p", "model": "m",
            "usage": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 0,
                      "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0}},
            "stopReason": stop, "timestamp": 1, "errorStatus": status,
            "errorMessage": status.map(|s| format!("Provider authentication failed (invalid_authentication_error, {s})")),
        }))
        .expect("assistant json")
    }

    /// A scripted stream: `start`, then either a 401 failure or a text answer.
    fn scripted(fail: bool) -> Box<dyn ModelStream> {
        let (handle, consumer) = event_stream();
        handle.push(E::Start {
            partial: message("", "stop", None),
        });
        if fail {
            let error = message("", "error", Some(401));
            handle.push(E::Error {
                reason: StopReason::Error,
                error: error.clone(),
            });
            handle.end(Some(error));
        } else {
            let done = message("hello", "stop", None);
            handle.push(E::Done {
                reason: StopReason::Stop,
                message: done.clone(),
            });
            handle.end(Some(done));
        }
        Box::new(consumer_wrapper(consumer))
    }

    fn consumer_wrapper(stream: pa_agent::stream::AssistantMessageEventStream) -> impl ModelStream {
        struct W(pa_agent::stream::AssistantMessageEventStream);
        impl ModelStream for W {
            fn next_event(&mut self) -> pa_agent::BoxFut<'_, Option<E>> {
                self.0.next_event()
            }
            fn result(&mut self) -> pa_agent::BoxFut<'_, anyhow::Result<Msg>> {
                self.0.result()
            }
        }
        W(stream)
    }

    async fn drain(stream: &mut dyn ModelStream) -> Vec<&'static str> {
        let mut kinds = Vec::new();
        while let Some(event) = stream.next_event().await {
            kinds.push(match event {
                E::Start { .. } => "start",
                E::Done { .. } => "done",
                E::Error { .. } => "error",
                _ => "other",
            });
        }
        kinds
    }

    fn retrying(
        first: Box<dyn ModelStream>,
        reopen: Option<Box<dyn ModelStream>>,
    ) -> (
        AuthRetryStream,
        std::sync::Arc<std::sync::atomic::AtomicUsize>,
    ) {
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = calls.clone();
        let reopen: Reopen = Box::new(move || {
            counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Box::pin(async move { reopen })
        });
        (
            AuthRetryStream {
                inner: first,
                reopen: Some(reopen),
                start_forwarded: false,
            },
            calls,
        )
    }

    #[tokio::test]
    async fn a_401_before_content_reopens_once_and_hides_the_failed_attempt() {
        let (mut stream, calls) = retrying(scripted(true), Some(scripted(false)));
        assert_eq!(drain(&mut stream).await, ["start", "done"]);
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert_eq!(stream.result().await.unwrap().stop_reason, StopReason::Stop);
    }

    #[tokio::test]
    async fn a_second_401_is_surfaced_not_retried_again() {
        let (mut stream, calls) = retrying(scripted(true), Some(scripted(true)));
        assert_eq!(drain(&mut stream).await, ["start", "error"]);
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn no_different_bearer_surfaces_the_original_401() {
        let (mut stream, calls) = retrying(scripted(true), None);
        assert_eq!(drain(&mut stream).await, ["start", "error"]);
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert_eq!(stream.result().await.unwrap().error_status, Some(401));
    }

    #[tokio::test]
    async fn success_never_reopens() {
        let (mut stream, calls) = retrying(scripted(false), Some(scripted(false)));
        assert_eq!(drain(&mut stream).await, ["start", "done"]);
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 0);
    }
}
