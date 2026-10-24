//! Streaming one assistant response: context transform, LLM-bound message
//! conversion, the model stream event loop, and the aborted-message
//! finalize path.

use std::sync::Arc;

use crate::abort::{is_abort_error, AbortSignal};
use crate::stream::{LlmContext, StreamFn, StreamRequestOptions, ToolDefinition};
use crate::types::{
    AgentContext, AgentEvent, AgentMessage, AssistantMessage, ContextSnapshot, StopReason,
};

use super::abort::{create_aborted_assistant_message, race_with_abort};
use super::tools::Preparation;
use super::{AgentEventSink, AgentLoopConfig};

/// Provider-bound messages for one request (TS `prepareProviderCall`): `transformContext`
/// (the extension `context` event) then `convertToLlm`. The loop runs it before opening the turn.
pub(crate) async fn prepare_llm_messages(
    context: &AgentContext,
    config: &AgentLoopConfig,
    signal: Option<&AbortSignal>,
) -> anyhow::Result<Vec<crate::types::Message>> {
    crate::abort::throw_if_aborted_signal(signal)?;
    // The hooks own their input: one materialization per request, which
    // `convert_to_llm` consumes into the provider messages.
    let mut messages: Vec<AgentMessage> = context.messages.to_vec();
    if let Some(transform) = config.transform_context.as_ref() {
        messages = race_with_abort(
            transform(messages, signal.cloned().unwrap_or_default()),
            signal,
        )
        .await?;
    }
    race_with_abort((config.convert_to_llm)(messages), signal).await
}

/// TS `prepareToolCallDispatch`: once the final assistant message is known (before its
/// `message_end`), every tool call is validated and runs its `beforeToolCall` hook (the extension
/// `tool_call` event). A hook argument revision is baked into the message so every consumer
/// snapshots a single version of the call.
async fn prepare_dispatch(
    message: &mut AssistantMessage,
    context: &mut AgentContext,
    config: &AgentLoopConfig,
    signal: Option<&AbortSignal>,
) -> Vec<Preparation> {
    if matches!(message.stop_reason, StopReason::Error | StopReason::Aborted) {
        return Vec::new();
    }
    let calls: Vec<crate::types::ToolCall> = message.tool_calls().into_iter().cloned().collect();
    if calls.is_empty() {
        return Vec::new();
    }
    // Hooks and tool executions each snapshot the context: freeze first so they share the history.
    context.messages.freeze();
    let for_hooks = for_turn_end(message.clone());
    let mut prepared = Vec::with_capacity(calls.len());
    for call in &calls {
        prepared.push(
            super::tool_call::prepare_tool_call(context, &for_hooks, call, config, signal).await,
        );
    }
    let finalized: Vec<crate::types::ToolCall> = calls
        .iter()
        .zip(&prepared)
        .map(|(call, p)| match p {
            Preparation::Prepared(p) => p.tool_call.clone(),
            Preparation::Immediate { .. } => call.clone(),
        })
        .collect();
    if super::run::bake_revised_arguments(message, &finalized) {
        if let Some(last) = context.messages.last_mut() {
            *last = AgentMessage::from(message.clone());
        }
    }
    prepared
}

pub(crate) async fn stream_assistant_response(
    context: &mut AgentContext,
    config: &AgentLoopConfig,
    signal: Option<&AbortSignal>,
    emit: &AgentEventSink,
    stream_fn: Option<&StreamFn>,
    llm_messages: anyhow::Result<Vec<crate::types::Message>>,
    prepared_calls: &mut Vec<Preparation>,
) -> anyhow::Result<AssistantMessage> {
    let mut partial_event: Option<Arc<crate::stream::AssistantMessageEvent>> = None;
    let mut added_partial = false;
    let mut timing = Timing::start();

    // The TS closure captures `partialMessage`/`addedPartial` by reference;
    // here the finish helper runs inline in the abort path below.
    macro_rules! finish_aborted_message {
        () => {{
            let mut final_message = create_aborted_assistant_message(
                config,
                partial_event.as_deref().and_then(event_partial),
            );
            finalize_message(&mut final_message, config, context, &timing);
            // An aborted turn never reached `done`: the provider timing stays unset.
            final_message.meta.duration = None;
            final_message.meta.ttft = None;
            if added_partial {
                *context.messages.last_mut().unwrap() = AgentMessage::from(final_message.clone());
            } else {
                context
                    .messages
                    .push(AgentMessage::from(final_message.clone()));
                emit(AgentEvent::MessageStart {
                    message: AgentMessage::from(final_message.clone()),
                })
                .await?;
            }
            emit(AgentEvent::MessageEnd {
                message: AgentMessage::from(final_message.clone()),
            })
            .await?;
            for_turn_end(final_message)
        }};
    }

    let result = stream_assistant_response_inner(
        context,
        config,
        signal,
        emit,
        stream_fn,
        llm_messages,
        prepared_calls,
        &mut partial_event,
        &mut added_partial,
        &mut timing,
    )
    .await;

    match result {
        Ok(message) => Ok(message),
        Err(error) => {
            if signal.is_some_and(AbortSignal::is_aborted) && is_abort_error(&error) {
                return Ok(finish_aborted_message!());
            }
            Err(error)
        }
    }
}

/// Inner body of `streamAssistantResponse`.
// Direct port of the TS `try` block.
#[allow(clippy::too_many_lines)]
async fn stream_assistant_response_inner(
    context: &mut AgentContext,
    config: &AgentLoopConfig,
    signal: Option<&AbortSignal>,
    emit: &AgentEventSink,
    stream_fn: Option<&StreamFn>,
    llm_messages: anyhow::Result<Vec<crate::types::Message>>,
    prepared_calls: &mut Vec<Preparation>,
    partial_event: &mut Option<Arc<crate::stream::AssistantMessageEvent>>,
    added_partial: &mut bool,
    timing: &mut Timing,
) -> anyhow::Result<AssistantMessage> {
    crate::abort::throw_if_aborted_signal(signal)?;

    let llm_messages = llm_messages?;

    let stream_fn = stream_fn.ok_or_else(|| {
        anyhow::anyhow!(
            "No stream function provided; the agent loop requires a model stream function (pa-ai integration supplies the default)"
        )
    })?;

    let resolved_api_key = match config.get_api_key.as_ref() {
        Some(get_api_key) => {
            match race_with_abort(get_api_key(config.model.provider.clone()), signal).await? {
                Some(key) => Some(key),
                None => config.api_key.clone(),
            }
        }
        None => config.api_key.clone(),
    };

    let llm_context = LlmContext {
        system_prompt: Some(
            config
                .get_system_prompt
                .as_ref()
                .map_or_else(|| context.system_prompt.clone(), |hook| hook()),
        ),
        messages: llm_messages,
        tools: context
            .tools
            .iter()
            .map(|tool| ToolDefinition {
                name: tool.name().to_string(),
                description: tool.description().to_string(),
                parameters: tool.parameters().clone(),
            })
            .collect(),
    };

    let options = StreamRequestOptions {
        temperature: config.temperature,
        max_tokens: config.max_tokens,
        reasoning: config.reasoning,
        session_id: config.session_id.clone(),
        service_tier: config.service_tier,
        api_key: resolved_api_key,
        signal: signal.cloned().unwrap_or_default(),
        // The TS loop config's `onPayload`/`onResponse` ride every stream
        // call; the Rust request-timing seam composes them per request at
        // the `StreamFn` boundary instead.
        on_payload: None,
        on_response: None,
        headers: None,
    };

    timing.started = std::time::Instant::now();
    let mut response = race_with_abort(
        stream_fn(config.model.clone(), llm_context, options),
        signal,
    )
    .await?;

    loop {
        let next = match signal {
            Some(signal) => {
                // TS races the iterator with `closeIterator` as the abort
                // callback: cancel the stream when the user aborts mid-read.
                let result = crate::abort::race_with_abort(response.next_event(), signal).await;
                if result.is_err() {
                    response.close();
                }
                result?
            }
            None => response.next_event().await,
        };
        let Some(event) = next else {
            break;
        };

        match event {
            crate::stream::AssistantMessageEvent::Start { mut partial } => {
                stamp_credential(&mut partial, config);
                let message = AgentMessage::from(partial.clone());
                *added_partial = true;
                context.messages.push(message.clone());
                *partial_event = Some(Arc::new(crate::stream::AssistantMessageEvent::Start {
                    partial,
                }));
                emit(AgentEvent::MessageStart { message }).await?;
            }
            mut event if event.is_delta() => {
                timing.mark_first_token();
                if let Some(partial) = event_partial_mut(&mut event) {
                    stamp_credential(partial, config);
                }
                let event = Arc::new(event);
                if let Some(partial) = event_partial(&event) {
                    *partial_event = Some(Arc::clone(&event));
                    emit(AgentEvent::MessageUpdate {
                        message: Arc::new(AgentMessage::from(partial.clone())),
                        assistant_message_event: event,
                    })
                    .await?;
                }
            }
            ref event if event.terminal_message().is_some() => {
                let mut final_message = event.terminal_message().unwrap().clone();
                match race_with_abort(response.result(), signal).await {
                    Ok(result_message) => final_message = result_message,
                    Err(error) => {
                        let aborted = signal.is_some_and(AbortSignal::is_aborted);
                        if !(aborted && is_abort_error(&error)) {
                            return Err(error);
                        }
                    }
                }
                finalize_message(&mut final_message, config, context, timing);
                if *added_partial {
                    *context.messages.last_mut().unwrap() =
                        AgentMessage::from(final_message.clone());
                } else {
                    context
                        .messages
                        .push(AgentMessage::from(final_message.clone()));
                }
                *prepared_calls =
                    prepare_dispatch(&mut final_message, context, config, signal).await;
                if !*added_partial {
                    emit(AgentEvent::MessageStart {
                        message: AgentMessage::from(final_message.clone()),
                    })
                    .await?;
                }
                emit(AgentEvent::MessageEnd {
                    message: AgentMessage::from(final_message.clone()),
                })
                .await?;
                return Ok(for_turn_end(final_message));
            }
            _ => {}
        }
    }

    // Stream ended without a terminal event: resolve the final message (TS
    // awaits `response.result()` here too).
    let mut final_message = race_with_abort(response.result(), signal).await?;
    finalize_message(&mut final_message, config, context, timing);
    if *added_partial {
        *context.messages.last_mut().unwrap() = AgentMessage::from(final_message.clone());
    } else {
        context
            .messages
            .push(AgentMessage::from(final_message.clone()));
    }
    *prepared_calls = prepare_dispatch(&mut final_message, context, config, signal).await;
    if !*added_partial {
        emit(AgentEvent::MessageStart {
            message: AgentMessage::from(final_message.clone()),
        })
        .await?;
    }
    emit(AgentEvent::MessageEnd {
        message: AgentMessage::from(final_message.clone()),
    })
    .await?;
    Ok(for_turn_end(final_message))
}

/// omp's `turn_end` carries the message as the provider finished it: the
/// commit-time stamps (`completedAt`, `contextSnapshot`) land on the stored
/// message only.
fn for_turn_end(mut message: AssistantMessage) -> AssistantMessage {
    message.meta.completed_at = None;
    message.meta.context_snapshot = None;
    message
}

fn event_partial(event: &crate::stream::AssistantMessageEvent) -> Option<&AssistantMessage> {
    match event {
        crate::stream::AssistantMessageEvent::Start { partial }
        | crate::stream::AssistantMessageEvent::TextStart { partial, .. }
        | crate::stream::AssistantMessageEvent::TextDelta { partial, .. }
        | crate::stream::AssistantMessageEvent::TextEnd { partial, .. }
        | crate::stream::AssistantMessageEvent::ThinkingStart { partial, .. }
        | crate::stream::AssistantMessageEvent::ThinkingDelta { partial, .. }
        | crate::stream::AssistantMessageEvent::ThinkingEnd { partial, .. }
        | crate::stream::AssistantMessageEvent::ToolCallStart { partial, .. }
        | crate::stream::AssistantMessageEvent::ToolCallDelta { partial, .. }
        | crate::stream::AssistantMessageEvent::ToolCallEnd { partial, .. } => Some(partial),
        _ => None,
    }
}

/// Request timing for `duration` / `ttft` (milliseconds, provider-independent).
pub(crate) struct Timing {
    started: std::time::Instant,
    first_token_ms: Option<f64>,
}

impl Timing {
    fn start() -> Self {
        Timing {
            started: std::time::Instant::now(),
            first_token_ms: None,
        }
    }

    fn mark_first_token(&mut self) {
        if self.first_token_ms.is_none() {
            self.first_token_ms = Some(self.started.elapsed().as_secs_f64() * 1000.0);
        }
    }
}

fn stamp_credential(message: &mut AssistantMessage, config: &AgentLoopConfig) {
    if message.meta.credential_id.is_none() {
        message.meta.credential_id = config.credential_id;
    }
}

/// Stamp the omp turn metadata on a finished (or aborted) assistant message:
/// credential, request timing, local completion time, and (for non-failed turns)
/// the prompt-accounting `contextSnapshot` omp records when committing the message.
fn finalize_message(
    message: &mut AssistantMessage,
    config: &AgentLoopConfig,
    context: &AgentContext,
    timing: &Timing,
) {
    // TS's aborted message is fresh: no credential/timing/commit stamps at all.
    if message.stop_reason == StopReason::Aborted {
        return;
    }
    stamp_credential(message, config);
    if message.meta.duration.is_none() {
        message.meta.duration = Some(timing.started.elapsed().as_secs_f64() * 1000.0);
    }
    if message.meta.ttft.is_none() {
        message.meta.ttft = timing.first_token_ms;
    }
    message.meta.completed_at = Some(crate::now_ms());
    if !matches!(message.stop_reason, StopReason::Aborted | StopReason::Error)
        && message.meta.context_snapshot.is_none()
    {
        let u = &message.usage;
        let mut prompt = u.input + u.cache_read + u.cache_write;
        if prompt == 0 {
            prompt = u.total_tokens.saturating_sub(u.output);
        }
        let overhead = context.system_prompt.chars().count()
            + context
                .tools
                .iter()
                .map(|t| t.name().len() + t.description().len() + t.parameters().to_string().len())
                .sum::<usize>();
        message.meta.context_snapshot = Some(ContextSnapshot {
            prompt_tokens: prompt,
            non_message_tokens: u64::try_from(overhead.div_ceil(4)).unwrap_or(0),
            compaction_epoch: config.compaction_epoch,
        });
    }
}

fn event_partial_mut(
    event: &mut crate::stream::AssistantMessageEvent,
) -> Option<&mut AssistantMessage> {
    match event {
        crate::stream::AssistantMessageEvent::Start { partial }
        | crate::stream::AssistantMessageEvent::TextStart { partial, .. }
        | crate::stream::AssistantMessageEvent::TextDelta { partial, .. }
        | crate::stream::AssistantMessageEvent::TextEnd { partial, .. }
        | crate::stream::AssistantMessageEvent::ThinkingStart { partial, .. }
        | crate::stream::AssistantMessageEvent::ThinkingDelta { partial, .. }
        | crate::stream::AssistantMessageEvent::ThinkingEnd { partial, .. }
        | crate::stream::AssistantMessageEvent::ToolCallStart { partial, .. }
        | crate::stream::AssistantMessageEvent::ToolCallDelta { partial, .. }
        | crate::stream::AssistantMessageEvent::ToolCallEnd { partial, .. } => Some(partial),
        _ => None,
    }
}
