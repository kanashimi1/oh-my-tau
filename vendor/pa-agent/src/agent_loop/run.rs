//! The agent turn loop: assistant turns, tool-call batches,
//! steering/follow-up/continuation message polling, and stop-hook
//! evaluation.

use crate::abort::AbortSignal;
use crate::stream::StreamFn;
use crate::types::{
    AgentContext, AgentEvent, AgentMessage, AssistantContent, AssistantMessage, Message,
    ShouldStopAfterTurnContext, StopReason, ToolCall, ToolResultMessage,
};

use super::abort::{
    poll_messages_unless_aborted, race_with_abort, settle_post_turn, PostTurnResult,
};
use super::response::{prepare_llm_messages, stream_assistant_response};
use super::tools::execute_tool_calls;
use super::{AgentEventSink, AgentLoopConfig};

/// Copy revised arguments from finalized calls onto the assistant message's tool calls.
pub(super) fn bake_revised_arguments(
    message: &mut AssistantMessage,
    finalized: &[ToolCall],
) -> bool {
    let mut changed = false;
    for content in &mut message.content {
        if let AssistantContent::ToolCall(call) = content {
            if let Some(done) = finalized
                .iter()
                .find(|f| f.id == call.id && f.arguments != call.arguments)
            {
                call.arguments.clone_from(&done.arguments);
                changed = true;
            }
        }
    }
    changed
}

/// Developer message carrying passive tool context (TS `createAdditionalContextMessage`).
pub(crate) fn create_additional_context_message(text: &str) -> AgentMessage {
    AgentMessage::Custom(crate::types::CustomAgentMessage {
        role: "developer".to_string(),
        payload: serde_json::json!({
            "content": [{"type": "text", "text": text}],
            "attribution": "agent",
            "timestamp": crate::now_ms(),
        }),
    })
}

// Direct port of the TS turn loop.
#[allow(clippy::too_many_lines)]
pub(crate) async fn run_loop(
    current_context: &mut AgentContext,
    new_messages: &mut Vec<AgentMessage>,
    config: &AgentLoopConfig,
    signal: Option<&AbortSignal>,
    emit: &AgentEventSink,
    stream_fn: Option<&StreamFn>,
    initial_messages: Vec<AgentMessage>,
) -> anyhow::Result<()> {
    // Messages whose events open the next turn (the run's prompts, then queued/steering messages).
    let mut turn_messages = initial_messages;
    let mut first_turn = true;
    let mut last_turn: Option<ShouldStopAfterTurnContext> = None;
    let mut pending_messages =
        poll_messages_unless_aborted(config.get_steering_messages.as_ref(), signal).await?;

    macro_rules! should_stop_before_turn {
        () => {
            !first_turn
                && config
                    .should_stop_before_turn
                    .as_ref()
                    .map(|hook| hook())
                    .unwrap_or(false)
        };
    }

    loop {
        crate::abort::throw_if_aborted_signal(signal)?;
        let mut has_more_tool_calls = true;

        while has_more_tool_calls || !pending_messages.is_empty() {
            crate::abort::throw_if_aborted_signal(signal)?;
            first_turn = false;

            // TS order: queued messages join the context, the provider request is prepared
            // (`context` event), then the turn opens (`turn_start`) and its input messages emit.
            for message in pending_messages.drain(..) {
                current_context.messages.push(message.clone());
                new_messages.push(message.clone());
                turn_messages.push(message);
            }
            let llm_messages = prepare_llm_messages(current_context, config, signal).await;
            emit(AgentEvent::TurnStart).await?;
            for message in turn_messages.drain(..) {
                emit(AgentEvent::MessageStart {
                    message: message.clone(),
                })
                .await?;
                emit(AgentEvent::MessageEnd { message }).await?;
            }

            let mut prepared_calls = Vec::new();
            let mut message = stream_assistant_response(
                current_context,
                config,
                signal,
                emit,
                stream_fn,
                llm_messages,
                &mut prepared_calls,
            )
            .await?;
            new_messages.push(AgentMessage::from(message.clone()));

            if message.stop_reason == StopReason::Error
                || message.stop_reason == StopReason::Aborted
            {
                emit(AgentEvent::TurnEnd {
                    message: AgentMessage::from(message.clone()),
                    tool_results: Vec::new(),
                })
                .await?;
                emit(AgentEvent::AgentEnd {
                    messages: new_messages.clone(),
                })
                .await?;
                return Ok(());
            }

            let tool_calls = message
                .tool_calls()
                .into_iter()
                .cloned()
                .collect::<Vec<ToolCall>>();

            let mut tool_results: Vec<ToolResultMessage> = Vec::new();
            has_more_tool_calls = false;
            if !tool_calls.is_empty() {
                // Tool hooks/executions each snapshot the context: freeze first so
                // they share the history instead of copying it.
                current_context.messages.freeze();
                let executed_tool_batch = execute_tool_calls(
                    current_context,
                    &message,
                    config,
                    signal,
                    emit,
                    prepared_calls,
                )
                .await?;
                // A `beforeToolCall` revision is baked into the assistant message itself so
                // history (and the next provider request) holds a single version of the call.
                if bake_revised_arguments(&mut message, &executed_tool_batch.tool_calls) {
                    let baked = AgentMessage::from(message.clone());
                    if let Some(last) = current_context.messages.last_mut() {
                        *last = baked.clone();
                    }
                    if let Some(entry) = new_messages
                        .iter_mut()
                        .rev()
                        .find(|m| matches!(m, AgentMessage::Standard(Message::Assistant(_))))
                    {
                        *entry = baked;
                    }
                }
                tool_results.extend(executed_tool_batch.messages);
                has_more_tool_calls = !executed_tool_batch.terminate;

                for result in &tool_results {
                    current_context
                        .messages
                        .push(AgentMessage::from(result.clone()));
                    new_messages.push(AgentMessage::from(result.clone()));
                }
                if let Some(text) = executed_tool_batch.additional_context {
                    let context_message = create_additional_context_message(&text);
                    emit(AgentEvent::MessageStart {
                        message: context_message.clone(),
                    })
                    .await?;
                    emit(AgentEvent::MessageEnd {
                        message: context_message.clone(),
                    })
                    .await?;
                    current_context.messages.push(context_message.clone());
                    new_messages.push(context_message);
                }
            }

            emit(AgentEvent::TurnEnd {
                message: AgentMessage::from(message.clone()),
                tool_results: tool_results.clone(),
            })
            .await?;
            if signal.is_some_and(AbortSignal::is_aborted) {
                emit(AgentEvent::AgentEnd {
                    messages: new_messages.clone(),
                })
                .await?;
                return Ok(());
            }
            current_context.messages.freeze();
            last_turn = Some(ShouldStopAfterTurnContext {
                message: message.clone(),
                tool_results: tool_results.clone(),
                context: clone_context(current_context),
                new_messages: new_messages.clone(),
            });

            let should_stop_result = settle_post_turn(
                race_with_abort(
                    async {
                        match config.should_stop_after_turn.as_ref() {
                            Some(hook) => hook(last_turn.clone().unwrap()).await,
                            None => Ok(false),
                        }
                    },
                    signal,
                ),
                signal,
            )
            .await?;
            match should_stop_result {
                PostTurnResult::Aborted | PostTurnResult::Completed(true) => {
                    emit(AgentEvent::AgentEnd {
                        messages: new_messages.clone(),
                    })
                    .await?;
                    return Ok(());
                }
                PostTurnResult::Completed(false) => {}
            }
            if should_stop_before_turn!() {
                emit(AgentEvent::AgentEnd {
                    messages: new_messages.clone(),
                })
                .await?;
                return Ok(());
            }

            let steering_messages_result = settle_post_turn(
                poll_messages_unless_aborted(config.get_steering_messages.as_ref(), signal),
                signal,
            )
            .await?;
            match steering_messages_result {
                PostTurnResult::Aborted => {
                    emit(AgentEvent::AgentEnd {
                        messages: new_messages.clone(),
                    })
                    .await?;
                    return Ok(());
                }
                PostTurnResult::Completed(messages) => {
                    pending_messages = messages;
                    // Steering drained by this poll owns the turn boundary;
                    // stop only when it was empty.
                    if pending_messages.is_empty() && should_stop_before_turn!() {
                        emit(AgentEvent::AgentEnd {
                            messages: new_messages.clone(),
                        })
                        .await?;
                        return Ok(());
                    }
                }
            }
        }

        if should_stop_before_turn!() {
            break;
        }
        let follow_up_messages_result = settle_post_turn(
            poll_messages_unless_aborted(config.get_follow_up_messages.as_ref(), signal),
            signal,
        )
        .await?;
        let follow_up_messages = match follow_up_messages_result {
            PostTurnResult::Aborted => {
                emit(AgentEvent::AgentEnd {
                    messages: new_messages.clone(),
                })
                .await?;
                return Ok(());
            }
            PostTurnResult::Completed(messages) => messages,
        };
        if !follow_up_messages.is_empty() {
            pending_messages = follow_up_messages;
            continue;
        }

        if should_stop_before_turn!() {
            break;
        }
        let continuation_messages_result = match last_turn.clone() {
            Some(context) => {
                let continuation_op: crate::BoxFut<'static, anyhow::Result<Vec<AgentMessage>>> =
                    match config.get_continuation_messages.as_ref() {
                        Some(hook) => hook(context, signal.cloned().unwrap_or_default()),
                        None => Box::pin(async { Ok(Vec::new()) }),
                    };
                settle_post_turn(race_with_abort(continuation_op, signal), signal).await?
            }
            None => PostTurnResult::Completed(Vec::new()),
        };
        let continuation_messages = match continuation_messages_result {
            PostTurnResult::Aborted => {
                emit(AgentEvent::AgentEnd {
                    messages: new_messages.clone(),
                })
                .await?;
                return Ok(());
            }
            PostTurnResult::Completed(messages) => messages,
        };
        if !continuation_messages.is_empty() {
            pending_messages = continuation_messages;
            continue;
        }

        break;
    }

    emit(AgentEvent::AgentEnd {
        messages: new_messages.clone(),
    })
    .await?;
    Ok(())
}

/// Snapshot for hooks and tool executions: shares the frozen transcript segments
/// (see [`crate::types::ContextMessages`]); only the unfrozen tail is copied.
pub(crate) fn clone_context(context: &AgentContext) -> AgentContext {
    context.clone()
}
