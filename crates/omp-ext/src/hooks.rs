//! `pa-agent` hook points backed by the extension host.
//!
//! Event names/payloads follow omp: `context` {messages}, `tool_call` {toolName, toolCallId, input},
//! `tool_result` {toolName, toolCallId, input, content, details, isError}.

use std::sync::Arc;

use anyhow::Result;
use pa_agent::abort::AbortSignal;
use pa_agent::agent_loop::{AfterToolCallFn, BeforeToolCallFn, TransformContextFn};
use pa_agent::types::{AfterToolCallResult, AgentMessage, BeforeToolCallResult, ToolResultContent};
use serde_json::{json, Value};

use crate::ExtHost;

/// Aggregated `tool_call` outcome; [`ExtHooks::before_tool_call`] maps it onto
/// [`BeforeToolCallResult`] (`block`/`reason`, `input` -> `args`, `additional_context`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ToolCallOutcome {
    pub block: bool,
    pub reason: Option<String>,
    pub input: Option<Value>,
    pub additional_context: Option<String>,
}

impl ToolCallOutcome {
    fn from_value(v: &Value) -> Self {
        Self {
            block: v.get("block").and_then(Value::as_bool).unwrap_or(false),
            reason: v.get("reason").and_then(Value::as_str).map(str::to_owned),
            input: v.get("input").cloned(),
            additional_context: v
                .get("additionalContext")
                .and_then(Value::as_str)
                .map(str::to_owned),
        }
    }
}

/// Emit `tool_call` before execution.
///
/// # Errors
/// Transport failure (a failing/timed-out handler comes back as `block: true`, like omp).
pub async fn tool_call(
    host: &ExtHost,
    tool_name: &str,
    tool_call_id: &str,
    input: &Value,
    ctx: Option<Value>,
) -> Result<ToolCallOutcome> {
    let r = host
        .emit(
            json!({"type":"tool_call","toolName":tool_name,"toolCallId":tool_call_id,"input":input}),
            ctx,
        )
        .await?;
    Ok(ToolCallOutcome::from_value(&r.result))
}

/// Emit `context`; returns the transformed message list.
///
/// # Errors
/// Transport failure or messages that do not round-trip into [`AgentMessage`].
pub async fn context(
    host: &ExtHost,
    messages: &[AgentMessage],
    ctx: Option<Value>,
) -> Result<Vec<AgentMessage>> {
    if ctx.is_none() && !host.has_handlers("context") {
        return Ok(messages.to_vec());
    }
    let r = host
        .emit(json!({"type":"context","messages":messages}), ctx)
        .await?;
    match r.result.get("messages") {
        Some(m) if !m.is_null() => Ok(serde_json::from_value(m.clone())?),
        _ => Ok(messages.to_vec()),
    }
}

/// [`context`] for an owned list: no copy of the messages is made. When no
/// extension subscribes to `context` the list is returned as is without being
/// serialized; otherwise it is serialized by reference and handed back
/// untouched unless a handler replaced it.
///
/// # Errors
/// Transport failure or messages that do not round-trip into [`AgentMessage`].
pub async fn context_owned(
    host: &ExtHost,
    messages: Vec<AgentMessage>,
    ctx: Option<Value>,
) -> Result<Vec<AgentMessage>> {
    context_owned_skipping(host, messages, ctx, &[]).await
}

/// [`context_owned`] with the handlers of the extensions named in `skip` left
/// out for this emit (`skipExtensions`).
///
/// # Errors
/// Transport failure or messages that do not round-trip into [`AgentMessage`].
pub async fn context_owned_skipping(
    host: &ExtHost,
    messages: Vec<AgentMessage>,
    ctx: Option<Value>,
    skip: &[String],
) -> Result<Vec<AgentMessage>> {
    if ctx.is_none() && !host.has_handlers("context") {
        return Ok(messages);
    }
    let r = host
        .emit_skipping(json!({"type":"context","messages":&messages}), ctx, skip)
        .await?;
    match r.result.get("messages") {
        Some(m) if !m.is_null() => Ok(serde_json::from_value(m.clone())?),
        _ => Ok(messages),
    }
}

/// Names of the extensions whose handlers skip the next `context` emit.
pub type SkipFn = Arc<dyn Fn() -> Vec<String> + Send + Sync>;

/// Hook factory bound to a host.
#[derive(Clone)]
pub struct ExtHooks {
    host: Arc<ExtHost>,
}

impl ExtHooks {
    #[must_use]
    pub fn new(host: Arc<ExtHost>) -> Self {
        Self { host }
    }

    /// omp `context` event. On abort or host failure the input is returned unchanged only for
    /// aborts; host failures surface as errors.
    #[must_use]
    pub fn transform_context(&self) -> TransformContextFn {
        self.transform_context_skipping(Arc::new(Vec::new))
    }

    /// [`transform_context`](Self::transform_context) that asks `skip` per
    /// request which extensions stand aside for that `context` emit.
    #[must_use]
    pub fn transform_context_skipping(&self, skip: SkipFn) -> TransformContextFn {
        let host = self.host.clone();
        Arc::new(move |messages, signal: AbortSignal| {
            let host = host.clone();
            let skip = skip();
            Box::pin(async move {
                if signal.is_aborted() {
                    return Ok(messages);
                }
                context_owned_skipping(&host, messages, None, &skip).await
            })
        })
    }

    /// omp `tool_call` event; `block` (including fail-closed handler errors) blocks the call,
    /// `input` rewrites the arguments and `additionalContext` is delivered after the results.
    #[must_use]
    pub fn before_tool_call(&self) -> BeforeToolCallFn {
        let host = self.host.clone();
        Arc::new(move |ctx, signal: AbortSignal| {
            let host = host.clone();
            Box::pin(async move {
                let call = tool_call(
                    &host,
                    &ctx.tool_call.name,
                    &ctx.tool_call.id,
                    &ctx.args,
                    None,
                );
                let outcome = tokio::select! {
                    r = call => r?,
                    () = signal.aborted() => {
                        return Ok(Some(BeforeToolCallResult {
                            block: true,
                            reason: Some("Tool execution was cancelled while an extension handler was pending".into()),
                            ..BeforeToolCallResult::default()
                        }));
                    }
                };
                if outcome.block {
                    return Ok(Some(BeforeToolCallResult {
                        block: true,
                        reason: outcome.reason,
                        ..BeforeToolCallResult::default()
                    }));
                }
                if outcome.input.is_none() && outcome.additional_context.is_none() {
                    return Ok(None);
                }
                Ok(Some(BeforeToolCallResult {
                    args: outcome.input,
                    additional_context: outcome.additional_context,
                    ..BeforeToolCallResult::default()
                }))
            })
        })
    }

    /// omp `tool_result` event; handler rewrites of `content`/`details`/`isError` are applied.
    #[must_use]
    pub fn after_tool_call(&self) -> AfterToolCallFn {
        let host = self.host.clone();
        Arc::new(move |ctx, signal: AbortSignal| {
            let host = host.clone();
            Box::pin(async move {
                if signal.is_aborted() {
                    return Ok(None);
                }
                // No `tool_result` subscriber: skip serializing the (possibly large) result.
                if !host.has_handlers("tool_result") {
                    return Ok(None);
                }
                let event = json!({
                    "type": "tool_result",
                    "toolName": ctx.tool_call.name,
                    "toolCallId": ctx.tool_call.id,
                    "input": ctx.args,
                    "content": ctx.result.content,
                    "details": ctx.result.details,
                    "isError": ctx.is_error,
                });
                let r = host.emit(event, None).await?.result;
                if r.is_null() {
                    return Ok(None);
                }
                let content = r
                    .get("content")
                    .filter(|c| !c.is_null())
                    .map(|c| serde_json::from_value::<Vec<ToolResultContent>>(c.clone()))
                    .transpose()?;
                Ok(Some(AfterToolCallResult {
                    content,
                    details: r.get("details").filter(|d| !d.is_null()).cloned(),
                    is_error: r.get("isError").and_then(Value::as_bool),
                    terminate: None,
                    additional_context: r
                        .get("additionalContext")
                        .and_then(Value::as_str)
                        .map(str::to_owned),
                }))
            })
        })
    }
}
