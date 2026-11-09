//! `completion()` handle ops: candidate resolution and one-shot executions.

use super::{
    json, mint_id, provider_target, AbortSignal, AgentDir, AssistantContentBlock, Context, Handle,
    HandleStatus, Map, Message, Model, OpCtx, ResolvedModel, SimpleStreamOptions, SimpleToolChoice,
    StopReason, Tool, UserContent, UserMessage, Value, COMPLETION_HANDLES, REQUEST_SLOTS,
    STRUCTURED_TOOL_NAME,
};
use tokio_util::sync::CancellationToken;

/// One candidate in the tier's fallback chain (TS `CompletionCandidate`).
pub(super) struct CompletionCandidate {
    model: Model,
    api_key: Option<String>,
    service_tier: Option<pa_types::ai::ServiceTier>,
    headers: Option<Vec<(String, String)>>,
    /// Effective reasoning from the resolved selector, if any.
    thinking: Option<pa_types::ai::ModelThinkingLevel>,
}

fn candidate_from_resolved(resolved: ResolvedModel) -> CompletionCandidate {
    CompletionCandidate {
        model: resolved.model,
        api_key: resolved.api_key,
        service_tier: None,
        headers: Some(resolved.headers),
        thinking: resolved.thinking,
    }
}

/// Walk a role's fallback chain to more candidates; the walk does blocking
/// config I/O (`resolve_in`), so it runs off the executor threads.
async fn walk_fallbacks(dir: AgentDir, primary: ResolvedModel) -> Vec<CompletionCandidate> {
    tokio::task::spawn_blocking(move || {
        let mut walker = omp_providers::fallback_walker(&dir, &primary);
        let mut out = Vec::new();
        while let Some(resolved) = walker.next_model() {
            out.push(candidate_from_resolved(resolved));
        }
        out
    })
    .await
    .unwrap_or_default()
}

/// TS `resolveTierCandidates`, omp-rs form: `default` answers from the live
/// provider target (then the `@default` role's fallback chain); `smol`/`slow`
/// resolve the role selector and walk its configured fallbacks.
async fn resolve_completion_candidates(tier: &str) -> Result<Vec<CompletionCandidate>, String> {
    let dir = AgentDir::default();
    let mut candidates = Vec::new();
    match tier {
        "default" => {
            // Primary: the session's live target; fallbacks: the `@default`
            // role's configured chain, walked when the primary fails.
            let target = provider_target()?;
            let (api_key, _) = crate::adapter::fresh_stored_key(
                &target.model.provider,
                target.credential_id,
                target.api_key,
            )
            .await;
            candidates.push(CompletionCandidate {
                model: target.model,
                api_key,
                service_tier: target.service_tier,
                headers: target.headers.map(|headers| headers.into_iter().collect()),
                thinking: None,
            });
            if let Ok(primary) =
                omp_providers::resolve_in_async(dir.clone(), "default".to_string()).await
            {
                candidates.extend(walk_fallbacks(dir, primary).await);
            }
        }
        "smol" | "slow" => {
            // `resolve_in` takes a bare role name (TS `@smol` alias form is
            // not a selector it parses).
            let primary = omp_providers::resolve_in_async(dir.clone(), tier.to_string())
                .await
                .map_err(|e| {
                    format!(
                        "completion() could not resolve a model for the \"{tier}\" tier. Configure modelRoles.{tier} or ensure a provider is available. ({e})"
                    )
                })?;
            candidates.push(candidate_from_resolved(primary.clone()));
            candidates.extend(walk_fallbacks(dir, primary).await);
        }
        other => {
            return Err(format!(
                "completion() received invalid arguments: model must be \"smol\", \"default\", or \"slow\", got {other:?}"
            ));
        }
    }
    if candidates.is_empty() {
        return Err(format!(
            "completion() could not resolve a model for the \"{tier}\" tier. Configure modelRoles.{tier} or ensure a provider is available."
        ));
    }
    Ok(candidates)
}

/// TS `executeCompletion`: one stateless, toolless completion over the
/// candidate chain (each fallback that issues a request consumes one retry
/// attempt; keyless candidates are skipped without consuming budget).
#[allow(clippy::too_many_lines)] // completion request: candidate chain, schema check, retries, usage
pub(super) async fn execute_completion(
    prompt: &str,
    system: Option<&str>,
    schema: Option<&Value>,
    candidates: &[CompletionCandidate],
    cancel: &CancellationToken,
) -> Result<(String, Option<Value>, pa_types::ai::Usage, Model), String> {
    let tools = schema.map(|schema| {
        vec![Tool {
            name: STRUCTURED_TOOL_NAME.to_string(),
            description:
                "Return your answer by calling this tool with the requested structured fields."
                    .to_string(),
            parameters: schema.clone(),
        }]
    });
    let system_prompt = system.unwrap_or("You are a helpful assistant.").to_string();
    let message = Message::User(UserMessage {
        content: UserContent::Text(prompt.to_string()),
        timestamp: pa_agent::now_ms().cast_unsigned(),
        rest: Map::default(),
    });

    let mut last_error: Option<String> = None;
    for (index, candidate) in candidates.iter().enumerate() {
        if cancel.is_cancelled() {
            return Err("completion() request aborted.".to_string());
        }
        if candidate.api_key.is_none() {
            last_error = Some(format!(
                "completion() has no API key for {}/{}. Configure credentials for this provider or choose another tier.",
                candidate.model.provider, candidate.model.id
            ));
            continue;
        }
        let context = Context {
            system_prompt: Some(system_prompt.clone()),
            messages: vec![message.clone()],
            tools: tools.clone(),
        };
        let options = SimpleStreamOptions {
            base: pa_ai::types::StreamOptions {
                api_key: candidate.api_key.clone(),
                service_tier: candidate.service_tier,
                headers: candidate
                    .headers
                    .clone()
                    .map(|headers| headers.into_iter().collect()),
                signal: Some(cancel.clone()),
                ..pa_ai::types::StreamOptions::default()
            },
            reasoning: candidate.thinking,
            thinking_budgets: None,
            // Utility one-shot: mirror the title/judgment path (no thinking)
            // for tiers without an explicit level; `slow` keeps the resolved
            // selector's level.
            disable_reasoning: candidate.thinking.is_none(),
            tool_choice: schema.map(|_| SimpleToolChoice::Function {
                name: STRUCTURED_TOOL_NAME.to_string(),
            }),
        };
        let response = match pa_ai::complete_simple(&candidate.model, &context, Some(options)).await
        {
            Ok(response) => response,
            Err(error) => {
                last_error = Some(format!("{error:?}"));
                continue;
            }
        };
        if response.stop_reason == StopReason::Aborted {
            return Err("completion() request aborted.".to_string());
        }
        if response.stop_reason == StopReason::Error {
            let message = response
                .error_message
                .clone()
                .unwrap_or_else(|| "completion() request failed.".to_string());
            last_error = Some(message.clone());
            if index + 1 >= candidates.len() {
                return Err(message);
            }
            continue;
        }
        if schema.is_some() {
            let call = response.content.iter().find_map(|block| match block {
                AssistantContentBlock::ToolCall(call) if call.name == STRUCTURED_TOOL_NAME => {
                    Some(call)
                }
                _ => None,
            });
            let value = if let Some(call) = call {
                Value::Object(call.arguments.clone())
            } else {
                let text = assistant_text(&response.content);
                if text.is_empty() {
                    return Err("completion() returned no structured response.".to_string());
                }
                serde_json::from_str::<Value>(&text).map_err(|_| {
                    "completion() did not return a structured response matching the schema."
                        .to_string()
                })?
            };
            return Ok((
                serde_json::to_string(&value).unwrap_or_default(),
                Some(value),
                response.usage,
                candidate.model.clone(),
            ));
        }
        let text = assistant_text(&response.content);
        if text.is_empty() {
            return Err("completion() returned no text output.".to_string());
        }
        return Ok((text, None, response.usage, candidate.model.clone()));
    }
    Err(last_error.unwrap_or_else(|| "completion() request failed.".to_string()))
}

/// Resolve the candidate chain for an `llm_query` model: a tier name
/// (`smol`/`default`/`slow`) or any model selector `resolve_in` understands
/// (`provider/id[:level]`, other role names).
pub(super) async fn resolve_query_candidates(
    model: &str,
) -> Result<Vec<CompletionCandidate>, String> {
    if matches!(model, "smol" | "default" | "slow") {
        return resolve_completion_candidates(model).await;
    }
    let dir = AgentDir::default();
    let primary = omp_providers::resolve_in_async(dir.clone(), model.to_string())
        .await
        .map_err(|e| format!("llm_query() could not resolve model {model:?}: {e}"))?;
    let mut candidates = vec![candidate_from_resolved(primary.clone())];
    candidates.extend(walk_fallbacks(dir, primary).await);
    Ok(candidates)
}

fn assistant_text(content: &[AssistantContentBlock]) -> String {
    content
        .iter()
        .filter_map(|block| match block {
            AssistantContentBlock::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

pub(super) async fn run_completion(args: &Value, ctx: &OpCtx) -> Result<Value, String> {
    let prompt = args
        .get("prompt")
        .and_then(Value::as_str)
        .filter(|prompt| !prompt.is_empty())
        .ok_or_else(|| {
            "completion() received invalid arguments: prompt must be a non-empty string".to_string()
        })?;
    let tier = args
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("default");
    let system = args.get("system").and_then(Value::as_str);
    let schema = args.get("schema").cloned();

    let candidates = resolve_completion_candidates(tier).await?;

    let id = mint_id("cmp");
    let handle = Handle::new("completion", id.clone());
    COMPLETION_HANDLES.lock().insert(id.clone(), handle.clone());

    let cell_signal = ctx.signal.clone();
    let cancel = handle.cancel.clone();
    let (prompt, system) = (prompt.to_string(), system.map(str::to_string));
    tokio::spawn(async move {
        match completion_job(
            prompt,
            system,
            schema,
            candidates,
            cancel.clone(),
            cell_signal,
        )
        .await
        {
            Ok((text, data, _usage)) => {
                handle.settle(HandleStatus::Completed, Some(text), data, None);
            }
            Err(error) => {
                let status = if cancel.is_cancelled() {
                    HandleStatus::Cancelled
                } else {
                    HandleStatus::Failed
                };
                handle.settle(status, None, None, Some(error));
            }
        }
        handle.mark_run_finished();
    });

    Ok(json!({"id": id}))
}

/// One retained completion run under the shared request-slot ceiling (TS
/// `retainCompletionHandle` + `evalRequestSlots`). The handle's cancellation
/// token reaches the provider stream mid-flight (`StreamOptions.signal`) and
/// stops the candidate walk; the cell's abort signal cancels the handle (TS
/// `AbortSignal.any([options.signal, controller.signal])`).
pub(super) async fn completion_job(
    prompt: String,
    system: Option<String>,
    schema: Option<Value>,
    candidates: Vec<CompletionCandidate>,
    cancel: CancellationToken,
    cell_signal: AbortSignal,
) -> Result<(String, Option<Value>, pa_types::ai::Usage), String> {
    let _permit = REQUEST_SLOTS
        .acquire()
        .await
        .map_err(|_| "completion() request aborted.".to_string())?;
    if cancel.is_cancelled() {
        return Err("completion() request aborted.".to_string());
    }
    tokio::select! {
        () = cell_signal.aborted() => {
            cancel.cancel();
            Err("completion() request aborted.".to_string())
        }
        result = execute_completion(&prompt, system.as_deref(), schema.as_ref(), &candidates, &cancel) => {
            result.map(|(text, data, usage, _model)| (text, data, usage))
        },
    }
}

// ---------------------------------------------------------------------------
// `__judge__` (TS `judgment-bridge.ts` `runEvalJudgment`)
// ---------------------------------------------------------------------------
