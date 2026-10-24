//! Transport selection for `stream_simple` (TS `streamSimple` / `mapOptionsForApi` routing):
//! which wire API a model streams through, with the provider-native options mapped from the
//! caller's simple options. [`plan_simple_request`] exposes the same decision as a pure request
//! body so it can be checked against the TS oracle without any I/O.

use serde_json::Value;

use crate::compat::ModelView;
use crate::providers::anthropic::{self, AnthropicOptions};
use crate::providers::kimi::{self, KimiFormat};
use crate::providers::openai_completions::{self, OpenAICompletionsOptions, ToolChoice};
use crate::providers::responses_request::{self, Flavor, ResponsesOptions, Summary};
use crate::types::{Context, Model, ModelThinkingLevel, SimpleStreamOptions, SimpleToolChoice};

/// Process switches the TS transport table reads.
#[derive(Clone, Debug)]
pub struct TransportEnv {
    /// `PI_OPENROUTER_RESPONSES !== "0"`: openrouter models stream through the Responses API.
    pub openrouter_responses: bool,
    /// `AZURE_OPENAI_DEPLOYMENT_NAME_MAP` (`model=deployment,...`).
    pub azure_deployment_name_map: Option<String>,
}

impl TransportEnv {
    #[must_use]
    pub fn from_process() -> Self {
        Self {
            openrouter_responses: std::env::var("PI_OPENROUTER_RESPONSES").as_deref() != Ok("0"),
            azure_deployment_name_map: std::env::var("AZURE_OPENAI_DEPLOYMENT_NAME_MAP").ok(),
        }
    }
}

/// The wire request a simple stream resolves to.
pub(crate) enum Resolved {
    Anthropic {
        model: Model,
        view: ModelView,
        options: AnthropicOptions,
    },
    Responses {
        model: Model,
        view: ModelView,
        options: ResponsesOptions,
        flavor: Flavor,
    },
    Completions {
        model: Model,
        options: OpenAICompletionsOptions,
    },
    /// Every other api streams through its registered provider unchanged.
    Registry,
}

/// TS `streamSimple` preamble: a session id for every inference (`withInferenceSessionId`) and the
/// sampling filter (`withSupportedSamplingParams`).
pub(crate) fn prepare_options(
    model: &Model,
    options: Option<&SimpleStreamOptions>,
    fresh_id: &dyn Fn() -> String,
) -> SimpleStreamOptions {
    let mut simple = options.cloned().unwrap_or_default();
    if simple.base.session_id.as_deref().is_none_or(str::is_empty) {
        simple.base.session_id = Some(fresh_id());
    }
    if ModelView::of(model).flag("supportsSamplingParams") == Some(false) {
        simple.base.temperature = None;
    }
    simple
}

/// Pick the transport for `model` and map `options` onto it.
///
/// # Errors
/// The options cannot form a valid request for the chosen transport.
pub(crate) fn resolve(
    model: &Model,
    options: &SimpleStreamOptions,
    env: &TransportEnv,
) -> Result<Resolved, String> {
    let view = ModelView::of(model);
    if model.provider == kimi::PROVIDER {
        let simple = anthropic::normalize_mandatory_reasoning(model, &view, options);
        if kimi::resolve_format(model, &view)? == KimiFormat::Anthropic {
            let shim = kimi::anthropic_shim(model, &view, &simple)?;
            let mut options = shim.options;
            options.view = Some(shim.view.clone());
            return Ok(Resolved::Anthropic {
                model: shim.model,
                view: shim.view,
                options,
            });
        }
        return Ok(Resolved::Completions {
            model: model.clone(),
            options: kimi::openai_options(model, &simple),
        });
    }
    match model.api.as_str() {
        "openrouter" if env.openrouter_responses => {
            let options = map_responses_options(model, &view, options)?;
            Ok(Resolved::Responses {
                model: model.clone(),
                view,
                options,
                flavor: Flavor::OpenAi,
            })
        }
        "openrouter" => {
            let o = anthropic::normalize_mandatory_reasoning(model, &view, options);
            Ok(Resolved::Completions {
                model: model.clone(),
                options: map_completions_options(model, &view, &o)?,
            })
        }
        "openai-responses" => {
            let options = map_responses_options(model, &view, options)?;
            Ok(Resolved::Responses {
                model: model.clone(),
                view,
                options,
                flavor: Flavor::OpenAi,
            })
        }
        "azure-openai-responses" => {
            let mapped = map_responses_options(model, &view, options)?;
            let deployment = azure_deployment_name(model, env.azure_deployment_name_map.as_deref());
            Ok(Resolved::Responses {
                model: model.clone(),
                view,
                options: mapped,
                flavor: Flavor::Azure { deployment },
            })
        }
        "anthropic-messages" => {
            let mapped = anthropic::map_simple_options(model, &view, options)?;
            Ok(Resolved::Anthropic {
                model: model.clone(),
                view,
                options: mapped,
            })
        }
        _ => Ok(Resolved::Registry),
    }
}

/// TS `resolveDeploymentName` (no per-call override): `AZURE_OPENAI_DEPLOYMENT_NAME_MAP` entry or the id.
fn azure_deployment_name(model: &Model, map: Option<&str>) -> String {
    map.into_iter()
        .flat_map(|m| m.split(','))
        .filter_map(|entry| entry.split_once('='))
        .map(|(k, v)| (k.trim(), v.trim()))
        .find(|(k, v)| *k == model.id && !v.is_empty())
        .map_or_else(|| model.id.clone(), |(_, v)| v.to_string())
}

/// TS `resolveOpenAiReasoningEffort`.
fn resolve_openai_reasoning_effort(
    model: &Model,
    view: &ModelView,
    options: &SimpleStreamOptions,
) -> Result<Option<String>, String> {
    let Some(reasoning) = options.reasoning.filter(|l| *l != ModelThinkingLevel::Off) else {
        return Ok(None);
    };
    if !model.reasoning || view.thinking.is_none() {
        return Ok(None);
    }
    let name = reasoning.wire_name();
    let efforts = view.efforts();
    if efforts.contains(&name) {
        return Ok(Some(name.to_string()));
    }
    let compat_mapped = view
        .compat
        .get("reasoningEffortMap")
        .and_then(|m| m.get(name))
        .and_then(Value::as_str);
    if let Some(mapped) = compat_mapped.filter(|m| efforts.contains(m)) {
        return Ok(Some(mapped.to_string()));
    }
    if compat_mapped.is_some() || view.effort_map(name).is_some() {
        return Ok(Some(name.to_string()));
    }
    anthropic::request::require_supported_effort(model, view, name)?;
    Ok(Some(name.to_string()))
}

fn openai_tool_choice(choice: Option<&SimpleToolChoice>) -> Option<ToolChoice> {
    Some(match choice? {
        SimpleToolChoice::Auto => ToolChoice::Auto,
        SimpleToolChoice::None => ToolChoice::None,
        SimpleToolChoice::Any | SimpleToolChoice::Required => ToolChoice::Required,
        SimpleToolChoice::Function { name } if !name.is_empty() => {
            ToolChoice::Function { name: name.clone() }
        }
        SimpleToolChoice::Function { .. } => return None,
    })
}

/// TS `mapOptionsForApi` for the Responses-family apis.
fn map_responses_options(
    model: &Model,
    view: &ModelView,
    raw: &SimpleStreamOptions,
) -> Result<ResponsesOptions, String> {
    let options = anthropic::normalize_mandatory_reasoning(model, view, raw);
    let mut base = options.base.clone();
    base.max_tokens = options
        .base
        .max_tokens
        .or((model.max_tokens > 0).then_some(model.max_tokens));
    Ok(ResponsesOptions {
        base,
        reasoning: resolve_openai_reasoning_effort(model, view, &options)?,
        reasoning_summary: Summary::Unset,
        disable_reasoning: options.disable_reasoning,
        tool_choice: openai_tool_choice(options.tool_choice.as_ref()),
        max_tokens_explicit: raw.base.max_tokens.is_some(),
        openrouter_variant: None,
    })
}

/// TS `mapOptionsForApi` for the `openrouter` chat-completions fallback.
fn map_completions_options(
    model: &Model,
    view: &ModelView,
    options: &SimpleStreamOptions,
) -> Result<OpenAICompletionsOptions, String> {
    let mut base = options.base.clone();
    base.max_tokens = options
        .base
        .max_tokens
        .or((model.max_tokens > 0).then_some(model.max_tokens));
    let mut o = OpenAICompletionsOptions::from_base(base);
    o.max_tokens_explicit = options.base.max_tokens.is_some();
    o.reasoning_effort = resolve_openai_reasoning_effort(model, view, options)?
        .as_deref()
        .and_then(crate::models::thinking_level_from_str);
    if options.disable_reasoning {
        o.reasoning_enabled = Some(false);
    }
    o.tool_choice = openai_tool_choice(options.tool_choice.as_ref());
    Ok(o)
}

/// The request a simple stream sends: the wire api and its JSON body.
pub struct PlannedRequest {
    pub api: String,
    pub body: Value,
}

/// The wire request `stream_simple` builds for these inputs, without sending it.
///
/// # Errors
/// The inputs cannot form a request, or the model's api has no planned transport.
pub fn plan_simple_request(
    model: &Model,
    context: &Context,
    options: Option<&SimpleStreamOptions>,
    env: &TransportEnv,
    fresh_id: &dyn Fn() -> String,
) -> Result<PlannedRequest, String> {
    let simple = prepare_options(model, options, fresh_id);
    let api_key = simple.base.api_key.clone().unwrap_or_default();
    match resolve(model, &simple, env)? {
        Resolved::Anthropic {
            model,
            view,
            options,
        } => Ok(PlannedRequest {
            api: model.api.clone(),
            body: anthropic::build_request_body(&model, &view, context, &options, &api_key)?,
        }),
        Resolved::Responses {
            model,
            view,
            options,
            flavor,
        } => Ok(PlannedRequest {
            api: model.api.clone(),
            body: responses_request::build_body(&model, &view, context, &options, &flavor)?,
        }),
        Resolved::Completions { model, options } => Ok(PlannedRequest {
            api: model.api.clone(),
            body: openai_completions::build_request_body(&model, context, &options)?,
        }),
        Resolved::Registry => Err(format!("no planned transport for {}", model.api)),
    }
}

/// Stream the `api` of `model` through the transport table (the registry providers of the apis the
/// table owns delegate here).
pub(crate) fn stream_simple_for_api(
    model: &Model,
    context: &Context,
    options: Option<&SimpleStreamOptions>,
) -> crate::event_stream::AssistantMessageEventStream {
    let simple = prepare_options(
        model,
        options,
        &crate::providers::anthropic::request::uuid_v4,
    );
    match dispatch(model, context, &simple, &TransportEnv::from_process()) {
        Ok(Some(stream)) => stream,
        Ok(None) => crate::providers::anthropic::error_stream_for(
            model,
            format!("no transport for {}", model.api),
        ),
        Err(message) => crate::providers::anthropic::error_stream_for(model, message),
    }
}

/// Start the stream `options` resolve to; `Ok(None)` when the api streams through its registered
/// provider unchanged.
///
/// # Errors
/// The options cannot form a valid request for the chosen transport.
pub(crate) fn dispatch(
    model: &Model,
    context: &Context,
    options: &SimpleStreamOptions,
    env: &TransportEnv,
) -> Result<Option<crate::event_stream::AssistantMessageEventStream>, String> {
    use crate::stream::clamp_output_budget;
    Ok(Some(match resolve(model, options, env)? {
        Resolved::Anthropic {
            model, mut options, ..
        } => {
            clamp_output_budget(&model, context, Some(&mut options.base));
            anthropic::stream_anthropic(&model, context, Some(&options))
        }
        Resolved::Responses {
            model,
            mut options,
            flavor,
            ..
        } => {
            clamp_output_budget(&model, context, Some(&mut options.base));
            match flavor {
                Flavor::OpenAi => crate::providers::openai_responses::stream_openai_responses(&model, context, Some(&options)),
                Flavor::Azure { deployment } => crate::providers::azure_openai_responses::stream_azure_openai_responses(
                    &model,
                    context,
                    Some(&crate::providers::azure_openai_responses::AzureOpenAIResponsesOptions::from_responses(options, deployment)),
                ),
            }
        }
        Resolved::Completions { model, mut options } => {
            clamp_output_budget(&model, context, Some(&mut options.base));
            openai_completions::stream_openai_completions(&model, context, Some(&options))
        }
        Resolved::Registry => return Ok(None),
    }))
}
