//! Azure `OpenAI` Responses API streaming provider: deployment name resolution (options/env map),
//! base-URL normalization with the /openai/v1 path, api-version query parameter, and the shared
//! Responses stream processor.

use std::collections::HashMap;

use serde_json::Map;

use crate::env_api_keys::get_env_api_key;
use crate::event_stream::{
    create_assistant_message_event_stream, AssistantMessageEvent, AssistantMessageEventStream,
    AssistantMessageEventWriter,
};
use crate::providers::openai_responses_shared::ResponsesStreamHooks;
use crate::providers::responses_request::{build_body, Flavor, ResponsesOptions};
use crate::registry::Provider;
use crate::types::{
    done_reason, error_reason, AssistantMessage, Context, Model, SimpleStreamOptions, StopReason,
    StreamOptions, Usage,
};
use crate::utils_inner::diagnostics::now_ms;
use crate::utils_inner::http::{send, HttpResponse, RequestOptions};
use crate::utils_inner::http_inspector::{finalize_http_error, RawHttpRequestDump};
use crate::utils_inner::json_parse::{parse_json_with_repair, parse_streaming_json};
use crate::utils_inner::sse::SseDecoder;
use crate::utils_inner::stream_failure::{
    format_stream_failure_message, record_stream_failure, stream_failure_from_stop_reason,
    ProviderError,
};

pub const API_AZURE_OPENAI_RESPONSES: &str = "azure-openai-responses";

const DEFAULT_AZURE_API_VERSION: &str = "v1";

fn parse_deployment_name_map(value: Option<&str>) -> HashMap<String, String> {
    let mut map = HashMap::new();
    let Some(value) = value else {
        return map;
    };
    for entry in value.split(',') {
        let trimmed = entry.trim();
        if trimmed.is_empty() {
            continue;
        }
        let (model_id, deployment_name) = trimmed.split_once('=').unwrap_or(("", ""));
        if model_id.is_empty() || deployment_name.is_empty() {
            continue;
        }
        map.insert(
            model_id.trim().to_string(),
            deployment_name.trim().to_string(),
        );
    }
    map
}

fn resolve_deployment_name(model: &Model, options: Option<&AzureOpenAIResponsesOptions>) -> String {
    if let Some(deployment) = options.and_then(|options| options.azure_deployment_name.as_ref()) {
        return deployment.clone();
    }
    let mapped = parse_deployment_name_map(
        std::env::var("AZURE_OPENAI_DEPLOYMENT_NAME_MAP")
            .ok()
            .as_deref(),
    )
    .get(&model.id)
    .cloned();
    mapped.unwrap_or_else(|| model.id.clone())
}

/// Provider-native options: the shared Responses options plus the Azure endpoint overrides.
#[derive(Clone, Default)]
pub struct AzureOpenAIResponsesOptions {
    pub responses: ResponsesOptions,
    pub azure_api_version: Option<String>,
    pub azure_resource_name: Option<String>,
    pub azure_base_url: Option<String>,
    pub azure_deployment_name: Option<String>,
}

impl AzureOpenAIResponsesOptions {
    pub fn from_base(base: StreamOptions) -> Self {
        Self {
            responses: ResponsesOptions::from_base(base),
            ..Self::default()
        }
    }

    /// The options the transport table mapped, pinned to the resolved deployment.
    pub(crate) fn from_responses(responses: ResponsesOptions, deployment: String) -> Self {
        Self {
            responses,
            azure_deployment_name: Some(deployment),
            ..Self::default()
        }
    }
}

fn normalize_azure_base_url(base_url: &str) -> Result<String, String> {
    let trimmed = base_url.trim().trim_end_matches('/');
    let url = url::Url::parse(trimmed)
        .map_err(|_| format!("Invalid Azure OpenAI base URL: {base_url}"))?;
    let host = url.host_str().unwrap_or_default();
    let is_azure_host =
        host.ends_with(".openai.azure.com") || host.ends_with(".cognitiveservices.azure.com");
    let normalized_path = url.path().trim_end_matches('/');

    // Ensure Azure hosts have /openai/v1 as base path so the deployment URL resolves as
    // <base>/deployments/<name>/responses?api-version=<v>.
    if is_azure_host && (normalized_path.is_empty() || normalized_path == "/openai") {
        let mut normalized = url.clone();
        normalized.set_path("/openai/v1");
        normalized.set_query(None);
        return Ok(normalized.to_string().trim_end_matches('/').to_string());
    }

    Ok(trimmed.to_string())
}

fn build_default_base_url(resource_name: &str) -> String {
    format!("https://{resource_name}.openai.azure.com/openai/v1")
}

fn resolve_azure_config(
    model: &Model,
    options: Option<&AzureOpenAIResponsesOptions>,
) -> Result<(String, String), String> {
    let api_version = options
        .and_then(|options| options.azure_api_version.clone())
        .or_else(|| std::env::var("AZURE_OPENAI_API_VERSION").ok())
        .unwrap_or_else(|| DEFAULT_AZURE_API_VERSION.to_string());

    let base_url = options
        .and_then(|options| options.azure_base_url.as_ref())
        .map(|url| url.trim().to_string())
        .or_else(|| {
            std::env::var("AZURE_OPENAI_BASE_URL")
                .ok()
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty())
        });
    let resource_name = options
        .and_then(|options| options.azure_resource_name.clone())
        .or_else(|| std::env::var("AZURE_OPENAI_RESOURCE_NAME").ok())
        .filter(|value| !value.is_empty());

    let mut resolved_base_url = base_url;
    if resolved_base_url.is_none() {
        if let Some(resource) = resource_name {
            resolved_base_url = Some(build_default_base_url(&resource));
        }
    }
    if resolved_base_url.is_none() && !model.base_url.is_empty() {
        resolved_base_url = Some(model.base_url.clone());
    }
    let Some(resolved_base_url) = resolved_base_url else {
        return Err(
            "Azure OpenAI base URL is required. Set AZURE_OPENAI_BASE_URL or AZURE_OPENAI_RESOURCE_NAME, or pass azureBaseUrl, azureResourceName, or model.baseUrl."
                .to_string(),
        );
    };

    Ok((normalize_azure_base_url(&resolved_base_url)?, api_version))
}

pub fn stream_azure_openai_responses(
    model: &Model,
    context: &Context,
    options: Option<&AzureOpenAIResponsesOptions>,
) -> AssistantMessageEventStream {
    let options = options.cloned();
    let model = model.clone();
    let context = context.clone();
    let (writer, reader) = create_assistant_message_event_stream();

    tokio::spawn(async move {
        let mut output = AssistantMessage {
            content: Vec::new(),
            api: API_AZURE_OPENAI_RESPONSES.to_string(),
            provider: model.provider.clone(),
            model: model.id.clone(),
            response_model: None,
            response_id: None,
            diagnostics: None,
            usage: Usage::default(),
            stop_reason: StopReason::Stop,
            stop_reason_raw: None,
            error_status: None,
            error_message: None,
            timestamp: now_ms(),
            rest: Map::default(),
        };

        let result = run_stream(&model, &context, options.as_ref(), &mut output, &writer).await;
        match result {
            Ok(()) => {
                writer.push(AssistantMessageEvent::Done {
                    reason: done_reason(output.stop_reason),
                    message: output,
                });
                writer.end(None);
            }
            Err(error) => {
                output.stop_reason = if error == ProviderError::Aborted {
                    StopReason::Aborted
                } else {
                    StopReason::Error
                };
                output.error_message = Some(format_stream_failure_message(&error));
                record_stream_failure(
                    (&model.provider, &model.id, &model.api),
                    &mut output,
                    &error,
                );
                writer.push(AssistantMessageEvent::Error {
                    reason: error_reason(output.stop_reason),
                    error: output.clone(),
                });
                writer.end(Some(output));
            }
        }
    });

    reader
}

// Long by design: mirrors the provider's stream shape.
#[allow(clippy::too_many_lines)]
async fn run_stream(
    model: &Model,
    context: &Context,
    options: Option<&AzureOpenAIResponsesOptions>,
    output: &mut AssistantMessage,
    writer: &AssistantMessageEventWriter,
) -> Result<(), ProviderError> {
    let options = options.cloned().unwrap_or_default();
    let deployment_name = resolve_deployment_name(model, Some(&options));

    let api_key = options
        .responses
        .base
        .api_key
        .clone()
        .or_else(|| get_env_api_key(&model.provider))
        .unwrap_or_default();
    let (base_url, api_version) =
        resolve_azure_config(model, Some(&options)).map_err(ProviderError::Message)?;

    let view = crate::compat::ModelView::of(model);
    let mut params = build_body(
        model,
        &view,
        context,
        &options.responses,
        &Flavor::Azure {
            deployment: deployment_name.clone(),
        },
    )
    .map_err(ProviderError::Message)?;
    if let Some(on_payload) = &options.responses.base.on_payload {
        if let Some(next) = on_payload(params.clone(), model) {
            params = next;
        }
    }

    let url = format!(
        "{}/deployments/{deployment_name}/responses?api-version={api_version}",
        base_url.trim_end_matches('/')
    );
    let mut headers: Vec<(String, String)> = Vec::new();
    for (name, value) in model.headers.iter().flatten() {
        headers.push((name.clone(), value.clone()));
    }
    if let Some(options_headers) = &options.responses.base.headers {
        for (name, value) in options_headers {
            headers.retain(|(existing, _)| !existing.eq_ignore_ascii_case(name));
            headers.push((name.clone(), value.clone()));
        }
    }
    headers.push(("api-key".into(), api_key));

    let body = params.to_string();
    // TS `rawRequestDump` (`finalizeErrorMessage` on a 400/413 rejection).
    let request_dump = RawHttpRequestDump {
        provider: model.provider.to_string(),
        api: model.api.to_string(),
        model: model.id.clone(),
        method: "POST".to_owned(),
        url: url.clone(),
        headers: headers.clone(),
        body: body.clone(),
    };
    let mut response: HttpResponse = send(RequestOptions {
        method: reqwest::Method::POST,
        url,
        headers,
        body: Some(body),
        signal: options.responses.base.signal.clone(),
        timeout_ms: options.responses.base.timeout_ms,
        connection: crate::utils_inner::stream_failure::ConnectionErrorProfile::Sdk,
        transport: crate::utils_inner::http::Transport::Http1,
    })
    .await?;

    if let Some(on_response) = &options.responses.base.on_response {
        on_response(
            crate::types::ProviderResponse {
                status: response.status,
                headers: response.headers.clone().into_iter().collect(),
            },
            model,
        );
    }
    let request_id = response.headers.get("x-request-id").cloned();

    if response.status >= 400 {
        let body = response.read_all_text().await.unwrap_or_default();
        let error =
            ProviderError::from_http_status_body(response.status, &body, response.headers.clone());
        return Err(finalize_http_error(error, true, Some(&request_dump)));
    }

    writer.push(AssistantMessageEvent::Start {
        partial: output.clone(),
    });

    let stream_result: Result<(), ProviderError>;
    {
        let hooks = ResponsesStreamHooks::default();
        let mut processor =
            crate::providers::openai_responses_shared::ResponsesStreamProcessor::new(
                model, output, writer, hooks,
            );
        // The TS try/catch encloses the streaming section and the abort and stop-reason checks; the
        // catch settles partial tool calls before the error event carries the message.
        stream_result = async {
            let mut decoder = SseDecoder::new();
            loop {
                let Some(chunk) = response.next_text().await? else {
                    break;
                };
                for sse in decoder.push_text(&chunk) {
                    if sse.data.trim().is_empty() {
                        continue;
                    }
                    let event = match parse_json_with_repair(&sse.data) {
                        Ok(event) => event,
                        Err(_) => parse_streaming_json(Some(&sse.data)),
                    };
                    processor.handle_event(&event)?;
                }
            }
            for sse in decoder.finish() {
                if sse.data.trim().is_empty() {
                    continue;
                }
                let event = match parse_json_with_repair(&sse.data) {
                    Ok(event) => event,
                    Err(_) => parse_streaming_json(Some(&sse.data)),
                };
                processor.handle_event(&event)?;
            }
            processor.finish()?;
            if options
                .responses
                .base
                .signal
                .as_ref()
                .is_some_and(tokio_util::sync::CancellationToken::is_cancelled)
            {
                return Err(ProviderError::Aborted);
            }
            if matches!(
                processor.stop_reason(),
                StopReason::Aborted | StopReason::Error
            ) {
                return Err(ProviderError::StreamFailure(
                    stream_failure_from_stop_reason(
                        processor.stop_reason_raw(),
                        request_id.as_deref(),
                    ),
                ));
            }
            Ok(())
        }
        .await;
        if stream_result.is_err() {
            processor.settle_partial_tool_calls();
        }
    }
    stream_result?;

    Ok(())
}

pub fn stream_simple_azure_openai_responses(
    model: &Model,
    context: &Context,
    options: Option<&SimpleStreamOptions>,
) -> AssistantMessageEventStream {
    crate::transport::stream_simple_for_api(model, context, options)
}

/// Registry provider for the `azure-openai-responses` API.
pub struct AzureOpenAIResponsesProvider;

impl Provider for AzureOpenAIResponsesProvider {
    fn api(&self) -> &str {
        API_AZURE_OPENAI_RESPONSES
    }

    fn stream(
        &self,
        model: &Model,
        context: &Context,
        options: Option<&StreamOptions>,
    ) -> AssistantMessageEventStream {
        let options = options.map(|base| AzureOpenAIResponsesOptions::from_base(base.clone()));
        stream_azure_openai_responses(model, context, options.as_ref())
    }

    fn stream_simple(
        &self,
        model: &Model,
        context: &Context,
        options: Option<&SimpleStreamOptions>,
    ) -> AssistantMessageEventStream {
        stream_simple_azure_openai_responses(model, context, options)
    }
}
