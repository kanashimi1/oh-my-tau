//! `OpenAI` Responses API streaming provider: session-affinity headers, prompt-cache retention,
//! reasoning params with encrypted-content include, service-tier pricing, and the shared Responses
//! stream processor.

use serde_json::Map;

use crate::env_api_keys::get_env_api_key;
use crate::event_stream::{
    create_assistant_message_event_stream, AssistantMessageEvent, AssistantMessageEventStream,
    AssistantMessageEventWriter,
};
use crate::providers::openai_responses_shared::{apply_service_tier_pricing, ResponsesStreamHooks};
use crate::providers::responses_request::{
    build_body, responses_headers, Flavor, ResponsesOptions,
};
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

pub const API_OPENAI_RESPONSES: &str = "openai-responses";

/// Provider-native options (see [`ResponsesOptions`]).
pub type OpenAIResponsesOptions = ResponsesOptions;

pub fn stream_openai_responses(
    model: &Model,
    context: &Context,
    options: Option<&OpenAIResponsesOptions>,
) -> AssistantMessageEventStream {
    let options = options.cloned();
    let model = model.clone();
    let context = context.clone();
    let (writer, reader) = create_assistant_message_event_stream();

    tokio::spawn(async move {
        let mut output = AssistantMessage {
            content: Vec::new(),
            api: model.api.clone(),
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
    options: Option<&OpenAIResponsesOptions>,
    output: &mut AssistantMessage,
    writer: &AssistantMessageEventWriter,
) -> Result<(), ProviderError> {
    let options = options.cloned().unwrap_or_default();
    let api_key = options
        .base
        .api_key
        .clone()
        .or_else(|| get_env_api_key(&model.provider))
        .unwrap_or_default();
    let view = crate::compat::ModelView::of(model);
    let mut params = build_body(model, &view, context, &options, &Flavor::OpenAi)
        .map_err(ProviderError::Message)?;
    if let Some(on_payload) = &options.base.on_payload {
        if let Some(next) = on_payload(params.clone(), model) {
            params = next;
        }
    }

    let url = format!("{}/responses", model.base_url.trim_end_matches('/'));
    let headers = responses_headers(
        model,
        &view,
        &api_key,
        options.base.headers.as_ref(),
        options.base.session_id.as_deref(),
    );
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
        signal: options.base.signal.clone(),
        timeout_ms: options.base.timeout_ms,
        connection: crate::utils_inner::stream_failure::ConnectionErrorProfile::Sdk,
        transport: crate::utils_inner::http::Transport::Http1,
    })
    .await?;

    if let Some(on_response) = &options.base.on_response {
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

    let model_id = model.id.clone();
    let hooks = ResponsesStreamHooks {
        request_service_tier: options.base.service_tier,
        resolve_service_tier: None,
        apply_service_tier_pricing: Some(Box::new(move |usage, service_tier| {
            apply_service_tier_pricing(usage, service_tier.as_deref(), &model_id);
        })),
    };

    let stream_result: Result<(), ProviderError>;
    {
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

pub fn stream_simple_openai_responses(
    model: &Model,
    context: &Context,
    options: Option<&SimpleStreamOptions>,
) -> AssistantMessageEventStream {
    crate::transport::stream_simple_for_api(model, context, options)
}

/// Registry provider for the `openai-responses` API.
pub struct OpenAIResponsesProvider;

impl Provider for OpenAIResponsesProvider {
    fn api(&self) -> &str {
        API_OPENAI_RESPONSES
    }

    fn stream(
        &self,
        model: &Model,
        context: &Context,
        options: Option<&StreamOptions>,
    ) -> AssistantMessageEventStream {
        let options = options.map(|base| OpenAIResponsesOptions::from_base(base.clone()));
        stream_openai_responses(model, context, options.as_ref())
    }

    fn stream_simple(
        &self,
        model: &Model,
        context: &Context,
        options: Option<&SimpleStreamOptions>,
    ) -> AssistantMessageEventStream {
        stream_simple_openai_responses(model, context, options)
    }
}
