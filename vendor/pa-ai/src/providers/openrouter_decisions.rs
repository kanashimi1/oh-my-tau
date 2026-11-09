//! OpenRouter decisions transport (TS api id `openrouter-decisions`): the typed-judgment
//! protocol of `POST {baseUrl}/decisions` (e.g. `https://openrouter.ai/api/alpha/decisions`
//! for TypeSafe jev and `inception/mercury-decide[:free]`). Same wire contract as systemone —
//! one `state` plus a schema of typed `questions`, probabilities over options in one forward
//! pass — so this mirrors `systemone.rs` with the `/decisions` path and the `model` field
//! forwarded from the request body verbatim.

use crate::event_stream::{
    create_assistant_message_event_stream, AssistantMessageEventStream, AssistantMessageEventWriter,
};
use crate::types::{Context, Model, SimpleStreamOptions, StreamOptions};
use crate::Provider;
use pa_types::ai::{AssistantContentBlock, AssistantMessage, StopReason, TextContent, Usage};
use serde_json::Value;

/// One request round-trip's wall-clock bound when the caller sets none.
const DEFAULT_TIMEOUT_MS: u64 = 30_000;
/// The error body's bounded excerpt.
const ERROR_BODY_CHARS: usize = 500;

pub struct OpenRouterDecisionsProvider;

impl Provider for OpenRouterDecisionsProvider {
    fn api(&self) -> &'static str {
        "openrouter-decisions"
    }

    fn stream(
        &self,
        model: &Model,
        context: &Context,
        options: Option<&StreamOptions>,
    ) -> AssistantMessageEventStream {
        let (writer, reader) = create_assistant_message_event_stream();
        let model = model.clone();
        let request = decisions_request_body(context);
        let options = options.cloned();
        tokio::spawn(async move {
            serve_decisions(&writer, &model, request, options.as_ref()).await;
        });
        reader
    }

    fn stream_simple(
        &self,
        model: &Model,
        context: &Context,
        options: Option<&SimpleStreamOptions>,
    ) -> AssistantMessageEventStream {
        self.stream(model, context, options.map(|options| &options.base))
    }
}

/// The decision request body: the newest user message's JSON text.
fn decisions_request_body(context: &Context) -> Value {
    context
        .messages
        .iter()
        .rev()
        .find_map(|message| match message {
            pa_types::ai::Message::User(user) => {
                let text = user.content.text();
                serde_json::from_str::<Value>(&text).ok()
            }
            _ => None,
        })
        .unwrap_or(Value::Null)
}

/// One POST to the decisions endpoint; the reply envelope (or the surfaced
/// HTTP error) ends the stream as the final assistant message.
async fn serve_decisions(
    writer: &AssistantMessageEventWriter,
    model: &Model,
    request: Value,
    options: Option<&StreamOptions>,
) {
    let timeout_ms = options
        .and_then(|options| options.timeout_ms)
        .unwrap_or(DEFAULT_TIMEOUT_MS);
    let mut headers = reqwest::header::HeaderMap::new();
    headers.insert(
        reqwest::header::CONTENT_TYPE,
        "application/json".parse().unwrap(),
    );
    if let Some(api_key) = options.and_then(|options| options.api_key.as_deref()) {
        if let Ok(value) = reqwest::header::HeaderValue::from_str(&format!("Bearer {api_key}")) {
            headers.insert(reqwest::header::AUTHORIZATION, value);
        }
    }
    for (name, value) in options
        .and_then(|options| options.headers.as_ref())
        .into_iter()
        .flatten()
    {
        if let (Ok(name), Ok(value)) = (
            reqwest::header::HeaderName::from_bytes(name.as_bytes()),
            reqwest::header::HeaderValue::from_str(value),
        ) {
            headers.insert(name, value);
        }
    }
    let url = format!("{}/decisions", model.base_url.trim_end_matches('/'));
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_millis(timeout_ms))
        .default_headers(headers)
        .build()
        .expect("openrouter-decisions client builds");
    let response = client.post(&url).body(request.to_string()).send().await;
    let message = match response {
        Ok(response) => {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            // The decisions endpoint answers a malformed request with HTTP 400
            // AND a JSON error envelope the agent can reason about; surface the
            // body either way so the harness sees the server-reported cause.
            if status.is_success() || body.trim_start().starts_with('{') {
                envelope_message(model, body)
            } else {
                let detail: String = body.chars().take(ERROR_BODY_CHARS).collect();
                let mut message = error_message(
                    model,
                    &format!("decisions returned HTTP {status}: {detail}"),
                );
                message.error_status = Some(status.as_u16());
                message
            }
        }
        Err(error) => error_message(model, &format!("the decisions request failed: {error}")),
    };
    writer.end(Some(message));
}

/// The reply envelope as the final assistant message.
fn envelope_message(model: &Model, body: String) -> AssistantMessage {
    let mut message = crate::event_stream::initial_assistant_message(
        "openrouter-decisions",
        &model.provider,
        &model.id,
    );
    message
        .content
        .push(AssistantContentBlock::Text(TextContent {
            text: body,
            text_signature: None,
            rest: serde_json::Map::default(),
        }));
    message.usage = Usage::default();
    message
}

/// The surfaced transport error as the final assistant message.
fn error_message(model: &Model, error: &str) -> AssistantMessage {
    let mut message = crate::event_stream::initial_assistant_message(
        "openrouter-decisions",
        &model.provider,
        &model.id,
    );
    message.stop_reason = StopReason::Error;
    message.error_message = Some(error.to_string());
    message
}

#[cfg(test)]
mod tests {
    use super::*;
    use pa_types::ai::{Message, UserContent, UserMessage};
    use serde_json::json;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    #[derive(Debug, Clone)]
    struct Seen {
        path: String,
        authorization: Option<String>,
        body: Value,
    }

    /// A loopback server answering POST `<base>/decisions`.
    async fn serve(
        status: u16,
        reply: Value,
        seen: std::sync::Arc<std::sync::Mutex<Vec<Seen>>>,
    ) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move {
            while let Ok((mut socket, _)) = listener.accept().await {
                let mut raw = Vec::new();
                let mut chunk = [0u8; 4096];
                let (head, body) = loop {
                    let read = socket.read(&mut chunk).await.unwrap();
                    assert_ne!(read, 0, "the client closed mid-request");
                    raw.extend_from_slice(&chunk[..read]);
                    let text = String::from_utf8_lossy(&raw).to_string();
                    let Some((head, body)) = text.split_once("\r\n\r\n") else {
                        continue;
                    };
                    let length = head
                        .lines()
                        .find_map(|l| l.to_ascii_lowercase().strip_prefix("content-length:").map(str::to_owned))
                        .and_then(|l| l.trim().parse::<usize>().ok());
                    match length {
                        Some(n) if body.len() >= n => break (head.to_string(), body[..n].to_string()),
                        None => break (head.to_string(), body.to_string()),
                        _ => continue,
                    }
                };
                seen.lock().unwrap().push(Seen {
                    path: head.lines().next().unwrap_or("").to_string(),
                    authorization: head
                        .lines()
                        .find(|l| l.to_ascii_lowercase().starts_with("authorization:"))
                        .map(str::to_owned),
                    body: serde_json::from_str(&body).unwrap_or(Value::Null),
                });
                let reply = reply.to_string();
                let response = format!(
                    "HTTP/1.1 {status} Status\r\ncontent-type: application/json\r\n\
                     content-length: {}\r\nconnection: close\r\n\r\n{reply}",
                    reply.len()
                );
                socket.write_all(response.as_bytes()).await.unwrap();
            }
        });
        base
    }

    fn fixture_model(base_url: &str) -> Model {
        serde_json::from_value(json!({
            "id": "inception/mercury-decide:free", "name": "Mercury Decide (free)",
            "api": "openrouter-decisions", "provider": "openrouter",
            "baseUrl": base_url, "reasoning": false, "input": ["text"],
            "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
            "contextWindow": 65536, "maxTokens": 58982,
        }))
        .unwrap()
    }

    fn decision_context(request: &Value) -> Context {
        Context {
            system_prompt: None,
            messages: vec![Message::User(UserMessage {
                content: UserContent::Text(request.to_string()),
                timestamp: 0,
                rest: serde_json::Map::default(),
            })],
            tools: None,
        }
    }

    #[tokio::test]
    async fn posts_the_decision_request_to_the_decisions_path_with_bearer_auth() {
        let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let envelope = json!({
            "model": "inception/mercury-decide-20260930",
            "answers": {"q": {"type": "noul", "noul": 0.9}},
            "usage": {"input_tokens": 42, "output_tokens": 1, "cost": 0.0},
        });
        let base = serve(200, envelope.clone(), std::sync::Arc::clone(&seen)).await;
        let model = fixture_model(&base);
        let request = json!({
            "model": "inception/mercury-decide:free",
            "state": "the harness asks one question",
            "questions": {"q": {"type": "noul", "instructions": "is it a good idea?"}}
        });
        let stream = OpenRouterDecisionsProvider.stream(
            &model,
            &decision_context(&request),
            Some(&StreamOptions {
                api_key: Some("test-key".to_string()),
                timeout_ms: Some(5_000),
                ..Default::default()
            }),
        );
        let message = stream.result().await;
        assert_eq!(message.stop_reason, StopReason::Stop);
        let text = message
            .content
            .iter()
            .find_map(|b| match b {
                AssistantContentBlock::Text(t) => Some(t.text.clone()),
                _ => None,
            })
            .unwrap_or_default();
        let parsed: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(parsed, envelope);
        let seen = seen.lock().unwrap();
        assert_eq!(seen.len(), 1);
        assert!(seen[0].path.contains("/decisions"), "{}", seen[0].path);
        assert_eq!(
            seen[0].authorization.as_deref(),
            Some("authorization: Bearer test-key")
        );
        assert_eq!(seen[0].body, request);
    }

    #[tokio::test]
    async fn surfaces_a_json_error_envelope_from_a_400_as_text() {
        let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let err = json!({"error": {"message": "JSON parsing failed", "code": 400}});
        let base = serve(400, err.clone(), std::sync::Arc::clone(&seen)).await;
        let model = fixture_model(&base);
        let stream = OpenRouterDecisionsProvider.stream(
            &model,
            &decision_context(&json!({"state": "x", "questions": {}})),
            Some(&StreamOptions {
                api_key: Some("k".to_string()),
                timeout_ms: Some(5_000),
                ..Default::default()
            }),
        );
        let message = stream.result().await;
        assert_eq!(message.stop_reason, StopReason::Stop);
        let text = message
            .content
            .iter()
            .find_map(|b| match b {
                AssistantContentBlock::Text(t) => Some(t.text.clone()),
                _ => None,
            })
            .unwrap_or_default();
        assert_eq!(serde_json::from_str::<Value>(&text).unwrap(), err);
    }
}
