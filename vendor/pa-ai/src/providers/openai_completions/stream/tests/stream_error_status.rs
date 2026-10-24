//! The HTTP status of a rejected request is carried on the failed message
//! (TS `errorStatus`), so consumers classify retryability without parsing text.

use super::*;

/// Serve one non-streaming error response with `status_line` and `body`.
async fn serve_error(status_line: &'static str, body: &'static str) -> SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut request = vec![0u8; 8192];
        let _ = socket.read(&mut request).await.unwrap();
        let response = format!(
            "HTTP/1.1 {status_line}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        socket.write_all(response.as_bytes()).await.unwrap();
    });
    addr
}

/// Stream the provider against an error response and return the terminal error message.
async fn failed_message(status_line: &'static str, body: &'static str) -> AssistantMessage {
    let addr = serve_error(status_line, body).await;
    let mut model = completions_model("gpt-5.5", "openai", 1.0, 1.0);
    model["baseUrl"] = json!(format!("http://{addr}"));
    let model: Model = serde_json::from_value(model).unwrap();
    let options = OpenAICompletionsOptions::from_base(crate::types::StreamOptions {
        api_key: Some("test".into()),
        ..Default::default()
    });
    let mut reader = stream_openai_completions(
        &model,
        &Context {
            system_prompt: None,
            messages: vec![],
            tools: None,
        },
        Some(&options),
    );
    loop {
        let event = reader.next_event().await.unwrap();
        if let AssistantMessageEvent::Error { error, .. } = event {
            return error;
        }
        assert!(
            !matches!(event, AssistantMessageEvent::Done { .. }),
            "an HTTP error must not complete"
        );
    }
}

#[tokio::test]
async fn rate_limited_request_records_http_status() {
    let message = failed_message(
        "429 Too Many Requests",
        r#"{"error":{"message":"Rate limit reached","type":"rate_limit_exceeded"}}"#,
    )
    .await;
    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(message.error_status, Some(429));
    assert!(message
        .error_message
        .as_deref()
        .is_some_and(|text| text.contains("Rate limit reached")));
}

#[tokio::test]
async fn rejected_request_records_http_status() {
    let message = failed_message(
        "400 Bad Request",
        r#"{"error":{"message":"Invalid value: 'x'.","type":"invalid_request_error"}}"#,
    )
    .await;
    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(message.error_status, Some(400));
}

#[test]
fn error_status_round_trips_as_ts_error_status_key() {
    let mut message =
        crate::event_stream::initial_assistant_message("openai-completions", "openai", "gpt-5.5");
    message.stop_reason = StopReason::Error;
    message.error_message = Some("Rate limit reached".into());
    message.error_status = Some(429);

    let value = serde_json::to_value(&message).unwrap();
    assert_eq!(value["errorStatus"], json!(429));
    let back: AssistantMessage = serde_json::from_value(value).unwrap();
    assert_eq!(back.error_status, Some(429));

    // A session line TS wrote carries the key; an unset status stays absent.
    let ts_line = json!({
        "role": "assistant",
        "content": [],
        "api": "openai-completions",
        "provider": "openai",
        "model": "gpt-5.5",
        "usage": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 0,
                  "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0}},
        "stopReason": "error",
        "errorMessage": "429 status code",
        "errorStatus": 429,
        "timestamp": 1
    });
    let parsed: AssistantMessage = serde_json::from_value(ts_line).unwrap();
    assert_eq!(parsed.error_status, Some(429));
    let reparsed = serde_json::to_value(&parsed).unwrap();
    assert_eq!(reparsed["errorStatus"], json!(429));

    message.error_status = None;
    let unset = serde_json::to_value(&message).unwrap();
    assert!(unset.get("errorStatus").is_none());
}
