//! `cargo run -p omp-providers --example stream_compare [-- <model_ref> [<reasoning>]]`
//! One short prompt; prints the normalized event sequence (kinds, no text) in the
//! same format as `stream_compare_ts.ts`.
use futures::StreamExt;
use pa_ai::types::{AssistantMessageEvent, Context, Message, UserMessage, UserMessageContent};
use serde_json::Value;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let model_ref = args.next().unwrap_or_else(|| "zai/glm-5.3-flash".into());
    let mut m = omp_providers::resolve(&model_ref)?;
    if let Some(level) = args.next() {
        m.thinking = serde_json::from_value(Value::String(level))?;
    }
    println!(
        "model: {}/{} api={} reasoning={:?}",
        m.provider, m.id, m.api, m.thinking
    );
    let ctx = Context {
        system_prompt: None,
        messages: vec![Message::User(UserMessage {
            content: UserMessageContent::Text("Reply with the single word: ok".into()),
            ..serde_json::from_value(serde_json::json!({"role":"user","content":"","timestamp":0}))?
        })],
        tools: None,
    };
    let mut s = omp_providers::stream(&m, ctx, None).await?;
    let mut seq: Vec<String> = Vec::new();
    let mut last: Option<Value> = None;
    while let Some(ev) = s.next().await {
        let v = serde_json::to_value(&ev)?;
        let kind = v["type"].as_str().unwrap_or("?").to_string();
        let tag = if matches!(
            ev,
            AssistantMessageEvent::Done { .. } | AssistantMessageEvent::Error { .. }
        ) {
            kind
        } else if kind.ends_with("_delta") {
            format!("{kind}+")
        } else {
            kind
        };
        if seq.last() != Some(&tag) || !tag.ends_with('+') {
            seq.push(tag);
        }
        last = Some(v);
    }
    println!("events: {}", seq.join(" "));
    if let Some(v) = last {
        let msg = v
            .get("message")
            .or_else(|| v.get("error"))
            .cloned()
            .unwrap_or(Value::Null);
        let blocks: Vec<&str> = msg["content"]
            .as_array()
            .map(|a| a.iter().filter_map(|b| b["type"].as_str()).collect())
            .unwrap_or_default();
        println!("blocks: {}", blocks.join(","));
        println!("stopReason: {}", msg["stopReason"].as_str().unwrap_or("?"));
        let u = &msg["usage"];
        let keys: Vec<String> = ["input", "output", "cacheRead", "cacheWrite", "totalTokens"]
            .iter()
            .map(|k| {
                format!(
                    "{k}{}",
                    if u[*k].as_u64().unwrap_or(0) > 0 {
                        ">0"
                    } else {
                        "=0"
                    }
                )
            })
            .collect();
        println!("usage: {}", keys.join(" "));
        println!(
            "errorMessage: {}",
            msg["errorMessage"].as_str().map_or("none", |_| "present")
        );
    }
    Ok(())
}
