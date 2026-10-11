//! `cargo run -p omp-providers --example probe -- <model_ref> "<prompt>"`
use std::io::Write;

use futures::StreamExt;
use pa_ai::types::{AssistantMessageEvent, Context, Message, UserMessage, UserMessageContent};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let model_ref = args.next().ok_or_else(|| anyhow::anyhow!("usage: probe <model_ref> <prompt>"))?;
    let prompt = args.next().unwrap_or_else(|| "Say hi in five words.".into());
    let m = omp_providers::resolve(&model_ref)?;
    eprintln!(
        "[resolved] {}/{} api={} base_url={} key={} thinking={:?}",
        m.provider,
        m.id,
        m.api,
        m.base_url,
        if m.api_key.is_some() { "set" } else { "MISSING" },
        m.thinking
    );
    let ctx = Context {
        system_prompt: None,
        messages: vec![Message::User(UserMessage {
            content: UserMessageContent::Text(prompt),
            ..serde_json::from_value(serde_json::json!({"role":"user","content":"","timestamp":0}))?
        })],
        tools: None,
    };
    let mut s = omp_providers::stream(&m, ctx, None).await?;
    let mut out = std::io::stdout();
    while let Some(ev) = s.next().await {
        match ev {
            AssistantMessageEvent::TextDelta { delta, .. } => {
                write!(out, "{delta}")?;
                out.flush()?;
            }
            AssistantMessageEvent::Done { message, .. } => {
                println!("\n[usage] {}", serde_json::to_string(&message.usage)?);
            }
            AssistantMessageEvent::Error { error, .. } => {
                println!(
                    "\n[error] stop={:?} {}",
                    error.stop_reason,
                    error.error_message.as_deref().unwrap_or("")
                );
            }
            _ => {}
        }
    }
    Ok(())
}
