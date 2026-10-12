//! Token estimation (pi-agent-core `Tokenizer.countMessage`).
//!
//! omp counts with a model-specific native tokenizer when the catalog names one and with a
//! `ceil(bytes / 4)` estimate otherwise (and always in tests). The estimate is the default here;
//! inject a [`Tokenizer`] for exact counts.

use serde_json::Value;

use crate::json::{message_role, str_field};

/// Counts tokens in one text fragment. Fragments of a message are counted independently and
/// summed, like the TS `countTokens(string[])`.
pub trait Tokenizer: Sync {
    fn count_text(&self, text: &str) -> u64;

    /// Estimated tokens for one image block. omp decodes the image header for its pixel size; the
    /// default charges the unknown-size patch budget (never undercounts).
    fn count_image(&self, block: &Value) -> u64 {
        match str_field(block, "detail") {
            Some("low") => 308,
            Some("high") => 3000,
            _ => 12_000,
        }
    }
}

/// `(utf8 bytes + 3) >> 2`: omp's fallback when no native tokenizer applies.
#[derive(Debug, Clone, Copy, Default)]
pub struct ByteEstimate;

impl Tokenizer for ByteEstimate {
    fn count_text(&self, text: &str) -> u64 {
        (text.len() as u64 + 3) >> 2
    }
}

/// Token cost of one snapcompact frame (`snapcompact.FRAME_TOKEN_ESTIMATE`).
const FRAME_TOKEN_ESTIMATE: u64 = 5024;

/// Options for [`count_message`] (`MessageCountOptions`).
#[derive(Debug, Clone, Copy, Default)]
pub struct CountOptions {
    /// Drop opaque provider reasoning payloads (signatures, redacted thinking, native server-tool
    /// blocks) from the estimate.
    pub exclude_encrypted_reasoning: bool,
}

fn json_string(v: &Value) -> String {
    omp_session::entries::to_js_string(v).unwrap_or_else(|_| "null".to_owned())
}

fn content_blocks(
    content: &Value,
    fragments: &mut Vec<String>,
    extra: &mut u64,
    t: &dyn Tokenizer,
) {
    match content {
        Value::String(s) => fragments.push(s.clone()),
        Value::Array(blocks) => {
            for block in blocks {
                match str_field(block, "type") {
                    Some("text") => {
                        if let Some(text) = str_field(block, "text").filter(|s| !s.is_empty()) {
                            fragments.push(text.to_owned());
                        }
                    }
                    Some("image") => *extra += t.count_image(block),
                    _ => {}
                }
            }
        }
        _ => {}
    }
}

/// Port of `Tokenizer.countMessage` over a JSON agent message (user, developer, assistant,
/// toolResult, custom, hookMessage, bashExecution, branchSummary, compactionSummary; any other
/// role counts as 0).
#[must_use]
pub fn count_message(t: &dyn Tokenizer, message: &Value, options: CountOptions) -> u64 {
    let mut fragments: Vec<String> = Vec::new();
    let mut extra = 0u64;
    let sum = |fragments: &[String]| fragments.iter().map(|f| t.count_text(f)).sum::<u64>();
    match message_role(message) {
        "bashExecution" => {
            for key in ["command", "output"] {
                if let Some(s) = str_field(message, key) {
                    fragments.push(s.to_owned());
                }
            }
            return sum(&fragments);
        }
        "assistant" => {
            for block in message
                .get("content")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                match str_field(block, "type") {
                    Some("text") => {
                        fragments.push(str_field(block, "text").unwrap_or("").to_owned());
                    }
                    Some("thinking") => {
                        fragments.push(str_field(block, "thinking").unwrap_or("").to_owned());
                        if let Some(sig) =
                            str_field(block, "thinkingSignature").filter(|s| !s.is_empty())
                        {
                            if !options.exclude_encrypted_reasoning {
                                fragments.push(sig.to_owned());
                            }
                        }
                    }
                    Some("toolCall") => {
                        fragments.push(str_field(block, "name").unwrap_or("").to_owned());
                        fragments.push(
                            block
                                .get("arguments")
                                .map_or_else(|| "null".to_owned(), json_string),
                        );
                    }
                    Some("redactedThinking") => {
                        if !options.exclude_encrypted_reasoning {
                            fragments.push(str_field(block, "data").unwrap_or("").to_owned());
                        }
                    }
                    Some("anthropicServerTool") if !options.exclude_encrypted_reasoning => {
                        fragments.push(
                            block
                                .get("block")
                                .map_or_else(|| "null".to_owned(), json_string),
                        );
                    }
                    _ => {}
                }
            }
        }
        "user" | "developer" | "custom" | "hookMessage" | "toolResult" => {
            if let Some(content) = message.get("content") {
                content_blocks(content, &mut fragments, &mut extra, t);
            }
        }
        role @ ("branchSummary" | "compactionSummary") => {
            fragments.push(str_field(message, "summary").unwrap_or("").to_owned());
            if role == "compactionSummary" {
                if let Some(blocks) = message.get("blocks").and_then(Value::as_array) {
                    for block in blocks {
                        if str_field(block, "type") == Some("text") {
                            fragments.push(str_field(block, "text").unwrap_or("").to_owned());
                        } else {
                            extra += FRAME_TOKEN_ESTIMATE;
                        }
                    }
                } else if let Some(images) = message.get("images").and_then(Value::as_array) {
                    extra += images.len() as u64 * FRAME_TOKEN_ESTIMATE;
                }
            }
        }
        _ => return 0,
    }
    extra + sum(&fragments)
}

/// Sum of [`count_message`] over `messages`.
#[must_use]
pub fn count_messages(t: &dyn Tokenizer, messages: &[Value], options: CountOptions) -> u64 {
    messages.iter().map(|m| count_message(t, m, options)).sum()
}
