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

/// Sum token estimates over text fragments without collecting them: identical
/// results to buffering `String`s, but no per-message text is cloned (this runs
/// over the whole session context at every post-turn threshold check).
fn content_blocks(content: &Value, total: &mut u64, extra: &mut u64, t: &dyn Tokenizer) {
    match content {
        Value::String(s) => *total += t.count_text(s),
        Value::Array(blocks) => {
            for block in blocks {
                match str_field(block, "type") {
                    Some("text") => {
                        if let Some(text) = str_field(block, "text").filter(|s| !s.is_empty()) {
                            *total += t.count_text(text);
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
    let mut total = 0u64;
    let mut extra = 0u64;
    match message_role(message) {
        "bashExecution" => {
            for key in ["command", "output"] {
                if let Some(s) = str_field(message, key) {
                    total += t.count_text(s);
                }
            }
            return total;
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
                        total += t.count_text(str_field(block, "text").unwrap_or(""));
                    }
                    Some("thinking") => {
                        total += t.count_text(str_field(block, "thinking").unwrap_or(""));
                        if let Some(sig) =
                            str_field(block, "thinkingSignature").filter(|s| !s.is_empty())
                        {
                            if !options.exclude_encrypted_reasoning {
                                total += t.count_text(sig);
                            }
                        }
                    }
                    Some("toolCall") => {
                        total += t.count_text(str_field(block, "name").unwrap_or(""));
                        total += t.count_text(
                            &block
                                .get("arguments")
                                .map_or_else(|| "null".to_owned(), json_string),
                        );
                    }
                    Some("redactedThinking") => {
                        if !options.exclude_encrypted_reasoning {
                            total += t.count_text(str_field(block, "data").unwrap_or(""));
                        }
                    }
                    Some("anthropicServerTool") if !options.exclude_encrypted_reasoning => {
                        total += t.count_text(
                            &block
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
                content_blocks(content, &mut total, &mut extra, t);
            }
        }
        role @ ("branchSummary" | "compactionSummary") => {
            total += t.count_text(str_field(message, "summary").unwrap_or(""));
            if role == "compactionSummary" {
                if let Some(blocks) = message.get("blocks").and_then(Value::as_array) {
                    for block in blocks {
                        if str_field(block, "type") == Some("text") {
                            total += t.count_text(str_field(block, "text").unwrap_or(""));
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
    extra + total
}

/// Sum of [`count_message`] over `messages`.
#[must_use]
pub fn count_messages(t: &dyn Tokenizer, messages: &[Value], options: CountOptions) -> u64 {
    messages.iter().map(|m| count_message(t, m, options)).sum()
}
