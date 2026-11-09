//! Live per-turn token tally for the loader row.

use serde_json::Value;

/// Live per-turn token accounting for the loader row (TS mirrors the
/// event-controller's usage tally, but reported while streaming).
///
/// Output/thinking tokens are estimated from the streamed deltas (chars/4)
/// until the provider's final `usage` arrives on the message frame, then
/// exact. Input comes from the request usage or the `context_usage`
/// estimate.
#[derive(Debug, Default, Clone)]
pub(super) struct TurnUsage {
    /// Streamed char high-water marks per message id: `(text, thinking)`.
    /// `message_update` frames carry the accumulated text, so the max per
    /// id is the message total; summing across ids spans a whole turn.
    streamed: std::collections::HashMap<String, (u64, u64)>,
    /// Same marks for frames whose message has no id.
    anonymous: (u64, u64),
    /// Additive delta tallies per message id: `message_update` frames sent
    /// without a full snapshot (`delta_updates` mode) carry only the
    /// `assistantMessageEvent` delta string, so those are summed, not maxed.
    delta_streamed: std::collections::HashMap<String, (u64, u64)>,
    /// Same additive tallies for delta frames whose message has no id.
    delta_anonymous: (u64, u64),
    /// Provider-reported usage from the latest assistant frame.
    exact: Option<ExactUsage>,
}

/// Provider-reported usage (`message.usage` on assistant frames).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct ExactUsage {
    input: u64,
    output: u64,
    cache_read: u64,
    cache_write: u64,
}

const CHARS_PER_TOKEN: u64 = 4;

impl TurnUsage {
    /// Clear all per-turn tallies (turn start).
    pub(super) fn reset(&mut self) {
        *self = Self::default();
    }

    /// Fold one assistant message frame (streaming partial or settled).
    pub(super) fn observe_message(&mut self, message: &Value) {
        let mut text = 0u64;
        let mut think = 0u64;
        if let Some(content) = message.get("content").and_then(Value::as_array) {
            for part in content {
                match part.get("type").and_then(Value::as_str) {
                    Some("text") => text += str_chars(part.get("text")),
                    Some("thinking") => think += str_chars(part.get("thinking")),
                    _ => {}
                }
            }
        }
        let slot = match message.get("id").and_then(Value::as_str) {
            Some(id) if !id.is_empty() => self.streamed.entry(id.to_owned()).or_default(),
            _ => &mut self.anonymous,
        };
        slot.0 = slot.0.max(text);
        slot.1 = slot.1.max(think);
        // A cumulative snapshot supersedes any deltas tallied for this message
        // (`delta_updates` mode never sends both on one frame, and the settled
        // `message_end` snapshot that closes a stream is the complete total).
        self.delta_streamed.clear();
        self.delta_anonymous = (0, 0);
        // Streaming partials carry the provider's running `usage` from the first
        // chunk (e.g. output 9) and would freeze the live count; only a settled
        // message (`stopReason`) or a bare usage frame is authoritative.
        let settled = message.get("stopReason").is_some() || message.get("content").is_none();
        if let Some(usage) = message.get("usage").filter(|_| settled) {
            let get = |k: &str| usage.get(k).and_then(Value::as_u64).unwrap_or(0);
            let exact = ExactUsage {
                input: get("input"),
                output: get("output"),
                cache_read: get("cacheRead"),
                cache_write: get("cacheWrite"),
            };
            if exact.input > 0 || exact.output > 0 {
                self.exact = Some(exact);
            }
        }
    }
    /// Fold one streaming delta frame (`delta_updates` mode): the runtime
    /// stripped the cumulative `message` snapshot, leaving only
    /// `assistantMessageEvent.delta`. These add (never max) into per-message
    /// tallies; a later cumulative snapshot resets them (see `observe_message`).
    pub(super) fn observe_delta(&mut self, frame: &Value) {
        let mut text = 0u64;
        let mut think = 0u64;
        if let Some(ev) = frame.get("assistantMessageEvent") {
            match ev.get("type").and_then(Value::as_str) {
                Some("text_delta") => text = str_chars(ev.get("delta")),
                Some("thinking_delta") => think = str_chars(ev.get("delta")),
                _ => return,
            }
        } else {
            return;
        }
        let slot = match frame.get("messageId").and_then(Value::as_str) {
            Some(id) if !id.is_empty() => self.delta_streamed.entry(id.to_owned()).or_default(),
            _ => &mut self.delta_anonymous,
        };
        slot.0 += text;
        slot.1 += think;
    }

    /// Estimated output tokens so far (text + thinking chars/4) or the exact
    /// provider count. Thinking is included so the live `↓` ticks up while
    /// the model reasons (the `(think N)` suffix still breaks it out).
    pub(super) fn output_tokens(&self) -> u64 {
        if let Some(exact) = self.exact {
            return exact.output;
        }
        self.estimated_output()
    }

    /// Estimated thinking tokens (chars/4); providers do not report these.
    pub(super) fn thinking_tokens(&self) -> u64 {
        self.estimated_thinking()
    }

    /// Input tokens: the provider's request usage, else the `context_usage`
    /// estimate for the active turn.
    pub(super) fn input_tokens(&self, context_estimate: Option<u64>) -> u64 {
        match self.exact {
            Some(exact) if exact.input > 0 => exact.input + exact.cache_read + exact.cache_write,
            _ => context_estimate.unwrap_or(0),
        }
    }

    /// Estimated turn cost in dollars from $/M pricing, `None` when free.
    #[allow(clippy::cast_precision_loss, clippy::unused_self)] // token counts are far below 2^52; keeps the method call shape
    pub(super) fn cost(
        &self,
        input: u64,
        output: u64,
        in_per_m: f64,
        out_per_m: f64,
    ) -> Option<f64> {
        let cost = input as f64 / 1e6 * in_per_m + output as f64 / 1e6 * out_per_m;
        (cost > 0.0).then_some(cost)
    }

    pub(super) fn estimated_output(&self) -> u64 {
        let (text, think) = self.totals();
        (text + think) / CHARS_PER_TOKEN
    }

    pub(super) fn estimated_thinking(&self) -> u64 {
        let (_, think) = self.totals();
        think / CHARS_PER_TOKEN
    }

    pub(super) fn totals(&self) -> (u64, u64) {
        let mut acc = self.anonymous;
        for (text, think) in self.streamed.values() {
            acc.0 += *text;
            acc.1 += *think;
        }
        acc.0 += self.delta_anonymous.0;
        acc.1 += self.delta_anonymous.1;
        for (text, think) in self.delta_streamed.values() {
            acc.0 += *text;
            acc.1 += *think;
        }
        acc
    }
}

/// Character count of an optional JSON string field.
fn str_chars(value: Option<&Value>) -> u64 {
    value
        .and_then(Value::as_str)
        .map_or(0, |s| s.chars().count() as u64)
}

/// Compact token count (`1234` → `1.2k`, `18200` → `18.2k`).
#[allow(clippy::cast_precision_loss)] // display rounding to one decimal
pub(super) fn format_tokens(n: u64) -> String {
    if n < 1_000 {
        n.to_string()
    } else if n < 1_000_000 {
        format!("{:.1}k", n as f64 / 1_000.0)
    } else {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    }
}
