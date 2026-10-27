//! Context-window usage accounting and the context-usage frames.

use omp_tui::magic::Ultra;
use pa_agent::types::{AgentMessage, AssistantContent, Message, StopReason};
use serde_json::{json, Value};
use std::sync::Arc;

use super::{level_name, message_text, Runtime};

impl Runtime {
    /// TS `ctx.getContextUsage()`: `{tokens, contextWindow, percent}` (None without a window).
    pub async fn context_usage_value(&self) -> Option<Value> {
        let (s, _) = self.agent.status().await;
        let base = prompt_overhead_chars(&s.system_prompt, &s.tools);
        let window = s.model.context_window;
        self.agent
            .with_messages(|m| context_usage(m.iter(), window, base))
            .await
    }

    /// The `context_usage` frame the TUI status line renders: active model
    /// (selector + display name), context tokens/percent/window and the
    /// auto-compaction gauge boundaries (TS `getContextUsage` +
    /// `getSessionCompactionBoundaries`).
    pub async fn context_usage_frame(&self) -> Value {
        let (s, _) = self.agent.status().await;
        let base = prompt_overhead_chars(&s.system_prompt, &s.tools);
        let window = s.model.context_window;
        let usage = self
            .agent
            .with_messages(|m| context_usage(m.iter(), window, base))
            .await;
        let mut frame = json!({
            "type": "context_usage",
            "model": format!("{}/{}", s.model.provider, s.model.id),
            "modelName": s.model.name,
            "contextWindow": window,
            "thinkingLevel": level_name(s.thinking_level),
            "thinkingLevels": self.supported_efforts(),
            "autoThinking": self.auto_thinking.load(std::sync::atomic::Ordering::Relaxed),
            "ultra": self.ultra().map(Ultra::name),
        });
        if let Some(Value::Object(u)) = usage {
            for (k, v) in u {
                frame[k.as_str()] = v;
            }
        }
        let compaction = crate::compact::load_settings();
        if window > 0 && compaction.enabled && compaction.strategy.as_deref() != Some("off") {
            let threshold = omp_compact::settings::resolve_threshold_tokens(window, &compaction);
            #[allow(clippy::cast_precision_loss)]
            if threshold > 0.0 && threshold <= window as f64 {
                let w = window as f64;
                frame["thresholdPercent"] = json!(threshold / w * 100.0);
                let async_enabled = omp_settings::Settings::load(&self.cwd)
                    .get_bool("compaction.asyncEnabled")
                    .unwrap_or(true);
                if async_enabled {
                    let lead = ((threshold * 0.125).floor()).clamp(8192.0, 32000.0);
                    frame["speculationPercent"] = json!((threshold - lead).max(0.0) / w * 100.0);
                }
            }
        }
        frame
    }

    /// Turn on `context_usage` frame emission (interactive TUI host).
    pub fn enable_context_usage_frames(&self) {
        self.ui_context_frames
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// Publish a fresh `context_usage` frame to the UI stream (off the
    /// caller's task: the agent's state lock must not be re-entered from an
    /// event callback).
    pub fn publish_context_usage(self: &Arc<Self>) {
        if !self
            .ui_context_frames
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            return;
        }
        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            return;
        };
        let rt = self.clone();
        handle.spawn(async move {
            let frame = rt.context_usage_frame().await;
            rt.send(frame);
        });
    }
}

/// Characters the system prompt and tool schemas add to every request.
pub(super) fn prompt_overhead_chars(
    system_prompt: &str,
    tools: &[Arc<dyn pa_agent::types::AgentTool>],
) -> usize {
    system_prompt.chars().count()
        + tools
            .iter()
            .map(|t| t.name().len() + t.description().len() + t.parameters().to_string().len())
            .sum::<usize>()
}

/// Rough token count of a message (omp `estimateTokens`: ~4 chars per token).
fn estimate_tokens(m: &AgentMessage) -> u64 {
    let chars = match m {
        AgentMessage::Standard(Message::Assistant(a)) => a
            .content
            .iter()
            .map(|c| match c {
                AssistantContent::Text(t) => t.text.chars().count(),
                AssistantContent::Thinking(t) => t.thinking.chars().count(),
                AssistantContent::ToolCall(c) => c.name.len() + c.arguments.to_string().len(),
            })
            .sum(),
        other => message_text(other).chars().count(),
    };
    u64::try_from(chars.div_ceil(4)).unwrap_or(u64::MAX)
}

/// omp `getContextUsage`: the last assistant usage plus an estimate of what followed it, or an
/// estimate of the whole request (`overhead_chars` = system prompt + tools) before any usage exists.
/// Last provider-reported usage plus an estimate of everything after it (or the whole
/// transcript + prompt overhead when no assistant reported usage). One reverse pass, so
/// it reads the transcript in place (`Agent::with_messages`).
pub(super) fn context_usage<'a>(
    messages: impl DoubleEndedIterator<Item = &'a AgentMessage>,
    window: u64,
    overhead_chars: usize,
) -> Option<Value> {
    if window == 0 {
        return None;
    }
    let mut trailing = 0u64;
    let mut reported = None;
    for m in messages.rev() {
        match m {
            AgentMessage::Standard(Message::Assistant(a))
                if a.usage.total_tokens > 0 && a.stop_reason != StopReason::Aborted =>
            {
                reported = Some(a.usage.total_tokens);
                break;
            }
            _ => trailing += estimate_tokens(m),
        }
    }
    let tokens = match reported {
        Some(total) => total + trailing,
        None => u64::try_from(overhead_chars.div_ceil(4)).unwrap_or(u64::MAX) + trailing,
    };
    #[allow(clippy::cast_precision_loss)]
    let percent = tokens as f64 / window as f64 * 100.0;
    Some(json!({"tokens": tokens, "contextWindow": window, "percent": percent}))
}

#[cfg(test)]
#[allow(clippy::disallowed_methods)] // tests: throwaway runtimes drive async code
mod context_usage_tests {
    use super::*;
    use serde_json::json;

    /// The previous slice-based implementation, kept as the equivalence oracle.
    fn reference(messages: &[AgentMessage], window: u64, overhead_chars: usize) -> Option<Value> {
        if window == 0 {
            return None;
        }
        let last_usage = messages.iter().rposition(|m| {
            matches!(m, AgentMessage::Standard(Message::Assistant(a)) if a.usage.total_tokens > 0 && a.stop_reason != StopReason::Aborted)
        });
        let tokens = match last_usage {
            Some(i) => {
                let AgentMessage::Standard(Message::Assistant(a)) = &messages[i] else {
                    return None;
                };
                a.usage.total_tokens + messages[i + 1..].iter().map(estimate_tokens).sum::<u64>()
            }
            None => {
                u64::try_from(overhead_chars.div_ceil(4)).unwrap_or(u64::MAX)
                    + messages.iter().map(estimate_tokens).sum::<u64>()
            }
        };
        #[allow(clippy::cast_precision_loss)]
        let percent = tokens as f64 / window as f64 * 100.0;
        Some(json!({"tokens": tokens, "contextWindow": window, "percent": percent}))
    }

    fn assistant(text: &str, total: u64, stop: &str) -> AgentMessage {
        let a: pa_agent::types::AssistantMessage = serde_json::from_value(json!({
            "content": [{"type": "text", "text": text}], "api": "openai-completions", "provider": "p", "model": "a",
            "usage": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "totalTokens": total,
                      "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0}},
            "stopReason": stop, "timestamp": 1
        }))
        .expect("assistant json");
        AgentMessage::Standard(Message::Assistant(a))
    }

    #[test]
    fn reverse_pass_matches_slice_reference() {
        let u = |t: &str| AgentMessage::user(t);
        let cases: Vec<Vec<AgentMessage>> = vec![
            vec![],
            vec![u("hello there")],
            vec![
                u("q"),
                assistant("a1", 120, "stop"),
                u("follow up question"),
            ],
            vec![
                u("q"),
                assistant("a1", 120, "stop"),
                assistant("a2", 0, "stop"),
                u("x"),
            ],
            vec![
                u("q"),
                assistant("a1", 120, "stop"),
                assistant("aborted", 999, "aborted"),
                u("y"),
            ],
            vec![assistant("only", 50, "stop")],
        ];
        for (k, msgs) in cases.iter().enumerate() {
            for window in [0u64, 1000] {
                assert_eq!(
                    context_usage(msgs.iter(), window, 37),
                    reference(msgs, window, 37),
                    "case {k} window {window}"
                );
            }
        }
    }
}
