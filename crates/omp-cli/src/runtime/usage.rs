//! Context-window usage accounting and the context-usage frames.

use omp_tui::magic::Ultra;
use pa_agent::types::{AgentMessage, AssistantContent, Message, StopReason};
use serde_json::{json, Value};
use std::sync::Arc;

use super::{level_name, lock, message_text, Runtime};

#[derive(Default)]
pub(super) struct UsageCache {
    overhead: Option<(
        (u64, crate::methodology::ToolProjection, u64),
        PromptOverhead,
    )>,
    history: Option<HistoryUsage>,
}

#[derive(Clone)]
struct HistoryUsage {
    revision: u64,
    window: u64,
    wire_chars: usize,
    catalog_chars: usize,
    usage: Value,
}

impl Runtime {
    /// Cache prompt serialization and transcript estimation independently of
    /// provider traffic. Agent transcript revisions catch appends, truncation,
    /// retries and compaction; the MCP revision catches lazy-schema loads.
    pub(super) async fn cached_context_usage(&self, window: u64) -> Option<Value> {
        let overhead = self
            .agent
            .with_prompt_revision(|system_prompt, tools, prompt_revision| {
                let key = (
                    prompt_revision,
                    self.tool_projection,
                    crate::tools::mcp::projection_revision(),
                );
                let mut cache = lock(&self.usage_cache);
                match &cache.overhead {
                    Some((cached_key, overhead)) if *cached_key == key => *overhead,
                    _ => {
                        let overhead = prompt_overhead(system_prompt, tools, self.tool_projection);
                        cache.overhead = Some((key, overhead));
                        overhead
                    }
                }
            })
            .await;
        self.agent
            .with_messages_revision(|messages, revision| {
                {
                    let cache = lock(&self.usage_cache);
                    if let Some(cached) = &cache.history {
                        if cached.revision == revision
                            && cached.window == window
                            && cached.wire_chars == overhead.wire_chars
                            && cached.catalog_chars == overhead.catalog_chars
                        {
                            return Some(cached.usage.clone());
                        }
                    }
                }
                let usage = context_usage(messages.iter(), window, overhead);
                if let Some(usage) = &usage {
                    lock(&self.usage_cache).history = Some(HistoryUsage {
                        revision,
                        window,
                        wire_chars: overhead.wire_chars,
                        catalog_chars: overhead.catalog_chars,
                        usage: usage.clone(),
                    });
                }
                usage
            })
            .await
    }

    /// TS `ctx.getContextUsage()`: `{tokens, contextWindow, percent}`.
    pub async fn context_usage_value(&self) -> Option<Value> {
        let (s, _) = self.agent.status().await;
        self.cached_context_usage(s.model.context_window).await
    }

    /// The `context_usage` frame the TUI status line renders.
    pub async fn context_usage_frame(&self) -> Value {
        let (s, _) = self.agent.status().await;
        let window = s.model.context_window;
        let usage = self.cached_context_usage(window).await;
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

    /// Publish a fresh frame without holding the caller's agent-state lock.
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

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct PromptOverhead {
    /// Characters serialized into the provider request: system prompt plus
    /// schemas exposed by the active tool projection.
    pub wire_chars: usize,
    /// Characters in every registered schema, including deferred tools.
    pub catalog_chars: usize,
}

pub(super) fn prompt_overhead(
    system_prompt: &str,
    tools: &[Arc<dyn pa_agent::types::AgentTool>],
    projection: crate::methodology::ToolProjection,
) -> PromptOverhead {
    let mut overhead = PromptOverhead {
        wire_chars: system_prompt.chars().count(),
        catalog_chars: 0,
    };
    for tool in tools {
        let chars =
            tool.name().len() + tool.description().len() + tool.parameters().to_string().len();
        overhead.catalog_chars += chars;
        if projection.includes(tool.name()) {
            overhead.wire_chars += chars;
        }
    }
    overhead
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
/// Last provider-reported usage plus an estimate of everything after it (or the whole
/// transcript + prompt overhead when no assistant reported usage). One reverse pass, so
/// it reads the transcript in place (`Agent::with_messages`).
pub(super) fn context_usage<'a>(
    messages: impl DoubleEndedIterator<Item = &'a AgentMessage>,
    window: u64,
    overhead: PromptOverhead,
) -> Option<Value> {
    if window == 0 {
        return None;
    }
    let mut trailing = 0u64;
    let mut resident = 0u64;
    let mut reported = None;
    for m in messages.rev() {
        let estimated = estimate_tokens(m);
        resident += estimated;
        if reported.is_none() {
            match m {
                AgentMessage::Standard(Message::Assistant(a))
                    if a.usage.total_tokens > 0 && a.stop_reason != StopReason::Aborted =>
                {
                    reported = Some(a.usage.total_tokens);
                }
                _ => trailing += estimated,
            }
        }
    }
    let tokens = match reported {
        Some(total) => total + trailing,
        None => u64::try_from(overhead.wire_chars.div_ceil(4)).unwrap_or(u64::MAX) + trailing,
    };
    #[allow(clippy::cast_precision_loss)]
    let percent = tokens as f64 / window as f64 * 100.0;
    Some(json!({
        "tokens": tokens,
        "contextWindow": window,
        "percent": percent,
        "wireTokens": tokens,
        "residentHistoryTokens": resident,
        "availableCatalogTokens": u64::try_from(overhead.catalog_chars.div_ceil(4)).unwrap_or(u64::MAX),
        "source": if reported.is_some() { "provider+estimate" } else { "estimate" },
    }))
}

#[cfg(test)]
#[allow(clippy::disallowed_methods)] // tests: throwaway runtimes drive async code
mod context_usage_tests {
    use super::*;
    use serde_json::json;

    /// The previous slice-based implementation, kept as the equivalence oracle.
    fn reference(
        messages: &[AgentMessage],
        window: u64,
        overhead: PromptOverhead,
    ) -> Option<Value> {
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
                u64::try_from(overhead.wire_chars.div_ceil(4)).unwrap_or(u64::MAX)
                    + messages.iter().map(estimate_tokens).sum::<u64>()
            }
        };
        #[allow(clippy::cast_precision_loss)]
        let percent = tokens as f64 / window as f64 * 100.0;
        let resident = messages.iter().map(estimate_tokens).sum::<u64>();
        Some(json!({
            "tokens": tokens,
            "contextWindow": window,
            "percent": percent,
            "wireTokens": tokens,
            "residentHistoryTokens": resident,
            "availableCatalogTokens": u64::try_from(overhead.catalog_chars.div_ceil(4)).unwrap_or(u64::MAX),
            "source": if last_usage.is_some() { "provider+estimate" } else { "estimate" },
        }))
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
    fn deferred_tool_schemas_stay_out_of_wire_overhead() {
        let cwd = std::path::Path::new("/tmp");
        let settings = omp_settings::Settings::load(cwd);
        let tools = crate::tools::session_builtin_tools(cwd, pi_edit::EditMode::Replace, &settings);
        let full = prompt_overhead("system", &tools, crate::methodology::ToolProjection::Full);
        let code = prompt_overhead(
            "system",
            &tools,
            crate::methodology::ToolProjection::CodeMode,
        );
        let expected_wire = "system".len()
            + tools
                .iter()
                .filter(|tool| crate::methodology::is_code_mode_tool(tool.name()))
                .map(|tool| {
                    tool.name().len()
                        + tool.description().len()
                        + tool.parameters().to_string().len()
                })
                .sum::<usize>();
        assert!(tools.iter().any(|tool| tool.name() == "eval"));
        assert_eq!(code.wire_chars, expected_wire);
        assert_eq!(code.catalog_chars, full.wire_chars - "system".len());
        assert!(code.wire_chars < full.wire_chars);
    }

    #[test]
    fn wire_and_catalog_tokens_are_independent() {
        let messages = [AgentMessage::user("hello")];
        let usage = context_usage(
            messages.iter(),
            1000,
            PromptOverhead {
                wire_chars: 40,
                catalog_chars: 4000,
            },
        )
        .unwrap();
        assert_eq!(usage["tokens"], 12);
        assert_eq!(usage["wireTokens"], 12);
        assert_eq!(usage["residentHistoryTokens"], 2);
        assert_eq!(usage["availableCatalogTokens"], 1000);
        assert_eq!(usage["source"], "estimate");
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
                let overhead = PromptOverhead {
                    wire_chars: 37,
                    catalog_chars: 148,
                };
                assert_eq!(
                    context_usage(msgs.iter(), window, overhead),
                    reference(msgs, window, overhead),
                    "case {k} window {window}"
                );
            }
        }
    }
}
