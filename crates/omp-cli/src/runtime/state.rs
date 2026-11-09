//! Read-only status and accessors: `state`, messages, session stats.

use omp_tui::magic::Ultra;
use pa_agent::types::{AgentMessage, Message, StopReason};
use serde_json::{json, Map, Value};

use super::queue::queue_mode;
use super::usage::{context_usage, prompt_overhead_chars};
use super::{level_name, lock, message_text, Runtime};

impl Runtime {
    pub async fn state(&self) -> Value {
        // Status without the transcript: this answers every `get_state`/slash busy check.
        let (s, message_count) = self.agent.status().await;
        let retrying = lock(&self.recovery).saga_open();
        let v = {
            let st = lock(&self.settings);
            let (steering, follow) = self.queued();
            let queued = steering.len() + follow.len();
            let mut v = json!({
                "isStreaming": s.is_streaming,
                "isCompacting": self.compacting.load(std::sync::atomic::Ordering::SeqCst),
                "steeringMode": queue_mode(self.agent.steering_mode()),
                "followUpMode": queue_mode(self.agent.follow_up_mode()),
                "interruptMode": st.interrupt_mode,
                "sessionId": self.persist.session_id(),
                "autoCompactionEnabled": st.auto_compaction,
                "queuedMessageCount": queued,
                "hasPendingAsyncWork": false,
                "isSettled": !s.is_streaming && queued == 0 && !retrying,
                "queuedMessages": {"steering": steering, "followUp": follow},
                "todoPhases": st.todo_phases,
                "fastModeEnabled": false,
                "tokensPerSecond": null,
                "fastModeActive": false,
                "messageCount": message_count,
                "systemPrompt": [s.system_prompt],
                "dumpTools": s.tools.iter().map(|t| json!({
                    "name": t.name(), "description": t.description(), "parameters": t.parameters(),
                })).collect::<Vec<_>>(),
                "goal": null,
            });
            if let Some(name) = &st.session_name {
                v["sessionName"] = json!(name);
            }
            if let Some(file) = self.persist.file() {
                v["sessionFile"] = json!(file.display().to_string());
            }
            v
        };
        let mut out = Map::new();
        out.insert("model".into(), self.model_info());
        out.insert("thinkingLevel".into(), json!(level_name(s.thinking_level)));
        out.insert("ultra".into(), json!(self.ultra().map(Ultra::name)));
        if let Some(obj) = v.as_object() {
            out.extend(obj.clone());
        }
        let base = prompt_overhead_chars(&s.system_prompt, &s.tools);
        let window = s.model.context_window;
        if let Some(usage) = self
            .agent
            .with_messages(|m| context_usage(m.iter(), window, base))
            .await
        {
            out.insert("contextUsage".into(), usage);
        }
        Value::Object(out)
    }

    pub async fn messages(&self) -> Vec<Value> {
        // Shared snapshot: serialize outside the state lock, without a deep copy first.
        self.agent
            .messages_snapshot()
            .await
            .iter()
            .map(crate::events::message_value)
            .collect()
    }

    pub async fn last_assistant_text(&self) -> Option<String> {
        self.agent
            .with_messages(|messages| {
                messages.iter().rev().find_map(|m| match m {
                    AgentMessage::Standard(Message::Assistant(a)) => {
                        let t = message_text(m);
                        (a.stop_reason != StopReason::Aborted || !t.is_empty()).then_some(t)
                    }
                    _ => None,
                })
            })
            .await
    }

    pub async fn session_stats(&self) -> Value {
        let (s, total) = self.agent.status().await;
        let base = prompt_overhead_chars(&s.system_prompt, &s.tools);
        let window = s.model.context_window;
        // One in-place pass over the transcript under the state lock (no copy).
        let (user, assistant, calls, results, input, output, cache_read, cache_write, cost, usage) =
            self.agent
                .with_messages(|messages| {
                    let (mut user, mut assistant, mut calls, mut results) =
                        (0u64, 0u64, 0u64, 0u64);
                    let (mut input, mut output, mut cache_read, mut cache_write, mut cost) =
                        (0u64, 0u64, 0u64, 0u64, 0.0f64);
                    for m in messages.iter() {
                        match m {
                            AgentMessage::Standard(Message::User(_)) => user += 1,
                            AgentMessage::Standard(Message::Assistant(a)) => {
                                assistant += 1;
                                calls += a.tool_calls().len() as u64;
                                input += a.usage.input;
                                output += a.usage.output;
                                cache_read += a.usage.cache_read;
                                cache_write += a.usage.cache_write;
                                cost += a.usage.cost.total;
                            }
                            AgentMessage::Standard(Message::ToolResult(_)) => results += 1,
                            AgentMessage::Standard(Message::Developer(_))
                            | AgentMessage::Custom(_) => {}
                        }
                    }
                    let usage = context_usage(messages.iter(), window, base);
                    (
                        user,
                        assistant,
                        calls,
                        results,
                        input,
                        output,
                        cache_read,
                        cache_write,
                        cost,
                        usage,
                    )
                })
                .await;
        let mut v = json!({
            "sessionId": self.persist.session_id(),
            "userMessages": user,
            "assistantMessages": assistant,
            "toolCalls": calls,
            "toolResults": results,
            "totalMessages": total,
            "tokens": {
                "input": input, "output": output, "reasoning": 0,
                "cacheRead": cache_read, "cacheWrite": cache_write,
                "total": input + output + cache_read + cache_write,
            },
            "premiumRequests": 0,
            "cost": cost,
        });
        if let Some(file) = self.persist.file() {
            v["sessionFile"] = json!(file.display().to_string());
        }
        if let Some(u) = usage {
            v["contextUsage"] = u;
        }
        v
    }
}
