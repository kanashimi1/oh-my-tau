//! `/btw`: ask the current model a side question over the live conversation as
//! read-only context, without tools; neither question nor answer enters the
//! session or the agent transcript. TS: session/btw-history.ts,
//! slash-commands/builtin-lifecycle.ts (`btw`).
//!
//! Bare `/btw` opens this session's BTW history as a runtime dialog (the
//! desktop picker) when there is one, so side questions are browsable the way
//! the TS `btw` mode's Esc-return list was.

use std::sync::Arc;

use pa_types::ai::{AssistantContentBlock, Context, Message, StopReason, UserContent, UserMessage};

use crate::rpc::slash::Outcome;
use crate::runtime::Runtime;

/// Answer ceiling (TS btw answers run to a few paragraphs).
const BTW_MAX_TOKENS: u64 = 4096;

/// Most recent side questions kept per session (TS btw-history keeps the
/// session's list; one live process instance serves one session at a time).
const BTW_HISTORY_LIMIT: usize = 50;

/// Side-question system prompt: answer only, the transcript is context.
const BTW_SYSTEM_PROMPT: &str = "You are answering a side question about an ongoing coding-agent session. The conversation transcript is provided as read-only context; you have no tools and cannot change anything. Answer the question directly and concisely based on the transcript. Do not repeat large transcript verbatim; summarize what matters. Output plain prose only.";

/// One remembered side question (question, answer, first line for the picker).
#[derive(Clone)]
struct BtwEntry {
    question: String,
    answer: String,
}

fn history() -> &'static std::sync::Mutex<Vec<BtwEntry>> {
    static HISTORY: std::sync::OnceLock<std::sync::Mutex<Vec<BtwEntry>>> =
        std::sync::OnceLock::new();
    HISTORY.get_or_init(|| std::sync::Mutex::new(Vec::new()))
}

fn first_line(text: &str, max: usize) -> String {
    let line = text.lines().find(|l| !l.trim().is_empty()).unwrap_or("");
    let mut out: String = line.chars().take(max).collect();
    if line.chars().count() > max {
        out.push('…');
    }
    out
}

/// Slash-command entry point.
pub async fn command(rt: &Arc<Runtime>, args: &str) -> Outcome {
    let question = args.trim();
    if question.is_empty() {
        return bare_command(rt).await;
    }
    // Wait for any running turn so the snapshot is a settled conversation.
    rt.wait_settled().await;

    let messages = rt.agent.messages_snapshot().await.into_vec();
    if messages.is_empty() {
        rt.send(serde_json::json!({"type": "command_output", "text": "btw: nothing to ask about yet — the session has no messages."}));
        return Outcome::Handled {
            agent_invoked: false,
        };
    }
    let mut context_messages = match convert_messages(messages).await {
        Ok(m) => m,
        Err(e) => {
            rt.send(serde_json::json!({"type": "command_output", "text": format!("btw: failed to read the conversation context: {e}")}));
            return Outcome::Handled {
                agent_invoked: false,
            };
        }
    };
    context_messages.push(Message::User(UserMessage {
        content: UserContent::Text(question.to_string()),
        timestamp: pa_agent::now_ms().cast_unsigned(),
        rest: serde_json::Map::default(),
    }));

    let target = rt.provider_target();
    let live = match crate::compact::LiveCompleter::from_target(&target) {
        Ok(live) => live,
        Err(e) => {
            rt.send(serde_json::json!({"type": "command_output", "text": format!("btw: no model available: {e}")}));
            return Outcome::Handled {
                agent_invoked: false,
            };
        }
    };
    rt.send(serde_json::json!({
        "type": "command_output",
        "text": format!(
            "btw: asking {}/{} over {} messages…",
            live.model.provider,
            live.model.id,
            context_messages.len()
        ),
    }));
    let (api_key, _) = crate::adapter::fresh_stored_key(
        &live.model.provider,
        live.credential_id,
        live.api_key.clone(),
    )
    .await;
    let options = pa_ai::types::SimpleStreamOptions {
        base: pa_ai::types::StreamOptions {
            max_tokens: Some(BTW_MAX_TOKENS),
            api_key,
            headers: live.headers.clone().map(|h| h.into_iter().collect()),
            ..pa_ai::types::StreamOptions::default()
        },
        reasoning: live
            .model
            .reasoning
            .then_some(pa_types::ai::ModelThinkingLevel::Medium),
        thinking_budgets: None,
        disable_reasoning: false,
        tool_choice: None,
    };
    let context = Context {
        system_prompt: Some(BTW_SYSTEM_PROMPT.to_string()),
        messages: context_messages,
        tools: None,
    };
    let (answer, answered) =
        match pa_ai::complete_simple(&live.model, &context, Some(options)).await {
            Ok(response)
                if !matches!(
                    response.stop_reason,
                    StopReason::Aborted | StopReason::Error
                ) =>
            {
                let text: String = response
                    .content
                    .iter()
                    .filter_map(|b| match b {
                        AssistantContentBlock::Text(t) => Some(t.text.as_str()),
                        _ => None,
                    })
                    .collect();
                let text = text.trim().to_string();
                if text.is_empty() {
                    (
                        "btw: the model returned an empty answer.".to_string(),
                        false,
                    )
                } else {
                    (text, true)
                }
            }
            Ok(_) => (
                "btw: the side question was aborted or failed.".to_string(),
                false,
            ),
            Err(e) => (format!("btw: request failed: {e:?}"), false),
        };
    if answered {
        let mut h = history().lock().unwrap_or_else(|e| e.into_inner());
        h.push(BtwEntry {
            question: question.to_string(),
            answer: answer.clone(),
        });
        if h.len() > BTW_HISTORY_LIMIT {
            let drop = h.len() - BTW_HISTORY_LIMIT;
            h.drain(0..drop);
        }
    }
    // Neither the question nor the answer enters the transcript: we never
    // `set_messages`/`append_messages`, and the outcome is `Handled`, not a
    // prompt.
    rt.send(serde_json::json!({"type": "command_output", "text": format!("btw — {question}\n\n{answer}")}));
    Outcome::Handled {
        agent_invoked: false,
    }
}

/// Bare `/btw`: browse this session's side-question history (newest first) in
/// the interactive picker; falls back to usage text without a UI or history.
async fn bare_command(rt: &Arc<Runtime>) -> Outcome {
    let entries: Vec<(String, String)> = {
        let h = history().lock().unwrap_or_else(|e| e.into_inner());
        h.iter()
            .rev()
            .map(|e| (e.question.clone(), first_line(&e.answer, 80)))
            .collect()
    };
    if entries.is_empty() {
        rt.send(serde_json::json!({"type": "command_output", "text": "usage: /btw <question> — ask a side question (answer does not enter the session); after the first one, bare /btw browses this session's BTW history."}));
        return Outcome::Handled {
            agent_invoked: false,
        };
    }
    if !crate::slash_extra::dialog::has_ui() {
        let mut text = String::from("btw history (newest first):\n");
        for (i, (q, a)) in entries.iter().enumerate() {
            text.push_str(&format!("{}. {q} — {a}\n", i + 1));
        }
        rt.send(serde_json::json!({"type": "command_output", "text": text}));
        return Outcome::Handled {
            agent_invoked: false,
        };
    }
    let choices: Vec<crate::slash_extra::dialog::Choice> = entries
        .iter()
        .map(|(q, a)| crate::slash_extra::dialog::Choice::with(q, a))
        .collect();
    let Some(index) = crate::slash_extra::dialog::select_with(
        rt,
        "BTW history (newest first)",
        &choices,
        crate::slash_extra::dialog::SelectStyle {
            searchable: true,
            ..Default::default()
        },
    )
    .await
    else {
        return Outcome::Handled {
            agent_invoked: false,
        };
    };
    // The picker indexes newest-first; the store is oldest-first.
    let stored = history()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .rev()
        .nth(index)
        .cloned();
    if let Some(e) = stored {
        rt.send(serde_json::json!({"type": "command_output", "text": format!("btw — {}\n\n{}", e.question, e.answer)}));
    }
    Outcome::Handled {
        agent_invoked: false,
    }
}

/// Agent transcript → LLM messages through the same converter the live loop
/// uses (developer/custom-role handling included).
async fn convert_messages(
    messages: Vec<pa_agent::types::AgentMessage>,
) -> anyhow::Result<Vec<Message>> {
    // `convert_to_llm` yields `pa_agent::types::Message`; the provider-facing
    // `Context` carries the wire-shape `pa_types::ai::Message`, so convert each
    // message back through its serialization (lossless: same JSON contract).
    let converted = crate::agent::convert_to_llm()(messages).await?;
    converted
        .into_iter()
        .map(|m| serde_json::from_value::<Message>(serde_json::to_value(&m)?).map_err(Into::into))
        .collect()
}
