//! `/shake`: strip heavy blocks (images, thinking) from the live agent context
//! in memory; the session file is untouched. TS: slash-commands/builtin-lifecycle.ts
//! (`shake` → `AgentSession.shake`).

use std::sync::Arc;

use pa_agent::types::{
    AgentMessage, AssistantContent, Message, ToolResultContent, UserContent, UserPart,
};

use crate::rpc::slash::Outcome;
use crate::runtime::Runtime;

/// What `/shake` removes (TS `ShakeMode`; omt's default is `all`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShakeMode {
    Images,
    Thinking,
    All,
}

/// Parse the `/shake` argument (`None` = default `all`).
pub fn parse_shake_mode(arg: &str) -> Result<ShakeMode, String> {
    match arg.trim().to_ascii_lowercase().as_str() {
        "" | "all" => Ok(ShakeMode::All),
        "images" => Ok(ShakeMode::Images),
        "thinking" => Ok(ShakeMode::Thinking),
        other => Err(format!(
            "Unknown /shake mode \"{other}\". Use images, thinking, or all."
        )),
    }
}

/// Approximate dropped size in bytes: serialized block when it serializes,
/// else a rough char/4 estimate from text content.
fn block_bytes(value: Option<&serde_json::Value>, fallback_chars: usize) -> usize {
    value
        .and_then(|v| serde_json::to_vec(v).ok())
        .map_or(fallback_chars.div_ceil(4), |b| b.len())
}

fn thinking_bytes(block: &AssistantContent) -> usize {
    match block {
        AssistantContent::Thinking(t) => {
            block_bytes(serde_json::to_value(t).ok().as_ref(), t.thinking.len())
        }
        _ => 0,
    }
}

fn image_bytes(block: &UserPart) -> usize {
    match block {
        UserPart::Image(image) => {
            block_bytes(serde_json::to_value(image).ok().as_ref(), image.data.len())
        }
        UserPart::Text(_) => 0,
    }
}

fn tool_result_image_bytes(block: &ToolResultContent) -> usize {
    match block {
        ToolResultContent::Image(image) => {
            block_bytes(serde_json::to_value(image).ok().as_ref(), image.data.len())
        }
        ToolResultContent::Text(_) => 0,
    }
}

/// Drop image parts from one user/developer content in place.
fn shake_user_parts(
    content: &mut UserContent,
    drop_images: bool,
    removed: &mut usize,
    bytes: &mut usize,
) {
    if !drop_images {
        return;
    }
    if let UserContent::Parts(parts) = content {
        let before = parts.len();
        *bytes += parts.iter().map(image_bytes).sum::<usize>();
        parts.retain(|p| !matches!(p, UserPart::Image(_)));
        *removed += before - parts.len();
    }
}

/// Filter one transcript in place; returns (blocks removed, approx bytes).
/// Images go from user/developer and tool-result content, thinking blocks from
/// assistant messages (TS `AgentSession.shake`).
#[must_use]
pub fn shake_messages(messages: &mut [AgentMessage], mode: ShakeMode) -> (usize, usize) {
    let mut removed = 0usize;
    let mut bytes = 0usize;
    let drop_images = matches!(mode, ShakeMode::Images | ShakeMode::All);
    let drop_thinking = matches!(mode, ShakeMode::Thinking | ShakeMode::All);

    for message in messages.iter_mut() {
        let AgentMessage::Standard(m) = message else {
            continue;
        };
        match m {
            Message::User(um) => {
                shake_user_parts(&mut um.content, drop_images, &mut removed, &mut bytes)
            }
            Message::Developer(dm) => {
                shake_user_parts(&mut dm.content, drop_images, &mut removed, &mut bytes)
            }
            Message::Assistant(am) => {
                if drop_thinking {
                    let before = am.content.len();
                    bytes += am.content.iter().map(thinking_bytes).sum::<usize>();
                    am.content
                        .retain(|b| !matches!(b, AssistantContent::Thinking(_)));
                    removed += before - am.content.len();
                }
            }
            Message::ToolResult(tr) => {
                if drop_images {
                    let before = tr.content.len();
                    bytes += tr
                        .content
                        .iter()
                        .map(tool_result_image_bytes)
                        .sum::<usize>();
                    tr.content
                        .retain(|b| !matches!(b, ToolResultContent::Image(_)));
                    removed += before - tr.content.len();
                }
            }
        }
    }
    (removed, bytes)
}

/// Slash-command entry point.
pub async fn command(rt: &Arc<Runtime>, args: &str) -> Outcome {
    let mode = match parse_shake_mode(args) {
        Ok(mode) => mode,
        Err(message) => {
            rt.send(serde_json::json!({"type": "command_output", "text": message}));
            return Outcome::Handled {
                agent_invoked: false,
            };
        }
    };
    // Wait for a settled turn: mutating the transcript mid-run would race the
    // loop's snapshot (`set_messages` surgery happens after the run is idle).
    rt.wait_settled().await;

    let mut counts = (0usize, 0usize);
    rt.agent
        .mutate_messages(|messages| {
            counts = shake_messages(messages, mode);
        })
        .await;
    let (blocks, bytes) = counts;
    let mode_name = match mode {
        ShakeMode::Images => "images",
        ShakeMode::Thinking => "thinking",
        ShakeMode::All => "all",
    };
    if blocks == 0 {
        rt.send(serde_json::json!({"type": "command_output", "text": format!("shake ({mode_name}): nothing to remove.")}));
    } else {
        rt.send(serde_json::json!({"type": "command_output", "text": format!(
            "shake ({mode_name}): removed {blocks} block(s), ~{} KB freed from the in-memory context (session file untouched).",
            bytes.div_ceil(1024)
        )}));
    }
    Outcome::Handled {
        agent_invoked: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pa_agent::types::{
        AssistantMessage, DeveloperMessage, ImageContent, TextContent, ThinkingContent,
        ToolResultMessage, Usage, UserMessage,
    };

    fn assistant_message(content: Vec<AssistantContent>) -> AssistantMessage {
        serde_json::from_value(serde_json::json!({
            "content": content, "api": "openai-completions", "provider": "p", "model": "a",
            "usage": Usage::zero(), "stopReason": "stop", "timestamp": 1
        }))
        .expect("assistant json")
    }

    fn user_with_images(n: usize) -> AgentMessage {
        let mut parts = vec![UserPart::Text(TextContent {
            text: "look".to_string(),
            text_signature: None,
        })];
        for _ in 0..n {
            parts.push(UserPart::Image(ImageContent {
                data: "QUJD".to_string(),
                mime_type: "image/png".to_string(),
            }));
        }
        AgentMessage::Standard(Message::User(UserMessage {
            content: UserContent::Parts(parts),
            timestamp: 0,
        }))
    }

    fn assistant_with_thinking(n: usize) -> AgentMessage {
        let mut content = vec![AssistantContent::Text(TextContent {
            text: "answer".to_string(),
            text_signature: None,
        })];
        for _ in 0..n {
            content.push(AssistantContent::Thinking(ThinkingContent {
                thinking: "hmm".repeat(100),
                thinking_signature: None,
                redacted: None,
            }));
        }
        AgentMessage::Standard(Message::Assistant(assistant_message(content)))
    }

    fn tool_result_with_image() -> AgentMessage {
        AgentMessage::Standard(Message::ToolResult(ToolResultMessage {
            tool_call_id: "t1".to_string(),
            tool_name: "read".to_string(),
            content: vec![ToolResultContent::Image(ImageContent {
                data: "QUJD".to_string(),
                mime_type: "image/png".to_string(),
            })],
            details: None,
            is_error: false,
            timestamp: 1,
        }))
    }

    #[test]
    fn parse_modes() {
        assert_eq!(parse_shake_mode("").ok(), Some(ShakeMode::All));
        assert_eq!(parse_shake_mode("  IMAGES ").ok(), Some(ShakeMode::Images));
        assert_eq!(parse_shake_mode("thinking").ok(), Some(ShakeMode::Thinking));
        assert!(parse_shake_mode("elide").is_err());
    }

    #[test]
    fn all_removes_images_and_thinking_but_keeps_text() {
        let mut messages = vec![
            user_with_images(2),
            assistant_with_thinking(1),
            tool_result_with_image(),
        ];
        let (blocks, bytes) = shake_messages(&mut messages, ShakeMode::All);
        assert_eq!(blocks, 4);
        assert!(bytes > 0);
        let AgentMessage::Standard(Message::User(um)) = &messages[0] else {
            panic!("expected user message");
        };
        let UserContent::Parts(p) = &um.content else {
            panic!("expected parts");
        };
        assert_eq!(p.len(), 1, "text part survives");
        let AgentMessage::Standard(Message::Assistant(am)) = &messages[1] else {
            panic!("expected assistant message");
        };
        assert_eq!(am.content.len(), 1, "text block survives");
        let AgentMessage::Standard(Message::ToolResult(tr)) = &messages[2] else {
            panic!("expected tool result");
        };
        assert!(tr.content.is_empty());
    }

    #[test]
    fn images_only_keeps_thinking() {
        let mut messages = vec![user_with_images(1), assistant_with_thinking(1)];
        let (blocks, _) = shake_messages(&mut messages, ShakeMode::Images);
        assert_eq!(blocks, 1);
        let AgentMessage::Standard(Message::Assistant(am)) = &messages[1] else {
            panic!("expected assistant message");
        };
        assert_eq!(am.content.len(), 2, "thinking survives");
    }

    #[test]
    fn thinking_only_keeps_images() {
        let mut messages = vec![user_with_images(1), assistant_with_thinking(1)];
        let (blocks, _) = shake_messages(&mut messages, ShakeMode::Thinking);
        assert_eq!(blocks, 1);
        let AgentMessage::Standard(Message::User(um)) = &messages[0] else {
            panic!("expected user message");
        };
        let UserContent::Parts(p) = &um.content else {
            panic!("expected parts");
        };
        assert_eq!(p.len(), 2, "image survives");
    }

    #[test]
    fn developer_images_are_shaken() {
        let mut messages = vec![AgentMessage::Standard(Message::Developer(
            DeveloperMessage {
                content: UserContent::Parts(vec![UserPart::Image(ImageContent {
                    data: "QUJD".to_string(),
                    mime_type: "image/png".to_string(),
                })]),
                timestamp: 0,
                attribution: None,
            },
        ))];
        let (blocks, _) = shake_messages(&mut messages, ShakeMode::All);
        assert_eq!(blocks, 1);
    }

    #[test]
    fn nothing_to_remove_is_zero() {
        let mut messages = vec![AgentMessage::Standard(Message::User(UserMessage {
            content: UserContent::Text("hi".to_string()),
            timestamp: 0,
        }))];
        assert_eq!(shake_messages(&mut messages, ShakeMode::All), (0, 0));
    }
}
