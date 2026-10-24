//! TS `transformMessages` (`providers/transform-messages.ts`): the cross-model history rewrite the
//! Anthropic Messages and OpenAI Responses request builders run before conversion — thinking
//! blocks re-bound to the target model, tool-call ids normalized, orphan tool calls answered and
//! stale tool results demoted to notes.

use std::collections::{HashMap, HashSet};

use serde_json::{json, Map, Value};

use crate::compat::ModelView;
use crate::types::{
    AssistantContent, AssistantMessage, Message, Model, StopReason, TextContent, ThinkingContent,
    ToolCall, ToolResultMessage, UserMessage, UserMessageContent, UserOrToolContent,
};

/// Marker key on a synthesized user message (TS `kSyntheticUser`).
pub(crate) const SYNTHETIC_USER_KEY: &str = "__synthetic";
/// Marker key on a thinking block demoted to text (TS `kDemotedThinking`).
pub(crate) const DEMOTED_THINKING_KEY: &str = "__demoted_thinking";

const MAX_TOOL_CALL_ID_LENGTH: usize = 64;
const MAX_TOOL_CALL_NAME_LENGTH: usize = 128;

/// Tool-call id normalizer: `(id, target model, source message) -> normalized id`.
pub(crate) type IdNormalizer<'a> = dyn Fn(&str, &Model, &AssistantMessage) -> Option<String> + 'a;

fn is_malformed_tool_call(call: &ToolCall) -> bool {
    call.id.trim().is_empty()
        || call.name.is_empty()
        || call.name.chars().count() > MAX_TOOL_CALL_NAME_LENGTH
        || call
            .name
            .chars()
            .any(|c| c.is_whitespace() || c.is_control())
}

fn sanitize_malformed_tool_calls(messages: &[Message]) -> Vec<Message> {
    let has_malformed = messages.iter().any(|m| {
        matches!(m, Message::Assistant(a)
            if a.content.iter().any(|b| matches!(b, AssistantContent::ToolCall(c) if is_malformed_tool_call(c))))
    });
    if !has_malformed {
        return messages.to_vec();
    }
    let mut drop_queues: HashMap<String, Vec<bool>> = HashMap::new();
    let mut result = Vec::new();
    for msg in messages {
        match msg {
            Message::Assistant(a) => {
                drop_queues.clear();
                let mut filtered = Vec::new();
                for block in &a.content {
                    if let AssistantContent::ToolCall(call) = block {
                        let malformed = is_malformed_tool_call(call);
                        drop_queues
                            .entry(call.id.clone())
                            .or_default()
                            .push(malformed);
                        if malformed {
                            continue;
                        }
                    }
                    filtered.push(block.clone());
                }
                if filtered.is_empty() {
                    continue;
                }
                let mut next = a.clone();
                next.content = filtered;
                result.push(Message::Assistant(next));
            }
            Message::ToolResult(r) => {
                if let Some(queue) = drop_queues.get_mut(&r.tool_call_id) {
                    if !queue.is_empty() {
                        let drop = queue.remove(0);
                        if queue.is_empty() {
                            drop_queues.remove(&r.tool_call_id);
                        }
                        if drop {
                            continue;
                        }
                    }
                }
                result.push(msg.clone());
            }
            Message::User(_) | Message::Developer(_) => {
                drop_queues.clear();
                result.push(msg.clone());
            }
        }
    }
    result
}

fn should_drop_truncated_thinking_only(a: &AssistantMessage) -> bool {
    matches!(
        a.stop_reason,
        StopReason::Length | StopReason::Error | StopReason::Aborted
    ) && !a
        .content
        .iter()
        .any(|b| matches!(b, AssistantContent::ToolCall(_) | AssistantContent::Text(_)))
}

fn latest_surviving_assistant_index(messages: &[Message]) -> Option<usize> {
    messages
        .iter()
        .enumerate()
        .rev()
        .find_map(|(i, m)| match m {
            Message::Assistant(a) if !should_drop_truncated_thinking_only(a) => Some(i),
            _ => None,
        })
}

fn is_valid_anthropic_tool_call_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 64
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

fn normalize_anthropic_target_tool_call_id(
    id: &str,
    model: &Model,
    source: &AssistantMessage,
    normalize: &IdNormalizer<'_>,
) -> String {
    if is_valid_anthropic_tool_call_id(id) {
        return id.to_string();
    }
    let normalized = normalize(id, model, source).unwrap_or_else(|| {
        id.chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                    c
                } else {
                    '_'
                }
            })
            .take(MAX_TOOL_CALL_ID_LENGTH)
            .collect()
    });
    if is_valid_anthropic_tool_call_id(&normalized) {
        return normalized;
    }
    format!("toolu_{}", base36_hash(id))
}

fn base36_hash(value: &str) -> String {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(value.as_bytes());
    let mut n = u64::from_be_bytes(digest[..8].try_into().expect("8 bytes"));
    if n == 0 {
        return "0".into();
    }
    let mut out = Vec::new();
    while n > 0 {
        out.push(b"0123456789abcdefghijklmnopqrstuvwxyz"[(n % 36) as usize]);
        n /= 36;
    }
    out.reverse();
    String::from_utf8(out).expect("ascii")
}

/// TS `targetReadsForeignThinking`.
fn target_reads_foreign_thinking(model: &Model, view: &ModelView) -> bool {
    match model.api.as_str() {
        "anthropic-messages" => view.is("replayUnsignedThinking"),
        "openai-completions" => {
            !view.is("requiresThinkingAsText")
                && model.reasoning
                && view.compat_str("thinkingFormat") == Some("zai")
        }
        _ => false,
    }
}

/// `renderDemotedThinking`.
fn render_demoted_thinking(model_id: &str, text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }
    let class = crate::compat::taxonomy::classify_model(crate::compat::rules(), "", model_id, true)
        .map(|i| i.class)
        .unwrap_or_default();
    match class.as_str() {
        "anthropic" => text.to_string(),
        _ => format!("<think>\n{text}\n</think>"),
    }
}

fn demoted_text(model_id: &str, text: &str) -> AssistantContent {
    let mut rest = Map::new();
    rest.insert(DEMOTED_THINKING_KEY.into(), Value::Bool(true));
    AssistantContent::Text(TextContent {
        text: render_demoted_thinking(model_id, text),
        text_signature: None,
        rest,
    })
}

fn is_demoted(block: &AssistantContent) -> bool {
    matches!(block, AssistantContent::Text(t) if t.rest.get(DEMOTED_THINKING_KEY) == Some(&Value::Bool(true)))
}

/// TS `transformMessages` for the message shapes this runtime carries.
// Long by design: mirrors the TS pass structure.
#[allow(clippy::too_many_lines)]
pub(crate) fn transform_messages(
    messages: &[Message],
    model: &Model,
    view: &ModelView,
    normalize_tool_call_id: &IdNormalizer<'_>,
) -> Vec<Message> {
    let messages = sanitize_malformed_tool_calls(messages);
    let mut tool_call_id_map: HashMap<String, String> = HashMap::new();
    let latest_surviving = latest_surviving_assistant_index(&messages);
    let is_anthropic_target = model.api == "anthropic-messages";

    let normalized: Vec<Message> = messages
        .iter()
        .enumerate()
        .map(|(index, msg)| match msg {
            Message::User(_) | Message::Developer(_) => msg.clone(),
            Message::ToolResult(r) => match tool_call_id_map.get(&r.tool_call_id) {
                Some(id) if id != &r.tool_call_id => Message::ToolResult(ToolResultMessage {
                    tool_call_id: id.clone(),
                    ..r.clone()
                }),
                _ => msg.clone(),
            },
            Message::Assistant(a) => {
                let is_same_model =
                    a.provider == model.provider && a.api == model.api && a.model == model.id;
                let is_anthropic_replay = is_anthropic_target && a.api == "anthropic-messages";
                let same_anthropic_deployment = is_anthropic_replay
                    && a.provider == model.provider
                    && (view.is("officialEndpoint")
                        || view.thinking_flag("prefixBinding") == Some(true));
                let is_latest_surviving = Some(index) == latest_surviving;
                let is_official_anthropic_source = is_anthropic_replay && a.provider == "anthropic";
                let is_signing_target = is_anthropic_target && view.is("signingEndpoint");
                let signing_involved = is_official_anthropic_source || is_signing_target;
                let replays_unsigned = is_anthropic_target && view.is("replayUnsignedThinking");
                let invalid_stop = matches!(a.stop_reason, StopReason::Aborted | StopReason::Error);
                let abandoned_tool_use = !invalid_stop
                    && a.stop_reason != StopReason::ToolUse
                    && a.content
                        .iter()
                        .any(|b| matches!(b, AssistantContent::ToolCall(_)));
                let last_block_index = a.content.len().saturating_sub(1);

                let signature_of = |t: &ThinkingContent, idx: usize| -> Option<String> {
                    let untrustworthy =
                        abandoned_tool_use || (invalid_stop && idx == last_block_index);
                    if untrustworthy {
                        None
                    } else {
                        t.thinking_signature.clone()
                    }
                };
                let visible_thinking_survives = |block: &AssistantContent, idx: usize| -> bool {
                    let AssistantContent::Thinking(t) = block else {
                        return false;
                    };
                    if t.redacted.unwrap_or(false) || !is_anthropic_replay {
                        return false;
                    }
                    if is_latest_surviving && abandoned_tool_use {
                        return true;
                    }
                    let sig = signature_of(t, idx).filter(|s| !s.is_empty());
                    if sig.is_none() && t.thinking.trim().is_empty() {
                        return false;
                    }
                    if is_same_model
                        && is_signing_target
                        && sig.as_deref().is_none_or(|s| s.trim().is_empty())
                    {
                        return false;
                    }
                    true
                };
                let has_visible_thinking = a.content.iter().any(
                    |b| matches!(b, AssistantContent::Thinking(t) if !t.redacted.unwrap_or(false)),
                );
                let drops_all_same_model_visible_thinking = is_anthropic_replay
                    && is_same_model
                    && is_signing_target
                    && has_visible_thinking
                    && !a
                        .content
                        .iter()
                        .enumerate()
                        .any(|(i, b)| visible_thinking_survives(b, i));

                let mut out: Vec<AssistantContent> = Vec::new();
                for (block_index, block) in a.content.iter().enumerate() {
                    match block {
                        AssistantContent::Thinking(t) if t.redacted.unwrap_or(false) => {
                            if is_anthropic_replay {
                                if drops_all_same_model_visible_thinking {
                                    continue;
                                }
                                if is_same_model
                                    || same_anthropic_deployment
                                    || (is_latest_surviving && a.provider == model.provider)
                                    || replays_unsigned
                                {
                                    out.push(block.clone());
                                }
                                continue;
                            }
                            if is_same_model {
                                out.push(block.clone());
                            }
                        }
                        AssistantContent::Thinking(t) => {
                            let signature_untrustworthy = abandoned_tool_use
                                || (invalid_stop && block_index == last_block_index);
                            let mut sanitized = t.clone();
                            if signature_untrustworthy
                                && sanitized
                                    .thinking_signature
                                    .as_deref()
                                    .is_some_and(|s| !s.is_empty())
                            {
                                sanitized.thinking_signature = None;
                            }
                            let has_sig = |t: &ThinkingContent| {
                                t.thinking_signature
                                    .as_deref()
                                    .is_some_and(|s| !s.is_empty())
                            };
                            if is_anthropic_replay {
                                let cross_provider_source = a.provider != model.provider;
                                if is_latest_surviving
                                    && abandoned_tool_use
                                    && !cross_provider_source
                                {
                                    out.push(block.clone());
                                    continue;
                                }
                                let stale_signature = !same_anthropic_deployment
                                    && if is_latest_surviving {
                                        cross_provider_source
                                    } else {
                                        !is_same_model
                                    };
                                if stale_signature && signing_involved && has_sig(&sanitized) {
                                    sanitized.thinking_signature = None;
                                }
                                if !has_sig(&sanitized) && sanitized.thinking.trim().is_empty() {
                                    continue;
                                }
                                if is_same_model
                                    && is_signing_target
                                    && sanitized
                                        .thinking_signature
                                        .as_deref()
                                        .is_none_or(|s| s.trim().is_empty())
                                {
                                    continue;
                                }
                                out.push(AssistantContent::Thinking(sanitized));
                                continue;
                            }
                            if is_same_model && has_sig(&sanitized) {
                                out.push(AssistantContent::Thinking(sanitized));
                                continue;
                            }
                            if sanitized.thinking.trim().is_empty() {
                                continue;
                            }
                            if is_same_model {
                                out.push(AssistantContent::Thinking(sanitized));
                                continue;
                            }
                            if target_reads_foreign_thinking(model, view) {
                                if has_sig(&sanitized) {
                                    sanitized.thinking_signature = None;
                                }
                                out.push(AssistantContent::Thinking(sanitized));
                                continue;
                            }
                            out.push(demoted_text(&model.id, &sanitized.thinking));
                        }
                        AssistantContent::Text(_) => out.push(block.clone()),
                        AssistantContent::ToolCall(call) => {
                            let mut next = call.clone();
                            if !is_same_model && call.thought_signature.is_some() {
                                next.thought_signature = None;
                            }
                            let normalized_id = if is_anthropic_target {
                                (!is_same_model || view.is("officialEndpoint")).then(|| {
                                    normalize_anthropic_target_tool_call_id(
                                        &call.id,
                                        model,
                                        a,
                                        normalize_tool_call_id,
                                    )
                                })
                            } else if !is_same_model {
                                normalize_tool_call_id(&call.id, model, a)
                            } else {
                                None
                            };
                            if let Some(id) = normalized_id {
                                if id != call.id {
                                    tool_call_id_map.insert(call.id.clone(), id.clone());
                                    next.id = id;
                                }
                            }
                            out.push(AssistantContent::ToolCall(next));
                        }
                    }
                }
                if out.last().is_some_and(is_demoted) {
                    if let Some(AssistantContent::Text(t)) = out.last_mut() {
                        t.text = t.text.trim_end().to_string();
                    }
                }
                let mut next = a.clone();
                next.content = out;
                Message::Assistant(next)
            }
        })
        .collect();

    let valid_tool_use_ids: HashSet<String> = normalized
        .iter()
        .filter_map(|m| match m {
            Message::Assistant(a) => Some(a),
            _ => None,
        })
        .flat_map(|a| a.content.iter())
        .filter_map(|b| match b {
            AssistantContent::ToolCall(c) => Some(c.id.clone()),
            _ => None,
        })
        .collect();

    // Real tool results by call id, consumable once and only after the call's message.
    let mut real_results: HashMap<String, Vec<(usize, ToolResultMessage, bool)>> = HashMap::new();
    for (i, m) in normalized.iter().enumerate() {
        if let Message::ToolResult(r) = m {
            real_results
                .entry(r.tool_call_id.clone())
                .or_default()
                .push((i, r.clone(), false));
        }
    }
    let take_real = |real: &mut HashMap<String, Vec<(usize, ToolResultMessage, bool)>>,
                     id: &str,
                     after: i64|
     -> Option<ToolResultMessage> {
        let entries = real.get_mut(id)?;
        for entry in entries.iter_mut() {
            if entry.2 || (entry.0 as i64) <= after {
                continue;
            }
            entry.2 = true;
            return Some(entry.1.clone());
        }
        None
    };

    let mut result: Vec<Message> = Vec::new();
    let mut pending_calls: Vec<ToolCall> = Vec::new();
    let mut pending_start: i64 = -1;
    let mut pending_aborted: Vec<ToolCall> = Vec::new();
    let mut pending_aborted_ts: Option<u64> = None;
    let mut pending_aborted_start: i64 = -1;
    let mut status: HashSet<String> = HashSet::new();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(0));

    macro_rules! synthetic {
        ($tc:expr, $text:expr, $ts:expr) => {
            Message::ToolResult(ToolResultMessage {
                tool_call_id: $tc.id.clone(),
                tool_name: $tc.name.clone(),
                content: vec![UserOrToolContent::Text(TextContent {
                    text: $text.to_string(),
                    text_signature: None,
                    rest: Map::default(),
                })],
                details: None,
                is_error: true,
                timestamp: $ts,
                rest: Map::default(),
            })
        };
    }

    macro_rules! flush_pending {
        ($ts:expr) => {
            for tc in std::mem::take(&mut pending_calls) {
                if status.contains(&tc.id) {
                    continue;
                }
                if let Some(real) = take_real(&mut real_results, &tc.id, pending_start) {
                    result.push(Message::ToolResult(real));
                } else {
                    result.push(synthetic!(tc, "No result provided", $ts));
                }
                status.insert(tc.id.clone());
            }
        };
    }
    macro_rules! flush_aborted {
        () => {
            if let Some(ts) = pending_aborted_ts.take() {
                for tc in std::mem::take(&mut pending_aborted) {
                    if status.contains(&tc.id) {
                        continue;
                    }
                    if let Some(real) = take_real(&mut real_results, &tc.id, pending_aborted_start)
                    {
                        result.push(Message::ToolResult(real));
                    } else {
                        result.push(synthetic!(tc, "aborted", ts));
                    }
                    status.insert(tc.id.clone());
                }
            }
        };
    }

    for (i, msg) in normalized.iter().enumerate() {
        let message_ts = match msg {
            Message::User(u) | Message::Developer(u) => u.timestamp,
            Message::Assistant(a) => a.timestamp,
            Message::ToolResult(r) => r.timestamp,
        };
        match msg {
            Message::Assistant(a) => {
                flush_pending!(message_ts);
                flush_aborted!();
                if let Message::Assistant(original) = &messages[i] {
                    if should_drop_truncated_thinking_only(original) {
                        continue;
                    }
                }
                let calls: Vec<ToolCall> = a
                    .content
                    .iter()
                    .filter_map(|b| match b {
                        AssistantContent::ToolCall(c) => Some(c.clone()),
                        _ => None,
                    })
                    .collect();
                if matches!(a.stop_reason, StopReason::Error | StopReason::Aborted) {
                    result.push(msg.clone());
                    pending_aborted = calls;
                    pending_aborted_ts = Some(a.timestamp);
                    pending_aborted_start = i as i64;
                    continue;
                }
                if !calls.is_empty() {
                    pending_calls = calls;
                    pending_start = i as i64;
                }
                result.push(msg.clone());
            }
            Message::ToolResult(r) => {
                let key = &r.tool_call_id;
                if status.contains(key) {
                    continue;
                }
                if let Some(pos) = pending_aborted.iter().position(|tc| &tc.id == key) {
                    pending_aborted.remove(pos);
                    status.insert(key.clone());
                    result.push(msg.clone());
                    continue;
                }
                if pending_calls.iter().any(|tc| &tc.id == key) {
                    status.insert(key.clone());
                    result.push(msg.clone());
                    continue;
                }
                if !valid_tool_use_ids.contains(key) {
                    if pending_calls.iter().any(|tc| !status.contains(&tc.id))
                        || !pending_aborted.is_empty()
                    {
                        continue;
                    }
                    let parts: Vec<&str> = r
                        .content
                        .iter()
                        .filter_map(|p| match p {
                            UserOrToolContent::Text(t) if !t.text.trim().is_empty() => {
                                Some(t.text.as_str())
                            }
                            _ => None,
                        })
                        .collect();
                    if !parts.is_empty() {
                        let error_attr = if r.is_error { " is-error=\"true\"" } else { "" };
                        let mut rest = Map::new();
                        rest.insert(SYNTHETIC_USER_KEY.into(), json!(true));
                        result.push(Message::User(UserMessage {
                            content: UserMessageContent::Text(format!(
                                "<stale-tool-result tool=\"{}\" id=\"{}\"{error_attr}>\n{}\n</stale-tool-result>",
                                r.tool_name,
                                r.tool_call_id,
                                parts.join("\n")
                            )),
                            timestamp: message_ts,
                            rest,
                        }));
                    }
                }
            }
            Message::User(_) | Message::Developer(_) => {
                flush_pending!(message_ts);
                flush_aborted!();
                result.push(msg.clone());
            }
        }
    }
    flush_pending!(now);
    flush_aborted!();
    result
}
