//! Representative conversation histories for the request-body equivalence tests: text with
//! escapes and astral characters, images, raw blocks, thinking (signed, unsigned, redacted,
//! reasoning-details, foreign-model), tool calls with nested/numeric arguments, tool results
//! (multi-text, image-only, empty, errored), unanswered calls and errored turns.

use serde_json::{json, Value};

use crate::types::{Context, Message, Model, ModelInput, Tool, Usage};

pub(crate) fn model(api: &str, provider: &str, id: &str, reasoning: bool, images: bool) -> Model {
    Model {
        id: id.into(),
        name: id.into(),
        api: api.into(),
        provider: provider.into(),
        base_url: "http://localhost".into(),
        reasoning,
        thinking_level_map: None,
        input: if images {
            vec![ModelInput::Text, ModelInput::Image]
        } else {
            vec![ModelInput::Text]
        },
        cost: crate::types::zero_model_cost(),
        context_window: 128_000,
        max_tokens: 8192,
        max_tokens_explicit: false,
        omit_max_output_tokens: None,
        request_model_id: None,
        featured: None,
        headers: None,
        compat: None,
    }
}

fn message(value: Value) -> Message {
    serde_json::from_value(value.clone())
        .unwrap_or_else(|error| panic!("fixture message {value} deserializes: {error}"))
}

fn assistant(model: &Model, content: Value, stop_reason: &str) -> Message {
    message(json!({
        "role": "assistant",
        "content": content,
        "api": model.api,
        "provider": model.provider,
        "model": model.id,
        "usage": serde_json::to_value(Usage::default()).expect("usage serializes"),
        "stopReason": stop_reason,
        "timestamp": 2,
    }))
}

fn user(content: Value) -> Message {
    message(json!({ "role": "user", "content": content, "timestamp": 1 }))
}

fn tool_result(id: &str, name: &str, content: Value, is_error: bool) -> Message {
    message(json!({
        "role": "toolResult",
        "toolCallId": id,
        "toolName": name,
        "content": content,
        "isError": is_error,
        "timestamp": 3,
    }))
}

/// A history exercising every conversion branch, produced by `model` (thinking signatures are
/// only replayed for the same model) with one foreign-model turn. `reasoning_details_signature`
/// is an `OpenAI` reasoning-details encoded signature.
pub(crate) fn history(model: &Model, reasoning_details_signature: &str) -> Vec<Message> {
    let foreign = {
        let mut foreign = model.clone();
        foreign.id = "other-model".into();
        foreign.provider = "other".into();
        foreign
    };
    let image = json!({ "type": "image", "data": "iVBORw0KGgoAAAANSUhEUg==", "mimeType": "image/png" });
    vec![
        user(json!("Hello \"world\"\n\u{1F648} <tag> \\ \u{7} \u{2028} \u{e9}")),
        user(json!("   ")),
        user(json!([
            { "type": "text", "text": "look:" },
            image,
            { "type": "text", "text": "  " },
            { "type": "video", "url": "https://example.com/v.mp4" },
        ])),
        assistant(
            model,
            json!([
                { "type": "thinking", "thinking": "plan \"step\"", "thinkingSignature": "sig-abc" },
                { "type": "thinking", "thinking": "second thought", "thinkingSignature": "sig-abc" },
                { "type": "text", "text": "Sure." },
                { "type": "text", "text": "  " },
                { "type": "text", "text": "Reading\tnow." },
                {
                    "type": "toolCall",
                    "id": "call_1|item_77",
                    "name": "read",
                    "arguments": {
                        "path": "/tmp/a\"b\\c",
                        "n": 1.5,
                        "big": 12_345_678_901_234_u64,
                        "neg": -3,
                        "tiny": 1e-7,
                        "nested": { "x": [true, null, "\u{1F600}"] },
                    },
                },
                {
                    "type": "toolCall",
                    "id": "call_2",
                    "name": "bash",
                    "arguments": { "cmd": "echo \u{e9}\n" },
                    "thoughtSignature": "{\"type\":\"reasoning.encrypted\",\"data\":\"zz\"}",
                },
            ]),
            "toolUse",
        ),
        tool_result(
            "call_1|item_77",
            "read",
            json!([
                { "type": "text", "text": "line1" },
                { "type": "text", "text": "" },
                { "type": "text", "text": "line3 \u{1F600}" },
            ]),
            false,
        ),
        tool_result("call_2", "bash", json!([image]), true),
        user(json!("next")),
        assistant(
            model,
            json!([
                { "type": "thinking", "thinking": "", "thinkingSignature": reasoning_details_signature },
                { "type": "text", "text": "" },
            ]),
            "stop",
        ),
        assistant(
            &foreign,
            json!([
                { "type": "thinking", "thinking": "foreign thoughts", "thinkingSignature": "foreign-sig" },
                { "type": "text", "text": "ok" },
            ]),
            "stop",
        ),
        assistant(model, json!([{ "type": "thinking", "thinking": "   " }]), "stop"),
        // A signature naming an existing key replaces that key's value in place.
        assistant(
            model,
            json!([
                { "type": "thinking", "thinking": "keyed", "thinkingSignature": "content" },
                { "type": "text", "text": "visible" },
            ]),
            "stop",
        ),
        assistant(
            model,
            json!([
                { "type": "thinking", "thinking": "", "thinkingSignature": "opaque-redacted", "redacted": true },
                { "type": "thinking", "thinking": "unsigned thought" },
                { "type": "text", "text": "after" },
            ]),
            "stop",
        ),
        assistant(model, json!([{ "type": "text", "text": "partial" }]), "error"),
        user(json!([{ "type": "text", "text": "do it" }])),
        assistant(
            model,
            json!([
                { "type": "toolCall", "id": "call_3", "name": "write", "arguments": {} },
                { "type": "toolCall", "id": "call_4", "name": "write", "arguments": { "content": "x\ny" } },
            ]),
            "toolUse",
        ),
        tool_result("call_3", "write", json!([]), false),
        tool_result("call_4", "", json!([{ "type": "text", "text": "written" }, image]), false),
        user(json!("unanswered next")),
        assistant(
            model,
            json!([{ "type": "toolCall", "id": "call_5", "name": "read", "arguments": { "path": "z" } }]),
            "toolUse",
        ),
        user(json!([{ "type": "text", "text": "final" }, image])),
    ]
}

/// History endings that drive the last-message cache-control markers differently.
pub(crate) fn endings(model: &Model, reasoning_details_signature: &str) -> Vec<Vec<Message>> {
    let full = history(model, reasoning_details_signature);
    let mut text_tail = full.clone();
    text_tail.push(user(json!("plain tail")));
    let mut blank_tail = full.clone();
    blank_tail.push(user(json!("")));
    let mut tool_tail = full.clone();
    tool_tail.push(assistant(
        model,
        json!([{ "type": "toolCall", "id": "call_6", "name": "read", "arguments": { "path": "q" } }]),
        "toolUse",
    ));
    tool_tail.push(tool_result(
        "call_6",
        "read",
        json!([{ "type": "text", "text": "tail result" }]),
        false,
    ));
    let mut assistant_tail = full.clone();
    assistant_tail.push(assistant(
        model,
        json!([{ "type": "text", "text": "assistant tail" }]),
        "stop",
    ));
    vec![
        full,
        text_tail,
        blank_tail,
        tool_tail,
        assistant_tail,
        Vec::new(),
    ]
}

pub(crate) fn tools() -> Vec<Tool> {
    vec![
        Tool {
            name: "read".into(),
            description: "Read \"files\"\nfrom disk".into(),
            parameters: json!({
                "type": "object",
                "properties": { "path": { "type": "string", "description": "\u{1F4C1} path" } },
                "required": ["path"],
                "additionalProperties": false,
            }),
        },
        Tool {
            name: "Bash".into(),
            description: String::new(),
            parameters: json!({ "type": "object" }),
        },
    ]
}

pub(crate) fn contexts(model: &Model, reasoning_details_signature: &str) -> Vec<Context> {
    let mut contexts = Vec::new();
    for messages in endings(model, reasoning_details_signature) {
        for system_prompt in [
            None,
            Some("You are \"helpful\".\n\u{1F648}".to_string()),
            Some(String::new()),
        ] {
            for tools in [None, Some(Vec::new()), Some(tools())] {
                contexts.push(Context {
                    system_prompt: system_prompt.clone(),
                    messages: messages.clone(),
                    tools,
                });
            }
        }
    }
    contexts
}
