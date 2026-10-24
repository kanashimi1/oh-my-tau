//! Per-model transcript dialects used to serialize a conversation for summarization
//! (pi-ai `dialect/*.ts` `renderTranscript`, pi-agent-core `utils.ts#serializeConversation`
//! with a dialect, pi-catalog `preferredDialect`).
//!
//! omp serializes summary input in the *target model's* tool-call dialect, so the prompt text
//! depends on `preferredDialect(model.id)`. All ten dialects are ported; the in-band scanners
//! (parsing) are not needed here. Tool schemas are not threaded into summary serialization, so
//! tool arguments always render as JSON (`buildArgShapes([])`).
//!
//! [`preferred_dialect`] ports the class-matcher ranking of `classifyModel` (rules.json
//! `taxonomy.classes`) with lenient ambiguity handling; it does not apply the reviewed identity
//! overrides or suffix collapse, which only matter for ids outside the matcher vocabulary.

// Arms mirror the TS dialect tables one-to-one; collecting per-call strings keeps each dialect's
// wire template visible.
#![allow(
    clippy::match_same_arms,
    clippy::format_collect,
    clippy::nonminimal_bool
)]

use serde_json::Value;

use crate::json::{message_role, str_field};
use crate::serialize::{escape_summary_boundary_tags, truncate_tool_result_for_summary};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dialect {
    Xml,
    Anthropic,
    Glm,
    Gemini,
    Gemma,
    Kimi,
    Qwen3,
    Deepseek,
    Minimax,
    Harmony,
}

#[derive(Clone, Copy)]
enum Kind {
    Exact,
    Bounded,
    Namespace,
    Prefix,
    Glob,
}

impl Kind {
    fn rank(self) -> u32 {
        match self {
            Self::Exact => 4,
            Self::Bounded => 3,
            Self::Namespace => 2,
            Self::Prefix => 1,
            Self::Glob => 0,
        }
    }
}

/// `(class id, its dialect, matcher kind, namespace-bounded flag, token)` from rules.json.
const MATCHERS: &[(&str, Dialect, Kind, bool, &str)] = &[
    ("ai21", Dialect::Xml, Kind::Bounded, false, "jamba"),
    ("ai21", Dialect::Xml, Kind::Namespace, false, "ai21"),
    ("amazon", Dialect::Xml, Kind::Bounded, false, "nova"),
    ("amazon", Dialect::Xml, Kind::Bounded, false, "titan"),
    ("amazon", Dialect::Xml, Kind::Namespace, true, "amazon"),
    (
        "anthropic",
        Dialect::Anthropic,
        Kind::Namespace,
        true,
        "anthropic",
    ),
    (
        "anthropic",
        Dialect::Anthropic,
        Kind::Bounded,
        false,
        "anthropic",
    ),
    (
        "anthropic",
        Dialect::Anthropic,
        Kind::Bounded,
        false,
        "claude",
    ),
    ("baidu", Dialect::Xml, Kind::Bounded, false, "ernie"),
    ("bytedance", Dialect::Xml, Kind::Bounded, false, "doubao"),
    ("cohere", Dialect::Xml, Kind::Bounded, false, "command"),
    ("cohere", Dialect::Xml, Kind::Namespace, false, "cohere"),
    (
        "deepseek",
        Dialect::Deepseek,
        Kind::Namespace,
        true,
        "deepseek",
    ),
    (
        "deepseek",
        Dialect::Deepseek,
        Kind::Bounded,
        false,
        "deepseek",
    ),
    ("gemini", Dialect::Gemini, Kind::Bounded, false, "gemini"),
    ("gemma", Dialect::Gemma, Kind::Bounded, false, "gemma"),
    ("glm", Dialect::Glm, Kind::Bounded, false, "glm"),
    ("glm", Dialect::Glm, Kind::Bounded, false, "zai-glm"),
    (
        "gpt-oss",
        Dialect::Harmony,
        Kind::Namespace,
        true,
        "gpt-oss",
    ),
    ("gpt-oss", Dialect::Harmony, Kind::Bounded, false, "gpt-oss"),
    ("kimi", Dialect::Kimi, Kind::Namespace, false, "moonshotai"),
    ("kimi", Dialect::Kimi, Kind::Bounded, false, "kimi"),
    ("kimi", Dialect::Kimi, Kind::Exact, false, "k3"),
    ("kimi", Dialect::Kimi, Kind::Exact, false, "k3-256k"),
    ("meta", Dialect::Xml, Kind::Bounded, false, "muse-spark"),
    ("meta", Dialect::Xml, Kind::Bounded, false, "llama"),
    ("meta", Dialect::Xml, Kind::Namespace, false, "meta-llama"),
    ("mimo", Dialect::Xml, Kind::Bounded, false, "mimo"),
    (
        "minimax",
        Dialect::Minimax,
        Kind::Namespace,
        false,
        "minimax",
    ),
    ("minimax", Dialect::Minimax, Kind::Bounded, false, "minimax"),
    ("minimax", Dialect::Minimax, Kind::Bounded, false, "hailuo"),
    ("mistral", Dialect::Xml, Kind::Bounded, false, "mistral"),
    ("mistral", Dialect::Xml, Kind::Bounded, false, "mixtral"),
    ("openai", Dialect::Harmony, Kind::Namespace, false, "openai"),
    ("openai", Dialect::Harmony, Kind::Namespace, true, "openai"),
    ("openai", Dialect::Harmony, Kind::Prefix, false, "gpt-"),
    ("openai", Dialect::Harmony, Kind::Prefix, false, "chatgpt-"),
    ("openai", Dialect::Harmony, Kind::Prefix, false, "codex-"),
    ("openai", Dialect::Harmony, Kind::Exact, false, "o1"),
    ("openai", Dialect::Harmony, Kind::Prefix, false, "o1-"),
    ("openai", Dialect::Harmony, Kind::Prefix, false, "o1."),
    ("openai", Dialect::Harmony, Kind::Exact, false, "o3"),
    ("openai", Dialect::Harmony, Kind::Prefix, false, "o3-"),
    ("openai", Dialect::Harmony, Kind::Prefix, false, "o3."),
    ("openai", Dialect::Harmony, Kind::Exact, false, "o4"),
    ("openai", Dialect::Harmony, Kind::Prefix, false, "o4-"),
    ("openai", Dialect::Harmony, Kind::Prefix, false, "o4."),
    ("qwen", Dialect::Qwen3, Kind::Bounded, false, "qwen"),
    (
        "qwen",
        Dialect::Qwen3,
        Kind::Bounded,
        false,
        "deepseek-r1-distill-qwen",
    ),
    ("qwen", Dialect::Qwen3, Kind::Glob, false, "*distill-qwen*"),
    ("stepfun", Dialect::Xml, Kind::Bounded, false, "step"),
    ("xai", Dialect::Xml, Kind::Prefix, false, "cursor-grok-"),
    ("xai", Dialect::Xml, Kind::Bounded, false, "grok"),
    ("xai", Dialect::Xml, Kind::Namespace, false, "x-ai"),
    ("xai", Dialect::Xml, Kind::Namespace, true, "xai"),
];

fn bare_of(id: &str) -> &str {
    id.rsplit_once('/').map_or(id, |(_, b)| b)
}

fn bounded_match(value: &str, token: &str) -> bool {
    if value == token {
        return true;
    }
    let Some(rest) = value.strip_prefix(token) else {
        return false;
    };
    rest.bytes()
        .next()
        .is_some_and(|b| matches!(b, b'-' | b'_' | b'.' | b':' | b'0'..=b'9'))
}

/// `globMatch` (`*` wildcards only).
fn glob_match(pattern: &str, value: &str) -> bool {
    let segments: Vec<&str> = pattern.split('*').collect();
    if segments.len() == 1 {
        return value == pattern;
    }
    if !value.starts_with(segments[0]) {
        return false;
    }
    let mut remainder = &value[segments[0].len()..];
    for segment in &segments[1..segments.len() - 1] {
        if segment.is_empty() {
            continue;
        }
        let Some(found) = remainder.find(segment) else {
            return false;
        };
        remainder = &remainder[found + segment.len()..];
    }
    let last = segments[segments.len() - 1];
    last.is_empty() || remainder.ends_with(last)
}

fn matcher_matches(kind: Kind, bounded: bool, token: &str, lower: &str, bare: &str) -> bool {
    match kind {
        Kind::Exact => bare == token,
        Kind::Bounded => bounded_match(bare, token),
        Kind::Namespace => {
            if bounded {
                lower
                    .split(['/', '.', ':'])
                    .any(|p| !p.is_empty() && bounded_match(p, token))
            } else {
                lower.split('/').any(|p| !p.is_empty() && p == token)
            }
        }
        Kind::Prefix => bare.starts_with(token),
        Kind::Glob => glob_match(token, bare),
    }
}

/// Provider-agnostic reviewed identity overrides (rules.json `overrides`); `preferredDialect`
/// classifies with an empty provider, so provider-scoped ones never apply.
fn identity_override(bare: &str) -> Option<Dialect> {
    match bare {
        "daybreak-blue-latest" | "daybreak-red-latest" => return Some(Dialect::Xml),
        "gpt-daybreak-blue-latest" | "gpt-daybreak-red-latest" => return Some(Dialect::Harmony),
        _ => {}
    }
    let a = glob_match("*bonsai-27b*", bare);
    let b = glob_match("*bonsai-2-27b*", bare);
    // equal specificity would tie (lenient: no override); the longer glob otherwise wins
    match (a, b) {
        (_, true) => Some(Dialect::Qwen3),
        (true, false) => Some(Dialect::Qwen3),
        _ => None,
    }
}

/// `preferredDialect(modelId)`: the dialect of the model's class (`xml` when unknown/ambiguous).
#[must_use]
pub fn preferred_dialect(model_id: &str) -> Dialect {
    let lower = model_id.trim().to_lowercase();
    let bare = bare_of(&lower);
    if let Some(d) = identity_override(bare) {
        return d;
    }
    let mut winner: Option<((u32, usize), &str, Dialect)> = None;
    let mut tied = false;
    for (class, dialect, kind, bounded, token) in MATCHERS {
        if !matcher_matches(*kind, *bounded, token, &lower, bare) {
            continue;
        }
        let rank = (kind.rank(), token.len());
        match &winner {
            Some((w, wclass, _)) if *w == rank && wclass != class => tied = true,
            Some((w, _, _)) if *w >= rank => {}
            _ => {
                winner = Some((rank, class, *dialect));
                tied = false;
            }
        }
    }
    if tied {
        return Dialect::Xml;
    }
    winner.map_or(Dialect::Xml, |(_, _, d)| d)
}

// ---------------------------------------------------------------------------------------------
// Shared rendering (dialect/rendering.ts)
// ---------------------------------------------------------------------------------------------

fn stringify_json(v: &Value) -> String {
    omp_session::entries::to_js_string(v).unwrap_or_else(|_| "null".to_owned())
}

struct ToolResult {
    id: String,
    name: String,
    index: usize,
    text: String,
    is_error: bool,
}

struct AssistantParts<'a> {
    text: String,
    thinking: String,
    tool_calls: Vec<&'a Value>,
}

fn assistant_parts(message: &Value) -> AssistantParts<'_> {
    let (mut text, mut thinking, mut tool_calls) = (String::new(), Vec::new(), Vec::new());
    for block in message
        .get("content")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        match str_field(block, "type") {
            Some("text") => text.push_str(str_field(block, "text").unwrap_or("")),
            Some("thinking") => thinking.push(str_field(block, "thinking").unwrap_or("")),
            Some("toolCall") => tool_calls.push(block),
            _ => {}
        }
    }
    AssistantParts {
        text,
        thinking: thinking.join("\n"),
        tool_calls,
    }
}

/// `messageContentText`.
fn content_text(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(blocks)) => {
            let mut out = String::new();
            for block in blocks {
                match str_field(block, "type") {
                    Some("text") => {
                        if let Some(t) = str_field(block, "text") {
                            out.push_str(t);
                        }
                    }
                    Some("image") => match str_field(block, "mimeType").filter(|m| !m.is_empty()) {
                        Some(m) => out.push_str(&format!("[Image: {m}]")),
                        None => out.push_str("[Image]"),
                    },
                    _ => {}
                }
            }
            out
        }
        _ => String::new(),
    }
}

fn collect_tool_result_run(messages: &[&Value], start: usize) -> (Vec<ToolResult>, usize) {
    let mut results = Vec::new();
    let mut next = start;
    while next < messages.len() && message_role(messages[next]) == "toolResult" {
        let m = messages[next];
        results.push(ToolResult {
            id: str_field(m, "toolCallId").unwrap_or("").to_owned(),
            name: str_field(m, "toolName").unwrap_or("undefined").to_owned(),
            index: results.len(),
            text: content_text(m.get("content")),
            is_error: m.get("isError") == Some(&Value::Bool(true)),
        });
        next += 1;
    }
    (results, next)
}

fn escape_xml_attr(v: &str) -> String {
    v.replace('&', "&amp;")
        .replace('"', "&quot;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn escape_xml_text(v: &str) -> String {
    v.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn call_args(call: &Value) -> impl Iterator<Item = (&String, &Value)> {
    call.get("arguments")
        .and_then(Value::as_object)
        .into_iter()
        .flat_map(|o| o.iter())
}

fn call_name(call: &Value) -> &str {
    str_field(call, "name").unwrap_or("undefined")
}

fn call_arguments_json(call: &Value) -> String {
    call.get("arguments")
        .map_or_else(|| "null".to_owned(), stringify_json)
}

/// `renderInvoke` (no schema: every argument is JSON).
fn render_invoke(call: &Value) -> String {
    let mut body = format!("<invoke name=\"{}\">", escape_xml_attr(call_name(call)));
    for (key, value) in call_args(call) {
        body.push_str(&format!(
            "<parameter name=\"{}\">{}</parameter>",
            escape_xml_attr(key),
            stringify_json(value)
        ));
    }
    body.push_str("</invoke>");
    body
}

fn render_invokes(calls: &[&Value]) -> String {
    calls
        .iter()
        .map(|c| render_invoke(c))
        .collect::<Vec<_>>()
        .join("\n")
}

fn render_tool_response_results(results: &[ToolResult]) -> String {
    results
        .iter()
        .map(|r| format!("<tool_response>\n{}\n</tool_response>", r.text))
        .collect::<Vec<_>>()
        .join("\n")
}

fn is_ascii_ws(b: u8) -> bool {
    matches!(b, 9..=13 | 32)
}

fn trim_ascii_start(text: &str, start: usize, end: usize) -> usize {
    let mut c = start;
    while c < end && is_ascii_ws(text.as_bytes()[c]) {
        c += 1;
    }
    c
}

fn trim_ascii_end(text: &str, start: usize, end: usize) -> usize {
    let mut c = end;
    while c > start && is_ascii_ws(text.as_bytes()[c - 1]) {
        c -= 1;
    }
    c
}

fn find_delimited_thinking_close(
    open: &str,
    close: &str,
    text: &str,
    start: usize,
    end: usize,
) -> Option<usize> {
    let mut depth = 1;
    let mut cursor = start;
    while cursor < end {
        let next_close = text[cursor..]
            .find(close)
            .map(|i| i + cursor)
            .filter(|i| *i < end)?;
        if let Some(next_open) = text[cursor..]
            .find(open)
            .map(|i| i + cursor)
            .filter(|o| *o < next_close)
        {
            depth += 1;
            cursor = next_open + open.len();
            continue;
        }
        depth -= 1;
        if depth == 0 {
            return Some(next_close);
        }
        cursor = next_close + close.len();
    }
    None
}

fn unwrap_delimited_thinking(open: &str, close: &str, text: &str) -> String {
    let end = trim_ascii_end(text, 0, text.len());
    let mut cursor = trim_ascii_start(text, 0, end);
    if cursor >= end || !text[cursor..].starts_with(open) {
        return text.to_owned();
    }
    let mut segments = Vec::new();
    while cursor < end {
        if !text[cursor..].starts_with(open) {
            return text.to_owned();
        }
        let inner_start = cursor + open.len();
        let Some(inner_end) = find_delimited_thinking_close(open, close, text, inner_start, end)
        else {
            return text.to_owned();
        };
        let t_end = trim_ascii_end(text, inner_start, inner_end);
        let t_start = trim_ascii_start(text, inner_start, t_end);
        segments.push(unwrap_delimited_thinking(
            open,
            close,
            &text[t_start..t_end],
        ));
        cursor = trim_ascii_start(text, inner_end + close.len(), end);
    }
    segments.join("\n")
}

fn render_delimited_thinking(open: &str, close: &str, text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }
    format!(
        "{open}\n{}\n{close}",
        unwrap_delimited_thinking(open, close, text)
    )
}

fn chat_ml_turn(role: &str, body: &str) -> String {
    format!("<|im_start|>{role}\n{body}<|im_end|>\n")
}

fn join_user_bodies(left: &str, right: &str) -> String {
    if left.is_empty() {
        right.to_owned()
    } else if right.is_empty() {
        left.to_owned()
    } else {
        format!("{left}\n{right}")
    }
}

fn py_string(v: &str) -> String {
    let escaped = v
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t");
    format!("\"{escaped}\"")
}

fn py_value(v: &Value) -> String {
    match v {
        Value::Null => "None".to_owned(),
        Value::Bool(b) => if *b { "True" } else { "False" }.to_owned(),
        Value::Number(_) => stringify_json(v),
        Value::String(s) => py_string(s),
        Value::Array(a) => format!(
            "[{}]",
            a.iter().map(py_value).collect::<Vec<_>>().join(", ")
        ),
        Value::Object(o) => format!(
            "{{{}}}",
            o.iter()
                .map(|(k, val)| format!("{}: {}", py_string(k), py_value(val)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn gemma_value(v: &Value) -> String {
    match v {
        Value::Null => "null".to_owned(),
        Value::Bool(b) => b.to_string(),
        Value::Number(_) => stringify_json(v),
        Value::String(s) => format!("<|\"|>{s}<|\"|>"),
        Value::Array(a) => format!(
            "[{}]",
            a.iter().map(gemma_value).collect::<Vec<_>>().join(",")
        ),
        Value::Object(o) => format!(
            "{{{}}}",
            o.iter()
                .map(|(k, val)| format!("{k}:{}", gemma_value(val)))
                .collect::<Vec<_>>()
                .join(",")
        ),
    }
}

fn kimi_call_id(name: &str, id: &str, index: usize) -> String {
    let trimmed = id.trim();
    if trimmed.starts_with("functions.") {
        trimmed.to_owned()
    } else {
        format!("functions.{name}:{index}")
    }
}

fn harmony_recipient(name: &str) -> String {
    if name.starts_with("functions.") {
        name.to_owned()
    } else {
        format!("functions.{name}")
    }
}

// ---------------------------------------------------------------------------------------------
// renderTranscript per dialect
// ---------------------------------------------------------------------------------------------

/// Which XML-ish flavour the legacy `Human:` / `Assistant:` transcript uses.
#[derive(Clone, Copy)]
enum Legacy {
    Xml,
    Anthropic,
    Minimax,
}

fn legacy_calls(flavour: Legacy, calls: &[&Value]) -> String {
    match flavour {
        Legacy::Xml => render_invokes(calls),
        Legacy::Anthropic if calls.is_empty() => String::new(),
        Legacy::Anthropic => format!(
            "<function_calls>\n{}\n</function_calls>",
            render_invokes(calls)
        ),
        Legacy::Minimax if calls.is_empty() => String::new(),
        Legacy::Minimax => format!(
            "<minimax:tool_call>\n{}\n</minimax:tool_call>",
            render_invokes(calls)
        ),
    }
}

fn legacy_results(flavour: Legacy, results: &[ToolResult]) -> String {
    if matches!(flavour, Legacy::Xml) {
        return render_tool_response_results(results);
    }
    let body = results
        .iter()
        .map(|r| {
            let (tag, stream) = if r.is_error {
                ("error", "stderr")
            } else {
                ("result", "stdout")
            };
            format!(
                "<{tag}>\n<tool_name>{}</tool_name>\n<{stream}>{}</{stream}>\n</{tag}>",
                escape_xml_text(&r.name),
                r.text
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!("<function_results>\n{body}\n</function_results>")
}

fn render_legacy(messages: &[&Value], flavour: Legacy) -> String {
    let mut out = String::new();
    let mut i = 0;
    while i < messages.len() {
        let message = messages[i];
        match message_role(message) {
            "assistant" => {
                let p = assistant_parts(message);
                let thinking = render_delimited_thinking("<thinking>", "</thinking>", &p.thinking);
                // Append in place: `out = format!("{out}…")` re-copied the whole
                // transcript per message (quadratic on long sessions).
                out.push_str("\n\nAssistant: ");
                out.push_str(&thinking);
                out.push_str(&p.text);
                out.push_str(&legacy_calls(flavour, &p.tool_calls));
                i += 1;
            }
            "toolResult" => {
                let (results, next) = collect_tool_result_run(messages, i);
                out.push_str("\n\nHuman: ");
                out.push_str(&legacy_results(flavour, &results));
                i = next;
            }
            role => {
                let text = content_text(message.get("content"));
                if role == "developer" {
                    if !text.is_empty() {
                        if !out.is_empty() {
                            out.push_str("\n\n");
                        }
                        out.push_str(&text);
                    }
                } else {
                    out.push_str("\n\nHuman: ");
                    out.push_str(&text);
                }
                i += 1;
            }
        }
    }
    out
}

fn glm_calls(calls: &[&Value]) -> String {
    calls
        .iter()
        .map(|call| {
            let mut body = format!("<tool_call>{}", call_name(call));
            for (key, value) in call_args(call) {
                body.push_str(&format!(
                    "\n<arg_key>{key}</arg_key>\n<arg_value>{}</arg_value>",
                    stringify_json(value)
                ));
            }
            format!("{body}\n</tool_call>")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn render_glm(messages: &[&Value]) -> String {
    if messages.is_empty() {
        return String::new();
    }
    let mut out = "[gMASK]<sop>".to_owned();
    let mut i = 0;
    while i < messages.len() {
        let message = messages[i];
        match message_role(message) {
            "assistant" => {
                let p = assistant_parts(message);
                let thinking = if p.thinking.is_empty() {
                    String::new()
                } else {
                    format!("\n<think>\n{}\n</think>", p.thinking)
                };
                out.push_str(&format!(
                    "<|assistant|>\n{thinking}{}{}",
                    p.text,
                    glm_calls(&p.tool_calls)
                ));
                i += 1;
            }
            "toolResult" => {
                let (results, next) = collect_tool_result_run(messages, i);
                out.push_str(&format!(
                    "<|observation|>\n{}",
                    render_tool_response_results(&results)
                ));
                i = next;
            }
            role => {
                let role = if role == "developer" { "system" } else { role };
                out.push_str(&format!(
                    "<|{role}|>\n{}",
                    content_text(message.get("content"))
                ));
                i += 1;
            }
        }
    }
    out
}

fn gemini_turn(role: &str, body: &str) -> String {
    format!("<start_of_turn>{role}\n{body}<end_of_turn>\n")
}

fn gemini_call(call: &Value) -> String {
    let kwargs = call_args(call)
        .map(|(k, v)| format!("{k}={}", py_value(v)))
        .collect::<Vec<_>>()
        .join(", ");
    format!("default_api.{}({kwargs})", call_name(call))
}

fn render_gemini(messages: &[&Value]) -> String {
    if messages.is_empty() {
        return String::new();
    }
    let mut out = "<bos>".to_owned();
    let mut pending = String::new();
    let mut i = 0;
    while i < messages.len() {
        let message = messages[i];
        match message_role(message) {
            "developer" => {
                pending = join_user_bodies(&pending, &content_text(message.get("content")));
                i += 1;
                continue;
            }
            "user" => {
                out.push_str(&gemini_turn(
                    "user",
                    &join_user_bodies(&pending, &content_text(message.get("content"))),
                ));
                pending.clear();
                i += 1;
                continue;
            }
            _ => {}
        }
        if !pending.is_empty() {
            out.push_str(&gemini_turn("user", &pending));
            pending.clear();
        }
        if message_role(message) == "assistant" {
            let p = assistant_parts(message);
            let thinking = if p.thinking.is_empty() {
                String::new()
            } else {
                format!("```thinking\n{}\n```\n", p.thinking)
            };
            let body = if p.tool_calls.len() == 1 {
                gemini_call(p.tool_calls[0])
            } else {
                format!(
                    "[{}]",
                    p.tool_calls
                        .iter()
                        .map(|c| gemini_call(c))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            };
            out.push_str(&gemini_turn(
                "model",
                &format!("{thinking}{}```tool_code\n{body}\n```", p.text),
            ));
            i += 1;
            continue;
        }
        let (results, next) = collect_tool_result_run(messages, i);
        let body = results
            .iter()
            .map(|r| format!("```tool_outputs\n{}\n```", r.text))
            .collect::<Vec<_>>()
            .join("\n");
        out.push_str(&gemini_turn("user", &body));
        i = next;
    }
    if !pending.is_empty() {
        out.push_str(&gemini_turn("user", &pending));
    }
    out
}

fn gemma_call(call: &Value) -> String {
    let args = call_args(call)
        .map(|(k, v)| format!("{k}:{}", gemma_value(v)))
        .collect::<Vec<_>>()
        .join(",");
    format!("<|tool_call>call:{}{{{args}}}<tool_call|>", call_name(call))
}

fn gemma_results(results: &[ToolResult]) -> String {
    results
        .iter()
        .map(|r| {
            let parsed = serde_json::from_str::<Value>(&r.text)
                .unwrap_or_else(|_| Value::String(r.text.clone()));
            format!(
                "<|tool_response>response:{}{{output:{}}}<tool_response|>",
                r.name,
                gemma_value(&parsed)
            )
        })
        .collect()
}

fn gemma_turn(role: &str, body: &str) -> String {
    format!("<|turn>{role}\n{body}<turn|>")
}

fn render_gemma(messages: &[&Value]) -> String {
    if messages.is_empty() {
        return String::new();
    }
    let mut out = "<bos>".to_owned();
    let mut i = 0;
    while i < messages.len() {
        let message = messages[i];
        match message_role(message) {
            "assistant" => {
                let p = assistant_parts(message);
                let thinking = if p.thinking.is_empty() {
                    String::new()
                } else {
                    format!("<|channel>thought\n{}<channel|>", p.thinking)
                };
                let mut body = format!(
                    "{thinking}{}{}",
                    p.text,
                    p.tool_calls
                        .iter()
                        .map(|c| gemma_call(c))
                        .collect::<String>()
                );
                let mut next = i + 1;
                if next < messages.len() && message_role(messages[next]) == "toolResult" {
                    let (results, n) = collect_tool_result_run(messages, next);
                    body.push_str(&gemma_results(&results));
                    next = n;
                }
                out.push_str(&gemma_turn("model", &body));
                i = next;
            }
            "toolResult" => {
                let (results, next) = collect_tool_result_run(messages, i);
                out.push_str(&gemma_turn("model", &gemma_results(&results)));
                i = next;
            }
            role => {
                let role = if role == "developer" { "system" } else { role };
                out.push_str(&gemma_turn(role, &content_text(message.get("content"))));
                i += 1;
            }
        }
    }
    out
}

fn kimi_turn(role: &str, name: &str, body: &str) -> String {
    format!("<|im_{role}|>{name}<|im_middle|>{body}<|im_end|>")
}

fn render_kimi(messages: &[&Value]) -> String {
    let mut out = String::new();
    let mut i = 0;
    while i < messages.len() {
        let message = messages[i];
        match message_role(message) {
            "assistant" => {
                let p = assistant_parts(message);
                let thinking = if p.thinking.is_empty() {
                    String::new()
                } else {
                    format!("<think>\n{}\n</think>", p.thinking)
                };
                let calls = if p.tool_calls.is_empty() {
                    String::new()
                } else {
                    let body: String = p
                        .tool_calls
                        .iter()
                        .enumerate()
                        .map(|(idx, c)| {
                            format!(
                                "<|tool_call_begin|>{}<|tool_call_argument_begin|>{}<|tool_call_end|>",
                                kimi_call_id(call_name(c), str_field(c, "id").unwrap_or("undefined"), idx),
                                call_arguments_json(c)
                            )
                        })
                        .collect();
                    format!("<|tool_calls_section_begin|>{body}<|tool_calls_section_end|>")
                };
                out.push_str(&kimi_turn(
                    "assistant",
                    "assistant",
                    &format!("{thinking}{}{calls}", p.text),
                ));
                i += 1;
            }
            "toolResult" => {
                let (results, next) = collect_tool_result_run(messages, i);
                for r in &results {
                    out.push_str(&kimi_turn(
                        "system",
                        &r.name,
                        &format!(
                            "## Return of {}\n{}",
                            kimi_call_id(&r.name, &r.id, r.index),
                            r.text
                        ),
                    ));
                }
                i = next;
            }
            role => {
                let role = if role == "developer" { "system" } else { role };
                out.push_str(&kimi_turn(
                    role,
                    role,
                    &content_text(message.get("content")),
                ));
                i += 1;
            }
        }
    }
    out
}

fn render_qwen3(messages: &[&Value]) -> String {
    if messages.is_empty() {
        return String::new();
    }
    let mut out = String::new();
    let mut i = 0;
    while i < messages.len() {
        let message = messages[i];
        match message_role(message) {
            "assistant" => {
                let p = assistant_parts(message);
                let thinking = if p.thinking.is_empty() {
                    String::new()
                } else {
                    format!("<think>\n{}\n</think>", p.thinking)
                };
                let calls = p
                    .tool_calls
                    .iter()
                    .map(|c| {
                        let mut obj = serde_json::Map::new();
                        obj.insert("name".into(), call_name(c).into());
                        obj.insert(
                            "arguments".into(),
                            c.get("arguments").cloned().unwrap_or(Value::Null),
                        );
                        format!(
                            "<tool_call>\n{}\n</tool_call>",
                            stringify_json(&Value::Object(obj))
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                out.push_str(&chat_ml_turn(
                    "assistant",
                    &format!("{thinking}{}{calls}", p.text),
                ));
                i += 1;
            }
            "toolResult" => {
                let (results, next) = collect_tool_result_run(messages, i);
                out.push_str(&chat_ml_turn(
                    "user",
                    &render_tool_response_results(&results),
                ));
                i = next;
            }
            role => {
                let role = if role == "developer" { "system" } else { role };
                out.push_str(&chat_ml_turn(role, &content_text(message.get("content"))));
                i += 1;
            }
        }
    }
    out
}

fn render_deepseek(messages: &[&Value]) -> String {
    if messages.is_empty() {
        return String::new();
    }
    let mut out = "<｜begin▁of▁sentence｜>".to_owned();
    let mut i = 0;
    while i < messages.len() {
        let message = messages[i];
        match message_role(message) {
            "assistant" => {
                let p = assistant_parts(message);
                let thinking = if p.thinking.is_empty() {
                    String::new()
                } else {
                    format!("<think>\n{}\n</think>", p.thinking)
                };
                let calls = if p.tool_calls.is_empty() {
                    String::new()
                } else {
                    let body: String = p
                        .tool_calls
                        .iter()
                        .map(|c| {
                            format!(
                                "<｜tool▁call▁begin｜>{}<｜tool▁sep｜>{}<｜tool▁call▁end｜>",
                                call_name(c),
                                call_arguments_json(c)
                            )
                        })
                        .collect();
                    format!("<｜tool▁calls▁begin｜>{body}<｜tool▁calls▁end｜>")
                };
                out.push_str(&format!(
                    "<｜Assistant｜>{thinking}{}{calls}<｜end▁of▁sentence｜>",
                    p.text
                ));
                i += 1;
            }
            "toolResult" => {
                let (results, next) = collect_tool_result_run(messages, i);
                out.push_str(
                    &results
                        .iter()
                        .map(|r| format!("<｜tool▁output▁begin｜>{}<｜tool▁output▁end｜>", r.text))
                        .collect::<Vec<_>>()
                        .join("\n"),
                );
                i = next;
            }
            role => {
                if role == "developer" {
                    out.push_str(&content_text(message.get("content")));
                } else {
                    out.push_str(&format!(
                        "<｜User｜>{}",
                        content_text(message.get("content"))
                    ));
                }
                i += 1;
            }
        }
    }
    out
}

fn render_harmony(messages: &[&Value]) -> String {
    let mut out = String::new();
    let mut i = 0;
    while i < messages.len() {
        let message = messages[i];
        match message_role(message) {
            "assistant" => {
                let p = assistant_parts(message);
                let mut emitted = false;
                if !p.thinking.is_empty() {
                    out.push_str(&format!(
                        "<|start|>assistant<|channel|>analysis<|message|>{}<|end|>",
                        p.thinking
                    ));
                    emitted = true;
                }
                if !p.text.is_empty() {
                    out.push_str(&format!(
                        "<|start|>assistant<|channel|>final<|message|>{}<|end|>",
                        p.text
                    ));
                    emitted = true;
                }
                for c in &p.tool_calls {
                    out.push_str(&format!(
                        "<|start|>assistant<|channel|>commentary to={}<|message|>{}<|call|>",
                        harmony_recipient(call_name(c)),
                        call_arguments_json(c)
                    ));
                    emitted = true;
                }
                if !emitted {
                    out.push_str("<|start|>assistant<|channel|>final<|message|><|end|>");
                }
                i += 1;
            }
            "toolResult" => {
                let (results, next) = collect_tool_result_run(messages, i);
                for r in &results {
                    out.push_str(&format!(
                        "<|start|>{} to=assistant<|channel|>commentary<|message|>{}<|end|>",
                        harmony_recipient(&r.name),
                        r.text
                    ));
                }
                i = next;
            }
            role => {
                out.push_str(&format!(
                    "<|start|>{role}<|message|>{}<|end|>",
                    content_text(message.get("content"))
                ));
                i += 1;
            }
        }
    }
    out
}

/// `getDialectDefinition(dialect).renderTranscript(messages)` (no tool schemas).
#[must_use]
pub fn render_transcript(dialect: Dialect, messages: &[Value]) -> String {
    render_transcript_refs(dialect, &messages.iter().collect::<Vec<_>>())
}

fn render_transcript_refs(dialect: Dialect, messages: &[&Value]) -> String {
    match dialect {
        Dialect::Xml => render_legacy(messages, Legacy::Xml),
        Dialect::Anthropic => render_legacy(messages, Legacy::Anthropic),
        Dialect::Minimax => render_legacy(messages, Legacy::Minimax),
        Dialect::Glm => render_glm(messages),
        Dialect::Gemini => render_gemini(messages),
        Dialect::Gemma => render_gemma(messages),
        Dialect::Kimi => render_kimi(messages),
        Dialect::Qwen3 => render_qwen3(messages),
        Dialect::Deepseek => render_deepseek(messages),
        Dialect::Harmony => render_harmony(messages),
    }
}

/// `escapeHarmonyControlTokens`: `<|start|>` -> `<\|start|\>`-style inert spelling.
fn escape_harmony_control_tokens(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(pos) = rest.find("<|") {
        out.push_str(&rest[..pos]);
        let tail = &rest[pos + 2..];
        let hit = [
            "start",
            "end",
            "message",
            "channel",
            "constrain",
            "return",
            "call",
        ]
        .iter()
        .find(|t| tail.starts_with(*t) && tail[t.len()..].starts_with("|>"));
        if let Some(t) = hit {
            out.push_str(&format!("<\\|{t}\\|>"));
            rest = &tail[t.len() + 2..];
        } else {
            out.push_str("<|");
            rest = tail;
        }
    }
    out.push_str(rest);
    out
}

/// `serializeConversation(messages, dialect)`: drops useless tool results (and their calls),
/// drops reasoning for Anthropic targets, truncates tool results, then renders the transcript.
#[must_use]
pub fn serialize_conversation(messages: &[Value], dialect: Dialect) -> String {
    serialize_conversation_refs(&messages.iter().collect::<Vec<_>>(), dialect)
}

/// [`serialize_conversation`] over borrowed messages. Messages that pass
/// through unchanged are rendered in place; only the rewritten ones
/// (filtered assistant blocks, truncated tool results) are new values, so a
/// long transcript is never deep-copied as a whole.
fn serialize_conversation_refs(messages: &[&Value], dialect: Dialect) -> String {
    let useless: Vec<&str> = messages
        .iter()
        .filter(|m| {
            message_role(m) == "toolResult"
                && m.get("useless") == Some(&Value::Bool(true))
                && m.get("isError") != Some(&Value::Bool(true))
        })
        .filter_map(|m| str_field(m, "toolCallId"))
        .collect();
    let drop_thinking = dialect == Dialect::Anthropic;
    let mut processed: Vec<std::borrow::Cow<'_, Value>> = Vec::with_capacity(messages.len());
    for &msg in messages {
        match message_role(msg) {
            "assistant" => {
                let blocks: &[Value] = msg
                    .get("content")
                    .and_then(Value::as_array)
                    .map_or(&[], Vec::as_slice);
                let keep = |b: &Value| {
                    let t = str_field(b, "type");
                    !(t == Some("toolCall")
                        && str_field(b, "id").is_some_and(|id| useless.contains(&id)))
                        && !(drop_thinking && t == Some("thinking"))
                };
                let kept = blocks.iter().filter(|b| keep(b)).count();
                if kept == 0 {
                    continue;
                }
                if kept == blocks.len() {
                    // Every block survives: the rewrite would equal the original.
                    processed.push(std::borrow::Cow::Borrowed(msg));
                } else {
                    let mut m = msg.clone();
                    if let Some(o) = m.as_object_mut() {
                        let kept: Vec<Value> = blocks.iter().filter(|b| keep(b)).cloned().collect();
                        o.insert("content".into(), Value::Array(kept));
                    }
                    processed.push(std::borrow::Cow::Owned(m));
                }
            }
            "toolResult" => {
                if str_field(msg, "toolCallId").is_some_and(|id| useless.contains(&id)) {
                    continue;
                }
                let text: String = msg
                    .get("content")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .filter(|c| str_field(c, "type") == Some("text"))
                    .filter_map(|c| str_field(c, "text"))
                    .collect();
                if text.is_empty() {
                    continue;
                }
                let mut m = msg.clone();
                if let Some(o) = m.as_object_mut() {
                    o.insert("content".into(), serde_json::json!([{ "type": "text", "text": truncate_tool_result_for_summary(&text) }]));
                }
                processed.push(std::borrow::Cow::Owned(m));
            }
            _ => processed.push(std::borrow::Cow::Borrowed(msg)),
        }
    }
    let refs: Vec<&Value> = processed.iter().map(AsRef::as_ref).collect();
    render_transcript_refs(dialect, &refs)
}

/// `serializeConversationForSummary(messages, dialect)`.
#[must_use]
pub fn serialize_conversation_for_summary(messages: &[Value], dialect: Dialect) -> String {
    serialize_conversation_for_summary_refs(&messages.iter().collect::<Vec<_>>(), dialect)
}

/// [`serialize_conversation_for_summary`] over borrowed messages.
#[must_use]
pub(crate) fn serialize_conversation_for_summary_refs(
    messages: &[&Value],
    dialect: Dialect,
) -> String {
    let conversation = serialize_conversation_refs(messages, dialect);
    let escaped = if dialect == Dialect::Harmony {
        escape_harmony_control_tokens(&conversation)
    } else {
        conversation
    };
    escape_summary_boundary_tags(&escaped)
}
