//! Typed models for omp session entries in omp's own shape. They cover the kinds prime-agent's
//! `FileEntry` lacks (`title` slot, `session_init`, `credential_pin`, `model_usage`, `mode_change`,
//! `title_change`, `ttsr_injection`, `reset_boundary`, omp-form `model_change`, nullable
//! `thinking_level_change`, `developer`/`pythonExecution` messages) and re-model the shared ones
//! (`message`, `custom`, `custom_message`, `compaction`, `branch_summary`, `service_tier_change`)
//! with omp's key order, so a parse → serialize pass is byte-identical.
//!
//! Field order follows omp's writers (`session-manager.ts`). Unknown fields are kept in `rest`;
//! conversation messages keep their keys verbatim in file order (omp assembles them incrementally,
//! so the order differs per writer path) and offer a typed view via [`OmpMessage::to_agent`].
//! Numbers are written with JS `Number#toString` formatting ([`to_js_string`]).

use std::io;

use pa_types::session::AgentMessage;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::ser::Formatter;
use serde_json::{Map, Number, Value};

/// Deserialize a field that distinguishes "absent" (outer `None`) from explicit `null` (`Some(None)`).
fn nullable<'de, D: Deserializer<'de>, T: Deserialize<'de>>(
    d: D,
) -> Result<Option<Option<T>>, D::Error> {
    Option::<T>::deserialize(d).map(Some)
}

/// Deserialize an optional arbitrary JSON value, keeping an explicit `null` as `Some(Value::Null)`.
fn keep_null<'de, D: Deserializer<'de>>(d: D) -> Result<Option<Value>, D::Error> {
    Value::deserialize(d).map(Some)
}

/// Declares an entry struct. Header form: `id`, `parentId`, `timestamp` first, then the payload
/// (omp's `{ type, ...freshEntryFields, ...payload }`). `tail` form: payload first, header last
/// (omp's `custom` / `custom_message`, which spread the header after the payload).
macro_rules! entry {
    ($(#[$meta:meta])* $name:ident { $($(#[$fmeta:meta])* $field:ident : $ty:ty),* $(,)? }) => {
        $(#[$meta])*
        #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
        #[serde(rename_all = "camelCase")]
        pub struct $name {
            pub id: String,
            #[serde(default)]
            pub parent_id: Option<String>,
            pub timestamp: String,
            $($(#[$fmeta])* pub $field: $ty,)*
            #[serde(flatten)]
            pub rest: Map<String, Value>,
        }
    };
    (tail $(#[$meta:meta])* $name:ident { $($(#[$fmeta:meta])* $field:ident : $ty:ty),* $(,)? }) => {
        $(#[$meta])*
        #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
        #[serde(rename_all = "camelCase")]
        pub struct $name {
            $($(#[$fmeta])* pub $field: $ty,)*
            pub id: String,
            #[serde(default)]
            pub parent_id: Option<String>,
            pub timestamp: String,
            #[serde(flatten)]
            pub rest: Map<String, Value>,
        }
    };
}

/// The fixed-width first-line title slot (`{"type":"title","v":1,...}`, 256 bytes with newline).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TitleSlot {
    pub v: u32,
    pub title: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    pub updated_at: String,
    pub pad: String,
    #[serde(flatten)]
    pub rest: Map<String, Value>,
}

entry! {
    /// `type: "model_change"`; `model` is `provider/modelId`.
    ModelChange {
        model: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        role: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        resolved_model_is_fallback: Option<bool>,
    }
}

entry! {
    /// `type: "thinking_level_change"`; `thinkingLevel` / `configured` may be explicit nulls.
    ThinkingLevelChange {
        #[serde(default, deserialize_with = "nullable", skip_serializing_if = "Option::is_none")]
        thinking_level: Option<Option<String>>,
        #[serde(default, deserialize_with = "nullable", skip_serializing_if = "Option::is_none")]
        configured: Option<Option<String>>,
    }
}

entry! {
    /// `type: "service_tier_change"`; `serviceTier` is a raw value (null = default tier).
    ServiceTierChange {
        #[serde(default)]
        service_tier: Value,
    }
}

entry! {
    /// `type: "model_usage"`: usage of a call outside the transcript. `usage` stays a raw JSON value
    /// so provider-specific counters (`reasoningTokens`, `cttl`, ...) survive.
    ModelUsage {
        purpose: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        role: Option<String>,
        api: String,
        provider: String,
        model: String,
        usage: Value,
        stop_reason: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error_message: Option<String>,
    }
}

entry! {
    /// `type: "credential_pin"`.
    CredentialPin {
        provider: String,
        hash: String,
    }
}

entry! {
    /// `type: "mode_change"`; `mode` is `"none"` when leaving a mode.
    ModeChange {
        mode: String,
        #[serde(default, deserialize_with = "keep_null", skip_serializing_if = "Option::is_none")]
        data: Option<Value>,
    }
}

entry! {
    /// `type: "title_change"`.
    TitleChange {
        title: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        previous_title: Option<String>,
        source: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        trigger: Option<String>,
    }
}

entry! {
    /// `type: "ttsr_injection"`.
    TtsrInjection {
        injected_rules: Vec<String>,
    }
}

entry! {
    /// `type: "reset_boundary"`: pure marker recorded by `/clear`.
    ResetBoundary {}
}

entry! {
    /// `type: "session_init"`: initial context of a subagent session.
    SessionInit {
        system_prompt: String,
        task: String,
        tools: Vec<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        agent: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        model_role: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        resolved_model: Option<String>,
        #[serde(default, deserialize_with = "keep_null", skip_serializing_if = "Option::is_none")]
        retry_fallback: Option<Value>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        read_only: Option<bool>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        spawns: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        read_summarize: Option<bool>,
        #[serde(default, deserialize_with = "keep_null", skip_serializing_if = "Option::is_none")]
        output_schema: Option<Value>,
        #[serde(default, deserialize_with = "keep_null", skip_serializing_if = "Option::is_none")]
        output_schema_mode: Option<Value>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        restrict_tool_names: Option<bool>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        advisor: Option<String>,
        #[serde(default, deserialize_with = "keep_null", skip_serializing_if = "Option::is_none")]
        compaction_threshold: Option<Value>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        isolated: Option<bool>,
    }
}

entry! {
    /// `type: "compaction"`.
    Compaction {
        summary: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        short_summary: Option<String>,
        first_kept_entry_id: String,
        tokens_before: Number,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tokens_after: Option<Number>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        method: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider_replay_through_entry_id: Option<String>,
        #[serde(default, deserialize_with = "keep_null", skip_serializing_if = "Option::is_none")]
        details: Option<Value>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        from_extension: Option<bool>,
        #[serde(default, deserialize_with = "keep_null", skip_serializing_if = "Option::is_none")]
        preserve_data: Option<Value>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        warning: Option<String>,
    }
}

entry! {
    /// `type: "branch_summary"`.
    BranchSummary {
        from_id: String,
        summary: String,
        #[serde(default, deserialize_with = "keep_null", skip_serializing_if = "Option::is_none")]
        details: Option<Value>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        from_extension: Option<bool>,
    }
}

entry! {
    tail
    /// `type: "custom"`: extension state that does not enter the LLM context.
    Custom {
        custom_type: String,
        #[serde(default, deserialize_with = "keep_null", skip_serializing_if = "Option::is_none")]
        data: Option<Value>,
    }
}

entry! {
    tail
    /// `type: "custom_message"`: extension message that does enter the LLM context.
    CustomMessage {
        custom_type: String,
        content: Value,
        display: bool,
        #[serde(default, deserialize_with = "keep_null", skip_serializing_if = "Option::is_none")]
        details: Option<Value>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        attribution: Option<String>,
    }
}

/// `role: "developer"` message (system-authored reminders injected into the conversation).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DeveloperMessage {
    pub role: String,
    /// Kept as raw JSON so block key order (`type` first) survives.
    pub content: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attribution: Option<String>,
    pub timestamp: Number,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub synthetic: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_initiated: Option<bool>,
    #[serde(flatten)]
    pub rest: Map<String, Value>,
}

/// `role: "pythonExecution"` message (a `!!`/eval cell run recorded in the transcript).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PythonExecutionMessage {
    pub role: String,
    pub code: String,
    pub output: String,
    #[serde(
        default,
        deserialize_with = "keep_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub exit_code: Option<Value>,
    pub cancelled: bool,
    pub truncated: bool,
    pub timestamp: Number,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exclude_from_context: Option<bool>,
    #[serde(flatten)]
    pub rest: Map<String, Value>,
}

/// The `message` payload of a `message` entry.
#[derive(Debug, Clone, PartialEq)]
pub enum OmpMessage {
    Developer(DeveloperMessage),
    PythonExecution(PythonExecutionMessage),
    /// Every other role (`user`, `assistant`, `toolResult`, `bashExecution`, ...): keys verbatim in
    /// file order. [`OmpMessage::to_agent`] gives prime-agent's typed view.
    Agent(Map<String, Value>),
}

impl OmpMessage {
    /// Wrap a prime-agent message, reordering keys the way omp writes them.
    ///
    /// # Errors
    /// The message does not serialize to a JSON object.
    pub fn from_agent(message: &AgentMessage) -> serde_json::Result<Self> {
        let Value::Object(mut map) = serde_json::to_value(message)? else {
            return Err(serde::ser::Error::custom("message is not a JSON object"));
        };
        canonicalize_message(&mut map);
        Ok(Self::Agent(map))
    }

    /// prime-agent's typed view; `None` for `developer` / `pythonExecution` (which it lacks) or a
    /// payload it cannot model.
    #[must_use]
    pub fn to_agent(&self) -> Option<AgentMessage> {
        match self {
            Self::Agent(map) => serde_json::from_value(Value::Object(map.clone())).ok(),
            _ => None,
        }
    }
}

impl<'de> Deserialize<'de> for OmpMessage {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        use serde::de::Error;
        let Value::Object(map) = Value::deserialize(d)? else {
            return Err(D::Error::custom("message is not a JSON object"));
        };
        match map.get("role").and_then(Value::as_str) {
            Some("developer") => serde_json::from_value(Value::Object(map))
                .map(Self::Developer)
                .map_err(D::Error::custom),
            Some("pythonExecution") => serde_json::from_value(Value::Object(map))
                .map(Self::PythonExecution)
                .map_err(D::Error::custom),
            Some(_) => Ok(Self::Agent(map)),
            None => Err(D::Error::custom("message has no role")),
        }
    }
}

impl Serialize for OmpMessage {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Developer(m) => m.serialize(s),
            Self::PythonExecution(m) => m.serialize(s),
            Self::Agent(m) => m.serialize(s),
        }
    }
}

/// Move `keys` (those present) to the front of `map`, in the given order; the rest keep their order.
fn pull_front(map: &mut Map<String, Value>, keys: &[&str]) {
    let mut out = Map::new();
    for k in keys {
        if let Some(v) = map.shift_remove(*k) {
            out.insert((*k).to_owned(), v);
        }
    }
    out.extend(std::mem::take(map));
    *map = out;
}

/// omp's key order for messages built from typed values (assistant/user/toolResult, content blocks, usage).
fn canonicalize_message(map: &mut Map<String, Value>) {
    let role = map
        .get("role")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    match role.as_str() {
        "assistant" => pull_front(
            map,
            &[
                "role",
                "content",
                "api",
                "provider",
                "model",
                "usage",
                "stopReason",
                "timestamp",
            ],
        ),
        "user" => pull_front(
            map,
            &["role", "content", "steering", "attribution", "timestamp"],
        ),
        "toolResult" => pull_front(
            map,
            &[
                "role",
                "toolCallId",
                "toolName",
                "content",
                "details",
                "isError",
                "useless",
                "timestamp",
                "prunedAt",
            ],
        ),
        _ => {}
    }
    if let Some(Value::Array(blocks)) = map.get_mut("content") {
        for block in blocks.iter_mut().filter_map(Value::as_object_mut) {
            let kind = block
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned();
            match kind.as_str() {
                "toolCall" => pull_front(block, &["type", "id", "name", "arguments"]),
                "thinking" => pull_front(block, &["type", "thinking", "thinkingSignature"]),
                "image" => pull_front(block, &["type", "data", "mimeType"]),
                _ => pull_front(block, &["type", "text"]),
            }
        }
    }
    if let Some(Value::Object(usage)) = map.get_mut("usage") {
        pull_front(
            usage,
            &[
                "input",
                "output",
                "cacheRead",
                "cacheWrite",
                "totalTokens",
                "reasoningTokens",
                "cost",
                "cttl",
            ],
        );
    }
}

entry! {
    /// `type: "message"`.
    MessageEntry {
        message: OmpMessage,
    }
}

/// An omp entry in omp's own shape. The `session` header and other kinds not listed here are
/// handled by `pa_types::session::FileEntry`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OmpEntry {
    Title(TitleSlot),
    Message(Box<MessageEntry>),
    ModelChange(ModelChange),
    ThinkingLevelChange(ThinkingLevelChange),
    ServiceTierChange(ServiceTierChange),
    ModelUsage(Box<ModelUsage>),
    CredentialPin(CredentialPin),
    ModeChange(ModeChange),
    TitleChange(TitleChange),
    TtsrInjection(TtsrInjection),
    ResetBoundary(ResetBoundary),
    SessionInit(Box<SessionInit>),
    Compaction(Box<Compaction>),
    BranchSummary(Box<BranchSummary>),
    Custom(Box<Custom>),
    CustomMessage(Box<CustomMessage>),
}

impl OmpEntry {
    /// Parse one JSONL line; `None` when the kind is not modeled here or its payload does not fit.
    #[must_use]
    pub fn parse(raw: &str) -> Option<Self> {
        serde_json::from_str(raw).ok()
    }

    /// Serialize to one JSON line (no trailing newline), numbers formatted like JS.
    ///
    /// # Errors
    /// Serialization failure.
    pub fn to_line(&self) -> serde_json::Result<String> {
        to_js_string(self)
    }

    /// The entry id (`None` for the title slot).
    #[must_use]
    pub fn id(&self) -> Option<&str> {
        Some(match self {
            Self::Title(_) => return None,
            Self::Message(e) => &e.id,
            Self::ModelChange(e) => &e.id,
            Self::ThinkingLevelChange(e) => &e.id,
            Self::ServiceTierChange(e) => &e.id,
            Self::ModelUsage(e) => &e.id,
            Self::CredentialPin(e) => &e.id,
            Self::ModeChange(e) => &e.id,
            Self::TitleChange(e) => &e.id,
            Self::TtsrInjection(e) => &e.id,
            Self::ResetBoundary(e) => &e.id,
            Self::SessionInit(e) => &e.id,
            Self::Compaction(e) => &e.id,
            Self::BranchSummary(e) => &e.id,
            Self::Custom(e) => &e.id,
            Self::CustomMessage(e) => &e.id,
        })
    }
}

/// `type` tags routed through [`OmpEntry`] by `SessionWriter::append_entry`; a payload that does not
/// fit one of these is an error instead of a silent fallback.
pub(crate) const STRICT_KINDS: [&str; 9] = [
    "model_change",
    "thinking_level_change",
    "model_usage",
    "credential_pin",
    "mode_change",
    "title_change",
    "ttsr_injection",
    "reset_boundary",
    "session_init",
];

/// serde_json formatter writing floats like JS `Number#toString` (plain decimals down to 1e-6,
/// exponent form `1e-7` / `1e+21` beyond), so omp-written numbers survive byte-for-byte.
struct JsFormatter;

impl Formatter for JsFormatter {
    fn write_f64<W: ?Sized + io::Write>(&mut self, w: &mut W, value: f64) -> io::Result<()> {
        w.write_all(js_number(value).as_bytes())
    }
}

/// Rust's shortest-digits printer rounds an exact decimal tie up; ECMAScript picks the even digit.
/// When `digits` is such a rounded-up tie and the even neighbour parses back to `v`, return it.
fn tie_to_even(v: f64, digits: &str, exp: i32) -> Option<String> {
    let k = digits.len();
    if !digits.ends_with(['1', '3', '5', '7', '9']) {
        return None;
    }
    let exact = format!("{:.60e}", v);
    let (mantissa, e) = exact.split_once('e')?;
    let all: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    if e.parse::<i32>().ok()? != exp || all.len() <= k {
        return None;
    }
    let (prefix, tail) = all.split_at(k);
    let is_tie = tail.starts_with('5') && tail[1..].bytes().all(|b| b == b'0');
    if !is_tie || prefix == digits || !prefix.ends_with(['0', '2', '4', '6', '8']) {
        return None;
    }
    let candidate = format!("{}.{}e{exp}", &prefix[..1], &prefix[1..]);
    let parsed: f64 = candidate.parse().ok()?;
    #[allow(clippy::float_cmp)]
    let same = parsed == v;
    same.then(|| prefix.to_owned())
}

fn js_number(v: f64) -> String {
    if v == 0.0 {
        return "0".into();
    }
    let sci = format!("{:e}", v.abs());
    let (mantissa, exp) = sci.split_once('e').unwrap_or((&sci, "0"));
    let mut digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    let exp: i32 = exp.parse().unwrap_or(0);
    if let Some(even) = tie_to_even(v.abs(), &digits, exp) {
        digits = even;
    }
    let k = i32::try_from(digits.len()).unwrap_or(i32::MAX);
    let n = exp + 1;
    let sign = if v < 0.0 { "-" } else { "" };
    let body = if k <= n && n <= 21 {
        format!(
            "{digits}{}",
            "0".repeat(usize::try_from(n - k).unwrap_or(0))
        )
    } else if 0 < n && n <= 21 {
        let (int, frac) = digits.split_at(usize::try_from(n).unwrap_or(0));
        format!("{int}.{frac}")
    } else if -6 < n && n <= 0 {
        format!("0.{}{digits}", "0".repeat(usize::try_from(-n).unwrap_or(0)))
    } else {
        let e = n - 1;
        let (first, rest) = digits.split_at(1);
        let frac = if rest.is_empty() {
            String::new()
        } else {
            format!(".{rest}")
        };
        format!("{first}{frac}e{}{}", if e < 0 { "-" } else { "+" }, e.abs())
    };
    format!("{sign}{body}")
}

/// `serde_json::to_string` with JS number formatting.
///
/// # Errors
/// Serialization failure.
pub fn to_js_string<T: Serialize + ?Sized>(value: &T) -> serde_json::Result<String> {
    let mut buf = Vec::new();
    let mut ser = serde_json::Serializer::with_formatter(&mut buf, JsFormatter);
    value.serialize(&mut ser)?;
    String::from_utf8(buf).map_err(serde::ser::Error::custom)
}

pub const TITLE_SLOT_BYTES: usize = 256;

/// Serialize the fixed-width title slot line: exactly [`TITLE_SLOT_BYTES`] UTF-8 bytes including the
/// newline. The title is truncated (by code point) to fit, as omp does.
///
/// # Errors
/// Serialization failure, or `updated_at`/`source` too long for the slot.
pub fn serialize_title_slot(
    title: &str,
    source: Option<&str>,
    updated_at: &str,
) -> anyhow::Result<String> {
    let line = |t: &str, pad: usize| -> anyhow::Result<String> {
        let slot = TitleSlot {
            v: 1,
            title: t.to_owned(),
            source: source.map(str::to_owned),
            updated_at: updated_at.to_owned(),
            pad: " ".repeat(pad),
            rest: Map::new(),
        };
        let mut s = OmpEntry::Title(slot).to_line()?;
        s.push('\n');
        Ok(s)
    };
    let mut chars: Vec<char> = title.chars().collect();
    loop {
        let t: String = chars.iter().collect();
        let base = line(&t, 0)?;
        if base.len() <= TITLE_SLOT_BYTES {
            return line(&t, TITLE_SLOT_BYTES - base.len());
        }
        anyhow::ensure!(
            chars.pop().is_some(),
            "title slot metadata exceeds {TITLE_SLOT_BYTES} bytes"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::js_number;

    #[test]
    fn js_number_matches_javascript() {
        for (v, want) in [
            (3.75e-6, "0.00000375"),
            (1e-6, "0.000001"),
            (1e-7, "1e-7"),
            (1.5e-7, "1.5e-7"),
            (0.5, "0.5"),
            (123.456, "123.456"),
            (1e21, "1e+21"),
            (1.5e300, "1.5e+300"),
            (1e20, "100000000000000000000"),
            (-0.00029775, "-0.00029775"),
            (9.714_451_465_470_121e-17, "9.714451465470121e-17"),
            (1_218.192_687_988_281_2, "1218.1926879882812"),
            (1_788_847_823_509.031_2, "1788847823509.0312"),
        ] {
            assert_eq!(js_number(v), want);
        }
    }
}
