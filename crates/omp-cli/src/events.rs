//! `AgentEvent` -> `omp` RPC event frames (field names match omp).

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use pa_agent::stream::AssistantMessageEvent as Ev;
use pa_agent::types::{
    AgentEvent, AgentMessage, AssistantContent, AssistantMessage, Message, TextContent,
    ThinkingContent, ToolCall,
};
use serde_json::{json, Map, Value};

fn to_value<T: serde::Serialize>(v: &T) -> Value {
    serde_json::to_value(v).unwrap_or(Value::Null)
}

/// Wire form of a message. omp stamps `attribution: "user"` on user-typed turns (after `content`).
#[must_use]
pub fn message_value(m: &AgentMessage) -> Value {
    let mut v = to_value(m);
    stamp_attribution(&mut v);
    v
}

/// Add `attribution: "user"` to a user-role message value that lacks it.
pub fn stamp_attribution(v: &mut Value) {
    let Some(obj) = v.as_object_mut() else { return };
    if obj.get("role").and_then(Value::as_str) != Some("user") || obj.contains_key("attribution") {
        return;
    }
    // Keep omp's key order (role, content, attribution, timestamp) regardless of map ordering.
    let entries: Vec<(String, Value)> = std::mem::take(obj).into_iter().collect();
    for (k, val) in entries {
        let is_content = k == "content";
        obj.insert(k, val);
        if is_content {
            obj.insert("attribution".into(), Value::String("user".into()));
        }
    }
    if !obj.contains_key("attribution") {
        obj.insert("attribution".into(), Value::String("user".into()));
    }
}

fn stop_reason(r: pa_agent::types::StopReason) -> Value {
    to_value(&r)
}

/// omp `assistantMessageEvent` payload.
#[must_use]
pub fn assistant_event(e: &Ev) -> Value {
    assistant_event_with(e, &|p| to_value(p))
}

/// Extension-host form of [`assistant_event`]: the `partial` message (identical
/// to the enclosing `message_update.message`) is not serialized a second time;
/// `partialIsMessage: true` tells `ext-host/host.ts` to point `partial` at the
/// event's `message` before any handler runs.
#[must_use]
pub fn assistant_event_slim(e: &Ev) -> Value {
    let mut v = assistant_event_with(e, &|_| Value::Null);
    if let Some(obj) = v.as_object_mut() {
        if obj.remove("partial").is_some() {
            obj.insert("partialIsMessage".into(), Value::Bool(true));
        }
    }
    v
}

fn assistant_event_with(e: &Ev, pv: &dyn Fn(&AssistantMessage) -> Value) -> Value {
    match e {
        Ev::Start { partial } => json!({"type":"start","partial":pv(partial)}),
        Ev::TextStart {
            content_index,
            partial,
        } => {
            json!({"type":"text_start","contentIndex":content_index,"partial":pv(partial)})
        }
        Ev::TextDelta {
            content_index,
            delta,
            partial,
        } => {
            json!({"type":"text_delta","contentIndex":content_index,"delta":delta,"partial":pv(partial)})
        }
        Ev::TextEnd {
            content_index,
            content,
            partial,
        } => {
            json!({"type":"text_end","contentIndex":content_index,"content":content,"partial":pv(partial)})
        }
        Ev::ThinkingStart {
            content_index,
            partial,
        } => {
            json!({"type":"thinking_start","contentIndex":content_index,"partial":pv(partial)})
        }
        Ev::ThinkingDelta {
            content_index,
            delta,
            partial,
        } => {
            json!({"type":"thinking_delta","contentIndex":content_index,"delta":delta,"partial":pv(partial)})
        }
        Ev::ThinkingEnd {
            content_index,
            partial,
        } => {
            json!({"type":"thinking_end","contentIndex":content_index,"partial":pv(partial)})
        }
        Ev::ToolCallStart {
            content_index,
            partial,
        } => {
            json!({"type":"toolcall_start","contentIndex":content_index,"partial":pv(partial)})
        }
        Ev::ToolCallDelta {
            content_index,
            delta,
            partial,
        } => {
            json!({"type":"toolcall_delta","contentIndex":content_index,"delta":delta,"partial":pv(partial)})
        }
        Ev::ToolCallEnd {
            content_index,
            tool_call,
            partial,
        } => {
            json!({"type":"toolcall_end","contentIndex":content_index,"toolCall":to_value(tool_call),"partial":pv(partial)})
        }
        Ev::Done { reason, message } => {
            json!({"type":"done","reason":stop_reason(*reason),"message":to_value(message)})
        }
        Ev::Error { reason, error } => {
            json!({"type":"error","reason":stop_reason(*reason),"error":to_value(error)})
        }
    }
}

// ---------------------------------------------------------------------------
// Extension-host `message_update` deltas.
//
// A `message_update` used to carry the whole accumulated message on every
// streaming delta (O(turn²) bytes per turn). [`ExtUpdateEncoder`] sends the
// whole message once per message (or after any loss of sync) and afterwards
// only what moved; `ext-host/host.ts` (`rebuildMessageUpdate`) rebuilds the
// full `{type, message, assistantMessageEvent}` event before any handler runs,
// so extensions see exactly the shape a full frame gives them.
//
// Frame (the `event` param of a fire-and-forget `emit`):
//   full:  {type:"message_update", message, assistantMessageEvent, delta:{id, seq:0}}
//   delta: {type:"message_update", assistantMessageEvent,
//           delta:{id, seq, n, hdr?, b:[{i, blk?, v, x?}, ...]}}
// `assistantMessageEvent` never carries `partial` (`partialIsMessage: true`:
// the host points it at the rebuilt message). `id` names the full frame the
// deltas build on, `seq` counts frames since it (the host rejects a gap), `n`
// is the content length after applying `b`, `hdr` the message without its
// content (`content: []` placeholder) when any non-content field changed.
// Each `b` entry rebuilds content block `i` as `blk` (or the host's previous
// block when absent) with its streamed field (`text` / `thinking` /
// `arguments`) set by `v`: `k` keep the previous value, `a` append the event's
// `delta`, `+` append the string `x`, `=` the literal `x`, `p` apply the JSON
// patch `x` (see [`value_patch`]). A frame the host cannot apply is answered
// with `result: {resync: true}`, and the next update goes out full.
// ---------------------------------------------------------------------------

/// What the extension host last received for one content block: enough to
/// tell whether the live block moved and how to express the move.
#[derive(Debug, Clone, PartialEq)]
enum BlockShadow {
    Text {
        len: usize,
        signature: Option<String>,
    },
    Thinking {
        len: usize,
        signature: Option<String>,
        redacted: Option<bool>,
    },
    ToolCall {
        id: String,
        name: String,
        signature: Option<String>,
        arguments: Value,
    },
}

impl BlockShadow {
    fn of(block: &AssistantContent) -> Self {
        match block {
            AssistantContent::Text(t) => BlockShadow::Text {
                len: t.text.len(),
                signature: t.text_signature.clone(),
            },
            AssistantContent::Thinking(t) => BlockShadow::Thinking {
                len: t.thinking.len(),
                signature: t.thinking_signature.clone(),
                redacted: t.redacted,
            },
            AssistantContent::ToolCall(tc) => BlockShadow::ToolCall {
                id: tc.id.clone(),
                name: tc.name.clone(),
                signature: tc.thought_signature.clone(),
                arguments: tc.arguments.clone(),
            },
        }
    }

    /// Every field except the streamed one is unchanged.
    fn same_skeleton(&self, block: &AssistantContent) -> bool {
        match (self, block) {
            (BlockShadow::Text { signature, .. }, AssistantContent::Text(t)) => {
                *signature == t.text_signature
            }
            (
                BlockShadow::Thinking {
                    signature,
                    redacted,
                    ..
                },
                AssistantContent::Thinking(t),
            ) => *signature == t.thinking_signature && *redacted == t.redacted,
            (
                BlockShadow::ToolCall {
                    id,
                    name,
                    signature,
                    ..
                },
                AssistantContent::ToolCall(tc),
            ) => *id == tc.id && *name == tc.name && *signature == tc.thought_signature,
            _ => false,
        }
    }
}

/// The block with its streamed field emptied (`""` / `null`). The host writes
/// the rebuilt value into that existing key, which keeps the key order of a
/// full serialization.
fn block_skeleton(block: &AssistantContent) -> Value {
    let skeleton = match block {
        AssistantContent::Text(t) => AssistantContent::Text(TextContent {
            text: String::new(),
            text_signature: t.text_signature.clone(),
        }),
        AssistantContent::Thinking(t) => AssistantContent::Thinking(ThinkingContent {
            thinking: String::new(),
            thinking_signature: t.thinking_signature.clone(),
            redacted: t.redacted,
        }),
        AssistantContent::ToolCall(tc) => AssistantContent::ToolCall(ToolCall {
            id: tc.id.clone(),
            name: tc.name.clone(),
            arguments: Value::Null,
            thought_signature: tc.thought_signature.clone(),
        }),
    };
    to_value(&skeleton)
}

/// How the host derives a streamed string field from the `prev_len` bytes it
/// holds. Streams only append (the TS stream contract), so an unchanged length
/// means unchanged text; `delta` is the event's own fragment, if it targets
/// this block.
fn string_op(live: &str, prev_len: Option<usize>, delta: Option<&str>) -> (&'static str, Option<Value>) {
    match prev_len {
        Some(n) if live.len() == n => ("k", None),
        Some(n) if delta.is_some_and(|d| live.len() == n + d.len() && live.ends_with(d)) => {
            ("a", None)
        }
        Some(n) if live.len() > n && live.is_char_boundary(n) => {
            ("+", Some(Value::String(live[n..].to_owned())))
        }
        _ => ("=", Some(Value::String(live.to_owned()))),
    }
}

/// Structural patch turning `prev` into `next` (streamed tool-call arguments
/// grow by appended keys, array items and string suffixes):
/// `{"=": v}` literal, `{"+": s}` string append, `{"o": {key: patch}}` object
/// whose existing keys keep their order (only changed/new keys listed; new
/// keys append), `{"a": [[i, patch]], "n": len}` array. Anything else
/// (removed or reordered keys, shrunk arrays, type changes) is a literal.
fn value_patch(prev: &Value, next: &Value) -> Value {
    match (prev, next) {
        (Value::String(p), Value::String(n)) if n.len() > p.len() && n.starts_with(p.as_str()) => {
            json!({ "+": &n[p.len()..] })
        }
        (Value::Object(p), Value::Object(n))
            if p.len() <= n.len() && p.keys().zip(n.keys()).all(|(a, b)| a == b) =>
        {
            let mut o = Map::new();
            for (k, nv) in n {
                match p.get(k) {
                    Some(pv) if pv == nv => {}
                    Some(pv) => {
                        o.insert(k.clone(), value_patch(pv, nv));
                    }
                    None => {
                        o.insert(k.clone(), json!({ "=": nv }));
                    }
                }
            }
            json!({ "o": o })
        }
        (Value::Array(p), Value::Array(n)) if p.len() <= n.len() => {
            let mut items = Vec::new();
            for (i, nv) in n.iter().enumerate() {
                match p.get(i) {
                    Some(pv) if pv == nv => {}
                    Some(pv) => items.push(json!([i, value_patch(pv, nv)])),
                    None => items.push(json!([i, { "=": nv }])),
                }
            }
            json!({ "a": items, "n": n.len() })
        }
        _ => json!({ "=": next }),
    }
}

/// One `b` entry for content block `index`, or `None` when the host's copy is
/// current and the event does not target the block.
fn block_entry(
    index: usize,
    block: &AssistantContent,
    prev: Option<&BlockShadow>,
    ame: &Ev,
    targeted: bool,
) -> Option<Value> {
    let skeleton_changed = !prev.is_some_and(|p| p.same_skeleton(block));
    let (op, payload) = match block {
        AssistantContent::Text(t) => {
            let prev_len = match prev {
                Some(BlockShadow::Text { len, .. }) => Some(*len),
                _ => None,
            };
            let delta = match ame {
                Ev::TextDelta { delta, .. } if targeted => Some(delta.as_str()),
                _ => None,
            };
            string_op(&t.text, prev_len, delta)
        }
        AssistantContent::Thinking(t) => {
            let prev_len = match prev {
                Some(BlockShadow::Thinking { len, .. }) => Some(*len),
                _ => None,
            };
            let delta = match ame {
                Ev::ThinkingDelta { delta, .. } if targeted => Some(delta.as_str()),
                _ => None,
            };
            string_op(&t.thinking, prev_len, delta)
        }
        AssistantContent::ToolCall(tc) => match prev {
            Some(BlockShadow::ToolCall { arguments, .. }) if *arguments == tc.arguments => {
                ("k", None)
            }
            Some(BlockShadow::ToolCall { arguments, .. }) => {
                ("p", Some(value_patch(arguments, &tc.arguments)))
            }
            _ => ("=", Some(tc.arguments.clone())),
        },
    };
    if op == "k" && !skeleton_changed && !targeted {
        return None;
    }
    let mut entry = Map::new();
    entry.insert("i".into(), json!(index));
    if skeleton_changed {
        entry.insert("blk".into(), block_skeleton(block));
    }
    entry.insert("v".into(), Value::String(op.into()));
    if let Some(x) = payload {
        entry.insert("x".into(), x);
    }
    Some(Value::Object(entry))
}

fn content_index(e: &Ev) -> Option<usize> {
    match e {
        Ev::TextStart { content_index, .. }
        | Ev::TextDelta { content_index, .. }
        | Ev::TextEnd { content_index, .. }
        | Ev::ThinkingStart { content_index, .. }
        | Ev::ThinkingDelta { content_index, .. }
        | Ev::ThinkingEnd { content_index, .. }
        | Ev::ToolCallStart { content_index, .. }
        | Ev::ToolCallDelta { content_index, .. }
        | Ev::ToolCallEnd { content_index, .. } => Some(*content_index),
        Ev::Start { .. } | Ev::Done { .. } | Ev::Error { .. } => None,
    }
}

/// The message without its content. Exhaustive destructuring: a new
/// `AssistantMessage` field fails to compile here instead of silently never
/// reaching the host.
fn header_of(m: &AssistantMessage) -> AssistantMessage {
    let AssistantMessage {
        content: _,
        api,
        provider,
        model,
        response_model,
        response_id,
        diagnostics,
        usage,
        stop_reason,
        stop_reason_raw,
        error_message,
        timestamp,
        meta,
    } = m;
    AssistantMessage {
        content: Vec::new(),
        api: api.clone(),
        provider: provider.clone(),
        model: model.clone(),
        response_model: response_model.clone(),
        response_id: response_id.clone(),
        diagnostics: diagnostics.clone(),
        usage: usage.clone(),
        stop_reason: *stop_reason,
        stop_reason_raw: stop_reason_raw.clone(),
        error_message: error_message.clone(),
        timestamp: *timestamp,
        meta: meta.clone(),
    }
}

/// `header` (content-less, from [`header_of`]) equals `m` outside `content`.
fn header_eq(header: &AssistantMessage, m: &AssistantMessage) -> bool {
    let AssistantMessage {
        content: _,
        api,
        provider,
        model,
        response_model,
        response_id,
        diagnostics,
        usage,
        stop_reason,
        stop_reason_raw,
        error_message,
        timestamp,
        meta,
    } = m;
    header.api == *api
        && header.provider == *provider
        && header.model == *model
        && header.response_model == *response_model
        && header.response_id == *response_id
        && header.diagnostics == *diagnostics
        && header.usage == *usage
        && header.stop_reason == *stop_reason
        && header.stop_reason_raw == *stop_reason_raw
        && header.error_message == *error_message
        && header.timestamp == *timestamp
        && header.meta == *meta
}

/// Full `message_update` frame; `id` starts a delta chain the host tracks.
fn update_frame_full(message: &AgentMessage, ame: &Ev, id: Option<u64>) -> Value {
    let mut f = Map::new();
    f.insert("type".into(), Value::String("message_update".into()));
    f.insert("message".into(), message_value(message));
    f.insert("assistantMessageEvent".into(), assistant_event_slim(ame));
    if let Some(id) = id {
        f.insert("delta".into(), json!({ "id": id, "seq": 0 }));
    }
    Value::Object(f)
}

/// The message the host holds for the current delta chain.
struct UpdateCursor {
    id: u64,
    seq: u64,
    header: AssistantMessage,
    blocks: Vec<BlockShadow>,
}

impl UpdateCursor {
    /// Delta frame for `live`, or `None` when the move cannot be expressed
    /// (content shrank): the caller sends a full frame instead.
    fn delta(
        &mut self,
        live: &AssistantMessage,
        ame: &Ev,
        index: Option<usize>,
        open: &BTreeSet<usize>,
    ) -> Option<Value> {
        let content = &live.content;
        let known = self.blocks.len();
        if content.len() < known {
            return None;
        }
        // Like TS `snapshotAssistantMessageIncremental`: the targeted block,
        // blocks still open (a provider may patch them without an event), and
        // every block appended since the previous frame. Finalized blocks are
        // never mutated after their end event, so the host's copy stands.
        let mut candidates: Vec<(usize, bool)> = Vec::new();
        if let Some(i) = index.filter(|&i| i < known) {
            candidates.push((i, true));
        }
        for &i in open {
            if i < known && Some(i) != index {
                candidates.push((i, false));
            }
        }
        for i in known..content.len() {
            candidates.push((i, index == Some(i)));
        }
        let mut entries = Vec::new();
        for (i, targeted) in candidates {
            let block = &content[i];
            if let Some(e) = block_entry(i, block, self.blocks.get(i), ame, targeted) {
                entries.push(e);
                let shadow = BlockShadow::of(block);
                if i < self.blocks.len() {
                    self.blocks[i] = shadow;
                } else {
                    self.blocks.push(shadow);
                }
            }
        }
        let mut d = Map::new();
        self.seq += 1;
        d.insert("id".into(), json!(self.id));
        d.insert("seq".into(), json!(self.seq));
        d.insert("n".into(), json!(content.len()));
        if !header_eq(&self.header, live) {
            self.header = header_of(live);
            d.insert(
                "hdr".into(),
                message_value(&AgentMessage::from(self.header.clone())),
            );
        }
        d.insert("b".into(), Value::Array(entries));
        let mut f = Map::new();
        f.insert("type".into(), Value::String("message_update".into()));
        f.insert("assistantMessageEvent".into(), assistant_event_slim(ame));
        f.insert("delta".into(), Value::Object(d));
        Some(Value::Object(f))
    }
}

/// Encodes `message_update` events for the extension host as deltas against
/// the message the host last rebuilt (frame schema above). Every
/// `message_update` the host receives must go through [`Self::encode`] in
/// wire order; any update the host does not receive must be reported with
/// [`Self::invalidate`] so the next one goes out full.
#[derive(Default)]
pub struct ExtUpdateEncoder {
    next_id: u64,
    cursor: Option<UpdateCursor>,
    /// Content indices started but not yet ended (TS `openBlocks`).
    open: BTreeSet<usize>,
    /// Host connection generation the cursor belongs to.
    generation: usize,
    /// Set when the host answered `resync` or the emit failed.
    resync: Arc<AtomicBool>,
}

impl ExtUpdateEncoder {
    /// `message_start` / `message_end` / run end: the next update starts a
    /// new chain with a full frame.
    pub fn message_boundary(&mut self) {
        self.cursor = None;
        self.open.clear();
    }

    /// The host missed an update (or lost its state): next frame is full.
    pub fn invalidate(&mut self) {
        self.cursor = None;
    }

    /// Flag an emit-completion callback sets when the host could not apply a
    /// frame (answer `{resync: true}`) or the emit failed.
    #[must_use]
    pub fn resync_flag(&self) -> Arc<AtomicBool> {
        self.resync.clone()
    }

    /// A reconnected host (new generation) holds no state.
    pub fn set_generation(&mut self, generation: usize) {
        if generation != self.generation {
            self.generation = generation;
            self.cursor = None;
        }
    }

    /// Frame for one `message_update` (`message` == the event's `partial`).
    pub fn encode(&mut self, message: &AgentMessage, ame: &Ev) -> Value {
        if self.resync.swap(false, Ordering::AcqRel) {
            self.cursor = None;
        }
        let index = content_index(ame);
        if let Some(i) = index {
            match ame {
                Ev::TextStart { .. } | Ev::ThinkingStart { .. } | Ev::ToolCallStart { .. } => {
                    self.open.insert(i);
                }
                Ev::TextEnd { .. } | Ev::ThinkingEnd { .. } | Ev::ToolCallEnd { .. } => {
                    self.open.remove(&i);
                }
                _ => {}
            }
        }
        let AgentMessage::Standard(Message::Assistant(live)) = message else {
            self.cursor = None;
            return update_frame_full(message, ame, None);
        };
        let open = &self.open;
        if let Some(frame) = self
            .cursor
            .as_mut()
            .and_then(|c| c.delta(live, ame, index, open))
        {
            return frame;
        }
        self.next_id += 1;
        self.cursor = Some(UpdateCursor {
            id: self.next_id,
            seq: 0,
            header: header_of(live),
            blocks: live.content.iter().map(BlockShadow::of).collect(),
        });
        update_frame_full(message, ame, Some(self.next_id))
    }
}

/// Stamps `messageId` on message lifecycle frames (one id shared by the
/// start, updates and end of a message) and marks terminal run ends, like
/// omp's session event forwarder.
#[derive(Default)]
pub struct Framer {
    counter: u64,
    current: Option<String>,
}

impl Framer {
    pub fn frame(&mut self, e: &AgentEvent) -> Value {
        let mut v = frame(e);
        let is_message = matches!(
            e,
            AgentEvent::MessageStart { .. }
                | AgentEvent::MessageUpdate { .. }
                | AgentEvent::MessageEnd { .. }
        );
        if matches!(e, AgentEvent::MessageStart { .. }) {
            self.counter += 1;
            self.current = Some(format!("msg-{:016x}", self.counter));
        }
        if is_message {
            if let (Some(id), Some(obj)) = (&self.current, v.as_object_mut()) {
                obj.insert("messageId".into(), Value::String(id.clone()));
            }
        }
        if matches!(e, AgentEvent::MessageEnd { .. }) {
            self.current = None;
        }
        if matches!(e, AgentEvent::AgentEnd { .. }) {
            if let Some(obj) = v.as_object_mut() {
                obj.insert("isTerminal".into(), Value::Bool(true));
                obj.insert("yielded".into(), Value::Bool(true));
            }
        }
        v
    }
}

#[must_use]
pub fn frame(e: &AgentEvent) -> Value {
    match e {
        AgentEvent::AgentStart => json!({"type":"agent_start"}),
        AgentEvent::AgentEnd { messages } => {
            json!({"type":"agent_end","messages":messages.iter().map(message_value).collect::<Vec<_>>()})
        }
        AgentEvent::TurnStart => json!({"type":"turn_start"}),
        AgentEvent::TurnEnd {
            message,
            tool_results,
        } => {
            json!({"type":"turn_end","message":message_value(message),"toolResults":to_value(tool_results)})
        }
        AgentEvent::MessageStart { message } => {
            json!({"type":"message_start","message":message_value(message)})
        }
        AgentEvent::MessageUpdate {
            message,
            assistant_message_event,
        } => json!({
            "type":"message_update",
            "message": to_value::<AgentMessage>(message),
            "assistantMessageEvent": assistant_event(assistant_message_event),
        }),
        AgentEvent::MessageEnd { message } => {
            json!({"type":"message_end","message":message_value(message)})
        }
        AgentEvent::ToolExecutionStart {
            tool_call_id,
            tool_name,
            args,
        } => {
            json!({"type":"tool_execution_start","toolCallId":tool_call_id,"toolName":tool_name,"args":args})
        }
        AgentEvent::ToolExecutionUpdate {
            tool_call_id,
            tool_name,
            args,
            partial_result,
        } => json!({
            "type":"tool_execution_update","toolCallId":tool_call_id,"toolName":tool_name,
            "args":args,"partialResult":to_value(partial_result)
        }),
        AgentEvent::ToolExecutionEnd {
            tool_call_id,
            tool_name,
            result,
            is_error,
        } => json!({
            "type":"tool_execution_end","toolCallId":tool_call_id,"toolName":tool_name,
            "result":to_value(result),"isError":is_error
        }),
    }
}

// ---------------------------------------------------------------------------
// Session-level frames `Runtime` emits around the agent's own event stream
// (TS `AgentSession` / `TodoTracker` shapes, wire field names match omp).
// ---------------------------------------------------------------------------

/// omp `tool_stream_update` frame: an incremental preview chunk from a
/// streaming tool (`pi-agent-core` `tool_stream_update`). `update` is
/// tool-specific opaque data (e.g. an edit-tool diff hunk); the wire shape
/// matches TS exactly (`{type, toolCallId, toolName, update?}`).
#[must_use]
pub fn tool_stream_update_frame(tool_call_id: &str, tool_name: &str, update: &Value) -> Value {
    json!({"type":"tool_stream_update","toolCallId":tool_call_id,"toolName":tool_name,"update":update})
}

/// omp `notice` frame: transient info/warning/error row (TS `AgentSession.emitNotice`).
#[must_use]
pub fn notice_frame(level: &str, message: &str, source: Option<&str>) -> Value {
    let mut v = json!({"type":"notice","level":level,"message":message});
    if let Some(s) = source {
        v["source"] = json!(s);
    }
    v
}

/// omp `todo_reminder` frame (`TodoTracker.checkCompletion`): the run stopped
/// with `todos` still incomplete; `attempt`/`maxAttempts` count the nudges.
#[must_use]
pub fn todo_reminder_frame(todos: &Value, attempt: u32, max_attempts: u32) -> Value {
    json!({"type":"todo_reminder","todos":todos,"attempt":attempt,"maxAttempts":max_attempts})
}

/// omp `todo_auto_clear` frame: the todo plan auto-cleared (nothing left open).
#[must_use]
pub fn todo_auto_clear_frame() -> Value {
    json!({"type":"todo_auto_clear"})
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Rust mirror of `ext-host/host.ts` `rebuildMessageUpdate`: applies one
    /// encoder frame to the host state and returns the rebuilt message.
    fn rebuild(state: &mut Option<(u64, u64, Value)>, frame: &Value) -> Value {
        let d = &frame["delta"];
        if let Some(msg) = frame.get("message") {
            *state = Some((d["id"].as_u64().expect("full frame id"), 0, msg.clone()));
            return msg.clone();
        }
        let (id, seq, prev) = state.as_mut().expect("delta without a base");
        assert_eq!(d["id"].as_u64(), Some(*id), "chain id");
        assert_eq!(d["seq"].as_u64(), Some(*seq + 1), "no seq gap");
        let ame = &frame["assistantMessageEvent"];
        let prev_content = prev["content"].as_array().expect("content").clone();
        let mut content = prev_content.clone();
        for e in d["b"].as_array().expect("entries") {
            let i = usize::try_from(e["i"].as_u64().unwrap()).unwrap();
            let old = prev_content.get(i);
            let mut nb = e.get("blk").or(old).expect("block base").clone();
            let key = match nb["type"].as_str().unwrap() {
                "text" => "text",
                "thinking" => "thinking",
                "toolCall" => "arguments",
                t => panic!("block type {t}"),
            };
            let old_str = || old.unwrap()[key].as_str().unwrap().to_owned();
            let val = match e["v"].as_str().unwrap() {
                "k" => old.unwrap()[key].clone(),
                "a" => Value::String(old_str() + ame["delta"].as_str().unwrap()),
                "+" => Value::String(old_str() + e["x"].as_str().unwrap()),
                "=" => e["x"].clone(),
                "p" => apply_patch(&old.unwrap()[key], &e["x"]),
                v => panic!("op {v}"),
            };
            nb[key] = val;
            if i == content.len() {
                content.push(nb);
            } else {
                content[i] = nb;
            }
        }
        assert_eq!(content.len() as u64, d["n"].as_u64().unwrap(), "content length");
        let mut msg = d.get("hdr").cloned().unwrap_or_else(|| prev.clone());
        msg["content"] = Value::Array(content);
        *seq += 1;
        *prev = msg.clone();
        msg
    }

    fn apply_patch(prev: &Value, p: &Value) -> Value {
        if let Some(v) = p.get("=") {
            return v.clone();
        }
        if let Some(s) = p.get("+") {
            return Value::String(prev.as_str().unwrap().to_owned() + s.as_str().unwrap());
        }
        if let Some(o) = p.get("o") {
            let mut out = prev.clone();
            for (k, sp) in o.as_object().unwrap() {
                let pv = prev.get(k).cloned().unwrap_or(Value::Null);
                out[k.as_str()] = apply_patch(&pv, sp);
            }
            return out;
        }
        if let Some(a) = p.get("a") {
            let mut out = prev.as_array().unwrap().clone();
            out.resize(usize::try_from(p["n"].as_u64().unwrap()).unwrap(), Value::Null);
            for item in a.as_array().unwrap() {
                let i = usize::try_from(item[0].as_u64().unwrap()).unwrap();
                out[i] = apply_patch(prev.get(i).unwrap_or(&Value::Null), &item[1]);
            }
            return Value::Array(out);
        }
        panic!("bad patch {p}");
    }

    fn empty_message() -> AssistantMessage {
        serde_json::from_value(json!({
            "content": [], "api": "openai-completions", "provider": "p", "model": "m",
            "usage": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 0,
                      "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0}},
            "stopReason": "stop", "timestamp": 1
        }))
        .expect("message json")
    }

    /// Drives one encoder + mirror host through `live`, checking the rebuilt
    /// message is byte-identical (key order included) to a full serialization.
    struct Harness {
        enc: ExtUpdateEncoder,
        host: Option<(u64, u64, Value)>,
        live: AssistantMessage,
        frames: Vec<Value>,
    }

    impl Harness {
        fn new() -> Self {
            Self {
                enc: ExtUpdateEncoder::default(),
                host: None,
                live: empty_message(),
                frames: Vec::new(),
            }
        }

        fn send(&mut self, make: impl FnOnce(AssistantMessage) -> Ev) -> Value {
            let ame = make(self.live.clone());
            let message = AgentMessage::from(self.live.clone());
            let frame = self.enc.encode(&message, &ame);
            let rebuilt = rebuild(&mut self.host, &frame);
            assert_eq!(
                serde_json::to_string(&rebuilt).unwrap(),
                serde_json::to_string(&message_value(&message)).unwrap(),
                "rebuilt message differs after frame {frame}"
            );
            assert!(
                frame["assistantMessageEvent"].get("partial").is_none(),
                "partial never travels"
            );
            self.frames.push(frame.clone());
            frame
        }

        fn text(&mut self, i: usize) -> &mut String {
            match &mut self.live.content[i] {
                AssistantContent::Text(t) => &mut t.text,
                AssistantContent::Thinking(t) => &mut t.thinking,
                AssistantContent::ToolCall(_) => panic!("tool call"),
            }
        }
    }

    fn is_full(frame: &Value) -> bool {
        frame.get("message").is_some()
    }

    #[test]
    fn deltas_rebuild_the_full_message() {
        let mut h = Harness::new();
        h.live.content.push(AssistantContent::Thinking(ThinkingContent {
            thinking: String::new(),
            thinking_signature: None,
            redacted: None,
        }));
        let f = h.send(|partial| Ev::ThinkingStart { content_index: 0, partial });
        assert!(is_full(&f), "first update of a message is full");
        for chunk in ["plan ", "the ", "reads"] {
            h.text(0).push_str(chunk);
            let f = h.send(|partial| Ev::ThinkingDelta {
                content_index: 0,
                delta: chunk.into(),
                partial,
            });
            assert!(!is_full(&f));
            assert_eq!(f["delta"]["b"][0]["v"], "a", "append from the event's delta");
            assert!(f["delta"]["b"][0].get("x").is_none(), "fragment not duplicated");
        }
        if let AssistantContent::Thinking(t) = &mut h.live.content[0] {
            t.thinking_signature = Some("sig".into());
        }
        let f = h.send(|partial| Ev::ThinkingEnd { content_index: 0, partial });
        assert_eq!(f["delta"]["b"][0]["v"], "k");
        assert_eq!(f["delta"]["b"][0]["blk"]["thinkingSignature"], "sig");

        h.live.content.push(AssistantContent::Text(TextContent {
            text: String::new(),
            text_signature: None,
        }));
        h.send(|partial| Ev::TextStart { content_index: 1, partial });
        for chunk in ["Hello", ", wörld"] {
            h.text(1).push_str(chunk);
            let f = h.send(|partial| Ev::TextDelta {
                content_index: 1,
                delta: chunk.into(),
                partial,
            });
            let b = f["delta"]["b"].as_array().unwrap();
            assert_eq!(b.len(), 1, "finalized thinking block is not resent");
            assert!(!f.to_string().contains("plan the reads"));
        }
        let text = h.text(1).clone();
        h.send(|partial| Ev::TextEnd { content_index: 1, content: text, partial });

        h.live.content.push(AssistantContent::ToolCall(ToolCall {
            id: "call-1".into(),
            name: "read".into(),
            arguments: json!({}),
            thought_signature: None,
        }));
        h.send(|partial| Ev::ToolCallStart { content_index: 2, partial });
        for args in [
            json!({"path": "src"}),
            json!({"path": "src/main.rs"}),
            json!({"path": "src/main.rs", "ranges": [1]}),
            json!({"path": "src/main.rs", "ranges": [1, 20]}),
            json!({"path": "src/main.rs", "ranges": [1, 20], "limit": 5}),
        ] {
            if let AssistantContent::ToolCall(tc) = &mut h.live.content[2] {
                tc.arguments = args;
            }
            let f = h.send(|partial| Ev::ToolCallDelta {
                content_index: 2,
                delta: "…".into(),
                partial,
            });
            assert_eq!(f["delta"]["b"][0]["v"], "p");
        }
        let tool_call = match &h.live.content[2] {
            AssistantContent::ToolCall(tc) => tc.clone(),
            _ => unreachable!(),
        };
        h.send(|partial| Ev::ToolCallEnd { content_index: 2, tool_call, partial });

        // A header-only move (usage update) ships `hdr`, no content entry.
        h.live.usage.output = 42;
        h.live.stop_reason = pa_agent::types::StopReason::ToolUse;
        let text = h.text(1).clone();
        let f = h.send(|partial| Ev::TextEnd { content_index: 1, content: text, partial });
        assert_eq!(f["delta"]["hdr"]["usage"]["output"], 42);
        assert_eq!(f["delta"]["hdr"]["content"], json!([]));
        assert_eq!(h.frames.iter().filter(|f| is_full(f)).count(), 1);
    }

    #[test]
    fn missed_updates_and_resync_stay_exact() {
        let mut h = Harness::new();
        h.live.content.push(AssistantContent::Text(TextContent {
            text: String::new(),
            text_signature: None,
        }));
        h.send(|partial| Ev::TextStart { content_index: 0, partial });
        // An update the host never saw (no frame), then one whose own delta
        // does not cover the gap: the missing suffix travels explicitly.
        h.text(0).push_str("skipped ");
        h.text(0).push_str("seen");
        let f = h.send(|partial| Ev::TextDelta {
            content_index: 0,
            delta: "seen".into(),
            partial,
        });
        assert_eq!(f["delta"]["b"][0]["v"], "+");
        assert_eq!(f["delta"]["b"][0]["x"], "skipped seen");
        // Rewritten (non-append) text goes out as a literal.
        *h.text(0) = "rewritten".into();
        let f = h.send(|partial| Ev::TextDelta {
            content_index: 0,
            delta: "x".into(),
            partial,
        });
        assert_eq!(f["delta"]["b"][0]["v"], "=");
        // Reordered argument keys cannot be patched: literal arguments.
        h.live.content.push(AssistantContent::ToolCall(ToolCall {
            id: "c".into(),
            name: "grep".into(),
            arguments: json!({"b": 1}),
            thought_signature: None,
        }));
        h.send(|partial| Ev::ToolCallStart { content_index: 1, partial });
        if let AssistantContent::ToolCall(tc) = &mut h.live.content[1] {
            tc.arguments = json!({"a": 1, "b": 1});
        }
        let f = h.send(|partial| Ev::ToolCallDelta {
            content_index: 1,
            delta: String::new(),
            partial,
        });
        assert!(f["delta"]["b"][0]["x"].get("=").is_some());
        // Host answered resync: next frame is full with a new chain id.
        h.enc.resync_flag().store(true, Ordering::SeqCst);
        let f = h.send(|partial| Ev::ToolCallDelta {
            content_index: 1,
            delta: String::new(),
            partial,
        });
        assert!(is_full(&f));
        assert_eq!(f["delta"]["id"], 2);
        // invalidate() (host skipped an update) and a reconnect do the same.
        h.enc.invalidate();
        assert!(is_full(&h.send(|partial| Ev::ToolCallDelta {
            content_index: 1,
            delta: String::new(),
            partial,
        })));
        h.enc.set_generation(1);
        assert!(is_full(&h.send(|partial| Ev::ToolCallDelta {
            content_index: 1,
            delta: String::new(),
            partial,
        })));
        // Shrunk content cannot be expressed as a delta.
        h.live.content.pop();
        assert!(is_full(&h.send(|partial| Ev::TextDelta {
            content_index: 0,
            delta: String::new(),
            partial,
        })));
        // A new message always starts with a full frame.
        h.enc.message_boundary();
        assert!(is_full(&h.send(|partial| Ev::TextDelta {
            content_index: 0,
            delta: String::new(),
            partial,
        })));
    }

    #[test]
    fn value_patch_round_trips() {
        let cases = [
            (json!({}), json!({"a": "x"})),
            (json!({"a": "x"}), json!({"a": "xyz"})),
            (json!({"a": [1]}), json!({"a": [1, {"b": "c"}]})),
            (json!({"a": [1, {"b": "c"}]}), json!({"a": [1, {"b": "cd", "e": null}]})),
            (json!({"a": 1, "b": 2}), json!({"b": 2})),
            (json!([1, 2]), json!([1])),
            (json!("ab"), json!(3)),
        ];
        for (prev, next) in cases {
            let p = value_patch(&prev, &next);
            let got = apply_patch(&prev, &p);
            assert_eq!(
                serde_json::to_string(&got).unwrap(),
                serde_json::to_string(&next).unwrap(),
                "patch {p}"
            );
        }
    }
}
