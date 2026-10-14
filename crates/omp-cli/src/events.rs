//! `AgentEvent` -> `omp` RPC event frames (field names match omp).

use pa_agent::stream::AssistantMessageEvent as Ev;
use pa_agent::types::{AgentEvent, AgentMessage, AssistantMessage};
use serde_json::{json, Value};

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
