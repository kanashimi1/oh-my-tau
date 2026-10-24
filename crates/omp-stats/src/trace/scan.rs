//! Single-transcript scan (`scanTranscript` in trace.ts): model/tool/turn/
//! background spans plus markers, and the cross-track facts (task results,
//! request counts). Entry shapes are loose [`Value`] views — session JSONL is
//! outside-controlled data, so every field is optional and re-checked.

use std::collections::HashMap;

use serde_json::Value;

use crate::types::{TraceMarker, TraceMarkerKind, TraceSpan, TraceSpanKind, TraceTrack};

use super::{compact_tokens, content_head, head_text, js_truthy, to_ms, DETAIL_MAX, LABEL_MAX};

/// `tool_execution_start` marker facts joined to tool spans by call id.
struct ToolStartFact {
    start: i64,
    detail: Option<String>,
}

/// Pending tool call recorded from an assistant turn's `toolCall` blocks.
struct PendingToolCall {
    call_id: String,
    name: String,
    args_preview: Option<String>,
    assistant_entry_id: Option<String>,
    model_end: i64,
}

/// Background job opened by an async-running tool result.
struct PendingBackgroundJob {
    job_id: String,
    start: i64,
    label: String,
    entry_id: Option<String>,
    end: Option<i64>,
}

/// Task tool result row, used to place subagent spans on the parent track.
#[derive(Debug, Clone)]
pub(crate) struct TaskResultFact {
    pub child_id: String,
    pub agent: Option<String>,
    pub task: Option<String>,
    pub duration_ms: Option<i64>,
    pub result_entry_id: Option<String>,
    pub tool_span_end: i64,
}

/// Everything one transcript contributes before cross-track assembly.
#[derive(Debug, Clone)]
pub(crate) struct TrackScan {
    pub track: TraceTrack,
    pub task_results: Vec<TaskResultFact>,
    pub requests: i64,
    pub tool_calls: i64,
    /// Model requests whose zero cost is unknown spend (mirrors db's unpriced predicate).
    pub unpriced_requests: i64,
    pub cwd: Option<String>,
    pub title: Option<String>,
    /// Session-header envelope timestamp (ms), for spanless traces.
    pub header_ts: Option<i64>,
    pub first_activity: Option<i64>,
    pub last_activity: Option<i64>,
}

/// Keep only the active branch: the parentId chain walked back from the last
/// entry with an id.
fn active_chain(views: &[Value]) -> Vec<&Value> {
    let mut by_id: HashMap<&str, &Value> = HashMap::new();
    let mut last: Option<&Value> = None;
    for entry in views {
        if let Some(id) = entry.get("id").and_then(Value::as_str) {
            if !id.is_empty() {
                by_id.insert(id, entry);
                last = Some(entry);
            }
        }
    }
    let Some(mut cursor) = last else {
        return Vec::new();
    };
    let mut keep = std::collections::HashSet::new();
    while let Some(id) = cursor.get("id").and_then(Value::as_str) {
        if !keep.insert(id) {
            break;
        }
        cursor = match cursor.get("parentId").and_then(Value::as_str) {
            Some(parent) => match by_id.get(parent) {
                Some(parent_entry) => parent_entry,
                None => break,
            },
            None => break,
        };
    }
    views
        .iter()
        .filter(|e| {
            e.get("id")
                .and_then(Value::as_str)
                .map(|id| keep.contains(id))
                .unwrap_or(false)
        })
        .collect()
}

/// Argument projection for tool-span details: command/path/intent, else nothing.
fn tool_start_detail(data: &Value) -> Option<String> {
    let mut candidate: Option<&str> = None;
    if let Some(args) = data.get("args").filter(|a| a.is_object()) {
        candidate = args
            .get("command")
            .and_then(Value::as_str)
            .or_else(|| args.get("path").and_then(Value::as_str));
    }
    candidate = candidate.or_else(|| data.get("intent").and_then(Value::as_str));
    candidate
        .filter(|s| !s.trim().is_empty())
        .map(|s| head_text(s, DETAIL_MAX))
}

/// Serialize tool-call arguments into a bounded preview.
fn args_preview(args: Option<&Value>) -> Option<String> {
    // TS: `if (args === undefined || args === null) return undefined`.
    let args = args.filter(|a| !a.is_null())?;
    serde_json::to_string(args)
        .ok()
        .map(|s| head_text(&s, DETAIL_MAX))
}

/// A finite JS number field (`typeof x === "number" && Number.isFinite(x)`).
fn finite_number(value: Option<&Value>) -> Option<f64> {
    value.and_then(Value::as_f64).filter(|v| v.is_finite())
}

/**
 * Scan one transcript into a track: model/tool/turn/background spans plus
 * markers, and the cross-track facts (task results, request counts).
 */
pub(crate) fn scan_transcript(
    entries: Vec<Value>,
    track_id: &str,
    parent_track_id: Option<&str>,
    file: &str,
) -> TrackScan {
    let chain = active_chain(&entries);
    let mut spans: Vec<TraceSpan> = Vec::new();
    let mut markers: Vec<TraceMarker> = Vec::new();
    let mut pending_tools: Vec<PendingToolCall> = Vec::new();
    let mut tool_starts: HashMap<String, ToolStartFact> = HashMap::new();
    let mut tool_results: HashMap<String, (i64, bool, Option<String>)> = HashMap::new();
    let mut background_jobs: Vec<PendingBackgroundJob> = Vec::new();
    let mut active_background_jobs: HashMap<String, usize> = HashMap::new();
    let mut turns: Vec<(i64, String, Option<String>)> = Vec::new();
    let mut task_results: Vec<TaskResultFact> = Vec::new();

    let mut session_exit: Option<(i64, std::collections::HashSet<String>)> = None;
    let mut cwd: Option<String> = None;
    let mut header_title: Option<String> = None;
    let mut header_ts: Option<i64> = None;
    let mut slot_title: Option<String> = None;
    let mut init_agent: Option<String> = None;
    let mut init_model: Option<String> = None;
    let mut first_assistant_model: Option<String> = None;
    let mut requests = 0i64;
    let mut unpriced_requests = 0i64;
    let mut last_chain_ts = 0i64;

    // Title slot and session header may sit outside the id chain — scan all entries.
    for raw in &entries {
        match raw.get("type").and_then(Value::as_str) {
            Some("title") => {
                if let Some(title) = raw.get("title").and_then(Value::as_str) {
                    slot_title = Some(title.to_string());
                }
            }
            Some("session") => {
                if let Some(c) = raw.get("cwd").and_then(Value::as_str) {
                    cwd = Some(c.to_string());
                }
                if let Some(t) = raw.get("title").and_then(Value::as_str) {
                    header_title = Some(t.to_string());
                }
                header_ts = to_ms(raw.get("timestamp"));
            }
            Some("session_init") => {
                if let Some(a) = raw.get("agent").and_then(Value::as_str) {
                    init_agent = Some(a.to_string());
                }
                if let Some(m) = raw.get("resolvedModel").and_then(Value::as_str) {
                    init_model = Some(m.to_string());
                }
            }
            _ => {}
        }
    }

    for entry in chain {
        let envelope_ts = to_ms(entry.get("timestamp"));
        if let Some(ts) = envelope_ts {
            last_chain_ts = last_chain_ts.max(ts);
        }

        match entry.get("type").and_then(Value::as_str) {
            Some("custom") => {
                let data = entry.get("data").cloned().unwrap_or(Value::Null);
                match entry.get("customType").and_then(Value::as_str) {
                    Some("tool_execution_start") => {
                        if let Some(call_id) = data.get("toolCallId").and_then(Value::as_str) {
                            let start = to_ms(data.get("startedAt")).or(envelope_ts);
                            if let Some(start) = start {
                                tool_starts.insert(
                                    call_id.to_string(),
                                    ToolStartFact {
                                        start,
                                        detail: tool_start_detail(&data),
                                    },
                                );
                            }
                        }
                    }
                    Some("session_exit") => {
                        let recorded_at = to_ms(data.get("recordedAt")).or(envelope_ts);
                        let mut pending_ids = std::collections::HashSet::new();
                        if let Some(pending) =
                            data.get("pendingToolCalls").and_then(Value::as_array)
                        {
                            for p in pending {
                                if let Some(id) = p.get("toolCallId").and_then(Value::as_str) {
                                    pending_ids.insert(id.to_string());
                                }
                            }
                        }
                        if let Some(recorded_at) = recorded_at {
                            session_exit = Some((recorded_at, pending_ids));
                            markers.push(TraceMarker {
                                time: recorded_at,
                                kind: TraceMarkerKind::SessionExit,
                                label: data
                                    .get("kind")
                                    .and_then(Value::as_str)
                                    .unwrap_or("exit")
                                    .to_string(),
                            });
                        }
                    }
                    _ => {}
                }
                continue;
            }
            Some("compaction") => {
                if let Some(ts) = envelope_ts {
                    markers.push(TraceMarker {
                        time: ts,
                        kind: TraceMarkerKind::Compaction,
                        label: format!(
                            "compaction {}→{}",
                            compact_tokens(entry.get("tokensBefore")),
                            compact_tokens(entry.get("tokensAfter"))
                        ),
                    });
                }
                continue;
            }
            Some("model_change") => {
                if let Some(ts) = envelope_ts {
                    markers.push(TraceMarker {
                        time: ts,
                        kind: TraceMarkerKind::ModelChange,
                        label: entry
                            .get("model")
                            .and_then(Value::as_str)
                            .unwrap_or("model change")
                            .to_string(),
                    });
                }
                continue;
            }
            Some("mode_change") => {
                if let Some(ts) = envelope_ts {
                    markers.push(TraceMarker {
                        time: ts,
                        kind: TraceMarkerKind::ModeChange,
                        label: entry
                            .get("mode")
                            .and_then(Value::as_str)
                            .unwrap_or("mode change")
                            .to_string(),
                    });
                }
                continue;
            }
            Some("reset_boundary") => {
                if let Some(ts) = envelope_ts {
                    markers.push(TraceMarker {
                        time: ts,
                        kind: TraceMarkerKind::Reset,
                        label: "reset".into(),
                    });
                }
                continue;
            }
            _ => {}
        }

        if entry.get("type").and_then(Value::as_str) != Some("message") {
            continue;
        }
        let Some(msg) = entry.get("message") else {
            continue;
        };
        let entry_id = entry
            .get("id")
            .and_then(Value::as_str)
            .map(ToString::to_string);

        match msg.get("role").and_then(Value::as_str) {
            Some("assistant") => {
                requests += 1;
                // Mirror the db's unpriced predicate so the trace headline agrees
                // with Costs/Recent Requests: a zero that is unknown spend, not
                // free. Counted before the span-timing bail so a fully dateless
                // entry still surfaces as N/A rather than vanishing.
                let total_tokens = crate::parser::resolve_usage_total(msg.get("usage"));
                let cost_obj = msg.get("usage").and_then(|u| u.get("cost"));
                let cost_total = finite_number(cost_obj.and_then(|c| c.get("total")));
                if total_tokens > 0.0 && cost_total.is_none_or(|c| c == 0.0) {
                    let provider = msg.get("provider").and_then(Value::as_str).unwrap_or("");
                    let model = msg.get("model").and_then(Value::as_str).unwrap_or("");
                    if provider == "xai-oauth" && !model.is_empty() {
                        // getBundledModel("xai", model)?.cost — direct lookup, no
                        // provider fallback; priced = any per-token rate nonzero.
                        let priced = crate::catalog::bundled_model_cost("xai", model)
                            .map(|c| crate::catalog::has_billable_cost(&c))
                            .unwrap_or(false);
                        if !priced {
                            unpriced_requests += 1;
                        }
                    } else if cost_obj.is_none() && !provider.is_empty() && !model.is_empty() {
                        let has_timestamp = finite_number(msg.get("timestamp"))
                            .map(|t| t > 0.0)
                            .unwrap_or(false)
                            || envelope_ts.is_some();
                        if !has_timestamp
                            && crate::catalog::is_scheduled_catalog_model(provider, model)
                        {
                            unpriced_requests += 1;
                        }
                    }
                }
                let mut start = finite_number(msg.get("timestamp")).map(|t| t as i64);
                let mut end = finite_number(msg.get("completedAt"))
                    .map(|t| t as i64)
                    .or_else(|| {
                        start.and_then(|s| {
                            finite_number(msg.get("duration")).map(|d| s.saturating_add(d as i64))
                        })
                    });
                if start.is_none() {
                    let fallback_end = end.or(envelope_ts);
                    let Some(fallback_end) = fallback_end else {
                        continue;
                    };
                    end = Some(fallback_end);
                    start = Some(
                        fallback_end.saturating_sub(
                            finite_number(msg.get("duration"))
                                .map(|d| d as i64)
                                .unwrap_or(0),
                        ),
                    );
                }
                let mut end = end.unwrap_or_else(|| envelope_ts.unwrap_or(start.unwrap()));
                let start = start.unwrap();
                end = end.max(start);
                if first_assistant_model.is_none() {
                    first_assistant_model = msg
                        .get("model")
                        .and_then(Value::as_str)
                        .map(ToString::to_string);
                }

                let mut span = TraceSpan {
                    id: format!("{track_id}:{}", entry_id.as_deref().unwrap_or("")),
                    kind: TraceSpanKind::Model,
                    start,
                    end,
                    label: head_text(
                        msg.get("model").and_then(Value::as_str).unwrap_or("model"),
                        LABEL_MAX,
                    ),
                    ..TraceSpan::default()
                };
                span.detail = content_head(msg.get("content"), DETAIL_MAX);
                span.entry_id = entry_id.clone();
                span.model = msg
                    .get("model")
                    .and_then(Value::as_str)
                    .map(ToString::to_string);
                if total_tokens > 0.0 {
                    span.tokens = Some(total_tokens as i64);
                }
                if let Some(cost) = cost_total {
                    span.cost = Some(cost);
                }
                span.ttft = finite_number(msg.get("ttft"));
                if msg.get("stopReason").and_then(Value::as_str) == Some("error")
                    || js_truthy(msg.get("errorMessage"))
                {
                    span.is_error = Some(true);
                }
                let span_end = span.end;
                spans.push(span);
                if let Some(blocks) = msg.get("content").and_then(Value::as_array) {
                    for block in blocks {
                        if block.get("type").and_then(Value::as_str) != Some("toolCall") {
                            continue;
                        }
                        let Some(call_id) = block.get("id").and_then(Value::as_str) else {
                            continue;
                        };
                        let Some(name) = block.get("name").and_then(Value::as_str) else {
                            continue;
                        };
                        pending_tools.push(PendingToolCall {
                            call_id: call_id.to_string(),
                            name: name.to_string(),
                            args_preview: args_preview(block.get("arguments")),
                            assistant_entry_id: entry_id.clone(),
                            model_end: span_end,
                        });
                    }
                }
            }
            Some("toolResult") => {
                let Some(call_id) = msg.get("toolCallId").and_then(Value::as_str) else {
                    continue;
                };
                let Some(end) = finite_number(msg.get("timestamp"))
                    .map(|t| t as i64)
                    .or(envelope_ts)
                else {
                    continue;
                };
                let tool_name = msg
                    .get("toolName")
                    .and_then(Value::as_str)
                    .unwrap_or("tool");
                tool_results.insert(
                    call_id.to_string(),
                    (
                        end,
                        msg.get("isError") == Some(&Value::Bool(true)),
                        entry_id.clone(),
                    ),
                );

                let details = msg.get("details");
                let async_info = details.and_then(|d| d.get("async"));
                if async_info
                    .filter(|a| a.get("state").and_then(Value::as_str) == Some("running"))
                    .and_then(|a| a.get("jobId"))
                    .and_then(Value::as_str)
                    .is_some()
                {
                    let job_id = async_info
                        .unwrap()
                        .get("jobId")
                        .unwrap()
                        .as_str()
                        .unwrap()
                        .to_string();
                    let job = PendingBackgroundJob {
                        job_id: job_id.clone(),
                        start: end,
                        label: head_text(&format!("{tool_name} job"), LABEL_MAX),
                        entry_id: entry_id.clone(),
                        end: None,
                    };
                    active_background_jobs.insert(job_id, background_jobs.len());
                    background_jobs.push(job);
                }
                if tool_name == "task" {
                    if let Some(results) = details
                        .and_then(|d| d.get("results"))
                        .and_then(Value::as_array)
                    {
                        for result in results {
                            let Some(child_id) = result.get("id").and_then(Value::as_str) else {
                                continue;
                            };
                            task_results.push(TaskResultFact {
                                child_id: child_id.to_string(),
                                agent: result
                                    .get("agent")
                                    .and_then(Value::as_str)
                                    .map(ToString::to_string),
                                task: result
                                    .get("task")
                                    .and_then(Value::as_str)
                                    .map(ToString::to_string),
                                duration_ms: finite_number(result.get("durationMs"))
                                    .map(|d| d as i64),
                                result_entry_id: entry_id.clone(),
                                tool_span_end: end,
                            });
                        }
                    }
                }
            }
            Some("custom")
                if msg.get("customType").and_then(Value::as_str) == Some("async-result") =>
            {
                let close_at = finite_number(msg.get("timestamp"))
                    .map(|t| t as i64)
                    .or(envelope_ts);
                let jobs = msg
                    .get("details")
                    .and_then(|d| d.get("jobs"))
                    .and_then(Value::as_array);
                if let (Some(close_at), Some(jobs)) = (close_at, jobs) {
                    for job in jobs {
                        let Some(job_id) = job.get("jobId").and_then(Value::as_str) else {
                            continue;
                        };
                        let Some(&idx) = active_background_jobs.get(job_id) else {
                            continue;
                        };
                        if background_jobs[idx].end.is_some() {
                            continue;
                        }
                        background_jobs[idx].end = Some(close_at);
                        active_background_jobs.remove(job_id);
                    }
                }
            }
            Some("user") if msg.get("synthetic") != Some(&Value::Bool(true)) => {
                let Some(time) = finite_number(msg.get("timestamp"))
                    .map(|t| t as i64)
                    .or(envelope_ts)
                else {
                    continue;
                };
                turns.push((
                    time,
                    content_head(msg.get("content"), LABEL_MAX)
                        .unwrap_or_else(|| "user".to_string()),
                    entry_id.clone(),
                ));
            }
            _ => {}
        }
    }

    // Tool spans: join call blocks with start markers and results.
    for pending in &pending_tools {
        let marker = tool_starts.get(&pending.call_id);
        let result = tool_results.get(&pending.call_id);
        let start = marker.map(|m| m.start).unwrap_or(pending.model_end);
        let (end, unterminated) = match result {
            Some((end, _, _)) => (*end, false),
            None => match &session_exit {
                Some((recorded_at, pending_ids)) if pending_ids.contains(&pending.call_id) => {
                    (*recorded_at, true)
                }
                _ => (
                    if last_chain_ts != 0 {
                        last_chain_ts
                    } else {
                        start
                    },
                    true,
                ),
            },
        };
        let mut span = TraceSpan {
            id: format!(
                "{track_id}:{}:{}",
                pending.assistant_entry_id.as_deref().unwrap_or(""),
                pending.call_id
            ),
            kind: TraceSpanKind::Tool,
            start,
            end: end.max(start),
            label: head_text(&pending.name, LABEL_MAX),
            ..TraceSpan::default()
        };
        span.tool_call_id = Some(pending.call_id.clone());
        span.detail = marker
            .and_then(|m| m.detail.clone())
            .or_else(|| pending.args_preview.clone());
        span.entry_id = result
            .and_then(|(_, _, e)| e.clone())
            .or_else(|| pending.assistant_entry_id.clone());
        if let Some((_, true, _)) = result {
            span.is_error = Some(true);
        }
        if unterminated {
            span.unterminated = Some(true);
        }
        spans.push(span);
    }

    // Background spans: opened by an async-running tool result, closed by the async-result delivery.
    for (background_index, job) in background_jobs.iter().enumerate() {
        let end = job.end.unwrap_or(if last_chain_ts != 0 {
            last_chain_ts
        } else {
            job.start
        });
        let mut span = TraceSpan {
            id: format!("{track_id}:bg:{background_index}:{}", job.job_id),
            kind: TraceSpanKind::Background,
            start: job.start,
            end: end.max(job.start),
            label: job.label.clone(),
            ..TraceSpan::default()
        };
        span.entry_id = job.entry_id.clone();
        if job.end.is_none() {
            span.unterminated = Some(true);
        }
        spans.push(span);
    }

    // Turn spans: user message → max end of spans starting before the next turn.
    turns.sort_by_key(|t| t.0);
    for (i, (time, label, entry_id)) in turns.iter().enumerate() {
        let next_time = turns.get(i + 1).map(|t| t.0).unwrap_or(i64::MAX);
        let mut end = *time;
        for span in &spans {
            if span.start >= *time && span.start < next_time && span.end > end {
                end = span.end;
            }
        }
        let mut span = TraceSpan {
            id: format!(
                "{track_id}:{}",
                entry_id.as_deref().unwrap_or(&format!("turn{i}"))
            ),
            kind: TraceSpanKind::Turn,
            start: *time,
            end,
            label: label.clone(),
            ..TraceSpan::default()
        };
        span.entry_id = entry_id.clone();
        spans.push(span);
    }

    spans.sort_by(|a, b| a.start.cmp(&b.start).then_with(|| a.end.cmp(&b.end)));
    markers.sort_by_key(|m| m.time);

    let mut first_activity: Option<i64> = None;
    let mut last_activity: Option<i64> = None;
    for span in &spans {
        first_activity = Some(first_activity.map_or(span.start, |s| s.min(span.start)));
        last_activity = Some(last_activity.map_or(span.end, |e| e.max(span.end)));
    }

    TrackScan {
        track: TraceTrack {
            id: track_id.to_string(),
            parent_id: parent_track_id.map(ToString::to_string),
            label: String::new(),
            agent: init_agent,
            model: init_model.or(first_assistant_model),
            file: file.to_string(),
            spans,
            markers,
        },
        task_results,
        requests,
        tool_calls: pending_tools.len() as i64,
        unpriced_requests,
        cwd,
        title: slot_title.or(header_title),
        header_ts,
        first_activity,
        last_activity,
    }
}
