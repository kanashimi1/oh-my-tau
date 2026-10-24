//! `todo`: structured task phases for tracking progress within a session.
//! Port of omp `tools/todo.ts`: the ops (init/start/done/drop/rm/block/
//! unblock/append/view), the in-progress normalization, completion
//! transitions and the summary text the model reads back.
//!
//! State: the phases of the last successful state-changing call. Like TS, a
//! successful call's `details.phases` is the durable snapshot (the runtime
//! persists tool results); `phases_from_entries` rebuilds it from transcript
//! entries on resume/fork, mirroring `getLatestTodoPhasesFromEntries`.
//! Failed batches are discarded wholesale and reported as errors.

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::sync::Arc;

use parking_lot::Mutex;

use anyhow::{anyhow, Result};
use pa_agent::abort::AbortSignal;
use pa_agent::types::{AgentTool, AgentToolResult, AgentToolUpdateCallback, ToolResultContent};
use pa_agent::BoxFut;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

const DESCRIPTION: &str = "Tasks identified by verbatim content, NEVER generated IDs (task-1). Unique, stable task/phase names; lost text: view, NEVER guess.\nBefore work, init for 3+ steps, requested task sets, or new instructions. MUST list EVERY user item separately (phased/numbered/bulleted/N); NEVER omit or remember leftovers.\nAfter successful mutation: no active means earliest pending starts (phase order); multiple active means only earliest stays. Blocked NEVER starts automatically; unblock returns pending. Done out of order may rewind pointer but NEVER reopen completed. Mark done immediately; follow phase order.\nExternal waits (user/agent/service): block with optional reason suppresses stop reminder, starts next pending. Unblock when actionable; append a clearing task for agent-actionable blocker.\nNEVER call todo alone: init with first work; done/start with next action.";

/// Default phase name for a flat `items` init (omp `DEFAULT_INIT_PHASE`).
const DEFAULT_INIT_PHASE: &str = "Tasks";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Pending,
    InProgress,
    Completed,
    Abandoned,
    Blocked,
}

impl Status {
    fn as_str(self) -> &'static str {
        match self {
            Status::Pending => "pending",
            Status::InProgress => "in_progress",
            Status::Completed => "completed",
            Status::Abandoned => "abandoned",
            Status::Blocked => "blocked",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TodoItem {
    pub content: String,
    pub status: Status,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blocker: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TodoPhase {
    pub name: String,
    pub tasks: Vec<TodoItem>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct Transition {
    phase: String,
    content: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Op {
    Init,
    Start,
    Done,
    Rm,
    Drop,
    Block,
    Unblock,
    Append,
    View,
}

impl Op {
    fn as_str(self) -> &'static str {
        match self {
            Op::Init => "init",
            Op::Start => "start",
            Op::Done => "done",
            Op::Rm => "rm",
            Op::Drop => "drop",
            Op::Block => "block",
            Op::Unblock => "unblock",
            Op::Append => "append",
            Op::View => "view",
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct ListEntry {
    phase: String,
    items: Vec<String>,
}

/// One todo op: the tool's params object.
#[derive(Debug, Clone, Deserialize)]
struct Entry {
    op: Op,
    list: Option<Vec<ListEntry>>,
    task: Option<String>,
    phase: Option<String>,
    items: Option<Vec<String>>,
    reason: Option<String>,
}

// =============================================================================
// State helpers
// =============================================================================

/// `truthy` in the TS source: an empty string counts as absent.
fn present(value: Option<&str>) -> Option<&str> {
    value.filter(|v| !v.is_empty())
}

fn find_task(phases: &[TodoPhase], content: &str) -> Option<(usize, usize)> {
    phases.iter().enumerate().find_map(|(pi, phase)| {
        phase
            .tasks
            .iter()
            .position(|t| t.content == content)
            .map(|ti| (pi, ti))
    })
}

fn total_tasks(phases: &[TodoPhase]) -> usize {
    phases.iter().map(|p| p.tasks.len()).sum()
}

/// `/^task-\d+$/`: an ID-shaped reference, which the model must not use.
fn is_task_id(content: &str) -> bool {
    content
        .strip_prefix("task-")
        .is_some_and(|digits| !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()))
}

fn resolve_task(
    phases: &[TodoPhase],
    content: Option<&str>,
    errors: &mut Vec<String>,
) -> Option<(usize, usize)> {
    let Some(content) = present(content) else {
        errors.push("Missing task content".to_string());
        return None;
    };
    let hit = find_task(phases, content);
    if hit.is_none() {
        if is_task_id(content) {
            errors.push(format!(
                "Task \"{content}\" not found. Tasks are referenced by content, not by IDs — pass the task's full text from the previous result."
            ));
        } else {
            let empty_hint = if total_tasks(phases) == 0 {
                " (todo list is empty — was it replaced or not yet created?)"
            } else {
                ""
            };
            errors.push(format!("Task \"{content}\" not found{empty_hint}"));
        }
    }
    hit
}

fn resolve_phase(
    phases: &[TodoPhase],
    name: Option<&str>,
    errors: &mut Vec<String>,
) -> Option<usize> {
    let Some(name) = present(name) else {
        errors.push("Missing phase name".to_string());
        return None;
    };
    let hit = phases.iter().position(|p| p.name == name);
    if hit.is_none() {
        errors.push(format!("Phase \"{name}\" not found"));
    }
    hit
}

/// Task coordinates an op targets: one task, one phase's tasks, or every task.
fn task_targets(
    phases: &[TodoPhase],
    entry: &Entry,
    errors: &mut Vec<String>,
) -> Vec<(usize, usize)> {
    if present(entry.task.as_deref()).is_some() {
        return resolve_task(phases, entry.task.as_deref(), errors)
            .into_iter()
            .collect();
    }
    if present(entry.phase.as_deref()).is_some() {
        return resolve_phase(phases, entry.phase.as_deref(), errors)
            .map(|pi| (0..phases[pi].tasks.len()).map(|ti| (pi, ti)).collect())
            .unwrap_or_default();
    }
    phases
        .iter()
        .enumerate()
        .flat_map(|(pi, p)| (0..p.tasks.len()).map(move |ti| (pi, ti)))
        .collect()
}

fn init_phases(entry: &Entry, errors: &mut Vec<String>) -> Vec<TodoPhase> {
    // Models often flatten a single-phase init into `{op:"init", items:[...]}`;
    // accept that shape as a one-phase list, like TS.
    let list: Vec<(String, Vec<String>)> = match (&entry.list, &entry.items) {
        (Some(list), _) => list
            .iter()
            .map(|e| (e.phase.clone(), e.items.clone()))
            .collect(),
        (None, Some(items)) if !items.is_empty() => vec![(
            entry
                .phase
                .clone()
                .unwrap_or_else(|| DEFAULT_INIT_PHASE.to_string()),
            items.clone(),
        )],
        _ => {
            errors.push("Missing list for init operation".to_string());
            return Vec::new();
        }
    };
    // Duplicate phase names / task contents would be unaddressable, so reject them.
    let mut seen_phases: HashSet<&str> = HashSet::new();
    let mut seen_tasks: HashSet<&str> = HashSet::new();
    for (phase, items) in &list {
        if !seen_phases.insert(phase) {
            errors.push(format!("Duplicate phase \"{phase}\" in init list"));
        }
        for content in items {
            if !seen_tasks.insert(content) {
                errors.push(format!("Duplicate task \"{content}\" in init list"));
            }
        }
    }
    list.into_iter()
        .map(|(name, items)| TodoPhase {
            name,
            tasks: items
                .into_iter()
                .map(|content| TodoItem {
                    content,
                    status: Status::Pending,
                    blocker: None,
                })
                .collect(),
        })
        .collect()
}

fn append_items(phases: &mut Vec<TodoPhase>, entry: &Entry, errors: &mut Vec<String>) {
    let Some(phase_name) = present(entry.phase.as_deref()) else {
        errors.push("Missing phase name for append operation".to_string());
        return;
    };
    let items = match &entry.items {
        Some(items) if !items.is_empty() => items,
        _ => {
            errors.push("Missing items for append operation".to_string());
            return;
        }
    };
    // Validate the whole batch before mutating: a failing op reports every duplicate.
    let mut seen: HashSet<&str> = HashSet::new();
    let mut has_duplicate = false;
    for content in items {
        if !seen.insert(content) || find_task(phases, content).is_some() {
            errors.push(format!("Task \"{content}\" already exists"));
            has_duplicate = true;
        }
    }
    if has_duplicate {
        return;
    }
    let pi = if let Some(pi) = phases.iter().position(|p| p.name == phase_name) {
        pi
    } else {
        phases.push(TodoPhase {
            name: phase_name.to_string(),
            tasks: Vec::new(),
        });
        phases.len() - 1
    };
    for content in items {
        phases[pi].tasks.push(TodoItem {
            content: content.clone(),
            status: Status::Pending,
            blocker: None,
        });
    }
}

fn remove_tasks(phases: &mut [TodoPhase], entry: &Entry, errors: &mut Vec<String>) {
    if present(entry.task.as_deref()).is_some() {
        if let Some((pi, ti)) = resolve_task(phases, entry.task.as_deref(), errors) {
            phases[pi].tasks.remove(ti);
        }
        return;
    }
    if present(entry.phase.as_deref()).is_some() {
        if let Some(pi) = resolve_phase(phases, entry.phase.as_deref(), errors) {
            phases[pi].tasks.clear();
        }
        return;
    }
    for phase in phases.iter_mut() {
        phase.tasks.clear();
    }
}

/// Collapse whitespace runs (incl. newlines) so a blocker note stays one line.
fn normalize_reason(reason: Option<&str>) -> Option<String> {
    let collapsed = reason?.split_whitespace().collect::<Vec<_>>().join(" ");
    (!collapsed.is_empty()).then_some(collapsed)
}

fn apply_entry(phases: &mut Vec<TodoPhase>, entry: &Entry, errors: &mut Vec<String>) {
    match entry.op {
        Op::Init => *phases = init_phases(entry, errors),
        Op::Start => {
            let Some(hit) = resolve_task(phases, entry.task.as_deref(), errors) else {
                return;
            };
            for (pi, phase) in phases.iter_mut().enumerate() {
                for (ti, candidate) in phase.tasks.iter_mut().enumerate() {
                    if candidate.status == Status::InProgress && (pi, ti) != hit {
                        candidate.status = Status::Pending;
                    }
                }
            }
            phases[hit.0].tasks[hit.1].status = Status::InProgress;
        }
        Op::Done | Op::Drop => {
            let status = if entry.op == Op::Done {
                Status::Completed
            } else {
                Status::Abandoned
            };
            for (pi, ti) in task_targets(phases, entry, errors) {
                phases[pi].tasks[ti].status = status;
            }
        }
        Op::Block => {
            if present(entry.task.as_deref()).is_none() && present(entry.phase.as_deref()).is_none()
            {
                errors.push("block requires a task or phase target".to_string());
                return;
            }
            let reason = normalize_reason(entry.reason.as_deref());
            for (pi, ti) in task_targets(phases, entry, errors) {
                // Only open work can be blocked; an already-blocked task may refine its note.
                let task = &mut phases[pi].tasks[ti];
                if !matches!(
                    task.status,
                    Status::Pending | Status::InProgress | Status::Blocked
                ) {
                    continue;
                }
                task.status = Status::Blocked;
                task.blocker.clone_from(&reason);
            }
        }
        Op::Unblock => {
            if present(entry.task.as_deref()).is_none() && present(entry.phase.as_deref()).is_none()
            {
                errors.push("unblock requires a task or phase target".to_string());
                return;
            }
            for (pi, ti) in task_targets(phases, entry, errors) {
                let task = &mut phases[pi].tasks[ti];
                if task.status == Status::Blocked {
                    task.status = Status::Pending;
                    task.blocker = None;
                }
            }
        }
        Op::Rm => remove_tasks(phases, entry, errors),
        Op::Append => append_items(phases, entry, errors),
        Op::View => {}
    }
}

/// Keep exactly one in-progress task (the first); if none, start the first pending one.
fn normalize_in_progress(phases: &mut [TodoPhase]) {
    let ordered: Vec<(usize, usize)> = phases
        .iter()
        .enumerate()
        .flat_map(|(pi, p)| (0..p.tasks.len()).map(move |ti| (pi, ti)))
        .collect();
    if ordered.is_empty() {
        return;
    }
    let in_progress: Vec<(usize, usize)> = ordered
        .iter()
        .copied()
        .filter(|&(pi, ti)| phases[pi].tasks[ti].status == Status::InProgress)
        .collect();
    if in_progress.len() > 1 {
        for &(pi, ti) in &in_progress[1..] {
            phases[pi].tasks[ti].status = Status::Pending;
        }
    }
    if !in_progress.is_empty() {
        return;
    }
    if let Some(&(pi, ti)) = ordered
        .iter()
        .find(|&&(pi, ti)| phases[pi].tasks[ti].status == Status::Pending)
    {
        phases[pi].tasks[ti].status = Status::InProgress;
    }
}

fn apply_params(phases: Vec<TodoPhase>, entry: &Entry) -> (Vec<TodoPhase>, Vec<String>) {
    let mut next = phases;
    let mut errors = Vec::new();
    apply_entry(&mut next, entry, &mut errors);
    normalize_in_progress(&mut next);
    (next, errors)
}

/// Tasks that became `completed` in this call (were open before).
fn completion_transitions(previous: &[TodoPhase], updated: &[TodoPhase]) -> Vec<Transition> {
    let mut previous_status: HashMap<(&str, &str), Status> = HashMap::new();
    for phase in previous {
        for task in &phase.tasks {
            previous_status.insert((phase.name.as_str(), task.content.as_str()), task.status);
        }
    }
    let mut out = Vec::new();
    for phase in updated {
        for task in &phase.tasks {
            if task.status != Status::Completed {
                continue;
            }
            if let Some(prev) = previous_status.get(&(phase.name.as_str(), task.content.as_str())) {
                if *prev != Status::Completed {
                    out.push(Transition {
                        phase: phase.name.clone(),
                        content: task.content.clone(),
                    });
                }
            }
        }
    }
    out
}

// =============================================================================
// Parameter resolution
// =============================================================================

fn parse_entry(raw: &Value) -> Result<Entry, String> {
    let entry: Entry = serde_json::from_value(raw.clone()).map_err(|e| e.to_string())?;
    if let Some(list) = &entry.list {
        if list.iter().any(|e| e.items.is_empty()) {
            return Err("list[].items: must contain at least 1 task".to_string());
        }
    }
    Ok(entry)
}

/// Infer a missing `op` from an unambiguous payload (omp `inferTodoOp`).
fn infer_op(raw: &Value, has_existing: bool) -> Option<&'static str> {
    let non_empty = |key: &str| {
        raw.get(key)
            .and_then(Value::as_array)
            .is_some_and(|a| !a.is_empty())
    };
    if non_empty("list") {
        return Some("init");
    }
    if non_empty("items") {
        if raw
            .get("phase")
            .and_then(Value::as_str)
            .is_some_and(|p| !p.is_empty())
        {
            return Some("append");
        }
        if !has_existing {
            return Some("init");
        }
    }
    None
}

/// Validate the arguments, repairing one recoverable shape: a missing `op`
/// alongside an unambiguous payload.
fn resolve_params(raw: &Value, has_existing: bool) -> Result<Entry, String> {
    match parse_entry(raw) {
        Ok(entry) => Ok(entry),
        Err(err) => {
            if raw.get("op").is_none() {
                if let Some(op) = infer_op(raw, has_existing) {
                    let mut repaired = raw.clone();
                    repaired["op"] = json!(op);
                    if let Ok(entry) = parse_entry(&repaired) {
                        return Ok(entry);
                    }
                }
            }
            Err(format!("Invalid todo arguments: {err}"))
        }
    }
}

// =============================================================================
// Summary text
// =============================================================================

/// The text the model reads back after each call (omp `formatSummary`).
fn format_summary(phases: &[TodoPhase], errors: &[String], read_only: bool) -> String {
    let tasks: Vec<&TodoItem> = phases.iter().flat_map(|p| p.tasks.iter()).collect();
    if tasks.is_empty() {
        if !errors.is_empty() {
            return format!("Errors: {}", errors.join("; "));
        }
        return if read_only {
            "Todo list is empty.".to_string()
        } else {
            "Todo list cleared.".to_string()
        };
    }
    let is_open = |s: Status| matches!(s, Status::Pending | Status::InProgress);
    let is_closed = |s: Status| matches!(s, Status::Completed | Status::Abandoned);

    let remaining: Vec<(&str, &TodoItem)> = phases
        .iter()
        .flat_map(|p| {
            p.tasks
                .iter()
                .filter(|t| is_open(t.status))
                .map(move |t| (p.name.as_str(), t))
        })
        .collect();

    let current_idx = phases
        .iter()
        .position(|p| p.tasks.iter().any(|t| is_open(t.status)))
        .unwrap_or(phases.len() - 1);
    let current = &phases[current_idx];
    let done = current.tasks.iter().filter(|t| is_closed(t.status)).count();

    let mut lines: Vec<String> = Vec::new();
    if !errors.is_empty() {
        lines.push(format!("Errors: {}", errors.join("; ")));
    }
    if remaining.is_empty() {
        lines.push("Remaining items: none.".to_string());
    } else {
        lines.push(format!("Remaining items ({}):", remaining.len()));
        for (phase, task) in &remaining {
            lines.push(format!(
                "  - {} [{}] ({phase})",
                task.content,
                task.status.as_str()
            ));
        }
    }
    let closed_all = tasks.iter().filter(|t| is_closed(t.status)).count();
    let blocked_all = tasks.iter().filter(|t| t.status == Status::Blocked).count();
    let worked_ahead = phases
        .iter()
        .enumerate()
        .any(|(idx, p)| idx > current_idx && p.tasks.iter().any(|t| is_closed(t.status)));
    let blocked_part = if blocked_all > 0 {
        format!(", {blocked_all} blocked")
    } else {
        String::new()
    };
    lines.push(format!(
        "Overall: {closed_all}/{} done, {} open{blocked_part}.",
        tasks.len(),
        remaining.len()
    ));
    let suffix = if worked_ahead {
        " — earliest phase with open tasks; the in-progress pointer auto-advances to the earliest open task on each completion, so it can sit behind out-of-order work (nothing was un-completed)."
    } else {
        "."
    };
    lines.push(format!(
        "Active phase {}/{} \"{}\" ({done}/{}){suffix}",
        current_idx + 1,
        phases.len(),
        current.name,
        current.tasks.len()
    ));
    for phase in phases {
        lines.push(format!("  {}:", phase.name));
        for task in &phase.tasks {
            let checkbox = if task.status == Status::Completed {
                "[X]"
            } else {
                "[ ]"
            };
            let tag = match task.status {
                Status::InProgress => " (in progress)".to_string(),
                Status::Abandoned => " (dropped)".to_string(),
                Status::Blocked => match &task.blocker {
                    Some(blocker) => format!(" (blocked: {blocker})"),
                    None => " (blocked)".to_string(),
                },
                _ => String::new(),
            };
            let mut line = String::new();
            let _ = write!(line, "    - {checkbox} {}{tag}", task.content);
            lines.push(line);
        }
    }
    lines.join("\n")
}

// =============================================================================
// Session snapshot
// =============================================================================

/// Rebuild the todo phases from transcript entries (omp
/// `getLatestTodoPhasesFromEntries`): the latest successful, state-changing
/// `todo` tool result (or `user_todo_edit` custom entry) on the branch.
/// Entries are the JSON session entries in branch order. The runtime calls
/// this on session load/fork and builds the tool via [`tool_with_session`].
#[must_use]
pub fn phases_from_entries(entries: &[Value]) -> Vec<TodoPhase> {
    for entry in entries.iter().rev() {
        if let Some(phases) = canonical_phases(entry) {
            return phases;
        }
    }
    Vec::new()
}

#[allow(dead_code)] // helper of phases_from_entries
fn canonical_phases(entry: &Value) -> Option<Vec<TodoPhase>> {
    if entry.get("type").and_then(Value::as_str) == Some("custom")
        && entry.get("customType").and_then(Value::as_str) == Some("user_todo_edit")
    {
        return parse_phases(entry.get("data")?.get("phases")?);
    }
    if entry.get("type").and_then(Value::as_str) != Some("message") {
        return None;
    }
    let message = entry.get("message")?;
    if message.get("role").and_then(Value::as_str) != Some("toolResult")
        || message.get("toolName").and_then(Value::as_str) != Some("todo")
        || message.get("isError").and_then(Value::as_bool) == Some(true)
    {
        return None;
    }
    let details = message.get("details")?;
    if details.get("op").and_then(Value::as_str) == Some("view") {
        return None;
    }
    parse_phases(details.get("phases")?)
}

#[allow(dead_code)] // helper of phases_from_entries
fn parse_phases(value: &Value) -> Option<Vec<TodoPhase>> {
    serde_json::from_value(value.clone()).ok()
}

// =============================================================================
// Tool
// =============================================================================

/// The `todo` tool: owns the phases of its session.
pub struct TodoTool {
    schema: Value,
    state: Mutex<Vec<TodoPhase>>,
    /// omp reports `"session"` when a session file exists, `"memory"` otherwise.
    storage: &'static str,
}

impl TodoTool {
    /// A tool starting from `phases` (empty for a new session; rebuild from the
    /// transcript with [`phases_from_entries`] on resume).
    #[must_use]
    pub fn with_phases(phases: Vec<TodoPhase>) -> Self {
        Self::with_session(phases, Some("session"))
    }

    /// A tool with explicit storage reporting: `session_file` mirrors omp's
    /// `session.getSessionFile()` (`Some` → `"session"`, `None` → `"memory"`).
    #[must_use]
    pub fn with_session(phases: Vec<TodoPhase>, session_file: Option<&str>) -> Self {
        Self {
            schema: schema(),
            state: Mutex::new(phases),
            storage: if session_file.is_some() {
                "session"
            } else {
                "memory"
            },
        }
    }

    fn run(&self, raw: &Value) -> Result<AgentToolResult> {
        let previous = self.state.lock().clone();
        let entry = resolve_params(raw, !previous.is_empty()).map_err(|e| anyhow!(e))?;
        let read_only = entry.op == Op::View;
        let (updated, errors) = if read_only {
            (previous.clone(), Vec::new())
        } else {
            apply_params(previous.clone(), &entry)
        };
        // A batch with any error is discarded wholesale; state and summary stay at previous.
        let failed = !errors.is_empty();
        let effective = if failed {
            previous.clone()
        } else {
            updated.clone()
        };
        let completed = if read_only || failed {
            Vec::new()
        } else {
            completion_transitions(&previous, &updated)
        };
        if !read_only && !failed {
            *self.state.lock() = updated;
        }
        let summary = format_summary(&effective, &errors, read_only);
        let mut details = json!({
            "op": entry.op.as_str(),
            "phases": effective,
            "storage": self.storage,
        });
        if !completed.is_empty() {
            details["completedTasks"] = json!(completed);
        }
        // TS: soft failures are `isError: true` results keeping content/details.
        Ok(AgentToolResult {
            content: vec![ToolResultContent::text(summary)],
            details,
            terminate: None,
            is_error: if failed { Some(true) } else { None },
        })
    }
}

impl AgentTool for TodoTool {
    fn name(&self) -> &'static str {
        "todo"
    }
    fn description(&self) -> &str {
        DESCRIPTION
    }
    fn parameters(&self) -> &Value {
        &self.schema
    }
    fn label(&self) -> &'static str {
        "Todo"
    }
    fn execute(
        self: Arc<Self>,
        _tool_call_id: String,
        params: Value,
        _signal: AbortSignal,
        _on_update: AgentToolUpdateCallback,
    ) -> BoxFut<'static, Result<AgentToolResult>> {
        Box::pin(async move { self.run(&params) })
    }
}

/// The `todo` tool, ready to register.
#[must_use]
pub fn tool() -> Arc<dyn AgentTool> {
    Arc::new(TodoTool::with_phases(Vec::new()))
}

/// Parameter schema, byte-equal (as JSON) to omp's `todoSchema`.
#[must_use]
pub fn schema() -> Value {
    json!({"type":"object","properties":{
        "op":{"type":"string","enum":["init","start","done","rm","drop","block","unblock","append","view"]},
        "list":{"type":"array","description":"init phases","items":{"type":"object","properties":{
            "phase":{"type":"string"},
            "items":{"type":"array","items":{"type":"string"},"minItems":1}
        },"required":["phase","items"]}},
        "task":{"type":"string","description":"exact task text"},
        "phase":{"type":"string"},
        "items":{"type":"array","description":"flat init/append tasks","items":{"type":"string"}},
        "reason":{"type":"string","description":"block reason"}
    },"required":["op"]})
}
#[cfg(test)]
mod tests {
    use super::*;

    /// Recorded TS `todo` transcripts (`tests-tools/todo_golden.ts`).
    const GOLDEN: &str = include_str!("../../../../tests-tools/todo_golden.json");

    #[test]
    fn replays_ts_golden_transcripts() {
        let scenarios: Vec<Value> = serde_json::from_str(GOLDEN).unwrap();
        assert!(!scenarios.is_empty());
        for scenario in &scenarios {
            let name = scenario["name"].as_str().unwrap();
            let tool = TodoTool::with_phases(Vec::new());
            for (i, step) in scenario["steps"].as_array().unwrap().iter().enumerate() {
                let ctx = format!("{name} step {i}");
                let outcome = tool.run(&step["params"]);
                let expected_text = step["text"].as_str().unwrap();
                let result = outcome.unwrap_or_else(|e| panic!("{ctx}: unexpected error {e}"));
                assert_eq!(
                    result.content,
                    vec![ToolResultContent::text(expected_text)],
                    "{ctx}: text"
                );
                assert_eq!(result.details, step["details"], "{ctx}: details");
                assert_eq!(
                    result.is_error,
                    if step["isError"].as_bool() == Some(true) {
                        Some(true)
                    } else {
                        None
                    },
                    "{ctx}: isError"
                );
            }
        }
    }

    #[test]
    fn missing_op_is_repaired_only_when_unambiguous() {
        let list = json!({"list": [{"phase": "A", "items": ["x"]}]});
        assert_eq!(resolve_params(&list, true).unwrap().op, Op::Init);
        let items = json!({"items": ["x"]});
        assert_eq!(resolve_params(&items, false).unwrap().op, Op::Init);
        assert!(resolve_params(&items, true)
            .unwrap_err()
            .starts_with("Invalid todo arguments: "));
        assert!(resolve_params(&json!({"task": "x"}), true).is_err());
    }

    #[test]
    fn phases_rebuild_from_branch_entries() {
        let entries = vec![
            json!({"type": "message", "message": {"role": "toolResult", "toolName": "todo", "isError": false,
                "details": {"op": "init", "phases": [{"name": "Old", "tasks": [{"content": "a", "status": "pending"}]}]}}}),
            json!({"type": "message", "message": {"role": "toolResult", "toolName": "todo", "isError": false,
                "details": {"op": "view", "phases": [{"name": "View", "tasks": []}]}}}),
            json!({"type": "custom", "customType": "user_todo_edit", "data": {"phases": [{"name": "Edited", "tasks": []}]}}),
        ];
        assert_eq!(
            phases_from_entries(&entries),
            vec![TodoPhase {
                name: "Edited".into(),
                tasks: vec![]
            }]
        );
        assert_eq!(phases_from_entries(&entries[..2])[0].name, "Old");
    }

    #[test]
    fn storage_reflects_session_file_presence() {
        let init = json!({"op": "init", "list": [{"phase": "P", "items": ["a"]}]});
        let with_file = TodoTool::with_session(Vec::new(), Some("/tmp/s.jsonl"));
        assert_eq!(
            with_file.run(&init).unwrap().details["storage"],
            json!("session")
        );
        let no_file = TodoTool::with_session(Vec::new(), None);
        assert_eq!(
            no_file.run(&init).unwrap().details["storage"],
            json!("memory")
        );
    }

    #[test]
    fn rehydrated_tool_continues_from_transcript_state() {
        let entries = vec![
            json!({"type": "message", "message": {"role": "toolResult", "toolName": "todo", "isError": false,
                "details": {"op": "init", "phases": [{"name": "P", "tasks": [
                    {"content": "a", "status": "completed"},
                    {"content": "b", "status": "pending"}]}]}}}),
        ];
        let tool = TodoTool::with_session(phases_from_entries(&entries), Some("s.jsonl"));
        let result = tool.run(&json!({"op": "done", "task": "b"})).unwrap();
        assert_eq!(result.details["op"], json!("done"));
        let phases = result.details["phases"].as_array().unwrap();
        assert_eq!(phases[0]["tasks"][1]["status"], json!("completed"));
        // State unknown to a fresh tool resolves after rehydration (TS isError).
        let fresh = TodoTool::with_phases(Vec::new());
        let missing = fresh.run(&json!({"op": "done", "task": "b"})).unwrap();
        assert_eq!(missing.is_error, Some(true));
    }

    #[test]
    fn schema_matches_ts_shape() {
        let s = schema();
        assert_eq!(s["required"], json!(["op"]));
        assert_eq!(s["properties"]["op"]["enum"].as_array().unwrap().len(), 9);
        assert_eq!(
            s["properties"]["list"]["items"]["properties"]["items"]["minItems"],
            json!(1)
        );
    }
}
