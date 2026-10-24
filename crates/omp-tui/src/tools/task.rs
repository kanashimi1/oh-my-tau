//! `task` (subagent) renderer — port of `pi-tui/src/tools/task.ts`,
//! `tools/agent-tree.ts`, `tools/task-yield-assembly.ts`, and
//! `tools/task-repair-args.ts`.
//!
//! Covers the P0/P1 surface: the pending/streaming task card (dispatch glyph,
//! agent header label, per-item batch rows), live per-subagent progress rows
//! fed by `tool_execution_update.partialResult` (status glyph, model badge,
//! description, current tool + elapsed, retry / rate-limit badges, token/cost
//! stats, completion percent), nested subagent trees with the render-depth
//! cap ([`MAX_NESTED_TASK_RENDER_DEPTH`]), settled per-agent result rows, and
//! final yield assembly ([`yield_assembly::assemble_yield_result`]).
//! Double-encoded prose repair lives in [`repair`].
//!
//! Widths are measured in code points (chars), matching the TS code-point cap
//! in `oneLineLabel`; the crate has no `unicode-width` dependency yet.

use std::collections::HashMap;

use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use serde_json::Value;

use crate::theme::Theme;

use super::{error_detail_lines, ToolRenderer};

/// Nested-agent rendering stops at this depth (TS `MAX_NESTED_TASK_RENDER_DEPTH`).
pub const MAX_NESTED_TASK_RENDER_DEPTH: usize = 8;
/// Display cap for a normalized one-line label (TS `LABEL_MAX`).
pub const LABEL_MAX: usize = 80;
/// Collapsed agent lists show this many rows before folding (TS `COLLAPSED_AGENT_LIMIT`).
pub const COLLAPSED_AGENT_LIMIT: usize = 4;
/// A tool call running longer than this gets a warning-colored elapsed (TS, 5000).
const TOOL_ELAPSED_WARN_MS: u64 = 5_000;

/// Spinner frames used for running rows (TS default spinner set).
pub const SPINNER_FRAMES: &[char] = &['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];

/// Lifecycle phase of one subagent (TS `AgentProgress.status`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum AgentStatus {
    /// Spawned, not yet running.
    #[default]
    Pending,
    /// Actively working.
    Running,
    /// Finished successfully.
    Completed,
    /// Finished with an error.
    Failed,
    /// Aborted (Esc / cancel).
    Aborted,
}

impl AgentStatus {
    /// Parse a status string; unknown values map to [`AgentStatus::Pending`].
    #[must_use]
    pub fn from_str(status: &str) -> Self {
        match status {
            "running" | "started" | "active" => Self::Running,
            "completed" | "done" => Self::Completed,
            "failed" | "error" => Self::Failed,
            "aborted" | "cancelled" | "canceled" => Self::Aborted,
            _ => Self::Pending,
        }
    }

    /// The wire spelling of this status.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Aborted => "aborted",
        }
    }

    fn is_live(self) -> bool {
        matches!(self, Self::Pending | Self::Running)
    }

    fn is_failed(self) -> bool {
        matches!(self, Self::Failed | Self::Aborted)
    }
}

/// A recently completed tool call on a subagent (TS `AgentProgress.recentTools`).
#[derive(Debug, Clone, Default)]
pub struct RecentTool {
    /// Tool name.
    pub tool: String,
    /// Serialized arguments (preview string).
    pub args: String,
    /// Intent the model attached to the call, if any.
    pub intent: Option<String>,
}

/// Auto-retry sleep state while a subagent waits on a provider retry.
#[derive(Debug, Clone, Default)]
pub struct RetryState {
    /// Current attempt (1-based).
    pub attempt: u64,
    /// Maximum attempts.
    pub max_attempts: u64,
    /// Delay before the next attempt.
    pub delay_ms: u64,
    /// Provider error message.
    pub error_message: String,
    /// When the sleep started (epoch ms).
    pub started_at_ms: u64,
}

/// Terminal retry failure once the subagent gave up retrying.
#[derive(Debug, Clone, Default)]
pub struct RetryFailure {
    /// Attempts made.
    pub attempt: u64,
    /// Final provider error message.
    pub error_message: String,
}

/// Live progress snapshot for one subagent (TS `AgentProgress`), parsed
/// defensively because args and updates stream in token by token.
#[derive(Debug, Clone, Default)]
pub struct AgentProgress {
    /// Dispatch index within the spawn.
    pub index: u64,
    /// Registry id (`AdjectiveNoun` or caller-provided `name`).
    pub id: String,
    /// Agent type (e.g. `"scout"`); empty means the generic worker.
    pub agent: String,
    /// Current lifecycle phase.
    pub status: AgentStatus,
    /// The work assignment.
    pub task: String,
    /// Expanded assignment text, when the agent produced one.
    pub assignment: Option<String>,
    /// Latest activity gist.
    pub description: Option<String>,
    /// Currently executing tool name, if any.
    pub current_tool: Option<String>,
    /// Serialized arguments of the current tool call.
    pub current_tool_args: Option<String>,
    /// Intent attached to the current call, if any.
    pub current_tool_intent: Option<String>,
    /// When the current tool call started (epoch ms).
    pub current_tool_start_ms: Option<u64>,
    /// Most recent completed tool calls.
    pub recent_tools: Vec<RecentTool>,
    /// Cumulative tool-call count.
    pub tool_count: u64,
    /// Assistant requests (message_end events) across the run.
    pub requests: u64,
    /// Cumulative input+output+cacheWrite tokens.
    pub tokens: u64,
    /// Latest per-turn context size.
    pub context_tokens: Option<u64>,
    /// Model context window, when known.
    pub context_window: Option<u64>,
    /// Cumulative cost in USD.
    pub cost: f64,
    /// Elapsed run time.
    pub duration_ms: u64,
    /// Resolved model display string (`<provider>/<id>`).
    pub resolved_model: Option<String>,
    /// Resolved model identity without the thinking suffix.
    pub resolved_model_identity: Option<String>,
    /// True when a live advisor is attached.
    pub advisor: bool,
    /// The agent's self-estimate of completion (0-100).
    pub completion_percent: Option<u8>,
    /// Auto-retry sleep state, while waiting on a provider retry.
    pub retry_state: Option<RetryState>,
    /// Terminal retry failure.
    pub retry_failure: Option<RetryFailure>,
    /// Per-tool data extracted from nested calls (`yield`, `task`, ...).
    pub extracted_tool_data: Option<Value>,
    /// Snapshot of the agent's own in-flight `task` call details.
    pub inflight_task_details: Option<Value>,
}

fn get_string(v: &Value, key: &str) -> Option<String> {
    v.get(key).and_then(Value::as_str).map(ToOwned::to_owned)
}

fn get_u64(v: &Value, key: &str) -> Option<u64> {
    v.get(key)
        .and_then(Value::as_u64)
        .or_else(|| v.get(key).and_then(Value::as_f64).map(|f| f as u64))
}

fn get_f64(v: &Value, key: &str) -> Option<f64> {
    v.get(key)
        .and_then(Value::as_f64)
        .or_else(|| v.get(key).and_then(Value::as_u64).map(|n| n as f64))
}

fn get_bool(v: &Value, key: &str) -> bool {
    v.get(key).and_then(Value::as_bool).unwrap_or(false)
}

impl AgentProgress {
    /// Parse one progress snapshot from a wire value; `None` for non-objects.
    #[must_use]
    pub fn from_value(v: &Value) -> Option<Self> {
        let obj = v.as_object()?;
        // Lifecycle payloads nest under `progress`; accept both shapes.
        let src = v.get("progress").filter(|p| p.is_object()).unwrap_or(v);
        let mut p = Self {
            index: get_u64(src, "index").unwrap_or(0),
            id: get_string(src, "id")
                .or_else(|| get_string(v, "id"))
                .unwrap_or_default(),
            agent: get_string(src, "agent")
                .or_else(|| get_string(v, "agent"))
                .unwrap_or_default(),
            status: src
                .get("status")
                .and_then(Value::as_str)
                .map(AgentStatus::from_str)
                .unwrap_or(AgentStatus::Pending),
            task: get_string(src, "task")
                .or_else(|| get_string(v, "task"))
                .unwrap_or_default(),
            assignment: get_string(src, "assignment"),
            description: get_string(src, "description"),
            current_tool: get_string(src, "currentTool")
                .or_else(|| get_string(src, "current_tool")),
            current_tool_args: get_string(src, "currentToolArgs")
                .or_else(|| get_string(src, "current_tool_args")),
            current_tool_intent: get_string(src, "currentToolIntent")
                .or_else(|| get_string(src, "current_tool_intent")),
            current_tool_start_ms: get_u64(src, "currentToolStartMs")
                .or_else(|| get_u64(src, "current_tool_start_ms")),
            recent_tools: Vec::new(),
            tool_count: get_u64(src, "toolCount")
                .or_else(|| get_u64(src, "tool_count"))
                .unwrap_or(0),
            requests: get_u64(src, "requests").unwrap_or(0),
            tokens: get_u64(src, "tokens").unwrap_or(0),
            context_tokens: get_u64(src, "contextTokens")
                .or_else(|| get_u64(src, "context_tokens")),
            context_window: get_u64(src, "contextWindow")
                .or_else(|| get_u64(src, "context_window")),
            cost: get_f64(src, "cost").unwrap_or(0.0),
            duration_ms: get_u64(src, "durationMs")
                .or_else(|| get_u64(src, "duration_ms"))
                .unwrap_or(0),
            resolved_model: get_string(src, "resolvedModel")
                .or_else(|| get_string(src, "resolved_model")),
            resolved_model_identity: get_string(src, "resolvedModelIdentity")
                .or_else(|| get_string(src, "resolved_model_identity")),
            advisor: get_bool(src, "advisor"),
            completion_percent: get_u64(src, "completionPercent")
                .or_else(|| get_u64(src, "completion_percent"))
                .map(|n| u8::try_from(n.min(100)).unwrap_or(100)),
            retry_state: None,
            retry_failure: None,
            extracted_tool_data: src
                .get("extractedToolData")
                .or_else(|| src.get("extracted_tool_data"))
                .cloned(),
            inflight_task_details: src
                .get("inflightTaskDetails")
                .or_else(|| src.get("inflight_task_details"))
                .cloned(),
        };
        if let Some(recent) = src
            .get("recentTools")
            .or_else(|| src.get("recent_tools"))
            .and_then(Value::as_array)
        {
            p.recent_tools = recent
                .iter()
                .filter_map(|t| {
                    t.as_object().map(|_| RecentTool {
                        tool: get_string(t, "tool").unwrap_or_default(),
                        args: get_string(t, "args").unwrap_or_default(),
                        intent: get_string(t, "intent"),
                    })
                })
                .collect();
        }
        if let Some(rs) = src
            .get("retryState")
            .or_else(|| src.get("retry_state"))
            .filter(|r| r.is_object())
        {
            p.retry_state = Some(RetryState {
                attempt: get_u64(rs, "attempt").unwrap_or(0),
                max_attempts: get_u64(rs, "maxAttempts")
                    .or_else(|| get_u64(rs, "max_attempts"))
                    .unwrap_or(0),
                delay_ms: get_u64(rs, "delayMs")
                    .or_else(|| get_u64(rs, "delay_ms"))
                    .unwrap_or(0),
                error_message: get_string(rs, "errorMessage")
                    .or_else(|| get_string(rs, "error_message"))
                    .unwrap_or_default(),
                started_at_ms: get_u64(rs, "startedAtMs")
                    .or_else(|| get_u64(rs, "started_at_ms"))
                    .unwrap_or(0),
            });
        }
        if let Some(rf) = src
            .get("retryFailure")
            .or_else(|| src.get("retry_failure"))
            .filter(|r| r.is_object())
        {
            p.retry_failure = Some(RetryFailure {
                attempt: get_u64(rf, "attempt").unwrap_or(0),
                error_message: get_string(rf, "errorMessage")
                    .or_else(|| get_string(rf, "error_message"))
                    .unwrap_or_default(),
            });
        }
        let _ = obj;
        Some(p)
    }
}
/// Width in code points (chars); the crate's stand-in for TS `visibleWidth`.
fn visible_width(s: &str) -> usize {
    s.chars().count()
}

/// Code-point-capped truncation appending `suffix` (usually `""` or `"…"`).
fn truncate_to_width(s: &str, max: usize, suffix: &str) -> String {
    if visible_width(s) <= max {
        return s.to_string();
    }
    let cap = max.saturating_sub(suffix.chars().count());
    let mut out: String = s.chars().take(cap).collect();
    out.push_str(suffix);
    out
}

/// Compact duration (TS `formatDuration` in `pi-utils/format.ts`).
#[must_use]
pub fn format_duration(ms: u64) -> String {
    const SEC: u64 = 1_000;
    const MIN: u64 = 60 * SEC;
    const HOUR: u64 = 60 * MIN;
    const DAY: u64 = 24 * HOUR;
    if ms < SEC {
        return format!("{ms}ms");
    }
    if ms < MIN {
        return format!("{:.1}s", ms as f64 / SEC as f64);
    }
    if ms < HOUR {
        let mins = ms / MIN;
        let secs = (ms % MIN) / SEC;
        return if secs > 0 {
            format!("{mins}m{secs}s")
        } else {
            format!("{mins}m")
        };
    }
    if ms < DAY {
        let hours = ms / HOUR;
        let mins = (ms % HOUR) / MIN;
        return if mins > 0 {
            format!("{hours}h{mins}m")
        } else {
            format!("{hours}h")
        };
    }
    let days = ms / DAY;
    let hours = (ms % DAY) / HOUR;
    if hours > 0 {
        format!("{days}d{hours}h")
    } else {
        format!("{days}d")
    }
}

/// Compact number with K/M/B suffix (TS `formatNumber`).
#[must_use]
pub fn format_number(n: u64) -> String {
    const UNITS: [(u64, &str); 3] = [(1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K")];
    for (div, suffix) in UNITS {
        if n >= div {
            let v = n as f64 / div as f64;
            let rounded = (v * 10.0).round() / 10.0;
            return if (rounded.fract() - rounded.trunc()).abs() < f64::EPSILON {
                format!("{}{}", rounded as u64, suffix)
            } else {
                format!("{rounded:.1}{suffix}")
            };
        }
    }
    n.to_string()
}

/// Display cap for a normalized one-line label (roster line, header).
pub const fn label_max() -> usize {
    LABEL_MAX
}

/// One-line, length-capped label safe for a single roster line (TS `oneLineLabel`):
/// collapses every run of whitespace, control, and format characters (the
/// zero-width separators `\s` misses) to one space, then caps by code point.
#[must_use]
pub fn one_line_label(text: &str, max: usize) -> String {
    let mut one_line = String::with_capacity(text.len());
    let mut pending_space = false;
    for c in text.chars() {
        let is_junk = c.is_whitespace() || c.is_control() || is_format_char(c);
        if is_junk {
            pending_space = true;
        } else {
            if pending_space && !one_line.is_empty() {
                one_line.push(' ');
            }
            pending_space = false;
            one_line.push(c);
        }
    }
    let cap = max.max(1);
    let mut chars = one_line.chars();
    if one_line.chars().count() > cap {
        let mut out: String = chars.by_ref().take(cap - 1).collect();
        out.push('…');
        out
    } else {
        one_line
    }
}

/// Unicode general-category Cf (format) — matches TS `\p{Cf}`.
fn is_format_char(c: char) -> bool {
    matches!(
        c,
        '\u{00AD}'
            | '\u{0600}'..='\u{0605}'
            | '\u{061C}'
            | '\u{06DD}'
            | '\u{070F}'
            | '\u{0890}'..='\u{0891}'
            | '\u{08E2}'
            | '\u{180E}'
            | '\u{200B}'..='\u{200F}'
            | '\u{202A}'..='\u{202E}'
            | '\u{2060}'..='\u{2064}'
            | '\u{2066}'..='\u{206F}'
            | '\u{FEFF}'
            | '\u{FFF9}'..='\u{FFFB}'
            | '\u{110BD}'
            | '\u{110CD}'
            | '\u{13430}'..='\u{1343F}'
            | '\u{1BCA0}'..='\u{1BCA3}'
            | '\u{1D173}'..='\u{1D17A}'
            | '\u{E0001}'
            | '\u{E0020}'..='\u{E007F}'
    )
}

/// First line of a streamed task brief (TS `taskFirstLine`): trimmed, with
/// control/format chars collapsed so a streamed blob can't break the row.
#[must_use]
pub fn task_first_line(task: &Value) -> String {
    let Some(s) = task.as_str() else {
        return String::new();
    };
    one_line_label(s, usize::MAX)
}

/// Header label for a task call while nothing has spawned: the flat form's
/// `agent` type; batch calls return `None` (TS `formatAgentHeaderLabel`).
#[must_use]
pub fn format_agent_header_label(args: Option<&Value>) -> Option<String> {
    let flat = args?
        .get("agent")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())?;
    Some(flat.to_string())
}

/// Dim `⟨agent⟩` badge for a non-default agent type; empty for the worker.
#[must_use]
pub fn agent_type_badge(agent: &str) -> String {
    let trimmed = agent.trim();
    if trimmed.is_empty() || trimmed == "task" {
        String::new()
    } else {
        format!(" ⟨{trimmed}⟩")
    }
}

/// One task batch item (TS `TaskItem`), parsed defensively.
#[derive(Debug, Clone, Default)]
pub struct TaskItemView {
    /// Stable agent name.
    pub name: Option<String>,
    /// Agent type.
    pub agent: Option<String>,
    /// The work.
    pub task: Option<Value>,
    /// Isolated worktree flag.
    pub isolated: bool,
}

/// Task tool params: flat form, batch form, shared context (TS `TaskParams`).
#[derive(Debug, Clone, Default)]
pub struct TaskParamsView {
    /// Flat-form stable name.
    pub name: Option<String>,
    /// Flat-form agent type.
    pub agent: Option<String>,
    /// Flat-form work.
    pub task: Option<Value>,
    /// Batch-form items.
    pub tasks: Vec<TaskItemView>,
    /// Batch-form shared background.
    pub context: Option<String>,
    /// Isolated worktree flag (flat form).
    pub isolated: bool,
}

impl TaskParamsView {
    /// Parse params from a wire value (already-repaired strings preferred).
    #[must_use]
    pub fn from_value(v: Option<&Value>) -> Self {
        let Some(v) = v.filter(|a| a.is_object()) else {
            return Self::default();
        };
        let mut view = Self {
            name: get_string(v, "name"),
            agent: get_string(v, "agent"),
            task: v.get("task").cloned(),
            tasks: Vec::new(),
            context: get_string(v, "context"),
            isolated: get_bool(v, "isolated"),
        };
        if let Some(items) = v.get("tasks").and_then(Value::as_array) {
            view.tasks = items
                .iter()
                .filter_map(|i| {
                    i.as_object().map(|_| TaskItemView {
                        name: get_string(i, "name"),
                        agent: get_string(i, "agent"),
                        task: i.get("task").cloned(),
                        isolated: get_bool(i, "isolated"),
                    })
                })
                .collect();
        }
        view
    }
}

fn format_more_items(n: usize, unit: &str) -> String {
    format!("… {n} more {unit}{}", if n == 1 { "" } else { "s" })
}

/// Render the call preview lines: the flat agent row plus batch item rows
/// (TS `renderTaskCallLines` + `renderTaskItemLines`). Spans carry their own
/// styles; the caller assembles [`Line`]s.
fn render_task_call_lines(args: Option<&Value>, theme: &Theme) -> Vec<Vec<Span<'static>>> {
    let bullet = Span::styled("•", Style::default().fg(theme.color("text_muted")));
    let mut lines: Vec<Vec<Span<'static>>> = Vec::new();
    let view = TaskParamsView::from_value(args);
    let name = view.name.as_deref().unwrap_or("").trim();
    let brief = view
        .task
        .as_ref()
        .map(|v| task_first_line(v))
        .unwrap_or_default();
    if !name.is_empty() || !brief.is_empty() {
        let mut spans = vec![bullet.clone()];
        spans.push(Span::styled(
            if name.is_empty() {
                "agent".to_string()
            } else {
                name.to_string()
            },
            Style::default()
                .fg(theme.color("accent"))
                .add_modifier(Modifier::BOLD),
        ));
        if !brief.is_empty() {
            spans.push(Span::raw(": "));
            spans.push(Span::styled(
                truncate_to_width(&brief, 64, "…"),
                Style::default().fg(theme.color("text_muted")),
            ));
        }
        let badge = agent_type_badge(view.agent.as_deref().unwrap_or(""));
        if !badge.is_empty() {
            spans.push(Span::styled(
                badge,
                Style::default().fg(theme.color("text_muted")),
            ));
        }
        lines.push(spans);
    }
    if !view.tasks.is_empty() {
        let cap = view.tasks.len().min(COLLAPSED_AGENT_LIMIT);
        for (i, item) in view.tasks.iter().take(cap).enumerate() {
            let mut spans = vec![bullet.clone()];
            let id_label = item
                .name
                .as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(ToOwned::to_owned)
                .unwrap_or_else(|| format!("#{}", i + 1));
            spans.push(Span::styled(
                id_label,
                Style::default()
                    .fg(theme.color("accent"))
                    .add_modifier(Modifier::BOLD),
            ));
            let brief = item
                .task
                .as_ref()
                .map(|v| task_first_line(v))
                .unwrap_or_default();
            if !brief.is_empty() {
                spans.push(Span::raw(": "));
                spans.push(Span::styled(
                    truncate_to_width(&brief, 64, "…"),
                    Style::default().fg(theme.color("text_muted")),
                ));
            }
            let badge = agent_type_badge(item.agent.as_deref().unwrap_or(""));
            if !badge.is_empty() {
                spans.push(Span::styled(
                    badge,
                    Style::default().fg(theme.color("text_muted")),
                ));
            }
            if item.isolated {
                spans.push(Span::styled(
                    " [isolated]",
                    Style::default().fg(theme.color("text_muted")),
                ));
            }
            lines.push(spans);
        }
        if cap < view.tasks.len() {
            lines.push(vec![
                bullet,
                Span::styled(
                    format_more_items(view.tasks.len() - cap, "agent"),
                    Style::default().fg(theme.color("text_muted")),
                ),
            ]);
        }
    }
    lines
}

/// Counters displayed alongside an agent name (TS `AgentStats`).
#[derive(Debug, Clone, Copy, Default)]
pub struct AgentStats {
    /// Tool-call count.
    pub tool_count: u64,
    /// Assistant request count.
    pub requests: u64,
    /// Latest per-turn context size.
    pub context_tokens: u64,
    /// Model context window.
    pub context_window: u64,
    /// Cumulative cost in USD.
    pub cost: f64,
}

/// Format the shared stat run: `·N tools ·N req ·ctx ·$cost` (TS `formatAgentStatRun`).
#[must_use]
pub fn format_agent_stat_run(stats: &AgentStats, theme: &Theme) -> Vec<Span<'static>> {
    let mut spans = Vec::new();
    let dim = Style::default().fg(theme.color("text_muted"));
    if stats.tool_count > 0 {
        spans.push(Span::styled(
            format!(" ·{} tools", format_number(stats.tool_count)),
            dim,
        ));
    }
    if stats.requests > 0 {
        spans.push(Span::styled(
            format!(" ·{} req", format_number(stats.requests)),
            dim,
        ));
    }
    if stats.context_tokens > 0 {
        let ctx = if stats.context_window > 0 {
            format!(
                " ·{}/{}",
                format_number(stats.context_tokens),
                format_number(stats.context_window)
            )
        } else {
            format!(" ·{}", format_number(stats.context_tokens))
        };
        spans.push(Span::styled(ctx, dim));
    }
    if stats.cost > 0.0 {
        spans.push(Span::styled(format!(" ·${:.2}", stats.cost), dim));
    }
    spans
}

/// One bounded agent progress row (TS `renderAgentTreeRow`, task presentation).
#[derive(Debug, Clone)]
pub struct AgentTreeRow {
    /// Lifecycle phase.
    pub status: AgentStatus,
    /// Tree prefix (e.g. `"└─"`); empty at the top level.
    pub prefix: String,
    /// Agent registry id.
    pub id: String,
    /// Total row budget in code points.
    pub width: usize,
    /// Resolved model display string.
    pub model: Option<String>,
    /// A live advisor is attached.
    pub advisor: bool,
    /// Spinner frame index for running rows.
    pub spinner_frame: usize,
    /// Row left the live region (clock stopped, name dimmed).
    pub frozen: bool,
    /// Preformatted role badge (from [`agent_type_badge`]).
    pub role_badge: String,
    /// Preformatted status badge (retry / rate-limit); `None` derives from status.
    pub status_badge: Option<String>,
    /// Activity gist, shown when it fits.
    pub description: Option<String>,
    /// Fallback preview (assignment brief) while running without a description.
    pub preview: Option<String>,
    /// Stat counters appended after the title.
    pub stats: Option<AgentStats>,
    /// Live completion self-estimate (0-100).
    pub completion_percent: Option<u8>,
    /// Elapsed run time; appended dim when `> 0`.
    pub duration_ms: Option<u64>,
}

/// Output of [`render_agent_tree_row`].
#[derive(Debug, Clone)]
pub struct AgentRowRender {
    /// The styled spans of the row.
    pub spans: Vec<Span<'static>>,
    /// Whether the description fit on the row.
    pub description_shown: bool,
}

fn status_icon_spans(
    status: AgentStatus,
    spinner_frame: usize,
    theme: &Theme,
) -> Vec<Span<'static>> {
    let style = |color: &str| Style::default().fg(theme.color(color));
    match status {
        AgentStatus::Completed => vec![Span::styled(
            theme.glyph("success").to_string(),
            style("success"),
        )],
        AgentStatus::Failed | AgentStatus::Aborted => {
            vec![Span::styled(
                theme.glyph("failure").to_string(),
                style("error"),
            )]
        }
        AgentStatus::Running => {
            let frame = SPINNER_FRAMES[spinner_frame % SPINNER_FRAMES.len()];
            vec![Span::styled(frame.to_string(), style("accent"))]
        }
        AgentStatus::Pending => vec![Span::styled("◌".to_string(), style("text_muted"))],
    }
}

/// Render the agent row, reserving the id and required status badges first
/// (TS `renderAgentTreeRow` for `presentation: "task"`).
#[must_use]
pub fn render_agent_tree_row(row: &AgentTreeRow, theme: &Theme) -> AgentRowRender {
    let live = row.status.is_live();
    let failed = row.status.is_failed();
    let name_color = if live && row.frozen {
        "text_muted"
    } else if row.status == AgentStatus::Completed {
        "text"
    } else {
        "accent"
    };

    let mut spans: Vec<Span<'static>> = Vec::new();
    if !row.prefix.is_empty() {
        spans.push(Span::styled(
            row.prefix.clone(),
            Style::default().fg(theme.color("text_muted")),
        ));
        spans.push(Span::raw(" "));
    }
    spans.extend(status_icon_spans(row.status, row.spinner_frame, theme));
    spans.push(Span::raw(" "));

    let lead_width = spans
        .iter()
        .map(|s| visible_width(s.content.as_ref()))
        .sum::<usize>();
    let badges_width = visible_width(&row.role_badge)
        + row
            .status_badge
            .as_deref()
            .map_or(0, |b| visible_width(b) + 1);
    let id = truncate_to_width(
        &row.id,
        row.width.saturating_sub(lead_width + badges_width + 4),
        "…",
    );
    let model = row
        .model
        .as_deref()
        .map(|m| truncate_to_width(m, 24, "…"))
        .unwrap_or_default();
    let model_lead_width = if model.is_empty() {
        0
    } else {
        visible_width(&model) + 1
    };

    let description = row
        .description
        .as_deref()
        .map(str::trim)
        .filter(|d| !d.is_empty());
    let description_shown = description.is_some()
        && lead_width
            + model_lead_width
            + visible_width(&id)
            + visible_width(description.unwrap_or(""))
            + badges_width
            + 2
            <= row.width;
    if let Some(preview) = row.preview.as_deref().filter(|_| description.is_none()) {
        spans.push(Span::styled(
            truncate_to_width(preview, 40, "…"),
            Style::default().fg(theme.color("text_muted")),
        ));
        spans.push(Span::raw(" "));
    }
    if !model.is_empty() {
        spans.push(Span::styled(
            model,
            Style::default().fg(theme.color("text_muted")),
        ));
        spans.push(Span::raw(" "));
    }
    let name_style = Style::default()
        .fg(theme.color(name_color))
        .add_modifier(Modifier::BOLD);
    if description_shown {
        spans.push(Span::styled(format!("{id}:"), name_style));
        spans.push(Span::raw(" "));
        spans.push(Span::styled(
            description.unwrap_or_default().to_string(),
            name_style,
        ));
    } else {
        spans.push(Span::styled(id, name_style));
    }
    if !row.role_badge.is_empty() {
        spans.push(Span::styled(
            row.role_badge.clone(),
            Style::default().fg(theme.color("text_muted")),
        ));
    }
    if let Some(badge) = row.status_badge.as_deref() {
        spans.push(Span::raw(" "));
        let color = if badge.contains("rate-limited") {
            "error"
        } else {
            "warning"
        };
        spans.push(Span::styled(
            badge.to_string(),
            Style::default().fg(theme.color(color)),
        ));
    } else if failed {
        let color = if row.status == AgentStatus::Aborted {
            "warning"
        } else {
            "error"
        };
        spans.push(Span::styled(
            format!(" {}", row.status.as_str()),
            Style::default().fg(theme.color(color)),
        ));
    }
    if let Some(stats) = &row.stats {
        spans.extend(format_agent_stat_run(stats, theme));
    }
    if live {
        if let Some(pct) = row.completion_percent {
            spans.push(Span::raw(" ·"));
            spans.push(Span::styled(
                format!("{pct}%"),
                Style::default().fg(theme.color("accent")),
            ));
        }
    }
    if let Some(ms) = row.duration_ms.filter(|ms| *ms > 0) {
        spans.push(Span::raw(" ·"));
        spans.push(Span::styled(
            format_duration(ms),
            Style::default().fg(theme.color("text_muted")),
        ));
    }
    AgentRowRender {
        spans,
        description_shown,
    }
}

/// A settled subagent result (TS `SingleResult`), parsed defensively.
#[derive(Debug, Clone, Default)]
pub struct SingleResult {
    /// Dispatch index.
    pub index: u64,
    /// Registry id.
    pub id: String,
    /// Agent type.
    pub agent: String,
    /// The work.
    pub task: String,
    /// Expanded assignment text.
    pub assignment: Option<String>,
    /// Latest activity gist.
    pub description: Option<String>,
    /// Process exit code.
    pub exit_code: u64,
    /// Output text.
    pub output: String,
    /// True when the run was aborted.
    pub aborted: bool,
    /// Merge/failure error text.
    pub error: Option<String>,
    /// Output was truncated to the byte/line cap.
    pub truncated: bool,
    /// Elapsed run time.
    pub duration_ms: u64,
    /// Assistant requests across the run.
    pub requests: u64,
    /// Latest per-turn context size.
    pub context_tokens: Option<u64>,
    /// Model context window, when known.
    pub context_window: Option<u64>,
    /// Cumulative cost in USD.
    pub cost: f64,
    /// Resolved model display string.
    pub resolved_model: Option<String>,
    /// Resolved model identity without the thinking suffix.
    pub resolved_model_identity: Option<String>,
    /// Per-tool data extracted from nested calls.
    pub extracted_tool_data: Option<Value>,
}

impl SingleResult {
    /// Parse one result from a wire value; `None` for non-objects.
    #[must_use]
    pub fn from_value(v: &Value) -> Option<Self> {
        v.as_object()?;
        let usage_cost = v
            .get("usage")
            .and_then(|u| u.get("cost"))
            .and_then(|c| c.get("total"))
            .and_then(Value::as_f64)
            .unwrap_or(0.0);
        Some(Self {
            index: get_u64(v, "index").unwrap_or(0),
            id: get_string(v, "id").unwrap_or_default(),
            agent: get_string(v, "agent").unwrap_or_default(),
            task: get_string(v, "task").unwrap_or_default(),
            assignment: get_string(v, "assignment"),
            description: get_string(v, "description"),
            exit_code: get_u64(v, "exitCode")
                .or_else(|| get_u64(v, "exit_code"))
                .unwrap_or(0),
            output: get_string(v, "output").unwrap_or_default(),
            aborted: get_bool(v, "aborted"),
            error: get_string(v, "error"),
            truncated: get_bool(v, "truncated"),
            duration_ms: get_u64(v, "durationMs")
                .or_else(|| get_u64(v, "duration_ms"))
                .unwrap_or(0),
            requests: get_u64(v, "requests").unwrap_or(0),
            context_tokens: get_u64(v, "contextTokens").or_else(|| get_u64(v, "context_tokens")),
            context_window: get_u64(v, "contextWindow").or_else(|| get_u64(v, "context_window")),
            cost: get_f64(v, "cost").unwrap_or(usage_cost),
            resolved_model: get_string(v, "resolvedModel")
                .or_else(|| get_string(v, "resolved_model")),
            resolved_model_identity: get_string(v, "resolvedModelIdentity")
                .or_else(|| get_string(v, "resolved_model_identity")),
            extracted_tool_data: v
                .get("extractedToolData")
                .or_else(|| v.get("extracted_tool_data"))
                .cloned(),
        })
    }

    fn success(&self) -> bool {
        !self.aborted && self.exit_code == 0 && self.error.is_none()
    }
}

/// Settled or in-flight `task` tool details (TS `TaskToolDetails`).
#[derive(Debug, Clone, Default)]
pub struct TaskToolDetails {
    /// Settled per-agent results.
    pub results: Vec<SingleResult>,
    /// Live per-agent progress snapshots.
    pub progress: Vec<AgentProgress>,
    /// Total wall-clock across all agents.
    pub total_duration_ms: u64,
    /// Detached background-job state, when the spawn is async.
    pub async_state: Option<String>,
    /// Detached background-job id.
    pub async_job_id: Option<String>,
}

impl TaskToolDetails {
    /// Parse details from a wire value; missing lists stay empty.
    #[must_use]
    pub fn from_value(v: &Value) -> Self {
        let mut details = Self::default();
        if !v.is_object() {
            return details;
        }
        if let Some(results) = v.get("results").and_then(Value::as_array) {
            details.results = results
                .iter()
                .filter_map(SingleResult::from_value)
                .collect();
        }
        if let Some(progress) = v.get("progress").and_then(Value::as_array) {
            details.progress = progress
                .iter()
                .filter_map(AgentProgress::from_value)
                .collect();
        }
        details.total_duration_ms = get_u64(v, "totalDurationMs")
            .or_else(|| get_u64(v, "total_duration_ms"))
            .unwrap_or(0);
        if let Some(a) = v.get("async").filter(|a| a.is_object()) {
            details.async_state = get_string(a, "state");
            details.async_job_id = get_string(a, "jobId").or_else(|| get_string(a, "job_id"));
        }
        details
    }

    fn is_empty(&self) -> bool {
        self.results.is_empty() && self.progress.is_empty()
    }
}

/// Extract task details from a tool call / update / result frame, tolerating
/// every placement the runtime uses: `details`, `partialResult`, `result`,
/// and bare `progress`/`results` arrays (TS merges call+result).
#[must_use]
pub fn details_of(call: &Value) -> TaskToolDetails {
    for key in ["details", "partialResult", "partial_result", "result"] {
        if let Some(arr) = call.get(key).and_then(Value::as_array) {
            let progress: Vec<AgentProgress> =
                arr.iter().filter_map(AgentProgress::from_value).collect();
            if !progress.is_empty() {
                return TaskToolDetails {
                    progress,
                    ..TaskToolDetails::default()
                };
            }
            let results: Vec<SingleResult> =
                arr.iter().filter_map(SingleResult::from_value).collect();
            if !results.is_empty() {
                return TaskToolDetails {
                    results,
                    ..TaskToolDetails::default()
                };
            }
        }
        if let Some(v) = call.get(key).filter(|v| v.is_object()) {
            let d = TaskToolDetails::from_value(v);
            if !d.is_empty() || v.get("totalDurationMs").is_some() {
                return d;
            }
            // A single progress payload: `{index, agent, task, progress}` or a
            // lifecycle payload `{id, agent, status}`.
            if let Some(p) = AgentProgress::from_value(v) {
                if !p.id.is_empty() || !p.agent.is_empty() {
                    return TaskToolDetails {
                        progress: vec![p],
                        ..TaskToolDetails::default()
                    };
                }
            }
        }
    }
    let mut d = TaskToolDetails::default();
    if let Some(progress) = call.get("progress").and_then(Value::as_array) {
        d.progress = progress
            .iter()
            .filter_map(AgentProgress::from_value)
            .collect();
    }
    if let Some(results) = call.get("results").and_then(Value::as_array) {
        d.results = results
            .iter()
            .filter_map(SingleResult::from_value)
            .collect();
    }
    d
}

/// Options shared by the live-progress and result line builders.
#[derive(Debug, Clone, Copy)]
pub struct RenderCtx {
    /// Available row width in code points.
    pub width: usize,
    /// Spinner frame for running rows.
    pub spinner_frame: usize,
    /// True to show full (expanded) bodies instead of collapsed previews.
    pub expanded: bool,
    /// Rows have left the live region: clocks stop, names gray out.
    pub frozen: bool,
    /// Current time (epoch ms) for elapsed-tool math.
    pub now_ms: u64,
}

impl RenderCtx {
    /// A live-region context at the given width.
    #[must_use]
    pub fn live(width: usize, now_ms: u64) -> Self {
        Self {
            width,
            spinner_frame: 0,
            expanded: false,
            frozen: false,
            now_ms,
        }
    }
}

/// Tree guides for nested levels (TS `nestedMarkers`).
fn nested_markers(is_last: bool, theme: &Theme) -> (String, String) {
    let _ = theme;
    if is_last {
        ("└─".to_string(), "   ".to_string())
    } else {
        ("├─".to_string(), "│  ".to_string())
    }
}

fn dim_line(prefix: &str, text: String, theme: &Theme, out: &mut Vec<Line<'static>>) {
    let mut spans = vec![Span::raw(prefix.to_string())];
    spans.push(Span::styled(
        text,
        Style::default().fg(theme.color("text_muted")),
    ));
    out.push(Line::from(spans));
}

fn hook_line(prefix: &str, text: String, color: &str, theme: &Theme, out: &mut Vec<Line<'static>>) {
    let mut spans = vec![Span::raw(prefix.to_string())];
    spans.push(Span::styled(
        "└ ",
        Style::default().fg(theme.color("text_muted")),
    ));
    spans.push(Span::styled(text, Style::default().fg(theme.color(color))));
    out.push(Line::from(spans));
}

/// First `max` lines of a body plus a `… N more lines` tail (TS `renderTaskSection`).
fn section_lines(
    body: &str,
    prefix: &str,
    max: usize,
    theme: &Theme,
    out: &mut Vec<Line<'static>>,
) {
    let lines: Vec<&str> = body.lines().collect();
    for line in lines.iter().take(max) {
        dim_line(
            &format!("{prefix}  "),
            one_line_label(line, usize::MAX),
            theme,
            out,
        );
    }
    if lines.len() > max {
        dim_line(
            &format!("{prefix}  "),
            format_more_items(lines.len() - max, "line"),
            theme,
            out,
        );
    }
}

/// Order live progress rows: finished agents by duration, then unfinished in
/// dispatch order (TS `orderProgressForDisplay`).
fn order_progress(progress: &[AgentProgress]) -> Vec<&AgentProgress> {
    let mut finished: Vec<&AgentProgress> = Vec::new();
    let mut unfinished: Vec<&AgentProgress> = Vec::new();
    for p in progress {
        if p.status.is_live() {
            unfinished.push(p);
        } else {
            finished.push(p);
        }
    }
    finished.sort_by_key(|p| (p.duration_ms, p.index));
    finished.into_iter().chain(unfinished).collect()
}

/// Order settled results by duration ascending, dispatch index tie-break
/// (TS `orderResultsForDisplay`).
fn order_results(results: &[SingleResult]) -> Vec<&SingleResult> {
    let mut ordered: Vec<&SingleResult> = results.iter().collect();
    ordered.sort_by_key(|r| (r.duration_ms, r.index));
    ordered
}

/// Collapsed result cap: failing results first, then the leading run, up to
/// [`COLLAPSED_AGENT_LIMIT`] (TS `selectCollapsedResults`).
fn select_collapsed_results<'a>(ordered: &'a [&'a SingleResult]) -> Vec<&'a SingleResult> {
    if ordered.len() <= COLLAPSED_AGENT_LIMIT {
        return ordered.to_vec();
    }
    let mut picked: Vec<usize> = Vec::new();
    for (i, r) in ordered.iter().enumerate() {
        if picked.len() >= COLLAPSED_AGENT_LIMIT {
            break;
        }
        if r.aborted || r.exit_code != 0 || r.error.is_some() {
            picked.push(i);
        }
    }
    for i in 0..ordered.len() {
        if picked.len() >= COLLAPSED_AGENT_LIMIT {
            break;
        }
        if !picked.contains(&i) {
            picked.push(i);
        }
    }
    ordered
        .iter()
        .enumerate()
        .filter(|(i, _)| picked.contains(i))
        .map(|(_, r)| *r)
        .collect()
}

/// Summary line for progress rows folded away by the collapsed cap
/// (TS `formatHiddenProgressLine`): `… N more agents (18 pending · 3 done)`.
fn hidden_progress_line(hidden: &[&AgentProgress]) -> String {
    let mut counts: HashMap<&'static str, usize> = HashMap::new();
    for p in hidden {
        *counts.entry(p.status.as_str()).or_default() += 1;
    }
    let mut parts: Vec<String> = Vec::new();
    for (status, label) in [
        ("completed", "done"),
        ("failed", "failed"),
        ("aborted", "aborted"),
        ("running", "running"),
        ("pending", "pending"),
    ] {
        if let Some(n) = counts.get(status) {
            parts.push(format!("{n} {label}"));
        }
    }
    if parts.is_empty() {
        format_more_items(hidden.len(), "agent")
    } else {
        format!(
            "{} ({})",
            format_more_items(hidden.len(), "agent"),
            parts.join(" · ")
        )
    }
}

/// Render streaming progress for one agent (TS `renderAgentProgress`).
#[allow(clippy::too_many_lines)]
fn render_agent_progress_lines(
    progress: &AgentProgress,
    prefix: &str,
    continue_prefix: &str,
    ctx: &RenderCtx,
    theme: &Theme,
    depth: usize,
    out: &mut Vec<Line<'static>>,
) {
    let full_description = progress
        .description
        .as_deref()
        .map(str::trim)
        .filter(|d| !d.is_empty())
        .map(ToOwned::to_owned);
    let status_badge = if progress.retry_state.is_some() && progress.status == AgentStatus::Running
    {
        Some("retrying".to_string())
    } else if progress.retry_failure.is_some()
        && matches!(progress.status, AgentStatus::Failed | AgentStatus::Aborted)
    {
        Some("rate-limited".to_string())
    } else {
        None
    };
    let preview = (progress.status == AgentStatus::Running && full_description.is_none())
        .then(|| one_line_label(progress.assignment.as_deref().unwrap_or(&progress.task), 40));
    let stats = matches!(
        progress.status,
        AgentStatus::Running | AgentStatus::Completed
    )
    .then_some(AgentStats {
        tool_count: progress.tool_count,
        requests: progress.requests,
        context_tokens: progress.context_tokens.unwrap_or(0),
        context_window: progress.context_window.unwrap_or(0),
        cost: progress.cost,
    });
    let row = render_agent_tree_row(
        &AgentTreeRow {
            status: progress.status,
            prefix: prefix.to_string(),
            id: progress.id.clone(),
            width: ctx.width,
            model: progress
                .resolved_model_identity
                .clone()
                .or_else(|| progress.resolved_model.clone()),
            advisor: progress.advisor,
            spinner_frame: ctx.spinner_frame,
            frozen: ctx.frozen,
            role_badge: agent_type_badge(&progress.agent),
            status_badge,
            description: full_description.clone(),
            preview,
            stats,
            completion_percent: progress.completion_percent,
            duration_ms: Some(progress.duration_ms),
        },
        theme,
    );
    out.push(Line::from(row.spans));
    if let Some(desc) = full_description.filter(|_| !row.description_shown) {
        section_lines(&desc, continue_prefix, 3, theme, out);
    }

    // Assignment brief: 3 collapsed / 12 expanded lines.
    let brief = progress.assignment.as_deref().unwrap_or(&progress.task);
    if !brief.trim().is_empty() {
        section_lines(
            brief,
            continue_prefix,
            if ctx.expanded { 12 } else { 3 },
            theme,
            out,
        );
    }

    // Current tool (while running) or the most recent completed tool.
    if progress.status == AgentStatus::Running {
        if let Some(tool) = progress.current_tool.as_deref() {
            let mut text = one_line_label(tool, usize::MAX);
            let detail = progress
                .current_tool_intent
                .clone()
                .or_else(|| progress.current_tool_args.clone());
            if let Some(d) = detail.filter(|d| !d.is_empty()) {
                text.push_str(&format!(
                    ": {}",
                    truncate_to_width(&one_line_label(&d, usize::MAX), 40, "…")
                ));
            }
            if let Some(start) = progress.current_tool_start_ms {
                let elapsed = ctx.now_ms.saturating_sub(start);
                if elapsed > TOOL_ELAPSED_WARN_MS {
                    text.push_str(&format!(" ·{}", format_duration(elapsed)));
                }
            }
            hook_line(continue_prefix, text, "text_muted", theme, out);
        } else if let Some(recent) = progress.recent_tools.first() {
            let mut text = one_line_label(&recent.tool, usize::MAX);
            let detail = recent
                .intent
                .clone()
                .or_else(|| (!recent.args.is_empty()).then(|| recent.args.clone()));
            if let Some(d) = detail.filter(|d| !d.is_empty()) {
                text.push_str(&format!(
                    ": {}",
                    truncate_to_width(&one_line_label(&d, usize::MAX), 40, "…")
                ));
            }
            hook_line(continue_prefix, text, "text_muted", theme, out);
        }
    }

    // Retry detail: why the subagent is paused and when it resumes.
    if let Some(rs) = &progress.retry_state {
        if progress.status == AgentStatus::Running {
            let remaining = rs
                .started_at_ms
                .saturating_add(rs.delay_ms)
                .saturating_sub(ctx.now_ms);
            let wait = if remaining > 0 {
                format!("in {}", format_duration(remaining))
            } else {
                "now".to_string()
            };
            hook_line(
                continue_prefix,
                format!(
                    "retrying {}/{} {wait}: {}",
                    rs.attempt,
                    rs.max_attempts,
                    truncate_to_width(&one_line_label(&rs.error_message, usize::MAX), 60, "…")
                ),
                "warning",
                theme,
                out,
            );
        }
    } else if let Some(rf) = &progress.retry_failure {
        if progress.status != AgentStatus::Running {
            hook_line(
                continue_prefix,
                format!(
                    "auto-retry gave up after {} attempt{}: {}",
                    rf.attempt,
                    if rf.attempt == 1 { "" } else { "s" },
                    truncate_to_width(&one_line_label(&rf.error_message, usize::MAX), 80, "…")
                ),
                "error",
                theme,
                out,
            );
        }
    }

    // Nested `task` tree: completed sub-calls plus the in-flight snapshot.
    let mut snapshots: Vec<Value> = progress
        .extracted_tool_data
        .as_ref()
        .and_then(|d| d.get("task"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if let Some(inflight) = &progress.inflight_task_details {
        snapshots.push(inflight.clone());
    }
    if !snapshots.is_empty() {
        let details: Vec<TaskToolDetails> =
            snapshots.iter().map(TaskToolDetails::from_value).collect();
        render_nested_task_tree(&details, ctx, theme, depth, out);
    }
}

/// Render a list of `task` detail snapshots as an interleaved tree, completed
/// results or live progress (TS `renderNestedTaskTree`). Depth-capped at
/// [`MAX_NESTED_TASK_RENDER_DEPTH`]; deeper levels collapse to one dim line.
fn render_nested_task_tree(
    details_list: &[TaskToolDetails],
    ctx: &RenderCtx,
    theme: &Theme,
    depth: usize,
    out: &mut Vec<Line<'static>>,
) {
    for details in details_list {
        if depth >= MAX_NESTED_TASK_RENDER_DEPTH {
            dim_line(
                "",
                "… nested task depth limit reached".to_string(),
                theme,
                out,
            );
            continue;
        }
        if !details.results.is_empty() {
            render_nested_results(&details.results, ctx, theme, depth, out);
        } else {
            render_nested_progress(&details.progress, ctx, theme, depth, out);
        }
    }
}

fn render_nested_results(
    results: &[SingleResult],
    ctx: &RenderCtx,
    theme: &Theme,
    depth: usize,
    out: &mut Vec<Line<'static>>,
) {
    let ordered = order_results(results);
    let visible: Vec<&SingleResult> = if ctx.expanded {
        ordered.clone()
    } else {
        select_collapsed_results(&ordered)
    };
    let hidden = ordered.len().saturating_sub(visible.len());
    let last_index = visible.len().saturating_sub(1);
    for (index, result) in visible.iter().enumerate() {
        let (prefix, continue_prefix) = nested_markers(hidden == 0 && index == last_index, theme);
        render_agent_result_lines(result, &prefix, &continue_prefix, ctx, theme, depth, out);
    }
    if hidden > 0 {
        let (prefix, _) = nested_markers(true, theme);
        dim_line(
            &format!("{prefix} "),
            format_more_items(hidden, "agent"),
            theme,
            out,
        );
    }
}

fn render_nested_progress(
    progress: &[AgentProgress],
    ctx: &RenderCtx,
    theme: &Theme,
    depth: usize,
    out: &mut Vec<Line<'static>>,
) {
    let ordered = order_progress(progress);
    let hidden_count = if ctx.expanded {
        0
    } else {
        ordered.len().saturating_sub(COLLAPSED_AGENT_LIMIT)
    };
    let visible: Vec<&AgentProgress> = ordered[hidden_count..].to_vec();
    let last_index = visible.len().saturating_sub(1);
    for (index, prog) in visible.iter().enumerate() {
        let (prefix, continue_prefix) =
            nested_markers(hidden_count == 0 && index == last_index, theme);
        render_agent_progress_lines(prog, &prefix, &continue_prefix, ctx, theme, depth + 1, out);
    }
    if hidden_count > 0 {
        let (prefix, _) = nested_markers(true, theme);
        dim_line(
            &format!("{prefix} "),
            hidden_progress_line(&ordered[..hidden_count]),
            theme,
            out,
        );
    }
}

/// Render the settled result row plus body for one agent (TS `renderAgentResult`).
#[allow(clippy::too_many_lines)]
fn render_agent_result_lines(
    result: &SingleResult,
    prefix: &str,
    continue_prefix: &str,
    ctx: &RenderCtx,
    theme: &Theme,
    depth: usize,
    out: &mut Vec<Line<'static>>,
) {
    let aborted = result.aborted;
    let merge_failed = !aborted && result.exit_code == 0 && result.error.is_some();
    let success = result.success();
    let (icon, icon_color, status_text) = if aborted {
        (theme.glyph("failure"), "warning", "aborted")
    } else if success {
        (theme.glyph("success"), "success", "done")
    } else if merge_failed {
        (theme.glyph("failure"), "warning", "merge failed")
    } else {
        (theme.glyph("failure"), "error", "failed")
    };

    let full_description = result
        .description
        .as_deref()
        .map(str::trim)
        .filter(|d| !d.is_empty())
        .map(ToOwned::to_owned);
    let mut spans: Vec<Span<'static>> = Vec::new();
    if !prefix.is_empty() {
        spans.push(Span::styled(
            prefix.to_string(),
            Style::default().fg(theme.color("text_muted")),
        ));
        spans.push(Span::raw(" "));
    }
    spans.push(Span::styled(
        icon.to_string(),
        Style::default().fg(theme.color(icon_color)),
    ));
    spans.push(Span::raw(" "));
    let lead_width: usize = spans
        .iter()
        .map(|s| visible_width(s.content.as_ref()))
        .sum();
    let status_badge = format!(" {status_text}");
    let role_badge = agent_type_badge(&result.agent);
    let id_budget = ctx
        .width
        .saturating_sub(lead_width + visible_width(&status_badge) + visible_width(&role_badge) + 4);
    let id = truncate_to_width(&result.id, id_budget, "…");
    let model = result
        .resolved_model_identity
        .clone()
        .or_else(|| result.resolved_model.clone())
        .map(|m| truncate_to_width(&m, 24, "…"))
        .unwrap_or_default();
    if !model.is_empty() {
        spans.push(Span::styled(
            model,
            Style::default().fg(theme.color("text_muted")),
        ));
        spans.push(Span::raw(" "));
    }
    let name_color = if success { "text" } else { "accent" };
    let desc = full_description.clone();
    let desc_fits = desc.as_deref().is_some_and(|d| {
        lead_width + visible_width(&id) + visible_width(d) + visible_width(&status_badge) + 2
            <= ctx.width
    });
    let name_style = Style::default()
        .fg(theme.color(name_color))
        .add_modifier(Modifier::BOLD);
    if desc_fits {
        spans.push(Span::styled(format!("{id}:"), name_style));
        spans.push(Span::raw(" "));
        spans.push(Span::styled(desc.unwrap_or_default(), name_style));
    } else {
        spans.push(Span::styled(id, name_style));
    }
    if !role_badge.is_empty() {
        spans.push(Span::styled(
            role_badge,
            Style::default().fg(theme.color("text_muted")),
        ));
    }
    spans.push(Span::styled(
        status_badge,
        Style::default().fg(theme.color(icon_color)),
    ));
    spans.extend(format_agent_stat_run(
        &AgentStats {
            tool_count: 0,
            requests: result.requests,
            context_tokens: result.context_tokens.unwrap_or(0),
            context_window: result.context_window.unwrap_or(0),
            cost: result.cost,
        },
        theme,
    ));
    if result.duration_ms > 0 {
        spans.push(Span::raw(" ·"));
        spans.push(Span::styled(
            format_duration(result.duration_ms),
            Style::default().fg(theme.color("text_muted")),
        ));
    }
    if result.truncated {
        spans.push(Span::raw(" "));
        spans.push(Span::styled(
            "[truncated]",
            Style::default().fg(theme.color("warning")),
        ));
    }
    out.push(Line::from(spans));
    if let Some(desc) = full_description.filter(|_| !desc_fits) {
        section_lines(&desc, continue_prefix, 3, theme, out);
    }

    // Assignment brief.
    let brief = result.assignment.as_deref().unwrap_or(&result.task);
    if !brief.trim().is_empty() {
        section_lines(
            brief,
            continue_prefix,
            if ctx.expanded { 12 } else { 3 },
            theme,
            out,
        );
    }

    if aborted {
        if let Some(reason) = result.error.as_deref() {
            hook_line(
                continue_prefix,
                truncate_to_width(&one_line_label(reason, usize::MAX), 80, "…"),
                "text_muted",
                theme,
                out,
            );
        }
    } else if !success && result.output.trim().is_empty() {
        // Failure with no output: the error text is the whole story — render
        // it as an error-detail block (TS `formatErrorDetail` semantics).
        if let Some(err) = result.error.as_deref().filter(|e| !e.trim().is_empty()) {
            out.extend(error_detail_lines(err, usize::MAX, theme));
        }
    }

    // Yield sections assembled from incremental yields, else an output preview.
    if !render_yield_sections(result, continue_prefix, ctx, theme, out) {
        let max_lines = if ctx.expanded { 12 } else { 3 };
        section_lines(&result.output, continue_prefix, max_lines, theme, out);
    }

    // Nested subagent runs settle into `extractedToolData.task`.
    let nested: Vec<Value> = result
        .extracted_tool_data
        .as_ref()
        .and_then(|d| d.get("task"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if !nested.is_empty() {
        let details: Vec<TaskToolDetails> =
            nested.iter().map(TaskToolDetails::from_value).collect();
        render_nested_task_tree(&details, ctx, theme, depth + 1, out);
    }
}

/// Render typed yield sections for a settled result; true when anything was
/// rendered (TS `renderTypedYieldSections` + `assembleYieldResult`).
fn render_yield_sections(
    result: &SingleResult,
    continue_prefix: &str,
    ctx: &RenderCtx,
    theme: &Theme,
    out: &mut Vec<Line<'static>>,
) -> bool {
    let Some(data) = result
        .extracted_tool_data
        .as_ref()
        .and_then(|d| d.get("yield"))
    else {
        return false;
    };
    let Some(array) = data.as_array() else {
        return false;
    };
    let items: Vec<yield_assembly::YieldItem> = array
        .iter()
        .filter_map(yield_assembly::YieldItem::from_value)
        .collect();
    let Some(assembled) = yield_assembly::assemble_yield_result(&items, None, None) else {
        return false;
    };
    dim_line(continue_prefix, "yield:".to_string(), theme, out);
    let body = serde_json::to_string_pretty(&assembled.data).unwrap_or_default();
    let max_lines = if ctx.expanded { 24 } else { 6 };
    for line in body.lines().take(max_lines) {
        dim_line(
            &format!("{continue_prefix}  "),
            line.to_string(),
            theme,
            out,
        );
    }
    let total = body.lines().count();
    if total > max_lines {
        dim_line(
            &format!("{continue_prefix}  "),
            format_more_items(total - max_lines, "line"),
            theme,
            out,
        );
    }
    if assembled.raw_text {
        dim_line(
            continue_prefix,
            "(from last assistant turn)".to_string(),
            theme,
            out,
        );
    }
    if assembled.schema_overridden {
        dim_line(
            continue_prefix,
            "(schema override: invalid data accepted)".to_string(),
            theme,
            out,
        );
    }
    let _ = ctx;
    true
}

/// Assemble the full task card body (TS `renderCall` / `renderResult`).
#[must_use]
pub fn card_lines(call: &Value, theme: &Theme, ctx: &RenderCtx) -> Vec<Line<'static>> {
    let mut out: Vec<Line<'static>> = Vec::new();
    let args = call.get("arguments").or_else(|| call.get("input"));
    let params = TaskParamsView::from_value(args.filter(|a| a.is_object()));
    let header = format_agent_header_label(args.filter(|a| a.is_object()))
        .unwrap_or_else(|| "task".to_string());
    let mut head = vec![
        Span::styled("◆ ", Style::default().fg(theme.color("accent"))),
        Span::styled(
            format!("task {header}"),
            Style::default()
                .fg(theme.color("accent"))
                .add_modifier(Modifier::BOLD),
        ),
    ];
    if params.isolated || params.tasks.iter().any(|t| t.isolated) {
        head.push(Span::styled(
            "  isolated".to_string(),
            Style::default().fg(theme.color("text_muted")),
        ));
    }
    out.push(Line::from(head));

    // Shared batch context renders before the assignment briefs.
    if let Some(context) = params.context.as_deref().filter(|c| !c.trim().is_empty()) {
        section_lines(
            context,
            "",
            if ctx.expanded { 12 } else { 3 },
            theme,
            &mut out,
        );
    }
    for spans in render_task_call_lines(args.filter(|a| a.is_object()), theme) {
        out.push(Line::from(spans));
    }

    let details = details_of(call);
    if !details.results.is_empty() {
        for result in order_results(&details.results) {
            render_agent_result_lines(result, "", "  ", ctx, theme, 0, &mut out);
        }
    } else if !details.progress.is_empty() {
        let ordered = order_progress(&details.progress);
        let hidden_count = if ctx.expanded {
            0
        } else {
            ordered.len().saturating_sub(COLLAPSED_AGENT_LIMIT)
        };
        for prog in &ordered[hidden_count..] {
            render_agent_progress_lines(prog, "", "  ", ctx, theme, 1, &mut out);
        }
        if hidden_count > 0 {
            dim_line(
                "  ",
                hidden_progress_line(&ordered[..hidden_count]),
                theme,
                &mut out,
            );
        }
    }
    if let (Some(state), Some(job)) = (&details.async_state, &details.async_job_id) {
        dim_line("  ", format!("async job {job}: {state}"), theme, &mut out);
    }
    if details.total_duration_ms > 0 && !details.results.is_empty() {
        dim_line(
            "  ",
            format!("total {}", format_duration(details.total_duration_ms)),
            theme,
            &mut out,
        );
    }
    out
}

/// Current epoch time in milliseconds.
fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Renderer for subagent spawns (`tool.task`).
pub struct Task;

impl ToolRenderer for Task {
    fn summary(&self, call: &Value) -> String {
        let args = call.get("arguments").or_else(|| call.get("input"));
        let params = TaskParamsView::from_value(args.filter(|a| a.is_object()));
        let brief = params
            .task
            .as_ref()
            .map(|v| task_first_line(v))
            .or_else(|| params.context.clone())
            .unwrap_or_default();
        let header = format_agent_header_label(args.filter(|a| a.is_object()))
            .or_else(|| {
                (!params.tasks.is_empty()).then(|| format!("batch ×{}", params.tasks.len()))
            })
            .unwrap_or_else(|| "task".to_string());
        let summary = if brief.is_empty() {
            header
        } else {
            format!("{header}: {}", one_line_label(&brief, 60))
        };
        if params.isolated {
            format!("{summary} [isolated]")
        } else {
            summary
        }
    }

    fn render(
        &self,
        frame: &mut ratatui::Frame<'_>,
        area: ratatui::layout::Rect,
        call: &Value,
        theme: &Theme,
    ) {
        let ctx = RenderCtx::live(area.width as usize, now_ms());
        frame.render_widget(Paragraph::new(card_lines(call, theme, &ctx)), area);
    }
}

/// Final yield assembly (port of `pi-tui/src/tools/task-yield-assembly.ts`):
/// folds a run's `yield` tool calls into one payload consumed by validation
/// and rendering.
pub mod yield_assembly {
    use std::collections::HashMap;

    use serde_json::Value;

    /// Output-schema shape of a declared top-level property, keyed by yield
    /// label: `array` sections accumulate; `scalar` sections keep the latest.
    pub type YieldSectionShapes = HashMap<String, SectionShape>;

    /// Merge shape for one yield section.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum SectionShape {
        /// Every yield appends to a list.
        Array,
        /// The latest yield wins.
        Scalar,
    }

    /// A yield call's type: a string label is terminal; a non-empty label
    /// array is incremental.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum YieldType {
        /// Terminal label (e.g. `"result"`).
        Terminal(String),
        /// Incremental section labels.
        Incremental(Vec<String>),
    }

    /// One extracted yield tool call (TS `YieldItem`).
    #[derive(Debug, Clone, Default)]
    pub struct YieldItem {
        /// Payload data; `None` when the yield defers to the last turn.
        pub data: Option<Value>,
        /// Outcome status.
        pub status: Option<String>,
        /// Error text on failure.
        pub error: Option<String>,
        /// Terminal vs incremental typing.
        pub item_type: Option<YieldType>,
        /// Resolve the payload from the latest durable assistant text.
        pub use_last_turn: bool,
        /// True when an incremental workpool yield finished its whole batch.
        pub complete: bool,
        /// The in-tool validator accepted schema-invalid data anyway.
        pub schema_overridden: bool,
    }

    impl YieldItem {
        /// Parse one yield call from wire data; `None` for non-objects.
        #[must_use]
        pub fn from_value(v: &Value) -> Option<Self> {
            v.as_object()?;
            let item_type = match v.get("type") {
                Some(Value::String(s)) => Some(YieldType::Terminal(s.trim().to_string())),
                Some(Value::Array(a)) => {
                    let labels: Vec<String> = a
                        .iter()
                        .filter_map(Value::as_str)
                        .map(str::trim)
                        .filter(|s| !s.is_empty())
                        .map(ToOwned::to_owned)
                        .collect();
                    Some(YieldType::Incremental(labels))
                }
                _ => None,
            };
            Some(Self {
                data: v.get("data").cloned(),
                status: v
                    .get("status")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned),
                error: v
                    .get("error")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned),
                item_type,
                use_last_turn: v
                    .get("useLastTurn")
                    .or_else(|| v.get("use_last_turn"))
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
                complete: v.get("complete").and_then(Value::as_bool).unwrap_or(false),
                schema_overridden: v
                    .get("schemaOverridden")
                    .or_else(|| v.get("schema_overridden"))
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
            })
        }

        fn incremental_labels(&self) -> Option<Vec<String>> {
            match self.item_type.as_ref()? {
                YieldType::Incremental(labels) if !labels.is_empty() => Some(labels.clone()),
                _ => None,
            }
        }

        fn is_incremental(&self) -> bool {
            matches!(self.item_type, Some(YieldType::Incremental(ref l)) if !l.is_empty())
        }
    }

    /// Outcome of folding a run's yields, with provenance flags.
    #[derive(Debug, Clone, PartialEq)]
    pub struct AssembledYieldResult {
        /// The assembled payload.
        pub data: Value,
        /// A schema-invalid value was accepted somewhere.
        pub schema_overridden: bool,
        /// The payload came from the last assistant turn as raw text.
        pub raw_text: bool,
        /// Some yield carried no data and no last-turn fallback existed.
        pub missing_data: bool,
    }

    fn get_labels(item_type: Option<&YieldType>) -> Vec<String> {
        match item_type {
            Some(YieldType::Terminal(s)) => {
                let label = s.trim();
                if label.is_empty() {
                    Vec::new()
                } else {
                    vec![label.to_string()]
                }
            }
            Some(YieldType::Incremental(labels)) => labels.clone(),
            None => Vec::new(),
        }
    }

    struct ResolvedPayload {
        value: Value,
        from_last_assistant_text: bool,
        missing_data: bool,
    }

    fn resolve_payload(
        item: &YieldItem,
        last_assistant_text: Option<&str>,
        labels: &[String],
    ) -> ResolvedPayload {
        let has_data = item.data.is_some();
        let should_use_last_turn = item.use_last_turn || (!labels.is_empty() && !has_data);
        if should_use_last_turn {
            if let Some(text) = last_assistant_text {
                return ResolvedPayload {
                    value: Value::String(text.to_string()),
                    from_last_assistant_text: true,
                    missing_data: text.is_empty(),
                };
            }
        }
        let value = item.data.clone().unwrap_or(Value::Null);
        ResolvedPayload {
            missing_data: item.data.is_none() || value.is_null(),
            value,
            from_last_assistant_text: false,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn append_section(
        sections: &mut serde_json::Map<String, Value>,
        section_counts: &mut HashMap<String, usize>,
        label: &str,
        value: Value,
        shape: Option<SectionShape>,
    ) {
        let count = section_counts.get(label).copied().unwrap_or(0);
        if shape == Some(SectionShape::Scalar) {
            sections.insert(label.to_string(), value);
        } else if count == 0 {
            sections.insert(
                label.to_string(),
                if shape == Some(SectionShape::Array) {
                    Value::Array(vec![value])
                } else {
                    value
                },
            );
        } else if let Some(existing) = sections.get_mut(label) {
            if let Some(array) = existing.as_array_mut() {
                array.push(value);
            } else {
                let taken = std::mem::take(existing);
                *existing = Value::Array(vec![taken, value]);
            }
        }
        section_counts.insert(label.to_string(), count + 1);
    }

    /// Assemble typed yield calls into the final payload (TS `assembleYieldResult`).
    ///
    /// An explicit terminal payload (untyped final result or a string-typed
    /// finalize carrying `data`) wins verbatim. Otherwise incremental
    /// (array-typed) yields merge per `section_shapes`; a data-less terminal
    /// keeps accumulated sections; with neither, the last assistant turn
    /// becomes the raw result. Aborted yields are skipped.
    #[must_use]
    pub fn assemble_yield_result(
        yield_items: &[YieldItem],
        last_assistant_text: Option<&str>,
        section_shapes: Option<&YieldSectionShapes>,
    ) -> Option<AssembledYieldResult> {
        if yield_items.is_empty() {
            return None;
        }

        // Terminal = the last non-incremental yield.
        let terminal_item = yield_items.iter().rev().find(|item| !item.is_incremental());

        let mut sections = serde_json::Map::new();
        let mut section_counts: HashMap<String, usize> = HashMap::new();
        let mut overridden_scalars: Vec<String> = Vec::new();
        let mut schema_overridden = false;
        let mut missing_data = false;
        let mut has_sections = false;
        for item in yield_items {
            if item.status.as_deref() == Some("aborted") {
                continue;
            }
            let Some(labels) = item.incremental_labels() else {
                continue;
            };
            let overridden = item.schema_overridden;
            let resolved = resolve_payload(item, last_assistant_text, &labels);
            missing_data |= resolved.missing_data;
            if labels.is_empty() {
                schema_overridden |= overridden;
            }
            for label in &labels {
                let shape = section_shapes.and_then(|shapes| shapes.get(label)).copied();
                append_section(
                    &mut sections,
                    &mut section_counts,
                    label,
                    resolved.value.clone(),
                    shape,
                );
                if shape == Some(SectionShape::Scalar) {
                    if overridden && !overridden_scalars.contains(label) {
                        overridden_scalars.push(label.clone());
                    } else if !overridden {
                        overridden_scalars.retain(|l| l != label);
                    }
                } else {
                    schema_overridden |= overridden;
                }
                has_sections = true;
            }
        }

        if let Some(terminal) = terminal_item {
            if terminal.data.is_some() {
                let resolved = resolve_payload(terminal, last_assistant_text, &[]);
                return Some(AssembledYieldResult {
                    raw_text: resolved.from_last_assistant_text && resolved.value.is_string(),
                    data: resolved.value,
                    schema_overridden: terminal.schema_overridden,
                    missing_data: resolved.missing_data,
                });
            }
        }

        if has_sections {
            return Some(AssembledYieldResult {
                data: Value::Object(sections),
                schema_overridden: schema_overridden || !overridden_scalars.is_empty(),
                raw_text: false,
                missing_data,
            });
        }

        let terminal = terminal_item?;
        let resolved = resolve_payload(
            terminal,
            last_assistant_text,
            &get_labels(terminal.item_type.as_ref()),
        );
        Some(AssembledYieldResult {
            raw_text: resolved.from_last_assistant_text && resolved.value.is_string(),
            data: resolved.value,
            schema_overridden: terminal.schema_overridden,
            missing_data: resolved.missing_data,
        })
    }

    #[cfg(test)]
    mod tests {
        use serde_json::json;

        use super::*;

        fn item(item_type: YieldType, data: Value) -> YieldItem {
            YieldItem {
                data: Some(data),
                item_type: Some(item_type),
                ..YieldItem::default()
            }
        }

        #[test]
        fn empty_yields_assemble_to_nothing() {
            assert!(assemble_yield_result(&[], None, None).is_none());
        }

        #[test]
        fn terminal_data_wins_verbatim() {
            let items = vec![
                item(YieldType::Incremental(vec!["notes".into()]), json!("a")),
                item(YieldType::Terminal("result".into()), json!({"ok": true})),
            ];
            let assembled = assemble_yield_result(&items, None, None).unwrap();
            assert_eq!(assembled.data, json!({"ok": true}));
            assert!(!assembled.schema_overridden && !assembled.raw_text && !assembled.missing_data);
        }

        #[test]
        fn incremental_sections_accumulate_and_scalar_keeps_latest() {
            let mut shapes = YieldSectionShapes::new();
            shapes.insert("notes".into(), SectionShape::Array);
            shapes.insert("verdict".into(), SectionShape::Scalar);
            let items = vec![
                item(YieldType::Incremental(vec!["notes".into()]), json!(1)),
                item(YieldType::Incremental(vec!["notes".into()]), json!(2)),
                item(YieldType::Incremental(vec!["verdict".into()]), json!("old")),
                item(YieldType::Incremental(vec!["verdict".into()]), json!("new")),
            ];
            let assembled = assemble_yield_result(&items, None, Some(&shapes)).unwrap();
            assert_eq!(assembled.data, json!({"notes": [1, 2], "verdict": "new"}));
        }

        #[test]
        fn data_less_terminal_keeps_sections() {
            let items = vec![
                item(
                    YieldType::Incremental(vec!["findings".into()]),
                    json!(["f1"]),
                ),
                YieldItem {
                    data: None,
                    item_type: Some(YieldType::Terminal("result".into())),
                    ..YieldItem::default()
                },
            ];
            let assembled = assemble_yield_result(&items, None, None).unwrap();
            assert_eq!(assembled.data, json!({"findings": ["f1"]}));
        }

        #[test]
        fn use_last_turn_falls_back_to_assistant_text() {
            let items = vec![YieldItem {
                data: None,
                item_type: Some(YieldType::Terminal(String::new())),
                use_last_turn: true,
                ..YieldItem::default()
            }];
            let assembled = assemble_yield_result(&items, Some("final answer"), None).unwrap();
            assert_eq!(assembled.data, json!("final answer"));
            assert!(assembled.raw_text);
        }

        #[test]
        fn aborted_yields_are_skipped() {
            let items = vec![
                YieldItem {
                    data: Some(json!("bad")),
                    status: Some("aborted".into()),
                    item_type: Some(YieldType::Incremental(vec!["notes".into()])),
                    ..YieldItem::default()
                },
                item(YieldType::Incremental(vec!["notes".into()]), json!("good")),
            ];
            let assembled = assemble_yield_result(&items, None, None).unwrap();
            assert_eq!(assembled.data, json!({"notes": "good"}));
        }
    }
}

/// Double-encoded prose repair (port of `pi-tui/src/tools/task-repair-args.ts`).
///
/// Models occasionally JSON-escape a natural-language field twice; the object
/// still parses, so the validator never fires, and the subagent receives the
/// garbled prompt. Repair is deliberately scoped to the task tool's prose
/// fields (`task`, shared `context`, batch items' `task`) — identifier fields
/// and code-bearing tools are never touched.
pub mod repair {
    use std::borrow::Cow;

    use serde_json::Value;

    /// Whether `value` carries the signature of whole-string double-encoding
    /// rather than an incidental escape mention (TS `hasDoubleEncodeSignature`):
    /// a structural escape (`\"`, `\\`, `\/`, `\uXXXX`) or two-plus escapes.
    fn has_double_encode_signature(value: &str) -> bool {
        let bytes = value.as_bytes();
        let mut i = 0;
        let mut count = 0;
        while i < bytes.len() {
            if bytes[i] == b'\\' {
                if i + 1 < bytes.len() {
                    let next = bytes[i + 1];
                    if matches!(next, b'"' | b'\\' | b'/')
                        || (next == b'u'
                            && i + 5 < bytes.len()
                            && bytes[i + 2..=i + 5].iter().all(|b| b.is_ascii_hexdigit()))
                    {
                        return true;
                    }
                }
                count += 1;
                if count >= 2 {
                    return true;
                }
                i += 2; // skip the escaped char so `\\` counts once
            } else {
                i += 1;
            }
        }
        false
    }

    /// Return the once-unescaped string when `value` is uniformly
    /// double-encoded JSON (a well-formed JSON string body decoding to a
    /// different string); otherwise return `value` unchanged (TS
    /// `repairDoubleEncodedJsonString`). The round-trip only succeeds when
    /// every backslash begins a valid JSON escape and no bare double-quote
    /// exists, so genuine prose (`C:\Users`, regexes) passes through untouched.
    #[must_use]
    pub fn repair_double_encoded_json_string(value: &str) -> Cow<'_, str> {
        if !value.contains('\\') || !has_double_encode_signature(value) {
            return Cow::Borrowed(value);
        }
        // Wrap in quotes and decode as a JSON string, mirroring `JSON.parse(`"${value}"`)`.
        let mut quoted = String::with_capacity(value.len() + 2);
        quoted.push('"');
        quoted.push_str(value);
        quoted.push('"');
        match serde_json::from_str::<String>(&quoted) {
            Ok(decoded) if decoded != value => Cow::Owned(decoded),
            _ => Cow::Borrowed(value),
        }
    }

    /// Repair double-encoded prose in task-tool params in place (flat `task`,
    /// shared `context`, each batch item's `task`); identifier fields are
    /// never repaired (TS `repairTaskParams`).
    pub fn repair_task_params(params: &mut Value) {
        let Some(obj) = params.as_object_mut() else {
            return;
        };
        for key in ["task", "context"] {
            if let Some(Value::String(s)) = obj.get(key) {
                if let Cow::Owned(fixed) = repair_double_encoded_json_string(s) {
                    obj.insert(key.to_string(), Value::String(fixed));
                }
            }
        }
        if let Some(Value::Array(items)) = obj.get_mut("tasks") {
            for item in items {
                let Some(item_obj) = item.as_object_mut() else {
                    continue;
                };
                if let Some(Value::String(s)) = item_obj.get("task") {
                    if let Cow::Owned(fixed) = repair_double_encoded_json_string(s) {
                        item_obj.insert("task".to_string(), Value::String(fixed));
                    }
                }
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use serde_json::json;

        use super::*;

        #[test]
        fn repairs_double_encoded_prose() {
            let garbled = "# Role\\nYou are a judge \\\"describe this\\\" \\u2014 done";
            let repaired = repair_double_encoded_json_string(garbled);
            assert_eq!(repaired, "# Role\nYou are a judge \"describe this\" — done");
        }

        #[test]
        fn leaves_genuine_prose_untouched() {
            for genuine in [
                "split lines on \\n",
                "match C:\\Users\\name",
                "use the regex \\d+ here",
                "no escapes at all",
            ] {
                assert_eq!(repair_double_encoded_json_string(genuine), genuine);
            }
        }

        #[test]
        fn repairs_task_params_fields_only() {
            let mut params = json!({
                "name": "agent\\none",
                "task": "say \\\"hi\\\"",
                "context": "ctx\\u00e9",
                "tasks": [
                    {"name": "keep\\nme", "task": "a\\nb"},
                    {"agent": "scout"}
                ]
            });
            repair_task_params(&mut params);
            assert_eq!(params["name"], json!("agent\\none")); // identifiers untouched
            assert_eq!(params["task"], json!("say \"hi\""));
            assert_eq!(params["context"], json!("ctxé"));
            assert_eq!(params["tasks"][0]["name"], json!("keep\\nme"));
            assert_eq!(params["tasks"][0]["task"], json!("a\\nb")); // lone \\n: untouched
        }
    }
}

#[cfg(test)]
mod tests {
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;
    use serde_json::json;

    use super::*;

    const NOW: u64 = 1_800_000_000_000;

    fn line_text(lines: &[Line<'static>]) -> Vec<String> {
        lines
            .iter()
            .map(|l| {
                l.spans
                    .iter()
                    .map(|s| s.content.as_ref())
                    .collect::<String>()
            })
            .collect()
    }

    #[test]
    fn one_line_label_collapses_junk_and_caps() {
        assert_eq!(one_line_label("  hello \n\t world  ", 80), "hello world");
        assert_eq!(one_line_label("a\u{200B}b\u{85}c", 80), "a b c");
        let long = "x".repeat(100);
        assert_eq!(one_line_label(&long, 80).chars().count(), 80);
        assert!(one_line_label(&long, 80).ends_with('…'));
    }

    #[test]
    fn format_duration_matches_ts() {
        assert_eq!(format_duration(0), "0ms");
        assert_eq!(format_duration(900), "900ms");
        assert_eq!(format_duration(1_500), "1.5s");
        assert_eq!(format_duration(65_000), "1m5s");
        assert_eq!(format_duration(120_000), "2m");
        assert_eq!(format_duration(3_600_000), "1h");
        assert_eq!(format_duration(86_400_000 * 2), "2d");
    }

    #[test]
    fn header_label_only_for_flat_agent() {
        assert_eq!(
            format_agent_header_label(Some(&json!({"agent": "scout", "task": "x"}))),
            Some("scout".to_string())
        );
        assert_eq!(format_agent_header_label(Some(&json!({"task": "x"}))), None);
        assert_eq!(format_agent_header_label(None), None);
    }

    #[test]
    fn summary_shows_agent_and_brief() {
        let call =
            json!({"name":"task","arguments":{"agent":"scout","task":"Fix the login\nflow"}});
        assert_eq!(Task.summary(&call), "scout: Fix the login flow");
        let batch = json!({"name":"task","arguments":{"tasks":[{"task":"a"},{"task":"b"}]}});
        assert_eq!(Task.summary(&batch), "batch ×2");
    }

    #[test]
    fn batch_call_lines_fold_after_collapsed_limit() {
        let theme = Theme::default();
        let args = json!({"tasks": (1..=6).map(|i| json!({"name": format!("a{i}"), "task": format!("job {i}")})).collect::<Vec<_>>()});
        let lines = render_task_call_lines(Some(&args), &theme);
        assert_eq!(lines.len(), COLLAPSED_AGENT_LIMIT + 1);
        let text = line_text(
            &lines
                .iter()
                .map(|spans| Line::from(spans.clone()))
                .collect::<Vec<_>>(),
        );
        assert!(text.last().unwrap().contains("… 2 more agents"));
    }

    #[test]
    fn progress_row_shows_status_model_tool_and_elapsed() {
        let theme = Theme::default();
        let update = json!({
            "type": "tool_execution_update",
            "partialResult": {
                "index": 0,
                "id": "BraveFox",
                "agent": "scout",
                "status": "running",
                "task": "survey the repo",
                "resolvedModelIdentity": "zai/glm-5.3-flash",
                "currentTool": "bash",
                "currentToolArgs": "cargo test",
                "currentToolStartMs": NOW - 12_000,
                "toolCount": 3,
                "requests": 2,
                "durationMs": 30_000,
                "completionPercent": 40
            }
        });
        let details = details_of(&update);
        assert_eq!(details.progress.len(), 1);
        let ctx = RenderCtx::live(120, NOW);
        let mut out = Vec::new();
        render_agent_progress_lines(&details.progress[0], "", "  ", &ctx, &theme, 1, &mut out);
        let text = line_text(&out).join("\n");
        assert!(text.contains("BraveFox"), "{text}");
        assert!(text.contains("zai/glm-5.3-flash"), "{text}");
        assert!(text.contains("bash: cargo test"), "{text}");
        assert!(text.contains("12.0s"), "elapsed tool time shown: {text}");
        assert!(text.contains("40%"), "{text}");
        assert!(text.contains("3 tools"), "{text}");
        assert!(text.contains("30.0s"), "run duration shown: {text}");
    }

    #[test]
    fn retry_state_renders_warning_badge_and_line() {
        let theme = Theme::default();
        let update = json!({
            "partialResult": [{
                "index": 0, "id": "CalmOak", "agent": "", "status": "running",
                "task": "t",
                "retryState": {"attempt": 2, "maxAttempts": 5, "delayMs": 60_000,
                               "errorMessage": "429 rate limit", "startedAtMs": NOW - 10_000}
            }]
        });
        let details = details_of(&update);
        let ctx = RenderCtx::live(120, NOW);
        let mut out = Vec::new();
        render_agent_progress_lines(&details.progress[0], "", "  ", &ctx, &theme, 1, &mut out);
        let text = line_text(&out).join("\n");
        assert!(text.contains("retrying"), "{text}");
        assert!(text.contains("429 rate limit"), "{text}");
        assert!(text.contains("2/5"), "{text}");
    }

    #[test]
    fn lifecycle_payload_maps_to_progress_row() {
        let theme = Theme::default();
        let frame =
            json!({"partialResult": {"id": "SwiftPine", "agent": "scout", "status": "started"}});
        let details = details_of(&frame);
        assert_eq!(details.progress.len(), 1);
        assert_eq!(details.progress[0].status, AgentStatus::Running);
        let mut out = Vec::new();
        render_agent_progress_lines(
            &details.progress[0],
            "",
            "  ",
            &RenderCtx::live(80, NOW),
            &theme,
            1,
            &mut out,
        );
        assert!(line_text(&out).join("\n").contains("SwiftPine"));
    }

    #[test]
    fn nested_progress_caps_at_max_depth() {
        let theme = Theme::default();
        // Build an in-flight nested chain 10 levels deep via inflightTaskDetails.
        let mut leaf = json!({"progress": [{"index": 0, "id": "leaf", "agent": "", "status": "running", "task": "t"}]});
        for i in 0..10 {
            leaf = json!({"progress": [{
                "index": 0, "id": format!("level{i}"), "agent": "", "status": "running", "task": "t",
                "inflightTaskDetails": leaf
            }]});
        }
        let details = details_of(&json!({"partialResult": leaf}));
        let ctx = RenderCtx::live(120, NOW);
        let mut out = Vec::new();
        for p in &details.progress {
            render_agent_progress_lines(p, "", "  ", &ctx, &theme, 1, &mut out);
        }
        let text = line_text(&out).join("\n");
        let cap_lines = text.matches("nested task depth limit reached").count();
        assert!(cap_lines >= 1, "depth cap line rendered: {text}");
        // The leaf beyond the cap must never appear.
        assert!(!text.contains("leaf"), "{text}");
    }

    #[test]
    fn result_rows_render_status_and_output_preview() {
        let theme = Theme::default();
        let call = json!({
            "name": "task",
            "arguments": {"agent": "reviewer", "task": "review the diff"},
            "details": {
                "totalDurationMs": 90_000,
                "results": [{
                    "index": 0, "id": "WiseOwl", "agent": "reviewer", "exitCode": 0,
                    "output": "line1\nline2\nline3\nline4", "durationMs": 45_000,
                    "requests": 3, "resolvedModel": "zai/glm-5.3-flash"
                }]
            }
        });
        let lines = card_lines(&call, &theme, &RenderCtx::live(120, NOW));
        let text = line_text(&lines).join("\n");
        assert!(text.contains("◆ task reviewer"), "{text}");
        assert!(text.contains("WiseOwl"), "{text}");
        assert!(text.contains("done"), "{text}");
        assert!(text.contains("45.0s"), "{text}");
        assert!(text.contains("… 1 more line"), "{text}");
        assert!(text.contains("total 1m30s"), "{text}");
    }

    #[test]
    fn failed_result_without_output_renders_error_detail_block() {
        let theme = Theme::default();
        let call = json!({
            "name": "task",
            "arguments": {"task": "explore"},
            "details": {"results": [{
                "index": 0, "id": "Gruff", "agent": "scout", "exitCode": 1,
                "output": "", "error": "Error: provider exploded",
                "durationMs": 1_000
            }]}
        });
        let lines = card_lines(&call, &theme, &RenderCtx::live(120, NOW));
        let text = line_text(&lines).join("\n");
        assert!(text.contains("failed"), "{text}");
        assert!(text.contains("  provider exploded"), "{text}");
        assert!(!text.contains("Error: provider"), "{text}");
    }

    #[test]
    fn collapsed_progress_folds_with_status_counts() {
        let theme = Theme::default();
        let progress: Vec<Value> = (0..7)
            .map(|i| {
                json!({"index": i, "id": format!("agent{i}"), "agent": "", "status": "completed",
                       "task": "t", "durationMs": (i + 1) * 1000})
            })
            .collect();
        let call = json!({"name": "task", "arguments": {"task": "batch"}, "details": {"progress": progress}});
        let lines = card_lines(&call, &theme, &RenderCtx::live(120, NOW));
        let text = line_text(&lines).join("\n");
        assert!(text.contains("… 3 more agents (3 done)"), "{text}");
    }

    #[test]
    fn render_produces_card_in_buffer() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let call = json!({
            "name": "task",
            "arguments": {"agent": "scout", "task": "map the module"},
            "partialResult": {"index": 0, "id": "KindMaple", "agent": "scout",
                              "status": "running", "task": "map the module",
                              "currentTool": "read", "durationMs": 5000}
        });
        terminal
            .draw(|f| {
                let area = f.area();
                Task.render(f, area, &call, &Theme::default());
            })
            .unwrap();
        let buffer = terminal.backend().buffer().clone();
        let mut rendered = String::new();
        for y in 0..buffer.area.height {
            for x in 0..buffer.area.width {
                rendered.push_str(buffer[(x, y)].symbol());
            }
        }
        assert!(rendered.contains("task scout"), "{rendered}");
        assert!(rendered.contains("KindMaple"), "{rendered}");
    }
}
