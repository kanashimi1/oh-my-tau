//! `/plan`: plan mode (read-only planning, then execute on approval). TS: plan-mode/, slash-commands/builtin-modes.ts.
//!
//! `/plan [task]` enters plan mode: the agent is told it is planning and the
//! side-effecting tools (`bash`/`edit`/`write`/…) are blocked by a tool-call
//! gate while the mode is active. The produced plan is appended to the session
//! (a `custom` entry) and shown. `/plan approve` (or `/plan go`) leaves the
//! mode and sends the saved plan as the next prompt to execute; `/plan off`
//! leaves without executing; `/plan status` reports the mode.
//!
//! Plan state lives in a process-wide map keyed by session id (the gate is
//! process-wide and reflects the single live runtime/session).

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use pa_agent::agent_loop::BeforeToolCallFn;
use pa_agent::types::BeforeToolCallResult;
use serde_json::json;

use crate::rpc::slash::Outcome;
use crate::runtime::Runtime;

/// Tools allowed while planning. Everything else (write side effects: `bash`,
/// `edit`, `write`, `eval`, `ast_edit`, `task`, `checkpoint`, `rewind`,
/// `wait`, …) is refused with a clear error. `ast_grep` is the read-only
/// structural-search half of `ast`; `ast_edit` rewrites files and is blocked.
const ALLOWED_TOOLS: &[&str] = &[
    "read",
    "glob",
    "grep",
    "find",
    "ast_grep",
    "lsp",
    "web_search",
    "think",
    "todo",
    "ask",
];

/// Whether `name` may run during plan mode.
#[must_use]
pub fn is_plan_tool(name: &str) -> bool {
    ALLOWED_TOOLS.contains(&name)
}

/// Per-session plan state.
#[derive(Debug, Clone, Default)]
struct PlanSessionState {
    enabled: bool,
    task: String,
    plan: String,
}

static PLAN_STATE: OnceLock<Mutex<HashMap<String, PlanSessionState>>> = OnceLock::new();

fn state() -> &'static Mutex<HashMap<String, PlanSessionState>> {
    PLAN_STATE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn lock_state() -> std::sync::MutexGuard<'static, HashMap<String, PlanSessionState>> {
    state()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Is any session currently in plan mode? The gate (installed per agent via
/// [`gate`]) consults this so it never needs the session id itself.
fn is_any_plan_active() -> bool {
    lock_state().values().any(|p| p.enabled)
}

/// Wrap the agent's `before_tool_call` to refuse side-effecting tools while
/// plan mode is active. Composed after [`crate::tools::relevance::wrap_before`]
/// in `agent::build_agent`.
#[must_use]
pub fn gate(inner: Option<BeforeToolCallFn>) -> BeforeToolCallFn {
    Arc::new(move |ctx, signal| {
        if is_any_plan_active() && !is_plan_tool(&ctx.tool_call.name) {
            let reason = format!(
                "Plan mode is read-only: `{}` is blocked. Use the read-only tools \
                 (read/search/find/ast/lsp/web_search/think/todo/ask) to investigate, \
                 then run /plan approve to execute.",
                ctx.tool_call.name
            );
            let blocked = BeforeToolCallResult {
                block: true,
                reason: Some(reason),
                args: None,
                additional_context: None,
            };
            return Box::pin(async move { Ok(Some(blocked)) });
        }
        match &inner {
            Some(f) => f(ctx, signal),
            None => Box::pin(async { Ok(None) }),
        }
    })
}

/// Parsed `/plan` invocation.
#[derive(Debug, Clone, PartialEq)]
pub enum PlanCommand {
    /// Bare `/plan`: toggle the mode on (or report when already on).
    Toggle,
    /// `/plan approve` / `/plan go`: leave plan mode and run the saved plan.
    Approve,
    /// `/plan off`: leave plan mode without executing.
    Off,
    /// `/plan status`: report the mode.
    Status,
    /// `/plan <task>`: enter plan mode and plan this task.
    Task(String),
}

/// Parse `/plan` arguments (TS `handlePlanModeCommand` subcommand set). The first
/// whitespace-delimited token selects the subcommand; anything else is a task.
#[must_use]
pub fn parse_args(args: &str) -> PlanCommand {
    let trimmed = args.trim();
    if trimmed.is_empty() {
        return PlanCommand::Toggle;
    }
    match trimmed.split_whitespace().next().unwrap_or_default() {
        "approve" | "go" => PlanCommand::Approve,
        "off" => PlanCommand::Off,
        "status" | "show" => PlanCommand::Status,
        _ => PlanCommand::Task(trimmed.to_string()),
    }
}

pub async fn command(rt: &Arc<Runtime>, args: &str) -> Outcome {
    match parse_args(args) {
        PlanCommand::Task(task) => enter_and_plan(rt, task).await,
        PlanCommand::Approve => approve(rt).await,
        PlanCommand::Off => leave(rt),
        PlanCommand::Status => status(rt),
        PlanCommand::Toggle => toggle(rt),
    }
}

fn output(rt: &Runtime, text: &str) {
    rt.send(json!({"type": "command_output", "text": text}));
}

fn handled() -> Outcome {
    Outcome::Handled {
        agent_invoked: false,
    }
}

fn set_session(session_id: &str, enabled: bool, task: &str) {
    let mut guard = lock_state();
    let entry = guard.entry(session_id.to_string()).or_default();
    entry.enabled = enabled;
    if !task.is_empty() {
        entry.task = task.to_string();
    }
}

/// The TS plan-mode instruction, prefixing the submitted planning task.
fn instruction(task: &str) -> String {
    format!(
        "<critical>\nPlan mode active.\n- Working tree/system read-only: NEVER create, edit, \
         delete, or rename working-tree files; NEVER run state-changing commands (`git commit`, \
         `npm install`, migrations) or otherwise change the system.\n- Use only the read-only \
         investigation tools (read/search/find/ast/lsp/web_search/think/todo/ask).\n- Produce a \
         concrete implementation plan as your final message.\n</critical>\n\nPlanning task:\n{task}"
    )
}

/// `/plan <task>`: enable the gate, submit the task with the planning
/// instruction, wait for the planning turn, then save + show the plan.
async fn enter_and_plan(rt: &Arc<Runtime>, task: String) -> Outcome {
    let session_id = rt.session_id();
    if is_any_plan_active() {
        output(
            rt,
            "Plan mode is already active. Run /plan approve, /plan off, or /plan status.",
        );
        return handled();
    }
    set_session(&session_id, true, &task);
    if let Err(e) = rt
        .prompt(None, &instruction(&task), Some("steer"), Vec::new())
        .await
    {
        set_session(&session_id, false, "");
        output(
            rt,
            &format!("plan: failed to start planning: {}", e.message),
        );
        return handled();
    }
    rt.wait_settled().await;
    let plan = rt.last_assistant_text().await.unwrap_or_default();
    {
        let mut guard = lock_state();
        if let Some(p) = guard.get_mut(&session_id) {
            p.plan.clone_from(&plan);
        }
    }
    let _ = rt.append_entry(
        "custom",
        json!({"customType": "plan", "data": {"task": task, "plan": plan}}),
    );
    output(rt, "─ Plan ─────────────────────────────");
    if plan.trim().is_empty() {
        output(rt, "(no plan text produced)");
    } else {
        output(rt, &plan);
    }
    output(rt, "─────────────────────────────────────");
    output(
        rt,
        "Plan mode: ON. Review the plan above, then /plan approve to execute or /plan off to cancel.",
    );
    handled()
}

/// `/plan approve` / `/plan go`: leave plan mode and send the saved plan as the
/// next prompt.
async fn approve(rt: &Arc<Runtime>) -> Outcome {
    let session_id = rt.session_id();
    let (plan, task) = {
        let guard = lock_state();
        match guard.get(&session_id) {
            Some(p) if p.enabled => (p.plan.clone(), p.task.clone()),
            _ => (String::new(), String::new()),
        }
    };
    if plan.trim().is_empty() {
        output(rt, "No plan to approve yet. Run /plan <task> first.");
        return handled();
    }
    set_session(&session_id, false, &task);
    if let Err(e) = rt.prompt(None, &plan, Some("steer"), Vec::new()).await {
        output(rt, &format!("plan: failed to submit plan: {}", e.message));
        return handled();
    }
    output(rt, "Plan approved. Plan mode off; executing the plan.");
    handled()
}

/// `/plan off`: leave plan mode without executing.
fn leave(rt: &Arc<Runtime>) -> Outcome {
    let session_id = rt.session_id();
    let was_active = {
        let mut guard = lock_state();
        let was = guard.get(&session_id).is_some_and(|p| p.enabled);
        if was {
            if let Some(p) = guard.get_mut(&session_id) {
                p.enabled = false;
            }
        }
        was
    };
    output(
        rt,
        if was_active {
            "Plan mode disabled (plan not executed)."
        } else {
            "Plan mode is not active."
        },
    );
    handled()
}

/// `/plan status`.
fn status(rt: &Runtime) -> Outcome {
    let session_id = rt.session_id();
    let state = lock_state().get(&session_id).cloned();
    match state {
        Some(p) if p.enabled => {
            output(rt, "Plan mode: ON");
            if !p.task.is_empty() {
                output(rt, &format!("Task: {}", p.task));
            }
            if p.plan.trim().is_empty() {
                output(rt, "Plan: none captured yet");
            } else {
                output(
                    rt,
                    "Plan: captured (run /plan approve to execute, /plan off to cancel)",
                );
            }
        }
        _ => output(rt, "Plan mode: OFF"),
    }
    handled()
}

/// Bare `/plan`: enable plan mode (no task submitted; the next prompt still
/// plans because the gate stays active).
fn toggle(rt: &Arc<Runtime>) -> Outcome {
    let session_id = rt.session_id();
    if is_any_plan_active() {
        output(
            rt,
            "Plan mode is already active. Run /plan <task>, /plan approve, or /plan off.",
        );
        return handled();
    }
    set_session(&session_id, true, "");
    output(
        rt,
        "Plan mode: ON. Submit a task to plan, or /plan off to disable.",
    );
    handled()
}

#[cfg(test)]
mod tests {
    use super::{is_plan_tool, parse_args, PlanCommand};

    #[test]
    fn parses_subcommands() {
        assert_eq!(parse_args(""), PlanCommand::Toggle);
        assert_eq!(parse_args("   "), PlanCommand::Toggle);
        assert_eq!(parse_args("approve"), PlanCommand::Approve);
        assert_eq!(parse_args("go"), PlanCommand::Approve);
        assert_eq!(parse_args("off"), PlanCommand::Off);
        assert_eq!(parse_args("status"), PlanCommand::Status);
        assert_eq!(parse_args(" show "), PlanCommand::Status);
    }

    #[test]
    fn parses_task() {
        assert_eq!(
            parse_args("refactor X"),
            PlanCommand::Task("refactor X".into())
        );
        assert_eq!(
            parse_args("  plan the migration  "),
            PlanCommand::Task("plan the migration".into())
        );
    }

    #[test]
    fn allowlist_allows_read_only_tools() {
        for name in [
            "read",
            "glob",
            "grep",
            "find",
            "ast_grep",
            "lsp",
            "web_search",
            "think",
            "todo",
            "ask",
        ] {
            assert!(is_plan_tool(name), "{name} should be allowed in plan mode");
        }
    }

    #[test]
    fn allowlist_blocks_side_effecting_tools() {
        for name in [
            "bash",
            "edit",
            "write",
            "eval",
            "ast_edit",
            "task",
            "checkpoint",
            "rewind",
            "wait",
        ] {
            assert!(!is_plan_tool(name), "{name} should be blocked in plan mode");
        }
    }
}
