//! `task`: delegate work to subagents (omp `TaskTool`, `task/index.ts`).
//!
//! Each call carries `context` plus a `tasks[]` batch (the flat single-spawn
//! shape is accepted at runtime, mirroring omp's stale-transcript path); every
//! item spawns one child `omp-rs -p <prompt>` subprocess via [`omp_task`]'s
//! [`TaskRunner`], with agent definitions discovered from
//! `~/.omp/agent/agents` and `.omp/agents`. Results come back as
//! `<task-result>` envelopes (`agent://<id>` pointers + inline preview).

use std::path::{Path, PathBuf};

use anyhow::{bail, Result};
use omp_task::discovery::{discover_project_agents, discover_user_agents};
use omp_task::params::{SchemaMode, TaskItem};
use omp_task::runner::{RunStatus, TaskRunner, TaskSpec};
use omp_task::envelope::format_task_result;
use pa_agent::types::AgentToolResult;
use serde::Deserialize;
use serde_json::{json, Value};

use super::Env;

/// omp `READ_ONLY_TOOL_NAMES` (`task/read-only-policy.ts`): approval tier
/// "read" tools; an agent is read-only iff its tools are a non-empty subset.
const READ_ONLY_TOOLS: &[&str] = &[
    "read", "wait", "grep", "glob", "find", "web_search", "ast_grep", "yield", "ask", "todo",
    "recall", "reflect", "retain", "memory_edit", "checkpoint", "rewind",
];

/// omp `task.maxConcurrency` protocol default.
const MAX_CONCURRENCY: usize = 32;

/// Static header of the tool description (the `task.md` template rendered
/// with the protocol defaults: batch + async enabled, no isolation, default
/// agent `task`). The discovered agent list is appended dynamically.
pub const DESCRIPTION_HEADER: &str = r"Spawn `tasks[]` concurrently; IDs return immediately.

# Results
`outputSchema` parsed payload, even invalid: `agent://<id>` (field `/<field>`, nested `/reports/0/data`); invalid preview inline.

# Delegation
Use most specific agent. Read-only research MUST use `scout` only when files unknown. Prefer one agent to investigate + edit. Omit `agent` only for default (`task`); NEVER specify it.
Shared edits need one integration owner; siblings coordinate via `write agent://<id>`. Set interfaces in `context`. Every task MUST skip build/lint/tests/formatters mid-flight; run once afterward.

# Inputs
`name`: CamelCase ≤32, auto-generated if omitted; address agent by name. `outputSchema` overrides agent/session schemas.
`solutionSpace`: describe how open-ended the child's problem is: whether the fix or design is given, or which causes or designs remain open. Volume of work does not widen it; NEVER mention sibling agents or coordination. (`one fix: rename, names given`; `one fix: slice end in paginate`; `single-flight cache load; races easy to miss`; `several retry API shapes; error classes to choose`; `deadlock cause open, no repro`)
`tools`: eval-defined, run in your kernel.
`model`: selector or ordered array overriding the agent's model for this spawn (`provider/model[:level]`, role alias like `@smol`, `@default` = parent's live model); set per `tasks[]` item, never top-level. Omit unless a specific model is needed. Requested model unavailable → stop and report; NEVER substitute another.
`schemaMode`: default permissive warns after retries; strict fails.
Children start blank; parent IRC steers immediately; large payloads via `local://<path>`, NEVER inline.

# Format
`context`: shared (`# Goal`, `# Contract` interfaces); NEVER repeat per task.
`task`: self-contained (`# Target` files/non-goals, `# Change` steps/APIs, `# Acceptance` observable result).";

/// Wire schema, byte-matching the TS `getTaskSchema` render under protocol
/// defaults (batch enabled, isolation/effort/eval-tools off paths excluded):
/// `context` + `tasks[]` batch shape; a top-level `model` is rejected
/// (`"not": {}` — model selectors are per-item only).
#[must_use]
pub fn schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "context": {"type": "string"},
            "tasks": {"type": "array", "items": {
                "type": "object",
                "properties": {
                    "task": {"type": "string"},
                    "solutionSpace": {"type": "string"},
                    "name": {"type": "string"},
                    "agent": {"type": "string", "default": "task"},
                    "model": {"anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}]},
                    "outputSchema": {"anyOf": [{"type": "object"}, {"type": "boolean"}, {"type": "string"}, {"type": "null"}]},
                    "schemaMode": {"enum": ["permissive", "strict"], "type": "string"},
                    "tools": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["task", "solutionSpace"]
            }},
            "model": {"not": {}}
        },
        "required": ["context", "tasks"]
    })
}

fn agent_is_read_only(tools: Option<&Vec<String>>) -> bool {
    match tools {
        Some(tools) if !tools.is_empty() => {
            tools.iter().all(|t| READ_ONLY_TOOLS.contains(&t.as_str()))
        }
        _ => false,
    }
}

/// Tool description: static header plus the discovered agent list
/// (`~/.omp/agent/agents` + `.omp/agents`, omp's `renderDescription`).
#[must_use]
pub fn description(cwd: &Path) -> String {
    let mut out = String::from(DESCRIPTION_HEADER);
    out.push_str("# Available Agents\n");
    let home = std::env::var_os("HOME").map_or_else(|| PathBuf::from("/"), PathBuf::from);
    let mut agents: Vec<_> = discover_user_agents(&home)
        .agents
        .into_iter()
        .chain(discover_project_agents(cwd).agents)
        .collect();
    agents.sort_by(|a, b| a.fields.name.cmp(&b.fields.name));
    for agent in &agents {
        out.push_str("- `");
        out.push_str(&agent.fields.name);
        out.push('`');
        if agent_is_read_only(agent.fields.tools.as_ref()) {
            out.push_str(" (READ-ONLY; investigation only, no edits)");
        }
        if agent.fields.blocking == Some(true) {
            out.push_str(" (BLOCKING; inline result)");
        }
        out.push_str(": ");
        out.push_str(&agent.fields.description);
        out.push('\n');
    }
    out
}

/// One `tasks[]` item as the wire delivers it (`solutionSpace` required by
/// the schema; `outputSchema` kept raw for [`omp_task::schema`]).
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct WireTaskItem {
    #[serde(default)]
    name: Option<String>,
    #[serde(default = "default_agent")]
    agent: String,
    task: String,
    #[serde(default)]
    solution_space: Option<String>,
    #[serde(default)]
    output_schema: Option<Value>,
    #[serde(default)]
    schema_mode: SchemaMode,
}

fn default_agent() -> String {
    "task".to_string()
}

impl WireTaskItem {
    fn to_item(&self, index: usize) -> Result<TaskItem> {
        let Some(solution_space) = self.solution_space.as_deref().map(str::trim).filter(|s| !s.is_empty()) else {
            bail!("Task {} is missing `solutionSpace`.", index + 1);
        };
        let item = TaskItem {
            name: self.name.clone(),
            agent: self.agent.clone(),
            task: self.task.clone(),
            solution_space: solution_space.to_string(),
            model: None,
            output_schema: self.output_schema.clone(),
            schema_mode: self.schema_mode,
            tools: None,
        };
        item.validate().map_err(|e| anyhow::anyhow!(e))?;
        Ok(item)
    }
}

#[derive(Debug, Deserialize)]
struct WireParams {
    #[serde(default)]
    context: Option<String>,
    #[serde(default)]
    tasks: Vec<WireTaskItem>,
    /// Flat single-spawn shape (stale transcripts / internal callers).
    #[serde(default)]
    task: Option<String>,
    #[serde(default)]
    solution_space: Option<String>,
    #[serde(default = "default_agent")]
    agent: String,
}

/// Execute a `task` call: validate, spawn every item, render envelopes.
pub async fn task(env: std::sync::Arc<Env>, args: Value) -> Result<AgentToolResult> {
    let params: WireParams = serde_json::from_value(args)
        .map_err(|e| anyhow::anyhow!("invalid task params: {e}"))?;

    let mut specs = Vec::new();
    if params.tasks.is_empty() {
        let Some(task_text) = params.task.as_deref().map(str::trim).filter(|t| !t.is_empty()) else {
            bail!("Missing `task`. Provide `tasks[]` items or a flat `task` string.");
        };
        let solution_space = params.solution_space.as_deref().unwrap_or_default();
        let item = TaskItem {
            name: None,
            agent: params.agent.clone(),
            task: task_text.to_string(),
            solution_space: solution_space.to_string(),
            model: None,
            output_schema: None,
            schema_mode: SchemaMode::Permissive,
            tools: None,
        };
        item.validate().map_err(|e| anyhow::anyhow!(e))?;
        specs.push(TaskSpec::from_item(&item, params.context.as_deref(), env.cwd.clone()));
    } else {
        if params.task.is_some() {
            bail!("Top-level `task` is not part of the batch shape. Put the work in `tasks[]` items.");
        }
        for (index, item) in params.tasks.iter().enumerate() {
            specs.push(TaskSpec::from_item(&item.to_item(index)?, params.context.as_deref(), env.cwd.clone()));
        }
    }

    let binary = std::env::current_exe().unwrap_or_else(|_| std::path::PathBuf::from("omp-rs"));
    let runner = TaskRunner::new(binary, MAX_CONCURRENCY);
    let outputs = runner.spawn_all(specs).await;

    let mut text = String::new();
    let mut failed = 0usize;
    for out in &outputs {
        if out.status != RunStatus::Completed {
            failed += 1;
        }
        if outputs.len() > 1 {
            use std::fmt::Write as _;
            let _ = writeln!(text, "=== {} ({})", out.id, out.agent);
        }
        text.push_str(&serde_json::to_string_pretty(&format_task_result(out))?);
        text.push('\n');
    }
    if failed > 0 {
        use std::fmt::Write as _;
        let _ = write!(text, "\n{failed} of {} task(s) failed.", outputs.len());
    }
    Ok(AgentToolResult {
        content: vec![pa_agent::types::ToolResultContent::text(text.trim_end())],
        details: Value::Null,
        terminate: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_matches_ts_reference() {
        let s = schema();
        let props = s["properties"].as_object().unwrap();
        assert_eq!(props.len(), 3);
        assert!(props["context"]["type"] == "string");
        assert!(props["tasks"]["items"]["required"]
            .as_array()
            .unwrap()
            .iter()
            .any(|v| v == "solutionSpace"));
        assert_eq!(s["required"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn flat_shape_parses_as_single_item() {
        let params: WireParams = serde_json::from_value(json!({
            "task": "do the thing", "solutionSpace": "one fix"
        }))
        .unwrap();
        assert!(params.tasks.is_empty() && params.task.is_some());
    }
}
