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
use omp_task::envelope::format_task_result;
use omp_task::params::{SchemaMode, TaskItem};
use omp_task::runner::{RunStatus, TaskRunner, TaskSpec};
use pa_agent::types::AgentToolResult;
use serde::Deserialize;
use serde_json::{json, Value};

use super::Env;

/// omp `READ_ONLY_TOOL_NAMES` (`task/read-only-policy.ts`): approval tier
/// "read" tools; an agent is read-only iff its tools are a non-empty subset.
const READ_ONLY_TOOLS: &[&str] = &[
    "read",
    "wait",
    "grep",
    "glob",
    "find",
    "web_search",
    "ast_grep",
    "yield",
    "ask",
    "todo",
    "recall",
    "reflect",
    "retain",
    "memory_edit",
    "checkpoint",
    "rewind",
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
    /// Per-item model selector (string or prioritized list), as advertised
    /// by the wire schema.
    #[serde(default)]
    model: Option<Value>,
    /// Per-item tool allow-list override.
    #[serde(default)]
    tools: Option<Vec<String>>,
}

fn default_agent() -> String {
    "task".to_string()
}

impl WireTaskItem {
    fn to_item(&self, index: usize) -> Result<TaskItem> {
        let Some(solution_space) = self
            .solution_space
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
        else {
            bail!("Task {} is missing `solutionSpace`.", index + 1);
        };
        let label = match self
            .name
            .as_deref()
            .map(str::trim)
            .filter(|n| !n.is_empty())
        {
            Some(name) => format!("Task {} (`{name}`)", index + 1),
            None => format!("Task {}", index + 1),
        };
        if let Some(model) = &self.model {
            if let Some(reason) = invalid_model_selector_reason(model, &label) {
                bail!("{reason}");
            }
        }
        let item = TaskItem {
            name: self.name.clone(),
            agent: self.agent.clone(),
            task: self.task.clone(),
            solution_space: solution_space.to_string(),
            model: self.model.clone(),
            output_schema: self.output_schema.clone(),
            schema_mode: self.schema_mode,
            tools: self.tools.clone(),
        };
        item.validate().map_err(|e| anyhow::anyhow!(e))?;
        Ok(item)
    }
}

/// Port of TS `invalidModelSelectorReason` (task/structured-subagent.ts:313):
/// a selector is a string or a non-empty array of strings; comma-separated
/// lists flatten; the literals `default`/`inherit` (with or without a
/// `:thinking` suffix) are ambiguous — `@default` states the inherit intent.
fn invalid_model_selector_reason(model: &Value, label: &str) -> Option<String> {
    let invalid = format!(
        "{label} has an invalid `model` value {model}. Use a selector or a non-empty array of selectors such as \"openai/gpt-5.4:high\" or \"@smol\"."
    );
    let values: Vec<&Value> = model
        .as_array()
        .map_or_else(|| vec![model], |a| a.iter().collect());
    let mut patterns: Vec<&str> = Vec::new();
    for value in values {
        let Some(selector) = value.as_str() else {
            return Some(invalid);
        };
        let normalized: Vec<&str> = selector
            .split(',')
            .map(str::trim)
            .filter(|p| !p.is_empty())
            .collect();
        if normalized.is_empty() {
            return Some(invalid);
        }
        patterns.extend(normalized);
    }
    if patterns.is_empty() {
        return Some(invalid);
    }
    if patterns.iter().any(|pattern| {
        let base = pattern.split(':').next().unwrap_or(pattern);
        base.eq_ignore_ascii_case("default") || base.eq_ignore_ascii_case("inherit")
    }) {
        return Some(format!(
            "{label} has an ambiguous `model` value {model}. Use \"@default\" to inherit the parent session's model, or name a model explicitly."
        ));
    }
    None
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
    /// Flat-shape model selector (TS validates the call-level selector too).
    #[serde(default)]
    model: Option<Value>,
}

/// Execute a `task` call: validate, spawn every item, render envelopes.
pub async fn task(env: std::sync::Arc<Env>, args: Value) -> Result<AgentToolResult> {
    let params: WireParams =
        serde_json::from_value(args).map_err(|e| anyhow::anyhow!("invalid task params: {e}"))?;

    let mut specs = Vec::new();
    if params.tasks.is_empty() {
        let Some(task_text) = params
            .task
            .as_deref()
            .map(str::trim)
            .filter(|t| !t.is_empty())
        else {
            bail!("Missing `task`. Provide `tasks[]` items or a flat `task` string.");
        };
        let solution_space = params.solution_space.as_deref().unwrap_or_default();
        if let Some(model) = &params.model {
            if let Some(reason) = invalid_model_selector_reason(model, "The call") {
                bail!("{reason}");
            }
        }
        let item = TaskItem {
            name: None,
            agent: params.agent.clone(),
            task: task_text.to_string(),
            solution_space: solution_space.to_string(),
            model: params.model.clone(),
            output_schema: None,
            schema_mode: SchemaMode::Permissive,
            tools: None,
        };
        item.validate().map_err(|e| anyhow::anyhow!(e))?;
        specs.push(TaskSpec::from_item(
            &item,
            params.context.as_deref(),
            env.cwd.clone(),
        ));
    } else {
        if params.task.is_some() {
            bail!(
                "Top-level `task` is not part of the batch shape. Put the work in `tasks[]` items."
            );
        }
        for (index, item) in params.tasks.iter().enumerate() {
            specs.push(TaskSpec::from_item(
                &item.to_item(index)?,
                params.context.as_deref(),
                env.cwd.clone(),
            ));
        }
    }

    // TS `canSpawnAtDepth` (`task.maxRecursionDepth`, default 2): the task
    // tool spawns subagents too, so it carries the same recursion gate as
    // agent()/workpool().
    if let Err(error) = omp_task::recursion_gate() {
        bail!("{error}");
    }
    for spec in &mut specs {
        // Children inherit the parent's extension/no-extensions flags and
        // tool restriction (EVAL-CHILD-POLICY-NOT-PROPAGATED).
        super::eval_agents::apply_child_policy(spec);
    }

    let binary = std::env::current_exe().unwrap_or_else(|_| std::path::PathBuf::from("omp-rs"));
    let runner = TaskRunner::new(binary, MAX_CONCURRENCY);
    let outputs = runner.spawn_all(specs).await;
    merge_outputs(&outputs)
}

/// Merge per-child outputs into the tool result (TS `mergeSyncPayloads`):
/// every envelope is rendered; any failed/cancelled child keeps the merged
/// view's content/details and sets `isError: true` so the agent loop can
/// retry (TS `...(merged.isError ? { isError: true } : {})`).
fn merge_outputs(outputs: &[omp_task::runner::RunOutput]) -> Result<AgentToolResult> {
    let mut text = String::new();
    let mut failed = 0usize;
    for out in outputs {
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
    let is_error = if failed > 0 {
        use std::fmt::Write as _;
        let _ = write!(text, "\n{failed} of {} task(s) failed.", outputs.len());
        Some(true)
    } else {
        None
    };
    let details = json!({
        "results": outputs.iter().map(format_task_result).collect::<Vec<_>>(),
    });
    Ok(AgentToolResult {
        content: vec![pa_agent::types::ToolResultContent::text(text.trim_end())],
        details,
        terminate: None,
        is_error,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wire_item(args: serde_json::Value) -> WireTaskItem {
        serde_json::from_value(args).unwrap()
    }

    #[test]
    fn per_item_model_and_tools_are_honored() {
        let item = wire_item(json!({
            "agent": "task",
            "task": "do it",
            "solutionSpace": ".",
            "model": "openai/gpt-5.4:high",
            "tools": ["read", "grep"],
        }));
        let item = item.to_item(0).expect("valid item");
        assert_eq!(item.model.as_ref(), Some(&json!("openai/gpt-5.4:high")));
        assert_eq!(
            item.tools.as_deref(),
            Some(["read".to_string(), "grep".to_string()].as_slice())
        );
        let spec = omp_task::runner::TaskSpec::from_item(&item, None, ".");
        assert_eq!(spec.model.as_deref(), Some("openai/gpt-5.4:high"));
        assert_eq!(
            spec.tools.as_deref(),
            Some(["read".to_string(), "grep".to_string()].as_slice())
        );
    }

    #[test]
    fn per_item_model_list_flattens_to_selector() {
        let item = wire_item(json!({
            "task": "do it",
            "solutionSpace": ".",
            "model": ["openai/gpt-5.4:high", "anthropic/claude-sonnet-4"],
        }));
        let item = item.to_item(0).expect("valid item");
        let spec = omp_task::runner::TaskSpec::from_item(&item, None, ".");
        assert_eq!(
            spec.model.as_deref(),
            Some("openai/gpt-5.4:high,anthropic/claude-sonnet-4")
        );
    }

    #[test]
    fn invalid_model_selectors_are_rejected() {
        for model in [
            json!("default"),
            json!("DEFAULT"),
            json!("default:high"),
            json!("inherit"),
            json!(42),
            json!(""),
            json!([]),
            json!(["openai/gpt-5.4", 7]),
        ] {
            let item = wire_item(json!({
                "task": "do it",
                "solutionSpace": ".",
                "model": model,
            }));
            assert!(item.to_item(0).is_err(), "model {model} must be rejected");
        }
        for model in [
            json!("openai/gpt-5.4:high"),
            json!("@default"),
            json!("@smol"),
            json!(["a", "b"]),
        ] {
            let item = wire_item(json!({
                "task": "do it",
                "solutionSpace": ".",
                "model": model,
            }));
            assert!(item.to_item(0).is_ok(), "model {model} must be accepted");
        }
    }

    fn run_output(id: &str, status: RunStatus) -> omp_task::runner::RunOutput {
        omp_task::runner::RunOutput {
            id: id.to_string(),
            agent: "task".to_string(),
            status,
            exit_code: None,
            output: "out".to_string(),
            stderr: String::new(),
            schema: omp_task::SchemaValidation::NoSchema,
            duration: std::time::Duration::ZERO,
            truncated: false,
        }
    }

    #[test]
    fn failed_children_surface_as_tool_error() {
        let outputs = vec![
            run_output("a", RunStatus::Completed),
            run_output("b", RunStatus::Failed),
        ];
        let result = merge_outputs(&outputs).expect("merged result");
        let text = result
            .content
            .iter()
            .find_map(|part| match part {
                pa_agent::types::ToolResultContent::Text(text) => Some(text.text.clone()),
                pa_agent::types::ToolResultContent::Image(_) => None,
            })
            .expect("text content");
        assert_eq!(result.is_error, Some(true));
        assert!(text.contains("1 of 2 task(s) failed."), "{text}");
        assert!(
            text.contains("=== b (task)"),
            "per-child envelopes kept: {text}"
        );
        assert!(
            result.details["results"].as_array().is_some_and(|r| r.len() == 2),
            "details.results kept: {}",
            result.details
        );
        let ok = merge_outputs(&[run_output("a", RunStatus::Completed)]).expect("ok");
        assert_eq!(ok.is_error, None);
    }

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
