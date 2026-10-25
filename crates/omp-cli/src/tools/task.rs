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

/// Fan-out ceiling for `tasks[]` batches: the `task.maxConcurrency` setting
/// wins when it is a positive number; otherwise `min(available_parallelism, 8)`
/// (`TaskRunner::default_max_concurrency`; the schema default `0` = auto).
fn effective_max_concurrency(cwd: &std::path::Path) -> usize {
    let settings = omp_settings::Settings::load(cwd);
    TaskRunner::resolve_max_concurrency(settings.get_number("task.maxConcurrency"))
}

/// Static header of the tool description (the `task.md` template rendered
/// with the protocol defaults: batch + async enabled, no isolation, default
/// agent `task`). The discovered agent list is appended dynamically.
pub const DESCRIPTION_HEADER: &str = r"Spawn `tasks[]` concurrently; IDs return immediately.

Subagents run as child processes with bounded concurrency: `task.maxConcurrency` when set to a positive number, else `min(CPU count, 8)`.

# Results
`outputSchema` parsed payload, even invalid: `agent://<id>` (field `/<field>`, nested `/reports/0/data`); invalid preview inline.

# Delegation
Use most specific agent. Prefer one agent to investigate + edit. Omit `agent` only for default (`task`); NEVER specify it.
Shared edits need one integration owner; siblings coordinate via `write agent://<id>`. Set interfaces in `context`.

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
    let mut schema = base_schema();
    // SERA rubric is advertised only when opted in (`OMP_TASK_RUBRIC=1`), so the
    // default tool schema (and request body) stays byte-identical to TS.
    if std::env::var("OMP_TASK_RUBRIC").is_ok_and(|v| v == "1") {
        schema["properties"]["tasks"]["items"]["properties"]["rubric"] = json!({
            "type": "array",
            "items": {"type": "object", "properties": {"criterion": {"type": "string"}, "weight": {"type": "number"}}, "required": ["criterion"]},
        });
    }
    schema
}

fn base_schema() -> Value {
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

/// Agent descriptions are condensed for the listing: enough to pick an
/// agent; the full text stays in the agent definition file.
const AGENT_SUMMARY_MAX: usize = 120;

/// `description` without parenthesized asides (route/model details),
/// whitespace collapsed, capped at [`AGENT_SUMMARY_MAX`] chars on a word
/// boundary.
fn agent_summary(description: &str) -> String {
    let mut flat = String::with_capacity(description.len());
    let mut depth = 0usize;
    for c in description.chars() {
        match c {
            '(' => depth += 1,
            ')' if depth > 0 => depth -= 1,
            _ if depth == 0 => flat.push(c),
            _ => {}
        }
    }
    let flat = flat
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .replace(" ;", ";")
        .replace(" ,", ",")
        .replace(" .", ".")
        .replace(" :", ":");
    let mut out = String::with_capacity(AGENT_SUMMARY_MAX);
    let mut len = 0usize;
    for word in flat.split(' ') {
        let add = word.chars().count() + usize::from(len > 0);
        if len + add > AGENT_SUMMARY_MAX {
            break;
        }
        if len > 0 {
            out.push(' ');
        }
        out.push_str(word);
        len += add;
    }
    if out.len() < flat.len() {
        // Truncated: end on the last clause boundary when one keeps at least
        // half the budget, so the line never stops mid-phrase.
        if let Some(i) = out
            .rfind(['.', ';'])
            .filter(|&i| i >= AGENT_SUMMARY_MAX / 2)
        {
            out.truncate(i);
        }
    }
    out.trim_end_matches(['.', ';', ',']).to_string()
}

/// Tool description: static header plus the discovered agent list
/// (`~/.omp/agent/agents` + `.omp/agents`), one summary line per agent.
#[must_use]
pub fn description(cwd: &Path) -> String {
    let home = std::env::var_os("HOME").map_or_else(|| PathBuf::from("/"), PathBuf::from);
    let mut agents: Vec<_> = discover_user_agents(&home)
        .agents
        .into_iter()
        .chain(discover_project_agents(cwd).agents)
        .collect();
    agents.sort_by(|a, b| a.fields.name.cmp(&b.fields.name));
    let mut out = String::from(DESCRIPTION_HEADER);
    out.push_str("\n\n# Available Agents\n");
    push_agent_list(&mut out, &agents);
    out
}

/// One summary line per `listed` agent; agents with `listed: false` stay
/// spawnable but are only named on a trailing `Also:` line.
fn push_agent_list(out: &mut String, agents: &[omp_task::agent::AgentDefinition]) {
    let mut unlisted: Vec<&str> = Vec::new();
    for agent in agents {
        if !agent.fields.listed {
            unlisted.push(&agent.fields.name);
            continue;
        }
        out.push_str("- `");
        out.push_str(&agent.fields.name);
        out.push('`');
        if agent_is_read_only(agent.fields.tools.as_ref()) {
            out.push_str(" (read-only)");
        }
        if agent.fields.blocking == Some(true) {
            out.push_str(" (blocking)");
        }
        out.push_str(": ");
        out.push_str(&agent_summary(&agent.fields.description));
        out.push('\n');
    }
    if !unlisted.is_empty() {
        out.push_str("Also: ");
        out.push_str(&unlisted.join(", "));
        out.push_str(" (definitions: ~/.omp/agent/agents/<name>.md)\n");
    }
}

/// Agent definition lookup by name for a spawn: project `.omp/agents`
/// shadows the user-level agents dir (TS discovery precedence).
fn find_agent(cwd: &Path, home: &Path, name: &str) -> Option<omp_task::agent::AgentDefinition> {
    let project = discover_project_agents(cwd);
    if let Some(agent) = project.get_agent(name) {
        return Some(agent.clone());
    }
    let user = discover_user_agents(home);
    user.get_agent(name).cloned()
}

/// The `--model` selector chain an agent definition contributes (TS
/// `resolveEffectiveAgentModelSelection`): its `model` frontmatter — a single
/// selector or an ordered fallback list — comma-joined so the child resolves
/// the entries in order (see [`crate::model::select`]).
fn agent_model_chain(agent: &omp_task::agent::AgentDefinition) -> Option<String> {
    agent
        .fields
        .model
        .as_ref()
        .filter(|chain| !chain.is_empty())
        .map(|chain| chain.join(","))
}

/// Effective `--model` for one spawn: the caller's per-task selector first,
/// then the agent definition's chain; `None` leaves the child on its
/// configured default (`modelRoles.default`), the TS last-resort fallback.
fn effective_model(item_model: Option<&str>, agent_model: Option<&str>) -> Option<String> {
    item_model
        .or(agent_model)
        .filter(|model| !model.trim().is_empty())
        .map(str::to_owned)
}

/// Stamp one spawn with its agent definition's model chain. [`TaskSpec`]
/// carries only the caller's selector, so without this every child ran on the
/// config default regardless of its agent's `model` frontmatter.
pub(crate) fn apply_agent_model(spec: &mut TaskSpec) {
    let home = std::env::var_os("HOME").map(PathBuf::from);
    let agent_model = home
        .as_deref()
        .and_then(|home| find_agent(&spec.cwd, home, &spec.agent))
        .and_then(|agent| agent_model_chain(&agent));
    spec.model = effective_model(spec.model.as_deref(), agent_model.as_deref());
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
    /// SERA rubric `[{criterion, weight}]`, frozen before the child spawns (opt-in).
    #[serde(default)]
    rubric: Option<Value>,
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
    /// Flat-shape SERA rubric (see [`WireTaskItem::rubric`]).
    #[serde(default)]
    rubric: Option<Value>,
}

/// Execute a `task` call: validate, spawn every item, render envelopes.
pub async fn task(env: std::sync::Arc<Env>, args: Value) -> Result<AgentToolResult> {
    let params: WireParams =
        serde_json::from_value(args).map_err(|e| anyhow::anyhow!("invalid task params: {e}"))?;

    let mut specs = Vec::new();
    let mut rubrics: Vec<Option<omp_task::rubric::FrozenRubric>> = Vec::new();
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
        rubrics.push(
            super::rlm_combinators::freeze_wire(params.rubric.as_ref())
                .map_err(|e| anyhow::anyhow!("task: rubric: {e}"))?,
        );
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
            rubrics.push(
                super::rlm_combinators::freeze_wire(item.rubric.as_ref())
                    .map_err(|e| anyhow::anyhow!("task: rubric: {e}"))?,
            );
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
        // The agent definition's `model` chain applies when the item gave no
        // selector (TS `resolveEffectiveAgentModelSelection`).
        apply_agent_model(spec);
    }

    let binary = std::env::current_exe().unwrap_or_else(|_| std::path::PathBuf::from("omt"));
    let runner = TaskRunner::new(binary, effective_max_concurrency(&env.cwd));
    if rubrics.iter().all(Option::is_none) {
        let outputs = runner.spawn_all(specs).await;
        return merge_outputs(&outputs);
    }
    // SERA: rubrics were frozen above, before any child spawned; score afterwards.
    let scorer = super::rlm_combinators::SessionScorer::new(pa_agent::abort::AbortSignal::never());
    let scored = omp_task::rubric::spawn_all_with_rubrics(
        &runner,
        specs.into_iter().zip(rubrics).collect(),
        &scorer,
    )
    .await;
    let (outputs, records): (Vec<_>, Vec<_>) = scored.into_iter().unzip();
    let mut result = merge_outputs(&outputs)?;
    attach_rubric_scores(&mut result, &outputs, &records);
    Ok(result)
}

/// Append the SERA verdicts to the merged result: one `rubric:` line per
/// scored child in the text and a `details.rubric` array.
fn attach_rubric_scores(
    result: &mut AgentToolResult,
    outputs: &[omp_task::runner::RunOutput],
    records: &[Option<Result<omp_task::rubric::RubricRecord, String>>],
) {
    use std::fmt::Write as _;
    let mut lines = String::new();
    let mut rows = Vec::new();
    for (out, rec) in outputs.iter().zip(records) {
        match rec {
            Some(Ok(r)) => {
                let _ = writeln!(
                    lines,
                    "rubric {} [{}]: score={:.3} gate={} per_criterion={:?}",
                    out.id, r.rubric_id, r.score, r.gate, r.per_criterion
                );
                rows.push(json!(r));
            }
            Some(Err(e)) => {
                let _ = writeln!(lines, "rubric {}: scoring failed: {e}", out.id);
                rows.push(json!({"run_id": out.id, "error": e}));
            }
            None => {}
        }
    }
    if let Some(first) = result.content.first_mut() {
        if let pa_agent::types::ToolResultContent::Text(t) = first {
            let _ = write!(t.text, "\n\n{}", lines.trim_end());
        }
    }
    if let Some(details) = result.details.as_object_mut() {
        details.insert("rubric".into(), Value::Array(rows));
    }
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
    fn agent_summary_drops_asides_and_caps_on_words() {
        assert_eq!(
            agent_summary("Cheap scout on zai (GLM-5.1); for extraction.\n  Use when busy."),
            "Cheap scout on zai; for extraction. Use when busy"
        );
        // Over the cap with a clause boundary past half the budget: cut there.
        let clauses = format!("{}; {}", "a".repeat(70), "b ".repeat(40));
        assert_eq!(agent_summary(&clauses), "a".repeat(70));
        // Over the cap: cut on a word boundary, never inside a multi-byte char.
        let long = "слово ".repeat(40);
        let s = agent_summary(&long);
        assert!(s.chars().count() <= AGENT_SUMMARY_MAX);
        assert!(long.starts_with(&s) && s.ends_with("слово"));
    }

    #[test]
    fn unlisted_agents_move_to_also_line() {
        use omp_task::agent::{parse_agent, AgentSource};
        let def = |name: &str, extra: &str| {
            parse_agent(
                format!("{name}.md"),
                &format!("---\nname: {name}\ndescription: Does {name}.\n{extra}---\nbody\n"),
                AgentSource::User,
            )
            .unwrap()
        };
        let agents = [
            def("alpha", ""),
            def("beta", "listed: false\n"),
            def("gamma", "listed: true\n"),
            def("delta", "listed: false\n"),
        ];
        let mut out = String::new();
        push_agent_list(&mut out, &agents);
        assert_eq!(
            out,
            "- `alpha`: Does alpha\n- `gamma`: Does gamma\n\
             Also: beta, delta (definitions: ~/.omp/agent/agents/<name>.md)\n"
        );
        // No unlisted agents: no `Also:` line.
        let mut out = String::new();
        push_agent_list(&mut out, &agents[..1]);
        assert_eq!(out, "- `alpha`: Does alpha\n");
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

    /// `model:` frontmatter reaches the spawn chain in both shapes: a single
    /// selector string and an ordered fallback list.
    #[test]
    fn agent_model_chain_takes_string_and_list_frontmatter() {
        use omp_task::agent::{parse_agent, AgentSource};
        let single = parse_agent(
            "single.md",
            "---\nname: single\ndescription: d\nmodel: zai/glm-5.3-flash\n---\nbody\n",
            AgentSource::User,
        )
        .unwrap();
        assert_eq!(
            agent_model_chain(&single).as_deref(),
            Some("zai/glm-5.3-flash")
        );
        let chain = parse_agent(
            "chain.md",
            "---\nname: chain\ndescription: d\nmodel:\n  - kimi-code/kimi-for-coding\n  - kimi-code/kimi-k2.6\n---\n",
            AgentSource::User,
        )
        .unwrap();
        assert_eq!(
            agent_model_chain(&chain).as_deref(),
            Some("kimi-code/kimi-for-coding,kimi-code/kimi-k2.6")
        );
        let bare = parse_agent(
            "bare.md",
            "---\nname: bare\ndescription: d\n---\n",
            AgentSource::User,
        )
        .unwrap();
        assert_eq!(agent_model_chain(&bare), None);
    }

    /// Precedence: the task item's selector > the agent's `model` chain >
    /// no override (the child's configured default).
    #[test]
    fn model_precedence_task_over_agent_over_default() {
        let agent = Some("anthropic/claude-haiku-5-5");
        assert_eq!(
            effective_model(Some("zai/glm-5.3-flash"), agent).as_deref(),
            Some("zai/glm-5.3-flash")
        );
        assert_eq!(
            effective_model(None, agent).as_deref(),
            Some("anthropic/claude-haiku-5-5")
        );
        assert_eq!(effective_model(None, None), None);
        // An empty selector is no override at all.
        assert_eq!(effective_model(Some("  "), agent), None);
    }

    /// A project `.omp/agents` definition shadows the user-level one, and a
    /// spawn without either keeps its own default.
    #[test]
    fn project_agent_shadows_user_agent() {
        let root = tempfile::tempdir().unwrap();
        let (project, user) = (root.path().join("proj"), root.path().join("home"));
        std::fs::create_dir_all(project.join(".omp/agents")).unwrap();
        std::fs::create_dir_all(user.join(".omp/agent/agents")).unwrap();
        std::fs::write(
            project.join(".omp/agents/dup.md"),
            "---\nname: dup\ndescription: d\nmodel: zai/glm-5.3-flash\n---\n",
        )
        .unwrap();
        std::fs::write(
            user.join(".omp/agent/agents/dup.md"),
            "---\nname: dup\ndescription: d\nmodel: anthropic/claude-haiku-5-5\n---\n",
        )
        .unwrap();
        std::fs::write(
            user.join(".omp/agent/agents/user-only.md"),
            "---\nname: user-only\ndescription: d\nmodel: kimi-code/kimi-k2.6\n---\n",
        )
        .unwrap();
        let chain = |name: &str| {
            find_agent(&project, &user, name).and_then(|agent| agent_model_chain(&agent))
        };
        assert_eq!(chain("dup").as_deref(), Some("zai/glm-5.3-flash"));
        assert_eq!(chain("user-only").as_deref(), Some("kimi-code/kimi-k2.6"));
        assert_eq!(chain("undefined-agent"), None);
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
            result.details["results"]
                .as_array()
                .is_some_and(|r| r.len() == 2),
            "details.results kept: {}",
            result.details
        );
        let ok = merge_outputs(&[run_output("a", RunStatus::Completed)]).expect("ok");
        assert_eq!(ok.is_error, None);
    }

    #[test]
    fn rubric_is_parsed_per_item_and_scores_are_attached() {
        let item = wire_item(json!({
            "task": "x", "solutionSpace": ".",
            "rubric": [{"criterion": "correct", "weight": 3}, {"criterion": "short", "weight": 1}],
        }));
        let frozen = super::super::rlm_combinators::freeze_wire(item.rubric.as_ref())
            .unwrap()
            .unwrap();
        assert!((frozen.criteria[0].weight - 0.75).abs() < 1e-12);
        let none = wire_item(json!({"task": "x", "solutionSpace": "."}));
        assert!(
            super::super::rlm_combinators::freeze_wire(none.rubric.as_ref())
                .unwrap()
                .is_none()
        );
        let bad = wire_item(json!({"task": "x", "solutionSpace": ".", "rubric": []}));
        assert!(super::super::rlm_combinators::freeze_wire(bad.rubric.as_ref()).is_err());

        let out = run_output("agent-1", RunStatus::Completed);
        let rec = omp_task::rubric::RubricRecord::from_verdict(
            "task",
            &frozen,
            "agent-1",
            1,
            1,
            true,
            vec![1.0, 0.0],
            Some(1.0),
        )
        .unwrap();
        let mut result = merge_outputs(std::slice::from_ref(&out)).unwrap();
        attach_rubric_scores(&mut result, std::slice::from_ref(&out), &[Some(Ok(rec))]);
        assert_eq!(result.details["rubric"][0]["score"], 0.75);
        let pa_agent::types::ToolResultContent::Text(t) = &result.content[0] else {
            panic!("text")
        };
        assert!(t.text.contains("score=0.750"), "{}", t.text);
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
