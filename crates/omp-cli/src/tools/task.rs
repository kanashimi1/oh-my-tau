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
#[cfg(test)]
use omp_task::runner::RunStatus;
use omp_task::runner::{ChildMode, TaskRunner, TaskSpec};
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
pub const DESCRIPTION_HEADER: &str = r"Spawn `tasks[]` concurrently; the call returns immediately with the children's ids/handles while they keep running in the background.

Subagents run as child processes with bounded concurrency: `task.maxConcurrency` when set to a positive number, else `min(CPU count, 8)`.

# Results
Each child's result auto-delivers to you as a message at your next turn boundary: one combined delivery per settled batch (`<task-results>` for finished children, `<agent-messages>` for what they sent you). Print (`omt -p`) and rpc sessions stay alive while your task children run — a settled child starts a turn with its delivery. Keep working; read results as they land, do not poll.
`outputSchema` parsed payload, even invalid: `agent://<id>` (field `/<field>`, nested `/reports/0/data`); invalid preview inline.

# Waiting
`wait` blocks until the first result or message arrives (like for eval agents/jobs). Nothing of yours running? It errors; NEVER poll while work remains.

# Delegation
Use most specific agent. Prefer one agent to investigate + edit. Omit `agent` only for default (`task`); NEVER specify it.
Shared edits need one integration owner. Set interfaces in `context`.

# Inputs
`name`: CamelCase ≤32, auto-generated if omitted; address agent by name. `outputSchema` overrides agent/session schemas.
`solutionSpace`: describe how open-ended the child's problem is: whether the fix or design is given, or which causes or designs remain open. Volume of work does not widen it; NEVER mention sibling agents or coordination. (`one fix: rename, names given`; `one fix: slice end in paginate`; `single-flight cache load; races easy to miss`; `several retry API shapes; error classes to choose`; `deadlock cause open, no repro`)
`tools`: eval-defined, run in your kernel.
`model`: selector or ordered array overriding the agent's model for this spawn (`provider/model[:level]`, role alias like `@smol`, `@default` = parent's live model); set per `tasks[]` item, never top-level. Omit unless a specific model is needed. Requested model unavailable → stop and report; NEVER substitute another.
`schemaMode`: default permissive warns after retries; strict fails.
Children start blank; large payloads via `local://<path>`, NEVER inline.

# Messaging
Each running child is `agent://<name>` (unnamed: `agent-N`; give items a `name` so siblings can address each other; names must be unused).
`write agent://<name>` / `agent://all` steers running children at their next turn boundary — from this or any later turn. A finished child cannot be messaged.
Inside a child: `write agent://<sibling-name>` messages a sibling (delivered at its next turn boundary); `write agent://parent` messages your parent and lands in its next `<agent-messages>` delivery; an id your own subtree does not know forwards up through your parent (nearest match up the tree wins, so keep names unique per subtree).

# Format
`context`: shared (`# Goal`, `# Contract` interfaces); NEVER repeat per task.
`task`: self-contained (`# Target` files/non-goals, `# Change` steps/APIs, `# Acceptance` observable result).

# Git worktrees
`worktree: true` on an item (batch-wide, or `worktree: true` in the agent definition's frontmatter — item wins over agent wins over batch) runs the child in a fresh `git worktree add` on branch `agent/<id>` from your HEAD. The checkout lives at `<repo>/../<repo-name>-wt/<id>` (override the parent dir with `OMP_TASK_WORKTREE_DIR`) and the child's cwd maps into it. Your cwd must be inside a git repo — otherwise the call fails with a clear error before anything spawns.
The child's prompt carries a footer: commit its work on this branch; every commit ends with trailers `Agent: <id>`, `Envelope: agent://<id>`, `Verified-by: <test command or evidence path>`.
When the child settles, its result shows the branch, its commits (sha + subject), the base..tip diffstat and whether the tree was clean; uncommitted leftovers are auto-committed as `wip(<id>)` so nothing is lost.
Merge a finished branch yourself with `omt agent-merge <id>` (or `--all`; `--squash`, `--discard`, `--require-review`): a merge commit aggregates the branch's trailers, conflicts abort cleanly (files listed, nothing half-merged), and a successful merge removes the worktree + branch. Review verdicts are git notes: `omt review note <commit> --verdict pass|fail|needs-work --by <agent> [--evidence <path>] [--summary <text>]`, read them with `omt review show <commit|range>`. Worktrees are removed only by the merge or by `--discard` — never while the child runs.";

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
        if agent.fields.worktree == Some(true) {
            out.push_str(" (worktree)");
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

/// Effective `worktree` for one spawn: the item's explicit setting first
/// (`Some(false)` disables even an agent-level `worktree: true`), then the
/// agent definition's `worktree` frontmatter, then the batch default.
fn effective_worktree(item: Option<bool>, agent: Option<bool>, batch: Option<bool>) -> bool {
    match item {
        Some(value) => value,
        None => agent.or(batch).unwrap_or(false),
    }
}

/// Stamp one spawn with its agent definition (TS `executor.ts`): the `model`
/// chain, the `tools` allowlist, the body as the child's system-prompt
/// addition and `thinkingLevel`. [`TaskSpec`] carries only the caller's
/// selectors, so without this every child ran as a generic session. Call after
/// [`super::eval_agents::apply_child_policy`], which resets `extra_args`.
pub(crate) fn apply_agent_definition(spec: &mut TaskSpec) {
    let home = std::env::var_os("HOME").map(PathBuf::from);
    let agent = home
        .as_deref()
        .and_then(|home| find_agent(&spec.cwd, home, &spec.agent));
    let agent_model = agent.as_ref().and_then(agent_model_chain);
    spec.model = effective_model(spec.model.as_deref(), agent_model.as_deref());
    if let Some(agent) = agent {
        stamp_agent_definition(spec, &agent);
    }
}

/// The non-model part of [`apply_agent_definition`], pure for tests.
fn stamp_agent_definition(spec: &mut TaskSpec, agent: &omp_task::agent::AgentDefinition) {
    if let Some(agent_tools) = agent.fields.tools.clone() {
        // The agent's list is the ceiling: the item's own request and the
        // parent's `--tools` restriction (already in `spec.tools`) narrow it.
        spec.tools = Some(match spec.tools.take() {
            None => agent_tools,
            Some(current) => {
                // `yield` stays: the parser adds it to every explicit list.
                let kept: Vec<String> = agent_tools
                    .into_iter()
                    .filter(|tool| tool == "yield" || current.contains(tool))
                    .collect();
                if kept.iter().all(|tool| tool == "yield")
                    && !current.contains(&"yield".to_string())
                {
                    vec!["__none__".to_string()]
                } else {
                    kept
                }
            }
        });
    }
    let body = agent.system_prompt.trim();
    if !body.is_empty() {
        spec.extra_args
            .extend(["--append-system-prompt".to_string(), body.to_string()]);
    }
    let level = match agent.fields.thinking_level {
        Some(omp_task::agent::ThinkingLevel::Low) => Some("low"),
        Some(omp_task::agent::ThinkingLevel::Medium) => Some("medium"),
        Some(omp_task::agent::ThinkingLevel::High) => Some("high"),
        Some(omp_task::agent::ThinkingLevel::Max) => Some("max"),
        Some(omp_task::agent::ThinkingLevel::Other(_)) | None => None,
    };
    // A `provider/model:level` selector already pins the level.
    let pinned = spec
        .model
        .as_deref()
        .is_some_and(|model| model.split(',').next().unwrap_or("").contains(':'));
    if let (Some(level), false) = (level, pinned) {
        spec.extra_args
            .extend(["--thinking".to_string(), level.to_string()]);
    }
}

/// Stamp one spawn with its effective `worktree` flag: item explicit >
/// agent definition frontmatter > batch default ([`effective_worktree`]).
/// `item` is the wire item's own `worktree` (`spec.worktree` already carries
/// it via [`TaskSpec::from_item`], so `Some(false)` must win over the agent).
pub(crate) fn apply_agent_worktree(spec: &mut TaskSpec, item: Option<bool>, batch: Option<bool>) {
    if item.is_none() {
        let home = std::env::var_os("HOME").map(PathBuf::from);
        let agent_wt = home
            .as_deref()
            .and_then(|home| find_agent(&spec.cwd, home, &spec.agent))
            .and_then(|agent| agent.fields.worktree);
        spec.worktree = effective_worktree(None, agent_wt, batch);
    } else {
        spec.worktree = effective_worktree(item, None, batch);
    }
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
    /// `worktree: true` (item wins over the agent definition's `worktree`
    /// frontmatter, which wins over the batch default): the child runs in a
    /// fresh `git worktree` on branch `agent/<id>` (see [`omp_task::worktree`]).
    #[serde(default)]
    worktree: Option<bool>,
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
            worktree: self.worktree,
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
    /// Batch-wide (or flat-shape) `worktree` default; an item's explicit
    /// `worktree` and the agent definition's `worktree` frontmatter win.
    #[serde(default)]
    worktree: Option<bool>,
}

/// Execute a `task` call: validate, spawn every item, render envelopes.
#[allow(clippy::too_many_lines)] // linear validate, spawn, render; each stage consumes the last
#[allow(clippy::unused_async)] // tool entry: the registry boxes this as a future
pub async fn task(env: std::sync::Arc<Env>, args: Value) -> Result<AgentToolResult> {
    let params: WireParams =
        serde_json::from_value(args).map_err(|e| anyhow::anyhow!("invalid task params: {e}"))?;

    let mut specs = Vec::new();
    let mut rubrics: Vec<Option<omp_task::rubric::FrozenRubric>> = Vec::new();
    // Per-item `worktree` flags (item > agent frontmatter > batch default);
    // aligned with `specs`. The flat shape has no per-item flag.
    let mut item_worktrees: Vec<Option<bool>> = Vec::new();
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
            worktree: params.worktree,
        };
        item.validate().map_err(|e| anyhow::anyhow!(e))?;
        specs.push(TaskSpec::from_item(
            &item,
            params.context.as_deref(),
            env.cwd.clone(),
        ));
        item_worktrees.push(params.worktree);
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
            item_worktrees.push(item.worktree);
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
    for (spec, item_worktree) in specs.iter_mut().zip(&item_worktrees) {
        // Children inherit the parent's extension/no-extensions flags and
        // tool restriction (EVAL-CHILD-POLICY-NOT-PROPAGATED).
        super::eval_agents::apply_child_policy(spec);
        // The agent definition's `model` chain applies when the item gave no
        // selector (TS `resolveEffectiveAgentModelSelection`).
        apply_agent_definition(spec);
        // `worktree`: item > agent `worktree` frontmatter > batch default.
        apply_agent_worktree(spec, *item_worktree, params.worktree);
    }

    let binary = crate::respawn_exe();
    let runner = TaskRunner::new(binary, effective_max_concurrency(&env.cwd));
    let prompts: Vec<String> = specs.iter().map(|s| s.prompt.clone()).collect();
    // Non-blocking (TS async task semantics): the children keep running; each
    // result is scored (SERA — the rubrics were frozen above, before any spawn)
    // and auto-delivers to the parent at its next turn boundary.
    let spawned = spawn_children(
        &runner,
        specs,
        rubrics,
        prompts,
        // This Env's session token: the children and their deliveries belong
        // to the session whose `task` call spawned them.
        &env.a2a_owner(),
    )?;
    let mut text = format!(
        "Spawned {} task child agent{}; results auto-deliver as messages when they settle:",
        spawned.len(),
        if spawned.len() == 1 { "" } else { "s" }
    );
    for child in &spawned {
        use std::fmt::Write as _;
        let _ = writeln!(text, "\n- agent://{} ({})", child.id, child.agent);
    }
    text.push_str("\nUse `wait` to block for the first result; `write agent://<id>` steers a running child in this or any later turn.");
    let details = json!({
        "handles": spawned.iter().map(|child| json!({
            "kind": "agent",
            "id": child.id,
            "agent": child.agent,
            "agent_url": format!("agent://{}", child.id),
        })).collect::<Vec<_>>(),
    });
    Ok(AgentToolResult {
        content: vec![pa_agent::types::ToolResultContent::text(text.trim_end())],
        details,
        terminate: None,
        is_error: None,
    })
}

/// Process-wide counter behind generated child ids (`agent-N`), so ids stay
/// unique across concurrent `task` calls sharing the agent registry.
static CHILD_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// One child a `task` call started (the immediate-return handle).
#[derive(Debug)]
pub(crate) struct SpawnedChild {
    pub(crate) id: String,
    pub(crate) agent: String,
}

/// One running child and its kill handle.
struct ChildRecord {
    owner: super::a2a::Owner,
    id: String,
    control: omp_task::runner::RunControl,
}

/// Process-wide state of this process's running `task` children. Scoped by
/// session owner ([`super::a2a::Owner`]) so concurrent sessions (unit tests)
/// never see each other's children; production runs one session.
struct AsyncChildren {
    children: Vec<ChildRecord>,
    /// One bus lease per session: a session's bus stays up while any of its
    /// children run.
    leases: Vec<(super::a2a::Owner, super::a2a::BusLease)>,
}

static ASYNC_CHILDREN: parking_lot::Mutex<AsyncChildren> = parking_lot::Mutex::new(AsyncChildren {
    children: Vec::new(),
    leases: Vec::new(),
});

/// Process-wide counter behind spawn-batch ids (one per `spawn_children`
/// call), so `wait` can tell which children were spawned together.
static BATCH_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Which spawn batch each child belongs to. A child's entry outlives its own
/// settle (until the whole batch settles) so its result delivery can still be
/// attributed to a possibly still-running batch.
#[derive(Default)]
struct BatchRegistry {
    /// Child id -> its spawn batch.
    child_batch: std::collections::HashMap<String, u64>,
    /// Batch id -> children of the batch not settled yet.
    remaining: std::collections::HashMap<u64, usize>,
}

static BATCHES: std::sync::LazyLock<parking_lot::Mutex<BatchRegistry>> =
    std::sync::LazyLock::new(|| parking_lot::Mutex::new(BatchRegistry::default()));

/// Claim a fresh batch id for one `spawn_children` call and remember its
/// children: `wait` holds each settled child's result until the child's whole
/// spawn batch settles, so one parent turn gets the batch's results together
/// instead of one extra wait/turn per child.
fn register_batch(child_ids: &[String]) -> u64 {
    let batch = BATCH_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
    let mut reg = BATCHES.lock();
    for id in child_ids {
        reg.child_batch.insert(id.clone(), batch);
    }
    reg.remaining.insert(batch, child_ids.len());
    batch
}

/// One child settled: drop it from its batch; the last settle dissolves the
/// batch so nothing about it is held anymore.
fn note_batch_child_settled(child_id: &str) {
    let mut reg = BATCHES.lock();
    let Some(batch) = reg.child_batch.get(child_id).copied() else {
        return;
    };
    let Some(remaining) = reg.remaining.get_mut(&batch) else {
        return;
    };
    *remaining -= 1;
    if *remaining == 0 {
        // Last settle dissolves the batch: drop it wholesale so nothing
        // about it is held anymore.
        reg.remaining.remove(&batch);
        reg.child_batch.retain(|_, member| *member != batch);
    }
}

/// Whether the task-result delivery `from` belongs to a spawn batch that
/// still has running children (its result is held for one combined delivery).
#[must_use]
pub(crate) fn batch_still_running(from: &str) -> bool {
    let reg = BATCHES.lock();
    reg.child_batch
        .get(from)
        .and_then(|batch| reg.remaining.get(batch))
        .is_some_and(|remaining| *remaining > 0)
}

/// Split drained deliveries into what may be returned now and task results
/// whose spawn batch still has running children (held so one `wait` returns
/// the whole settled batch together). Messages/steering/questions and results
/// of completed or batch-less children always return now.
pub(crate) fn split_batch_ready(
    deliveries: Vec<super::a2a::PendingDelivery>,
) -> (
    Vec<super::a2a::PendingDelivery>,
    Vec<super::a2a::PendingDelivery>,
) {
    deliveries.into_iter().partition(|delivery| {
        delivery.kind != super::a2a::KIND_TASK_RESULT || !batch_still_running(&delivery.from)
    })
}

/// Re-queue held deliveries (relative order preserved) so a later `wait` (or
/// turn boundary) returns them.
pub(crate) fn requeue_deliveries(deliveries: Vec<super::a2a::PendingDelivery>) {
    for delivery in deliveries {
        super::a2a::queue_delivery(delivery);
    }
}

/// How many `task` children are still running (all sessions; tests).
#[cfg(test)]
#[allow(clippy::disallowed_methods)] // tests: throwaway runtimes drive async code
#[must_use]
pub(crate) fn live_children() -> usize {
    ASYNC_CHILDREN.lock().children.len()
}

/// How many `task` children of one session are still running.
#[must_use]
pub(crate) fn live_children_for(owner: &super::a2a::Owner) -> usize {
    ASYNC_CHILDREN
        .lock()
        .children
        .iter()
        .filter(|child| &child.owner == owner)
        .count()
}

/// Whether the session must not settle yet: its children are running or its
/// results are still waiting for a turn boundary.
#[must_use]
pub(crate) fn async_children_active_for(owner: &super::a2a::Owner) -> bool {
    live_children_for(owner) > 0 || super::a2a::has_pending_for(owner)
}

/// Kill every running `task` child of one session (Ctrl-C / abort / session
/// end). The children's group is `SIGKILL`ed by the runner's drive loop or its
/// drop guard, so this is safe even while the runtime is tearing down.
pub(crate) fn kill_async_children_for(owner: &super::a2a::Owner) {
    let (mine, others): (Vec<_>, Vec<_>) = {
        let mut reg = ASYNC_CHILDREN.lock();
        let children = std::mem::take(&mut reg.children);
        children
            .into_iter()
            .partition(|child| &child.owner == owner)
    };
    {
        let mut reg = ASYNC_CHILDREN.lock();
        reg.children = others;
        let mine_left = reg.children.iter().any(|child| &child.owner == owner);
        if !mine_left {
            reg.leases.retain(|(lease_owner, _)| lease_owner != owner);
        }
    }
    for child in &mine {
        let _ = child.control.cancel();
    }
}

/// Park until at least one of the session's deliveries is pending, then drain
/// them all (one combined delivery per settled batch). Returns empty only
/// when the race "children settled and their deliveries were consumed
/// elsewhere" wins — the caller then settles normally.
pub(crate) async fn next_delivery_batch(
    owner: &super::a2a::Owner,
) -> Vec<super::a2a::PendingDelivery> {
    loop {
        let pending = super::a2a::take_deliveries_for(owner);
        if !pending.is_empty() {
            let (ready, held) = split_batch_ready(pending);
            requeue_deliveries(held);
            if !ready.is_empty() {
                return ready;
            }
        }
        if live_children_for(owner) == 0 && !super::a2a::has_pending_for(owner) {
            return Vec::new();
        }
        // Park on the queue with a tick as a missed-notify safety net.
        tokio::select! {
            () = super::a2a::wait_delivery_notify() => {}
            () = tokio::time::sleep(std::time::Duration::from_millis(250)) => {}
        }
    }
}

/// Render one combined delivery batch for a turn-start message / `wait`
/// result ([`super::a2a::render_deliveries`]).
#[must_use]
pub(crate) fn render_delivery_batch(deliveries: &[super::a2a::PendingDelivery]) -> String {
    super::a2a::render_deliveries(deliveries)
}

/// Spawn every spec as a controllable RPC child ([`ChildMode::Rpc`]) and
/// return immediately. Each live child is registered under its id in the
/// shared agent registry (`write agent://<id>` steers it; unregistered once
/// it settles) and gets the parent's a2a bus, so it can message siblings and
/// `agent://parent`. A spec with `worktree: true` first gets a fresh
/// `git worktree` on branch `agent/<id>` from the parent's HEAD: the child's
/// cwd is remapped into the checkout and its prompt carries the commit/
/// trailer footer; when the child settles its result includes the worktree
/// summary (branch, commits, diffstat, clean/dirty, auto-committed
/// `wip(<id>)` leftovers). When a child settles its result is scored against
/// its frozen rubric (SERA) and auto-delivered to the parent at the next
/// turn boundary. Cancelling the session kills the children
/// ([`kill_async_children`]).
#[allow(clippy::too_many_lines)] // linear spawn sequence: validate, prepare worktrees, launch, register
pub(crate) fn spawn_children(
    runner: &TaskRunner,
    mut specs: Vec<TaskSpec>,
    rubrics: Vec<Option<omp_task::rubric::FrozenRubric>>,
    prompts: Vec<String>,
    owner: &super::a2a::Owner,
) -> Result<Vec<SpawnedChild>> {
    use std::sync::atomic::Ordering;

    use super::eval_agents::TaskSlot;

    // Claim every id first: a clash fails the whole call before any spawn.
    let mut slots = Vec::with_capacity(specs.len());
    for spec in &mut specs {
        let slot = if let Some(id) = spec.id.clone() {
            TaskSlot::claim(&id).map_err(|e| anyhow::anyhow!(e))?
        } else {
            loop {
                let id = format!("agent-{}", CHILD_SEQ.fetch_add(1, Ordering::Relaxed) + 1);
                if let Ok(slot) = TaskSlot::claim(&id) {
                    spec.id = Some(id);
                    break slot;
                }
            }
        };
        slots.push(slot);
    }
    // Worktree-per-writing-agent: create every branch + checkout BEFORE any
    // spawn, so a failure (not a repo, branch taken) fails the whole call
    // without half-spawned batches. The worktree (and branch) live until the
    // parent's `omt agent-merge <id>` (or `--discard`) removes them.
    let mut worktrees: Vec<Option<omp_task::worktree::Worktree>> = Vec::with_capacity(specs.len());
    for spec in &mut specs {
        if !spec.worktree {
            worktrees.push(None);
            continue;
        }
        let id = spec.id.clone().unwrap_or_default();
        let parent_cwd = spec.cwd.clone();
        let repo_root = omp_task::worktree::repo_root(&parent_cwd)
            .map_err(|e| anyhow::anyhow!("task child `{id}`: {e}"))?;
        let wt = omp_task::worktree::Worktree::create(&repo_root, &id)
            .map_err(|e| anyhow::anyhow!("task child `{id}`: {e}"))?;
        spec.prompt = format!("{}{}", spec.prompt, wt.prompt_footer());
        spec.cwd = wt.map_cwd(&parent_cwd);
        worktrees.push(Some(wt));
    }
    // Children and their deliveries belong to the session that spawned them
    // (the registered runtime, when there is one).
    // The bus lease is shared by every concurrent call of the session and
    // dropped when its last child settles.
    let call_lease = super::a2a::BusLease::acquire_for(owner)
        .map_err(|e| anyhow::anyhow!("task: cannot start the agent message bus: {e}"))?;
    // Register only after every fallible setup step. Settlement can now
    // only observe a batch whose children are about to be spawned.
    register_batch(
        &specs
            .iter()
            .map(|spec| spec.id.clone().unwrap_or_default())
            .collect::<Vec<_>>(),
    );
    let sock = {
        let mut reg = ASYNC_CHILDREN.lock();
        if let Some((_, lease)) = reg.leases.iter().find(|(l_owner, _)| l_owner == owner) {
            lease.sock().to_path_buf()
        } else {
            let sock = call_lease.sock().to_path_buf();
            reg.leases.push((owner.clone(), call_lease));
            sock
        }
    };
    let mut spawned = Vec::with_capacity(specs.len());
    for ((((mut spec, slot), rubric), prompt), worktree) in specs
        .into_iter()
        .zip(slots)
        .zip(rubrics)
        .zip(prompts)
        .zip(worktrees)
    {
        let id = spec.id.clone().unwrap_or_default();
        spec.env.push((
            super::a2a::SOCK_ENV.to_string(),
            sock.to_string_lossy().into_owned(),
        ));
        spec.env
            .push((super::a2a::SELF_ENV.to_string(), id.clone()));
        let agent = spec.agent.clone();
        let (control, join) = runner.start(spec, ChildMode::Rpc);
        slot.attach(control.clone());
        ASYNC_CHILDREN.lock().children.push(ChildRecord {
            owner: owner.clone(),
            id: id.clone(),
            control,
        });
        spawned.push(SpawnedChild {
            id: id.clone(),
            agent,
        });
        let task_owner = owner.clone();
        tokio::spawn(async move {
            let output = join.await;
            slot.settle(&output);
            // SERA: score when the child settles (rubric frozen before spawn).
            let record = match rubric {
                Some(rubric) => {
                    let scorer = super::rlm_combinators::SessionScorer::new(
                        pa_agent::abort::AbortSignal::never(),
                    );
                    Some(omp_task::rubric::score_output(&rubric, &scorer, &prompt, &output).await)
                }
                None => None,
            };
            // Worktree child: summarize branch/commits/diffstat and
            // auto-commit leftovers (git calls block; run off the executor).
            let worktree_summary = match worktree {
                Some(wt) => tokio::task::spawn_blocking(move || wt.settle_summary())
                    .await
                    .ok(),
                None => None,
            };
            let mut text = result_delivery_text(&output, record.as_ref());
            if let Some(summary) = worktree_summary {
                use std::fmt::Write as _;
                let _ = writeln!(text, "{}", worktree_delivery_line(&summary));
            }
            // Record the settle before queueing: a single-child (or last)
            // batch is already dissolved when its delivery lands, so its
            // result is never held; earlier siblings' results stay held
            // until this last settle.
            note_batch_child_settled(&id);
            super::a2a::queue_delivery(super::a2a::PendingDelivery {
                owner: task_owner.clone(),
                from: id,
                kind: super::a2a::KIND_TASK_RESULT.to_string(),
                text,
            });
            // Last settle of this session tears its bus down.
            let mut reg = ASYNC_CHILDREN.lock();
            reg.children.retain(|child| child.id != output.id);
            if !reg.children.iter().any(|child| child.owner == task_owner) {
                reg.leases
                    .retain(|(lease_owner, _)| lease_owner != &task_owner);
            }
        });
    }
    Ok(spawned)
}

/// One `worktree` line for the settle delivery: compact JSON carrying the
/// branch, base, commits (sha + subject), the base..tip diffstat, and
/// whether the tree was clean when the child settled.
#[must_use]
pub(crate) fn worktree_delivery_line(summary: &omp_task::worktree::WorktreeSummary) -> String {
    let line = serde_json::json!({
        "branch": summary.branch,
        "base": summary.base,
        "path": summary.path,
        "clean": summary.clean,
        "leftovers_auto_committed": summary.leftovers_auto_committed,
        "commits": summary.commits,
        "diffstat": summary.diffstat,
    });
    match serde_json::to_string(&line) {
        Ok(json) => format!("worktree {json}"),
        Err(e) => format!("worktree {{\"error\":\"render failed: {e}\"}}"),
    }
}

/// Render one settled child's `<task-results>` payload: the result envelope
/// plus, when a rubric was declared, its SERA verdict line.
fn result_delivery_text(
    output: &omp_task::runner::RunOutput,
    record: Option<&Result<omp_task::rubric::RubricRecord, String>>,
) -> String {
    let mut text = match serde_json::to_string_pretty(&format_task_result(output)) {
        Ok(envelope) => envelope,
        Err(e) => format!("{{\"error\":\"render failed: {e}\"}}"),
    };
    match record {
        Some(Ok(rec)) => {
            use std::fmt::Write as _;
            let _ = writeln!(
                text,
                "\nrubric {} [{}]: score={:.3} gate={} per_criterion={:?}",
                output.id, rec.rubric_id, rec.score, rec.gate, rec.per_criterion
            );
        }
        Some(Err(error)) => {
            use std::fmt::Write as _;
            let _ = writeln!(text, "\nrubric {}: scoring failed: {error}", output.id);
        }
        None => {}
    }
    text
}

#[cfg(test)]
#[allow(clippy::disallowed_methods)] // tests: throwaway runtimes drive async code
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

    /// The definition's `tools`, body and `thinkingLevel` reach the child's
    /// argv; the agent list is a ceiling the item/parent restriction narrows.
    #[test]
    fn agent_definition_stamps_tools_prompt_and_thinking() {
        use omp_task::agent::{parse_agent, AgentSource};
        let agent = parse_agent(
            "bb.md",
            "---\nname: bb\ndescription: d\ntools: [read, bash, mcp__blockbench_bb_preview]\nthinkingLevel: high\n---\nCheck every edit with bb_preview.\n",
            AgentSource::User,
        )
        .unwrap();
        let spec = |model: Option<&str>, tools: Option<Vec<&str>>| TaskSpec {
            id: None,
            agent: "bb".into(),
            prompt: "go".into(),
            cwd: PathBuf::from("/tmp"),
            output_schema: None,
            schema_mode: SchemaMode::Permissive,
            model: model.map(str::to_string),
            tools: tools.map(|t| t.into_iter().map(str::to_string).collect()),
            depth: 1,
            extra_args: vec!["--no-extensions".into()],
            env: Vec::new(),
            worktree: false,
        };

        let mut plain = spec(Some("zai/glm-5.3-flash"), None);
        stamp_agent_definition(&mut plain, &agent);
        assert_eq!(
            plain.tools.as_deref(),
            Some(&["read", "bash", "mcp__blockbench_bb_preview", "yield"].map(String::from)[..])
        );
        assert_eq!(
            plain.extra_args,
            [
                "--no-extensions",
                "--append-system-prompt",
                "Check every edit with bb_preview.",
                "--thinking",
                "high"
            ]
        );

        let mut narrowed = spec(Some("zai/glm-5.3-flash:low"), Some(vec!["read", "edit"]));
        stamp_agent_definition(&mut narrowed, &agent);
        assert_eq!(
            narrowed.tools,
            Some(vec!["read".to_string(), "yield".to_string()])
        );
        assert!(!narrowed.extra_args.contains(&"--thinking".to_string()));

        let mut disjoint = spec(None, Some(vec!["edit"]));
        stamp_agent_definition(&mut disjoint, &agent);
        assert_eq!(disjoint.tools, Some(vec!["__none__".to_string()]));
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
    fn result_delivery_text_renders_envelopes_and_failures() {
        let completed = result_delivery_text(&run_output("a", RunStatus::Completed), None);
        assert!(completed.contains("\"id\": \"a\""), "{completed}");
        assert!(
            completed.contains("\"status\": \"completed\""),
            "{completed}"
        );
        let failed = run_output("b", RunStatus::Failed);
        let failed = result_delivery_text(&failed, None);
        assert!(failed.contains("\"status\": \"failed\""), "{failed}");
        // A scored rubric appends the SERA verdict line.
        let frozen = omp_task::rubric::FrozenRubric::freeze(&json!([
            {"criterion": "correct", "weight": 3},
            {"criterion": "short", "weight": 1},
        ]))
        .unwrap();
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
        let scored =
            result_delivery_text(&run_output("agent-1", RunStatus::Completed), Some(&Ok(rec)));
        assert!(scored.contains("score=0.750"), "{scored}");
        let scorer_failed = result_delivery_text(
            &run_output("agent-1", RunStatus::Completed),
            Some(&Err("scorer down".to_string())),
        );
        assert!(
            scorer_failed.contains("rubric agent-1: scoring failed: scorer down"),
            "{scorer_failed}"
        );
    }

    #[test]
    fn rubric_is_parsed_per_item_and_verdict_rides_the_delivery() {
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
        // The verdict rides the settle-time delivery as a `rubric:` line.
        let text = result_delivery_text(&out, Some(&Ok(rec)));
        assert!(text.contains("score=0.750"), "{text}");
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

    /// Fake `--mode rpc` child that speaks the real a2a envelope wire over the
    /// bus it inherits: `e2e-a` messages its sibling and the parent,
    /// `agt-*` (eval `agent()` spawn) does the same to prove one channel for
    /// every spawner, `e2e-b` waits for `FAKE_STEERS` steers and answers with
    /// everything it received, sorted.
    const FAKE_A2A_CHILD: &str = r#"#!/usr/bin/env python3
import json, os, socket, sys, uuid
me, sock = os.environ["OMP_A2A_SELF"], os.environ["OMP_A2A_SOCK"]
steers_wanted = int(os.environ.get("FAKE_STEERS", "2"))
def send(to, msg):
    s = socket.socket(socket.AF_UNIX); s.connect(sock)
    env = {"message_id": str(uuid.uuid4()), "sender": me, "recipient": to,
           "kind": "agent.message", "payload": msg}
    s.sendall((json.dumps(env) + "\n").encode())
    reply = s.makefile().readline(); s.close(); return json.loads(reply)
def out(o): print(json.dumps(o), flush=True)
stdin = sys.stdin
stdin.readline()
out({"type": "response", "command": "prompt", "success": True, "id": "omp-task-prompt"})
if me == "e2e-b":
    got = []
    while len(got) < steers_wanted:
        frame = json.loads(stdin.readline())
        if frame.get("type") == "steer":
            got.append(frame["message"])
    answer = "|".join(sorted(got))
else:
    target = "e2e-b"
    answer = json.dumps([send(target, "hello-from-" + me)["ok"], send("parent", "report-from-" + me)["ok"]])
out({"type": "prompt_result", "status": "completed", "sessionSettled": True})
for _ in range(2):
    q = json.loads(stdin.readline())
    if q["type"] == "get_last_assistant_text":
        out({"type": "response", "command": "get_last_assistant_text", "success": True, "data": {"text": answer}})
    else:
        out({"type": "response", "command": "get_session_stats", "success": True, "data": {"tokens": {"output": 1}}})
stdin.readline()
"#;

    #[tokio::test]
    async fn spawn_children_results_auto_deliver() {
        let _serial = super::super::a2a::tests::LOCK.lock().await;
        super::super::a2a::tests::without_parent_bus(async {
            use std::os::unix::fs::PermissionsExt;
            let dir = tempfile::tempdir().unwrap();
            let bin = dir.path().join("fake-a2a-child");
            std::fs::write(&bin, FAKE_A2A_CHILD).unwrap();
            std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755)).unwrap();
            let runner = TaskRunner::new(&bin, 4);
            let spec = TaskSpec {
                id: Some("e2e-b".to_string()),
                agent: "task".into(),
                prompt: "go".into(),
                cwd: dir.path().to_path_buf(),
                output_schema: None,
                schema_mode: SchemaMode::Permissive,
                model: None,
                tools: None,
                depth: 1,
                extra_args: Vec::new(),
                env: vec![("FAKE_STEERS".to_string(), "1".to_string())],
                worktree: false,
            };
            let _ = super::super::a2a::take_deliveries();

            // The call returns immediately while the child is still running:
            // the immediate return names the child and the child is still
            // steerable afterwards.
            let spawned = spawn_children(
                &runner,
                vec![spec],
                vec![None],
                vec!["go".to_string()],
                &None,
            )
            .unwrap();
            assert_eq!(spawned.len(), 1);
            assert_eq!(spawned[0].id, "e2e-b");
            assert!(
                live_children() >= 1,
                "child still running after the call returned"
            );

            // A `write agent://e2e-b` from a later turn steers the running child.
            let receipt = super::super::eval_agents::deliver_message("e2e-b", "from-parent")
                .await
                .unwrap();
            assert_eq!(receipt, "Delivered to e2e-b.");

            // The settle auto-delivers the result (poll the unscoped queue:
            // a concurrent test runtime may re-register an owner mid-test, so
            // the delivery is not guaranteed to be unowned).
            let mut result: Option<super::super::a2a::PendingDelivery> = None;
            for _ in 0..50 {
                for delivery in super::super::a2a::take_deliveries() {
                    if delivery.kind == super::super::a2a::KIND_TASK_RESULT
                        && delivery.from == "e2e-b"
                    {
                        result = Some(delivery);
                    }
                }
                if result.is_some() {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            }
            let result = result.expect("e2e-b result delivered");
            assert!(
                result.text.contains("from-parent"),
                "steer text in the child's answer: {}",
                result.text
            );
            let rendered = render_delivery_batch(std::slice::from_ref(&result));
            assert!(rendered.starts_with("<task-results>"), "{rendered}");

            // Settled: unregistered, duplicate ids still rejected (two
            // same-named specs in ONE call fail the claim before any spawn).
            let gone = super::super::eval_agents::deliver_from_peer("e2e-b", "late", "x").await;
            assert!(gone.is_err());
            let dup_spec = || TaskSpec {
                id: Some("dup-1".to_string()),
                agent: "task".into(),
                prompt: "go".into(),
                cwd: dir.path().to_path_buf(),
                output_schema: None,
                schema_mode: SchemaMode::Permissive,
                model: None,
                tools: None,
                depth: 1,
                extra_args: Vec::new(),
                env: Vec::new(),
                worktree: false,
            };
            let dup = spawn_children(
                &runner,
                vec![dup_spec(), dup_spec()],
                vec![None, None],
                vec!["go".to_string(), "go".to_string()],
                &None,
            );
            assert!(dup.unwrap_err().to_string().contains("already in use"));
            assert_eq!(live_children(), 0, "failed claim spawns nothing");
        })
        .await;
    }

    #[tokio::test]
    async fn spawn_children_child_to_parent_message_auto_delivers() {
        let _serial = super::super::a2a::tests::LOCK.lock().await;
        super::super::a2a::tests::without_parent_bus(async {
            use std::os::unix::fs::PermissionsExt;
            let dir = tempfile::tempdir().unwrap();
            let bin = dir.path().join("fake-a2a-child");
            std::fs::write(&bin, FAKE_A2A_CHILD).unwrap();
            std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755)).unwrap();
            let runner = TaskRunner::new(&bin, 4);
            let _ = super::super::a2a::take_deliveries();
            let spawned = spawn_children(
                &runner,
                vec![TaskSpec {
                    id: Some("e2e-a".to_string()),
                    agent: "task".into(),
                    prompt: "go".into(),
                    cwd: dir.path().to_path_buf(),
                    output_schema: None,
                    schema_mode: SchemaMode::Permissive,
                    model: None,
                    tools: None,
                    depth: 1,
                    extra_args: Vec::new(),
                    env: Vec::new(),
                    worktree: false,
                }],
                vec![None],
                vec!["go".to_string()],
                &None,
            )
            .unwrap();
            assert_eq!(spawned[0].id, "e2e-a");
            // The child settles; its `agent://parent` message and its result both
            // deliver (message first — it is sent before the answer — then the
            // result; the combined batch renders both blocks).
            let mut saw_message = false;
            let mut saw_result = false;
            for _ in 0..40 {
                for delivery in super::super::a2a::take_deliveries() {
                    if delivery.kind == super::super::a2a::kind::MESSAGE {
                        assert_eq!(delivery.from, "e2e-a");
                        assert_eq!(delivery.text, "report-from-e2e-a");
                        saw_message = true;
                    }
                    if delivery.kind == super::super::a2a::KIND_TASK_RESULT {
                        saw_result = true;
                    }
                }
                if saw_message && saw_result {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            }
            assert!(saw_message, "agent://parent message delivered");
            assert!(saw_result, "child result delivered");
            // Combined rendering keeps both blocks in one delivery text.
            let combined = render_delivery_batch(&[
                super::super::a2a::PendingDelivery {
                    owner: None,
                    from: "e2e-a".into(),
                    kind: super::super::a2a::kind::MESSAGE.to_string(),
                    text: "report-from-e2e-a".into(),
                },
                super::super::a2a::PendingDelivery {
                    owner: None,
                    from: "e2e-a".into(),
                    kind: super::super::a2a::KIND_TASK_RESULT.to_string(),
                    text: "{\"id\": \"e2e-a\"}".into(),
                },
            ]);
            assert!(
                combined.contains("<agent-messages>\nfrom agent://e2e-a: report-from-e2e-a"),
                "{combined}"
            );
            assert!(combined.contains("<task-results>"), "{combined}");
            assert_eq!(live_children(), 0);
        })
        .await;
    }

    /// One channel for every spawner: an eval `agent()` child spawned through
    /// the real subprocess runner messages its `task` sibling and the parent
    /// over the same bus.
    #[tokio::test]
    async fn agent_child_talks_to_task_sibling_over_the_same_bus() {
        let _serial = super::super::a2a::tests::LOCK.lock().await;
        super::super::a2a::tests::without_parent_bus(async {
            use std::os::unix::fs::PermissionsExt;
            use std::sync::LazyLock;
            static ENV: LazyLock<std::sync::Arc<Env>> =
                LazyLock::new(|| std::sync::Arc::new(Env::new(std::path::Path::new("."), false)));
            let dir = tempfile::tempdir().unwrap();
            let bin = dir.path().join("fake-a2a-child");
            std::fs::write(&bin, FAKE_A2A_CHILD).unwrap();
            std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755)).unwrap();
            let launcher: std::sync::Arc<dyn super::super::eval_agents::Launcher> =
                std::sync::Arc::new(super::super::eval_agents::SubprocessLauncher::for_test(
                    &bin,
                ));
            let _ = super::super::a2a::take_deliveries();

            // The `task` sibling waits for exactly one steer (from the agent child).
            let spawned = spawn_children(
                &TaskRunner::new(&bin, 4),
                vec![TaskSpec {
                    id: Some("e2e-b".to_string()),
                    agent: "task".into(),
                    prompt: "go".into(),
                    cwd: dir.path().to_path_buf(),
                    output_schema: None,
                    schema_mode: SchemaMode::Permissive,
                    model: None,
                    tools: None,
                    depth: 1,
                    extra_args: Vec::new(),
                    env: vec![("FAKE_STEERS".to_string(), "1".to_string())],
                    worktree: false,
                }],
                vec![None],
                vec!["go".to_string()],
                &None,
            )
            .unwrap();
            assert_eq!(spawned[0].id, "e2e-b");

            // The eval `agent()` child (fake binary, real env hand-off).
            let ctx = super::super::eval_bridge::OpCtx {
                cwd: dir.path().to_path_buf(),
                signal: pa_agent::abort::AbortSignal::never(),
                on_update: None,
                tool_call_id: "test-agent-child".to_string(),
            };
            let handle = super::super::eval_agents::run_agent_with(
                &ENV,
                &json!({"prompt": "ping the task sibling", "agent": "task"}),
                &ctx,
                &launcher,
                &None,
            )
            .unwrap();
            let agt_id = handle["id"].as_str().unwrap().to_string();
            assert!(agt_id.starts_with("agt-"), "{handle}");

            // Collect the deliveries until both children settled. The eval
            // agent() child's *result* is a handle read (wait()/status()),
            // not a bus delivery; what must deliver over the shared bus is
            // its `agent://parent` message, and the task sibling's result
            // must show the steer it received from the agent() child.
            let mut saw_sibling_result = false;
            let mut saw_parent_message = false;
            for _ in 0..40 {
                for delivery in super::super::a2a::take_deliveries() {
                    if delivery.kind == super::super::a2a::KIND_TASK_RESULT
                        && delivery.from == "e2e-b"
                    {
                        assert!(
                            delivery.text.contains("hello-from-"),
                            "sibling got the agent child's message: {}",
                            delivery.text
                        );
                        saw_sibling_result = true;
                    }
                    if delivery.kind == super::super::a2a::kind::MESSAGE
                        && delivery.from == agt_id
                        && delivery.text.contains("report-from-")
                    {
                        saw_parent_message = true;
                    }
                }
                if saw_sibling_result && saw_parent_message {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            }
            assert!(saw_sibling_result, "task sibling result delivered");
            assert!(saw_parent_message, "agent() child parent message delivered");
        })
        .await;
    }

    /// Fake `--mode rpc` child for the worktree e2e: it works INSIDE its cwd
    /// (which must be the git worktree it was spawned with): commits a file
    /// with the `Agent:`/`Envelope:`/`Verified-by:` trailers, leaves another
    /// file uncommitted, and settles.
    const FAKE_WORKTREE_CHILD: &str = r#"#!/usr/bin/env python3
import json, os, subprocess, sys

def git(*args):
    subprocess.run(["git", "-c", "user.name=fake-child", "-c",
                    "user.email=fake@child"] + list(args),
                   cwd=os.getcwd(), check=True, capture_output=True)

stdin = sys.stdin
stdin.readline()
print(json.dumps({"type": "response", "command": "prompt", "success": True, "id": "omp-task-prompt"}), flush=True)
me = os.environ["OMP_A2A_SELF"]
with open("work.txt", "w") as f:
    f.write("work by " + me + "\n")
git("add", "-A")
git("commit", "--no-verify", "-m",
    "child: the work\n\nAgent: " + me + "\nEnvelope: agent://" + me +
    "\nVerified-by: fake-worktree-child")
with open("leftover.txt", "w") as f:
    f.write("uncommitted\n")
print(json.dumps({"type": "prompt_result", "status": "completed", "sessionSettled": True}), flush=True)
for _ in range(2):
    q = json.loads(stdin.readline())
    if q["type"] == "get_last_assistant_text":
        print(json.dumps({"type": "response", "command": "get_last_assistant_text", "success": True, "data": {"text": "worktree-child-answer"}}), flush=True)
    else:
        print(json.dumps({"type": "response", "command": "get_session_stats", "success": True, "data": {"tokens": {"output": 1}}}), flush=True)
stdin.readline()
"#;

    fn git_in(dir: &Path, args: &[&str]) -> String {
        let out = std::process::Command::new("git")
            .args(args)
            .current_dir(dir)
            .output()
            .unwrap_or_else(|e| panic!("spawn git: {e}"));
        assert!(
            out.status.success(),
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8_lossy(&out.stdout).into_owned()
    }

    /// A `worktree: true` child gets a fresh git worktree on `agent/<id>`
    /// from the parent's HEAD, its cwd mapped into it, and a prompt footer
    /// with the commit/trailer contract; when it settles, the delivery shows
    /// branch, commits (sha + subject), the diffstat and clean/dirty, with
    /// leftovers auto-committed as `wip(<id>)`. The worktree survives until
    /// the parent's `agent-merge`.
    #[tokio::test]
    #[allow(clippy::too_many_lines)] // one end-to-end scenario: worktree, child run, settle summary
    async fn worktree_child_runs_in_its_branch_and_reports_a_settle_summary() {
        let _serial = super::super::a2a::tests::LOCK.lock().await;
        super::super::a2a::tests::without_parent_bus(async {
            use std::os::unix::fs::PermissionsExt;
            let dir = tempfile::tempdir().unwrap();
            // The parent's repo: one commit on main.
            let repo = dir.path().join("repo");
            std::fs::create_dir(&repo).unwrap();
            git_in(&repo, &["init", "-b", "main"]);
            git_in(&repo, &["config", "user.name", "omt-test"]);
            git_in(&repo, &["config", "user.email", "omt-test@example.com"]);
            std::fs::write(repo.join("file.txt"), "one\n").unwrap();
            git_in(&repo, &["add", "-A"]);
            git_in(&repo, &["commit", "-m", "initial"]);
            let bin = dir.path().join("fake-worktree-child");
            std::fs::write(&bin, FAKE_WORKTREE_CHILD).unwrap();
            std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755)).unwrap();
            let _ = super::super::a2a::take_deliveries();

            let spawned = spawn_children(
                &TaskRunner::new(&bin, 4),
                vec![TaskSpec {
                    id: Some("wt-1".to_string()),
                    agent: "task".into(),
                    prompt: "go".into(),
                    cwd: repo.clone(),
                    output_schema: None,
                    schema_mode: SchemaMode::Permissive,
                    model: None,
                    tools: None,
                    depth: 1,
                    extra_args: Vec::new(),
                    env: Vec::new(),
                    worktree: true,
                }],
                vec![None],
                vec!["go".to_string()],
                &None,
            )
            .unwrap();
            assert_eq!(spawned[0].id, "wt-1");

            // Wait for the settle delivery.
            let mut delivery_text: Option<String> = None;
            for _ in 0..50 {
                for delivery in super::super::a2a::take_deliveries() {
                    if delivery.kind == super::super::a2a::KIND_TASK_RESULT
                        && delivery.from == "wt-1"
                    {
                        delivery_text = Some(delivery.text);
                    }
                }
                if delivery_text.is_some() {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            }
            let text = delivery_text.expect("worktree child result delivered");
            assert!(text.contains("worktree-child-answer"), "{text}");
            // The summary line: branch, base, commits, diffstat, dirty + wip.
            assert!(text.contains("\"branch\":\"agent/wt-1\""), "{text}");
            assert!(text.contains("\"clean\":false"), "{text}");
            assert!(text.contains("\"leftovers_auto_committed\":true"), "{text}");
            assert!(
                text.contains("child: the work")
                    && text.contains("wip(wt-1): uncommitted leftovers"),
                "child + wip commits listed: {text}"
            );
            assert!(
                text.contains("2 files changed"),
                "diffstat covers work.txt + leftover.txt: {text}"
            );

            // The branch exists in the parent's repo with the trailer commit,
            // and the worktree still exists (removed only by agent-merge).
            let subjects = git_in(&repo, &["log", "--format=%s", "agent/wt-1"]);
            assert!(subjects.contains("child: the work"), "{subjects}");
            assert!(
                subjects.contains("wip(wt-1): uncommitted leftovers"),
                "{subjects}"
            );
            // The child's own commit carries the trailers (the tip is the
            // trailer-less wip commit).
            let trailers = git_in(&repo, &["log", "--format=%(trailers)", "agent/wt-1"]);
            assert!(trailers.contains("Agent: wt-1"), "{trailers}");
            assert!(trailers.contains("Envelope: agent://wt-1"), "{trailers}");
            let wt_path = dir.path().join("repo-wt").join("wt-1");
            assert!(
                wt_path.is_dir(),
                "worktree kept until the merge: {wt_path:?}"
            );

            // The parent's merge helper merges it and cleans up.
            let outcome = omp_task::worktree::merge_agent(
                &repo,
                "wt-1",
                &omp_task::worktree::MergeOptions::default(),
            )
            .unwrap();
            let commit = match outcome.status {
                omp_task::worktree::MergeStatus::Merged { commit } => commit,
                other => panic!("expected a clean merge, got {other:?}"),
            };
            assert_eq!(commit, git_in(&repo, &["rev-parse", "HEAD"]).trim());
            let body = git_in(&repo, &["log", "-1", "--format=%B"]);
            assert!(body.contains("Agent: wt-1"), "{body}");
            assert!(!wt_path.exists(), "worktree removed after the merge");
            assert!(git_in(&repo, &["branch", "--list", "agent/wt-1"])
                .trim()
                .is_empty());
            assert_eq!(
                std::fs::read_to_string(repo.join("leftover.txt")).unwrap(),
                "uncommitted\n",
                "the wip commit carried the leftover into main"
            );
            assert_eq!(live_children(), 0);
        })
        .await;
    }

    /// `worktree: true` with the parent cwd outside any git repo fails the
    /// whole call with a clear error before anything spawns.
    #[tokio::test]
    async fn worktree_true_outside_a_git_repo_fails_before_any_spawn() {
        let _serial = super::super::a2a::tests::LOCK.lock().await;
        super::super::a2a::tests::without_parent_bus(async {
            let dir = tempfile::tempdir().unwrap();
            let bin = dir.path().join("no-such-child");
            let err = spawn_children(
                &TaskRunner::new(&bin, 4),
                vec![TaskSpec {
                    id: Some("wt-x".to_string()),
                    agent: "task".into(),
                    prompt: "go".into(),
                    cwd: dir.path().to_path_buf(),
                    output_schema: None,
                    schema_mode: SchemaMode::Permissive,
                    model: None,
                    tools: None,
                    depth: 1,
                    extra_args: Vec::new(),
                    env: Vec::new(),
                    worktree: true,
                }],
                vec![None],
                vec!["go".to_string()],
                &None,
            )
            .unwrap_err();
            assert!(
                err.to_string().contains("not inside a git repository"),
                "clear error, got: {err}"
            );
            assert_eq!(live_children(), 0, "nothing spawned");
        })
        .await;
    }
}
