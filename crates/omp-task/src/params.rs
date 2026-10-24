//! Task tool parameter contract, ported from `task/types.ts` (`ArkType`
//! `taskItemSchema` / `taskSchema`).
//!
//! Wire shape (one spawn):
//! ```json
//! { "name": "optional-id", "agent": "task", "task": "the prompt",
//!   "solutionSpace": "...", "outputSchema": {...}, "schemaMode": "strict" }
//! ```
//! Batch form: `{ "tasks": [ <item>, ... ] }` plus optional shared `context`.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Agent names must be filesystem/URL-handle safe (mirrors
/// `TASK_AGENT_NAME_PATTERN`).
const AGENT_NAME_PATTERN: &str = "^[A-Za-z0-9_-]+$";

/// Whether a child output must satisfy its `outputSchema`.
///
/// Mirrors `schemaMode?: "permissive" | "strict"` on the task item schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SchemaMode {
    /// Default: schema violations are reported but do not fail the run.
    #[default]
    Permissive,
    /// A non-JSON or schema-violating output fails the task item.
    Strict,
}

/// One entry of a task call's `tasks[]` array (or a single non-batch call,
/// which is the same shape without the wrapper).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskItem {
    /// Optional caller-chosen id; doubles as the `agent://<id>` handle.
    pub name: Option<String>,
    /// Agent definition to spawn; defaults to `"task"` (the bundled general agent).
    #[serde(default = "default_agent")]
    pub agent: String,
    /// The prompt / task text for the child.
    pub task: String,
    /// Working-directory hint for the child (solution space), mirroring the
    /// TS required `solutionSpace` string.
    pub solution_space: String,
    /// Model override: a pattern or prioritized list.
    pub model: Option<Value>,
    /// Declared JSON schema for the child's final output.
    pub output_schema: Option<Value>,
    /// How strictly to enforce `output_schema`.
    #[serde(default)]
    pub schema_mode: SchemaMode,
    /// Tool allow-list override for this spawn.
    pub tools: Option<Vec<String>>,
}

fn default_agent() -> String {
    "task".to_string()
}

impl TaskItem {
    /// Validate the item: required fields non-empty, agent name matches the
    /// safe-name pattern (mirrors the TS `ArkType` rules).
    ///
    /// # Errors
    /// Returns a human-readable reason when the item is not spawnable.
    pub fn validate(&self) -> Result<(), String> {
        if self.task.trim().is_empty() {
            return Err("task: `task` must be a non-empty string".to_string());
        }
        if !is_valid_agent_name(&self.agent) {
            return Err(format!(
                "task: `agent` must match /{AGENT_NAME_PATTERN}/, got {:?}",
                self.agent
            ));
        }
        if let Some(name) = &self.name {
            if name.trim().is_empty() {
                return Err("task: `name` must not be blank".to_string());
            }
            if !is_valid_handle_name(name) {
                return Err(format!(
                    "task: `name` must be handle-safe (dot-qualified ids allowed), got {name:?}"
                ));
            }
        }
        Ok(())
    }
}

/// Full task tool call parameters: either a single item or a `tasks[]` batch
/// with shared context.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskParams {
    /// Batch of spawns. Empty/absent means this is a single-item call.
    #[serde(default)]
    pub tasks: Vec<TaskItem>,
    /// Shared context prepended to every batched task's prompt.
    pub context: Option<String>,
    /// Caller-side name for the whole call (envelope id).
    pub name: Option<String>,
    /// Default agent for items that do not set one.
    #[serde(default = "default_agent")]
    pub agent: String,
    /// Call-wide output schema default.
    pub output_schema: Option<Value>,
    /// Single-item form (used when `tasks` is empty).
    #[serde(flatten)]
    pub single: TaskItemOptional,
}

/// Optional single-item fields, accepted alongside the batch form so both
/// `task` call shapes deserialize into one struct.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskItemOptional {
    pub task: Option<String>,
    pub solution_space: Option<String>,
}

/// Whether these params are a batch (`tasks[]`) or a single spawn.
#[must_use]
pub fn is_batch(params: &TaskParams) -> bool {
    !params.tasks.is_empty()
}

/// Check an agent name against the TS safe-name pattern
/// (`TASK_AGENT_NAME_PATTERN`).
#[must_use]
pub fn is_valid_agent_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

/// Caller-chosen run ids double as `agent://<id>` handles, which may be
/// dot-qualified (`Parent.Child-1_x`) per the TS internal-URL convention.
#[must_use]
pub fn is_valid_handle_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.')
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn deserializes_batch_form() {
        let raw = json!({
            "name": "sweep",
            "context": "shared background",
            "agent": "scout-glm",
            "outputSchema": {"type": "object"},
            "tasks": [
                {"name": "a", "task": "do a", "solutionSpace": ".",
                 "schemaMode": "strict"},
                {"task": "do b", "solutionSpace": ".", "agent": "kimi-coder"}
            ]
        });
        let params: TaskParams = serde_json::from_value(raw).unwrap();
        assert!(is_batch(&params));
        assert_eq!(params.tasks.len(), 2);
        assert_eq!(params.tasks[0].schema_mode, SchemaMode::Strict);
        // Items default their own agent to "task"; call-wide agent overrides
        // are applied by the caller composing TaskSpecs (like the TS task
        // tool injects its schema default).
        assert_eq!(params.tasks[0].agent, "task");
        assert_eq!(params.tasks[1].agent, "kimi-coder"); // item override
        assert_eq!(params.output_schema, Some(json!({"type": "object"})));
        params.tasks.iter().for_each(|t| t.validate().unwrap());
    }

    #[test]
    fn default_agent_is_task() {
        let item: TaskItem = serde_json::from_value(json!({
            "task": "x", "solutionSpace": "."
        }))
        .unwrap();
        assert_eq!(item.agent, "task");
        assert_eq!(item.schema_mode, SchemaMode::Permissive);
    }

    #[test]
    fn schema_mode_deserializes() {
        let item: TaskItem = serde_json::from_value(json!({
            "task": "x", "solutionSpace": ".", "schemaMode": "strict"
        }))
        .unwrap();
        assert_eq!(item.schema_mode, SchemaMode::Strict);
        // Unknown modes are rejected (TS ArkType literal is case-sensitive).
        let bad: Result<TaskItem, _> = serde_json::from_value(json!({
            "task": "x", "solutionSpace": ".", "schemaMode": "Strict"
        }));
        assert!(bad.is_err());
    }

    #[test]
    fn validate_rejects_bad_agent_names() {
        let mut item: TaskItem = serde_json::from_value(json!({
            "task": "x", "solutionSpace": ".", "agent": "scout glm"
        }))
        .unwrap();
        assert!(item.validate().is_err());
        item.agent = "scout-glm".to_string();
        item.validate().unwrap();
        let mut blank = item;
        blank.task = "  ".to_string();
        assert!(blank.validate().is_err());
    }

    #[test]
    fn agent_name_pattern() {
        assert!(is_valid_handle_name("Parent.Child-1_x"));
        assert!(!is_valid_agent_name("Parent.Child-1_x")); // dots: handle-only
        assert!(is_valid_agent_name("scout-glm_2"));
        assert!(!is_valid_agent_name("has space"));
        assert!(!is_valid_agent_name("slash/name"));
        assert!(!is_valid_agent_name(""));
    }
}
