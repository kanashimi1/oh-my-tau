//! The `think` tool (omp `ThinkTool`, hidden): records private scratchpad
//! thoughts while native model reasoning is disabled. The result is a bare
//! separator; the thoughts are never shown to the user.

use std::sync::Arc;

use anyhow::Result;
use pa_agent::abort::AbortSignal;
use pa_agent::types::{AgentTool, AgentToolResult, AgentToolUpdateCallback};
use pa_agent::BoxFut;
use serde_json::{json, Value};

const DESCRIPTION: &str = "private scratchpad; not shown to user";

/// Parameter schema, byte-equal (as JSON) to omp's `thinkSchema`.
#[must_use]
pub fn schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "thoughts": {"type": "string", "description": "private scratchpad; not shown to user"}
        },
        "required": ["thoughts"],
        "additionalProperties": false,
        "description": "private scratchpad; not shown to user"
    })
}

/// The `think` tool.
pub struct ThinkTool {
    schema: Value,
}

/// TS `supportsExternalThinking` (tools/think.ts): whether the model transport
/// can suppress native reasoning while private scratchpad thoughts are active.
/// The port's `Model` has no `thinking` field, so the `thinking.requiresEffort`
/// / `mode: "budget"` legs collapse to the compat/reasoning/api checks.
#[must_use]
pub fn supports_external_thinking(model: &pa_types::ai::Model) -> bool {
    use serde_json::Value;
    // The resolved record (not the model's sparse `compat` overrides): the flags below are
    // resolver outputs.
    let compat = pa_ai::compat::resolve_for_model(model)
        .ok()
        .and_then(|p| p.compat);
    let compat = compat.as_ref();
    let compat_true =
        |key: &str| compat.is_some_and(|raw| raw.get(key).is_some_and(|v| v == &Value::Bool(true)));
    let anthropic = model.api == "anthropic-messages";
    let requires_thinking = anthropic && compat_true("requiresThinkingEnabled");
    if model.reasoning
        && (requires_thinking
            || compat_true("omitReasoningEffort")
            || compat
                .is_some_and(|raw| raw.get("supportsReasoningEffort") == Some(&Value::Bool(false))))
    {
        return false;
    }
    if model.api == "google-generative-ai"
        || model.api == "google-gemini-cli"
        || model.api == "google-vertex"
    {
        return !model.reasoning;
    }
    anthropic
        || model.api == "openai-responses"
        || model.api == "azure-openai-responses"
        || model.api == "openai-codex-responses"
}

/// The hidden `think` tool (omp `HIDDEN_TOOLS.think`).
#[must_use]
pub fn tool() -> Arc<dyn AgentTool> {
    Arc::new(ThinkTool { schema: schema() })
}

impl AgentTool for ThinkTool {
    fn name(&self) -> &'static str {
        "think"
    }
    fn description(&self) -> &str {
        DESCRIPTION
    }
    fn parameters(&self) -> &Value {
        &self.schema
    }
    fn label(&self) -> &'static str {
        "Think"
    }
    fn execute(
        self: Arc<Self>,
        _tool_call_id: String,
        _params: Value,
        _signal: AbortSignal,
        _on_update: AgentToolUpdateCallback,
    ) -> BoxFut<'static, Result<AgentToolResult>> {
        Box::pin(async move { Ok(AgentToolResult::text("------")) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_matches_omp() {
        assert_eq!(
            serde_json::to_value(schema()).unwrap(),
            json!({
                "type": "object",
                "properties": {
                    "thoughts": {"type": "string", "description": "private scratchpad; not shown to user"}
                },
                "required": ["thoughts"],
                "additionalProperties": false,
                "description": "private scratchpad; not shown to user"
            })
        );
    }

    #[test]
    fn execute_returns_separator() {
        let result = tool();
        assert_eq!(result.name(), "think");
        assert_eq!(
            result.description(),
            "private scratchpad; not shown to user"
        );
    }
}
