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

/// The hidden `think` tool (omp `HIDDEN_TOOLS.think`).
#[must_use]
pub fn tool() -> Arc<dyn AgentTool> {
    Arc::new(ThinkTool {
        schema: schema(),
    })
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
        assert_eq!(result.description(), "private scratchpad; not shown to user");
    }
}
