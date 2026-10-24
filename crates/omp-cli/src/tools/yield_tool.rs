//! The `yield` tool (omp `YieldTool`, hidden): subagent result submission.
//!
//! Subset of omp's tool that matches this port's runtime surface: no
//! caller-supplied output schema (structured-output tasks don't exist here) and
//! no workpool, so the parameters schema is omp's no-schema loose form and the
//! schema-retry ladder is inert. The `data`/`error`/`type` contract, the
//! mutual-exclusion checks and the empty-result abort ladder mirror
//! `tools/yield.ts`.

use std::sync::Arc;

use anyhow::{anyhow, Result};
use pa_agent::abort::AbortSignal;
use pa_agent::types::{AgentTool, AgentToolResult, AgentToolUpdateCallback};
use pa_agent::BoxFut;
use serde_json::{json, Map, Value};

const DESCRIPTION: &str = "Submit subagent output: `{ data: <your output> }` for success, `{ error: \"message\" }` for failure. Never both; never a bare payload outside `data`.\n\nOmit `type` for the usual single terminal structured result. Pass `type: [\"section\"]` to submit an incremental, non-terminal section that accumulates.\nPass `type: \"result\"` to finalize; when `data` is omitted, your last assistant turn becomes the raw final result.";

const YIELD_FORMAT_HINT: &str =
    "Submit success as {\"data\":<your output>} or failure as {\"error\":\"message\"}.";
/// Max consecutive untyped empty-result submissions before the child is failed
/// explicitly (omp `MAX_EMPTY_RESULT_RETRIES`).
const MAX_EMPTY_RESULT_RETRIES: u32 = 3;

/// Parameter schema, byte-equal (as JSON) to omp's `buildYieldParameters` with
/// no caller output schema (`looseRecordSchema` `data`).
#[must_use]
pub fn schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "description": "submit data or error",
        "properties": {
            "type": {
                "anyOf": [
                    {"type": "string"},
                    {"type": "array", "minItems": 1, "items": {"type": "string"}}
                ],
                "description": "Optional result type. A non-empty string array is incremental; a string is terminal."
            },
            "data": {
                "type": "object",
                "additionalProperties": true,
                "description": "Structured JSON output (no schema specified)"
            },
            "error": {"type": "string", "description": "Failure reason; mutually exclusive with data"}
        },
        "required": []
    })
}

/// omp `parseYieldError`: strict-mode providers send omitted optionals as
/// `null`; `""` next to a valid payload also means "no error".
fn parse_yield_error(value: Option<&Value>) -> Result<Option<String>> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) if s.is_empty() => Ok(None),
        Some(Value::String(s)) => Ok(Some(s.clone())),
        Some(_) => Err(anyhow!("error must be a string")),
    }
}

/// omp `parseYieldType`: `null` (strict-mode omitted optional) is absent.
fn parse_yield_type(value: Option<&Value>) -> Result<Option<Value>> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(_)) => Ok(Some(value.unwrap().clone())),
        Some(Value::Array(items)) if !items.is_empty() && items.iter().all(Value::is_string) => {
            Ok(Some(value.unwrap().clone()))
        }
        Some(_) => Err(anyhow!(
            "type must be a string or non-empty array of strings"
        )),
    }
}

/// The `yield` tool.
pub struct YieldTool {
    schema: Value,
    /// Consecutive untyped empty-result submissions (omp `#emptyResultFailures`).
    empty_result_failures: std::sync::Mutex<u32>,
}

/// The hidden `yield` tool (omp `HIDDEN_TOOLS.yield`).
#[must_use]
pub fn tool() -> Arc<dyn AgentTool> {
    Arc::new(YieldTool {
        schema: schema(),
        empty_result_failures: std::sync::Mutex::new(0),
    })
}

impl AgentTool for YieldTool {
    fn name(&self) -> &'static str {
        "yield"
    }
    fn description(&self) -> &str {
        DESCRIPTION
    }
    fn parameters(&self) -> &Value {
        &self.schema
    }
    fn label(&self) -> &'static str {
        "Submit Result"
    }
    fn execute(
        self: Arc<Self>,
        _tool_call_id: String,
        params: Value,
        _signal: AbortSignal,
        _on_update: AgentToolUpdateCallback,
    ) -> BoxFut<'static, Result<AgentToolResult>> {
        Box::pin(async move { self.run(params) })
    }
}

impl YieldTool {
    #[allow(clippy::too_many_lines)] // mirrors tools/yield.ts execute()
    fn run(self: Arc<Self>, params: Value) -> Result<AgentToolResult> {
        let Value::Object(raw) = params else {
            return Err(anyhow!("yield arguments must be an object"));
        };
        let raw = &raw;
        // Strict-mode providers send omitted optionals as `null`; treat as absent.
        let data: Option<&Value> = match raw.get("data") {
            Some(Value::Null) | None => None,
            Some(v) => Some(v),
        };
        let error_message = parse_yield_error(raw.get("error"))?;
        let yield_type = parse_yield_type(raw.get("type"))?;
        let use_last_turn = error_message.is_none() && data.is_none() && yield_type.is_some();

        if error_message.is_some() && data.is_some() {
            return Err(anyhow!("yield cannot contain both data and error"));
        }
        if error_message.is_none() && data.is_none() && yield_type.is_none() {
            let mut failures = self
                .empty_result_failures
                .lock()
                .expect("yield retry counter poisoned");
            *failures += 1;
            if *failures > MAX_EMPTY_RESULT_RETRIES {
                let attempt_count = *failures;
                *failures = 0;
                let error = format!(
                    "yield result stayed empty after {attempt_count} consecutive attempt(s); aborting child instead of retrying forever. {YIELD_FORMAT_HINT}"
                );
                return Ok(aborted_result(&error, yield_type));
            }
            let remaining = MAX_EMPTY_RESULT_RETRIES - *failures;
            return Err(anyhow!(
                "yield must contain either `data` or `error`. {YIELD_FORMAT_HINT} Empty untyped result retries remaining before abort: {remaining}."
            ));
        }
        *self
            .empty_result_failures
            .lock()
            .expect("yield retry counter poisoned") = 0;

        if let Some(error) = error_message {
            Ok(aborted_result(&error, yield_type))
        } else {
            let mut details = Map::new();
            if let Some(data) = data {
                details.insert("data".into(), data.clone());
            }
            details.insert("status".into(), json!("success"));
            details.insert("error".into(), Value::Null);
            if let Some(yield_type) = yield_type {
                details.insert("type".into(), yield_type);
            }
            if use_last_turn {
                // No last-assistant-turn accessor on this runtime: omp skips
                // its empty-last-turn guard the same way when the session
                // exposes no resolver, and finalization reads the turn.
                details.insert("useLastTurn".into(), json!(true));
            }
            Ok(AgentToolResult {
                content: vec![pa_agent::types::ToolResultContent::text(
                    "Result submitted.",
                )],
                details: Value::Object(details),
                terminate: None,
                is_error: None,
            })
        }
    }
}

/// omp's aborted-yield result frame (`status: "aborted"` details).
fn aborted_result(error: &str, yield_type: Option<Value>) -> AgentToolResult {
    let mut details = Map::new();
    details.insert("status".into(), json!("aborted"));
    details.insert("error".into(), json!(error));
    if let Some(yield_type) = yield_type {
        details.insert("type".into(), yield_type);
    }
    AgentToolResult {
        content: vec![pa_agent::types::ToolResultContent::text(format!(
            "Task aborted: {error}"
        ))],
        details: Value::Object(details),
        terminate: None,
        is_error: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(params: Value) -> Result<AgentToolResult> {
        YieldTool::run(
            Arc::new(YieldTool {
                schema: schema(),
                empty_result_failures: std::sync::Mutex::new(0),
            }),
            params,
        )
    }

    #[test]
    fn schema_matches_omp() {
        assert_eq!(
            schema(),
            json!({
                "type": "object",
                "additionalProperties": false,
                "description": "submit data or error",
                "properties": {
                    "type": {
                        "anyOf": [
                            {"type": "string"},
                            {"type": "array", "minItems": 1, "items": {"type": "string"}}
                        ],
                        "description": "Optional result type. A non-empty string array is incremental; a string is terminal."
                    },
                    "data": {
                        "type": "object",
                        "additionalProperties": true,
                        "description": "Structured JSON output (no schema specified)"
                    },
                    "error": {"type": "string", "description": "Failure reason; mutually exclusive with data"}
                },
                "required": []
            })
        );
    }

    #[test]
    fn success_frame() {
        let r = run(json!({"data": {"n": 1}, "type": "result"})).unwrap();
        assert_eq!(
            r.content[0],
            pa_agent::types::ToolResultContent::text("Result submitted.")
        );
        assert_eq!(r.details["status"], "success");
        assert_eq!(r.details["data"]["n"], 1);
        assert_eq!(r.details["type"], "result");
    }

    #[test]
    fn abort_frame_and_mutual_exclusion() {
        let r = run(json!({"error": "boom"})).unwrap();
        assert_eq!(
            r.content[0],
            pa_agent::types::ToolResultContent::text("Task aborted: boom")
        );
        assert_eq!(r.details["status"], "aborted");
        let e = run(json!({"data": {}, "error": "x"})).unwrap_err();
        assert_eq!(e.to_string(), "yield cannot contain both data and error");
    }

    #[test]
    fn null_optionals_and_type_validation() {
        // Strict-mode providers send omitted optionals as null.
        let r = run(json!({"data": {"ok": true}, "type": null, "error": null})).unwrap();
        assert_eq!(r.details["status"], "success");
        let e = run(json!({"type": 3})).unwrap_err();
        assert_eq!(
            e.to_string(),
            "type must be a string or non-empty array of strings"
        );
        let e = run(json!({"error": 3})).unwrap_err();
        assert_eq!(e.to_string(), "error must be a string");
    }

    #[test]
    fn empty_result_ladder() {
        let tool = Arc::new(YieldTool {
            schema: schema(),
            empty_result_failures: std::sync::Mutex::new(0),
        });
        for remaining in (0..3).rev() {
            let e = YieldTool::run(tool.clone(), json!({})).unwrap_err();
            assert!(
                e.to_string()
                    .ends_with(&format!("before abort: {remaining}.")),
                "{e}"
            );
        }
        let r = YieldTool::run(tool, json!({})).unwrap();
        assert_eq!(r.details["status"], "aborted");
    }
}
