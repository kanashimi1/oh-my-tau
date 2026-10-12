//! `ask`: ask the user a question during execution. Port of omp `tools/ask.ts`.
//!
//! omp shows questions through the UI's rich ask dialog. In RPC UI mode that is
//! one `extension_ui_request` frame with `method: "ask"` carrying the questions,
//! answered by an `extension_ui_response` with `answers` (or `cancelled`, with
//! `timedOut` when the deadline passed). The frame transport lives in the host
//! and is injected through [`AskUi`]. Without a UI, the tool fails with omp's
//! headless error (`Ask tool requires interactive mode`); omp's `-p` mode
//! never registers the tool, so this is only reached by a misconfigured host.
//!
//! Not ported yet: the select/editor fallback used when the host never opts in
//! to the rich dialog (`set_ask_dialog`), images in answers, and `chat` results.

use std::sync::Arc;

use anyhow::{anyhow, Result};
use pa_agent::abort::AbortSignal;
use pa_agent::types::{AgentTool, AgentToolResult, AgentToolUpdateCallback, ToolResultContent};
use pa_agent::BoxFut;
use serde::Deserialize;
use serde_json::{json, Map, Value};

const DESCRIPTION: &str = r#"Ask only for materially different tradeoffs the user must decide. Default: act using code/config/docs/history and conventions. Several viable choices: pick conservative/standard, proceed, state choice.

<instruction>
- Batch related questions; 2–5 distinct options each; short labels, tradeoffs in `description`.
- `recommended` auto-adds " (Recommended)"; `multi: true` permits multiple selections.
- NEVER supply "Other": UI adds "Other (type your own)". Clarifying custom input? Answer first; re-ask unresolved questions.
</instruction>"#;

/// Labels the runtime adds to every question; options may not reuse them.
const RESERVED_LABELS: [&str; 3] = ["Other (type your own)", "Chat about this", "Next →"];

/// Host transport for the rich ask dialog.
pub trait AskUi: Send + Sync {
    /// Present `questions` (the frame's `questions` array) with an optional
    /// deadline in milliseconds, and resolve with the answer frame: the
    /// `extension_ui_response` object (`answers`, or `cancelled` + `timedOut`).
    /// `Ok(None)` means the host could not deliver a response at all.
    fn ask(
        &self,
        questions: Value,
        timeout_ms: Option<u64>,
    ) -> BoxFut<'static, Result<Option<Value>>>;
}

#[derive(Debug, Clone, Deserialize)]
struct OptionItem {
    label: String,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    preview: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct QuestionItem {
    id: String,
    question: String,
    #[serde(default)]
    header: Option<String>,
    options: Vec<OptionItem>,
    #[serde(default)]
    multi: Option<bool>,
    #[serde(default)]
    recommended: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct Params {
    questions: Vec<QuestionItem>,
}

/// One answered question (omp `QuestionResult`).
#[derive(Debug, Clone, PartialEq)]
struct QuestionResult {
    id: String,
    question: String,
    options: Vec<String>,
    multi: bool,
    selected: Vec<String>,
    custom: Option<String>,
    timed_out: bool,
}

/// The `ask` tool.
pub struct AskTool {
    schema: Value,
    ui: Option<Arc<dyn AskUi>>,
}

/// The `ask` tool without a UI: every call fails with the headless error.
#[must_use]
pub fn tool() -> Arc<dyn AgentTool> {
    Arc::new(AskTool {
        schema: schema(),
        ui: None,
    })
}

/// The `ask` tool wired to a host dialog transport.
#[must_use]
#[allow(dead_code)] // host-transport variant, wired when a dialog UI backend is present
pub fn tool_with_ui(ui: Arc<dyn AskUi>) -> Arc<dyn AgentTool> {
    Arc::new(AskTool {
        schema: schema(),
        ui: Some(ui),
    })
}

fn parse_params(raw: Value) -> Result<Params> {
    let params: Params =
        serde_json::from_value(raw).map_err(|e| anyhow!("Invalid ask arguments: {e}"))?;
    if params.questions.is_empty() {
        return Err(anyhow!(
            "Invalid ask arguments: questions must contain at least 1 item"
        ));
    }
    for question in &params.questions {
        if let Some(reserved) = question
            .options
            .iter()
            .find(|o| RESERVED_LABELS.contains(&o.label.as_str()))
        {
            return Err(anyhow!(
                "Invalid ask arguments: option labels must not collide with reserved runtime labels: {}",
                reserved.label
            ));
        }
    }
    Ok(params)
}

/// Text-level validation omp reports as a non-error result (`Error: ...`).
fn validate_questions(questions: &[QuestionItem]) -> Option<String> {
    let mut seen_ids: Vec<&str> = Vec::new();
    for question in questions {
        if seen_ids.contains(&question.id.as_str()) {
            return Some(format!(
                "Error: question ids must be unique: {}",
                json!(question.id)
            ));
        }
        seen_ids.push(&question.id);
        let mut seen_labels: Vec<&str> = Vec::new();
        for option in &question.options {
            if seen_labels.contains(&option.label.as_str()) {
                return Some(format!(
                    "Error: option labels must be unique within a question: {}",
                    json!(option.label)
                ));
            }
            seen_labels.push(&option.label);
        }
    }
    None
}

fn trimmed_nonempty(value: Option<&String>) -> Option<&str> {
    value.map(|v| v.trim()).filter(|v| !v.is_empty())
}

/// One question in the `ask` frame (omp `ExtensionAskDialogQuestion`).
fn frame_question(question: &QuestionItem) -> Value {
    let options: Vec<Value> = question
        .options
        .iter()
        .map(|option| {
            let mut obj = Map::new();
            obj.insert("label".into(), json!(option.label));
            if let Some(description) = trimmed_nonempty(option.description.as_ref()) {
                obj.insert("description".into(), json!(description));
            }
            if let Some(preview) = trimmed_nonempty(option.preview.as_ref()) {
                obj.insert("preview".into(), json!(preview));
            }
            Value::Object(obj)
        })
        .collect();
    let mut obj = Map::new();
    obj.insert("id".into(), json!(question.id));
    obj.insert("question".into(), json!(question.question));
    if let Some(header) = trimmed_nonempty(question.header.as_ref()) {
        obj.insert("header".into(), json!(header));
    }
    obj.insert("options".into(), Value::Array(options));
    if let Some(multi) = question.multi {
        obj.insert("multi".into(), json!(multi));
    }
    if let Some(recommended) = question.recommended {
        obj.insert("recommended".into(), json!(recommended));
    }
    Value::Object(obj)
}

/// omp `timedOutAskDialogResult`: each question takes its recommended option, else the first.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)] // index math on omp's clamped recommended option, values are small and range-validated
fn timed_out_results(questions: &[QuestionItem]) -> Vec<QuestionResult> {
    questions
        .iter()
        .map(|q| {
            let labels: Vec<String> = q.options.iter().map(|o| o.label.clone()).collect();
            let fallback = if labels.is_empty() {
                None
            } else {
                let last = labels.len() - 1;
                let index = q.recommended.unwrap_or(0.0).max(0.0).min(last as f64) as usize;
                labels.get(index).cloned()
            };
            QuestionResult {
                id: q.id.clone(),
                question: q.question.clone(),
                options: labels,
                multi: q.multi.unwrap_or(false),
                selected: fallback.into_iter().collect(),
                custom: None,
                timed_out: true,
            }
        })
        .collect()
}

/// omp `parseAskDialogResponse`: validates each answer against its question.
fn parse_answers(questions: &[QuestionItem], answers: &Value) -> Result<Vec<QuestionResult>> {
    let answers = answers
        .as_array()
        .filter(|a| a.len() == questions.len())
        .ok_or_else(|| {
            anyhow!(
                "Ask dialog response must carry {} answers in question order",
                questions.len()
            )
        })?;
    questions
        .iter()
        .zip(answers)
        .enumerate()
        .map(|(index, (question, answer))| {
            let id_ok = answer.get("id").and_then(Value::as_str) == Some(question.id.as_str());
            if !answer.is_object() || !id_ok {
                return Err(anyhow!(
                    "Ask dialog answer {index} must have id {}",
                    json!(question.id)
                ));
            }
            let labels: Vec<&str> = question.options.iter().map(|o| o.label.as_str()).collect();
            let multi = question.multi.unwrap_or(false);
            let selected_raw = answer
                .get("selectedOptions")
                .and_then(Value::as_array)
                .ok_or_else(|| {
                    anyhow!(
                        "Ask dialog answer {} must carry a selectedOptions array",
                        json!(question.id)
                    )
                })?;
            let mut selected: Vec<String> = Vec::new();
            for label in selected_raw {
                let known = label.as_str().filter(|l| labels.contains(l));
                let Some(label) = known else {
                    return Err(anyhow!(
                        "Ask dialog answer {} selected unknown option {}",
                        json!(question.id),
                        label
                    ));
                };
                if selected.iter().any(|s| s == label) {
                    return Err(anyhow!(
                        "Ask dialog answer {} selected {} twice",
                        json!(question.id),
                        json!(label)
                    ));
                }
                selected.push(label.to_string());
            }
            let custom_raw = answer.get("customInput").filter(|v| !v.is_null());
            let custom = match custom_raw {
                None => None,
                Some(Value::String(s)) => Some(s.trim().to_string()).filter(|s| !s.is_empty()),
                Some(_) => {
                    return Err(anyhow!(
                        "Ask dialog answer {} customInput must be a string",
                        json!(question.id)
                    ))
                }
            };
            if !multi && (selected.len() > 1 || (!selected.is_empty() && custom.is_some())) {
                return Err(anyhow!(
                    "Ask dialog answer {} is single-select but carries more than one answer",
                    json!(question.id)
                ));
            }
            Ok(QuestionResult {
                id: question.id.clone(),
                question: question.question.clone(),
                options: labels
                    .iter()
                    .map(std::string::ToString::to_string)
                    .collect(),
                multi,
                selected,
                custom,
                timed_out: false,
            })
        })
        .collect()
}

/// Interprets the host's answer frame. `Ok(None)` = cancelled by the user.
fn results_from_response(
    questions: &[QuestionItem],
    response: Option<Value>,
) -> Result<Option<Vec<QuestionResult>>> {
    let Some(frame) = response else {
        return Ok(None);
    };
    if frame.get("cancelled").and_then(Value::as_bool) == Some(true) {
        if frame.get("timedOut").and_then(Value::as_bool) == Some(true) {
            return Ok(Some(timed_out_results(questions)));
        }
        return Ok(None);
    }
    let answers = frame.get("answers").cloned().unwrap_or(Value::Null);
    parse_answers(questions, &answers).map(Some)
}

fn format_question_result(result: &QuestionResult) -> String {
    if let Some(custom) = &result.custom {
        return format!("{}: \"{custom}\"", result.id);
    }
    if !result.selected.is_empty() {
        let suffix = if result.timed_out {
            " (auto-selected after timeout)"
        } else {
            ""
        };
        return if result.multi {
            format!("{}: [{}]{suffix}", result.id, result.selected.join(", "))
        } else {
            format!("{}: {}{suffix}", result.id, result.selected[0])
        };
    }
    if result.multi {
        format!("{}: []", result.id)
    } else {
        format!("{}: (cancelled)", result.id)
    }
}

fn indent_lines(text: &str) -> String {
    text.split('\n')
        .map(|line| format!("  {line}"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Text for a single question (omp `formatSingleQuestionResponse`).
fn format_single_response(result: &QuestionResult) -> String {
    let mut parts: Vec<String> = Vec::new();
    if !result.selected.is_empty() {
        let text = if result.multi {
            format!("User selected: {}", result.selected.join(", "))
        } else {
            format!("User selected: {}", result.selected[0])
        };
        parts.push(if result.timed_out {
            format!("{text} (auto-selected after timeout)")
        } else {
            text
        });
    }
    if let Some(custom) = &result.custom {
        parts.push(if custom.contains('\n') {
            format!("User provided custom input:\n{}", indent_lines(custom))
        } else {
            format!("User provided custom input: {custom}")
        });
    }
    if !parts.is_empty() {
        return parts.join("\n");
    }
    if result.multi {
        "User did not select any options".to_string()
    } else {
        "User cancelled the selection".to_string()
    }
}

fn result_json(result: &QuestionResult) -> Value {
    let mut obj = Map::new();
    obj.insert("id".into(), json!(result.id));
    obj.insert("question".into(), json!(result.question));
    obj.insert("options".into(), json!(result.options));
    obj.insert("multi".into(), json!(result.multi));
    obj.insert("selectedOptions".into(), json!(result.selected));
    if let Some(custom) = &result.custom {
        obj.insert("customInput".into(), json!(custom));
    }
    if result.timed_out {
        obj.insert("timedOut".into(), json!(true));
    }
    Value::Object(obj)
}

fn cancelled() -> anyhow::Error {
    anyhow!("Ask tool was cancelled by the user")
}

impl AskTool {
    async fn run(&self, raw: Value) -> Result<AgentToolResult> {
        let Some(ui) = self.ui.clone() else {
            return Err(anyhow!("Ask tool requires interactive mode"));
        };
        let params = parse_params(raw)?;
        if let Some(text) = validate_questions(&params.questions) {
            return Ok(AgentToolResult::text(text));
        }
        let frame: Vec<Value> = params.questions.iter().map(frame_question).collect();
        let response = ui.ask(Value::Array(frame), None).await?;
        let Some(results) = results_from_response(&params.questions, response)? else {
            return Err(cancelled());
        };
        if results.len() != params.questions.len() {
            return Err(anyhow!(
                "Ask dialog returned a result count that does not match the requested questions"
            ));
        }
        if let [single] = results.as_slice() {
            if !single.timed_out
                && !single.multi
                && single.selected.is_empty()
                && single.custom.is_none()
            {
                return Err(cancelled());
            }
            let text = format_single_response(single);
            let details = json!({
                "question": single.question,
                "options": single.options,
                "multi": single.multi,
                "selectedOptions": single.selected,
                "customInput": single.custom,
                "timedOut": single.timed_out.then_some(true),
            });
            return Ok(result_with_text(text, details));
        }
        let text = format!(
            "User answers:\n{}",
            results
                .iter()
                .map(format_question_result)
                .collect::<Vec<_>>()
                .join("\n")
        );
        let details = json!({"results": results.iter().map(result_json).collect::<Vec<_>>()});
        Ok(result_with_text(text, details))
    }
}

fn result_with_text(text: String, details: Value) -> AgentToolResult {
    AgentToolResult {
        content: vec![ToolResultContent::text(text)],
        details: strip_nulls(details),
        terminate: None,
    }
}

/// Drop `null` members, as JSON serialization of omitted TS `undefined` fields does.
fn strip_nulls(value: Value) -> Value {
    match value {
        Value::Object(map) => {
            Value::Object(map.into_iter().filter(|(_, v)| !v.is_null()).collect())
        }
        other => other,
    }
}

impl AgentTool for AskTool {
    fn name(&self) -> &'static str {
        "ask"
    }
    fn description(&self) -> &str {
        DESCRIPTION
    }
    fn parameters(&self) -> &Value {
        &self.schema
    }
    fn label(&self) -> &'static str {
        "Ask"
    }
    fn execute(
        self: Arc<Self>,
        _tool_call_id: String,
        params: Value,
        _signal: AbortSignal,
        _on_update: AgentToolUpdateCallback,
    ) -> BoxFut<'static, Result<AgentToolResult>> {
        Box::pin(async move { self.run(params).await })
    }
}

/// Parameter schema, byte-equal (as JSON) to omp's `askSchema`.
#[must_use]
pub fn schema() -> Value {
    json!({"type":"object","properties":{
        "questions":{"type":"array","minItems":1,"items":{"type":"object","properties":{
            "id":{"type":"string"},
            "question":{"type":"string"},
            "header":{"type":"string","description":"display chip"},
            "options":{"type":"array","items":{"type":"object","properties":{
                "label":{"type":"string"},
                "description":{"type":"string"},
                "preview":{"type":"string","description":"rich preview"}
            },"required":["label"]}},
            "multi":{"type":"boolean"},
            "recommended":{"type":"number","description":"0-based default index"}
        },"required":["id","question","options"]}}
    },"required":["questions"]})
}
#[cfg(test)]
mod tests {
    use parking_lot::Mutex;

    use super::*;

    /// Host transport double: records the frame and answers with a fixed response.
    struct FakeUi {
        response: Option<Value>,
        frames: Mutex<Vec<(Value, Option<u64>)>>,
    }

    impl AskUi for FakeUi {
        fn ask(
            &self,
            questions: Value,
            timeout_ms: Option<u64>,
        ) -> BoxFut<'static, Result<Option<Value>>> {
            self.frames.lock().push((questions, timeout_ms));
            let response = self.response.clone();
            Box::pin(async move { Ok(response) })
        }
    }

    fn fake(response: Option<Value>) -> (Arc<FakeUi>, AskTool) {
        let ui = Arc::new(FakeUi {
            response,
            frames: Mutex::new(Vec::new()),
        });
        let tool = AskTool {
            schema: schema(),
            ui: Some(ui.clone()),
        };
        (ui, tool)
    }

    fn storage_args() -> Value {
        json!({"questions": [{
            "id": "storage", "question": "Database?",
            "options": [{"label": "SQLite"}, {"label": "Postgres", "description": "  shared  "}],
            "recommended": 0
        }]})
    }

    #[tokio::test]
    async fn headless_without_ui_fails_like_omp() {
        let err = AskTool {
            schema: schema(),
            ui: None,
        }
        .run(storage_args())
        .await
        .unwrap_err();
        assert_eq!(err.to_string(), "Ask tool requires interactive mode");
    }

    #[tokio::test]
    async fn single_answer_sends_ask_frame_and_formats_result() {
        let (ui, tool) = fake(Some(
            json!({"answers": [{"id": "storage", "selectedOptions": ["SQLite"]}]}),
        ));
        let result = tool.run(storage_args()).await.unwrap();
        assert_eq!(
            result.content,
            vec![ToolResultContent::text("User selected: SQLite")]
        );
        assert_eq!(result.details["selectedOptions"], json!(["SQLite"]));
        assert!(result.details.get("timedOut").is_none());
        let frames = ui.frames.lock();
        assert_eq!(frames.len(), 1);
        assert_eq!(
            frames[0].0,
            json!([{
                "id": "storage", "question": "Database?",
                "options": [{"label": "SQLite"}, {"label": "Postgres", "description": "shared"}],
                "recommended": 0.0
            }])
        );
    }

    #[tokio::test]
    async fn cancel_and_empty_single_select_are_cancellations() {
        let (_, tool) = fake(Some(json!({"cancelled": true})));
        assert_eq!(
            tool.run(storage_args()).await.unwrap_err().to_string(),
            "Ask tool was cancelled by the user"
        );
        let (_, tool) = fake(Some(
            json!({"answers": [{"id": "storage", "selectedOptions": []}]}),
        ));
        assert_eq!(
            tool.run(storage_args()).await.unwrap_err().to_string(),
            "Ask tool was cancelled by the user"
        );
    }

    #[tokio::test]
    async fn timeout_auto_selects_recommended_option() {
        let (_, tool) = fake(Some(json!({"cancelled": true, "timedOut": true})));
        let mut args = storage_args();
        args["questions"][0]["recommended"] = json!(1);
        let result = tool.run(args).await.unwrap();
        assert_eq!(
            result.content,
            vec![ToolResultContent::text(
                "User selected: Postgres (auto-selected after timeout)"
            )]
        );
        assert_eq!(result.details["timedOut"], json!(true));
    }

    #[tokio::test]
    async fn multi_question_answers_use_result_lines() {
        let args = json!({"questions": [
            {"id": "a", "question": "A?", "options": [{"label": "x"}, {"label": "y"}]},
            {"id": "b", "question": "B?", "multi": true, "options": [{"label": "p"}, {"label": "q"}]}
        ]});
        let response = json!({"answers": [
            {"id": "a", "selectedOptions": [], "customInput": "  mine  "},
            {"id": "b", "selectedOptions": ["p", "q"]}
        ]});
        let (_, tool) = fake(Some(response));
        let result = tool.run(args).await.unwrap();
        assert_eq!(
            result.content,
            vec![ToolResultContent::text(
                "User answers:\na: \"mine\"\nb: [p, q]"
            )]
        );
        assert_eq!(result.details["results"][0]["customInput"], json!("mine"));
    }

    #[tokio::test]
    async fn invalid_answers_fail_closed() {
        let (_, tool) = fake(Some(
            json!({"answers": [{"id": "storage", "selectedOptions": ["Nope"]}]}),
        ));
        let err = tool.run(storage_args()).await.unwrap_err().to_string();
        assert_eq!(
            err,
            "Ask dialog answer \"storage\" selected unknown option \"Nope\""
        );
        let (_, tool) = fake(Some(json!({"answers": []})));
        assert!(tool.run(storage_args()).await.is_err());
    }

    #[tokio::test]
    async fn reserved_labels_and_duplicate_ids_are_rejected() {
        let (_, tool) = fake(None);
        let reserved =
            json!({"questions": [{"id": "a", "question": "?", "options": [{"label": "Next →"}]}]});
        assert!(tool.run(reserved).await.is_err());
        let dup = json!({"questions": [
            {"id": "a", "question": "?", "options": [{"label": "x"}]},
            {"id": "a", "question": "?", "options": [{"label": "y"}]}
        ]});
        let result = tool.run(dup).await.unwrap();
        assert_eq!(
            result.content,
            vec![ToolResultContent::text(
                "Error: question ids must be unique: \"a\""
            )]
        );
    }
}
