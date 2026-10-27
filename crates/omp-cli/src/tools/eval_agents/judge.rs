//! `judge()` op: question parsing and single judgments.

use super::{
    json, AbortSignal, AgentDir, JudgeRequest, JudgmentCache, Map, OpCtx, Value, REQUEST_SLOTS,
};

/// TS `parseState`: a non-empty string or any JSON value (object/array).
pub(super) fn parse_state(value: &Value) -> Result<Value, String> {
    match value {
        Value::String(text) if text.is_empty() => {
            Err("judge() received invalid arguments: state must not be empty".to_string())
        }
        Value::String(_) | Value::Array(_) | Value::Object(_) => Ok(value.clone()),
        _ => Err("judge() received invalid arguments: state must be a string, a JSON object, or a JSON array".to_string()),
    }
}

pub(super) fn parse_instructions(id: &str, value: Option<&Value>) -> Result<String, String> {
    match value.and_then(Value::as_str) {
        Some(text) if !text.is_empty() => Ok(text.to_string()),
        _ => Err(format!(
            "judge() received invalid arguments: question \"{id}\" needs non-empty string instructions"
        )),
    }
}

/// TS `parseQuestions`: keyed questions of type `choice` | `bool` | `score`;
/// the cell-facing `bool` maps to the library's `noul` wire type.
pub(super) fn parse_questions(value: Option<&Value>) -> Result<Map<String, Value>, String> {
    let Some(questions) = value.and_then(Value::as_object) else {
        return Err(
            "judge() received invalid arguments: questions must be an object keyed by question id"
                .to_string(),
        );
    };
    if questions.is_empty() {
        return Err(
            "judge() received invalid arguments: questions must contain at least one question"
                .to_string(),
        );
    }
    let mut parsed = Map::new();
    for (id, question) in questions {
        let Some(question) = question.as_object() else {
            return Err(format!(
                "judge() received invalid arguments: question \"{id}\" must be an object"
            ));
        };
        let instructions = parse_instructions(id, question.get("instructions"))?;
        let invalid = |detail: &str| format!("judge() received invalid arguments: {detail}");
        let wire = match question.get("type").and_then(Value::as_str) {
            Some("choice") => {
                let Some(criteria) = question.get("criteria").and_then(Value::as_object) else {
                    return Err(invalid(&format!(
                        "choice question \"{id}\" needs criteria: {{ label: rubric | null }}"
                    )));
                };
                if criteria.len() < 2 {
                    return Err(invalid(&format!(
                        "choice question \"{id}\" needs at least two options"
                    )));
                }
                for (label, rubric) in criteria {
                    if !rubric.is_null() && !rubric.is_string() {
                        return Err(invalid(&format!(
                            "choice question \"{id}\" criteria \"{label}\" must be a string or null"
                        )));
                    }
                }
                json!({"type": "choice", "instructions": instructions, "criteria": question.get("criteria").cloned().unwrap_or(Value::Null)})
            }
            Some("bool") => {
                // TS `parseBool`: no `criteria` key when absent (the System One
                // endpoint rejects `criteria: null`), else only the string sides.
                let mut wire = json!({"type": "noul", "instructions": instructions});
                if let Some(criteria) = question.get("criteria") {
                    let Some(criteria) = criteria.as_object() else {
                        return Err(invalid(&format!(
                            "bool question \"{id}\" criteria must be {{ true?: string, false?: string }}"
                        )));
                    };
                    let mut sides = Map::new();
                    for side in ["true", "false"] {
                        if let Some(description) = criteria.get(side) {
                            if !description.is_string() {
                                return Err(invalid(&format!(
                                    "bool question \"{id}\" criteria.{side} must be a string"
                                )));
                            }
                            sides.insert(side.to_string(), description.clone());
                        }
                    }
                    wire["criteria"] = Value::Object(sides);
                }
                wire
            }
            Some("score") => {
                let Some(levels) = question.get("criteria").and_then(Value::as_array) else {
                    return Err(invalid(&format!(
                        "score question \"{id}\" needs criteria: [lowest, ..., highest] level descriptions"
                    )));
                };
                if levels.len() < 2 || !levels.iter().all(Value::is_string) {
                    return Err(invalid(&format!(
                        "score question \"{id}\" needs at least two levels"
                    )));
                }
                json!({"type": "score", "instructions": instructions, "criteria": question.get("criteria").cloned().unwrap_or(Value::Null)})
            }
            _ => {
                return Err(invalid(&format!(
                    "question \"{id}\" type must be \"choice\", \"bool\", or \"score\""
                )));
            }
        };
        parsed.insert(id.clone(), wire);
    }
    Ok(parsed)
}

/// TS `toEvalJudgmentResult`: the cell sees `bool` where the library answers
/// `noul`; plus the `provider/model` backend label.
pub(super) fn judgment_result_value(result: &omp_providers::JudgeResult) -> Value {
    let mut answers = Map::new();
    for (id, answer) in &result.answers {
        let mut answer = answer.clone();
        if answer.get("type").and_then(Value::as_str) == Some("noul") {
            let mut as_bool = Map::new();
            as_bool.insert("type".to_string(), Value::String("bool".to_string()));
            as_bool.insert(
                "bool".to_string(),
                answer.get("noul").cloned().unwrap_or(Value::Null),
            );
            answer = Value::Object(as_bool);
        }
        answers.insert(id.clone(), answer);
    }
    json!({"answers": Value::Object(answers), "model": format!("{}/{}", result.provider, result.model)})
}

pub(super) fn judgment_cache() -> Option<JudgmentCache> {
    let dir = AgentDir::default();
    JudgmentCache::open(&omp_providers::judgment_cache_path(&dir)).ok()
}

pub(super) async fn run_judgment_op(args: &Value, ctx: &OpCtx) -> Result<Value, String> {
    if !args.is_object() {
        return Err(
            "judge() received invalid arguments: expected { state, questions }".to_string(),
        );
    }
    let state = parse_state(args.get("state").unwrap_or(&Value::Null))?;
    let questions = parse_questions(args.get("questions"))?;
    let dir = AgentDir::default();
    let chain = omp_providers::resolve_judge_chain(&dir)
        .map_err(|error| format!("judge() has no model registry: {error}"))?;
    throw_if_aborted(&ctx.signal)?;
    let _permit = REQUEST_SLOTS
        .acquire()
        .await
        .map_err(|_| "judge() request aborted.".to_string())?;
    let result = omp_providers::judge(
        &chain,
        &JudgeRequest { state, questions },
        judgment_cache().as_ref(),
    )
    .await
    .map_err(|error| error.to_string())?;
    throw_if_aborted(&ctx.signal)?;
    Ok(judgment_result_value(&result))
}

pub(super) fn throw_if_aborted(signal: &AbortSignal) -> Result<(), String> {
    if signal.is_aborted() {
        return Err("aborted".to_string());
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// `__judge_batch__` (TS `judgment-batch-bridge.ts` `runEvalJudgmentBatch`)
// ---------------------------------------------------------------------------
