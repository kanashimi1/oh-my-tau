//! SERA rubrics (arXiv 2610.04902): a parent commits a weighted success rubric
//! for a child BEFORE the child runs; afterwards the child's output is scored
//! against that frozen rubric and the score is logged as a training signal.
//!
//! Paper contract (§3.2): `ρ = {(k_j, w_j)}`, `w_j ≥ 0`, `Σ w_j = 1`; the scorer
//! returns a binary success gate `g ∈ {0,1}` and per-criterion scores
//! `c_j ∈ [0,1]`; node score `s = g · Σ w_j c_j` (a failed child scores exactly
//! 0). The signal is not a free judge: when an external check exists (an
//! `outputSchema` verdict, a program verifier) it is logged next to the score as
//! the verified outcome `y`, and [`rank_loss`] is the paper's ranking objective
//! `mean max(0, δ − s_i + s_j)` over pairs with `y_i > y_j` (eq. 3).
//!
//! This module is provider-free: the scoring model call is injected through
//! [`RubricScorer`]. [`spawn_all_with_rubrics`] is the runner hook: freeze →
//! spawn → score → log.

use std::future::Future;
use std::io::Write as _;
use std::pin::Pin;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;
use serde_json::{json, Value};

use crate::runner::{RunOutput, RunStatus, TaskRunner, TaskSpec};
use crate::schema::SchemaValidation;

/// Hard cap on criteria per rubric (keeps the scorer prompt bounded).
pub const MAX_CRITERIA: usize = 12;
/// Env var: JSONL file receiving one [`RubricRecord`] per scored child.
pub const LOG_ENV: &str = "OMP_RUBRIC_LOG";

/// One weighted success criterion; weights are normalized to sum to 1.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Criterion {
    pub criterion: String,
    pub weight: f64,
}

/// An immutable rubric, committed before the child it scores starts.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FrozenRubric {
    /// Stable content hash (criteria text + normalized weights).
    pub id: String,
    pub criteria: Vec<Criterion>,
    /// Unix millis at which the rubric was frozen (always < `scored_ms`).
    pub frozen_ms: u64,
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

impl FrozenRubric {
    /// Validate and freeze a wire rubric: `[{"criterion": str, "weight": number≥0}]`
    /// (`weight` defaults to 1). Weights are normalized to sum to 1.
    ///
    /// # Errors
    /// Not an array, empty, over [`MAX_CRITERIA`], blank criterion, negative or
    /// non-finite weight, or all-zero weights.
    pub fn freeze(raw: &Value) -> Result<Self, String> {
        let list = raw
            .as_array()
            .ok_or("rubric must be an array of {criterion, weight}")?;
        if list.is_empty() {
            return Err("rubric must contain at least one criterion".into());
        }
        if list.len() > MAX_CRITERIA {
            return Err(format!(
                "rubric has {} criteria; max is {MAX_CRITERIA}",
                list.len()
            ));
        }
        let mut parsed = Vec::with_capacity(list.len());
        for (i, item) in list.iter().enumerate() {
            let text = item
                .get("criterion")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .ok_or_else(|| format!("rubric[{i}].criterion must be a non-empty string"))?;
            let weight = match item.get("weight") {
                None | Some(Value::Null) => 1.0,
                Some(w) => w
                    .as_f64()
                    .ok_or_else(|| format!("rubric[{i}].weight must be a number"))?,
            };
            if !weight.is_finite() || weight < 0.0 {
                return Err(format!("rubric[{i}].weight must be a finite number ≥ 0"));
            }
            parsed.push((text.to_string(), weight));
        }
        let total: f64 = parsed.iter().map(|(_, w)| w).sum();
        if total <= 0.0 {
            return Err("rubric weights must not all be zero".into());
        }
        let criteria: Vec<Criterion> = parsed
            .into_iter()
            .map(|(criterion, w)| Criterion {
                criterion,
                weight: w / total,
            })
            .collect();
        let canon = criteria
            .iter()
            .map(|c| format!("{}\u{1f}{:.9}", c.criterion, c.weight))
            .collect::<Vec<_>>()
            .join("\u{1e}");
        Ok(Self {
            id: format!("rb-{:016x}", fnv1a(canon.as_bytes())),
            criteria,
            frozen_ms: now_ms(),
        })
    }

    /// Node score `g · Σ w_j c_j`; the gate zeroes the score exactly.
    ///
    /// # Errors
    /// `c` length differs from the criteria count.
    pub fn score(&self, gate: bool, c: &[f64]) -> Result<f64, String> {
        if c.len() != self.criteria.len() {
            return Err(format!(
                "expected {} criterion scores, got {}",
                self.criteria.len(),
                c.len()
            ));
        }
        if !gate {
            return Ok(0.0);
        }
        Ok(self
            .criteria
            .iter()
            .zip(c)
            .map(|(k, v)| k.weight * v.clamp(0.0, 1.0))
            .sum::<f64>()
            .clamp(0.0, 1.0))
    }

    /// Scorer prompt: the frozen criteria, the delegated task and the child's
    /// output; the scorer must answer with one JSON object.
    #[must_use]
    pub fn scorer_prompt(&self, task: &str, output: &str) -> String {
        let mut p = String::from(
            "You score a delegated subtask against a rubric that was fixed before the work started.\n\
             Reply with ONE JSON object and nothing else: {\"success\": true|false, \"scores\": [c1, c2, ...]}.\n\
             `success` is false when the subtask was not completed or produced no usable output; \
             each c_j in [0,1] grades criterion j in order.\n\n<rubric>\n",
        );
        for (i, k) in self.criteria.iter().enumerate() {
            p.push_str(&format!(
                "{}. {} (weight {:.3})\n",
                i + 1,
                k.criterion,
                k.weight
            ));
        }
        p.push_str("</rubric>\n\n<task>\n");
        p.push_str(task);
        p.push_str("\n</task>\n\n<output>\n");
        p.push_str(output);
        p.push_str("\n</output>\n");
        p
    }
}

/// Parse the scorer's reply: first JSON object found, `{success, scores[n]}`.
///
/// # Errors
/// No JSON object, missing/invalid fields, or wrong score count.
pub fn parse_scorer_reply(text: &str, n: usize) -> Result<(bool, Vec<f64>), String> {
    let start = text.find('{').ok_or("scorer reply has no JSON object")?;
    let mut stream = serde_json::Deserializer::from_str(&text[start..]).into_iter::<Value>();
    let value = stream
        .next()
        .ok_or("scorer reply is empty")?
        .map_err(|e| format!("scorer reply is not valid JSON: {e}"))?;
    let gate = match value.get("success") {
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(x)) => x.as_f64().unwrap_or(0.0) >= 0.5,
        _ => return Err("scorer reply lacks boolean `success`".into()),
    };
    let scores = value
        .get("scores")
        .and_then(Value::as_array)
        .ok_or("scorer reply lacks `scores` array")?
        .iter()
        .map(|v| {
            v.as_f64()
                .filter(|f| f.is_finite())
                .ok_or("scores must be finite numbers")
        })
        .collect::<Result<Vec<_>, _>>()?;
    if scores.len() != n {
        return Err(format!(
            "scorer returned {} scores for {n} criteria",
            scores.len()
        ));
    }
    Ok((gate, scores))
}

/// Ranking objective of SERA eq. 3: `mean_{(i,j): y_i>y_j} max(0, δ − s_i + s_j)`.
/// `None` when no pair has distinct verified outcomes (group skipped).
#[must_use]
pub fn rank_loss(scores: &[f64], verified: &[f64], margin: f64) -> Option<f64> {
    let n = scores.len().min(verified.len());
    let (mut sum, mut pairs) = (0.0, 0usize);
    for i in 0..n {
        for j in 0..n {
            if verified[i] > verified[j] {
                sum += (margin - scores[i] + scores[j]).max(0.0);
                pairs += 1;
            }
        }
    }
    (pairs > 0).then(|| sum / pairs as f64)
}

/// One scored child: the training-signal row appended to [`LOG_ENV`].
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RubricRecord {
    /// `task` | `llm_query` | any caller-chosen origin.
    pub source: String,
    pub rubric_id: String,
    pub criteria: Vec<Criterion>,
    /// Child/run id or candidate label.
    pub run_id: String,
    pub prompt_chars: usize,
    pub output_chars: usize,
    pub gate: bool,
    pub per_criterion: Vec<f64>,
    pub score: f64,
    /// External verified outcome `y ∈ [0,1]` (schema verdict / program check).
    pub verified: Option<f64>,
    pub frozen_ms: u64,
    pub scored_ms: u64,
}

impl RubricRecord {
    /// Build a record from a scorer verdict.
    ///
    /// # Errors
    /// Score-count mismatch.
    #[allow(clippy::too_many_arguments)]
    pub fn from_verdict(
        source: &str,
        rubric: &FrozenRubric,
        run_id: &str,
        prompt_chars: usize,
        output_chars: usize,
        gate: bool,
        per_criterion: Vec<f64>,
        verified: Option<f64>,
    ) -> Result<Self, String> {
        let score = rubric.score(gate, &per_criterion)?;
        Ok(Self {
            source: source.to_string(),
            rubric_id: rubric.id.clone(),
            criteria: rubric.criteria.clone(),
            run_id: run_id.to_string(),
            prompt_chars,
            output_chars,
            gate,
            per_criterion,
            score,
            verified,
            frozen_ms: rubric.frozen_ms,
            scored_ms: now_ms().max(rubric.frozen_ms + 1),
        })
    }

    /// Append to the `OMP_RUBRIC_LOG` JSONL file (failures ignored, like the RLM ledger log).
    pub fn append_log(&self) {
        let Some(path) = std::env::var_os(LOG_ENV) else {
            return;
        };
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
        {
            let _ = writeln!(f, "{}", json!(self));
        }
    }
}

/// Provider seam: run the scorer model on a prompt and return its text.
pub trait RubricScorer: Send + Sync {
    fn complete<'a>(
        &'a self,
        prompt: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<String, String>> + Send + 'a>>;
}

/// Verified outcome from a child's schema verdict: the only external check the
/// task tool has. `None` when no schema was declared.
#[must_use]
pub fn verified_from_schema(v: &SchemaValidation) -> Option<f64> {
    match v {
        SchemaValidation::NoSchema => None,
        SchemaValidation::Valid => Some(1.0),
        SchemaValidation::NotJson | SchemaValidation::Invalid(_) => Some(0.0),
    }
}

/// Score one settled child against its frozen rubric. A child that did not
/// complete takes gate 0 without a scorer call (paper §3.2).
///
/// # Errors
/// Scorer failure or an unparsable verdict.
pub async fn score_output(
    rubric: &FrozenRubric,
    scorer: &dyn RubricScorer,
    task: &str,
    out: &RunOutput,
) -> Result<RubricRecord, String> {
    let verified = verified_from_schema(&out.schema);
    let usable = out.status == RunStatus::Completed && !out.output.trim().is_empty();
    let (gate, per) = if usable {
        let reply = scorer
            .complete(&rubric.scorer_prompt(task, &out.output))
            .await?;
        parse_scorer_reply(&reply, rubric.criteria.len())?
    } else {
        (false, vec![0.0; rubric.criteria.len()])
    };
    let rec = RubricRecord::from_verdict(
        "task",
        rubric,
        &out.id,
        task.len(),
        out.output.len(),
        gate,
        per,
        verified,
    )?;
    rec.append_log();
    Ok(rec)
}

/// Runner hook: rubrics are frozen by the caller before this is invoked (the
/// [`FrozenRubric`] is the proof), children run, then each rubric-bearing
/// child is scored. A scorer failure leaves that child's record `Err`.
pub async fn spawn_all_with_rubrics(
    runner: &TaskRunner,
    items: Vec<(TaskSpec, Option<FrozenRubric>)>,
    scorer: &dyn RubricScorer,
) -> Vec<(RunOutput, Option<Result<RubricRecord, String>>)> {
    let (specs, rubrics): (Vec<_>, Vec<_>) = items.into_iter().unzip();
    let prompts: Vec<String> = specs.iter().map(|s| s.prompt.clone()).collect();
    let outputs = runner.spawn_all(specs).await;
    let mut scored = Vec::with_capacity(outputs.len());
    for (i, out) in outputs.into_iter().enumerate() {
        let rec = match rubrics.get(i).and_then(Option::as_ref) {
            Some(r) => {
                Some(score_output(r, scorer, prompts.get(i).map_or("", String::as_str), &out).await)
            }
            None => None,
        };
        scored.push((out, rec));
    }
    scored
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rb() -> FrozenRubric {
        FrozenRubric::freeze(&json!([
            {"criterion": "correct answer", "weight": 3},
            {"criterion": "concise", "weight": 1}
        ]))
        .unwrap()
    }

    #[test]
    fn weights_normalize_and_id_is_stable() {
        let a = rb();
        assert!((a.criteria[0].weight - 0.75).abs() < 1e-12);
        assert!((a.criteria.iter().map(|c| c.weight).sum::<f64>() - 1.0).abs() < 1e-12);
        assert_eq!(a.id, rb().id);
        let other = FrozenRubric::freeze(&json!([{"criterion": "x"}])).unwrap();
        assert_ne!(a.id, other.id);
    }

    #[test]
    fn invalid_rubrics_are_rejected() {
        for bad in [
            json!("x"),
            json!([]),
            json!([{"criterion": " "}]),
            json!([{"criterion": "a", "weight": -1}]),
            json!([{"criterion": "a", "weight": 0}]),
            json!([{"weight": 1}]),
            json!([{"criterion": "a", "weight": "hi"}]),
        ] {
            assert!(FrozenRubric::freeze(&bad).is_err(), "{bad}");
        }
        let many = Value::Array(
            (0..13)
                .map(|i| json!({"criterion": format!("c{i}")}))
                .collect(),
        );
        assert!(FrozenRubric::freeze(&many).is_err());
    }

    #[test]
    fn gate_zeroes_score_and_weights_apply() {
        let r = rb();
        assert!((r.score(true, &[1.0, 0.0]).unwrap() - 0.75).abs() < 1e-12);
        assert_eq!(r.score(false, &[1.0, 1.0]).unwrap(), 0.0);
        assert!((r.score(true, &[2.0, -1.0]).unwrap() - 0.75).abs() < 1e-12); // clamped
        assert!(r.score(true, &[1.0]).is_err());
    }

    #[test]
    fn scorer_reply_parsing() {
        let (g, s) = parse_scorer_reply(
            "Sure:\n```json\n{\"success\": true, \"scores\": [1, 0.5]}\n```",
            2,
        )
        .unwrap();
        assert!(g);
        assert_eq!(s, vec![1.0, 0.5]);
        assert!(parse_scorer_reply("{\"success\": true, \"scores\": [1]}", 2).is_err());
        assert!(parse_scorer_reply("no json", 1).is_err());
        assert!(parse_scorer_reply("{\"scores\": [1]}", 1).is_err());
    }

    #[test]
    fn rank_loss_matches_paper_eq3() {
        // perfect order with margin: zero loss
        assert_eq!(rank_loss(&[0.9, 0.1], &[1.0, 0.0], 0.5), Some(0.0));
        // inverted order: max(0, 0.5 - 0.1 + 0.9) = 1.3
        assert!((rank_loss(&[0.1, 0.9], &[1.0, 0.0], 0.5).unwrap() - 1.3).abs() < 1e-12);
        // all-equal verified outcomes: group skipped
        assert_eq!(rank_loss(&[0.1, 0.9], &[1.0, 1.0], 0.5), None);
    }

    #[test]
    fn schema_verdict_is_the_verified_outcome() {
        assert_eq!(verified_from_schema(&SchemaValidation::NoSchema), None);
        assert_eq!(verified_from_schema(&SchemaValidation::Valid), Some(1.0));
        assert_eq!(verified_from_schema(&SchemaValidation::NotJson), Some(0.0));
    }

    struct Fixed(&'static str);
    impl RubricScorer for Fixed {
        fn complete<'a>(
            &'a self,
            _p: &'a str,
        ) -> Pin<Box<dyn Future<Output = Result<String, String>> + Send + 'a>> {
            Box::pin(async move { Ok(self.0.to_string()) })
        }
    }

    fn out(status: RunStatus, text: &str) -> RunOutput {
        RunOutput {
            id: "agent-1".into(),
            agent: "task".into(),
            status,
            exit_code: Some(0),
            output: text.into(),
            stderr: String::new(),
            schema: SchemaValidation::Valid,
            duration: std::time::Duration::ZERO,
            truncated: false,
        }
    }

    #[tokio::test]
    async fn failed_child_scores_zero_without_scorer_call() {
        let rec = score_output(
            &rb(),
            &Fixed("{\"success\":true,\"scores\":[1,1]}"),
            "t",
            &out(RunStatus::Failed, "x"),
        )
        .await
        .unwrap();
        assert!(!rec.gate);
        assert_eq!(rec.score, 0.0);
        assert_eq!(rec.verified, Some(1.0));
        assert!(rec.scored_ms > rec.frozen_ms);
    }

    #[tokio::test]
    async fn completed_child_is_scored_and_logged() {
        // Serialize env-var + log-file access with any other test touching
        // `LOG_ENV` (CI runs tests in parallel threads; interleaved appends
        // produced torn reads).
        static LOG_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());
        let _guard = LOG_LOCK.lock().await;
        let dir = std::env::temp_dir().join(format!("omp-rubric-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let log = dir.join("r.jsonl");
        let _ = std::fs::remove_file(&log);
        std::env::set_var(LOG_ENV, &log);
        let rec = score_output(
            &rb(),
            &Fixed("{\"success\":true,\"scores\":[1,0]}"),
            "t",
            &out(RunStatus::Completed, "ans"),
        )
        .await
        .unwrap();
        std::env::remove_var(LOG_ENV);
        assert!((rec.score - 0.75).abs() < 1e-12);
        // Take the last *complete* line: a concurrent writer could have
        // appended a partial record at the tail when we read.
        let text = std::fs::read_to_string(&log).unwrap();
        let line = text
            .lines()
            .rev()
            .find(|l| serde_json::from_str::<Value>(l).is_ok())
            .expect("at least one complete JSONL record");
        let v: Value = serde_json::from_str(line).unwrap();
        assert_eq!(v["rubric_id"], rec.rubric_id);
        assert_eq!(v["verified"], 1.0);
        let _ = std::fs::remove_dir_all(dir);
    }
}
