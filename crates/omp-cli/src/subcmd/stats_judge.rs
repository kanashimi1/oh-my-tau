//! The dashboard's paid frustration judge (TS `openStandaloneJudge(cwd, "stats_frustration")`):
//! the `judge` role chain of `omp-providers`, answering the frustration questions through the
//! shared judgment cache. Registered with `omp-stats`, which holds no LLM client itself.

use std::collections::BTreeMap;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use omp_providers::{AgentDir, JudgeCandidate, JudgeRequest, JudgmentCache};
use omp_stats::frustration::{
    set_stats_judge_provider, JudgeAnswer, JudgeFuture, JudgeModel, Question, Questions, StatsJudge,
};
use serde_json::{json, Map, Value};

/// Register the judge provider; the chain is resolved lazily on the first estimate or run.
pub fn register() {
    set_stats_judge_provider(Some(Arc::new(open)));
}

struct HostJudge {
    chain: Vec<JudgeCandidate>,
    cache: Option<JudgmentCache>,
}

/// Resolve the `judge` role chain. Like TS (`resolveCandidates`), a role without a credentialed
/// model is an empty chain: the dashboard reports "No judge model is available".
#[allow(clippy::unnecessary_wraps)]
fn open() -> Result<Arc<dyn StatsJudge>> {
    let dir = AgentDir::default();
    let chain = omp_providers::resolve_judge_chain(&dir).unwrap_or_else(|error| {
        tracing::warn!(error = %error, "stats judge: no usable judge model");
        Vec::new()
    });
    let cache = JudgmentCache::open(&omp_providers::judgment_cache_path(&dir)).ok();
    Ok(Arc::new(HostJudge { chain, cache }))
}

/// The wire form of a question (the shape `eval`'s `judge()` bridge forwards).
fn question_wire(question: &Question) -> Value {
    match question {
        Question::Score {
            instructions,
            criteria,
        } => {
            json!({"type": "score", "instructions": instructions, "criteria": criteria})
        }
        Question::Noul {
            instructions,
            criteria,
        } => {
            let mut wire = json!({"type": "noul", "instructions": instructions});
            if let Some(criteria) = criteria {
                wire["criteria"] = json!(criteria);
            }
            wire
        }
        Question::Choice {
            instructions,
            criteria,
        } => {
            json!({"type": "choice", "instructions": instructions, "criteria": criteria})
        }
    }
}

fn levels(answer: &Value) -> Option<BTreeMap<String, f64>> {
    answer
        .get("probabilities")?
        .as_object()?
        .iter()
        .map(|(level, p)| Some((level.clone(), p.as_f64()?)))
        .collect()
}

impl StatsJudge for HostJudge {
    fn primary_model(&self) -> Option<JudgeModel> {
        let model = &self.chain.first()?.model;
        Some(JudgeModel {
            provider: model.provider.clone(),
            id: model.id.clone(),
            cost_input: model.model.cost.input.as_f64(),
            cost_output: model.model.cost.output.as_f64(),
        })
    }

    fn judge<'a>(&'a self, text: &'a str, questions: &'a Questions) -> JudgeFuture<'a> {
        Box::pin(async move {
            let wire: Map<String, Value> = questions
                .iter()
                .map(|(id, question)| (id.clone(), question_wire(question)))
                .collect();
            let request = JudgeRequest {
                state: Value::String(text.to_owned()),
                questions: wire,
            };
            let result = omp_providers::judge(&self.chain, &request, self.cache.as_ref()).await?;
            let annoyed_levels = result
                .answers
                .get("annoyed")
                .and_then(levels)
                .ok_or_else(|| anyhow!("judge answered without `annoyed` probabilities"))?;
            let target = result
                .answers
                .get("target")
                .and_then(|answer| answer.get("choice"))
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow!("judge answered without a `target` choice"))?
                .to_owned();
            Ok(JudgeAnswer {
                provider: result.provider,
                model: result.model,
                annoyed_levels,
                target,
                cost: result.usage.cost,
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frustration_questions_have_the_wire_shape_the_judge_expects() {
        let questions = &*omp_stats::frustration::FRUSTRATION_QUESTIONS;
        let annoyed = question_wire(&questions["annoyed"]);
        assert_eq!(annoyed["type"], "score");
        assert_eq!(annoyed["criteria"].as_array().map(Vec::len), Some(4));
        let target = question_wire(&questions["target"]);
        assert_eq!(target["type"], "choice");
        assert!(target["criteria"]["assistant"].is_string());
    }

    #[test]
    fn levels_reads_score_probabilities() {
        let answer =
            json!({"type": "score", "score": 2.0, "probabilities": {"0": 0.1, "2": 0.7, "3": 0.2}});
        let parsed = levels(&answer).expect("levels");
        assert_eq!(parsed["2"], 0.7);
        assert!(levels(&json!({"type": "choice"})).is_none());
    }

    #[test]
    fn empty_chain_has_no_primary_model() {
        let judge = HostJudge {
            chain: Vec::new(),
            cache: None,
        };
        assert!(judge.primary_model().is_none());
    }
}
