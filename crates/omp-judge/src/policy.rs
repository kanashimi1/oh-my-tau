//! Laya context-policy inference: one `action` question (`noop | clm_edit | rlm_offload`) over a
//! controller-state text rendered by `omp-cli`'s `context_policy` module (and, byte-identically,
//! by `bench/context/ctxp.py` when the training set is built).

use std::path::Path;

use anyhow::{Context, Result};
use serde_json::Value;

use crate::scorer::ModelPaths;

/// The single question every ctx_policy record carries (also read by the Python dataset builder).
pub const QUESTIONS: &str = include_str!("policy_questions.json");

/// Loaded policy head. Owns the ONNX session; not `Sync`.
pub struct PolicyScorer {
    judge: laya_ort::Judge,
    questions: Value,
}

impl PolicyScorer {
    /// # Errors
    /// Missing/invalid checkpoint or ONNX file.
    pub fn load(paths: &ModelPaths) -> Result<Self> {
        let judge = laya_ort::Judge::load(
            Path::new(&paths.ckpt),
            Path::new(&paths.onnx),
            paths.threads.max(1),
        )
        .with_context(|| {
            format!(
                "load laya policy (ckpt {}, onnx {})",
                paths.ckpt.display(),
                paths.onnx.display()
            )
        })?;
        Ok(Self {
            judge,
            questions: serde_json::from_str(QUESTIONS).context("policy_questions.json")?,
        })
    }

    /// `(option, probability)` in label order for one controller state.
    ///
    /// # Errors
    /// Inference failure.
    pub fn probs(&mut self, state: &str) -> Result<Vec<(String, f64)>> {
        let qs = self.judge.judge(state, &self.questions)?;
        let q = qs
            .into_iter()
            .find(|q| q.qid == "action")
            .context("missing question `action`")?;
        Ok(q.options
            .into_iter()
            .zip(q.probs.into_iter().map(f64::from))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn questions_parse_and_have_three_actions() {
        let v: serde_json::Value = serde_json::from_str(super::QUESTIONS).unwrap();
        let keys: Vec<_> = v["action"]["criteria"]
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect();
        assert_eq!(keys, ["noop", "clm_edit", "rlm_offload"]);
    }
}
