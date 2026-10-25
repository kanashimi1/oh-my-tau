//! Laya judge inference (ONNX via `laya-ort`) for judge views.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde_json::{json, Map, Value};

/// Evidence-selection checkpoint (tokenizer + `rl_agent_config.json` temperatures):
/// `$OMP_JUDGE_CKPT` if set, else `$HOME/.omp-rs/judge/tj_sel/ckpt`. A missing
/// checkpoint is reported by [`Scorer::load`], never by this lookup.
#[must_use]
pub fn default_ckpt() -> PathBuf {
    if let Some(p) = std::env::var_os("OMP_JUDGE_CKPT") {
        return PathBuf::from(p);
    }
    let home = std::env::var_os("HOME").map_or_else(|| PathBuf::from("."), PathBuf::from);
    home.join(".omp-rs/judge/tj_sel/ckpt")
}
/// ONNX exported from the checkpoint at [`default_ckpt`].
pub const DEFAULT_ONNX: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/onnx/laya_judge_tjsel.onnx");

const QUESTIONS: &str = include_str!("questions.json");

/// Model locations.
#[derive(Debug, Clone)]
pub struct ModelPaths {
    pub ckpt: PathBuf,
    pub onnx: PathBuf,
    pub threads: usize,
}

impl Default for ModelPaths {
    fn default() -> Self {
        Self {
            ckpt: default_ckpt(),
            onnx: DEFAULT_ONNX.into(),
            threads: 2,
        }
    }
}

/// One scored turn.
#[derive(Debug, Clone, PartialEq)]
pub struct Scores {
    pub success_p: f64,
    pub rework_p: f64,
    /// `(option, probability)` in label order (`none`, `wrong_result`, ...).
    pub failure_mode_probs: Vec<(String, f64)>,
}

impl Scores {
    /// `failure_mode_probs` as a JSON object (insertion-ordered).
    #[must_use]
    pub fn failure_mode_json(&self) -> Value {
        let mut m = Map::new();
        for (k, p) in &self.failure_mode_probs {
            m.insert(k.clone(), json!(p));
        }
        Value::Object(m)
    }
}

/// Loaded judge. Owns the ONNX session; not `Sync` (use from one thread).
pub struct Scorer {
    judge: crate::onnx::Judge,
    questions: Value,
}

impl Scorer {
    /// # Errors
    /// Missing/invalid checkpoint or ONNX file.
    pub fn load(paths: &ModelPaths) -> Result<Self> {
        let judge = crate::onnx::Judge::load(
            Path::new(&paths.ckpt),
            Path::new(&paths.onnx),
            paths.threads.max(1),
        )
        .with_context(|| {
            format!(
                "load laya judge (ckpt {}, onnx {})",
                paths.ckpt.display(),
                paths.onnx.display()
            )
        })?;
        Ok(Self {
            judge,
            questions: serde_json::from_str(QUESTIONS).context("questions.json")?,
        })
    }

    /// Score one judge view.
    ///
    /// # Errors
    /// Inference failure.
    pub fn score(&mut self, view: &str) -> Result<Scores> {
        let qs = self.judge.judge(view, &self.questions)?;
        let find = |qid: &str| {
            qs.iter()
                .find(|q| q.qid == qid)
                .with_context(|| format!("missing question {qid}"))
        };
        let p_true = |qid: &str| -> Result<f64> {
            let q = find(qid)?;
            let i = q
                .options
                .iter()
                .position(|o| o == "true")
                .context("no `true` option")?;
            Ok(f64::from(q.probs[i]))
        };
        let fm = find("failure_mode")?;
        Ok(Scores {
            success_p: p_true("success")?,
            rework_p: p_true("rework")?,
            failure_mode_probs: fm
                .options
                .iter()
                .cloned()
                .zip(fm.probs.iter().map(|p| f64::from(*p)))
                .collect(),
        })
    }
}
