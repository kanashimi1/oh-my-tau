//! ONNX backend seam: `laya_ort::Judge` with the `onnx` feature, otherwise a
//! stub whose `load` fails cleanly (builds without prebuilt ONNX Runtime, e.g. musl).

#[cfg(feature = "onnx")]
pub use laya_ort::Judge;

#[cfg(not(feature = "onnx"))]
pub use stub::Judge;

#[cfg(not(feature = "onnx"))]
mod stub {
    use std::path::Path;

    use anyhow::{bail, Result};
    use serde_json::Value;

    /// Per-question probabilities (mirror of `laya_ort::QuestionProbs`).
    #[derive(Debug, Clone)]
    pub struct QuestionProbs {
        pub qid: String,
        pub options: Vec<String>,
        pub probs: Vec<f32>,
    }

    /// Placeholder judge: this build has no ONNX Runtime.
    pub struct Judge;

    impl Judge {
        /// # Errors
        /// Always: the binary was built without the `onnx` feature.
        pub fn load(_ckpt: &Path, _onnx: &Path, _threads: usize) -> Result<Self> {
            bail!("this omt build has no ONNX Runtime (built without the `onnx` feature)")
        }

        /// # Errors
        /// Unreachable in practice: [`Judge::load`] never succeeds.
        pub fn judge(&mut self, _state: &str, _questions: &Value) -> Result<Vec<QuestionProbs>> {
            bail!("this omt build has no ONNX Runtime")
        }
    }
}
