//! Opt-in exact token accounting for Blaze tool results.
//!
//! `OMT_BLAZE_TOKENIZER_JSON` selects the target model's Hugging Face
//! tokenizer.json. Without it the existing `(bytes + 3) / 4` estimate is used
//! and labelled as such; an exact count is never claimed for a different model.

use anyhow::{Context, Result};
use serde_json::{json, Value};

#[derive(Debug)]
pub(crate) struct TokenMeter {
    exact: Option<tokenizers::Tokenizer>,
}

impl TokenMeter {
    pub(crate) fn from_env() -> Result<Self> {
        let Some(path) = std::env::var_os("OMT_BLAZE_TOKENIZER_JSON") else {
            return Ok(Self { exact: None });
        };
        let tokenizer = tokenizers::Tokenizer::from_file(&path)
            .map_err(|error| anyhow::anyhow!(error.to_string()))
            .with_context(|| format!("cannot load OMT_BLAZE_TOKENIZER_JSON {}", std::path::Path::new(&path).display()))?;
        Ok(Self { exact: Some(tokenizer) })
    }

    fn count(&self, text: &str) -> Result<u64> {
        match &self.exact {
            Some(tokenizer) => {
                let encoded = tokenizer
                    .encode(text, false)
                    .map_err(|error| anyhow::anyhow!(error.to_string()))?;
                Ok(encoded.len() as u64)
            }
            None => Ok(((text.len() as u64) + 3) >> 2),
        }
    }

    pub(crate) fn add_counts(&self, details: &mut Value, raw: &str, model_facing: &str) -> Result<()> {
        details["rawBytes"] = json!(raw.len());
        details["modelFacingBytes"] = json!(model_facing.len());
        details["rawTokens"] = json!(self.count(raw)?);
        details["modelFacingTokens"] = json!(self.count(model_facing)?);
        details["tokenCountKind"] = json!(if self.exact.is_some() { "tokenizer.json" } else { "bytes-div-4-estimate" });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configured_tokenizer_produces_exact_counts() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokenizer.json");
        std::fs::write(
            &path,
            r#"{"version":"1.0","added_tokens":[],"model":{"type":"WordLevel","vocab":{"[UNK]":0,"hello":1,"world":2},"unk_token":"[UNK]"},"pre_tokenizer":{"type":"Whitespace"}}"#,
        )
        .unwrap();
        let tokenizer = tokenizers::Tokenizer::from_file(&path).unwrap();
        let meter = TokenMeter { exact: Some(tokenizer) };
        let mut details = json!({});
        meter.add_counts(&mut details, "hello world", "hello").unwrap();
        assert_eq!(details["rawTokens"], 2);
        assert_eq!(details["modelFacingTokens"], 1);
        assert_eq!(details["tokenCountKind"], "tokenizer.json");
    }

    #[test]
    fn unconfigured_counts_are_labelled_estimates() {
        let meter = TokenMeter { exact: None };
        let mut details = json!({});
        meter.add_counts(&mut details, "12345678", "1234").unwrap();
        assert_eq!(details["rawTokens"], 2);
        assert_eq!(details["modelFacingTokens"], 1);
        assert_eq!(details["tokenCountKind"], "bytes-div-4-estimate");
    }
}
