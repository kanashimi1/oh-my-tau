//! Model selection via `omp-providers` (models.yml / config.yml roles /
//! agent.db credentials). Setting OMP_RS_BASE_URL switches to an explicit
//! openai-completions endpoint (OMP_RS_API_KEY / OMP_RS_MODEL).

use anyhow::{anyhow, Result};
use pa_agent::types::ThinkingLevel;
use serde_json::json;

pub struct Selected {
    pub model: pa_types::ai::Model,
    pub api_key: Option<String>,
    pub thinking: ThinkingLevel,
    pub headers: Option<std::collections::BTreeMap<String, String>>,
}

fn map_level(l: pa_types::ai::ModelThinkingLevel) -> ThinkingLevel {
    crate::adapter::map_thinking_level(l)
}

fn from_providers(reference: &str) -> Result<Selected> {
    let r = omp_providers::resolve(reference)?;
    let headers = (!r.headers.is_empty()).then(|| r.headers.iter().cloned().collect());
    Ok(Selected {
        // No explicit level: reasoning models default to medium (some, e.g. zai
        // glm, reject thinking=off), others stay off.
        thinking: r.thinking.map_or(
            if r.reasoning { ThinkingLevel::Medium } else { ThinkingLevel::Off },
            map_level,
        ),
        api_key: r.api_key.clone(),
        model: r.model,
        headers,
    })
}

fn parse_thinking(s: &str) -> Option<ThinkingLevel> {
    Some(match s {
        "off" => ThinkingLevel::Off,
        "minimal" => ThinkingLevel::Minimal,
        "low" => ThinkingLevel::Low,
        "medium" => ThinkingLevel::Medium,
        "high" => ThinkingLevel::High,
        "xhigh" => ThinkingLevel::Xhigh,
        "max" => ThinkingLevel::Max,
        _ => return None,
    })
}

/// Resolve `provider/id[:thinking]`, falling back to `OMP_RS_MODEL`.
///
/// # Errors
/// Fails when no model reference is available or the model cannot be built.
pub fn select(model_ref: Option<&str>) -> Result<Selected> {
    let env_model = std::env::var("OMP_RS_MODEL").ok();
    if std::env::var_os("OMP_RS_BASE_URL").is_none() {
        let reference = model_ref
            .map(str::to_string)
            .or(env_model)
            .unwrap_or_else(|| "default".to_string());
        return from_providers(&reference);
    }
    let reference = model_ref
        .map(str::to_string)
        .or(env_model)
        .ok_or_else(|| anyhow!("no model: pass --model provider/id or set OMP_RS_MODEL"))?;
    let (spec, thinking) = match reference.rsplit_once(':') {
        Some((head, tail)) if parse_thinking(tail).is_some() => (head, parse_thinking(tail)),
        _ => (reference.as_str(), None),
    };
    let (provider, id) = spec.split_once('/').unwrap_or(("custom", spec));
    let base_url = std::env::var("OMP_RS_BASE_URL")?;
    let api_key = std::env::var("OMP_RS_API_KEY")
        .ok()
        .or_else(|| pa_ai::env_api_keys::get_env_api_key(provider));
    let model: pa_types::ai::Model = serde_json::from_value(json!({
        "id": id,
        "name": id,
        "api": "openai-completions",
        "provider": provider,
        "baseUrl": base_url,
        "reasoning": false,
        "input": ["text"],
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        "contextWindow": 128_000,
        "maxTokens": 16_384,
    }))?;
    Ok(Selected {
        model,
        api_key,
        thinking: thinking.unwrap_or(ThinkingLevel::Off),
        headers: None,
    })
}
