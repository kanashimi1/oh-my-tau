//! Model selection via `omp-providers` (`models.yml` / `config.yml` roles /
//! agent.db credentials). Setting `OMP_RS_BASE_URL` switches to an explicit
//! openai-completions endpoint (`OMP_RS_API_KEY` / `OMP_RS_MODEL`).

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

async fn from_providers(reference: &str) -> Result<Selected> {
    let r = omp_providers::resolve_async(reference.to_string()).await?;
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
pub async fn select(model_ref: Option<&str>) -> Result<Selected> {
    let env_model = std::env::var("OMP_RS_MODEL").ok();
    if std::env::var_os("OMP_RS_BASE_URL").is_none() {
        let reference = model_ref
            .map(str::to_string)
            .or(env_model)
            .unwrap_or_else(|| "default".to_string());
        return from_providers(&reference).await;
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

/// Catalog metadata omp's `Model` carries beyond pa-ai's descriptor, in catalog key order.
const CATALOG_EXTRA_KEYS: [&str; 7] =
    ["int", "tps", "thinking", "identity", "requiresGlyphTokenization", "tokenizer", "supportsComputerUse"];

fn catalog_path() -> Option<std::path::PathBuf> {
    if let Some(p) = std::env::var_os("OMP_RS_CATALOG") {
        return Some(p.into());
    }
    let home = std::env::var_os("HOME")?;
    Some(std::path::PathBuf::from(home).join(".bun/install/global/node_modules/@oh-my-pi/pi-catalog/src/models.json"))
}

fn catalog() -> Option<&'static serde_json::Value> {
    static CATALOG: std::sync::OnceLock<Option<serde_json::Value>> = std::sync::OnceLock::new();
    CATALOG
        .get_or_init(|| serde_json::from_slice(&std::fs::read(catalog_path()?).ok()?).ok())
        .as_ref()
}

/// `int`/`tps`/`thinking`/`identity`/`tokenizer`/... of `provider/id` from omp's bundled
/// catalog (`pi-catalog/src/models.json`; override the path with `OMP_RS_CATALOG`).
/// `None` when the catalog is not installed or lists no such model.
#[must_use]
pub fn catalog_extras(provider: &str, id: &str) -> Option<Vec<(&'static str, serde_json::Value)>> {
    let entry = catalog()?.get(provider)?.get(id)?;
    let extras: Vec<_> = CATALOG_EXTRA_KEYS
        .iter()
        .filter_map(|k| entry.get(*k).map(|v| (*k, v.clone())))
        .collect();
    (!extras.is_empty()).then_some(extras)
}

fn parse_effort(s: &str) -> Option<ThinkingLevel> {
    parse_thinking(s)
}

/// Catalog `thinking.efforts` of `provider/id` (least to most intensive), when listed.
#[must_use]
pub fn catalog_efforts(provider: &str, id: &str) -> Option<Vec<ThinkingLevel>> {
    let thinking = catalog_extras(provider, id)?.into_iter().find(|(k, _)| *k == "thinking")?.1;
    let efforts: Vec<_> = thinking.get("efforts")?.as_array()?.iter().filter_map(|e| parse_effort(e.as_str()?)).collect();
    (!efforts.is_empty()).then_some(efforts)
}

/// Lowest effort of a model that cannot think "off" (`thinking.requiresEffort`).
#[must_use]
pub fn always_thinking_floor(provider: &str, id: &str) -> Option<ThinkingLevel> {
    let thinking = catalog_extras(provider, id)?.into_iter().find(|(k, _)| *k == "thinking")?.1;
    if thinking.get("requiresEffort") != Some(&serde_json::Value::Bool(true)) {
        return None;
    }
    catalog_efforts(provider, id)?.into_iter().next()
}

fn ladder_index(l: ThinkingLevel) -> usize {
    [ThinkingLevel::Off, ThinkingLevel::Minimal, ThinkingLevel::Low, ThinkingLevel::Medium, ThinkingLevel::High, ThinkingLevel::Xhigh, ThinkingLevel::Max]
        .iter()
        .position(|x| *x == l)
        .unwrap_or(0)
}

/// `resolveThinkingLevelForModel`: `off` stays; otherwise the nearest higher supported effort,
/// else the nearest lower one (catalog efforts when listed, else the pa-ai capability map).
#[must_use]
pub fn clamp_level(model: &pa_types::ai::Model, level: ThinkingLevel) -> ThinkingLevel {
    if level == ThinkingLevel::Off {
        return level;
    }
    match catalog_efforts(&model.provider, &model.id) {
        Some(efforts) if !efforts.contains(&level) => efforts
            .iter()
            .copied()
            .find(|e| ladder_index(*e) > ladder_index(level))
            .or_else(|| efforts.iter().rev().copied().find(|e| ladder_index(*e) < ladder_index(level)))
            .unwrap_or(level),
        Some(_) => level,
        None => map_level(pa_types::ai::clamp_thinking_level(model, crate::adapter::model_thinking_level(level))),
    }
}
