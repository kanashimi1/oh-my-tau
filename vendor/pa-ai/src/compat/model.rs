//! Glue between the wire [`Model`] and the policy resolver.
//!
//! `Model.compat` carries the *sparse* compat overrides (TS `Model.compatConfig`: what a
//! `models.yml` `compat:` block or a catalog row's `compatConfig` authored); the complete resolved
//! record is always derived from it by [`resolve_model_policy`], never stored. Thinking metadata
//! the resolver derives is materialized into `thinking_level_map` ([`materialize`]), the
//! request-side shape of TS `Model.thinking`.

use std::collections::BTreeMap;

use serde_json::{Map, Value};

use pa_types::ai::{Model, ModelThinkingLevel};

use super::resolve::{resolve_model_policy, ModelSpec, ResolveError, ResolvedModelPolicy};

fn spec_of(model: &Model, explicit_thinking: Option<&Value>) -> ModelSpec {
    ModelSpec {
        provider: model.provider.clone(),
        id: model.id.clone(),
        api: model.api.clone(),
        base_url: model.base_url.clone(),
        provider_type: None,
        reasoning: model.reasoning,
        thinking: explicit_thinking.cloned(),
        compat: model
            .compat
            .as_ref()
            .map(|c| c.raw.clone().into_iter().collect()),
        lenient: true,
    }
}

/// Resolve the full policy for a model (identity, compat, thinking, catalog, request).
///
/// # Errors
/// A cascade tie between equal-rank rules (a rule-authoring defect) or an empty thinking range.
pub fn resolve_for_model(model: &Model) -> Result<ResolvedModelPolicy, ResolveError> {
    resolve_model_policy(&spec_of(model, None), None)
}

/// The thinking-level map a resolved TS thinking config denotes: every ladder effort addressable
/// (wire value = its `effortMap` entry or itself), every other effort explicitly unsupported, and
/// `off` unsupported on `requiresEffort` models.
fn thinking_level_map_of(thinking: &Value) -> BTreeMap<ModelThinkingLevel, Option<String>> {
    let efforts: Vec<&str> = thinking
        .get("efforts")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    let effort_map = thinking.get("effortMap").and_then(Value::as_object);
    let mut map = BTreeMap::new();
    for level in [
        ModelThinkingLevel::Minimal,
        ModelThinkingLevel::Low,
        ModelThinkingLevel::Medium,
        ModelThinkingLevel::High,
        ModelThinkingLevel::Xhigh,
        ModelThinkingLevel::Max,
    ] {
        let name = level.wire_name();
        let value = efforts.contains(&name).then(|| {
            effort_map
                .and_then(|m| m.get(name))
                .and_then(Value::as_str)
                .unwrap_or(name)
                .to_string()
        });
        map.insert(level, value);
    }
    if thinking.get("requiresEffort").and_then(Value::as_bool) == Some(true) {
        map.insert(ModelThinkingLevel::Off, None);
    }
    map
}

/// TS `buildModel` for a model that arrived as a spec: resolve the policy once and apply it. A
/// reviewed `thinking-upgrade-neutral` correction promotes `reasoning`; the resolved thinking
/// ladder becomes the model's `thinking_level_map` unless the model already carries an explicit
/// map (a `models.yml` `thinkingLevelMap`).
///
/// `explicit_thinking` is the row's authored TS `thinking` object (catalog/cache rows).
///
/// # Errors
/// See [`resolve_for_model`].
pub fn materialize(
    model: &mut Model,
    explicit_thinking: Option<&Value>,
) -> Result<ResolvedModelPolicy, ResolveError> {
    let policy = resolve_model_policy(&spec_of(model, explicit_thinking), None)?;
    // TS `applyCatalogAssignments`: rule-assigned catalog data the request builders read.
    if model.omit_max_output_tokens.is_none() {
        if let Some(omit) = policy
            .catalog
            .get("omitMaxOutputTokens")
            .and_then(Value::as_bool)
        {
            model.omit_max_output_tokens = Some(omit);
        }
    }
    if let Some(thinking) = &policy.thinking {
        model.reasoning = true;
        if model.thinking_level_map.is_none() {
            model.thinking_level_map = Some(thinking_level_map_of(thinking));
        }
    }
    Ok(policy)
}

/// A `Model` from one cached/bundled TS catalog row, not yet materialized ([`materialize`] it once
/// every override has been applied). The stored resolved `compat` is dropped (it may predate the
/// current rules); the row's `compatConfig` is the authored override set and the row's `thinking`
/// the authored thinking, returned alongside.
#[must_use]
pub fn model_from_catalog_row(mut row: Value) -> Option<(Model, Option<Value>)> {
    let obj = row.as_object_mut()?;
    let thinking = obj.remove("thinking").filter(|t| !t.is_null());
    let config = obj.remove("compatConfig").filter(Value::is_object);
    obj.remove("compat");
    if let Some(config) = config {
        obj.insert("compat".into(), config);
    }
    // TS rows may carry `contextWindow`/`maxTokens: null` (model known by id only). The Rust
    // `Model` models "unknown" as 0, which every consumer already treats like TS `?? 0` /
    // `?? UNKNOWN_MODEL_MAX_OUTPUT_TOKENS`.
    let obj = row.as_object_mut()?;
    for key in ["contextWindow", "maxTokens"] {
        if obj.get(key).is_some_and(Value::is_null) {
            obj.insert(key.into(), Value::from(0u64));
        }
    }
    let model: Model = serde_json::from_value(row).ok()?;
    Some((model, thinking))
}

/// Sparse compat overrides as a plain JSON map (empty when none).
#[must_use]
pub fn compat_overrides(model: &Model) -> Map<String, Value> {
    model
        .compat
        .as_ref()
        .map(|c| c.raw.clone().into_iter().collect())
        .unwrap_or_default()
}
