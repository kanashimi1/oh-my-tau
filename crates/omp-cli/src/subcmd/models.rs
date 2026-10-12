//! `omp-rs models` — list/search the model catalog, mirroring TS
//! `cli/models-cli.ts` (`omp models [ls|find|refresh] [pattern] [--json]`).
//!
//! Sources, lowest to highest precedence (TS `findModel` order inverted for
//! listing): pa-ai's compiled catalog, omp's `models.db` `model_cache`, then
//! `models.yml` custom providers. Unlike the TS `refresh` action, omp-rs has
//! no online catalog fetch: `refresh` re-reads the on-disk caches.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::Result;
use pa_types::ai::{Model, ModelInput};
use serde_json::{json, Map, Value};

/// One catalog row: the model plus its supported effort names, when known.
struct ListedModel {
    model: Model,
    efforts: Option<Vec<String>>,
}

/// `omp models` action, mirroring TS `ModelsAction`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Action {
   Ls,
    Find,
    Refresh,
}

/// Entry point: `omp-rs models [ls|list|find|refresh] [pattern] [--json] [--kind k]`.
///
/// # Errors
/// Unreadable `models.db`/`models.yml` (parse problems are skipped per source).
pub fn run(args: &[String]) -> Result<i32> {
    let mut json = false;
    let mut positional: Vec<&str> = Vec::new();
    let mut it = args.iter().peekable();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--json" => json = true,
            "--kind" => {
                // Accepted for flag compatibility; omp-rs only tracks chat models.
                let _ = it.next();
            }
            other if other.starts_with("--kind=") => {}
            other if other.starts_with('-') => anyhow::bail!("unknown flag: {other}"),
            other => positional.push(other),
        }
    }
    let (action, pattern) = resolve_args(positional.first().copied(), positional.get(1).copied());
    if action == Action::Refresh {
        eprintln!("omp-rs: online catalog refresh is not supported; listing on-disk caches");
    }

    let dir = omp_providers::AgentDir::default();
    let catalog = collect_catalog(&dir);
    render(&catalog, action, pattern, json);
    Ok(0)
}

fn resolve_args<'a>(first: Option<&'a str>, second: Option<&'a str>) -> (Action, Option<&'a str>) {
    match first {
        Some("ls") | Some("list") => (Action::Ls, second),
        Some("find") => (Action::Find, second),
        Some("refresh") => (Action::Refresh, second),
        Some(other) => (Action::Ls, Some(other)),
        None => (Action::Ls, None),
    }
}

/// Merge every source into one `(provider, id)`-keyed map (later sources win),
/// restricted to *available* providers like TS `ModelRegistry.getAvailable`:
/// a provider is available when it is declared in `models.yml` or has a
/// credential (env API key, or an enabled `agent.db` `auth_credentials` row).
fn collect_catalog(dir: &omp_providers::AgentDir) -> BTreeMap<(String, String), ListedModel> {
    let mut out: BTreeMap<(String, String), ListedModel> = BTreeMap::new();
    let yml_path = dir.0.join("models.yml");
    let available = available_providers(&yml_path, &dir.0.join("agent.db"));

    for provider in pa_ai::models_generated::get_providers() {
        for model in pa_ai::models_generated::get_models(provider) {
            let efforts = model.thinking_level_map.as_ref().map(|map| {
                map.keys().map(level_name).collect::<Vec<_>>()
            });
            out.insert(
                (model.provider.clone(), model.id.clone()),
                ListedModel { model: model.clone(), efforts },
            );
        }
    }

    for (provider_id, entries) in db_cache_models(&dir.0.join("models.db")) {
        let provider = provider_id.split(':').next().unwrap_or(&provider_id).to_string();
        for entry in entries {
            // TS `getAvailable("chat")` drops non-chat catalog rows (`kind` set
            // to image/tts/stt/embedding/rerank/video).
            if entry.get("kind").and_then(Value::as_str).is_some_and(is_non_chat_kind) {
                continue;
            }
            let Some(mut model) = lenient_model(&entry, &provider) else { continue };
            if model.provider.is_empty() {
                model.provider = provider.clone();
            }
            let efforts = entry
                .get("thinking")
                .and_then(|t| t.get("efforts"))
                .and_then(Value::as_array)
                .map(|a| a.iter().filter_map(|e| e.as_str().map(str::to_string)).collect());
            out.insert((model.provider.clone(), model.id.clone()), ListedModel { model, efforts });
        }
    }

    for (provider, entry) in yml_custom_models(&yml_path) {
        if let Some(model) = custom_model(&provider, &entry) {
            out.insert((model.provider.clone(), model.id.clone()), ListedModel { model, efforts: None });
        }
    }
    apply_overrides(&mut out, &model_overrides(&yml_path));
    out.retain(|(provider, _), _| available.contains(provider));
    out
}

/// TS `KIND_API_KINDS`: non-chat catalog kinds excluded from the chat listing.
fn is_non_chat_kind(kind: &str) -> bool {
    matches!(kind, "image" | "tts" | "stt" | "embedding" | "rerank" | "video" | "judge")
}

/// Parse a `models.db` cache entry tolerantly: rows may carry `null` limits
/// (TS renders them as `-`), which the strict `Model` serde shape rejects.
fn lenient_model(entry: &Value, provider: &str) -> Option<Model> {
    if let Ok(mut model) = serde_json::from_value::<Model>(entry.clone()) {
        return Some(model);
    }
    let id = entry.get("id").and_then(Value::as_str)?.to_string();
    let name = entry.get("name").and_then(Value::as_str).unwrap_or(&id).to_string();
    let api = entry.get("api").and_then(Value::as_str).unwrap_or("openai-completions").to_string();
    let num = |key: &str| entry.get(key).and_then(Value::as_u64).unwrap_or(0);
    let input = entry
        .get("input")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|v| match v.as_str()? {
                    "image" => Some(ModelInput::Image),
                    _ => Some(ModelInput::Text),
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| vec![ModelInput::Text]);
    let zero = pa_types::JsNumber(0.0);
    Some(Model {
        id,
        name,
        api,
        provider: provider.to_string(),
        base_url: entry.get("baseUrl").and_then(Value::as_str).unwrap_or_default().to_string(),
        reasoning: entry.get("reasoning").and_then(Value::as_bool).unwrap_or(false),
        thinking_level_map: None,
        input,
        cost: entry
            .get("cost")
            .and_then(|c| serde_json::from_value::<pa_types::ai::ModelCost>(c.clone()).ok())
            .unwrap_or(pa_types::ai::ModelCost { input: zero, output: zero, cache_read: zero, cache_write: zero }),
        context_window: num("contextWindow"),
        max_tokens: num("maxTokens"),
        max_tokens_explicit: true,
        featured: None,
        headers: None,
        compat: None,
    })
}

/// `providers.*.modelOverrides` from `models.yml`: `"prov/id" -> patch fields`.
fn model_overrides(path: &Path) -> Vec<(String, String, Value)> {
    let Ok(text) = std::fs::read_to_string(path) else { return Vec::new() };
    let Ok(Value::Object(root)) = serde_yaml_ng::from_str::<serde_yaml_ng::Value>(&text).map(yaml_to_json) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    if let Some(providers) = root.get("providers").and_then(Value::as_object) {
        for (name, pcfg) in providers {
            if let Some(overrides) = pcfg.get("modelOverrides").and_then(Value::as_object) {
                for (selector, patch) in overrides {
                    // Selectors are "provider/id" but historically also bare ids.
                    let (prov, id) = match selector.split_once('/') {
                        Some((p, i)) => (p.to_string(), i.to_string()),
                        None => (name.clone(), selector.clone()),
                    };
                    out.push((prov, id, patch.clone()));
                }
            }
        }
    }
    out
}

/// Apply `modelOverrides` patches onto matching catalog rows (TS registry
/// merge). Selectors name the *originating* provider, but a model may live in
/// the catalog under another provider's cache (e.g. an override written for
/// `inception/mercury-2.5` applies to the row reachable through any provider),
/// so a provider miss falls back to an id-only match.
fn apply_overrides(catalog: &mut BTreeMap<(String, String), ListedModel>, overrides: &[(String, String, Value)]) {
    for (provider, id, patch) in overrides {
        let key = (provider.clone(), id.clone());
        let target = if catalog.contains_key(&key) {
            key
        } else {
            // Model ids may themselves contain slashes (openrouter-style
            // "inception/mercury-2.5"), so match the raw selector against the
            // id alone or the "provider/id" pair before giving up.
            let selector = format!("{provider}/{id}");
            let Some(found) = catalog
                .keys()
                .find(|(prov, cid)| cid == &selector || selector == format!("{prov}/{cid}"))
                .cloned()
            else {
                continue;
            };
            found
        };
        let Some(listed) = catalog.get_mut(&target) else { continue };
        let m = &mut listed.model;
        if let Some(v) = patch.get("contextWindow").and_then(Value::as_u64) {
            m.context_window = v;
            m.max_tokens_explicit = true;
        }
        if let Some(v) = patch.get("maxTokens").and_then(Value::as_u64) {
            m.max_tokens = v;
            m.max_tokens_explicit = true;
        }
        if let Some(v) = patch.get("name").and_then(Value::as_str) {
            m.name = v.to_string();
        }
        if let Some(v) = patch.get("baseUrl").and_then(Value::as_str) {
            m.base_url = v.to_string();
        }
        if let Some(v) = patch.get("reasoning").and_then(Value::as_bool) {
            m.reasoning = v;
        }
        if let Some(v) = patch.get("cost").and_then(|c| serde_json::from_value::<pa_types::ai::ModelCost>(c.clone()).ok()) {
            m.cost = v;
        }
    }
}

/// TS availability: `models.yml` providers ∪ credential-backed providers.
fn available_providers(models_yml: &Path, agent_db: &Path) -> std::collections::BTreeSet<String> {
    let mut set: std::collections::BTreeSet<String> = yml_custom_models(models_yml)
        .into_iter()
        .map(|(provider, _)| provider)
        .collect();
    if let Ok(conn) = rusqlite::Connection::open_with_flags(
        agent_db,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    ) {
        if let Ok(mut stmt) = conn.prepare("SELECT DISTINCT provider FROM auth_credentials WHERE disabled_cause IS NULL") {
            let rows = stmt.query_map([], |r| r.get::<_, String>(0));
            if let Ok(rows) = rows {
                for row in rows.filter_map(std::result::Result::ok) {
                    set.insert(row);
                }
            }
        }
    }
    for provider in pa_ai::models_generated::get_providers() {
        if pa_ai::env_api_keys::get_env_api_key(provider).is_some() {
            set.insert(provider.to_string());
        }
    }
    set
}

/// Read every `model_cache` row of `models.db`: `(provider_id, model entries)`.
fn db_cache_models(db: &Path) -> Vec<(String, Vec<Value>)> {
    let Ok(conn) = rusqlite::Connection::open_with_flags(
        db,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    ) else {
        return Vec::new();
    };
    let mut stmt = match conn.prepare("SELECT provider_id, models FROM model_cache ORDER BY updated_at DESC") {
        Ok(s) => s,
        Err(_) => return Vec::new(),
    };
    let rows = stmt
        .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))
        .map(|rows| rows.filter_map(std::result::Result::ok).collect::<Vec<_>>())
        .unwrap_or_default();
    rows.into_iter()
        .filter_map(|(id, json)| {
            serde_json::from_str::<Value>(&json)
                .ok()
                .and_then(|v| match v {
                    Value::Array(a) => Some(a),
                    _ => None,
                })
                .map(|a| (id, a))
        })
        .collect()
}

/// `providers.<name>.models[]` entries from `models.yml` (TS `Config.providers`).
fn yml_custom_models(path: &Path) -> Vec<(String, Value)> {
    let Ok(text) = std::fs::read_to_string(path) else { return Vec::new() };
    let Ok(Value::Object(root)) = serde_yaml_ng::from_str::<serde_yaml_ng::Value>(&text).map(yaml_to_json) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    if let Some(providers) = root.get("providers").and_then(Value::as_object) {
        for (name, pcfg) in providers {
            if let Some(models) = pcfg.get("models").and_then(Value::as_array) {
                for entry in models {
                    out.push((name.clone(), entry.clone()));
                }
            }
        }
    }
    out
}

/// Build a `Model` from a `models.yml` custom entry (TS `custom_model` defaults).
fn custom_model(provider: &str, entry: &Value) -> Option<Model> {
    let id = entry.get("id").and_then(Value::as_str)?.to_string();
    let name = entry.get("name").and_then(Value::as_str).unwrap_or(&id).to_string();
    let num = |key: &str| entry.get(key).and_then(Value::as_u64).unwrap_or(0);
    let input = entry
        .get("input")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|v| match v.as_str()? {
                    "image" => Some(ModelInput::Image),
                    _ => Some(ModelInput::Text),
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| vec![ModelInput::Text]);
    let zero = pa_types::JsNumber(0.0);
    let cost = entry
        .get("cost")
        .and_then(|c| serde_json::from_value::<pa_types::ai::ModelCost>(c.clone()).ok())
        .unwrap_or(pa_types::ai::ModelCost {
            input: zero,
            output: zero,
            cache_read: zero,
            cache_write: zero,
        });
    Some(Model {
        id,
        name,
        api: "openai-completions".to_string(),
        provider: provider.to_string(),
        base_url: String::new(),
        reasoning: entry.get("reasoning").and_then(Value::as_bool).unwrap_or(false),
        thinking_level_map: None,
        input,
        cost,
        context_window: num("contextWindow"),
        max_tokens: num("maxTokens"),
        max_tokens_explicit: true,
        featured: None,
        headers: None,
        compat: None,
    })
}

/// TS `getSupportedEfforts` display name for a thinking level (wire name).
fn level_name(level: &pa_types::ai::ModelThinkingLevel) -> String {
    level.wire_name().to_string()
}

/// TS `formatNumber`: compact K/M/B suffixes (`262K`, `8.2K`, `1M`).
fn format_number(n: u64) -> String {
    let f = n as f64;
    let trim1 = |x: f64| {
        let s = format!("{x:.1}");
        match s.strip_suffix(".0") {
            Some(stripped) => stripped.to_string(),
            None => s,
        }
    };
    if n < 1_000 {
        n.to_string()
    } else if n < 10_000 {
        format!("{}K", trim1(f / 1_000.0))
    } else if n < 1_000_000 {
        format!("{}K", (f / 1_000.0).round() as u64)
    } else if n < 10_000_000 {
        format!("{}M", trim1(f / 1_000_000.0))
    } else if n < 1_000_000_000 {
        format!("{}M", (f / 1_000_000.0).round() as u64)
    } else if n < 10_000_000_000 {
        format!("{}B", trim1(f / 1_000_000_000.0))
    } else {
        format!("{}B", (f / 1_000_000_000.0).round() as u64)
    }
}

fn format_limit(n: u64) -> String {
    if n == 0 { "-".to_string() } else { format_number(n) }
}

/// TS `sendsImageInputOnWire`: declared image input minus `compat.stripImageInput`.
fn sends_image_input(m: &Model) -> bool {
    let declared = m.input.contains(&ModelInput::Image);
    let stripped = m
        .compat
        .as_ref()
        .and_then(|c| c.raw.get("stripImageInput"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    declared && !stripped
}

fn filter_models<'a>(
    catalog: &'a BTreeMap<(String, String), ListedModel>,
    action: Action,
    pattern: Option<&str>,
) -> Vec<&'a ListedModel> {
    let mut all: Vec<&ListedModel> = catalog.values().collect();
    let Some(needle) = pattern.map(str::to_lowercase) else { return all };
    let exact_found = action != Action::Find
        && all.iter().any(|m| m.model.provider.to_lowercase() == needle);
    if exact_found {
        all.retain(|m| m.model.provider.to_lowercase() == needle);
    } else {
        all.retain(|m| {
            let m = &m.model;
            m.id.to_lowercase().contains(&needle)
                || m.provider.to_lowercase().contains(&needle)
                || format!("{}/{}", m.provider, m.id).to_lowercase().contains(&needle)
                || m.name.to_lowercase().contains(&needle)
        });
    }
    all
}

/// TS `localeCompare` (ICU root as resolved by bun/Node): case-insensitive,
/// with primary weights `_` < `-` < `:` < `.` < `@` < `/` < digits < letters
/// (measured empirically) and digit runs compared by numeric value
/// (`k10` < `k9` is false — numeric runs sort numerically, so `k9` < `k10`).
fn ts_cmp(a: &str, b: &str) -> std::cmp::Ordering {
    collation_key(a).cmp(&collation_key(b)).then_with(|| a.cmp(b))
}

#[derive(PartialEq, Eq, PartialOrd, Ord)]
enum Element {
    /// Punctuation and symbols below digits (weight per measured DUCET order).
    Punct(u8),
    /// Digits, ordered by codepoint (`20250514` < `5` because `2` < `5`).
    Digit(u8),
    /// ASCII letters.
    Letter(u8),
    /// Anything else, by codepoint, slotted between symbols and digits.
    Other(u32),
}

fn collation_key(s: &str) -> Vec<Element> {
    fn punct_rank(c: char) -> Option<u8> {
        Some(match c {
            '_' => 1,
            '-' => 2,
            ':' => 3,
            '.' => 4,
            '@' => 5,
            '/' => 6,
            '~' => 7,
            _ => return None,
        })
    }
    s.chars()
        .map(|c| {
            let lower = c.to_lowercase().next().unwrap_or(c);
            if lower.is_ascii_digit() {
                Element::Digit(lower as u8 - b'0')
            } else if lower.is_ascii_alphabetic() {
                Element::Letter(lower as u8)
            } else if let Some(rank) = punct_rank(lower) {
                Element::Punct(rank)
            } else {
                Element::Other(lower as u32)
            }
        })
        .collect()
}

fn render(catalog: &BTreeMap<(String, String), ListedModel>, action: Action, pattern: Option<&str>, json: bool) {
    let filtered = filter_models(catalog, action, pattern);
    let mut sorted = filtered;
    sorted.sort_by(|a, b| {
        ts_cmp(&a.model.provider, &b.model.provider).then(ts_cmp(&a.model.id, &b.model.id))
    });

    if json {
        let models: Vec<Value> = sorted.iter().map(model_json).collect();
        println!("{}", json!({ "models": models }));
        return;
    }
    if catalog.is_empty() {
        println!("No models available. Set API keys in environment variables.");
        return;
    }
    if sorted.is_empty() {
        println!("No models matching {:?}", pattern.unwrap_or_default());
        return;
    }

    let mut by_provider: BTreeMap<&str, Vec<&ListedModel>> = BTreeMap::new();
    for m in sorted {
        by_provider.entry(&m.model.provider).or_default().push(m);
    }
    let mut first = true;
    for (provider, models) in by_provider {
        if !first {
            println!();
        }
        first = false;
        println!("{provider} ({})", models.len());
        let rows: Vec<Vec<String>> = models
            .iter()
            .map(|m| {
                let thinking = match &m.efforts {
                    Some(efforts) if !efforts.is_empty() => efforts.join(","),
                    _ if m.model.reasoning => "yes".to_string(),
                    _ => "-".to_string(),
                };
                vec![
                    m.model.id.clone(),
                    format_limit(m.model.context_window),
                    format_limit(m.model.max_tokens),
                    thinking,
                    if sends_image_input(&m.model) { "yes".to_string() } else { "no".to_string() },
                ]
            })
            .collect();
        for line in box_table(
            &[
                ("model", false),
                ("context", true),
                ("max-out", true),
                ("thinking", false),
                ("images", false),
            ],
            &rows,
        ) {
            println!("{line}");
        }
    }
}

/// TS `toModelJson`.
fn model_json(m: &&ListedModel) -> Value {
    let model = &m.model;
    json!({
        "provider": model.provider,
        "kind": "chat",
        "id": model.id,
        "selector": format!("{}/{id}", model.provider, id = model.id),
        "name": model.name,
        "contextWindow": model.context_window,
        "maxTokens": model.max_tokens,
        "reasoning": model.reasoning,
        "thinking": m.efforts,
        "input": model
            .input
            .iter()
            .map(|i| match i { ModelInput::Text => "text", ModelInput::Image => "image" })
            .collect::<Vec<_>>(),
        "cost": model.cost,
        "pricingStatus": if model.cost.input.0 > 0.0 || model.cost.output.0 > 0.0 { "priced" } else { "unknown" },
    })
}

/// TS `boxTable`: one box-drawing table; header row and plain cells (no ANSI here).
fn box_table(columns: &[(&str, bool)], rows: &[Vec<String>]) -> Vec<String> {
    let widths: Vec<usize> = columns
        .iter()
        .enumerate()
        .map(|(i, (header, _))| {
            rows.iter()
                .map(|row| row.get(i).map_or(0, String::len))
                .chain(std::iter::once(header.len()))
                .max()
                .unwrap_or(0)
        })
        .collect();
    let bar = "│";
    let segments: Vec<String> = widths.iter().map(|w| "─".repeat(w + 2)).collect();
    let render_row = |cells: &[String]| {
        let padded: Vec<String> = columns
            .iter()
            .enumerate()
            .map(|(i, (_, right))| {
                let cell = cells.get(i).map_or("", String::as_str);
                let space = widths[i].saturating_sub(cell.len());
                if *right {
                    format!("{}{cell}", " ".repeat(space))
                } else {
                    format!("{cell}{}", " ".repeat(space))
                }
            })
            .collect();
        format!("{bar} {} {bar}", padded.join(&format!(" {bar} ")))
    };
    let mut lines = vec![format!("┌{}┐", segments.join("┬"))];
    lines.push(render_row(
        &columns.iter().map(|(header, _)| (*header).to_string()).collect::<Vec<_>>(),
    ));
    lines.push(format!("├{}┤", segments.join("┼")));
    for row in rows {
        lines.push(render_row(row));
    }
    lines.push(format!("└{}┘", segments.join("┴")));
    lines
}

/// Convert a YAML value to JSON (TS treats both as the same document model).
fn yaml_to_json(value: serde_yaml_ng::Value) -> Value {
    match value {
        serde_yaml_ng::Value::Null => Value::Null,
        serde_yaml_ng::Value::Bool(b) => Value::Bool(b),
        serde_yaml_ng::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::Number(i.into())
            } else if let Some(u) = n.as_u64() {
                Value::Number(u.into())
            } else {
                n.as_f64()
                    .and_then(serde_json::Number::from_f64)
                    .map_or(Value::Null, Value::Number)
            }
        }
        serde_yaml_ng::Value::String(s) => Value::String(s),
        serde_yaml_ng::Value::Sequence(seq) => Value::Array(seq.into_iter().map(yaml_to_json).collect()),
        serde_yaml_ng::Value::Mapping(map) => Value::Object(
            map.into_iter()
                .filter_map(|(k, v)| k.as_str().map(|key| (key.to_string(), yaml_to_json(v))))
                .collect::<Map<String, Value>>(),
        ),
        serde_yaml_ng::Value::Tagged(tagged) => yaml_to_json(tagged.value),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_number_matches_ts() {
        assert_eq!(format_number(0), "0");
        assert_eq!(format_number(999), "999");
        assert_eq!(format_number(1000), "1K");
        assert_eq!(format_number(8192), "8.2K");
        assert_eq!(format_number(200_000), "200K");
        assert_eq!(format_number(262_144), "262K");
        assert_eq!(format_number(999_999), "1000K");
        assert_eq!(format_number(1_000_000), "1M");
        assert_eq!(format_number(9_500_000), "9.5M");
    }

    #[test]
    fn format_limit_zero_is_dash() {
        assert_eq!(format_limit(0), "-");
        assert_eq!(format_limit(8192), "8.2K");
    }

    #[test]
    fn resolve_args_falls_through_to_ls() {
        let (action, pattern) = resolve_args(Some("find"), Some("gpt"));
        assert_eq!(action, Action::Find);
        assert_eq!(pattern, Some("gpt"));
        let (action, pattern) = resolve_args(Some("openai"), None);
        assert_eq!(action, Action::Ls);
        assert_eq!(pattern, Some("openai"));
        let (action, pattern) = resolve_args(None, None);
        assert_eq!(action, Action::Ls);
        assert_eq!(pattern, None);
    }

    #[test]
    fn box_table_shapes_borders() {
        let lines = box_table(
            &[("model", false), ("context", true)],
            &[vec!["a".to_string(), format_limit(8192)]],
        );
        assert_eq!(lines[0], "┌───────┬─────────┐");
        assert!(lines[1].starts_with("│ model"));
        assert_eq!(lines[2], "├───────┼─────────┤");
        assert!(lines[3].contains("8.2K"));
        assert_eq!(lines[4], "└───────┴─────────┘");
    }

    #[test]
    fn yaml_numbers_convert() {
        assert_eq!(yaml_to_json(serde_yaml_ng::from_str("42").unwrap()), Value::Number(42.into()));
        assert_eq!(
            yaml_to_json(serde_yaml_ng::from_str("0.5").unwrap()),
            Value::Number(serde_json::Number::from_f64(0.5).unwrap())
        );
    }
}
