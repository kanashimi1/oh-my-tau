//! Model selection via `omp-providers` (`models.yml` / `config.yml` roles /
//! agent.db credentials). Setting `OMP_RS_BASE_URL` switches to an explicit
//! openai-completions endpoint (`OMP_RS_API_KEY` / `OMP_RS_MODEL`).

use anyhow::{anyhow, Result};
use pa_agent::types::ThinkingLevel;
use serde_json::json;
use std::future::Future;

pub struct Selected {
    pub model: pa_types::ai::Model,
    pub api_key: Option<String>,
    pub thinking: ThinkingLevel,
    pub headers: Option<std::collections::BTreeMap<String, String>>,
    /// Stored `auth_credentials` row `api_key` came from; see `ProviderTarget::credential_id`.
    pub credential_id: Option<i64>,
    /// The `--model` chain entries after the selected one: the runtime retry
    /// fallback chain of this model (TS `installSubagentRetryFallbackChain`), so a
    /// quota-exhausted 429 on the winner switches to the next entry mid-run.
    pub fallbacks: Vec<String>,
}

fn map_level(l: pa_types::ai::ModelThinkingLevel) -> ThinkingLevel {
    crate::adapter::map_thinking_level(l)
}

async fn from_providers(reference: String) -> Result<Selected> {
    let r = omp_providers::resolve_async(reference).await?;
    let headers = (!r.headers.is_empty()).then(|| r.headers.iter().cloned().collect());
    Ok(Selected {
        // No explicit level: reasoning models default to medium (some, e.g. zai
        // glm, reject thinking=off), others stay off.
        thinking: r.thinking.map_or(
            if r.reasoning {
                ThinkingLevel::Medium
            } else {
                ThinkingLevel::Off
            },
            map_level,
        ),
        api_key: r.api_key.clone(),
        credential_id: r.credential_id,
        model: r.model,
        headers,
        fallbacks: Vec::new(),
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
        return select_chain(&reference).await;
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
        credential_id: None,
        fallbacks: Vec::new(),
    })
}

/// Ordered candidates of a model reference (TS `normalizeModelPatternList`):
/// a comma-separated reference is a priority list, first entry first. A
/// leading `@` marks a role alias (TS `MODEL_ROLE_ALIAS_PREFIX`); the bare
/// role name is what `modelRoles` lookup takes, so `@smol:high` becomes
/// `smol:high`.
fn model_candidates(reference: &str) -> Vec<String> {
    reference
        .split(',')
        .map(str::trim)
        .filter(|candidate| !candidate.is_empty())
        .map(|candidate| candidate.strip_prefix('@').unwrap_or(candidate))
        .map(str::to_owned)
        .collect()
}

/// Resolve one reference by walking its ordered candidate list: the first
/// entry whose provider has credentials wins — TS `resolveModelOverride`
/// skips models whose provider cannot be called — and an entry that resolves
/// without credentials is used only when no credentialed entry resolved, so
/// keyless-by-design local providers stay selectable. When nothing resolved,
/// the first failure surfaces (the requested model is never silently
/// substituted with the config default).
async fn select_chain(reference: &str) -> Result<Selected> {
    select_chain_with(reference, from_providers).await
}

/// [`select_chain`] with the per-candidate resolver injected (tests substitute
/// a fake; production resolves through [`from_providers`]).
async fn select_chain_with<F, Fut>(reference: &str, resolve: F) -> Result<Selected>
where
    F: Fn(String) -> Fut,
    Fut: Future<Output = Result<Selected>>,
{
    let candidates = model_candidates(reference);
    let rest = |index: usize| candidates[index + 1..].to_vec();
    let mut resolved: Option<(usize, Selected)> = None;
    let mut failure: Option<anyhow::Error> = None;
    for (index, candidate) in candidates.iter().enumerate() {
        match resolve(candidate.clone()).await {
            Ok(mut selected) if selected.api_key.is_some() => {
                selected.fallbacks = rest(index);
                return Ok(selected);
            }
            Ok(selected) => {
                if resolved.is_none() {
                    resolved = Some((index, selected));
                }
            }
            Err(error) => {
                if failure.is_none() {
                    failure = Some(error);
                }
            }
        }
    }
    match resolved {
        Some((index, mut selected)) => {
            selected.fallbacks = rest(index);
            Ok(selected)
        }
        None => Err(failure.unwrap_or_else(|| anyhow!("no model reference in {reference:?}"))),
    }
}

fn catalog_path() -> Option<std::path::PathBuf> {
    if let Some(p) = std::env::var_os("OMP_RS_CATALOG") {
        return Some(p.into());
    }
    let home = std::env::var_os("HOME")?;
    Some(
        std::path::PathBuf::from(home)
            .join(".bun/install/global/node_modules/@oh-my-pi/pi-catalog/src/models.json"),
    )
}

/// Compact view of one catalog model: only the `CATALOG_EXTRA_KEYS` fields survive the
/// one-shot build, and every non-bool value is an index into the store's single intern table
/// (identity JSON, tokenizer names, thinking levels/modes — a few hundred distinct strings
/// process-wide) or the raw JSON number token (`int`/`tps`, preserving the int/float
/// distinction `serde_json::Value` makes). The whole 5618-entry store retains well under
/// 1 MB, versus ~80 MB for the original process-lifetime `Value` tree (~23 MB for cloned
/// per-entry `Value`s).
#[derive(Default)]
struct CatalogEntry {
    int: Option<Str>,
    tps: Option<Str>,
    thinking: Option<Str>,
    identity: Option<Str>,
    requires_glyph_tokenization: Option<Str>,
    tokenizer: Option<Str>,
    supports_computer_use: Option<Str>,
}

/// Index into [`CatalogStore::strings`]; `Copy` so extras building borrows cheaply.
type Str = u32;

/// Owner of the interned strings and the per-provider entry maps: one process-lifetime
/// static holding every allocation the catalog needs.
struct CatalogStore {
    strings: Vec<String>,
    providers: std::collections::HashMap<String, std::collections::HashMap<String, CatalogEntry>>,
}

impl CatalogStore {
    fn str(&self, s: Str) -> &str {
        &self.strings[s as usize]
    }

    /// Deduplicate `s` into the intern table, returning its index.
    fn intern(
        strings: &mut Vec<String>,
        seen: &mut std::collections::HashMap<String, Str>,
        s: String,
    ) -> Str {
        if let Some(&i) = seen.get(&s) {
            return i;
        }
        // The table holds a few hundred entries; degrade to a duplicate index only past
        // 4G entries (impossible for the 5618-model catalog).
        let idx = u32::try_from(strings.len()).unwrap_or(u32::MAX);
        seen.insert(s.clone(), idx);
        strings.push(s);
        idx
    }

    /// Present extras of `provider`/`id` in `CATALOG_EXTRA_KEYS` order (catalog key order).
    fn extras(&self, provider: &str, id: &str) -> Option<Vec<(&'static str, serde_json::Value)>> {
        let entry = self.providers.get(provider)?.get(id)?;
        let mut out = Vec::new();
        let push =
            |out: &mut Vec<(&'static str, serde_json::Value)>, k: &'static str, raw: &str| {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(raw) {
                    out.push((k, v));
                }
            };
        if let Some(raw) = &entry.int {
            push(&mut out, "int", self.str(*raw));
        }
        if let Some(raw) = &entry.tps {
            push(&mut out, "tps", self.str(*raw));
        }
        if let Some(raw) = &entry.thinking {
            push(&mut out, "thinking", self.str(*raw));
        }
        if let Some(raw) = &entry.identity {
            push(&mut out, "identity", self.str(*raw));
        }
        if let Some(raw) = &entry.requires_glyph_tokenization {
            push(&mut out, "requiresGlyphTokenization", self.str(*raw));
        }
        if let Some(raw) = &entry.tokenizer {
            push(&mut out, "tokenizer", self.str(*raw));
        }
        if let Some(raw) = &entry.supports_computer_use {
            push(&mut out, "supportsComputerUse", self.str(*raw));
        }
        (!out.is_empty()).then_some(out)
    }
}

fn catalog() -> Option<&'static CatalogStore> {
    static CATALOG: std::sync::LazyLock<Option<CatalogStore>> =
        std::sync::LazyLock::new(build_catalog);
    CATALOG.as_ref()
}

/// Owned view of one catalog entry: `Box<RawValue>` copies out the raw JSON text of each
/// extra field (no DOM, no whole-file buffer), and serde skips everything else at zero
/// retained cost.
#[derive(serde::Deserialize)]
struct RawEntry {
    int: Option<Box<serde_json::value::RawValue>>,
    tps: Option<Box<serde_json::value::RawValue>>,
    thinking: Option<Box<serde_json::value::RawValue>>,
    identity: Option<Box<serde_json::value::RawValue>>,
    #[serde(rename = "requiresGlyphTokenization")]
    requires_glyph_tokenization: Option<Box<serde_json::value::RawValue>>,
    tokenizer: Option<Box<serde_json::value::RawValue>>,
    #[serde(rename = "supportsComputerUse")]
    supports_computer_use: Option<Box<serde_json::value::RawValue>>,
}

/// Streaming build: `from_reader` pulls `models.json` straight from the file (the 11.8 MB
/// never sits in the allocator — `fs::read` + `from_slice` left the freed buffer resident
/// under mimalloc), and the visitors below push each entry into the interned store with no
/// intermediate maps. Only [`CatalogStore`] survives — no full-catalog `Value` tree is ever
/// constructed.
fn build_catalog() -> Option<CatalogStore> {
    use serde::de::{DeserializeSeed, MapAccess, Visitor};
    use serde::Deserializer as _;

    struct CatalogVisitor<'a> {
        store: &'a mut CatalogStore,
        seen: &'a mut std::collections::HashMap<String, Str>,
    }

    impl<'de> Visitor<'de> for CatalogVisitor<'_> {
        type Value = ();

        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("a catalog of providers to model entries")
        }

        fn visit_map<A: MapAccess<'de>>(self, mut m: A) -> Result<(), A::Error> {
            while let Some(provider) = m.next_key::<String>()? {
                let mut entries = std::collections::HashMap::new();
                m.next_value_seed(ModelsVisitor {
                    store: self.store,
                    seen: self.seen,
                    entries: &mut entries,
                })?;
                self.store.providers.insert(provider, entries);
            }
            Ok(())
        }
    }

    struct ModelsVisitor<'a> {
        store: &'a mut CatalogStore,
        seen: &'a mut std::collections::HashMap<String, Str>,
        entries: &'a mut std::collections::HashMap<String, CatalogEntry>,
    }

    impl<'de> DeserializeSeed<'de> for ModelsVisitor<'_> {
        type Value = ();

        fn deserialize<D: serde::Deserializer<'de>>(self, d: D) -> Result<Self::Value, D::Error> {
            d.deserialize_map(self)
        }
    }

    impl<'de> Visitor<'de> for ModelsVisitor<'_> {
        type Value = ();

        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("a provider's model entries")
        }

        fn visit_map<A: MapAccess<'de>>(self, mut m: A) -> Result<(), A::Error> {
            while let Some(id) = m.next_key::<String>()? {
                let raw = m.next_value::<RawEntry>()?;
                let mut interned = |rv: Option<Box<serde_json::value::RawValue>>| {
                    rv.map(|rv| {
                        CatalogStore::intern(
                            &mut self.store.strings,
                            self.seen,
                            rv.get().to_string(),
                        )
                    })
                };
                let entry = CatalogEntry {
                    int: interned(raw.int),
                    tps: interned(raw.tps),
                    thinking: interned(raw.thinking),
                    identity: interned(raw.identity),
                    requires_glyph_tokenization: interned(raw.requires_glyph_tokenization),
                    tokenizer: interned(raw.tokenizer),
                    supports_computer_use: interned(raw.supports_computer_use),
                };
                self.entries.insert(id, entry);
            }
            Ok(())
        }
    }

    let file = std::fs::File::open(catalog_path()?).ok()?;
    let mut store = CatalogStore {
        strings: Vec::new(),
        providers: std::collections::HashMap::new(),
    };
    let mut seen: std::collections::HashMap<String, Str> = std::collections::HashMap::new();
    let mut de = serde_json::Deserializer::from_reader(std::io::BufReader::new(file));
    de.deserialize_map(CatalogVisitor {
        store: &mut store,
        seen: &mut seen,
    })
    .ok()?;
    Some(store)
}

/// `int`/`tps`/`thinking`/`identity`/`tokenizer`/... of `provider/id` from omp's bundled
/// catalog (`pi-catalog/src/models.json`; override the path with `OMP_RS_CATALOG`).
/// `None` when the catalog is not installed or lists no such model.
#[must_use]
pub fn catalog_extras(provider: &str, id: &str) -> Option<Vec<(&'static str, serde_json::Value)>> {
    catalog()?.extras(provider, id)
}

fn parse_effort(s: &str) -> Option<ThinkingLevel> {
    parse_thinking(s)
}

/// Catalog `thinking.efforts` of `provider/id` (least to most intensive), when listed.
#[must_use]
pub fn catalog_efforts(provider: &str, id: &str) -> Option<Vec<ThinkingLevel>> {
    let thinking = catalog_extras(provider, id)?
        .into_iter()
        .find(|(k, _)| *k == "thinking")?
        .1;
    let efforts: Vec<_> = thinking
        .get("efforts")?
        .as_array()?
        .iter()
        .filter_map(|e| parse_effort(e.as_str()?))
        .collect();
    (!efforts.is_empty()).then_some(efforts)
}

/// Lowest effort of a model that cannot think "off" (`thinking.requiresEffort`).
#[must_use]
pub fn always_thinking_floor(provider: &str, id: &str) -> Option<ThinkingLevel> {
    let thinking = catalog_extras(provider, id)?
        .into_iter()
        .find(|(k, _)| *k == "thinking")?
        .1;
    if thinking.get("requiresEffort") != Some(&serde_json::Value::Bool(true)) {
        return None;
    }
    catalog_efforts(provider, id)?.into_iter().next()
}

fn ladder_index(l: ThinkingLevel) -> usize {
    [
        ThinkingLevel::Off,
        ThinkingLevel::Minimal,
        ThinkingLevel::Low,
        ThinkingLevel::Medium,
        ThinkingLevel::High,
        ThinkingLevel::Xhigh,
        ThinkingLevel::Max,
    ]
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
            .or_else(|| {
                efforts
                    .iter()
                    .rev()
                    .copied()
                    .find(|e| ladder_index(*e) < ladder_index(level))
            })
            .unwrap_or(level),
        Some(_) => level,
        None => map_level(pa_types::ai::clamp_thinking_level(
            model,
            crate::adapter::model_thinking_level(level),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Catalog metadata omp's `Model` carries beyond pa-ai's descriptor, in catalog key order.
    const CATALOG_EXTRA_KEYS: [&str; 7] = [
        "int",
        "tps",
        "thinking",
        "identity",
        "requiresGlyphTokenization",
        "tokenizer",
        "supportsComputerUse",
    ];

    #[test]
    fn model_reference_splits_into_ordered_candidates() {
        assert_eq!(
            model_candidates("anthropic/claude-haiku-5-5, kimi-code/kimi-k2.6"),
            vec![
                "anthropic/claude-haiku-5-5".to_string(),
                "kimi-code/kimi-k2.6".to_string()
            ]
        );
        assert_eq!(
            model_candidates("zai/glm-5.3-flash:high"),
            ["zai/glm-5.3-flash:high".to_string()]
        );
        assert!(model_candidates("  ,  ,").is_empty());
    }

    /// A role alias keeps its `:level` suffix and drops the `@` marker
    /// (`@smol` -> the `modelRoles.smol` selector, `@default` -> the default
    /// role).
    #[test]
    fn model_reference_expands_role_aliases() {
        assert_eq!(
            model_candidates("@smol:high, @default"),
            vec!["smol:high".to_string(), "default".to_string()]
        );
    }

    /// A `--model` chain walks its entries in order and skips entries whose
    /// provider has no credentials (TS `resolveModelOverride`).
    #[tokio::test]
    async fn chain_prefers_the_first_credentialed_candidate() {
        async fn resolve(reference: String) -> Result<Selected> {
            match reference.as_str() {
                "local/keyless" => Ok(fake_selected("keyless", None)),
                "zai/glm-5.3-flash" => Ok(fake_selected("glm", Some("key"))),
                other => Err(anyhow!("model '{other}' not found")),
            }
        }
        // Credentialed entry wins even when a keyless one comes first.
        let selected = select_chain_with("local/keyless, zai/glm-5.3-flash", resolve)
            .await
            .expect("chain resolves");
        assert_eq!(selected.model.id, "glm");
        assert_eq!(selected.api_key.as_deref(), Some("key"));
        assert!(selected.fallbacks.is_empty(), "last entry has no fallbacks");
        // The entries after the winner become its runtime fallback chain.
        let selected = select_chain_with("zai/glm-5.3-flash, local/keyless, @smol:high", resolve)
            .await
            .expect("chain resolves");
        assert_eq!(selected.model.id, "glm");
        assert_eq!(selected.fallbacks, vec!["local/keyless".to_string(), "smol:high".to_string()]);
        // No credentialed entry: the first resolvable entry is kept so
        // keyless-by-design local providers stay selectable.
        let selected = select_chain_with("local/keyless, nope/model", resolve)
            .await
            .expect("chain resolves");
        assert_eq!(selected.model.id, "keyless");
        assert_eq!(selected.api_key, None);
        // Nothing resolved: the first failure surfaces, never a silent
        // substitution.
        let error = match select_chain_with("nope/one, nope/two", resolve).await {
            Ok(selected) => panic!("unresolvable chain resolved {}", selected.model.id),
            Err(error) => error,
        };
        assert!(error.to_string().contains("nope/one"), "{error}");
        assert!(select_chain_with(" , ", resolve).await.is_err());
    }

    fn fake_selected(id: &str, api_key: Option<&str>) -> Selected {
        let model: pa_types::ai::Model = serde_json::from_value(json!({
            "id": id,
            "name": id,
            "api": "openai-completions",
            "provider": "local",
            "baseUrl": "http://127.0.0.1:1/v1",
            "reasoning": false,
            "input": ["text"],
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
            "contextWindow": 1_000,
            "maxTokens": 100,
        }))
        .expect("fake model");
        Selected {
            model,
            api_key: api_key.map(str::to_owned),
            thinking: ThinkingLevel::Off,
            headers: None,
            credential_id: None,
            fallbacks: Vec::new(),
        }
    }

    /// Reference: untyped `Value` parse of the catalog (the pre-fix implementation),
    /// reduced to the `CATALOG_EXTRA_KEYS` fields. Must cover every provider/id of the
    /// real 5618-model `models.json`, and the typed [`Catalog`] must agree exactly.
    #[test]
    fn catalog_matches_value_reference_parse() {
        let Some(path) = catalog_path().filter(|p| p.is_file()) else {
            eprintln!("catalog not installed; skipping");
            return;
        };
        let bytes = std::fs::read(&path).expect("read catalog");
        let reference: serde_json::Value = serde_json::from_slice(&bytes).expect("value parse");
        let reference = reference.as_object().expect("catalog is an object");
        let mut models = 0usize;
        for (provider, ids) in reference {
            let ids = ids.as_object().expect("provider is an object");
            for (id, entry) in ids {
                models += 1;
                let expected: Vec<(&'static str, serde_json::Value)> = CATALOG_EXTRA_KEYS
                    .iter()
                    .filter_map(|k| {
                        entry
                            .get(*k)
                            .map(|v| (*k, v.clone()))
                            .filter(|(_, v)| !v.is_null())
                    })
                    .collect();
                let got = catalog()
                    .and_then(|c| c.extras(provider, id))
                    .unwrap_or_default();
                assert_eq!(
                    expected, got,
                    "mismatch for {provider}/{id}: typed catalog parse diverged from Value reference"
                );
                if !expected.is_empty() {
                    assert!(
                        catalog_extras(provider, id).is_some(),
                        "{provider}/{id}: catalog_extras must be Some for a non-empty entry"
                    );
                }
            }
        }
        assert_eq!(models, 5618, "expected the full 5618-model catalog");
        assert!(catalog().is_some(), "typed catalog must be installed");
    }
}
