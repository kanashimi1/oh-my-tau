//! omp-providers: resolve omp model references (`provider/id[:thinking]` or a
//! role name from `config.yml` `modelRoles`) into `pa-ai` models + credentials,
//! and stream through `pa-ai`.
//!
//! Sources, all read-only (nothing is ever written under `~/.omp/agent`):
//! * `models.yml`  - custom providers / `modelOverrides` / provider `baseUrl` overrides
//! * `models.db`   - omp's dynamic model catalog cache (`model_cache`)
//! * `pa-ai` compiled fallback catalog
//! * credentials: `models.yml` `apiKey` -> env vars -> `agent.db` `auth_credentials`
//!   (api keys and non-expired OAuth access tokens; OAuth is never refreshed here).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context as _, Result};
use serde_json::{json, Map, Value};

pub use pa_ai::types::{Context, Model, ModelThinkingLevel, SimpleStreamOptions, StreamOptions};
pub use pa_ai::AssistantMessageEventStream;

/// A model reference resolved to everything needed to call it.
#[derive(Debug, Clone)]
pub struct ResolvedModel {
    pub provider: String,
    pub id: String,
    /// pa-ai api id: "anthropic-messages" | "openai-completions" | "openai-responses" | ...
    pub api: String,
    pub base_url: String,
    pub api_key: Option<String>,
    /// Provider + model extra request headers.
    pub headers: Vec<(String, String)>,
    pub context_window: u64,
    pub max_tokens: u64,
    pub reasoning: bool,
    /// Thinking level from the `:suffix` (or role); `None` = provider default / `auto`.
    pub thinking: Option<ModelThinkingLevel>,
    /// Full pa-ai model descriptor (what `to_pa_model` returns).
    pub model: Model,
}

/// Where omp's config lives. `Default` = `$OMP_AGENT_DIR` or `~/.omp/agent`.
#[derive(Debug, Clone)]
pub struct AgentDir(pub PathBuf);

impl Default for AgentDir {
    fn default() -> Self {
        if let Some(dir) = std::env::var_os("OMP_AGENT_DIR") {
            return Self(PathBuf::from(dir));
        }
        let home = std::env::var_os("HOME").map_or_else(|| PathBuf::from("/"), PathBuf::from);
        Self(home.join(".omp").join("agent"))
    }
}

/// Resolve `model_ref` against the default omp agent dir.
///
/// # Errors
/// Unknown role/provider/model, or unreadable/invalid config files.
pub fn resolve(model_ref: &str) -> Result<ResolvedModel> {
    resolve_in(&AgentDir::default(), model_ref)
}

/// The pa-ai model for streaming.
#[must_use]
pub fn to_pa_model(m: &ResolvedModel) -> Model {
    m.model.clone()
}

/// Start a streaming completion (thin wrapper over `pa_ai::stream_simple`).
/// Fills in the resolved api key, headers and thinking level unless `opts` set them.
///
/// # Errors
/// No pa-ai provider registered for the model's api.
pub async fn stream(
    m: &ResolvedModel,
    context: Context,
    opts: Option<SimpleStreamOptions>,
) -> Result<AssistantMessageEventStream> {
    let mut o = opts.unwrap_or_default();
    if o.base.api_key.is_none() {
        o.base.api_key.clone_from(&m.api_key);
    }
    if o.reasoning.is_none() {
        o.reasoning = m.thinking;
    }
    if !m.headers.is_empty() && o.base.headers.is_none() {
        o.base.headers = Some(m.headers.iter().cloned().collect());
    }
    pa_ai::stream_simple(&m.model, &context, Some(o)).map_err(|e| anyhow!("{e:?}"))
}

/// [`resolve`] with an explicit agent dir (tests, `--agent-dir`).
///
/// # Errors
/// See [`resolve`].
pub fn resolve_in(dir: &AgentDir, model_ref: &str) -> Result<ResolvedModel> {
    let cfg = Config::load(&dir.0)?;
    let (provider, id, thinking) = parse_ref(&cfg, model_ref, 0)?;
    let (mut model, meta) = find_model(&dir.0, &cfg, &provider, &id)?;
    let thinking = adjust_thinking(thinking, meta.as_ref());
    apply_overrides(&cfg, &provider, &mut model);
    // omp "pseudo apis" that are plain OpenAI-compatible chat endpoints.
    if model.api == "openrouter" {
        model.api = "openai-completions".to_string();
    }
    let api_key = cfg
        .provider_key(&provider)
        .or_else(|| env_key(&provider))
        .or_else(|| db_key(&dir.0.join("agent.db"), &provider));
    let mut headers: Vec<(String, String)> = cfg
        .provider(&provider)
        .and_then(|p| p.get("headers"))
        .and_then(Value::as_object)
        .map(|h| h.iter().filter_map(|(k, v)| Some((k.clone(), v.as_str()?.to_string()))).collect())
        .unwrap_or_default();
    if let Some(h) = &model.headers {
        for (k, v) in h {
            headers.retain(|(hk, _)| hk != k);
            headers.push((k.clone(), v.clone()));
        }
    }
    if !headers.is_empty() {
        model.headers = Some(headers.iter().cloned().collect());
    }
    Ok(ResolvedModel {
        provider: model.provider.clone(),
        id: model.id.clone(),
        api: model.api.clone(),
        base_url: model.base_url.clone(),
        api_key,
        headers,
        context_window: model.context_window,
        max_tokens: model.max_tokens,
        reasoning: model.reasoning,
        thinking,
        model,
    })
}

// ---------------------------------------------------------------- ref parsing

fn parse_level(s: &str) -> Option<Option<ModelThinkingLevel>> {
    use ModelThinkingLevel as L;
    Some(match s {
        "auto" | "inherit" => None,
        "off" | "none" => Some(L::Off),
        "minimal" => Some(L::Minimal),
        "low" => Some(L::Low),
        "medium" => Some(L::Medium),
        "high" => Some(L::High),
        "xhigh" => Some(L::Xhigh),
        "max" => Some(L::Max),
        _ => return None,
    })
}

type Parsed = (String, String, Option<ModelThinkingLevel>);

fn parse_ref(cfg: &Config, model_ref: &str, depth: u8) -> Result<Parsed> {
    let model_ref = model_ref.trim();
    let (base, thinking_override) = match model_ref.rsplit_once(':') {
        Some((b, suffix)) => match parse_level(suffix) {
            Some(level) => (b, Some(level)),
            None => (model_ref, None),
        },
        None => (model_ref, None),
    };
    if let Some((provider, id)) = base.split_once('/') {
        if !provider.is_empty() && !id.is_empty() {
            return Ok((provider.to_string(), id.to_string(), thinking_override.flatten()));
        }
    }
    // Role name.
    if depth > 4 {
        bail!("modelRoles alias loop at '{model_ref}'");
    }
    let target = cfg
        .roles
        .get(base)
        .ok_or_else(|| anyhow!("'{model_ref}' is neither 'provider/id' nor a modelRoles role ({:?})", cfg.roles.keys().collect::<Vec<_>>()))?;
    let (p, i, t) = parse_ref(cfg, target, depth + 1)?;
    Ok((p, i, thinking_override.map_or(t, |o| o)))
}

// ---------------------------------------------------------------- config

struct Config {
    roles: BTreeMap<String, String>,
    providers: Map<String, Value>,
}

impl Config {
    fn load(dir: &Path) -> Result<Self> {
        let read_yaml = |name: &str| -> Result<Value> {
            let path = dir.join(name);
            match std::fs::read_to_string(&path) {
                Ok(s) if s.trim().is_empty() => Ok(Value::Null),
                Ok(s) => serde_yaml_ng::from_str::<Value>(&s).with_context(|| format!("parsing {}", path.display())),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Value::Null),
                Err(e) => Err(e).with_context(|| format!("reading {}", path.display())),
            }
        };
        let config = read_yaml("config.yml")?;
        let models = read_yaml("models.yml")?;
        let roles = config
            .get("modelRoles")
            .and_then(Value::as_object)
            .map(|m| m.iter().filter_map(|(k, v)| Some((k.clone(), v.as_str()?.to_string()))).collect())
            .unwrap_or_default();
        let providers = models.get("providers").and_then(Value::as_object).cloned().unwrap_or_default();
        Ok(Self { roles, providers })
    }

    fn provider(&self, p: &str) -> Option<&Map<String, Value>> {
        self.providers.get(p).and_then(Value::as_object)
    }

    /// `apiKey` from models.yml: an env var name if that var is set, else the literal.
    fn provider_key(&self, p: &str) -> Option<String> {
        let raw = self.provider(p)?.get("apiKey")?.as_str()?;
        if raw.is_empty() || raw.starts_with('!') {
            return None; // shell-command keys are not executed by this port
        }
        match std::env::var(raw) {
            Ok(v) if !v.is_empty() => Some(v),
            _ => Some(raw.to_string()),
        }
    }
}

// ---------------------------------------------------------------- model lookup

fn default_api_for(provider: &str) -> &'static str {
    match provider {
        "anthropic" => "anthropic-messages",
        "openai-codex" => "openai-codex-responses",
        "openai" => "openai-responses",
        "google" => "google-generative-ai",
        _ => "openai-completions",
    }
}

fn base_url_for(provider: &str) -> Option<&'static str> {
    Some(match provider {
        "zai" => "https://api.z.ai/api/coding/paas/v4",
        "kimi-code" => "https://api.kimi.com/coding/v1",
        "openrouter" => "https://openrouter.ai/api/v1",
        "opencode-zen" => "http://127.0.0.1:8787/v1",
        "anthropic" => "https://api.anthropic.com",
        "openai" => "https://api.openai.com/v1",
        _ => return None,
    })
}

fn find_model(dir: &Path, cfg: &Config, provider: &str, id: &str) -> Result<(Model, Option<Value>)> {
    let pcfg = cfg.provider(provider);
    // 1. custom model from models.yml
    if let Some(entry) = pcfg
        .and_then(|p| p.get("models"))
        .and_then(Value::as_array)
        .and_then(|a| a.iter().find(|m| m.get("id").and_then(Value::as_str) == Some(id)))
    {
        return Ok((custom_model(provider, pcfg.expect("checked"), entry)?, entry.get("thinking").cloned()));
    }
    // 2. omp's dynamic catalog cache, 3. pa-ai compiled catalog
    let (mut model, meta) = match cache_model(&dir.join("models.db"), provider, id) {
        Some((m, t)) => (Some(m), t),
        None => (pa_ai::models_generated::get_model(provider, id).cloned(), None),
    };
    // 4. provider configured in models.yml (baseUrl) but model unknown: synthesize.
    if model.is_none() {
        if let Some(p) = pcfg {
            if p.get("baseUrl").is_some() {
                let entry = json!({"id": id, "name": id});
                model = Some(custom_model(provider, p, &entry)?);
            }
        }
    }
    let mut model = model.ok_or_else(|| {
        anyhow!("model '{provider}/{id}' not found in models.yml, models.db cache or built-in catalog")
    })?;
    // provider-level overrides apply to catalog models too
    if let Some(p) = pcfg {
        if let Some(u) = p.get("baseUrl").and_then(Value::as_str) {
            model.base_url = u.to_string();
        }
        if let Some(a) = p.get("api").and_then(Value::as_str) {
            model.api = a.to_string();
        }
    }
    Ok((model, meta))
}

/// omp's catalog `thinking` metadata: `efforts` (supported levels), `requiresEffort`
/// (reasoning cannot be disabled) and `defaultLevel`. Clamp the requested level to a
/// supported effort and fill the default when the model requires one.
fn adjust_thinking(level: Option<ModelThinkingLevel>, meta: Option<&Value>) -> Option<ModelThinkingLevel> {
    use ModelThinkingLevel as L;
    let Some(meta) = meta else { return level };
    let efforts: Vec<L> = meta
        .get("efforts")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(|e| parse_level(e.as_str()?).flatten()).collect())
        .unwrap_or_default();
    let requires = meta.get("requiresEffort").and_then(Value::as_bool).unwrap_or(false);
    let default = meta.get("defaultLevel").and_then(Value::as_str).and_then(|s| parse_level(s).flatten());
    let level = match level {
        None if requires => default,
        Some(L::Off) if requires => default.or(efforts.first().copied()),
        other => other,
    };
    let Some(l) = level else { return None };
    if l == L::Off || efforts.is_empty() || efforts.contains(&l) {
        return Some(l);
    }
    // nearest supported: next higher, else highest lower
    efforts.iter().copied().filter(|e| *e > l).min().or_else(|| efforts.iter().copied().max()).or(Some(l))
}

fn custom_model(provider: &str, pcfg: &Map<String, Value>, entry: &Value) -> Result<Model> {
    let s = |v: &Value, k: &str| v.get(k).and_then(Value::as_str).map(str::to_string);
    let id = s(entry, "id").ok_or_else(|| anyhow!("model without id in provider {provider}"))?;
    let api = s(entry, "api")
        .or_else(|| pcfg.get("api").and_then(Value::as_str).map(str::to_string))
        .unwrap_or_else(|| default_api_for(provider).to_string());
    let base_url = s(entry, "baseUrl")
        .or_else(|| pcfg.get("baseUrl").and_then(Value::as_str).map(str::to_string))
        .or_else(|| base_url_for(provider).map(str::to_string))
        .ok_or_else(|| anyhow!("provider '{provider}' has no baseUrl"))?;
    let u = |k: &str, d: u64| entry.get(k).and_then(Value::as_u64).unwrap_or(d);
    let mut v = json!({
        "id": id,
        "name": s(entry, "name").unwrap_or_else(|| id.clone()),
        "api": api,
        "provider": provider,
        "baseUrl": base_url,
        "reasoning": entry.get("reasoning").and_then(Value::as_bool).unwrap_or(false),
        "input": entry.get("input").cloned().unwrap_or_else(|| json!(["text"])),
        "cost": entry.get("cost").cloned().unwrap_or_else(|| json!({"input":0,"output":0,"cacheRead":0,"cacheWrite":0})),
        "contextWindow": u("contextWindow", 128_000),
        "maxTokens": u("maxTokens", 16_384),
        "maxTokensExplicit": entry.get("maxTokens").is_some(),
    });
    for k in ["headers", "compat", "thinkingLevelMap"] {
        if let Some(x) = entry.get(k) {
            v[k] = x.clone();
        }
    }
    serde_json::from_value(v).with_context(|| format!("invalid model entry {provider}/{id} in models.yml"))
}

fn apply_overrides(cfg: &Config, provider: &str, m: &mut Model) {
    let Some(o) = cfg
        .provider(provider)
        .and_then(|p| p.get("modelOverrides"))
        .and_then(|o| o.get(&m.id))
        .and_then(Value::as_object)
    else {
        return;
    };
    if let Some(n) = o.get("maxTokens").and_then(Value::as_u64) {
        m.max_tokens = n;
        m.max_tokens_explicit = true;
    }
    if let Some(n) = o.get("contextWindow").and_then(Value::as_u64) {
        m.context_window = n;
    }
    if let Some(n) = o.get("name").and_then(Value::as_str) {
        m.name = n.to_string();
    }
    if let Some(b) = o.get("reasoning").and_then(Value::as_bool) {
        m.reasoning = b;
    }
    if let Some(h) = o.get("headers").and_then(Value::as_object) {
        let hm = m.headers.get_or_insert_with(BTreeMap::new);
        for (k, v) in h {
            if let Some(v) = v.as_str() {
                hm.insert(k.clone(), v.to_string());
            }
        }
    }
    if let Some(c) = o.get("cost").and_then(|c| serde_json::from_value(c.clone()).ok()) {
        m.cost = c;
    }
}

fn open_ro(path: &Path) -> Option<rusqlite::Connection> {
    rusqlite::Connection::open_with_flags(
        path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .ok()
}

/// Look up `provider/id` in omp's `models.db` `model_cache` (provider_id is
/// `provider` or `provider:<version>:<hash>`).
fn cache_model(db: &Path, provider: &str, id: &str) -> Option<(Model, Option<Value>)> {
    let conn = open_ro(db)?;
    let mut stmt = conn
        .prepare("SELECT models FROM model_cache WHERE provider_id = ?1 OR provider_id LIKE ?2 ESCAPE '\\' ORDER BY updated_at DESC")
        .ok()?;
    let like = format!("{}:%", provider.replace('\\', "\\\\").replace('%', "\\%").replace('_', "\\_"));
    let rows: Vec<String> = stmt
        .query_map([provider, like.as_str()], |r| r.get::<_, String>(0))
        .ok()?
        .filter_map(std::result::Result::ok)
        .collect();
    for json in rows {
        let Ok(Value::Array(models)) = serde_json::from_str::<Value>(&json) else { continue };
        if let Some(entry) = models.into_iter().find(|m| m.get("id").and_then(Value::as_str) == Some(id)) {
            let thinking = entry.get("thinking").cloned();
            if let Ok(m) = serde_json::from_value::<Model>(entry) {
                return Some((m, thinking));
            }
        }
    }
    None
}

// ---------------------------------------------------------------- credentials

/// Env var for providers whose omp id differs from pa-ai's.
fn env_key(provider: &str) -> Option<String> {
    let alias = match provider {
        "kimi-code" => Some("KIMI_API_KEY"),
        "opencode-zen" => Some("OPENCODE_API_KEY"),
        "inception" => Some("INCEPTION_API_KEY"),
        "atria" => Some("ATRIA_API_KEY"),
        _ => None,
    };
    if let Some(v) = alias.and_then(|a| std::env::var(a).ok()).filter(|v| !v.is_empty()) {
        return Some(v);
    }
    pa_ai::env_api_keys::get_env_api_key(provider)
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
}

/// First enabled credential for `provider` in `agent.db`: an api key, or an
/// OAuth access token that has not expired.
fn db_key(db: &Path, provider: &str) -> Option<String> {
    let conn = open_ro(db)?;
    let mut stmt = conn
        .prepare("SELECT credential_type, data FROM auth_credentials WHERE provider = ?1 AND disabled_cause IS NULL ORDER BY id")
        .ok()?;
    let rows: Vec<(String, String)> = stmt
        .query_map([provider], |r| Ok((r.get(0)?, r.get(1)?)))
        .ok()?
        .filter_map(std::result::Result::ok)
        .collect();
    let now = now_ms();
    for (kind, data) in rows {
        let Ok(d) = serde_json::from_str::<Value>(&data) else { continue };
        match kind.as_str() {
            "api_key" => {
                if let Some(k) = d.get("key").and_then(Value::as_str).filter(|k| !k.is_empty()) {
                    return Some(k.to_string());
                }
            }
            "oauth" => {
                let expired = d.get("expires").and_then(Value::as_u64).is_some_and(|e| e <= now);
                if let (false, Some(a)) = (expired, d.get("access").and_then(Value::as_str)) {
                    return Some(a.to_string());
                }
            }
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (tempfile::TempDir, AgentDir) {
        let t = tempfile::tempdir().unwrap();
        std::fs::write(
            t.path().join("config.yml"),
            "modelRoles:\n  default: anthropic/claude-opus-5-5:medium\n  smol: zz/m1:minimal\n",
        )
        .unwrap();
        std::fs::write(
            t.path().join("models.yml"),
            "providers:\n  zz:\n    baseUrl: http://127.0.0.1:1/v1\n    apiKey: literal-key\n    api: openai-completions\n    headers:\n      X-A: b\n    models:\n      - id: m1\n        name: M One\n        contextWindow: 32768\n        maxTokens: 8192\n    modelOverrides:\n      m1:\n        maxTokens: 100\n  anthropic:\n    baseUrl: http://localhost:9/anthropic\n",
        )
        .unwrap();
        let dir = AgentDir(t.path().to_path_buf());
        (t, dir)
    }

    #[test]
    fn custom_provider_with_override_and_headers() {
        let (_t, dir) = fixture();
        let m = resolve_in(&dir, "zz/m1").unwrap();
        assert_eq!((m.provider.as_str(), m.id.as_str(), m.api.as_str()), ("zz", "m1", "openai-completions"));
        assert_eq!(m.base_url, "http://127.0.0.1:1/v1");
        assert_eq!(m.api_key.as_deref(), Some("literal-key"));
        assert_eq!(m.context_window, 32768);
        assert_eq!(m.max_tokens, 100); // modelOverrides
        assert_eq!(m.headers, vec![("X-A".to_string(), "b".to_string())]);
        assert_eq!(m.thinking, None);
        assert_eq!(to_pa_model(&m).max_tokens, 100);
    }

    #[test]
    fn thinking_suffix() {
        let (_t, dir) = fixture();
        let m = resolve_in(&dir, "zz/m1:high").unwrap();
        assert_eq!(m.id, "m1");
        assert_eq!(m.thinking, Some(ModelThinkingLevel::High));
        // non-level suffix stays part of the id
        let m = resolve_in(&dir, "zz/m1:free").unwrap();
        assert_eq!(m.id, "m1:free"); // synthesized from provider baseUrl
    }

    #[test]
    fn role_lookup_and_suffix_override() {
        let (_t, dir) = fixture();
        let m = resolve_in(&dir, "smol").unwrap();
        assert_eq!((m.provider.as_str(), m.id.as_str()), ("zz", "m1"));
        assert_eq!(m.thinking, Some(ModelThinkingLevel::Minimal));
        let m = resolve_in(&dir, "smol:low").unwrap();
        assert_eq!(m.thinking, Some(ModelThinkingLevel::Low));
        assert!(resolve_in(&dir, "nosuchrole").is_err());
    }

    #[test]
    fn provider_base_url_override_and_effort_clamp() {
        let t = tempfile::tempdir().unwrap();
        std::fs::write(t.path().join("models.yml"), "providers:\n  groq:\n    baseUrl: http://localhost:9/v1\n").unwrap();
        let dir = AgentDir(t.path().to_path_buf());
        let id = pa_ai::models_generated::get_models("groq")[0].id.clone();
        let m = resolve_in(&dir, &format!("groq/{id}")).unwrap();
        assert_eq!(m.base_url, "http://localhost:9/v1");
        let meta = json!({"efforts":["low","high","max"],"requiresEffort":true,"defaultLevel":"max"});
        use ModelThinkingLevel as L;
        assert_eq!(adjust_thinking(None, Some(&meta)), Some(L::Max));
        assert_eq!(adjust_thinking(Some(L::Off), Some(&meta)), Some(L::Max));
        assert_eq!(adjust_thinking(Some(L::Minimal), Some(&meta)), Some(L::Low));
        assert_eq!(adjust_thinking(Some(L::Medium), Some(&meta)), Some(L::High));
        assert_eq!(adjust_thinking(Some(L::Medium), None), Some(L::Medium));
    }

    #[test]
    fn db_credentials_skip_disabled_and_expired() {
        let t = tempfile::tempdir().unwrap();
        let db = t.path().join("agent.db");
        let c = rusqlite::Connection::open(&db).unwrap();
        c.execute_batch(
            "CREATE TABLE auth_credentials (id INTEGER PRIMARY KEY AUTOINCREMENT, provider TEXT NOT NULL, credential_type TEXT NOT NULL, data TEXT NOT NULL, disabled_cause TEXT DEFAULT NULL);
             INSERT INTO auth_credentials(provider,credential_type,data,disabled_cause) VALUES
              ('p','api_key','{\"key\":\"disabled\"}','x'),
              ('p','oauth','{\"access\":\"expired\",\"expires\":1000}',NULL),
              ('p','oauth','{\"access\":\"live\",\"expires\":99999999999999}',NULL),
              ('q','api_key','{\"key\":\"k\"}',NULL);",
        )
        .unwrap();
        drop(c);
        assert_eq!(db_key(&db, "p").as_deref(), Some("live"));
        assert_eq!(db_key(&db, "q").as_deref(), Some("k"));
        assert_eq!(db_key(&db, "none"), None);
    }
}
