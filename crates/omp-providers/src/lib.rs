//! omp-providers: resolve omp model references (`provider/id[:thinking]` or a
//! role name from `config.yml` `modelRoles`) into `pa-ai` models + credentials,
//! and stream through `pa-ai`.
//!
//! Sources (only OAuth refresh write-back to `agent.db` writes under `~/.omp/agent`, see `auth`):
//! * `models.yml`  - custom providers / `modelOverrides` / provider `baseUrl` overrides
//! * `models.db`   - omp's dynamic model catalog cache (`model_cache`)
//! * `pa-ai` compiled fallback catalog
//! * credentials: `models.yml` `apiKey` -> env vars -> the credential store (`auth`: the auth
//!   broker when configured, else `agent.db` `auth_credentials`; expired OAuth tokens are refreshed
//!   per lookup, and request paths re-resolve per request and retry once after a 401).

pub mod auth;
mod command;
mod fallback;
mod judge;
pub mod login;
pub mod usage;

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context as _, Result};
use serde_json::{json, Map, Value};

pub use pa_ai::types::{Context, Model, ModelThinkingLevel, SimpleStreamOptions, StreamOptions};
pub use pa_ai::AssistantMessageEventStream;

pub use command::{
    invalidate as invalidate_command_value, resolve_config_value, resolve_config_value_async,
};

mod discovery;
pub use discovery::{
    append_models_path, extract_model_ids, models_hash, refresh, refresh_async, ProviderRefresh,
    MATERIALIZATION_POLICY,
};
pub use fallback::{
    fallback_cooldown_ms, is_retryable_error, parse_rate_limit_reason, rate_limit_backoff_ms,
    ActiveFallback, CandidateOptions, FallbackConfig, FallbackState, FallbackWalker, ModelLookup,
    RateLimitReason, RestorePlan, RetryConfig, RevertPolicy,
};
pub use judge::{
    judge, judge_native, judge_online, judgment_cache_path, judgment_route, resolve_judge,
    resolve_judge_chain, stable_json, CachedAnswers, JudgeCandidate, JudgeKind, JudgeRequest,
    JudgeResult, JudgeUsage, JudgmentCache,
};

/// Where the effective thinking level came from (omp `pickInitialThinkingLevel` order).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkingSource {
    /// `:level` suffix, `--thinking`, or the level in the role's selector.
    Selector,
    /// The model catalog's `thinking.defaultLevel`.
    ModelDefault,
    /// `defaultThinkingLevel` in `config.yml` (omp default: `high`).
    ConfigDefault,
    /// Non-reasoning model or no controllable effort: nothing is sent.
    None,
}

/// A model reference resolved to everything needed to call it.
///
/// `credential_id` pins the store row the key came from; when that row sits in a multi-account
/// pool the runtime rotates through [`auth::CredentialPool::alternates`] on a provider rate
/// limit instead of only backing off (same provider, same model, next account).
#[derive(Debug, Clone)]
pub struct ResolvedModel {
    pub provider: String,
    pub id: String,
    /// pa-ai api id: "anthropic-messages" | "openai-completions" | "openai-responses" | ...
    pub api: String,
    pub base_url: String,
    pub api_key: Option<String>,
    /// `agent.db` `auth_credentials` row that supplied `api_key` (None for config/env keys).
    pub credential_id: Option<i64>,
    /// Provider + model extra request headers.
    pub headers: Vec<(String, String)>,
    pub context_window: u64,
    pub max_tokens: u64,
    pub reasoning: bool,
    /// Effective thinking level, resolved as omp does: selector suffix / role level, else the
    /// model's catalog default, else `defaultThinkingLevel` (`auto` resolves to a provisional
    /// level, see `thinking_auto`), clamped to the model's effort ladder. `None` = send nothing.
    pub thinking: Option<ModelThinkingLevel>,
    /// The selector/config asked for `auto` thinking (omp then re-classifies each turn; `thinking`
    /// holds the provisional level).
    pub thinking_auto: bool,
    pub thinking_source: ThinkingSource,
    /// The selector this model was asked for, role-expanded (`provider/id[:level]`).
    pub selector: String,
    /// Role name when resolved through `modelRoles` (chain-key hint for fallbacks).
    pub role: Option<String>,
    /// `retry.fallbackChains` candidates after this model, in order (selectors; see
    /// [`FallbackWalker`] to resolve them lazily).
    pub fallbacks: Vec<String>,
    pub retry: RetryConfig,
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

/// [`resolve`] on the blocking-thread pool. `!command`-backed api keys and
/// headers shell out for up to 10 s; awaiting the sync [`resolve`] inside an
/// async fn parks an executor thread (and `available_models` parks one per
/// provider). Async call sites must use this instead.
///
/// # Errors
/// See [`resolve`].
pub async fn resolve_async(model_ref: String) -> Result<ResolvedModel> {
    resolve_in_async(AgentDir::default(), model_ref).await
}

/// [`resolve_in`] on the blocking-thread pool (see [`resolve_async`]).
///
/// # Errors
/// See [`resolve_in`].
pub async fn resolve_in_async(dir: AgentDir, model_ref: String) -> Result<ResolvedModel> {
    tokio::task::spawn_blocking(move || resolve_in(&dir, &model_ref)).await?
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

/// Walker over the fallback chain of `m` (see [`FallbackWalker`]); `dir` must be the agent dir `m` came from.
#[must_use]
pub fn fallback_walker(dir: &AgentDir, m: &ResolvedModel) -> FallbackWalker {
    FallbackWalker::new(dir.clone(), m.fallbacks.clone())
}

/// [`resolve`] with an explicit agent dir (tests, `--agent-dir`).
///
/// # Errors
/// See [`resolve`].
pub fn resolve_in(dir: &AgentDir, model_ref: &str) -> Result<ResolvedModel> {
    let cfg = Config::load(&dir.0)?;
    let (provider, id, selected) = parse_ref(&cfg, model_ref, 0)?;
    let (mut model, meta) = find_model(&dir.0, &cfg, &provider, &id)?;
    let (thinking, thinking_auto, thinking_source) =
        pick_thinking(selected, &model, meta.as_ref(), cfg.default_thinking);
    let role = {
        let base = split_level(model_ref.trim()).0;
        (!base.contains('/')).then(|| base.to_string())
    };
    let selector = role_selector(&cfg, model_ref);
    let fallbacks = cfg.fallback(&dir.0).candidates(&selector, role.as_deref());
    // `api: "openrouter"` stays as is: the pa-ai transport table streams it through the Responses
    // API (or chat completions with `PI_OPENROUTER_RESPONSES=0`), exactly as TS does.
    let (api_key, credential_id) = match cfg.provider_key(&provider).or_else(|| env_key(&provider))
    {
        Some(key) => (Some(key), None),
        None => match auth::credential(dir, &provider) {
            Some(auth::Credential { key, id }) => {
                // TS `getOAuthApiKey` structured format: the Cloud Code Assist providers parse
                // the OAuth row payload as JSON and need `projectId`/`token` from it (`_Kn` in
                // the TS bundle) — the bare access token alone yields `project: ""` and a 404.
                let key = if provider.as_str() == "google-gemini-cli" || provider.as_str() == "google-antigravity" {
                    auth::structured_key(dir, &provider).unwrap_or(key)
                } else {
                    key
                };
                (Some(key), Some(id))
            }
            None => (None, None),
        },
    };
    let mut headers: Vec<(String, String)> = cfg
        .provider(&provider)
        .and_then(|p| p.get("headers"))
        .and_then(Value::as_object)
        .map(|h| {
            h.iter()
                .filter_map(|(k, v)| Some((k.clone(), v.as_str()?.to_string())))
                .collect()
        })
        .unwrap_or_default();
    if let Some(h) = &model.headers {
        for (k, v) in h {
            headers.retain(|(hk, _)| hk != k);
            headers.push((k.clone(), v.clone()));
        }
    }
    // `!command` header values run through the shell; empty / failed values are omitted.
    let headers: Vec<(String, String)> = headers
        .into_iter()
        .filter_map(|(k, v)| Some((k, resolve_config_value(&v)?)))
        .collect();
    model.headers = (!headers.is_empty()).then(|| headers.iter().cloned().collect());
    Ok(ResolvedModel {
        provider: model.provider.clone(),
        id: model.id.clone(),
        api: model.api.clone(),
        base_url: model.base_url.clone(),
        api_key,
        credential_id,
        headers,
        context_window: model.context_window,
        max_tokens: model.max_tokens,
        reasoning: model.reasoning,
        thinking,
        thinking_auto,
        thinking_source,
        selector,
        role,
        fallbacks,
        retry: cfg.retry.clone(),
        model,
    })
}

// ---------------------------------------------------------------- ref parsing

/// A thinking selector as written in a `:suffix`, role value or `defaultThinkingLevel`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Sel {
    /// omp's session-level `auto`: classify per turn.
    Auto,
    Inherit,
    Level(ModelThinkingLevel),
}

/// Exact names plus omp's unambiguous (>= 2 char) abbreviations (`med`, `hi`, `xhi`).
fn parse_sel(s: &str) -> Option<Sel> {
    use ModelThinkingLevel as L;
    const NAMES: [(&str, Sel); 9] = [
        ("inherit", Sel::Inherit),
        ("off", Sel::Level(L::Off)),
        ("minimal", Sel::Level(L::Minimal)),
        ("low", Sel::Level(L::Low)),
        ("medium", Sel::Level(L::Medium)),
        ("high", Sel::Level(L::High)),
        ("xhigh", Sel::Level(L::Xhigh)),
        ("max", Sel::Level(L::Max)),
        ("auto", Sel::Auto),
    ];
    if s == "none" {
        return Some(Sel::Level(L::Off));
    }
    if let Some((_, v)) = NAMES.iter().find(|(n, _)| *n == s) {
        return Some(*v);
    }
    if s.len() < 2 {
        return None;
    }
    let mut it = NAMES.iter().filter(|(n, _)| n.starts_with(s));
    match (it.next(), it.next()) {
        (Some((_, v)), None) => Some(*v),
        _ => None,
    }
}

/// Concrete effort name (catalog `efforts` / `defaultLevel`).
fn effort(s: &str) -> Option<ModelThinkingLevel> {
    match parse_sel(s)? {
        Sel::Level(l) => Some(l),
        _ => None,
    }
}

/// Split a trailing `:level` off a selector; a non-level suffix stays in the id.
pub(crate) fn split_level(s: &str) -> (&str, Option<Sel>) {
    match s.rsplit_once(':') {
        Some((b, suffix)) => match parse_sel(suffix) {
            Some(sel) => (b, Some(sel)),
            None => (s, None),
        },
        None => (s, None),
    }
}

type Parsed = (String, String, Option<Sel>);

fn parse_ref(cfg: &Config, model_ref: &str, depth: u8) -> Result<Parsed> {
    let model_ref = model_ref.trim();
    let (base, thinking_override) = split_level(model_ref);
    if let Some((provider, id)) = base.split_once('/') {
        if !provider.is_empty() && !id.is_empty() {
            return Ok((provider.to_string(), id.to_string(), thinking_override));
        }
    }
    // Role name.
    if depth > 4 {
        bail!("modelRoles alias loop at '{model_ref}'");
    }
    let target = cfg.roles.get(base).ok_or_else(|| {
        anyhow!(
            "'{model_ref}' is neither 'provider/id' nor a modelRoles role ({:?})",
            cfg.roles.keys().collect::<Vec<_>>()
        )
    })?;
    let (p, i, t) = parse_ref(cfg, target, depth + 1)?;
    Ok((p, i, thinking_override.or(t)))
}

/// The selector string a ref stands for: role value for a role, else the ref itself.
fn role_selector(cfg: &Config, model_ref: &str) -> String {
    let (base, level) = split_level(model_ref.trim());
    match cfg.roles.get(base) {
        Some(target) if !base.contains('/') => {
            let mut s = role_selector(cfg, target);
            if level.is_some() {
                // an explicit suffix on the role ref replaces the role's own level
                s = format!(
                    "{}:{}",
                    split_level(&s).0,
                    model_ref.trim().rsplit(':').next().unwrap_or_default()
                );
            }
            s
        }
        _ => model_ref.trim().to_string(),
    }
}

// ---------------------------------------------------------------- config

struct Config {
    roles: BTreeMap<String, String>,
    providers: Map<String, Value>,
    /// `defaultThinkingLevel` (omp default `high`).
    default_thinking: Sel,
    retry: RetryConfig,
    /// `retry.fallbackChains` in file order.
    chains: Vec<(String, Vec<String>)>,
}

impl Config {
    fn load(dir: &Path) -> Result<Self> {
        let read_yaml = |name: &str| -> Result<Value> {
            let path = dir.join(name);
            match std::fs::read_to_string(&path) {
                Ok(s) if s.trim().is_empty() => Ok(Value::Null),
                Ok(s) => serde_yaml_ng::from_str::<Value>(&s)
                    .with_context(|| format!("parsing {}", path.display())),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Value::Null),
                Err(e) => Err(e).with_context(|| format!("reading {}", path.display())),
            }
        };
        let models = read_yaml("models.yml")?;
        // Registry-layered view of the agent config (defaults < config.yml);
        // serde_json's `preserve_order` map keeps fallback-chain order, so the
        // old raw-YAML double-read is no longer needed.
        let settings = omp_settings::Settings::load_from(
            Some(&dir.join("config.yml")),
            &dir.join(".no-project-layer"),
        );
        let roles: std::collections::BTreeMap<String, String> = settings.model_roles();
        let providers = models
            .get("providers")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        let default_thinking = settings
            .get_string("defaultThinkingLevel")
            .as_deref()
            .and_then(parse_sel)
            .filter(|s| *s != Sel::Inherit)
            .unwrap_or(Sel::Level(ModelThinkingLevel::High));
        let chains = settings
            .get_record("retry.fallbackChains")
            .iter()
            .filter_map(|(k, v)| {
                let list = v
                    .as_array()?
                    .iter()
                    .filter_map(|e| e.as_str().map(str::to_string))
                    .collect();
                Some((k.clone(), list))
            })
            .collect();
        Ok(Self {
            roles,
            providers,
            default_thinking,
            retry: RetryConfig::from_settings(&settings),
            chains,
        })
    }

    fn fallback(&self, dir: &Path) -> FallbackConfig {
        let mut known_providers: BTreeSet<String> = self.providers.keys().cloned().collect();
        known_providers.extend(
            [
                "anthropic",
                "openai",
                "openai-codex",
                "google",
                "zai",
                "kimi-code",
                "openrouter",
                "opencode-zen",
            ]
            .map(String::from),
        );
        FallbackConfig {
            chains: self.chains.clone(),
            roles: self.roles.clone(),
            known_providers,
            model_exists: Some(fallback::ModelLookup::catalog(dir)),
        }
    }

    fn provider(&self, p: &str) -> Option<&Map<String, Value>> {
        self.providers.get(p).and_then(Value::as_object)
    }

    /// `apiKey` from models.yml: `!command` stdout, an env var name if that var is set, else the literal.
    /// A keyless provider (`auth: none`, TS `ProviderConfig.auth`) without `apiKey` yields an empty
    /// key: requests go out without an `Authorization` header, like TS.
    fn provider_key(&self, p: &str) -> Option<String> {
        let pc = self.provider(p)?;
        match pc.get("apiKey").and_then(Value::as_str) {
            Some(raw) => resolve_config_value(raw),
            None => (pc.get("auth").and_then(Value::as_str) == Some("none")).then(String::new),
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

fn find_model(
    dir: &Path,
    cfg: &Config,
    provider: &str,
    id: &str,
) -> Result<(Model, Option<Value>)> {
    let pcfg = cfg.provider(provider);
    // 1. custom model from models.yml
    if let Some(entry) = pcfg
        .and_then(|p| p.get("models"))
        .and_then(Value::as_array)
        .and_then(|a| {
            a.iter()
                .find(|m| m.get("id").and_then(Value::as_str) == Some(id))
        })
    {
        let mut model = custom_model(provider, pcfg.expect("checked"), entry)?;
        let authored = entry.get("thinking").cloned();
        let resolved = finish_model(cfg, provider, &mut model, authored.as_ref())?;
        return Ok((model, resolved));
    }
    // 2. omp's dynamic catalog cache, 3. pa-ai compiled catalog
    let (mut model, meta) = match cache_model(&dir.join("models.db"), provider, id) {
        Some((m, t)) => (Some(m), t),
        None => (
            pa_ai::models_generated::get_model(provider, id).cloned(),
            None,
        ),
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
        anyhow!(
            "model '{provider}/{id}' not found in models.yml, models.db cache or built-in catalog"
        )
    })?;
    // provider-level overrides apply to catalog models too. A provider-level `baseUrl`
    // may be a sandbox/relay override; model rows pinned to a googleapis CCA host keep
    // their host (the antigravity provider rewrites daily->sandbox itself), while an
    // explicit non-googleapis override (local mock, relay) always wins.
    if let Some(p) = pcfg {
        if let Some(u) = p.get("baseUrl").and_then(Value::as_str) {
            let model_is_cca = model.base_url.contains("googleapis.com");
            let override_is_cca = u.contains("googleapis.com");
            if !model_is_cca || !override_is_cca {
                model.base_url = u.to_string();
            }
        }
        if let Some(a) = p.get("api").and_then(Value::as_str) {
            model.api = a.to_string();
        }
    }
    let resolved = finish_model(cfg, provider, &mut model, meta.as_ref())?;
    Ok((model, resolved))
}

/// Every model source ends here (TS `buildModel`): `modelOverrides` first (they feed the policy
/// input), then the one-time policy resolution — thinking ladder, `reasoning` correction and
/// catalog assignments — through the shared resolver in `pa_ai::compat`.
///
/// Returns the resolved thinking config (TS `model.thinking`): the ladder every effort clamp and
/// request builder reads, whatever stale `thinking` a cached row stored.
fn finish_model(
    cfg: &Config,
    provider: &str,
    model: &mut Model,
    thinking: Option<&Value>,
) -> Result<Option<Value>> {
    // TS `mergeCompat(model.compatConfig, providerOverride.compat)`: the provider-level `compat:`
    // block (and `disableStrictTools`) layers over the model's authored overrides.
    if let Some(p) = cfg.provider(provider) {
        if let Some(c) = p.get("compat").and_then(Value::as_object) {
            merge_compat(model, c);
        }
        if p.get("disableStrictTools").and_then(Value::as_bool) == Some(true) {
            let mut c = Map::new();
            c.insert("disableStrictTools".into(), Value::Bool(true));
            merge_compat(model, &c);
        }
    }
    apply_overrides(cfg, provider, model);
    pa_ai::compat::materialize(model, thinking)
        .map(|policy| policy.thinking)
        .map_err(|e| anyhow!("model policy for {provider}/{}: {e}", model.id))
}

/// Effective thinking level for a model, in omp's order (`sdk.ts` `pickInitialThinkingLevel`):
/// selector level → catalog `thinking.defaultLevel` → `defaultThinkingLevel` → clamp to the
/// model (`resolveThinkingLevelForModel` / `resolveProvisionalAutoLevel`).
/// Returns `(level, auto, source)`.
fn pick_thinking(
    selected: Option<Sel>,
    model: &Model,
    meta: Option<&Value>,
    config_default: Sel,
) -> (Option<ModelThinkingLevel>, bool, ThinkingSource) {
    let model_default = meta
        .and_then(|m| m.get("defaultLevel"))
        .and_then(Value::as_str)
        .and_then(effort);
    let (sel, source) = match selected {
        Some(Sel::Inherit) | None => match model_default {
            Some(l) => (Sel::Level(l), ThinkingSource::ModelDefault),
            None => (config_default, ThinkingSource::ConfigDefault),
        },
        Some(s) => (s, ThinkingSource::Selector),
    };
    if !model.reasoning {
        return (None, false, ThinkingSource::None);
    }
    match sel {
        Sel::Auto => {
            let level = provisional_auto(meta, model_default);
            (
                level,
                true,
                if level.is_some() {
                    source
                } else {
                    ThinkingSource::None
                },
            )
        }
        Sel::Inherit => (None, false, ThinkingSource::None),
        Sel::Level(l) => {
            let level = clamp_level(l, meta, model_default);
            (
                level,
                false,
                if level.is_some() {
                    source
                } else {
                    ThinkingSource::None
                },
            )
        }
    }
}

fn efforts_of(meta: Option<&Value>) -> Option<Vec<ModelThinkingLevel>> {
    let a = meta?.get("efforts")?.as_array()?;
    Some(
        a.iter()
            .filter_map(|e| effort(e.as_str()?))
            .filter(|l| *l != ModelThinkingLevel::Off)
            .collect(),
    )
}

/// `clampThinkingLevelForModel`: the highest supported effort not above the request, else the lowest
/// supported. `off` stays `off` unless the model `requiresEffort`. Without catalog metadata the
/// level passes through; with an empty effort list there is nothing to control.
fn clamp_level(
    l: ModelThinkingLevel,
    meta: Option<&Value>,
    model_default: Option<ModelThinkingLevel>,
) -> Option<ModelThinkingLevel> {
    use ModelThinkingLevel as L;
    let requires = meta
        .and_then(|m| m.get("requiresEffort"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let efforts = efforts_of(meta);
    let l = if l == L::Off && requires {
        model_default.or_else(|| efforts.as_ref().and_then(|e| e.iter().copied().min()))?
    } else {
        l
    };
    if l == L::Off {
        return Some(l);
    }
    let Some(mut efforts) = efforts else {
        return Some(l);
    };
    efforts.sort();
    if efforts.contains(&l) {
        return Some(l);
    }
    efforts
        .iter()
        .copied()
        .filter(|e| *e < l)
        .max()
        .or_else(|| efforts.first().copied())
}

/// `resolveProvisionalAutoLevel`: model default (max → xhigh, fallback high), floored at `low`,
/// capped at `xhigh`, within the model's efforts.
fn provisional_auto(
    meta: Option<&Value>,
    model_default: Option<ModelThinkingLevel>,
) -> Option<ModelThinkingLevel> {
    use ModelThinkingLevel as L;
    let preferred = match model_default.unwrap_or(L::High) {
        L::Max => L::Xhigh,
        o => o,
    };
    let mut pool: Vec<L> = efforts_of(meta)
        .unwrap_or_else(|| vec![L::Minimal, L::Low, L::Medium, L::High, L::Xhigh, L::Max]);
    pool.sort();
    let floored: Vec<L> = pool.iter().copied().filter(|e| *e >= L::Low).collect();
    let floored = if floored.is_empty() { pool } else { floored };
    let capped: Vec<L> = floored.into_iter().filter(|e| *e <= L::Xhigh).collect();
    let mut chosen = *capped.first()?;
    for c in capped {
        if c > preferred {
            break;
        }
        chosen = c;
    }
    Some(chosen)
}

fn custom_model(provider: &str, pcfg: &Map<String, Value>, entry: &Value) -> Result<Model> {
    let s = |v: &Value, k: &str| v.get(k).and_then(Value::as_str).map(str::to_string);
    let id = s(entry, "id").ok_or_else(|| anyhow!("model without id in provider {provider}"))?;
    let api = s(entry, "api")
        .or_else(|| pcfg.get("api").and_then(Value::as_str).map(str::to_string))
        .unwrap_or_else(|| default_api_for(provider).to_string());
    let base_url = s(entry, "baseUrl")
        .or_else(|| {
            pcfg.get("baseUrl")
                .and_then(Value::as_str)
                .map(str::to_string)
        })
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
    serde_json::from_value(v)
        .with_context(|| format!("invalid model entry {provider}/{id} in models.yml"))
}

/// TS `mergeCompat`: `over` layered over the model's sparse compat, nested objects merged
/// recursively, everything else replaced.
fn merge_compat(m: &mut Model, over: &Map<String, Value>) {
    fn merge(base: &mut Map<String, Value>, over: &Map<String, Value>) {
        for (k, v) in over {
            match (base.get_mut(k), v) {
                (Some(Value::Object(b)), Value::Object(o)) => merge(b, o),
                _ => {
                    base.insert(k.clone(), v.clone());
                }
            }
        }
    }
    let mut raw = m.compat.take().map(|c| c.raw).unwrap_or_default();
    merge(&mut raw, over);
    m.compat = Some(pa_types::ai::ModelCompat { raw });
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
    if let Some(c) = o
        .get("cost")
        .and_then(|c| serde_json::from_value(c.clone()).ok())
    {
        m.cost = c;
    }
    if let Some(c) = o.get("compat").and_then(Value::as_object) {
        merge_compat(m, c);
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
    let like = format!(
        "{}:%",
        provider
            .replace('\\', "\\\\")
            .replace('%', "\\%")
            .replace('_', "\\_")
    );
    let rows: Vec<String> = stmt
        .query_map([provider, like.as_str()], |r| r.get::<_, String>(0))
        .ok()?
        .filter_map(std::result::Result::ok)
        .collect();
    for json in rows {
        let Ok(Value::Array(models)) = serde_json::from_str::<Value>(&json) else {
            continue;
        };
        if let Some(entry) = models
            .into_iter()
            .find(|m| m.get("id").and_then(Value::as_str) == Some(id))
        {
            // The stored resolved `compat` may predate the current rules; the policy resolver
            // re-derives it from the row's authored `compatConfig` (see `materialize`).
            if let Some((model, thinking)) = pa_ai::compat::model_from_catalog_row(entry) {
                return Some((model, thinking));
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
    if let Some(v) = alias
        .and_then(|a| std::env::var(a).ok())
        .filter(|v| !v.is_empty())
    {
        return Some(v);
    }
    pa_ai::env_api_keys::get_env_api_key(provider)
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
}

/// How the active credential for a provider is supplied, as `/session` reports it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthDetails {
    /// TS `resolveProviderAuthMode`: `oauth` | `api key` | `env api key` | `runtime/fallback` | `unknown`.
    pub mode: &'static str,
    /// TS `KeyCascade.describe`: where the active credential came from; `None` when unknown.
    pub source: Option<String>,
}

/// [`AuthDetails`] for `provider` from `models.yml` `apiKey`, env vars and the
/// enabled `agent.db` `auth_credentials` rows (the omp `AuthStorage` strata).
#[must_use]
pub fn auth_details(dir: &AgentDir, provider: &str) -> AuthDetails {
    let config_key = Config::load(&dir.0).ok().is_some_and(|cfg| {
        cfg.provider(provider)
            .and_then(|p| p.get("apiKey"))
            .and_then(Value::as_str)
            .is_some()
    });
    // (type, data, id) of every enabled stored credential, in pool order.
    let rows: Vec<(String, Value, i64)> = open_ro(&dir.0.join("agent.db"))
        .and_then(|conn| {
            let mut stmt = conn
                .prepare("SELECT credential_type, data, id FROM auth_credentials WHERE provider = ?1 AND disabled_cause IS NULL ORDER BY id")
                .ok()?;
            let rows = stmt
                .query_map([provider], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?, r.get::<_, i64>(2)?)))
                .ok()?
                .filter_map(std::result::Result::ok)
                .map(|(kind, data, id)| (kind, serde_json::from_str(&data).unwrap_or(Value::Null), id))
                .collect();
            Some(rows)
        })
        .unwrap_or_default();
    let env = env_key(provider).is_some();
    let mode = if rows.iter().any(|(kind, ..)| kind == "oauth") {
        "oauth"
    } else if !rows.is_empty() {
        "api key"
    } else if env {
        "env api key"
    } else if config_key {
        "runtime/fallback"
    } else {
        "unknown"
    };
    const BASE_LABEL: &str = "local store";
    let describe = |kind: &str, filter: &dyn Fn(&Value) -> bool| {
        let (_, data, id) = rows.iter().find(|(k, d, _)| k == kind && filter(d))?;
        let identity = if kind == "oauth" {
            ["email", "accountId", "projectId"]
                .iter()
                .find_map(|f| data.get(*f).and_then(Value::as_str).map(str::to_string))
                .unwrap_or_else(|| format!("cred {id}"))
        } else {
            format!("cred {id}")
        };
        Some(format!("{BASE_LABEL} · {kind} #{id} ({identity})"))
    };
    let is_login = |d: &Value| d.get("source").and_then(Value::as_str) == Some("login");
    let source = if config_key {
        Some("config override (models.yml)".to_string())
    } else {
        describe("oauth", &|_| true)
            .or_else(|| describe("api_key", &is_login))
            .or_else(|| env.then(|| format!("env (over {BASE_LABEL})")))
            .or_else(|| describe("api_key", &|d| !is_login(d)))
    };
    AuthDetails { mode, source }
}

#[cfg(test)]
mod tests {
    /// Async resolution never parks the executor: two concurrent `!command`
    /// values (0.4 s each) finish well under 0.8 s even on a single-threaded
    /// runtime.
    #[tokio::test(flavor = "current_thread")]
    async fn async_command_resolution_runs_off_executor() {
        let a = resolve_config_value_async("!sleep 0.4; echo one");
        let b = resolve_config_value_async("!sleep 0.4; echo two");
        let (a, b) = futures::join!(a, b);
        assert_eq!(a.as_deref(), Some("one"));
        assert_eq!(b.as_deref(), Some("two"));
    }

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
            "providers:\n  zz:\n    baseUrl: http://127.0.0.1:1/v1\n    apiKey: literal-key\n    api: openai-completions\n    headers:\n      X-A: b\n    models:\n      - id: m1\n        name: M One\n        reasoning: true\n        contextWindow: 32768\n        maxTokens: 8192\n    modelOverrides:\n      m1:\n        maxTokens: 100\n  anthropic:\n    baseUrl: http://localhost:9/anthropic\n",
        )
        .unwrap();
        let dir = AgentDir(t.path().to_path_buf());
        (t, dir)
    }

    #[test]
    fn custom_provider_with_override_and_headers() {
        let (_t, dir) = fixture();
        let m = resolve_in(&dir, "zz/m1").unwrap();
        assert_eq!(
            (m.provider.as_str(), m.id.as_str(), m.api.as_str()),
            ("zz", "m1", "openai-completions")
        );
        assert_eq!(m.base_url, "http://127.0.0.1:1/v1");
        assert_eq!(m.api_key.as_deref(), Some("literal-key"));
        assert_eq!(m.context_window, 32768);
        assert_eq!(m.max_tokens, 100); // modelOverrides
        assert_eq!(m.headers, vec![("X-A".to_string(), "b".to_string())]);
        assert_eq!(m.thinking, Some(ModelThinkingLevel::High)); // config default for a reasoning model
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
        std::fs::write(
            t.path().join("models.yml"),
            "providers:\n  groq:\n    baseUrl: http://localhost:9/v1\n",
        )
        .unwrap();
        let dir = AgentDir(t.path().to_path_buf());
        let id = pa_ai::models_generated::get_models("groq")[0].id.clone();
        let m = resolve_in(&dir, &format!("groq/{id}")).unwrap();
        assert_eq!(m.base_url, "http://localhost:9/v1");
    }

    #[test]
    fn clamp_follows_omp_semantics() {
        use ModelThinkingLevel as L;
        let meta =
            json!({"efforts":["low","high","max"],"requiresEffort":true,"defaultLevel":"max"});
        let m = Some(&meta);
        let dm = Some(L::Max);
        assert_eq!(clamp_level(L::Off, m, dm), Some(L::Max));
        assert_eq!(clamp_level(L::Minimal, m, dm), Some(L::Low)); // below ladder -> lowest
        assert_eq!(clamp_level(L::Medium, m, dm), Some(L::Low)); // highest supported not above
        assert_eq!(clamp_level(L::Xhigh, m, dm), Some(L::High));
        assert_eq!(clamp_level(L::Medium, None, None), Some(L::Medium));
        assert_eq!(clamp_level(L::Off, None, None), Some(L::Off));
        assert_eq!(provisional_auto(m, dm), Some(L::High)); // max -> xhigh -> highest supported <= xhigh
        assert_eq!(provisional_auto(None, None), Some(L::High));
    }

    #[test]
    fn thinking_precedence_and_sources() {
        use ModelThinkingLevel as L;
        let (_t, dir) = fixture();
        // no suffix, no catalog default -> config default (omp default: high), reasoning model
        let m = resolve_in(&dir, "zz/m1").unwrap();
        assert_eq!(
            (m.thinking, m.thinking_source),
            (Some(L::High), ThinkingSource::ConfigDefault)
        );
        // selector beats everything
        let m = resolve_in(&dir, "zz/m1:low").unwrap();
        assert_eq!(
            (m.thinking, m.thinking_source),
            (Some(L::Low), ThinkingSource::Selector)
        );
        // abbreviations as in omp
        assert_eq!(
            resolve_in(&dir, "zz/m1:med").unwrap().thinking,
            Some(L::Medium)
        );
        // explicit auto -> provisional level + flag
        let m = resolve_in(&dir, "zz/m1:auto").unwrap();
        assert_eq!((m.thinking, m.thinking_auto), (Some(L::High), true));
        // non-reasoning model sends nothing
        std::fs::write(
            dir.0.join("models.yml"),
            "providers:\n  zz:\n    baseUrl: http://127.0.0.1:1/v1\n    apiKey: k\n    models:\n      - id: plain\n",
        )
        .unwrap();
        let m = resolve_in(&dir, "zz/plain:high").unwrap();
        assert_eq!(
            (m.thinking, m.thinking_source),
            (None, ThinkingSource::None)
        );
    }

    #[test]
    fn config_default_thinking_and_fallback_chain_from_yaml() {
        use ModelThinkingLevel as L;
        let (_t, dir) = fixture();
        std::fs::write(
            dir.0.join("config.yml"),
            "defaultThinkingLevel: low\nmodelRoles:\n  smol: zz/m1:minimal\n  judge: zz/m1\nretry:\n  maxRetries: 3\n  fallbackChains:\n    smol:\n      - zz/m2\n      - zz/m3\n",
        )
        .unwrap();
        let m = resolve_in(&dir, "zz/m1").unwrap();
        assert_eq!(m.thinking, Some(L::Low));
        assert_eq!(m.retry.max_retries, 3);
        assert_eq!(m.fallbacks, ["zz/m2", "zz/m3"]); // m1 is smol's primary: its chain follows the model
        let m = resolve_in(&dir, "smol").unwrap();
        assert_eq!(
            (m.role.as_deref(), m.selector.as_str()),
            (Some("smol"), "zz/m1:minimal")
        );
        assert_eq!(m.fallbacks, ["zz/m2", "zz/m3"]);
        let mut w = fallback_walker(&dir, &m);
        assert_eq!(w.next_model().unwrap().id, "m2"); // synthesized from provider baseUrl
        assert_eq!(w.pending(), ["zz/m3"]);
    }

    #[test]
    fn judge_chain_native_cutoff() {
        let (_t, dir) = fixture();
        std::fs::write(
            dir.0.join("models.yml"),
            "providers:\n  nat:\n    baseUrl: https://x.test/api/alpha/\n    apiKey: k\n    api: openrouter-decisions\n    models:\n      - id: j1\n  chat:\n    baseUrl: http://c.test/v1\n    apiKey: k\n    models:\n      - id: c1\n  nokey:\n    baseUrl: http://n.test/v1\n    models:\n      - id: n1\n",
        )
        .unwrap();
        std::fs::write(
            dir.0.join("config.yml"),
            "modelRoles:\n  judge: chat/c1\nretry:\n  fallbackChains:\n    judge:\n      - nat/j1\n      - chat/c2\n      - nokey/n1\n      - nat/j2\n",
        )
        .unwrap();
        let chain = resolve_judge_chain(&dir).unwrap();
        let got: Vec<_> = chain
            .iter()
            .map(|c| (c.kind, c.model.id.as_str()))
            .collect();
        // chat/c1 (online), then first native j1; chat/c2 dropped by the native cutoff, nokey skipped.
        assert_eq!(
            got,
            [
                (JudgeKind::Online, "c1"),
                (JudgeKind::Native, "j1"),
                (JudgeKind::Native, "j2")
            ]
        );
        assert_eq!(
            chain[1].url.as_deref(),
            Some("https://x.test/api/alpha/decisions")
        );
    }

    #[test]
    fn command_api_key_and_headers() {
        let (_t, dir) = fixture();
        std::fs::write(
            dir.0.join("models.yml"),
            "providers:\n  zz:\n    baseUrl: http://127.0.0.1:1/v1\n    apiKey: \"!echo cmd-key\"\n    headers:\n      X-T: \"!echo hv\"\n      X-Bad: \"!exit 1\"\n    models:\n      - id: m1\n",
        )
        .unwrap();
        let m = resolve_in(&dir, "zz/m1").unwrap();
        assert_eq!(m.api_key.as_deref(), Some("cmd-key"));
        assert_eq!(m.headers, vec![("X-T".to_string(), "hv".to_string())]);
    }

    #[test]
    fn auth_none_provider_resolves_keyless() {
        let (_t, dir) = fixture();
        std::fs::write(
            dir.0.join("models.yml"),
            "providers:\n  kl:\n    baseUrl: http://127.0.0.1:1/v1\n    api: openai-completions\n    auth: none\n    models:\n      - id: m1\n",
        )
        .unwrap();
        let m = resolve_in(&dir, "kl/m1").unwrap();
        assert_eq!(m.api_key.as_deref(), Some(""));
        assert_eq!(m.credential_id, None);
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
        let key = |p: &str| auth::local_credential(&db, p).map(|c| c.key);
        assert_eq!(key("p").as_deref(), Some("live"));
        assert_eq!(key("q").as_deref(), Some("k"));
        assert_eq!(key("none"), None);
        // The row id travels with the key (the live oauth row is the 3rd insert for `p`).
        assert_eq!(auth::local_credential(&db, "q").map(|c| c.id), Some(4));
    }

    // ---- kimi-code compat (regression: `400 role 'developer' is not allowed`)

    /// The TS-resolved compat record `models.db` stores for `kimi-code/kimi-for-coding`
    /// (`thinkingFormat: "kimi"` is not a spelling older Rust enums knew).
    const KIMI_CACHE_COMPAT: &str = r#"{"supportsStore":false,"supportsDeveloperRole":false,"supportsMultipleSystemMessages":false,"supportsReasoningEffort":true,"reasoningEffortMap":{},"supportsUsageInStreaming":true,"alwaysSendMaxTokens":true,"maxTokensField":"max_tokens","thinkingFormat":"kimi","kimiApiFormat":"anthropic","reasoningDisableMode":"zai-thinking-disabled","supportsStrictMode":false,"toolSchemaFlavor":"moonshot-mfjs"}"#;

    fn kimi_dir(source: &str, base_url: &str) -> (tempfile::TempDir, AgentDir) {
        let t = tempfile::tempdir().unwrap();
        match source {
            "cache" => {
                let c = rusqlite::Connection::open(t.path().join("models.db")).unwrap();
                c.execute_batch(
                    "CREATE TABLE model_cache (provider_id TEXT PRIMARY KEY, version INTEGER NOT NULL, updated_at INTEGER NOT NULL, authoritative INTEGER NOT NULL DEFAULT 0, static_fingerprint TEXT NOT NULL DEFAULT '', models TEXT NOT NULL);",
                )
                .unwrap();
                let models = json!([{
                    "id": "kimi-for-coding", "name": "K2.8 Preview", "api": "openai-completions",
                    "provider": "kimi-code", "baseUrl": base_url, "reasoning": true,
                    "input": ["text", "image"],
                    "cost": {"input": 0.95, "output": 4, "cacheRead": 0.19, "cacheWrite": 0},
                    "contextWindow": 1_048_576, "maxTokens": 32768,
                    "thinking": {"mode": "effort", "efforts": ["low", "high", "max"], "requiresEffort": true, "defaultLevel": "max"},
                    "identity": {"class": "kimi"},
                    "compat": serde_json::from_str::<Value>(KIMI_CACHE_COMPAT).unwrap(),
                }]);
                c.execute(
                    "INSERT INTO model_cache(provider_id, version, updated_at, models) VALUES ('kimi-code', 1, 1, ?1)",
                    [models.to_string()],
                )
                .unwrap();
                std::fs::write(t.path().join("models.yml"), format!("providers:\n  kimi-code:\n    apiKey: k\n    baseUrl: {base_url}\n")).unwrap();
            }
            "yml-custom" => std::fs::write(
                t.path().join("models.yml"),
                format!("providers:\n  kimi-code:\n    baseUrl: {base_url}\n    apiKey: k\n    api: openai-completions\n    models:\n      - id: kimi-for-coding\n        reasoning: true\n"),
            )
            .unwrap(),
            // Provider configured, model unknown to every catalog: synthesized.
            _ => std::fs::write(t.path().join("models.yml"), format!("providers:\n  kimi-code:\n    baseUrl: {base_url}\n    apiKey: k\n    api: openai-completions\n")).unwrap(),
        }
        let dir = AgentDir(t.path().to_path_buf());
        (t, dir)
    }

    /// Request over a real socket: a model whose resolved compat names `kimiApiFormat: "anthropic"`
    /// (the whole kimi-code catalog) streams through `/v1/messages` with the Anthropic body and the
    /// Kimi device headers, as TS `streamKimi` does — never chat completions.
    #[tokio::test]
    async fn kimi_code_anthropic_format_streams_through_the_messages_endpoint() {
        use futures::StreamExt as _;
        use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
        for source in ["cache", "yml-custom", "synthesized"] {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let base = format!("http://{}/coding/v1", listener.local_addr().unwrap());
            let server = tokio::spawn(async move {
                let (mut sock, _) = listener.accept().await.unwrap();
                let mut buf = Vec::new();
                let mut head_text = String::new();
                let body = loop {
                    let mut chunk = [0u8; 4096];
                    let n = sock.read(&mut chunk).await.unwrap();
                    buf.extend_from_slice(&chunk[..n]);
                    let text = String::from_utf8_lossy(&buf).to_string();
                    if let Some((head, rest)) = text.split_once("\r\n\r\n") {
                        head_text = head.to_string();
                        let len = head
                            .lines()
                            .find_map(|l| {
                                l.to_ascii_lowercase()
                                    .strip_prefix("content-length:")
                                    .map(|v| v.trim().parse::<usize>().unwrap())
                            })
                            .unwrap_or(0);
                        if rest.len() >= len {
                            break rest[..len].to_string();
                        }
                    }
                    if n == 0 {
                        break String::new();
                    }
                };
                let sse = "data: {\"id\":\"1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";
                let resp = format!("HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{sse}", sse.len());
                sock.write_all(resp.as_bytes()).await.unwrap();
                (head_text, body)
            });
            let (_t, dir) = kimi_dir(source, &base);
            let m = resolve_in(&dir, "kimi-code/kimi-for-coding").unwrap();
            let ctx: Context = serde_json::from_value(json!({
                "systemPrompt": "SYS PROMPT",
                "messages": [{"role": "user", "content": "hi", "timestamp": 0}]
            }))
            .unwrap();
            let mut s = stream(&m, ctx, None).await.unwrap();
            while s.next().await.is_some() {}
            let (head, body) = server.await.unwrap();
            let body: Value = serde_json::from_str(&body).unwrap();
            assert!(
                head.starts_with("POST /coding/v1/messages "),
                "{source}: {head}"
            );
            assert!(
                head.to_ascii_lowercase()
                    .contains("x-msh-platform: kimi_cli"),
                "{source}: {head}"
            );
            assert_eq!(body["system"][0]["text"], "SYS PROMPT", "{source}: {body}");
            assert_eq!(body["messages"][0]["role"], "user", "{source}: {body}");
            assert!(body.get("max_tokens").is_some(), "{source}: {body}");
            assert!(body.get("store").is_none(), "{source}");
            if source != "synthesized" {
                // Synthesized models are non-reasoning (as in TS); the others enable thinking.
                assert!(
                    matches!(
                        body["thinking"]["type"].as_str(),
                        Some("enabled" | "adaptive")
                    ),
                    "{source}: {body}"
                );
            }
        }
    }

    // ---- every model source resolves its compat through the shared policy resolver

    fn cache_dir(row: Value, models_yml: &str) -> (tempfile::TempDir, AgentDir) {
        let t = tempfile::tempdir().unwrap();
        let c = rusqlite::Connection::open(t.path().join("models.db")).unwrap();
        c.execute_batch(
            "CREATE TABLE model_cache (provider_id TEXT PRIMARY KEY, version INTEGER NOT NULL, updated_at INTEGER NOT NULL, authoritative INTEGER NOT NULL DEFAULT 0, static_fingerprint TEXT NOT NULL DEFAULT '', models TEXT NOT NULL);",
        )
        .unwrap();
        c.execute(
            "INSERT INTO model_cache(provider_id, version, updated_at, models) VALUES ('groq', 1, 1, ?1)",
            [json!([row]).to_string()],
        )
        .unwrap();
        std::fs::write(t.path().join("models.yml"), models_yml).unwrap();
        let dir = AgentDir(t.path().to_path_buf());
        (t, dir)
    }

    fn groq_row(compat: Value, compat_config: Option<Value>) -> Value {
        let mut row = json!({
            "id": "llama-3.3-70b-versatile", "name": "L", "api": "openai-completions",
            "provider": "groq", "baseUrl": "https://api.groq.com/openai/v1", "reasoning": false,
            "input": ["text"], "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
            "contextWindow": 131_072, "maxTokens": 32_768, "compat": compat,
        });
        if let Some(c) = compat_config {
            row["compatConfig"] = c;
        }
        row
    }

    fn resolved_flag(m: &ResolvedModel, key: &str) -> Option<Value> {
        pa_ai::compat::resolve_for_model(&m.model)
            .unwrap()
            .compat
            .unwrap()
            .get(key)
            .cloned()
    }

    /// A cache row's stored `compat` is the TS-resolved record as of the row's write: it can be
    /// stale, so it is never used as an override set. Only the authored `compatConfig` is.
    #[test]
    fn cache_rows_resolve_from_compat_config_not_the_stale_stored_record() {
        let stale = json!({"supportsDeveloperRole": true, "supportsStore": true});
        let (_t, dir) = cache_dir(
            groq_row(stale.clone(), None),
            "providers:\n  groq:\n    apiKey: k\n",
        );
        let m = resolve_in(&dir, "groq/llama-3.3-70b-versatile").unwrap();
        assert_eq!(
            resolved_flag(&m, "supportsDeveloperRole"),
            Some(json!(false)),
            "groq takes `system`"
        );
        assert_eq!(
            resolved_flag(&m, "supportsMultipleSystemMessages"),
            Some(json!(true))
        );

        let (_t, dir) = cache_dir(
            groq_row(stale, Some(json!({"supportsDeveloperRole": true}))),
            "providers:\n  groq:\n    apiKey: k\n",
        );
        let m = resolve_in(&dir, "groq/llama-3.3-70b-versatile").unwrap();
        assert_eq!(
            resolved_flag(&m, "supportsDeveloperRole"),
            Some(json!(true)),
            "authored override wins"
        );
    }

    /// TS `mergeCompat(model.compatConfig, providerOverride.compat)`: a provider-level `compat:`
    /// block and a `modelOverrides` `compat:` layer over the model's authored overrides.
    #[test]
    fn models_yml_provider_and_model_override_compat_layer_over_the_row() {
        let yml = "providers:\n  groq:\n    apiKey: k\n    compat:\n      supportsStore: true\n      extraBody:\n        a: 1\n    modelOverrides:\n      llama-3.3-70b-versatile:\n        compat:\n          extraBody:\n            b: 2\n";
        let (_t, dir) = cache_dir(
            groq_row(json!({}), Some(json!({"supportsDeveloperRole": true}))),
            yml,
        );
        let m = resolve_in(&dir, "groq/llama-3.3-70b-versatile").unwrap();
        assert_eq!(resolved_flag(&m, "supportsStore"), Some(json!(true)));
        assert_eq!(
            resolved_flag(&m, "supportsDeveloperRole"),
            Some(json!(true)),
            "row override survives"
        );
        assert_eq!(
            resolved_flag(&m, "extraBody"),
            Some(json!({"a": 1, "b": 2})),
            "nested objects merge"
        );
    }

    /// A custom `models.yml` model resolves through the same resolver: a loopback endpoint is a
    /// local serving backend (grammar tool schemas, no first-event timeout), and the model's
    /// authored `compat` overrides the resolved record.
    #[test]
    fn custom_models_resolve_host_policy_and_authored_overrides() {
        let t = tempfile::tempdir().unwrap();
        std::fs::write(
            t.path().join("models.yml"),
            "providers:\n  local:\n    baseUrl: http://127.0.0.1:8080/v1\n    apiKey: k\n    api: openai-completions\n    models:\n      - id: qwen3-coder\n        reasoning: true\n        compat:\n          supportsDeveloperRole: true\n",
        )
        .unwrap();
        let dir = AgentDir(t.path().to_path_buf());
        let m = resolve_in(&dir, "local/qwen3-coder").unwrap();
        assert_eq!(
            resolved_flag(&m, "toolSchemaFlavor"),
            Some(json!("grammar"))
        );
        assert_eq!(
            resolved_flag(&m, "streamFirstEventTimeoutMs"),
            Some(json!(0))
        );
        assert_eq!(
            resolved_flag(&m, "supportsDeveloperRole"),
            Some(json!(true))
        );
        assert_eq!(resolved_flag(&m, "thinkingFormat"), Some(json!("qwen")));
        // The resolved ladder is materialized onto the model (qwen: reasoning effort ladder).
        assert!(m.model.thinking_level_map.is_some());
    }

    /// A `models.db` row known only by id (`contextWindow`/`maxTokens: null`, no thinking — what TS
    /// caches for `anthropic/claude-haiku-5-5`) resolves with "unknown" (0) limits, like TS `?? 0`.
    #[test]
    fn cached_row_with_null_limits_resolves() {
        let t = tempfile::tempdir().unwrap();
        let conn = rusqlite::Connection::open(t.path().join("models.db")).unwrap();
        conn.execute_batch(
            "CREATE TABLE model_cache (provider_id TEXT PRIMARY KEY, version INTEGER NOT NULL, updated_at INTEGER NOT NULL, authoritative INTEGER NOT NULL DEFAULT 0, static_fingerprint TEXT NOT NULL DEFAULT '', models TEXT NOT NULL);",
        )
        .unwrap();
        let row = json!([{
            "id": "claude-haiku-9-9", "name": "Claude Haiku 9.9", "api": "anthropic-messages",
            "provider": "anthropic", "baseUrl": "https://api.anthropic.com", "reasoning": false,
            "input": ["text"], "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
            "contextWindow": null, "maxTokens": null,
        }]);
        conn.execute(
            "INSERT INTO model_cache (provider_id, version, updated_at, authoritative, models) VALUES ('anthropic', 1, 1, 0, ?1)",
            [row.to_string()],
        )
        .unwrap();
        drop(conn);
        let dir = AgentDir(t.path().to_path_buf());
        let m = resolve_in(&dir, "anthropic/claude-haiku-9-9").unwrap();
        assert_eq!(
            (m.api.as_str(), m.context_window, m.max_tokens, m.reasoning),
            ("anthropic-messages", 0, 0, false)
        );
    }
}
