//! `resolve_model_policy` — the single model-policy resolver (TS `compat/resolve.ts`). Layering,
//! later layers win:
//!
//!   1. unconditional per-API compat defaults;
//!   2. host-derived flags (URL/provider detection, compound host×identity branches);
//!   3. compat-cascade axes compiled from the rule tree;
//!   4. spec-authored sparse overrides, then the spec-undefined fixups.
//!
//! Compat records are dynamic JSON objects. TS distinguishes "key declared with value `undefined`"
//! (an override may still land on it) from "key not declared" (override ignored), so [`Rec`]
//! tracks declared keys separately from defined values.

use std::collections::HashSet;

use serde_json::{json, Map, Value};

use super::cascade::{resolve_cascade, AmbiguousOverlap, ResolvedAxes, Target};
use super::hosts::{
    has_local_loopback_base_url, host_matches_url, is_anthropic_signing_proxy_url,
    is_azure_anthropic_route, is_bedrock_anthropic_route, is_official_anthropic_api_url,
    model_matches_host,
};
use super::revision::{compare_revision, parse_revision, Revision};
use super::taxonomy::{classify_model, strip_thinking_variant_suffix, AmbiguousIdentity, Identity};
use super::{axes, rules};

#[derive(Debug, thiserror::Error)]
pub enum ResolveError {
    #[error(transparent)]
    Identity(#[from] AmbiguousIdentity),
    #[error(transparent)]
    Cascade(#[from] AmbiguousOverlap),
    #[error("{0}")]
    Thinking(String),
}

/// Input to [`resolve_model_policy`] (the TS `ModelSpec` fields the resolver reads).
#[derive(Debug, Clone, Default)]
pub struct ModelSpec {
    pub provider: String,
    pub id: String,
    pub api: String,
    /// Empty = unset.
    pub base_url: String,
    /// Backend policy identity when it differs from the credential-bearing provider id.
    pub provider_type: Option<String>,
    pub reasoning: bool,
    /// Explicit spec thinking config (JSON object), when authored.
    pub thinking: Option<Value>,
    /// Sparse compat overrides (TS `spec.compat` / `Model.compatConfig`).
    pub compat: Option<Map<String, Value>>,
    /// Swallow identity ambiguity instead of failing (TS discovery normalization uses
    /// `classifyModel` leniently; catalog compilation stays strict).
    pub lenient: bool,
}

/// Result of resolving one model spec.
#[derive(Debug, Clone)]
pub struct ResolvedModelPolicy {
    pub identity: Identity,
    /// Complete resolved compat record; `None` for APIs without a compat shape.
    pub compat: Option<Map<String, Value>>,
    pub thinking: Option<Value>,
    pub catalog: Map<String, Value>,
    pub request: Map<String, Value>,
}

impl ResolvedModelPolicy {
    pub fn to_json(&self) -> Value {
        let mut m = Map::new();
        m.insert("identity".into(), self.identity.to_json());
        if let Some(c) = &self.compat {
            m.insert("compat".into(), Value::Object(c.clone()));
        }
        if let Some(t) = &self.thinking {
            m.insert("thinking".into(), t.clone());
        }
        m.insert("catalog".into(), Value::Object(self.catalog.clone()));
        m.insert("request".into(), Value::Object(self.request.clone()));
        Value::Object(m)
    }
}

// ---------------------------------------------------------------------------
// Compat record with declared-but-undefined keys
// ---------------------------------------------------------------------------

#[derive(Clone, Default)]
struct Rec {
    map: Map<String, Value>,
    declared: HashSet<String>,
}

impl Rec {
    fn d(&mut self, k: &str, v: impl Into<Value>) {
        self.declared.insert(k.to_string());
        self.map.insert(k.to_string(), v.into());
    }
    /// Declare a key whose value is `undefined`.
    fn u(&mut self, k: &str) {
        self.declared.insert(k.to_string());
        self.map.shift_remove(k);
    }
    fn opt(&mut self, k: &str, v: Option<Value>) {
        match v {
            Some(v) => self.d(k, v),
            None => self.u(k),
        }
    }
    fn get(&self, k: &str) -> Option<&Value> {
        self.map.get(k)
    }
    /// TS `k in compat && compat[k] === value`.
    fn is(&self, k: &str, v: bool) -> bool {
        self.map.get(k).and_then(Value::as_bool) == Some(v)
    }
    fn truthy(&self, k: &str) -> bool {
        self.map.get(k).is_some_and(truthy)
    }
    fn str_of(&self, k: &str) -> Option<&str> {
        self.map.get(k).and_then(Value::as_str)
    }
    fn declares(&self, k: &str) -> bool {
        self.declared.contains(k)
    }
}

fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
        Value::String(s) => !s.is_empty(),
        _ => true,
    }
}

/// Assign defined override values onto a freshly built record; undeclared keys are ignored.
fn apply_compat_overrides(rec: &mut Rec, overrides: Option<&Map<String, Value>>) {
    let Some(overrides) = overrides else { return };
    for (k, v) in overrides {
        if rec.declares(k) {
            rec.map.insert(k.clone(), v.clone());
        }
    }
}

/// Assign resolved wire axes onto a record when the model's API maps to one of the axis's records.
fn apply_wire_axes(rec: &mut Rec, wire: &Map<String, Value>, api: &str) {
    let ax = axes();
    let Some(records) = ax.api_records.get(api) else {
        return;
    };
    for (key, value) in wire {
        let Some(axis_records) = ax.wire.get(key) else {
            continue;
        };
        if !axis_records.iter().any(|r| records.contains(r)) {
            continue;
        }
        rec.d(key, value.clone());
    }
}

// ---------------------------------------------------------------------------
// Effort vocabulary helpers
// ---------------------------------------------------------------------------

const THINKING_EFFORTS: [&str; 6] = ["minimal", "low", "medium", "high", "xhigh", "max"];
const DEFAULT_REASONING_EFFORTS: [&str; 4] = ["minimal", "low", "medium", "high"];
const DEFAULT_REASONING_EFFORTS_WITH_XHIGH: [&str; 5] =
    ["minimal", "low", "medium", "high", "xhigh"];

fn effort_value(v: &Value) -> Option<&'static str> {
    let s = v.as_str()?;
    THINKING_EFFORTS.iter().find(|e| **e == s).copied()
}

fn effort_list(v: &Value) -> Option<Vec<&'static str>> {
    v.as_array()?.iter().map(effort_value).collect()
}

fn effort_record(v: &Value) -> Option<Map<String, Value>> {
    let obj = v.as_object()?;
    let mut out = Map::new();
    for (k, mapped) in obj {
        let Some(effort) = THINKING_EFFORTS.iter().find(|e| **e == k.as_str()) else {
            continue;
        };
        if mapped.is_string() {
            out.insert((*effort).to_string(), mapped.clone());
        }
    }
    (!out.is_empty()).then_some(out)
}

fn effort_number_record(v: &Value) -> Option<Map<String, Value>> {
    let obj = v.as_object()?;
    let mut out = Map::new();
    for (k, mapped) in obj {
        let Some(effort) = THINKING_EFFORTS.iter().find(|e| **e == k.as_str()) else {
            continue;
        };
        if mapped.is_number() {
            out.insert((*effort).to_string(), mapped.clone());
        }
    }
    (!out.is_empty()).then_some(out)
}

fn thinking_mode(v: &Value) -> Option<&'static str> {
    match v.as_str()? {
        "effort" => Some("effort"),
        "budget" => Some("budget"),
        "google-level" => Some("google-level"),
        "anthropic-adaptive" => Some("anthropic-adaptive"),
        "anthropic-budget-effort" => Some("anthropic-budget-effort"),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Identity facts
// ---------------------------------------------------------------------------

struct Facts<'a> {
    identity: &'a Identity,
    revision: Option<Revision>,
}

impl<'a> Facts<'a> {
    fn new(identity: &'a Identity) -> Self {
        Self {
            identity,
            revision: identity.revision.as_deref().and_then(parse_revision),
        }
    }
    fn is(&self, cls: &str) -> bool {
        self.identity.class == cls
    }
    fn family(&self, families: &[&str]) -> bool {
        self.identity
            .family
            .as_deref()
            .is_some_and(|f| families.contains(&f))
    }
    fn rev_gte(&self, min: &str) -> bool {
        let Some(r) = &self.revision else {
            return false;
        };
        parse_revision(min).is_some_and(|b| compare_revision(r, &b).is_ge())
    }
    fn rev_major(&self) -> Option<u32> {
        self.revision.map(|r| r[0])
    }
    fn kimi_mandatory_thinking(&self) -> bool {
        self.is("kimi") && self.family(&["k2.7-code", "k3"])
    }
    fn anthropic_adaptive_gen_at_least(&self, opus_min: &str) -> bool {
        if !self.is("anthropic") {
            return false;
        }
        if self.family(&["opus"]) {
            return self.rev_gte(opus_min);
        }
        if self.family(&["sonnet", "fable", "mythos"]) {
            return self.rev_gte("5");
        }
        if self.family(&["haiku"]) {
            return self.rev_gte("5.5");
        }
        false
    }
}

// ---------------------------------------------------------------------------
// Detected baselines
// ---------------------------------------------------------------------------

const GLM_CODING_PLAN_STREAM_IDLE_TIMEOUT_MS: i64 = 600_000;
const LOCAL_OPENAI_COMPAT_STREAM_IDLE_TIMEOUT_MS: i64 = 300_000;
const BEDROCK_REASONING_STREAM_IDLE_TIMEOUT_MS: i64 = 600_000;

fn is_local_provider(p: &str) -> bool {
    matches!(p, "llama.cpp" | "lm-studio" | "vllm" | "ollama")
}
fn is_proxy_provider(p: &str) -> bool {
    p == "litellm"
}

fn resolve_reasoning_disable_mode(format: &str) -> &'static str {
    match format {
        "openrouter" => "openrouter-enabled-false",
        "zai" | "kimi" => "zai-thinking-disabled",
        "qwen" => "qwen-enable-thinking-false",
        "qwen-chat-template" => "qwen-template-false",
        "chat-template" => "chat-template-thinking-false",
        _ => "lowest-effort",
    }
}

fn is_official_openai_endpoint(provider: &str, base_url: &str) -> bool {
    if provider != "openai" {
        return false;
    }
    if base_url.is_empty() {
        return true;
    }
    url::Url::parse(base_url).is_ok_and(|u| u.host_str() == Some("api.openai.com"))
}

fn compat_reasoning(spec: &ModelSpec, axes: &ResolvedAxes) -> bool {
    if spec.reasoning {
        return true;
    }
    axes.reasoning && read_rule_thinking(axes).upgrade_neutral == Some(true)
}

struct OpenAIDetection {
    is_cline_pass: bool,
    is_zai: bool,
    is_zhipu: bool,
    is_moonshot_native: bool,
    is_opencode_host: bool,
    is_opencode_provider: bool,
    is_direct_deepseek_api: bool,
    is_deepseek_reasoning: bool,
    is_direct_deepseek_reasoning: bool,
    is_venice: bool,
    requires_enabled_thinking: bool,
    is_xiaomi_mimo: bool,
    is_local_openai_compat_backend: bool,
    is_local_serving_backend: bool,
    is_open_router: bool,
}

fn detect_openai(spec: &ModelSpec, facts: &Facts<'_>, reasoning_capable: bool) -> OpenAIDetection {
    let provider = spec.provider.as_str();
    let backend_provider = spec.provider_type.as_deref().unwrap_or(provider);
    let base_url = spec.base_url.as_str();
    let m = |host: &str| model_matches_host(provider, base_url, host);
    let is_moonshot_native = m("moonshotNative");
    let is_xiaomi_host = m("xiaomi");
    let is_direct_deepseek_api = m("deepseekDirect");
    let is_deepseek_family = m("deepseekFamily") || facts.is("deepseek");
    let is_deepseek_reasoning = is_deepseek_family && reasoning_capable;
    let is_local_openai_compat_backend = !is_proxy_provider(backend_provider)
        && (is_local_provider(backend_provider) || has_local_loopback_base_url(base_url));
    OpenAIDetection {
        is_cline_pass: provider == "cline-pass",
        is_zai: m("zai"),
        is_zhipu: m("zhipu"),
        is_moonshot_native,
        is_opencode_host: m("opencode"),
        is_opencode_provider: provider == "opencode-go" || provider == "opencode-zen",
        is_direct_deepseek_api,
        is_deepseek_reasoning,
        is_direct_deepseek_reasoning: is_direct_deepseek_api && is_deepseek_reasoning,
        is_venice: m("venice"),
        requires_enabled_thinking: is_moonshot_native
            && facts.is("kimi")
            && facts.family(&["k2.7-code"]),
        is_xiaomi_mimo: is_xiaomi_host && facts.is("mimo"),
        is_local_openai_compat_backend,
        is_local_serving_backend: is_local_openai_compat_backend
            || has_local_loopback_base_url(base_url),
        is_open_router: m("openrouter"),
    }
}

fn detect_stream_markup_healing(
    provider: &str,
    facts: &Facts<'_>,
    base_url: &str,
) -> Option<&'static str> {
    let is_kimi_k2 = facts.is("kimi")
        && facts
            .identity
            .family
            .as_deref()
            .is_some_and(|f| f.starts_with("k2"));
    if provider == "kimi-code" || provider == "moonshot" || is_kimi_k2 {
        return Some("kimi");
    }
    if facts.is("deepseek") {
        return Some("dsml");
    }
    if is_official_openai_endpoint(provider, base_url) {
        return None;
    }
    Some("thinking")
}

#[allow(clippy::too_many_lines, clippy::fn_params_excessive_bools)]
fn detect_openai_compat(
    spec: &ModelSpec,
    d: &OpenAIDetection,
    facts: &Facts<'_>,
    reasoning_capable: bool,
) -> Rec {
    let provider = spec.provider.as_str();
    let base_url = spec.base_url.as_str();
    let m = |host: &str| model_matches_host(provider, base_url, host);
    let u = |host: &str| host_matches_url(base_url, host);
    let is_cerebras = m("cerebras");
    let is_cerebras_host = u("cerebras");
    let is_kilo = m("kilo");
    let is_alibaba = m("alibabaDashscope");
    let is_xiaomi_host = m("xiaomi");
    let is_grok = m("xai");
    let is_mistral = m("mistral");
    let is_google_aistudio_openai = u("googleAistudio");
    let is_openai_host = m("openai");
    let is_azure_host = m("azureOpenAI");
    let is_vercel_gateway = m("vercelAIGateway");
    let is_together = m("together");
    let is_fireworks = u("fireworks");
    let is_groq_host = m("groq");
    let is_minimax_host = m("minimax");
    let is_qwen_portal = m("qwenPortal");
    let is_moonshot_kimi = facts.is("kimi") && d.is_moonshot_native;
    let is_moonshot_kimi_k3 = is_moonshot_kimi && facts.family(&["k3"]);
    let uses_moonshot_kimi_preserved_thinking = is_moonshot_kimi && facts.family(&["k2.6"]);
    let is_anthropic_model = m("anthropic") || facts.is("anthropic");
    let is_qwen = facts.is("qwen");
    let is_deepseek_family = m("deepseekFamily") || facts.is("deepseek");
    let supports_zai_reasoning_effort =
        (d.is_zai || d.is_zhipu) && facts.is("glm") && facts.rev_gte("5.2");

    let is_non_standard = is_cerebras
        || is_grok
        || is_mistral
        || is_google_aistudio_openai
        || u("chutes")
        || u("deepseekFamily")
        || is_fireworks
        || is_alibaba
        || d.is_zai
        || d.is_zhipu
        || is_kilo
        || is_qwen
        || is_xiaomi_host
        || d.is_moonshot_native
        || d.is_opencode_host;

    let use_max_tokens = is_mistral
        || d.is_moonshot_native
        || d.is_zai
        || d.is_zhipu
        || u("chutes")
        || is_fireworks
        || d.is_direct_deepseek_api;

    let supports_prompt_cache_breakpoints = is_official_openai_endpoint(provider, base_url)
        && facts.is("openai")
        && facts.rev_gte("5.6");

    let supports_multiple_system_messages_default = !is_minimax_host
        && !is_alibaba
        && !is_qwen_portal
        && !is_qwen
        && (is_openai_host
            || is_azure_host
            || d.is_open_router
            || is_cerebras
            || is_together
            || is_fireworks
            || is_groq_host
            || is_deepseek_family
            || is_mistral
            || is_grok
            || d.is_zai
            || d.is_zhipu
            || provider == "github-copilot"
            || provider == "zenmux");

    let stream_idle_timeout_ms: Option<i64> = if facts.is("glm")
        && facts.rev_gte("5")
        && !facts.family(&["vision"])
        && (u("zai") || u("zhipu") || u("opencode"))
    {
        Some(GLM_CODING_PLAN_STREAM_IDLE_TIMEOUT_MS)
    } else if facts.is("mimo") && u("xiaomi") {
        Some(300_000)
    } else if reasoning_capable
        && facts.is("kimi")
        && (facts.family(&["k3"]) || facts.family(&["k2.7-code"]))
        && u("moonshotNative")
    {
        Some(300_000)
    } else if reasoning_capable && facts.is("deepseek") && u("deepseekDirect") {
        Some(300_000)
    } else if d.is_local_serving_backend {
        Some(LOCAL_OPENAI_COMPAT_STREAM_IDLE_TIMEOUT_MS)
    } else {
        None
    };

    let wire_model_id_mode = if u("openrouter") { "openrouter" } else { "raw" };

    let thinking_format = if (facts.is("kimi") && !facts.family(&["k3"]) && u("moonshotNative"))
        || u("zai")
        || u("zhipu")
        || (facts.is("mimo") && u("xiaomi"))
    {
        "zai"
    } else if u("openrouter") {
        "openrouter"
    } else if is_qwen && u("nvidia") {
        "qwen-chat-template"
    } else if is_qwen && (is_fireworks || u("venice")) {
        "openai"
    } else if u("alibabaDashscope") || is_qwen {
        "qwen"
    } else {
        "openai"
    };

    let mut r = Rec::default();
    r.d("supportsStore", !is_non_standard);
    r.d("supportsDeveloperRole", is_openai_host || is_azure_host);
    r.d(
        "supportsMultipleSystemMessages",
        supports_multiple_system_messages_default,
    );
    r.d(
        "supportsReasoningEffort",
        !is_grok
            && !d.is_xiaomi_mimo
            && (!(d.is_zai || d.is_zhipu) || supports_zai_reasoning_effort),
    );
    r.u("trustExplicitThinkingOnly");
    r.d("supportsReasoningParams", provider != "github-copilot");
    r.d("supportsSamplingParams", true);
    r.d(
        "supportsPenaltyAndStopParams",
        !(is_grok && reasoning_capable),
    );
    r.d("reasoningEffortMap", json!({}));
    r.d("supportsUsageInStreaming", !is_cerebras_host);
    r.d("alwaysSendMaxTokens", facts.is("kimi"));
    r.d(
        "disableReasoningOnForcedToolChoice",
        !d.is_cline_pass && ((facts.is("kimi") && !is_moonshot_kimi_k3) || is_anthropic_model),
    );
    r.d(
        "disableReasoningOnToolChoice",
        !d.is_cline_pass && is_deepseek_family && reasoning_capable && !d.is_open_router,
    );
    r.d("disableReasoningWithTools", false);
    r.d(
        "supportsToolChoice",
        d.is_cline_pass || !d.is_direct_deepseek_reasoning,
    );
    r.d(
        "supportsForcedToolChoice",
        !d.requires_enabled_thinking
            && !(d.is_opencode_host && d.is_deepseek_reasoning)
            && !(d.is_cline_pass && is_qwen),
    );
    r.d("supportsNamedToolChoice", true);
    r.d(
        "maxTokensField",
        if use_max_tokens {
            "max_tokens"
        } else {
            "max_completion_tokens"
        },
    );
    r.d("requiresToolResultName", is_mistral);
    r.d("requiresAssistantAfterToolResult", is_mistral);
    r.d("requiresThinkingAsText", is_mistral);
    r.d("requiresMistralToolIds", is_mistral);
    r.d("thinkingFormat", thinking_format);
    r.u("kimiApiFormat");
    r.d(
        "reasoningDisableMode",
        if d.is_cline_pass {
            "cline-enabled-false"
        } else if d.is_venice {
            "venice-disable-thinking"
        } else {
            resolve_reasoning_disable_mode(thinking_format)
        },
    );
    r.d("omitReasoningEffort", false);
    r.d("includeEncryptedReasoning", true);
    r.d(
        "filterReasoningHistory",
        d.is_open_router && is_anthropic_model,
    );
    r.opt(
        "thinkingKeep",
        uses_moonshot_kimi_preserved_thinking.then(|| json!("all")),
    );
    r.d(
        "reasoningContentField",
        if d.is_cline_pass {
            "reasoning"
        } else {
            "reasoning_content"
        },
    );
    r.u("mistralReasoningContentParts");
    r.d(
        "requiresReasoningContentForToolCalls",
        (facts.is("kimi") && !d.is_opencode_provider)
            || (is_deepseek_family && reasoning_capable)
            || d.is_xiaomi_mimo
            || (d.is_open_router && reasoning_capable),
    );
    r.d(
        "requiresReasoningContentForAllAssistantTurns",
        ((is_deepseek_family && reasoning_capable) || d.is_xiaomi_mimo) && !d.is_open_router,
    );
    r.d(
        "allowsSyntheticReasoningContentForToolCalls",
        (!is_deepseek_family || !reasoning_capable) && !d.is_xiaomi_mimo,
    );
    r.u("syntheticReasoningContentFallback");
    r.d("replayReasoningContent", d.is_local_openai_compat_backend);
    r.d(
        "qwenPreserveThinking",
        (thinking_format == "qwen" || thinking_format == "qwen-chat-template")
            && d.is_local_openai_compat_backend,
    );
    r.d("qwenTemplateReasoningEffort", false);
    r.d(
        "requiresAssistantContentForToolCalls",
        facts.is("kimi") || d.is_direct_deepseek_reasoning,
    );
    r.opt(
        "cacheControlFormat",
        ((d.is_cline_pass && (is_qwen || is_anthropic_model))
            || (d.is_open_router && is_anthropic_model))
            .then(|| json!("anthropic")),
    );
    r.d(
        "supportsPromptCacheBreakpoints",
        supports_prompt_cache_breakpoints,
    );
    r.opt(
        "promptCacheBreakpointTtl",
        supports_prompt_cache_breakpoints.then(|| json!("30m")),
    );
    r.u("openRouterRouting");
    r.u("vercelGatewayRouting");
    r.d("isOpenRouterHost", d.is_open_router);
    r.d("wireModelIdMode", wire_model_id_mode);
    r.d("isVercelGatewayHost", is_vercel_gateway);
    r.d(
        "supportsStrictMode",
        u("openai")
            || u("azureOpenAI")
            || u("cerebras")
            || u("together")
            || u("openrouter")
            || u("deepseekFamily"),
    );
    r.u("extraBody");
    r.d(
        "toolStrictMode",
        if is_cerebras_host {
            "all_strict"
        } else {
            "mixed"
        },
    );
    r.opt(
        "toolSchemaFlavor",
        if d.is_moonshot_native || facts.is("kimi") {
            Some(json!("moonshot-mfjs"))
        } else if d.is_local_openai_compat_backend {
            Some(json!("grammar"))
        } else {
            None
        },
    );
    r.opt(
        "streamFirstEventTimeoutMs",
        d.is_local_serving_backend.then(|| json!(0)),
    );
    r.opt(
        "streamIdleTimeoutMs",
        stream_idle_timeout_ms.map(|v| json!(v)),
    );
    r.d(
        "stripDeepseekSpecialTokens",
        facts.is("deepseek") && (provider == "nvidia" || provider == "deepseek"),
    );
    r.opt(
        "streamMarkupHealingPattern",
        detect_stream_markup_healing(provider, facts, base_url).map(|v| json!(v)),
    );
    r.d("reasoningDeltasMayBeCumulative", false);
    r.d("emptyLengthFinishIsContextError", false);
    r.d("usesOpenAIToolCallIdLimit", false);
    r.opt(
        "promptCacheSessionHeader",
        u("xai").then(|| json!("x-grok-conv-id")),
    );
    r.d("dropThinkingWhenReasoningEffort", false);
    r.d("nativeKimiK3Reasoning", false);
    r.d("zaiReasoningEffortDialect", false);
    r.d("clampOutputToModelMax", d.is_local_openai_compat_backend);
    r.d("stripImageInput", false);
    r.u("thinkingLoopGuard");
    r.d("rejectRootObjectUnion", false);
    r.d("retryWithoutStrictOnGrammarError", false);
    r.d("supportsPromptCacheKey", false);
    r
}

/// Merge a sparse effort map over a base (spread semantics).
fn spread_maps(parts: &[Option<&Map<String, Value>>]) -> Map<String, Value> {
    let mut out = Map::new();
    for p in parts.iter().flatten() {
        for (k, v) in *p {
            out.insert(k.clone(), v.clone());
        }
    }
    out
}

/// Per-key overlay: spec `reasoningEffortMap` keys win over the cascade-provided map.
fn overlay_effort_map_axis(
    rec: &mut Rec,
    axes: &ResolvedAxes,
    spec_compat: Option<&Map<String, Value>>,
) {
    let axis_map = axes.wire.get("reasoningEffortMap").and_then(effort_record);
    let spec_map = spec_compat
        .and_then(|c| c.get("reasoningEffortMap"))
        .and_then(Value::as_object);
    if let (Some(axis_map), Some(spec_map)) = (axis_map, spec_map) {
        rec.d(
            "reasoningEffortMap",
            Value::Object(spread_maps(&[Some(&axis_map), Some(spec_map)])),
        );
    }
}

fn spec_compat_defined(spec: &ModelSpec, key: &str) -> bool {
    spec.compat.as_ref().is_some_and(|c| c.contains_key(key))
}

fn fixup_openai_compat(spec: &ModelSpec, rec: &mut Rec, d: &OpenAIDetection, axes: &ResolvedAxes) {
    // Direct DeepSeek reasoning: the rules own the `thinking: enabled` toggle via `whenThinking`.
    if d.is_direct_deepseek_reasoning {
        if let Some(extra) = rec.get("extraBody").and_then(Value::as_object) {
            let enabled = extra
                .get("thinking")
                .and_then(Value::as_object)
                .and_then(|t| t.get("type"))
                .and_then(Value::as_str)
                == Some("enabled");
            if enabled {
                let mut extra = extra.clone();
                extra.shift_remove("thinking");
                if extra.is_empty() {
                    rec.u("extraBody");
                } else {
                    rec.d("extraBody", Value::Object(extra));
                }
            }
        }
    }
    if !spec_compat_defined(spec, "reasoningDisableMode")
        && !axes.wire.contains_key("reasoningDisableMode")
    {
        let fmt = rec.str_of("thinkingFormat").unwrap_or("openai").to_string();
        let mode = if d.is_cline_pass {
            "cline-enabled-false"
        } else if d.requires_enabled_thinking {
            "omit"
        } else if d.is_direct_deepseek_reasoning {
            "zai-thinking-disabled"
        } else if d.is_venice {
            "venice-disable-thinking"
        } else {
            resolve_reasoning_disable_mode(&fmt)
        };
        rec.d("reasoningDisableMode", mode);
    }
    if !spec_compat_defined(spec, "omitReasoningEffort")
        && !axes.wire.contains_key("omitReasoningEffort")
        && !rec.truthy("supportsReasoningEffort")
    {
        rec.d("omitReasoningEffort", true);
    }

    let axis_when_thinking = if compat_reasoning(spec, axes) {
        axes.wire
            .get("whenThinking")
            .filter(|v| v.is_object())
            .cloned()
    } else {
        None
    };
    let spec_when = spec
        .compat
        .as_ref()
        .and_then(|c| c.get("whenThinking"))
        .filter(|v| !v.is_null())
        .cloned();
    let when_policy = spec_when.or(axis_when_thinking).or_else(|| {
        d.is_direct_deepseek_reasoning.then(|| {
            let mut extra = rec
                .get("extraBody")
                .and_then(Value::as_object)
                .cloned()
                .unwrap_or_default();
            extra.insert("thinking".into(), json!({"type": "enabled"}));
            json!({ "extraBody": extra })
        })
    });
    match when_policy.as_ref().and_then(Value::as_object) {
        Some(policy) => {
            let mut variant = rec.clone();
            variant.u("whenThinking");
            apply_compat_overrides(&mut variant, Some(policy));
            if policy.get("reasoningDisableMode").is_none() {
                let fmt = variant
                    .str_of("thinkingFormat")
                    .unwrap_or("openai")
                    .to_string();
                variant.d(
                    "reasoningDisableMode",
                    if d.is_venice {
                        "venice-disable-thinking"
                    } else {
                        resolve_reasoning_disable_mode(&fmt)
                    },
                );
            }
            if policy.get("omitReasoningEffort").is_none()
                && !variant.truthy("supportsReasoningEffort")
            {
                variant.d("omitReasoningEffort", true);
            }
            rec.d("whenThinking", Value::Object(variant.map));
        }
        None => rec.u("whenThinking"),
    }
}

fn resolve_openai_completions_policy(
    spec: &ModelSpec,
    facts: &Facts<'_>,
    axes: &ResolvedAxes,
) -> Rec {
    let reasoning_capable = compat_reasoning(spec, axes);
    let d = detect_openai(spec, facts, reasoning_capable);
    let mut rec = detect_openai_compat(spec, &d, facts, reasoning_capable);
    apply_wire_axes(&mut rec, &axes.wire, "openai-completions");
    apply_compat_overrides(&mut rec, spec.compat.as_ref());
    overlay_effort_map_axis(&mut rec, axes, spec.compat.as_ref());
    fixup_openai_compat(spec, &mut rec, &d, axes);
    rec
}

#[allow(clippy::too_many_lines)]
fn resolve_openai_responses_policy(
    spec: &ModelSpec,
    facts: &Facts<'_>,
    axes: &ResolvedAxes,
    api: &str,
) -> Rec {
    let base_url = spec.base_url.as_str();
    let provider = spec.provider.as_str();
    let backend_provider = spec.provider_type.as_deref().unwrap_or(provider);
    let m = |host: &str| model_matches_host(provider, base_url, host);
    let u = |host: &str| host_matches_url(base_url, host);
    let is_azure = m("azureOpenAI");
    let is_open_router = m("openrouter");
    let is_openai_url = u("openai");
    let is_vercel_gateway = m("vercelAIGateway");
    let is_xai_host = m("xai");
    let supports_prompt_cache_breakpoints = is_official_openai_endpoint(provider, base_url)
        && facts.is("openai")
        && facts.rev_gte("5.6");
    let thinking_format = if is_open_router {
        "openrouter"
    } else {
        "openai"
    };
    let reasoning_capable = compat_reasoning(spec, axes);
    let is_local_serving_backend = (!is_proxy_provider(backend_provider)
        && is_local_provider(backend_provider))
        || has_local_loopback_base_url(base_url);
    let is_anthropic_model = facts.is("anthropic");
    let is_deepseek_family = facts.is("deepseek");

    let mut r = Rec::default();
    r.d(
        "supportsDeveloperRole",
        is_azure || is_openai_url || u("githubCopilot"),
    );
    r.d(
        "supportsStrictMode",
        is_azure
            || u("openai")
            || u("azureOpenAI")
            || u("cerebras")
            || u("together")
            || u("openrouter")
            || u("deepseekFamily"),
    );
    r.d("supportsReasoningEffort", !is_xai_host);
    r.u("trustExplicitThinkingOnly");
    r.d("supportsLongPromptCacheRetention", is_openai_url);
    r.d(
        "supportsPromptCacheBreakpoints",
        supports_prompt_cache_breakpoints,
    );
    r.opt(
        "promptCacheBreakpointTtl",
        supports_prompt_cache_breakpoints.then(|| json!("30m")),
    );
    r.d(
        "strictResponsesPairing",
        is_azure || provider == "github-copilot",
    );
    r.d(
        "supportsImageDetailOriginal",
        is_openai_url || u("azureOpenAI") || (is_azure && base_url.is_empty()) || u("openaiCodex"),
    );
    r.d("supportsReasoningSummary", !is_xai_host);
    r.u("statefulResponses");
    r.d("supportsAllTurnsReasoningContext", false);
    r.d("supportsConfigurationUpdate", false);
    r.d("supportsSteering", false);
    r.d("requiresReasoningOffJuiceInstruction", false);
    r.d("stripImageInput", false);
    r.u("thinkingLoopGuard");
    r.d("reasoningEffortMap", json!({}));
    r.d("supportsReasoningParams", true);
    r.d("supportsSamplingParams", true);
    r.d("supportsPenaltyAndStopParams", !is_xai_host);
    r.d("thinkingFormat", thinking_format);
    r.d(
        "reasoningDisableMode",
        resolve_reasoning_disable_mode(thinking_format),
    );
    r.d("omitReasoningEffort", false);
    r.d("includeEncryptedReasoning", true);
    r.d(
        "filterReasoningHistory",
        is_open_router && is_anthropic_model,
    );
    r.d("disableReasoningOnForcedToolChoice", facts.is("kimi"));
    r.d(
        "disableReasoningOnToolChoice",
        is_deepseek_family && reasoning_capable && !is_open_router,
    );
    r.d("disableReasoningWithTools", false);
    r.d("supportsToolChoice", true);
    r.d(
        "supportsForcedToolChoice",
        provider != "opencode-go" && provider != "opencode-zen",
    );
    r.d("supportsNamedToolChoice", true);
    r.d("reasoningContentField", "reasoning_content");
    r.d(
        "requiresReasoningContentForToolCalls",
        (facts.is("kimi")
            || (is_deepseek_family && reasoning_capable)
            || (is_open_router && reasoning_capable))
            && reasoning_capable,
    );
    r.d(
        "requiresReasoningContentForAllAssistantTurns",
        is_deepseek_family && reasoning_capable && !is_open_router,
    );
    r.d(
        "allowsSyntheticReasoningContentForToolCalls",
        !is_deepseek_family || !reasoning_capable,
    );
    r.u("syntheticReasoningContentFallback");
    r.d("replayReasoningContent", false);
    r.d("qwenPreserveThinking", false);
    r.d("qwenTemplateReasoningEffort", false);
    r.d("requiresThinkingAsText", false);
    r.d("requiresMistralToolIds", false);
    r.d("requiresToolResultName", false);
    r.d("requiresAssistantAfterToolResult", false);
    r.d("requiresAssistantContentForToolCalls", facts.is("kimi"));
    r.u("openRouterRouting");
    r.u("vercelGatewayRouting");
    r.d("isOpenRouterHost", is_open_router);
    r.d("isVercelGatewayHost", is_vercel_gateway);
    r.d(
        "wireModelIdMode",
        if is_open_router { "openrouter" } else { "raw" },
    );
    r.opt(
        "toolSchemaFlavor",
        facts.is("kimi").then(|| json!("moonshot-mfjs")),
    );
    r.d("alwaysSendMaxTokens", facts.is("kimi"));
    r.d(
        "clampOutputToModelMax",
        !is_proxy_provider(backend_provider)
            && (is_local_provider(backend_provider) || has_local_loopback_base_url(base_url)),
    );
    r.d(
        "supportsObfuscationOptOut",
        is_openai_url || provider == "openai",
    );
    r.d("storeResponses", false);
    r.d("connectionBoundNativeHistory", false);
    r.d(
        "officialEndpoint",
        is_official_openai_endpoint(provider, base_url),
    );
    r.d("harmonyLeakMitigation", false);
    r.d("rejectRootObjectUnion", false);
    r.d("retryWithoutStrictOnGrammarError", false);
    r.opt(
        "cacheControlFormat",
        (is_open_router && is_anthropic_model).then(|| json!("anthropic")),
    );
    r.d(
        "stripDeepseekSpecialTokens",
        facts.is("deepseek") && (provider == "nvidia" || provider == "deepseek"),
    );
    r.opt(
        "streamMarkupHealingPattern",
        detect_stream_markup_healing(provider, facts, base_url).map(|v| json!(v)),
    );
    r.d("reasoningDeltasMayBeCumulative", false);
    r.d("emptyLengthFinishIsContextError", false);
    r.d("usesOpenAIToolCallIdLimit", false);
    r.opt(
        "promptCacheSessionHeader",
        u("xai").then(|| json!("x-grok-conv-id")),
    );
    r.opt(
        "streamFirstEventTimeoutMs",
        if is_local_serving_backend {
            Some(json!(0))
        } else {
            spec.compat
                .as_ref()
                .and_then(|c| c.get("streamFirstEventTimeoutMs"))
                .cloned()
        },
    );
    r.opt(
        "streamIdleTimeoutMs",
        if is_local_serving_backend {
            Some(json!(LOCAL_OPENAI_COMPAT_STREAM_IDLE_TIMEOUT_MS))
        } else {
            spec.compat
                .as_ref()
                .and_then(|c| c.get("streamIdleTimeoutMs"))
                .cloned()
        },
    );
    apply_wire_axes(&mut r, &axes.wire, api);
    apply_compat_overrides(&mut r, spec.compat.as_ref());
    overlay_effort_map_axis(&mut r, axes, spec.compat.as_ref());
    if is_xai_host && axes.wire.contains_key("reasoningEffortMap") {
        let canonical = axes
            .wire
            .get("reasoningEffortMap")
            .and_then(effort_record)
            .unwrap_or_default();
        let current = r
            .get("reasoningEffortMap")
            .and_then(Value::as_object)
            .cloned();
        let mut merged = spread_maps(&[current.as_ref(), Some(&canonical)]);
        for key in ["xhigh", "max"] {
            if !canonical.contains_key(key) {
                merged.shift_remove(key);
            }
        }
        r.d("reasoningEffortMap", Value::Object(merged));
    }
    if !spec_compat_defined(spec, "reasoningDisableMode")
        && !axes.wire.contains_key("reasoningDisableMode")
    {
        let fmt = r.str_of("thinkingFormat").unwrap_or("openai").to_string();
        r.d("reasoningDisableMode", resolve_reasoning_disable_mode(&fmt));
    }
    if !spec_compat_defined(spec, "omitReasoningEffort")
        && !axes.wire.contains_key("omitReasoningEffort")
        && !r.truthy("supportsReasoningEffort")
    {
        r.d("omitReasoningEffort", true);
    }
    if provider == "xai-oauth"
        && axes
            .wire
            .get("supportsReasoningEffort")
            .and_then(Value::as_bool)
            == Some(true)
        && spec
            .compat
            .as_ref()
            .and_then(|c| c.get("supportsReasoningEffort"))
            .and_then(Value::as_bool)
            != Some(false)
    {
        r.d("supportsReasoningEffort", true);
        r.d("omitReasoningEffort", false);
    }
    r
}

const RESPONSES_ONLY_KEYS: [&str; 16] = [
    "supportsLongPromptCacheRetention",
    "strictResponsesPairing",
    "supportsImageDetailOriginal",
    "supportsObfuscationOptOut",
    "storeResponses",
    "connectionBoundNativeHistory",
    "supportsAllTurnsReasoningContext",
    "supportsConfigurationUpdate",
    "supportsSteering",
    "officialEndpoint",
    "harmonyLeakMitigation",
    "cacheControlFormat",
    "requiresReasoningOffJuiceInstruction",
    "supportsReasoningSummary",
    "statefulResponses",
    "isVercelGatewayHost",
];

fn resolve_anthropic_policy(spec: &ModelSpec, facts: &Facts<'_>, axes: &ResolvedAxes) -> Rec {
    let base_url = spec.base_url.as_str();
    let provider = spec.provider.as_str();
    let m = |host: &str| model_matches_host(provider, base_url, host);
    let official = is_official_anthropic_api_url(base_url);
    let is_copilot = m("githubCopilot");
    let is_zenmux = m("zenmux");
    let requires_thinking_enabled = m("moonshotNative") && facts.kimi_mandatory_thinking();
    let bedrock_messages_api = is_bedrock_anthropic_route(base_url);
    let rejects_strict_tools = is_azure_anthropic_route(base_url) || bedrock_messages_api;
    let signing_endpoint =
        official || is_copilot || is_zenmux || is_anthropic_signing_proxy_url(base_url);
    let mut r = Rec::default();
    r.d("officialEndpoint", official);
    r.d("signingEndpoint", signing_endpoint);
    r.d("supportsContextManagement", true);
    r.d("supportsServerCompaction", false);
    r.d("firstPartyProvider", false);
    r.d("supportsOutputEffort", true);
    r.d("disableStrictTools", rejects_strict_tools);
    r.opt(
        "bedrockMessagesApi",
        bedrock_messages_api.then(|| json!(true)),
    );
    r.d("disableAdaptiveThinking", false);
    r.d("allowAnthropicHeaderOverrides", false);
    r.d("supportsEagerToolInputStreaming", official);
    r.d("supportsLongCacheRetention", official);
    r.d(
        "supportsMidConversationSystem",
        official && !facts.family(&["sonnet"]) && facts.anthropic_adaptive_gen_at_least("4.8"),
    );
    r.d("supportsTurnScopedSystem", false);
    r.d("supportsMidConversationToolChanges", false);
    r.d("supportsPerMessageEffort", false);
    r.d("supportsThinkingBindingControls", false);
    r.d("supportsBetweenToolsThinking", false);
    r.d(
        "supportsForcedToolChoice",
        !requires_thinking_enabled && !facts.family(&["fable", "mythos"]),
    );
    r.d("supportsSamplingParams", true);
    r.d("requiresToolResultId", false);
    r.d("requiresThinkingEnabled", requires_thinking_enabled);
    r.d(
        "replayUnsignedThinking",
        !signing_endpoint && (compat_reasoning(spec, axes) || m("deepseekFamily")),
    );
    r.d("escapeBuiltinToolNames", false);
    r.d("injectClaudeCodeInstruction", true);
    r.d("stripImageInput", false);
    r.u("maxImageDimension");
    r.u("maxImagePayloadBytes");
    r.u("thinkingLoopGuard");
    r.u("stripThinkingHistory");
    r.u("fastMode");
    r.opt(
        "streamIdleTimeoutMs",
        spec.compat
            .as_ref()
            .and_then(|c| c.get("streamIdleTimeoutMs"))
            .cloned(),
    );
    apply_wire_axes(&mut r, &axes.wire, "anthropic-messages");
    apply_compat_overrides(&mut r, spec.compat.as_ref());
    r
}

fn resolve_bedrock_policy(spec: &ModelSpec, axes: &ResolvedAxes) -> Rec {
    let mut r = Rec::default();
    r.d("promptCacheMode", "none");
    r.d("supportsLongPromptCacheRetention", false);
    r.d("promptCacheMinimumTokens", 0);
    r.d("promptCacheMaximumCheckpoints", 0);
    r.d("supportsForcedToolChoice", true);
    r.d("supportsThinkingBindingControls", true);
    r.opt(
        "streamIdleTimeoutMs",
        compat_reasoning(spec, axes).then(|| json!(BEDROCK_REASONING_STREAM_IDLE_TIMEOUT_MS)),
    );
    apply_wire_axes(&mut r, &axes.wire, "bedrock-converse-stream");
    apply_compat_overrides(&mut r, spec.compat.as_ref());
    r
}

fn resolve_devin_policy(spec: &ModelSpec, axes: &ResolvedAxes) -> Rec {
    let mut r = Rec::default();
    r.d("trustExplicitThinkingOnly", true);
    r.d("modelRouter", false);
    r.d("supportsParallelToolCalls", false);
    apply_wire_axes(&mut r, &axes.wire, "devin-agent");
    apply_compat_overrides(&mut r, spec.compat.as_ref());
    r
}

fn resolve_google_policy(spec: &ModelSpec, axes: &ResolvedAxes) -> Rec {
    let mut r = Rec::default();
    for k in [
        "supportsFunctionPartId",
        "requiresSkipThoughtSignature",
        "requiresSkipThoughtSignatureOnFirstFunctionCall",
        "dropUnsignedThinking",
        "ccaLegacyParametersSchema",
        "multimodalFunctionResponse",
        "flashStreamLeakWorkaround",
        "claudeThinkingBetaHeader",
        "antigravityClaudeToolMode",
        "stripImageInput",
    ] {
        r.d(k, false);
    }
    apply_wire_axes(&mut r, &axes.wire, &spec.api);
    apply_compat_overrides(&mut r, spec.compat.as_ref());
    r
}

// ---------------------------------------------------------------------------
// Thinking resolution
// ---------------------------------------------------------------------------

fn omits_wire_reasoning_effort(api: &str, compat: Option<&Rec>) -> bool {
    if !matches!(
        api,
        "openai-responses" | "openai-codex-responses" | "azure-openai-responses"
    ) {
        return false;
    }
    compat.is_some_and(|c| c.is("supportsReasoningEffort", false))
}

fn read_compat_effort_map(compat: Option<&Rec>) -> Option<Map<String, Value>> {
    let map = compat?.get("reasoningEffortMap")?.as_object()?;
    (!map.is_empty()).then(|| map.clone())
}

fn default_thinking_mode(spec: &ModelSpec, facts: &Facts<'_>) -> &'static str {
    match spec.api.as_str() {
        "google-generative-ai" | "google-gemini-cli" | "google-vertex" => {
            if facts.is("gemini") && facts.rev_major() == Some(3) {
                "google-level"
            } else {
                "budget"
            }
        }
        "anthropic-messages" => {
            if facts.is("minimax") && facts.family(&["m2", "m3"]) {
                return "anthropic-adaptive";
            }
            if facts.is("glm")
                && facts.rev_gte("5.2")
                && (spec.provider == "umans" || spec.provider == "zai")
            {
                return "anthropic-budget-effort";
            }
            if facts.is("anthropic") {
                if facts.rev_gte(if facts.family(&["haiku"]) {
                    "5.5"
                } else {
                    "4.6"
                }) {
                    return "anthropic-adaptive";
                }
                if facts.family(&["opus"]) && facts.rev_gte("4.5") {
                    return "anthropic-budget-effort";
                }
            }
            "budget"
        }
        "bedrock-converse-stream" => {
            if facts.is("anthropic") {
                if facts.anthropic_adaptive_gen_at_least("4.6") {
                    return "anthropic-adaptive";
                }
                if facts.family(&["opus"]) && facts.rev_gte("4.5") {
                    return "anthropic-budget-effort";
                }
            }
            if facts.is("openai") {
                return "effort";
            }
            "budget"
        }
        _ => "effort",
    }
}

fn fallback_efforts(spec: &ModelSpec, compat: Option<&Rec>) -> Vec<&'static str> {
    let api = spec.api.as_str();
    if api == "anthropic-messages" {
        return DEFAULT_REASONING_EFFORTS_WITH_XHIGH.to_vec();
    }
    if api == "bedrock-converse-stream" {
        return DEFAULT_REASONING_EFFORTS.to_vec();
    }
    if matches!(api, "openai-completions" | "openrouter") {
        if let Some(c) = compat.filter(|c| c.declares("thinkingFormat")) {
            if c.str_of("thinkingFormat") == Some("openai") && c.truthy("supportsReasoningEffort") {
                return DEFAULT_REASONING_EFFORTS_WITH_XHIGH.to_vec();
            }
            return DEFAULT_REASONING_EFFORTS.to_vec();
        }
    }
    if matches!(
        api,
        "openai-responses" | "openai-codex-responses" | "azure-openai-responses"
    ) {
        return DEFAULT_REASONING_EFFORTS_WITH_XHIGH.to_vec();
    }
    DEFAULT_REASONING_EFFORTS.to_vec()
}

fn filter_effort_map(map: &Map<String, Value>, efforts: &[&str]) -> Option<Map<String, Value>> {
    let mut filtered: Option<Map<String, Value>> = None;
    for effort in efforts {
        let Some(mapped) = map.get(*effort) else {
            continue;
        };
        filtered
            .get_or_insert_with(Map::new)
            .insert((*effort).to_string(), mapped.clone());
    }
    filtered
}

#[derive(Default)]
struct RuleThinking {
    mode: Option<&'static str>,
    efforts: Option<Vec<&'static str>>,
    default_level: Option<&'static str>,
    effort_map: Option<Map<String, Value>>,
    effort_budgets: Option<Map<String, Value>>,
    requires_effort: Option<bool>,
    suppress_when_off: Option<bool>,
    supports_display: Option<bool>,
    prefix_binding: Option<bool>,
    upgrade_neutral: Option<bool>,
}

fn read_rule_thinking(axes: &ResolvedAxes) -> RuleThinking {
    let raw = &axes.thinking;
    let b = |k: &str| raw.get(k).and_then(Value::as_bool);
    RuleThinking {
        mode: raw.get("mode").and_then(thinking_mode),
        efforts: raw.get("efforts").and_then(effort_list),
        default_level: raw.get("defaultLevel").and_then(effort_value),
        effort_map: raw.get("effortMap").and_then(effort_record),
        effort_budgets: raw.get("effortBudgets").and_then(effort_number_record),
        requires_effort: b("requiresEffort"),
        suppress_when_off: b("suppressWhenOff"),
        supports_display: b("supportsDisplay"),
        prefix_binding: b("prefixBinding"),
        upgrade_neutral: b("upgradeNeutral"),
    }
}

fn implies_mandatory_reasoning(facts: &Facts<'_>, model_id: &str) -> bool {
    facts.identity.thinking_variant || strip_thinking_variant_suffix(rules(), model_id).is_some()
}

fn default_supports_display(spec: &ModelSpec, facts: &Facts<'_>) -> bool {
    matches!(
        spec.api.as_str(),
        "anthropic-messages" | "bedrock-converse-stream"
    ) && facts.anthropic_adaptive_gen_at_least("4.7")
}

fn merge_effort_map(
    spec: &ModelSpec,
    rule_map: Option<&Map<String, Value>>,
    compat: Option<&Rec>,
    efforts: &[&str],
) -> Option<Map<String, Value>> {
    let detected: Option<Map<String, Value>> =
        (matches!(spec.api.as_str(), "openai-completions" | "openrouter")
            && model_matches_host(&spec.provider, &spec.base_url, "fireworks"))
        .then(|| {
            let mut m = Map::new();
            m.insert("minimal".into(), json!("none"));
            m
        });
    let configured = read_compat_effort_map(compat);
    if detected.is_none() && rule_map.is_none() && configured.is_none() {
        return None;
    }
    filter_effort_map(
        &spread_maps(&[detected.as_ref(), rule_map, configured.as_ref()]),
        efforts,
    )
}

fn is_qwen_template_reasoning_effort_compat(compat: Option<&Rec>) -> bool {
    compat.is_some_and(|c| c.is("qwenTemplateReasoningEffort", true))
}

fn resolve_thinking_policy(
    spec: &ModelSpec,
    facts: &Facts<'_>,
    axes: &ResolvedAxes,
    compat: Option<&Rec>,
) -> Result<Option<Value>, ResolveError> {
    let rule = read_rule_thinking(axes);
    let explicit = spec.thinking.as_ref().filter(|t| {
        t.get("efforts")
            .and_then(Value::as_array)
            .is_some_and(|a| !a.is_empty())
    });
    if !spec.reasoning && (explicit.is_some() || rule.upgrade_neutral != Some(true)) {
        return Ok(None);
    }
    if spec.provider == "cline-pass"
        && compat.is_some_and(|c| c.is("supportsReasoningEffort", false))
    {
        return Ok(None);
    }
    if omits_wire_reasoning_effort(&spec.api, compat) {
        return Ok(None);
    }
    if let Some(explicit) = explicit {
        return Ok(Some(fill_explicit_thinking(
            spec, facts, compat, explicit, &rule,
        )));
    }
    if compat.is_some_and(|c| c.is("trustExplicitThinkingOnly", true)) {
        return Ok(None);
    }
    if rule.efforts.as_ref().is_some_and(Vec::is_empty) {
        return Ok(None);
    }
    let mode = rule
        .mode
        .unwrap_or_else(|| default_thinking_mode(spec, facts));
    let efforts = rule
        .efforts
        .clone()
        .unwrap_or_else(|| fallback_efforts(spec, compat));
    if efforts.is_empty() {
        return Err(ResolveError::Thinking(format!(
            "Model {}/{} resolved to an empty thinking range",
            spec.provider, spec.id
        )));
    }
    let mut config = Map::new();
    config.insert("mode".into(), json!(mode));
    config.insert("efforts".into(), json!(efforts));
    if let Some(d) = rule.default_level {
        config.insert("defaultLevel".into(), json!(d));
    }
    if let Some(map) = merge_effort_map(spec, rule.effort_map.as_ref(), compat, &efforts) {
        config.insert("effortMap".into(), Value::Object(map));
    }
    if let Some(b) = &rule.effort_budgets {
        config.insert("effortBudgets".into(), Value::Object(b.clone()));
    }
    if rule
        .supports_display
        .unwrap_or_else(|| default_supports_display(spec, facts))
    {
        config.insert("supportsDisplay".into(), json!(true));
    }
    if rule.prefix_binding == Some(true) {
        config.insert("prefixBinding".into(), json!(true));
    }
    let requires_effort = rule.requires_effort.unwrap_or_else(|| {
        implies_mandatory_reasoning(facts, &spec.id)
            || is_qwen_template_reasoning_effort_compat(compat)
    });
    if requires_effort {
        config.insert("requiresEffort".into(), json!(true));
    }
    if rule.suppress_when_off == Some(true) {
        config.insert("suppressWhenOff".into(), json!(true));
    }
    Ok(Some(Value::Object(config)))
}

fn fill_explicit_thinking(
    spec: &ModelSpec,
    facts: &Facts<'_>,
    compat: Option<&Rec>,
    thinking: &Value,
    rule: &RuleThinking,
) -> Value {
    let obj = thinking.as_object().cloned().unwrap_or_default();
    let efforts: Vec<&str> = obj
        .get("efforts")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    let effort_map = if obj.get("effortMap").is_none() {
        merge_effort_map(spec, rule.effort_map.as_ref(), compat, &efforts)
    } else {
        None
    };
    let needs_display = obj.get("supportsDisplay").is_none()
        && rule
            .supports_display
            .unwrap_or_else(|| default_supports_display(spec, facts));
    let needs_requires_effort = obj.get("requiresEffort").is_none()
        && rule.requires_effort.unwrap_or_else(|| {
            implies_mandatory_reasoning(facts, &spec.id)
                || is_qwen_template_reasoning_effort_compat(compat)
        });
    let needs_default_level = obj.get("defaultLevel").is_none() && rule.default_level.is_some();
    let needs_prefix_binding =
        obj.get("prefixBinding").is_none() && rule.prefix_binding == Some(true);
    let needs_effort_budgets = obj.get("effortBudgets").is_none() && rule.effort_budgets.is_some();
    if effort_map.is_none()
        && !needs_display
        && !needs_requires_effort
        && !needs_default_level
        && !needs_prefix_binding
        && !needs_effort_budgets
    {
        return thinking.clone();
    }
    let mut filled = obj;
    if let Some(m) = effort_map {
        filled.insert("effortMap".into(), Value::Object(m));
    }
    if needs_display {
        filled.insert("supportsDisplay".into(), json!(true));
    }
    if let (true, Some(d)) = (needs_default_level, rule.default_level) {
        filled.insert("defaultLevel".into(), json!(d));
    }
    if needs_requires_effort {
        filled.insert("requiresEffort".into(), json!(true));
    }
    if needs_prefix_binding {
        filled.insert("prefixBinding".into(), json!(true));
    }
    if let (true, Some(b)) = (needs_effort_budgets, &rule.effort_budgets) {
        filled.insert("effortBudgets".into(), Value::Object(b.clone()));
    }
    Value::Object(filled)
}

// ---------------------------------------------------------------------------
// Entry
// ---------------------------------------------------------------------------

fn build_resolve_target(spec: &ModelSpec, identity: &Identity, provider_type: &str) -> Target {
    Target {
        provider: provider_type.to_string(),
        api: spec.api.clone(),
        upstream: None,
        class: identity.class.clone(),
        family: identity.family.clone(),
        revision: identity.revision.clone(),
        model: spec.id.clone(),
        reasoning: spec.reasoning,
    }
}

/// Catalog-data axis assignments for one model spec.
pub fn resolve_catalog_axes(spec: &ModelSpec) -> Result<Map<String, Value>, ResolveError> {
    let identity = classify_model(rules(), &spec.provider, &spec.id, spec.lenient)?;
    let provider_type = spec
        .provider_type
        .clone()
        .unwrap_or_else(|| spec.provider.clone());
    Ok(resolve_cascade(
        rules(),
        &build_resolve_target(spec, &identity, &provider_type),
    )?
    .catalog)
}

/// Resolve the full policy surface for one model spec: structured identity, complete compat
/// record, thinking metadata, catalog-data corrections and request policy.
pub fn resolve_model_policy(
    spec: &ModelSpec,
    upstream: Option<&str>,
) -> Result<ResolvedModelPolicy, ResolveError> {
    let identity = classify_model(rules(), &spec.provider, &spec.id, spec.lenient)?;
    let facts = Facts::new(&identity);
    let provider_type = spec
        .provider_type
        .clone()
        .unwrap_or_else(|| spec.provider.clone());
    let mut target = build_resolve_target(spec, &identity, &provider_type);
    target.upstream = upstream.map(str::to_string);
    let axes = resolve_cascade(rules(), &target)?;
    let compat: Option<Rec> = match spec.api.as_str() {
        "openrouter" => {
            let mut chat = resolve_openai_completions_policy(spec, &facts, &axes);
            let responses = resolve_openai_responses_policy(spec, &facts, &axes, "openrouter");
            for key in RESPONSES_ONLY_KEYS {
                chat.opt(key, responses.get(key).cloned());
            }
            Some(chat)
        }
        "openai-completions" => Some(resolve_openai_completions_policy(spec, &facts, &axes)),
        "openai-responses" | "azure-openai-responses" | "openai-codex-responses" => Some(
            resolve_openai_responses_policy(spec, &facts, &axes, &spec.api),
        ),
        "anthropic-messages" => Some(resolve_anthropic_policy(spec, &facts, &axes)),
        "bedrock-converse-stream" => Some(resolve_bedrock_policy(spec, &axes)),
        "devin-agent" => Some(resolve_devin_policy(spec, &axes)),
        "google-generative-ai" | "google-vertex" | "google-gemini-cli" => {
            Some(resolve_google_policy(spec, &axes))
        }
        _ => None,
    };
    let thinking = resolve_thinking_policy(spec, &facts, &axes, compat.as_ref())?;
    let mut request = Map::new();
    for key in &super::axes().request {
        if let Some(v) = axes.wire.get(key) {
            request.insert(key.clone(), v.clone());
        }
    }
    Ok(ResolvedModelPolicy {
        identity,
        compat: compat.map(|c| c.map),
        thinking,
        catalog: axes.catalog,
        request,
    })
}
