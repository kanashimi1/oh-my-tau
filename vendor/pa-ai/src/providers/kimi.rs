//! Kimi Code transport selection (TS `providers/kimi.ts` + `openai-anthropic-shim.ts`).
//!
//! Kimi serves both an OpenAI-compatible (`/coding/v1/chat/completions`) and an
//! Anthropic-compatible (`/coding/v1/messages`) surface. Each model's resolved compat names the
//! protocol (`kimiApiFormat`); the Anthropic format streams through the Anthropic Messages
//! provider against a model rebuilt for that API.

use serde_json::{json, Value};

use crate::compat::ModelView;
use crate::providers::anthropic::{map_tool_choice, AnthropicOptions};
use crate::providers::openai_completions::{OpenAICompletionsOptions, ToolChoice};
use crate::types::{
    Model, ModelThinkingLevel, SimpleStreamOptions, SimpleToolChoice, StreamOptions,
};

pub const PROVIDER: &str = "kimi-code";

/// The wire protocol a Kimi model streams through.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KimiFormat {
    OpenAi,
    Anthropic,
}

/// TS `streamKimi`'s format choice: `model.compat.kimiApiFormat ?? options.format`.
///
/// # Errors
/// The model carries no resolved API format (TS throws the same).
pub fn resolve_format(model: &Model, view: &ModelView) -> Result<KimiFormat, String> {
    match view.compat_str("kimiApiFormat") {
        Some("anthropic") => Ok(KimiFormat::Anthropic),
        Some("openai") => Ok(KimiFormat::OpenAi),
        _ => Err(format!(
            "Kimi Code model {} has no resolved API format",
            model.id
        )),
    }
}

fn sanitize_header_value(value: &str, fallback: &str) -> String {
    let cleaned: String = value.chars().filter(|c| (' '..='~').contains(c)).collect();
    let cleaned = cleaned.trim();
    if cleaned.is_empty() {
        fallback.to_string()
    } else {
        cleaned.to_string()
    }
}

fn proc_string(name: &str) -> String {
    std::fs::read_to_string(format!("/proc/sys/kernel/{name}"))
        .map(|s| s.trim().to_string())
        .unwrap_or_default()
}

/// TS `getKimiCommonHeaders`: the device identity every Kimi request carries.
#[must_use]
pub fn common_headers() -> Vec<(String, String)> {
    const VERSION: &str = "18.9.1";
    let release = proc_string("osrelease");
    let arch = match std::env::consts::ARCH {
        "x86_64" => "x64",
        "aarch64" => "arm64",
        other => other,
    };
    let device_model = ["Linux", release.as_str(), arch]
        .iter()
        .filter(|s| !s.is_empty())
        .copied()
        .collect::<Vec<_>>()
        .join(" ");
    let agent_dir = std::env::var_os("OMP_AGENT_DIR").map_or_else(
        || {
            std::path::PathBuf::from(std::env::var_os("HOME").unwrap_or_default())
                .join(".omp")
                .join("agent")
        },
        std::path::PathBuf::from,
    );
    let device_id = std::fs::read_to_string(agent_dir.join("kimi-device-id"))
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    vec![
        ("User-Agent".into(), format!("KimiCLI/{VERSION}")),
        ("X-Msh-Platform".into(), "kimi_cli".into()),
        ("X-Msh-Version".into(), VERSION.into()),
        (
            "X-Msh-Device-Name".into(),
            sanitize_header_value(&proc_string("hostname"), "unknown"),
        ),
        (
            "X-Msh-Device-Model".into(),
            sanitize_header_value(&device_model, "unknown"),
        ),
        (
            "X-Msh-Os-Version".into(),
            sanitize_header_value(&proc_string("version"), "unknown"),
        ),
        (
            "X-Msh-Device-Id".into(),
            sanitize_header_value(&device_id, "unknown"),
        ),
    ]
}

/// The Anthropic-format request a Kimi model sends: the shim's rebuilt model, its policy view and
/// the provider-native options mapped from the caller's simple options.
pub struct AnthropicShim {
    pub model: Model,
    pub view: ModelView,
    pub options: AnthropicOptions,
}

/// TS `streamOpenAIAnthropicShim` (anthropic branch, Kimi configuration).
///
/// # Errors
/// The thinking budget leaves no room for the response.
pub fn anthropic_shim(
    model: &Model,
    view: &ModelView,
    options: &SimpleStreamOptions,
) -> Result<AnthropicShim, String> {
    let mut merged: std::collections::BTreeMap<String, String> =
        common_headers().into_iter().collect();
    for (k, v) in options.base.headers.iter().flatten() {
        merged.insert(k.clone(), v.clone());
    }
    let anthropic_thinking_mode =
        (view.compat_str("thinkingFormat") == Some("kimi")).then_some("anthropic-adaptive");
    let explicit_thinking: Option<Value> = match (anthropic_thinking_mode, &view.thinking) {
        (Some(mode), Some(Value::Object(t))) => {
            let mut t = t.clone();
            t.insert("mode".into(), json!(mode));
            Some(Value::Object(t))
        }
        _ => None,
    };
    let mut shim = model.clone();
    shim.api = "anthropic-messages".to_string();
    shim.base_url = strip_v1(&model.base_url);
    shim.headers = Some(merged.clone());
    shim.compat = None;
    shim.request_model_id = None;
    shim.thinking_level_map = None;
    shim.omit_max_output_tokens = None;
    let shim_view = ModelView::with_thinking(&shim, explicit_thinking);

    // The shim model re-resolves its ladder (`anthropic-adaptive` mode over the Kimi host), which can
    // be narrower than the ladder a stale cached row advertised to the caller; clamp the request to
    // it (`clampThinkingLevelForModel`) instead of failing the turn.
    let reasoning = options
        .reasoning
        .filter(|l| *l != ModelThinkingLevel::Off)
        .map(|l| clamp_to_ladder(l, &shim_view.efforts()));
    let thinking_enabled = reasoning.is_some() && model.reasoning && !options.disable_reasoning;
    let model_max = (model.max_tokens > 0).then_some(model.max_tokens);
    let mut base: StreamOptions = options.base.clone();
    base.max_tokens = options.base.max_tokens.or(model_max);
    base.headers = Some(merged.into_iter().collect());
    let mut native = AnthropicOptions::from_base(base);
    native.thinking_enabled = Some(thinking_enabled);
    native.thinking_budget_tokens =
        reasoning.map(|level| crate::providers::anthropic::thinking_budget_for(options, level));
    if anthropic_thinking_mode.is_some() {
        native.reasoning = reasoning;
    }
    native.tool_choice = map_tool_choice(options.tool_choice.as_ref());
    Ok(AnthropicShim {
        model: shim,
        view: shim_view,
        options: native,
    })
}

/// `baseUrl.replace(/\/v1\/?$/, "")`.
fn strip_v1(url: &str) -> String {
    if let Some(stripped) = url.strip_suffix("/v1/") {
        return stripped.to_string();
    }
    url.strip_suffix("/v1").unwrap_or(url).to_string()
}

/// TS `streamOpenAIAnthropicShim` (openai branch, Kimi configuration): the chat-completions
/// options. The shim forwards the caller's raw reasoning/tool-choice (no `mapOptionsForApi`).
#[must_use]
pub fn openai_options(model: &Model, options: &SimpleStreamOptions) -> OpenAICompletionsOptions {
    let mut merged: std::collections::BTreeMap<String, String> =
        common_headers().into_iter().collect();
    for (k, v) in options.base.headers.iter().flatten() {
        merged.insert(k.clone(), v.clone());
    }
    let mut base: StreamOptions = options.base.clone();
    base.max_tokens = options
        .base
        .max_tokens
        .or((model.max_tokens > 0).then_some(model.max_tokens));
    base.headers = Some(merged.into_iter().collect());
    let mut native = OpenAICompletionsOptions::from_base(base);
    native.max_tokens_explicit = options.base.max_tokens.is_some();
    native.reasoning_effort = options.reasoning.filter(|l| *l != ModelThinkingLevel::Off);
    if options.disable_reasoning {
        native.reasoning_enabled = Some(false);
    }
    native.tool_choice = match options.tool_choice.as_ref() {
        Some(SimpleToolChoice::Auto) => Some(ToolChoice::Auto),
        Some(SimpleToolChoice::None) => Some(ToolChoice::None),
        Some(SimpleToolChoice::Any | SimpleToolChoice::Required) => Some(ToolChoice::Required),
        Some(SimpleToolChoice::Function { name }) if !name.is_empty() => {
            Some(ToolChoice::Function { name: name.clone() })
        }
        _ => None,
    };
    native
}

/// `clampThinkingLevelForModel`: the highest supported effort not above `level`, else the lowest.
fn clamp_to_ladder(level: ModelThinkingLevel, efforts: &[&str]) -> ModelThinkingLevel {
    const ORDER: [ModelThinkingLevel; 6] = [
        ModelThinkingLevel::Minimal,
        ModelThinkingLevel::Low,
        ModelThinkingLevel::Medium,
        ModelThinkingLevel::High,
        ModelThinkingLevel::Xhigh,
        ModelThinkingLevel::Max,
    ];
    let supported: Vec<ModelThinkingLevel> = ORDER
        .into_iter()
        .filter(|l| efforts.contains(&l.wire_name()))
        .collect();
    if supported.is_empty() || supported.contains(&level) {
        return level;
    }
    let rank = |l: ModelThinkingLevel| ORDER.iter().position(|o| *o == l).unwrap_or(0);
    supported
        .iter()
        .copied()
        .filter(|l| rank(*l) < rank(level))
        .max_by_key(|l| rank(*l))
        .unwrap_or(supported[0])
}
