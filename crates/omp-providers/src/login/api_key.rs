//! The TS `login "api-key"` engine (`registry/engine/api-key.ts` + `api-key-validation.ts`):
//! optional key-page notice, paste prompt, optional probe against the provider, then the key.
//! The per-provider rules live in `api_key_providers.json`, generated from the TS
//! `rules/auth/*.kdl` by `gen_api_key_providers.py`.

use std::sync::LazyLock;

use pa_ai::oauth::{
    OAuthLoginUi, OAuthPrompt, ProviderHttp, ProviderHttpMethod, ProviderHttpRequest,
};
use serde::Deserialize;
use serde_json::{json, Value};

pub(crate) const VALIDATION_TIMEOUT_MS: u64 = 15_000;
pub(crate) const LOGIN_CANCELLED: &str = "Login cancelled";

#[derive(Debug, Clone, Deserialize)]
pub struct Validate {
    pub kind: String,
    pub label: Option<String>,
    pub base_url: Option<String>,
    pub url: Option<String>,
    pub model: Option<String>,
    pub max_tokens_field: Option<String>,
    pub max_tokens: Option<u64>,
    pub base_url_env: Option<String>,
    pub headers_hook: Option<String>,
    #[serde(default)]
    pub optional: bool,
    #[serde(default)]
    pub trust_forbidden: bool,
    #[serde(default)]
    pub tolerate_model_denied: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ApiKeyProvider {
    pub id: String,
    pub name: String,
    pub auth_url: Option<String>,
    pub instructions: Option<String>,
    pub prompt: String,
    pub placeholder: Option<String>,
    pub normalize: Option<String>,
    pub empty_fallback: Option<String>,
    pub validate: Option<Validate>,
}

static TABLE: LazyLock<Vec<ApiKeyProvider>> = LazyLock::new(|| {
    serde_json::from_str(include_str!("api_key_providers.json"))
        .expect("api_key_providers.json parses")
});

/// The api-key rule of `id`, if the provider logs in by pasted key.
#[must_use]
pub fn provider(id: &str) -> Option<&'static ApiKeyProvider> {
    TABLE.iter().find(|p| p.id == id)
}

/// Every api-key provider id.
pub fn ids() -> impl Iterator<Item = &'static str> {
    TABLE.iter().map(|p| p.id.as_str())
}

/// TS `rule.normalize "strip-bearer"`: `/^bearer\b\s*/i`.
fn strip_bearer(value: &str) -> &str {
    let Some(head) = value.get(..6) else {
        return value;
    };
    if !head.eq_ignore_ascii_case("bearer") {
        return value;
    }
    let rest = &value[6..];
    match rest.chars().next() {
        Some(c) if c.is_alphanumeric() || c == '_' => value,
        _ => rest.trim_start(),
    }
}

/// Run one api-key login; `Ok("")` means "no key entered" (empty fallback `""`).
///
/// `env` resolves environment variables (`COREWEAVE_PROJECT`, `*_BASE_URL`).
///
/// # Errors
/// Cancelled prompt, empty key, or a probe the provider rejected.
pub async fn login(
    rule: &ApiKeyProvider,
    ui: &dyn OAuthLoginUi,
    http: &dyn ProviderHttp,
    env: &(dyn Fn(&str) -> Option<String> + Sync),
) -> Result<String, String> {
    if let (Some(url), Some(instructions)) = (&rule.auth_url, &rule.instructions) {
        ui.on_auth(url, Some(instructions));
    }
    let answer = ui
        .on_prompt(&OAuthPrompt {
            message: rule.prompt.clone(),
            placeholder: rule.placeholder.clone(),
            allow_empty: rule.empty_fallback.is_some(),
        })
        .await
        .ok_or_else(|| LOGIN_CANCELLED.to_string())?;
    if ui.is_cancelled() {
        return Err(LOGIN_CANCELLED.to_string());
    }
    let mut key = answer.trim().to_string();
    if rule.normalize.as_deref() == Some("strip-bearer") && !key.is_empty() {
        key = strip_bearer(&key).to_string();
        if key.is_empty() {
            return Err(format!(
                "{} API key is empty after stripping Bearer prefix",
                rule.name
            ));
        }
    }
    if key.is_empty() {
        return rule
            .empty_fallback
            .clone()
            .ok_or_else(|| "API key is required".to_string());
    }
    if let Some(validate) = &rule.validate {
        ui.on_progress(if validate.optional {
            "Validating API key (optional)..."
        } else {
            "Validating API key..."
        });
        if let Err(error) = run_validation(validate, &rule.name, &key, http, env).await {
            let trusts_forbidden = validate.trust_forbidden && error.status == Some(403);
            let auth_failed = matches!(error.status, Some(401 | 403));
            if !validate.optional || (auth_failed && !trusts_forbidden) {
                return Err(error.message);
            }
            ui.on_progress(&format!(
                "Skipping {} validation endpoint; continuing with provided API key.",
                rule.name
            ));
        }
    }
    Ok(key)
}

pub(crate) struct ValidationError {
    status: Option<u16>,
    pub(crate) message: String,
}

fn headers_for_hook(
    hook: Option<&str>,
    env: &(dyn Fn(&str) -> Option<String> + Sync),
    provider: &str,
) -> Result<Vec<(String, String)>, ValidationError> {
    let fail = |message: String| ValidationError {
        status: None,
        message,
    };
    match hook {
        None => Ok(Vec::new()),
        Some("cline-pass-client") => Ok([
            ("HTTP-Referer", "https://cline.bot"),
            ("X-Title", "Cline"),
            ("X-IS-MULTIROOT", "false"),
            ("X-CLIENT-TYPE", "cline-sdk"),
            ("User-Agent", "Cline/3.0.58"),
            ("X-CLIENT-VERSION", "3.0.58"),
            ("X-PLATFORM", "linux"),
            ("X-PLATFORM-VERSION", "3.0.54"),
            ("X-CORE-VERSION", "0.0.79"),
        ]
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .to_vec()),
        Some("coreweave-project") => {
            let clean = |k: &str| {
                env(k)
                    .map(|v| v.trim().to_string())
                    .filter(|v| !v.is_empty())
            };
            let project = clean("COREWEAVE_PROJECT")
                .or_else(|| clean("WANDB_INFERENCE_PROJECT"))
                .or_else(|| {
                    let wandb = clean("WANDB_PROJECT")?;
                    if wandb.contains('/') {
                        return Some(wandb);
                    }
                    clean("WANDB_ENTITY").map(|entity| format!("{entity}/{wandb}"))
                });
            match project {
                Some(project) => Ok(vec![("OpenAI-Project".to_string(), project)]),
                None => Err(fail(format!(
                    "CoreWeave Serverless Inference requires OpenAI-Project. Set COREWEAVE_PROJECT=<team>/<project> before running /login {provider}. To persist it, add export COREWEAVE_PROJECT=<team>/<project> to your shell startup file (for example ~/.zshrc, ~/.bashrc, or your shell's profile/rc file)."
                ))),
            }
        }
        Some(other) => Err(fail(format!("unknown headers hook {other}"))),
    }
}

pub(crate) async fn run_validation(
    validate: &Validate,
    login_label: &str,
    key: &str,
    http: &dyn ProviderHttp,
    env: &(dyn Fn(&str) -> Option<String> + Sync),
) -> Result<(), ValidationError> {
    let label = validate.label.as_deref().unwrap_or(login_label);
    let fail = |message: String| ValidationError {
        status: None,
        message,
    };
    let (method, url, headers, body) = match validate.kind.as_str() {
        "chat-completions" => {
            let base = validate
                .base_url
                .as_deref()
                .ok_or_else(|| fail("missing base-url".into()))?;
            let field = validate.max_tokens_field.as_deref().unwrap_or("max_tokens");
            // TS key order: model, messages, <max tokens field>, temperature.
            let ordered = json!({
                "model": validate.model,
                "messages": [{"role": "user", "content": "ping"}],
                field: validate.max_tokens.unwrap_or(1),
                "temperature": 0,
            });
            (
                ProviderHttpMethod::Post,
                format!("{base}/chat/completions"),
                vec![
                    ("Content-Type".to_string(), "application/json".to_string()),
                    ("Authorization".to_string(), format!("Bearer {key}")),
                ],
                Some(ordered.to_string()),
            )
        }
        "anthropic-messages" => {
            let base = validate
                .base_url
                .as_deref()
                .ok_or_else(|| fail("missing base-url".into()))?;
            let trimmed = base.trim().trim_end_matches('/');
            let base = trimmed.strip_suffix("/v1").unwrap_or(trimmed);
            (
                ProviderHttpMethod::Post,
                format!("{base}/v1/messages"),
                vec![
                    ("Content-Type".to_string(), "application/json".to_string()),
                    ("anthropic-version".to_string(), "2023-06-01".to_string()),
                    ("x-api-key".to_string(), key.to_string()),
                ],
                Some(json!({"model": validate.model, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 1}).to_string()),
            )
        }
        "models-endpoint" => {
            let env_base = validate
                .base_url_env
                .as_deref()
                .and_then(env)
                .map(|v| v.trim().to_string())
                .filter(|v| !v.is_empty());
            let url = match env_base {
                Some(base) => format!("{}/models", base.trim_end_matches('/')),
                None => validate
                    .url
                    .clone()
                    .ok_or_else(|| fail("missing url".into()))?,
            };
            let mut headers = headers_for_hook(validate.headers_hook.as_deref(), env, login_label)?;
            headers.push(("Authorization".to_string(), format!("Bearer {key}")));
            (ProviderHttpMethod::Get, url, headers, None)
        }
        other => return Err(fail(format!("unknown validation kind {other}"))),
    };
    let response = http
        .request(
            ProviderHttpRequest {
                method,
                url,
                headers,
                body,
                follow_redirects: true,
            },
            VALIDATION_TIMEOUT_MS,
        )
        .await
        .map_err(|message| fail(message))?;
    if response.ok() {
        return Ok(());
    }
    let details = response.body.trim();
    if validate.tolerate_model_denied && response.status == 401 {
        let code = serde_json::from_str::<Value>(details).ok().and_then(|v| {
            v.pointer("/error/code")
                .or_else(|| v.get("code"))
                .and_then(Value::as_str)
                .map(str::to_string)
        });
        if code.as_deref() == Some("invalid_model") {
            return Ok(());
        }
    }
    let message = if details.is_empty() {
        format!("{label} API key validation failed ({})", response.status)
    } else {
        format!(
            "{label} API key validation failed ({}): {details}",
            response.status
        )
    };
    Err(ValidationError {
        status: Some(response.status),
        message,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_bearer_matches_the_ts_regex() {
        assert_eq!(strip_bearer("Bearer sk-1"), "sk-1");
        assert_eq!(strip_bearer("bearer   sk-1"), "sk-1");
        assert_eq!(strip_bearer("bearerxyz"), "bearerxyz");
        assert_eq!(strip_bearer("sk-1"), "sk-1");
    }

    #[test]
    fn table_covers_the_ts_api_key_roster() {
        assert_eq!(TABLE.len(), 59);
        let exa = provider("exa").unwrap();
        assert!(exa.validate.is_none());
        assert_eq!(
            provider("llama.cpp").unwrap().empty_fallback.as_deref(),
            Some("llama-cpp-local")
        );
        assert_eq!(
            provider("ollama").unwrap().empty_fallback.as_deref(),
            Some("")
        );
    }

    #[test]
    fn the_ported_key_providers_build_the_ts_probe_requests() {
        // zai-coding-plan probes the coding-plan base with the coding model…
        let zai = provider("zai-coding-plan").unwrap();
        assert_eq!(
            zai.auth_url.as_deref(),
            Some("https://z.ai/manage-apikey/apikey-list")
        );
        assert_eq!(
            zai.validate.as_ref().unwrap().base_url.as_deref(),
            Some("https://api.z.ai/api/coding/paas/v4")
        );
        assert_eq!(
            zai.validate.as_ref().unwrap().model.as_deref(),
            Some("glm-5.2")
        );
        // …and stores under `zai` like TS `store-as "zai"`.
        assert_eq!(super::super::providers::store_as("zai-coding-plan"), "zai");
        // openrouter keys authenticate against the key endpoint, not /models.
        let openrouter = provider("openrouter").unwrap();
        assert_eq!(
            openrouter.validate.as_ref().unwrap().url.as_deref(),
            Some("https://openrouter.ai/api/v1/auth/key")
        );
        assert_eq!(
            openrouter.validate.as_ref().unwrap().kind,
            "models-endpoint"
        );
    }
}
