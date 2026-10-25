//! The `login "custom"` flows that are key prompts with an endpoint choice,
//! ported from TS `registry/oauth/alibaba-coding-plan.ts` / `alibaba-token-plan.ts`.
//! Both end in a plain API key (the Token Plan optionally wraps its cookie and
//! region base URL into the TS `serializeAlibabaTokenPlanCredential` payload),
//! which the caller stores like any api-key login.

use pa_ai::oauth::{OAuthLoginUi, OAuthPrompt, ProviderHttp};
use serde_json::{json, Value};

use super::api_key::{run_validation, Validate, LOGIN_CANCELLED};

/// Alibaba Coding Plan (TS `DEFAULT_AUTH_URL` / `CHINA_AUTH_URL`).
const CODING_PLAN_INTL_AUTH_URL: &str = "https://modelstudio.console.alibabacloud.com/";
const CODING_PLAN_CN_AUTH_URL: &str = "https://bailian.console.aliyun.com/?tab=model#/api-key";
const CODING_PLAN_INTL_BASE_URL: &str = "https://coding-intl.dashscope.aliyuncs.com/v1";
const CODING_PLAN_CN_BASE_URL: &str = "https://coding.dashscope.aliyuncs.com/v1";
/// TS `VALIDATION_MODEL`.
const CODING_PLAN_VALIDATION_MODEL: &str = "qwen3.5-plus";

/// QwenCloud Token Plan (TS `INTERNATIONAL_AUTH_URL` / `CHINA_AUTH_URL` and the
/// catalog `wire/alibaba-token-plan` endpoints).
const TOKEN_PLAN_INTL_AUTH_URL: &str =
    "https://home.qwencloud.com/billing/subscription/token-plan-individual";
const TOKEN_PLAN_CN_AUTH_URL: &str = "https://www.aliyun.com/benefit/scene/tokenplan";
pub(crate) const TOKEN_PLAN_BASE_URL: &str =
    "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1";
pub(crate) const TOKEN_PLAN_CN_BASE_URL: &str =
    "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1";

async fn prompt(
    ui: &dyn OAuthLoginUi,
    message: &str,
    placeholder: Option<&str>,
    allow_empty: bool,
) -> Result<String, String> {
    let answer = ui
        .on_prompt(&OAuthPrompt {
            message: message.to_string(),
            placeholder: placeholder.map(str::to_string),
            allow_empty,
        })
        .await
        .ok_or_else(|| LOGIN_CANCELLED.to_string())?;
    if ui.is_cancelled() {
        return Err(LOGIN_CANCELLED.to_string());
    }
    Ok(answer.trim().to_string())
}

/// One `{1|2|3}` endpoint choice; `Ok(None)` = the default (any non-2/3 answer).
async fn endpoint_choice(ui: &dyn OAuthLoginUi, message: &str) -> Result<Option<char>, String> {
    let choice = prompt(ui, message, Some("1"), false).await?;
    match choice.as_str() {
        "2" => Ok(Some('2')),
        "3" => Ok(Some('3')),
        _ => Ok(None),
    }
}

async fn custom_base_url(
    ui: &dyn OAuthLoginUi,
    message: &str,
    placeholder: &str,
) -> Result<String, String> {
    let raw = prompt(ui, message, Some(placeholder), false).await?;
    let trimmed = raw.trim_end_matches('/').to_string();
    if trimmed.is_empty() {
        return Err("Custom URL is required for option 3".to_string());
    }
    Ok(trimmed)
}

fn validate_build(kind: &str, label: &str, base_url: &str, model: Option<&str>) -> Validate {
    serde_json::from_value(json!({
        "kind": kind,
        "label": label,
        "base_url": base_url,
        "model": model,
        "url": if kind == "models-endpoint" { Value::String(format!("{base_url}/models")) } else { Value::Null },
    }))
    .expect("inline validate descriptor parses")
}

async fn validate(
    ui: &dyn OAuthLoginUi,
    http: &dyn ProviderHttp,
    env: &(dyn Fn(&str) -> Option<String> + Sync),
    validate: &Validate,
    label: &str,
    key: &str,
) -> Result<(), String> {
    ui.on_progress("Validating API key...");
    match run_validation(validate, label, key, http, env).await {
        Ok(()) => Ok(()),
        Err(error) => Err(error.message),
    }
}

/// TS `loginAlibabaCodingPlan`: endpoint choice, key paste, chat-completions
/// probe (models-endpoint probe for a custom base URL).
pub(crate) async fn alibaba_coding_plan(
    ui: &dyn OAuthLoginUi,
    http: &dyn ProviderHttp,
    env: &(dyn Fn(&str) -> Option<String> + Sync),
) -> Result<String, String> {
    let choice = endpoint_choice(
        ui,
        "Select Alibaba Coding Plan endpoint: 1=International (default), 2=China, 3=Custom — enter 1, 2, or 3",
    )
    .await?;
    let (base_url, auth_url, instructions): (String, &str, &str) = match choice {
        Some('2') => (
            CODING_PLAN_CN_BASE_URL.to_string(),
            CODING_PLAN_CN_AUTH_URL,
            "Copy your API key from the Alibaba Cloud Bailian console (China mainland)",
        ),
        Some('3') => {
            let base =
                custom_base_url(ui, "Enter custom base URL", "https://your-proxy.com/v1").await?;
            (
                base,
                CODING_PLAN_INTL_AUTH_URL,
                "Copy your API key from the Alibaba Cloud DashScope console",
            )
        }
        _ => (
            CODING_PLAN_INTL_BASE_URL.to_string(),
            CODING_PLAN_INTL_AUTH_URL,
            "Copy your API key from the Alibaba Cloud DashScope console (International)",
        ),
    };
    ui.on_auth(auth_url, Some(instructions));
    let key = prompt(
        ui,
        "Paste your Alibaba Coding Plan API key",
        Some("sk-..."),
        false,
    )
    .await?;
    if key.is_empty() {
        return Err("API key is required".to_string());
    }
    let kind = if choice == Some('3') {
        "models-endpoint"
    } else {
        "chat-completions"
    };
    let descriptor = validate_build(
        kind,
        "Alibaba Coding Plan",
        &base_url,
        Some(CODING_PLAN_VALIDATION_MODEL),
    );
    validate(ui, http, env, &descriptor, "Alibaba Coding Plan", &key).await?;
    Ok(key)
}

/// TS `serializeAlibabaTokenPlanCredential`: a bare token until a quota cookie
/// or a diverging region base URL joins it.
fn serialize_token_plan(token: &str, cookie: &str, base_url: Option<&str>) -> String {
    if cookie.is_empty() && base_url.is_none() {
        return token.to_string();
    }
    let mut payload = json!({ "token": token });
    let object = payload.as_object_mut().expect("object literal");
    if !cookie.is_empty() {
        object.insert("cookie".into(), Value::String(cookie.to_string()));
    }
    if let Some(base_url) = base_url {
        object.insert("baseUrl".into(), Value::String(base_url.to_string()));
    }
    payload.to_string()
}

/// TS `loginAlibabaTokenPlan`: region choice, key paste, models-endpoint probe,
/// optional quota-reporting cookie.
pub(crate) async fn alibaba_token_plan(
    ui: &dyn OAuthLoginUi,
    http: &dyn ProviderHttp,
    env: &(dyn Fn(&str) -> Option<String> + Sync),
) -> Result<String, String> {
    let choice = endpoint_choice(
        ui,
        "Select QwenCloud Token Plan region: 1=International (default), 2=China (Beijing), 3=Custom — enter 1, 2, or 3",
    )
    .await?;
    let (base_url, auth_url, instructions): (String, &str, &str) = match choice {
        Some('2') => (
            TOKEN_PLAN_CN_BASE_URL.to_string(),
            TOKEN_PLAN_CN_AUTH_URL,
            "Subscribe to the China (Beijing) 百炼 Token Plan and copy its dedicated API key",
        ),
        Some('3') => {
            let base = custom_base_url(
                ui,
                "Enter custom Token Plan base URL",
                "https://token-plan.<region>.maas.aliyuncs.com/compatible-mode/v1",
            )
            .await?;
            (
                base,
                TOKEN_PLAN_INTL_AUTH_URL,
                "Copy your Token Plan API key for the custom endpoint",
            )
        }
        _ => (
            TOKEN_PLAN_BASE_URL.to_string(),
            TOKEN_PLAN_INTL_AUTH_URL,
            "Subscribe to Token Plan Individual and copy its dedicated API key. Keep this page open; the next prompt explains how to enable optional quota reporting.",
        ),
    };
    ui.on_auth(auth_url, Some(instructions));
    let key = prompt(
        ui,
        "Paste your QwenCloud Token Plan API key",
        Some("sk-sp-..."),
        false,
    )
    .await?;
    if key.is_empty() {
        return Err("API key is required".to_string());
    }
    let descriptor = validate_build("models-endpoint", "QwenCloud Token Plan", &base_url, None);
    validate(ui, http, env, &descriptor, "QwenCloud Token Plan", &key).await?;

    let cookie_host = if base_url == TOKEN_PLAN_CN_BASE_URL {
        "bailian-cs.console.aliyun.com"
    } else {
        "cs-data.qwencloud.com"
    };
    let raw = prompt(
        ui,
        &format!(
            "Optional quota reporting: open browser DevTools → Network, reload the Token Plan page, \
             filter for api.json, and select the {cookie_host}/data/api.json request whose api query \
             ends in /tokenplan/personal/api/v2/usage. Copy Request Headers → Cookie, then paste the \
             complete name=value; ... value here, or press Enter to skip."
        ),
        Some("name=value; name=value; ..."),
        true,
    )
    .await?;
    let stripped = raw
        .trim()
        .strip_prefix("Cookie:")
        .or_else(|| raw.trim().strip_prefix("cookie:"))
        .map_or(raw.trim(), str::trim);
    let cookie = stripped.to_string();
    let cookie_ok = cookie.is_empty()
        || cookie.split(';').any(|segment| match segment.find('=') {
            Some(at) if at > 0 => {
                !segment[..at].trim().is_empty() && !segment[at + 1..].trim().is_empty()
            }
            _ => false,
        });
    if !cookie_ok {
        return Err(format!(
            "Invalid QwenCloud Cookie header. Copy the complete Cookie request header from the \
             {cookie_host} usage request, not a single cookie value."
        ));
    }
    let region = (base_url != TOKEN_PLAN_BASE_URL).then_some(base_url.as_str());
    Ok(serialize_token_plan(&key, &cookie, region))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_plan_serialization_matches_the_ts_helper() {
        assert_eq!(serialize_token_plan("sk-sp-1", "", None), "sk-sp-1");
        assert_eq!(
            serialize_token_plan("sk-sp-1", "a=1; b=2", None),
            r#"{"token":"sk-sp-1","cookie":"a=1; b=2"}"#
        );
        assert_eq!(
            serialize_token_plan("sk-sp-1", "", Some(TOKEN_PLAN_CN_BASE_URL)),
            format!(r#"{{"token":"sk-sp-1","baseUrl":"{TOKEN_PLAN_CN_BASE_URL}"}}"#)
        );
    }
}
