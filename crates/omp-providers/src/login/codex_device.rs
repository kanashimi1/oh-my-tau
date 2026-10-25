//! The headless ChatGPT (Codex) login (TS `loginOpenAICodexDevice` in
//! `registry/oauth/openai-codex.ts`): request a device code from
//! `auth.openai.com/api/accounts/deviceauth/usercode`, have the user approve
//! `auth.openai.com/codex/device`, then poll the device token endpoint until
//! it returns an authorization code that is exchanged at the regular token
//! endpoint with the device redirect URI.

use serde_json::{json, Value};

use pa_ai::oauth::{OAuthLoginUi, ProviderHttp, ProviderHttpMethod, ProviderHttpRequest};

const CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
const TOKEN_URL: &str = "https://auth.openai.com/oauth/token";
const DEVICE_USERCODE_URL: &str = "https://auth.openai.com/api/accounts/deviceauth/usercode";
const DEVICE_TOKEN_URL: &str = "https://auth.openai.com/api/accounts/deviceauth/token";
const DEVICE_REDIRECT_URI: &str = "https://auth.openai.com/deviceauth/callback";
const DEVICE_AUTH_URL: &str = "https://auth.openai.com/codex/device";
/// TS `DEVICE_POLL_SAFETY_MARGIN_MS`, added to the provider's poll interval.
const POLL_SAFETY_MARGIN_MS: u64 = 3_000;
const MAX_POLLS: u32 = 120;
const REQUEST_TIMEOUT_MS: u64 = 15_000;

pub(crate) struct CodexDeviceCredentials {
    pub access: String,
    pub refresh: String,
    /// Wall-clock epoch milliseconds.
    pub expires: i64,
    /// The ChatGPT workspace id (`chatgpt_account_id` JWT claim, TS
    /// `getTokenProfile`).
    pub account_id: String,
}

fn field(body: &Value, name: &str) -> Option<String> {
    body.get(name)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

async fn post_json(
    http: &dyn ProviderHttp,
    url: &str,
    payload: Value,
) -> Result<pa_ai::oauth::ProviderHttpResponse, String> {
    http.request(
        ProviderHttpRequest {
            method: ProviderHttpMethod::Post,
            url: url.to_string(),
            headers: vec![("Content-Type".to_string(), "application/json".to_string())],
            body: Some(payload.to_string()),
            follow_redirects: false,
        },
        REQUEST_TIMEOUT_MS,
    )
    .await
}

/// TS `loginOpenAICodexDevice`.
///
/// # Errors
/// The login was cancelled, initiation failed, the response was malformed, or
/// polling failed/timed out.
pub(crate) async fn login(
    ui: &dyn OAuthLoginUi,
    http: &dyn ProviderHttp,
    now_ms: i64,
    expires_in_seconds: impl Fn(&Value) -> Option<i64>,
) -> Result<CodexDeviceCredentials, String> {
    if ui.is_cancelled() {
        return Err("Login cancelled".to_string());
    }
    ui.on_progress("Initiating device authorization…");
    let init_response =
        post_json(http, DEVICE_USERCODE_URL, json!({ "client_id": CLIENT_ID })).await?;
    if !init_response.ok() {
        return Err(format!(
            "Device authorization initiation failed: {}",
            init_response.status
        ));
    }
    let init: Value = serde_json::from_str(&init_response.body)
        .map_err(|_| "Device authorization response is not valid JSON".to_string())?;
    let (Some(device_auth_id), Some(user_code)) =
        (field(&init, "device_auth_id"), field(&init, "user_code"))
    else {
        return Err("Device authorization response missing required fields".to_string());
    };
    let interval_ms = match init.get("interval") {
        Some(Value::Number(seconds)) => seconds.as_u64().unwrap_or(5),
        Some(Value::String(seconds)) => seconds.parse().unwrap_or(5),
        _ => 5,
    } * 1000
        + POLL_SAFETY_MARGIN_MS;

    ui.on_auth(DEVICE_AUTH_URL, Some(&format!("Enter code: {user_code}")));
    ui.on_progress(&format!(
        "Waiting for browser authorization (code: {user_code})…"
    ));

    for _poll in 0..MAX_POLLS {
        if ui.is_cancelled() {
            return Err("Device authorization cancelled".to_string());
        }
        let poll_response = post_json(
            http,
            DEVICE_TOKEN_URL,
            json!({
                "device_auth_id": device_auth_id,
                "user_code": user_code,
            }),
        )
        .await?;
        // 403/404 = authorization pending, keep polling.
        if poll_response.status == 403 || poll_response.status == 404 {
            tokio::time::sleep(std::time::Duration::from_millis(interval_ms)).await;
            continue;
        }
        if !poll_response.ok() {
            return Err(format!(
                "Device token polling failed: {}",
                poll_response.status
            ));
        }
        let body: Value = serde_json::from_str(&poll_response.body)
            .map_err(|_| "Device token response is not valid JSON".to_string())?;
        let (Some(authorization_code), Some(code_verifier)) = (
            field(&body, "authorization_code"),
            field(&body, "code_verifier"),
        ) else {
            return Err(
                "Device token response missing authorization_code or code_verifier".to_string(),
            );
        };
        ui.on_progress("Exchanging authorization code for tokens…");
        let exchange = post_json(
            http,
            TOKEN_URL,
            json!({
                "grant_type": "authorization_code",
                "code": authorization_code,
                "code_verifier": code_verifier,
                "client_id": CLIENT_ID,
                "redirect_uri": DEVICE_REDIRECT_URI,
            }),
        )
        .await?;
        if !exchange.ok() {
            return Err(format!("Token exchange failed: {}", exchange.status));
        }
        let token: Value = serde_json::from_str(&exchange.body)
            .map_err(|_| "Token exchange response is not valid JSON".to_string())?;
        let (Some(access), Some(refresh)) = (
            field(&token, "access_token"),
            field(&token, "refresh_token"),
        ) else {
            return Err(
                "Token exchange response missing access_token or refresh_token".to_string(),
            );
        };
        let expires = expires_in_seconds(&token).map_or(now_ms, |seconds| {
            now_ms.saturating_add(seconds.saturating_mul(1000))
        });
        let account_id = super::store::jwt_payload(&access)
            .and_then(|payload| {
                payload
                    .get("https://api.openai.com/auth")
                    .and_then(|claim| claim.get("chatgpt_account_id"))
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_string)
            })
            .unwrap_or_default();
        return Ok(CodexDeviceCredentials {
            access,
            refresh,
            expires,
            account_id,
        });
    }
    Err("Device authorization timed out — user did not complete login in time".to_string())
}
