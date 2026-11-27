//! The Muse Code (Meta) login (TS `login "device-code"` rule
//! `rules/auth/muse-code.kdl` run by `registry/engine/device-code.ts`, plus
//! the `attachMuseCodeApiKey` after-exchange hook in
//! `registry/oauth/muse-code.ts`): an RFC 8628 device grant at
//! `auth.meta.com/oidc/device/*`, then the subscription key mint at
//! `api.meta.ai/muse-code/key`. The stored access is the JSON credential
//! `{oauthAccessToken, apiKey}` and Meta's device response omits expiry, so
//! the row expires never.

use std::time::{Duration, Instant};

use serde_json::{json, Value};

use pa_ai::oauth::{OAuthLoginUi, ProviderHttp, ProviderHttpMethod, ProviderHttpRequest};

const CLIENT_ID: &str = "1031625952748946";
const DEVICE_URL: &str = "https://auth.meta.com/oidc/device/authorization/";
const TOKEN_URL: &str = "https://auth.meta.com/oidc/device/token/";
const MUSE_KEY_URL: &str = "https://api.meta.ai/muse-code/key";
const API_VERSION: &str = "1.0.0";
const AUTH_HEADERS: [(&str, &str); 2] = [
    ("Accept", "application/json"),
    ("x-api-version", API_VERSION),
];
const REQUEST_TIMEOUT_MS: u64 = 20_000;
/// TS `MINIMUM_DEVICE_FLOW_INTERVAL_MS` / default 5s / `slow_down` +5s.
const MIN_POLL_INTERVAL_MS: u64 = 1000;
const DEFAULT_POLL_INTERVAL_MS: u64 = 5000;
const SLOW_DOWN_INCREMENT_MS: u64 = 5000;
/// TS `NEVER_EXPIRES` (max safe Date value).
pub(crate) const NEVER_EXPIRES: i64 = 8_640_000_000_000_000;

fn field(body: &Value, name: &str) -> Option<String> {
    body.get(name)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

/// TS `attachMuseCodeApiKey`: mint (or reuse) the Model API key and pack the
/// stored credential. Returns the final data fields to store.
///
/// # Errors
/// The key exchange failed, the subscription is inactive/required, or the
/// response has no key or stable account identity.
pub async fn attach_api_key(
    http: &dyn ProviderHttp,
    ui: &dyn OAuthLoginUi,
    oauth_access_token: &str,
) -> Result<(String, String, String), String> {
    ui.on_progress("Exchanging Meta account for the Muse Code key...");
    let request = ProviderHttpRequest {
        method: ProviderHttpMethod::Post,
        url: MUSE_KEY_URL.to_string(),
        headers: vec![
            ("Accept".to_string(), "application/json".to_string()),
            (
                "Authorization".to_string(),
                format!("Bearer {oauth_access_token}"),
            ),
            ("Content-Type".to_string(), "application/json".to_string()),
            ("x-api-version".to_string(), API_VERSION.to_string()),
        ],
        body: Some(json!({ "onboard": true }).to_string()),
        follow_redirects: false,
    };
    let response = http
        .request(request, REQUEST_TIMEOUT_MS)
        .await
        .map_err(|e| format!("Muse Code key exchange failed: {e}"))?;
    let text = response.body.trim().to_string();
    if !response.ok() {
        let excerpt = if text.is_empty() {
            String::new()
        } else {
            format!(" {}", text.chars().take(500).collect::<String>())
        };
        return Err(format!(
            "Muse Code key exchange failed: {}{excerpt}",
            response.status
        ));
    }
    let body: Value = serde_json::from_str(&text)
        .map_err(|_| "Muse Code key exchange returned invalid JSON".to_string())?;
    if body.get("is_subs_active") == Some(&Value::Bool(false)) {
        return Err("invalid_grant: Muse Code subscription is inactive".to_string());
    }
    let api_key = field(&body, "api_key").unwrap_or_default();
    if api_key.is_empty() {
        let action_url =
            field(&body, "action_url").or_else(|| field(&body, "require_payment_action_url"));
        if body.get("require_payment") == Some(&Value::Bool(true)) || action_url.is_some() {
            return Err(match action_url {
                Some(url) => format!("Muse Code subscription is required: {url}"),
                None => "Muse Code subscription is required".to_string(),
            });
        }
        return Err("Muse Code key response is missing api_key".to_string());
    }
    let email = field(&body, "user_email").map(|email| email.trim().to_lowercase());
    let account_id = field(&body, "user_id").or_else(|| email.clone());
    let Some(account_id) = account_id else {
        return Err("Muse Code key response is missing a stable account identity".to_string());
    };
    let credential =
        json!({ "oauthAccessToken": oauth_access_token, "apiKey": api_key }).to_string();
    Ok((credential, account_id, email.unwrap_or_default()))
}

/// The device-code login up to (and including) the key mint.
///
/// # Errors
/// Cancelled, device authorization failed/malformed, polling failed/expired,
/// or the key mint failed.
pub async fn login(
    ui: &dyn OAuthLoginUi,
    http: &dyn ProviderHttp,
) -> Result<(String, Option<String>, Option<String>), String> {
    // (stored access, accountId, email)
    if ui.is_cancelled() {
        return Err("Login cancelled".to_string());
    }
    ui.on_progress("Requesting device authorization...");
    let device = post_form(
        http,
        DEVICE_URL,
        &[("client_id", CLIENT_ID)],
        REQUEST_TIMEOUT_MS,
    )
    .await?;
    if !device.ok() {
        return Err(format!(
            "muse-code device authorization failed: {} {}",
            device.status, device.body
        ));
    }
    let body: Value = serde_json::from_str(&device.body)
        .map_err(|_| "muse-code device authorization response is not valid JSON".to_string())?;
    let (Some(user_code), Some(device_code), Some(verification_uri)) = (
        field(&body, "user_code"),
        field(&body, "device_code"),
        field(&body, "verification_uri"),
    ) else {
        return Err("muse-code device authorization response missing required fields".to_string());
    };
    let url = field(&body, "verification_uri_complete").unwrap_or(verification_uri);
    ui.on_auth(&url, Some(&format!("Enter code: {user_code}")));
    ui.on_progress("Waiting for device authorization...");

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let mut interval_ms = body
        .get("interval")
        .and_then(Value::as_f64)
        .filter(|seconds| seconds.is_finite())
        .map_or(DEFAULT_POLL_INTERVAL_MS, |seconds| {
            (seconds * 1000.0) as u64
        })
        .max(MIN_POLL_INTERVAL_MS);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let deadline = body
        .get("expires_in")
        .and_then(Value::as_f64)
        .filter(|seconds| seconds.is_finite())
        .map(|seconds| Instant::now() + Duration::from_millis((seconds * 1000.0) as u64));
    let mut slow_downs = 0u32;
    loop {
        if ui.is_cancelled() {
            return Err("Login cancelled".to_string());
        }
        if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
            break;
        }
        let reply = post_form(
            http,
            TOKEN_URL,
            &[
                ("grant_type", "urn:ietf:params:oauth:grant-type:device_code"),
                ("client_id", CLIENT_ID),
                ("device_code", device_code.as_str()),
            ],
            REQUEST_TIMEOUT_MS,
        )
        .await?;
        let token: Value = serde_json::from_str(&reply.body).unwrap_or(Value::Null);
        let error = token.get("error").and_then(Value::as_str);
        if reply.ok() && error.is_none() {
            // Meta's device response omits expiry and rejects refresh_token grants.
            let Some(access) = field(&token, "access_token") else {
                return Err("muse-code token response missing access token".to_string());
            };
            let (access, account_id, email) = attach_api_key(http, ui, &access).await?;
            return Ok((
                access,
                (!account_id.is_empty()).then_some(account_id),
                (!email.is_empty()).then_some(email),
            ));
        }
        match error {
            Some("authorization_pending") => {}
            Some("slow_down") => {
                slow_downs += 1;
                interval_ms = (interval_ms + SLOW_DOWN_INCREMENT_MS).max(MIN_POLL_INTERVAL_MS);
            }
            Some("expired_token") => {
                return Err("muse-code device code expired; restart the login".to_string())
            }
            Some("access_denied") => {
                return Err("muse-code device authorization was denied".to_string())
            }
            other => {
                let description = token
                    .get("error_description")
                    .and_then(Value::as_str)
                    .or(other)
                    .unwrap_or("");
                return Err(format!(
                    "muse-code device token request failed: {} {description}",
                    reply.status
                )
                .trim()
                .to_string());
            }
        }
        tokio::time::sleep(Duration::from_millis(interval_ms)).await;
    }
    Err(if slow_downs > 0 {
        "Device flow timed out after one or more slow_down responses. This is often caused by clock \
         drift in WSL or VM environments. Please sync or restart the VM clock and try again."
            .to_string()
    } else {
        "Device flow timed out".to_string()
    })
}

async fn post_form(
    http: &dyn ProviderHttp,
    url: &str,
    pairs: &[(&str, &str)],
    timeout_ms: u64,
) -> Result<pa_ai::oauth::ProviderHttpResponse, String> {
    let body = pairs
        .iter()
        .map(|(key, value)| format!("{key}={}", super::cursor::encode_component(value)))
        .collect::<Vec<_>>()
        .join("&");
    http.request(
        ProviderHttpRequest {
            method: ProviderHttpMethod::Post,
            url: url.to_string(),
            headers: AUTH_HEADERS
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .chain([(
                    "Content-Type".to_string(),
                    "application/x-www-form-urlencoded".to_string(),
                )])
                .collect(),
            body: Some(body),
            follow_redirects: false,
        },
        timeout_ms,
    )
    .await
}
