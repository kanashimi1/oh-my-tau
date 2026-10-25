//! The Factory Droid login (`rules/auth/factory-droid.kdl`, run by TS's
//! generic device-code engine, plus its `factory-droid-region` after-exchange
//! hook): a WorkOS RFC 8628 device grant, JWT `exp` expiry with a one-day
//! fallback, then the `whoami` identity check that pins the organization and
//! the residency/inference regions.

use std::time::{Duration, Instant};

use base64::Engine as _;
use serde_json::Value;

use pa_ai::oauth::{OAuthLoginUi, ProviderHttp, ProviderHttpMethod, ProviderHttpRequest};

use super::api_key::LOGIN_CANCELLED;

/// WorkOS public client: the access token authorizes the Factory LLM proxy.
const CLIENT_ID: &str = "client_01HNM792M5G5G1A2THWPXKFMXB";
const DEVICE_URL: &str = "https://api.workos.com/user_management/authorize/device";
const TOKEN_URL: &str = "https://api.workos.com/user_management/authenticate";
/// TS `factoryDroidApiBaseUrl`: EU residency is served from the EU host.
const FACTORY_API_BASE_URL: &str = "https://api.factory.ai";
const FACTORY_API_EU_BASE_URL: &str = "https://api.eu.factory.ai";
const REQUEST_TIMEOUT_MS: u64 = 15_000;
const WHOAMI_TIMEOUT_MS: u64 = 15_000;
/// TS `credential` map: `expires "jwt" skew-ms=0 fallback-ms=86400000`.
const EXPIRY_FALLBACK_MS: i64 = 86_400_000;
/// TS `MINIMUM_DEVICE_FLOW_INTERVAL_MS`.
const MIN_POLL_INTERVAL_MS: u64 = 1000;
/// TS `DEFAULT_DEVICE_FLOW_INTERVAL_SECONDS`.
const DEFAULT_POLL_INTERVAL_MS: u64 = 5000;
/// TS `SLOW_DOWN_INTERVAL_INCREMENT_MS`.
const SLOW_DOWN_INCREMENT_MS: u64 = 5000;
const CANCEL_POLL_INTERVAL: Duration = Duration::from_millis(250);

/// The projected credentials, including the whoami-resolved scope.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FactoryCredentials {
    pub access: String,
    pub refresh: String,
    /// Wall-clock epoch milliseconds.
    pub expires: i64,
    pub email: Option<String>,
    pub account_id: Option<String>,
    /// The Factory org claim (`external_org_id`), confirmed by whoami.
    pub org_id: Option<String>,
    /// Account residency: `eu` or `global`.
    pub region: Option<String>,
    /// Inference serving region: `global`, `eu`, or `us`.
    pub inference_region: Option<String>,
}

/// # Errors
/// Cancel ([`LOGIN_CANCELLED`]), device authorization/polling failures, an
/// unusable token response, or a failed identity check.
pub(crate) async fn login_factory(
    http: &dyn ProviderHttp,
    ui: &dyn OAuthLoginUi,
) -> Result<FactoryCredentials, String> {
    if ui.is_cancelled() {
        return Err(LOGIN_CANCELLED.to_string());
    }
    ui.on_progress("Requesting device authorization...");
    let device = post_form(http, DEVICE_URL, &[("client_id", CLIENT_ID)]).await?;
    if !device.ok {
        return Err(format!(
            "factory-droid device authorization failed: {} {}",
            device.status, device.text
        ));
    }
    let field = |name: &str| {
        device
            .body
            .get(name)
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
    };
    let (Some(user_code), Some(device_code), Some(verification_uri)) = (
        field("user_code"),
        field("device_code"),
        field("verification_uri"),
    ) else {
        return Err(
            "factory-droid device authorization response missing required fields".to_string(),
        );
    };
    let url = field("verification_uri_complete").unwrap_or_else(|| verification_uri.clone());
    // The rule declares no instructions; the code travels in the URL/progress.
    ui.on_auth(&url, None);
    ui.on_progress(&format!("Enter code: {user_code}"));
    ui.on_progress("Waiting for device authorization...");

    // The wire's interval/expiry are second counts (JSON numbers); ms integers are the poll's unit.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let mut interval_ms = device
        .body
        .get("interval")
        .and_then(Value::as_f64)
        .filter(|seconds| seconds.is_finite())
        .map_or(DEFAULT_POLL_INTERVAL_MS, |seconds| {
            (seconds * 1000.0).max(0.0) as u64
        })
        .max(MIN_POLL_INTERVAL_MS);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let deadline = device
        .body
        .get("expires_in")
        .and_then(Value::as_f64)
        .filter(|seconds| seconds.is_finite())
        .map(|seconds| Instant::now() + Duration::from_millis((seconds * 1000.0).max(0.0) as u64));
    let mut slow_downs = 0u32;
    let granted = loop {
        if deadline.is_some_and(|d| Instant::now() >= d) {
            return Err(timeout_message(slow_downs));
        }
        if ui.is_cancelled() {
            return Err(LOGIN_CANCELLED.to_string());
        }
        let reply = post_form(
            http,
            TOKEN_URL,
            &[
                ("grant_type", "urn:ietf:params:oauth:grant-type:device_code"),
                ("client_id", CLIENT_ID),
                ("device_code", device_code.as_str()),
            ],
        )
        .await?;
        let error = reply
            .body
            .get("error")
            .and_then(Value::as_str)
            .map(str::to_string);
        if reply.ok && error.is_none() {
            break reply;
        }
        match error.as_deref() {
            Some("authorization_pending") => {}
            Some("slow_down") => {
                slow_downs += 1;
                interval_ms = (interval_ms + SLOW_DOWN_INCREMENT_MS).max(MIN_POLL_INTERVAL_MS);
            }
            Some("expired_token") => {
                return Err("factory-droid device code expired; restart the login".to_string());
            }
            Some("access_denied") => {
                return Err("factory-droid device authorization was denied".to_string());
            }
            other => {
                let description = reply
                    .body
                    .get("error_description")
                    .and_then(Value::as_str)
                    .or(other)
                    .unwrap_or("");
                return Err(format!(
                    "factory-droid device token request failed: {} {description}",
                    reply.status
                )
                .trim()
                .to_string());
            }
        }
        let wait = Duration::from_millis(interval_ms);
        let wait = deadline.map_or(wait, |d| {
            wait.min(d.saturating_duration_since(Instant::now()))
        });
        if wait.is_zero() {
            return Err(timeout_message(slow_downs));
        }
        cancel_aware_sleep(ui, wait).await?;
    };
    if ui.is_cancelled() {
        return Err(LOGIN_CANCELLED.to_string());
    }
    let mut credentials = credentials_from(&granted.body)?;
    // TS `attachFactoryDroidRegion` after-exchange hook (login phase): resolve
    // the organization and the residency/inference regions from whoami.
    enrich_region(http, ui, &mut credentials).await?;
    Ok(credentials)
}

fn timeout_message(slow_downs: u32) -> String {
    if slow_downs > 0 {
        "Device flow timed out after one or more slow_down responses. This is often caused by clock \
         drift in WSL or VM environments. Please sync or restart the VM clock and try again."
            .to_string()
    } else {
        "Device flow timed out".to_string()
    }
}

fn credentials_from(body: &Value) -> Result<FactoryCredentials, String> {
    let scalar = |value: Option<&Value>| -> Option<String> {
        match value {
            Some(Value::String(string)) if !string.is_empty() => Some(string.clone()),
            Some(Value::Number(number)) if number.as_f64().is_some_and(f64::is_finite) => {
                Some(number.to_string())
            }
            _ => None,
        }
    };
    let access = scalar(body.get("access_token"))
        .ok_or_else(|| "factory-droid token response missing access token".to_string())?;
    let refresh = scalar(body.get("refresh_token")).unwrap_or_default();
    // `expires "jwt" skew-ms=0 fallback-ms=86400000`.
    let expires = jwt_expiry_ms(&access)
        .unwrap_or_else(|| super::now_ms_i64().saturating_add(EXPIRY_FALLBACK_MS));
    let claims = jwt_payload(&access).unwrap_or_default();
    // `email "user.email" claim="email"`: body dot-path first, then the claim.
    let email = scalar(body.get("user").and_then(|user| user.get("email")))
        .or_else(|| scalar(claims.get("email")));
    let account_id = scalar(body.get("user").and_then(|user| user.get("id")))
        .or_else(|| scalar(claims.get("sub")));
    // WorkOS organization_id is internal; only this claim can be X-Factory-Org-Id.
    let org_id = scalar(claims.get("external_org_id"));
    Ok(FactoryCredentials {
        access,
        refresh,
        expires,
        email,
        account_id,
        org_id,
        region: None,
        inference_region: None,
    })
}

/// TS `attachFactoryDroidRegion` (login phase): the whoami body wins; a login
/// that cannot resolve an organization fails.
async fn enrich_region(
    http: &dyn ProviderHttp,
    ui: &dyn OAuthLoginUi,
    credentials: &mut FactoryCredentials,
) -> Result<(), String> {
    if ui.is_cancelled() {
        return Err(LOGIN_CANCELLED.to_string());
    }
    if credentials.refresh.is_empty() {
        return Err("Factory token response missing refresh token".to_string());
    }
    let mut headers = vec![
        (
            "Authorization".to_string(),
            format!("Bearer {}", credentials.access),
        ),
        ("Accept".to_string(), "application/json".to_string()),
    ];
    if let Some(org_id) = &credentials.org_id {
        headers.push(("X-Factory-Org-Id".to_string(), org_id.clone()));
    }
    let base = match credentials.region.as_deref() {
        Some("eu") => FACTORY_API_EU_BASE_URL,
        _ => FACTORY_API_BASE_URL,
    };
    let response = http
        .request(
            ProviderHttpRequest {
                method: ProviderHttpMethod::Get,
                url: format!("{base}/api/cli/whoami"),
                headers,
                body: None,
                follow_redirects: true,
            },
            WHOAMI_TIMEOUT_MS,
        )
        .await
        .map_err(|error| format!("Factory identity check failed: {error}"))?;
    if !response.ok() {
        return Err(format!(
            "Factory identity check failed ({}): {}",
            response.status,
            response.body.chars().take(500).collect::<String>()
        ));
    }
    let body: Value = serde_json::from_str(&response.body)
        .map_err(|error| format!("Factory whoami response invalid JSON: {error}"))?;
    let org_id = match body.get("orgId").and_then(Value::as_str) {
        Some(value) if !value.is_empty() => Some(value.to_string()),
        _ => {
            return Err("Factory login did not resolve an organization".to_string());
        }
    };
    let region = match body.get("region").and_then(Value::as_str) {
        Some(value) if value == "eu" || value == "global" => Some(value.to_string()),
        _ => credentials.region.clone(),
    };
    let inference_region = match body.get("inferenceRegion").and_then(Value::as_str) {
        Some(value) if value == "global" || value == "eu" || value == "us" => {
            Some(value.to_string())
        }
        // TS `resolveFactoryDroidInferenceRegion`: explicit scope wins, else EU
        // residency infers EU, else global.
        _ => credentials
            .inference_region
            .clone()
            .or_else(|| (region.as_deref() == Some("eu")).then(|| "eu".to_string()))
            .or_else(|| Some("global".to_string())),
    };
    credentials.org_id = org_id;
    credentials.region = region;
    credentials.inference_region = inference_region;
    Ok(())
}

/// An unverified JWT payload, `None` for malformed tokens.
fn jwt_payload(token: &str) -> Option<serde_json::Map<String, Value>> {
    let payload = token.split('.').nth(1)?;
    let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(payload.trim_end_matches('='))
        .ok()?;
    serde_json::from_slice(&bytes).ok()
}

/// Epoch ms of the `exp` claim (skew 0), or `None` when absent.
fn jwt_expiry_ms(token: &str) -> Option<i64> {
    let exp = jwt_payload(token)?.get("exp")?.as_f64()?;
    if !exp.is_finite() {
        return None;
    }
    #[allow(clippy::cast_possible_truncation)]
    Some((exp * 1000.0) as i64)
}

async fn post_form(
    http: &dyn ProviderHttp,
    url: &str,
    fields: &[(&str, &str)],
) -> Result<Reply, String> {
    let body = form_body(fields);
    let response = http
        .request(
            ProviderHttpRequest {
                method: ProviderHttpMethod::Post,
                url: url.to_string(),
                headers: vec![
                    ("Accept".to_string(), "application/json".to_string()),
                    (
                        "Content-Type".to_string(),
                        "application/x-www-form-urlencoded".to_string(),
                    ),
                ],
                body: Some(body),
                follow_redirects: true,
            },
            REQUEST_TIMEOUT_MS,
        )
        .await
        .map_err(|error| format!("factory-droid request to {url} failed: {error}"))?;
    let parsed = serde_json::from_str(&response.body).unwrap_or(Value::Null);
    Ok(Reply {
        ok: response.ok(),
        status: response.status,
        text: response.body,
        body: parsed,
    })
}

fn form_body(fields: &[(&str, &str)]) -> String {
    let mut out = String::new();
    for (key, value) in fields {
        if !out.is_empty() {
            out.push('&');
        }
        out.push_str(&percent_encode(key.as_bytes()));
        out.push('=');
        out.push_str(&percent_encode(value.as_bytes()));
    }
    out
}

fn percent_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len());
    for byte in bytes {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(*byte as char);
            }
            other => out.push_str(&format!("%{other:02X}")),
        }
    }
    out
}

struct Reply {
    ok: bool,
    status: u16,
    text: String,
    body: Value,
}

/// The cooperative cancel flag ends the wait mid-sleep.
async fn cancel_aware_sleep(ui: &dyn OAuthLoginUi, total: Duration) -> Result<(), String> {
    let deadline = Instant::now() + total;
    loop {
        if ui.is_cancelled() {
            return Err(LOGIN_CANCELLED.to_string());
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Ok(());
        }
        tokio::time::sleep(remaining.min(CANCEL_POLL_INTERVAL)).await;
    }
}
