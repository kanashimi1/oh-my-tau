//! The Kilo Gateway device login (TS `registry/oauth/kilo.ts`): request a
//! device code from `api.kilo.ai`, show the verification URL, and poll until
//! approved. The result is a bearer token with a one-year expiry (no refresh).

use std::time::{Duration, Instant};

use pa_ai::oauth::{OAuthLoginUi, ProviderHttp, ProviderHttpMethod, ProviderHttpRequest};

use super::api_key::LOGIN_CANCELLED;

const DEVICE_AUTH_BASE_URL: &str = "https://api.kilo.ai/api/device-auth";
/// TS `POLL_INTERVAL_MS`.
const POLL_INTERVAL_MS: u64 = 5000;
/// TS `ONE_YEAR_MS`.
const ONE_YEAR_MS: i64 = 365 * 24 * 60 * 60 * 1000;
const REQUEST_TIMEOUT_MS: u64 = 30_000;
const CANCEL_POLL_INTERVAL: Duration = Duration::from_millis(250);

/// The approved access token; `refresh` is empty and the expiry is one year.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KiloCredentials {
    pub access: String,
    /// Wall-clock epoch milliseconds.
    pub expires: i64,
}

/// # Errors
/// Cancel ([`LOGIN_CANCELLED`]), initiation/polling failures, denial, expiry,
/// or the timeout message.
pub(crate) async fn login_kilo(
    http: &dyn ProviderHttp,
    ui: &dyn OAuthLoginUi,
) -> Result<KiloCredentials, String> {
    if ui.is_cancelled() {
        return Err(LOGIN_CANCELLED.to_string());
    }
    let initiate = post_initiate(http).await?;
    if initiate.status == 429 {
        return Err("Too many pending authorization requests. Please try again later.".to_string());
    }
    if !initiate.ok() {
        return Err(format!(
            "Failed to initiate device authorization: {}",
            initiate.status
        ));
    }
    let body: serde_json::Value =
        serde_json::from_str(&initiate.body).unwrap_or(serde_json::Value::Null);
    let string_at = |name: &str| {
        body.get(name)
            .and_then(serde_json::Value::as_str)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
    };
    let (Some(user_code), Some(verification_url)) =
        (string_at("code"), string_at("verificationUrl"))
    else {
        return Err("Kilo device authorization response missing required fields".to_string());
    };
    let expires_in_seconds = body.get("expiresIn").and_then(serde_json::Value::as_f64);
    let Some(expires_in_seconds) =
        expires_in_seconds.filter(|seconds| seconds.is_finite() && *seconds > 0.0)
    else {
        return Err("Kilo device authorization response missing required fields".to_string());
    };
    ui.on_auth(&verification_url, Some(&format!("Enter code: {user_code}")));
    // The wire's expiresIn is a second count (JSON number); ms is the loop's unit.
    #[allow(clippy::cast_possible_truncation)]
    let deadline =
        Instant::now() + Duration::from_millis((expires_in_seconds * 1000.0).max(0.0) as u64);
    let poll_url = format!(
        "{DEVICE_AUTH_BASE_URL}/codes/{}",
        percent_encode(user_code.as_bytes())
    );
    loop {
        if Instant::now() >= deadline {
            break;
        }
        if ui.is_cancelled() {
            return Err(LOGIN_CANCELLED.to_string());
        }
        let poll = get(http, &poll_url).await?;
        match poll.status {
            202 => {
                cancel_aware_sleep(ui, Duration::from_millis(POLL_INTERVAL_MS)).await?;
                continue;
            }
            403 => return Err("Authorization was denied".to_string()),
            410 => return Err("Authorization code expired. Please try again.".to_string()),
            status if !(200..300).contains(&status) => {
                return Err(format!("Failed to poll device authorization: {status}"));
            }
            _ => {}
        }
        let body: serde_json::Value =
            serde_json::from_str(&poll.body).unwrap_or(serde_json::Value::Null);
        let status = body
            .get("status")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default();
        let token = body
            .get("token")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default();
        if status == "approved" && !token.is_empty() {
            return Ok(KiloCredentials {
                access: token.to_string(),
                expires: super::now_ms_i64().saturating_add(ONE_YEAR_MS),
            });
        }
        if status == "denied" {
            return Err("Authorization was denied".to_string());
        }
        if status == "expired" {
            return Err("Authorization code expired. Please try again.".to_string());
        }
        cancel_aware_sleep(ui, Duration::from_millis(POLL_INTERVAL_MS)).await?;
    }
    Err("Authentication timed out. Please try again.".to_string())
}

/// TS initiates with an empty JSON POST.
async fn post_initiate(
    http: &dyn ProviderHttp,
) -> Result<pa_ai::oauth::ProviderHttpResponse, String> {
    http.request(
        ProviderHttpRequest {
            method: ProviderHttpMethod::Post,
            url: format!("{DEVICE_AUTH_BASE_URL}/codes"),
            headers: vec![("Content-Type".to_string(), "application/json".to_string())],
            body: None,
            follow_redirects: true,
        },
        REQUEST_TIMEOUT_MS,
    )
    .await
    .map_err(|error| format!("Kilo request to {DEVICE_AUTH_BASE_URL}/codes failed: {error}"))
}

/// `encodeURIComponent`: unreserved ASCII passes, everything else is `%XX`.
fn percent_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len());
    for byte in bytes {
        match byte {
            b'A'..=b'Z'
            | b'a'..=b'z'
            | b'0'..=b'9'
            | b'-'
            | b'_'
            | b'.'
            | b'~'
            | b'!'
            | b'*'
            | b'\''
            | b'('
            | b')' => {
                out.push(*byte as char);
            }
            other => out.push_str(&format!("%{other:02X}")),
        }
    }
    out
}

async fn get(
    http: &dyn ProviderHttp,
    url: &str,
) -> Result<pa_ai::oauth::ProviderHttpResponse, String> {
    http.request(
        ProviderHttpRequest {
            method: ProviderHttpMethod::Get,
            url: url.to_string(),
            headers: Vec::new(),
            body: None,
            follow_redirects: true,
        },
        REQUEST_TIMEOUT_MS,
    )
    .await
    .map_err(|error| format!("Kilo request to {url} failed: {error}"))
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
