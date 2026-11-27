//! The Perplexity login (TS `registry/oauth/perplexity.ts`), HTTP email-OTP
//! path only: CSRF token, `signin-email`, then `signin-otp` (with the
//! authenticator challenge branch), yielding a session JWT. The macOS
//! `defaults read ai.perplexity.mac authToken` borrowing and the
//! `onBrowserSession` SSO capture have no omt seam and stay unported.

use base64::Engine as _;
use serde_json::{json, Value};

use pa_ai::oauth::{
    OAuthLoginUi, OAuthPrompt, ProviderHttp, ProviderHttpMethod, ProviderHttpRequest,
    ProviderHttpResponse,
};

const API_VERSION: &str = "2.18";
const APP_USER_AGENT: &str = "Perplexity/641 CFNetwork/1568 Darwin/25.2.0";
const BASE_URL: &str = "https://www.perplexity.ai";
const REQUEST_TIMEOUT_MS: u64 = 30_000;
/// TS `NEVER_EXPIRES` (max safe Date value).
pub(crate) const NEVER_EXPIRES: i64 = 8_640_000_000_000_000;

pub struct PerplexityCredentials {
    pub access: String,
    /// Perplexity has no refresh token; TS mirrors the JWT into `refresh`.
    pub refresh: String,
    /// Wall-clock epoch milliseconds.
    pub expires: i64,
    pub email: String,
}

async fn prompt(ui: &dyn OAuthLoginUi, message: &str, placeholder: &str) -> Result<String, String> {
    ui.on_prompt(&OAuthPrompt {
        message: message.to_string(),
        placeholder: Some(placeholder.to_string()),
        allow_empty: false,
    })
    .await
    .ok_or_else(|| "Login cancelled".to_string())
}

/// TS `getJwtExpiry`: `exp` minus 5 minutes when present, else never.
fn jwt_expiry(token: &str) -> i64 {
    let Ok(payload) = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(token.split('.').nth(1).unwrap_or_default())
    else {
        return NEVER_EXPIRES;
    };
    let Ok(payload) = serde_json::from_str::<Value>(&String::from_utf8_lossy(&payload)) else {
        return NEVER_EXPIRES;
    };
    payload
        .get("exp")
        .and_then(Value::as_i64)
        .map_or(NEVER_EXPIRES, |exp| {
            exp.saturating_mul(1000).saturating_sub(300_000)
        })
}

async fn request(
    http: &dyn ProviderHttp,
    method: ProviderHttpMethod,
    url: &str,
    body: Option<String>,
) -> Result<ProviderHttpResponse, String> {
    http.request(
        ProviderHttpRequest {
            method,
            url: url.to_string(),
            headers: vec![
                ("User-Agent".to_string(), APP_USER_AGENT.to_string()),
                ("X-App-ApiVersion".to_string(), API_VERSION.to_string()),
                ("Content-Type".to_string(), "application/json".to_string()),
            ],
            body,
            follow_redirects: false,
        },
        REQUEST_TIMEOUT_MS,
    )
    .await
    .map_err(|error| format!("Perplexity request to {url} failed: {error}"))
}

/// The TS `httpEmailLogin` path.
///
/// # Errors
/// Cancelled, an HTTP failure, or an OTP/challenge rejection.
pub async fn login(
    ui: &dyn OAuthLoginUi,
    http: &dyn ProviderHttp,
) -> Result<PerplexityCredentials, String> {
    if ui.is_cancelled() {
        return Err("Login cancelled".to_string());
    }
    let email = prompt(
        ui,
        "Enter your Perplexity email address",
        "user@example.com",
    )
    .await?;
    let email = email.trim().to_string();
    if email.is_empty() {
        return Err("Email is required for Perplexity login".to_string());
    }

    ui.on_progress("Fetching Perplexity CSRF token...");
    let csrf_response = request(
        http,
        ProviderHttpMethod::Get,
        &format!("{BASE_URL}/api/auth/csrf"),
        None,
    )
    .await?;
    if !csrf_response.ok() {
        return Err(format!(
            "Perplexity CSRF request failed: {}",
            csrf_response.status
        ));
    }
    let csrf: Value = serde_json::from_str(&csrf_response.body)
        .map_err(|_| "Perplexity CSRF response is not valid JSON".to_string())?;
    let Some(csrf_token) = csrf
        .get("csrfToken")
        .and_then(Value::as_str)
        .filter(|t| !t.is_empty())
    else {
        return Err("Perplexity CSRF response missing csrfToken".to_string());
    };

    ui.on_progress("Sending login code to your email...");
    let send = request(
        http,
        ProviderHttpMethod::Post,
        &format!("{BASE_URL}/api/auth/signin-email"),
        Some(json!({ "email": email, "csrfToken": csrf_token }).to_string()),
    )
    .await?;
    if !send.ok() {
        return Err(format!(
            "Perplexity send login code failed ({}): {}",
            send.status,
            send.body.chars().take(500).collect::<String>()
        ));
    }

    let otp = prompt(ui, "Enter the code sent to your email", "123456").await?;
    let otp = otp.trim().to_string();
    if otp.is_empty() {
        return Err("OTP code is required".to_string());
    }
    ui.on_progress("Verifying login code...");
    let mut verify = request(
        http,
        ProviderHttpMethod::Post,
        &format!("{BASE_URL}/api/auth/signin-otp"),
        Some(json!({ "email": email, "otp": otp, "csrfToken": csrf_token }).to_string()),
    )
    .await?;
    let mut data: Value = serde_json::from_str(&verify.body).unwrap_or(Value::Null);

    if !verify.ok() {
        let reason = rejection_reason(&data);
        return Err(format!("Perplexity OTP verification failed: {reason}"));
    }

    if data.get("status").and_then(Value::as_str) == Some("totp_challenge_required")
        && data
            .get("challenge_token")
            .and_then(Value::as_str)
            .is_some()
    {
        let totp = prompt(ui, "Enter the code from your authenticator app", "123456").await?;
        let totp = totp.trim().to_string();
        if totp.is_empty() {
            return Err("Authenticator code is required".to_string());
        }
        ui.on_progress("Verifying authenticator code...");
        let challenge_token = data
            .get("challenge_token")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        verify = request(
            http,
            ProviderHttpMethod::Post,
            &format!("{BASE_URL}/api/auth/totp/challenge-verify"),
            Some(json!({ "token": challenge_token, "code": totp }).to_string()),
        )
        .await?;
        data = serde_json::from_str(&verify.body).unwrap_or(Value::Null);
        if !verify.ok() {
            let reason = rejection_reason(&data);
            return Err(format!(
                "Perplexity authenticator verification failed: {reason}"
            ));
        }
    }

    let token = data
        .get("token")
        .and_then(Value::as_str)
        .or_else(|| data.get("challenge_token").and_then(Value::as_str))
        .unwrap_or_default();
    let error_code = data
        .get("error_code")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let status = data
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or("success");
    if token.is_empty()
        || !error_code.is_empty()
        || (status != "success" && status != "totp_challenge_required")
    {
        let reason = rejection_reason(&data);
        return Err(format!(
            "Perplexity OTP verification response rejected: {reason}"
        ));
    }
    Ok(PerplexityCredentials {
        access: token.to_string(),
        refresh: token.to_string(),
        expires: jwt_expiry(token),
        email,
    })
}

fn rejection_reason(data: &Value) -> String {
    for key in ["text", "error_code", "status"] {
        if let Some(reason) = data
            .get(key)
            .and_then(Value::as_str)
            .filter(|r| !r.is_empty())
        {
            return reason.to_string();
        }
    }
    "missing token".to_string()
}
