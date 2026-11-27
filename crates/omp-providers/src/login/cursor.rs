//! The Cursor login flow (TS `registry/oauth/cursor.ts`): a PKCE + uuid
//! login deep-link that the browser approves, then polling
//! `api2.cursor.sh/auth/poll` with the uuid and verifier until Cursor mints
//! the token pair.

use std::time::Duration;

use sha2::{Digest, Sha256};

use crate::login::store;
use pa_ai::oauth::{
    generate_pkce, OAuthLoginUi, ProviderHttp, ProviderHttpMethod, ProviderHttpRequest,
};

const LOGIN_URL: &str = "https://cursor.com/loginDeepControl";
const POLL_URL: &str = "https://api2.cursor.sh/auth/poll";
const REQUEST_TIMEOUT_MS: u64 = 30_000;
/// TS `POLL_MAX_ATTEMPTS` / `POLL_BASE_DELAY` / `POLL_MAX_DELAY` / multiplier.
const POLL_MAX_ATTEMPTS: u32 = 150;
const POLL_BASE_DELAY: Duration = Duration::from_millis(1000);
const POLL_MAX_DELAY: Duration = Duration::from_millis(10_000);
const POLL_BACKOFF_MULTIPLIER: f64 = 1.2;
/// TS: three consecutive transport errors abort the poll.
const MAX_CONSECUTIVE_ERRORS: u32 = 3;

pub(crate) struct CursorCredentials {
    pub access: String,
    pub refresh: String,
    /// Wall-clock epoch milliseconds.
    pub expires: i64,
}

/// TS `getTokenExpiry`: JWT `exp` minus 5 minutes, else now + 1 hour.
fn token_expiry(access: &str, now_ms: i64) -> i64 {
    jwt_exp(access).map_or(now_ms + 3_600_000, |exp| {
        exp.saturating_mul(1000).saturating_sub(300_000)
    })
}

/// The `exp` claim in epoch seconds, when the access token is a JWT carrying one.
fn jwt_exp(token: &str) -> Option<i64> {
    let payload = store::jwt_payload(token)?;
    payload.get("exp").and_then(serde_json::Value::as_i64)
}

/// TS `generateCursorAuthParams` + `pollCursorAuth` + `loginCursor`.
///
/// # Errors
/// The login was cancelled, polling timed out, or the poll endpoint failed.
pub(crate) async fn login(
    ui: &dyn OAuthLoginUi,
    http: &dyn ProviderHttp,
    now_ms: i64,
) -> Result<CursorCredentials, String> {
    if ui.is_cancelled() {
        return Err("Login cancelled".to_string());
    }
    let (verifier, challenge) = generate_pkce();
    let uuid = random_uuid(&verifier);
    let login_url = format!(
        "{LOGIN_URL}?challenge={}&uuid={uuid}&mode=login&redirectTarget=cli",
        encode_component(&challenge)
    );
    ui.on_auth(&login_url, None);
    ui.on_progress("Waiting for browser authentication...");

    let mut delay = POLL_BASE_DELAY;
    let mut consecutive_errors = 0u32;
    for _ in 0..POLL_MAX_ATTEMPTS {
        tokio::time::sleep(delay).await;
        if ui.is_cancelled() {
            return Err("Login cancelled".to_string());
        }
        let request = ProviderHttpRequest {
            method: ProviderHttpMethod::Get,
            url: format!("{POLL_URL}?uuid={uuid}&verifier={verifier}"),
            headers: Vec::new(),
            body: None,
            follow_redirects: true,
        };
        match http.request(request, REQUEST_TIMEOUT_MS).await {
            Err(_) => {
                consecutive_errors += 1;
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS {
                    return Err(
                        "Too many consecutive errors during Cursor auth polling".to_string()
                    );
                }
            }
            Ok(response) if response.status == 404 => {
                consecutive_errors = 0;
                #[allow(clippy::cast_precision_loss)]
                let stretched = delay.as_millis() as f64 * POLL_BACKOFF_MULTIPLIER;
                delay = Duration::from_millis(stretched as u64).min(POLL_MAX_DELAY);
                continue;
            }
            Ok(response) if response.ok() => {
                let body: serde_json::Value = serde_json::from_str(&response.body)
                    .map_err(|_| "Cursor poll response is not valid JSON".to_string())?;
                let field = |name: &str| {
                    body.get(name)
                        .and_then(serde_json::Value::as_str)
                        .filter(|value| !value.is_empty())
                        .map(str::to_string)
                };
                let (Some(access), Some(refresh)) = (field("accessToken"), field("refreshToken"))
                else {
                    return Err(
                        "Cursor poll response missing accessToken or refreshToken".to_string()
                    );
                };
                let expires = token_expiry(&access, now_ms);
                return Ok(CursorCredentials {
                    access,
                    refresh,
                    expires,
                });
            }
            Ok(response) => {
                return Err(format!("Poll failed: {}", response.status));
            }
        }
    }
    Err("Cursor authentication polling timeout".to_string())
}

/// TS keeps the random uuid opaque; ours is derived from the fresh PKCE
/// secret, which is random per login and never leaves the process raw.
fn random_uuid(secret: &str) -> String {
    let digest = Sha256::digest(secret.as_bytes());
    let mut bytes = [0u8; 16];
    bytes.copy_from_slice(&digest[..16]);
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    let hex: String = bytes.iter().map(|byte| format!("{byte:02x}")).collect();
    format!(
        "{}-{}-{}-{}-{}",
        &hex[..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..32]
    )
}

/// JS `encodeURIComponent` semantics (the KDL/TS query building uses it).
pub(crate) fn encode_component(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for byte in value.bytes() {
        match byte {
            b'A'..=b'Z'
            | b'a'..=b'z'
            | b'0'..=b'9'
            | b'-'
            | b'_'
            | b'.'
            | b'!'
            | b'~'
            | b'*'
            | b'\''
            | b'('
            | b')' => out.push(byte as char),
            other => out.push_str(&format!("%{other:02X}")),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_derived_uuid_has_the_v4_shape() {
        let (verifier, _) = generate_pkce();
        let uuid = random_uuid(&verifier);
        let parts: Vec<&str> = uuid.split('-').collect();
        assert_eq!(parts.len(), 5);
        assert_eq!(
            parts.iter().map(|p| p.len()).collect::<Vec<_>>(),
            vec![8, 4, 4, 4, 12]
        );
        assert!(uuid.chars().all(|c| c == '-' || c.is_ascii_hexdigit()));
        assert_eq!(&uuid[14..15], "4");
        assert_ne!(uuid, random_uuid(&generate_pkce().0));
    }

    #[test]
    fn jwt_expiry_tracks_the_exp_claim_with_the_ts_skew() {
        use base64::Engine as _;
        let payload = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(br#"{"exp":1000}"#);
        let token = format!("a.{payload}.b");
        assert_eq!(token_expiry(&token, 0), 700_000);
        assert_eq!(token_expiry("not-a-jwt", 5), 3_600_005);
    }

    #[test]
    fn encode_component_matches_encode_uri_component() {
        assert_eq!(encode_component("a b&c=d"), "a%20b%26c%3Dd");
        assert_eq!(encode_component("kept-_.!~*'()"), "kept-_.!~*'()");
    }
}
