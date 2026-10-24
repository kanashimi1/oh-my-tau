//! The Kimi Code OAuth flow: an RFC 8628 device authorization grant against
//! `{base}/api/oauth/device_authorization` / `{base}/api/oauth/token`
//! (TS `rules/auth/kimi-code.kdl`, run by the generic device-code engine
//! `registry/engine/device-code.ts`). Every request carries the Kimi device
//! identity headers.

use std::time::{Duration, Instant};

use base64::Engine as _;

use super::provider_http::{ProviderHttp, ProviderHttpMethod, ProviderHttpRequest};
use super::types::OAuthLoginUi;

pub const KIMI_CLIENT_ID: &str = "17e5f671-d194-4dfb-9706-5516cb48c098";
/// The default OAuth host (`KIMI_CODE_OAUTH_HOST` / `KIMI_OAUTH_HOST` override it).
pub const KIMI_DEFAULT_HOST: &str = "https://auth.kimi.com";
const REQUEST_TIMEOUT_MS: u64 = 30_000;
/// TS `mapCredentials` expiry skew of the `seconds` mode.
const EXPIRY_SKEW_MS: i64 = 300_000;
/// TS `MINIMUM_DEVICE_FLOW_INTERVAL_MS`.
const MIN_POLL_INTERVAL_MS: u64 = 1000;
/// TS `DEFAULT_DEVICE_FLOW_INTERVAL_SECONDS`.
const DEFAULT_POLL_INTERVAL_MS: u64 = 5000;
/// TS `SLOW_DOWN_INTERVAL_INCREMENT_MS`.
const SLOW_DOWN_INCREMENT_MS: u64 = 5000;
const CANCEL_POLL_INTERVAL: Duration = Duration::from_millis(250);
/// The cancel error the driving surface maps to the silent cancelled outcome.
pub const LOGIN_CANCELLED: &str = "Login cancelled";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KimiCredentials {
    pub access: String,
    pub refresh: String,
    /// Wall-clock epoch milliseconds.
    pub expires: i64,
    /// The `user_id`/`sub` claim of the access token, when it is a JWT carrying one.
    pub account_id: Option<String>,
}

struct Reply {
    ok: bool,
    status: u16,
    text: String,
    body: serde_json::Value,
}

/// The device authorization, the verification URL, and the token poll.
///
/// `base` is the OAuth host (`https://auth.kimi.com` unless overridden).
///
/// # Errors
///
/// Returns an error when the login is cancelled ([`LOGIN_CANCELLED`]), the device
/// authorization fails, the grant is denied/expires, or the token response is unusable.
pub async fn login_kimi(
    http: &dyn ProviderHttp,
    ui: &dyn OAuthLoginUi,
    base: &str,
    headers: &[(String, String)],
) -> Result<KimiCredentials, String> {
    if ui.is_cancelled() {
        return Err(LOGIN_CANCELLED.to_string());
    }
    let base = base.trim_end_matches('/');
    ui.on_progress("Requesting device authorization...");
    let device = post_form(
        http,
        ui,
        &format!("{base}/api/oauth/device_authorization"),
        &[("client_id", KIMI_CLIENT_ID)],
        headers,
        REQUEST_TIMEOUT_MS,
    )
    .await?;
    if !device.ok {
        return Err(format!(
            "kimi-code device authorization failed: {} {}",
            device.status, device.text
        ));
    }
    let field = |name: &str| {
        device
            .body
            .get(name)
            .and_then(serde_json::Value::as_str)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
    };
    let (Some(user_code), Some(device_code), Some(verification_uri)) = (
        field("user_code"),
        field("device_code"),
        field("verification_uri"),
    ) else {
        return Err("kimi-code device authorization response missing required fields".to_string());
    };
    let url = field("verification_uri_complete").unwrap_or(verification_uri);
    ui.on_auth(&url, Some(&format!("Enter code: {user_code}")));
    ui.on_progress("Waiting for device authorization...");

    // The wire's interval/expiry are second counts (JSON numbers); ms integers are the poll's unit.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let mut interval_ms = device
        .body
        .get("interval")
        .and_then(serde_json::Value::as_f64)
        .filter(|seconds| seconds.is_finite())
        .map_or(DEFAULT_POLL_INTERVAL_MS, |seconds| {
            (seconds * 1000.0).max(0.0) as u64
        })
        .max(MIN_POLL_INTERVAL_MS);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let deadline = device
        .body
        .get("expires_in")
        .and_then(serde_json::Value::as_f64)
        .filter(|seconds| seconds.is_finite())
        .map(|seconds| Instant::now() + Duration::from_millis((seconds * 1000.0).max(0.0) as u64));
    let mut slow_downs = 0u32;
    let token_url = format!("{base}/api/oauth/token");

    while deadline.is_none_or(|d| Instant::now() < d) {
        if ui.is_cancelled() {
            return Err(LOGIN_CANCELLED.to_string());
        }
        let reply = post_form(
            http,
            ui,
            &token_url,
            &[
                ("grant_type", "urn:ietf:params:oauth:grant-type:device_code"),
                ("client_id", KIMI_CLIENT_ID),
                ("device_code", device_code.as_str()),
            ],
            headers,
            REQUEST_TIMEOUT_MS,
        )
        .await?;
        let error = reply.body.get("error").cloned();
        if reply.ok && error.is_none() {
            return credentials_from(&reply);
        }
        if deadline.is_some_and(|d| Instant::now() >= d) {
            break;
        }
        match error.as_ref().and_then(serde_json::Value::as_str) {
            Some("authorization_pending") => {}
            Some("slow_down") => {
                slow_downs += 1;
                interval_ms = (interval_ms + SLOW_DOWN_INCREMENT_MS).max(MIN_POLL_INTERVAL_MS);
            }
            Some("expired_token") => {
                return Err("kimi-code device code expired; restart the login".to_string());
            }
            Some("access_denied") => {
                return Err("kimi-code device authorization was denied".to_string());
            }
            other => {
                let description = reply
                    .body
                    .get("error_description")
                    .and_then(serde_json::Value::as_str)
                    .or(other)
                    .unwrap_or("");
                return Err(format!(
                    "kimi-code device token request failed: {} {description}",
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
            break;
        }
        cancel_aware_sleep(ui, wait).await?;
    }
    Err(if slow_downs > 0 {
        "Device flow timed out after one or more slow_down responses. This is often caused by clock \
         drift in WSL or VM environments. Please sync or restart the VM clock and try again."
            .to_string()
    } else {
        "Device flow timed out".to_string()
    })
}

fn credentials_from(reply: &Reply) -> Result<KimiCredentials, String> {
    let access = reply
        .body
        .get("access_token")
        .and_then(serde_json::Value::as_str)
        .filter(|token| !token.is_empty())
        .ok_or_else(|| {
            format!(
                "kimi-code token response missing access token: {}",
                reply.text.chars().take(500).collect::<String>()
            )
        })?
        .to_string();
    let seconds = reply
        .body
        .get("expires_in")
        .and_then(serde_json::Value::as_f64)
        .filter(|seconds| seconds.is_finite())
        .ok_or_else(|| "kimi-code token response missing expires_in".to_string())?;
    let refresh = reply
        .body
        .get("refresh_token")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .to_string();
    // `expires_in` seconds fit i64 millis for any sane lifetime.
    #[allow(clippy::cast_possible_truncation)]
    let expires = now_ms()
        .saturating_add((seconds * 1000.0) as i64)
        .saturating_sub(EXPIRY_SKEW_MS);
    let account_id = jwt_claim(&access, &["user_id", "sub"]);
    Ok(KimiCredentials {
        access,
        refresh,
        expires,
        account_id,
    })
}

/// The first non-empty string claim among `names` of an unverified JWT payload.
fn jwt_claim(token: &str, names: &[&str]) -> Option<String> {
    let payload = token.split('.').nth(1)?;
    let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(payload.trim_end_matches('='))
        .ok()?;
    let claims: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
    names.iter().find_map(|name| {
        claims
            .get(name)
            .and_then(serde_json::Value::as_str)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
    })
}

async fn post_form(
    http: &dyn ProviderHttp,
    ui: &dyn OAuthLoginUi,
    url: &str,
    fields: &[(&str, &str)],
    extra_headers: &[(String, String)],
    timeout_ms: u64,
) -> Result<Reply, String> {
    let body = url::form_urlencoded::Serializer::new(String::new())
        .extend_pairs(fields.iter().map(|(key, value)| (*key, *value)))
        .finish();
    let mut headers: Vec<(String, String)> = extra_headers.to_vec();
    headers.push((
        "Content-Type".to_string(),
        "application/x-www-form-urlencoded".to_string(),
    ));
    let response = http
        .request(
            ProviderHttpRequest {
                method: ProviderHttpMethod::Post,
                url: url.to_string(),
                headers,
                body: Some(body),
                follow_redirects: true,
            },
            timeout_ms,
        )
        .await
        .map_err(|message| format!("kimi-code request to {url} failed: {message}"))?;
    if ui.is_cancelled() {
        return Err(LOGIN_CANCELLED.to_string());
    }
    let parsed = serde_json::from_str(&response.body).unwrap_or(serde_json::Value::Null);
    Ok(Reply {
        ok: response.ok(),
        status: response.status,
        text: response.body,
        body: parsed,
    })
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

// Epoch millis fit i64 for ~292 million years.
#[allow(clippy::cast_possible_truncation)]
fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(i64::MAX, |elapsed| elapsed.as_millis() as i64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use parking_lot::Mutex;
    use std::collections::VecDeque;
    use std::future::Future;
    use std::pin::Pin;

    use super::super::provider_http::ProviderHttpResponse;
    use super::super::types::OAuthPrompt;

    struct Scripted {
        queued: Mutex<VecDeque<(String, ProviderHttpResponse)>>,
        seen: Mutex<Vec<ProviderHttpRequest>>,
    }

    impl Scripted {
        fn new(replies: Vec<(&str, u16, &str)>) -> Self {
            Self {
                queued: Mutex::new(
                    replies
                        .into_iter()
                        .map(|(url, status, body)| {
                            (
                                url.to_string(),
                                ProviderHttpResponse {
                                    status,
                                    body: body.to_string(),
                                },
                            )
                        })
                        .collect(),
                ),
                seen: Mutex::new(Vec::new()),
            }
        }
    }

    impl ProviderHttp for Scripted {
        fn request(
            &self,
            request: ProviderHttpRequest,
            _timeout_ms: u64,
        ) -> Pin<Box<dyn Future<Output = Result<ProviderHttpResponse, String>> + Send + '_>>
        {
            self.seen.lock().push(request.clone());
            let next = self.queued.lock().pop_front();
            Box::pin(async move {
                match next {
                    Some((url, response)) if url == request.url => Ok(response),
                    Some((url, _)) => Err(format!("expected {url}, got {}", request.url)),
                    None => Err("nothing scripted".to_string()),
                }
            })
        }
    }

    #[derive(Default)]
    struct Ui {
        auth: Mutex<Vec<(String, Option<String>)>>,
        progress: Mutex<Vec<String>>,
    }

    impl OAuthLoginUi for Ui {
        fn on_auth(&self, url: &str, instructions: Option<&str>) {
            self.auth
                .lock()
                .push((url.to_string(), instructions.map(str::to_string)));
        }
        fn on_prompt(
            &self,
            _prompt: &OAuthPrompt,
        ) -> Pin<Box<dyn Future<Output = Option<String>> + Send + '_>> {
            Box::pin(std::future::pending())
        }
        fn on_progress(&self, message: &str) {
            self.progress.lock().push(message.to_string());
        }
        fn on_manual_code_input(
            &self,
        ) -> Option<Pin<Box<dyn Future<Output = Option<String>> + Send + '_>>> {
            None
        }
    }

    fn jwt(claims: &str) -> String {
        let enc = |s: &str| base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(s);
        format!("{}.{}.sig", enc("{}"), enc(claims))
    }

    #[tokio::test]
    async fn device_flow_polls_then_maps_credentials() {
        let access = jwt(r#"{"user_id":"u-42","sub":"s"}"#);
        let token_ok =
            format!(r#"{{"access_token":"{access}","refresh_token":"r1","expires_in":3600}}"#);
        let http = Scripted::new(vec![
            (
                "https://auth.example/api/oauth/device_authorization",
                200,
                r#"{"user_code":"ABCD-1234","device_code":"dev1","verification_uri":"https://k/verify","verification_uri_complete":"https://k/verify?c=ABCD-1234","interval":1,"expires_in":60}"#,
            ),
            (
                "https://auth.example/api/oauth/token",
                400,
                r#"{"error":"authorization_pending"}"#,
            ),
            ("https://auth.example/api/oauth/token", 200, &token_ok),
        ]);
        let ui = Ui::default();
        let headers = vec![("X-Msh-Platform".to_string(), "kimi_cli".to_string())];
        let before = now_ms();
        let creds = login_kimi(&http, &ui, "https://auth.example/", &headers)
            .await
            .unwrap();
        assert_eq!(creds.access, access);
        assert_eq!(creds.refresh, "r1");
        assert_eq!(creds.account_id.as_deref(), Some("u-42"));
        assert!(creds.expires >= before + 3_600_000 - EXPIRY_SKEW_MS);
        let auth = ui.auth.lock();
        assert_eq!(auth[0].0, "https://k/verify?c=ABCD-1234");
        assert_eq!(auth[0].1.as_deref(), Some("Enter code: ABCD-1234"));
        let seen = http.seen.lock();
        assert_eq!(seen.len(), 3);
        assert!(seen[0]
            .body
            .as_deref()
            .unwrap()
            .contains(&format!("client_id={KIMI_CLIENT_ID}")));
        let poll = seen[2].body.as_deref().unwrap();
        assert!(poll.contains("grant_type=urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Adevice_code"));
        assert!(poll.contains("device_code=dev1"));
        assert!(seen.iter().all(|r| r
            .headers
            .iter()
            .any(|(k, v)| k == "X-Msh-Platform" && v == "kimi_cli")));
    }

    #[tokio::test]
    async fn denied_and_malformed_authorizations_fail() {
        let http = Scripted::new(vec![
            (
                "https://auth.example/api/oauth/device_authorization",
                200,
                r#"{"user_code":"A","device_code":"d","verification_uri":"https://k/v","interval":1}"#,
            ),
            (
                "https://auth.example/api/oauth/token",
                400,
                r#"{"error":"access_denied"}"#,
            ),
        ]);
        let err = login_kimi(&http, &Ui::default(), "https://auth.example", &[])
            .await
            .unwrap_err();
        assert_eq!(err, "kimi-code device authorization was denied");

        let http = Scripted::new(vec![(
            "https://auth.example/api/oauth/device_authorization",
            200,
            r#"{"user_code":"A"}"#,
        )]);
        let err = login_kimi(&http, &Ui::default(), "https://auth.example", &[])
            .await
            .unwrap_err();
        assert_eq!(
            err,
            "kimi-code device authorization response missing required fields"
        );

        let http = Scripted::new(vec![(
            "https://auth.example/api/oauth/device_authorization",
            503,
            "down",
        )]);
        let err = login_kimi(&http, &Ui::default(), "https://auth.example", &[])
            .await
            .unwrap_err();
        assert_eq!(err, "kimi-code device authorization failed: 503 down");
    }
}
