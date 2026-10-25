//! The Xiaomi MiMo login (TS `registry/oauth/xiaomi.ts`): paste an API key
//! (`sk-…` or token-plan `tp-…`), then validate it with a one-token
//! chat-completions probe — regional `tp-` keys probe the SGP → AMS → CN
//! Token Plan clusters in fallback order, standard keys the standard host.

use pa_ai::oauth::{
    OAuthLoginUi, OAuthPrompt, ProviderHttp, ProviderHttpMethod, ProviderHttpRequest,
};

use super::api_key::{LOGIN_CANCELLED, VALIDATION_TIMEOUT_MS};

const PROVIDER_NAME: &str = "Xiaomi MiMo";
const AUTH_URL: &str = "https://platform.xiaomimimo.com/#/console/api-keys";
const STANDARD_API_BASE_URL: &str = "https://api.xiaomimimo.com/v1";
const VALIDATION_MODEL: &str = "mimo-v2.5";
const TOKEN_PLAN_SGP_API_BASE_URL: &str = "https://token-plan-sgp.xiaomimimo.com/v1";
const TOKEN_PLAN_AMS_API_BASE_URL: &str = "https://token-plan-ams.xiaomimimo.com/v1";
const TOKEN_PLAN_CN_API_BASE_URL: &str = "https://token-plan-cn.xiaomimimo.com/v1";
/// TS `TOKEN_PLAN_KEY_PREFIX`.
const TOKEN_PLAN_KEY_PREFIX: &str = "tp-";

/// # Errors
/// Cancel ([`LOGIN_CANCELLED`]), an empty key, or a failed validation probe.
pub(crate) async fn login_xiaomi(
    ui: &dyn OAuthLoginUi,
    http: &dyn ProviderHttp,
) -> Result<String, String> {
    ui.on_auth(
        AUTH_URL,
        Some("Copy your API key from the Xiaomi MiMo console"),
    );
    let answer = ui
        .on_prompt(&OAuthPrompt {
            message: "Paste your Xiaomi API key (sk-... or token-plan tp-...)".to_string(),
            placeholder: Some("sk-... or tp-...".to_string()),
            allow_empty: false,
        })
        .await
        .ok_or_else(|| LOGIN_CANCELLED.to_string())?;
    if ui.is_cancelled() {
        return Err(LOGIN_CANCELLED.to_string());
    }
    let key = answer.trim().to_string();
    if key.is_empty() {
        return Err("API key is required".to_string());
    }
    ui.on_progress("Validating xiaomi API key...");
    validate_xiaomi_api_key(&key, http, &|| ui.is_cancelled()).await?;
    Ok(key)
}

/// One endpoint attempt: `Ok(None)` = accepted, `Ok(Some(message))` = 401
/// (this endpoint did not accept the key; try the next), `Err` = real failure.
async fn probe(
    base_url: &str,
    key: &str,
    http: &dyn ProviderHttp,
) -> Result<Option<String>, String> {
    let body = serde_json::json!({
        "model": VALIDATION_MODEL,
        "max_tokens": 1,
        "messages": [{"role": "user", "content": "ping"}],
    })
    .to_string();
    let response = http
        .request(
            ProviderHttpRequest {
                method: ProviderHttpMethod::Post,
                url: format!("{base_url}/chat/completions"),
                headers: vec![
                    ("Content-Type".to_string(), "application/json".to_string()),
                    ("Authorization".to_string(), format!("Bearer {key}")),
                ],
                body: Some(body),
                follow_redirects: true,
            },
            VALIDATION_TIMEOUT_MS,
        )
        .await
        .map_err(|error| format!("{PROVIDER_NAME} API key validation failed: {error}"))?;
    if response.ok() {
        return Ok(None);
    }
    // 401: this endpoint did not accept the key; try the next one.
    if response.status == 401 {
        let details = response.body.trim().to_string();
        let message = if details.is_empty() {
            format!(
                "{PROVIDER_NAME} API key validation failed ({})",
                response.status
            )
        } else {
            format!(
                "{PROVIDER_NAME} API key validation failed ({}): {details}",
                response.status
            )
        };
        return Ok(Some(message));
    }
    // Non-auth errors are real failures.
    let details = response.body.trim().to_string();
    if details.is_empty() {
        Err(format!(
            "{PROVIDER_NAME} API key validation failed ({})",
            response.status
        ))
    } else {
        Err(format!(
            "{PROVIDER_NAME} API key validation failed ({}): {details}",
            response.status
        ))
    }
}

/// # Errors
/// Every probed endpoint rejected the key or the transport failed.
pub(crate) async fn validate_xiaomi_api_key(
    key: &str,
    http: &dyn ProviderHttp,
    cancelled: &(dyn Fn() -> bool + Sync),
) -> Result<(), String> {
    // Regional keys keep the historical SGP → AMS → CN fallback; a shared
    // timeout would stay aborted, so each probe gets a fresh one.
    let endpoints: &[&str] = if key.starts_with(TOKEN_PLAN_KEY_PREFIX) {
        &[
            TOKEN_PLAN_SGP_API_BASE_URL,
            TOKEN_PLAN_AMS_API_BASE_URL,
            TOKEN_PLAN_CN_API_BASE_URL,
        ]
    } else {
        &[STANDARD_API_BASE_URL]
    };
    let mut last_error = format!("{PROVIDER_NAME} API key validation failed");
    for endpoint in endpoints {
        match probe(endpoint, key, http).await {
            Ok(None) => return Ok(()),
            Ok(Some(message)) => {
                last_error = message;
            }
            Err(error) => {
                // Only a caller cancel stops the fallback; timeouts and other
                // transport failures fall through to the next endpoint.
                if cancelled() {
                    return Err(LOGIN_CANCELLED.to_string());
                }
                last_error = error;
            }
        }
    }
    Err(last_error)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::future::Future;
    use std::pin::Pin;

    struct Scripted(Vec<(u16, String)>);
    impl ProviderHttp for Scripted {
        fn request(
            &self,
            _request: ProviderHttpRequest,
            _timeout_ms: u64,
        ) -> Pin<
            Box<
                dyn Future<Output = Result<pa_ai::oauth::ProviderHttpResponse, String>> + Send + '_,
            >,
        > {
            let (status, body) = self.0.first().cloned().unwrap_or((500, String::new()));
            Box::pin(async move { Ok(pa_ai::oauth::ProviderHttpResponse { status, body }) })
        }
    }

    #[tokio::test]
    async fn standard_key_probes_only_the_standard_host() {
        let http = Scripted(vec![(200, "{}".into())]);
        validate_xiaomi_api_key("sk-1", &http, &|| false)
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn token_plan_key_falls_back_from_sgp_to_ams() {
        // 401 on SGP is a fallback, not a failure; a transport error also falls through.
        struct Fallback;
        impl ProviderHttp for Fallback {
            fn request(
                &self,
                request: ProviderHttpRequest,
                _timeout_ms: u64,
            ) -> Pin<
                Box<
                    dyn Future<Output = Result<pa_ai::oauth::ProviderHttpResponse, String>>
                        + Send
                        + '_,
                >,
            > {
                let reply = if request.url.contains("token-plan-sgp") {
                    Err("the request timed out".to_string())
                } else if request.url.contains("token-plan-ams") {
                    Ok(pa_ai::oauth::ProviderHttpResponse {
                        status: 401,
                        body: "no".to_string(),
                    })
                } else {
                    Ok(pa_ai::oauth::ProviderHttpResponse {
                        status: 200,
                        body: "{}".to_string(),
                    })
                };
                Box::pin(async move { reply })
            }
        }
        validate_xiaomi_api_key("tp-1", &Fallback, &|| false)
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn hard_failure_surfaces_the_status_and_details() {
        let http = Scripted(vec![(500, "boom".into())]);
        let error = validate_xiaomi_api_key("sk-1", &http, &|| false)
            .await
            .unwrap_err();
        assert_eq!(error, "Xiaomi MiMo API key validation failed (500): boom");
    }

    #[tokio::test]
    async fn all_endpoints_rejecting_carries_the_last_details() {
        struct Reject;
        impl ProviderHttp for Reject {
            fn request(
                &self,
                _request: ProviderHttpRequest,
                _timeout_ms: u64,
            ) -> Pin<
                Box<
                    dyn Future<Output = Result<pa_ai::oauth::ProviderHttpResponse, String>>
                        + Send
                        + '_,
                >,
            > {
                Box::pin(async move {
                    Ok(pa_ai::oauth::ProviderHttpResponse {
                        status: 401,
                        body: "bad".to_string(),
                    })
                })
            }
        }
        let error = validate_xiaomi_api_key("tp-1", &Reject, &|| false)
            .await
            .unwrap_err();
        assert_eq!(error, "Xiaomi MiMo API key validation failed (401): bad");
    }
}
