//! The `login "oauth-code"` flows of the four providers omt ports
//! (TS `registry/engine/oauth-code.ts` + the `rules/auth/*.kdl` rules):
//! google-gemini-cli, google-antigravity, devin, and stencil. The browser
//! completes the grant on a fixed loopback callback port; a paste of the
//! final redirect URL (or bare code) races the callback like the TS
//! `OAuthCallbackFlow`.

use std::time::Duration;

use base64::Engine as _;
use serde_json::{json, Map, Value};

use pa_ai::oauth::{
    generate_pkce, OAuthLoginUi, ProviderHttp, ProviderHttpMethod, ProviderHttpRequest,
    ProviderHttpResponse,
};

use super::cursor::encode_component;

const REQUEST_TIMEOUT_MS: u64 = 30_000;
/// TS `mapCredentials` skew of the `seconds` mode.
const EXPIRY_SKEW_MS: i64 = 300_000;
/// The cancel poll cadence of the pa-ai flows.
const CANCEL_POLL_MS: u64 = 250;

/// One fixed loopback callback registration (TS `callback port=… path=…`).
pub(crate) struct Loopback {
    host: &'static str,
    port: u16,
    path: &'static str,
}

/// Per-provider static configuration; dynamic values (env overrides) are
/// resolved when the flow runs.
pub(crate) struct CodeFlow {
    pub provider: &'static str,
    pub client_id: Option<String>,
    pub client_secret: Option<String>,
    pub authorize_url: String,
    pub scopes: &'static [&'static str],
    pub extra_authorize_params: Vec<(&'static str, &'static str)>,
    pub pkce: bool,
    pub state_uuid: bool,
    pub callback: Loopback,
    /// A full redirect-URI override from the environment (`stencil`).
    pub redirect_uri_env: Option<(&'static str, Option<String>)>,
    pub token_url: String,
    /// TS `token body="json" standard=#false`: only the listed params, as JSON.
    pub token_json_params: Option<&'static [&'static str]>,
    pub userinfo_url: Option<String>,
    /// The email / account-id field names of the userinfo response.
    pub userinfo_fields: (&'static str, &'static str),
}

pub(crate) struct FlowOutcome {
    pub access: String,
    pub refresh: String,
    pub expires: i64,
    /// The raw token-endpoint response (project hooks read `refresh_token`).
    pub raw: Value,
    pub email: Option<String>,
    pub account_id: Option<String>,
    pub project: Option<String>,
}

impl FlowOutcome {
    fn with_project(mut self, project: String) -> Self {
        self.project = Some(project);
        self
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

pub(crate) fn form_encode(pairs: &[(&str, &str)]) -> String {
    pairs
        .iter()
        .map(|(key, value)| format!("{}={}", encode_component(key), encode_component(value)))
        .collect::<Vec<_>>()
        .join("&")
}

async fn post(
    http: &dyn ProviderHttp,
    url: &str,
    content_type: &str,
    accept_json: bool,
    body: String,
    provider: &str,
) -> Result<ProviderHttpResponse, String> {
    let mut headers = vec![("Content-Type".to_string(), content_type.to_string())];
    if accept_json {
        headers.push(("Accept".to_string(), "application/json".to_string()));
    }
    http.request(
        ProviderHttpRequest {
            method: ProviderHttpMethod::Post,
            url: url.to_string(),
            headers,
            body: Some(body),
            follow_redirects: false,
        },
        REQUEST_TIMEOUT_MS,
    )
    .await
    .map_err(|e| format!("{provider} request to {url} failed: {e}"))
}

fn json_field(body: &Value, name: &str) -> Option<String> {
    body.get(name)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

pub(crate) fn jwt_exp_ms(token: &str, skew_ms: i64) -> Option<i64> {
    super::store::jwt_payload(token)?
        .get("exp")
        .and_then(Value::as_i64)
        .map(|exp| exp.saturating_mul(1000).saturating_sub(skew_ms))
}

/// The `code` query parameter of a pasted final redirect URL, or the bare code.
fn code_from_paste(value: &str) -> Option<String> {
    let value = value.trim();
    if value.is_empty() {
        return None;
    }
    let query = value
        .split_once('?')
        .map(|(_, query)| query)
        .unwrap_or(value);
    for pair in query.split('&') {
        let (name, code) = pair.split_once('=').unwrap_or((pair, ""));
        if name == "code" {
            let code = code.split('#').next().unwrap_or("");
            if !code.is_empty() {
                return Some(code.to_string());
            }
        }
    }
    Some(value.to_string())
}

// ---------------------------------------------------------------------------
// The loopback callback server
// ---------------------------------------------------------------------------

/// The running callback server: an accept loop task that settles a channel
/// once (first valid redirect wins); the receiver races the manual paste.
struct CallbackServer {
    code_rx: tokio::sync::mpsc::Receiver<Result<String, String>>,
    _task: tokio::task::JoinHandle<()>,
}

impl CallbackServer {
    async fn bind(loopback: &Loopback, state: &str) -> Result<Self, String> {
        let listener = tokio::net::TcpListener::bind((loopback.host, loopback.port))
            .await
            .map_err(|error| format!("port {}: {error}", loopback.port))?;
        let (tx, code_rx) = tokio::sync::mpsc::channel(1);
        let state = state.to_string();
        let path = loopback.path.to_string();
        let host = loopback.host.to_string();
        let task = tokio::spawn(async move {
            loop {
                let Ok((mut stream, _)) = listener.accept().await else {
                    break;
                };
                let head = read_head(&mut stream).await;
                let target = head
                    .lines()
                    .next()
                    .unwrap_or_default()
                    .split_whitespace()
                    .nth(1)
                    .unwrap_or_default()
                    .to_string();
                let (path_seen, query) = target.split_once('?').unwrap_or((target.as_str(), ""));
                if path_seen != path {
                    respond(&mut stream, "404 Not Found", "Callback route not found.").await;
                    continue;
                }
                let param = |name: &str| {
                    query.split('&').find_map(|pair| {
                        let (key, value) = pair.split_once('=').unwrap_or((pair, ""));
                        (key == name && !value.is_empty()).then(|| value.to_string())
                    })
                };
                if param("state").as_deref() != Some(state.as_str()).filter(|s| !s.is_empty()) {
                    respond(&mut stream, "400 Bad Request", "State mismatch.").await;
                    continue;
                }
                let settled = match param("code") {
                    Some(code) => {
                        respond(
                            &mut stream,
                            "200 OK",
                            "Authentication completed. You can close this window.",
                        )
                        .await;
                        Ok(code)
                    }
                    None => {
                        respond(
                            &mut stream,
                            "400 Bad Request",
                            "Missing authorization code.",
                        )
                        .await;
                        Err(format!(
                            "{host} token exchange failed: the callback carried no authorization code"
                        ))
                    }
                };
                let _ = tx.send(settled).await;
                break;
            }
        });
        Ok(Self {
            code_rx,
            _task: task,
        })
    }
}

async fn read_head(stream: &mut tokio::net::TcpStream) -> String {
    use tokio::io::AsyncReadExt;
    let mut head = Vec::with_capacity(1024);
    let mut buffer = [0u8; 8192];
    loop {
        if head.windows(4).any(|window| window == b"\r\n\r\n") || head.len() >= buffer.len() {
            return String::from_utf8_lossy(&head).to_string();
        }
        match stream.read(&mut buffer).await {
            Ok(0) | Err(_) => return String::from_utf8_lossy(&head).to_string(),
            Ok(read) => head.extend_from_slice(&buffer[..read]),
        }
    }
}

async fn respond(stream: &mut tokio::net::TcpStream, status: &str, message: &str) {
    use tokio::io::AsyncWriteExt;
    let body = format!("<html><body><p>{message}</p></body></html>");
    let response = format!(
        "HTTP/1.1 {status}\r\nContent-Type: text/html\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let _ = stream.write_all(response.as_bytes()).await;
    let _ = stream.shutdown().await;
}

// ---------------------------------------------------------------------------
// The flow
// ---------------------------------------------------------------------------

/// Run one `login "oauth-code"` flow end to end (browser callback or paste).
///
/// # Errors
/// The callback port could not be bound, the login was cancelled, or the
/// token exchange failed.
pub(crate) async fn run(
    http: &dyn ProviderHttp,
    ui: &dyn OAuthLoginUi,
    flow: &CodeFlow,
) -> Result<FlowOutcome, String> {
    if ui.is_cancelled() {
        return Err("Login cancelled".to_string());
    }
    // `redirectUri` overrides (stencil `redirect-uri-env`) must stay loopback http.
    let default_redirect = format!(
        "http://{}:{}{}",
        flow.callback.host, flow.callback.port, flow.callback.path
    );
    let redirect_uri = match &flow.redirect_uri_env {
        Some((_, Some(uri))) if !uri.trim().is_empty() => {
            let uri = uri.trim().to_string();
            if !uri.starts_with("http://127.0.0.1") && !uri.starts_with("http://localhost") {
                return Err(format!(
                    "Loopback redirect URI overrides must use http://, got: {uri}"
                ));
            }
            uri
        }
        Some((_, _)) => default_redirect,
        None => default_redirect,
    };

    let (verifier, challenge) = if flow.pkce {
        let (verifier, challenge) = generate_pkce();
        (Some(verifier), Some(challenge))
    } else {
        (None, None)
    };
    let scope = if flow.scopes.is_empty() {
        None
    } else {
        Some(flow.scopes.join(" "))
    };
    let state = if flow.state_uuid {
        Some(random_uuid(&verifier.clone().unwrap_or_default()))
    } else {
        None
    };

    let mut params: Vec<(&str, String)> = vec![
        ("response_type", "code".to_string()),
        ("redirect_uri", redirect_uri.clone()),
    ];
    if let Some(client_id) = &flow.client_id {
        params.push(("client_id", client_id.clone()));
    }
    if let Some(scope) = &scope {
        params.push(("scope", scope.clone()));
    }
    if let Some(challenge) = &challenge {
        params.push(("code_challenge", challenge.clone()));
        params.push(("code_challenge_method", "S256".to_string()));
    }
    if let Some(state) = &state {
        params.push(("state", state.clone()));
    }
    for (key, value) in &flow.extra_authorize_params {
        params.push((key, value.to_string()));
    }
    let query = params
        .iter()
        .map(|(key, value)| format!("{}={}", encode_component(key), encode_component(value)))
        .collect::<Vec<_>>()
        .join("&");
    let authorize_url = format!("{}?{query}", flow.authorize_url);

    let mut server = CallbackServer::bind(&flow.callback, state.as_deref().unwrap_or("")).await?;
    ui.on_auth(&authorize_url, Some(flow_instructions(flow)));

    let mut manual = ui.on_manual_code_input();
    let code = loop {
        tokio::select! {
            _ = tokio::time::sleep(Duration::from_millis(CANCEL_POLL_MS)) => {
                if ui.is_cancelled() {
                    return Err("Login cancelled".to_string());
                }
            }
            settled = server.code_rx.recv() => {
                match settled {
                    Some(Ok(code)) => break code,
                    Some(Err(message)) => return Err(message),
                    // Server task gone: the paste branch below is the
                    // remaining path.
                    None => return Err("Login cancelled".to_string()),
                }
            }
            pasted = async {
                match manual.as_mut() {
                    Some(future) => paste_code(future).await,
                    None => std::future::pending().await,
                }
            } => break pasted?,
        }
    };
    drop(server);

    exchange_token(http, ui, flow, &code, &redirect_uri, verifier.as_deref()).await
}

async fn paste_code(
    future: &mut std::pin::Pin<Box<dyn std::future::Future<Output = Option<String>> + Send + '_>>,
) -> Result<String, String> {
    match future.as_mut().await {
        Some(pasted) => code_from_paste(&pasted).ok_or_else(|| "Login cancelled".to_string()),
        None => Err("Login cancelled".to_string()),
    }
}

fn flow_instructions(flow: &CodeFlow) -> &'static str {
    match flow.provider {
        "devin" => "Sign in to Devin in your browser.",
        "stencil" => "Sign in to your stencil.so account in the browser. If the browser cannot reach this machine, paste the final redirect URL when prompted.",
        _ => "Complete the sign-in in your browser.",
    }
}

/// TS `DeclarativeOAuthCodeFlow.exchangeToken`.
#[allow(clippy::too_many_arguments)]
async fn exchange_token(
    http: &dyn ProviderHttp,
    ui: &dyn OAuthLoginUi,
    flow: &CodeFlow,
    code: &str,
    redirect_uri: &str,
    verifier: Option<&str>,
) -> Result<FlowOutcome, String> {
    let (body, content_type) = if let Some(json_params) = flow.token_json_params {
        // TS `body="json" standard=#false`: only the templated params, as JSON.
        let mut map = Map::new();
        for name in json_params {
            let value = match *name {
                "code" => code.to_string(),
                "code_verifier" => verifier.unwrap_or_default().to_string(),
                other => {
                    let _ = other;
                    String::new()
                }
            };
            map.insert((*name).to_string(), Value::String(value));
        }
        (Value::Object(map).to_string(), "application/json")
    } else {
        let mut pairs = vec![
            ("grant_type", "authorization_code".to_string()),
            ("code", code.to_string()),
            ("redirect_uri", redirect_uri.to_string()),
        ];
        if let Some(client_id) = &flow.client_id {
            pairs.push(("client_id", client_id.clone()));
        }
        if let Some(client_secret) = &flow.client_secret {
            pairs.push(("client_secret", client_secret.clone()));
        }
        if let Some(verifier) = verifier {
            pairs.push(("code_verifier", verifier.to_string()));
        }
        let pairs: Vec<(&str, &str)> = pairs.iter().map(|(k, v)| (*k, v.as_str())).collect();
        (form_encode(&pairs), "application/x-www-form-urlencoded")
    };
    let response = post(
        http,
        &flow.token_url,
        content_type,
        true,
        body,
        flow.provider,
    )
    .await?;
    if !response.ok() {
        return Err(format!(
            "{} token exchange failed: {} {}",
            flow.provider,
            response.status,
            response.body.chars().take(500).collect::<String>()
        ));
    }
    let raw: Value = serde_json::from_str(&response.body)
        .map_err(|_| format!("{} token response is not valid JSON", flow.provider))?;
    let access = json_field(&raw, "access_token")
        .ok_or_else(|| format!("{} token response missing access token", flow.provider))?;
    let refresh = json_field(&raw, "refresh_token").unwrap_or_default();
    let expires = raw
        .get("expires_in")
        .and_then(Value::as_i64)
        .map_or(super::now_ms_i64() + 3_600_000, |seconds| {
            super::now_ms_i64() + seconds * 1000 - EXPIRY_SKEW_MS
        });

    // TS `userinfo` step.
    let mut email = None;
    let mut account_id = None;
    if let Some(userinfo_url) = &flow.userinfo_url {
        ui.on_progress("Fetching account info...");
        let request = ProviderHttpRequest {
            method: ProviderHttpMethod::Get,
            url: userinfo_url.clone(),
            headers: vec![("Authorization".to_string(), format!("Bearer {access}"))],
            body: None,
            follow_redirects: false,
        };
        if let Ok(response) = http.request(request, REQUEST_TIMEOUT_MS).await {
            if response.ok() {
                if let Ok(userinfo) = serde_json::from_str::<Value>(&response.body) {
                    email = json_field(&userinfo, flow.userinfo_fields.0);
                    account_id = json_field(&userinfo, flow.userinfo_fields.1);
                }
            }
        }
    }
    Ok(FlowOutcome {
        access,
        refresh,
        expires,
        raw,
        email,
        account_id,
        project: None,
    })
}

/// TS `generateState()` of the uuid mode, derived from the fresh PKCE secret
/// (random per login; the raw secret never leaves the process).
pub(crate) fn random_uuid(secret: &str) -> String {
    use sha2::{Digest, Sha256};
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

pub(crate) fn decode_base64(value: &str) -> String {
    String::from_utf8(
        base64::engine::general_purpose::STANDARD
            .decode(value)
            .unwrap_or_default(),
    )
    .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paste_parsing_extracts_the_code_from_a_redirect_url() {
        assert_eq!(
            code_from_paste("http://127.0.0.1:54547/callback?state=s&code=abc#x"),
            Some("abc".to_string())
        );
        assert_eq!(code_from_paste("bare-code"), Some("bare-code".to_string()));
        assert_eq!(code_from_paste("  "), None);
    }

    #[test]
    fn form_encoding_matches_url_search_params() {
        assert_eq!(
            form_encode(&[("grant_type", "authorization_code"), ("code", "a/b c")]),
            "grant_type=authorization_code&code=a%2Fb%20c"
        );
    }
}
// ---------------------------------------------------------------------------
// Provider flows
// ---------------------------------------------------------------------------

const GEMINI_CODE_ASSIST_ENDPOINT: &str = "https://cloudcode-pa.googleapis.com";
const ANTIGRAVITY_CODE_ASSIST_ENDPOINT: &str = "https://daily-cloudcode-pa.googleapis.com";

// TS keeps the installed-app clients base64 in the KDL rules; kept the same
// way (see `refresh_google` in `crate::auth`).
pub(crate) const GEMINI_CLIENT_ID_B64: &str =
    "NjgxMjU1ODA5Mzk1LW9vOGZ0Mm9wcmRybnA5ZTNhcWY2YXYzaG1kaWIxMzVqLmFwcHMuZ29vZ2xldXNlcmNvbnRlbnQuY29t";
pub(crate) const GEMINI_CLIENT_SECRET_B64: &str =
    "R09DU1BYLTR1SGdNUG0tMW83U2stZ2VWNkN1NWNsWEZzeGw=";
const ANTIGRAVITY_CLIENT_ID_B64: &str =
    "MTA3MTAwNjA2MDU5MS10bWhzc2luMmgyMWxjcmUyMzV2dG9sb2poNGc0MDNlcC5hcHBzLmdvb2dsZXVzZXJjb250ZW50LmNvbQ==";
const ANTIGRAVITY_CLIENT_SECRET_B64: &str = "R09DU1BYLUs1OEZXUjQ4NkxkTEoxbUxCOHNYQzR6NnFEQWY=";

/// TS `getGeminiCliHeaders` (pi-catalog `wire/gemini-headers.ts`).
fn gemini_cli_headers(access: &str) -> Vec<(String, String)> {
    vec![
        ("Authorization".to_string(), format!("Bearer {access}")),
        ("Content-Type".to_string(), "application/json".to_string()),
        (
            "User-Agent".to_string(),
            "GeminiCLI/0.46.0/gemini-3.1-pro-preview (linux; x86_64; terminal)".to_string(),
        ),
        (
            "Client-Metadata".to_string(),
            "ideType=IDE_UNSPECIFIED,platform=PLATFORM_UNSPECIFIED,pluginType=GEMINI".to_string(),
        ),
    ]
}

/// TS `getAntigravityUserAgent` with the pinned fallback version/changelist.
fn antigravity_user_agent() -> String {
    format!(
        "antigravity/hub/2.19.1 (aidev_client; os_type={}; arch={}; cl=963137146)",
        std::env::consts::OS,
        std::env::consts::ARCH
    )
}

fn antigravity_headers(access: &str) -> Vec<(String, String)> {
    vec![
        ("Authorization".to_string(), format!("Bearer {access}")),
        ("Content-Type".to_string(), "application/json".to_string()),
        ("User-Agent".to_string(), antigravity_user_agent()),
    ]
}

fn google_flow(provider: &'static str) -> CodeFlow {
    CodeFlow {
        provider,
        client_id: None,
        client_secret: None,
        authorize_url: "https://accounts.google.com/o/oauth2/v2/auth".to_string(),
        scopes: &[],
        extra_authorize_params: Vec::new(),
        pkce: true,
        state_uuid: true,
        callback: Loopback {
            host: "127.0.0.1",
            port: 0,
            path: "/callback",
        },
        redirect_uri_env: None,
        token_url: "https://oauth2.googleapis.com/token".to_string(),
        token_json_params: None,
        userinfo_url: None,
        userinfo_fields: ("email", "sub"),
    }
}

/// TS `auth "google-gemini-cli"` login rule.
///
/// # Errors
/// Callback bind failure, cancellation, exchange, or project discovery failure.
pub(crate) async fn login_google_gemini_cli(
    http: &dyn ProviderHttp,
    ui: &dyn OAuthLoginUi,
    env: &(dyn Fn(&str) -> Option<String> + Sync),
) -> Result<FlowOutcome, String> {
    let mut flow = google_flow("google-gemini-cli");
    flow.client_id = Some(decode_base64(GEMINI_CLIENT_ID_B64));
    flow.client_secret = Some(decode_base64(GEMINI_CLIENT_SECRET_B64));
    flow.scopes = &[
        "https://www.googleapis.com/auth/cloud-platform",
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/userinfo.profile",
    ];
    flow.extra_authorize_params = vec![("access_type", "offline"), ("prompt", "consent")];
    flow.callback = Loopback {
        host: "127.0.0.1",
        port: 8085,
        path: "/oauth2callback",
    };
    flow.userinfo_url = Some("https://www.googleapis.com/oauth2/v1/userinfo?alt=json".to_string());
    let outcome = run(http, ui, &flow).await?;
    let project = discover_gemini_cli_project(http, ui, env, &outcome).await?;
    Ok(outcome.with_project(project))
}

/// TS `auth "google-antigravity"` login rule.
///
/// # Errors
/// Callback bind failure, cancellation, exchange, or project discovery failure.
pub(crate) async fn login_google_antigravity(
    http: &dyn ProviderHttp,
    ui: &dyn OAuthLoginUi,
) -> Result<FlowOutcome, String> {
    let mut flow = google_flow("google-antigravity");
    flow.client_id = Some(decode_base64(ANTIGRAVITY_CLIENT_ID_B64));
    flow.client_secret = Some(decode_base64(ANTIGRAVITY_CLIENT_SECRET_B64));
    flow.scopes = &[
        "https://www.googleapis.com/auth/cloud-platform",
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/userinfo.profile",
        "https://www.googleapis.com/auth/cclog",
        "https://www.googleapis.com/auth/experimentsandconfigs",
    ];
    flow.extra_authorize_params = vec![("access_type", "offline"), ("prompt", "consent")];
    flow.callback = Loopback {
        host: "127.0.0.1",
        port: 51121,
        path: "/oauth-callback",
    };
    flow.userinfo_url = Some("https://www.googleapis.com/oauth2/v1/userinfo?alt=json".to_string());
    let outcome = run(http, ui, &flow).await?;
    let project = discover_antigravity_project(http, ui, &outcome).await?;
    Ok(outcome.with_project(project))
}

/// TS `auth "devin"` login rule (`login "oauth-code"`, `state "uuid"`, JSON
/// token body with only `code`/`code_verifier`, `expires "jwt"` with a
/// one-year fallback).
///
/// # Errors
/// Callback bind failure, cancellation, or exchange failure.
pub(crate) async fn login_devin(
    http: &dyn ProviderHttp,
    ui: &dyn OAuthLoginUi,
) -> Result<FlowOutcome, String> {
    let mut flow = google_flow("devin");
    flow.authorize_url = "https://app.devin.ai/auth/cli/continue".to_string();
    flow.extra_authorize_params = vec![("prompt", "select_account")];
    flow.callback = Loopback {
        host: "127.0.0.1",
        port: 59653,
        path: "/callback",
    };
    flow.token_url = "https://api.devin.ai/auth/cli/token".to_string();
    flow.token_json_params = Some(&["code", "code_verifier"]);
    run(http, ui, &flow).await
}

/// TS `auth "stencil"` login rule (client id `omp`, paste fallback documented
/// in the instructions, env-overridable auth/redirect URLs).
///
/// # Errors
/// Callback bind failure, cancellation, or exchange failure.
pub(crate) async fn login_stencil(
    http: &dyn ProviderHttp,
    ui: &dyn OAuthLoginUi,
    env: &(dyn Fn(&str) -> Option<String> + Sync),
) -> Result<FlowOutcome, String> {
    let mut flow = google_flow("stencil");
    flow.client_id = Some(
        env("STENCIL_CLIENT_ID")
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| "omp".to_string()),
    );
    flow.scopes = &["openid", "profile", "email", "offline_access"];
    let auth_base = env("STENCIL_AUTH_URL")
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "https://auth.stencil.so".to_string())
        .trim_end_matches('/')
        .to_string();
    flow.authorize_url = format!("{auth_base}/oauth/authorize");
    flow.callback = Loopback {
        host: "127.0.0.1",
        port: 54547,
        path: "/callback",
    };
    flow.redirect_uri_env = Some(("STENCIL_REDIRECT_URI", env("STENCIL_REDIRECT_URI")));
    flow.token_url = format!("{auth_base}/oauth/token");
    flow.userinfo_url = Some(format!("{auth_base}/oauth/userinfo"));
    run(http, ui, &flow).await
}

// ---------------------------------------------------------------------------
// Google Cloud Code Assist project discovery (the TS after-exchange hooks)
// ---------------------------------------------------------------------------

/// TS `discoverProject` + `googleGeminiCliProjectHook` for gemini-cli.
async fn discover_gemini_cli_project(
    http: &dyn ProviderHttp,
    ui: &dyn OAuthLoginUi,
    env: &(dyn Fn(&str) -> Option<String> + Sync),
    outcome: &FlowOutcome,
) -> Result<String, String> {
    if outcome
        .raw
        .get("refresh_token")
        .and_then(Value::as_str)
        .unwrap_or("")
        .is_empty()
    {
        return Err("No refresh token received. Please try again.".to_string());
    }
    let env_project = env_project_id(env);
    let headers = gemini_cli_headers(&outcome.access);
    ui.on_progress("Checking for existing Cloud Code Assist project...");
    let load_body = match &env_project {
        Some(project) => json!({
            "cloudaicompanionProject": project,
            "metadata": {
                "ideType": "IDE_UNSPECIFIED",
                "platform": "PLATFORM_UNSPECIFIED",
                "pluginType": "GEMINI",
                "duetProject": project,
            }
        }),
        None => json!({
            "metadata": {
                "ideType": "IDE_UNSPECIFIED",
                "platform": "PLATFORM_UNSPECIFIED",
                "pluginType": "GEMINI",
            }
        }),
    };
    let load_url = format!("{GEMINI_CODE_ASSIST_ENDPOINT}/v1internal:loadCodeAssist");
    let data = match post_code_assist(http, &load_url, &headers, load_body).await? {
        Ok(data) => data,
        Err((_status, body)) => {
            // TS `isVpcScAffectedUser`: a SECURITY_POLICY_VIOLATED RPC error
            // behaves like a provisioned standard-tier account.
            if body.contains("SECURITY_POLICY_VIOLATED") {
                json!({ "currentTier": { "id": "standard-tier" } })
            } else {
                return Err(format!("loadCodeAssist failed: {_status} {body}"));
            }
        }
    };
    if has_message_field(&data, "currentTier") {
        if let Some(project) = project_from(&data) {
            return Ok(project);
        }
        if let Some(project) = env_project {
            return Ok(project);
        }
        return Err(REQUIRES_ENV_PROJECT.to_string());
    }
    let tier_id = default_tier_id(&data);
    if tier_id != "free-tier" && env_project.is_none() {
        return Err(REQUIRES_ENV_PROJECT.to_string());
    }
    ui.on_progress("Provisioning Cloud Code Assist project (this may take a moment)...");
    let mut onboard_body = json!({
        "tierId": tier_id,
        "metadata": {
            "ideType": "IDE_UNSPECIFIED",
            "platform": "PLATFORM_UNSPECIFIED",
            "pluginType": "GEMINI",
        }
    });
    if tier_id != "free-tier" {
        if let Some(project) = &env_project {
            onboard_body["cloudaicompanionProject"] = Value::String(project.clone());
            onboard_body["metadata"]["duetProject"] = Value::String(project.clone());
        }
    }
    let onboard = post_code_assist(
        http,
        &format!("{GEMINI_CODE_ASSIST_ENDPOINT}/v1internal:onboardUser"),
        &headers,
        onboard_body,
    )
    .await?
    .map_err(|(status, body)| format!("onboardUser failed: {status} {body}"))?;
    let operation = if onboard.get("done") == Some(&Value::Bool(true)) {
        onboard
    } else {
        match onboard.get("name").and_then(Value::as_str) {
            Some(name) => {
                poll_operation(http, ui, GEMINI_CODE_ASSIST_ENDPOINT, &headers, name).await?
            }
            None => onboard,
        }
    };
    if let Some(project) = operation
        .get("response")
        .and_then(|response| response.get("cloudaicompanionProject"))
        .and_then(|project| project.get("id"))
        .and_then(Value::as_str)
        .filter(|project| !project.is_empty())
    {
        return Ok(project.to_string());
    }
    if let Some(project) = env_project {
        return Ok(project);
    }
    Err(COULD_NOT_DISCOVER_PROJECT.to_string())
}

async fn load_code_assist(
    http: &dyn ProviderHttp,
    headers: &[(String, String)],
) -> Result<Value, String> {
    post_code_assist(
        http,
        &format!("{ANTIGRAVITY_CODE_ASSIST_ENDPOINT}/v1internal:loadCodeAssist"),
        headers,
        json!({ "metadata": { "ideType": "ANTIGRAVITY" } }),
    )
    .await?
    .map_err(|(status, body)| format!("loadCodeAssist failed: {status} {body}"))
}

/// TS `discoverProject` + `googleAntigravityProjectHook` for antigravity.
async fn discover_antigravity_project(
    http: &dyn ProviderHttp,
    ui: &dyn OAuthLoginUi,
    outcome: &FlowOutcome,
) -> Result<String, String> {
    if outcome
        .raw
        .get("refresh_token")
        .and_then(Value::as_str)
        .unwrap_or("")
        .is_empty()
    {
        return Err("No refresh token received. Please try again.".to_string());
    }
    let headers = antigravity_headers(&outcome.access);
    ui.on_progress("Checking Cloud Code Assist account status...");
    let initial = load_code_assist(http, &headers)
        .await
        .map_err(discovery_error)?;
    assert_free_tier_eligible(&initial)?;
    if !has_message_field(&initial, "currentTier") {
        ui.on_progress("Provisioning the Antigravity free tier...");
        onboard_antigravity(http, ui, &headers).await?;
        ui.on_progress("Refreshing Cloud Code Assist project...");
    }
    let refreshed = load_code_assist(http, &headers)
        .await
        .map_err(discovery_error)?;
    project_from(&refreshed)
        .ok_or_else(|| "loadCodeAssist did not return a cloudaicompanionProject".to_string())
}

fn discovery_error(message: String) -> String {
    format!("Could not discover an Antigravity project. {message}")
}

/// TS `assertFreeTierEligible`.
fn assert_free_tier_eligible(payload: &Value) -> Result<(), String> {
    if has_message_field(payload, "currentTier") || has_message_field(payload, "paidTier") {
        return Ok(());
    }
    let ineligible = payload
        .get("ineligibleTiers")
        .and_then(Value::as_array)
        .and_then(|tiers| {
            tiers.iter().find(|tier| {
                tier.get("tierId").and_then(Value::as_str) == Some("free-tier")
                    && tier.get("reasonMessage").and_then(Value::as_str).is_some()
            })
        });
    let Some(ineligible) = ineligible else {
        return Ok(());
    };
    let reason = ineligible
        .get("reasonMessage")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let message = match ineligible.get("validationUrl").and_then(Value::as_str) {
        Some(url) if !url.is_empty() => format!("{reason}\n{url}"),
        _ => reason,
    };
    if message.is_empty() {
        Ok(())
    } else {
        Err(message)
    }
}

/// TS `onboardUser`: free-tier onboarding with a 30s cap and 1s poll cadence.
async fn onboard_antigravity(
    http: &dyn ProviderHttp,
    ui: &dyn OAuthLoginUi,
    headers: &[(String, String)],
) -> Result<(), String> {
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    let url = format!("{ANTIGRAVITY_CODE_ASSIST_ENDPOINT}/v1internal:onboardUser");
    let mut operation = post_code_assist(
        http,
        &url,
        headers,
        json!({ "tierId": "free-tier", "metadata": { "ideType": "ANTIGRAVITY" } }),
    )
    .await?
    .map_err(|(status, body)| format!("onboardUser failed: {status} {body}"))?;
    loop {
        if operation.get("done") == Some(&Value::Bool(true)) {
            if let Some(error) = operation.get("error").filter(|error| !error.is_null()) {
                let message = error
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or("OnboardUser operation failed");
                let code = error.get("code").and_then(Value::as_i64).unwrap_or(0);
                return Err(format!("OnboardUser operation failed: {code}: {message}"));
            }
            if operation
                .get("response")
                .is_none_or(|response| response.is_null())
            {
                return Err("failed to unmarshal OnboardUserResponse".to_string());
            }
            return Ok(());
        }
        let now = std::time::Instant::now();
        if now >= deadline {
            return Err("onboardUser timed out after 30000ms".to_string());
        }
        tokio::time::sleep(Duration::from_millis(1000).min(deadline - now)).await;
        if ui.is_cancelled() {
            return Err("Login cancelled".to_string());
        }
        match operation.get("name").and_then(Value::as_str) {
            Some(name) => {
                operation = post_code_assist(
                    http,
                    &format!("{ANTIGRAVITY_CODE_ASSIST_ENDPOINT}/v1internal/{name}"),
                    headers,
                    json!({}),
                )
                .await?
                .map_err(|(status, body)| {
                    format!("onboardUser operation failed: {status} {body}")
                })?;
            }
            None => return Err("onboardUser returned an operation without a name".to_string()),
        }
    }
}

const REQUIRES_ENV_PROJECT: &str = "This account requires setting the GOOGLE_CLOUD_PROJECT or GOOGLE_CLOUD_PROJECT_ID environment variable. See https://goo.gle/gemini-cli-auth-docs#workspace-gca";
const COULD_NOT_DISCOVER_PROJECT: &str = "Could not discover or provision a Google Cloud project. Try setting the GOOGLE_CLOUD_PROJECT or GOOGLE_CLOUD_PROJECT_ID environment variable. See https://goo.gle/gemini-cli-auth-docs#workspace-gca";

fn env_project_id(env: &(dyn Fn(&str) -> Option<String> + Sync)) -> Option<String> {
    env("GOOGLE_CLOUD_PROJECT")
        .filter(|value| !value.is_empty())
        .or_else(|| env("GOOGLE_CLOUD_PROJECT_ID").filter(|value| !value.is_empty()))
}

/// TS `getDefaultTier`: the default allowed tier, else `legacy-tier`.
fn default_tier_id(payload: &Value) -> &str {
    payload
        .get("allowedTiers")
        .and_then(Value::as_array)
        .and_then(|tiers| {
            tiers
                .iter()
                .find(|tier| tier.get("isDefault") == Some(&Value::Bool(true)))
                .or_else(|| tiers.first())
        })
        .and_then(|tier| tier.get("id"))
        .and_then(Value::as_str)
        .unwrap_or("legacy-tier")
}

fn project_from(payload: &Value) -> Option<String> {
    payload
        .get("cloudaicompanionProject")
        .and_then(Value::as_str)
        .filter(|project| !project.is_empty())
        .map(str::to_string)
}

fn has_message_field(payload: &Value, field: &str) -> bool {
    payload.get(field).is_some_and(|value| !value.is_null())
}

/// `Ok(payload)` or `Err((status, body))`; a transport error is already
/// formatted.
async fn post_code_assist(
    http: &dyn ProviderHttp,
    url: &str,
    headers: &[(String, String)],
    body: Value,
) -> Result<Result<Value, (u16, String)>, String> {
    let response = http
        .request(
            ProviderHttpRequest {
                method: ProviderHttpMethod::Post,
                url: url.to_string(),
                headers: headers.to_vec(),
                body: Some(body.to_string()),
                follow_redirects: false,
            },
            REQUEST_TIMEOUT_MS,
        )
        .await
        .map_err(|error| format!("request to {url} failed: {error}"))?;
    if !response.ok() {
        return Ok(Err((response.status, response.body)));
    }
    serde_json::from_str(&response.body)
        .map(Ok)
        .map_err(|_| format!("{url} returned invalid JSON"))
}

/// TS `pollOperation` of google-gemini-cli: 24 attempts, 5s apart.
async fn poll_operation(
    http: &dyn ProviderHttp,
    ui: &dyn OAuthLoginUi,
    endpoint: &str,
    headers: &[(String, String)],
    operation_name: &str,
) -> Result<Value, String> {
    const POLL_MAX_ATTEMPTS: u32 = 24;
    for attempt in 0..POLL_MAX_ATTEMPTS {
        if attempt > 0 {
            ui.on_progress(&format!(
                "Waiting for project provisioning (attempt {}/{POLL_MAX_ATTEMPTS})...",
                attempt + 1
            ));
            tokio::time::sleep(Duration::from_secs(5)).await;
        }
        if ui.is_cancelled() {
            return Err("Login cancelled".to_string());
        }
        let request = ProviderHttpRequest {
            method: ProviderHttpMethod::Get,
            url: format!("{endpoint}/v1internal/{operation_name}"),
            headers: headers.to_vec(),
            body: None,
            follow_redirects: false,
        };
        let response = http
            .request(request, REQUEST_TIMEOUT_MS)
            .await
            .map_err(|error| format!("Failed to poll operation: {error}"))?;
        if !response.ok() {
            return Err(format!("Failed to poll operation: {}", response.status));
        }
        let data: Value = serde_json::from_str(&response.body)
            .map_err(|_| "Operation response is not valid JSON".to_string())?;
        if data.get("done") == Some(&Value::Bool(true)) {
            return Ok(data);
        }
    }
    Err(format!(
        "Project provisioning did not complete after {POLL_MAX_ATTEMPTS} attempts"
    ))
}

#[cfg(test)]
mod google_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn the_google_clients_decode_to_the_installed_app_ids() {
        assert!(decode_base64(GEMINI_CLIENT_ID_B64).ends_with(".apps.googleusercontent.com"));
        assert!(decode_base64(GEMINI_CLIENT_SECRET_B64).starts_with("GOCSPX-"));
        assert!(decode_base64(ANTIGRAVITY_CLIENT_ID_B64).ends_with(".apps.googleusercontent.com"));
        assert!(decode_base64(ANTIGRAVITY_CLIENT_SECRET_B64).starts_with("GOCSPX-"));
    }

    #[test]
    fn default_tier_picks_the_default_allowed_tier() {
        assert_eq!(default_tier_id(&json!({})), "legacy-tier");
        assert_eq!(
            default_tier_id(&json!({"allowedTiers":[{"id":"a"},{"id":"b","isDefault":true}]})),
            "b"
        );
        assert_eq!(
            default_tier_id(&json!({"allowedTiers":[{"id":"free-tier"}]})),
            "free-tier"
        );
    }
}
