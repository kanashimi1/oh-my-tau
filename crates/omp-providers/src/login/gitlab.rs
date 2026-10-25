//! The GitLab Duo OAuth flows (`rules/auth/gitlab-duo.kdl` /
//! `gitlab-duo-agent.kdl`, run by TS's generic `login "oauth-code"` engine):
//! PKCE authorization code through either a loopback callback server
//! (`gitlab-duo`) or the manual paste of the VS Code redirect
//! (`gitlab-duo-agent`), then the form-encoded token exchange.

use std::future::Future;
use std::net::ToSocketAddrs;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use pa_ai::oauth::{
    generate_pkce, OAuthLoginUi, ProviderHttp, ProviderHttpMethod, ProviderHttpRequest,
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::Notify;

use super::api_key::LOGIN_CANCELLED;

const AUTHORIZE_URL: &str = "https://gitlab.com/oauth/authorize";
const TOKEN_URL: &str = "https://gitlab.com/oauth/token";
const SCOPE: &str = "api";
/// TS `da4edff2…` (`GITLAB_CLIENT_ID` overrides it; GitLab rejects the bundled
/// client when its registered redirect URI drifts).
const DUO_CLIENT_ID: &str = "da4edff2e6ebd2bc3208611e2768bc1c1dd7be791dc5ff26ca34ca9ee44f7d4b";
/// TS `36f2a70c…`: GitLab's official VS Code OAuth application.
const AGENT_CLIENT_ID: &str = "36f2a70cddeb5a0889d4fd8295c241b7e9848e89cf9e599d0eed2d8e5350fbf5";
const CALLBACK_HOST: &str = "127.0.0.1";
const CALLBACK_PORT: u16 = 8080;
const CALLBACK_PATH: &str = "/callback";
const REDIRECT_URI_DUO: &str = "http://localhost:8080/callback";
const REDIRECT_URI_AGENT: &str = "vscode://gitlab.gitlab-workflow/authentication";
/// TS `credential` map: `expires "seconds" path="expires_in" from="created_at" skew-ms=300000`.
const EXPIRY_SKEW_MS: i64 = 300_000;
const TOKEN_TIMEOUT_MS: u64 = 30_000;
const CANCEL_POLL_INTERVAL: Duration = Duration::from_millis(250);
const DUO_INSTRUCTIONS: &str = "Complete GitLab login in browser. If GitLab responds with \"The redirect URI included is not valid\", register your own GitLab OAuth application and set GITLAB_CLIENT_ID + GITLAB_REDIRECT_URI, or use a Personal Access Token via GITLAB_TOKEN.";
const AGENT_INSTRUCTIONS: &str = "Complete GitLab login in your browser. This uses GitLab's official VS Code OAuth application. If the redirect opens VS Code instead of returning to OMP, copy the full vscode://gitlab.gitlab-workflow/authentication?... callback URL from VS Code/browser and paste it back into OMP.";

/// The projected `credential` map of the rule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GitlabCredentials {
    pub access: String,
    pub refresh: String,
    /// Wall-clock epoch milliseconds.
    pub expires: i64,
}

/// `gitlab-duo`: loopback callback (port 8080) raced against the manual paste.
///
/// # Errors
/// Cancel ([`LOGIN_CANCELLED`]), bind without a paste surface, state mismatch,
/// or a failed token exchange.
pub(crate) async fn gitlab_duo(
    ui: &dyn OAuthLoginUi,
    http: &dyn ProviderHttp,
    env: &(dyn Fn(&str) -> Option<String> + Sync),
) -> Result<GitlabCredentials, String> {
    if ui.is_cancelled() {
        return Err(LOGIN_CANCELLED.to_string());
    }
    let client_id = env("GITLAB_CLIENT_ID")
        .filter(|id| !id.trim().is_empty())
        .unwrap_or_else(|| DUO_CLIENT_ID.to_string());
    let callback = callback_override(env);
    let (redirect_uri, host, port, path) = callback.unwrap_or((
        REDIRECT_URI_DUO.to_string(),
        CALLBACK_HOST.to_string(),
        CALLBACK_PORT,
        CALLBACK_PATH.to_string(),
    ));
    let (verifier, challenge) = generate_pkce();
    let state = generate_pkce().0;
    let server = CallbackServer::bind(&host, port, &path, &state);
    login_oauth_code(
        ui,
        http,
        &client_id,
        &redirect_uri,
        &verifier,
        &challenge,
        &state,
        server,
        DUO_INSTRUCTIONS,
    )
    .await
}

/// `gitlab-duo-agent`: manual paste of the `vscode://` redirect; no local server.
///
/// # Errors
/// Cancel or a failed token exchange.
pub(crate) async fn gitlab_duo_agent(
    ui: &dyn OAuthLoginUi,
    http: &dyn ProviderHttp,
) -> Result<GitlabCredentials, String> {
    if ui.is_cancelled() {
        return Err(LOGIN_CANCELLED.to_string());
    }
    let (verifier, challenge) = generate_pkce();
    let state = generate_pkce().0;
    login_oauth_code(
        ui,
        http,
        AGENT_CLIENT_ID,
        REDIRECT_URI_AGENT,
        &verifier,
        &challenge,
        &state,
        Err("gitlab-duo-agent requires manual paste of the redirect URL".to_string()),
        AGENT_INSTRUCTIONS,
    )
    .await
}

/// A `GITLAB_REDIRECT_URI` http-loopback override binds its own host/port/path
/// (TS `resolveCallbackOptions`); a non-loopback or non-http override keeps the
/// registered loopback server and advertises the override verbatim.
fn callback_override(
    env: &(dyn Fn(&str) -> Option<String> + Sync),
) -> Option<(String, String, u16, String)> {
    let raw = env("GITLAB_REDIRECT_URI").filter(|uri| !uri.trim().is_empty())?;
    let url = reqwest::Url::parse(raw.trim()).ok()?;
    if url.scheme() != "http" {
        return None;
    }
    let host = url.host_str()?.to_string();
    if !matches!(host.as_str(), "localhost" | "127.0.0.1" | "[::1]") {
        return None;
    }
    let port = url.port_or_known_default().unwrap_or(80);
    let path = if url.path().is_empty() {
        CALLBACK_PATH.to_string()
    } else {
        url.path().to_string()
    };
    Some((raw.trim().to_string(), host, port, path))
}

/// The shared tail of both flows: present the URL, collect the code
/// (callback server / manual paste / prompt), exchange it.
#[allow(clippy::too_many_arguments)]
async fn login_oauth_code(
    ui: &dyn OAuthLoginUi,
    http: &dyn ProviderHttp,
    client_id: &str,
    redirect_uri: &str,
    verifier: &str,
    challenge: &str,
    state: &str,
    server: Result<CallbackServer, String>,
    instructions: &str,
) -> Result<GitlabCredentials, String> {
    let manual = ui.on_manual_code_input();
    let server = match server {
        Ok(server) => Some(server),
        Err(error) if manual.is_some() => {
            ui.on_progress(&format!(
                "Callback server unavailable ({error}); paste the redirect URL from the browser."
            ));
            None
        }
        Err(error) => return Err(error),
    };
    ui.on_auth(
        &authorize_url(client_id, redirect_uri, challenge, state),
        Some(instructions),
    );
    let (code, _echoed) = wait_for_code(ui, server.as_ref(), manual, state).await?;
    if ui.is_cancelled() {
        return Err(LOGIN_CANCELLED.to_string());
    }
    ui.on_progress("Exchanging authorization code for tokens...");
    exchange(http, client_id, redirect_uri, &code, verifier).await
}

fn authorize_url(client_id: &str, redirect_uri: &str, challenge: &str, state: &str) -> String {
    let mut url = reqwest::Url::parse(AUTHORIZE_URL).expect("static authorize url parses");
    for (name, value) in [
        ("client_id", client_id),
        ("response_type", "code"),
        ("redirect_uri", redirect_uri),
        ("scope", SCOPE),
        ("code_challenge", challenge),
        ("code_challenge_method", "S256"),
        ("state", state),
    ] {
        url.query_pairs_mut().append_pair(name, value);
    }
    url.to_string()
}

enum Race {
    /// The callback's code, or `None` when the wait settled empty.
    Callback(Option<(String, String)>),
    /// The pasted input, or `None` when cancelled.
    Manual(Option<String>),
}

/// The raced input: the server settle wins over a late manual paste; a surface
/// without either falls back to a blocking prompt.
async fn wait_for_code(
    ui: &dyn OAuthLoginUi,
    server: Option<&CallbackServer>,
    manual: Option<Pin<Box<dyn Future<Output = Option<String>> + Send + '_>>>,
    state: &str,
) -> Result<(String, Option<String>), String> {
    let manual_available = manual.is_some();
    let manual_answer = async {
        match manual {
            Some(future) => future.await,
            None => std::future::pending::<Option<String>>().await,
        }
    };
    tokio::pin!(manual_answer);
    let server_wait = async {
        match server {
            Some(server) => server.wait().await,
            None => std::future::pending::<Option<(String, String)>>().await,
        }
    };
    tokio::pin!(server_wait);
    let mut tick = tokio::time::interval(CANCEL_POLL_INTERVAL);
    let raced = loop {
        if ui.is_cancelled() {
            return Err(LOGIN_CANCELLED.to_string());
        }
        tokio::select! {
            _ = tick.tick() => {}
            code = &mut server_wait => break Race::Callback(code),
            answer = &mut manual_answer => break Race::Manual(answer),
        }
    };
    let input = match raced {
        Race::Callback(Some((code, echoed))) => return Ok((code, Some(echoed))),
        Race::Callback(None) => {
            if manual_available {
                match manual_answer.await {
                    Some(input) => Some(input),
                    None => return Err(LOGIN_CANCELLED.to_string()),
                }
            } else {
                None
            }
        }
        Race::Manual(None) => return Err(LOGIN_CANCELLED.to_string()),
        Race::Manual(Some(input)) => Some(input),
    };
    let input = match input {
        Some(input) => input,
        None => {
            let answer = ui
                .on_prompt(&pa_ai::oauth::OAuthPrompt {
                    message: "Paste the authorization code or full redirect URL:".to_string(),
                    placeholder: None,
                    allow_empty: false,
                })
                .await;
            answer.ok_or_else(|| LOGIN_CANCELLED.to_string())?
        }
    };
    parse_paste(&input, state)
}

fn parse_paste(input: &str, state: &str) -> Result<(String, Option<String>), String> {
    let value = input.trim();
    let (code, echoed) = if let Ok(url) = reqwest::Url::parse(value) {
        let get = |name: &str| {
            url.query_pairs()
                .find(|(key, _)| key == name)
                .map(|(_, value)| value.to_string())
                .filter(|value| !value.is_empty())
        };
        (get("code"), get("state"))
    } else if let Some((code, echoed)) = value.split_once('#') {
        (
            Some(code.to_string()),
            Some(echoed.to_string()).filter(|value| !value.is_empty()),
        )
    } else {
        (Some(value.to_string()), None)
    };
    if let Some(echoed) = &echoed {
        if echoed != state {
            return Err("OAuth state mismatch".to_string());
        }
    }
    code.filter(|code| !code.is_empty())
        .map(|code| (code, echoed))
        .ok_or_else(|| "Missing authorization code".to_string())
}

async fn exchange(
    http: &dyn ProviderHttp,
    client_id: &str,
    redirect_uri: &str,
    code: &str,
    verifier: &str,
) -> Result<GitlabCredentials, String> {
    let body = form(&[
        ("grant_type", "authorization_code"),
        ("client_id", client_id),
        ("code", code),
        ("redirect_uri", redirect_uri),
        ("code_verifier", verifier),
    ]);
    let response = http
        .request(
            ProviderHttpRequest {
                method: ProviderHttpMethod::Post,
                url: TOKEN_URL.to_string(),
                headers: vec![(
                    "Content-Type".to_string(),
                    "application/x-www-form-urlencoded".to_string(),
                )],
                body: Some(body),
                follow_redirects: true,
            },
            TOKEN_TIMEOUT_MS,
        )
        .await
        .map_err(|error| format!("GitLab token exchange request failed: {error}"))?;
    if !response.ok() {
        return Err(format!(
            "GitLab token exchange failed ({}): {}",
            response.status,
            response.body.chars().take(500).collect::<String>()
        ));
    }
    let json: serde_json::Value = serde_json::from_str(&response.body)
        .map_err(|error| format!("GitLab token response invalid JSON: {error}"))?;
    let get = |name: &str| {
        json.get(name)
            .and_then(serde_json::Value::as_str)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
    };
    let access = get("access_token")
        .ok_or_else(|| "GitLab token response missing access_token".to_string())?;
    let refresh = get("refresh_token").unwrap_or_default();
    let seconds = json
        .get("expires_in")
        .and_then(serde_json::Value::as_f64)
        .filter(|seconds| seconds.is_finite() && *seconds > 0.0)
        .ok_or_else(|| "GitLab token response missing expires_in".to_string())?;
    let created_at = json
        .get("created_at")
        .and_then(serde_json::Value::as_f64)
        .filter(|seconds| seconds.is_finite() && *seconds > 0.0);
    // Seconds fit i64 millis for any sane lifetime; clamp a hostile sum.
    #[allow(clippy::cast_possible_truncation)]
    let base_ms = created_at
        .map(|seconds| seconds * 1000.0)
        .unwrap_or_else(|| super::now_ms_i64() as f64);
    let expires = (base_ms + seconds * 1000.0).clamp(0.0, i64::MAX as f64) as i64;
    Ok(GitlabCredentials {
        access,
        refresh,
        expires: expires.saturating_sub(EXPIRY_SKEW_MS),
    })
}

pub(crate) fn form(fields: &[(&str, &str)]) -> String {
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

struct CallbackCode {
    code: String,
    state: String,
}

/// Settled once per login; the first settle wins.
#[derive(Default)]
struct CallbackShared {
    slot: tokio::sync::Mutex<Option<Option<CallbackCode>>>,
    notify: Notify,
}

/// A loopback callback server. Dropping it aborts the accept loop, releasing
/// the port.
struct CallbackServer {
    shared: Arc<CallbackShared>,
    task: Option<tokio::task::JoinHandle<()>>,
}

impl Drop for CallbackServer {
    fn drop(&mut self) {
        if let Some(task) = self.task.take() {
            task.abort();
        }
    }
}

impl CallbackServer {
    /// # Errors
    /// A bind failure (the login's hard error unless the surface pastes).
    fn bind(host: &str, port: u16, path: &str, state: &str) -> Result<Self, String> {
        let addr: std::net::SocketAddr = (host, port)
            .to_socket_addrs()
            .map_err(|error| format!("port {port}: {error}"))?
            .next()
            .ok_or_else(|| format!("port {port}: no addresses"))?;
        let socket = if addr.is_ipv4() {
            tokio::net::TcpSocket::new_v4()
        } else {
            tokio::net::TcpSocket::new_v6()
        }
        .map_err(|error| format!("port {port}: {error}"))?;
        #[cfg(unix)]
        socket
            .set_reuseaddr(true)
            .map_err(|error| format!("port {port}: {error}"))?;
        socket
            .bind(addr)
            .map_err(|error| format!("port {port}: {error}"))?;
        let listener = socket
            .listen(16)
            .map_err(|error| format!("port {port}: {error}"))?;
        let shared = Arc::new(CallbackShared::default());
        let task_shared = Arc::clone(&shared);
        let state = state.to_string();
        let path = path.to_string();
        let task = tokio::spawn(async move {
            loop {
                let Ok((stream, _)) = listener.accept().await else {
                    break;
                };
                let shared = Arc::clone(&task_shared);
                let state = state.clone();
                let path = path.clone();
                tokio::spawn(async move {
                    serve(stream, &shared, &path, &state).await;
                });
            }
        });
        Ok(Self {
            shared,
            task: Some(task),
        })
    }

    /// The code and state, or `None` when the wait was cancelled.
    async fn wait(&self) -> Option<(String, String)> {
        loop {
            if let Some(result) = self.shared.slot.lock().await.take() {
                return result.map(|code| (code.code, code.state));
            }
            self.shared.notify.notified().await;
        }
    }
}

/// One browser request: read it, answer it, settle the login only on a
/// matching redirect (path, `error`, `code`, echoed `state` checked in order).
async fn serve(
    mut stream: tokio::net::TcpStream,
    shared: &CallbackShared,
    path: &str,
    state: &str,
) {
    let mut buf = [0u8; 4096];
    let read = stream.read(&mut buf).await.unwrap_or(0);
    let head = String::from_utf8_lossy(&buf[..read]);
    let target = head
        .lines()
        .next()
        .unwrap_or_default()
        .split_whitespace()
        .nth(1)
        .unwrap_or_default()
        .to_string();
    let (route, query) = target.split_once('?').unwrap_or((target.as_str(), ""));
    let (status, page) = if route != path {
        ("400 Bad Request", "Callback route not found.".to_string())
    } else if let Some(error) = query_param(query, "error") {
        ("400 Bad Request", format!("Error: {error}"))
    } else {
        match (query_param(query, "code"), query_param(query, "state")) {
            (Some(code), Some(echoed)) if !code.is_empty() && echoed == state => {
                let mut slot = shared.slot.lock().await;
                if slot.is_none() {
                    *slot = Some(Some(CallbackCode {
                        code: code.clone(),
                        state: echoed.clone(),
                    }));
                    drop(slot);
                    shared.notify.notify_one();
                }
                (
                    "200 OK",
                    "GitLab authentication completed. You can close this window.".to_string(),
                )
            }
            (Some(_), Some(ref echoed)) if *echoed != state => {
                ("400 Bad Request", "State mismatch.".to_string())
            }
            _ => (
                "400 Bad Request",
                "Missing code or state parameter.".to_string(),
            ),
        }
    };
    let body = format!("<html><body>{page}</body></html>");
    let response = format!(
        "HTTP/1.1 {status}\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let _ = stream.write_all(response.as_bytes()).await;
    let _ = stream.shutdown().await;
}

fn query_param(query: &str, name: &str) -> Option<String> {
    query
        .split('&')
        .filter_map(|pair| pair.split_once('='))
        .find(|(key, _)| *key == name)
        .map(|(_, value)| percent_decode(value))
        .filter(|value| !value.is_empty())
}

fn percent_decode(value: &str) -> String {
    let bytes = value.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' && index + 2 < bytes.len() {
            let hex = std::str::from_utf8(&bytes[index + 1..index + 3]).unwrap_or("");
            if let Ok(byte) = u8::from_str_radix(hex, 16) {
                out.push(byte);
                index += 3;
                continue;
            }
        }
        if bytes[index] == b'+' {
            out.push(b' ');
        } else {
            out.push(bytes[index]);
        }
        index += 1;
    }
    String::from_utf8_lossy(&out).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn form_encoding_percent_encodes_unreserved_safe() {
        assert_eq!(form(&[("a", "b c")]), "a=b%20c");
        assert_eq!(form(&[("k", "v&=x")]), "k=v%26%3Dx");
        assert_eq!(percent_decode("a%20b+c"), "a b c");
        assert_eq!(percent_decode("vscode%3A%2F%2Fgitlab"), "vscode://gitlab");
    }

    #[test]
    fn paste_parsing_handles_urls_fragments_and_bare_codes() {
        assert_eq!(
            parse_paste("http://localhost:8080/callback?code=c1&state=s1", "s1").unwrap(),
            ("c1".to_string(), Some("s1".to_string()))
        );
        assert_eq!(
            parse_paste("c1#s1", "s1").unwrap(),
            ("c1".to_string(), Some("s1".to_string()))
        );
        assert_eq!(parse_paste("c1", "s1").unwrap(), ("c1".to_string(), None));
        assert_eq!(
            parse_paste("c1#other", "s1").unwrap_err(),
            "OAuth state mismatch"
        );
    }
}
