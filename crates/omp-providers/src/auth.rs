//! Stored-credential access for every request path (TS `AuthStorage.getApiKey`
//! + `auth-retry`): credentials are resolved per request, never once per
//! process.
//!
//! * A configured auth broker (`OMP_AUTH_BROKER_URL`/`_TOKEN`, `auth.broker.url`/`.token` in
//!   `config.yml`, `<config root>/auth-broker.token`; TS `resolveAuthBrokerConfig`) *replaces*
//!   the local store: snapshots come from `GET /v1/snapshot`, near-expiry OAuth rows are
//!   refreshed through `POST /v1/credential/:id/refresh` and the broker owns persistence.
//! * Without a broker the local `agent.db` is the store. An OAuth row that is expired (or within
//!   the 60 s skew) is refreshed locally under the row's refresh lease and written back with the
//!   compare-and-set TS uses (`auth_credential_refresh_leases`, `UPDATE .. WHERE data = ?`).
//! * After a provider 401 [`recover`] force-refreshes the failing credential once
//!   (`?reason=auth-recovery` through the broker) and hands back the new bearer, if it differs.

use std::collections::HashMap;
use std::future::Future;
use std::path::Path;
use std::sync::{LazyLock, Mutex};
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context as _, Result};
use rusqlite::OptionalExtension as _;
use serde_json::{Map, Value};

use crate::{now_ms, AgentDir};

/// TS `OAUTH_REFRESH_SKEW_MS`.
const REFRESH_SKEW_MS: i64 = 60_000;
/// TS `OAUTH_REMINT_COOLDOWN_MS`: auth recovery reuses a token this process minted this recently.
const REMINT_COOLDOWN: Duration = Duration::from_secs(5 * 60);
const LEASE_TTL_MS: i64 = 15_000;
const LEASE_POLL: Duration = Duration::from_millis(50);
const LEASE_WAIT: Duration = Duration::from_secs(20);
const BROKER_TIMEOUT: Duration = Duration::from_secs(10);
/// How long one broker snapshot answers repeated lookups (discovery asks per provider).
const SNAPSHOT_TTL: Duration = Duration::from_secs(2);
const CAPABILITIES_HEADER: &str = "OMP-Auth-Broker-Capabilities";
const CAPABILITY_CODEX_METER: &str = "codex-meter-block-scopes";
/// Credential rows carry this in `refresh` when the broker withholds the real refresh token.
const REMOTE_REFRESH_SENTINEL: &str = "__remote__";

/// A usable bearer for one provider and the `auth_credentials` row it came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Credential {
    pub key: String,
    pub id: i64,
}

/// Every usable credential of one provider, in store order (the row id order the
/// credential store iterates). `current` is [`credential`]'s pick; `alternates` are the
/// other rows (skip lists `current` by row id).
#[derive(Debug, Clone)]
pub struct CredentialPool {
    pub current: Credential,
    pub alternates: Vec<Credential>,
}

impl CredentialPool {
    /// Whether another account can take over when `current` is rate-limited.
    #[must_use]
    pub fn has_alternates(&self) -> bool {
        !self.alternates.is_empty()
    }
}

/// Broker connection (TS `AuthBrokerClientConfig`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BrokerConfig {
    pub url: String,
    pub token: String,
}

/// OAuth token-endpoint hosts a refresh talks to (overridable like TS `KIMI_CODE_OAUTH_HOST`).
#[derive(Debug, Clone, Default)]
pub struct Hosts {
    pub kimi: Option<String>,
}

impl Hosts {
    pub(crate) fn from_env() -> Self {
        let kimi = ["KIMI_CODE_OAUTH_HOST", "KIMI_OAUTH_HOST"]
            .iter()
            .find_map(|k| {
                std::env::var(k)
                    .ok()
                    .map(|v| v.trim().to_string())
                    .filter(|v| !v.is_empty())
            });
        Self { kimi }
    }
}

// ------------------------------------------------------------ broker config

fn read_dotted<'a>(record: &'a Value, dotted: &str) -> Option<&'a Value> {
    let flat = record.get(dotted);
    let mut current = record;
    for segment in dotted.split('.') {
        match current.as_object().and_then(|o| o.get(segment)) {
            Some(next) => current = next,
            None => return flat,
        }
    }
    Some(current)
}

fn read_dotted_string(record: &Value, dotted: &str) -> Option<String> {
    read_dotted(record, dotted)
        .and_then(Value::as_str)
        .map(str::to_string)
}

/// `(auth.broker.url, auth.broker.token)` from `<agent dir>/config.yml` (or `.yaml`).
fn read_config_broker(dir: &AgentDir) -> Result<(Option<String>, Option<String>)> {
    for name in ["config.yml", "config.yaml"] {
        let path = dir.0.join(name);
        let raw = match std::fs::read_to_string(&path) {
            Ok(raw) => raw,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(e) => bail!("Unable to read {}: {e}", path.display()),
        };
        let parsed: Value = serde_yaml_ng::from_str(&raw)
            .with_context(|| format!("{} contains invalid YAML", path.display()))?;
        return match parsed {
            Value::Null => Ok((None, None)),
            Value::Object(_) => Ok((
                read_dotted_string(&parsed, "auth.broker.url"),
                read_dotted_string(&parsed, "auth.broker.token"),
            )),
            _ => bail!("{} must contain a YAML object", path.display()),
        };
    }
    Ok((None, None))
}

/// `<config root>/auth-broker.token` (the config root is the agent dir's parent, `~/.omp`).
fn token_file(dir: &AgentDir) -> std::path::PathBuf {
    dir.0.parent().unwrap_or(&dir.0).join("auth-broker.token")
}

/// TS `resolveAuthBrokerConfig`: env, then `config.yml`, then the token file. `Ok(None)` when no
/// broker URL is configured; an error when a URL is configured without any bearer token.
///
/// # Errors
/// Unreadable/invalid `config.yml`, or a broker URL without a token.
pub fn resolve_broker_config(
    dir: &AgentDir,
    env: &dyn Fn(&str) -> Option<String>,
) -> Result<Option<BrokerConfig>> {
    let non_empty = |v: Option<String>| v.filter(|v| !v.is_empty());
    let env_token = non_empty(env("OMP_AUTH_BROKER_TOKEN"));
    let mut url = non_empty(env("OMP_AUTH_BROKER_URL"));
    let mut config_token = None;
    if url.is_none() || env_token.is_none() {
        let (config_url, token) = read_config_broker(dir)?;
        if url.is_none() {
            url = config_url
                .and_then(|raw| crate::resolve_config_value(&raw))
                .filter(|v| !v.is_empty());
        }
        config_token = token
            .and_then(|raw| crate::resolve_config_value(&raw))
            .filter(|v| !v.is_empty());
    }
    let Some(url) = url else { return Ok(None) };
    let token = env_token.or(config_token).or_else(|| {
        std::fs::read_to_string(token_file(dir))
            .ok()
            .map(|t| t.trim().to_string())
            .filter(|t| !t.is_empty())
    });
    match token {
        Some(token) => Ok(Some(BrokerConfig { url, token })),
        None => bail!(
            "OMP_AUTH_BROKER_URL is set ({url}) but no bearer token is available. Set OMP_AUTH_BROKER_TOKEN, the \
             `auth.broker.token` config entry, or place one at {}.",
            token_file(dir).display()
        ),
    }
}

fn process_env(key: &str) -> Option<String> {
    std::env::var(key).ok()
}

// ------------------------------------------------------------ broker client

struct SnapshotEntry {
    id: i64,
    provider: String,
    credential: Value,
}

fn parse_entry(v: &Value) -> Option<SnapshotEntry> {
    Some(SnapshotEntry {
        id: v.get("id")?.as_i64()?,
        provider: v.get("provider")?.as_str()?.to_string(),
        credential: v.get("credential")?.clone(),
    })
}

type SnapshotCache = HashMap<String, (Instant, Vec<(i64, String, Value)>)>;
static SNAPSHOTS: LazyLock<Mutex<SnapshotCache>> = LazyLock::new(|| Mutex::new(HashMap::new()));

fn cache_key(cfg: &BrokerConfig) -> String {
    format!("{}\u{0}{}", cfg.url, cfg.token)
}

fn invalidate_snapshot(cfg: &BrokerConfig) {
    SNAPSHOTS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .remove(&cache_key(cfg));
}

async fn broker_request(cfg: &BrokerConfig, post: bool, path: &str) -> Result<Value> {
    let url = format!("{}{path}", cfg.url.trim_end_matches('/'));
    let client = reqwest::Client::builder().timeout(BROKER_TIMEOUT).build()?;
    let mut last = None;
    // TS client: one retry on transport errors; HTTP statuses are final.
    for _ in 0..2 {
        let request = if post {
            client.post(&url)
        } else {
            client.get(&url)
        }
        .bearer_auth(&cfg.token)
        .header("Accept", "application/json")
        .header(CAPABILITIES_HEADER, CAPABILITY_CODEX_METER);
        match request.send().await {
            Ok(response) => {
                let status = response.status();
                let text = response.text().await.unwrap_or_default();
                if !status.is_success() {
                    bail!(
                        "Auth broker request failed: {status}: {}",
                        text.chars().take(300).collect::<String>()
                    );
                }
                return serde_json::from_str(&text)
                    .map_err(|e| anyhow!("Auth broker returned malformed JSON: {e}"));
            }
            Err(e) => last = Some(e),
        }
    }
    Err(anyhow!(
        "Auth broker request failed after 2 attempt(s): {}",
        last.map(|e| e.to_string()).unwrap_or_default()
    ))
}

async fn broker_snapshot(cfg: &BrokerConfig) -> Result<Vec<SnapshotEntry>> {
    let key = cache_key(cfg);
    {
        let cache = SNAPSHOTS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some((at, rows)) = cache.get(&key) {
            if at.elapsed() < SNAPSHOT_TTL {
                return Ok(rows
                    .iter()
                    .map(|(id, provider, credential)| SnapshotEntry {
                        id: *id,
                        provider: provider.clone(),
                        credential: credential.clone(),
                    })
                    .collect());
            }
        }
    }
    let body = broker_request(cfg, false, "/v1/snapshot").await?;
    let mut entries: Vec<SnapshotEntry> = body
        .get("credentials")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(parse_entry)
        .collect();
    entries.sort_by_key(|e| e.id);
    SNAPSHOTS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .insert(
            key,
            (
                Instant::now(),
                entries
                    .iter()
                    .map(|e| (e.id, e.provider.clone(), e.credential.clone()))
                    .collect(),
            ),
        );
    Ok(entries)
}

async fn broker_refresh(cfg: &BrokerConfig, id: i64, recovery: bool) -> Result<Value> {
    let suffix = if recovery {
        "?reason=auth-recovery"
    } else {
        ""
    };
    let body = broker_request(cfg, true, &format!("/v1/credential/{id}/refresh{suffix}")).await?;
    invalidate_snapshot(cfg);
    body.get("entry")
        .and_then(|e| e.get("credential"))
        .cloned()
        .ok_or_else(|| anyhow!("Auth broker refresh response has no entry.credential"))
}

fn access_of(credential: &Value) -> Option<&str> {
    credential
        .get("access")
        .and_then(Value::as_str)
        .filter(|a| !a.is_empty())
}

fn expires_of(credential: &Value) -> Option<i64> {
    credential
        .get("expires")
        .and_then(|e| e.as_i64().or_else(|| e.as_f64().map(|f| f as i64)))
}

fn is_fresh(credential: &Value, now: i64) -> bool {
    expires_of(credential).is_none_or(|e| e > now + REFRESH_SKEW_MS)
}

async fn broker_resolve(
    cfg: &BrokerConfig,
    provider: &str,
    force: Option<i64>,
) -> Result<Option<Credential>> {
    let now = now_ms() as i64;
    for entry in broker_snapshot(cfg)
        .await?
        .into_iter()
        .filter(|e| e.provider == provider)
    {
        if let Some((c, _)) = broker_entry_credential(cfg, entry, now, force).await? {
            return Ok(Some(c));
        }
    }
    Ok(None)
}
async fn broker_resolve_row(
    cfg: &BrokerConfig,
    provider: &str,
    force: Option<i64>,
) -> Result<Option<(Credential, Value)>> {
    let now = now_ms() as i64;
    for entry in broker_snapshot(cfg)
        .await?
        .into_iter()
        .filter(|e| e.provider == provider)
    {
        if let Some(pair) = broker_entry_credential(cfg, entry, now, force).await? {
            return Ok(Some(pair));
        }
    }
    Ok(None)
}

/// One broker snapshot entry reduced to a usable `Credential`, or `None` when it cannot serve
/// a request (a forced refresh that fails or re-mints the same token yields `None` when
/// `force` targets this entry — the caller has nothing new to retry with).
async fn broker_entry_credential(
    cfg: &BrokerConfig,
    entry: SnapshotEntry,
    now: i64,
    force: Option<i64>,
) -> Result<Option<(Credential, Value)>> {
    match entry.credential.get("type").and_then(Value::as_str) {
        Some("api_key") => Ok(entry
            .credential
            .get("key")
            .and_then(Value::as_str)
            .filter(|k| !k.is_empty())
            .map(|key| {
                (
                    Credential {
                        key: key.to_string(),
                        id: entry.id,
                    },
                    entry.credential.clone(),
                )
            })),
        Some("oauth") => {
            let forced = force == Some(entry.id);
            let mut credential = entry.credential.clone();
            if forced || !is_fresh(&credential, now) {
                match broker_refresh(cfg, entry.id, forced).await {
                    Ok(refreshed) => credential = refreshed,
                    Err(e) if !forced && expires_of(&credential).is_none_or(|x| x > now) => {
                        tracing::warn!(
                            "auth broker refresh of credential {} failed: {e:#}",
                            entry.id
                        );
                    }
                    Err(e) => {
                        tracing::warn!(
                            "auth broker refresh of credential {} failed: {e:#}",
                            entry.id
                        );
                        return Ok(None);
                    }
                }
            }
            let Some(access) = access_of(&credential) else {
                return Ok(None);
            };
            if expires_of(&credential).is_some_and(|x| x <= now) {
                return Ok(None);
            }
            Ok(Some((
                Credential {
                    key: access.to_string(),
                    id: entry.id,
                },
                credential,
            )))
        }
        _ => Ok(None),
    }
}

/// Every usable credential of `provider` from the broker snapshot: the first pick
/// ([`broker_resolve`]'s ordering) plus every other entry that yields a bearer.
async fn broker_pool(cfg: &BrokerConfig, provider: &str) -> Result<Option<CredentialPool>> {
    let now = now_ms() as i64;
    let mut usable: Vec<Credential> = Vec::new();
    for entry in broker_snapshot(cfg)
        .await?
        .into_iter()
        .filter(|e| e.provider == provider)
    {
        if let Some((c, _)) = broker_entry_credential(cfg, entry, now, None).await? {
            if usable.iter().all(|u| u.id != c.id) {
                usable.push(c);
            }
        }
    }
    let Some(first) = usable.first().cloned() else {
        return Ok(None);
    };
    Ok(Some(CredentialPool {
        alternates: usable.into_iter().skip(1).collect(),
        current: first,
    }))
}

// ------------------------------------------------------------ local store

struct Row {
    id: i64,
    kind: String,
    data: String,
}

fn open_rw(db: &Path) -> Result<rusqlite::Connection> {
    let conn = rusqlite::Connection::open_with_flags(
        db,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_WRITE | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )?;
    conn.busy_timeout(Duration::from_secs(5))?;
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS auth_credential_refresh_leases (
            credential_id INTEGER PRIMARY KEY,
            owner TEXT NOT NULL,
            expires_at_ms INTEGER NOT NULL,
            updated_at INTEGER NOT NULL DEFAULT (CAST(strftime('%s','now') AS INTEGER))
        );",
    )?;
    Ok(conn)
}

fn read_rows(db: &Path, provider: &str) -> Result<Vec<Row>> {
    let conn = rusqlite::Connection::open_with_flags(
        db,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )?;
    conn.busy_timeout(Duration::from_secs(5))?;
    let mut stmt = conn.prepare(
        "SELECT credential_type, data, id FROM auth_credentials WHERE provider = ?1 AND disabled_cause IS NULL ORDER BY id",
    )?;
    let rows = stmt
        .query_map([provider], |r| {
            Ok(Row {
                kind: r.get(0)?,
                data: r.get(1)?,
                id: r.get(2)?,
            })
        })?
        .filter_map(std::result::Result::ok)
        .collect();
    Ok(rows)
}

/// What a provider token endpoint handed back for a refresh grant.
struct Refreshed {
    access: String,
    refresh: Option<String>,
    expires: i64,
    account_id: Option<String>,
    enterprise_url: Option<String>,
}

/// Tokens this process minted, keyed by (credential store path, row id): row ids are only unique
/// within one `agent.db`.
type MintKey = (String, i64);
static MINTS: LazyLock<Mutex<HashMap<MintKey, (String, Instant)>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

fn remember_mint(db: &str, id: i64, access: &str) {
    let mut mints = MINTS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    mints.retain(|_, (_, at)| at.elapsed() < REMINT_COOLDOWN);
    mints.insert((db.to_string(), id), (access.to_string(), Instant::now()));
}

fn recent_mint(db: &str, id: i64, access: &str) -> bool {
    MINTS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get(&(db.to_string(), id))
        .is_some_and(|(a, at)| a == access && at.elapsed() < REMINT_COOLDOWN)
}

fn db_key(db: &Path) -> String {
    db.to_string_lossy().into_owned()
}

fn form_encode(pairs: &[(&str, &str)]) -> String {
    let mut ser = reqwest::Url::parse("http://x/").expect("static url");
    ser.query_pairs_mut().extend_pairs(pairs);
    ser.query().unwrap_or_default().to_string()
}

/// TS `refresh {}` of `auth "kimi-code"`: form refresh grant at `{base}/api/oauth/token`.
async fn refresh_kimi(hosts: &Hosts, refresh_token: &str) -> Result<Refreshed> {
    const CLIENT_ID: &str = "17e5f671-d194-4dfb-9706-5516cb48c098";
    let base = hosts.kimi.as_deref().unwrap_or("https://auth.kimi.com");
    let url = format!("{base}/api/oauth/token");
    let body = form_encode(&[
        ("grant_type", "refresh_token"),
        ("client_id", CLIENT_ID),
        ("refresh_token", refresh_token),
    ]);
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()?;
    let mut request = client
        .post(&url)
        .header("Content-Type", "application/x-www-form-urlencoded");
    for (k, v) in pa_ai::kimi_common_headers() {
        request = request.header(k, v);
    }
    let response = request
        .body(body)
        .send()
        .await
        .map_err(|e| anyhow!("kimi-code request to {url} failed: {e}"))?;
    let status = response.status();
    let text = response.text().await.unwrap_or_default();
    if !status.is_success() {
        bail!(
            "kimi-code token refresh failed: {} {}",
            status.as_u16(),
            text.chars().take(500).collect::<String>()
        );
    }
    let body: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
    let access = body
        .get("access_token")
        .and_then(Value::as_str)
        .filter(|a| !a.is_empty())
        .ok_or_else(|| {
            anyhow!(
                "kimi-code token response missing access token: {}",
                text.chars().take(500).collect::<String>()
            )
        })?;
    let expires_in = body
        .get("expires_in")
        .and_then(Value::as_f64)
        .ok_or_else(|| anyhow!("kimi-code token response missing expires_in"))?;
    Ok(Refreshed {
        access: access.to_string(),
        refresh: body
            .get("refresh_token")
            .and_then(Value::as_str)
            .filter(|r| !r.is_empty())
            .map(str::to_string),
        expires: now_ms() as i64 + (expires_in * 1000.0) as i64 - 300_000,
        account_id: None,
        enterprise_url: None,
    })
}

/// TS `refresh { require "projectId" }` of `auth "google-gemini-cli"` / `"google-antigravity"`
/// (pi-catalog `rules/auth/*.kdl`): form refresh grant at Google's token endpoint with the
/// provider's installed-app client; `projectId` (and the rest of the row) survives via
/// [`merge_refreshed`].
async fn refresh_google(provider: &str, data: &Value, refresh_token: &str) -> Result<Refreshed> {
    use base64::Engine as _;
    const TOKEN_URL: &str = "https://oauth2.googleapis.com/token";
    let (client_id, client_secret) = match provider {
        "google-gemini-cli" => (
            "NjgxMjU1ODA5Mzk1LW9vOGZ0Mm9wcmRybnA5ZTNhcWY2YXYzaG1kaWIxMzVqLmFwcHMuZ29vZ2xldXNlcmNvbnRlbnQuY29t",
            "R09DU1BYLTR1SGdNUG0tMW83U2stZ2VWNkN1NWNsWEZzeGw=",
        ),
        _ => (
            "MTA3MTAwNjA2MDU5MS10bWhzc2luMmgyMWxjcmUyMzV2dG9sb2poNGc0MDNlcC5hcHBzLmdvb2dsZXVzZXJjb250ZW50LmNvbQ==",
            "R09DU1BYLUs1OEZXUjQ4NkxkTEoxbUxCOHNYQzR6NnFEQWY=",
        ),
    };
    if data
        .get("projectId")
        .and_then(Value::as_str)
        .is_none_or(str::is_empty)
    {
        bail!("{provider} credentials are missing projectId; sign in again");
    }
    let decode = |b: &str| -> Result<String> {
        Ok(String::from_utf8(
            base64::engine::general_purpose::STANDARD.decode(b)?,
        )?)
    };
    let (client_id, client_secret) = (decode(client_id)?, decode(client_secret)?);
    let body = form_encode(&[
        ("grant_type", "refresh_token"),
        ("client_id", &client_id),
        ("client_secret", &client_secret),
        ("refresh_token", refresh_token),
    ]);
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()?;
    let response = client
        .post(TOKEN_URL)
        .header("Content-Type", "application/x-www-form-urlencoded")
        .header("Accept", "application/json")
        .body(body)
        .send()
        .await
        .map_err(|e| anyhow!("{provider} request to {TOKEN_URL} failed: {e}"))?;
    let status = response.status();
    let text = response.text().await.unwrap_or_default();
    if !status.is_success() {
        bail!(
            "{provider} token refresh failed: {} {}",
            status.as_u16(),
            text.chars().take(500).collect::<String>()
        );
    }
    let body: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
    let access = body
        .get("access_token")
        .and_then(Value::as_str)
        .filter(|a| !a.is_empty())
        .ok_or_else(|| anyhow!("{provider} token response missing access token"))?;
    let expires_in = body
        .get("expires_in")
        .and_then(Value::as_f64)
        .ok_or_else(|| anyhow!("{provider} token response missing expires_in"))?;
    Ok(Refreshed {
        access: access.to_string(),
        refresh: body
            .get("refresh_token")
            .and_then(Value::as_str)
            .filter(|r| !r.is_empty())
            .map(str::to_string),
        expires: now_ms() as i64 + (expires_in * 1000.0) as i64 - 300_000,
        account_id: None,
        enterprise_url: None,
    })
}

/// The provider's refresh flow, or `None` when omp-rs has none for it (the stored token is then
/// used until it expires).
async fn refresh_flow(provider: &str, data: &Value, hosts: &Hosts) -> Option<Result<Refreshed>> {
    let refresh = data
        .get("refresh")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let http = pa_ai::oauth::ReqwestProviderHttp::new();
    match provider {
        "kimi-code" => Some(refresh_kimi(hosts, &refresh).await),
        "google-gemini-cli" | "google-antigravity" => {
            Some(refresh_google(provider, data, &refresh).await)
        }
        "anthropic" => Some(
            pa_ai::oauth::refresh_anthropic_token(&http, &refresh)
                .await
                .map(|c| Refreshed {
                    access: c.access,
                    refresh: Some(c.refresh),
                    expires: c.expires,
                    account_id: None,
                    enterprise_url: None,
                })
                .map_err(|e| anyhow!(e)),
        ),
        "openai-codex" => Some(
            pa_ai::oauth::refresh_openai_codex_token(
                &pa_ai::oauth::ReqwestCodexHttp::new(),
                &refresh,
            )
            .await
            .map(|c| Refreshed {
                access: c.access,
                refresh: Some(c.refresh),
                expires: c.expires,
                account_id: Some(c.account_id),
                enterprise_url: None,
            })
            .map_err(|e| anyhow!(e)),
        ),
        "xai-oauth" => Some(
            pa_ai::oauth::refresh_xai_token(&http, &refresh)
                .await
                .map(|c| Refreshed {
                    access: c.access,
                    refresh: Some(c.refresh),
                    expires: c.expires,
                    account_id: None,
                    enterprise_url: None,
                })
                .map_err(|e| anyhow!(e)),
        ),
        "github-copilot" => {
            let enterprise = data
                .get("enterpriseUrl")
                .and_then(Value::as_str)
                .map(str::to_string);
            Some(
                pa_ai::oauth::refresh_github_copilot_token(&http, &refresh, enterprise.as_deref())
                    .await
                    .map(|c| Refreshed {
                        access: c.access,
                        refresh: Some(c.refresh),
                        expires: c.expires,
                        account_id: None,
                        enterprise_url: c.enterprise_url,
                    })
                    .map_err(|e| anyhow!(e)),
            )
        }
        _ => None,
    }
}

/// TS `mergeRefreshedCredential`: refreshed bytes over the stored row, key order preserved.
fn merge_refreshed(mut data: Map<String, Value>, refreshed: &Refreshed) -> Map<String, Value> {
    data.insert("access".into(), Value::String(refreshed.access.clone()));
    if let Some(refresh) = &refreshed.refresh {
        data.insert("refresh".into(), Value::String(refresh.clone()));
    }
    data.insert("expires".into(), Value::from(refreshed.expires));
    if let Some(id) = &refreshed.account_id {
        data.insert("accountId".into(), Value::String(id.clone()));
    }
    if let Some(url) = &refreshed.enterprise_url {
        data.insert("enterpriseUrl".into(), Value::String(url.clone()));
    }
    data
}

fn try_lease(conn: &rusqlite::Connection, id: i64, owner: &str) -> Result<bool> {
    let now = now_ms() as i64;
    conn.execute(
        "INSERT INTO auth_credential_refresh_leases (credential_id, owner, expires_at_ms, updated_at)
         VALUES (?1, ?2, ?3, CAST(strftime('%s','now') AS INTEGER))
         ON CONFLICT(credential_id) DO UPDATE SET owner = excluded.owner, expires_at_ms = excluded.expires_at_ms,
             updated_at = excluded.updated_at
         WHERE auth_credential_refresh_leases.expires_at_ms <= ?4",
        rusqlite::params![id, owner, now + LEASE_TTL_MS, now],
    )?;
    let holder: Option<String> = conn
        .query_row(
            "SELECT owner FROM auth_credential_refresh_leases WHERE credential_id = ?1",
            [id],
            |r| r.get(0),
        )
        .ok();
    Ok(holder.as_deref() == Some(owner))
}

fn row_state(conn: &rusqlite::Connection, id: i64) -> Option<(String, Option<String>)> {
    conn.query_row(
        "SELECT data, disabled_cause FROM auth_credentials WHERE id = ?1",
        [id],
        |r| Ok((r.get(0)?, r.get(1)?)),
    )
    .ok()
}

fn owner_tag() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    format!("omp-rs-{}-{nanos}", std::process::id())
}

/// Refresh one OAuth row under its lease (TS `OAuthRefresher.refreshStored`). `Ok(None)` when
/// omp-rs has no refresh flow for the provider.
async fn refresh_local_row(
    db: &Path,
    provider: &str,
    row: &Row,
    forced: bool,
    hosts: &Hosts,
) -> Result<Option<Credential>> {
    let conn = open_rw(db)?;
    let owner = owner_tag();
    let deadline = Instant::now() + LEASE_WAIT;
    let mut leased = false;
    while Instant::now() < deadline {
        if try_lease(&conn, row.id, &owner)? {
            leased = true;
            break;
        }
        tokio::time::sleep(LEASE_POLL).await;
        // A peer finished the refresh while we waited: adopt its usable result.
        if let Some((data, None)) = row_state(&conn, row.id) {
            let v: Value = serde_json::from_str(&data).unwrap_or(Value::Null);
            if data != row.data && is_fresh(&v, now_ms() as i64) {
                return Ok(access_of(&v).map(|a| Credential {
                    key: a.to_string(),
                    id: row.id,
                }));
            }
        }
    }
    if !leased {
        bail!(
            "timed out waiting for the OAuth refresh lease of credential {}",
            row.id
        );
    }
    let result = refresh_leased(&conn, db, provider, row, forced, hosts, &owner).await;
    let _ = conn.execute(
        "DELETE FROM auth_credential_refresh_leases WHERE credential_id = ?1 AND owner = ?2",
        rusqlite::params![row.id, owner],
    );
    result
}

/// Outcome of [`force_refresh_row`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForceRefresh {
    /// The row was re-minted (by us, or by a peer while we waited for the lease).
    Refreshed,
    /// `reuse_recent` and this process minted the stored token within the remint cooldown.
    ReusedRecent,
    /// omp-rs has no refresh flow for the row's provider.
    Unsupported,
}

/// Auth-broker server side (TS `OAuthRefresher.refreshById`): force-refresh one active OAuth row
/// under its refresh lease, whatever its expiry. `reuse_recent` is the `?reason=auth-recovery`
/// `reuseRecentMint` option. Errors carry the TS messages (`No credential with id=N` → 404).
///
/// # Errors
/// Unknown or disabled id, a non-OAuth row, a lease timeout, or the provider refresh failure.
pub async fn force_refresh_row(db: &Path, id: i64, reuse_recent: bool) -> Result<ForceRefresh> {
    let (provider, row) = {
        let conn = open_rw(db)?;
        conn.query_row(
            "SELECT provider, credential_type, data FROM auth_credentials WHERE id = ?1 AND disabled_cause IS NULL",
            [id],
            |r| Ok((r.get::<_, String>(0)?, Row { kind: r.get(1)?, data: r.get(2)?, id })),
        )
        .map_err(|_| anyhow!("No credential with id={id}"))?
    };
    if row.kind != "oauth" {
        bail!("Credential id={id} is not an OAuth credential");
    }
    if reuse_recent {
        let data: Value = serde_json::from_str(&row.data).unwrap_or(Value::Null);
        if access_of(&data).is_some_and(|a| recent_mint(&db_key(db), id, a)) {
            return Ok(ForceRefresh::ReusedRecent);
        }
    }
    let hosts = Hosts::from_env();
    Ok(
        match refresh_local_row(db, &provider, &row, true, &hosts).await? {
            Some(_) => ForceRefresh::Refreshed,
            None => ForceRefresh::Unsupported,
        },
    )
}

async fn refresh_leased(
    conn: &rusqlite::Connection,
    db: &Path,
    provider: &str,
    row: &Row,
    forced: bool,
    hosts: &Hosts,
    owner: &str,
) -> Result<Option<Credential>> {
    let Some((current_data, None)) = row_state(conn, row.id) else {
        return Ok(None);
    };
    let current: Value = serde_json::from_str(&current_data)?;
    let now = now_ms() as i64;
    // Re-check under the lease: a peer may have rotated the row since we read it.
    if is_fresh(&current, now) && (!forced || current_data != row.data) {
        return Ok(access_of(&current).map(|a| Credential {
            key: a.to_string(),
            id: row.id,
        }));
    }
    if current
        .get("refresh")
        .and_then(Value::as_str)
        .is_none_or(|r| r.is_empty() || r == REMOTE_REFRESH_SENTINEL)
    {
        bail!("credential {} has no refresh token", row.id);
    }
    let Some(refreshed) = refresh_flow(provider, &current, hosts).await else {
        return Ok(None);
    };
    let refreshed = refreshed?;
    let Value::Object(map) = current else {
        bail!("credential {} is not an object", row.id)
    };
    let merged = serde_json::to_string(&Value::Object(merge_refreshed(map, &refreshed)))?;
    let changed = conn.execute(
        "UPDATE auth_credentials SET data = ?1, updated_at = CAST(strftime('%s','now') AS INTEGER)
         WHERE id = ?2 AND data = ?3 AND disabled_cause IS NULL
           AND EXISTS (SELECT 1 FROM auth_credential_refresh_leases WHERE credential_id = ?2 AND owner = ?4 AND expires_at_ms > ?5)",
        rusqlite::params![merged, row.id, current_data, owner, now_ms() as i64],
    )?;
    if changed == 0 {
        // CAS lost: the stored row moved under us; the stored copy wins when usable.
        if let Some((data, None)) = row_state(conn, row.id) {
            let v: Value = serde_json::from_str(&data).unwrap_or(Value::Null);
            if is_fresh(&v, now_ms() as i64) {
                return Ok(access_of(&v).map(|a| Credential {
                    key: a.to_string(),
                    id: row.id,
                }));
            }
        }
        bail!("credential {} changed during refresh", row.id);
    }
    remember_mint(&db_key(db), row.id, &refreshed.access);
    Ok(Some(Credential {
        key: refreshed.access,
        id: row.id,
    }))
}

/// The first usable credential of `rows`, refreshing an expired OAuth row. Iterates rows in
/// store order and stops at the first usable one (provider order, TS `localResolve`).
async fn local_resolve(
    db: &Path,
    provider: &str,
    force: Option<i64>,
    hosts: &Hosts,
) -> Result<Option<Credential>> {
    let rows = match read_rows(db, provider) {
        Ok(rows) => rows,
        Err(_) => return Ok(None),
    };
    let now = now_ms() as i64;
    for row in rows {
        if let Some((c, _)) = local_row_credential(db, provider, &row, now, force, hosts).await {
            return Ok(Some(c));
        }
    }
    Ok(None)
}
async fn local_resolve_row(
    db: &Path,
    provider: &str,
    force: Option<i64>,
    hosts: &Hosts,
) -> Result<Option<(Credential, Value)>> {
    let rows = match read_rows(db, provider) {
        Ok(rows) => rows,
        Err(_) => return Ok(None),
    };
    let now = now_ms() as i64;
    for row in rows {
        if let Some(pair) = local_row_credential(db, provider, &row, now, force, hosts).await {
            return Ok(Some(pair));
        }
    }
    Ok(None)
}

/// One credential row reduced to a usable `Credential`, or `None` when the row cannot serve a
/// request. `force` targets one row (auth recovery): it alone is consulted, and a re-minted
/// bearer is not re-used (`reuseRecentMint`).
async fn local_row_credential(
    db: &Path,
    provider: &str,
    row: &Row,
    now: i64,
    force: Option<i64>,
    hosts: &Hosts,
) -> Option<(Credential, Value)> {
    let data: Value = serde_json::from_str(&row.data).ok()?;
    match row.kind.as_str() {
        "api_key" => data
            .get("key")
            .and_then(Value::as_str)
            .filter(|k| !k.is_empty())
            .map(|k| {
                (
                    Credential {
                        key: k.to_string(),
                        id: row.id,
                    },
                    data.clone(),
                )
            }),
        "oauth" => {
            let access = access_of(&data)?;
            let forced = force == Some(row.id);
            if forced && recent_mint(&db_key(db), row.id, access) {
                return Some((
                    Credential {
                        key: access.to_string(),
                        id: row.id,
                    },
                    data,
                ));
            }
            if !forced && is_fresh(&data, now) {
                return Some((
                    Credential {
                        key: access.to_string(),
                        id: row.id,
                    },
                    data,
                ));
            }
            match refresh_local_row(db, provider, row, forced, hosts).await {
                Ok(Some(c)) => Some((c, data)),
                Ok(None) | Err(_) if forced => None,
                Ok(None) => None,
                Err(e) => {
                    tracing::warn!(
                        "OAuth refresh of {provider} credential {} failed: {e:#}",
                        row.id
                    );
                    if expires_of(&data).is_none_or(|x| x > now) {
                        Some((
                            Credential {
                                key: access.to_string(),
                                id: row.id,
                            },
                            data,
                        ))
                    } else {
                        None
                    }
                }
            }
        }
        _ => None,
    }
}

/// Every usable credential of `provider`: the first pick ([`local_resolve`]'s ordering) plus
/// every other row that yields a bearer. Stale OAuth rows are refreshed on the way (best
/// effort); rows that fail to resolve are left out rather than failing the pool.
async fn local_pool(
    db: &Path,
    provider: &str,
    rows: Vec<Row>,
    hosts: &Hosts,
) -> Result<Option<CredentialPool>> {
    let now = now_ms() as i64;
    let mut usable: Vec<Credential> = Vec::new();
    for row in &rows {
        if let Some((c, _)) = local_row_credential(db, provider, row, now, None, hosts).await {
            if usable.iter().all(|u| u.id != c.id) {
                usable.push(c);
            }
        }
    }
    let Some(first) = usable.first().cloned() else {
        return Ok(None);
    };
    Ok(Some(CredentialPool {
        alternates: usable.into_iter().skip(1).collect(),
        current: first,
    }))
}

// ------------------------------------------------------------ entry points

/// Run `fut` to completion on a private runtime thread: callable from sync code and from inside
/// a tokio worker alike.
fn block_on<M, F>(make: M) -> F::Output
where
    M: FnOnce() -> F + Send,
    F: Future,
    F::Output: Send,
{
    std::thread::scope(|s| {
        s.spawn(|| {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("auth runtime")
                .block_on(make())
        })
        .join()
        .expect("auth thread panicked")
    })
}

async fn resolve_with(
    dir: &AgentDir,
    provider: &str,
    broker: Option<&BrokerConfig>,
    force: Option<i64>,
    hosts: &Hosts,
) -> Result<Option<Credential>> {
    match broker {
        Some(cfg) => broker_resolve(cfg, provider, force).await,
        None => local_resolve(&dir.0.join("agent.db"), provider, force, hosts).await,
    }
}
async fn resolve_with_row(
    dir: &AgentDir,
    provider: &str,
    broker: Option<&BrokerConfig>,
    force: Option<i64>,
    hosts: &Hosts,
) -> Result<Option<(Credential, Value)>> {
    match broker {
        Some(cfg) => broker_resolve_row(cfg, provider, force).await,
        None => local_resolve_row(&dir.0.join("agent.db"), provider, force, hosts).await,
    }
}

/// One specific `auth_credentials` row / broker snapshot entry by id: what a pinned
/// (`credential_id`) target re-resolves through, so a rotated account does not drift back to
/// the store's first row on the next request. OAuth rows refresh on expiry like [`resolve_with`].
async fn resolve_by_id_with(
    dir: &AgentDir,
    provider: &str,
    id: i64,
    broker: Option<&BrokerConfig>,
    hosts: &Hosts,
) -> Result<Option<Credential>> {
    match broker {
        Some(cfg) => {
            let now = now_ms() as i64;
            for entry in broker_snapshot(cfg)
                .await?
                .into_iter()
                .filter(|e| e.provider == provider && e.id == id)
            {
                if let Some((c, _)) = broker_entry_credential(cfg, entry, now, None).await? {
                    return Ok(Some(c));
                }
            }
            Ok(None)
        }
        None => {
            let db = dir.0.join("agent.db");
            let row = read_row_by_id(&db, provider, id)?;
            let Some(row) = row else { return Ok(None) };
            let now = now_ms() as i64;
            Ok(local_row_credential(&db, provider, &row, now, None, hosts)
                .await
                .map(|(c, _)| c))
        }
    }
}

fn read_row_by_id(db: &Path, provider: &str, id: i64) -> Result<Option<Row>> {
    let conn = rusqlite::Connection::open_with_flags(
        db,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )?;
    conn.busy_timeout(Duration::from_secs(5))?;
    let mut stmt = conn.prepare(
        "SELECT credential_type, data FROM auth_credentials WHERE id = ?1 AND provider = ?2 AND disabled_cause IS NULL",
    )?;
    let row = stmt
        .query_row(rusqlite::params![id, provider], |r| {
            Ok(Row {
                id,
                kind: r.get(0)?,
                data: r.get(1)?,
            })
        })
        .optional()?;
    Ok(row)
}

fn resolve_blocking(dir: &AgentDir, provider: &str, force: Option<i64>) -> Option<Credential> {
    let broker = match resolve_broker_config(dir, &process_env) {
        Ok(broker) => broker,
        Err(e) => {
            // A configured broker never silently falls back to the local store (TS).
            tracing::warn!("auth broker unavailable: {e:#}");
            return None;
        }
    };
    let hosts = Hosts::from_env();
    match block_on(|| resolve_with(dir, provider, broker.as_ref(), force, &hosts)) {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!("credential lookup for {provider} failed: {e:#}");
            None
        }
    }
}

/// The current bearer for `provider` from the credential store (broker when configured, else
/// `agent.db`), refreshing an expired OAuth token first.
#[must_use]
pub fn credential(dir: &AgentDir, provider: &str) -> Option<Credential> {
    resolve_blocking(dir, provider, None)
}

/// The bearer of one specific credential row (`broker snapshot entry`), refreshing an expired
/// OAuth token first. `None` when the row is gone, disabled or unusable.
#[must_use]
pub fn credential_by_id(dir: &AgentDir, provider: &str, id: i64) -> Option<Credential> {
    let broker = match resolve_broker_config(dir, &process_env) {
        Ok(broker) => broker,
        Err(e) => {
            tracing::warn!("auth broker unavailable: {e:#}");
            return None;
        }
    };
    let hosts = Hosts::from_env();
    match block_on(|| resolve_by_id_with(dir, provider, id, broker.as_ref(), &hosts)) {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!("credential lookup for {provider}/{id} failed: {e:#}");
            None
        }
    }
}

/// Local-store lookup only (no broker): the seam tests use so the process env cannot redirect them.
#[cfg(test)]
pub(crate) fn local_credential(db: &Path, provider: &str) -> Option<Credential> {
    let hosts = Hosts::default();
    block_on(|| local_resolve(db, provider, None, &hosts))
        .ok()
        .flatten()
}

/// [`credential`] plus every other usable credential of the provider — the account-rotation
/// pool. Without a broker this is one `read_rows` sweep; with a broker it is `current` plus
/// the other snapshot credentials of the same provider.
#[must_use]
pub fn credential_pool(dir: &AgentDir, provider: &str) -> Option<CredentialPool> {
    let broker = match resolve_broker_config(dir, &process_env) {
        Ok(broker) => broker,
        Err(e) => {
            tracing::warn!("auth broker unavailable: {e:#}");
            return None;
        }
    };
    match broker {
        Some(cfg) => block_on(|| broker_pool(&cfg, provider)).ok().flatten(),
        None => {
            let hosts = Hosts::from_env();
            let db = dir.0.join("agent.db");
            let rows = read_rows(&db, provider).unwrap_or_default();
            block_on(|| local_pool(&db, provider, rows, &hosts))
                .ok()
                .flatten()
        }
    }
}

/// [`credential_pool`] off the async executor.
pub async fn credential_pool_async(dir: AgentDir, provider: String) -> Option<CredentialPool> {
    tokio::task::spawn_blocking(move || credential_pool(&dir, &provider))
        .await
        .ok()
        .flatten()
}

/// The TS `getOAuthApiKey` structured credential format for the Cloud Code Assist providers
/// (`google-gemini-cli` and `google-antigravity`, both `api-key-format "structured"`): the raw
/// `auth_credentials` row payload serialized as `{"apiEndpoint","token","enterpriseUrl",
/// "projectId","refreshToken","expiresAt","email","accountId"}`. The gemini-cli provider parses
/// this JSON to recover the bearer and the Cloud project (`_Kn` in the TS bundle); the bare
/// access token alone would send `project: ""` and fail every request. `None` when the row is
/// not an OAuth payload with a project id.
#[must_use]
pub fn structured_key(dir: &AgentDir, provider: &str) -> Option<String> {
    if !matches!(provider, "google-gemini-cli" | "google-antigravity") {
        return None;
    }
    structured_key_by_id(dir, provider, credential(dir, provider)?.id)
}

/// [`structured_key`] for one pinned `auth_credentials` row (the per-request refresh path
/// re-resolves a stored target through its row, not the provider-wide pick).
#[must_use]
pub fn structured_key_by_id(dir: &AgentDir, provider: &str, id: i64) -> Option<String> {
    if !matches!(provider, "google-gemini-cli" | "google-antigravity") {
        return None;
    }
    let row = read_row_by_id(&dir.0.join("agent.db"), provider, id).ok()??;
    let data: Value = serde_json::from_str(&row.data).ok()?;
    let token = data.get("access").and_then(Value::as_str)?.to_string();
    let project_id = data
        .get("projectId")
        .or_else(|| data.get("project_id"))
        .and_then(Value::as_str)?
        .to_string();
    let pick = |key: &str| {
        data.get(key)
            .and_then(Value::as_str)
            .map(std::string::ToString::to_string)
    };
    Some(
        serde_json::to_string(&serde_json::json!({
            "apiEndpoint": pick("apiEndpoint"),
            "token": token,
            "enterpriseUrl": pick("enterpriseUrl"),
            "projectId": project_id,
            "refreshToken": pick("refresh"),
            "expiresAt": data.get("expires").and_then(|e| e.as_i64().or_else(|| e.as_f64().map(|f| f as i64))),
            "email": pick("email"),
            "accountId": pick("accountId"),
        }))
        .ok()?,
    )
}

/// After a provider 401 for `failed`: force-refresh that credential and return the replacement,
/// or `None` when there is no different bearer to retry with (TS `auth-retry.ts`).
#[must_use]
pub fn recover(dir: &AgentDir, provider: &str, failed: &Credential) -> Option<Credential> {
    resolve_blocking(dir, provider, Some(failed.id)).filter(|c| c.key != failed.key)
}

/// [`credential`] off the async executor.
pub async fn credential_async(dir: AgentDir, provider: String) -> Option<Credential> {
    tokio::task::spawn_blocking(move || credential(&dir, &provider))
        .await
        .ok()
        .flatten()
}

/// The resolved credential plus the parsed `auth_credentials.data` JSON of the row it came from
/// (orgId/projectId/accountId/email/expires live there; usage probes need them). Refreshes an
/// expired oauth row like [`credential_async`].
pub async fn credential_row_async(dir: AgentDir, provider: String) -> Option<(Credential, Value)> {
    tokio::task::spawn_blocking(move || {
        let broker = match resolve_broker_config(&dir, &process_env) {
            Ok(broker) => broker,
            Err(e) => {
                tracing::warn!("auth broker unavailable: {e:#}");
                return None;
            }
        };
        let hosts = Hosts::from_env();
        block_on(|| resolve_with_row(&dir, &provider, broker.as_ref(), None, &hosts))
            .ok()
            .flatten()
    })
    .await
    .ok()
    .flatten()
}

/// [`credential_by_id`] off the async executor.
pub async fn credential_by_id_async(
    dir: AgentDir,
    provider: String,
    id: i64,
) -> Option<Credential> {
    tokio::task::spawn_blocking(move || credential_by_id(&dir, &provider, id))
        .await
        .ok()
        .flatten()
}

/// [`recover`] off the async executor.
pub async fn recover_async(
    dir: AgentDir,
    provider: String,
    failed: Credential,
) -> Option<Credential> {
    tokio::task::spawn_blocking(move || recover(&dir, &provider, &failed))
        .await
        .ok()
        .flatten()
}

#[cfg(test)]
mod tests;
