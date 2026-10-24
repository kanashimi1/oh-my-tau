//! Credential persistence for `login`/`logout`: the TS `SqliteAuthCredentialStore` write paths
//! (`upsertAuthCredential`, `deleteAuthCredential(s)`, identity keys, superseded-tombstone purge)
//! against `agent.db`, and the auth-broker equivalents (`POST /v1/credential`,
//! `POST /v1/credential/:id/disable`). Rows written here are byte-compatible with TS: same
//! columns, `data` JSON key order, `identity_key` and `disabled_cause` handling.

use std::path::Path;
use std::time::Duration;

use anyhow::{anyhow, bail, Context as _, Result};
use base64::Engine as _;
use rusqlite::{params, Connection, OpenFlags, TransactionBehavior};
use serde::Deserialize;
use serde_json::{json, Map, Value};

use crate::auth::BrokerConfig;

/// TS `AUTH_SCHEMA_VERSION`.
const AUTH_SCHEMA_VERSION: i64 = 8;
const NOW_EPOCH: &str = "CAST(strftime('%s','now') AS INTEGER)";
const BROKER_TIMEOUT: Duration = Duration::from_secs(10);

/// What a login stores.
#[derive(Debug, Clone, PartialEq)]
pub enum Credential {
    /// `{"key":…,"source":"login"}` (TS `storeLoginApiKey`).
    ApiKey(String),
    /// The OAuth `data` object without `type`, key order as TS serializes it.
    OAuth(Map<String, Value>),
}

/// One active `auth_credentials` row.
#[derive(Debug, Clone, PartialEq)]
pub struct StoredRow {
    pub id: i64,
    pub provider: String,
    pub credential: Credential,
}

impl StoredRow {
    fn from_parts(id: i64, provider: String, kind: &str, data: &str) -> Option<Self> {
        let parsed: Value = serde_json::from_str(data).ok()?;
        let Value::Object(map) = parsed else {
            return None;
        };
        let credential = match kind {
            "api_key" => Credential::ApiKey(map.get("key")?.as_str()?.to_string()),
            "oauth" => Credential::OAuth(map),
            _ => return None,
        };
        Some(Self {
            id,
            provider,
            credential,
        })
    }

    fn str_field(&self, key: &str) -> Option<&str> {
        match &self.credential {
            Credential::OAuth(map) => map
                .get(key)
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty()),
            Credential::ApiKey(_) => None,
        }
    }

    /// TS `oauthLabel` (slash-commands/helpers/logout.ts): what a logout row is called.
    #[must_use]
    pub fn label(&self) -> String {
        if matches!(self.credential, Credential::ApiKey(_)) {
            return format!("API key #{}", self.id);
        }
        let base = ["email", "accountId", "projectId", "enterpriseUrl"]
            .iter()
            .find_map(|k| self.str_field(k))
            .map_or_else(|| format!("OAuth credential #{}", self.id), str::to_string);
        match ["orgName", "orgId"].iter().find_map(|k| self.str_field(k)) {
            Some(org) if org != base => format!("{base} ({org})"),
            _ => base,
        }
    }
}

fn norm_account(v: Option<&str>) -> Option<String> {
    v.map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

fn norm_email(v: Option<&str>) -> Option<String> {
    v.map(|s| s.trim().to_lowercase()).filter(|s| !s.is_empty())
}

/// Decoded JWT payload object, if `token` is a three-part JWT.
#[must_use]
pub fn jwt_payload(token: &str) -> Option<Map<String, Value>> {
    let mut parts = token.split('.');
    let (_, payload, _, None) = (parts.next()?, parts.next()?, parts.next()?, parts.next()) else {
        return None;
    };
    let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(payload.trim_end_matches('='))
        .ok()?;
    match serde_json::from_slice(&bytes).ok()? {
        Value::Object(map) => Some(map),
        _ => None,
    }
}

/// TS `extractOAuthTokenIdentifiers`.
fn token_identifiers(token: Option<&str>) -> Vec<String> {
    let Some(payload) = token.and_then(jwt_payload) else {
        return Vec::new();
    };
    let mut out: Vec<String> = Vec::new();
    let mut push = |id: String| {
        if !out.contains(&id) {
            out.push(id);
        }
    };
    if let Some(email) = norm_email(payload.get("email").and_then(Value::as_str)) {
        push(format!("email:{email}"));
    }
    if let Some(email) = norm_email(
        payload
            .get("https://api.openai.com/profile")
            .and_then(Value::as_object)
            .and_then(|p| p.get("email"))
            .and_then(Value::as_str),
    ) {
        push(format!("email:{email}"));
    }
    let claim = payload
        .get("https://api.openai.com/auth")
        .and_then(Value::as_object)
        .and_then(|a| a.get("chatgpt_account_id"))
        .and_then(Value::as_str);
    let account = ["account_id", "accountId", "user_id", "sub"]
        .iter()
        .find_map(|k| payload.get(*k).and_then(Value::as_str))
        .or(claim);
    if let Some(account) = norm_account(account) {
        push(format!("account:{account}"));
    }
    out
}

/// TS `extractOAuthCredentialIdentifiers`.
fn oauth_identifiers(data: &Map<String, Value>) -> Vec<String> {
    let get = |k: &str| data.get(k).and_then(Value::as_str);
    let mut out: Vec<String> = Vec::new();
    let mut push = |id: String| {
        if !out.contains(&id) {
            out.push(id);
        }
    };
    if let Some(v) = norm_account(get("accountId")) {
        push(format!("account:{v}"));
    }
    if let Some(v) = norm_email(get("email")) {
        push(format!("email:{v}"));
    }
    if let Some(v) = norm_account(get("projectId")) {
        push(format!("project:{v}"));
    }
    if let Some(v) = norm_account(get("orgId")) {
        push(format!("org:{v}"));
    }
    for id in token_identifiers(get("access"))
        .into_iter()
        .chain(token_identifiers(get("refresh")))
    {
        push(id);
    }
    out
}

/// Providers declaring `org-scoped-identity #true` in the TS auth rules.
fn org_scoped(provider: &str) -> bool {
    matches!(provider, "anthropic" | "openai-codex" | "factory-droid")
}

/// TS `resolveProviderCredentialIdentityKey`.
fn provider_identity_key(provider: &str, ids: &[String]) -> Option<String> {
    let find = |prefix: &str| ids.iter().find(|i| i.starts_with(prefix));
    if org_scoped(provider) {
        let base = find("email:")
            .or_else(|| find("account:"))
            .or_else(|| find("project:"));
        let org = find("org:");
        return match (base, org) {
            (Some(base), Some(org)) => Some(format!("{base}|{org}")),
            (Some(base), None) => Some(base.clone()),
            (None, org) => org.cloned(),
        };
    }
    find("account:")
        .or_else(|| find("email:"))
        .or_else(|| find("project:"))
        .cloned()
}

/// TS `resolveCredentialIdentityKey`.
#[must_use]
pub fn identity_key(provider: &str, credential: &Credential) -> Option<String> {
    match credential {
        Credential::ApiKey(_) => None,
        Credential::OAuth(data) => provider_identity_key(provider, &oauth_identifiers(data)),
    }
}

/// `data` column for a credential (TS `serializeCredential`).
fn serialize(credential: &Credential) -> String {
    match credential {
        Credential::ApiKey(key) => json!({"key": key, "source": "login"}).to_string(),
        Credential::OAuth(data) => Value::Object(data.clone()).to_string(),
    }
}

fn kind_of(credential: &Credential) -> &'static str {
    match credential {
        Credential::ApiKey(_) => "api_key",
        Credential::OAuth(_) => "oauth",
    }
}

/// TS `matchesReplacementCredential` (the alibaba/cloudflare composite-key matchers are for
/// providers whose login is not ported).
fn matches_replacement(
    provider: &str,
    existing: Option<&Credential>,
    existing_key: Option<&str>,
    incoming: &Credential,
) -> bool {
    let Some(existing) = existing else {
        return false;
    };
    if kind_of(existing) != kind_of(incoming) {
        return false;
    }
    let Credential::OAuth(incoming_data) = incoming else {
        return existing == incoming;
    };
    let incoming_ids = oauth_identifiers(incoming_data);
    let Some(incoming_key) = provider_identity_key(provider, &incoming_ids) else {
        return false;
    };
    if Some(incoming_key.as_str()) == existing_key {
        return true;
    }
    let Some(existing_key) = existing_key else {
        return false;
    };
    let Some(org) = incoming_ids.iter().find(|i| i.starts_with("org:")) else {
        return false;
    };
    if incoming_key != *org && !incoming_key.ends_with(&format!("|{org}")) {
        return false;
    }
    if existing_key == org {
        return true;
    }
    let existing_ids = match existing {
        Credential::OAuth(data) if existing_key.ends_with(&format!("|{org}")) => {
            Some(oauth_identifiers(data))
        }
        _ => None,
    };
    let org_id = &org["org:".len()..];
    for id in &incoming_ids {
        let is_base =
            id.starts_with("email:") || id.starts_with("account:") || id.starts_with("project:");
        if !is_base {
            continue;
        }
        let value = &id[id.find(':').map_or(0, |i| i + 1)..];
        if value == org_id {
            continue;
        }
        if existing_key == id || existing_key == format!("{id}|{org}") {
            return true;
        }
        if existing_ids.as_ref().is_some_and(|ids| ids.contains(id)) {
            return true;
        }
    }
    false
}

fn normalize_cause(cause: &str) -> String {
    let trimmed = cause.trim();
    if trimmed.is_empty() {
        "disabled".to_string()
    } else {
        trimmed.to_string()
    }
}

/// The local `agent.db` store.
pub struct LocalStore {
    conn: Connection,
}

impl LocalStore {
    /// Open (creating the directory, file and the `auth_credentials` table when absent) like TS
    /// `SqliteAuthCredentialStore.open`. An existing database is left to TS to migrate.
    ///
    /// # Errors
    /// The directory or database cannot be created/opened.
    pub fn open(db: &Path) -> Result<Self> {
        if let Some(dir) = db.parent() {
            if !dir.is_dir() {
                std::fs::create_dir_all(dir)
                    .with_context(|| format!("creating {}", dir.display()))?;
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt as _;
                    let _ = std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700));
                }
            }
        }
        let conn = Connection::open_with_flags(
            db,
            OpenFlags::SQLITE_OPEN_READ_WRITE
                | OpenFlags::SQLITE_OPEN_CREATE
                | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
        .with_context(|| format!("opening {}", db.display()))?;
        conn.busy_timeout(Duration::from_secs(5))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            let _ = std::fs::set_permissions(db, std::fs::Permissions::from_mode(0o600));
        }
        let present: bool = conn
            .query_row(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'auth_credentials'",
                [],
                |_| Ok(true),
            )
            .unwrap_or(false);
        if !present {
            conn.execute_batch(&format!(
                "PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;
                 CREATE TABLE IF NOT EXISTS auth_schema_version (id INTEGER PRIMARY KEY CHECK (id = 1), version INTEGER NOT NULL);
                 CREATE TABLE IF NOT EXISTS auth_credentials (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    provider TEXT NOT NULL,
                    credential_type TEXT NOT NULL,
                    data TEXT NOT NULL,
                    disabled_cause TEXT DEFAULT NULL,
                    identity_key TEXT DEFAULT NULL,
                    created_at INTEGER NOT NULL DEFAULT ({NOW_EPOCH}),
                    updated_at INTEGER NOT NULL DEFAULT ({NOW_EPOCH})
                 );
                 CREATE INDEX IF NOT EXISTS idx_auth_provider ON auth_credentials(provider);
                 CREATE INDEX IF NOT EXISTS idx_auth_provider_identity ON auth_credentials(provider, identity_key) WHERE identity_key IS NOT NULL;
                 INSERT OR REPLACE INTO auth_schema_version(id, version) VALUES (1, {AUTH_SCHEMA_VERSION});"
            ))?;
        }
        Ok(Self { conn })
    }

    /// Active rows, optionally for one provider, by id (TS `listAuthCredentials`).
    ///
    /// # Errors
    /// SQLite failure.
    pub fn list(&self, provider: Option<&str>) -> Result<Vec<StoredRow>> {
        Self::list_on(&self.conn, provider)
    }

    fn list_on(conn: &Connection, provider: Option<&str>) -> Result<Vec<StoredRow>> {
        let mut stmt = conn.prepare(
            "SELECT id, provider, credential_type, data FROM auth_credentials \
             WHERE disabled_cause IS NULL AND (?1 IS NULL OR provider = ?1) ORDER BY id ASC",
        )?;
        let rows = stmt
            .query_map([provider], |r| {
                Ok((
                    r.get::<_, i64>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, String>(3)?,
                ))
            })?
            .filter_map(std::result::Result::ok)
            .filter_map(|(id, provider, kind, data)| {
                StoredRow::from_parts(id, provider, &kind, &data)
            })
            .collect();
        Ok(rows)
    }

    /// Providers with at least one active row, first-seen order (TS `listProviders`).
    ///
    /// # Errors
    /// SQLite failure.
    pub fn providers(&self) -> Result<Vec<String>> {
        let mut out: Vec<String> = Vec::new();
        for row in self.list(None)? {
            if !out.contains(&row.provider) {
                out.push(row.provider);
            }
        }
        Ok(out)
    }

    /// TS `upsertAuthCredential`: one transaction that retires api-key rows an OAuth login
    /// supersedes, updates (or inserts) the matching row, then purges superseded tombstones.
    ///
    /// # Errors
    /// SQLite failure.
    pub fn upsert(&mut self, provider: &str, credential: &Credential) -> Result<Vec<StoredRow>> {
        let data = serialize(credential);
        let key = identity_key(provider, credential);
        let tx = self
            .conn
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let existing: Vec<(i64, Option<Credential>, Option<String>)> = {
            let mut stmt = tx.prepare(
                "SELECT id, credential_type, data, identity_key FROM auth_credentials \
                 WHERE provider = ?1 AND disabled_cause IS NULL ORDER BY id ASC",
            )?;
            let rows = stmt
                .query_map([provider], |r| {
                    Ok((
                        r.get::<_, i64>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, String>(2)?,
                        r.get::<_, Option<String>>(3)?,
                    ))
                })?
                .filter_map(std::result::Result::ok)
                .map(|(id, kind, data, stored_key)| {
                    let credential = StoredRow::from_parts(id, provider.to_string(), &kind, &data)
                        .map(|r| r.credential);
                    let key = stored_key
                        .map(|k| k.trim().to_string())
                        .filter(|k| !k.is_empty())
                        .or_else(|| credential.as_ref().and_then(|c| identity_key(provider, c)));
                    (id, credential, key)
                })
                .collect();
            rows
        };
        if matches!(credential, Credential::OAuth(_)) {
            for (id, existing, _) in &existing {
                if matches!(existing, Some(Credential::ApiKey(_))) {
                    disable_on(&tx, *id, "replaced by oauth login")?;
                }
            }
        }
        let mut target: Option<i64> = None;
        for (id, existing, existing_key) in &existing {
            if !matches_replacement(
                provider,
                existing.as_ref(),
                existing_key.as_deref(),
                credential,
            ) {
                continue;
            }
            if target.is_none() {
                target = Some(*id);
                tx.execute(
                    &format!(
                        "UPDATE auth_credentials SET credential_type = ?, data = ?, identity_key = ?, updated_at = {NOW_EPOCH} WHERE id = ?"
                    ),
                    params![kind_of(credential), data, key, id],
                )?;
                continue;
            }
            disable_on(&tx, *id, "replaced by newer credential")?;
        }
        if target.is_none() {
            tx.execute(
                &format!(
                    "INSERT INTO auth_credentials (provider, credential_type, data, identity_key, created_at, updated_at) VALUES (?, ?, ?, ?, {NOW_EPOCH}, {NOW_EPOCH})"
                ),
                params![provider, kind_of(credential), data, key],
            )?;
        }
        let active = Self::list_on(&tx, Some(provider))?;
        tx.commit()?;
        let _ = self.purge_superseded(provider, &active);
        Ok(active)
    }

    /// TS `#purgeSupersededDisabledRows`: best-effort.
    fn purge_superseded(&self, provider: &str, active: &[StoredRow]) -> Result<()> {
        let has_api_key = active
            .iter()
            .any(|r| matches!(r.credential, Credential::ApiKey(_)));
        let oauth: Vec<&Credential> = active
            .iter()
            .map(|r| &r.credential)
            .filter(|c| matches!(c, Credential::OAuth(_)))
            .collect();
        let active_keys: Vec<String> = oauth
            .iter()
            .filter_map(|c| identity_key(provider, c))
            .collect();
        if !has_api_key && active_keys.is_empty() {
            return Ok(());
        }
        let mut stmt = self.conn.prepare(
            "SELECT id, credential_type, data, identity_key FROM auth_credentials WHERE provider = ?1 AND disabled_cause IS NOT NULL ORDER BY id ASC",
        )?;
        let disabled: Vec<(i64, String, String, Option<String>)> = stmt
            .query_map([provider], |r| {
                Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?))
            })?
            .filter_map(std::result::Result::ok)
            .collect();
        for (id, kind, data, stored_key) in disabled {
            if has_api_key && kind == "api_key" {
                self.conn
                    .execute("DELETE FROM auth_credentials WHERE id = ?", [id])?;
                continue;
            }
            let credential =
                StoredRow::from_parts(id, provider.to_string(), &kind, &data).map(|r| r.credential);
            let key = stored_key
                .map(|k| k.trim().to_string())
                .filter(|k| !k.is_empty())
                .or_else(|| credential.as_ref().and_then(|c| identity_key(provider, c)));
            if key.as_ref().is_some_and(|k| active_keys.contains(k)) {
                self.conn
                    .execute("DELETE FROM auth_credentials WHERE id = ?", [id])?;
                continue;
            }
            let Some(credential) = credential else {
                continue;
            };
            if oauth.iter().any(|active| {
                matches_replacement(provider, Some(&credential), key.as_deref(), active)
            }) {
                self.conn
                    .execute("DELETE FROM auth_credentials WHERE id = ?", [id])?;
            }
        }
        Ok(())
    }

    /// TS `deleteAuthCredential`: soft-delete one row; `false` when it was not active.
    ///
    /// # Errors
    /// SQLite failure.
    pub fn disable(&self, id: i64, cause: &str) -> Result<bool> {
        Ok(disable_on(&self.conn, id, cause)? > 0)
    }

    /// TS `deleteAuthCredentials`: soft-delete every active row of a provider.
    ///
    /// # Errors
    /// SQLite failure.
    pub fn disable_provider(&self, provider: &str, cause: &str) -> Result<()> {
        self.conn.execute(
            &format!("UPDATE auth_credentials SET disabled_cause = ?, updated_at = {NOW_EPOCH} WHERE provider = ? AND disabled_cause IS NULL"),
            params![normalize_cause(cause), provider],
        )?;
        Ok(())
    }
}

fn disable_on(conn: &Connection, id: i64, cause: &str) -> Result<usize> {
    Ok(conn.execute(
        &format!("UPDATE auth_credentials SET disabled_cause = ?, updated_at = {NOW_EPOCH} WHERE id = ? AND disabled_cause IS NULL"),
        params![normalize_cause(cause), id],
    )?)
}

// ------------------------------------------------------------------ broker

async fn broker_call(
    cfg: &BrokerConfig,
    method: reqwest::Method,
    path: &str,
    body: Option<&Value>,
) -> Result<Value> {
    let url = format!("{}{path}", cfg.url.trim_end_matches('/'));
    let client = reqwest::Client::builder().timeout(BROKER_TIMEOUT).build()?;
    let mut request = client
        .request(method, &url)
        .bearer_auth(&cfg.token)
        .header("Accept", "application/json")
        .header("OMP-Auth-Broker-Capabilities", "codex-meter-block-scopes");
    if let Some(body) = body {
        request = request.json(body);
    }
    let response = request
        .send()
        .await
        .map_err(|e| anyhow!("Auth broker request failed: {e}"))?;
    let status = response.status();
    let text = response.text().await.unwrap_or_default();
    if !status.is_success() {
        bail!(
            "Auth broker request failed: {status}: {}",
            text.chars().take(300).collect::<String>()
        );
    }
    serde_json::from_str(&text).map_err(|e| anyhow!("Auth broker returned malformed JSON: {e}"))
}

fn credential_wire(credential: &Credential) -> Value {
    match credential {
        Credential::ApiKey(key) => json!({"type": "api_key", "key": key, "source": "login"}),
        Credential::OAuth(data) => {
            let mut wire = Map::new();
            wire.insert("type".into(), json!("oauth"));
            wire.extend(data.clone());
            Value::Object(wire)
        }
    }
}

/// Broker `POST /v1/credential`.
///
/// # Errors
/// Transport or HTTP failure.
pub async fn broker_upload(
    cfg: &BrokerConfig,
    provider: &str,
    credential: &Credential,
) -> Result<()> {
    let body = json!({"provider": provider, "credential": credential_wire(credential)});
    broker_call(cfg, reqwest::Method::POST, "/v1/credential", Some(&body)).await?;
    Ok(())
}

// ------------------------------------------------------- broker usage reads

/// One `/v1/usage/history` entry (TS `UsageHistoryEntry`); only the fields the
/// stats dashboard's window analytics consume.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BrokerUsageHistoryEntry {
    pub recorded_at: i64,
    pub provider: String,
    pub account_key: String,
    pub email: Option<String>,
    pub account_id: Option<String>,
    pub limit_id: String,
    pub label: String,
    pub window_label: Option<String>,
    pub used_fraction: Option<f64>,
    pub status: Option<String>,
}

#[derive(Debug, Deserialize)]
struct UsageHistoryResponse {
    entries: Vec<BrokerUsageHistoryEntry>,
}

/// Per-provider aggregate of one client's recorded usage (TS
/// `ClientProviderUsage`); only the token fields `sumFleetTokens` folds.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BrokerClientProviderUsage {
    pub provider: String,
    pub input_tokens: f64,
    pub output_tokens: f64,
    pub cache_read_tokens: f64,
    pub cache_write_tokens: f64,
}

/// One known client with its usage aggregates over the queried window (TS
/// `ClientUsageClientSummary`); only the fields the fleet-token fold uses.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BrokerClientUsageClientSummary {
    pub install_id: String,
    pub providers: Vec<BrokerClientProviderUsage>,
}

#[derive(Debug, Deserialize)]
struct ClientUsageSummaryResponse {
    clients: Vec<BrokerClientUsageClientSummary>,
}

/// Broker `GET /v1/usage/history?sinceMs=`: recorded usage-limit snapshots
/// from the broker host's durable history, oldest first. The TS client's
/// default 10 s timeout applies (`BROKER_TIMEOUT`).
///
/// # Errors
/// Transport, HTTP or schema failure.
pub async fn broker_usage_history(
    cfg: &BrokerConfig,
    since_ms: i64,
) -> Result<Vec<BrokerUsageHistoryEntry>> {
    let body = broker_call(
        cfg,
        reqwest::Method::GET,
        &format!("/v1/usage/history?sinceMs={since_ms}"),
        None,
    )
    .await?;
    let response: UsageHistoryResponse = serde_json::from_value(body)
        .map_err(|e| anyhow!("Auth broker returned malformed JSON: {e}"))?;
    Ok(response.entries)
}

/// Broker `GET /v1/usage/clients?sinceMs=`: per-client token burn aggregates
/// over the window. The TS client's default 10 s timeout applies.
///
/// # Errors
/// Transport, HTTP or schema failure.
pub async fn broker_client_usage_summary(
    cfg: &BrokerConfig,
    since_ms: i64,
) -> Result<Vec<BrokerClientUsageClientSummary>> {
    let body = broker_call(
        cfg,
        reqwest::Method::GET,
        &format!("/v1/usage/clients?sinceMs={since_ms}"),
        None,
    )
    .await?;
    let response: ClientUsageSummaryResponse = serde_json::from_value(body)
        .map_err(|e| anyhow!("Auth broker returned malformed JSON: {e}"))?;
    Ok(response.clients)
}

/// Broker `GET /v1/snapshot` rows (secrets redacted by the broker), by id.
///
/// # Errors
/// Transport or HTTP failure.
pub async fn broker_list(cfg: &BrokerConfig, provider: Option<&str>) -> Result<Vec<StoredRow>> {
    let body = broker_call(cfg, reqwest::Method::GET, "/v1/snapshot", None).await?;
    let mut rows: Vec<StoredRow> = body
        .get("credentials")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|entry| {
            let id = entry.get("id")?.as_i64()?;
            let row_provider = entry.get("provider")?.as_str()?.to_string();
            if provider.is_some_and(|p| p != row_provider) {
                return None;
            }
            let credential = entry.get("credential")?;
            let kind = credential.get("type")?.as_str()?.to_string();
            let mut data = credential.as_object()?.clone();
            data.remove("type");
            StoredRow::from_parts(id, row_provider, &kind, &Value::Object(data).to_string())
        })
        .collect();
    rows.sort_by_key(|r| r.id);
    Ok(rows)
}

/// Broker `POST /v1/credential/:id/disable`.
///
/// # Errors
/// Transport or HTTP failure.
pub async fn broker_disable(cfg: &BrokerConfig, id: i64, cause: &str) -> Result<()> {
    let body = json!({"cause": cause});
    broker_call(
        cfg,
        reqwest::Method::POST,
        &format!("/v1/credential/{id}/disable"),
        Some(&body),
    )
    .await?;
    Ok(())
}

#[cfg(test)]
mod usage_tests {
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};

    use super::*;

    /// Canned HTTP/1.1 server (the `auth/tests.rs` pattern): records the
    /// request line and headers of every request, replies from `reply`.
    fn serve(
        reply: impl Fn(&str) -> (u16, String) + Send + 'static,
    ) -> (String, Arc<Mutex<Vec<String>>>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let seen = Arc::new(Mutex::new(Vec::new()));
        let log = seen.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { continue };
                let mut buf = Vec::new();
                let mut chunk = [0u8; 4096];
                loop {
                    let n = stream.read(&mut chunk).unwrap_or(0);
                    buf.extend_from_slice(&chunk[..n]);
                    if buf.windows(4).any(|w| w == b"\r\n\r\n") {
                        break;
                    }
                    if n == 0 {
                        break;
                    }
                }
                let text = String::from_utf8_lossy(&buf).to_string();
                let line = text.lines().next().unwrap_or_default().to_string();
                log.lock().unwrap().push(text);
                let (status, body) = reply(&line);
                let _ = write!(
                    stream,
                    "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
            }
        });
        (url, seen)
    }

    fn cfg(url: &str) -> BrokerConfig {
        BrokerConfig {
            url: url.to_string(),
            token: "tok".to_string(),
        }
    }

    #[tokio::test]
    async fn usage_history_gets_since_ms_and_parses_entries() {
        let (url, seen) = serve(|line| {
            assert_eq!(line, "GET /v1/usage/history?sinceMs=1728000000000 HTTP/1.1");
            (
                200,
                r#"{"generatedAt":1728000001,"entries":[
                    {"recordedAt":1728000001000,"provider":"google-antigravity","accountKey":"k1","email":"a@b.c",
                     "accountId":"acc1","limitId":"daily","label":"Usage","windowLabel":"Daily",
                     "usedFraction":0.5,"status":"ok"},
                    {"recordedAt":1728000002000,"provider":"kimi","accountKey":"k2","limitId":"weekly","label":"L"}
                ]}"#
                .to_string(),
            )
        });
        let entries = broker_usage_history(&cfg(&url), 1_728_000_000_000)
            .await
            .unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].provider, "google-antigravity");
        assert_eq!(entries[0].recorded_at, 1_728_000_001_000);
        assert_eq!(entries[0].email.as_deref(), Some("a@b.c"));
        assert_eq!(entries[0].used_fraction, Some(0.5));
        assert_eq!(entries[0].status.as_deref(), Some("ok"));
        // Absent optionals are None (TS `?? null`).
        assert_eq!(entries[1].email, None);
        assert_eq!(entries[1].account_id, None);
        assert_eq!(entries[1].window_label, None);
        assert_eq!(entries[1].used_fraction, None);
        assert_eq!(entries[1].status, None);
        let request = seen.lock().unwrap().join("\n");
        assert!(request.contains("authorization: Bearer tok"), "{request}");
    }

    #[tokio::test]
    async fn client_usage_summary_gets_since_ms_and_parses_clients() {
        let (url, seen) = serve(|line| {
            assert_eq!(line, "GET /v1/usage/clients?sinceMs=42 HTTP/1.1");
            (
                200,
                r#"{"generatedAt":1728000001,"clients":[
                    {"installId":"i1","firstSeen":1727990000,"lastSeen":1728000000,"providers":[
                        {"provider":"google-antigravity","inputTokens":10,"outputTokens":5,
                         "cacheReadTokens":2,"cacheWriteTokens":1},
                        {"provider":"kimi","inputTokens":1,"outputTokens":1,"cacheReadTokens":0,"cacheWriteTokens":0}
                    ]},
                    {"installId":"i2","firstSeen":1727990000,"lastSeen":1728000000,"providers":[]}
                ]}"#
                .to_string(),
            )
        });
        let clients = broker_client_usage_summary(&cfg(&url), 42).await.unwrap();
        assert_eq!(clients.len(), 2);
        assert_eq!(clients[0].install_id, "i1");
        assert_eq!(clients[0].providers.len(), 2);
        assert_eq!(clients[0].providers[0].provider, "google-antigravity");
        assert_eq!(clients[0].providers[0].cache_write_tokens, 1.0);
        assert!(clients[1].providers.is_empty());
        let request = seen.lock().unwrap().join("\n");
        assert!(request.contains("authorization: Bearer tok"), "{request}");
    }

    #[tokio::test]
    async fn usage_history_http_error_bails() {
        let (url, _seen) = serve(|_| (500, r#"{"error":"boom"}"#.to_string()));
        let err = broker_usage_history(&cfg(&url), 0)
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("500"), "{err}");
    }
}
