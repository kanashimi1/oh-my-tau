//! `agent.db` reads and writes behind the broker routes (TS `SqliteAuthCredentialStore` +
//! `CredentialPool.snapshot`). Every call opens its own short-lived connection: SQLite in WAL
//! mode with a busy timeout, shared with omp-rs clients refreshing locally and other processes.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};
use std::time::Duration;

use anyhow::{Context, Result};
use omp_providers::login::store::identity_key;
use omp_providers::login::{Credential, LocalStore};
use rusqlite::{params, Connection, OpenFlags, OptionalExtension};
use serde_json::{json, Map, Value};

pub const REMOTE_REFRESH_SENTINEL: &str = "__remote__";
/// TS `CLIENT_USAGE_BUCKET_MS`: observed usage merges into a row younger than this.
const CLIENT_USAGE_BUCKET_MS: i64 = 5 * 60_000;
const NOW_EPOCH: &str = "CAST(strftime('%s','now') AS INTEGER)";

/// One long-lived connection serialized behind a mutex: opening a connection per call (WAL
/// attach + schema parse) cost more than the queries themselves under snapshot load.
#[derive(Debug)]
pub struct Store {
    pub db: PathBuf,
    conn: Mutex<Connection>,
}

/// One active credential as the snapshot exposes it (TS `AuthCredentialSnapshotEntry`, before
/// `rotatesInMs`/`blocks`).
#[derive(Debug, Clone)]
pub struct Entry {
    pub id: i64,
    pub provider: String,
    /// The deserialized credential (`type` first), refresh token NOT yet redacted.
    pub credential: Map<String, Value>,
}

impl Entry {
    pub fn is_oauth(&self) -> bool {
        self.credential.get("type").and_then(Value::as_str) == Some("oauth")
    }

    pub fn expires(&self) -> Option<f64> {
        self.credential
            .get("expires")
            .and_then(Value::as_f64)
            .filter(|e| e.is_finite())
    }

    /// TS `snapshotEntry`: refresh token replaced by the remote sentinel, identity key resolved
    /// from the credential itself.
    pub fn wire(&self) -> Map<String, Value> {
        let mut credential = self.credential.clone();
        let identity = if self.is_oauth() {
            credential.insert(
                "refresh".into(),
                Value::String(REMOTE_REFRESH_SENTINEL.into()),
            );
            let mut data = self.credential.clone();
            data.shift_remove("type");
            identity_key(&self.provider, &Credential::OAuth(data))
        } else {
            None
        };
        let mut out = Map::new();
        out.insert("id".into(), json!(self.id));
        out.insert("provider".into(), json!(self.provider));
        out.insert("credential".into(), Value::Object(credential));
        out.insert(
            "identityKey".into(),
            identity.map_or(Value::Null, Value::String),
        );
        out
    }
}

/// TS `deserializeCredential`.
fn deserialize(kind: &str, data: &str) -> Option<Map<String, Value>> {
    let Value::Object(parsed) = serde_json::from_str::<Value>(data).ok()? else {
        return None;
    };
    let mut out = Map::new();
    match kind {
        "api_key" => {
            let key = parsed.get("key")?.as_str()?;
            out.insert("type".into(), json!("api_key"));
            out.insert("key".into(), json!(key));
            if parsed.get("source").and_then(Value::as_str) == Some("login") {
                out.insert("source".into(), json!("login"));
            }
        }
        "oauth" => {
            out.insert("type".into(), json!("oauth"));
            out.extend(parsed);
        }
        _ => return None,
    }
    Some(out)
}

pub struct Block {
    pub credential_id: i64,
    pub provider_key: String,
    pub block_scope: String,
    pub blocked_until_ms: i64,
    pub updated_at_ms: i64,
}

impl Store {
    fn conn(&self) -> Result<MutexGuard<'_, Connection>> {
        Ok(self
            .conn
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner))
    }

    /// Open the store, creating what TS `SqliteAuthCredentialStore` would on a fresh database
    /// (the credential table via omp-providers, then the broker-owned tables and the
    /// `auth_change_revision` triggers peers poll).
    pub fn open(db: &Path) -> Result<Self> {
        drop(LocalStore::open(db)?);
        let conn = Connection::open_with_flags(
            db,
            OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
        .with_context(|| format!("opening {}", db.display()))?;
        conn.busy_timeout(Duration::from_secs(5))?;
        let store = Self {
            db: db.to_path_buf(),
            conn: Mutex::new(conn),
        };
        let conn = store.conn()?;
        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             CREATE TABLE IF NOT EXISTS auth_change_revision (id INTEGER PRIMARY KEY CHECK (id = 1), revision INTEGER NOT NULL);
             INSERT OR IGNORE INTO auth_change_revision (id, revision) VALUES (1, 0);
             CREATE TABLE IF NOT EXISTS auth_credential_blocks (
                credential_id INTEGER NOT NULL, provider_key TEXT NOT NULL, block_scope TEXT NOT NULL DEFAULT '',
                blocked_until_ms INTEGER NOT NULL, updated_at INTEGER NOT NULL,
                PRIMARY KEY (credential_id, provider_key, block_scope));
             CREATE INDEX IF NOT EXISTS idx_auth_credential_blocks_expires ON auth_credential_blocks(blocked_until_ms);
             CREATE TABLE IF NOT EXISTS auth_credential_refresh_leases (
                credential_id INTEGER PRIMARY KEY, owner TEXT NOT NULL, expires_at_ms INTEGER NOT NULL, updated_at INTEGER NOT NULL);
             CREATE TABLE IF NOT EXISTS usage_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT, recorded_at INTEGER NOT NULL, provider TEXT NOT NULL,
                account_key TEXT NOT NULL, email TEXT, account_id TEXT, limit_id TEXT NOT NULL, label TEXT NOT NULL,
                window_label TEXT, used_fraction REAL, status TEXT, resets_at INTEGER);
             CREATE INDEX IF NOT EXISTS idx_usage_history_recorded ON usage_history(recorded_at);
             CREATE TABLE IF NOT EXISTS clients (install_id TEXT PRIMARY KEY, hostname TEXT, first_seen INTEGER NOT NULL, last_seen INTEGER NOT NULL);
             CREATE TABLE IF NOT EXISTS client_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT, recorded_at INTEGER NOT NULL, install_id TEXT NOT NULL,
                provider TEXT NOT NULL, model TEXT NOT NULL, requests INTEGER NOT NULL, input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL, cache_read_tokens INTEGER NOT NULL, cache_write_tokens INTEGER NOT NULL,
                cost_usd REAL NOT NULL DEFAULT 0);
             CREATE INDEX IF NOT EXISTS idx_client_usage_recorded ON client_usage(recorded_at);",
        )?;
        for table in ["auth_credentials", "auth_credential_blocks"] {
            for event in ["INSERT", "UPDATE", "DELETE"] {
                conn.execute_batch(&format!(
                    "CREATE TRIGGER IF NOT EXISTS auth_change_revision_{table}_{ev} AFTER {event} ON {table}
                     BEGIN UPDATE auth_change_revision SET revision = revision + 1 WHERE id = 1; END;",
                    ev = event.to_lowercase()
                ))?;
            }
        }
        let has_app: bool = conn
            .prepare("SELECT 1 FROM pragma_table_info('client_usage') WHERE name = 'app'")?
            .exists([])?;
        if !has_app {
            conn.execute_batch("ALTER TABLE client_usage ADD COLUMN app TEXT NOT NULL DEFAULT ''")?;
        }
        drop(conn);
        Ok(store)
    }

    /// The store-wide change counter every writer bumps through triggers.
    pub fn revision(&self) -> Result<i64> {
        let conn = self.conn()?;
        let mut stmt =
            conn.prepare_cached("SELECT revision FROM auth_change_revision WHERE id = 1")?;
        Ok(stmt.query_row([], |r| r.get(0))?)
    }

    /// Active credentials grouped by provider in first-appearance order of `id ASC` (the TS
    /// pool's provider `Map` after a reload).
    pub fn entries(&self) -> Result<Vec<Entry>> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare_cached(
            "SELECT id, provider, credential_type, data FROM auth_credentials WHERE disabled_cause IS NULL ORDER BY id ASC",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, String>(3)?,
            ))
        })?;
        let mut groups: Vec<(String, Vec<Entry>)> = Vec::new();
        for row in rows {
            let (id, provider, kind, data) = row?;
            let Some(credential) = deserialize(&kind, &data) else {
                continue;
            };
            let entry = Entry {
                id,
                provider: provider.clone(),
                credential,
            };
            match groups.iter_mut().find(|(p, _)| *p == provider) {
                Some((_, list)) => list.push(entry),
                None => groups.push((provider, vec![entry])),
            }
        }
        Ok(groups.into_iter().flat_map(|(_, list)| list).collect())
    }

    pub fn entry(&self, id: i64) -> Result<Option<Entry>> {
        Ok(self.entries()?.into_iter().find(|e| e.id == id))
    }

    /// TS `listDisabledCredentials`.
    pub fn disabled(&self, provider: Option<&str>) -> Result<Vec<Value>> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare(
            "SELECT id, provider, credential_type, data, disabled_cause, updated_at FROM auth_credentials
             WHERE disabled_cause IS NOT NULL AND (?1 IS NULL OR provider = ?1) ORDER BY id ASC",
        )?;
        let rows = stmt.query_map([provider], |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, String>(3)?,
                r.get::<_, Option<String>>(4)?,
                r.get::<_, Option<i64>>(5)?,
            ))
        })?;
        let mut out = Vec::new();
        for row in rows {
            let (id, provider, kind, data, cause, updated_at) = row?;
            let mut s = Map::new();
            s.insert("id".into(), json!(id));
            s.insert("provider".into(), json!(provider));
            s.insert(
                "type".into(),
                json!(if kind == "api_key" {
                    "api_key"
                } else {
                    "oauth"
                }),
            );
            s.insert(
                "cause".into(),
                json!(cause.unwrap_or_else(|| "disabled".into())),
            );
            if let Some(cred) = deserialize(&kind, &data).filter(|_| kind == "oauth") {
                for key in ["email", "accountId", "orgId", "orgName"] {
                    if let Some(v) = cred
                        .get(key)
                        .filter(|v| v.as_str().is_some_and(|s| !s.is_empty()))
                    {
                        s.insert(key.into(), v.clone());
                    }
                }
            }
            if let Some(at) = updated_at {
                s.insert("disabledAtMs".into(), json!(at * 1000));
            }
            out.push(Value::Object(s));
        }
        Ok(out)
    }

    /// TS `credentials.upsert` → the provider's active entries after the write.
    pub fn upsert(&self, provider: &str, credential: &Map<String, Value>) -> Result<Vec<Entry>> {
        let mut data = credential.clone();
        let kind = data.shift_remove("type");
        let cred = match kind.as_ref().and_then(Value::as_str) {
            Some("api_key") => Credential::ApiKey(
                data.get("key")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
            ),
            _ => Credential::OAuth(data),
        };
        let mut store = LocalStore::open(&self.db)?;
        let rows = store.upsert(provider, &cred)?;
        // omp-providers writes login keys as `{"key","source":"login"}`; TS `serializeCredential`
        // keeps `source` only when the caller sent it.
        if kind.as_ref().and_then(Value::as_str) == Some("api_key")
            && credential.get("source").is_none()
        {
            let conn = self.conn()?;
            for r in &rows {
                if let Credential::ApiKey(key) = &r.credential {
                    conn.execute(
                        "UPDATE auth_credentials SET data = ?1 WHERE id = ?2 AND data <> ?1",
                        params![json!({"key": key}).to_string(), r.id],
                    )?;
                }
            }
        }
        let all = self.entries()?;
        Ok(rows
            .iter()
            .filter_map(|r| all.iter().find(|e| e.id == r.id).cloned())
            .collect())
    }

    pub fn disable(&self, id: i64, cause: &str) -> Result<bool> {
        LocalStore::open(&self.db)?.disable(id, cause)
    }

    /// Unexpired blocks (TS `listCredentialBlocks`), `updated_at` seconds → ms. Expired rows are
    /// purged first and the legacy Codex `shared` row is hidden (its meters are mirrored by the
    /// `auth_codex_*` triggers), as TS does.
    pub fn blocks(&self, now_ms: i64) -> Result<Vec<Block>> {
        let conn = self.conn()?;
        conn.execute(
            "DELETE FROM auth_credential_blocks WHERE blocked_until_ms <= ?1",
            [now_ms],
        )?;
        let mut stmt = conn.prepare_cached(
            "SELECT credential_id, provider_key, block_scope, blocked_until_ms, updated_at FROM auth_credential_blocks
             WHERE blocked_until_ms > ?1 AND NOT (provider_key = 'openai-codex:oauth' AND block_scope = 'shared')
             ORDER BY credential_id, provider_key ASC, block_scope ASC",
        )?;
        let rows = stmt.query_map([now_ms], |r| {
            Ok(Block {
                credential_id: r.get(0)?,
                provider_key: r.get(1)?,
                block_scope: r.get(2)?,
                blocked_until_ms: r.get(3)?,
                updated_at_ms: r.get::<_, i64>(4)? * 1000,
            })
        })?;
        Ok(rows.collect::<std::result::Result<_, _>>()?)
    }

    pub fn block_upsert(
        &self,
        id: i64,
        provider_key: &str,
        scope: &str,
        until_ms: i64,
    ) -> Result<()> {
        self.conn()?.execute(
            &format!(
                "INSERT INTO auth_credential_blocks (credential_id, provider_key, block_scope, blocked_until_ms, updated_at)
                 VALUES (?1, ?2, ?3, ?4, {NOW_EPOCH})
                 ON CONFLICT(credential_id, provider_key, block_scope) DO UPDATE SET
                    blocked_until_ms = MAX(blocked_until_ms, excluded.blocked_until_ms), updated_at = excluded.updated_at"
            ),
            params![id, provider_key, scope, until_ms],
        )?;
        Ok(())
    }

    pub fn block_delete(&self, id: i64, provider_key: &str, scope: &str) -> Result<()> {
        self.conn()?.execute(
            "DELETE FROM auth_credential_blocks WHERE credential_id = ?1 AND provider_key = ?2 AND block_scope = ?3",
            params![id, provider_key, scope],
        )?;
        Ok(())
    }

    pub fn blocks_delete_all(&self, id: i64) -> Result<()> {
        self.conn()?.execute(
            "DELETE FROM auth_credential_blocks WHERE credential_id = ?1",
            [id],
        )?;
        Ok(())
    }

    /// TS `listUsageHistory` (errors read as an empty list, as TS does).
    pub fn usage_history(&self, since_ms: i64, provider: Option<&str>) -> Vec<Value> {
        let run = || -> Result<Vec<Value>> {
            let conn = self.conn()?;
            let mut stmt = conn.prepare(
                "SELECT recorded_at, provider, account_key, email, account_id, limit_id, label, window_label, used_fraction,
                        status, resets_at
                 FROM usage_history WHERE recorded_at >= ?1 AND (?2 IS NULL OR provider = ?2) ORDER BY recorded_at ASC",
            )?;
            let rows = stmt.query_map(params![since_ms, provider], |r| {
                let mut m = Map::new();
                let opt = |m: &mut Map<String, Value>, k: &str, v: Option<Value>| {
                    if let Some(v) = v {
                        m.insert(k.into(), v);
                    }
                };
                m.insert("recordedAt".into(), json!(r.get::<_, i64>(0)?));
                m.insert("provider".into(), json!(r.get::<_, String>(1)?));
                m.insert("accountKey".into(), json!(r.get::<_, String>(2)?));
                opt(
                    &mut m,
                    "email",
                    r.get::<_, Option<String>>(3)?.map(Value::String),
                );
                opt(
                    &mut m,
                    "accountId",
                    r.get::<_, Option<String>>(4)?.map(Value::String),
                );
                m.insert("limitId".into(), json!(r.get::<_, String>(5)?));
                m.insert("label".into(), json!(r.get::<_, String>(6)?));
                opt(
                    &mut m,
                    "windowLabel",
                    r.get::<_, Option<String>>(7)?.map(Value::String),
                );
                opt(
                    &mut m,
                    "usedFraction",
                    r.get::<_, Option<f64>>(8)?.map(|f| json!(f)),
                );
                opt(
                    &mut m,
                    "status",
                    r.get::<_, Option<String>>(9)?.map(Value::String),
                );
                opt(
                    &mut m,
                    "resetsAt",
                    r.get::<_, Option<i64>>(10)?.map(|v| json!(v)),
                );
                Ok(Value::Object(m))
            })?;
            Ok(rows.collect::<std::result::Result<_, _>>()?)
        };
        run().unwrap_or_default()
    }

    /// TS `recordClientUsage`.
    pub fn record_client(&self, report: &ClientReport, now_ms: i64) -> Result<()> {
        let mut conn = self.conn()?;
        let tx = conn.transaction()?;
        tx.execute(
            "INSERT INTO clients (install_id, hostname, first_seen, last_seen) VALUES (?1, ?2, ?3, ?3)
             ON CONFLICT(install_id) DO UPDATE SET hostname = COALESCE(excluded.hostname, hostname), last_seen = excluded.last_seen",
            params![report.install_id, report.hostname, now_ms],
        )?;
        let app = report.app.as_deref().map(str::trim).unwrap_or_default();
        for e in &report.entries {
            let existing: Option<i64> = tx
                .query_row(
                    "SELECT id FROM client_usage WHERE install_id = ?1 AND app = ?2 AND provider = ?3 AND model = ?4
                       AND recorded_at >= ?5 ORDER BY recorded_at DESC LIMIT 1",
                    params![report.install_id, app, e.provider, e.model, e.at - CLIENT_USAGE_BUCKET_MS],
                    |r| r.get(0),
                )
                .optional()?;
            match existing {
                Some(id) => tx.execute(
                    "UPDATE client_usage SET recorded_at = ?1, requests = requests + ?2, input_tokens = input_tokens + ?3,
                       output_tokens = output_tokens + ?4, cache_read_tokens = cache_read_tokens + ?5,
                       cache_write_tokens = cache_write_tokens + ?6, cost_usd = cost_usd + ?7 WHERE id = ?8",
                    params![e.at, e.requests, e.input, e.output, e.cache_read, e.cache_write, e.cost, id],
                )?,
                None => tx.execute(
                    "INSERT INTO client_usage (recorded_at, install_id, app, provider, model, requests, input_tokens,
                       output_tokens, cache_read_tokens, cache_write_tokens, cost_usd)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
                    params![
                        e.at,
                        report.install_id,
                        app,
                        e.provider,
                        e.model,
                        e.requests,
                        e.input,
                        e.output,
                        e.cache_read,
                        e.cache_write,
                        e.cost
                    ],
                )?,
            };
        }
        tx.commit()?;
        Ok(())
    }

    /// TS `getClientUsageSummary`.
    pub fn client_summary(&self, since_ms: i64) -> Result<Vec<Value>> {
        let conn = self.conn()?;
        let num = |r: &rusqlite::Row<'_>, i: usize| -> rusqlite::Result<Value> {
            Ok(match r.get_ref(i)? {
                rusqlite::types::ValueRef::Integer(v) => json!(v),
                rusqlite::types::ValueRef::Real(v) => json!(v),
                _ => Value::Null,
            })
        };
        let mut by_install: Vec<(String, Vec<Value>)> = Vec::new();
        {
            let mut aggs = conn.prepare(
                "SELECT install_id, app, provider, SUM(requests), SUM(input_tokens), SUM(output_tokens),
                        SUM(cache_read_tokens), SUM(cache_write_tokens), SUM(cost_usd)
                 FROM client_usage WHERE recorded_at >= ?1 GROUP BY install_id, app, provider
                 ORDER BY install_id, SUM(input_tokens + output_tokens + cache_read_tokens + cache_write_tokens) DESC",
            )?;
            let rows = aggs.query_map([since_ms], |r| {
                let install: String = r.get(0)?;
                let app: String = r.get(1)?;
                let mut m = Map::new();
                if !app.is_empty() {
                    m.insert("app".into(), json!(app));
                }
                m.insert("provider".into(), json!(r.get::<_, String>(2)?));
                for (i, k) in [
                    "requests",
                    "inputTokens",
                    "outputTokens",
                    "cacheReadTokens",
                    "cacheWriteTokens",
                    "costUsd",
                ]
                .iter()
                .enumerate()
                {
                    m.insert((*k).into(), num(r, i + 3)?);
                }
                Ok((install, Value::Object(m)))
            })?;
            for row in rows {
                let (install, v) = row?;
                match by_install.iter_mut().find(|(i, _)| *i == install) {
                    Some((_, list)) => list.push(v),
                    None => by_install.push((install, vec![v])),
                }
            }
        }
        let mut stmt = conn.prepare("SELECT install_id, hostname, first_seen, last_seen FROM clients ORDER BY last_seen DESC")?;
        let clients = stmt.query_map([], |r| {
            let install: String = r.get(0)?;
            let mut m = Map::new();
            m.insert("installId".into(), json!(install));
            if let Some(h) = r.get::<_, Option<String>>(1)? {
                m.insert("hostname".into(), json!(h));
            }
            m.insert("firstSeen".into(), json!(r.get::<_, i64>(2)?));
            m.insert("lastSeen".into(), json!(r.get::<_, i64>(3)?));
            Ok((install, m))
        })?;
        let mut out = Vec::new();
        for c in clients {
            let (install, mut m) = c?;
            let providers = by_install
                .iter()
                .find(|(i, _)| *i == install)
                .map(|(_, l)| l.clone())
                .unwrap_or_default();
            m.insert("providers".into(), Value::Array(providers));
            out.push(Value::Object(m));
        }
        Ok(out)
    }
}

/// `POST /v1/usage/observed` body (TS `ClientUsageReport`).
pub struct ClientReport {
    pub install_id: String,
    pub hostname: Option<String>,
    pub app: Option<String>,
    pub entries: Vec<ObservedEntry>,
}

pub struct ObservedEntry {
    pub at: i64,
    pub provider: String,
    pub model: String,
    pub requests: f64,
    pub input: f64,
    pub output: f64,
    pub cache_read: f64,
    pub cache_write: f64,
    pub cost: f64,
}
