//! Online model discovery — a compact port of TS `model-discovery` +
//! `catalog/src/model-cache.ts` `writeModelCache`.
//!
//! `refresh` walks every provider that has a base URL and a resolvable
//! credential (models.yml `apiKey` → env var → `agent.db`
//! `auth_credentials`), fetches its `/models` endpoint (query string
//! preserved, TS `appendModelsPath`), merges the returned ids with the
//! provider's existing `models.db` cache row and writes the row back in the
//! exact TS `model_cache` schema — including the `materialization_policy`
//! string TS computes at build time (see `build.rs`); TS readers delete any
//! row whose policy differs.
//!
//! TS reference: `coding-agent/src/config/model-discovery.ts`,
//! `catalog/src/model-cache.ts`.

use std::path::Path;
use std::time::Duration;

use anyhow::{bail, Context as _, Result};
use serde_json::{json, Value};

use crate::AgentDir;

/// TS `CACHE_SCHEMA_VERSION` (`catalog/src/model-cache.ts`).
const CACHE_SCHEMA_VERSION: i64 = 13;
/// TS `HEADER_RESTORE_VERSION`.
const HEADER_RESTORE_VERSION: i64 = 1;
/// TS `BUILT_IN_DISCOVERY_CACHE_TTL_MS`: a fresh authoritative row skips the network.
const CACHE_TTL_MS: u64 = 2 * 60 * 60 * 1000;
/// TS `BUILT_IN_DISCOVERY_NON_AUTHORITATIVE_RETRY_MS`.
const NON_AUTHORITATIVE_RETRY_MS: u64 = 5 * 60 * 1000;
/// TS `RUNTIME_DYNAMIC_MODEL_FETCH_TIMEOUT_MS`.
const FETCH_TIMEOUT: Duration = Duration::from_secs(15);
/// TS `DISCOVERY_DEFAULT_CONTEXT_WINDOW` (`provider-models/openai-compat.ts`).
const DEFAULT_CONTEXT_WINDOW: u64 = 128_000;
/// TS `DISCOVERY_DEFAULT_MAX_TOKENS`.
const DEFAULT_MAX_TOKENS: u64 = 32_768;

/// The exact TS `materializationPolicy()` string (computed in `build.rs`).
pub const MATERIALIZATION_POLICY: &str = env!("OMP_MODEL_MATERIALIZATION_POLICY");

/// Outcome of refreshing one provider.
#[derive(Debug, Clone)]
pub struct ProviderRefresh {
    pub provider: String,
    /// Model ids discovered from the endpoint.
    pub discovered: Vec<String>,
    /// Model ids in the row after the merge (discovered ∪ previously cached).
    pub total: usize,
    /// When set, the provider was skipped (fresh cache) or failed; the reason.
    pub note: Option<String>,
}

/// `Bun.hash(models).toString(36)` with a base-36 length prefix
/// (`model-cache.ts` `modelsHash`) — recomputed by TS readers on the bytes it
/// reads, so it only needs to be self-consistent with our serialization.
#[must_use]
pub fn models_hash(serialized: &str) -> String {
    format!(
        "{}:{}",
        radix(serialized.len() as u64, 36),
        radix(omp_stats::parser::wyhash::wyhash(serialized.as_bytes()), 36)
    )
}

fn radix(mut v: u64, base: u32) -> String {
    const DIGITS: &[u8; 36] = b"0123456789abcdefghijklmnopqrstuvwxyz";
    if v == 0 {
        return "0".to_owned();
    }
    let mut out = Vec::new();
    while v > 0 {
        out.push(DIGITS[usize::try_from(v % u64::from(base)).expect("fits")]);
        v /= u64::from(base);
    }
    out.reverse();
    String::from_utf8(out).expect("ascii")
}

/// Refresh every discoverable provider (or just `only`, when given) against
/// their live `/models` endpoints and update `models.db`.
///
/// Blocking; async call sites should use [`refresh_async`].
///
/// # Errors
/// Only when `only` names an unknown provider; per-provider HTTP failures are
/// reported in [`ProviderRefresh::note`] (TS treats discovery as best-effort).
pub fn refresh(dir: &AgentDir, only: Option<&str>) -> Result<Vec<ProviderRefresh>> {
    // Run on a dedicated plain thread with a scratch current-thread runtime:
    // the sync CLI path is often itself inside a tokio context, where both
    // `block_on` on the ambient handle and building a nested runtime panic.
    std::thread::scope(|scope| {
        scope
            .spawn(|| {
                tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()?
                    .block_on(refresh_run(dir, only))
            })
            .join()
            .expect("refresh thread panicked")
    })
}

/// [`refresh`] on the blocking thread pool (async call sites; a blocking
/// fetch inside an async fn parks an executor thread).
///
/// # Errors
/// See [`refresh`].
pub async fn refresh_async(dir: &AgentDir, only: Option<&str>) -> Result<Vec<ProviderRefresh>> {
    let dir = dir.clone();
    let only = only.map(str::to_owned);
    tokio::task::spawn_blocking(move || {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?
            .block_on(refresh_run(&dir, only.as_deref()))
    })
    .await?
}

async fn refresh_run(dir: &AgentDir, only: Option<&str>) -> Result<Vec<ProviderRefresh>> {
    let targets = discoverable_providers(dir)?;
    if let Some(only) = only {
        if !targets.iter().any(|(p, _, _)| p == only) {
            bail!("provider '{only}' has no base URL + credential to refresh");
        }
    }
    let db = dir.0.join("models.db");
    let client = reqwest::Client::builder()
        .timeout(FETCH_TIMEOUT)
        .build()
        .context("building http client")?;
    let mut out = Vec::new();
    for (provider, base_url, key) in targets {
        if only.is_some_and(|o| o != provider.as_str()) {
            continue;
        }
        let now = now_ms();
        if let Some(note) = fresh_skip(&db, &provider, now) {
            out.push(ProviderRefresh {
                provider,
                discovered: Vec::new(),
                total: 0,
                note: Some(note),
            });
            continue;
        }
        let url = append_models_path(&base_url);
        let mut req = client
            .get(&url)
            .header("Accept", "application/json")
            .timeout(FETCH_TIMEOUT);
        req = if base_url.contains("anthropic.com") {
            req.header("x-api-key", &key)
                .header("anthropic-version", "2023-06-01")
        } else {
            req.header("Authorization", format!("Bearer {key}"))
        };
        let discovered = match req.send().await {
            Ok(resp) if resp.status().is_success() => match resp.json::<Value>().await {
                Ok(payload) => extract_model_ids(&payload),
                Err(e) => {
                    out.push(ProviderRefresh {
                        provider,
                        discovered: vec![],
                        total: 0,
                        note: Some(format!("bad payload: {e}")),
                    });
                    continue;
                }
            },
            Ok(resp) => {
                let status = resp.status();
                out.push(ProviderRefresh {
                    provider,
                    discovered: vec![],
                    total: 0,
                    note: Some(format!("HTTP {status} from {url}")),
                });
                continue;
            }
            Err(e) => {
                out.push(ProviderRefresh {
                    provider,
                    discovered: vec![],
                    total: 0,
                    note: Some(format!("{e}")),
                });
                continue;
            }
        };
        let total = merge_and_write(&db, &provider, &base_url, &discovered, now);
        out.push(ProviderRefresh {
            provider,
            discovered,
            total,
            note: None,
        });
    }
    Ok(out)
}

/// Every provider with both a base URL and a credential: models.yml providers
/// (`baseUrl`, `apiKey`) plus the known built-ins with env/db credentials.
/// TS `#addImplicitDiscoverableProviders`; keyless providers (`local`, `web`,
/// `ollama`) are intentionally out of scope here.
fn discoverable_providers(dir: &AgentDir) -> Result<Vec<(String, String, String)>> {
    let mut out: Vec<(String, String, String)> = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    let yml = dir.0.join("models.yml");
    let providers = match std::fs::read_to_string(&yml) {
        Ok(s) if s.trim().is_empty() => Value::Null,
        Ok(s) => serde_yaml_ng::from_str::<Value>(&s)
            .with_context(|| format!("parsing {}", yml.display()))?
            .get("providers")
            .cloned()
            .unwrap_or(Value::Null),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Value::Null,
        Err(e) => return Err(e).with_context(|| format!("reading {}", yml.display())),
    };
    if let Some(map) = providers.as_object() {
        for (name, pcfg) in map {
            let Some(base) = pcfg.get("baseUrl").and_then(Value::as_str) else {
                continue;
            };
            let Some(key) = provider_key(dir, name, pcfg) else {
                continue;
            };
            if seen.insert(name.clone()) {
                out.push((name.clone(), base.trim_end_matches('/').to_owned(), key));
            }
        }
    }
    for (name, base) in builtin_base_urls() {
        if seen.contains(name) {
            continue;
        }
        if let Some(key) = builtin_key(dir, name) {
            out.push((name.to_owned(), base.to_owned(), key));
        }
    }
    Ok(out)
}

/// models.yml `apiKey`: `!command` stdout, a set env var name, else the literal.
fn provider_key(dir: &AgentDir, name: &str, pcfg: &Value) -> Option<String> {
    if let Some(raw) = pcfg.get("apiKey").and_then(Value::as_str) {
        if let Some(k) = crate::resolve_config_value(raw).filter(|k| !k.is_empty()) {
            return Some(k);
        }
    }
    pa_ai::env_api_keys::get_env_api_key(name)
        .or_else(|| crate::auth::credential(dir, name).map(|c| c.key))
}

/// Env var for providers whose omp id differs from pa-ai's, plus `agent.db`.
fn builtin_key(dir: &AgentDir, name: &str) -> Option<String> {
    let alias = match name {
        "kimi-code" => "KIMI_API_KEY",
        "opencode-zen" => "OPENCODE_API_KEY",
        "inception" => "INCEPTION_API_KEY",
        "atria" => "ATRIA_API_KEY",
        _ => "",
    };
    std::env::var(alias)
        .ok()
        .filter(|v| !v.is_empty())
        .or_else(|| pa_ai::env_api_keys::get_env_api_key(name))
        .or_else(|| crate::auth::credential(dir, name).map(|c| c.key))
}

fn builtin_base_urls() -> Vec<(&'static str, &'static str)> {
    vec![
        ("zai", "https://api.z.ai/api/coding/paas/v4"),
        ("kimi-code", "https://api.kimi.com/coding/v1"),
        ("openrouter", "https://openrouter.ai/api/v1"),
        ("opencode-zen", "http://127.0.0.1:8787/v1"),
        ("anthropic", "https://api.anthropic.com"),
        ("openai", "https://api.openai.com/v1"),
    ]
}

/// TS `appendModelsPath`: `/models` appended to the pathname so a query
/// string survives (`…/root?token=x` → `…/root/models?token=x`).
#[must_use]
pub fn append_models_path(base_url: &str) -> String {
    let (base, query) = match base_url.split_once('?') {
        Some((b, q)) => (b, Some(q)),
        None => (base_url, None),
    };
    let mut url = format!("{}/models", base.trim_end_matches('/'));
    if let Some(q) = query {
        url.push('?');
        url.push_str(q);
    }
    url
}

/// Model ids from a discovery payload: OpenAI `{"data": [{"id": …}]}`,
/// Ollama `{"models": [{"name"/"model": …}]}`, or a bare id array.
#[must_use]
pub fn extract_model_ids(payload: &Value) -> Vec<String> {
    let items = if payload.is_array() {
        payload.as_array().expect("checked").clone()
    } else {
        payload
            .get("data")
            .or_else(|| payload.get("models"))
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default()
    };
    let mut ids = Vec::new();
    for item in items {
        let id = item
            .as_str()
            .map(str::to_owned)
            .or_else(|| {
                item.get("id")
                    .or_else(|| item.get("name"))
                    .or_else(|| item.get("model"))
                    .and_then(Value::as_str)
                    .map(str::to_owned)
            })
            .filter(|id| !id.is_empty() && !is_non_chat_id(id));
        if let Some(id) = id {
            ids.push(id);
        }
    }
    ids.sort();
    ids.dedup();
    ids
}

/// TS `KIND_API_KINDS` proxies for raw /models listings (no kind metadata).
fn is_non_chat_id(id: &str) -> bool {
    let l = id.to_ascii_lowercase();
    [
        "embed",
        "whisper",
        "tts",
        "rerank",
        "moderation",
        "dall-e",
        "image",
    ]
    .iter()
    .any(|k| l.contains(k))
}

/// Why the fetch was skipped: an authoritative row fresher than the TTL, or a
/// non-authoritative row fresher than the retry window.
fn fresh_skip(db: &Path, provider: &str, now: u64) -> Option<String> {
    let conn = rusqlite::Connection::open_with_flags(
        db,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .ok()?;
    let (updated_at, authoritative): (i64, i64) = conn
        .query_row(
            "SELECT updated_at, authoritative FROM model_cache WHERE provider_id = ?1",
            [provider],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .ok()?;
    let age = now.saturating_sub(u64::try_from(updated_at).unwrap_or(0));
    let ttl = if authoritative == 1 {
        CACHE_TTL_MS
    } else {
        NON_AUTHORITATIVE_RETRY_MS
    };
    (age < ttl).then(|| format!("cache fresh ({age} ms old, ttl {ttl} ms)"))
}

/// One materialized `Model` row, valid for TS `isMaterializedModel` and the
/// Rust `Model` serde shape (see `pa_ai::compat::model_from_catalog_row`).
fn discovered_row(provider: &str, base_url: &str, id: &str) -> Value {
    let reasoning = is_reasoning_id(id);
    json!({
        "id": id,
        "name": id,
        "api": "openai-completions",
        "provider": provider,
        "baseUrl": base_url,
        "reasoning": reasoning,
        "input": ["text"],
        "contextWindow": DEFAULT_CONTEXT_WINDOW,
        "maxTokens": DEFAULT_MAX_TOKENS,
        "identity": {"class": "openai-compat"},
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        "supportsComputerUseConfig": null,
    })
}

/// Conservative reasoning guess for raw /models listings (no metadata).
fn is_reasoning_id(id: &str) -> bool {
    let l = id.to_ascii_lowercase();
    ["think", "reason", "r1", "o1", "o3", "o4"]
        .iter()
        .any(|k| l.contains(k))
}

/// Merge `discovered` ids with the provider's existing cached rows and write
/// the `model_cache` row (TS `writeModelCache` schema, always a payload
/// write). Returns the merged model count.
fn merge_and_write(
    db: &Path,
    provider: &str,
    base_url: &str,
    discovered: &[String],
    now: u64,
) -> usize {
    // Existing rows are additive (TS `mergeDiscoveredModel`): previously
    // discovered / catalog-merged models survive a refresh that no longer
    // lists them.
    let mut models: Vec<String> = db_models(db, provider);
    for id in discovered {
        if !models.iter().any(|m| m == id) {
            models.push(id.clone());
        }
    }
    models.sort();
    let rows: Vec<Value> = models
        .iter()
        .map(|id| discovered_row(provider, base_url, id))
        .collect();
    let serialized = serde_json::to_string(&rows).unwrap_or_else(|_| "[]".to_owned());
    write_row(db, provider, now, &serialized);
    models.len()
}

/// Previously cached ids for `provider` (tolerant parse).
fn db_models(db: &Path, provider: &str) -> Vec<String> {
    let Some(conn) = rusqlite::Connection::open_with_flags(
        db,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .ok() else {
        return Vec::new();
    };
    let Ok(serialized) = conn.query_row(
        "SELECT models FROM model_cache WHERE provider_id = ?1",
        [provider],
        |r| r.get::<_, String>(0),
    ) else {
        return Vec::new();
    };
    match serde_json::from_str::<Value>(&serialized) {
        Ok(Value::Array(rows)) => rows
            .iter()
            .filter_map(|m| m.get("id").and_then(Value::as_str).map(str::to_owned))
            .collect(),
        _ => Vec::new(),
    }
}

/// Write one `model_cache` row exactly as TS `writeModelCache` does (headers
/// are never persisted: `header_omitted_model_ids` stays `[]`, and no model
/// carries headers so `unrestorable` is `[]` too).
fn write_row(db: &Path, provider: &str, now: u64, serialized_models: &str) {
    if let Some(parent) = db.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let Ok(conn) = rusqlite::Connection::open(db) else {
        return;
    };
    let _ = conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS model_cache (
            provider_id TEXT PRIMARY KEY,
            version INTEGER NOT NULL,
            materialization_policy TEXT NOT NULL DEFAULT '',
            updated_at INTEGER NOT NULL,
            authoritative INTEGER NOT NULL DEFAULT 0,
            static_fingerprint TEXT NOT NULL DEFAULT '',
            header_omitted_model_ids TEXT NOT NULL DEFAULT '[]',
            unrestorable_header_model_ids TEXT NOT NULL DEFAULT '[]',
            header_restore_version INTEGER NOT NULL DEFAULT 0,
            models_hash TEXT NOT NULL DEFAULT '',
            models TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS model_cache_refresh (
            provider_id TEXT PRIMARY KEY,
            payload_updated_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL,
            authoritative INTEGER NOT NULL
        );",
    );
    let _ = conn.execute(
        "INSERT OR REPLACE INTO model_cache (
            provider_id, version, materialization_policy, updated_at, authoritative, static_fingerprint,
            header_omitted_model_ids, unrestorable_header_model_ids,
            header_restore_version, models_hash, models
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
        rusqlite::params![
            provider,
            CACHE_SCHEMA_VERSION,
            MATERIALIZATION_POLICY,
            now as i64,
            1, // authoritative: a direct /models fetch is provider-native truth
            "", // static_fingerprint
            "[]",
            "[]",
            HEADER_RESTORE_VERSION,
            models_hash(serialized_models),
            serialized_models,
        ],
    );
    // A payload write invalidates the side-table freshness row (TS does the same).
    let _ = conn.execute(
        "DELETE FROM model_cache_refresh WHERE provider_id = ?1",
        [provider],
    );
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir() -> (tempfile::TempDir, AgentDir) {
        let tmp = tempfile::tempdir().expect("tempdir");
        let dir = AgentDir(tmp.path().to_path_buf());
        (tmp, dir)
    }

    /// Minimal HTTP/1.1 server for one request, serving `body` on `path`.
    fn mock_server(
        path: &'static str,
        body: &'static str,
    ) -> (String, std::thread::JoinHandle<()>) {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().expect("addr").port();
        let handle = std::thread::spawn(move || {
            let (mut sock, _) = listener.accept().expect("accept");
            let mut buf = [0u8; 4096];
            let _ = sock.read(&mut buf);
            let req = String::from_utf8_lossy(&buf);
            assert!(
                req.starts_with(&format!("GET {path}")),
                "unexpected request: {req}"
            );
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = sock.write_all(resp.as_bytes());
        });
        (format!("http://127.0.0.1:{port}/v1"), handle)
    }

    #[test]
    fn append_models_path_preserves_query() {
        assert_eq!(append_models_path("https://h/v1"), "https://h/v1/models");
        assert_eq!(
            append_models_path("https://h/root?token=x"),
            "https://h/root/models?token=x"
        );
        assert_eq!(append_models_path("https://h/v1/"), "https://h/v1/models");
    }

    #[test]
    fn extracts_ids_from_payload_shapes() {
        let openai = json!({"data": [{"id": "m-2"}, {"id": "text-embedding-3"}, {"id": "m-1"}]});
        assert_eq!(extract_model_ids(&openai), vec!["m-1", "m-2"]);
        let ollama = json!({"models": [{"name": "llama4"}, {"model": "qwen3"}]});
        assert_eq!(extract_model_ids(&ollama), vec!["llama4", "qwen3"]);
        let bare = json!(["solo"]);
        assert_eq!(extract_model_ids(&bare), vec!["solo"]);
    }

    #[test]
    fn refresh_writes_ts_schema_row_and_merges() {
        let body = r#"{"data":[{"id":"mock-new-model"},{"id":"mock-older"}]}"#;
        let (base, server) = mock_server("/v1/models", body);
        let (_tmp, dir) = temp_dir();
        std::fs::write(
            dir.0.join("models.yml"),
            format!("providers:\n  mockprov:\n    baseUrl: {base}\n    apiKey: test-key\n"),
        )
        .expect("write models.yml");

        let report = refresh(&dir, Some("mockprov")).expect("refresh");
        server.join().expect("server thread");
        let r = &report[0];
        assert_eq!(r.provider, "mockprov");
        assert!(r.note.is_none(), "unexpected failure: {:?}", r.note);
        assert_eq!(r.discovered, vec!["mock-new-model", "mock-older"]);

        // Row shape: policy, version, hash prefix and merged payload.
        let conn = rusqlite::Connection::open(dir.0.join("models.db")).expect("open");
        let (policy, version, hash, authoritative, models): (String, i64, String, i64, String) = conn
            .query_row(
                "SELECT materialization_policy, version, models_hash, authoritative, models FROM model_cache WHERE provider_id = 'mockprov'",
                [],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)),
            )
            .expect("row");
        assert_eq!(policy, MATERIALIZATION_POLICY);
        assert_eq!(version, CACHE_SCHEMA_VERSION);
        assert_eq!(authoritative, 1);
        let prefix = hash.split(':').next().unwrap_or_default();
        let expected_len = u64::from_str_radix(prefix, 36).expect("base36 length");
        assert_eq!(expected_len, models.len() as u64, "hash prefix {hash}");
        let rows: Vec<Value> = serde_json::from_str(&models).expect("parse models");
        let ids: Vec<&str> = rows
            .iter()
            .filter_map(|m| m.get("id").and_then(Value::as_str))
            .collect();
        assert_eq!(ids, vec!["mock-new-model", "mock-older"]);
        for row in &rows {
            // TS `isMaterializedModel` required fields.
            for key in [
                "id",
                "name",
                "api",
                "provider",
                "baseUrl",
                "reasoning",
                "input",
                "identity",
                "cost",
                "supportsComputerUseConfig",
            ] {
                assert!(row.get(key).is_some(), "row missing {key}: {row}");
            }
            assert_eq!(row["input"], json!(["text"]));
            assert_eq!(
                row["cost"],
                json!({"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0})
            );
        }
    }

    #[test]
    fn second_refresh_within_ttl_skips_network() {
        let body = r#"{"data":[{"id":"mock-m"}]}"#;
        let (base, server) = mock_server("/v1/models", body);
        let (_tmp, dir) = temp_dir();
        std::fs::write(
            dir.0.join("models.yml"),
            format!("providers:\n  mockprov:\n    baseUrl: {base}\n    apiKey: k\n"),
        )
        .expect("write models.yml");
        refresh(&dir, Some("mockprov")).expect("first refresh");
        server.join().expect("server");
        let report = refresh(&dir, Some("mockprov")).expect("second refresh");
        assert_eq!(report[0].discovered, Vec::<String>::new());
        assert!(report[0]
            .note
            .as_deref()
            .unwrap_or("")
            .starts_with("cache fresh"));
    }

    #[test]
    fn ttl_windows_match_ts_constants() {
        assert_eq!(CACHE_TTL_MS, 7_200_000);
        assert_eq!(NON_AUTHORITATIVE_RETRY_MS, 300_000);
        assert_eq!(DEFAULT_CONTEXT_WINDOW, 128_000);
        assert_eq!(DEFAULT_MAX_TOKENS, 32_768);
        assert!(!MATERIALIZATION_POLICY.is_empty());
        assert!(MATERIALIZATION_POLICY.starts_with("app-"));
        assert!(MATERIALIZATION_POLICY.contains(":builder-1:rules-"));
    }
}
