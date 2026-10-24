use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};

use super::*;

/// A canned HTTP/1.1 server: `(method, path+query, headers)` of every request is recorded and the
/// reply comes from `reply`. Runs on a detached thread for the rest of the test process.
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
                let text = String::from_utf8_lossy(&buf).to_string();
                if let Some(head_end) = text.find("\r\n\r\n") {
                    let len = text
                        .lines()
                        .find_map(|l| {
                            l.to_ascii_lowercase()
                                .strip_prefix("content-length:")
                                .map(|v| v.trim().parse::<usize>().unwrap_or(0))
                        })
                        .unwrap_or(0);
                    if buf.len() >= head_end + 4 + len {
                        break;
                    }
                }
                if n == 0 {
                    break;
                }
            }
            let text = String::from_utf8_lossy(&buf).to_string();
            let line = text.lines().next().unwrap_or_default().to_string();
            let mut parts = line.split(' ');
            let key = format!(
                "{} {}",
                parts.next().unwrap_or_default(),
                parts.next().unwrap_or_default()
            );
            log.lock().unwrap().push(text.clone());
            let (status, body) = reply(&key);
            let _ = write!(
                stream,
                "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
        }
    });
    (url, seen)
}

fn requests(seen: &Arc<Mutex<Vec<String>>>) -> Vec<String> {
    seen.lock()
        .unwrap()
        .iter()
        .map(|r| r.lines().next().unwrap_or_default().to_string())
        .collect()
}

fn agent_db(rows: &[(&str, &str, &str)]) -> (tempfile::TempDir, AgentDir) {
    let t = tempfile::tempdir().unwrap();
    let c = rusqlite::Connection::open(t.path().join("agent.db")).unwrap();
    c.execute_batch(
        "CREATE TABLE auth_credentials (id INTEGER PRIMARY KEY AUTOINCREMENT, provider TEXT NOT NULL, credential_type TEXT NOT NULL, data TEXT NOT NULL, disabled_cause TEXT DEFAULT NULL, identity_key TEXT DEFAULT NULL, created_at INTEGER NOT NULL DEFAULT 0, updated_at INTEGER NOT NULL DEFAULT 0);",
    )
    .unwrap();
    for (provider, kind, data) in rows {
        c.execute(
            "INSERT INTO auth_credentials(provider, credential_type, data) VALUES (?1, ?2, ?3)",
            [provider, kind, data],
        )
        .unwrap();
    }
    let dir = AgentDir(t.path().to_path_buf());
    (t, dir)
}

fn in_ms(delta: i64) -> i64 {
    now_ms() as i64 + delta
}

#[test]
fn broker_config_follows_env_config_then_token_file() {
    let t = tempfile::tempdir().unwrap();
    let agent = t.path().join("agent");
    std::fs::create_dir(&agent).unwrap();
    let dir = AgentDir(agent.clone());
    let none = |_: &str| None;
    assert_eq!(resolve_broker_config(&dir, &none).unwrap(), None);

    // URL without any token is an error, not a silent local fallback.
    let url_only = |k: &str| (k == "OMP_AUTH_BROKER_URL").then(|| "http://b:1".to_string());
    assert!(resolve_broker_config(&dir, &url_only)
        .unwrap_err()
        .to_string()
        .contains("no bearer token"));

    // The token file pairs with an env URL.
    std::fs::write(t.path().join("auth-broker.token"), "filetok\n").unwrap();
    assert_eq!(
        resolve_broker_config(&dir, &url_only).unwrap(),
        Some(BrokerConfig {
            url: "http://b:1".into(),
            token: "filetok".into()
        })
    );

    // Nested config.yml entries beat the token file; flat dotted keys work too.
    std::fs::write(
        agent.join("config.yml"),
        "auth:\n  broker:\n    url: http://cfg:2\n    token: cfgtok\n",
    )
    .unwrap();
    assert_eq!(
        resolve_broker_config(&dir, &none).unwrap(),
        Some(BrokerConfig {
            url: "http://cfg:2".into(),
            token: "cfgtok".into()
        })
    );
    std::fs::write(
        agent.join("config.yml"),
        "\"auth.broker.url\": http://flat:3\n",
    )
    .unwrap();
    assert_eq!(
        resolve_broker_config(&dir, &none).unwrap(),
        Some(BrokerConfig {
            url: "http://flat:3".into(),
            token: "filetok".into()
        })
    );

    // Env wins over config for both values.
    let env = |k: &str| match k {
        "OMP_AUTH_BROKER_URL" => Some("http://env:4".to_string()),
        "OMP_AUTH_BROKER_TOKEN" => Some("envtok".to_string()),
        _ => None,
    };
    assert_eq!(
        resolve_broker_config(&dir, &env).unwrap(),
        Some(BrokerConfig {
            url: "http://env:4".into(),
            token: "envtok".into()
        })
    );
}

#[test]
fn broker_refreshes_a_near_expiry_oauth_row_and_recovers_after_401() {
    let (url, seen) = serve(|req| {
        match req {
        "GET /v1/snapshot" => (
            200,
            serde_json::json!({"generation": 1, "credentials": [
                {"id": 3, "provider": "other", "credential": {"type": "api_key", "key": "k-other"}},
                {"id": 25, "provider": "kimi-code",
                 "credential": {"type": "oauth", "access": "stale", "refresh": "__remote__", "expires": now_ms() as i64 + 5_000}}
            ]})
            .to_string(),
        ),
        "POST /v1/credential/25/refresh" => (
            200,
            serde_json::json!({"entry": {"id": 25, "provider": "kimi-code",
                "credential": {"type": "oauth", "access": "fresh-1", "refresh": "__remote__", "expires": now_ms() as i64 + 900_000}}})
            .to_string(),
        ),
        "POST /v1/credential/25/refresh?reason=auth-recovery" => (
            200,
            serde_json::json!({"entry": {"id": 25, "provider": "kimi-code",
                "credential": {"type": "oauth", "access": "fresh-2", "refresh": "__remote__", "expires": now_ms() as i64 + 900_000}}})
            .to_string(),
        ),
        _ => (404, "{}".into()),
    }
    });
    let cfg = BrokerConfig {
        url,
        token: "secret".into(),
    };
    let dir = AgentDir(std::env::temp_dir());
    let hosts = Hosts::default();

    let got = block_on(|| resolve_with(&dir, "kimi-code", Some(&cfg), None, &hosts))
        .unwrap()
        .unwrap();
    assert_eq!(
        got,
        Credential {
            key: "fresh-1".into(),
            id: 25
        }
    );
    assert_eq!(
        requests(&seen),
        [
            "GET /v1/snapshot HTTP/1.1",
            "POST /v1/credential/25/refresh HTTP/1.1"
        ]
    );
    let first = seen.lock().unwrap()[0].to_ascii_lowercase();
    assert!(first.contains("authorization: bearer secret"), "{first}");

    // A 401 recovery refreshes the failing row with the auth-recovery reason.
    let recovered = block_on(|| resolve_with(&dir, "kimi-code", Some(&cfg), Some(25), &hosts))
        .unwrap()
        .unwrap();
    assert_eq!(recovered.key, "fresh-2");
    assert!(requests(&seen)
        .last()
        .unwrap()
        .starts_with("POST /v1/credential/25/refresh?reason=auth-recovery"));

    // api-key rows come straight from the snapshot.
    let other = block_on(|| resolve_with(&dir, "other", Some(&cfg), None, &hosts))
        .unwrap()
        .unwrap();
    assert_eq!(other.key, "k-other");
}

#[test]
fn broker_failure_is_an_error_not_a_local_fallback() {
    let (url, _seen) = serve(|_| (500, "{}".into()));
    let cfg = BrokerConfig {
        url,
        token: "t".into(),
    };
    let (_t, dir) = agent_db(&[("kimi-code", "api_key", r#"{"key":"local"}"#)]);
    let hosts = Hosts::default();
    assert!(block_on(|| resolve_with(&dir, "kimi-code", Some(&cfg), None, &hosts)).is_err());
}

#[test]
fn local_store_refreshes_an_expired_kimi_token_and_writes_it_back() {
    let (url, seen) = serve(|req| {
        match req {
        "POST /api/oauth/token" => (
            200,
            serde_json::json!({"access_token": "new-access", "refresh_token": "new-refresh", "expires_in": 900}).to_string(),
        ),
        _ => (404, "{}".into()),
    }
    });
    let hosts = Hosts { kimi: Some(url) };
    let data = serde_json::json!({"access": "old-access", "refresh": "old-refresh", "expires": in_ms(-1_000),
        "accountId": "acct", "authorizedAt": "t0"})
    .to_string();
    let (t, dir) = agent_db(&[("kimi-code", "oauth", &data)]);
    let db = t.path().join("agent.db");

    let got = block_on(|| local_resolve(&db, "kimi-code", None, &hosts))
        .unwrap()
        .unwrap();
    assert_eq!(
        got,
        Credential {
            key: "new-access".into(),
            id: 1
        }
    );

    // The grant is a form refresh with the client id and the stored refresh token.
    let req = seen.lock().unwrap()[0].clone();
    assert!(
        req.to_ascii_lowercase()
            .contains("content-type: application/x-www-form-urlencoded"),
        "{req}"
    );
    assert!(req.contains("grant_type=refresh_token"), "{req}");
    assert!(req.contains("refresh_token=old-refresh"), "{req}");
    assert!(
        req.contains("client_id=17e5f671-d194-4dfb-9706-5516cb48c098"),
        "{req}"
    );

    // Write-back: refreshed fields replace the row's, everything else and the key order stay.
    let conn = rusqlite::Connection::open(&db).unwrap();
    let stored: String = conn
        .query_row("SELECT data FROM auth_credentials WHERE id = 1", [], |r| {
            r.get(0)
        })
        .unwrap();
    let v: Value = serde_json::from_str(&stored).unwrap();
    assert_eq!(v["access"], "new-access");
    assert_eq!(v["refresh"], "new-refresh");
    assert_eq!(v["accountId"], "acct");
    assert_eq!(v["authorizedAt"], "t0");
    let expires = v["expires"].as_i64().unwrap();
    assert!(
        (expires - (in_ms(900_000) - 300_000)).abs() < 10_000,
        "expires {expires}"
    );
    assert!(
        stored.starts_with(r#"{"access""#),
        "key order kept: {stored}"
    );
    let leases: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM auth_credential_refresh_leases",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(leases, 0, "lease released");

    // Now fresh: served from the row, no second token request.
    let again = block_on(|| resolve_with(&dir, "kimi-code", None, None, &hosts))
        .unwrap()
        .unwrap();
    assert_eq!(again.key, "new-access");
    assert_eq!(requests(&seen).len(), 1);

    // 401 recovery re-uses the token this process just minted instead of burning the refresh token.
    let recovered = block_on(|| local_resolve(&db, "kimi-code", Some(1), &hosts))
        .unwrap()
        .unwrap();
    assert_eq!(recovered.key, "new-access");
    assert_eq!(requests(&seen).len(), 1);
}

#[test]
fn local_forced_refresh_mints_a_new_bearer_for_a_foreign_token() {
    let (url, seen) = serve(|_| {
        (
            200,
            serde_json::json!({"access_token": "minted", "expires_in": 900}).to_string(),
        )
    });
    let hosts = Hosts { kimi: Some(url) };
    // Still valid on paper (server-side revoked): only a forced refresh replaces it.
    let data = serde_json::json!({"access": "revoked", "refresh": "r", "expires": in_ms(600_000)})
        .to_string();
    let (t, _dir) = agent_db(&[("kimi-code", "oauth", &data)]);
    let db = t.path().join("agent.db");
    assert_eq!(
        block_on(|| local_resolve(&db, "kimi-code", None, &hosts))
            .unwrap()
            .unwrap()
            .key,
        "revoked"
    );
    assert!(requests(&seen).is_empty());
    let forced = block_on(|| local_resolve(&db, "kimi-code", Some(1), &hosts))
        .unwrap()
        .unwrap();
    assert_eq!(forced.key, "minted");
    // A missing `refresh_token` in the response keeps the stored one.
    let conn = rusqlite::Connection::open(&db).unwrap();
    let stored: String = conn
        .query_row("SELECT data FROM auth_credentials WHERE id = 1", [], |r| {
            r.get(0)
        })
        .unwrap();
    assert_eq!(
        serde_json::from_str::<Value>(&stored).unwrap()["refresh"],
        "r"
    );
}

#[test]
fn failed_refresh_keeps_a_still_valid_token_and_drops_an_expired_one() {
    let (url, _seen) = serve(|_| (400, r#"{"error":"invalid_grant"}"#.into()));
    let hosts = Hosts { kimi: Some(url) };
    let soon =
        serde_json::json!({"access": "soon", "refresh": "r", "expires": in_ms(30_000)}).to_string();
    let gone = serde_json::json!({"access": "gone", "refresh": "r", "expires": in_ms(-30_000)})
        .to_string();
    let (t, _d) = agent_db(&[("kimi-code", "oauth", &soon), ("zzz", "oauth", &gone)]);
    let db = t.path().join("agent.db");
    assert_eq!(
        block_on(|| local_resolve(&db, "kimi-code", None, &hosts))
            .unwrap()
            .unwrap()
            .key,
        "soon"
    );
    // No refresh flow for the provider and the token is expired: no credential.
    assert_eq!(
        block_on(|| local_resolve(&db, "zzz", None, &hosts)).unwrap(),
        None
    );
}
