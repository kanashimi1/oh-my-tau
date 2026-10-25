use std::io::{Read, Write};
use std::net::TcpListener;
use std::pin::Pin;
use std::sync::Arc;

use pa_ai::oauth::{
    OAuthLoginUi, OAuthPrompt, ProviderHttp, ProviderHttpMethod, ProviderHttpRequest,
    ProviderHttpResponse,
};
use parking_lot::Mutex;
use serde_json::{json, Value};

use super::*;

/// Scripted UI: answers prompts from a queue and records what it was shown.
#[derive(Default)]
struct ScriptedUi {
    answers: Mutex<Vec<String>>,
    auth: Mutex<Vec<(String, Option<String>)>>,
    prompts: Mutex<Vec<OAuthPrompt>>,
    progress: Mutex<Vec<String>>,
}

impl ScriptedUi {
    fn with(answers: &[&str]) -> Arc<Self> {
        let ui = Self::default();
        *ui.answers.lock() = answers.iter().rev().map(ToString::to_string).collect();
        Arc::new(ui)
    }
}

impl OAuthLoginUi for ScriptedUi {
    fn on_auth(&self, url: &str, instructions: Option<&str>) {
        self.auth
            .lock()
            .push((url.to_string(), instructions.map(str::to_string)));
    }
    fn on_prompt(
        &self,
        prompt: &OAuthPrompt,
    ) -> Pin<Box<dyn Future<Output = Option<String>> + Send + '_>> {
        self.prompts.lock().push(prompt.clone());
        let answer = self.answers.lock().pop();
        Box::pin(async move { answer })
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

/// Records requests, answers from a closure.
struct ScriptedHttp {
    seen: Mutex<Vec<ProviderHttpRequest>>,
    reply: Box<dyn Fn(&ProviderHttpRequest) -> (u16, String) + Send + Sync>,
}

impl ProviderHttp for ScriptedHttp {
    fn request(
        &self,
        request: ProviderHttpRequest,
        _timeout_ms: u64,
    ) -> Pin<Box<dyn Future<Output = Result<ProviderHttpResponse, String>> + Send + '_>> {
        let (status, body) = (self.reply)(&request);
        self.seen.lock().push(request);
        Box::pin(async move { Ok(ProviderHttpResponse { status, body }) })
    }
}

fn services(
    reply: impl Fn(&ProviderHttpRequest) -> (u16, String) + Send + Sync + 'static,
) -> (Services, Arc<ScriptedHttp>) {
    let http = Arc::new(ScriptedHttp {
        seen: Mutex::new(Vec::new()),
        reply: Box::new(reply),
    });
    struct Shared(Arc<ScriptedHttp>);
    impl ProviderHttp for Shared {
        fn request(
            &self,
            request: ProviderHttpRequest,
            timeout_ms: u64,
        ) -> Pin<Box<dyn Future<Output = Result<ProviderHttpResponse, String>> + Send + '_>>
        {
            self.0.request(request, timeout_ms)
        }
    }
    let services = Services {
        http: Box::new(Shared(http.clone())),
        codex_http: Box::new(pa_ai::oauth::ReqwestCodexHttp::new()),
        hosts: Hosts::default(),
        env: Box::new(|_| None),
    };
    (services, http)
}

fn temp_target() -> (tempfile::TempDir, Target) {
    let dir = tempfile::tempdir().unwrap();
    let target = Target::Local(dir.path().join("agent").join("agent.db"));
    (dir, target)
}

fn raw_rows(target: &Target) -> Vec<(i64, String, String, String, Option<String>, Option<String>)> {
    let Target::Local(db) = target else {
        unreachable!()
    };
    let conn = rusqlite::Connection::open(db).unwrap();
    let mut stmt = conn
        .prepare("SELECT id, provider, credential_type, data, disabled_cause, identity_key FROM auth_credentials ORDER BY id")
        .unwrap();
    stmt.query_map([], |r| {
        Ok((
            r.get(0)?,
            r.get(1)?,
            r.get(2)?,
            r.get(3)?,
            r.get(4)?,
            r.get(5)?,
        ))
    })
    .unwrap()
    .map(Result::unwrap)
    .collect()
}

#[tokio::test]
async fn api_key_login_validates_then_stores_a_ts_login_row() {
    let (_dir, target) = temp_target();
    let (services, http) = services(|_| (200, "{}".into()));
    let ui = ScriptedUi::with(&["  sk-test-1  "]);
    let identity = login(&target, "cerebras", ui.clone(), &services)
        .await
        .unwrap();
    assert_eq!(identity, Some(Identity::ApiKey));

    let seen = http.seen.lock().clone();
    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0].url, "https://api.cerebras.ai/v1/chat/completions");
    assert!(seen[0]
        .headers
        .contains(&("Authorization".to_string(), "Bearer sk-test-1".to_string())));
    let body: Value = serde_json::from_str(seen[0].body.as_deref().unwrap()).unwrap();
    assert_eq!(body["model"], "gpt-oss-120b");
    assert_eq!(body["max_tokens"], 1);
    assert_eq!(ui.prompts.lock()[0].message, "Paste your Cerebras API key");
    assert_eq!(ui.progress.lock()[0], "Validating API key...");
    // The key page is shown first.
    assert_eq!(ui.auth.lock()[0].0, "https://cloud.cerebras.ai/platform/");

    let rows = raw_rows(&target);
    assert_eq!(rows.len(), 1);
    assert_eq!(
        (rows[0].1.as_str(), rows[0].2.as_str()),
        ("cerebras", "api_key")
    );
    assert_eq!(rows[0].3, r#"{"key":"sk-test-1","source":"login"}"#);
    assert_eq!((rows[0].4.clone(), rows[0].5.clone()), (None, None));

    // The same key again updates the row instead of duplicating it.
    let ui = ScriptedUi::with(&["sk-test-1"]);
    login(&target, "cerebras", ui, &services).await.unwrap();
    assert_eq!(raw_rows(&target).len(), 1);
    // A different key is a second active row (TS only replaces an identical key).
    let ui = ScriptedUi::with(&["sk-test-2"]);
    login(&target, "cerebras", ui, &services).await.unwrap();
    assert_eq!(raw_rows(&target).len(), 2);
}

#[tokio::test]
async fn api_key_rejected_by_the_probe_stores_nothing() {
    let (_dir, target) = temp_target();
    let (services, _) = services(|_| (401, r#"{"error":"bad key"}"#.into()));
    let err = login(&target, "cerebras", ScriptedUi::with(&["nope"]), &services)
        .await
        .unwrap_err();
    assert_eq!(
        err,
        r#"Cerebras API key validation failed (401): {"error":"bad key"}"#
    );
    let Target::Local(db) = &target else {
        unreachable!()
    };
    assert!(!db.exists());
}

#[tokio::test]
async fn optional_probe_failure_is_skipped_unless_it_is_an_auth_failure() {
    let (_dir, target) = temp_target();
    // commandcode: optional probe, trust-forbidden.
    let (services, _) = services(|_| (500, "boom".into()));
    let ui = ScriptedUi::with(&["user_1"]);
    assert_eq!(
        login(&target, "commandcode", ui.clone(), &services)
            .await
            .unwrap(),
        Some(Identity::ApiKey)
    );
    assert_eq!(
        ui.progress.lock()[1],
        "Skipping Command Code validation endpoint; continuing with provided API key."
    );

    let (services, _) = services_with_status(403);
    assert_eq!(
        login(
            &target,
            "commandcode",
            ScriptedUi::with(&["user_2"]),
            &services
        )
        .await
        .unwrap(),
        Some(Identity::ApiKey)
    );
    let (services, _) = services_with_status(401);
    let err = login(
        &target,
        "commandcode",
        ScriptedUi::with(&["user_3"]),
        &services,
    )
    .await
    .unwrap_err();
    assert!(
        err.starts_with("Command Code API key validation failed (401)"),
        "{err}"
    );
}

fn services_with_status(status: u16) -> (Services, Arc<ScriptedHttp>) {
    services(move |_| (status, "x".into()))
}

#[tokio::test]
async fn empty_key_cancel_and_fallback_follow_the_ts_engine() {
    let (_dir, target) = temp_target();
    let (services, _) = services(|_| (200, "{}".into()));
    assert_eq!(
        login(&target, "exa", ScriptedUi::with(&["  "]), &services)
            .await
            .unwrap_err(),
        "API key is required"
    );
    assert_eq!(
        login(&target, "exa", ScriptedUi::with(&[]), &services)
            .await
            .unwrap_err(),
        "Login cancelled"
    );
    // ollama: empty fallback "" = nothing stored; llama.cpp: stores the fallback key.
    assert_eq!(
        login(&target, "ollama", ScriptedUi::with(&[""]), &services)
            .await
            .unwrap(),
        None
    );
    assert!(raw_rows(&target_after_touch(&target)).is_empty());
    assert_eq!(
        login(&target, "llama.cpp", ScriptedUi::with(&[""]), &services)
            .await
            .unwrap(),
        Some(Identity::ApiKey)
    );
    assert_eq!(
        raw_rows(&target)[0].3,
        r#"{"key":"llama-cpp-local","source":"login"}"#
    );
    // strip-bearer (deepseek)
    login(
        &target,
        "deepseek",
        ScriptedUi::with(&["Bearer sk-ds"]),
        &services,
    )
    .await
    .unwrap();
    assert!(raw_rows(&target)
        .iter()
        .any(|r| r.3 == r#"{"key":"sk-ds","source":"login"}"#));
}

/// `raw_rows` needs the db to exist; the ollama case may not have created it.
fn target_after_touch(target: &Target) -> Target {
    let Target::Local(db) = target else {
        unreachable!()
    };
    if !db.exists() {
        LocalStore::open(db).unwrap();
    }
    target.clone()
}

#[tokio::test]
async fn unknown_and_unported_providers_fail_with_a_clear_message() {
    let (_dir, target) = temp_target();
    let (services, _) = services(|_| (200, "{}".into()));
    let err = login(&target, "nope", ScriptedUi::with(&[]), &services)
        .await
        .unwrap_err();
    assert_eq!(err, "Unknown OAuth provider: nope");
    let err = login(&target, "cursor", ScriptedUi::with(&[]), &services)
        .await
        .unwrap_err();
    assert!(err.contains("not ported to omt yet"), "{err}");
}

#[test]
fn every_listed_provider_is_either_supported_or_explicitly_unported() {
    for (id, _) in providers::PROVIDERS {
        assert!(
            is_supported(id) ^ providers::UNPORTED.contains(&id),
            "{id}: supported={} unported={}",
            is_supported(id),
            providers::UNPORTED.contains(&id)
        );
    }
    for id in api_key::ids() {
        assert!(
            providers::display_name(id).is_some(),
            "{id} missing from the roster"
        );
    }
}

/// A one-shot mock Kimi OAuth host: device authorization then an immediately-complete token poll.
fn mock_kimi_host(access: &str) -> (String, Arc<Mutex<Vec<String>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let seen = Arc::new(Mutex::new(Vec::new()));
    let log = seen.clone();
    let access = access.to_string();
    let host = base.clone();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let mut buf = vec![0u8; 8192];
            let n = stream.read(&mut buf).unwrap_or(0);
            let text = String::from_utf8_lossy(&buf[..n]).to_string();
            let path = text
                .split_whitespace()
                .nth(1)
                .unwrap_or_default()
                .to_string();
            log.lock().push(text.clone());
            let body = if path.ends_with("/api/oauth/device_authorization") {
                json!({
                    "user_code": "ABCD-1234",
                    "device_code": "dev-code",
                    "verification_uri": format!("{host}/verify"),
                    "verification_uri_complete": format!("{host}/verify?code=ABCD-1234"),
                    "interval": 1,
                    "expires_in": 60
                })
            } else {
                json!({"access_token": access, "refresh_token": "ref-1", "expires_in": 3600})
            };
            let body = body.to_string();
            let _ = write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
        }
    });
    (base, seen)
}

fn jwt(payload: &Value) -> String {
    use base64::Engine as _;
    let enc = |v: &str| base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(v);
    format!(
        "{}.{}.sig",
        enc(r#"{"alg":"none"}"#),
        enc(&payload.to_string())
    )
}

#[tokio::test]
async fn kimi_device_login_against_a_mock_host_writes_a_ts_oauth_row() {
    let access = jwt(&json!({"user_id": "user-42"}));
    let (base, seen) = mock_kimi_host(&access);
    let (_dir, target) = temp_target();
    let mut services = Services::production();
    services.hosts = Hosts {
        kimi: Some(base.clone()),
    };
    let ui = ScriptedUi::with(&[]);
    let before = crate::now_ms();
    let identity = login(&target, "kimi-code", ui.clone(), &services)
        .await
        .unwrap();
    assert_eq!(
        identity,
        Some(Identity::OAuth {
            email: None,
            account_id: Some("user-42".into()),
            org_id: None,
            org_name: None
        })
    );

    // Device authorization request, then the device-code token poll, both form-encoded.
    let requests = seen.lock();
    assert!(requests[0].starts_with("POST /api/oauth/device_authorization"));
    assert!(requests[0].contains("client_id=17e5f671-d194-4dfb-9706-5516cb48c098"));
    assert!(requests[1].starts_with("POST /api/oauth/token"));
    assert!(
        requests[1].contains("grant_type=urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Adevice_code")
    );
    assert!(requests[1].contains("device_code=dev-code"));
    assert!(requests[0]
        .to_lowercase()
        .contains("x-msh-platform: kimi_cli"));
    // The verification URL goes to the UI with the user-code instructions.
    assert_eq!(
        ui.auth.lock()[0],
        (
            format!("{base}/verify?code=ABCD-1234"),
            Some("Enter code: ABCD-1234".to_string())
        )
    );

    let rows = raw_rows(&target);
    assert_eq!(rows.len(), 1);
    assert_eq!(
        (rows[0].1.as_str(), rows[0].2.as_str()),
        ("kimi-code", "oauth")
    );
    assert_eq!(rows[0].5.as_deref(), Some("account:user-42"));
    let data: Value = serde_json::from_str(&rows[0].3).unwrap();
    let keys: Vec<&str> = data
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(
        keys,
        ["access", "refresh", "expires", "accountId", "authorizedAt"]
    );
    assert_eq!(data["access"], access);
    assert_eq!(data["refresh"], "ref-1");
    let expires = data["expires"].as_u64().unwrap();
    assert!(
        expires + 3_600_000
            > before
                .saturating_add(3_600_000)
                .saturating_sub(300_000 + 5000)
            && expires < before + 3_700_000
    );
    assert!(data["authorizedAt"].as_u64().unwrap() >= before);
}

#[test]
fn oauth_upsert_matches_by_identity_and_keeps_org_scoped_rows_apart() {
    let (_dir, target) = temp_target();
    let Target::Local(db) = &target else {
        unreachable!()
    };
    let mut store = LocalStore::open(db).unwrap();
    let oauth = |email: &str, org: &str, access: &str| {
        let mut data = Map::new();
        data.insert("access".into(), json!(access));
        data.insert("refresh".into(), json!("r"));
        data.insert("expires".into(), json!(1));
        data.insert("email".into(), json!(email));
        data.insert("accountId".into(), json!("acct"));
        data.insert("orgId".into(), json!(org));
        data.insert("orgName".into(), json!(format!("Org {org}")));
        Credential::OAuth(data)
    };
    store
        .upsert("anthropic", &oauth("A@x.io", "o1", "t1"))
        .unwrap();
    // Same account + org: updated in place.
    let rows = store
        .upsert("anthropic", &oauth("a@x.io", "o1", "t2"))
        .unwrap();
    assert_eq!(rows.len(), 1);
    // Same email, other org: a second subscription.
    let rows = store
        .upsert("anthropic", &oauth("a@x.io", "o2", "t3"))
        .unwrap();
    assert_eq!(rows.len(), 2);
    let raw = raw_rows(&target);
    assert_eq!(raw[0].5.as_deref(), Some("email:a@x.io|org:o1"));
    assert_eq!(raw[1].5.as_deref(), Some("email:a@x.io|org:o2"));
    assert_eq!(rows[0].label(), "a@x.io (Org o1)");

    // An api key present is retired by an OAuth login.
    store
        .upsert("openai-codex", &Credential::ApiKey("k".into()))
        .unwrap();
    let rows = store
        .upsert("openai-codex", &oauth("c@x.io", "w", "t"))
        .unwrap();
    assert_eq!(rows.len(), 1);
    let raw = raw_rows(&target);
    let retired = raw
        .iter()
        .find(|r| r.1 == "openai-codex" && r.2 == "api_key")
        .unwrap();
    assert_eq!(retired.4.as_deref(), Some("replaced by oauth login"));
}

#[tokio::test]
async fn logout_soft_deletes_with_the_ts_causes_and_a_relogin_purges_the_tombstone() {
    let (_dir, target) = temp_target();
    let (services, _) = services(|_| (200, "{}".into()));
    login(&target, "exa", ScriptedUi::with(&["k1"]), &services)
        .await
        .unwrap();
    login(&target, "exa", ScriptedUi::with(&["k2"]), &services)
        .await
        .unwrap();
    login(&target, "tavily", ScriptedUi::with(&["t1"]), &services)
        .await
        .unwrap();
    assert_eq!(target.providers().await.unwrap(), ["exa", "tavily"]);

    let rows = target.list(Some("exa")).await.unwrap();
    assert!(target.remove_by_id("exa", rows[0].id).await.unwrap());
    assert!(
        !target.remove_by_id("exa", rows[0].id).await.unwrap(),
        "already removed"
    );
    let raw = raw_rows(&target);
    assert_eq!(raw[0].4.as_deref(), Some("deleted by user"));
    assert_eq!(raw[1].4, None);

    target
        .logout_provider("exa", "logged out by user")
        .await
        .unwrap();
    let raw = raw_rows(&target);
    assert_eq!(
        raw[0].4.as_deref(),
        Some("deleted by user"),
        "already-disabled rows keep their cause"
    );
    assert_eq!(raw[1].4.as_deref(), Some("logged out by user"));
    assert_eq!(target.providers().await.unwrap(), ["tavily"]);

    // Logging in again with an api key hard-deletes the superseded api-key tombstones.
    login(&target, "exa", ScriptedUi::with(&["k3"]), &services)
        .await
        .unwrap();
    let raw = raw_rows(&target);
    assert_eq!(raw.iter().filter(|r| r.1 == "exa").count(), 1);
    assert_eq!(
        target.list(Some("exa")).await.unwrap()[0].credential,
        Credential::ApiKey("k3".into())
    );
}

#[tokio::test]
async fn alibaba_coding_plan_prompts_endpoints_then_stores_a_validated_key() {
    let (_dir, target) = temp_target();
    let (services, http) = services(|_| (200, "{}".into()));
    // 1=International, then the key.
    let ui = ScriptedUi::with(&["1", "  sk-ali-1  "]);
    let identity = login(&target, "alibaba-coding-plan", ui.clone(), &services)
        .await
        .unwrap();
    assert_eq!(identity, Some(Identity::ApiKey));
    let seen = http.seen.lock().clone();
    assert_eq!(seen.len(), 1);
    assert_eq!(
        seen[0].url,
        "https://coding-intl.dashscope.aliyuncs.com/v1/chat/completions"
    );
    assert!(seen[0]
        .headers
        .contains(&("Authorization".to_string(), "Bearer sk-ali-1".to_string())));
    let body: Value = serde_json::from_str(seen[0].body.as_deref().unwrap()).unwrap();
    assert_eq!(body["model"], "qwen3.5-plus");
    assert_eq!(body["max_tokens"], 1);
    assert_eq!(ui.prompts.lock()[0].message, "Select Alibaba Coding Plan endpoint: 1=International (default), 2=China, 3=Custom — enter 1, 2, or 3");
    assert_eq!(
        ui.auth.lock()[0].0,
        "https://modelstudio.console.alibabacloud.com/"
    );
    let rows = raw_rows(&target);
    assert_eq!(rows.len(), 1);
    assert_eq!(
        (rows[0].1.as_str(), rows[0].2.as_str()),
        ("alibaba-coding-plan", "api_key")
    );
    assert_eq!(rows[0].3, r#"{"key":"sk-ali-1","source":"login"}"#);
}

#[tokio::test]
async fn alibaba_coding_plan_china_choice_targets_the_china_base() {
    let (_dir, target) = temp_target();
    let (services, http) = services(|_| (200, "{}".into()));
    login(
        &target,
        "alibaba-coding-plan",
        ScriptedUi::with(&["2", "sk-ali-cn"]),
        &services,
    )
    .await
    .unwrap();
    let seen = http.seen.lock().clone();
    assert_eq!(
        seen[0].url,
        "https://coding.dashscope.aliyuncs.com/v1/chat/completions"
    );
    assert_eq!(
        raw_rows(&target)[0].1,
        "alibaba-coding-plan",
        "no region redirect: stored under its own id"
    );
}

#[tokio::test]
async fn alibaba_coding_plan_custom_choice_probes_the_models_endpoint() {
    let (_dir, target) = temp_target();
    let (services, http) = services(|r| {
        // The chat-completions probe must not fire for a custom base URL.
        assert!(r.url.ends_with("/models"), "{}", r.url);
        (200, r#"{"data":[]}"#.into())
    });
    login(
        &target,
        "alibaba-coding-plan",
        ScriptedUi::with(&["3", "https://proxy.example/v1/", "sk-ali-x"]),
        &services,
    )
    .await
    .unwrap();
    let seen = http.seen.lock().clone();
    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0].url, "https://proxy.example/v1/models");
    assert_eq!(seen[0].method, ProviderHttpMethod::Get);
}

#[tokio::test]
async fn alibaba_token_plan_wraps_cookie_and_region_into_the_stored_key() {
    let (_dir, target) = temp_target();
    let (services, http) = services(|r| {
        assert!(r.url.ends_with("/models"), "{}", r.url);
        (200, r#"{"data":[]}"#.into())
    });
    // 2=China, key, then the optional cookie.
    let ui = ScriptedUi::with(&["2", "sk-sp-cn", "Cookie: t=1; u=2"]);
    login(&target, "alibaba-token-plan", ui.clone(), &services)
        .await
        .unwrap();
    let seen = http.seen.lock().clone();
    assert_eq!(
        seen[0].url,
        "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/models"
    );
    let row = serde_json::from_str::<Value>(&raw_rows(&target)[0].3).unwrap();
    let stored: Value = serde_json::from_str(row["key"].as_str().unwrap()).unwrap();
    assert_eq!(stored["token"], "sk-sp-cn");
    assert_eq!(stored["cookie"], "t=1; u=2");
    assert_eq!(
        stored["baseUrl"],
        "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
    );

    // International default with no cookie stores the bare key, TS form.
    let (services, _) = self::services(|_| (200, r#"{"data":[]}"#.into()));
    let ui = ScriptedUi::with(&["1", "sk-sp-intl", ""]);
    login(&target, "alibaba-token-plan", ui, &services)
        .await
        .unwrap();
    let rows = raw_rows(&target);
    assert_eq!(rows[1].3, r#"{"key":"sk-sp-intl","source":"login"}"#);
}

#[tokio::test]
async fn alibaba_token_plan_rejects_a_single_cookie_value_like_ts() {
    let (_dir, target) = temp_target();
    let (services, _) = services(|_| (200, r#"{"data":[]}"#.into()));
    let err = login(
        &target,
        "alibaba-token-plan",
        ScriptedUi::with(&["1", "sk-sp-1", "just-a-value"]),
        &services,
    )
    .await
    .unwrap_err();
    assert!(err.starts_with("Invalid QwenCloud Cookie header"), "{err}");
    // The rejection happens after validation; only the earlier successful
    // login's row (if any) exists, nothing new was stored for this attempt.
    assert!(raw_rows(&target_after_touch(&target))
        .iter()
        .all(|r| !r.3.contains("just-a-value")));
}

#[tokio::test]
async fn alibaba_key_logins_cancel_and_empty_key_follow_the_ts_engine() {
    let (_dir, target) = temp_target();
    let (services, _) = services(|_| (200, "{}".into()));
    assert_eq!(
        login(
            &target,
            "alibaba-coding-plan",
            ScriptedUi::with(&[]),
            &services
        )
        .await
        .unwrap_err(),
        "Login cancelled"
    );
    assert_eq!(
        login(
            &target,
            "alibaba-coding-plan",
            ScriptedUi::with(&["1", "  "]),
            &services
        )
        .await
        .unwrap_err(),
        "API key is required"
    );
    assert_eq!(
        login(
            &target,
            "alibaba-token-plan",
            ScriptedUi::with(&["1", "  "]),
            &services
        )
        .await
        .unwrap_err(),
        "API key is required"
    );
}

#[tokio::test]
async fn openrouter_and_zai_coding_plan_log_in_like_api_key_providers() {
    let (_dir, target) = temp_target();
    let (services, http) = services(|_| (200, "{}".into()));
    let ui = ScriptedUi::with(&["sk-or-v1-abc"]);
    assert_eq!(
        login(&target, "openrouter", ui, &services).await.unwrap(),
        Some(Identity::ApiKey)
    );
    let seen = http.seen.lock().clone();
    assert_eq!(seen[0].url, "https://openrouter.ai/api/v1/auth/key");
    assert!(seen[0].headers.contains(&(
        "Authorization".to_string(),
        "Bearer sk-or-v1-abc".to_string()
    )));
    assert_eq!(raw_rows(&target)[0].1, "openrouter");

    // zai-coding-plan stores under `zai` (TS `store-as "zai"`).
    let (services, http) = self::services(|_| (200, "{}".into()));
    let ui = ScriptedUi::with(&["sk-zai-1"]);
    assert_eq!(
        login(&target, "zai-coding-plan", ui, &services)
            .await
            .unwrap(),
        Some(Identity::ApiKey)
    );
    let seen = http.seen.lock().clone();
    assert_eq!(
        seen[0].url,
        "https://api.z.ai/api/coding/paas/v4/chat/completions"
    );
    let rows = raw_rows(&target);
    assert!(rows.iter().any(|r| r.1 == "zai" && r.2 == "api_key"));
}

#[tokio::test]
async fn rejected_probe_blocks_the_new_key_logins_too() {
    let (_dir, target) = temp_target();
    let (services, _) = services(|_| (401, "nope".into()));
    let err = login(
        &target,
        "zai-coding-plan",
        ScriptedUi::with(&["sk-zai-1"]),
        &services,
    )
    .await
    .unwrap_err();
    assert_eq!(err, r#"Z.AI API key validation failed (401): nope"#);
    assert!(!target
        .providers()
        .await
        .unwrap()
        .contains(&"zai".to_string()));
}
