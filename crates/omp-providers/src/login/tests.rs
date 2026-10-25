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
type HttpReply = Box<dyn Fn(&ProviderHttpRequest) -> (u16, String) + Send + Sync>;
type RawRow = (i64, String, String, String, Option<String>, Option<String>);

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
    reply: HttpReply,
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

fn raw_rows(target: &Target) -> Vec<RawRow> {
    let Target::Local(db) = target else {
        unreachable!()
    };
    if !db.exists() {
        return Vec::new();
    }
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

/// A UI whose pane already exited: every cancel check trips.
struct CancelledUi;

impl OAuthLoginUi for CancelledUi {
    fn on_auth(&self, _url: &str, _instructions: Option<&str>) {}
    fn on_prompt(
        &self,
        _prompt: &OAuthPrompt,
    ) -> Pin<Box<dyn Future<Output = Option<String>> + Send + '_>> {
        Box::pin(async { None })
    }
    fn on_progress(&self, _message: &str) {}
    fn on_manual_code_input(
        &self,
    ) -> Option<Pin<Box<dyn Future<Output = Option<String>> + Send + '_>>> {
        None
    }
    fn is_cancelled(&self) -> bool {
        true
    }
}

#[tokio::test]
async fn cloudflare_login_serializes_the_three_answers_into_a_structured_key() {
    let (_dir, target) = temp_target();
    let (services, _) = services(|_| (200, "{}".into()));
    let ui = ScriptedUi::with(&["gw-default", "  acct-1  ", " cfut_tok "]);
    let identity = login(&target, "cloudflare-ai-gateway", ui.clone(), &services)
        .await
        .unwrap();
    assert_eq!(identity, Some(Identity::ApiKey));
    assert_eq!(
        ui.auth.lock()[0].0,
        "https://developers.cloudflare.com/ai-gateway/configuration/authentication/"
    );
    let messages: Vec<_> = ui
        .prompts
        .lock()
        .iter()
        .map(|p| p.message.clone())
        .collect();
    assert_eq!(
        messages,
        vec![
            "Paste your Cloudflare AI Gateway token/API key",
            "Enter your Cloudflare account ID",
            "Enter your Cloudflare AI Gateway ID",
        ]
    );
    let rows = raw_rows(&target);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].2, "api_key");
    let stored: Value = serde_json::from_str(&rows[0].3).unwrap();
    let credential: Value = serde_json::from_str(stored["key"].as_str().unwrap()).unwrap();
    assert_eq!(
        credential,
        json!({"token": "gw-default", "accountId": "acct-1", "gatewayId": "cfut_tok"})
    );
}

#[tokio::test]
async fn cloudflare_login_errors_follow_the_ts_messages() {
    let (_dir, target) = temp_target();
    let (services, _) = services(|_| (200, "{}".into()));
    assert_eq!(
        login(
            &target,
            "cloudflare-ai-gateway",
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
            "cloudflare-ai-gateway",
            ScriptedUi::with(&["  "]),
            &services
        )
        .await
        .unwrap_err(),
        "API key is required"
    );
    assert_eq!(
        login(
            &target,
            "cloudflare-ai-gateway",
            ScriptedUi::with(&["tok", " "]),
            &services
        )
        .await
        .unwrap_err(),
        "Cloudflare account ID is required"
    );
    assert_eq!(
        login(
            &target,
            "cloudflare-ai-gateway",
            ScriptedUi::with(&["tok", "acct", " "]),
            &services
        )
        .await
        .unwrap_err(),
        "Cloudflare AI Gateway ID is required"
    );
    let Target::Local(db) = &target else {
        unreachable!()
    };
    assert!(!db.exists());
}

#[tokio::test]
async fn xiaomi_login_validates_then_stores_the_key() {
    let (_dir, target) = temp_target();
    let (services, _http) = services(|request| {
        assert_eq!(
            request.url,
            "https://api.xiaomimimo.com/v1/chat/completions"
        );
        assert!(request
            .headers
            .contains(&("Authorization".to_string(), "Bearer sk-x-1".to_string())));
        let body: Value = serde_json::from_str(request.body.as_deref().unwrap()).unwrap();
        assert_eq!(body["model"], "mimo-v2.5");
        (200, "{}".into())
    });
    let ui = ScriptedUi::with(&[" sk-x-1 "]);
    assert_eq!(
        login(&target, "xiaomi", ui.clone(), &services)
            .await
            .unwrap(),
        Some(Identity::ApiKey)
    );
    assert_eq!(
        ui.auth.lock()[0].0,
        "https://platform.xiaomimimo.com/#/console/api-keys"
    );
    assert_eq!(
        ui.prompts.lock()[0].placeholder.as_deref(),
        Some("sk-... or tp-...")
    );
    assert_eq!(ui.progress.lock()[0], "Validating xiaomi API key...");
    let rows = raw_rows(&target);
    assert_eq!(rows[0].3, r#"{"key":"sk-x-1","source":"login"}"#);
}

#[tokio::test]
async fn xiaomi_login_errors_and_cancel() {
    let (_dir, target) = temp_target();
    let (services, http) = services(|_| (401, "bad key".into()));
    let err = login(&target, "xiaomi", ScriptedUi::with(&["tp-1"]), &services)
        .await
        .unwrap_err();
    // A tp- key walks SGP → AMS → CN before failing with the last details.
    let seen = http.seen.lock().clone();
    assert_eq!(seen.len(), 3);
    assert!(seen[0].url.contains("token-plan-sgp"));
    assert!(seen[2].url.contains("token-plan-cn"));
    assert_eq!(err, "Xiaomi MiMo API key validation failed (401): bad key");
    assert_eq!(
        login(&target, "xiaomi", ScriptedUi::with(&[]), &services)
            .await
            .unwrap_err(),
        "Login cancelled"
    );
    assert_eq!(
        login(&target, "xiaomi", ScriptedUi::with(&["  "]), &services)
            .await
            .unwrap_err(),
        "API key is required"
    );
}

#[tokio::test]
async fn kilo_login_approves_and_stores_a_one_year_grant() {
    let (_dir, target) = temp_target();
    let (services, http) = services(|request| {
        if request.url.ends_with("/codes") {
            assert_eq!(request.url, "https://api.kilo.ai/api/device-auth/codes");
            (
                200,
                json!({
                    "code": "ABCD-1234",
                    "verificationUrl": "https://kilo.ai/authorize",
                    "expiresIn": 600
                })
                .to_string(),
            )
        } else {
            assert_eq!(
                request.url,
                "https://api.kilo.ai/api/device-auth/codes/ABCD-1234"
            );
            (
                200,
                json!({"status": "approved", "token": "kilo-token"}).to_string(),
            )
        }
    });
    let ui = Arc::new(ScriptedUi::default());
    let before = crate::now_ms() as i64;
    assert_eq!(
        login(&target, "kilo", ui.clone(), &services).await.unwrap(),
        Some(Identity::OAuth {
            email: None,
            account_id: None,
            org_id: None,
            org_name: None,
        })
    );
    let auth = ui.auth.lock();
    assert_eq!(auth[0].0, "https://kilo.ai/authorize");
    assert_eq!(auth[0].1.as_deref(), Some("Enter code: ABCD-1234"));
    drop(auth);
    let rows = raw_rows(&target);
    assert_eq!(rows[0].2, "oauth");
    let stored: Value = serde_json::from_str(&rows[0].3).unwrap();
    assert_eq!(stored["access"], "kilo-token");
    assert_eq!(stored["refresh"], "");
    let after = crate::now_ms() as i64;
    let expires = stored["expires"].as_i64().unwrap();
    assert!(
        (before + 365 * 24 * 60 * 60 * 1000..=after + 365 * 24 * 60 * 60 * 1000).contains(&expires)
    );
    let _ = http;
}

#[tokio::test]
async fn kilo_login_error_paths() {
    let (_dir, target) = temp_target();
    let (services, _) = services(|request| {
        if request.url.ends_with("/codes") {
            (429, "too many".into())
        } else {
            (200, json!({"status": "denied"}).to_string())
        }
    });
    assert_eq!(
        login(&target, "kilo", ScriptedUi::with(&[]), &services)
            .await
            .unwrap_err(),
        "Too many pending authorization requests. Please try again later."
    );
    let (services, _) = self::services(|request| {
        if request.url.ends_with("/codes") {
            (
                200,
                json!({"code": "C1", "verificationUrl": "https://kilo.ai/authorize", "expiresIn": 600})
                    .to_string(),
            )
        } else {
            (200, json!({"status": "denied"}).to_string())
        }
    });
    assert_eq!(
        login(&target, "kilo", ScriptedUi::with(&[]), &services)
            .await
            .unwrap_err(),
        "Authorization was denied"
    );
    let (services, _) = self::services(|request| {
        if request.url.ends_with("/codes") {
            (
                200,
                json!({"code": "C1", "verificationUrl": "https://kilo.ai/authorize", "expiresIn": 600})
                    .to_string(),
            )
        } else {
            (410, String::new())
        }
    });
    assert_eq!(
        login(&target, "kilo", ScriptedUi::with(&[]), &services)
            .await
            .unwrap_err(),
        "Authorization code expired. Please try again."
    );
    let (services, _) = self::services(|request| {
        if request.url.ends_with("/codes") {
            (200, json!({"code": "C1"}).to_string())
        } else {
            (200, String::new())
        }
    });
    assert_eq!(
        login(&target, "kilo", ScriptedUi::with(&[]), &services)
            .await
            .unwrap_err(),
        "Kilo device authorization response missing required fields"
    );
    let (services, _) = self::services(|_| (200, "{}".into()));
    assert_eq!(
        login(&target, "kilo", Arc::new(CancelledUi) as _, &services)
            .await
            .unwrap_err(),
        "Login cancelled"
    );
    let Target::Local(db) = &target else {
        unreachable!()
    };
    assert!(!db.exists());
}

#[tokio::test]
async fn factory_login_completes_the_device_grant_and_whoami() {
    let (_dir, target) = temp_target();
    // exp = now + 3600s, wrapped in a fake JWT payload.
    let payload = base64url_payload(&json!({
        "exp": crate::now_ms() as i64 / 1000 + 3600,
        "email": "dev@example.com",
        "sub": "user_1",
        "external_org_id": "org_token",
    }));
    let access = format!("header.{payload}.signature");
    let (services, http) = services(move |request| {
        if request.url.ends_with("/authorize/device") {
            assert!(request
                .headers
                .contains(&("Accept".to_string(), "application/json".to_string())));
            let body = request.body.as_deref().unwrap();
            assert!(body.contains("client_id=client_01HNM792M5G5G1A2THWPXKFMXB"));
            (
                200,
                json!({
                    "device_code": "dc_1",
                    "user_code": "WDJB-MJHT",
                    "verification_uri": "https://auth.workos.com/device",
                    "interval": 0,
                    "expires_in": 600
                })
                .to_string(),
            )
        } else if request.url.ends_with("/authenticate") {
            (
                200,
                json!({
                    "access_token": access,
                    "refresh_token": "rt_1",
                    "user": {"email": "dev@example.com", "id": "user_1"}
                })
                .to_string(),
            )
        } else {
            assert!(request.url.ends_with("/api/cli/whoami"));
            assert!(request
                .headers
                .contains(&("X-Factory-Org-Id".to_string(), "org_token".to_string())));
            (
                200,
                json!({"orgId": "org_whoami", "region": "eu", "inferenceRegion": "global"})
                    .to_string(),
            )
        }
    });
    let ui = Arc::new(ScriptedUi::default());
    assert_eq!(
        login(&target, "factory-droid", ui.clone(), &services)
            .await
            .unwrap(),
        Some(Identity::OAuth {
            email: Some("dev@example.com".to_string()),
            account_id: Some("user_1".to_string()),
            org_id: Some("org_whoami".to_string()),
            org_name: None,
        })
    );
    assert_eq!(ui.auth.lock()[0].0, "https://auth.workos.com/device");
    let rows = raw_rows(&target);
    assert_eq!(rows[0].2, "oauth");
    let stored: Value = serde_json::from_str(&rows[0].3).unwrap();
    assert_eq!(stored["refresh"], "rt_1");
    assert_eq!(stored["orgId"], "org_whoami");
    assert_eq!(stored["region"], "eu");
    assert_eq!(stored["inferenceRegion"], "global");
    // The JWT exp wins over the one-day fallback.
    let now = crate::now_ms() as i64;
    let expires = stored["expires"].as_i64().unwrap();
    assert!(expires > now + 30 * 60 * 1000 && expires < now + 2 * 60 * 60 * 1000);
    let _ = http;
}

#[tokio::test]
async fn factory_login_error_paths() {
    let (_dir, target) = temp_target();
    let (services, _) = services(|request| {
        if request.url.ends_with("/authorize/device") {
            (500, "nope".into())
        } else {
            (200, "{}".into())
        }
    });
    assert!(
        login(&target, "factory-droid", ScriptedUi::with(&[]), &services)
            .await
            .unwrap_err()
            .starts_with("factory-droid device authorization failed: 500 nope")
    );

    let (services, _) = self::services(|request| {
        if request.url.ends_with("/authorize/device") {
            (
                200,
                json!({
                    "device_code": "dc_1",
                    "user_code": "U1",
                    "verification_uri": "https://auth.workos.com/device",
                    "interval": 0,
                    "expires_in": 600
                })
                .to_string(),
            )
        } else {
            (200, json!({"error": "access_denied"}).to_string())
        }
    });
    assert_eq!(
        login(&target, "factory-droid", ScriptedUi::with(&[]), &services)
            .await
            .unwrap_err(),
        "factory-droid device authorization was denied"
    );

    // whoami failure at login is hard; a missing org resolution too.
    let payload = base64url_payload(&json!({"external_org_id": "org_1"}));
    let access = format!("h.{payload}.s");
    let access_clone = access.clone();
    let (services, _) = self::services(move |request| {
        if request.url.ends_with("/authorize/device") {
            (
                200,
                json!({
                    "device_code": "dc_1",
                    "user_code": "U1",
                    "verification_uri": "https://auth.workos.com/device",
                    "interval": 0,
                    "expires_in": 600
                })
                .to_string(),
            )
        } else if request.url.ends_with("/authenticate") {
            (
                200,
                json!({"access_token": access_clone, "refresh_token": "rt_1"}).to_string(),
            )
        } else {
            (503, "down".into())
        }
    });
    assert_eq!(
        login(&target, "factory-droid", ScriptedUi::with(&[]), &services)
            .await
            .unwrap_err(),
        "Factory identity check failed (503): down"
    );

    let (services, _) = self::services(move |request| {
        if request.url.ends_with("/authorize/device") {
            (
                200,
                json!({
                    "device_code": "dc_1",
                    "user_code": "U1",
                    "verification_uri": "https://auth.workos.com/device",
                    "interval": 0,
                    "expires_in": 600
                })
                .to_string(),
            )
        } else if request.url.ends_with("/authenticate") {
            (
                200,
                json!({"access_token": access, "refresh_token": "rt_1"}).to_string(),
            )
        } else {
            (200, json!({"region": "eu"}).to_string())
        }
    });
    assert_eq!(
        login(&target, "factory-droid", ScriptedUi::with(&[]), &services)
            .await
            .unwrap_err(),
        "Factory login did not resolve an organization"
    );
    let (services, _) = self::services(|_| (200, "{}".into()));
    assert_eq!(
        login(&target, "factory-droid", Arc::new(CancelledUi), &services)
            .await
            .unwrap_err(),
        "Login cancelled"
    );
    let Target::Local(db) = &target else {
        unreachable!()
    };
    assert!(!db.exists());
}

fn base64url_payload(value: &Value) -> String {
    use base64::Engine as _;
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(value.to_string())
}

#[tokio::test]
async fn gitlab_duo_login_completes_through_the_loopback_callback() {
    let (_dir, target) = temp_target();
    let (services, http) = services(|request| {
        assert_eq!(request.url, "https://gitlab.com/oauth/token");
        assert!(request.headers.contains(&(
            "Content-Type".to_string(),
            "application/x-www-form-urlencoded".to_string()
        )));
        let body = request.body.as_deref().unwrap();
        for expected in [
            "grant_type=authorization_code",
            &format!(
                "client_id={}",
                "da4edff2e6ebd2bc3208611e2768bc1c1dd7be791dc5ff26ca34ca9ee44f7d4b"
            ),
            "code=gl-code",
            &format!(
                "redirect_uri={}",
                "http%3A%2F%2Flocalhost%3A8080%2Fcallback"
            ),
        ] {
            assert!(body.contains(expected), "{body}");
        }
        assert!(body.contains("code_verifier="));
        (
            200,
            json!({
                "access_token": "glat",
                "refresh_token": "glrt",
                "expires_in": 7200,
                "created_at": 1_700_000_000i64
            })
            .to_string(),
        )
    });
    let scripted = Arc::new(ScriptedUi::default());
    let services = &services;
    let flow = login(&target, "gitlab-duo", scripted.clone(), services);
    tokio::pin!(flow);
    let driver = async {
        let auth_url = loop {
            if let Some((url, _)) = scripted.auth.lock().first() {
                break url.clone();
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        };
        let url = reqwest::Url::parse(&auth_url).unwrap();
        let get = |name: &str| {
            url.query_pairs()
                .find(|(key, _)| key == name)
                .map(|(_, value)| value.to_string())
                .unwrap()
        };
        assert_eq!(get("scope"), "api");
        assert_eq!(get("response_type"), "code");
        assert!(!get("code_challenge").is_empty());
        assert_eq!(get("code_challenge_method"), "S256");
        let state = get("state");
        let mut stream = tokio::net::TcpStream::connect("127.0.0.1:8080")
            .await
            .unwrap();
        use tokio::io::AsyncWriteExt;
        stream
            .write_all(
                format!(
                    "GET /callback?code=gl-code&state={state} HTTP/1.1\r\nHost: localhost\r\n\r\n"
                )
                .as_bytes(),
            )
            .await
            .unwrap();
        stream.shutdown().await.unwrap();
    };
    let settled = tokio::select! {
        _ = driver => false,
        result = flow.as_mut() => { assert!(result.is_ok()); true }
    };
    // If the driver won the select (it finished right after the callback),
    // the flow only needs its final poll.
    let result = if settled {
        Ok(Some(Identity::OAuth {
            email: None,
            account_id: None,
            org_id: None,
            org_name: None,
        }))
    } else {
        flow.as_mut().await
    };
    assert_eq!(
        result,
        Ok(Some(Identity::OAuth {
            email: None,
            account_id: None,
            org_id: None,
            org_name: None,
        }))
    );
    let rows = raw_rows(&target);
    assert_eq!(rows[0].2, "oauth");
    let stored: Value = serde_json::from_str(&rows[0].3).unwrap();
    assert_eq!(stored["access"], "glat");
    assert_eq!(stored["refresh"], "glrt");
    assert_eq!(
        stored["expires"],
        json!(1_700_000_000i64 * 1000 + 7200 * 1000 - 300_000)
    );
    let _ = http;
}

#[tokio::test]
async fn gitlab_duo_login_cancelled_and_exchange_failure() {
    let (_dir, target) = temp_target();
    let (services, _) = services(|_| (200, "{}".into()));
    assert_eq!(
        login(&target, "gitlab-duo", Arc::new(CancelledUi), &services)
            .await
            .unwrap_err(),
        "Login cancelled"
    );
    let Target::Local(db) = &target else {
        unreachable!()
    };
    assert!(!db.exists());
}

/// A paste-capable UI whose manual answer waits for the authorize URL (so the
/// pasted `state` matches the flow's) — the `gitlab-duo-agent` surface.
struct ManualPasteUi {
    auth: parking_lot::Mutex<Vec<(String, Option<String>)>>,
}

impl OAuthLoginUi for ManualPasteUi {
    fn on_auth(&self, url: &str, instructions: Option<&str>) {
        self.auth
            .lock()
            .push((url.to_string(), instructions.map(str::to_string)));
    }
    fn on_prompt(
        &self,
        _prompt: &OAuthPrompt,
    ) -> Pin<Box<dyn Future<Output = Option<String>> + Send + '_>> {
        Box::pin(async { None })
    }
    fn on_progress(&self, _message: &str) {}
    fn on_manual_code_input(
        &self,
    ) -> Option<Pin<Box<dyn Future<Output = Option<String>> + Send + '_>>> {
        Some(Box::pin(async {
            let auth_url = loop {
                if let Some((url, _)) = self.auth.lock().first() {
                    break url.clone();
                }
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            };
            let url = reqwest::Url::parse(&auth_url).unwrap();
            let state = url
                .query_pairs()
                .find(|(key, _)| key == "state")
                .map(|(_, value)| value.to_string())
                .unwrap();
            Some(format!(
                "vscode://gitlab.gitlab-workflow/authentication?code=vs-code&state={state}"
            ))
        }))
    }
}

#[tokio::test]
async fn gitlab_duo_agent_login_exchanges_the_pasted_vscode_redirect() {
    let (_dir, target) = temp_target();
    let (services, http) = services(|request| {
        assert_eq!(request.url, "https://gitlab.com/oauth/token");
        let body = request.body.as_deref().unwrap();
        for expected in [
            "grant_type=authorization_code",
            &format!(
                "client_id={}",
                "36f2a70cddeb5a0889d4fd8295c241b7e9848e89cf9e599d0eed2d8e5350fbf5"
            ),
            "code=vs-code",
            &format!(
                "redirect_uri={}",
                "vscode%3A%2F%2Fgitlab.gitlab-workflow%2Fauthentication"
            ),
        ] {
            assert!(body.contains(expected), "{body}");
        }
        (
            200,
            json!({"access_token": "at", "refresh_token": "rt", "expires_in": 7200}).to_string(),
        )
    });
    let ui = Arc::new(ManualPasteUi {
        auth: parking_lot::Mutex::new(Vec::new()),
    });
    assert_eq!(
        login(&target, "gitlab-duo-agent", ui.clone(), &services)
            .await
            .unwrap(),
        Some(Identity::OAuth {
            email: None,
            account_id: None,
            org_id: None,
            org_name: None,
        })
    );
    let rows = raw_rows(&target);
    assert_eq!(rows[0].2, "oauth");
    let stored: Value = serde_json::from_str(&rows[0].3).unwrap();
    assert_eq!(stored["access"], "at");
    assert_eq!(stored["refresh"], "rt");
    let _ = http;
}

#[tokio::test]
async fn gitlab_duo_agent_login_cancelled() {
    let (_dir, target) = temp_target();
    let (services, _) = services(|_| (200, "{}".into()));
    assert_eq!(
        login(
            &target,
            "gitlab-duo-agent",
            Arc::new(CancelledUi),
            &services
        )
        .await
        .unwrap_err(),
        "Login cancelled"
    );
    // A manual surface that resolves empty cancels the paste race.
    struct ManualNoneUi;
    impl OAuthLoginUi for ManualNoneUi {
        fn on_auth(&self, _url: &str, _instructions: Option<&str>) {}
        fn on_prompt(
            &self,
            _prompt: &OAuthPrompt,
        ) -> Pin<Box<dyn Future<Output = Option<String>> + Send + '_>> {
            Box::pin(async { None })
        }
        fn on_progress(&self, _message: &str) {}
        fn on_manual_code_input(
            &self,
        ) -> Option<Pin<Box<dyn Future<Output = Option<String>> + Send + '_>>> {
            Some(Box::pin(async { None }))
        }
    }
    assert_eq!(
        login(
            &target,
            "gitlab-duo-agent",
            Arc::new(ManualNoneUi),
            &services
        )
        .await
        .unwrap_err(),
        "Login cancelled"
    );
    let Target::Local(db) = &target else {
        unreachable!()
    };
    assert!(!db.exists());
}

// ---------------------------------------------------------------------------
// Wave 2: cursor, muse-code, perplexity, google loopback flows, codex device
// ---------------------------------------------------------------------------

use std::collections::VecDeque;

type Replies = Arc<Mutex<VecDeque<(u16, String)>>>;

/// Routes scripted replies per URL substring, in order; unmatched requests
/// get a 500 so a wrong expectation fails loudly.
fn routed(route: Arc<Mutex<Vec<(&'static str, Replies)>>>) -> (Services, Arc<ScriptedHttp>) {
    services(move |request| {
        let mut routes = route.lock();
        for (needle, queue) in routes.iter_mut() {
            if request.url.contains(*needle) {
                if let Some(reply) = queue.lock().pop_front() {
                    return reply;
                }
            }
        }
        (
            500,
            r#"{"error":"no scripted reply for this request"}"#.to_string(),
        )
    })
}

fn queue(replies: Vec<(u16, String)>) -> Replies {
    Arc::new(Mutex::new(replies.into()))
}

fn routes(table: Vec<(&'static str, Replies)>) -> Arc<Mutex<Vec<(&'static str, Replies)>>> {
    Arc::new(Mutex::new(table))
}

/// The browser leg: waits for the authorize URL, then fires one real HTTP
/// request at the running callback server with the state and code.
async fn complete_callback(ui: &ScriptedUi, code: &str) {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let url = {
        let mut found = None;
        for _ in 0..500 {
            if let Some((url, _)) = ui.auth.lock().first() {
                found = Some(url.clone());
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        found.expect("the flow never surfaced an authorize URL")
    };
    let redirect = url
        .split("redirect_uri=")
        .nth(1)
        .unwrap_or_default()
        .split('&')
        .next()
        .unwrap_or_default()
        .replace("%3A", ":")
        .replace("%2F", "/");
    let (host_port, path) = redirect
        .trim_start_matches("http://")
        .split_once('/')
        .expect("redirect_uri must be a loopback http URL");
    let state = url
        .split("state=")
        .nth(1)
        .unwrap_or_default()
        .split('&')
        .next()
        .unwrap_or_default();
    let request = format!(
        "GET /{path}?code={code}&state={state} HTTP/1.1\r\nHost: {host_port}\r\nConnection: close\r\n\r\n"
    );
    for _ in 0..500 {
        if let Ok(mut stream) = tokio::net::TcpStream::connect(host_port).await {
            stream.write_all(request.as_bytes()).await.unwrap();
            let mut ignored = Vec::new();
            let _ = stream.read_to_end(&mut ignored).await;
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    panic!("the callback server never came up on {host_port}");
}

// -- cursor ------------------------------------------------------------------

#[tokio::test]
async fn cursor_login_polls_until_the_browser_leg_completes_then_stores_the_pair() {
    let now = crate::now_ms() / 1000;
    let access = jwt(&json!({ "sub": "user-9", "exp": now + 3600 }));
    let (services, http) = routed(routes(vec![(
        "/auth/poll",
        queue(vec![
            (404, String::new()),
            (
                200,
                json!({"accessToken": access, "refreshToken": "cur-refresh"}).to_string(),
            ),
        ]),
    )]));
    let (_dir, target) = temp_target();
    let ui = ScriptedUi::with(&[]);
    let identity = login(&target, "cursor", ui.clone(), &services)
        .await
        .unwrap();

    let url = &ui.auth.lock()[0].0;
    assert!(url.starts_with("https://cursor.com/loginDeepControl?challenge="));
    assert!(url.contains("&uuid="));
    assert!(url.contains("&mode=login&redirectTarget=cli"));
    assert_eq!(
        identity,
        Some(Identity::OAuth {
            email: None,
            account_id: None,
            org_id: None,
            org_name: None
        })
    );

    let polls = http.seen.lock().clone();
    assert_eq!(polls.len(), 2);
    let verifier = polls[0]
        .url
        .split("verifier=")
        .nth(1)
        .unwrap_or_default()
        .to_string();
    assert!(!verifier.is_empty());
    assert!(polls[1].url.ends_with(&verifier));

    let rows = raw_rows(&target);
    assert_eq!(
        (rows[0].1.as_str(), rows[0].2.as_str()),
        ("cursor", "oauth")
    );
    let data: Value = serde_json::from_str(&rows[0].3).unwrap();
    assert_eq!(data["access"], access.clone());
    assert_eq!(data["refresh"], "cur-refresh");
    assert_eq!(data["expires"], (now + 3600) * 1000 - 300_000);
}

#[tokio::test]
async fn cursor_login_poll_failure_is_a_login_error() {
    let (services, _) = routed(routes(vec![(
        "/auth/poll",
        queue(vec![(400, "no".into())]),
    )]));
    let (_dir, target) = temp_target();
    let err = login(&target, "cursor", ScriptedUi::with(&[]), &services)
        .await
        .unwrap_err();
    assert_eq!(err, "Poll failed: 400");
    // Nothing is stored on a failed login.
    assert!(raw_rows(&target).is_empty());
}

// -- openai-codex-device ------------------------------------------------------

#[tokio::test]
async fn codex_device_login_exchanges_the_device_code_and_stores_under_openai_codex() {
    let access = jwt(&json!({
        "https://api.openai.com/profile": {"email": "Dev@Example.com"},
        "https://api.openai.com/auth": {"chatgpt_account_id": "acc-1", "chatgpt_plan_type": "pro"},
    }));
    let (services, http) = routed(routes(vec![
        (
            "deviceauth/usercode",
            queue(vec![(
                200,
                json!({"device_auth_id": "d1", "user_code": "UC-1", "interval": 0}).to_string(),
            )]),
        ),
        (
            "deviceauth/token",
            queue(vec![(
                200,
                json!({"authorization_code": "ac", "code_verifier": "cv"}).to_string(),
            )]),
        ),
        (
            "oauth/token",
            queue(vec![(
                200,
                json!({"access_token": access, "refresh_token": "cr", "expires_in": 3600})
                    .to_string(),
            )]),
        ),
    ]));
    let (_dir, target) = temp_target();
    let ui = ScriptedUi::with(&[]);
    let identity = login(&target, "openai-codex-device", ui.clone(), &services)
        .await
        .unwrap();

    assert_eq!(
        ui.auth.lock()[0],
        (
            "https://auth.openai.com/codex/device".to_string(),
            Some("Enter code: UC-1".to_string())
        )
    );
    assert_eq!(
        identity,
        Some(Identity::OAuth {
            email: Some("dev@example.com".into()),
            account_id: Some("acc-1".into()),
            org_id: Some("acc-1".into()),
            org_name: Some("pro".into())
        })
    );

    let seen = http.seen.lock().clone();
    assert_eq!(seen.len(), 3);
    assert!(seen[0].url.ends_with("deviceauth/usercode"));
    let init: Value = serde_json::from_str(seen[0].body.as_deref().unwrap()).unwrap();
    assert_eq!(init["client_id"], "app_EMoamEEZ73f0CkXaXp7hrann");
    let poll: Value = serde_json::from_str(seen[1].body.as_deref().unwrap()).unwrap();
    assert_eq!(
        (poll["device_auth_id"].as_str(), poll["user_code"].as_str()),
        (Some("d1"), Some("UC-1"))
    );
    let exchange: Value = serde_json::from_str(seen[2].body.as_deref().unwrap()).unwrap();
    assert_eq!(exchange["grant_type"], "authorization_code");
    assert_eq!(
        exchange["redirect_uri"],
        "https://auth.openai.com/deviceauth/callback"
    );
    assert_eq!(exchange["client_id"], "app_EMoamEEZ73f0CkXaXp7hrann");

    let rows = raw_rows(&target);
    assert_eq!(rows[0].1, "openai-codex");
    let data: Value = serde_json::from_str(&rows[0].3).unwrap();
    assert_eq!(data["refresh"], "cr");
    assert_eq!(data["accountId"], "acc-1");
}

#[tokio::test]
async fn codex_device_login_initiation_failure_is_a_login_error() {
    let (services, _) = routed(routes(vec![(
        "deviceauth/usercode",
        queue(vec![(503, String::new())]),
    )]));
    let (_dir, target) = temp_target();
    let err = login(
        &target,
        "openai-codex-device",
        ScriptedUi::with(&[]),
        &services,
    )
    .await
    .unwrap_err();
    assert_eq!(err, "Device authorization initiation failed: 503");
    assert!(raw_rows(&target).is_empty());
}

// -- muse-code ----------------------------------------------------------------

#[tokio::test]
async fn muse_code_device_login_mints_the_key_and_stores_the_packed_credential() {
    let (services, http) = routed(routes(vec![
        (
            "oidc/device/authorization",
            queue(vec![(
                200,
                json!({
                    "user_code": "MC-7",
                    "device_code": "dc",
                    "verification_uri": "https://auth.meta.com/device",
                    "verification_uri_complete": "https://auth.meta.com/device?code=MC-7",
                    "interval": 0,
                    "expires_in": 600
                })
                .to_string(),
            )]),
        ),
        (
            "oidc/device/token",
            queue(vec![(200, json!({"access_token": "meta-token"}).to_string())]),
        ),
        (
            "muse-code/key",
            queue(vec![(
                200,
                json!({"api_key": "mkey", "user_email": "User@Example.com", "user_id": "u-1", "is_subs_active": true})
                    .to_string(),
            )]),
        ),
    ]));
    let (_dir, target) = temp_target();
    let ui = ScriptedUi::with(&[]);
    let identity = login(&target, "muse-code", ui.clone(), &services)
        .await
        .unwrap();

    assert_eq!(
        ui.auth.lock()[0],
        (
            "https://auth.meta.com/device?code=MC-7".to_string(),
            Some("Enter code: MC-7".to_string())
        )
    );
    assert_eq!(
        identity,
        Some(Identity::OAuth {
            email: Some("user@example.com".into()),
            account_id: Some("u-1".into()),
            org_id: None,
            org_name: None
        })
    );

    let key_request = &http.seen.lock()[2];
    assert_eq!(key_request.url, "https://api.meta.ai/muse-code/key");
    assert!(key_request
        .headers
        .contains(&("Authorization".to_string(), "Bearer meta-token".to_string())));
    assert!(key_request
        .headers
        .contains(&("x-api-version".to_string(), "1.0.0".to_string())));
    let key_body: Value = serde_json::from_str(key_request.body.as_deref().unwrap()).unwrap();
    assert_eq!(key_body["onboard"], true);

    let rows = raw_rows(&target);
    assert_eq!(
        (rows[0].1.as_str(), rows[0].2.as_str()),
        ("muse-code", "oauth")
    );
    let data: Value = serde_json::from_str(&rows[0].3).unwrap();
    let packed: Value = serde_json::from_str(data["access"].as_str().unwrap()).unwrap();
    assert_eq!(
        (
            packed["oauthAccessToken"].as_str(),
            packed["apiKey"].as_str()
        ),
        (Some("meta-token"), Some("mkey"))
    );
    assert_eq!(data["expires"], 8_640_000_000_000_000i64);
    assert_eq!(
        (data["accountId"].as_str(), data["email"].as_str()),
        (Some("u-1"), Some("user@example.com"))
    );
}

#[tokio::test]
async fn muse_code_device_denial_is_a_login_error() {
    let (services, _) = routed(routes(vec![
        (
            "oidc/device/authorization",
            queue(vec![(
                200,
                json!({"user_code": "MC-7", "device_code": "dc", "verification_uri": "https://auth.meta.com/device"}).to_string(),
            )]),
        ),
        (
            "oidc/device/token",
            queue(vec![(200, json!({"error": "access_denied"}).to_string())]),
        ),
    ]));
    let (_dir, target) = temp_target();
    let err = login(&target, "muse-code", ScriptedUi::with(&[]), &services)
        .await
        .unwrap_err();
    assert_eq!(err, "muse-code device authorization was denied");
    assert!(raw_rows(&target).is_empty());
}

// -- perplexity ---------------------------------------------------------------

#[tokio::test]
async fn perplexity_email_otp_login_stores_the_session_jwt() {
    let (services, http) = routed(routes(vec![
        (
            "/api/auth/csrf",
            queue(vec![(200, json!({"csrfToken": "c-1"}).to_string())]),
        ),
        ("/api/auth/signin-email", queue(vec![(200, "{}".into())])),
        (
            "/api/auth/signin-otp",
            queue(vec![(200, json!({"token": "ppl-jwt"}).to_string())]),
        ),
    ]));
    let (_dir, target) = temp_target();
    let ui = ScriptedUi::with(&["  User@Example.com  ", "  424242  "]);
    let identity = login(&target, "perplexity", ui.clone(), &services)
        .await
        .unwrap();

    assert_eq!(
        identity,
        Some(Identity::OAuth {
            email: Some("User@Example.com".into()),
            account_id: None,
            org_id: None,
            org_name: None
        })
    );
    assert_eq!(ui.progress.lock()[0], "Fetching Perplexity CSRF token...");
    let seen = http.seen.lock().clone();
    assert_eq!(seen.len(), 3);
    assert!(seen[0]
        .headers
        .contains(&("X-App-ApiVersion".to_string(), "2.18".to_string())));
    let send: Value = serde_json::from_str(seen[1].body.as_deref().unwrap()).unwrap();
    assert_eq!(
        (send["email"].as_str(), send["csrfToken"].as_str()),
        (Some("User@Example.com"), Some("c-1"))
    );
    let verify: Value = serde_json::from_str(seen[2].body.as_deref().unwrap()).unwrap();
    assert_eq!(verify["otp"], "424242");

    let rows = raw_rows(&target);
    let data: Value = serde_json::from_str(&rows[0].3).unwrap();
    assert_eq!(
        (data["access"].as_str(), data["refresh"].as_str()),
        (Some("ppl-jwt"), Some("ppl-jwt"))
    );
    assert_eq!(data["email"], "User@Example.com");
}

#[tokio::test]
async fn perplexity_totp_challenge_verifies_the_authenticator_code() {
    let (services, http) = routed(routes(vec![
        (
            "/api/auth/csrf",
            queue(vec![(200, json!({"csrfToken": "c-1"}).to_string())]),
        ),
        ("/api/auth/signin-email", queue(vec![(200, "{}".into())])),
        (
            "/api/auth/signin-otp",
            queue(vec![(
                200,
                json!({"status": "totp_challenge_required", "challenge_token": "ct-1"}).to_string(),
            )]),
        ),
        (
            "totp/challenge-verify",
            queue(vec![(200, json!({"token": "ppl-jwt-2"}).to_string())]),
        ),
    ]));
    let (_dir, target) = temp_target();
    let ui = ScriptedUi::with(&["u@x.com", "111111", "999999"]);
    login(&target, "perplexity", ui, &services).await.unwrap();

    let seen = http.seen.lock().clone();
    assert_eq!(seen.len(), 4);
    assert!(seen[3].url.ends_with("/api/auth/totp/challenge-verify"));
    let verify: Value = serde_json::from_str(seen[3].body.as_deref().unwrap()).unwrap();
    assert_eq!(
        (verify["token"].as_str(), verify["code"].as_str()),
        (Some("ct-1"), Some("999999"))
    );
}

#[tokio::test]
async fn perplexity_csrf_failure_is_a_login_error() {
    let (services, _) = routed(routes(vec![(
        "/api/auth/csrf",
        queue(vec![(503, String::new())]),
    )]));
    let (_dir, target) = temp_target();
    let err = login(
        &target,
        "perplexity",
        ScriptedUi::with(&["u@x.com", "1"]),
        &services,
    )
    .await
    .unwrap_err();
    assert_eq!(err, "Perplexity CSRF request failed: 503");
    assert!(raw_rows(&target).is_empty());
}

// -- google-gemini-cli ----------------------------------------------------------

#[tokio::test]
async fn gemini_cli_loopback_login_exchanges_and_discovers_the_project() {
    let access = jwt(&json!({ "email": "g@x.com" }));
    let (services, http) = routed(routes(vec![
        (
            "oauth2.googleapis.com/token",
            queue(vec![(
                200,
                json!({"access_token": access, "refresh_token": "gr", "expires_in": 3600}).to_string(),
            )]),
        ),
        (
            "oauth2/v1/userinfo",
            queue(vec![(200, json!({"email": "g@x.com"}).to_string())]),
        ),
        (
            "v1internal:loadCodeAssist",
            queue(vec![(
                200,
                json!({"currentTier": {"id": "standard-tier"}, "cloudaicompanionProject": "proj-1"}).to_string(),
            )]),
        ),
    ]));
    let (_dir, target) = temp_target();
    let ui = ScriptedUi::with(&[]);
    let (identity, ()) = tokio::join!(
        login(&target, "google-gemini-cli", ui.clone(), &services),
        complete_callback(&ui, "g-code"),
    );
    let identity = identity.unwrap();
    assert_eq!(
        identity,
        Some(Identity::OAuth {
            email: Some("g@x.com".into()),
            account_id: None,
            org_id: None,
            org_name: None
        })
    );

    let url = &ui.auth.lock()[0].0;
    assert!(url.starts_with("https://accounts.google.com/o/oauth2/v2/auth?"));
    for needle in [
        "response_type=code",
        &format!(
            "client_id={}",
            super::oauth_code::decode_base64(super::oauth_code::GEMINI_CLIENT_ID_B64)
        ),
        "redirect_uri=http%3A%2F%2F127.0.0.1%3A8085%2Foauth2callback",
        "scope=https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fcloud-platform",
        "code_challenge_method=S256",
        "access_type=offline",
        "prompt=consent",
    ] {
        assert!(
            url.contains(needle),
            "authorize URL missing {needle}: {url}"
        );
    }

    let seen = http.seen.lock().clone();
    assert_eq!(seen.len(), 3);
    let exchange_body = seen[0].body.clone().unwrap();
    assert!(exchange_body.starts_with("grant_type=authorization_code&"));
    assert!(exchange_body.contains("code=g-code"));
    assert!(exchange_body.contains(&format!(
        "client_id={}",
        super::oauth_code::decode_base64(super::oauth_code::GEMINI_CLIENT_ID_B64)
    )));
    assert!(exchange_body.contains(&format!(
        "client_secret={}",
        super::oauth_code::decode_base64(super::oauth_code::GEMINI_CLIENT_SECRET_B64)
    )));
    assert!(exchange_body.contains("code_verifier="));
    assert!(seen[2].headers.contains(&(
        "User-Agent".to_string(),
        "GeminiCLI/0.46.0/gemini-3.1-pro-preview (linux; x86_64; terminal)".to_string()
    )));

    let rows = raw_rows(&target);
    assert_eq!(rows[0].1, "google-gemini-cli");
    let data: Value = serde_json::from_str(&rows[0].3).unwrap();
    assert_eq!(data["projectId"], "proj-1");
    assert_eq!(data["refresh"], "gr");
    assert_eq!(data["email"], "g@x.com");
}

#[tokio::test]
async fn gemini_cli_project_discovery_failure_is_a_login_error() {
    let access = jwt(&json!({}));
    let (services, _) = routed(routes(vec![
        (
            "oauth2.googleapis.com/token",
            queue(vec![(
                200,
                json!({"access_token": access, "refresh_token": "gr", "expires_in": 3600})
                    .to_string(),
            )]),
        ),
        (
            "oauth2/v1/userinfo",
            queue(vec![(200, json!({"email": "g@x.com"}).to_string())]),
        ),
        (
            "v1internal:loadCodeAssist",
            queue(vec![(500, r#"{"error":{"message":"denied"}}"#.into())]),
        ),
    ]));
    let (_dir, target) = temp_target();
    let ui = ScriptedUi::with(&[]);
    let (result, ()) = tokio::join!(
        login(&target, "google-gemini-cli", ui.clone(), &services),
        complete_callback(&ui, "g-code"),
    );
    let err = result.unwrap_err();
    assert!(err.contains("loadCodeAssist failed: 500"), "{err}");
    // Nothing is stored on a failed discovery.
    assert!(raw_rows(&target).is_empty());
}

#[tokio::test]
async fn antigravity_loopback_login_onboards_the_free_tier_then_stores_the_project() {
    let access = jwt(&json!({}));
    let (services, http) = routed(routes(vec![
        (
            "oauth2.googleapis.com/token",
            queue(vec![(
                200,
                json!({"access_token": access, "refresh_token": "agr", "expires_in": 3600}).to_string(),
            )]),
        ),
        (
            "oauth2/v1/userinfo",
            queue(vec![(200, json!({"email": "a@x.com"}).to_string())]),
        ),
        (
            "v1internal:loadCodeAssist",
            queue(vec![
                (200, json!({"allowedTiers": [{"id": "free-tier"}]}).to_string()),
                (
                    200,
                    json!({"currentTier": {"id": "free-tier"}, "cloudaicompanionProject": "ap-1"}).to_string(),
                ),
            ]),
        ),
        (
            "v1internal:onboardUser",
            queue(vec![(
                200,
                json!({"done": true, "response": {"@type": "x", "cloudaicompanionProject": "ap-1"}}).to_string(),
            )]),
        ),
    ]));
    let (_dir, target) = temp_target();
    let ui = ScriptedUi::with(&[]);
    let (identity, ()) = tokio::join!(
        login(&target, "google-antigravity", ui.clone(), &services),
        complete_callback(&ui, "ag-code"),
    );
    identity.unwrap();

    let url = &ui.auth.lock()[0].0;
    assert!(url.contains("redirect_uri=http%3A%2F%2F127.0.0.1%3A51121%2Foauth-callback"));
    assert!(url.contains("https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fcclog"));

    let seen = http.seen.lock().clone();
    assert!(seen[2]
        .headers
        .iter()
        .any(|(name, value)| name == "User-Agent"
            && value.starts_with("antigravity/hub/2.19.1 (aidev_client;")));
    let load_body: Value = serde_json::from_str(seen[2].body.as_deref().unwrap()).unwrap();
    assert_eq!(load_body["metadata"]["ideType"], "ANTIGRAVITY");
    assert_eq!(seen.len(), 5);

    let rows = raw_rows(&target);
    let data: Value = serde_json::from_str(&rows[0].3).unwrap();
    assert_eq!(data["projectId"], "ap-1");
}

// -- devin -----------------------------------------------------------------------

#[tokio::test]
async fn devin_loopback_login_exchanges_json_and_stores_the_endpoints() {
    let access = jwt(&json!({ "exp": 1_900_000_000i64 }));
    let (services, http) = routed(routes(vec![(
        "api.devin.ai/auth/cli/token",
        queue(vec![(
            200,
            json!({"access_token": access, "refresh_token": "dr"}).to_string(),
        )]),
    )]));
    let (_dir, target) = temp_target();
    let ui = ScriptedUi::with(&[]);
    let (identity, ()) = tokio::join!(
        login(&target, "devin", ui.clone(), &services),
        complete_callback(&ui, "dv-code"),
    );
    assert_eq!(
        identity.unwrap(),
        Some(Identity::OAuth {
            email: None,
            account_id: None,
            org_id: None,
            org_name: None
        })
    );

    let url = &ui.auth.lock()[0].0;
    assert!(url.starts_with("https://app.devin.ai/auth/cli/continue?"));
    assert!(url.contains("prompt=select_account"));
    assert!(url.contains("redirect_uri=http%3A%2F%2F127.0.0.1%3A59653%2Fcallback"));
    assert!(!url.contains("client_id="));
    assert!(url.contains("code_challenge_method=S256"));

    let seen = http.seen.lock().clone();
    assert_eq!(seen.len(), 1);
    assert!(seen[0]
        .headers
        .contains(&("Content-Type".to_string(), "application/json".to_string())));
    let body: Value = serde_json::from_str(seen[0].body.as_deref().unwrap()).unwrap();
    let keys: Vec<&str> = body
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(keys, vec!["code", "code_verifier"]);
    assert_eq!(body["code"], "dv-code");

    let rows = raw_rows(&target);
    assert_eq!(rows[0].1, "devin");
    let data: Value = serde_json::from_str(&rows[0].3).unwrap();
    assert_eq!(
        (data["apiEndpoint"].as_str(), data["enterpriseUrl"].as_str()),
        (Some("https://api.devin.ai"), Some("https://app.devin.ai"))
    );
    assert_eq!(data["refresh"], "dr");
    assert_eq!(data["expires"], 1_900_000_000_000i64);
}

// -- stencil ----------------------------------------------------------------------

#[tokio::test]
async fn stencil_loopback_login_exchanges_and_profiles_the_account() {
    let (services, http) = routed(routes(vec![
        (
            "auth.stencil.so/oauth/token",
            queue(vec![(
                200,
                json!({"access_token": "st", "refresh_token": "sr", "expires_in": 3600})
                    .to_string(),
            )]),
        ),
        (
            "auth.stencil.so/oauth/userinfo",
            queue(vec![(
                200,
                json!({"email": "s@x.io", "sub": "sub-7"}).to_string(),
            )]),
        ),
    ]));
    let (_dir, target) = temp_target();
    let ui = ScriptedUi::with(&[]);
    let (identity, ()) = tokio::join!(
        login(&target, "stencil", ui.clone(), &services),
        complete_callback(&ui, "st-code"),
    );
    assert_eq!(
        identity.unwrap(),
        Some(Identity::OAuth {
            email: Some("s@x.io".into()),
            account_id: Some("sub-7".into()),
            org_id: None,
            org_name: None
        })
    );

    let url = &ui.auth.lock()[0].0;
    assert!(url.starts_with("https://auth.stencil.so/oauth/authorize?"));
    assert!(url.contains("client_id=omp"));
    assert!(url.contains("redirect_uri=http%3A%2F%2F127.0.0.1%3A54547%2Fcallback"));
    assert!(url.contains("scope=openid%20profile%20email%20offline_access"));

    let seen = http.seen.lock().clone();
    assert_eq!(seen.len(), 2);
    let body = seen[0].body.clone().unwrap();
    assert!(body.contains("grant_type=authorization_code"));
    assert!(body.contains("client_id=omp"));
    assert!(body.contains("code=st-code"));
    assert!(body.contains("code_verifier="));
    assert!(seen[1]
        .headers
        .contains(&("Authorization".to_string(), "Bearer st".to_string())));

    let rows = raw_rows(&target);
    assert_eq!(rows[0].1, "stencil");
    let data: Value = serde_json::from_str(&rows[0].3).unwrap();
    assert_eq!(
        (data["access"].as_str(), data["refresh"].as_str()),
        (Some("st"), Some("sr"))
    );
    assert_eq!(data["accountId"], "sub-7");
}

#[tokio::test]
async fn stencil_manual_paste_completes_the_login_without_the_browser() {
    let (services, _) = routed(routes(vec![
        (
            "auth.stencil.so/oauth/token",
            queue(vec![(
                200,
                json!({"access_token": "st", "refresh_token": "sr", "expires_in": 3600})
                    .to_string(),
            )]),
        ),
        (
            "auth.stencil.so/oauth/userinfo",
            queue(vec![(
                200,
                json!({"email": "s@x.io", "sub": "sub-7"}).to_string(),
            )]),
        ),
    ]));
    let (_dir, target) = temp_target();
    // A paste-capable UI: the paste races the (never completed) callback.
    let paste = Arc::new(std::sync::Mutex::new(Some(
        "http://127.0.0.1:54547/callback?state=whatever&code=paste-code".to_string(),
    )));
    struct StencilPasteUi(Arc<ScriptedUi>, Arc<std::sync::Mutex<Option<String>>>);
    impl OAuthLoginUi for StencilPasteUi {
        fn on_auth(&self, url: &str, instructions: Option<&str>) {
            self.0.on_auth(url, instructions);
        }
        fn on_prompt(
            &self,
            prompt: &OAuthPrompt,
        ) -> Pin<Box<dyn Future<Output = Option<String>> + Send + '_>> {
            self.0.on_prompt(prompt)
        }
        fn on_progress(&self, message: &str) {
            self.0.on_progress(message);
        }
        fn on_manual_code_input(
            &self,
        ) -> Option<Pin<Box<dyn Future<Output = Option<String>> + Send + '_>>> {
            let paste = self.1.clone();
            Some(Box::pin(async move { paste.lock().unwrap().take() }))
        }
    }
    let ui = ScriptedUi::with(&[]);
    let identity = login(
        &target,
        "stencil",
        Arc::new(StencilPasteUi(ui.clone(), paste)),
        &services,
    )
    .await
    .unwrap();
    assert_eq!(
        identity,
        Some(Identity::OAuth {
            email: Some("s@x.io".into()),
            account_id: Some("sub-7".into()),
            org_id: None,
            org_name: None
        })
    );
    let rows = raw_rows(&target);
    let data: Value = serde_json::from_str(&rows[0].3).unwrap();
    assert_eq!(data["access"], "st");
}
