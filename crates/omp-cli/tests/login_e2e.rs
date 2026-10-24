//! `omp-rs login` / `logout` end to end in a temp HOME against a mock Kimi OAuth host: the row
//! the Rust CLI writes is what TS `omp` reads (`omp-rs`-side reader is `omp_providers::auth`),
//! and logout soft-deletes it like TS. Never touches the real `~/.omp`.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::process::{Command, Stdio};

fn mock_kimi_host() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
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
            let body = if path.ends_with("/api/oauth/device_authorization") {
                format!(
                    r#"{{"user_code":"ABCD-1234","device_code":"dev","verification_uri":"{host}/verify","interval":1,"expires_in":60}}"#
                )
            } else {
                r#"{"access_token":"kimi-access-token","refresh_token":"kimi-refresh","expires_in":3600}"#.to_string()
            };
            let _ = write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
        }
    });
    base
}

fn omp_rs(home: &std::path::Path) -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_omt"));
    cmd.env("HOME", home)
        .env_remove("OMP_AGENT_DIR")
        .env_remove("OMP_AUTH_BROKER_URL")
        .env_remove("OMP_AUTH_BROKER_TOKEN")
        .env("BROWSER", "true")
        .env("NO_COLOR", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    cmd
}

fn run(mut cmd: Command, stdin: &str) -> (i32, String, String) {
    let mut child = cmd.spawn().unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(stdin.as_bytes())
        .unwrap();
    let out = child.wait_with_output().unwrap();
    (
        out.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&out.stdout).to_string(),
        String::from_utf8_lossy(&out.stderr).to_string(),
    )
}

fn cred(home: &tempfile::TempDir, provider: &str) -> Option<String> {
    // Local store only: this process may have OMP_AUTH_BROKER_* set, which must never be consulted.
    let db = home.path().join(".omp/agent/agent.db");
    let rows = omp_providers::login::LocalStore::open(&db)
        .unwrap()
        .list(Some(provider))
        .unwrap();
    rows.into_iter().find_map(|row| match row.credential {
        omp_providers::login::Credential::OAuth(data) => data
            .get("access")
            .and_then(|a| a.as_str())
            .map(str::to_string),
        omp_providers::login::Credential::ApiKey(key) => Some(key),
    })
}

fn rows(db: &std::path::Path) -> Vec<(String, String, String, Option<String>)> {
    let conn = rusqlite::Connection::open(db).unwrap();
    let mut stmt = conn
        .prepare("SELECT provider, credential_type, data, disabled_cause FROM auth_credentials ORDER BY id")
        .unwrap();
    stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))
        .unwrap()
        .map(Result::unwrap)
        .collect()
}

#[test]
fn login_then_logout_roundtrip_in_a_temp_home() {
    let home = tempfile::tempdir().unwrap();
    let db = home.path().join(".omp/agent/agent.db");
    let base = mock_kimi_host();

    // login kimi-code (device flow) against the mock host.
    let mut cmd = omp_rs(home.path());
    cmd.args(["login", "kimi-code"])
        .env("KIMI_CODE_OAUTH_HOST", &base);
    let (code, stdout, stderr) = run(cmd, "");
    assert_eq!(code, 0, "stderr: {stderr}\nstdout: {stdout}");
    assert!(
        stdout.contains(&format!(
            "\nOpen this URL in your browser:\n{base}/verify\nEnter code: ABCD-1234\n\n"
        )),
        "{stdout}"
    );
    assert!(stdout.contains("Requesting device authorization..."));
    assert!(stdout.contains("Waiting for device authorization..."));
    assert!(
        stdout.contains("\nLogged in to Kimi Code\n")
            || stdout.contains("Logged in to Kimi Code as"),
        "{stdout}"
    );
    assert!(
        stdout.contains(&format!("Credentials saved to {}\n", db.display())),
        "{stdout}"
    );

    let stored = rows(&db);
    assert_eq!(stored.len(), 1);
    assert_eq!(
        (stored[0].0.as_str(), stored[0].1.as_str()),
        ("kimi-code", "oauth")
    );
    assert!(
        stored[0]
            .2
            .starts_with(r#"{"access":"kimi-access-token","refresh":"kimi-refresh","expires":"#),
        "{}",
        stored[0].2
    );
    assert!(stored[0].2.contains(r#","authorizedAt":"#));
    assert_eq!(stored[0].3, None);

    // The credential resolves through the omp-rs auth reader (same one the agent uses).
    assert_eq!(
        cred(&home, "kimi-code").as_deref(),
        Some("kimi-access-token")
    );

    // api-key login through the picker (stdin) for a provider without a probe.
    let mut cmd = omp_rs(home.path());
    cmd.arg("login");
    let exa = omp_providers::login::providers::PROVIDERS
        .iter()
        .position(|(id, _)| *id == "exa")
        .unwrap()
        + 1;
    let (code, stdout, stderr) = run(cmd, &format!("{exa}\nexa-key-1\n"));
    assert_eq!(code, 0, "stderr: {stderr}\nstdout: {stdout}");
    assert!(stdout.starts_with("Select a provider:\n\n  1. ChatGPT Plus/Pro (Codex Subscription)\n  2. Anthropic (Claude Pro/Max)\n"), "{stdout}");
    assert!(stdout.contains("Enter number (1-80): "));
    assert!(stdout.contains("Open this URL in your browser:\nhttps://dashboard.exa.ai/api-keys\nCreate or copy your API key from the Exa dashboard.\n"));
    assert!(stdout.contains("Paste your Exa API key (API key): "));
    assert!(stdout.contains("\nLogged in to Exa\n"), "{stdout}");
    let stored = rows(&db);
    assert_eq!(stored[1].2, r#"{"key":"exa-key-1","source":"login"}"#);

    // Failure paths: exit code 1, TS wording on stderr.
    let (code, _, stderr) = run(
        {
            let mut c = omp_rs(home.path());
            c.args(["login", "nope"]);
            c
        },
        "",
    );
    assert_eq!(code, 1);
    assert_eq!(
        stderr,
        "Login failed: Unknown OAuth provider 'nope'. Run `omt login` to pick one.\n"
    );
    let (code, _, stderr) = run(
        {
            let mut c = omp_rs(home.path());
            c.arg("login");
            c
        },
        "99\n",
    );
    assert_eq!(code, 1);
    assert_eq!(stderr, "Login failed: Invalid selection: 99\n");
    let (code, _, stderr) = run(
        {
            let mut c = omp_rs(home.path());
            c.args(["login", "cursor"]);
            c
        },
        "",
    );
    assert_eq!(code, 1);
    assert!(
        stderr.starts_with("Login failed: Login for 'cursor' is not ported"),
        "{stderr}"
    );

    // logout: soft delete with the TS cause.
    let (code, stdout, _) = run(
        {
            let mut c = omp_rs(home.path());
            c.args(["logout", "kimi-code"]);
            c
        },
        "",
    );
    assert_eq!(code, 0);
    assert_eq!(stdout, "Logged out of kimi-code\n");
    let stored = rows(&db);
    assert_eq!(stored[0].3.as_deref(), Some("logged out by user"));
    assert_eq!(stored[1].3, None);
    assert_eq!(cred(&home, "kimi-code"), None);

    // picker lists only providers that still have credentials.
    let (code, stdout, _) = run(
        {
            let mut c = omp_rs(home.path());
            c.arg("logout");
            c
        },
        "1\n",
    );
    assert_eq!(code, 0);
    assert!(
        stdout.starts_with("Select a provider to logout:\n\n  1. exa\n"),
        "{stdout}"
    );
    assert!(stdout.ends_with("Logged out of exa\n"));
    let (_, stdout, _) = run(
        {
            let mut c = omp_rs(home.path());
            c.arg("logout");
            c
        },
        "",
    );
    assert_eq!(stdout, "No credentials stored.\n");
}
