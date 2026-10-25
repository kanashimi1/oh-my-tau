//! `/browser`, `/computer`: browser automation and desktop screenshots.
//!
//! `/browser` drives a local Chromium/Chrome over the `DevTools` protocol
//! (launched headless with `--remote-debugging-port` and a temp profile);
//! screenshots are saved as PNGs in the temp dir and their path is printed.
//! `/computer` captures the desktop via whichever screenshot tool is
//! installed (`grim`, `import`, `gnome-screenshot`).

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::Mutex;

use anyhow::{anyhow, bail, Context, Result};
use base64::Engine;
use futures::{SinkExt, StreamExt};
use serde_json::{json, Value};
use tokio_tungstenite::tungstenite::Message;

use crate::rpc::slash::Outcome;
use crate::runtime::Runtime;

/// Chrome-family binaries tried in order.
const CHROME_CANDIDATES: [&str; 4] = [
    "google-chrome",
    "google-chrome-stable",
    "chromium",
    "chromium-browser",
];

/// Desktop screenshot tools tried in order (first found wins).
const SHOT_TOOLS: [(&str, &[&str]); 3] = [
    ("grim", &["grim"]),
    ("import", &["import", "-window", "root"]),
    ("gnome-screenshot", &["gnome-screenshot", "-f"]),
];

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Find an executable on `PATH`.
#[must_use]
pub fn find_on_path(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|dir| dir.join(name))
        .find(|p| p.is_file())
}

/// Revision number of a Playwright cache dir named `chromium-<rev>` (0 when
/// unparseable); used to try the newest browser first.
#[must_use]
fn playwright_rev(dir: &Path) -> u64 {
    dir.file_name()
        .and_then(|n| n.to_str())
        .and_then(|n| n.strip_prefix("chromium-"))
        .and_then(|rev| rev.parse().ok())
        .unwrap_or(0)
}

/// Pure ordering of chrome candidates: an explicit env override
/// (`OMT_CHROME` / `CHROME_PATH`) first, then PATH hits in candidate order,
/// then Playwright cache binaries, newest revision first.
#[must_use]
fn ordered_chrome_candidates(
    env_chrome: Option<PathBuf>,
    path_hits: Vec<PathBuf>,
    mut playwright: Vec<PathBuf>,
) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Some(p) = env_chrome.filter(|p| !p.as_os_str().is_empty()) {
        out.push(p);
    }
    out.extend(path_hits);
    playwright.sort_by_key(|d| std::cmp::Reverse(playwright_rev(d)));
    out.extend(playwright);
    out
}

/// Playwright cache chrome binaries (`~/.cache/ms-playwright/chromium-*/…`),
/// newest revision first.
fn playwright_chrome_bins() -> Vec<PathBuf> {
    let Some(home) = std::env::var_os("HOME")
        .map(PathBuf::from)
        .filter(|p| !p.as_os_str().is_empty())
    else {
        return Vec::new();
    };
    let Ok(entries) = std::fs::read_dir(home.join(".cache").join("ms-playwright")) else {
        return Vec::new();
    };
    let mut dirs: Vec<PathBuf> = entries
        .flatten()
        .filter(|e| {
            e.file_name().to_str().is_some_and(|n| {
                n.starts_with("chromium-") && n["chromium-".len()..].parse::<u64>().is_ok()
            })
        })
        .map(|e| e.path())
        .collect();
    dirs.sort_by_key(|d| std::cmp::Reverse(playwright_rev(d)));
    let mut bins = Vec::new();
    for dir in dirs {
        // chrome-linux64: newer layout; chrome-linux: legacy layout.
        for sub in ["chrome-linux64/chrome", "chrome-linux/chrome"] {
            let p = dir.join(sub);
            if p.is_file() {
                bins.push(p);
                break;
            }
        }
    }
    bins
}

/// Locate a usable Chromium/Chrome binary:
/// `OMT_CHROME` / `CHROME_PATH`, then PATH, then the Playwright cache.
fn discover_chrome() -> Option<PathBuf> {
    let env_chrome = std::env::var("OMT_CHROME")
        .or_else(|_| std::env::var("CHROME_PATH"))
        .ok()
        .map(PathBuf::from);
    let path_hits: Vec<PathBuf> = CHROME_CANDIDATES
        .iter()
        .filter_map(|n| find_on_path(n))
        .collect();
    ordered_chrome_candidates(env_chrome, path_hits, playwright_chrome_bins())
        .into_iter()
        .find(|p| p.is_file())
}

fn temp_stamp() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or_default()
}

/// One CDP request/response over a fresh websocket connection.
async fn cdp_call(ws_url: &str, method: &str, params: Value) -> Result<Value> {
    let (mut ws, _) = tokio_tungstenite::connect_async(ws_url)
        .await
        .with_context(|| format!("failed to connect to {ws_url}"))?;
    let id = 1;
    ws.send(Message::text(
        json!({"id": id, "method": method, "params": params}).to_string(),
    ))
    .await?;
    let deadline = tokio::time::timeout(Duration::from_secs(20), async {
        while let Some(msg) = ws.next().await {
            if let Message::Text(text) = msg? {
                let v: Value = serde_json::from_str(text.as_str())?;
                if v.get("id").and_then(Value::as_u64) == Some(id) {
                    if let Some(err) = v.get("error") {
                        bail!("CDP {method} failed: {err}");
                    }
                    return Ok(v.get("result").cloned().unwrap_or(Value::Null));
                }
            }
        }
        bail!("websocket closed before {method} responded")
    })
    .await
    .map_err(|_| anyhow!("timed out waiting for CDP {method} response"))?;
    deadline
}

// ---------------------------------------------------------------------------
// Browser session
// ---------------------------------------------------------------------------

struct BrowserSession {
    child: tokio::process::Child,
    port: u16,
    profile: PathBuf,
}

static SESSION: Mutex<Option<BrowserSession>> = Mutex::const_new(None);

async fn devtools_http(port: u16, path: &str) -> Result<Value> {
    let url = format!("http://127.0.0.1:{port}/{path}");
    let resp = reqwest::Client::new()
        .get(&url)
        .timeout(Duration::from_secs(5))
        .send()
        .await
        .with_context(|| format!("request to {url} failed"))?;
    let text = resp.text().await?;
    serde_json::from_str(&text).with_context(|| format!("bad JSON from {url}: {text}"))
}

/// List `DevTools` page targets.
async fn page_targets(port: u16) -> Result<Vec<Value>> {
    let v = devtools_http(port, "json/list").await?;
    Ok(v.as_array()
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .filter(|t| t.get("type").and_then(Value::as_str) == Some("page"))
        .collect())
}

/// Launch a browser if needed and return its `DevTools` port.
async fn session_port() -> Result<u16> {
    {
        let guard = SESSION.lock().await;
        if let Some(session) = guard.as_ref() {
            return Ok(session.port);
        }
    }
    let session = launch_session().await?;
    let port = session.port;
    let mut guard = SESSION.lock().await;
    if guard.is_none() {
        *guard = Some(session);
    } else {
        // Raced with another launch: keep the winner, kill ours.
        let mut extra = session;
        let _ = extra.child.start_kill();
        let _ = std::fs::remove_dir_all(&extra.profile);
    }
    Ok(port)
}

async fn launch_session() -> Result<BrowserSession> {
    let bin = discover_chrome().ok_or_else(|| {
        anyhow!(
            "no Chromium/Chrome found (tried PATH: {}, $OMT_CHROME, $CHROME_PATH, \
             ~/.cache/ms-playwright/chromium-*/); install one to use /browser",
            CHROME_CANDIDATES.join(", ")
        )
    })?;
    // Reserve a free TCP port for --remote-debugging-port.
    let port = {
        let l = tokio::net::TcpListener::bind(("127.0.0.1", 0)).await?;
        l.local_addr()?.port()
    };
    let profile = std::env::temp_dir().join(format!("omt-browser-profile-{}", temp_stamp()));
    let mut child = tokio::process::Command::new(&bin)
        .args([
            "--headless=new",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-gpu",
            &format!("--remote-debugging-port={port}"),
        ])
        .arg(format!("--user-data-dir={}", profile.display()))
        .arg("about:blank")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .with_context(|| format!("failed to launch {}", bin.display()))?;
    // Wait for the DevTools endpoint to come up.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(15);
    loop {
        if tokio::time::Instant::now() >= deadline {
            let _ = child.start_kill();
            bail!("{} did not open a DevTools port within 15s", bin.display());
        }
        if child.try_wait()?.is_some() {
            bail!("{} exited immediately", bin.display());
        }
        if devtools_http(port, "json/version").await.is_ok() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    Ok(BrowserSession {
        child,
        port,
        profile,
    })
}

async fn shutdown_session(guard: &mut Option<BrowserSession>) {
    if let Some(mut session) = guard.take() {
        let _ = session.child.start_kill();
        let _ = session.child.wait().await;
        let _ = std::fs::remove_dir_all(&session.profile);
    }
}

// ---------------------------------------------------------------------------
// Subcommands
// ---------------------------------------------------------------------------

async fn browser_open(port: u16, url: &str) -> Result<String> {
    if url.is_empty() {
        bail!("usage: /browser open <url>");
    }
    // Newer Chrome requires PUT for /json/new; older builds only GET.
    let new_url = format!("http://127.0.0.1:{port}/json/new?{url}");
    let client = reqwest::Client::new();
    let resp = client
        .put(&new_url)
        .timeout(Duration::from_secs(5))
        .send()
        .await
        .ok()
        .filter(|r| r.status().is_success());
    let text = match resp {
        Some(r) => r.text().await?,
        None => {
            client
                .get(&new_url)
                .timeout(Duration::from_secs(5))
                .send()
                .await
                .context("failed to create target")?
                .text()
                .await?
        }
    };
    let target: Value = serde_json::from_str(&text)
        .with_context(|| format!("unexpected response creating target: {text}"))?;
    Ok(format!(
        "browser: opened {} (id {})",
        target.get("url").and_then(Value::as_str).unwrap_or(url),
        target.get("id").and_then(Value::as_str).unwrap_or("?")
    ))
}

async fn browser_screenshot(port: u16) -> Result<String> {
    let targets = page_targets(port).await?;
    let target = targets
        .iter()
        .find_map(|t| t.get("webSocketDebuggerUrl").and_then(Value::as_str))
        .ok_or_else(|| anyhow!("no open page target; use /browser open <url> first"))?;
    let result = cdp_call(target, "Page.captureScreenshot", json!({"format": "png"})).await?;
    let data = result
        .get("data")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("screenshot response missing image data"))?;
    let png = base64::engine::general_purpose::STANDARD
        .decode(data)
        .context("invalid base64 screenshot data")?;
    let path = std::env::temp_dir().join(format!("omt-browser-{}.png", temp_stamp()));
    std::fs::write(&path, &png).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(format!("browser: screenshot saved to {}", path.display()))
}

// ---------------------------------------------------------------------------
// Slash command entry points
// ---------------------------------------------------------------------------

/// Slash-command entry point for `/browser`.
pub async fn command(rt: &std::sync::Arc<Runtime>, args: &str) -> Outcome {
    let out = match run_browser(args).await {
        Ok(text) | Err(text) => text,
    };
    rt.send(serde_json::json!({"type": "command_output", "text": out}));
    Outcome::Handled {
        agent_invoked: false,
    }
}

async fn run_browser(args: &str) -> std::result::Result<String, String> {
    let tokens: Vec<&str> = args.split_whitespace().collect();
    match tokens.first().copied() {
        None => Err("usage: /browser open <url> | screenshot | close".into()),
        Some("open") => {
            let url = args.trim_start_matches("open").trim();
            match session_port().await {
                Ok(port) => browser_open(port, url)
                    .await
                    .map_err(|e| format!("browser: {e:#}")),
                Err(e) => Err(format!("browser: {e:#}")),
            }
        }
        Some("screenshot" | "shot") => match session_port().await {
            Ok(port) => browser_screenshot(port)
                .await
                .map_err(|e| format!("browser: {e:#}")),
            Err(e) => Err(format!("browser: {e:#}")),
        },
        Some("close") => {
            let mut guard = SESSION.lock().await;
            shutdown_session(&mut guard).await;
            Ok("browser: closed".into())
        }
        Some(other) => Err(format!(
            "browser: unknown subcommand \"{other}\" (open <url> | screenshot | close)"
        )),
    }
}

/// Slash-command entry point for `/computer`: desktop screenshot.
pub async fn computer_command(rt: &std::sync::Arc<Runtime>, args: &str) -> Outcome {
    let out = run_computer(args).await;
    rt.send(serde_json::json!({"type": "command_output", "text": out}));
    Outcome::Handled {
        agent_invoked: false,
    }
}

async fn run_computer(_args: &str) -> String {
    match SHOT_TOOLS
        .iter()
        .find_map(|(name, argv)| find_on_path(name).map(|bin| (bin, *argv)))
    {
        Some((bin, argv)) => {
            let path = std::env::temp_dir().join(format!("omt-desktop-{}.png", temp_stamp()));
            let mut cmd = tokio::process::Command::new(&bin);
            cmd.args(argv).arg(&path);
            match cmd.output().await {
                Ok(out) if out.status.success() && path.exists() => {
                    format!("computer: desktop screenshot saved to {}", path.display())
                }
                Ok(out) => format!(
                    "computer: {} failed ({}): {}",
                    bin.display(),
                    out.status,
                    String::from_utf8_lossy(&out.stderr).trim()
                ),
                Err(e) => format!("computer: failed to run {}: {e}", bin.display()),
            }
        }
        None => format!(
            "computer: no desktop screenshot tool found; install one of: {}",
            SHOT_TOOLS
                .iter()
                .map(|(n, _)| *n)
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_real_binaries_and_misses_bogus_ones() {
        // `sh` is guaranteed in the test environment.
        assert!(find_on_path("sh").is_some());
        assert!(find_on_path("omt-definitely-not-a-binary").is_none());
    }

    #[test]
    fn posix_screenshot_tool_order_is_stable() {
        assert_eq!(SHOT_TOOLS[0].0, "grim");
        assert_eq!(SHOT_TOOLS[1].0, "import");
        assert_eq!(SHOT_TOOLS[2].0, "gnome-screenshot");
    }

    #[tokio::test]
    async fn browser_usage_errors() {
        assert!(run_browser("").await.unwrap_err().contains("usage"));
        assert!(run_browser("frobnicate")
            .await
            .unwrap_err()
            .contains("unknown subcommand"));
    }

    #[test]
    fn chrome_discovery_order() {
        let env = Some(PathBuf::from("/opt/custom/chrome"));
        let path_hits = vec![PathBuf::from("/usr/bin/google-chrome")];
        let playwright = vec![
            PathBuf::from("/cache/ms-playwright/chromium-1100"),
            PathBuf::from("/cache/ms-playwright/chromium-1200"),
        ];
        let order = ordered_chrome_candidates(env, path_hits.clone(), playwright.clone());
        assert_eq!(
            order,
            vec![
                PathBuf::from("/opt/custom/chrome"),
                PathBuf::from("/usr/bin/google-chrome"),
                PathBuf::from("/cache/ms-playwright/chromium-1200"),
                PathBuf::from("/cache/ms-playwright/chromium-1100"),
            ]
        );
        // No env override: PATH hits come first.
        let order = ordered_chrome_candidates(None, path_hits.clone(), playwright.clone());
        assert_eq!(order[0], PathBuf::from("/usr/bin/google-chrome"));
        // Empty env value is ignored.
        let order = ordered_chrome_candidates(Some(PathBuf::from("")), path_hits, playwright);
        assert_eq!(order[0], PathBuf::from("/usr/bin/google-chrome"));
    }

    #[test]
    fn playwright_rev_sorts_newest_first() {
        assert_eq!(playwright_rev(&PathBuf::from("/x/chromium-1148")), 1148);
        assert_eq!(playwright_rev(&PathBuf::from("/x/not-chromium")), 0);
        let mut dirs = [
            PathBuf::from("/c/chromium-9"),
            PathBuf::from("/c/chromium-1148"),
        ];
        dirs.sort_by_key(|d| std::cmp::Reverse(playwright_rev(d)));
        assert_eq!(dirs[0], PathBuf::from("/c/chromium-1148"));
    }
}
