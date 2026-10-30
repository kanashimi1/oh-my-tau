//! `/share`: share the session via a secret gist. TS:
//! `slash-commands/builtin-collaboration.ts` + `export/share.ts` (gist store).
//!
//! The TS flow seals and encrypts the session before pushing to a gist; the
//! port keeps the simpler documented slice: export the live session to
//! standalone HTML (`omp export`, `subcmd::export`) and upload it as a secret
//! gist. `gh gist create` is used when the `gh` CLI is present and
//! authenticated; otherwise a `GITHUB_TOKEN` POST to the gists API is used
//! with the same reqwest client the crate already uses. The gist URL is
//! printed on success; failures produce clear messages.

use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;

use base64::Engine as _;
use serde_json::json;

use crate::rpc::slash::Outcome;
use crate::runtime::Runtime;

const GIST_FILENAME: &str = "omp-session.html";

fn output(rt: &Runtime, text: &str) {
    rt.send(json!({"type": "command_output", "text": text}));
}

fn handled() -> Outcome {
    Outcome::Handled {
        agent_invoked: false,
    }
}

/// Slash-command entry point. `/share` or `/share gist` (same behavior;
/// the TS share-server store is not ported).
pub async fn command(rt: &Arc<Runtime>, args: &str) -> Outcome {
    let verb = args.trim().to_lowercase();
    if !verb.is_empty() && verb != "gist" {
        output(
            rt,
            "Usage: /share [gist]\n\nShares the session as a secret GitHub gist (the encrypted share-server store is not ported).",
        );
        return handled();
    }
    // 1) Export the live session to standalone HTML.
    let Some(session_file) = rt.live_session_file() else {
        output(
            rt,
            "Share: no session file to export (the session was started with --no-session).",
        );
        return handled();
    };
    let html = match crate::subcmd::export::render_html(&session_file) {
        Ok(html) => html,
        Err(e) => {
            output(rt, &format!("Share: failed to export the session: {e:#}"));
            return handled();
        }
    };
    // Stage to a temp file so `gh gist create` can read it.
    let tmp = std::env::temp_dir().join(format!("omp-share-{}.html", std::process::id()));
    if let Err(e) = std::fs::write(&tmp, &html) {
        output(
            rt,
            &format!("Share: failed to write {}: {e}", tmp.display()),
        );
        return handled();
    }
    // 2) Upload: gh CLI first, GITHUB_TOKEN second.
    let result = match upload_with_gh(&tmp) {
        Ok(url) => Ok(url),
        Err(GhError::Missing | GhError::Unauthenticated) => match std::env::var("GITHUB_TOKEN") {
            Ok(token) if !token.trim().is_empty() => {
                upload_with_token(token.trim(), GIST_FILENAME, &html).await
            }
            _ => Err("Share: no upload path available — install and authenticate the GitHub CLI (`gh auth login`) or set GITHUB_TOKEN.".to_string()),
        },
        Err(GhError::Failed(msg)) => Err(format!("Share: gh gist create failed: {msg}")),
    };
    let _ = std::fs::remove_file(&tmp);
    match result {
        Ok(url) => output(rt, &format!("Session shared: {url}")),
        Err(msg) => output(rt, &msg),
    }
    handled()
}

enum GhError {
    /// `gh` is not installed.
    Missing,
    /// `gh` exists but is not authenticated.
    Unauthenticated,
    /// `gh` ran but the gist creation failed.
    Failed(String),
}

fn gh_available() -> Result<(), GhError> {
    match Command::new("gh").arg("--version").output() {
        Ok(out) if out.status.success() => Ok(()),
        Ok(_) => Err(GhError::Failed("gh --version failed".to_string())),
        Err(_) => Err(GhError::Missing),
    }
}

fn upload_with_gh(file: &PathBuf) -> Result<String, GhError> {
    gh_available()?;
    // TS checks `gh auth status` before relying on gh.
    let auth = Command::new("gh").arg("auth").arg("status").output();
    match auth {
        Ok(out) if out.status.success() => {}
        Ok(_) => return Err(GhError::Unauthenticated),
        Err(e) => return Err(GhError::Failed(format!("run gh: {e}"))),
    }
    let out = Command::new("gh")
        .args(["gist", "create", "--public=false"])
        .arg(file)
        .output()
        .map_err(|e| GhError::Failed(format!("run gh: {e}")))?;
    if !out.status.success() {
        return Err(GhError::Failed(
            String::from_utf8_lossy(&out.stderr).trim().to_string(),
        ));
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    extract_gist_url(&stdout).ok_or_else(|| {
        GhError::Failed(format!(
            "could not parse a gist URL from gh output: {}",
            stdout.trim()
        ))
    })
}

/// Extract the first `https://gist.github.com/…` URL from free-form output.
pub fn extract_gist_url(text: &str) -> Option<String> {
    let start = text.find("https://gist.github.com/")?;
    let rest = &text[start..];
    let end = rest
        .char_indices()
        .find(|(_, c)| c.is_whitespace())
        .map_or(rest.len(), |(i, _)| i);
    let url = &rest[..end];
    let url = url.trim_end_matches(|c: char| ")]}.,".contains(c));
    Some(url.to_string())
}

/// Secret gist via the REST API with a `GITHUB_TOKEN` bearer.
async fn upload_with_token(token: &str, filename: &str, content: &str) -> Result<String, String> {
    let body = json!({
        "description": "omp session export",
        "public": false,
        "files": { filename: { "content": content } },
    });
    let resp = reqwest::Client::new()
        .post("https://api.github.com/gists")
        .bearer_auth(token)
        .header("User-Agent", "omp-rs")
        .header("Accept", "application/vnd.github+json")
        .header("X-GitHub-Api-Version", "2022-11-28")
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("Share: GitHub API unreachable: {e}"))?;
    let status = resp.status();
    let text = resp.text().await.unwrap_or_default();
    if !status.is_success() {
        let hint = match status.as_u16() {
            401 => " (token rejected; check GITHUB_TOKEN)",
            403 => " (forbidden; the token may lack gist scope or the rate limit is hit)",
            _ => "",
        };
        return Err(format!("Share: GitHub API error HTTP {status}{hint}"));
    }
    let value: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| format!("Share: GitHub API returned invalid JSON: {e}"))?;
    value
        .get("html_url")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| "Share: GitHub API response has no html_url".to_string())
}

/// Base64 helper kept for parity with the TS gist payload (unused today; the
/// HTML is stored verbatim). Reveals intent if the sealed-blob flow is ported.
#[allow(dead_code)]
fn b64(data: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_gist_url_from_gh_output() {
        assert_eq!(
            extract_gist_url("https://gist.github.com/octocat/abc123\n"),
            Some("https://gist.github.com/octocat/abc123".to_string())
        );
        // gh prints the URL on stderr in some versions; trailing junk tolerated.
        assert_eq!(
            extract_gist_url("\u{1b}some noise https://gist.github.com/u/deadbeef/ trailing"),
            Some("https://gist.github.com/u/deadbeef/".to_string())
        );
        assert_eq!(
            extract_gist_url("https://gist.github.com/u/x."),
            Some("https://gist.github.com/u/x".to_string())
        );
        assert_eq!(extract_gist_url("no url here"), None);
        assert_eq!(extract_gist_url(""), None);
    }
}
