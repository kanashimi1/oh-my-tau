//! Delivery into herdr panes (`SPEC.md` item 5): the hub itself delivers
//! envelopes addressed to `w4:pXX` recipients through the `herdr` CLI,
//! rendering the existing 4-line `[FROM]/[ID]/[POINTER]/[EXPECTED]` head so
//! current agents understand it.
//!
//! Policy:
//! - dead pane (not in `herdr agent list`) -> the send fails fast;
//! - urgent or reply-expected -> deliver the full text once the pane is
//!   idle (poll no faster than 2 s, give up to `failed` after the deadline);
//! - notes -> one short wake line once the pane is idle;
//! - `herdr agent prompt` first; omt panes on herdr 0.9.4 are never
//!   registered as active named agents (audit 2026-10-25), so fall back to
//!   `herdr pane send-text` + an `enter` key.

use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use serde::Deserialize;
use tokio::process::Command;

use crate::envelope::{pane_text, Envelope, State, PANE_WAKE_TOOL};
use crate::store::Store;

/// Minimum poll interval for pane status (`SPEC.md`: no faster than 2 s).
pub const POLL_INTERVAL: Duration = Duration::from_secs(2);
/// Upper bound for delivering a note that found no idle moment.
const NOTE_GIVE_UP_MS: u64 = 30 * 60 * 1000;

/// A pane as listed by `herdr agent list`.
#[derive(Debug, Deserialize)]
struct AgentList {
    #[serde(rename = "type")]
    _ty: Option<String>,
    result: AgentListResult,
}

#[derive(Debug, Deserialize)]
struct AgentListResult {
    agents: Vec<AgentInfo>,
}

#[derive(Debug, Deserialize)]
struct AgentInfo {
    pane_id: String,
}

/// `herdr agent get` response.
#[derive(Debug, Deserialize)]
struct AgentGet {
    result: AgentGetResult,
}

#[derive(Debug, Deserialize)]
struct AgentGetResult {
    agent: AgentStatus,
}

#[derive(Debug, Deserialize)]
struct AgentStatus {
    agent_status: String,
}

/// Is this pane currently listed by `herdr agent list`?
///
/// # Errors
/// When herdr itself is unreachable or answers garbage: the transport is
/// herdr, so fail closed (treat as dead).
pub async fn pane_exists(pane: &str) -> anyhow::Result<bool> {
    let out = Command::new("herdr")
        .args(["agent", "list"])
        .stdin(Stdio::null())
        .output()
        .await
        .map_err(|e| anyhow::anyhow!("herdr agent list: {e}"))?;
    if !out.status.success() {
        anyhow::bail!("herdr agent list: exit {}", out.status);
    }
    let parsed: AgentList = serde_json::from_slice(&out.stdout)
        .map_err(|e| anyhow::anyhow!("herdr agent list: bad JSON: {e}"))?;
    Ok(parsed.result.agents.iter().any(|a| a.pane_id == pane))
}

/// Status of a pane per `herdr agent get` ("unknown" on any failure).
pub async fn pane_status(pane: &str) -> String {
    let Ok(out) = Command::new("herdr")
        .args(["agent", "get", pane])
        .stdin(Stdio::null())
        .output()
        .await
    else {
        return "unknown".to_string();
    };
    if !out.status.success() {
        return "unknown".to_string();
    }
    serde_json::from_slice::<AgentGet>(&out.stdout)
        .map_or_else(|_| "unknown".to_string(), |g| g.result.agent.agent_status)
}

/// A settled pane accepts input ("idle" or "done" — same rule as
/// `herdr agent wait`).
#[must_use]
pub fn is_idle(status: &str) -> bool {
    status == "idle" || status == "done"
}

/// Push `text` into a pane: `herdr agent prompt` first, then the
/// send-text/send-keys fallback. Returns the mechanism used.
///
/// # Errors
/// Every herdr invocation failure (spawn, non-zero exit, missing pane).
pub async fn deliver_text(pane: &str, text: &str) -> anyhow::Result<&'static str> {
    let prompt = Command::new("herdr")
        .args(["agent", "prompt", pane, text])
        .stdin(Stdio::null())
        .output()
        .await
        .map_err(|e| anyhow::anyhow!("herdr agent prompt {pane}: {e}"))?;
    if prompt.status.success() {
        return Ok("agent-prompt");
    }
    let stderr = String::from_utf8_lossy(&prompt.stderr);
    tracing::debug!(pane, stderr = %stderr, "agent prompt failed; falling back to pane send-text");
    let send = Command::new("herdr")
        .args(["pane", "send-text", pane, text])
        .stdin(Stdio::null())
        .output()
        .await
        .map_err(|e| anyhow::anyhow!("herdr pane send-text {pane}: {e}"))?;
    if !send.status.success() {
        anyhow::bail!(
            "herdr pane send-text {pane}: exit {}: {}",
            send.status,
            String::from_utf8_lossy(&send.stderr)
        );
    }
    let enter = Command::new("herdr")
        .args(["pane", "send-keys", pane, "enter"])
        .stdin(Stdio::null())
        .output()
        .await
        .map_err(|e| anyhow::anyhow!("herdr pane send-keys {pane}: {e}"))?;
    if !enter.status.success() {
        anyhow::bail!(
            "herdr pane send-keys {pane}: exit {}: {}",
            enter.status,
            String::from_utf8_lossy(&enter.stderr)
        );
    }
    Ok("pane-send-text")
}

/// One short wake line for notes, pointing the pane at its inbox.
#[must_use]
pub fn wake_line(env: &Envelope) -> String {
    format!(
        "[hub] new envelope for you (id={}): {} {}",
        env.message_id, PANE_WAKE_TOOL, env.recipient
    )
}

/// Spawn the background delivery task for one envelope addressed to a pane.
/// Urgent or reply-expected envelopes get the full text, notes the wake
/// line; give up to `failed` after the deadline. `hold` lives as long as the
/// task (the server's busy guard: no idle exit mid-delivery).
pub fn spawn_delivery(store: Arc<Store>, env: Envelope, hold: impl Send + 'static) {
    let reply_expected = env.is_reply_expected();
    tokio::spawn(async move {
        let _hold = hold;
        deliver_to_pane(&store, env, reply_expected).await;
    });
}

#[allow(clippy::too_many_lines)] // poll loop: liveness -> idle -> deliver -> deadline.
async fn deliver_to_pane(store: &Arc<Store>, env: Envelope, reply_expected: bool) {
    let pane = env.recipient.clone();
    let id = env.message_id.clone();
    let full = reply_expected || env.priority.0 == 0;
    let deadline = env
        .deadline
        .unwrap_or_else(|| crate::now_ms() + NOTE_GIVE_UP_MS);
    loop {
        if !pane_exists(&pane).await.unwrap_or(false) {
            tracing::warn!(pane, id, "pane vanished; marking failed");
            let _ = store.transition(&id, State::FAILED, crate::now_ms());
            return;
        }
        if is_idle(&pane_status(&pane).await) {
            let text = if full {
                pane_text(&env)
            } else {
                wake_line(&env)
            };
            match deliver_text(&pane, &text).await {
                Ok(mechanism) => {
                    tracing::info!(pane, id, mechanism, "delivered to pane");
                    let now = crate::now_ms();
                    let _ = store.transition(&id, State::DELIVERED, now);
                    return;
                }
                Err(e) => {
                    tracing::warn!(pane, id, error = %e, "pane delivery failed; will retry");
                }
            }
        }
        if crate::now_ms() >= deadline {
            tracing::warn!(pane, id, "deadline reached; marking failed");
            let _ = store.transition(&id, State::FAILED, crate::now_ms());
            return;
        }
        tokio::time::sleep(POLL_INTERVAL).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idle_statuses() {
        assert!(is_idle("idle"));
        assert!(is_idle("done"));
        assert!(!is_idle("working"));
        assert!(!is_idle("blocked"));
        assert!(!is_idle("unknown"));
    }

    #[test]
    fn wake_line_points_at_inbox() {
        let env = Envelope {
            message_id: "m1".into(),
            recipient: "w4:p8T".into(),
            ..Envelope::default()
        };
        let line = wake_line(&env);
        assert!(line.contains("id=m1"));
        assert!(line.contains("omt hub inbox w4:p8T"));
        assert!(!line.contains('\n'));
    }
}
