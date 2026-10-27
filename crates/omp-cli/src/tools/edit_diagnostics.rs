//! Post-write advisory LSP diagnostics gate for `edit`/`write`.
//!
//! After a successful file write the gate asks any *already running*
//! `omp_lsp` clients ([`active_clients`]) that serve the changed files to
//! refresh their in-memory copy, then reports diagnostics that are new with
//! respect to the pre-write publish. Rules:
//!
//! - **Advisory-only**: the gate never fails the underlying edit/write call.
//! - **Baseline**: the gate only sees a pre-write publish when a client was
//!   already running; [`prewarm_spawn`] (wired into the `read` tool) starts
//!   the project's server on first read so the first edit is already gated.
//!   The gate itself still never cold-starts a server.
//! - **Budget**: at most [`GATE_BUDGET_MS`] (2000ms) is spent per call,
//!   split evenly across the changed files.
//! - **Escalation**: repeating an *identical* per-file diagnostic delta
//!   bumps the `Env` recurrence ledger and renders a `BLOCKED-RETRY` hint.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use omp_lsp::client::LspClient;
use omp_lsp::types::{Diagnostic, PublishedDiagnostics};
use omp_lsp::utils::file_to_uri;
use serde_json::{json, Value};

use super::util::format_path_relative_to_cwd;
use super::Env;

/// Total time budget for one `edit`/`write` gate call (split across files).
pub const GATE_BUDGET_MS: u64 = 2000;
/// Poll cadence while waiting for a refreshed publish.
const POLL_STEP_MS: u64 = 150;
/// Budget to start a server in [`prewarm`] (initialize handshake).
const PREWARM_INIT_TIMEOUT: Duration = Duration::from_secs(8);
/// Budget for a project-aware server (rust-analyzer) to finish loading.
const PREWARM_LOAD_TIMEOUT: Duration = Duration::from_secs(15);
/// Re-attempt prewarm at most this often per (cwd, server) — a healthy
/// client stays in the registry, so retries only matter after the idle
/// reaper killed it or the first start failed.
const PREWARM_RETRY_COOLDOWN: Duration = Duration::from_mins(5);
/// LSP `Severity` = Error|Warning; higher severities are informational/hint.
const MAX_GATED_SEVERITY: u32 = 2;

/// Outcome of a gate run: the text section to append and the detail rows.
pub struct GateResult {
    /// Lines to append to the tool's text output (empty when nothing is new).
    pub section: String,
    /// Full new-diagnostic list for `details["diagnostics"]`.
    pub diagnostics: Vec<Value>,
}

/// A normalized new diagnostic (subject-visible).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NewDiag {
    /// 0-based LSP line (display as `line + 1`).
    pub line: u32,
    /// 0-based LSP column (display as `col + 1`).
    pub col: u32,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct DiagKey {
    line: u32,
    col: u32,
    message: String,
}

/// Collapse any internal whitespace runs into single spaces so the same
/// diagnostic text published with different wrapping compares equal.
fn normalize_message(message: &str) -> String {
    message.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn keyed(diagnostics: &[Diagnostic]) -> HashSet<DiagKey> {
    diagnostics
        .iter()
        .filter(|d| d.severity.unwrap_or(1) <= MAX_GATED_SEVERITY)
        .map(|d| DiagKey {
            line: d.range.start.line,
            col: d.range.start.character,
            message: normalize_message(&d.message),
        })
        .collect()
}

/// Diagnostics present in `after` but not `before` (errors/warnings only),
/// in `after` order with duplicates collapsed.
pub fn compute_new_diagnostics(before: &[Diagnostic], after: &[Diagnostic]) -> Vec<NewDiag> {
    let before = keyed(before);
    let mut seen = HashSet::new();
    let mut news = Vec::new();
    for d in after {
        if d.severity.unwrap_or(1) > MAX_GATED_SEVERITY {
            continue;
        }
        let message = normalize_message(&d.message);
        let key = DiagKey {
            line: d.range.start.line,
            col: d.range.start.character,
            message: message.clone(),
        };
        if !before.contains(&key) && seen.insert(key) {
            news.push(NewDiag {
                line: d.range.start.line,
                col: d.range.start.character,
                message,
            });
        }
    }
    news
}

fn hash_delta(delta: &[NewDiag]) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    delta.hash(&mut hasher);
    hasher.finish()
}

/// Record one occurrence of `delta` for `path` in the recurrence ledger.
/// Returns `true` when this exact delta was already seen for the same file.
fn record_recurrence(
    ledger: &mut HashMap<(PathBuf, u64), usize>,
    path: &Path,
    delta: &[NewDiag],
) -> bool {
    let key = (path.to_path_buf(), hash_delta(delta));
    let entry = ledger.entry(key).or_insert(0);
    let escalated = *entry > 0;
    *entry += 1;
    escalated
}

/// Whether an active client's configured file types cover `path`.
fn client_serves(client: &LspClient, path: &Path) -> bool {
    let ext = path
        .extension()
        .map(|e| format!(".{}", e.to_string_lossy().to_lowercase()))
        .unwrap_or_default();
    let ext_no_dot = ext.trim_start_matches('.').to_string();
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().to_lowercase())
        .unwrap_or_default();
    client.config().file_types.iter().any(|ft| {
        let normalized = ft.to_lowercase();
        let no_dot = normalized.trim_start_matches('.').to_string();
        normalized == ext || normalized == name || no_dot == ext_no_dot || no_dot == name
    })
}

/// Ask a running client to re-sync the on-disk (post-write) content.
///
/// Open files get a full-text `didChange` + `didSave`. Files that are not
/// currently open are only re-opened when the server has analyzed them before
/// (evidence: a previous publish); completely unknown files are left alone so
/// the gate never cold-opens arbitrary paths.
async fn notify_refresh(client: &LspClient, path: &Path, uri: &str) {
    if client.is_open(uri) {
        let Ok(content) = std::fs::read_to_string(path) else {
            return;
        };
        let version = client.open_file_version(uri).unwrap_or(0) + 1;
        let _ = client
            .send_notification(
                "textDocument/didChange",
                json!({
                    "textDocument": { "uri": uri, "version": version },
                    "contentChanges": [{ "text": content }]
                }),
            )
            .await;
        let _ = client
            .send_notification(
                "textDocument/didSave",
                json!({ "textDocument": { "uri": uri }, "text": content }),
            )
            .await;
    } else if client.published_diagnostics(uri).is_some() {
        let Ok(content) = std::fs::read_to_string(path) else {
            return;
        };
        let language_id = client
            .config()
            .language_id
            .clone()
            .unwrap_or_else(|| omp_lsp::utils::detect_language_id(path));
        let _ = client
            .send_notification(
                "textDocument/didOpen",
                json!({
                    "textDocument": { "uri": uri, "languageId": language_id, "version": 1, "text": content }
                }),
            )
            .await;
    }
}

/// Run the advisory gate over `files`, returning `None` when there is nothing
/// to report (including when no LSP server is running at all).
#[allow(clippy::too_many_lines)] // gate pipeline: prewarm, cold budget, collect, filter, render
pub async fn run(env: &Env, files: &[PathBuf]) -> Option<GateResult> {
    /// Budget while a cold-started server (file opened by `prewarm` but never
    /// published yet) produces its first diagnostics. rust-analyzer needs
    /// 15-30s on a cold project; without this the gate would always lose the
    /// race against the model's first edit.
    const COLD_GATE_BUDGET_MS: u64 = 25_000;
    let clients = omp_lsp::client::active_clients();
    if files.is_empty() || clients.is_empty() {
        return None;
    }
    // Per-file budget: the normal fast path, or the cold path below.

    let mut sections = Vec::new();
    let mut all_diagnostics = Vec::new();

    for path in files {
        let relevant: Vec<&LspClient> = clients.iter().filter(|c| client_serves(c, path)).collect();
        if relevant.is_empty() {
            continue;
        }
        let uri = file_to_uri(path);

        // Pre-write snapshot. Nothing above has touched the client, so this is
        // still the pre-edit publish for the changed file.
        let before_snapshot: Vec<Option<PublishedDiagnostics>> = relevant
            .iter()
            .map(|c| c.published_diagnostics(&uri))
            .collect();
        let before: Vec<Diagnostic> = before_snapshot
            .iter()
            .filter_map(|p| p.as_ref())
            .flat_map(|p| p.diagnostics.iter().cloned())
            .collect();
        let before_versions: Vec<Option<i64>> = before_snapshot
            .iter()
            .map(|p| p.as_ref().and_then(|p| p.version))
            .collect();
        // Refresh feasibility: at least one client must know the file (open
        // or previously published), otherwise `notify_refresh` sent nothing
        // and no publish can ever arrive — skip instead of burning budget.
        if !relevant
            .iter()
            .any(|c| c.is_open(&uri) || c.published_diagnostics(&uri).is_some())
        {
            continue;
        }
        // Cold path: the file is open (via `prewarm`) but this client never
        // published for it — the first publish is still on its way. Warm
        // project-aware servers (rust-analyzer) can also lag a didChange
        // re-publish while still loading a cold project, so they get the
        // same extended budget; healthy warm servers answer in well under
        // a second and the poll loop exits early.
        let cold = relevant
            .iter()
            .zip(&before_versions)
            .any(|(c, v)| v.is_none() && c.is_open(&uri));
        let slow = relevant
            .iter()
            .any(|c| omp_lsp::servers::is_project_aware_lsp_server(c.config()));
        let budget = if cold || slow {
            Duration::from_millis(COLD_GATE_BUDGET_MS)
        } else {
            Duration::from_millis((GATE_BUDGET_MS / files.len() as u64).max(1))
        };

        // Ask each client to refresh with the new content, then poll for a
        // newer publish until the per-file budget runs out.
        for c in &relevant {
            notify_refresh(c, path, &uri).await;
        }
        let deadline = Instant::now() + budget;
        let mut updated = vec![false; relevant.len()];
        loop {
            for (i, c) in relevant.iter().enumerate() {
                if updated[i] {
                    continue;
                }
                let now = c.published_diagnostics(&uri);
                let now_v = now.as_ref().and_then(|p| p.version);
                let changed = match (before_versions[i], now_v) {
                    (Some(b), Some(n)) => n != b,
                    (None, Some(_)) => true,
                    _ => false,
                };
                // Cold path: a project-aware server's first publish may be
                // empty because analysis is still running. Treat an empty
                // publish as final only once the server is quiescent (or a
                // non-empty publish arrived, or the budget runs out).
                let strict = cold && omp_lsp::servers::is_project_aware_lsp_server(c.config());
                let settled = !strict
                    || now.as_ref().is_some_and(|p| !p.diagnostics.is_empty())
                    || c.is_quiescent();
                if changed && settled {
                    updated[i] = true;
                }
            }
            if updated.iter().all(|&u| u) || Instant::now() >= deadline {
                break;
            }
            tokio::time::sleep(Duration::from_millis(POLL_STEP_MS)).await;
        }

        let after: Vec<Diagnostic> = relevant
            .iter()
            .filter_map(|c| c.published_diagnostics(&uri))
            .flat_map(|p| p.diagnostics.into_iter())
            .collect();
        let delta: Vec<NewDiag> = compute_new_diagnostics(&before, &after);
        if delta.is_empty() {
            continue;
        }

        let escalated = {
            let mut ledger = env.diagnostics_ledger.lock();
            record_recurrence(&mut ledger, path, &delta)
        };
        let rel = format_path_relative_to_cwd(path, &env.cwd, false);
        if escalated {
            sections.push(format!(
                "[lsp BLOCKED-RETRY] this exact edit reproduces the same {} new diagnostics — change approach, do not repeat the same edit",
                delta.len()
            ));
        } else {
            let items = delta
                .iter()
                .map(|d| format!("{}:{} {}", d.line + 1, d.col + 1, d.message))
                .collect::<Vec<_>>()
                .join("; ");
            sections.push(format!(
                "[lsp] +{} new diagnostics in {}: {}",
                delta.len(),
                rel,
                items
            ));
        }
        all_diagnostics.extend(delta.iter().map(|d| {
            json!({
                "file": rel,
                "line": d.line + 1,
                "column": d.col + 1,
                "message": d.message,
                "severity": "warning",
            })
        }));
    }

    if sections.is_empty() {
        return None;
    }
    Some(GateResult {
        section: sections.join("\n"),
        diagnostics: all_diagnostics,
    })
}

// ============================================================================
// Autostart (prewarm)
// ============================================================================

/// One prewarm attempt per (cwd, server name) per process, with the last
/// attempt time for the [`PREWARM_RETRY_COOLDOWN`] backoff.
static PREWARM_ATTEMPTS: std::sync::OnceLock<
    parking_lot::Mutex<HashMap<(PathBuf, String), Instant>>,
> = std::sync::OnceLock::new();

/// Wall-clock cancel for the bounded project-load wait.
struct DeadlineCancel(Instant);

impl omp_lsp::Cancel for DeadlineCancel {
    fn is_cancelled(&self) -> bool {
        Instant::now() >= self.0
    }
    fn cancelled(&self) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send + '_>> {
        Box::pin(tokio::time::sleep_until(self.0.into()))
    }
}

/// Fire-and-forget [`prewarm_file`]: start the file's language server in the
/// background so it is warm (with pre-edit content) by the time the model
/// edits. Called by the `read` tool; must never block or fail the read —
/// without a runtime (unit tests, sync callers) it is a no-op.
pub fn prewarm_spawn(cwd: &Path, file: &Path) {
    let Ok(handle) = tokio::runtime::Handle::try_current() else {
        return;
    };
    let (cwd, file) = (cwd.to_path_buf(), file.to_path_buf());
    handle.spawn(async move {
        prewarm_file(&cwd, &file).await;
    });
}

/// Fire-and-forget [`prewarm_project`]: start every project server for `cwd`
/// at session start (wired into `session_builtin_tools`), so the first edit
/// already has a pre-edit diagnostics baseline without waiting for the model
/// to `read` anything. Same no-op guards as [`prewarm_spawn`].
pub fn prewarm_project_spawn(cwd: &Path) {
    let Ok(handle) = tokio::runtime::Handle::try_current() else {
        return;
    };
    let cwd = cwd.to_path_buf();
    handle.spawn(async move {
        prewarm_project(&cwd).await;
    });
}
/// Resolve the already-running client for `name`, if any.
fn active_client_named(name: &str) -> Option<LspClient> {
    omp_lsp::client::active_clients()
        .into_iter()
        .find(|c| c.name() == name)
}

/// Start the primary server for `file` (see [`prewarm_server`]) and `didOpen`
/// the file with its on-disk (pre-edit) content, so the server publishes a
/// baseline the post-write gate can diff against.
pub async fn prewarm_file(cwd: &Path, file: &Path) {
    let config = omp_lsp::config::get_config(cwd);
    let Some((name, server)) = omp_lsp::servers::get_lsp_server_for_file(&config, file) else {
        return;
    };
    let client = if let Some(c) = active_client_named(&name) {
        Some(c)
    } else if let Some(c) = omp_lsp::client::get_active_or_pending_client(&server, cwd, None).await
    {
        // The session-start prewarm's client, as soon as its init lands —
        // the cooldown in prewarm_server must not skip the didOpen.
        Some(c)
    } else {
        prewarm_server(cwd, &name, &server).await
    };
    if let Some(client) = client {
        client.refresh_file(file).await;
    }
}

/// Start every (primary, non-linter) server configured for `cwd`.
pub async fn prewarm_project(cwd: &Path) {
    let config = omp_lsp::config::get_config(cwd);
    for (name, server) in omp_lsp::servers::get_lsp_servers(&config) {
        prewarm_server(cwd, &name, &server).await;
    }
}

/// Start one server (once per session per server, see
/// [`PREWARM_RETRY_COOLDOWN`]) and wait for project-aware servers to load,
/// bounded by [`PREWARM_INIT_TIMEOUT`] and [`PREWARM_LOAD_TIMEOUT`]. Best
/// effort: any failure is cached by the cooldown and simply leaves the gate
/// without a baseline. Returns the client when one is (or became) available.
pub async fn prewarm_server(
    cwd: &Path,
    name: &str,
    server: &omp_lsp::types::ServerConfig,
) -> Option<LspClient> {
    if let Some(existing) = active_client_named(name) {
        return Some(existing); // already warm
    }
    let key = (cwd.to_path_buf(), name.to_string());
    let attempts = PREWARM_ATTEMPTS.get_or_init(|| parking_lot::Mutex::new(HashMap::new()));
    {
        let mut guard = attempts.lock();
        if let Some(last) = guard.get(&key) {
            if last.elapsed() < PREWARM_RETRY_COOLDOWN {
                return None;
            }
        }
        guard.insert(key, Instant::now());
    }
    let client =
        omp_lsp::client::get_or_create_client(server, cwd, Some(PREWARM_INIT_TIMEOUT), None)
            .await
            .ok()?;
    if omp_lsp::servers::is_project_aware_lsp_server(server) {
        let cancel = DeadlineCancel(Instant::now() + PREWARM_LOAD_TIMEOUT);
        client.wait_for_project_loaded(Some(&cancel)).await;
    }
    Some(client)
}

#[cfg(test)]
mod tests {
    use super::*;
    use omp_lsp::types::{Position, Range};

    fn diag(line: u32, col: u32, severity: u32, message: &str) -> Diagnostic {
        Diagnostic {
            range: Range {
                start: Position {
                    line,
                    character: col,
                },
                end: Position {
                    line,
                    character: col + 1,
                },
            },
            severity: Some(severity),
            code: None,
            code_description: None,
            source: None,
            message: message.to_string(),
            tags: None,
            related_information: None,
            data: None,
        }
    }

    #[test]
    fn delta_counts_only_new_diagnostics() {
        let before = vec![
            diag(12, 4, 1, "unresolved import Foo"),
            diag(30, 0, 2, "mismatched types"),
        ];
        let after = vec![
            diag(12, 4, 1, "unresolved import Foo"), // same -> excluded
            diag(30, 0, 2, "mismatched types"),      // gone -> excluded
            diag(40, 2, 1, "expected type `u8`"),    // new
            diag(55, 1, 3, "some info"),             // severity > warning -> excluded
        ];
        let delta = compute_new_diagnostics(&before, &after);
        assert_eq!(delta.len(), 1);
        assert_eq!(delta[0].line, 40);
        assert_eq!(delta[0].col, 2);
        assert_eq!(delta[0].message, "expected type `u8`");
    }

    #[test]
    fn message_normalization_collapses_whitespace() {
        let before = vec![diag(1, 0, 1, "a  b\nc")];
        let after = vec![diag(1, 0, 1, "a b c"), diag(2, 0, 1, "x")];
        let delta = compute_new_diagnostics(&before, &after);
        assert_eq!(delta.len(), 1);
        assert_eq!(delta[0].line, 2);
        assert_eq!(delta[0].col, 0);
    }

    #[test]
    fn recurrence_ledger_escalates_only_on_repeat() {
        let mut ledger: HashMap<(PathBuf, u64), usize> = HashMap::new();
        let path = PathBuf::from("/repo/src/x.rs");
        let delta = vec![NewDiag {
            line: 1,
            col: 0,
            message: "boom".into(),
        }];
        assert!(!record_recurrence(&mut ledger, &path, &delta));
        assert!(record_recurrence(&mut ledger, &path, &delta));
        assert!(!record_recurrence(
            &mut ledger,
            &path,
            &[NewDiag {
                line: 9,
                col: 0,
                message: "other".into(),
            }]
        ));
    }
}
