//! Blaze v2 process boundary and compact, model-facing diagnostics.
//!
//! No cargo/rustc fallback lives here: Blaze either answers or its failure is
//! returned verbatim as a bounded tool error. Full process output is archived;
//! only the structured diagnostic frontier enters the model's context.

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use pa_agent::abort::AbortSignal;
use pa_agent::types::AgentToolResult;
use serde_json::{json, Value};
use tokio::process::Command;

use super::Env;
use crate::approval::{deny_message, parse_mode, resolve_approval, tool_decision, Gate, Policy};
use omp_ext::lifecycle::ApprovalMode;

pub const DESCRIPTION: &str = "Run Blaze v2 directly in a Rust workspace: `check` emits compact structured diagnostics, `test` uses Blaze's reached-code/read selection (no manual --test heuristic), `verify` runs check then test, and `why` explains a stored file/item answer. No cargo/rustc fallback. `max_tokens` bounds model-facing diagnostics; full failure output is archived. Cargo/CI remains the final compatibility oracle while Blaze is young. Set OMT_BLAZE_BIN to the absolute Blaze binary/wrapper when it is not on PATH.";

#[must_use]
pub fn schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["check", "test", "verify", "why"]},
            "path": {"type": "string", "description": "Cargo workspace/package directory; defaults to the session cwd."},
            "item": {"type": "string", "description": "For `why`: path or path::item from a previous Blaze check."},
            "timeout": {"type": "number", "description": "Per-Blaze-command timeout in seconds (1-600, default 120)."},
            "max_tokens": {"type": "integer", "minimum": 128, "maximum": 16000, "default": 2000}
        },
        "required": ["action"],
        "additionalProperties": false
    })
}

/// Apply the same approval policy as the corresponding direct nested tool call.
/// Without an interactive gate, prompt/deny decisions fail closed.
pub(crate) async fn ensure_nested_allowed(
    tool: &str,
    tool_args: &Value,
    cwd: &Path,
    call_id: &str,
    signal: &AbortSignal,
) -> std::result::Result<(), String> {
    if let Some(gate) = Gate::get() {
        return gate.check(tool, call_id, tool_args, signal).await;
    }
    let settings = omp_settings::Settings::load(cwd);
    let policies = settings.get_record("tools.approval");
    let mode = settings
        .get_string("tools.approvalMode")
        .as_deref()
        .and_then(parse_mode)
        .unwrap_or(ApprovalMode::Yolo);
    let decision = tool_decision(tool, tool_args, &settings);
    let resolved = resolve_approval(tool, &decision, mode, &policies);
    if resolved.policy == Policy::Deny {
        return Err(deny_message(&resolved, tool));
    }
    if resolved.policy == Policy::Prompt {
        return Err(format!(
            "Tool \"{tool}\" requires approval but no interactive UI available.\nOptions:\n  1. Set tools.approvalMode: yolo in /settings\n  2. Add tools.approval.{tool}: allow to config\n  3. Use an interactive UI to approve the tool call"
        ));
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Finding {
    pub id: String,
    pub severity: String,
    pub code: String,
    pub path: String,
    pub line: Option<u64>,
    pub col: Option<u64>,
    pub message: String,
    pub fix: Option<Value>,
}

impl Finding {
    fn from_json(value: &Value) -> Option<Self> {
        Some(Self {
            id: value.get("id")?.as_str()?.to_owned(),
            severity: value
                .get("severity")
                .and_then(Value::as_str)
                .unwrap_or("error")
                .to_owned(),
            code: value
                .get("code")
                .and_then(Value::as_str)
                .unwrap_or("blaze")
                .to_owned(),
            path: value
                .get("path")
                .and_then(Value::as_str)
                .unwrap_or("<workspace>")
                .to_owned(),
            line: value.get("line").and_then(Value::as_u64),
            col: value.get("col").and_then(Value::as_u64),
            message: value
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_owned(),
            fix: value.get("fix").cloned().filter(|fix| !fix.is_null()),
        })
    }

    fn is_unused_import(&self) -> bool {
        matches!(self.code.as_str(), "unused-import" | "unused_imports")
    }

    fn compact(&self) -> String {
        let mut out = format!("{} {}[{}] {}", self.id, self.severity, self.code, self.path);
        if let (Some(line), Some(col)) = (self.line, self.col) {
            let _ = write!(out, ":{line}:{col}");
        }
        let _ = write!(out, "  {}", self.message);
        if let Some(reason) = self
            .fix
            .as_ref()
            .and_then(|fix| fix.get("reason"))
            .and_then(Value::as_str)
        {
            let _ = write!(out, "\n  fix: {reason}");
        }
        out
    }
}

#[derive(Debug, Default)]
pub(crate) struct Check {
    pub success: bool,
    pub status: Option<i32>,
    pub errors: usize,
    pub warnings: usize,
    pub findings: Vec<Finding>,
    pub process_error: Option<String>,
    pub log: String,
}

impl Check {
    pub fn unused_imports(&self) -> impl Iterator<Item = &Finding> {
        self.findings.iter().filter(|it| it.is_unused_import())
    }

    pub fn unused_import_count(&self) -> usize {
        self.unused_imports().count()
    }

    pub fn compact(&self, max_tokens: usize) -> String {
        let mut out = format!(
            "blaze check: {} (exit {:?}); {} error(s), {} warning(s), {} fix(es)\n",
            if self.success { "ok" } else { "failed" },
            self.status,
            self.errors,
            self.warnings,
            self.findings.iter().filter(|it| it.fix.is_some()).count()
        );
        if self.findings.is_empty() {
            if let Some(error) = &self.process_error {
                let remaining = max_tokens.saturating_mul(4).saturating_sub(out.len());
                if error.len() <= remaining {
                    out.push_str(error);
                    out.push('\n');
                } else if remaining > 80 {
                    let mut cut = remaining - 80;
                    while !error.is_char_boundary(cut) {
                        cut = cut.saturating_sub(1);
                    }
                    out.push_str(&error[..cut]);
                    out.push_str("\n… error truncated; see the archived report\n");
                }
            }
        }
        // Errors are the actionable frontier. Stable dedup removes repeated
        // configurations without pretending to have Blaze's internal DBSP graph.
        let mut seen = BTreeSet::new();
        let mut ordered: Vec<&Finding> = self.findings.iter().collect();
        ordered.sort_by_key(|it| (it.severity != "error", &it.path, it.line, it.col, &it.id));
        let mut omitted = 0;
        for finding in ordered {
            let key = (
                &finding.code,
                &finding.path,
                finding.line,
                finding.col,
                &finding.message,
            );
            if !seen.insert(key) {
                continue;
            }
            let line = finding.compact();
            if out.len() + line.len() + 1 > max_tokens.saturating_mul(4) {
                omitted += 1;
                continue;
            }
            out.push_str(&line);
            out.push('\n');
        }
        if omitted > 0 {
            let _ = writeln!(
                out,
                "… {omitted} diagnostic(s) omitted by max_tokens; see the archived report"
            );
        }
        out
    }
}

pub(crate) fn parse_check(output: &std::process::Output) -> Check {
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let mut check = Check {
        success: output.status.success(),
        status: output.status.code(),
        process_error: (!stderr.trim().is_empty()).then(|| stderr.trim().to_owned()),
        log: format!("$ blaze check --view json --each\n{stdout}{stderr}"),
        ..Check::default()
    };
    let mut closing = None;
    for line in stdout.lines().chain(stderr.lines()) {
        let Ok(value) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        if value.get("id").is_some() {
            if let Some(finding) = Finding::from_json(&value) {
                check.findings.push(finding);
            }
        } else if value.get("errors").and_then(Value::as_u64).is_some() {
            closing = Some(value);
        }
    }
    if let Some(summary) = closing {
        check.errors = usize::try_from(summary.get("errors").and_then(Value::as_u64).unwrap_or(0))
            .unwrap_or(usize::MAX);
        check.warnings = usize::try_from(summary.get("warnings").and_then(Value::as_u64).unwrap_or(0))
            .unwrap_or(usize::MAX);
    } else {
        check.errors = check
            .findings
            .iter()
            .filter(|it| it.severity == "error")
            .count();
        check.warnings = check
            .findings
            .iter()
            .filter(|it| it.severity == "warning")
            .count();
    }
    check
}

async fn run(project: &Path, args: &[String], timeout: Duration) -> Result<std::process::Output> {
    let shown = args.join(" ");
    let binary = std::env::var_os("OMT_BLAZE_BIN").unwrap_or_else(|| "blaze".into());
    tokio::time::timeout(
        timeout,
        Command::new(&binary)
            .args(args)
            .current_dir(project)
            .kill_on_drop(true)
            .output(),
    )
    .await
    .map_err(|_| anyhow!("blaze {shown} timed out after {}s", timeout.as_secs()))?
    .with_context(|| {
        format!("failed to spawn blaze {shown}; install Blaze v2 and its pinned toolchain")
    })
}

fn args_with_path(command: &str, _project: &Path, tail: &[&str]) -> Vec<String> {
    // Blaze resolves the checkout from the child cwd and rejects absolute roots.
    let mut args = vec![command.to_owned(), ".".to_owned()];
    args.extend(tail.iter().map(|it| (*it).to_owned()));
    args
}

pub(crate) async fn check(project: &Path, timeout: Duration) -> Result<Check> {
    let args = args_with_path("check", project, &["--view", "json", "--each"]);
    let output = run(project, &args, timeout).await?;
    Ok(parse_check(&output))
}

/// Apply only the selected finding IDs. Blaze performs its own second check
/// after one atomic fix transaction, so its output is the next `Check`.
pub(crate) async fn fix(project: &Path, timeout: Duration, ids: &[String]) -> Result<Check> {
    let mut args = args_with_path("check", project, &["--view", "json", "--each", "--fix"]);
    args.extend(ids.iter().cloned());
    let output = run(project, &args, timeout).await?;
    Ok(parse_check(&output))
}

fn compact_output(command: &str, output: &std::process::Output, max_tokens: usize) -> String {
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let mut text = format!("$ blaze {command}\n{stdout}{stderr}");
    let budget = max_tokens.saturating_mul(4);
    if text.len() > budget {
        let mut cut = budget.saturating_sub(80);
        while !text.is_char_boundary(cut) {
            cut = cut.saturating_sub(1);
        }
        text.truncate(cut);
        text.push_str("\n… output truncated by max_tokens; see the archived output\n");
    }
    text
}

fn result(
    action: &str,
    project: &Path,
    output: &std::process::Output,
    text: String,
    max_tokens: usize,
) -> AgentToolResult {
    let mut details = json!({
        "workflow": "blaze",
        "action": action,
        "path": project.display().to_string(),
        "status": output.status.code(),
        "success": output.status.success(),
        "maxTokens": max_tokens,
    });
    let failed = !output.status.success();
    if failed {
        let log = compact_output(action, output, usize::MAX / 4);
        if let Some(url) = super::util::save_artifact(&log) {
            details["log"] = json!(url);
        }
    }
    let mut result = AgentToolResult::text(text);
    result.details = details;
    result.is_error = failed.then_some(true);
    result
}

fn resolve_project(env: &Env, raw: &str) -> Result<PathBuf> {
    let requested = super::util::resolve_to_cwd(raw, &env.cwd);
    let project = requested
        .canonicalize()
        .with_context(|| format!("Blaze project does not exist: {}", requested.display()))?;
    let cwd = env
        .cwd
        .canonicalize()
        .context("cannot resolve session cwd")?;
    if !project.starts_with(&cwd) || !project.is_dir() {
        bail!("Blaze path must be a directory inside the session cwd");
    }
    Ok(project)
}

pub async fn blaze(env: Arc<Env>, args: Value, signal: AbortSignal) -> Result<AgentToolResult> {
    let action = args
        .get("action")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("blaze action is required"))?;
    if !matches!(action, "check" | "test" | "verify" | "why") {
        bail!("unknown blaze action; expected check, test, verify or why");
    }
    let project = resolve_project(
        &env,
        args.get("path").and_then(Value::as_str).unwrap_or("."),
    )?;
    let timeout = Duration::from_secs_f64(
        args.get("timeout")
            .and_then(Value::as_f64)
            .unwrap_or(120.0)
            .clamp(1.0, 600.0),
    );
    let max_tokens = args
        .get("max_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(2000)
        .clamp(128, 16000) as usize;
    let call_id = super::macro_step::next_call_id("blaze");
    let command_text = match action {
        "why" => format!(
            "blaze why {}",
            args.get("item").and_then(Value::as_str).unwrap_or("")
        ),
        _ => format!("blaze {action}"),
    };
    ensure_nested_allowed(
        "bash",
        &json!({"command": command_text, "cwd": project}),
        &project,
        &call_id,
        &signal,
    )
    .await
    .map_err(|msg| anyhow!("blaze {action} refused: {msg}"))?;

    match action {
        "check" => {
            let checked = check(&project, timeout).await?;
            let mut result = AgentToolResult::text(checked.compact(max_tokens));
            result.details = json!({
                "workflow": "blaze", "action": "check", "status": checked.status,
                "success": checked.success, "errors": checked.errors,
                "warnings": checked.warnings, "findings": checked.findings.len(),
                "maxTokens": max_tokens,
            });
            result.is_error = (!checked.success).then_some(true);
            if !checked.success {
                if let Some(url) = super::util::save_artifact(&checked.log) {
                    result.details["log"] = json!(url);
                }
            }
            Ok(result)
        }
        "test" | "verify" => {
            if action == "verify" {
                let checked = check(&project, timeout).await?;
                if !checked.success {
                    let mut result = AgentToolResult::text(checked.compact(max_tokens));
                    result.details = json!({"workflow":"blaze","action":"verify","stage":"check","success":false,"status":checked.status});
                    result.is_error = Some(true);
                    if let Some(url) = super::util::save_artifact(&checked.log) {
                        result.details["log"] = json!(url);
                    }
                    return Ok(result);
                }
            }
            let command = args_with_path("test", &project, &[]);
            let output = run(&project, &command, timeout).await?;
            let text = compact_output(action, &output, max_tokens);
            Ok(result(action, &project, &output, text, max_tokens))
        }
        "why" => {
            let item = args
                .get("item")
                .and_then(Value::as_str)
                .filter(|it| !it.is_empty())
                .ok_or_else(|| anyhow!("blaze why requires item"))?;
            let output = run(&project, &["why".to_owned(), item.to_owned()], timeout).await?;
            let text = compact_output("why", &output, max_tokens);
            Ok(result("why", &project, &output, text, max_tokens))
        }
        _ => unreachable!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::ExitStatus;

    #[cfg(unix)]
    fn status(code: i32) -> ExitStatus {
        use std::os::unix::process::ExitStatusExt;
        ExitStatus::from_raw(code << 8)
    }

    #[test]
    #[cfg(unix)]
    fn parses_structured_frontier_and_unused_import_fixes() {
        let output = std::process::Output {
            status: status(1),
            stdout: concat!(
                "{\"id\":\"a2345\",\"severity\":\"warning\",\"code\":\"unused-import\",\"path\":\"src/lib.rs\",\"line\":1,\"col\":1,\"message\":\"unused import\",\"fix\":{\"reason\":\"remove it\"}}\n",
                "{\"id\":\"b2345\",\"severity\":\"error\",\"code\":\"E0308\",\"path\":\"src/lib.rs\",\"line\":3,\"col\":5,\"message\":\"mismatched types\",\"fix\":null}\n",
                "{\"errors\":1,\"warnings\":1,\"fixes\":1,\"command\":\"blaze check --fix a2345\"}\n"
            ).as_bytes().to_vec(),
            stderr: vec![],
        };
        let parsed = parse_check(&output);
        assert_eq!(parsed.errors, 1);
        assert_eq!(parsed.warnings, 1);
        assert_eq!(parsed.unused_import_count(), 1);
        assert_eq!(parsed.unused_imports().next().unwrap().id, "a2345");
        let compact = parsed.compact(2000);
        assert!(compact.find("E0308").unwrap() < compact.find("unused-import").unwrap());
    }

    /// Opt-in host smoke: drives the actual tool boundary and installed Blaze.
    /// Supports either the stock release's bounded toolchain failure or a
    /// configured working binary's successful check, never a cargo fallback.
    #[tokio::test]
    #[allow(clippy::disallowed_methods)] // tokio::test drives the real async tool boundary
    async fn real_blaze_tool_result_is_bounded() {
        let expect_success = std::env::var("OMT_BLAZE_LIVE_SUCCESS").ok().as_deref() == Some("1");
        if !expect_success && std::env::var("OMT_BLAZE_LIVE_TEST").ok().as_deref() != Some("1") {
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("Cargo.toml"),
            "[package]\nname = \"blaze_tool_smoke\"\nversion = \"0.1.0\"\n",
        )
        .unwrap();
        std::fs::create_dir(dir.path().join("src")).unwrap();
        std::fs::write(dir.path().join("src/lib.rs"), "pub fn ok() {}\n").unwrap();
        let env = Arc::new(Env::with_mode(dir.path(), pi_edit::EditMode::Replace));
        let result = blaze(
            env,
            json!({"action":"check","max_tokens":128,"timeout":30}),
            AbortSignal::never(),
        )
        .await
        .unwrap();
        assert_eq!(result.is_error, (!expect_success).then_some(true), "details={} content={:?}", result.details, result.content);
        assert_eq!(result.details["workflow"], "blaze");
        assert_eq!(result.details["success"], expect_success);
        let pa_agent::types::ToolResultContent::Text(text) = &result.content[0] else {
            panic!("Blaze returned non-text content");
        };
        assert!(text.text.len() < 700, "unbounded process output: {}", text.text);
    }

    #[test]
    #[cfg(unix)]
    fn token_budget_keeps_summary_and_reports_omissions() {
        let stdout = (1..=30).fold(String::new(), |mut out, line| {
            let _ = writeln!(out, "{{\"id\":\"a{line}\",\"severity\":\"error\",\"code\":\"E0308\",\"path\":\"src/lib.rs\",\"line\":{line},\"col\":5,\"message\":\"a long diagnostic\",\"fix\":null}}");
            out
        });
        let output = std::process::Output {
            status: status(1),
            stdout: stdout.into_bytes(),
            stderr: vec![],
        };
        let compact = parse_check(&output).compact(128);
        assert!(compact.starts_with("blaze check: failed"));
        assert!(compact.contains("omitted by max_tokens"));
        assert!(compact.len() < 800);
    }
}
