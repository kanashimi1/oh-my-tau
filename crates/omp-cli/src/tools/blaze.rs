//! Blaze v2 process boundary and compact, model-facing diagnostics.
//!
//! No cargo/rustc fallback lives here: Blaze either answers or its failure is
//! returned verbatim as a bounded tool error. Full process output is archived;
//! only the structured diagnostic frontier enters the model's context.

use std::collections::{BTreeMap, BTreeSet};
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

    /// The blocking root behind this finding, when its documented message
    /// carries one. `macro-call-failed` quotes the proc-macro failure
    /// (`… the proc macro cannot run: `name`: <tool> …; build it with
    /// scripts/src/bin/proc_macro_modules.rs`); toolchain/source refusals
    /// name the toolchain, shelf or pinned source explicitly. Anything else
    /// returns None and stays an unclassified finding.
    fn cause(&self) -> Option<Cause> {
        if let Some(rest) = self.message.split("the proc macro cannot run:").nth(1) {
            let tool = rest
                .split(": ")
                .nth(1)
                .and_then(|tail| tail.split_whitespace().next())
                .map(|word| word.trim_matches('`').to_owned())
                .unwrap_or_else(|| String::from("proc-macro"));
            return Some(Cause::ProcMacroModule { tool });
        }
        if self.message.contains("cannot run") && self.message.contains("build script") {
            return Some(Cause::ProcMacroModule {
                tool: String::from("build-script"),
            });
        }
        let lower = self.message.to_ascii_lowercase();
        if toolchain_refusal_wording(&lower) {
            return Some(Cause::ToolchainOrSourceRefused);
        }
        None
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

/// A `not checked` note: a dependent item Blaze could not verify because a
/// root failure (missing crate scope, failed CrateScope, …) blocks it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Note {
    pub text: String,
}

impl Note {
    fn from_json(value: &Value) -> Option<Self> {
        Some(Self {
            text: value.get("note")?.as_str()?.to_owned(),
        })
    }

    /// The blocking root this note depends on, strictly from the documented
    /// note wording (`… not checked because its crate has no names record …`
    /// / `… its member's check scope holds `crate`, which has no names
    /// record …`). Everything else stays residual.
    fn cause(&self) -> Option<Cause> {
        let rest = self.text.split("not checked because ").nth(1)?;
        if let Some(inner) = rest.split("holds `").nth(1) {
            if let Some(crate_name) = inner.split("`, which has no names record").next() {
                return Some(Cause::CrateScope {
                    crate_name: crate_name.to_owned(),
                });
            }
        }
        rest.contains("its crate has no names record")
            .then(|| Cause::CrateScope {
                crate_name: String::from("<self>"),
            })
    }
}

/// A root/blocking cause. Strictly derived from documented finding/note
/// wording; anything else stays residual instead of being mapped onto a
/// hypothetical Blaze-internal graph (there is no public DBSP graph API).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Cause {
    /// A proc-macro or build-script module is absent from the store/cache
    /// (`… no compiled module in the store or the cache …; build it with
    /// scripts/src/bin/proc_macro_modules.rs`).
    ProcMacroModule { tool: String },
    /// A crate's check scope failed, so every item depending on it is
    /// `not checked` (e.g. `icu_properties_data`, `rustversion`).
    CrateScope { crate_name: String },
    /// The toolchain or pinned source refused the command (toolchain/shelf
    /// absence, source pin mismatch, explicit refusal wording such as
    /// `the toolchain manifest ...: not there; this blaze needs its own
    /// toolchain installed`).
    ToolchainOrSourceRefused,
}

/// Documented refusal wording: a toolchain/shelf/pinned-source word plus an
/// explicit absence/refusal word. Used for finding messages and raw stderr
/// evidence alike; anything short of both stays unclassified.
fn toolchain_refusal_wording(lower: &str) -> bool {
    let names_toolchain = ["toolchain", "shelf", "pinned source", "source pin"]
        .iter()
        .any(|word| lower.contains(word));
    let reads_as_refusal = [
        "refus",
        "missing",
        "not available",
        "no such",
        "absent",
        "not there",
    ]
    .iter()
    .any(|word| lower.contains(word));
    names_toolchain && reads_as_refusal
}

impl Cause {
    fn rank(&self) -> u8 {
        match self {
            // A missing proc-macro/build-script module blocks whole crates and
            // therefore every downstream check: most blocking first.
            Cause::ProcMacroModule { .. } => 0,
            Cause::CrateScope { .. } => 1,
            Cause::ToolchainOrSourceRefused => 2,
        }
    }

    fn label(&self) -> String {
        match self {
            Cause::ProcMacroModule { tool } => {
                format!("proc-macro/build-script module missing ({tool})")
            }
            Cause::CrateScope { crate_name } => format!("crate scope failed ({crate_name})"),
            Cause::ToolchainOrSourceRefused => {
                String::from("toolchain/source refused the command")
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RootCause {
    pub cause: Cause,
    pub findings: usize,
    pub notes: usize,
}

impl RootCause {
    fn headline(&self) -> String {
        let counted = match (self.findings, self.notes) {
            (f, 0) => format!("{f} finding(s)"),
            (0, n) => format!("{n} `not checked` item(s)"),
            (f, n) => format!("{f} finding(s), {n} `not checked` item(s)"),
        };
        let mut out = format!("root: {} — {counted}", self.cause.label());
        if matches!(self.cause, Cause::ProcMacroModule { .. }) {
            let _ = write!(
                out,
                "
  fix: build the missing modules with scripts/src/bin/proc_macro_modules.rs (D52)"
            );
        }
        out
    }
}

/// Root causes of one check, ranked most-blocking first and carrying their
/// dependent `not checked` note counts.
#[derive(Debug, Default)]
pub(crate) struct RootCauses {
    roots: BTreeMap<Cause, RootCause>,
    pub residual_notes: usize,
}

impl RootCauses {
    fn entry(&mut self, cause: &Cause) -> &mut RootCause {
        self.roots.entry(cause.clone()).or_insert_with(|| RootCause {
            cause: cause.clone(),
            findings: 0,
            notes: 0,
        })
    }

    fn record_finding(&mut self, cause: &Cause) {
        self.entry(cause).findings += 1;
    }

    fn record_note(&mut self, cause: &Cause) {
        self.entry(cause).notes += 1;
    }

    fn record_residual_note(&mut self) {
        self.residual_notes += 1;
    }

    fn any(&self) -> bool {
        !self.roots.is_empty()
    }

    fn ordered(&self) -> Vec<&RootCause> {
        let mut v: Vec<&RootCause> = self.roots.values().collect();
        v.sort_by_key(|r| (r.cause.rank(), std::cmp::Reverse(r.findings + r.notes)));
        v
    }
}

#[derive(Debug, Default)]
pub(crate) struct Check {
    pub success: bool,
    pub status: Option<i32>,
    pub errors: usize,
    pub warnings: usize,
    pub findings: Vec<Finding>,
    /// `not checked` notes: dependent items blocked by a root failure.
    pub notes: Vec<Note>,
    /// Non-JSON process output lines (toolchain wrappers, `time -v`, …).
    pub evidence: Vec<String>,
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

    /// Causal compaction: the root/blocking failures survive any budget and
    /// the dependent `not checked` noise collapses beneath them. Parsed
    /// strictly from the documented JSON finding/note fields; residual or
    /// unclassified diagnostics are reported as such — never mapped onto a
    /// pretend Blaze-internal graph.
    pub fn compact(&self, max_tokens: usize) -> String {
        let budget = max_tokens.saturating_mul(4);
        let mut out = format!(
            "blaze check: {} (exit {:?}); {} error(s), {} warning(s), {} fix(es), {} not checked\n",
            if self.success { "ok" } else { "failed" },
            self.status,
            self.errors,
            self.warnings,
            self.findings.iter().filter(|it| it.fix.is_some()).count(),
            self.notes.len()
        );
        if self.findings.is_empty() && self.notes.is_empty() {
            // No structured findings at all: raw stderr evidence can still
            // carry the blocking root (a toolchain/shelf refusal is the
            // common real case). Surface it above the verbatim error.
            let mut evidence_roots = RootCauses::default();
            for line in &self.evidence {
                if toolchain_refusal_wording(&line.to_ascii_lowercase()) {
                    evidence_roots.record_finding(&Cause::ToolchainOrSourceRefused);
                }
            }
            for root in evidence_roots.ordered() {
                let line = root.headline();
                if out.len() + line.len() + 1 > budget {
                    break;
                }
                let _ = writeln!(out, "{line}");
            }
            if let Some(error) = &self.process_error {
                let remaining = budget.saturating_sub(out.len());
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
            return out;
        }

        let mut roots = RootCauses::default();
        let mut seen = BTreeSet::new();
        let mut ordered: Vec<&Finding> = self.findings.iter().collect();
        ordered.sort_by_key(|it| (it.severity != "error", &it.path, it.line, it.col, &it.id));
        let mut unclassified: Vec<&Finding> = Vec::new();
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
            match finding.cause() {
                Some(cause) => roots.record_finding(&cause),
                None => unclassified.push(finding),
            }
        }
        for note in &self.notes {
            match note.cause() {
                Some(cause) => roots.record_note(&cause),
                None => roots.record_residual_note(),
            }
        }
        for line in &self.evidence {
            if toolchain_refusal_wording(&line.to_ascii_lowercase()) {
                roots.record_finding(&Cause::ToolchainOrSourceRefused);
            }
        }

        let mut omitted = 0usize;
        let mut rendered = 0usize;
        // 1. Root causes, most blocking first.
        for root in roots.ordered() {
            let line = root.headline();
            if out.len() + line.len() + 1 > budget {
                omitted += 1;
                continue;
            }
            let _ = writeln!(out, "{line}");
            rendered += 1;
            if root.notes > 0 {
                let dep = format!(
                    "  {} dependent `not checked` item(s) collapse under this root\n",
                    root.notes
                );
                if out.len() + dep.len() <= budget {
                    out.push_str(&dep);
                } else {
                    omitted += 1;
                }
            }
        }
        // 2. Unclassified findings stay visible and honest. When roots exist
        //    they cap at four exemplars; without roots every finding is the
        //    frontier and is listed until the budget runs out.
        let cap = if roots.any() { 4 } else { usize::MAX };
        let mut unlisted = 0usize;
        for (index, finding) in unclassified.iter().enumerate() {
            if index >= cap {
                unlisted += 1;
                continue;
            }
            let line = format!("  {}", finding.compact());
            if out.len() + line.len() + 1 > budget {
                omitted += 1;
                continue;
            }
            let _ = writeln!(out, "{line}");
            rendered += 1;
        }
        let _ = rendered;
        if roots.any() && unlisted > 0 {
            let _ = writeln!(
                out,
                "… {unlisted} unclassified finding(s) omitted by max_tokens; see the archived report"
            );
        }
        if roots.residual_notes > 0 {
            let _ = writeln!(
                out,
                "… {} `not checked` note(s) did not match a known root; see the archived report",
                roots.residual_notes
            );
        }
        if omitted > 0 {
            let _ = writeln!(
                out,
                "… {omitted} line(s) omitted by max_tokens; see the archived report"
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
            if !line.trim().is_empty() {
                // Non-JSON process output (toolchain wrappers, `time -v`, …):
                // kept as raw evidence and classified conservatively.
                check.evidence.push(line.trim().to_owned());
            }
            continue;
        };
        if value.get("id").is_some() {
            if let Some(finding) = Finding::from_json(&value) {
                check.findings.push(finding);
            }
        } else if let Some(note) = Note::from_json(&value) {
            check.notes.push(note);
        } else if value.get("errors").and_then(Value::as_u64).is_some() {
            closing = Some(value);
        } else if !line.trim().is_empty() {
            check.evidence.push(line.trim().to_owned());
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
    meter: &super::blaze_tokens::TokenMeter,
) -> Result<AgentToolResult> {
    let mut details = json!({
        "workflow": "blaze",
        "action": action,
        "path": project.display().to_string(),
        "status": output.status.code(),
        "success": output.status.success(),
        "maxTokens": max_tokens,
    });
    let raw = compact_output(action, output, usize::MAX / 4);
    meter.add_counts(&mut details, &raw, &text)?;
    let failed = !output.status.success();
    if failed {
        if let Some(url) = super::util::save_artifact(&raw) {
            details["log"] = json!(url);
        }
    }
    let mut result = AgentToolResult::text(text);
    result.details = details;
    result.is_error = failed.then_some(true);
    Ok(result)
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
    let meter = super::blaze_tokens::TokenMeter::from_env()?;
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
            let text = checked.compact(max_tokens);
            let mut details = json!({
                "workflow": "blaze", "action": "check", "status": checked.status,
                "success": checked.success, "errors": checked.errors,
                "warnings": checked.warnings, "findings": checked.findings.len(),
                "maxTokens": max_tokens,
            });
            meter.add_counts(&mut details, &checked.log, &text)?;
            let mut result = AgentToolResult::text(text);
            result.details = details;
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
                    let text = checked.compact(max_tokens);
                    let mut details = json!({"workflow":"blaze","action":"verify","stage":"check","success":false,"status":checked.status});
                    meter.add_counts(&mut details, &checked.log, &text)?;
                    let mut result = AgentToolResult::text(text);
                    result.details = details;
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
            Ok(result(action, &project, &output, text, max_tokens, &meter)?)
        }
        "why" => {
            let item = args
                .get("item")
                .and_then(Value::as_str)
                .filter(|it| !it.is_empty())
                .ok_or_else(|| anyhow!("blaze why requires item"))?;
            let output = run(&project, &["why".to_owned(), item.to_owned()], timeout).await?;
            let text = compact_output("why", &output, max_tokens);
            Ok(result("why", &project, &output, text, max_tokens, &meter)?)
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


    /// Cold-log regression (runs/omt-blaze-v2/omt-blaze-cold.log): the real
    /// omt check produced 1 root-blocking proc-macro failure family, missing
    /// crate scopes and a mountain of dependent `not checked` notes. Under a
    /// small budget the roots must survive verbatim and the noise must
    /// collapse under them.
    #[test]
    #[cfg(unix)]
    fn causal_compaction_roots_survive_small_budget_and_noise_collapses() {
        // Mirrors real lines from runs/omt-blaze-v2/omt-blaze-cold.log (and
        // runs/omt-blaze-v2/root-cause/cold-slice.jsonl).
        let tokio_macro = r#"{"id":"r3fhe","severity":"error","code":"macro-call-failed","path":"crates/omp-auth-broker/src/main.rs","line":618,"col":1,"message":"the call of `crate :: select_priv_clean_pattern` failed: the proc macro cannot run: `select_priv_clean_pattern`: tokio-macros 2.7.2: no compiled module in the store or the cache (module key 41c9de28dd66e5d18772b43c8a182702544acde9157d4ac87da4eb7122783813); build it with scripts/src/bin/proc_macro_modules.rs (D52)"}"#;
        let unresolved = r#"{"id":"g7hmx","severity":"error","code":"unresolved","path":"crates/omp-cli/src/compact.rs","line":354,"col":7,"message":"no `serde` in this scope"}"#;
        let note_icu = r#"{"note":"`crates/omp-cli/src/approval.rs`: body `mode_max_tier`: not checked because its member's check scope holds `icu_properties_data`, which has no names record this command (its `CrateScope` task failed: that failure is reported at it)"}"#;
        let note_rv = r#"{"note":"`crates/omp-tui/src/tools/task.rs`: body `AgentTreeRow`: not checked because its member's check scope holds `rustversion`, which has no names record this command (its `CrateScope` task failed: that failure is reported at it)"}"#;
        let note_self = r#"{"note":"`crates/omp-providers/src/command.rs`: body `FAILURE_RETRY`: not checked because its crate has no names record this command: its `CrateScope` task failed, and that failure is reported at it"}"#;
        let summary = r#"{"errors":1257,"warnings":0,"fixes":0,"command":"blaze check","not_checked":["types","borrows"]}"#;

        let mut stdout = String::new();
        // 1 tokio-macros root finding + 1 unresolved (unclassified), then the
        // dependent noise: 40 icu, 12 rustversion, 3 self-crate notes.
        stdout.push_str(tokio_macro);
        stdout.push('\n');
        stdout.push_str(unresolved);
        stdout.push('\n');
        for _ in 0..40 {
            stdout.push_str(note_icu);
            stdout.push('\n');
        }
        for _ in 0..12 {
            stdout.push_str(note_rv);
            stdout.push('\n');
        }
        for _ in 0..3 {
            stdout.push_str(note_self);
            stdout.push('\n');
        }
        stdout.push_str(summary);
        stdout.push('\n');

        let output = std::process::Output {
            status: status(1),
            stdout: stdout.into_bytes(),
            stderr: vec![],
        };
        let parsed = parse_check(&output);
        assert_eq!(parsed.findings.len(), 2);
        assert_eq!(parsed.notes.len(), 55);

        let compact = parsed.compact(700);
        // The root survives verbatim even at a tiny budget.
        assert!(
            compact.contains("root: proc-macro/build-script module missing (tokio-macros)"),
            "missing proc-macro root in: {compact}"
        );
        assert!(compact.contains("proc_macro_modules.rs"));
        // Dependent noise collapses into counts beneath its root.
        assert!(compact.contains("40 dependent `not checked` item(s) collapse under this root"));
        assert!(compact.contains("crate scope failed (icu_properties_data)"));
        assert!(compact.contains("crate scope failed (rustversion)"));
        assert!(compact.contains("crate scope failed (<self>)"));
        // The dependent notes themselves never flood the output.
        assert!(compact.matches("not checked because").count() == 0);
        // The unclassified finding is visible exactly once as an exemplar.
        assert_eq!(compact.matches("no `serde` in this scope").count(), 1);
        let head = compact.lines().next().unwrap();
        assert!(head.contains("not checked"), "{head}");
    }

    /// A missing build-script module and an explicit toolchain/source refusal
    /// classify into their own honest buckets; residual notes stay counted.
    #[test]
    #[cfg(unix)]
    fn causal_buckets_build_script_toolchain_and_residual() {
        let build_script = r#"{"id":"bs001","severity":"error","code":"macro-call-failed","path":"build.rs","line":1,"col":1,"message":"build script cannot run: the build script module is missing from the store"}"#;
        let tool_refused = r#"{"id":"tc001","severity":"error","code":"toolchain","path":"<workspace>","line":null,"col":null,"message":"the pinned source for this toolchain is missing; command refused"}"#;
        let odd_note = r#"{"note":"`src/x.rs`: body `y`: not checked because the config disables this lint"}"#;
        let stdout = format!(
            "{build_script}\n{tool_refused}\n{odd_note}\n{{\"errors\":2,\"warnings\":0}}\n"
        );
        let output = std::process::Output {
            status: status(1),
            stdout: stdout.clone().into_bytes(),
            stderr: vec![],
        };
        let parsed = parse_check(&output);
        assert_eq!(parsed.notes.len(), 1);
        let compact = parsed.compact(2000);
        assert!(compact.contains("proc-macro/build-script module missing (build-script)"));
        assert!(compact.contains("toolchain/source refused the command"));
        assert!(compact.contains("did not match a known root"));
        // Evidence line (non-JSON) with explicit toolchain refusal wording maps to its root.
        let with_stderr = std::process::Output {
            status: status(1),
            stdout: stdout.into_bytes(),
            stderr: b"error: toolchain shelf is missing; command refused\n".to_vec(),
        };
        let compact2 = parse_check(&with_stderr).compact(2000);
        assert!(compact2.contains("toolchain/source refused the command"));
    }

    /// The full-output artifact (checked.log) still carries every raw line:
    /// compaction may shrink the model view but never the archived report.
    #[test]
    #[cfg(unix)]
    fn artifact_log_keeps_raw_lines_while_compact_shrinks() {
        let mut stdout = String::from(
            r#"{"id":"r3fhe","severity":"error","code":"macro-call-failed","path":"a.rs","line":1,"col":1,"message":"the proc macro cannot run: `m`: tokio-macros 2.7.2: no compiled module"}"#,
        );
        stdout.push('\n');
        for i in 0..200 {
            let _ = writeln!(
                stdout,
                r#"{{"note":"`src/f{i}.rs`: body `b{i}`: not checked because its member's check scope holds `icu_properties_data`, which has no names record this command"}}"#
            );
        }
        let output = std::process::Output {
            status: status(1),
            stdout: stdout.into_bytes(),
            stderr: vec![],
        };
        let parsed = parse_check(&output);
        assert_eq!(parsed.notes.len(), 200);
        let compact = parsed.compact(256);
        assert!(compact.len() < 2000, "compact leaked raw noise: {}", compact.len());
        assert_eq!(parsed.log.matches("icu_properties_data").count(), 200);
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

    /// Real Blaze v2 wording (runs/omt-blaze-v2/quality/cases/toolchain-refusal):
    /// a missing shelf manifest is reported on stderr as `the toolchain
    /// manifest ...: not there; this blaze needs its own toolchain installed`.
    /// That wording must classify as a toolchain/source refusal root, not
    /// vanish into unclassified evidence.
    #[test]
    #[cfg(unix)]
    fn toolchain_manifest_not_there_wording_is_a_refusal_root() {
        let output = std::process::Output {
            status: status(1),
            stdout: vec![],
            stderr: b"error: the pinned toolchain: toolchain 099e991f: the toolchain manifest (file 099e991f in /home/u/.blaze/v2/toolchain): not there; this blaze needs its own toolchain installed, run `blaze-toolchain` from its tree or reinstall\n".to_vec(),
        };
        let parsed = parse_check(&output);
        // No JSON findings at all: the refusal arrives as raw stderr evidence.
        assert!(parsed.findings.is_empty());
        assert_eq!(parsed.evidence.len(), 1);
        let compact = parsed.compact(2000);
        assert!(
            compact.contains("root: toolchain/source refused the command"),
            "stderr refusal was not classified: {compact}"
        );
    }

    /// BLAZE-V2 tail-4 quality A/B harness (runs/omt-blaze-v2/quality/):
    /// replays each case's captured `blaze check` stdout/stderr/status through
    /// the real parse + causal-compaction path and writes the exact artifacts
    /// the two A/B arms see, plus the token-accounting details:
    ///   raw.txt      - the archived raw report (`Check::log`)
    ///   compact.txt  - the model-facing Root-Cause Frontier (max_tokens=2000)
    ///   details.json - TokenMeter counts (`tokenizer.json` or labelled estimate)
    /// Opt-in via OMT_BLAZE_QUALITY_CASES=<cases dir>; `#[ignore]` keeps the
    /// default suite hermetic.
    #[test]
    #[ignore]
    #[cfg(unix)]
    fn quality_dump_ab_artifacts() {
        let Ok(cases) = std::env::var("OMT_BLAZE_QUALITY_CASES") else {
            return;
        };
        let meter = crate::tools::blaze_tokens::TokenMeter::from_env().unwrap();
        let mut dirs: Vec<PathBuf> = std::fs::read_dir(&cases)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| path.is_dir())
            .collect();
        dirs.sort();
        for dir in dirs {
            let stdout = std::fs::read(dir.join("check.stdout.jsonl")).unwrap();
            let stderr = std::fs::read(dir.join("check.stderr.txt")).unwrap_or_default();
            let code: i32 = std::fs::read_to_string(dir.join("check.status"))
                .unwrap()
                .trim()
                .parse()
                .unwrap();
            let output = std::process::Output {
                status: status(code),
                stdout,
                stderr,
            };
            let parsed = parse_check(&output);
            let compact = parsed.compact(2000);
            let mut details = json!({
                "case": dir.file_name().unwrap().to_string_lossy(),
                "status": parsed.status,
                "success": parsed.success,
                "errors": parsed.errors,
                "warnings": parsed.warnings,
                "findings": parsed.findings.len(),
                "notes": parsed.notes.len(),
                "evidence": parsed.evidence.len(),
                "maxTokens": 2000,
            });
            meter.add_counts(&mut details, &parsed.log, &compact).unwrap();
            std::fs::write(dir.join("raw.txt"), &parsed.log).unwrap();
            std::fs::write(dir.join("compact.txt"), &compact).unwrap();
            std::fs::write(
                dir.join("details.json"),
                format!("{}\n", serde_json::to_string_pretty(&details).unwrap()),
            )
            .unwrap();
        }
    }
}
