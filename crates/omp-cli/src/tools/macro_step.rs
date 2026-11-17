//! Deterministic, approval-gated repair playbooks exposed as `macro_step`.
//!
//! The first playbook consumes rustc's machine-applicable `unused_imports`
//! spans, applies them in byte space, and re-checks until clean or explicitly
//! stuck. Only the bounded digest reaches the model; raw cargo output never
//! does. This is deliberately a fixed local controller, not a model-in-loop
//! workflow or a general `cargo fix` wrapper.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use pa_agent::abort::{throw_if_aborted, AbortSignal};
use pa_agent::types::AgentToolResult;
use serde_json::{json, Value};
use tokio::process::Command;

use super::Env;
use crate::approval::{
    deny_message, parse_mode, resolve_approval, tool_decision, Gate, Policy,
};
use omp_ext::lifecycle::ApprovalMode;

static CALL_SEQ: AtomicU64 = AtomicU64::new(0);

fn next_call_id(prefix: &str) -> String {
    format!("{prefix}-{}", CALL_SEQ.fetch_add(1, Ordering::Relaxed))
}

/// Apply the same approval policy as the corresponding direct nested tool call.
/// Without an interactive gate, prompt/deny decisions fail closed.
async fn ensure_nested_allowed(
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

pub const DESCRIPTION: &str = "Run a bounded deterministic code-repair playbook in ONE tool call. `rust_unused_imports` repeatedly runs cargo check, applies only rustc MachineApplicable unused-import byte spans, and re-verifies (default/max 4 repair rounds). Returns a compact before/after digest instead of compiler logs. Never rewrites unrelated code; refuses paths outside the project. Prefer this over manual grep/read/edit loops for unused Rust imports.";

#[must_use]
pub fn schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "playbook": {"type": "string", "enum": ["rust_unused_imports"]},
            "path": {"type": "string", "description": "Cargo project/workspace directory; defaults to the session cwd."},
            "max_iterations": {"type": "integer", "minimum": 1, "maximum": 8, "default": 4},
            "timeout": {"type": "number", "description": "Per-cargo-check timeout in seconds (1-600, default 120)."}
        },
        "required": ["playbook"],
        "additionalProperties": false
    })
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Suggestion {
    file: PathBuf,
    start: usize,
    end: usize,
    replacement: String,
}

#[derive(Debug, Default)]
struct Check {
    success: bool,
    unused_warnings: usize,
    suggestions: Vec<Suggestion>,
    errors: usize,
    log: String,
}

fn diagnostic_spans(message: &Value) -> Vec<Suggestion> {
    let mut spans = Vec::new();
    let primary = message
        .get("spans")
        .and_then(Value::as_array)
        .into_iter()
        .flatten();
    let children = message
        .get("children")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|child| child.get("spans").and_then(Value::as_array))
        .flatten();
    // Prefer primary spans. Children are fallback only: they often contain
    // alternative explanations whose overlapping edits must not both apply.
    let candidates: Vec<&Value> = primary.collect();
    let owned_children;
    let selected = if candidates.iter().any(|span| {
        span.get("suggested_replacement").is_some()
            && span.get("suggestion_applicability").and_then(Value::as_str)
                == Some("MachineApplicable")
    }) {
        candidates
    } else {
        owned_children = children.collect();
        owned_children
    };
    for span in selected {
        if span.get("suggestion_applicability").and_then(Value::as_str) != Some("MachineApplicable")
        {
            continue;
        }
        let Some(replacement) = span.get("suggested_replacement").and_then(Value::as_str) else {
            continue;
        };
        let (Some(file), Some(start), Some(end)) = (
            span.get("file_name").and_then(Value::as_str),
            span.get("byte_start").and_then(Value::as_u64),
            span.get("byte_end").and_then(Value::as_u64),
        ) else {
            continue;
        };
        let (Ok(start), Ok(end)) = (usize::try_from(start), usize::try_from(end)) else {
            continue;
        };
        if start <= end {
            spans.push(Suggestion {
                file: PathBuf::from(file),
                start,
                end,
                replacement: replacement.to_owned(),
            });
        }
    }
    spans
}

fn parse_check(output: &std::process::Output) -> Check {
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let mut check = Check {
        success: output.status.success(),
        log: format!("$ cargo check --message-format=json\n{stdout}{stderr}"),
        ..Check::default()
    };
    for line in stdout.lines() {
        let Ok(value) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        if value.get("reason").and_then(Value::as_str) != Some("compiler-message") {
            continue;
        }
        let Some(message) = value.get("message") else {
            continue;
        };
        if message.get("level").and_then(Value::as_str) == Some("error") {
            check.errors += 1;
        }
        if message.pointer("/code/code").and_then(Value::as_str) == Some("unused_imports") {
            check.unused_warnings += 1;
            check.suggestions.extend(diagnostic_spans(message));
        }
    }
    check.suggestions.sort();
    check.suggestions.dedup();
    check
}

async fn cargo_check(project: &Path, timeout: Duration) -> Result<Check> {
    let output = tokio::time::timeout(
        timeout,
        Command::new("cargo")
            .args(["check", "--quiet", "--message-format=json"])
            .current_dir(project)
            .env("CARGO_TERM_COLOR", "never")
            .kill_on_drop(true)
            .output(),
    )
    .await
    .map_err(|_| anyhow!("cargo check timed out after {}s", timeout.as_secs()))?
    .context("failed to spawn cargo check")?;
    Ok(parse_check(&output))
}

struct PendingWrite {
    path: PathBuf,
    bytes: Vec<u8>,
    count: usize,
}

fn prepare_writes(project: &Path, suggestions: &[Suggestion]) -> Result<Vec<PendingWrite>> {
    let mut grouped = BTreeMap::<PathBuf, Vec<Suggestion>>::new();
    for suggestion in suggestions {
        let path = if suggestion.file.is_absolute() {
            suggestion.file.clone()
        } else {
            project.join(&suggestion.file)
        };
        grouped.entry(path).or_default().push(suggestion.clone());
    }
    let mut writes = Vec::new();
    for (path, mut spans) in grouped {
        let canonical = path
            .canonicalize()
            .with_context(|| format!("cannot resolve rustc span file {}", path.display()))?;
        if !canonical.starts_with(project) {
            bail!("rustc suggestion escapes project: {}", canonical.display());
        }
        let original = std::fs::read(&canonical)
            .with_context(|| format!("cannot read {}", canonical.display()))?;
        spans.sort();
        spans.dedup();
        let mut previous_end = 0;
        for span in &spans {
            if span.start < previous_end || span.end > original.len() {
                bail!(
                    "overlapping or out-of-bounds rustc suggestions in {}",
                    canonical.display()
                );
            }
            previous_end = span.end;
        }
        let mut bytes = original;
        for span in spans.iter().rev() {
            bytes.splice(
                span.start..span.end,
                span.replacement.as_bytes().iter().copied(),
            );
        }
        writes.push(PendingWrite {
            path: canonical,
            bytes,
            count: spans.len(),
        });
    }
    Ok(writes)
}

#[derive(Clone, Copy)]
struct RunSummary<'a> {
    status: &'a str,
    before: usize,
    after: usize,
    applied: usize,
    checks: usize,
    started: Instant,
}

fn compact_result(summary: RunSummary<'_>, writes: &[PendingWrite], log: &str) -> AgentToolResult {
    let RunSummary {
        status,
        before,
        after,
        applied,
        checks,
        started,
    } = summary;
    let failed = status != "clean";
    let mut details = json!({
        "workflow": "macro_step",
        "playbook": "rust_unused_imports",
        "status": status,
        "beforeUnusedImports": before,
        "afterUnusedImports": after,
        "appliedSuggestions": applied,
        "cargoChecks": checks,
        "changedFiles": writes.iter().map(|w| w.path.display().to_string()).collect::<Vec<_>>(),
        "elapsedMs": started.elapsed().as_millis(),
    });
    if failed && !log.is_empty() {
        if let Some(url) = super::util::save_artifact(log) {
            details["log"] = json!(url);
        }
    }
    let files = writes.len();
    let text = format!(
        "macro_step rust_unused_imports: {status} — unused imports {before} → {after}; {applied} suggestion(s), {files} file(s), {checks} cargo check(s); {} ms.{}",
        started.elapsed().as_millis(),
        details.get("log").and_then(Value::as_str).map_or_else(String::new, |url| format!(" Full log: {url}"))
    );
    let mut result = AgentToolResult::text(text);
    result.details = details;
    result.is_error = failed.then_some(true);
    result
}

/// Run one deterministic macro-step playbook.
///
/// # Errors
/// Invalid arguments, cargo execution failures, invalid rustc spans, and
/// filesystem failures. Approval refusals and unresolved warnings are compact
/// tool errors rather than protocol errors.
#[allow(clippy::too_many_lines)] // bounded parse→approve→apply→recheck controller
pub async fn macro_step(
    env: Arc<Env>,
    args: Value,
    signal: AbortSignal,
) -> Result<AgentToolResult> {
    let started = Instant::now();
    if args.get("playbook").and_then(Value::as_str) != Some("rust_unused_imports") {
        bail!("unknown macro_step playbook; expected `rust_unused_imports`");
    }
    throw_if_aborted(&signal)?;
    let raw_path = args.get("path").and_then(Value::as_str).unwrap_or(".");
    let requested = super::util::resolve_to_cwd(raw_path, &env.cwd);
    let project = requested
        .canonicalize()
        .with_context(|| format!("macro_step project does not exist: {}", requested.display()))?;
    let cwd = env
        .cwd
        .canonicalize()
        .context("cannot resolve session cwd")?;
    if !project.starts_with(&cwd) || !project.is_dir() {
        bail!("macro_step path must be a directory inside the session cwd");
    }
    let max_iterations = args
        .get("max_iterations")
        .and_then(Value::as_u64)
        .unwrap_or(4)
        .clamp(1, 8) as usize;
    let timeout = Duration::from_secs_f64(
        args.get("timeout")
            .and_then(Value::as_f64)
            .unwrap_or(120.0)
            .clamp(1.0, 600.0),
    );
    let call_id = next_call_id("macro_step");
    let bash_args = json!({"command": "cargo check --message-format=json", "cwd": project});
    let mut before = None;
    let mut applied = 0;
    let mut checks = 0;
    let mut all_writes = Vec::<PendingWrite>::new();
    let mut full_log = String::new();

    for iteration in 0..=max_iterations {
        throw_if_aborted(&signal)?;
        ensure_nested_allowed(
            "bash",
            &bash_args,
            &project,
            &format!("{call_id}-check-{iteration}"),
            &signal,
        )
        .await
        .map_err(|msg| anyhow!("macro_step cargo check refused: {msg}"))?;
        let check = cargo_check(&project, timeout).await?;
        checks += 1;
        before.get_or_insert(check.unused_warnings);
        let _ = write!(full_log, "\n\n=== check {} ===\n{}", checks, check.log);
        let after = check.unused_warnings;
        if after == 0 {
            let status = if check.success {
                "clean"
            } else {
                "cargo_failed"
            };
            return Ok(compact_result(
                RunSummary {
                    status,
                    before: before.unwrap_or(0),
                    after: 0,
                    applied,
                    checks,
                    started,
                },
                &all_writes,
                &full_log,
            ));
        }
        if iteration == max_iterations || check.suggestions.is_empty() {
            return Ok(compact_result(
                RunSummary {
                    status: "stuck",
                    before: before.unwrap_or(after),
                    after,
                    applied,
                    checks,
                    started,
                },
                &all_writes,
                &full_log,
            ));
        }
        let writes = prepare_writes(&project, &check.suggestions)?;
        if writes.is_empty() {
            return Ok(compact_result(
                RunSummary {
                    status: "stuck",
                    before: before.unwrap_or(after),
                    after,
                    applied,
                    checks,
                    started,
                },
                &all_writes,
                &full_log,
            ));
        }
        let edit_args = json!({
            "path": writes.iter().map(|w| w.path.display().to_string()).collect::<Vec<_>>().join(","),
            "old_string": "rustc MachineApplicable unused-import spans",
            "new_string": ""
        });
        if let Err(msg) = ensure_nested_allowed(
            "edit",
            &edit_args,
            &project,
            &format!("{call_id}-edit-{iteration}"),
            &signal,
        )
        .await
        {
            let mut result = compact_result(
                RunSummary {
                    status: "refused",
                    before: before.unwrap_or(after),
                    after,
                    applied,
                    checks,
                    started,
                },
                &all_writes,
                &full_log,
            );
            result.content = vec![pa_agent::types::ToolResultContent::text(format!(
                "macro_step: edit refused before this repair round: {msg}"
            ))];
            return Ok(result);
        }
        for write in &writes {
            std::fs::write(&write.path, &write.bytes).with_context(|| {
                format!("cannot apply rustc suggestions to {}", write.path.display())
            })?;
        }
        applied += writes.iter().map(|w| w.count).sum::<usize>();
        all_writes.extend(writes);
    }
    unreachable!("bounded loop always returns at its final check")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_only_machine_applicable_unused_import_spans() {
        let message = json!({
            "spans": [
                {"file_name":"src/lib.rs","byte_start":4,"byte_end":9,"suggested_replacement":"","suggestion_applicability":"MachineApplicable"},
                {"file_name":"src/lib.rs","byte_start":10,"byte_end":12,"suggested_replacement":"x","suggestion_applicability":"MaybeIncorrect"}
            ]
        });
        assert_eq!(
            diagnostic_spans(&message),
            vec![Suggestion {
                file: "src/lib.rs".into(),
                start: 4,
                end: 9,
                replacement: String::new()
            }]
        );
    }

    #[test]
    fn applies_reverse_byte_spans_without_shifting_offsets() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("Cargo.toml"),
            "[package]\nname='x'\nversion='0.1.0'\n",
        )
        .unwrap();
        std::fs::create_dir(dir.path().join("src")).unwrap();
        std::fs::write(dir.path().join("src/lib.rs"), "abcdef\n").unwrap();
        let writes = prepare_writes(
            dir.path(),
            &[
                Suggestion {
                    file: "src/lib.rs".into(),
                    start: 1,
                    end: 2,
                    replacement: "XX".into(),
                },
                Suggestion {
                    file: "src/lib.rs".into(),
                    start: 4,
                    end: 5,
                    replacement: String::new(),
                },
            ],
        )
        .unwrap();
        std::fs::write(&writes[0].path, &writes[0].bytes).unwrap();
        assert_eq!(
            std::fs::read_to_string(dir.path().join("src/lib.rs")).unwrap(),
            "aXXcdf\n"
        );
    }
}
