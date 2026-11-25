//! Deterministic, approval-gated repair playbooks exposed as `macro_step`.
//!
//! `rust_unused_imports` consumes Blaze v2's structured findings and applies
//! only its `unused-import` fixes. Blaze rechecks after every atomic fix
//! transaction. No cargo/rustc fallback and no raw compiler wall reaches the
//! model; full diagnostics are archived on failure.

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use pa_agent::abort::{AbortSignal, throw_if_aborted};
use pa_agent::types::AgentToolResult;
use serde_json::{Value, json};

use super::Env;
use super::blaze::{check, ensure_nested_allowed, fix};

static CALL_SEQ: AtomicU64 = AtomicU64::new(0);

pub(crate) fn next_call_id(prefix: &str) -> String {
    format!("{prefix}-{}", CALL_SEQ.fetch_add(1, Ordering::Relaxed))
}

pub const DESCRIPTION: &str = "Run a bounded deterministic code-repair playbook in ONE tool call. `rust_unused_imports` uses Blaze v2 structured findings, applies only Blaze unused-import fixes by finding id, and re-verifies (default/max 4 repair rounds). Returns a compact before/after digest instead of compiler logs. No cargo fallback. Requires Blaze `unused-import` lint enabled in workspace metadata. Never rewrites unrelated code; refuses paths outside the project. Prefer this over manual grep/read/edit loops for unused Rust imports.";

#[must_use]
pub fn schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "playbook": {"type": "string", "enum": ["rust_unused_imports"]},
            "path": {"type": "string", "description": "Cargo project/workspace directory; defaults to the session cwd."},
            "max_iterations": {"type": "integer", "minimum": 1, "maximum": 8, "default": 4},
            "timeout": {"type": "number", "description": "Per-Blaze-command timeout in seconds (1-600, default 120)."}
        },
        "required": ["playbook"],
        "additionalProperties": false
    })
}

/// Blaze v2 runs no lints at its default `rust` step. Fail closed rather
/// than reporting a false clean when its native unused-import fix is disabled.
fn require_unused_import_lint(project: &Path, cwd: &Path) -> Result<()> {
    let mut package = None;
    for ancestor in project.ancestors() {
        if !ancestor.starts_with(cwd) {
            break;
        }
        let manifest = ancestor.join("Cargo.toml");
        if !manifest.is_file() {
            continue;
        }
        let text = std::fs::read_to_string(&manifest)
            .with_context(|| format!("cannot read {}", manifest.display()))?;
        let doc: toml::Value = toml::from_str(&text)
            .with_context(|| format!("cannot parse {}", manifest.display()))?;
        let blaze = doc
            .get("workspace")
            .and_then(|it| it.get("metadata"))
            .and_then(|it| it.get("blaze"));
        let enabled = blaze.is_some_and(|blaze| {
            blaze.get("lints").and_then(toml::Value::as_str) == Some("strict")
                || blaze
                    .get("lints-add")
                    .and_then(toml::Value::as_array)
                    .is_some_and(|rules| {
                        rules.iter().any(|rule| rule.as_str() == Some("unused-import"))
                    })
        });
        if doc.get("workspace").is_some() || package.is_none() {
            package = Some((manifest.clone(), enabled));
        }
        if doc.get("workspace").is_some() {
            break;
        }
    }
    let Some((manifest, enabled)) = package else {
        bail!("macro_step found no Cargo.toml for Blaze workspace");
    };
    if !enabled {
        bail!(
            "Blaze v2's default `rust` step runs no unused-import lint; enable it in {} with `[workspace.metadata.blaze] lints-add = [\"unused-import\"]` (or `lints = \"strict\"`)",
            manifest.display()
        );
    }
    Ok(())
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

fn compact_result(
    summary: RunSummary<'_>,
    changed_files: &BTreeSet<PathBuf>,
    log: &str,
) -> AgentToolResult {
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
        "engine": "blaze-v2",
        "status": status,
        "beforeUnusedImports": before,
        "afterUnusedImports": after,
        "appliedSuggestions": applied,
        "blazeChecks": checks,
        "changedFiles": changed_files.iter().map(|p| p.display().to_string()).collect::<Vec<_>>(),
        "elapsedMs": started.elapsed().as_millis(),
    });
    if failed && !log.is_empty() {
        if let Some(url) = super::util::save_artifact(log) {
            details["log"] = json!(url);
        }
    }
    let text = format!(
        "macro_step rust_unused_imports: {status} — unused imports {before} → {after}; {applied} Blaze fix(es), {} file(s), {checks} Blaze check(s); {} ms.{}",
        changed_files.len(),
        started.elapsed().as_millis(),
        details
            .get("log")
            .and_then(Value::as_str)
            .map_or_else(String::new, |url| format!(" Full log: {url}"))
    );
    let mut result = AgentToolResult::text(text);
    result.details = details;
    result.is_error = failed.then_some(true);
    result
}

/// Run one deterministic macro-step playbook.
///
/// # Errors
/// Invalid arguments, Blaze execution failures, approval refusals and output
/// failures. Unresolved findings are compact tool errors, not protocol errors.
#[allow(clippy::too_many_lines)] // bounded parse→approve→fix→recheck controller
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
    require_unused_import_lint(&project, &cwd)?;
    let call_id = next_call_id("macro_step");
    let bash_args = json!({"command": "blaze check --view json --each", "cwd": project});
    let mut before = None;
    let mut applied = 0;
    let mut checks = 0;
    let mut changed_files = BTreeSet::new();
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
        .map_err(|msg| anyhow!("macro_step Blaze check refused: {msg}"))?;
        let checked = check(&project, timeout).await?;
        checks += 1;
        before.get_or_insert(checked.unused_import_count());
        let _ = write!(full_log, "\n\n=== check {checks} ===\n{}", checked.log);
        let after = checked.unused_import_count();
        if after == 0 {
            return Ok(compact_result(
                RunSummary {
                    status: if checked.success {
                        "clean"
                    } else {
                        "blaze_failed"
                    },
                    before: before.unwrap_or(0),
                    after: 0,
                    applied,
                    checks,
                    started,
                },
                &changed_files,
                &full_log,
            ));
        }
        let ids: Vec<String> = checked
            .unused_imports()
            .filter(|finding| finding.fix.is_some())
            .map(|finding| finding.id.clone())
            .collect();
        if iteration == max_iterations || ids.is_empty() {
            return Ok(compact_result(
                RunSummary {
                    status: "stuck",
                    before: before.unwrap_or(after),
                    after,
                    applied,
                    checks,
                    started,
                },
                &changed_files,
                &full_log,
            ));
        }
        let edit_args = json!({
            "path": checked.unused_imports().map(|it| it.path.as_str()).collect::<Vec<_>>().join(","),
            "old_string": "Blaze unused-import findings",
            "new_string": ""
        });
        if let Err(msg) = ensure_nested_allowed(
            "edit",
            &edit_args,
            &project,
            &format!("{call_id}-fix-{iteration}"),
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
                &changed_files,
                &full_log,
            );
            result.content = vec![pa_agent::types::ToolResultContent::text(format!(
                "macro_step: edit refused before this repair round: {msg}"
            ))];
            return Ok(result);
        }
        for path in checked.unused_imports().map(|it| Path::new(&it.path)) {
            changed_files.insert(if path.is_absolute() {
                path.to_path_buf()
            } else {
                project.join(path)
            });
        }
        applied += ids.len();
        let fixed = fix(&project, timeout, &ids).await?;
        // `--fix` checks before and after its atomic write.
        checks += 2;
        let _ = write!(full_log, "\n\n=== fix + recheck ===\n{}", fixed.log);
        if fixed.success && fixed.unused_import_count() == 0 {
            return Ok(compact_result(
                RunSummary {
                    status: "clean",
                    before: before.unwrap_or(0),
                    after: 0,
                    applied,
                    checks,
                    started,
                },
                &changed_files,
                &full_log,
            ));
        }
        if fixed.unused_import_count() >= after {
            return Ok(compact_result(
                RunSummary {
                    status: if fixed.success {
                        "stuck"
                    } else {
                        "blaze_failed"
                    },
                    before: before.unwrap_or(after),
                    after: fixed.unused_import_count(),
                    applied,
                    checks,
                    started,
                },
                &changed_files,
                &full_log,
            ));
        }
    }
    unreachable!("bounded loop always returns at its final check")
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unused_import_repair_requires_an_enabled_blaze_lint() {
        let dir = tempfile::tempdir().unwrap();
        let manifest = dir.path().join("Cargo.toml");
        std::fs::write(&manifest, "[package]\nname = \"demo\"\nversion = \"0.1.0\"\n").unwrap();
        assert!(require_unused_import_lint(dir.path(), dir.path()).is_err());
        std::fs::write(
            &manifest,
            "[package]\nname = \"demo\"\nversion = \"0.1.0\"\n[workspace.metadata.blaze]\nlints-add = [\"unused-import\"]\n",
        )
        .unwrap();
        require_unused_import_lint(dir.path(), dir.path()).unwrap();
    }
}
