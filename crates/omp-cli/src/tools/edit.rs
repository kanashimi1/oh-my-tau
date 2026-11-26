//! `edit` (`hashline` / `replace` / `patch` / `apply_patch` / `sloppy`, run by the
//! vendored `pi-edit` engine exactly as `omp`'s native `EditSession` does)
//! and `write`.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use async_trait::async_trait;
use pa_agent::types::AgentToolResult;
use pi_edit::engine::FileOp;
use pi_edit::error::{EditError, EditResult};
use pi_edit::path_policy::PathPolicy;
use pi_edit::session::{
    ApplyRequest, EditWriter, Session, SessionConfig, WriteRequest, WriteResponse,
};
use pi_edit::EditMode;
use serde_json::{json, Value};

use super::edit_diagnostics;
use super::util::{format_path_relative_to_cwd, resolve_to_cwd, str_arg};
use super::Env;

/// Parameter schema of `edit` for `mode` (omp `edit/schemas.ts`).
#[must_use]
pub fn schema(mode: EditMode) -> Value {
    match mode {
        EditMode::Hashline | EditMode::ApplyPatch | EditMode::Sloppy => json!({
            "type": "object",
            "properties": {"input": {"type": "string"}},
            "required": ["input"],
            "additionalProperties": false
        }),
        EditMode::Replace => json!({
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
                "replace_all": {"type": "boolean"}
            },
            "required": ["path", "old_string", "new_string"],
            "additionalProperties": false
        }),
        EditMode::Patch => json!({
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "edits": {"type": "array", "items": {
                    "type": "object",
                    "properties": {
                        "op": {"type": "string", "enum": ["create", "delete", "update"]},
                        "rename": {"type": "string"},
                        "diff": {"type": "string"}
                    },
                    "additionalProperties": false
                }}
            },
            "required": ["path", "edits"],
            "additionalProperties": false
        }),
    }
}

fn policy(cwd: &Path) -> PathPolicy {
    PathPolicy {
        cwd: cwd.to_path_buf(),
        home_dir: std::env::var_os("HOME").map_or_else(|| PathBuf::from("/"), PathBuf::from),
        url_schemes: Vec::new(),
        url_alias_schemes: Vec::new(),
        plan_writable_roots: Vec::new(),
        plan_active: false,
        block_auto_generated: true,
    }
}

/// Original bytes for one successful host write. Kept in memory only: this is
/// rollback for a live tool call, not a crash-recovery journal.
struct Undo {
    source: PathBuf,
    source_bytes: Option<Vec<u8>>,
    destination: Option<(PathBuf, Option<Vec<u8>>)>,
}

impl Undo {
    fn capture(request: &WriteRequest) -> EditResult<Self> {
        let source_bytes = read_optional(&request.absolute)?;
        let target = request.move_to.as_deref().unwrap_or(&request.absolute);
        let destination = if target == request.absolute {
            None
        } else {
            Some((target.to_path_buf(), read_optional(target)?))
        };
        Ok(Self {
            source: request.absolute.clone(),
            source_bytes,
            destination,
        })
    }

    fn restore(self) -> EditResult<()> {
        if let Some((path, bytes)) = self.destination {
            restore_path(&path, bytes)?;
        }
        restore_path(&self.source, self.source_bytes)
    }
}

fn read_optional(path: &Path) -> EditResult<Option<Vec<u8>>> {
    match std::fs::read(path) {
        Ok(bytes) => Ok(Some(bytes)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(io_err(path, &error)),
    }
}

fn restore_path(path: &Path, bytes: Option<Vec<u8>>) -> EditResult<()> {
    match bytes {
        Some(bytes) => {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).map_err(|error| io_err(parent, &error))?;
            }
            std::fs::write(path, bytes).map_err(|error| io_err(path, &error))
        }
        None => match std::fs::remove_file(path) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(io_err(path, &error)),
        },
    }
}

/// Host-side byte owner for the engine. Each instance belongs to one apply.
#[derive(Default)]
struct FsWriter {
    undo: parking_lot::Mutex<Vec<Undo>>,
    #[cfg(test)]
    fail_at: Option<usize>,
    #[cfg(test)]
    write_calls: std::sync::atomic::AtomicUsize,
}

fn io_err(path: &Path, e: &std::io::Error) -> EditError {
    EditError::Writer(format!("{}: {e}", path.display()))
}

#[async_trait]
impl EditWriter for FsWriter {
    async fn write(&self, request: WriteRequest) -> EditResult<WriteResponse> {
        #[cfg(test)]
        {
            let call = self
                .write_calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                + 1;
            if self.fail_at == Some(call) {
                return Err(EditError::Writer(format!(
                    "injected write failure at call {call}"
                )));
            }
        }
        let undo = Undo::capture(&request)?;
        let target = request.move_to.as_deref().unwrap_or(&request.absolute);
        let result: EditResult<WriteResponse> = (|| match request.op {
            FileOp::Delete => {
                std::fs::remove_file(&request.absolute)
                    .map_err(|e| io_err(&request.absolute, &e))?;
                Ok(WriteResponse::default())
            }
            FileOp::Noop => Ok(WriteResponse::default()),
            FileOp::Create | FileOp::Update => {
                let content = request.content.unwrap_or_default();
                if let Some(parent) = target.parent() {
                    std::fs::create_dir_all(parent).map_err(|e| io_err(parent, &e))?;
                }
                std::fs::write(target, &content).map_err(|e| io_err(target, &e))?;
                if request.move_to.is_some() && request.absolute != target {
                    std::fs::remove_file(&request.absolute)
                        .map_err(|e| io_err(&request.absolute, &e))?;
                }
                Ok(WriteResponse {
                    written: content,
                    diagnostics_json: None,
                })
            }
        })();
        match result {
            Ok(response) => {
                self.undo.lock().push(undo);
                Ok(response)
            }
            Err(error) => match undo.restore() {
                Ok(()) => Err(error),
                Err(rollback) => Err(EditError::Writer(format!(
                    "{error}; failed-write rollback failed: {rollback}"
                ))),
            },
        }
    }

    async fn rollback(&self, _completed: &[WriteRequest]) -> EditResult<()> {
        let completed = std::mem::take(&mut *self.undo.lock());
        let mut errors = Vec::new();
        for undo in completed.into_iter().rev() {
            if let Err(error) = undo.restore() {
                errors.push(error.to_string());
            }
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(EditError::Writer(errors.join("; ")))
        }
    }
}

fn edit_session(env: &Env, args: &Value) -> Session {
    let config = SessionConfig {
        mode: env.edit_mode,
        policy: policy(&env.cwd),
        allow_fuzzy: true,
        fuzzy_threshold: 0.95,
        enforce_seen_lines: true,
        raw_input: false,
    };
    let mut session = Session::new(config, env.store.clone());
    session.set_args_json(&args.to_string());
    session.finish();
    session
}

#[cfg(test)]
async fn apply_with_writer(
    env: &Env,
    args: &Value,
    writer: &dyn EditWriter,
) -> EditResult<pi_edit::session::ApplyOutcome> {
    edit_session(env, args)
        .apply(ApplyRequest::default(), writer)
        .await
}

/// Run one edit call.
///
/// # Errors
/// The engine's model-facing message (parse, locate, apply or write failure).
pub async fn edit(env: &Env, args: &Value) -> Result<AgentToolResult> {
    let mut session = edit_session(env, args);
    // Claims guard (OMP_CLAIMS=block): refuse before the first write when
    // another live endpoint holds a claim on a target path. Warning mode
    // checks after the apply instead (the engine knows the exact paths then)
    // and never blocks.
    if crate::tools::a2a::claims_block_mode_enabled() {
        let targets = edit_target_paths(env, args, &mut session);
        if let Some(reason) = crate::tools::a2a::claims_block_reason(&targets).await {
            bail!("{reason}");
        }
    }
    let outcome = {
        let _transaction = env.begin_edit_transaction();
        session
            .apply(ApplyRequest::default(), &FsWriter::default())
            .await
    }
    .map_err(|e| anyhow!("{e}"))?;
    let mut result = AgentToolResult::text(outcome.text.clone());
    result.details = edit_details(&outcome);
    // Advisory post-write LSP gate: report diagnostics new to this edit.
    // Never fails the edit; no cold start; bounded by GATE_BUDGET_MS.
    let changed: Vec<PathBuf> = outcome
        .files
        .iter()
        .filter(|f| matches!(f.op, FileOp::Create | FileOp::Update))
        .map(|f| f.absolute.clone())
        .collect();
    if let Some(gate) = edit_diagnostics::run(env, &changed).await {
        let oracle = gate.details();
        result
            .content
            .push(pa_agent::types::ToolResultContent::text(gate.section));
        if !result.details.is_object() {
            result.details = json!({});
        }
        result.details["oracle"] = oracle;
        result.details["diagnostics"] = Value::Array(gate.diagnostics);
    }
    // Advisory post-write claims guard: a live claim held by another
    // endpoint rides the result as a warning line (apply already proceeded);
    // a successful write by this agent auto-claims the path (15 min lease).
    if !changed.is_empty() {
        for line in crate::tools::a2a::claims_check_files(&changed).await {
            result
                .content
                .push(pa_agent::types::ToolResultContent::text(line));
        }
        crate::tools::a2a::claims_note_writes(&changed);
    }
    Ok(result)
}

/// Best-effort target paths of an edit call, for the pre-apply claims block:
/// the explicit `path` argument when the mode takes one, else the paths the
/// payload names (patch markers), resolved against the cwd.
fn edit_target_paths(env: &Env, args: &Value, session: &mut Session) -> Vec<PathBuf> {
    if let Some(raw) = args.get("path").and_then(Value::as_str) {
        return vec![resolve_to_cwd(raw, &env.cwd)];
    }
    let input = args
        .get("input")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let mut out = Vec::new();
    for line in input.lines() {
        let path = line
            .strip_prefix("*** Update File: ")
            .or_else(|| line.strip_prefix("*** Add File: "))
            .or_else(|| line.strip_prefix("*** Delete File: "))
            .or_else(|| line.strip_prefix("*** Move to: "));
        if let Some(path) = path {
            out.push(resolve_to_cwd(path.trim(), &env.cwd));
        }
    }
    if out.is_empty() {
        // Last resort: ask the engine for the preview batch (reads inputs,
        // no writes) so blocking never fails open on an unrecognized format.
        for file in session.preview().files {
            if file.error.is_none() {
                out.push(resolve_to_cwd(&file.display, &env.cwd));
            }
        }
    }
    out
}

/// omp `EditToolDetails`: the numbered `diff` (+ `firstChangedLine`, `op`,
/// `path`) the transcript card renders; several files add `perFileResults`.
fn edit_details(outcome: &pi_edit::session::ApplyOutcome) -> Value {
    let one = |f: &pi_edit::session::FileOutcome| {
        let mut d = serde_json::Map::new();
        d.insert("path".into(), Value::from(f.display.clone()));
        d.insert("diff".into(), Value::from(f.diff.clone()));
        d.insert(
            "op".into(),
            Value::from(match f.op {
                FileOp::Create => "create",
                FileOp::Delete => "delete",
                FileOp::Update | FileOp::Noop => "update",
            }),
        );
        if let Some(line) = f.first_changed_line {
            d.insert("firstChangedLine".into(), Value::from(line));
        }
        if let Some(to) = &f.move_to {
            d.insert("move".into(), Value::from(to.display().to_string()));
        }
        Value::Object(d)
    };
    match outcome.files.as_slice() {
        [] => Value::Null,
        [only] => one(only),
        files => {
            let mut d = one(&files[0]);
            d["perFileResults"] = Value::Array(files.iter().map(one).collect());
            d
        }
    }
}

/// `write`: create or overwrite a file.
///
/// # Errors
/// Missing arguments or filesystem failures.
pub async fn write(env: &Env, args: &Value) -> Result<AgentToolResult> {
    let raw = str_arg(args, "path")?;
    let Some(content) = args.get("content").and_then(Value::as_str) else {
        bail!("`content` is required to write {raw}");
    };
    // Route internal URLs through the scheme registry (TS write tool
    // `writeTarget` + `peelWriteSelector`).
    let registry = super::read::url_registry();
    let normalized = registry.normalize(raw);
    if registry.can_handle(&normalized) {
        // `agent://<id>` / `agent://all` writes are peer messages (TS
        // `AgentProtocolHandler.write` → `executeSend`): steer a running eval
        // subagent instead of hitting the read-only output store.
        if let Some(to) = normalized.strip_prefix("agent://") {
            // Delivery receipts are async; `write` is async like `edit`, so
            // just await the delivery.
            let delivered = super::eval_agents::deliver_message(to, content).await;
            return delivered
                .map(AgentToolResult::text)
                .map_err(|error| anyhow!(error));
        }
        let (url_path, sel) = registry.split(&normalized);
        if let Some(sel) = sel {
            if !sel.eq_ignore_ascii_case("raw") && !sel.eq_ignore_ascii_case("conflicts") {
                bail!(
                    "write does not accept the trailing selector \":{sel}\" \u{2014} it addresses a whole file. Remove \":{sel}\", or if the filename truly ends with it, percent-encode the \":\" as %3A."
                );
            }
        }
        registry.write(&url_path, content, &super::read::url_ctx(env))?;
        return Ok(AgentToolResult::text(format!(
            "Successfully wrote {} bytes to {}",
            content.len(),
            url_path
        )));
    }
    let path = resolve_to_cwd(raw, &env.cwd);
    // Claims guard (OMP_CLAIMS=block): refuse before the write when another
    // live endpoint holds a claim on the path (the repo anchor is resolved
    // by the guard itself; the file may not exist yet).
    if super::a2a::claims_block_mode_enabled() {
        if let Some(reason) = super::a2a::claims_block_reason(std::slice::from_ref(&path)).await {
            bail!("{reason}");
        }
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("cannot create {}", parent.display()))?;
    }
    std::fs::write(&path, content).with_context(|| format!("cannot write {}", path.display()))?;
    env.store
        .invalidate(&pi_edit::path_policy::canonical_key(&path));
    let mut result = AgentToolResult::text(format!(
        "Successfully wrote {} bytes to {}",
        content.len(),
        format_path_relative_to_cwd(&path, &env.cwd, false)
    ));
    // Advisory post-write LSP gate (same rules as `edit`).
    if let Some(gate) = edit_diagnostics::run(env, std::slice::from_ref(&path)).await {
        let oracle = gate.details();
        result
            .content
            .push(pa_agent::types::ToolResultContent::text(gate.section));
        if !result.details.is_object() {
            result.details = json!({});
        }
        result.details["oracle"] = oracle;
        result.details["diagnostics"] = Value::Array(gate.diagnostics);
    }
    // Advisory post-write claims guard (warning line) + auto-claim.
    for line in super::a2a::claims_check_files(std::slice::from_ref(&path)).await {
        result
            .content
            .push(pa_agent::types::ToolResultContent::text(line));
    }
    super::a2a::claims_note_writes(std::slice::from_ref(&path));
    Ok(result)
}

#[cfg(test)]
#[allow(clippy::disallowed_methods)] // tests: throwaway runtimes drive async code
mod tests {
    use super::*;
    use crate::tools::read::read;

    fn text_of(r: &AgentToolResult) -> String {
        r.content
            .iter()
            .filter_map(|c| match c {
                pa_agent::types::ToolResultContent::Text(t) => Some(t.text.clone()),
                pa_agent::types::ToolResultContent::Image(_) => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn tag_of(read_out: &str) -> String {
        let header = read_out.lines().next().unwrap();
        header
            .split('#')
            .nth(1)
            .unwrap()
            .trim_end_matches(']')
            .to_string()
    }

    #[tokio::test]
    async fn hashline_replace_range_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("f.txt"), "alpha\nbeta\ngamma\n").unwrap();
        let env = Env::new(dir.path(), true);
        let out = text_of(&read(&env, &json!({"path": "f.txt"})).unwrap());
        let tag = tag_of(&out);
        let input = format!("[f.txt#{tag}]\nPUT 2.=2:\n+BETA\n+beta2\n");
        let res = edit(&env, &json!({"input": input})).await.unwrap();
        assert_eq!(
            std::fs::read_to_string(dir.path().join("f.txt")).unwrap(),
            "alpha\nBETA\nbeta2\ngamma\n"
        );
        let text = text_of(&res);
        // result header carries the new snapshot tag
        assert!(text.starts_with("[f.txt#"), "{text}");
    }

    #[tokio::test]
    async fn hashline_insert_after_and_eof_append() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("f.txt"), "one\ntwo\n").unwrap();
        let env = Env::new(dir.path(), true);
        let tag = tag_of(&text_of(&read(&env, &json!({"path": "f.txt"})).unwrap()));
        edit(
            &env,
            &json!({"input": format!("[f.txt#{tag}]\nPUT >1:\n+one-and-a-half\n")}),
        )
        .await
        .unwrap();
        let tag = tag_of(&text_of(&read(&env, &json!({"path": "f.txt"})).unwrap()));
        edit(
            &env,
            &json!({"input": format!("[f.txt#{tag}]\nPUT >$:\n+three\n")}),
        )
        .await
        .unwrap();
        assert_eq!(
            std::fs::read_to_string(dir.path().join("f.txt")).unwrap(),
            "one\none-and-a-half\ntwo\nthree\n"
        );
    }

    #[tokio::test]
    async fn hashline_cut_deletes_lines() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("f.txt"), "a\nb\nc\nd\n").unwrap();
        let env = Env::new(dir.path(), true);
        let tag = tag_of(&text_of(&read(&env, &json!({"path": "f.txt"})).unwrap()));
        edit(
            &env,
            &json!({"input": format!("[f.txt#{tag}]\nCUT 2.=3\n")}),
        )
        .await
        .unwrap();
        assert_eq!(
            std::fs::read_to_string(dir.path().join("f.txt")).unwrap(),
            "a\nd\n"
        );
    }

    #[tokio::test]
    async fn stale_tag_and_garbage_are_rejected_without_writing() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("f.txt"), "a\nb\n").unwrap();
        let env = Env::new(dir.path(), true);
        let _ = read(&env, &json!({"path": "f.txt"})).unwrap();
        let err = edit(&env, &json!({"input": "[f.txt#0000]\nPUT 1.=1:\n+x\n"}))
            .await
            .unwrap_err();
        assert!(!err.to_string().is_empty());
        let err = edit(&env, &json!({"input": "this is not hashline"}))
            .await
            .unwrap_err();
        assert!(!err.to_string().is_empty());
        assert_eq!(
            std::fs::read_to_string(dir.path().join("f.txt")).unwrap(),
            "a\nb\n"
        );
    }

    #[tokio::test]
    async fn hashline_requires_lines_to_have_been_seen() {
        let dir = tempfile::tempdir().unwrap();
        let mut body = String::new();
        for i in 1..=50 {
            use std::fmt::Write as _;
            let _ = writeln!(body, "l{i}");
        }
        std::fs::write(dir.path().join("f.txt"), body).unwrap();
        let env = Env::new(dir.path(), true);
        let tag = tag_of(&text_of(
            &read(&env, &json!({"path": "f.txt:1-3"})).unwrap(),
        ));
        let err = edit(
            &env,
            &json!({"input": format!("[f.txt#{tag}]\nPUT 40.=40:\n+x\n")}),
        )
        .await
        .unwrap_err();
        assert!(!err.to_string().is_empty());
    }

    #[tokio::test]
    async fn apply_patch_commits_two_files_as_one_patch_set() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "alpha\n").unwrap();
        std::fs::write(dir.path().join("b.txt"), "beta\n").unwrap();
        let env = Env::with_mode(dir.path(), EditMode::ApplyPatch);
        let input = "*** Begin Patch\n*** Update File: a.txt\n@@\n-alpha\n+ALPHA\n*** Update File: b.txt\n@@\n-beta\n+BETA\n*** End Patch\n";
        edit(&env, &json!({"input": input})).await.unwrap();
        assert_eq!(
            std::fs::read_to_string(dir.path().join("a.txt")).unwrap(),
            "ALPHA\n"
        );
        assert_eq!(
            std::fs::read_to_string(dir.path().join("b.txt")).unwrap(),
            "BETA\n"
        );
    }

    #[tokio::test]
    async fn second_write_failure_rolls_back_first_byte_for_byte() {
        let dir = tempfile::tempdir().unwrap();
        let original_a = "alpha\r\nhéllo\r\n";
        let original_b = "beta\r\n";
        std::fs::write(dir.path().join("a.txt"), original_a).unwrap();
        std::fs::write(dir.path().join("b.txt"), original_b).unwrap();
        let env = Env::new(dir.path(), true);
        let tag_a = tag_of(&text_of(&read(&env, &json!({"path": "a.txt"})).unwrap()));
        let tag_b = tag_of(&text_of(&read(&env, &json!({"path": "b.txt"})).unwrap()));
        let input = format!("[a.txt#{tag_a}]\nPUT 1.=1:\n+ALPHA\n[b.txt#{tag_b}]\nPUT 1.=1:\n+BETA\n");
        let writer = FsWriter {
            fail_at: Some(2),
            ..FsWriter::default()
        };
        let error = apply_with_writer(&env, &json!({"input": input}), &writer)
            .await
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("injected write failure at call 2"),
            "{error}"
        );
        assert!(error.contains("Rolled back 1 completed write"), "{error}");
        assert_eq!(
            std::fs::read(dir.path().join("a.txt")).unwrap(),
            original_a.as_bytes()
        );
        assert_eq!(
            std::fs::read(dir.path().join("b.txt")).unwrap(),
            original_b.as_bytes()
        );
        let key_a = pi_edit::path_policy::canonical_key(&dir.path().join("a.txt"));
        let key_b = pi_edit::path_policy::canonical_key(&dir.path().join("b.txt"));
        assert_eq!(env.store.head_hash(&key_a).as_deref(), Some(tag_a.as_str()));
        assert_eq!(env.store.head_hash(&key_b).as_deref(), Some(tag_b.as_str()));
    }

    #[tokio::test]
    async fn stale_second_file_rejects_whole_set_with_current_context() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "a0\na1\n").unwrap();
        std::fs::write(dir.path().join("b.txt"), "b0\nb1\n").unwrap();
        let env = Env::new(dir.path(), true);
        let tag_a = tag_of(&text_of(&read(&env, &json!({"path": "a.txt"})).unwrap()));
        let tag_b = tag_of(&text_of(&read(&env, &json!({"path": "b.txt"})).unwrap()));
        std::fs::write(dir.path().join("b.txt"), "b0 changed\nb1\n").unwrap();
        let input = format!("[a.txt#{tag_a}]\nPUT 1.=1:\n+A0\n[b.txt#{tag_b}]\nPUT 1.=1:\n+B0\n");
        let error = edit(&env, &json!({"input": input}))
            .await
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("file changed between read and edit"),
            "{error}"
        );
        assert!(error.contains("*1:b0 changed"), "{error}");
        assert_eq!(
            std::fs::read_to_string(dir.path().join("a.txt")).unwrap(),
            "a0\na1\n"
        );
        assert_eq!(
            std::fs::read_to_string(dir.path().join("b.txt")).unwrap(),
            "b0 changed\nb1\n"
        );
    }

    #[tokio::test]
    async fn duplicate_anchors_and_out_of_range_lines_reject_without_writes() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("dup.txt"), "dup\ndup\n").unwrap();
        let replace = Env::with_mode(dir.path(), EditMode::Replace);
        let args = json!({"path": "dup.txt", "old_string": "dup", "new_string": "x"});
        let first = edit(&replace, &args).await.unwrap_err().to_string();
        let second = edit(&replace, &args).await.unwrap_err().to_string();
        assert_eq!(first, second);
        assert!(first.contains("2 occurrences"), "{first}");
        assert_eq!(
            std::fs::read_to_string(dir.path().join("dup.txt")).unwrap(),
            "dup\ndup\n"
        );

        std::fs::write(dir.path().join("lines.txt"), "one\ntwo\n").unwrap();
        let hashline = Env::new(dir.path(), true);
        let tag = tag_of(&text_of(
            &read(&hashline, &json!({"path": "lines.txt"})).unwrap(),
        ));
        let error = edit(
            &hashline,
            &json!({"input": format!("[lines.txt#{tag}]\nPUT 9.=9:\n+x\n")}),
        )
        .await
        .unwrap_err()
        .to_string();
        assert!(error.contains("lines 9 of lines.txt"), "{error}");
        assert!(error.contains("never displayed"), "{error}");
        assert_eq!(
            std::fs::read_to_string(dir.path().join("lines.txt")).unwrap(),
            "one\ntwo\n"
        );
    }

    #[tokio::test]
    async fn duplicate_hashline_file_anchor_resolution_is_deterministic() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("f.txt"),
            "one
",
        )
        .unwrap();
        let env = Env::new(dir.path(), true);
        let tag = tag_of(&text_of(&read(&env, &json!({"path": "f.txt"})).unwrap()));
        let input = format!(
            "[f.txt#{tag}]
PUT 1.=1:
+ONE
[f.txt#{tag}]
PUT 1.=1:
+UNO
"
        );
        let result = edit(&env, &json!({"input": input})).await.unwrap();
        let text = text_of(&result);
        assert!(
            text.contains("Multiple hunks targeted the same exact range; kept only the last"),
            "{text}"
        );
        assert_eq!(
            std::fs::read_to_string(dir.path().join("f.txt")).unwrap(),
            "UNO
"
        );
    }

    #[tokio::test]
    async fn bash_rejects_uncommitted_overlay_then_runs_after_commit_scope() {
        let dir = tempfile::tempdir().unwrap();
        let env = std::sync::Arc::new(Env::new(dir.path(), true));
        let transaction = env.begin_edit_transaction();
        let error = super::super::bash::bash(
            env.clone(),
            json!({"command": "printf bad > overlay-escaped"}),
            pa_agent::abort::AbortSignal::never(),
        )
        .await
        .unwrap_err()
        .to_string();
        assert!(error.contains("no isolated filesystem exists"), "{error}");
        assert!(!dir.path().join("overlay-escaped").exists());
        drop(transaction);
        super::super::bash::bash(
            env,
            json!({"command": "printf good > overlay-escaped"}),
            pa_agent::abort::AbortSignal::never(),
        )
        .await
        .unwrap();
        assert_eq!(
            std::fs::read_to_string(dir.path().join("overlay-escaped")).unwrap(),
            "good"
        );
    }

    #[tokio::test]
    async fn replace_mode_edits_by_string() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("f.txt"), "hello world\n").unwrap();
        let env = Env::with_mode(dir.path(), EditMode::Replace);
        edit(
            &env,
            &json!({"path": "f.txt", "old_string": "world", "new_string": "rust"}),
        )
        .await
        .unwrap();
        assert_eq!(
            std::fs::read_to_string(dir.path().join("f.txt")).unwrap(),
            "hello rust\n"
        );
    }

    #[tokio::test]
    async fn write_creates_parents_and_reports_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let env = Env::new(dir.path(), true);
        let r = write(&env, &json!({"path": "a/b/c.txt", "content": "héllo"}))
            .await
            .unwrap();
        assert_eq!(text_of(&r), "Successfully wrote 6 bytes to a/b/c.txt");
        assert_eq!(
            std::fs::read_to_string(dir.path().join("a/b/c.txt")).unwrap(),
            "héllo"
        );
    }

    #[tokio::test]
    async fn write_agent_url_is_a_message_not_a_read_only_store() {
        let dir = tempfile::tempdir().unwrap();
        let env = Env::new(dir.path(), true);
        let err = write(
            &env,
            &json!({"path": "agent://nobody-here", "content": "hi"}),
        )
        .await
        .unwrap_err()
        .to_string();
        assert!(
            err.starts_with("Failed: nobody-here is not running."),
            "{err}"
        );
        assert!(std::fs::read_dir(dir.path()).unwrap().next().is_none());
    }
}
