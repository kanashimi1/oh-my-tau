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

/// Host-side byte owner for the engine: plain filesystem writes.
struct FsWriter;

fn io_err(path: &Path, e: &std::io::Error) -> EditError {
    EditError::Writer(format!("{}: {e}", path.display()))
}

#[async_trait]
impl EditWriter for FsWriter {
    async fn write(&self, request: WriteRequest) -> EditResult<WriteResponse> {
        let target = request.move_to.as_deref().unwrap_or(&request.absolute);
        match request.op {
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
        }
    }
}

/// Run one edit call.
///
/// # Errors
/// The engine's model-facing message (parse, locate, apply or write failure).
pub async fn edit(env: &Env, args: &Value) -> Result<AgentToolResult> {
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
    let outcome = session
        .apply(ApplyRequest::default(), &FsWriter)
        .await
        .map_err(|e| anyhow!("{e}"))?;
    let mut result = AgentToolResult::text(outcome.text.clone());
    result.details = edit_details(&outcome);
    Ok(result)
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
pub fn write(env: &Env, args: &Value) -> Result<AgentToolResult> {
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
            // Delivery receipts are async; `write` itself is a sync tool on a
            // `spawn_blocking` thread (see `tools.rs`), so drive the future to
            // completion on the current runtime (or a throwaway one, e.g. in
            // unit tests without a reactor).
            let delivered = match tokio::runtime::Handle::try_current() {
                Ok(handle) => handle.block_on(super::eval_agents::deliver_message(to, content)),
                Err(_) => tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("write agent:// delivery runtime")
                    .block_on(super::eval_agents::deliver_message(to, content)),
            };
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
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("cannot create {}", parent.display()))?;
    }
    std::fs::write(&path, content).with_context(|| format!("cannot write {}", path.display()))?;
    env.store
        .invalidate(&pi_edit::path_policy::canonical_key(&path));
    Ok(AgentToolResult::text(format!(
        "Successfully wrote {} bytes to {}",
        content.len(),
        format_path_relative_to_cwd(&path, &env.cwd, false)
    )))
}

#[cfg(test)]
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

    #[test]
    fn write_creates_parents_and_reports_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let env = Env::new(dir.path(), true);
        let r = write(&env, &json!({"path": "a/b/c.txt", "content": "héllo"})).unwrap();
        assert_eq!(text_of(&r), "Successfully wrote 6 bytes to a/b/c.txt");
        assert_eq!(
            std::fs::read_to_string(dir.path().join("a/b/c.txt")).unwrap(),
            "héllo"
        );
    }

    #[test]
    fn write_agent_url_is_a_message_not_a_read_only_store() {
        let dir = tempfile::tempdir().unwrap();
        let env = Env::new(dir.path(), true);
        let err = write(
            &env,
            &json!({"path": "agent://nobody-here", "content": "hi"}),
        )
        .unwrap_err()
        .to_string();
        assert!(
            err.starts_with("Failed: nobody-here is not running."),
            "{err}"
        );
        assert!(std::fs::read_dir(dir.path()).unwrap().next().is_none());
    }
}
