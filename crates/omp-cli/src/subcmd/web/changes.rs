//! Change center backend for the web UI.
//!
//! `GET  /api/changes?cwd=&path=` — changed files of a git repo, or the
//! unified diff of one file (untracked files diff against the empty file).
//! `POST /api/changes/checkpoint` — back up file bytes + metadata under the
//! user's data directory (never inside the project) and return the id.
//! `POST /api/changes/revert`     — batch-revert worktree changes for
//! `{cwd, paths: [...]}`, answering `{reverted, refused, error}`: tracked
//! modifications are reset to their index content (`git checkout --`),
//! untracked paths (`??` in porcelain) are refused and named in the
//! `error` text. Whole-repo checkout/reset is never used; git runs through
//! argument arrays with all disk work in `spawn_blocking`.

use std::io::Write as _;
use std::path::{Component, Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context as _, Result};
use hyper::body::Incoming;
use hyper::{Method, Request, Response, StatusCode};
use serde_json::{json, Value};

use super::{error, json_response, query_param, read_json, Body};

/// Diff / list caps so a huge repo cannot stall the hub.
const MAX_FILES: usize = 2000;
const MAX_DIFF_BYTES: usize = 512 * 1024;

pub async fn api(req: Request<Incoming>) -> Response<Body> {
    match (req.method(), req.uri().path()) {
        (&Method::GET, "/api/changes") => get_changes(req).await,
        (&Method::POST, "/api/changes/checkpoint") => checkpoint(req).await,
        (&Method::POST, "/api/changes/revert") => revert(req).await,
        _ => error(StatusCode::NOT_FOUND, "not found"),
    }
}

// ── git plumbing ─────────────────────────────────────────────────────────────

fn git(cwd: &Path, args: &[&str]) -> Result<String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(cwd)
        .args(args)
        .output()
        .context("spawn git")?;
    if !out.status.success() {
        bail!(
            "git {}: {}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Like [`git`] but accepts exit code 1 (`git diff --no-index` signals
/// "files differ" that way).
fn git_diff_ok(cwd: &Path, args: &[&str]) -> Result<String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(cwd)
        .args(args)
        .output()
        .context("spawn git")?;
    let code = out.status.code().unwrap_or(2);
    if code > 1 {
        bail!(
            "git {}: {}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

fn is_repo(cwd: &Path) -> bool {
    git(cwd, &["rev-parse", "--is-inside-work-tree"])
        .map(|s| s.trim() == "true")
        .unwrap_or(false)
}

/// Porcelain v1 with `-z`: entries are `XY <path>\0`, renames carry the
/// original path as an extra NUL field.
fn changed_files(cwd: &Path) -> Result<Vec<Value>> {
    let raw = git(
        cwd,
        &["status", "--porcelain=v1", "-z", "--untracked-files=normal"],
    )?;
    let mut out = Vec::new();
    let mut fields = raw.split('\0').filter(|f| !f.is_empty()).peekable();
    while let Some(entry) = fields.next() {
        if out.len() >= MAX_FILES {
            break;
        }
        if entry.len() < 4 {
            continue;
        }
        let mut chars = entry.chars();
        let x = chars.next().unwrap_or(' ');
        let y = chars.next().unwrap_or(' ');
        let path = entry[3..].to_owned();
        if x == 'R' || x == 'C' || y == 'R' || y == 'C' {
            fields.next(); // skip the rename source path
        }
        let untracked = x == '?' && y == '?';
        let code = if untracked {
            "?".to_owned()
        } else {
            format!("{}{}", x, y).trim().to_owned()
        };
        let label = if untracked {
            "untracked".to_owned()
        } else {
            match code.as_str() {
                "M" => "modified".into(),
                "A" => "added".into(),
                "D" => "deleted".into(),
                "T" => "typechange".into(),
                "R" => "renamed".into(),
                "C" => "copied".into(),
                c if c.contains('D') => "deleted".into(),
                c if c.contains('A') => "added".into(),
                c if c.contains('M') => "modified".into(),
                _ => "changed".into(),
            }
        };
        let mtime = std::fs::metadata(cwd.join(&path))
            .ok()
            .and_then(|m| m.modified().ok())
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_millis() as u64);
        out.push(json!({
            "path": path,
            "abs": cwd.join(&path),
            "code": code,
            "status": label,
            "untracked": untracked,
            "staged": !untracked && x != ' ',
            "mtimeMs": mtime,
        }));
    }
    Ok(out)
}

/// Unified diff of one file (tracked: vs `HEAD`, covering staged + unstaged;
/// untracked: vs the empty file). Adds/deletions counted from the body.
fn file_diff(cwd: &Path, rel: &str, abs: &Path, untracked: bool) -> Result<Value> {
    let text = if untracked {
        if abs.is_file() {
            git_diff_ok(
                cwd,
                &[
                    "diff",
                    "--no-color",
                    "--no-index",
                    "--",
                    "/dev/null",
                    &abs.to_string_lossy(),
                ],
            )?
        } else {
            String::new()
        }
    } else {
        git(cwd, &["diff", "--no-color", "HEAD", "--", rel])?
    };
    if text.len() > MAX_DIFF_BYTES {
        let cut = text
            .char_indices()
            .nth(MAX_DIFF_BYTES)
            .map_or(text.len(), |(i, _)| i);
        let mut t = text[..cut].to_owned();
        t.push_str("\n… diff truncated\n");
        return Ok(json!({"path": rel, "diff": t, "truncated": true}));
    }
    let mut adds = 0usize;
    let mut dels = 0usize;
    for line in text.lines() {
        match line.as_bytes().first() {
            Some(b'+') if !line.starts_with("+++") => adds += 1,
            Some(b'-') if !line.starts_with("---") => dels += 1,
            _ => {}
        }
    }
    Ok(json!({
        "path": rel,
        "diff": text,
        "additions": adds,
        "deletions": dels,
        "truncated": false,
    }))
}

// ── path safety ──────────────────────────────────────────────────────────────

fn has_escape(rel: &str) -> bool {
    Path::new(rel).components().any(|c| {
        matches!(
            c,
            Component::ParentDir | Component::RootDir | Component::Prefix(_)
        )
    })
}

/// Canonical git worktree root, also when the supplied cwd is a subdirectory.
fn repo_root(cwd: &Path) -> Option<PathBuf> {
    let root = git(cwd, &["rev-parse", "--show-toplevel"]).ok()?;
    Path::new(root.trim()).canonicalize().ok()
}

/// Resolve `repo/<rel>` canonically and verify it stays inside `repo`.
/// Works for missing files by canonicalizing the deepest existing ancestor.
fn safe_join(repo: &Path, rel: &str) -> Result<PathBuf> {
    if rel.is_empty() || has_escape(rel) {
        bail!("unsafe path {rel:?}");
    }
    let target = repo.join(rel);
    // canonicalize the deepest existing ancestor (the file itself may be
    // deleted or untracked) and require the target to stay inside the repo
    let mut probe = target.as_path();
    while !probe.exists() {
        probe = probe
            .parent()
            .context(format!("cannot resolve {rel:?} within the repository"))?;
    }
    let canon = probe.canonicalize().context("canonicalize repo path")?;
    if !canon.starts_with(repo) {
        bail!("path escapes the repository: {rel:?}");
    }
    if !target.starts_with(repo) {
        bail!("path escapes the repository: {rel:?}");
    }
    Ok(target)
}

// ── checkpoints ──────────────────────────────────────────────────────────────

fn home() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}

fn backup_root() -> Result<PathBuf> {
    let home = home().context("no home directory for checkpoints")?;
    Ok(home.join(".local/share/omp-web/changes-checkpoints"))
}

fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

fn new_checkpoint_id() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("cp-{nanos:x}")
}

/// Copy `file` into `dir` preserving the relative layout, plus its mtime.
fn backup_file(file: &Path, dir: &Path, rel: &str) -> Result<PathBuf> {
    let dest = dir.join(rel);
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("mkdir {}", parent.display()))?;
    }
    std::fs::copy(file, &dest).with_context(|| format!("copy {}", file.display()))?;
    if let Ok(meta) = std::fs::metadata(file) {
        if let Ok(modified) = meta.modified() {
            if let Ok(f) = std::fs::OpenOptions::new().write(true).open(&dest) {
                let _ = f.set_modified(modified);
            }
        }
    }
    Ok(dest)
}

fn write_manifest(dir: &Path, manifest: &Value) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let path = dir.join("manifest.json");
    let mut f =
        std::fs::File::create(&path).with_context(|| format!("create {}", path.display()))?;
    f.write_all(serde_json::to_string_pretty(manifest)?.as_bytes())?;
    Ok(())
}

/// Snapshot one file's current bytes + metadata. Returns `(id, backup rel)`.
/// Missing files produce a checkpoint that records their absence, so the
/// removed bytes stay recoverable.
fn checkpoint_one(repo: &Path, rel: &str, abs: &Path, note: &str) -> Result<(String, Value)> {
    let id = new_checkpoint_id();
    let dir = backup_root()?.join(&id);
    let existed = abs.is_file();
    let backup_rel = if existed {
        backup_file(abs, &dir, rel)?
    } else {
        PathBuf::new()
    };
    let mtime = existed
        .then(|| std::fs::metadata(abs).ok())
        .flatten()
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_millis() as u64);
    let entry = json!({
        "path": rel,
        "abs": abs,
        "repo": repo,
        "existed": existed,
        "backup": if existed { json!(backup_rel.to_string_lossy()) } else { Value::Null },
        "mtimeMs": mtime,
    });
    write_manifest(
        &dir,
        &json!({
            "id": id,
            "createdMs": now_ms() as u64,
            "note": note,
            "files": [entry],
        }),
    )?;
    Ok((id, entry))
}

// ── HTTP handlers ────────────────────────────────────────────────────────────

fn cwd_param(req: &Request<Incoming>) -> Result<PathBuf> {
    let raw = query_param(req, "cwd").context("missing cwd")?;
    Ok(PathBuf::from(raw))
}

async fn get_changes(req: Request<Incoming>) -> Response<Body> {
    let cwd = match cwd_param(&req) {
        Ok(c) => c,
        Err(e) => return error(StatusCode::BAD_REQUEST, &format!("{e:#}")),
    };
    let one = query_param(&req, "path");
    let res = tokio::task::spawn_blocking(move || list_or_diff(cwd, one)).await;
    match res {
        Ok(Ok(v)) => json_response(StatusCode::OK, &v),
        Ok(Err(e)) => error(StatusCode::INTERNAL_SERVER_ERROR, &format!("{e:#}")),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, &format!("{e:#}")),
    }
}

fn list_or_diff(cwd: PathBuf, one: Option<String>) -> Result<Value> {
    let repo = repo_root(&cwd).context("cwd does not exist")?;
    if !is_repo(&repo) {
        bail!("not a git repository: {}", repo.display());
    }
    match one.filter(|p| !p.is_empty()) {
        None => {
            let files = changed_files(&repo)?;
            Ok(json!({"cwd": repo, "files": files, "truncated": files.len() >= MAX_FILES}))
        }
        Some(rel) => {
            let entry = changed_files(&repo)?
                .into_iter()
                .find(|f| f.get("path").and_then(Value::as_str) == Some(rel.as_str()));
            let untracked = entry
                .as_ref()
                .and_then(|f| f.get("untracked"))
                .and_then(Value::as_bool)
                .unwrap_or_else(|| !repo.join(&rel).exists() || !tracked(&repo, &rel));
            let abs = safe_join(&repo, &rel)?;
            let mut v = file_diff(&repo, &rel, &abs, untracked)?;
            if let (Some(obj), Some(e)) = (v.as_object_mut(), entry) {
                obj.insert("file".into(), e);
            }
            Ok(v)
        }
    }
}

fn tracked(repo: &Path, rel: &str) -> bool {
    git(repo, &["ls-files", "--", rel])
        .map(|s| !s.trim().is_empty())
        .unwrap_or(false)
}

async fn checkpoint(req: Request<Incoming>) -> Response<Body> {
    let body = match read_json(req).await {
        Ok(v) => v,
        Err(e) => return error(StatusCode::BAD_REQUEST, &format!("read body: {e:#}")),
    };
    let res = tokio::task::spawn_blocking(move || do_checkpoint(&body)).await;
    match res {
        Ok(Ok(v)) => json_response(StatusCode::OK, &v),
        Ok(Err(e)) => error(StatusCode::INTERNAL_SERVER_ERROR, &format!("{e:#}")),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, &format!("{e:#}")),
    }
}

fn do_checkpoint(body: &Value) -> Result<Value> {
    let cwd = body
        .get("cwd")
        .and_then(Value::as_str)
        .context("missing cwd")?;
    let repo = repo_root(Path::new(cwd)).context("not a git worktree")?;
    let rel = body
        .get("path")
        .and_then(Value::as_str)
        .context("missing path")?;
    let abs = safe_join(&repo, rel)?;
    let (id, entry) = checkpoint_one(&repo, rel, &abs, "manual checkpoint")?;
    Ok(json!({
        "ok": true,
        "checkpoint": id,
        "file": entry,
    }))
}

async fn revert(req: Request<Incoming>) -> Response<Body> {
    let body = match read_json(req).await {
        Ok(v) => v,
        Err(e) => return error(StatusCode::BAD_REQUEST, &format!("read body: {e:#}")),
    };
    let res = tokio::task::spawn_blocking(move || do_revert(&body)).await;
    match res {
        Ok(Ok(v)) => json_response(StatusCode::OK, &v),
        Ok(Err(e)) => error(StatusCode::INTERNAL_SERVER_ERROR, &format!("{e:#}")),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, &format!("{e:#}")),
    }
}

/// Revert a batch of paths. Tracked modifications are reset to their index
/// content with `git checkout --`; untracked paths (`??` in porcelain
/// status) have no committed content to fall back to and are refused instead,
/// with the refused paths named in the response's `error` text.
fn do_revert(body: &Value) -> Result<Value> {
    let cwd = body
        .get("cwd")
        .and_then(Value::as_str)
        .context("missing cwd")?;
    let repo = repo_root(Path::new(cwd)).context("not a git worktree")?;
    let paths = body
        .get("paths")
        .and_then(Value::as_array)
        .context("missing paths")?;
    let mut reverted = Vec::new();
    let mut refused = Vec::new();
    for path in paths {
        let rel = match path.as_str().filter(|p| !p.is_empty()) {
            Some(rel) => rel,
            None => continue,
        };
        if safe_join(&repo, rel).is_err() {
            refused.push(rel.to_owned());
            continue;
        }
        let status = git(&repo, &["status", "--porcelain", "--", rel])?;
        if status.trim().is_empty() || status.starts_with("??") {
            // Unknown to git or untracked: there is nothing to check out.
            refused.push(rel.to_owned());
            continue;
        }
        git(&repo, &["checkout", "--", rel])?;
        reverted.push(rel.to_owned());
    }
    let error = if refused.is_empty() {
        Value::Null
    } else {
        json!(format!("refused paths: {}", refused.join(", ")))
    };
    Ok(json!({"reverted": reverted, "refused": refused, "error": error}))
}
