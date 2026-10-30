//! `/wt`, `/worktree`: per-session git worktrees. TS: slash-commands/builtin-lifecycle.ts
//! (session/session-worktree.ts). Also supports parallel child agents (`/wt run`).

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use serde_json::json;

use crate::rpc::slash::Outcome;
use crate::runtime::Runtime;

const USAGE: &str = "usage: /wt new [name] | /wt list | /wt remove <name> [--force] | /wt merge <name> | /wt run <n> <prompt> | /wt status";

/// Branch namespace used for worktrees.
const BRANCH_PREFIX: &str = "omt";

// ---------------------------------------------------------------------------
// Pure helpers (unit-tested)
// ---------------------------------------------------------------------------

/// Sanitise a worktree/branch name: collapse disallowed runs to `-`, trim
/// leading/trailing `-` and `.`, forbid git-unsafe names. Returns `None` when
/// nothing usable remains.
fn sanitize_name(raw: &str) -> Option<String> {
    // Collapse runs of `.` into a single `-` (a lone `.` is kept).
    let mut collapsed = String::with_capacity(raw.len());
    let mut dots = 0usize;
    for ch in raw.chars() {
        if ch == '.' {
            dots += 1;
            continue;
        }
        if dots == 1 {
            collapsed.push('.');
        } else if dots > 1 {
            collapsed.push('-');
        }
        dots = 0;
        collapsed.push(ch);
    }
    if dots == 1 {
        collapsed.push('.');
    } else if dots > 1 {
        collapsed.push('-');
    }
    let raw: &str = &collapsed;

    let mut out = String::with_capacity(raw.len());
    let mut dash = false;
    for ch in raw.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' || ch == '.' {
            out.push(ch);
            dash = false;
        } else if !dash && !out.is_empty() {
            out.push('-');
            dash = true;
        }
    }
    let trimmed = out.trim_matches(|c| c == '-' || c == '.');
    if trimmed.is_empty()
        || trimmed == "."
        || trimmed.starts_with('-')
        || trimmed.to_ascii_lowercase().ends_with(".lock")
        || trimmed.contains("..")
    {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// `<repo-parent>/<repo>-wt-<name>` for a repository root.
fn worktree_path(repo_root: &Path, name: &str) -> Option<PathBuf> {
    let parent = repo_root.parent()?;
    if parent.as_os_str().is_empty() {
        // Relative or parentless repo path: no stable sibling location.
        return None;
    }
    let repo = repo_root.file_name()?.to_str()?;
    Some(parent.join(format!("{repo}-wt-{name}")))
}

/// Default name for `/wt new` without an argument: `wt-<yyyymmdd-hhmmss>`
/// (mirrors TS `defaultSessionWorktreeBranch`, flattened for a path segment).
fn default_worktree_name() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let days = now / 86_400;
    let secs = now % 86_400;
    // Civil-from-days (Howard Hinnant's algorithm).
    // Days since epoch fits i64 for any representable system time.
    #[allow(clippy::cast_possible_wrap)]
    let shifted_days = days as i64 + 719_468;
    let era = shifted_days.div_euclid(146_097);
    let day_of_era = shifted_days.rem_euclid(146_097);
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
    let month = if month_prime < 10 {
        month_prime + 3
    } else {
        month_prime - 9
    };
    let year = if month <= 2 { year + 1 } else { year };
    let (hour, minute, second) = (secs / 3600, secs % 3600 / 60, secs % 60);
    format!("wt-{year:04}{month:02}{day:02}-{hour:02}{minute:02}{second:02}")
}

// ---------------------------------------------------------------------------
// Git plumbing
// ---------------------------------------------------------------------------

/// Run `git` in `cwd` and return trimmed stdout; a non-zero exit or spawn
/// failure becomes a user-facing message.
fn git(cwd: &Path, args: &[&str]) -> Result<String, String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(cwd)
        .args(args)
        .stdin(Stdio::null())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| format!("failed to run git: {e}"))?;
    if out.status.success() {
        Ok(String::from_utf8_lossy(&out.stdout).trim_end().to_string())
    } else {
        let err = String::from_utf8_lossy(&out.stderr);
        let err = err.trim();
        Err(if err.is_empty() {
            format!("git {} failed", args.first().unwrap_or(&""))
        } else {
            format!("git {}: {err}", args.first().unwrap_or(&""))
        })
    }
}

/// Repository root for `cwd`, or a user-facing error.
fn repo_root(cwd: &Path) -> Result<PathBuf, String> {
    let out = git(cwd, &["rev-parse", "--show-toplevel"])?;
    if out.is_empty() {
        return Err(format!("Not inside a git repository: {}", cwd.display()));
    }
    Ok(PathBuf::from(out))
}

/// `/wt list` body from `git worktree list --porcelain`.
fn list_worktrees(root: &Path) -> Result<String, String> {
    let raw = git(root, &["worktree", "list", "--porcelain"])?;
    let mut lines = Vec::new();
    let mut path = String::new();
    let mut head = String::new();
    let mut branch = String::new();
    let mut bare = false;
    let flush = |path: &mut String,
                 head: &mut String,
                 branch: &mut String,
                 bare: &mut bool,
                 lines: &mut Vec<String>| {
        if !path.is_empty() {
            let label = if *bare {
                "(bare)".to_string()
            } else if branch.is_empty() {
                format!("(detached HEAD {})", &head[..head.len().min(9)])
            } else {
                format!("[{branch}]")
            };
            lines.push(format!("{path}  {label}"));
        }
        path.clear();
        head.clear();
        branch.clear();
        *bare = false;
    };
    for line in raw.lines() {
        if let Some(p) = line.strip_prefix("worktree ") {
            flush(&mut path, &mut head, &mut branch, &mut bare, &mut lines);
            path = p.to_string();
        } else if let Some(h) = line.strip_prefix("HEAD ") {
            head = h.to_string();
        } else if let Some(b) = line.strip_prefix("branch ") {
            branch = b.trim_start_matches("refs/heads/").to_string();
        } else if line == "bare" {
            bare = true;
        }
    }
    flush(&mut path, &mut head, &mut branch, &mut bare, &mut lines);
    if lines.is_empty() {
        Ok("No worktrees found.".to_string())
    } else {
        Ok(lines.join("\n"))
    }
}

// ---------------------------------------------------------------------------
// Parallel runs (`/wt run`)
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct Slot {
    name: String,
    path: PathBuf,
    status: String, // running | done | failed
    detail: String,
    started: Instant,
    elapsed_secs: u64,
    last_line: String,
}

struct RunState {
    prompt: String,
    slots: Arc<Mutex<Vec<Slot>>>,
    finished: Arc<std::sync::atomic::AtomicBool>,
}

static RUN: Mutex<Option<RunState>> = Mutex::new(None);

fn format_slots(slots: &[Slot]) -> String {
    slots
        .iter()
        .map(|s| {
            let last = if s.last_line.is_empty() {
                "-"
            } else {
                s.last_line.as_str()
            };
            let detail = if s.detail.is_empty() {
                String::new()
            } else {
                format!(" ({})", s.detail)
            };
            format!(
                "{}  {}{}  {:>3}s  {}",
                s.name, s.status, detail, s.elapsed_secs, last
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Current-exe path with the Linux ` (deleted)` suffix (replaced binary) stripped.
fn current_binary() -> PathBuf {
    std::env::current_exe().map_or_else(
        |_| PathBuf::from("omt"),
        |p| {
            let s = p.to_string_lossy();
            match s.strip_suffix(" (deleted)") {
                Some(stripped) => PathBuf::from(stripped),
                None => p,
            }
        },
    )
}

/// Human summary of a worktree's changes: untracked files extracted from
/// `git status --short` output plus the `git diff --stat HEAD` body.
/// `"(no changes)"` when both are empty.
fn change_summary(status_short: &str, diff_stat: &str) -> String {
    let untracked: Vec<&str> = status_short
        .lines()
        .filter(|l| l.starts_with("?? "))
        .map(|l| l["?? ".len()..].trim())
        .collect();
    let mut out = String::new();
    if !untracked.is_empty() {
        out.push_str("Untracked:\n");
        for f in &untracked {
            let _ = writeln!(out, "  {f}");
        }
    }
    let stat = diff_stat.trim_end();
    while out.ends_with('\n') {
        out.pop();
    }
    if !stat.is_empty() {
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(stat);
    }
    if out.is_empty() {
        "(no changes)".into()
    } else {
        out
    }
}

/// Spawn one `omt -p --no-title <prompt>` child in `path` and track it in `slots[idx]`.
async fn spawn_child(
    binary: &Path,
    path: PathBuf,
    prompt: String,
    model: Option<String>,
    slots: Arc<Mutex<Vec<Slot>>>,
    idx: usize,
) {
    let mut cmd = tokio::process::Command::new(binary);
    cmd.args(["-p", "--no-title"]);
    // Children run on the parent session's model, not the configured default.
    if let Some(model) = model.as_deref() {
        cmd.args(["--model", model]);
    }
    cmd.arg(&prompt).current_dir(&path);
    cmd.stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    let child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            if let Ok(mut v) = slots.lock() {
                v[idx].status = "failed".into();
                v[idx].detail = format!("spawn: {e}");
            }
            return;
        }
    };
    let pid = child.id();
    if let Ok(mut v) = slots.lock() {
        if let Some(s) = v.get_mut(idx) {
            s.detail = pid.map(|p| format!("pid {p}")).unwrap_or_default();
        }
    }
    let mut child = child;
    let mut stdout = child.stdout.take();
    // Drain stdout, keeping the last non-empty line.
    let reader = {
        let slots = slots.clone();
        async move {
            use tokio::io::AsyncBufReadExt;
            let Some(out) = stdout.take() else { return };
            let mut lines = tokio::io::BufReader::new(out).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                let line = line.trim_end().to_string();
                if line.is_empty() {
                    continue;
                }
                if let Ok(mut v) = slots.lock() {
                    if let Some(s) = v.get_mut(idx) {
                        s.last_line = line;
                    }
                }
            }
        }
    };
    let (exit, ()) = tokio::join!(child.wait(), reader);
    if let Ok(mut v) = slots.lock() {
        if let Some(s) = v.get_mut(idx) {
            match &exit {
                Ok(code) if code.success() => s.status = "done".into(),
                Ok(code) => {
                    s.status = "failed".into();
                    s.detail = format!("exit {code}");
                }
                Err(e) => {
                    s.status = "failed".into();
                    s.detail = format!("wait: {e}");
                }
            }
            s.elapsed_secs = s.started.elapsed().as_secs();
        }
    }
}

// Validation, spawning and the status monitor share `slots`/`RUN` state and
// read best as one unit.
#[allow(clippy::too_many_lines)]
fn run_command(rt: &Arc<Runtime>, root: &Path, n: usize, prompt: &str) -> Outcome {
    if n == 0 || prompt.is_empty() {
        rt.send(
            json!({"type": "command_output", "text": "/wt run: need a worker count and a prompt."}),
        );
        rt.send(json!({"type": "command_output", "text": USAGE}));
        return Outcome::Handled {
            agent_invoked: false,
        };
    }
    if RUN
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .is_some()
    {
        rt.send(json!({"type": "command_output", "text": "/wt run: a run is already in progress (see /wt status)."}));
        return Outcome::Handled {
            agent_invoked: false,
        };
    }
    let binary = current_binary();
    let mut slots = Vec::new();
    let mut paths = Vec::new();
    for i in 1..=n {
        let name = format!("run-{i}");
        let Some(path) = worktree_path(root, &name) else {
            rt.send(json!({"type": "command_output", "text": format!("/wt run: cannot compute worktree path for '{name}'.")}));
            return Outcome::Handled {
                agent_invoked: false,
            };
        };
        if let Err(e) = git(
            root,
            &[
                "worktree",
                "add",
                &path.display().to_string(),
                "-b",
                &format!("{BRANCH_PREFIX}/{name}"),
            ],
        ) {
            rt.send(json!({"type": "command_output", "text": format!("/wt run: worktree '{name}' failed: {e}")}));
            return Outcome::Handled {
                agent_invoked: false,
            };
        }
        paths.push(path.clone());
        slots.push(Slot {
            name,
            path,
            status: "running".into(),
            detail: String::new(),
            started: Instant::now(),
            elapsed_secs: 0,
            last_line: String::new(),
        });
    }
    let slots = Arc::new(Mutex::new(slots));
    let finished = Arc::new(std::sync::atomic::AtomicBool::new(false));
    *RUN.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(RunState {
        prompt: prompt.to_string(),
        slots: slots.clone(),
        finished: finished.clone(),
    });
    rt.send(json!({"type": "command_output", "text": format!("Started {n} agent(s) on branches {BRANCH_PREFIX}/run-1..{BRANCH_PREFIX}/run-{n}; status updates follow.")}));
    let info = rt.model_info();
    let model = match (
        info.get("provider").and_then(serde_json::Value::as_str),
        info.get("id").and_then(serde_json::Value::as_str),
    ) {
        (Some(p), Some(id)) => Some(format!("{p}/{id}")),
        _ => None,
    };
    for (idx, path) in paths.into_iter().enumerate() {
        let binary = binary.clone();
        let model = model.clone();
        let prompt = prompt.to_string();
        let slots = slots.clone();
        tokio::spawn(async move {
            spawn_child(&binary, path, prompt, model, slots, idx).await;
        });
    }
    // Monitor: emit combined status until every child settles.
    let rt = rt.clone();
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
            let (snapshot, all_done);
            {
                let mut v = slots
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                for s in v.iter_mut() {
                    if s.status == "running" {
                        s.elapsed_secs = s.started.elapsed().as_secs();
                    }
                }
                snapshot = format_slots(&v);
                all_done = v.iter().all(|s| s.status != "running");
            }
            rt.send(
                json!({"type": "command_output", "text": format!("[/wt run status]\n{snapshot}")}),
            );
            if all_done {
                break;
            }
        }
        // Final per-worktree diff stats.
        let mut report = String::new();
        {
            let v = slots
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            for s in v.iter() {
                let status = git(&s.path, &["status", "--short"])
                    .unwrap_or_else(|e| format!("(git status failed: {e})"));
                let stat = git(&s.path, &["diff", "--stat", "HEAD"])
                    .unwrap_or_else(|e| format!("(diff failed: {e})"));
                let _ = write!(
                    report,
                    "--- {} ({})\n{}\n",
                    s.name,
                    s.status,
                    change_summary(&status, &stat)
                );
            }
        }
        rt.send(json!({"type": "command_output", "text": format!("[/wt run finished]\n{report}")}));
        finished.store(true, std::sync::atomic::Ordering::Relaxed);
    });
    Outcome::Handled {
        agent_invoked: false,
    }
}

fn status_command(rt: &Arc<Runtime>) -> Outcome {
    let guard = RUN
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    match guard.as_ref() {
        None => {
            rt.send(
                json!({"type": "command_output", "text": "/wt status: no /wt run in progress."}),
            );
        }
        Some(state) => {
            let mut v = state
                .slots
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            for s in v.iter_mut() {
                if s.status == "running" {
                    s.elapsed_secs = s.started.elapsed().as_secs();
                }
            }
            let body = format_slots(&v);
            let tail = if state.finished.load(std::sync::atomic::Ordering::Relaxed) {
                "\n(run finished)"
            } else {
                ""
            };
            rt.send(json!({"type": "command_output", "text": format!("[/wt run: {}]\n{body}{tail}", state.prompt)}));
        }
    }
    Outcome::Handled {
        agent_invoked: false,
    }
}

// ---------------------------------------------------------------------------
// Slash entry point
// ---------------------------------------------------------------------------

/// Slash-command entry point.
// Flat subcommand dispatch; splitting the arms into helpers would hide the
// shared control flow without making it shorter.
#[allow(clippy::too_many_lines)]
pub async fn command(rt: &Arc<Runtime>, args: &str) -> Outcome {
    let args = args.trim();
    let (verb, rest) = match args.split_once(char::is_whitespace) {
        Some((v, r)) => (v, r.trim()),
        None => (args, ""),
    };
    let output = |rt: &Arc<Runtime>, text: String| {
        rt.send(json!({"type": "command_output", "text": text}));
    };
    match verb {
        "new" | "" => {
            if verb.is_empty() && !args.is_empty() {
                output(rt, USAGE.to_string());
                return Outcome::Handled {
                    agent_invoked: false,
                };
            }
            let raw_name = if rest.is_empty() {
                default_worktree_name()
            } else {
                rest.to_string()
            };
            let Some(name) = sanitize_name(&raw_name) else {
                output(rt, format!("/wt new: invalid worktree name '{raw_name}'."));
                return Outcome::Handled {
                    agent_invoked: false,
                };
            };
            let Ok(root) = repo_root(&rt.cwd) else {
                output(
                    rt,
                    format!("Not inside a git repository: {}", rt.cwd.display()),
                );
                return Outcome::Handled {
                    agent_invoked: false,
                };
            };
            let Some(path) = worktree_path(&root, &name) else {
                output(
                    rt,
                    format!("/wt new: cannot compute worktree path for '{name}'."),
                );
                return Outcome::Handled {
                    agent_invoked: false,
                };
            };
            let branch = format!("{BRANCH_PREFIX}/{name}");
            if let Err(e) = git(
                &root,
                &[
                    "worktree",
                    "add",
                    &path.display().to_string(),
                    "-b",
                    &branch,
                ],
            ) {
                output(rt, format!("/wt new failed: {e}"));
                return Outcome::Handled {
                    agent_invoked: false,
                };
            }
            match rt.new_session(Some(&path)).await {
                Ok(()) => output(
                    rt,
                    format!(
                        "Created worktree {} on branch {branch}; started a new session there.",
                        path.display()
                    ),
                ),
                Err(e) => output(
                    rt,
                    format!(
                        "Created worktree {} on branch {branch}, but the new session failed: {e}",
                        path.display()
                    ),
                ),
            }
            Outcome::Handled {
                agent_invoked: false,
            }
        }
        "list" => {
            match repo_root(&rt.cwd).and_then(|root| list_worktrees(&root)) {
                Ok(body) => output(rt, body),
                Err(e) => output(rt, e),
            }
            Outcome::Handled {
                agent_invoked: false,
            }
        }
        "remove" => {
            let (name_arg, force) = match rest.strip_suffix("--force") {
                Some(r) => (r.trim(), true),
                None => (rest, false),
            };
            let Some(name) = sanitize_name(name_arg) else {
                output(
                    rt,
                    format!("/wt remove: invalid worktree name '{name_arg}'."),
                );
                return Outcome::Handled {
                    agent_invoked: false,
                };
            };
            let root = match repo_root(&rt.cwd) {
                Ok(r) => r,
                Err(e) => {
                    output(rt, e);
                    return Outcome::Handled {
                        agent_invoked: false,
                    };
                }
            };
            let Some(path) = worktree_path(&root, &name) else {
                output(
                    rt,
                    format!("/wt remove: cannot compute worktree path for '{name}'."),
                );
                return Outcome::Handled {
                    agent_invoked: false,
                };
            };
            if !path.is_dir() {
                output(
                    rt,
                    format!("/wt remove: worktree {} does not exist.", path.display()),
                );
                return Outcome::Handled {
                    agent_invoked: false,
                };
            }
            if !force {
                match git(&path, &["status", "--porcelain"]) {
                    Ok(dirty) if !dirty.is_empty() => {
                        output(
                            rt,
                            format!(
                                "/wt remove: {} has uncommitted changes; pass --force to discard them.",
                                path.display()
                            ),
                        );
                        return Outcome::Handled {
                            agent_invoked: false,
                        };
                    }
                    Ok(_) => {}
                    Err(e) => {
                        output(rt, format!("/wt remove: {e}"));
                        return Outcome::Handled {
                            agent_invoked: false,
                        };
                    }
                }
            }
            let mut git_args = vec!["worktree", "remove"];
            if force {
                git_args.push("--force");
            }
            let path_arg = path.display().to_string();
            git_args.push(&path_arg);
            match git(&root, &git_args) {
                Ok(_) => output(
                    rt,
                    format!(
                        "Removed worktree {} (branch {BRANCH_PREFIX}/{name} kept).",
                        path.display()
                    ),
                ),
                Err(e) => output(rt, format!("/wt remove failed: {e}")),
            }
            Outcome::Handled {
                agent_invoked: false,
            }
        }
        "merge" => {
            let Some(name) = sanitize_name(rest) else {
                output(rt, format!("/wt merge: invalid worktree name '{rest}'."));
                return Outcome::Handled {
                    agent_invoked: false,
                };
            };
            let branch = format!("{BRANCH_PREFIX}/{name}");
            let root = match repo_root(&rt.cwd) {
                Ok(r) => r,
                Err(e) => {
                    output(rt, e);
                    return Outcome::Handled {
                        agent_invoked: false,
                    };
                }
            };
            // Commit pending changes in the worktree first; otherwise the
            // merge brings nothing (local edits never reach the branch).
            let mut prelude = String::new();
            if let Some(wt) = worktree_path(&root, &name) {
                if wt.is_dir() {
                    let dirty = git(&wt, &["status", "--porcelain"]).unwrap_or_default();
                    if !dirty.trim().is_empty() {
                        let msg = format!("omt: {name}");
                        if let Err(e) =
                            git(&wt, &["add", "-A"]).and_then(|_| git(&wt, &["commit", "-m", &msg]))
                        {
                            output(
                                rt,
                                format!(
                                    "/wt merge: failed to commit pending changes in {}: {e}",
                                    wt.display()
                                ),
                            );
                            return Outcome::Handled {
                                agent_invoked: false,
                            };
                        }
                        prelude = format!("Committed pending changes in {}.\n", wt.display());
                    }
                }
            }
            if let Ok(out) = git(&root, &["merge", "--no-edit", &branch]) {
                output(
                    rt,
                    if out.is_empty() {
                        format!("{prelude}Merged {branch}.")
                    } else {
                        format!("{prelude}{out}")
                    },
                );
            } else {
                // Report conflicting files from the index.
                let conflicts =
                    git(&root, &["diff", "--name-only", "--diff-filter=U"]).unwrap_or_default();
                let body = if conflicts.is_empty() {
                    String::new()
                } else {
                    format!(
                        "Conflicting files:\n{}",
                        conflicts
                            .lines()
                            .map(|l| format!("  {l}"))
                            .collect::<Vec<_>>()
                            .join("\n")
                    )
                };
                output(
                    rt,
                    format!("Merge of {branch} has conflicts; resolve them, then `git add` + `git commit`.\n{body}"),
                );
            }
            Outcome::Handled {
                agent_invoked: false,
            }
        }
        "run" => {
            let (count, prompt) = match rest.split_once(char::is_whitespace) {
                Some((c, p)) => (c, p.trim()),
                None => (rest, ""),
            };
            let Ok(n) = count.parse::<usize>() else {
                output(rt, format!("/wt run: invalid worker count '{count}'."));
                rt.send(json!({"type": "command_output", "text": USAGE}));
                return Outcome::Handled {
                    agent_invoked: false,
                };
            };
            let root = match repo_root(&rt.cwd) {
                Ok(r) => r,
                Err(e) => {
                    output(rt, e);
                    return Outcome::Handled {
                        agent_invoked: false,
                    };
                }
            };
            run_command(rt, &root, n, prompt)
        }
        "status" => status_command(rt),
        _ => {
            output(rt, format!("/wt: unknown subcommand '{verb}'."));
            rt.send(json!({"type": "command_output", "text": USAGE}));
            Outcome::Handled {
                agent_invoked: false,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_keeps_safe_characters() {
        assert_eq!(sanitize_name("feature-1").as_deref(), Some("feature-1"));
        assert_eq!(sanitize_name("a_b.c").as_deref(), Some("a_b.c"));
        assert_eq!(sanitize_name("Fix Bug!!").as_deref(), Some("Fix-Bug"));
    }

    #[test]
    fn sanitize_collapses_runs_and_trims() {
        assert_eq!(sanitize_name("  a   b  ").as_deref(), Some("a-b"));
        assert_eq!(sanitize_name("--x--").as_deref(), Some("x"));
        assert_eq!(sanitize_name("a..b").as_deref(), Some("a-b"));
    }

    #[test]
    fn sanitize_rejects_unusable() {
        assert!(sanitize_name("").is_none());
        assert!(sanitize_name("///").is_none());
        assert!(sanitize_name("..").is_none());
        assert!(sanitize_name("stuff.lock").is_none());
    }

    #[test]
    fn worktree_path_uses_repo_parent() {
        let root = Path::new("/repo/checkout");
        assert_eq!(
            worktree_path(root, "run-1"),
            Some(PathBuf::from("/repo/checkout-wt-run-1"))
        );
        assert_eq!(worktree_path(Path::new("/"), "x"), None);
        assert_eq!(worktree_path(Path::new("relative"), "x"), None);
    }

    #[test]
    fn summary_lists_untracked_and_diff_stat() {
        assert_eq!(change_summary("", ""), "(no changes)");
        assert_eq!(change_summary("", " file.rs | 1 +\n"), " file.rs | 1 +");
        assert_eq!(
            change_summary("?? hello.txt\n M src/a.rs\n", ""),
            "Untracked:\n  hello.txt"
        );
        let both = change_summary(
            "?? hello.txt\n?? dir/\n",
            " hello.txt | 1 +\n 1 file changed, 1 insertion(+)\n",
        );
        assert!(both.starts_with("Untracked:\n  hello.txt\n  dir/\n"));
        assert!(both.contains("1 file changed, 1 insertion(+)"));
        // Tracked-only modifications are not listed as untracked.
        let tracked = change_summary(" M src/a.rs\n", "");
        assert_eq!(tracked, "(no changes)");
    }

    #[test]
    fn default_name_has_timestamp_shape() {
        let name = default_worktree_name();
        // wt-YYYYMMDD-HHMMSS
        assert_eq!(name.len(), 18);
        assert!(name.starts_with("wt-"));
        assert_eq!(name.as_bytes()[11], b'-');
    }
}
