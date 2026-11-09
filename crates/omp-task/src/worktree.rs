//! Git worktrees for `task` children and the parent's merge helpers
//! (`SPEC.md` batch 2, changes 1–2).
//!
//! A writing-agent child with `worktree: true` does not edit the parent's
//! checkout: the parent creates a fresh `git worktree add` on branch
//! `agent/<task-id>` from its own HEAD, the child works there, commits with
//! the `Agent:`/`Envelope:`/`Verified-by:` trailers, and at settle the
//! parent summarizes branch/commits/diffstat and auto-commits leftovers as
//! `wip(<task-id>)` so nothing is lost. The worktree and branch live until
//! `omt agent-merge <id>` merges them (or `--discard` throws them away).
//!
//! All git access shells out to the `git` binary (no git library in the
//! workspace); every call is scoped to one repository.

use std::path::{Path, PathBuf};
use std::process::Command;

use thiserror::Error;

/// Branch prefix for worktree children: `agent/<task-id>`.
pub const BRANCH_PREFIX: &str = "agent/";
/// Env var overriding the directory that holds per-task worktrees. Default:
/// `<repo>/../<repo-name>-wt` (e.g. `/x/omp-rs` -> `/x/omp-rs-wt/<id>`).
pub const WORKTREE_DIR_ENV: &str = "OMP_TASK_WORKTREE_DIR";

/// What can go wrong in the worktree/merge lifecycle.
#[derive(Debug, Error)]
pub enum WorktreeError {
    /// `worktree: true` but the parent cwd is not inside a git repository.
    #[error(
        "worktree: `{0}` is not inside a git repository; `worktree: true` needs one (init or clone a repo first)"
    )]
    NotARepo(PathBuf),
    /// A git invocation failed.
    #[error("worktree: git {0} failed: {1}")]
    Git(String, String),
    /// Miscellaneous lifecycle error (branch exists, path taken, ...).
    #[error("worktree: {0}")]
    Message(String),
}

/// Run `git` in `repo`, returning stdout on success.
fn git(repo: &Path, args: &[&str]) -> Result<String, WorktreeError> {
    let output = Command::new("git")
        .args(args)
        .current_dir(repo)
        .env("GIT_TERMINAL_PROMPT", "0")
        .output()
        .map_err(|e| WorktreeError::Message(format!("cannot spawn `git`: {e}")))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(WorktreeError::Git(
            args.join(" "),
            stderr.trim().to_string(),
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// Whether `git` in `cwd` addresses an existing ref.
fn ref_exists(repo: &Path, r#ref: &str) -> bool {
    Command::new("git")
        .args(["show-ref", "--verify", "--quiet", r#ref])
        .current_dir(repo)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

/// The repository root containing `cwd` (`git rev-parse --show-toplevel`).
///
/// # Errors
/// [`WorktreeError::NotARepo`] when `cwd` is not inside a work tree.
pub fn repo_root(cwd: &Path) -> Result<PathBuf, WorktreeError> {
    match git(cwd, &["rev-parse", "--show-toplevel"]) {
        Ok(out) => Ok(PathBuf::from(out.trim())),
        Err(e) => Err(match e {
            WorktreeError::Git(_, _) => WorktreeError::NotARepo(cwd.to_path_buf()),
            other => other,
        }),
    }
}

/// The directory holding per-task worktrees: [`WORKTREE_DIR_ENV`] when set,
/// else `<repo>/../<repo-name>-wt`.
#[must_use]
pub fn worktree_dir(repo_root: &Path) -> PathBuf {
    if let Some(dir) = std::env::var_os(WORKTREE_DIR_ENV) {
        return PathBuf::from(dir);
    }
    let name = repo_root
        .file_name()
        .map_or_else(|| "repo".to_string(), |n| n.to_string_lossy().into_owned());
    match repo_root.parent() {
        Some(parent) => parent.join(format!("{name}-wt")),
        None => PathBuf::from(format!("{name}-wt")),
    }
}

/// One child's worktree: the checkout, its branch, and the base commit it
/// was created from.
#[derive(Debug, Clone)]
pub struct Worktree {
    /// The task id the branch is named after (`agent/<id>`).
    pub id: String,
    /// The parent's repository root.
    pub repo_root: PathBuf,
    /// The worktree checkout path.
    pub path: PathBuf,
    /// The child's branch (`agent/<id>`).
    pub branch: String,
    /// The commit the branch was created from (the parent's HEAD at spawn).
    pub base: String,
}

/// One commit on the child's branch: abbreviated sha + subject.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct CommitInfo {
    pub sha: String,
    pub subject: String,
}

/// What the parent reports when a worktree child settles.
#[derive(Debug, Clone, serde::Serialize)]
pub struct WorktreeSummary {
    pub branch: String,
    pub base: String,
    pub path: PathBuf,
    /// Whether the working tree was clean when the child settled (`false`
    /// means leftovers were auto-committed as `wip(<id>)`).
    pub clean: bool,
    pub leftovers_auto_committed: bool,
    /// Commits the child made (sha + subject), base..tip.
    pub commits: Vec<CommitInfo>,
    /// `git diff --shortstat` of base..tip ("no changes" when empty).
    pub diffstat: String,
}

impl Worktree {
    /// Create the branch `agent/<id>` at the parent's HEAD and check it out
    /// as a fresh worktree under [`worktree_dir`].
    ///
    /// # Errors
    /// Not-a-repo callers must pass [`repo_root`] output; the branch already
    /// exists; or `git worktree add` failed.
    pub fn create(repo_root: &Path, id: &str) -> Result<Self, WorktreeError> {
        let branch = format!("{BRANCH_PREFIX}{id}");
        if ref_exists(repo_root, &format!("refs/heads/{branch}")) {
            return Err(WorktreeError::Message(format!(
                "branch `{branch}` already exists; merge or discard it first \
                 (`omt agent-merge {id}` or `omt agent-merge {id} --discard`)"
            )));
        }
        let base = git(repo_root, &["rev-parse", "HEAD"])?.trim().to_string();
        let path = worktree_dir(repo_root).join(id);
        if path.exists() {
            return Err(WorktreeError::Message(format!(
                "worktree path {} already exists; remove it or pick another id",
                path.display()
            )));
        }
        git(
            repo_root,
            &[
                "worktree",
                "add",
                path.to_string_lossy().as_ref(),
                "-b",
                &branch,
                &base,
            ],
        )?;
        Ok(Self {
            id: id.to_string(),
            repo_root: repo_root.to_path_buf(),
            path,
            branch,
            base,
        })
    }

    /// Map a path inside the parent's checkout to the same relative path in
    /// this worktree (the child's cwd).
    #[must_use]
    pub fn map_cwd(&self, parent_cwd: &Path) -> PathBuf {
        match parent_cwd.strip_prefix(&self.repo_root) {
            Ok(rel) if rel.as_os_str().is_empty() => self.path.clone(),
            Ok(rel) => self.path.join(rel),
            Err(_) => self.path.clone(),
        }
    }

    /// The footer appended to the child's prompt: where it runs and the
    /// commit/trailer contract.
    #[must_use]
    pub fn prompt_footer(&self) -> String {
        format!(
            "\n\n## Git worktree\n\
             You run in a fresh git worktree at `{}` (repo `{}`), on branch `{}` \
             created from `{}`. Do all work and all commits here; never switch \
             branches, never create other branches, and commit your work on this \
             branch.\n\
             Every commit must carry these trailers:\n\
             Agent: {id}\n\
             Envelope: agent://{id}\n\
             Verified-by: <test command or evidence path>\n\
             Uncommitted leftovers are auto-committed as `wip({id})` when you \
             settle, and the parent merges this branch with `omt agent-merge {id}`.",
            self.path.display(),
            self.repo_root.display(),
            self.branch,
            self.base,
            id = self.id,
        )
    }

    /// `git status --porcelain` (empty = clean), tolerating git failure.
    fn is_clean(&self) -> bool {
        git(&self.path, &["status", "--porcelain"])
            .map(|out| out.trim().is_empty())
            .unwrap_or(false)
    }

    /// Commit every uncommitted change as `wip(<id>)` so nothing is lost.
    /// Returns whether a wip commit was created.
    fn wip_leftovers(&self) -> bool {
        if self.is_clean() {
            return false;
        }
        let _ = git(&self.path, &["add", "-A"]);
        git(
            &self.path,
            &[
                "commit",
                "--no-verify",
                "-m",
                &format!("wip({}): uncommitted leftovers", self.id),
            ],
        )
        .is_ok()
    }

    /// Settle-time summary for the parent's delivery: auto-commit leftovers,
    /// then list branch commits (sha + subject) and the base..tip diffstat.
    #[must_use]
    pub fn settle_summary(&self) -> WorktreeSummary {
        let clean = self.is_clean();
        let leftovers_auto_committed = !clean && self.wip_leftovers();
        let commits = git(
            &self.path,
            &["log", "--format=%h%x1f%s", &format!("{}..HEAD", self.base)],
        )
        .map(|out| {
            out.lines()
                .filter_map(|line| {
                    let (sha, subject) = line.split_once('\x1f')?;
                    Some(CommitInfo {
                        sha: sha.trim().to_string(),
                        subject: subject.trim().to_string(),
                    })
                })
                .collect()
        })
        .unwrap_or_default();
        let diffstat = git(
            &self.path,
            &["diff", "--shortstat", &format!("{}..HEAD", self.base)],
        )
        .map(|out| {
            let trimmed = out.trim();
            if trimmed.is_empty() {
                "no changes".to_string()
            } else {
                trimmed.to_string()
            }
        })
        .unwrap_or_else(|_| "unknown".to_string());
        WorktreeSummary {
            branch: self.branch.clone(),
            base: self.base.clone(),
            path: self.path.clone(),
            clean,
            leftovers_auto_committed,
            commits,
            diffstat,
        }
    }

    /// Remove the worktree checkout (the branch survives until merged or
    /// discarded — see [`merge_agent`]/[`discard_agent`]).
    ///
    /// # Errors
    /// `git worktree remove` failed while the path still exists.
    pub fn remove(&self) -> Result<(), WorktreeError> {
        match git(
            &self.repo_root,
            &[
                "worktree",
                "remove",
                "--force",
                self.path.to_string_lossy().as_ref(),
            ],
        ) {
            Ok(_) => Ok(()),
            // Already gone (removed by hand): nothing to do.
            Err(e) if !self.path.exists() => {
                let _ = e;
                Ok(())
            }
            Err(e) => Err(e),
        }
    }
}

/// Outcome of one [`merge_agent`] / [`discard_agent`] run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MergeStatus {
    /// Merged with a merge commit (or a squash commit) — carries the sha.
    Merged { commit: String },
    /// The merge conflicted; the repo was aborted back to its pre-merge
    /// state and these files conflicted.
    Conflict { files: Vec<String> },
    /// `--discard`: worktree and branch removed without merging.
    Discarded,
}

/// One branch's merge outcome plus the cleanup that ran.
#[derive(Debug, Clone)]
pub struct MergeOutcome {
    pub id: String,
    pub branch: String,
    pub status: MergeStatus,
    pub squash: bool,
}

/// How [`merge_agent`] merges.
#[derive(Debug, Clone, Copy, Default)]
pub struct MergeOptions {
    /// Squash the branch into one commit carrying the aggregated trailers.
    pub squash: bool,
    /// Refuse to merge unless the branch tip carries a passing review note
    /// (`refs/notes/review`, see [`crate::notes`]).
    pub require_review: bool,
}

/// Merge the branch `refs/heads/<name>`; on conflict abort and report.
fn merge_branch(
    repo: &Path,
    branch: &str,
    message: &str,
    squash: bool,
) -> Result<MergeStatus, WorktreeError> {
    let merge_args: Vec<&str> = if squash {
        vec!["merge", "--squash", branch]
    } else {
        vec!["merge", "--no-edit", "--no-ff", "-m", message, branch]
    };
    if git(repo, &merge_args).is_ok() {
        if squash {
            // Nothing staged means the branch was already fully contained.
            let staged = git(repo, &["diff", "--cached", "--quiet"]).is_err();
            if !staged {
                return Ok(MergeStatus::Merged {
                    commit: git(repo, &["rev-parse", "HEAD"])?.trim().to_string(),
                });
            }
            git(repo, &["commit", "--no-verify", "--no-edit", "-m", message])?;
        }
        return Ok(MergeStatus::Merged {
            commit: git(repo, &["rev-parse", "HEAD"])?.trim().to_string(),
        });
    }
    // Conflict (or other merge failure): capture the conflicting files while
    // the merge is still in flight, then abort — never leave the repo
    // half-merged.
    let files: Vec<String> = git(repo, &["diff", "--name-only", "--diff-filter=U"])
        .map(|out| out.lines().map(str::trim).map(str::to_string).collect())
        .unwrap_or_default();
    let _ = git(repo, &["merge", "--abort"]);
    Ok(MergeStatus::Conflict { files })
}

/// Collect `Agent:`/`Envelope:`/`Verified-by:` trailers across the branch's
/// commits (merge-base..branch), unique, order-preserving; keyed lowercase.
///
/// The scan reads full commit bodies (`%B`), not `%(trailers:only)`: git's
/// trailer parser only recognizes the LAST contiguous trailer block, so a
/// commit that separates trailers with blank lines (e.g. several `git commit
/// -m` flags) would silently drop all but the final paragraph. Model-written
/// commits routinely do that; the three known keys are unambiguous.
fn branch_trailers(repo: &Path, branch: &str) -> Vec<(String, Vec<String>)> {
    let Ok(base) = git(repo, &["merge-base", "HEAD", branch]) else {
        return Vec::new();
    };
    let Ok(log) = git(
        repo,
        &[
            "log",
            "--format=%B%x00",
            &format!("{}..{branch}", base.trim()),
        ],
    ) else {
        return Vec::new();
    };
    let wanted = ["agent", "envelope", "verified-by"];
    let mut out: Vec<(String, Vec<String>)> = Vec::new();
    for line in log.lines() {
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let key = key.trim().to_lowercase();
        if !wanted.contains(&key.as_str()) {
            continue;
        }
        for value in value.split(',') {
            let value = value.trim();
            if value.is_empty() {
                continue;
            }
            let entry = out.iter_mut().find(|(k, _)| *k == key);
            match entry {
                Some((_, values)) => {
                    if !values.iter().any(|v| v == value) {
                        values.push(value.to_string());
                    }
                }
                None => out.push((key.clone(), vec![value.to_string()])),
            }
        }
    }
    out
}

/// The merge-commit message: subject plus the aggregated trailers.
fn merge_message(repo: &Path, branch: &str, id: &str) -> Result<String, WorktreeError> {
    let current = git(repo, &["branch", "--show-current"])?.trim().to_string();
    let mut trailers = branch_trailers(repo, branch);
    if !trailers.iter().any(|(k, _)| k == "agent") {
        trailers.insert(0, ("agent".to_string(), vec![id.to_string()]));
    }
    let mut message = format!("merge {branch} into {current}");
    for (key, values) in trailers {
        // Canonical SPEC trailer spelling (git trailers are case-insensitive,
        // but the byte form should match the shared rules).
        let key = match key.as_str() {
            "agent" => "Agent",
            "envelope" => "Envelope",
            "verified-by" => "Verified-by",
            other => other,
        };
        message.push_str(&format!("\n{key}: {}", values.join(", ")));
    }
    Ok(message)
}

/// Worktree registered for `branch` in `repo`, if any.
#[must_use]
pub fn find_worktree(repo: &Path, branch: &str) -> Option<PathBuf> {
    let out = git(repo, &["worktree", "list", "--porcelain"]).ok()?;
    let mut path: Option<PathBuf> = None;
    for line in out.lines() {
        if let Some(rest) = line.strip_prefix("worktree ") {
            // A new worktree block starts; its `branch` line follows.
            path = Some(PathBuf::from(rest));
        } else if let Some(rest) = line.strip_prefix("branch ") {
            if rest.trim() == format!("refs/heads/{branch}") {
                return path;
            }
        } else if line == "detached" || line.starts_with("HEAD ") {
            // Not a branch checkout; keep looking.
        } else if line.is_empty() {
            path = None;
        }
    }
    None
}

/// All `agent/*` branches in `repo` (short names), for `--all`.
#[must_use]
pub fn agent_branches(repo: &Path) -> Vec<String> {
    git(
        repo,
        &[
            "for-each-ref",
            "refs/heads/agent/",
            "--format=%(refname:short)",
        ],
    )
    .map(|out| {
        out.lines()
            .map(str::trim)
            .filter(|b| !b.is_empty())
            .map(str::to_string)
            .collect()
    })
    .unwrap_or_default()
}

/// Merge `agent/<id>` into the current branch, then remove the worktree and
/// delete the branch. A conflict aborts the merge (repo untouched) and
/// reports the conflicting files; nothing is cleaned up so the parent can
/// retry or discard.
///
/// # Errors
/// No such branch, a failing `git merge`/`--abort`, or `--require-review`
/// without a passing note on the branch tip.
pub fn merge_agent(
    repo: &Path,
    id: &str,
    options: &MergeOptions,
) -> Result<MergeOutcome, WorktreeError> {
    let branch = format!("{BRANCH_PREFIX}{id}");
    if !ref_exists(repo, &format!("refs/heads/{branch}")) {
        return Err(WorktreeError::Message(format!(
            "no agent branch `{branch}` in {}",
            repo.display()
        )));
    }
    if options.require_review {
        let tip = git(repo, &["rev-parse", &branch])?.trim().to_string();
        if !crate::notes::has_passing_review(repo, &tip) {
            return Err(WorktreeError::Message(format!(
                "branch tip {tip} has no passing review note; run \
                 `omt review note {tip} --verdict pass --by <agent>` (or drop \
                 --require-review)"
            )));
        }
    }
    // Commit leftovers before merging so nothing is lost.
    if let Some(path) = find_worktree(repo, &branch) {
        let wt = Worktree {
            id: id.to_string(),
            repo_root: repo.to_path_buf(),
            path,
            branch: branch.clone(),
            base: String::new(),
        };
        let _ = wt.wip_leftovers();
    }
    let message = merge_message(repo, &branch, id)?;
    let status = merge_branch(repo, &branch, &message, options.squash)?;
    if matches!(status, MergeStatus::Conflict { .. }) {
        return Ok(MergeOutcome {
            id: id.to_string(),
            branch,
            status,
            squash: options.squash,
        });
    }
    // Success: the worktree and branch are no longer needed.
    if let Some(path) = find_worktree(repo, &branch) {
        let wt = Worktree {
            id: id.to_string(),
            repo_root: repo.to_path_buf(),
            path,
            branch: branch.clone(),
            base: String::new(),
        };
        let _ = wt.remove();
    }
    let _ = git(repo, &["branch", "-D", &branch]);
    Ok(MergeOutcome {
        id: id.to_string(),
        branch,
        status,
        squash: options.squash,
    })
}

/// Throw away `agent/<id>` without merging: remove the worktree and delete
/// the branch.
///
/// # Errors
/// No such branch, or worktree removal failed.
pub fn discard_agent(repo: &Path, id: &str) -> Result<MergeOutcome, WorktreeError> {
    let branch = format!("{BRANCH_PREFIX}{id}");
    if !ref_exists(repo, &format!("refs/heads/{branch}")) {
        return Err(WorktreeError::Message(format!(
            "no agent branch `{branch}` in {}",
            repo.display()
        )));
    }
    if let Some(path) = find_worktree(repo, &branch) {
        let wt = Worktree {
            id: id.to_string(),
            repo_root: repo.to_path_buf(),
            path,
            branch: branch.clone(),
            base: String::new(),
        };
        wt.remove()?;
    }
    git(repo, &["branch", "-D", &branch])?;
    Ok(MergeOutcome {
        id: id.to_string(),
        branch,
        status: MergeStatus::Discarded,
        squash: false,
    })
}

#[cfg(test)]
mod tests;
