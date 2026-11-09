//! Unit tests for the worktree lifecycle and the parent merge helpers.
//! Every test builds a throwaway git repo under a temp dir.

use parking_lot::{Mutex, MutexGuard};
use std::path::{Path, PathBuf};
use std::process::Command;

use super::*;

/// Env-var tests mutate `OMP_TASK_WORKTREE_DIR` process-wide; serialize the
/// whole module.
static LOCK: Mutex<()> = Mutex::new(());

fn lock() -> MutexGuard<'static, ()> {
    LOCK.lock()
}

fn git_in(dir: &Path, args: &[&str]) -> String {
    let out = Command::new("git")
        .args(args)
        .current_dir(dir)
        .output()
        .unwrap_or_else(|e| panic!("spawn git: {e}"));
    assert!(
        out.status.success(),
        "git {} failed: {}",
        args.join(" "),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

struct Repo {
    _dir: tempfile::TempDir,
    root: PathBuf,
}

impl Repo {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("repo");
        std::fs::create_dir(&root).unwrap();
        git_in(&root, &["init", "-b", "main"]);
        git_in(&root, &["config", "user.name", "omt-test"]);
        git_in(&root, &["config", "user.email", "omt-test@example.com"]);
        std::fs::write(root.join("file.txt"), "one\n").unwrap();
        git_in(&root, &["add", "-A"]);
        git_in(&root, &["commit", "-m", "initial"]);
        Self { _dir: dir, root }
    }

    /// The main-branch tip sha.
    fn head(&self) -> String {
        git_in(&self.root, &["rev-parse", "HEAD"])
            .trim()
            .to_string()
    }

    /// A commit in `dir` (the worktree checkout) writing `file`.
    fn commit_in(&self, dir: &Path, file: &str, content: &str, message: &str) {
        std::fs::write(dir.join(file), content).unwrap();
        git_in(dir, &["add", "-A"]);
        git_in(dir, &["commit", "--no-verify", "-m", message]);
    }
}

#[test]
fn repo_root_errors_outside_a_repo() {
    let _g = lock();
    let dir = tempfile::tempdir().unwrap();
    let err = repo_root(dir.path()).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("not inside a git repository"),
        "clear error for worktree:true outside a repo: {msg}"
    );
}

#[test]
fn create_maps_cwd_and_pins_branch_and_base() {
    let _g = lock();
    let repo = Repo::new();
    let wt = Worktree::create(&repo.root, "alpha").unwrap();
    assert_eq!(wt.branch, "agent/alpha");
    assert_eq!(wt.base, repo.head());
    // Default location: <repo>/../<repo-name>-wt/<id>.
    let expected = repo.root.parent().unwrap().join("repo-wt").join("alpha");
    assert_eq!(wt.path, expected);
    // The checkout is on the new branch at the base commit.
    assert_eq!(
        git_in(&wt.path, &["branch", "--show-current"]).trim(),
        "agent/alpha"
    );
    assert_eq!(git_in(&wt.path, &["rev-parse", "HEAD"]).trim(), wt.base);
    // cwd mapping: repo root maps to the checkout, subdirs keep the relative path.
    assert_eq!(wt.map_cwd(&repo.root), wt.path);
    let sub = repo.root.join("crates/omp-task");
    assert_eq!(wt.map_cwd(&sub), wt.path.join("crates/omp-task"));
    // Outside the repo maps to the checkout root.
    let outside = repo.root.parent().unwrap().join("elsewhere");
    assert_eq!(wt.map_cwd(&outside), wt.path);
}

#[test]
fn env_overrides_the_worktree_dir() {
    let _g = lock();
    let repo = Repo::new();
    let custom = repo.root.parent().unwrap().join("custom-wt");
    std::env::set_var(WORKTREE_DIR_ENV, &custom);
    let wt = Worktree::create(&repo.root, "beta").unwrap();
    std::env::remove_var(WORKTREE_DIR_ENV);
    assert_eq!(wt.path, custom.join("beta"));
    wt.remove().unwrap();
}

#[test]
fn branch_or_path_already_exists_is_a_clear_error() {
    let _g = lock();
    let repo = Repo::new();
    let wt = Worktree::create(&repo.root, "gamma").unwrap();
    let again = Worktree::create(&repo.root, "gamma").unwrap_err();
    assert!(again.to_string().contains("already exists"), "{again}");
    wt.remove().unwrap();
    // After removal the branch still exists; removal alone does not unblock.
    assert!(Worktree::create(&repo.root, "gamma").is_err());
}

#[test]
fn settle_summary_lists_commits_and_wips_leftovers() {
    let _g = lock();
    let repo = Repo::new();
    let wt = Worktree::create(&repo.root, "delta").unwrap();
    repo.commit_in(
        &wt.path,
        "delta.md",
        "work\n",
        "delta: the work\n\nAgent: delta\nEnvelope: agent://delta\nVerified-by: cargo test\n",
    );
    std::fs::write(wt.path.join("scratch.txt"), "leftover\n").unwrap();
    let summary = wt.settle_summary();
    assert_eq!(summary.branch, "agent/delta");
    assert!(!summary.clean, "leftover made the tree dirty");
    assert!(
        summary.leftovers_auto_committed,
        "leftovers must be auto-committed"
    );
    let subjects: Vec<&str> = summary.commits.iter().map(|c| c.subject.as_str()).collect();
    // `git log` order: newest first (the wip commit lands last).
    assert_eq!(
        subjects,
        ["wip(delta): uncommitted leftovers", "delta: the work"]
    );
    assert!(summary.commits.iter().all(|c| !c.sha.is_empty()));
    assert!(
        summary.diffstat.contains("2 files changed"),
        "diffstat covers both files: {}",
        summary.diffstat
    );
    // Tree is clean after the wip commit and the wip is real.
    assert!(git_in(&wt.path, &["status", "--porcelain"])
        .trim()
        .is_empty());
}

#[test]
fn settle_summary_of_a_clean_child_is_empty() {
    let _g = lock();
    let repo = Repo::new();
    let wt = Worktree::create(&repo.root, "epsilon").unwrap();
    let summary = wt.settle_summary();
    assert!(summary.clean);
    assert!(!summary.leftovers_auto_committed);
    assert!(summary.commits.is_empty());
    assert_eq!(summary.diffstat, "no changes");
}

#[test]
fn remove_takes_the_checkout_but_keeps_the_branch() {
    let _g = lock();
    let repo = Repo::new();
    let wt = Worktree::create(&repo.root, "zeta").unwrap();
    wt.remove().unwrap();
    assert!(!wt.path.exists());
    assert!(git_in(
        &repo.root,
        &["show-ref", "--verify", "refs/heads/agent/zeta"]
    )
    .contains("agent/zeta"));
}

#[test]
fn merge_agent_merges_with_aggregated_trailers_and_cleans_up() {
    let _g = lock();
    let repo = Repo::new();
    let wt = Worktree::create(&repo.root, "eta").unwrap();
    repo.commit_in(
        &wt.path,
        "eta.md",
        "eta\n",
        "eta: feature\n\nAgent: eta\nEnvelope: agent://eta\nVerified-by: cargo test -p omp-task\n",
    );
    // Diverge main so the merge is real.
    repo.commit_in(&repo.root, "main.md", "main\n", "main: unrelated");
    let outcome = merge_agent(&repo.root, "eta", &MergeOptions::default()).unwrap();
    let commit = match outcome.status {
        MergeStatus::Merged { commit } => commit,
        other => panic!("expected a merge, got {other:?}"),
    };
    assert_eq!(commit, repo.head());
    // The merge commit carries the aggregated trailers.
    let body = git_in(&repo.root, &["log", "-1", "--format=%B"]);
    assert!(body.contains("Agent: eta"), "{body}");
    assert!(body.contains("Envelope: agent://eta"), "{body}");
    assert!(
        body.contains("Verified-by: cargo test -p omp-task"),
        "{body}"
    );
    assert!(body.starts_with("merge agent/eta into main"));
    // Merge parents: main + agent/eta.
    let parents = git_in(&repo.root, &["log", "-1", "--format=%P"]);
    assert_eq!(parents.trim().split(' ').count(), 2);
    // Cleanup: worktree and branch are gone.
    assert!(!wt.path.exists());
    assert!(!git_in(&repo.root, &["branch", "--list", "agent/eta"]).contains("agent/eta"));
    // The merged content is on main.
    assert_eq!(
        std::fs::read_to_string(repo.root.join("eta.md")).unwrap(),
        "eta\n"
    );
}

#[test]
fn merge_agent_conflict_aborts_and_reports_files() {
    let _g = lock();
    let repo = Repo::new();
    let wt = Worktree::create(&repo.root, "theta").unwrap();
    std::fs::write(wt.path.join("file.txt"), "theta edit\n").unwrap();
    git_in(&wt.path, &["add", "-A"]);
    git_in(
        &wt.path,
        &["commit", "--no-verify", "-m", "theta: change line"],
    );
    // Same line on main -> conflict.
    std::fs::write(repo.root.join("file.txt"), "main edit\n").unwrap();
    git_in(&repo.root, &["add", "-A"]);
    git_in(
        &repo.root,
        &["commit", "--no-verify", "-m", "main: change line"],
    );
    let outcome = merge_agent(&repo.root, "theta", &MergeOptions::default()).unwrap();
    assert_eq!(
        outcome.status,
        MergeStatus::Conflict {
            files: vec!["file.txt".to_string()]
        }
    );
    // Not half-merged: no MERGE_HEAD, the file is back to main's content.
    let merge_head = Command::new("git")
        .args(["rev-parse", "-q", "--verify", "MERGE_HEAD"])
        .current_dir(&repo.root)
        .output()
        .unwrap();
    assert!(!merge_head.status.success(), "MERGE_HEAD must be gone");
    assert!(git_in(&repo.root, &["status", "--porcelain"])
        .trim()
        .is_empty());
    assert_eq!(
        std::fs::read_to_string(repo.root.join("file.txt")).unwrap(),
        "main edit\n"
    );
    // Nothing was cleaned up: the parent can retry or discard.
    assert!(wt.path.exists());
    assert!(git_in(&repo.root, &["branch", "--list", "agent/theta"]).contains("agent/theta"));
    // --discard throws the branch away after the conflict.
    let discarded = discard_agent(&repo.root, "theta").unwrap();
    assert_eq!(discarded.status, MergeStatus::Discarded);
    assert!(!wt.path.exists());
    assert!(!git_in(&repo.root, &["branch", "--list", "agent/theta"]).contains("agent/theta"));
}

#[test]
fn merge_agent_squash_commits_once_with_trailers() {
    let _g = lock();
    let repo = Repo::new();
    let wt = Worktree::create(&repo.root, "iota").unwrap();
    repo.commit_in(&wt.path, "i1.md", "1\n", "iota: one\n\nAgent: iota\n");
    repo.commit_in(&wt.path, "i2.md", "2\n", "iota: two");
    let outcome = merge_agent(
        &repo.root,
        "iota",
        &MergeOptions {
            squash: true,
            require_review: false,
        },
    )
    .unwrap();
    let commit = match outcome.status {
        MergeStatus::Merged { commit } => commit,
        other => panic!("expected a squash merge, got {other:?}"),
    };
    assert_eq!(commit, repo.head());
    // A squash commit has one parent and carries the aggregated trailers.
    let parents = git_in(&repo.root, &["log", "-1", "--format=%P"]);
    assert_eq!(parents.trim().split(' ').count(), 1);
    let body = git_in(&repo.root, &["log", "-1", "--format=%B"]);
    assert!(body.contains("Agent: iota"), "{body}");
    assert!(body.starts_with("merge agent/iota into main"));
    assert!(!wt.path.exists());
    assert!(std::fs::read(repo.root.join("i2.md")).is_ok());
}

#[test]
fn merge_agent_commits_leftovers_before_merging() {
    let _g = lock();
    let repo = Repo::new();
    let wt = Worktree::create(&repo.root, "kappa").unwrap();
    std::fs::write(wt.path.join("uncommitted.md"), "draft\n").unwrap();
    let outcome = merge_agent(&repo.root, "kappa", &MergeOptions::default()).unwrap();
    assert!(matches!(outcome.status, MergeStatus::Merged { .. }));
    // The draft rode the wip commit into main.
    assert_eq!(
        std::fs::read_to_string(repo.root.join("uncommitted.md")).unwrap(),
        "draft\n"
    );
    let body = git_in(&repo.root, &["log", "-1", "--format=%B"]);
    assert!(body.contains("Agent: kappa"), "{body}");
}

#[test]
fn merge_agent_aggregates_blank_line_separated_trailers() {
    let _g = lock();
    let repo = Repo::new();
    let wt = Worktree::create(&repo.root, "nu").unwrap();
    // Several `-m` flags: git separates the paragraphs with BLANK LINES, so
    // `%(trailers:only)` would see only the last one — the aggregation must
    // still pick every known key up (model-written commits do this).
    std::fs::write(wt.path.join("nu.md"), "nu\n").unwrap();
    git_in(&wt.path, &["add", "-A"]);
    git_in(
        &wt.path,
        &[
            "commit",
            "--no-verify",
            "-m",
            "nu: work",
            "-m",
            "Agent: nu",
            "-m",
            "Envelope: agent://nu",
            "-m",
            "Verified-by: cargo test",
        ],
    );
    let outcome = merge_agent(&repo.root, "nu", &MergeOptions::default()).unwrap();
    assert!(matches!(outcome.status, MergeStatus::Merged { .. }));
    let body = git_in(&repo.root, &["log", "-1", "--format=%B"]);
    assert!(body.contains("Agent: nu"), "{body}");
    assert!(body.contains("Envelope: agent://nu"), "{body}");
    assert!(body.contains("Verified-by: cargo test"), "{body}");
}

#[test]
fn require_review_refuses_until_a_pass_note_lands() {
    let _g = lock();
    let repo = Repo::new();
    let wt = Worktree::create(&repo.root, "lambda").unwrap();
    repo.commit_in(&wt.path, "lambda.md", "l\n", "lambda: work");
    let options = MergeOptions {
        squash: false,
        require_review: true,
    };
    let err = merge_agent(&repo.root, "lambda", &options).unwrap_err();
    assert!(err.to_string().contains("no passing review note"), "{err}");
    // A needs-work note is not enough.
    let tip = git_in(&wt.path, &["rev-parse", "HEAD"]).trim().to_string();
    crate::notes::record_review(
        &repo.root,
        &tip,
        crate::notes::Verdict::NeedsWork,
        "reviewer",
        None,
        Some("fix it"),
    )
    .unwrap();
    assert!(merge_agent(&repo.root, "lambda", &options).is_err());
    // The pass note unblocks.
    crate::notes::record_review(
        &repo.root,
        &tip,
        crate::notes::Verdict::Pass,
        "reviewer",
        Some("/tmp/evidence.txt"),
        Some("No grounded findings."),
    )
    .unwrap();
    let outcome = merge_agent(&repo.root, "lambda", &options).unwrap();
    assert!(matches!(outcome.status, MergeStatus::Merged { .. }));
    wt.remove().ok();
}

#[test]
fn agent_branches_lists_only_agent_prefix() {
    let _g = lock();
    let repo = Repo::new();
    let wt = Worktree::create(&repo.root, "mu").unwrap();
    git_in(&repo.root, &["branch", "feature/other"]);
    let branches = agent_branches(&repo.root);
    assert_eq!(branches, vec!["agent/mu".to_string()]);
    assert_eq!(find_worktree(&repo.root, "agent/mu"), Some(wt.path.clone()));
    assert_eq!(find_worktree(&repo.root, "agent/unknown"), None);
}
