//! Unit tests for review verdicts on `refs/notes/review`.

use std::path::{Path, PathBuf};
use std::process::Command;

use super::*;

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
        std::fs::write(root.join("f.txt"), "1\n").unwrap();
        git_in(&root, &["add", "-A"]);
        git_in(&root, &["commit", "-m", "c1"]);
        Self { _dir: dir, root }
    }

    fn commit_file(&self, file: &str, content: &str) -> String {
        std::fs::write(self.root.join(file), content).unwrap();
        git_in(&self.root, &["add", "-A"]);
        git_in(&self.root, &["commit", "-m", &format!("add {file}")]);
        git_in(&self.root, &["rev-parse", "HEAD"])
            .trim()
            .to_string()
    }
}

#[test]
fn verdict_parse_roundtrip() {
    assert_eq!(Verdict::parse("pass"), Some(Verdict::Pass));
    assert_eq!(Verdict::parse("fail"), Some(Verdict::Fail));
    assert_eq!(Verdict::parse("needs-work"), Some(Verdict::NeedsWork));
    assert_eq!(Verdict::parse("nope"), None);
    assert_eq!(Verdict::Pass.as_str(), "pass");
    assert_eq!(Verdict::Fail.as_str(), "fail");
    assert_eq!(Verdict::NeedsWork.as_str(), "needs-work");
}

#[test]
fn record_append_and_read_back_multiple_reviewers() {
    let repo = Repo::new();
    let sha = repo.commit_file("a.txt", "a\n");
    record_review(
        &repo.root,
        &sha,
        Verdict::NeedsWork,
        "second-reviewer",
        None,
        Some("naming is off\nsee line 3"),
    )
    .unwrap();
    // One needs-work line: no passing gate.
    assert!(!has_passing_review(&repo.root, &sha));
    record_review(
        &repo.root,
        &sha,
        Verdict::Pass,
        "asymmetric-reviewer",
        Some("/tmp/evidence.txt"),
        Some("No grounded findings."),
    )
    .unwrap();
    let notes = reviews_for(&repo.root, &sha).unwrap();
    assert_eq!(notes.len(), 2);
    assert_eq!(notes[0].verdict, Verdict::NeedsWork);
    assert_eq!(notes[0].by, "second-reviewer");
    assert_eq!(notes[0].evidence, None);
    // Newlines in a summary are flattened to one line.
    assert_eq!(
        notes[0].summary.as_deref(),
        Some("naming is off see line 3")
    );
    assert_eq!(notes[1].verdict, Verdict::Pass);
    assert_eq!(notes[1].by, "asymmetric-reviewer");
    assert_eq!(notes[1].evidence.as_deref(), Some("/tmp/evidence.txt"));
    assert_eq!(notes[1].summary.as_deref(), Some("No grounded findings."));
    assert!(notes[1].date > 0);
    // Any pass line satisfies the gate.
    assert!(has_passing_review(&repo.root, &sha));
}

#[test]
fn unreviewed_commit_has_no_notes() {
    let repo = Repo::new();
    let sha = repo.commit_file("b.txt", "b\n");
    assert!(reviews_for(&repo.root, &sha).unwrap().is_empty());
    assert!(!has_passing_review(&repo.root, &sha));
}

#[test]
fn record_accepts_a_short_sha() {
    let repo = Repo::new();
    let sha = repo.commit_file("c.txt", "c\n");
    let short = &sha[..8];
    let resolved = record_review(&repo.root, short, Verdict::Pass, "rv", None, None).unwrap();
    assert_eq!(resolved, sha);
    assert!(has_passing_review(&repo.root, &sha));
}

#[test]
fn unknown_commit_is_an_error() {
    let repo = Repo::new();
    assert!(record_review(&repo.root, "deadbeef", Verdict::Pass, "rv", None, None).is_err());
    assert!(reviews_for(&repo.root, "deadbeef").is_err());
}

#[test]
fn notes_live_on_the_review_ref() {
    let repo = Repo::new();
    let sha = repo.commit_file("d.txt", "d\n");
    record_review(&repo.root, &sha, Verdict::Pass, "rv", None, None).unwrap();
    let listing = git_in(&repo.root, &["notes", "--ref", "review", "list"]);
    assert!(listing.contains(&sha[..40]), "{listing}");
}

#[test]
fn range_commits_lists_newest_first() {
    let repo = Repo::new();
    let first = git_in(&repo.root, &["rev-parse", "HEAD"])
        .trim()
        .to_string();
    let second = repo.commit_file("e.txt", "e\n");
    let range = format!("{first}..{second}");
    let commits = range_commits(&repo.root, &range).unwrap();
    assert_eq!(commits, vec![second]);
}
