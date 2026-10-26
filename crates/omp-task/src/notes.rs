//! Review verdicts as git notes on `refs/notes/review` (`SPEC.md` batch 2,
//! change 3).
//!
//! Every verdict is one append-only structured line inside the commit's
//! note, so multiple reviewers can record independently:
//!
//! ```text
//! verdict=pass by=asymmetric-reviewer date=1729999999 evidence=/abs/path summary=No grounded findings.
//! ```
//!
//! `omt review note` appends a line, `omt review show` prints them, and
//! `omt agent-merge --require-review` refuses to merge a branch whose tip
//! has no `verdict=pass` line.

use std::path::Path;
use std::process::Command;

use thiserror::Error;

/// The git notes ref holding review verdicts (`refs/notes/review`).
pub const REVIEW_NOTES_REF: &str = "review";

/// A recorded verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    Pass,
    Fail,
    NeedsWork,
}

impl Verdict {
    /// Parse the wire/`--verdict` spelling.
    #[must_use]
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim() {
            "pass" => Some(Self::Pass),
            "fail" => Some(Self::Fail),
            "needs-work" => Some(Self::NeedsWork),
            _ => None,
        }
    }

    /// The wire/`--verdict` spelling.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pass => "pass",
            Self::Fail => "fail",
            Self::NeedsWork => "needs-work",
        }
    }
}

/// What can go wrong recording or reading a verdict.
#[derive(Debug, Error)]
pub enum NotesError {
    #[error("review: git {0} failed: {1}")]
    Git(String, String),
    #[error("review: {0}")]
    Message(String),
}

/// One parsed verdict line from a commit's note.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReviewNote {
    pub verdict: Verdict,
    pub by: String,
    /// Unix seconds when the verdict was recorded.
    pub date: u64,
    pub evidence: Option<String>,
    pub summary: Option<String>,
}

fn git(repo: &Path, args: &[&str]) -> Result<String, NotesError> {
    let output = Command::new("git")
        .args(args)
        .current_dir(repo)
        .env("GIT_TERMINAL_PROMPT", "0")
        .output()
        .map_err(|e| NotesError::Message(format!("cannot spawn `git`: {e}")))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(NotesError::Git(args.join(" "), stderr.trim().to_string()));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// Resolve any committish to a full sha (`git rev-parse`).
fn resolve(repo: &Path, commit: &str) -> Result<String, NotesError> {
    let out = git(
        repo,
        &["rev-parse", "--verify", &format!("{commit}^{{commit}}")],
    )?;
    Ok(out.trim().to_string())
}

/// Read the raw review note of one commit; `None` when there is none.
fn read_note(repo: &Path, sha: &str) -> Option<String> {
    let output = Command::new("git")
        .args(["notes", "--ref", REVIEW_NOTES_REF, "show", sha])
        .current_dir(repo)
        .env("GIT_TERMINAL_PROMPT", "0")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// Record one verdict line on `commit`'s review note (appending to whatever
/// reviewers recorded before).
///
/// # Errors
/// `commit` does not resolve, or the notes write failed.
pub fn record_review(
    repo: &Path,
    commit: &str,
    verdict: Verdict,
    by: &str,
    evidence: Option<&str>,
    summary: Option<&str>,
) -> Result<String, NotesError> {
    let sha = resolve(repo, commit)?;
    let date = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let mut line = format!(
        "verdict={} by={} date={date}",
        verdict.as_str(),
        sanitize(by)
    );
    if let Some(evidence) = evidence.filter(|e| !e.trim().is_empty()) {
        line.push_str(&format!(" evidence={}", sanitize(evidence)));
    }
    if let Some(summary) = summary.filter(|s| !s.trim().is_empty()) {
        line.push_str(&format!(" summary={}", sanitize(summary)));
    }
    let mut content = read_note(repo, &sha).unwrap_or_default();
    if !content.trim().is_empty() {
        content.push('\n');
    }
    content.push_str(&line);
    content.push('\n');
    git(
        repo,
        &[
            "notes",
            "--ref",
            REVIEW_NOTES_REF,
            "add",
            "-f",
            "-m",
            &content,
            &sha,
        ],
    )?;
    Ok(sha)
}

/// Flatten a field for one line: no newlines/tabs, trimmed.
fn sanitize(raw: &str) -> String {
    raw.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Parse the structured lines of one note's body.
#[must_use]
pub fn parse_notes(body: &str) -> Vec<ReviewNote> {
    body.lines()
        .filter_map(|line| {
            let line = line.trim();
            if !line.starts_with("verdict=") {
                return None;
            }
            let rest = &line["verdict=".len()..];
            let verdict_raw = rest.split_whitespace().next().unwrap_or("");
            let verdict = Verdict::parse(verdict_raw)?;
            let by = field(rest, "by=").unwrap_or_default();
            let date = field(rest, "date=")
                .and_then(|d| d.parse().ok())
                .unwrap_or(0);
            let evidence = field(rest, "evidence=").filter(|v| !v.is_empty());
            let summary = field(rest, "summary=").filter(|v| !v.is_empty());
            Some(ReviewNote {
                verdict,
                by,
                date,
                evidence,
                summary,
            })
        })
        .collect()
}

/// Extract `name=`-style field `key` from a line: the run of text after the
/// first ` key` up to the next known field (or end of line). Values may
/// contain spaces.
fn field(line: &str, key: &str) -> Option<String> {
    let marker = format!(" {key}");
    let start = line.find(&marker)? + marker.len();
    let rest = &line[start..];
    for next in [" by=", " date=", " evidence=", " summary="] {
        if let Some(idx) = rest.find(next) {
            return Some(rest[..idx].trim().to_string());
        }
    }
    Some(rest.trim().to_string())
}

/// The recorded verdicts on one commit (empty when unreviewed).
///
/// # Errors
/// `commit` does not resolve.
pub fn reviews_for(repo: &Path, commit: &str) -> Result<Vec<ReviewNote>, NotesError> {
    let sha = resolve(repo, commit)?;
    Ok(read_note(repo, &sha)
        .map(|body| parse_notes(&body))
        .unwrap_or_default())
}

/// Whether `commit` carries at least one `verdict=pass` line.
#[must_use]
pub fn has_passing_review(repo: &Path, commit: &str) -> bool {
    reviews_for(repo, commit)
        .map(|notes| notes.iter().any(|n| n.verdict == Verdict::Pass))
        .unwrap_or(false)
}

/// Commits of `range` newest-first (`git rev-list`), for `review show a..b`.
///
/// # Errors
/// `range` is not a valid revision range.
pub fn range_commits(repo: &Path, range: &str) -> Result<Vec<String>, NotesError> {
    let out = git(repo, &["rev-list", range])?;
    Ok(out.lines().map(str::trim).map(str::to_string).collect())
}

#[cfg(test)]
mod tests;
