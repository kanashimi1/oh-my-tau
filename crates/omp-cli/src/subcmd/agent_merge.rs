//! `omt agent-merge` — the parent's merge helper for worktree children
//! (`SPEC.md` batch 2, change 2).
//!
//! ```text
//! omt agent-merge <task-id>... | --all [--squash] [--discard] [--require-review]
//! ```
//!
//! Merges `agent/<task-id>` into the current branch with a merge commit that
//! aggregates the branch's `Agent:`/`Envelope:`/`Verified-by:` trailers,
//! then removes the worktree and deletes the branch. A conflict aborts the
//! merge (the repo is never left half-merged) and prints the conflicting
//! files. `--discard` throws the branch away without merging. With
//! `--require-review` the branch tip must carry a `verdict=pass` note on
//! `refs/notes/review` (see `omt review note`).

use omp_task::worktree::{self, MergeOptions, MergeStatus};

#[allow(clippy::struct_excessive_bools)] // one bool per CLI switch; they are independent flags
struct Args {
    ids: Vec<String>,
    all: bool,
    squash: bool,
    discard: bool,
    require_review: bool,
}

fn parse_args(args: &[String]) -> Result<Args, String> {
    let mut parsed = Args {
        ids: Vec::new(),
        all: false,
        squash: false,
        discard: false,
        require_review: false,
    };
    for arg in args {
        match arg.as_str() {
            "--all" => parsed.all = true,
            "--squash" => parsed.squash = true,
            "--discard" => parsed.discard = true,
            "--require-review" => parsed.require_review = true,
            "--help" | "-h" => return Err("usage".to_string()),
            other if other.starts_with('-') => {
                return Err(format!("unknown flag: {other}"));
            }
            other => parsed.ids.push(other.to_string()),
        }
    }
    Ok(parsed)
}

fn usage() {
    eprintln!(
        "omt agent-merge — merge a worktree child's branch into the current one

Usage: omt agent-merge <task-id>... | --all [--squash] [--discard] [--require-review]

Merges agent/<task-id> into the current branch; the merge commit aggregates
the branch's Agent:/Envelope:/Verified-by: trailers. On conflict the merge is
aborted (nothing half-merged) and the conflicting files are printed. A
successful merge removes the worktree and deletes the branch; --discard does
that without merging. --require-review refuses branches whose tip has no
passing verdict note (`omt review note`)."
    );
}

/// Dispatch `omt agent-merge <args>`; returns the process exit code.
///
/// # Errors
/// Usage errors, not-a-repo, or merge failures (conflicts are reported on
/// stdout with exit code 1).
pub fn run(args: &[String]) -> anyhow::Result<i32> {
    let parsed = match parse_args(args) {
        Ok(parsed) => parsed,
        Err(e) => {
            if e == "usage" {
                usage();
                return Ok(2);
            }
            eprintln!("omt agent-merge: {e}\n");
            usage();
            return Ok(2);
        }
    };
    if parsed.ids.is_empty() && !parsed.all {
        eprintln!("omt agent-merge: give at least one <task-id> or --all\n");
        usage();
        return Ok(2);
    }
    let cwd = std::env::current_dir()?;
    let repo = worktree::repo_root(&cwd).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut ids = parsed.ids;
    if parsed.all {
        let mut branches = worktree::agent_branches(&repo);
        if branches.is_empty() {
            println!("no agent/* branches to merge in {}", repo.display());
            return Ok(0);
        }
        branches.sort();
        ids.extend(
            branches
                .into_iter()
                .map(|b| b.trim_start_matches("agent/").to_string()),
        );
    }
    let options = MergeOptions {
        squash: parsed.squash,
        require_review: parsed.require_review,
    };
    let mut had_conflict = false;
    for id in ids {
        if parsed.discard {
            match worktree::discard_agent(&repo, &id) {
                Ok(outcome) => println!(
                    "discarded {} (worktree removed, branch deleted)",
                    outcome.branch
                ),
                Err(e) => eprintln!("omt agent-merge {id}: {e}"),
            }
            continue;
        }
        match worktree::merge_agent(&repo, &id, &options) {
            Ok(outcome) => match outcome.status {
                MergeStatus::Merged { commit } => {
                    let mode = if outcome.squash {
                        "squash-merged"
                    } else {
                        "merged"
                    };
                    println!(
                        "{mode} {} (commit {})",
                        outcome.branch,
                        &commit[..7.min(commit.len())]
                    );
                    println!("  worktree + branch removed");
                }
                MergeStatus::Conflict { files } => {
                    had_conflict = true;
                    eprintln!(
                        "CONFLICT merging {}: merge aborted, repo unchanged.",
                        outcome.branch
                    );
                    eprintln!("Conflicting files:");
                    for file in &files {
                        eprintln!("  {file}");
                    }
                }
                MergeStatus::Discarded => unreachable!("discard handled above"),
            },
            Err(e) => {
                had_conflict = true;
                eprintln!("omt agent-merge {id}: {e}");
            }
        }
    }
    Ok(i32::from(had_conflict))
}
