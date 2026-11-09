//! `omt review` — review verdicts as git notes on `refs/notes/review`
//! (`SPEC.md` batch 2, change 3).
//!
//! ```text
//! omt review note <commit> --verdict pass|fail|needs-work --by <agent>
//!                 [--evidence <path>] [--summary <text>]
//! omt review show <commit|range>
//! ```
//!
//! Notes are append-only structured lines so multiple reviewers can record
//! independently; `omt review show` prints them per commit, and
//! `omt agent-merge --require-review` gates on a `verdict=pass` line.

use omp_task::notes::{self, Verdict};

fn usage() {
    eprintln!(
        "omt review — review verdicts as git notes (refs/notes/review)

Usage:
  omt review note <commit> --verdict pass|fail|needs-work --by <agent> \\
                [--evidence <path>] [--summary <text>]
  omt review show <commit|range>

Each note is an append-only line: verdict=... by=... date=... evidence=...
summary=... — several reviewers can record on one commit. A passing verdict
on a branch tip satisfies `omt agent-merge --require-review`."
    );
}

/// Dispatch `omt review <command> [args]`; returns the process exit code.
///
/// # Errors
/// Usage errors, unknown verdicts, or git failures.
pub fn run(args: &[String]) -> anyhow::Result<i32> {
    let Some(cmd) = args.first().map(String::as_str) else {
        usage();
        return Ok(2);
    };
    match cmd {
        "note" => cmd_note(&args[1..]),
        "show" => cmd_show(&args[1..]),
        "--help" | "-h" | "help" => {
            usage();
            Ok(2)
        }
        other => {
            eprintln!("unknown review command: {other}\n");
            usage();
            Ok(2)
        }
    }
}

struct NoteArgs {
    commit: Option<String>,
    verdict: Option<String>,
    by: Option<String>,
    evidence: Option<String>,
    summary: Option<String>,
}

fn parse_note_args(args: &[String]) -> Result<NoteArgs, String> {
    let mut parsed = NoteArgs {
        commit: None,
        verdict: None,
        by: None,
        evidence: None,
        summary: None,
    };
    let mut iter = args.iter().peekable();
    while let Some(arg) = iter.next() {
        let mut flag_value = |name: &str| -> Result<String, String> {
            iter.next()
                .map(String::to_string)
                .ok_or_else(|| format!("{name} needs a value"))
        };
        match arg.as_str() {
            "--verdict" => parsed.verdict = Some(flag_value("--verdict")?),
            "--by" => parsed.by = Some(flag_value("--by")?),
            "--evidence" => parsed.evidence = Some(flag_value("--evidence")?),
            "--summary" => parsed.summary = Some(flag_value("--summary")?),
            other if other.starts_with('-') => return Err(format!("unknown flag: {other}")),
            other => {
                if parsed.commit.is_some() {
                    return Err(format!("unexpected argument: {other}"));
                }
                parsed.commit = Some(other.to_string());
            }
        }
    }
    Ok(parsed)
}

fn cmd_note(args: &[String]) -> anyhow::Result<i32> {
    let parsed = match parse_note_args(args) {
        Ok(parsed) => parsed,
        Err(e) => {
            eprintln!("omt review note: {e}\n");
            usage();
            return Ok(2);
        }
    };
    let (Some(commit), Some(verdict_raw), Some(by)) = (parsed.commit, parsed.verdict, parsed.by)
    else {
        eprintln!("omt review note: <commit>, --verdict and --by are required\n");
        usage();
        return Ok(2);
    };
    let Some(verdict) = Verdict::parse(&verdict_raw) else {
        eprintln!("omt review note: unknown verdict {verdict_raw:?} (pass|fail|needs-work)\n");
        usage();
        return Ok(2);
    };
    let cwd = std::env::current_dir()?;
    let repo = omp_task::worktree::repo_root(&cwd).map_err(|e| anyhow::anyhow!("{e}"))?;
    let sha = notes::record_review(
        &repo,
        &commit,
        verdict,
        &by,
        parsed.evidence.as_deref(),
        parsed.summary.as_deref(),
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;
    println!("recorded {} on {sha} by {by}", verdict.as_str());
    Ok(0)
}

// invariant: write!/writeln! to a String is infallible.
#[allow(clippy::expect_used)]
fn cmd_show(args: &[String]) -> anyhow::Result<i32> {
    use std::fmt::Write as _;
    let [rev] = args else {
        eprintln!("omt review show: give exactly one <commit|range>\n");
        usage();
        return Ok(2);
    };
    let cwd = std::env::current_dir()?;
    let repo = omp_task::worktree::repo_root(&cwd).map_err(|e| anyhow::anyhow!("{e}"))?;
    // A range (`a..b`, `HEAD~3..`) lists every commit's notes; anything else
    // is one commit.
    let shas = if rev.contains("..") {
        notes::range_commits(&repo, rev).map_err(|e| anyhow::anyhow!("{e}"))?
    } else {
        vec![rev.clone()]
    };
    if shas.is_empty() {
        println!("no commits in range {rev}");
        return Ok(0);
    }
    let mut shown = 0usize;
    for sha in shas {
        let reviews = notes::reviews_for(&repo, &sha).map_err(|e| anyhow::anyhow!("{e}"))?;
        if reviews.is_empty() {
            continue;
        }
        shown += 1;
        println!("{sha}:");
        for note in reviews {
            let mut line = format!("  verdict={} by={}", note.verdict.as_str(), note.by);
            if let Some(evidence) = note.evidence {
                write!(line, " evidence={evidence}").expect("writing to a String is infallible");
            }
            if let Some(summary) = note.summary {
                write!(line, " summary={summary}").expect("writing to a String is infallible");
            }
            println!("{line}");
        }
    }
    if shown == 0 {
        println!("no review notes for {rev}");
    }
    Ok(0)
}
