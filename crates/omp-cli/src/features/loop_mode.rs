//! `/loop`: re-run a prompt until a condition holds. TS: slash-commands/builtin-modes.ts
//! (loop mode) + modes/loop-condition.ts.
//!
//! Forms:
//!   `/loop <n> <prompt>` — run `<prompt>` n times.
//!   `/loop until "<shell cmd>" <prompt>` — re-run until the shell command exits 0
//!     (default max 10 iterations, override with `--max N`).
//!   `/loop stop` — cancel a running loop.
//!
//! Each iteration is one `rt.prompt` followed by `rt.wait_settled`; the loop
//! runs in a spawned task so the slash command returns immediately. Exit-code
//! semantics follow TS `loop-condition.ts`: 0 satisfies `until`, 1 keeps
//! looping, anything else means the condition itself is broken and halts the
//! loop with an error instead of reading as "false".

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use serde_json::json;

use crate::rpc::slash::Outcome;
use crate::runtime::Runtime;

/// Default cap for `until` loops.
pub const DEFAULT_MAX_ITERATIONS: u32 = 10;

static CANCEL: AtomicBool = AtomicBool::new(false);
static RUNNING: AtomicBool = AtomicBool::new(false);

/// What a parsed `/loop` invocation should do.
#[derive(Debug, Clone, PartialEq)]
pub enum LoopKind {
    /// Run the prompt exactly n times.
    Count { n: u32, prompt: String },
    /// Run the prompt, then evaluate the shell condition; stop when it exits 0.
    Until {
        command: String,
        prompt: String,
        max: u32,
    },
}

fn usage() -> String {
    "Usage: /loop <n> <prompt> | /loop until \"<shell cmd>\" <prompt> [--max N] | /loop stop"
        .to_string()
}

/// Extract a leading `"..."` quoted string; returns the unquoted content.
fn quoted_prefix(s: &str) -> Option<&str> {
    let rest = s.strip_prefix('"')?;
    let end = rest.find('"')?;
    Some(&rest[..end])
}

/// Split a leading `--max N` (with any spacing) off the argument string;
/// returns the remainder without the flag.
fn strip_max(args: &str) -> Result<(String, u32), String> {
    let Some(pos) = args.find("--max") else {
        return Ok((args.trim().to_string(), DEFAULT_MAX_ITERATIONS));
    };
    let rest = args[pos..].trim_start_matches("--max").trim();
    let (value, leftover) = match rest.split_once(char::is_whitespace) {
        Some((v, tail)) => (v, Some(tail)),
        None => (rest, None),
    };
    let max: u32 = value
        .parse()
        .map_err(|_| format!("Invalid --max value: {value:?}"))?;
    if max == 0 {
        return Err("--max must be at least 1".to_string());
    }
    let head = match leftover {
        Some(tail) => format!("{}{}", &args[..pos], tail),
        None => args[..pos].to_string(),
    };
    Ok((head.trim().to_string(), max))
}

/// Parse `/loop` arguments. Returns `None` for `/loop stop`.
pub fn parse_args(args: &str) -> Result<Option<LoopKind>, String> {
    let (args, max) = strip_max(args)?;
    if args.is_empty() {
        return Err(usage());
    }
    if args == "stop" {
        return Ok(None);
    }

    if let Some(rest) = args.strip_prefix("until") {
        const UNTIL_USAGE: &str = "Usage: /loop until \"<shell cmd>\" <prompt>";
        let rest = rest.trim_start();
        let Some(cmd) = quoted_prefix(rest) else {
            return Err(UNTIL_USAGE.to_string());
        };
        let prompt = rest[cmd.len() + 2..].trim();
        if prompt.is_empty() {
            return Err(UNTIL_USAGE.to_string());
        }
        return Ok(Some(LoopKind::Until {
            command: cmd.to_string(),
            prompt: prompt.to_string(),
            max,
        }));
    }

    let (count, prompt) = args.split_once(char::is_whitespace).ok_or_else(usage)?;
    let n: u32 = count
        .parse()
        .map_err(|_| format!("Invalid loop count: {count:?}"))?;
    if n == 0 {
        return Err("Loop count must be at least 1".to_string());
    }
    let prompt = prompt.trim();
    if prompt.is_empty() {
        return Err(usage());
    }
    Ok(Some(LoopKind::Count {
        n,
        prompt: prompt.to_string(),
    }))
}

/// Verdict of one `until` condition evaluation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Verdict {
    /// Keep looping.
    Continue,
    /// Condition satisfied: stop cleanly.
    Halt,
    /// The condition command itself is broken (exit > 1): stop and report.
    Error,
}

/// Map a condition exit code onto a loop verdict (TS `evaluateLoopCondition`).
pub fn condition_verdict(exit_code: Option<i32>) -> Verdict {
    match exit_code {
        Some(0) => Verdict::Halt,
        Some(1) => Verdict::Continue,
        _ => Verdict::Error,
    }
}

fn output(rt: &Runtime, text: impl Into<String>) {
    rt.send(json!({"type": "command_output", "text": text.into()}));
}

/// Run one condition evaluation via `sh -c`, in the session's cwd.
async fn eval_condition(rt: &Runtime, command: &str) -> Verdict {
    let status = tokio::process::Command::new("sh")
        .arg("-c")
        .arg(command)
        .current_dir(&rt.cwd)
        .status()
        .await;
    match status {
        Ok(st) => condition_verdict(st.code()),
        Err(_) => Verdict::Error,
    }
}

/// Slash-command entry point.
pub async fn command(rt: &Arc<Runtime>, args: &str) -> Outcome {
    let trimmed = args.trim();
    if trimmed == "stop" {
        if RUNNING.load(Ordering::SeqCst) {
            CANCEL.store(true, Ordering::SeqCst);
            rt.abort().await;
            output(rt, "Loop stop requested: aborting the current iteration.");
        } else {
            output(rt, "No loop is running.");
        }
        return Outcome::Handled {
            agent_invoked: false,
        };
    }

    let kind = match parse_args(args) {
        Ok(k) => k,
        Err(e) => {
            output(rt, e);
            return Outcome::Handled {
                agent_invoked: false,
            };
        }
    };
    let Some(kind) = kind else {
        return Outcome::Handled {
            agent_invoked: false,
        };
    };

    if RUNNING.load(Ordering::SeqCst) {
        output(rt, "A loop is already running: /loop stop first.");
        return Outcome::Handled {
            agent_invoked: false,
        };
    }

    let describe = match &kind {
        LoopKind::Count { n, prompt } => format!("loop {n}x: {prompt}"),
        LoopKind::Until {
            command,
            prompt,
            max,
        } => format!("loop until `{command}` succeeds (max {max}): {prompt}"),
    };
    output(rt, format!("Started {describe}. /loop stop to cancel."));

    let rt2 = rt.clone();
    tokio::spawn(async move {
        RUNNING.store(true, Ordering::SeqCst);
        run_loop(&rt2, kind).await;
        RUNNING.store(false, Ordering::SeqCst);
        CANCEL.store(false, Ordering::SeqCst);
    });

    Outcome::Handled {
        agent_invoked: false,
    }
}

/// The spawned loop body: one prompt + settle per iteration.
async fn run_loop(rt: &Arc<Runtime>, kind: LoopKind) {
    let total = match &kind {
        LoopKind::Count { n, .. } => *n,
        LoopKind::Until { max, .. } => *max,
    };
    let prompt_text = match &kind {
        LoopKind::Count { prompt, .. } | LoopKind::Until { prompt, .. } => prompt,
    };
    for i in 1..=total {
        if CANCEL.load(Ordering::SeqCst) {
            output(rt, format!("Loop stopped at iteration {i}."));
            return;
        }
        if crate::features::budget::is_tripped() {
            output(
                rt,
                format!("Loop stopped at iteration {i}: budget limit reached."),
            );
            return;
        }
        output(rt, format!("[loop {i}/{total}]"));
        if let Err(e) = rt.prompt(None, prompt_text, None, Vec::new()).await {
            output(rt, format!("Loop stopped: prompt failed: {}", e.message));
            return;
        }
        rt.wait_settled().await;

        if let LoopKind::Until { command, .. } = &kind {
            if CANCEL.load(Ordering::SeqCst) {
                output(rt, format!("Loop stopped at iteration {i}."));
                return;
            }
            match eval_condition(rt, command).await {
                Verdict::Halt => {
                    output(
                        rt,
                        format!(
                            "Loop condition `{command}` is now satisfied. Done ({i} iteration(s))."
                        ),
                    );
                    return;
                }
                Verdict::Continue => {}
                Verdict::Error => {
                    output(
                        rt,
                        format!(
                            "Loop condition `{command}` failed (exit > 1): the condition is broken, not unsatisfied. Loop stopped."
                        ),
                    );
                    return;
                }
            }
        }
    }
    output(rt, format!("Loop finished: {total} iteration(s)."));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_count_loops() {
        assert_eq!(
            parse_args("3 fix the tests"),
            Ok(Some(LoopKind::Count {
                n: 3,
                prompt: "fix the tests".to_string()
            }))
        );
        assert_eq!(
            parse_args("  12   say hi "),
            Ok(Some(LoopKind::Count {
                n: 12,
                prompt: "say hi".to_string()
            }))
        );
    }

    #[test]
    fn parses_until_loops() {
        assert_eq!(
            parse_args("until \"test -f done\" build it"),
            Ok(Some(LoopKind::Until {
                command: "test -f done".to_string(),
                prompt: "build it".to_string(),
                max: DEFAULT_MAX_ITERATIONS,
            }))
        );
        assert_eq!(
            parse_args("until \"grep -q ok f\" go --max 25"),
            Ok(Some(LoopKind::Until {
                command: "grep -q ok f".to_string(),
                prompt: "go".to_string(),
                max: 25,
            }))
        );
    }

    #[test]
    fn parses_stop_and_rejects_garbage() {
        assert_eq!(parse_args("stop"), Ok(None));
        assert_eq!(parse_args("  stop "), Ok(None));
        assert!(parse_args("").is_err());
        assert!(parse_args("until unquoted x").is_err());
        assert!(parse_args("until \"c\"").is_err());
        assert!(parse_args("abc hi").is_err());
        assert!(parse_args("0 hi").is_err());
        assert!(parse_args("3").is_err());
        assert!(parse_args("until \"c\" p --max 0").is_err());
        assert!(parse_args("until \"c\" p --max x").is_err());
    }

    #[test]
    fn verdict_mapping_matches_ts() {
        assert_eq!(condition_verdict(Some(0)), Verdict::Halt);
        assert_eq!(condition_verdict(Some(1)), Verdict::Continue);
        assert_eq!(condition_verdict(Some(2)), Verdict::Error);
        assert_eq!(condition_verdict(Some(127)), Verdict::Error);
        assert_eq!(condition_verdict(None), Verdict::Error);
    }
}
