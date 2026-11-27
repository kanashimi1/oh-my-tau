//! Turn-economy experiment arms (`OMP_TURN_<ARM>=1`): one binary serves every
//! arm of the A/B, and each switch defaults to the deployed behaviour.
//!
//! | arm | idea | where it acts |
//! |---|---|---|
//! | `OMP_TURN_QUIET` | do not end a turn on a quiet success | `eval` description, rule first |
//! | `OMP_TURN_TODO` | the todo update rides with the next working call | `eval` description |
//! | `OMP_TURN_HARNESS` | errors the harness itself causes | `#!bash` cells, stale `asyncio.run` note, 180 s cell deadline |
//! | `OMP_TURN_WAIT` | wait inside the cell, not across turns | `eval` description |
//! | `OMP_TURN_READ` | fewer paging turns | `read` default limit 300 → 600, `eval` description |

use std::sync::LazyLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Arms {
    pub quiet: bool,
    pub todo: bool,
    pub harness: bool,
    pub wait: bool,
    pub read: bool,
}

fn flag(name: &str) -> bool {
    std::env::var_os(name).is_some_and(|v| !v.is_empty() && v != "0")
}

static ARMS: LazyLock<Arms> = LazyLock::new(|| Arms {
    quiet: flag("OMP_TURN_QUIET"),
    todo: flag("OMP_TURN_TODO"),
    harness: flag("OMP_TURN_HARNESS"),
    wait: flag("OMP_TURN_WAIT"),
    read: flag("OMP_TURN_READ"),
});

#[must_use]
pub(crate) fn arms() -> Arms {
    *ARMS
}

const QUIET_RULE: &str = "A short successful result with the task unfinished is a wasted turn. Before returning, continue in the same cell with \
every next step you can already write: run the validator right after the patch, read what a grep hit points to, chain the \
dependent commands, and stop at the first failure. Return only when the next step depends on output you must read, or when \
the task (or its reporting milestone) is reached.";
const TODO_NOTE: &str = "`todo` and `ask` are callable inside a cell like any other tool: `await tool.todo({op: \"done\", task: \"<exact \
task text>\"})`, then the next working call in the same cell. Never spend a turn on a call that only updates the plan: \
attach the update to the step it closes or to the step it starts.";
const HARNESS_NOTE: &str = "Pure shell step: start the cell with the line `#!bash` and write the script below it (`language: \"js\"`; no JS \
quoting). A cell gets 180 s by default.";
const WAIT_NOTE: &str = "A command that takes minutes: one call with its timeout raised — `tool.bash({command, timeout: 1800})`; the \
cell deadline pauses while a host call runs. Never background a job and poll it across turns; if a job is already in the \
background, poll it inside one cell until it reports EXIT.";
const READ_NOTE: &str = "`tool.read` returns up to 600 lines per call: request the whole range you need in one call (`path: \
\"src/a.rs:40-560\"`) instead of paging through a file 300 lines at a time.";

/// The deployed description plus the active arms' notes: the quiet rule goes
/// first (it is meant to shape how the cell is planned), the rest follow.
#[must_use]
fn compose(base: &str, a: Arms) -> String {
    let mut out = String::with_capacity(base.len() + 1200);
    if a.quiet {
        out.push_str(QUIET_RULE);
        out.push('\n');
    }
    // The deployed text still says `asyncio.run` fails; this line runs it fine.
    out.push_str(&base.replace("`asyncio.run` fails", "`asyncio.run` works"));
    for (on, note) in [
        (a.todo, TODO_NOTE),
        (a.harness, HARNESS_NOTE),
        (a.wait, WAIT_NOTE),
        (a.read, READ_NOTE),
    ] {
        if on {
            out.push('\n');
            out.push_str(note);
        }
    }
    out
}

/// The `eval` description for the active arms (read once per process).
pub(crate) fn description() -> &'static str {
    static TEXT: LazyLock<String> = LazyLock::new(|| compose(super::eval::DESCRIPTION, arms()));
    &TEXT
}

/// Cell deadline when the model passes none: the omp default (30 s), or 180 s
/// under the harness arm (a killed JS cell also loses its kernel state).
#[must_use]
pub(crate) fn eval_default_timeout_secs() -> f64 {
    if arms().harness {
        180.0
    } else {
        30.0
    }
}

/// `read` default line limit for the active arms.
#[must_use]
pub(crate) fn read_default_limit(configured: usize) -> usize {
    if arms().read {
        600
    } else {
        configured
    }
}

fn is_shell_shebang(line: &str) -> bool {
    matches!(
        line,
        "#!bash"
            | "#!sh"
            | "#!shell"
            | "#!/bin/bash"
            | "#!/bin/sh"
            | "#!/usr/bin/env bash"
            | "#!/usr/bin/env sh"
    )
}

pub(super) fn shell_cell_if(code: &str, harness: bool) -> Option<String> {
    if !harness {
        return None;
    }
    let trimmed = code.trim_start_matches(['\u{feff}', ' ', '\t', '\r']);
    let first = trimmed.lines().next()?.trim_end();
    if !is_shell_shebang(first) {
        return None;
    }
    let body = trimmed[first.len()..].trim_start_matches(['\r', '\n']);
    let mut js = String::with_capacity(body.len() + 160);
    js.push_str("const __ompShellScript = ");
    js.push_str(&serde_json::to_string(body).ok()?);
    // `tool.bash` throws on a non-zero exit: the exit status belongs to the
    // shell result the model reads, not to a JS error with a stack trace.
    js.push_str(
        ";\nreturn await (async () => {\n    const script = ",
    );
    js.push_str(&serde_json::to_string(body).ok()?);
    js.push_str(
        ";\n    try {\n        return (await tool.bash({ command: script, timeout: 1800 })).text;\n    } catch (error) {\n        return String(error && error.message ? error.message : error);\n    }\n})();\n",
    );
    Some(js)
}

/// A `#!bash` cell (harness arm): the script below the shebang runs as one
/// `tool.bash` call, so a pure shell step needs no JS quoting.
#[must_use]
pub(crate) fn shell_cell(code: &str) -> Option<String> {
    shell_cell_if(code, arms().harness)
}

#[cfg(test)]
mod tests {
    use super::*;

    const OFF: Arms = Arms {
        quiet: false,
        todo: false,
        harness: false,
        wait: false,
        read: false,
    };

    #[test]
    fn no_arm_keeps_the_deployed_description() {
        assert_eq!(compose("base", OFF), "base");
    }

    #[test]
    fn quiet_rule_comes_first_and_notes_follow_in_order() {
        let text = compose(
            "base",
            Arms {
                quiet: true,
                todo: true,
                harness: true,
                wait: true,
                read: true,
                ..OFF
            },
        );
        let base = text.find("base").expect("base present");
        assert!(text.starts_with(QUIET_RULE), "rule first");
        let mut at = base + "base".len();
        for note in [TODO_NOTE, HARNESS_NOTE, WAIT_NOTE, READ_NOTE] {
            let found = text[at..].find(note).expect("note after the previous one");
            at += found + note.len();
        }
    }

    #[test]
    fn shell_cell_runs_the_script_as_one_bash_call() {
        let js = shell_cell_if("#!/usr/bin/env bash\nset -e\necho \"quoted $HOME\" && ls | wc -l\n", true)
            .expect("expanded");
        assert!(js.starts_with("const __ompShellScript = \""));
        assert!(js.contains("await tool.bash({ command: script, timeout: 1800 })"));
        assert!(js.ends_with("})();\n"));
        assert!(!js.contains("\nlet "), "no top-level `let`: the runner demotes top-level declarations");
        assert_eq!(shell_cell_if("#!/usr/bin/env bun\nconsole.log(1)\n", true), None);
        assert_eq!(shell_cell_if("const x = 1;\n", true), None);
    }

    #[test]
    fn harness_arm_corrects_the_stale_asyncio_note() {
        let text = compose(
            "Python (IPython): top-level `await` works, `asyncio.run` fails. JS (Bun).",
            Arms {
                harness: true,
                ..OFF
            },
        );
        assert!(text.contains("`asyncio.run` works"));
        assert!(!text.contains("`asyncio.run` fails"));
        assert!(text.ends_with(HARNESS_NOTE));
    }

    #[test]
    fn shell_cell_is_off_without_the_arm() {
        assert_eq!(shell_cell_if("#!bash\necho hi\n", false), None);
    }
}
