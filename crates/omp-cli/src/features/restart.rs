//! `/restart`: exec the current binary with the original argv plus
//! `--resume <session file>`, dropping any existing resume/continue flags.
//! TS: slash-commands/builtin-lifecycle.ts (`restart`).

use std::sync::Arc;

use crate::rpc::slash::Outcome;
use crate::runtime::Runtime;

/// Resume/continue flags that must not survive into the restarted process.
/// `--resume`/`-r`/`--session` take an *optional* value (consumed only when
/// the next argv element is not flag-like, matching `cli.rs`).
const RESUME_FLAGS: &[&str] = &["--resume", "-r", "--session"];
/// Boolean continue flags (no value).
const CONTINUE_FLAGS: &[&str] = &["-c", "--continue"];

fn is_flag_like(arg: &str) -> bool {
    arg.starts_with('-') && arg != "-"
}

/// Rewrite `argv` for restart: drop resume/continue flags (and their optional
/// values), then append `--resume <session_file>`. argv[0] is kept.
#[must_use]
pub fn rewrite_argv<I, S>(argv: I, session_file: &str) -> Vec<String>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let argv: Vec<String> = argv.into_iter().map(Into::into).collect();
    let mut out: Vec<String> = Vec::with_capacity(argv.len() + 2);
    let mut i = 0;
    while i < argv.len() {
        let token = argv[i].as_str();
        if crate::cli::STRING_FLAGS.contains(&token) {
            // Value-taking flags: the next token is the value even when it
            // looks like a flag (`--model -r` ⇒ model is `-r`); keep both.
            out.push(argv[i].clone());
            if let Some(next) = argv.get(i + 1) {
                out.push(next.clone());
                i += 1;
            }
        } else {
            let drop = if CONTINUE_FLAGS.contains(&token) {
                true
            } else if RESUME_FLAGS.contains(&token) {
                // `--resume <value>`: consume a value-like successor (a bare
                // `--resume -p x` keeps `-p` as the next flag).
                if let Some(next) = argv.get(i + 1) {
                    if !is_flag_like(next) {
                        i += 1;
                    }
                }
                true
            } else if RESUME_FLAGS.iter().any(|f| {
                token
                    .strip_prefix(f)
                    .is_some_and(|rest| rest.starts_with('='))
            }) {
                // `--resume=<value>` inline form.
                true
            } else {
                false
            };
            if !drop {
                out.push(argv[i].clone());
            }
        }
        i += 1;
    }
    out.push("--resume".to_string());
    out.push(session_file.to_string());
    out
}

/// Exec target: the current binary, minus a trailing ` (deleted)` marker when
/// the stripped path exists (the binary was upgraded/replaced on disk).
#[must_use]
pub fn exec_path() -> Option<std::path::PathBuf> {
    let exe = std::env::current_exe().ok()?;
    if !exe.exists() {
        let raw = exe.to_string_lossy();
        if let Some(stripped) = raw.strip_suffix(" (deleted)") {
            let path = std::path::PathBuf::from(stripped);
            if path.exists() {
                return Some(path);
            }
        }
    }
    Some(exe)
}

/// Slash-command entry point.
pub async fn command(rt: &Arc<Runtime>, args: &str) -> Outcome {
    let _ = args;
    let Some(exe) = exec_path() else {
        rt.send(serde_json::json!({"type": "command_output", "text": "restart: failed to resolve the current executable path."}));
        return Outcome::Handled {
            agent_invoked: false,
        };
    };
    let session_file = rt.session_file();
    let argv = rewrite_argv(std::env::args(), &session_file.to_string_lossy());
    rt.send(
        serde_json::json!({"type": "command_output", "text": format!(
            "restart: exec {} (session {})",
            exe.display(),
            session_file.display()
        )}),
    );
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        // wait_settled first: never exec out from under a running turn.
        rt.wait_settled().await;
        let error = std::process::Command::new(&exe).args(&argv).exec();
        // exec only returns on failure.
        rt.send(serde_json::json!({"type": "command_output", "text": format!("restart: exec failed: {error}")}));
    }
    #[cfg(not(unix))]
    {
        rt.send(serde_json::json!({"type": "command_output", "text": "restart: not supported on this platform."}));
    }
    Outcome::Handled {
        agent_invoked: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn appends_resume() {
        assert_eq!(
            rewrite_argv(args(&["omt", "-p", "hi"]), "/s/session.jsonl"),
            args(&["omt", "-p", "hi", "--resume", "/s/session.jsonl"])
        );
    }

    #[test]
    fn drops_resume_flag_with_value() {
        assert_eq!(
            rewrite_argv(args(&["omt", "--resume", "abc", "-p", "hi"]), "/s.jsonl"),
            args(&["omt", "-p", "hi", "--resume", "/s.jsonl"])
        );
        assert_eq!(
            rewrite_argv(args(&["omt", "-r", "abc"]), "/s.jsonl"),
            args(&["omt", "--resume", "/s.jsonl"])
        );
        assert_eq!(
            rewrite_argv(args(&["omt", "--session", "abc"]), "/s.jsonl"),
            args(&["omt", "--resume", "/s.jsonl"])
        );
    }

    #[test]
    fn drops_bare_resume_without_eating_next_flag() {
        // `--resume` before another flag: the flag stays, the resume goes.
        assert_eq!(
            rewrite_argv(args(&["omt", "--resume", "-p", "x"]), "/s.jsonl"),
            args(&["omt", "-p", "x", "--resume", "/s.jsonl"])
        );
        // Trailing bare `--resume`.
        assert_eq!(
            rewrite_argv(args(&["omt", "x", "--resume"]), "/s.jsonl"),
            args(&["omt", "x", "--resume", "/s.jsonl"])
        );
    }

    #[test]
    fn drops_inline_resume_value() {
        assert_eq!(
            rewrite_argv(args(&["omt", "--resume=abc", "-p", "x"]), "/s.jsonl"),
            args(&["omt", "-p", "x", "--resume", "/s.jsonl"])
        );
    }

    #[test]
    fn drops_continue_flags() {
        assert_eq!(
            rewrite_argv(args(&["omt", "-c", "--continue", "x"]), "/s.jsonl"),
            args(&["omt", "x", "--resume", "/s.jsonl"])
        );
    }

    #[test]
    fn keeps_unrelated_lookalikes() {
        // A value that happens to follow a non-resume flag is untouched.
        assert_eq!(
            rewrite_argv(args(&["omt", "--model", "-r"]), "/s.jsonl"),
            args(&["omt", "--model", "-r", "--resume", "/s.jsonl"])
        );
    }
}
