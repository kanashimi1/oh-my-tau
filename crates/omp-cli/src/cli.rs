//! CLI argument parsing mirroring the TS omp CLI
//! (`pi-coding-agent/src/cli/flag-tables.ts` + `cli/args.ts`):
//! string setters consume the next token unconditionally, optional flags
//! (`--resume`/`--session`) consume it only when value-like, valueless flags
//! are booleans, `--flag=value` is split before dispatch, and `--` ends flag
//! parsing. Usage errors carry omp's exact message text.

use std::path::PathBuf;

use anyhow::{bail, Result};

/// `--resume` value: bare flag (`Yes`) or a session id prefix/path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Resume {
    Yes,
    Id(String),
}

/// Parsed CLI flags (TS `Args` shape, headless-only fields included).
#[derive(Debug, Default)]
#[allow(clippy::struct_excessive_bools)] // mirrors omp's parsed argv shape
pub struct Args {
    pub help: bool,
    pub version: bool,
    pub print: bool,
    /// Positional messages joined with a space (TS `messages`).
    pub prompt: Option<String>,
    pub cwd: Option<PathBuf>,
    pub model: Option<String>,
    pub provider: Option<String>,
    pub api_key: Option<String>,
    pub smol: Option<String>,
    pub slow: Option<String>,
    pub plan: Option<String>,
    pub goal: Option<String>,
    pub config: Vec<String>,
    pub add_dir: Vec<PathBuf>,
    pub mode: Option<String>,
    pub resume: Option<Resume>,
    pub continue_prev: bool,
    pub fork: Option<String>,
    pub from_claude: bool,
    pub from_codex: bool,
    pub session_dir: Option<PathBuf>,
    pub no_session: bool,
    pub models: Vec<String>,
    pub no_tools: bool,
    pub tools_csv: Option<String>,
    pub thinking: Option<String>,
    pub service_tier: Option<String>,
    pub system_prompt: Option<String>,
    pub system_prompt_template: Option<String>,
    pub append_system_prompt: Option<String>,
    pub allow_home: bool,
    pub profile: Option<String>,
    pub alias: Option<String>,
    pub hooks: Vec<String>,
    pub extension: Vec<PathBuf>,
    pub trusted_extensions: Vec<String>,
    pub no_extensions: bool,
    pub no_skills: bool,
    pub skills: Vec<String>,
    pub no_rules: bool,
    pub export: Option<String>,
    pub no_title: bool,
    pub no_ui: bool,
    pub print_thoughts: bool,
    pub hide_thinking: bool,
    pub max_time: Option<f64>,
    pub max_time_raw: Option<String>,
    pub auto_approve: bool,
    pub approval_mode: Option<String>,
    pub no_lsp: bool,
    pub no_pty: bool,
    pub advisor: bool,
    pub external_thinking: bool,
    pub prewalk: bool,
    pub no_prewalk: bool,
    pub prewalk_into: Option<String>,
    pub plan_yolo: bool,
    pub plan_yolo_into: Option<String>,
    pub dump_system_prompt: bool,
    /// `--compact-prompt`: render the compact system prompt template.
    pub compact_prompt: bool,
    /// `--sandbox`: run tool subprocesses under Landlock/seccomp confinement
    /// (`/sandbox`; features/sandbox.rs).
    pub sandbox: bool,
    /// `--context-strategy append|clm|clm-host` (opt-in; default `append`).
    pub context_strategy: Option<String>,
    /// Image attachments collected from `@<image>` positional args
    /// (kept out of the text prompt; passed to the agent as content).
    pub images: Vec<pa_agent::types::ImageContent>,
}

/// Flags whose value is taken from the next argv token, even when it looks
/// like a flag (`--system-prompt --profile foo` ⇒ prompt is `--profile`).
pub(crate) const STRING_FLAGS: &[&str] = &[
    "--cwd",
    "--config",
    "--add-dir",
    "--mode",
    "--fork",
    "--provider",
    "--model",
    "--smol",
    "--slow",
    "--goal",
    "--plan",
    "--prewalk-into",
    "--plan-yolo-into",
    "--max-time",
    "--service-tier",
    "--api-key",
    "--system-prompt",
    "--system-prompt-template",
    "--append-system-prompt",
    "--provider-session-id",
    "--prompt-cache-key",
    "--session-dir",
    "--models",
    "--tools",
    "--thinking",
    "--export",
    "--hook",
    "--extension",
    "-e",
    "--trusted-extension",
    "--plugin-dir",
    "--skills",
    "--approval-mode",
    "--profile",
    "--alias",
    "--context-strategy",
];

/// Optional-value flags: consume the next token only when it is value-like
/// (TS `OPTIONAL_FLAGS`; all three use `rejectEmpty`).
const OPTIONAL_FLAGS: &[&str] = &["--resume", "-r", "--session"];

/// Valueless boolean flags (TS `VALUELESS_FLAGS`; short aliases handled at
/// the dispatch site like the TS parser).
const VALUELESS_FLAGS: &[&str] = &[
    "--allow-home",
    "--from-claude",
    "--from-codex",
    "--no-session",
    "--no-tools",
    "--no-lsp",
    "--no-pty",
    "--hide-thinking",
    "--advisor",
    "--external-thinking",
    "--prewalk",
    "--no-prewalk",
    "--plan-yolo",
    "--compact-prompt",
    "--print-thoughts",
    "--no-skills",
    "--no-rules",
    "--no-title",
    "--no-ui",
    "--auto-approve",
    "--yolo",
    "--sandbox",
];

const SERVICE_TIER_VALUES: &[&str] = &[
    "none",
    "auto",
    "default",
    "flex",
    "scale",
    "priority",
    "ultrafast",
];

const MODE_VALUES: &[&str] = &["text", "json", "rpc", "acp", "rpc-ui"];

fn json_str(s: &str) -> String {
    serde_json::to_string(s).unwrap_or_else(|_| format!("\"{s}\""))
}

/// Usage error text, matching omp's `CliUsageError` report
/// ("Error: {msg}" plus a "Run `omp --help` for available flags." tail).
fn usage_err(msg: &str) -> anyhow::Error {
    anyhow::anyhow!("{msg}\nRun `omp --help` for available flags.")
}

fn parse_max_time(value: &str) -> Result<f64> {
    let trimmed = value.trim();
    let seconds = if let Some(caps) = max_time_captures(trimmed) {
        caps.0 * caps.1
    } else {
        match trimmed.parse::<f64>() {
            Ok(s) => s,
            Err(_) => {
                return Err(usage_err(&format!(
                    "Invalid --max-time value: {}. Expected a positive number of seconds or duration like \"5s\", \"10m\", \"1h\".",
                    json_str(value)
                )));
            }
        }
    };
    if seconds.is_finite() && seconds > 0.0 {
        Ok(seconds)
    } else {
        Err(usage_err(&format!(
            "Invalid --max-time value: {}. Expected a positive number of seconds or duration like \"5s\", \"10m\", \"1h\".",
            json_str(value)
        )))
    }
}

/// `^(\d+(?:\.\d+)?)([smh])$` with the multiplier applied (h=3600, m=60).
fn max_time_captures(s: &str) -> Option<(f64, f64)> {
    let last = s.chars().last()?;
    let mult = match last {
        'h' => 3600.0,
        'm' => 60.0,
        's' => 1.0,
        _ => return None,
    };
    let num = &s[..s.len() - last.len_utf8()];
    if num.is_empty() || !num.chars().next().is_some_and(|c| c.is_ascii_digit()) {
        return None;
    }
    if !num.chars().all(|c| c.is_ascii_digit() || c == '.') || num.matches('.').count() > 1 {
        return None;
    }
    num.parse::<f64>().ok().map(|v| (v, mult))
}

fn set_string(a: &mut Args, flag: &str, value: String, invalid: &mut Vec<String>) -> Result<()> {
    match flag {
        "--cwd" => a.cwd = Some(PathBuf::from(value)),
        "--config" => a.config.push(value),
        "--add-dir" => a.add_dir.push(PathBuf::from(value)),
        "--mode" => {
            if MODE_VALUES.contains(&value.as_str()) {
                a.mode = Some(value);
            } else {
                invalid.push(format!(
                    "Invalid --mode value: {}. Expected one of: text, json, rpc, rpc-ui, acp.",
                    json_str(&value)
                ));
            }
        }
        "--fork" => a.fork = Some(value),
        "--provider" => a.provider = Some(value),
        "--model" => a.model = Some(value),
        "--smol" => a.smol = Some(value),
        "--slow" => a.slow = Some(value),
        "--goal" => a.goal = Some(value),
        "--plan" => a.plan = Some(value),
        "--prewalk-into" => a.prewalk_into = Some(value),
        "--plan-yolo-into" => a.plan_yolo_into = Some(value),
        "--max-time" => {
            a.max_time = Some(parse_max_time(&value)?);
            a.max_time_raw = Some(value);
        }
        "--service-tier" => {
            if !SERVICE_TIER_VALUES.contains(&value.as_str()) {
                Err(usage_err(&format!(
                    "Invalid --service-tier value: {}. Expected one of: {}.",
                    json_str(&value),
                    SERVICE_TIER_VALUES.join(", ")
                )))?;
            }
            a.service_tier = Some(value);
        }
        "--api-key" => a.api_key = Some(value),
        "--context-strategy" => match crate::context::Kind::parse(&value) {
            Ok(_) => a.context_strategy = Some(value),
            Err(e) => invalid.push(format!("Invalid --context-strategy value: {e}")),
        },
        "--system-prompt" => a.system_prompt = Some(value),
        "--system-prompt-template" => a.system_prompt_template = Some(value),
        "--append-system-prompt" => a.append_system_prompt = Some(value),
        // Parsed but unused in omp-rs (no provider-session / plugin subsystem):
        "--provider-session-id" | "--prompt-cache-key" | "--plugin-dir" => {}
        "--session-dir" => a.session_dir = Some(PathBuf::from(value)),
        "--models" => {
            a.models = value.split(',').map(|s| s.trim().to_string()).collect();
        }
        "--tools" => {
            a.tools_csv = Some(
                value
                    .split(',')
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .collect::<Vec<_>>()
                    .join(","),
            );
        }
        "--thinking" => {
            if crate::runtime::parse_level(&value).is_some() || value == "auto" {
                a.thinking = Some(value);
            } else {
                invalid.push(format!(
                    "Invalid --thinking value: {}. Expected one of: off, minimal, low, medium, high, xhigh, max, auto.",
                    json_str(&value)
                ));
            }
        }
        "--export" => a.export = Some(value),
        "--hook" => a.hooks.push(value),
        "--extension" | "-e" => a.extension.push(PathBuf::from(value)),
        "--trusted-extension" => a.trusted_extensions.push(value),
        "--skills" => {
            a.skills = value.split(',').map(|s| s.trim().to_string()).collect();
        }
        "--approval-mode" => {
            if ["always-ask", "write", "yolo"].contains(&value.as_str()) {
                a.approval_mode = Some(value);
            } else {
                invalid.push(format!(
                    "Invalid --approval-mode value: {}. Expected one of: always-ask, write, yolo.",
                    json_str(&value)
                ));
            }
        }
        "--profile" => a.profile = Some(value),
        "--alias" => a.alias = Some(value),
        other => bail!("unknown flag: {other}"),
    }
    Ok(())
}

/// Parse argv exactly like the TS `parseArgs` dispatch (no extension flags —
/// omp-rs has no extension-registered flag table at parse time).
pub fn parse_from<I: Iterator<Item = String>>(it: I) -> Result<Args> {
    let mut args: Vec<String> = it.collect();
    let mut a = Args::default();
    let mut invalid: Vec<String> = Vec::new();
    let mut unrecognized: Vec<String> = Vec::new();
    let mut messages: Vec<String> = Vec::new();
    let mut images: Vec<pa_agent::types::ImageContent> = Vec::new();
    let mut trusted_flag_count = 0usize;
    let mut saw_separator = false;

    let mut i = 0usize;
    while i < args.len() {
        if saw_separator {
            push_message(&mut messages, &mut images, args[i].clone())?;
            i += 1;
            continue;
        }
        let mut arg = args[i].clone();
        let flag_index = i;

        // `--flag=value`: split the value into the next slot so the
        // value-consuming branches pick it up via `args[i + 1]` (TS splices
        // the same way); an unconsumed value is dropped after the loop body.
        let mut equals_value_index: Option<usize> = None;
        if arg.starts_with("--") && arg.contains('=') {
            let eq = arg.find('=').unwrap();
            let value = arg[eq + 1..].to_string();
            arg = arg[..eq].to_string();
            args.insert(i + 1, value);
            equals_value_index = Some(i + 1);
        }

        if STRING_FLAGS.contains(&arg.as_str()) {
            if arg == "--trusted-extension" {
                trusted_flag_count += 1;
            }
            if arg == "--goal"
                && (i + 1 >= args.len()
                    || (equals_value_index.is_none() && args[i + 1].starts_with('-')))
            {
                invalid.push("--goal requires an objective.".to_string());
                if i + 1 < args.len() {
                    i += 1; // a flag-looking token is the (rejected) value, not a flag
                }
            } else if i + 1 < args.len() {
                i += 1;
                let value = std::mem::take(&mut args[i]);
                set_string(&mut a, &arg, value, &mut invalid)?;
            } else {
                // TS string flags consume a missing value as "".
                set_string(&mut a, &arg, String::new(), &mut invalid)?;
            }
        } else if OPTIONAL_FLAGS.contains(&arg.as_str()) {
            let next = args.get(i + 1);
            let consume = next.is_some_and(|n| !n.starts_with('-') && !n.is_empty());
            a.resume = Some(if consume {
                i += 1;
                Resume::Id(std::mem::take(&mut args[i]))
            } else {
                Resume::Yes
            });
        } else if VALUELESS_FLAGS.contains(&arg.as_str()) {
            match arg.as_str() {
                "--allow-home" => a.allow_home = true,
                "--from-claude" => a.from_claude = true,
                "--from-codex" => a.from_codex = true,
                "--no-session" => a.no_session = true,
                "--no-tools" => a.no_tools = true,
                "--no-lsp" => a.no_lsp = true,
                "--no-pty" => a.no_pty = true,
                "--hide-thinking" => a.hide_thinking = true,
                "--advisor" => a.advisor = true,
                "--external-thinking" => a.external_thinking = true,
                "--prewalk" => a.prewalk = true,
                "--no-prewalk" => a.no_prewalk = true,
                "--plan-yolo" => a.plan_yolo = true,
                "--compact-prompt" => a.compact_prompt = true,
                "--sandbox" => a.sandbox = true,
                "--print-thoughts" => a.print_thoughts = true,
                "--no-skills" => a.no_skills = true,
                "--no-rules" => a.no_rules = true,
                "--no-title" => a.no_title = true,
                "--no-ui" => a.no_ui = true,
                "--auto-approve" | "--yolo" => a.auto_approve = true,
                _ => unreachable!("VALUELESS_FLAGS dispatch"),
            }
        } else {
            match arg.as_str() {
                "-h" | "--help" => a.help = true,
                "-v" | "--version" => a.version = true,
                "-c" | "--continue" => a.continue_prev = true,
                "-p" | "--print" => a.print = true,
                "--no-extensions" => a.no_extensions = true,
                "--dump-system-prompt" => a.dump_system_prompt = true,
                other if other.starts_with('@') => {
                    push_message(&mut messages, &mut images, other.to_string())?
                }
                "-" => push_message(&mut messages, &mut images, "-".to_string())?,
                other if !other.starts_with('-') => {
                    push_message(&mut messages, &mut images, other.to_string())?;
                }
                "--" => saw_separator = true,
                other => unrecognized.push(other.to_string()),
            }
        }
        // Drop an unconsumed `--flag=value` value (e.g. a boolean flag).
        let equals_value_index = equals_value_index.filter(|_| i == flag_index);
        if let Some(equals_value_index) = equals_value_index {
            args.remove(equals_value_index);
        }
        i += 1;
    }

    finish(
        a,
        &messages,
        images,
        &invalid,
        &unrecognized,
        trusted_flag_count,
    )
}

/// Post-parse validation (TS end-of-`parseArgs` checks) and error reporting
/// (`reportUnrecognizedFlags` / `reportInvalidFlagValues`).
fn finish(
    mut a: Args,
    messages: &[String],
    images: Vec<pa_agent::types::ImageContent>,
    invalid: &[String],
    unrecognized: &[String],
    trusted_flag_count: usize,
) -> Result<Args> {
    if !messages.is_empty() {
        a.prompt = Some(messages.join(" "));
    }
    a.images = images;
    if let Some(goal) = a.goal.take() {
        if goal.trim().is_empty() {
            return Err(usage_err("--goal requires a non-empty objective."));
        }
        a.goal = Some(goal);
    }
    let ext_or_hook_mentions_trusted = {
        let ext_hit = a
            .extension
            .iter()
            .any(|v| v.display().to_string() == "--trusted-extension");
        let hook_hit = a
            .hooks
            .iter()
            .any(|v| v == "--trusted-extension" || v.starts_with("--trusted-extension="));
        ext_hit || hook_hit
    };
    if a.trusted_extensions.len() != trusted_flag_count || ext_or_hook_mentions_trusted {
        return Err(usage_err(
            "--trusted-extension requires a non-empty, non-flag value",
        ));
    }
    if trusted_flag_count > 0 && (!a.extension.is_empty() || !a.hooks.is_empty()) {
        return Err(usage_err(
            "--trusted-extension cannot be combined with --extension, -e, or --hook",
        ));
    }
    for trusted in &a.trusted_extensions {
        if trusted.is_empty() {
            return Err(usage_err(
                "--trusted-extension requires a non-empty, non-flag value",
            ));
        }
        if !PathBuf::from(trusted).is_absolute() {
            return Err(usage_err(&format!(
                "--trusted-extension requires an absolute path: {trusted}"
            )));
        }
    }
    if a.system_prompt.is_some() && a.system_prompt_template.is_some() {
        return Err(usage_err(
            "--system-prompt and --system-prompt-template cannot be combined",
        ));
    }
    if !unrecognized.is_empty() {
        let plural = if unrecognized.len() == 1 { "" } else { "s" };
        bail!(
            "unknown flag{plural}: {}\nRun `omp --help` for available flags.",
            unrecognized.join(", ")
        );
    }
    if !invalid.is_empty() {
        bail!("{}", invalid.join("\n"));
    }
    Ok(a)
}

/// Append a positional message, expanding `@<path>` attachments the way the
/// TS CLI combines them into the initial prompt: text files are inlined,
/// image files become `ImageContent` entries in `images` (never joined into
/// the text prompt).
fn push_message(
    messages: &mut Vec<String>,
    images: &mut Vec<pa_agent::types::ImageContent>,
    arg: String,
) -> Result<()> {
    if let Some(path) = arg.strip_prefix('@') {
        let path = path
            .strip_prefix('"')
            .and_then(|p| p.strip_suffix('"'))
            .or_else(|| path.strip_prefix('\'').and_then(|p| p.strip_suffix('\'')))
            .unwrap_or(path);
        if !path.is_empty() && std::path::Path::new(path).is_file() {
            let p = std::path::Path::new(path);
            if crate::images::looks_like_image(p) {
                return match crate::images::read_image(p) {
                    Ok(img) => {
                        images.push(img);
                        Ok(())
                    }
                    Err(e) => bail!("cannot read @{path}: {e:#}"),
                };
            }
            let text = std::fs::read_to_string(path)
                .map_err(|e| anyhow::anyhow!("cannot read @{path}: {e}"))?;
            messages.push(text);
            return Ok(());
        }
    }
    messages.push(arg);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(argv: &[&str]) -> Result<Args> {
        parse_from(argv.iter().copied().map(ToString::to_string))
    }

    #[test]
    fn print_flag_is_boolean_and_prompt_is_positional() {
        let a = parse(&["-p", "--no-session", "--model", "zai/glm-5.3-flash", "do X"]).unwrap();
        assert!(a.print);
        assert!(a.no_session);
        assert_eq!(a.model.as_deref(), Some("zai/glm-5.3-flash"));
        assert_eq!(a.prompt.as_deref(), Some("do X"));
    }

    #[test]
    fn multiple_positional_messages_are_joined() {
        let a = parse(&["-p", "first", "second"]).unwrap();
        assert_eq!(a.prompt.as_deref(), Some("first second"));
    }

    #[test]
    fn at_file_expands_into_prompt() {
        let dir = std::env::temp_dir();
        let f = dir.join(format!("omp-rs-cli-parse-test-{}", std::process::id()));
        std::fs::write(&f, "file contents\n").unwrap();
        let arg = format!("@{}", f.display());
        let a = parse(&["-p", &arg, "tail"]).unwrap();
        assert_eq!(a.prompt.as_deref(), Some("file contents\n tail"));
        std::fs::remove_file(&f).unwrap();
    }

    #[test]
    fn double_dash_ends_flag_parsing() {
        let a = parse(&["-p", "--", "--not-a-flag"]).unwrap();
        assert_eq!(a.prompt.as_deref(), Some("--not-a-flag"));
    }

    #[test]
    fn unknown_flag_errors_with_omp_message() {
        let err = parse(&["--nosession", "x"]).unwrap_err().to_string();
        assert_eq!(
            err,
            "unknown flag: --nosession\nRun `omp --help` for available flags."
        );
        let err = parse(&["--foo", "--bar"]).unwrap_err().to_string();
        assert!(err.starts_with("unknown flags: --foo, --bar"), "{err}");
    }

    #[test]
    fn version_and_help_short_aliases() {
        assert!(parse(&["-v"]).unwrap().version);
        assert!(parse(&["--version"]).unwrap().version);
        assert!(parse(&["-h"]).unwrap().help);
    }

    #[test]
    fn continue_short_alias() {
        assert!(parse(&["-c", "-p", "x"]).unwrap().continue_prev);
        assert!(parse(&["--continue"]).unwrap().continue_prev);
    }

    #[test]
    fn resume_optional_value() {
        let a = parse(&["--resume", "abc123"]).unwrap();
        assert_eq!(a.resume, Some(Resume::Id("abc123".into())));
        // flag-looking successor is not consumed
        let a = parse(&["--resume", "-p", "x"]).unwrap();
        assert_eq!(a.resume, Some(Resume::Yes));
        assert!(a.print);
        // bare at end
        assert_eq!(parse(&["-r"]).unwrap().resume, Some(Resume::Yes));
        // --session aliases resume; empty successor rejected
        let a = parse(&["--session", ""]).unwrap();
        assert_eq!(a.resume, Some(Resume::Yes));
        // inline form
        let a = parse(&["--resume=def456"]).unwrap();
        assert_eq!(a.resume, Some(Resume::Id("def456".into())));
    }

    #[test]
    fn fork_takes_value() {
        assert_eq!(
            parse(&["--fork", "f1"]).unwrap().fork.as_deref(),
            Some("f1")
        );
        assert_eq!(parse(&["--fork=f2"]).unwrap().fork.as_deref(), Some("f2"));
    }

    #[test]
    fn mode_validates_against_ts_set() {
        for m in ["text", "json", "rpc", "acp", "rpc-ui"] {
            assert_eq!(parse(&["--mode", m]).unwrap().mode.as_deref(), Some(m));
        }
        let err = parse(&["--mode", "bogus"]).unwrap_err().to_string();
        assert_eq!(
            err,
            "Invalid --mode value: \"bogus\". Expected one of: text, json, rpc, rpc-ui, acp."
        );
    }

    #[test]
    fn max_time_parses_seconds_and_durations() {
        let a = parse(&["--max-time", "600"]).unwrap();
        assert_eq!(a.max_time, Some(600.0));
        assert_eq!(parse(&["--max-time", "5s"]).unwrap().max_time, Some(5.0));
        assert_eq!(parse(&["--max-time", "10m"]).unwrap().max_time, Some(600.0));
        assert_eq!(
            parse(&["--max-time", "1.5h"]).unwrap().max_time,
            Some(5400.0)
        );
        let err = parse(&["--max-time", "abc"]).unwrap_err().to_string();
        assert_eq!(
            err,
            "Invalid --max-time value: \"abc\". Expected a positive number of seconds or duration like \"5s\", \"10m\", \"1h\".\nRun `omp --help` for available flags."
        );
        assert!(parse(&["--max-time", "0"]).is_err());
        assert!(parse(&["--max-time", "-5"]).is_err());
    }

    #[test]
    fn string_flags_consume_flag_looking_values() {
        let a = parse(&["--system-prompt", "--profile", "msg"]).unwrap();
        assert_eq!(a.system_prompt.as_deref(), Some("--profile"));
        assert_eq!(a.prompt.as_deref(), Some("msg"));
        // TS args.ts: `--system-prompt --profile foo` ⇒ the prompt is the
        // literal `--profile`; a following value-like token is a positional.
        let a = parse(&["--system-prompt", "--profile", "work", "msg"]).unwrap();
        assert_eq!(a.system_prompt.as_deref(), Some("--profile"));
        assert_eq!(a.prompt.as_deref(), Some("work msg"));
        // missing value becomes "" like TS
        let a = parse(&["--model"]).unwrap();
        assert_eq!(a.model.as_deref(), Some(""));
    }

    #[test]
    fn equals_form_splits_value() {
        let a = parse(&["--model=zai/glm-5.3-flash", "--thinking=high"]).unwrap();
        assert_eq!(a.model.as_deref(), Some("zai/glm-5.3-flash"));
        assert_eq!(a.thinking.as_deref(), Some("high"));
        // valueless flag with inline value drops it (TS guard)
        let a = parse(&["--print=yes"]).unwrap();
        assert!(a.print);
        assert!(a.prompt.is_none());
    }

    #[test]
    fn role_model_flags_parse() {
        let a = parse(&["--smol", "haiku", "--slow", "opus", "--plan", "gpt-5"]).unwrap();
        assert_eq!(a.smol.as_deref(), Some("haiku"));
        assert_eq!(a.slow.as_deref(), Some("opus"));
        assert_eq!(a.plan.as_deref(), Some("gpt-5"));
    }

    #[test]
    fn config_and_add_dir_repeatable() {
        let a = parse(&[
            "--config",
            "a.yml",
            "--config",
            "b.yml",
            "--add-dir",
            "/tmp",
            "--add-dir",
            "/var",
        ])
        .unwrap();
        assert_eq!(a.config, ["a.yml", "b.yml"]);
        assert_eq!(a.add_dir, [PathBuf::from("/tmp"), PathBuf::from("/var")]);
    }

    #[test]
    fn models_tools_skills_split_csv() {
        let a = parse(&[
            "--models",
            " a , b ,,c ",
            "--tools",
            "read, bash",
            "--skills",
            "git-*",
        ])
        .unwrap();
        assert_eq!(a.models, ["a", "b", "", "c"]); // TS maps trim, keeps empty parts
        assert_eq!(a.tools_csv.as_deref(), Some("read,bash"));
        assert_eq!(a.skills, ["git-*"]);
    }

    #[test]
    fn approval_mode_validated() {
        for m in ["always-ask", "write", "yolo"] {
            assert_eq!(
                parse(&["--approval-mode", m])
                    .unwrap()
                    .approval_mode
                    .as_deref(),
                Some(m)
            );
        }
        let err = parse(&["--approval-mode", "xx"]).unwrap_err().to_string();
        assert_eq!(
            err,
            "Invalid --approval-mode value: \"xx\". Expected one of: always-ask, write, yolo."
        );
        assert!(parse(&["--yolo"]).unwrap().auto_approve);
        assert!(parse(&["--auto-approve"]).unwrap().auto_approve);
    }

    #[test]
    fn service_tier_validated_with_omp_message() {
        let err = parse(&["--service-tier", "xx"]).unwrap_err().to_string();
        assert_eq!(
            err,
            "Invalid --service-tier value: \"xx\". Expected one of: none, auto, default, flex, scale, priority, ultrafast.\nRun `omp --help` for available flags."
        );
        assert_eq!(
            parse(&["--service-tier", "flex"])
                .unwrap()
                .service_tier
                .as_deref(),
            Some("flex")
        );
    }

    #[test]
    fn thinking_validated() {
        for t in [
            "off", "minimal", "low", "medium", "high", "xhigh", "max", "auto",
        ] {
            assert_eq!(
                parse(&["--thinking", t]).unwrap().thinking.as_deref(),
                Some(t)
            );
        }
        let err = parse(&["--thinking", "huge"]).unwrap_err().to_string();
        assert!(err.contains("Invalid --thinking value"), "{err}");
    }

    #[test]
    fn trusted_extension_validations() {
        let err = parse(&["--trusted-extension", "/abs/path", "-e", "/x"])
            .unwrap_err()
            .to_string();
        assert_eq!(
            err,
            "--trusted-extension cannot be combined with --extension, -e, or --hook\nRun `omp --help` for available flags."
        );
        let err = parse(&["--trusted-extension", "relative/path"])
            .unwrap_err()
            .to_string();
        assert_eq!(
            err,
            "--trusted-extension requires an absolute path: relative/path\nRun `omp --help` for available flags."
        );
        let a = parse(&["--trusted-extension", "/abs/path"]).unwrap();
        assert_eq!(a.trusted_extensions, ["/abs/path".to_string()]);
    }

    #[test]
    fn system_prompt_template_conflicts() {
        let err = parse(&["--system-prompt", "a", "--system-prompt-template", "b"])
            .unwrap_err()
            .to_string();
        assert_eq!(
            err,
            "--system-prompt and --system-prompt-template cannot be combined\nRun `omp --help` for available flags."
        );
    }

    #[test]
    fn goal_requires_objective() {
        let err = parse(&["--goal"]).unwrap_err().to_string();
        assert_eq!(err, "--goal requires an objective.");
        let err = parse(&["--goal", "   "]).unwrap_err().to_string();
        assert_eq!(
            err,
            "--goal requires a non-empty objective.\nRun `omp --help` for available flags."
        );
        // flag-looking successor is consumed as the rejected value, not a flag
        let err = parse(&["--goal", "--print"]).unwrap_err().to_string();
        assert_eq!(err, "--goal requires an objective.");
    }

    #[test]
    fn valueless_flags_accepted() {
        let a = parse(&[
            "--allow-home",
            "--no-tools",
            "--no-lsp",
            "--no-pty",
            "--hide-thinking",
            "--advisor",
            "--external-thinking",
            "--prewalk",
            "--no-prewalk",
            "--plan-yolo",
            "--print-thoughts",
            "--no-skills",
            "--no-rules",
            "--no-title",
            "--no-ui",
        ])
        .unwrap();
        assert!(a.allow_home && a.no_tools && a.no_lsp && a.no_pty && a.hide_thinking);
        assert!(a.advisor && a.external_thinking && a.prewalk && a.no_prewalk);
        assert!(
            a.plan_yolo && a.print_thoughts && a.no_skills && a.no_rules && a.no_title && a.no_ui
        );
    }

    #[test]
    fn compact_prompt_flag_parses() {
        let a = parse(&["--compact-prompt", "msg"]).unwrap();
        assert!(a.compact_prompt);
        let a = parse(&["msg"]).unwrap();
        assert!(!a.compact_prompt);
    }

    #[test]
    fn sandbox_flag_parses() {
        let a = parse(&["--sandbox", "msg"]).unwrap();
        assert!(a.sandbox);
        let a = parse(&["msg"]).unwrap();
        assert!(!a.sandbox);
    }

    #[test]
    fn session_dir_and_extension_short_alias() {
        let a = parse(&["--session-dir", "/tmp/s", "-e", "/ext.ts", "--hook", "h.ts"]).unwrap();
        assert_eq!(a.session_dir, Some(PathBuf::from("/tmp/s")));
        assert_eq!(a.extension, [PathBuf::from("/ext.ts")]);
        assert_eq!(a.hooks, ["h.ts".to_string()]);
    }

    #[test]
    fn profile_and_alias_parse() {
        let a = parse(&["--profile", "work", "--alias", "omp-work"]).unwrap();
        assert_eq!(a.profile.as_deref(), Some("work"));
        assert_eq!(a.alias.as_deref(), Some("omp-work"));
        assert_eq!(
            parse(&["--profile=p2"]).unwrap().profile.as_deref(),
            Some("p2")
        );
    }

    #[test]
    fn export_and_fork_flags() {
        assert_eq!(
            parse(&["--export", "s.jsonl"]).unwrap().export.as_deref(),
            Some("s.jsonl")
        );
        assert!(parse(&["--from-claude"]).unwrap().from_claude);
        assert!(parse(&["--from-codex"]).unwrap().from_codex);
    }
}
