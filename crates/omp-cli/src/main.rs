//! omp-rs: Rust port of the omp coding agent (print + rpc modes).

mod adapter;
mod agent;
mod compact;
mod events;
mod ext;
mod model;
mod paging;
mod persist;
mod rpc;
mod runtime;
mod sysprompt;
mod tools;
mod tree;

use std::path::PathBuf;

use anyhow::{bail, Result};

#[allow(clippy::struct_excessive_bools)] // CLI flag struct mirroring omp's parsed argv shape
struct Args {
    print: bool,
    prompt: Option<String>,
    model: Option<String>,
    cwd: Option<PathBuf>,
    mode: Option<String>,
    no_session: bool,
    no_extensions: bool,
    extension: Vec<PathBuf>,
    thinking: Option<String>,
    system_prompt: Option<String>,
    append_system_prompt: Option<String>,
    tools_csv: Option<String>,
    dump_system_prompt: bool,
}

fn parse_args() -> Result<Args> {
    parse_from(std::env::args().skip(1))
}

/// Parse CLI arguments, matching the TS omp CLI: `-p/--print` is a boolean
/// flag and the prompt comes from positional message arguments (joined with a
/// space). `@<path>` positional args attach a text file's contents to the
/// prompt; `--` ends flag parsing.
fn parse_from<I: Iterator<Item = String>>(it: I) -> Result<Args> {
    let mut a = Args {
        print: false,
        prompt: None,
        model: None,
        cwd: None,
        mode: None,
        no_session: false,
        no_extensions: false,
        extension: Vec::new(),
        thinking: None,
        system_prompt: None,
        append_system_prompt: None,
        tools_csv: None,
        dump_system_prompt: false,
    };
    let mut messages: Vec<String> = Vec::new();
    let mut it = it.peekable();
    let mut flags_done = false;
    while let Some(arg) = it.next() {
        if flags_done {
            messages.push(arg);
            continue;
        }
        if arg == "--" {
            flags_done = true;
            continue;
        }
        if !arg.starts_with('-') || arg == "-" {
            push_message(&mut messages, arg)?;
            continue;
        }
        let mut value = |name: &str| {
            it.next()
                .ok_or_else(|| anyhow::anyhow!("{name} needs a value"))
        };
        match arg.as_str() {
            "-p" | "--print" => a.print = true,
            "--model" => a.model = Some(value("--model")?),
            "--cwd" => a.cwd = Some(PathBuf::from(value("--cwd")?)),
            "--mode" => a.mode = Some(value("--mode")?),
            "--thinking" => a.thinking = Some(value("--thinking")?),
            "--extension" => a.extension.push(PathBuf::from(value("--extension")?)),
            "--no-session" => a.no_session = true,
            "--system-prompt" => a.system_prompt = Some(value("--system-prompt")?),
            "--append-system-prompt" => {
                a.append_system_prompt = Some(value("--append-system-prompt")?);
            }
            "--tools" => a.tools_csv = Some(value("--tools")?),
            "--dump-system-prompt" => a.dump_system_prompt = true,
            "--no-extensions" => a.no_extensions = true,
            "-h" | "--help" => {
                println!(
                    "usage: omp-rs -p [--model provider/id[:thinking]] [--cwd DIR]\n       [--thinking off|minimal|low|medium|high|xhigh|max] [--extension <path>]...\n       [--no-extensions] [--no-session] [--] <prompt...|@file>\n       omp-rs --mode rpc [--model ...] [--cwd DIR]"
                );
                std::process::exit(0);
            }
            other => bail!("unknown argument: {other}"),
        }
    }
    if !messages.is_empty() {
        a.prompt = Some(messages.join(" "));
    }
    Ok(a)
}

/// Append a positional message, expanding `@<path>` text-file attachments the
/// way the TS CLI combines them into the initial prompt.
fn push_message(messages: &mut Vec<String>, arg: String) -> Result<()> {
    if let Some(path) = arg.strip_prefix('@') {
        if !path.is_empty() && std::path::Path::new(path).is_file() {
            let text = std::fs::read_to_string(path)
                .map_err(|e| anyhow::anyhow!("cannot read @{path}: {e}"))?;
            messages.push(text);
            return Ok(());
        }
    }
    messages.push(arg);
    Ok(())
}

#[tokio::main]
#[allow(clippy::too_many_lines)] // argv parse + dispatch; splitting would only add hops
async fn main() -> Result<()> {
    let args = parse_args()?;
    let cwd = match args.cwd {
        Some(c) => c.canonicalize()?,
        None => std::env::current_dir()?,
    };
    // System prompt options are assembled before model selection so
    // `--dump-system-prompt` can run without a provider.
    let sp_opts = sysprompt::Options {
        system_prompt: args.system_prompt.clone(),
        append_system_prompt: args.append_system_prompt.clone(),
        tools_csv: args.tools_csv.clone(),
        subagent: false,
        model: args.model.clone().unwrap_or_default(),
        generated_append: Vec::new(),
    };
    if args.dump_system_prompt {
        let metas: Vec<sysprompt::ToolMeta> = tools::builtin_tools(
            &cwd,
            tools::edit_mode_for(sp_opts.model.split('/').next().unwrap_or("")),
        )
        .iter()
        .map(|t| (t.name().to_string(), t.name().to_string()))
        .collect();
        let built = sysprompt::build(&cwd, &sp_opts, &metas);
        for (i, block) in built.blocks.iter().enumerate() {
            if i > 0 {
                println!();
            }
            println!("{block}");
        }
        return Ok(());
    }
    let mut selected = model::select(args.model.as_deref()).await?;
    if let Some(level) = args.thinking.as_deref() {
        selected.thinking = runtime::parse_level(level)
            .ok_or_else(|| anyhow::anyhow!("unknown thinking level: {level}"))?;
    }

    // Extension host: hooks + tools feed agent construction, so it starts first.
    let mut extensions: Option<std::sync::Arc<ext::Extensions>> = None;
    if !args.no_extensions {
        match ext::Extensions::start(&cwd, &args.extension).await {
            Ok(e) => extensions = Some(e),
            Err(e) => eprintln!("omp-rs: extensions disabled: {e:#}"),
        }
    }
    let extra_tools = match extensions.as_ref() {
        Some(e) => e.tools().await.unwrap_or_default(),
        None => Vec::new(),
    };
    let (agent, target) =
        agent::build_agent(&cwd, selected, extra_tools, extensions.as_ref(), &sp_opts)?;

    let session = persist::attach_with(&agent, &cwd, args.no_session).await;
    if let (Some(e), Ok(p)) = (extensions.as_ref(), &session) {
        let model = agent.state().await.model;
        let model_ref = format!("{}/{}", model.provider, model.id);
        if let Err(err) = e
            .set_session_ctx(&p.session_id(), p.file().as_deref(), &model_ref)
            .await
        {
            eprintln!("omp-rs: extension ctx: {err:#}");
        }
    }

    if args.mode.as_deref() == Some("rpc") {
        let persistence = session.map_err(|e| anyhow::anyhow!("session file: {e:#}"))?;
        let result = match extensions.as_ref() {
            Some(e) => rpc::run_with(agent, target, cwd, persistence, Some(e.rpc_on_start())).await,
            None => rpc::run(agent, target, cwd, persistence).await,
        };
        if let Some(e) = &extensions {
            e.shutdown().await;
        }
        return result;
    }
    if let Err(e) = &session {
        eprintln!("omp-rs: session persistence disabled: {e:#}");
    }
    // omp-rs has no interactive TUI: a positional prompt already implies
    // headless mode (as does an explicit -p), matching the TS CLI's non-TTY path.
    let headless = args.print || args.prompt.is_some();
    let Some(prompt) = args.prompt else {
        if let Some(e) = &extensions {
            e.shutdown().await;
        }
        let _ = headless;
        bail!("nothing to do: pass a prompt (e.g. -p \"...\") or --mode rpc");
    };

    let messages = if let Ok(p) = session {
        if let Some(e) = extensions.as_ref() {
            e.session_start().await;
        }
        if let Some(f) = p.file() {
            eprintln!("session: {}", f.display());
        }
        match extensions.as_ref() {
            Some(e) => run_print_with_extensions(agent, target, cwd, p, &prompt, Some(e)).await?,
            None => runtime::run_print(agent, target, cwd, p, &prompt).await?,
        }
    } else {
        let prompt = match extensions.as_ref() {
            Some(e) => match e.before_prompt(&agent, &prompt).await {
                Ok(ext::PromptGate::Run(text)) => text,
                Ok(ext::PromptGate::Handled) => String::new(),
                Err(e) => {
                    eprintln!("omp-rs: extension input: {e:#}");
                    prompt.clone()
                }
            },
            None => prompt.clone(),
        };
        if !prompt.is_empty() {
            agent.prompt(prompt.as_str()).await?;
        }
        agent.state().await.messages
    };
    if let Some(e) = &extensions {
        e.shutdown().await;
    }
    match agent::last_assistant(&messages) {
        Some((_, Some(err))) => {
            eprintln!("error: {err}");
            std::process::exit(1);
        }
        Some((text, None)) => println!("{text}"),
        None => bail!("no assistant response"),
    }
    Ok(())
}

/// Print-mode run with the extension host attached: runtime hook points
/// (serve loop, lifecycle forwarding) plus the pre-prompt lifecycle events.
async fn run_print_with_extensions(
    agent: pa_agent::agent::Agent,
    target: agent::Target,
    cwd: PathBuf,
    persist: persist::Persistence,
    prompt: &str,
    extensions: Option<&std::sync::Arc<ext::Extensions>>,
) -> Result<Vec<pa_agent::types::AgentMessage>> {
    let agent = std::sync::Arc::new(agent);
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    drop(rx);
    let rt = runtime::Runtime::start(agent.clone(), target, cwd, persist, tx).await;
    if let Some(e) = extensions {
        e.attach(rt.clone());
        let prompt = match e.before_prompt(&agent, prompt).await {
            Ok(ext::PromptGate::Run(text)) => text,
            Ok(ext::PromptGate::Handled) => String::new(),
            Err(e) => {
                eprintln!("omp-rs: extension input: {e:#}");
                prompt.to_string()
            }
        };
        if !prompt.is_empty() {
            rt.prompt(None, &prompt, None, Vec::new())
                .await
                .map_err(|e| anyhow::anyhow!(e.message))?;
        }
    } else {
        rt.prompt(None, prompt, None, Vec::new())
            .await
            .map_err(|e| anyhow::anyhow!(e.message))?;
    }
    rt.wait_settled().await;
    Ok(agent.state().await.messages)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(argv: &[&str]) -> Result<Args> {
        parse_from(argv.iter().copied().map(ToString::to_string))
    }

    /// TS parity: `-p` is a boolean flag; the prompt is positional, so flags
    /// may follow it (`omp-rs -p --no-session --model X "do X"`).
    #[test]
    fn print_flag_is_boolean_and_prompt_is_positional() {
        let a = parse(&["-p", "--no-session", "--model", "zai/glm-5.3-flash", "do X"]).unwrap();
        assert!(a.print);
        assert!(a.no_session);
        assert_eq!(a.model.as_deref(), Some("zai/glm-5.3-flash"));
        assert_eq!(a.prompt.as_deref(), Some("do X"));
    }

    /// Old `-p "prompt"` invocation parses identically under flag semantics.
    #[test]
    fn print_flag_with_direct_prompt_still_works() {
        let a = parse(&["-p", "hello world"]).unwrap();
        assert!(a.print);
        assert_eq!(a.prompt.as_deref(), Some("hello world"));
    }

    #[test]
    fn multiple_positional_messages_are_joined() {
        let a = parse(&["-p", "first", "second"]).unwrap();
        assert_eq!(a.prompt.as_deref(), Some("first second"));
    }

    #[test]
    fn at_file_expands_into_prompt() {
        let dir = std::env::temp_dir();
        let f = dir.join(format!("omp-rs-parse-test-{}", std::process::id()));
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
    fn unknown_flag_still_errors() {
        assert!(parse(&["--nosession", "x"]).is_err());
    }
}
