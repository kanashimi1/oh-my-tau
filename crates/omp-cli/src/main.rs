//! omp-rs: Rust port of the omp coding agent (print + rpc modes).

mod adapter;
mod agent;
mod cli;
mod compact;
mod events;
mod ext;
mod images;
mod model;
mod paging;
mod persist;
mod rpc;
mod runtime;
pub mod subcmd;
mod sysprompt;
mod title;
mod tools;
mod tree;
mod tui;

use std::path::{Path, PathBuf};

use anyhow::{bail, Result};
use cli::{Args, Resume};

// Allocation strategy, measured (runs/omp-rust/reports/AllocMem.md, Tails.md):
// glibc malloc retains the cross-thread per-turn churn in per-thread arenas;
// mimalloc cut that (197 -> 162 MB/session). It still keeps large per-turn
// wire-context transients committed (~0.15 MB/turn growth, Tails.md) — open.
#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

const HELP: &str = r"usage: omp-rs [--mode text|json|rpc] [-p|--print] [--model provider/id[:thinking]]
       [--provider <name>] [--api-key <key>] [--smol <m>] [--slow <m>] [--plan <m>]
       [--cwd DIR] [--add-dir DIR]... [--config FILE]... [--session-dir DIR]
       [-c|--continue] [-r|--resume [ID]] [--fork ID] [--from-claude] [--from-codex]
       [--thinking off|minimal|low|medium|high|xhigh|max|auto] [--tools a,b] [--no-tools]
       [--models a,b] [--max-time 600|10m|1h] [--print-thoughts] [--hide-thinking]
       [--system-prompt TEXT] [--system-prompt-template FILE] [--append-system-prompt TEXT]
       [-e|--extension PATH]... [--hook PATH]... [--trusted-extension ABS]...
       [--no-extensions] [--no-session] [--no-skills] [--no-rules] [--no-title]
       [--no-lsp] [--no-pty] [--no-ui] [--allow-home] [--auto-approve|--yolo]
       [--approval-mode always-ask|write|yolo] [--export FILE] [--version] [--] <prompt...|@file>

Session control (headless: pass a session id prefix or path; bare --resume needs
the interactive picker, which omp-rs does not have):
  -c, --continue       continue the newest previous session
  -r, --resume [ID]    resume a session (also --session)
  --fork ID            fork a session into a fresh file

Flags parsed for TS parity whose subsystems are not in omp-rs (no-op or
reported, see GapInventory): --config, --add-dir, --smol/--slow/--plan,
--models, --service-tier, --prewalk*/--plan-yolo*, --skills/--no-skills,
--no-rules, --no-title, --no-lsp, --no-pty, --advisor, --external-thinking,
--hide-thinking, --auto-approve/--approval-mode, --profile, --alias.
";

fn parse_args() -> Result<Args> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    // CLI subcommands (TS cli-commands.ts): a leading non-flag token naming a
    // known subcommand dispatches before flag parsing.
    if let Some(name) = argv.first().filter(|a| !a.starts_with('-')) {
        if subcmd::known(name) {
            match subcmd::run(name, &argv[1..]) {
                Ok(code) => exit(code),
                Err(e) => {
                    eprintln!("Error: {e:#}");
                    exit(1);
                }
            }
        }
    }
    cli::parse_from(argv.into_iter())
}

/// Directory sessions are stored in/looked up from, honoring --session-dir
/// (mirrors `persist::sessions_dir`, which is private to that module).
fn sessions_dir(cwd: &Path, session_dir: Option<&Path>) -> Result<PathBuf> {
    if let Some(d) = session_dir {
        return Ok(d.to_path_buf());
    }
    let home = std::env::var_os("HOME").ok_or_else(|| anyhow::anyhow!("HOME is not set"))?;
    let slug = cwd.display().to_string().replace('/', "-");
    Ok(PathBuf::from(home).join(".omp-rs/sessions").join(slug))
}

/// Resolve a `--resume ID`/`--fork ID` value to a session file: an existing
/// path wins; otherwise match the session id prefix inside `dir`.
fn resolve_session(dir: &Path, value: &str) -> Option<PathBuf> {
    let as_path = PathBuf::from(value);
    if as_path.is_file() {
        return Some(as_path);
    }
    let entries = std::fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
            continue;
        }
        let Ok(first) = std::fs::read_to_string(&path) else {
            continue;
        };
        let id = first
            .lines()
            .take(5)
            .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
            .find(|v| v.get("type").and_then(serde_json::Value::as_str) == Some("session"))
            .and_then(|v| v.get("id").and_then(serde_json::Value::as_str).map(ToString::to_string));
        if id.is_some_and(|id| id.starts_with(value)) {
            return Some(path);
        }
    }
    None
}

/// Resolve the session-source flags (--resume/--session/--continue/--fork) to
/// a session file to load, or None to start fresh.
fn resolve_resume_target(
    resume: Option<&Resume>,
    continue_prev: bool,
    fork: Option<&str>,
    dir: &Path,
) -> Result<Option<PathBuf>> {
    if let Some(fork) = fork {
        return match resolve_session(dir, fork) {
            Some(p) => Ok(Some(p)),
            None => bail!(
                "Session \"{fork}\" not found.\nRun `omp --resume` without an argument to pick from recent sessions, or `omp` to start a new one."
            ),
        };
    }
    match resume {
        Some(Resume::Id(value)) => match resolve_session(dir, value) {
            Some(p) => Ok(Some(p)),
            None => bail!(
                "Session \"{value}\" not found.\nRun `omp --resume` without an argument to pick from recent sessions, or `omp` to start a new one."
            ),
        },
        Some(Resume::Yes) => bail!(
            "omp-rs: bare --resume needs the interactive session picker; pass a session id prefix or path"
        ),
        None => {
            if continue_prev {
                // TS starts a fresh session when there is no previous one.
                Ok(runtime::newest_session(dir))
            } else {
                Ok(None)
            }
        }
    }
}

fn usage_err(msg: &str) -> anyhow::Error {
    anyhow::anyhow!("{msg}\nRun `omp --help` for available flags.")
}

/// mimalloc reads `MIMALLOC_*` options once at process load (`_mi_options_init`),
/// before `main` runs, so the tuning env must exist before exec. When unset,
/// set it and re-exec ourselves (`exec`, same pid/stdio/argv); the marker env
/// stops the recursion. The default 10s purge delay keeps the large per-turn
/// wire-context transients (request body, payload trees) committed in
/// mimalloc segments — measured as ~0.15 MB/turn linear RSS growth
/// (Tails.md, GrowthFix.md). A zero delay returns fully-free segments to the
/// OS immediately. An explicit user `MIMALLOC_PURGE_DELAY` or the
/// `OMP_RS_NO_MEM_REEXEC` opt-out skips the re-exec.
fn ensure_mimalloc_env() {
    if std::env::var_os("MIMALLOC_PURGE_DELAY").is_some()
        || std::env::var_os("OMP_RS_NO_MEM_REEXEC").is_some()
    {
        return;
    }
    std::env::set_var("MIMALLOC_PURGE_DELAY", "0");
    // Cross-heap collection every N generic allocations: per-turn large
    // wire buffers are allocated/freed across tokio worker threads, and
    // their segments otherwise stay committed in the owning thread's cache
    // (measured ~0.06 MB/turn residual after purge alone; GrowthFix.md).
    std::env::set_var("MIMALLOC_GENERIC_COLLECT", "100");
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        let exe = std::env::current_exe().unwrap_or_else(|_| "/proc/self/exe".into());
        let args: Vec<String> = std::env::args().skip(1).collect();
        let error = std::process::Command::new(exe).args(&args).exec();
        // `exec` only returns on failure; keep running without the tuning
        // rather than refuse to start.
        eprintln!("omp-rs: allocator-tuning re-exec failed: {error}");
    }
}

#[allow(clippy::too_many_lines)] // sequential startup guard clauses; mirrors omp's main flow
#[tokio::main]
async fn main() -> Result<()> {
    ensure_mimalloc_env();
    let mut args = parse_args()?;
    if args.no_title {
        // TS `PI_NO_TITLE`: read by the auto-title hook in `runtime::spawn_auto_title`.
        std::env::set_var("OMP_RS_NO_TITLE", "1");
    }
    if args.help {
        print!("{HELP}");
        return Ok(());
    }
    if args.version {
        println!("omp-rs/{}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }
    // Flags of subsystems omp-rs does not have: omp's exact error where one
    // exists, otherwise an explicit unsupported message.
    if args.goal.is_some() {
        return Err(usage_err("--goal requires an interactive terminal (not --print or --mode)."));
    }
    if args.alias.is_some() && args.profile.is_none() && std::env::var_os("OMP_PROFILE").is_none() {
        return Err(usage_err("--alias requires --profile <name> or OMP_PROFILE"));
    }
    if args.profile.is_some() {
        bail!("omp-rs: --profile is not supported (no profile subsystem)");
    }
    if let Some(path) = args.export.clone() {
        if !Path::new(&path).is_file() {
            // omp's exact error for a missing export source.
            bail!("File not found: {path}");
        }
        // TS exports the session to HTML and exits; the `export` subcommand
        // owns the exporter.
        return match subcmd::run("export", &[path]) {
            Ok(code) => {
                exit(code);
            }
            Err(e) => Err(e),
        };
    }
    if args.from_claude || args.from_codex {
        bail!("omp-rs: --from-claude/--from-codex are not supported (no importer subsystem)");
    }
    if let Some(mode @ ("acp" | "rpc-ui")) = args.mode.as_deref() {
        bail!("omp-rs: --mode {mode} is not supported (no {mode} subsystem)");
    }
    if !args.config.is_empty() || !args.add_dir.is_empty() {
        eprintln!(
            "omp-rs: --config/--add-dir parsed but their subsystems are not in omp-rs; ignored"
        );
    }

    let mut cwd = match args.cwd {
        Some(c) => c.canonicalize()?,
        None => std::env::current_dir()?,
    };
    // TS guard: launching in $HOME without --allow-home switches to a temp dir
    // so sessions and tool runs never treat the home directory as a project.
    if !args.allow_home {
        if let Some(home) = std::env::var_os("HOME").map(PathBuf::from) {
            if cwd == home.canonicalize().unwrap_or(home) {
                eprintln!("omp-rs: started in ~; switching to a temp dir (pass --allow-home to stay)");
                cwd = std::env::temp_dir();
            }
        }
    }
    let add_dirs: Vec<PathBuf> = args
        .add_dir
        .iter()
        .map(|d| d.canonicalize().unwrap_or_else(|_| d.clone()))
        .collect();
    drop(add_dirs); // parsed for parity; no workspace-subsystem consumer yet

    // Role model selection: expose omp's PI_*_MODEL env contract so any
    // role-aware subsystem (task/prewalk/plan ports) picks these up.
    for (flag, value) in [("--smol", &args.smol), ("--slow", &args.slow), ("--plan", &args.plan)] {
        if let Some(v) = value {
            let key = format!(
                "OMP_RS_{}_MODEL",
                flag.trim_start_matches("--").to_uppercase()
            );
            std::env::set_var(key, v);
        }
    }

    // System prompt options are assembled before model selection so
    // `--dump-system-prompt` can run without a provider.
    let tools_csv = if args.no_tools {
        // Restrict to nothing: the --tools filter retains only named tools.
        Some("__none__".to_string())
    } else {
        args.tools_csv.clone()
    };
    let sp_opts = sysprompt::Options {
        system_prompt: args.system_prompt.clone(),
        append_system_prompt: args.append_system_prompt.clone(),
        tools_csv,
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

    // Model reference: --model wins; a bare id is prefixed with --provider
    // (omp's legacy `--provider` flag); --provider alone names the provider.
    let reference = match (&args.model, &args.provider) {
        (Some(m), Some(p)) if !m.contains('/') => Some(format!("{p}/{m}")),
        (Some(m), _) => Some(m.clone()),
        (None, Some(p)) => Some(p.clone()),
        (None, None) => None,
    };
    let mut selected = model::select(reference.as_deref()).await?;
    if let Some(key) = args.api_key.clone() {
        selected.api_key = Some(key);
    }
    if let Some(level) = args.thinking.as_deref() {
        if level != "auto" {
            selected.thinking = runtime::parse_level(level)
                .ok_or_else(|| anyhow::anyhow!("unknown thinking level: {level}"))?;
        }
    }

    // Extension host: hooks + tools feed agent construction, so it starts first.
    let mut extensions: Option<std::sync::Arc<ext::Extensions>> = None;
    if !args.no_extensions {
        let extra: Vec<PathBuf> = args
            .extension
            .iter()
            .cloned()
            .chain(args.hooks.iter().map(PathBuf::from))
            .collect();
        match ext::Extensions::start(&cwd, &extra).await {
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
    // Resolve internal URLs (local://, artifact://, agent://, history://)
    // against this session's artifacts dir.
    if let Ok(p) = &session {
        omp_urls::set_global_session_file(p.file());
    }
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

    // Session-source flags: --resume/--session/--continue/--fork load a
    // previous transcript; --session-dir relocates the session store.
    let resume_target = if args.mode.as_deref() == Some("rpc") {
        None // the rpc runtime owns session lifecycle
    } else {
        let dir = sessions_dir(&cwd, args.session_dir.as_deref())?;
        std::fs::create_dir_all(&dir).ok();
        let target = resolve_resume_target(
            args.resume.as_ref(),
            args.continue_prev,
            args.fork.as_deref(),
            &dir,
        )?;
        if args.session_dir.is_some() {
            if let Ok(p) = &session {
                p.start_new(&cwd, args.session_dir.as_deref())?;
            }
        }
        target
    };

    if args.mode.as_deref() == Some("rpc") {
        let persistence = session.map_err(|e| anyhow::anyhow!("session file: {e:#}"))?;
        if let Some(e) = extensions.as_ref() {
            crate::rpc::slash::set_extensions(e.clone());
        }
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
    // Interactive TUI: TTY on stdin+stdout, no -p/--mode/prompt, nothing to
    // resume — TS enters `interactive-mode.ts` on exactly this condition.
    // Session resume in the TUI is a follow-up (the picker overlay), so
    // resume flags keep falling through to the headless guard below.
    if args.mode.is_none()
        && !args.print
        && args.prompt.is_none()
        && args.resume.is_none()
        && !args.continue_prev
        && args.fork.is_none()
        && std::io::IsTerminal::is_terminal(&std::io::stdin())
        && std::io::IsTerminal::is_terminal(&std::io::stdout())
    {
        let persistence = session.map_err(|e| anyhow::anyhow!("session file: {e:#}"))?;
        let result = tui::run(agent, target, cwd, persistence).await;
        if let Some(e) = &extensions {
            e.shutdown().await;
        }
        return result;
    }
    // A positional prompt already implies headless mode (as does an explicit
    // -p), matching the TS CLI's non-TTY path.
    let headless = args.print || args.prompt.is_some();
    let Some(prompt) = args.prompt else {
        if let Some(e) = &extensions {
            e.shutdown().await;
        }
        let _ = headless;
        bail!("nothing to do: pass a prompt (e.g. -p \"...\") or --mode rpc");
    };
    let session = session.ok();
    let session_path = session.as_ref().map(persist::Persistence::path);
    let mode_json = args.mode.as_deref() == Some("json");
    let print_thoughts = args.print_thoughts;
    let images = std::mem::take(&mut args.images);

    let run = run_print(
        agent,
        target,
        cwd,
        session,
        resume_target,
        args.session_dir.clone(),
        prompt,
        images,
        extensions.as_ref(),
    );
    let messages = if let Some(secs) = args.max_time {
        if let Ok(r) = tokio::time::timeout(std::time::Duration::from_secs_f64(secs), run).await { r? } else {
            if let Some(e) = &extensions {
                e.shutdown().await;
            }
            eprintln!("omp-rs: stopped after --max-time {:?}", args.max_time_raw.as_deref().unwrap_or(""));
            exit(124);
        }
    } else {
        run.await?
    };
    if let Some(e) = &extensions {
        e.shutdown().await;
    }

    if mode_json {
        // TS json mode: the session transcript as JSON lines on stdout.
        if let Some(p) = &session_path {
            if let Ok(text) = std::fs::read_to_string(p) {
                print!("{text}");
            }
        }
        shutdown_lsp();
        return Ok(());
    }
    match agent::last_assistant(&messages) {
        Some((_, Some(err))) => {
            eprintln!("error: {err}");
            exit(1);
        }
        Some((text, None)) => {
            if print_thoughts {
                if let Some(thinking) = last_thinking(&messages) {
                    println!("{thinking}");
                }
            }
            println!("{text}");
        }
        None => {
            shutdown_lsp();
            bail!("no assistant response");
        }
    }
    shutdown_lsp();
    Ok(())
}

/// Thinking blocks of the last assistant message (`--print-thoughts`).
fn last_thinking(messages: &[pa_agent::types::AgentMessage]) -> Option<String> {
    use pa_agent::types::{AgentMessage};
    use pa_agent::types::{AssistantContent, Message};
    messages.iter().rev().find_map(|m| match m {
        AgentMessage::Standard(Message::Assistant(a)) => {
            let thinking = a
                .content
                .iter()
                .filter_map(|c| match c {
                    AssistantContent::Thinking(t) => Some(t.thinking.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n");
            (!thinking.is_empty()).then_some(thinking)
        }
        _ => None,
    })
}

/// TS `postmortem.register("lsp-shutdown", …)`: the LSP client registry is a
/// process static (it never drops), so `shutdown_all` must run explicitly on
/// every exit path instead of relying on `kill_on_drop`.
fn shutdown_lsp() {
    // Exit paths run inside the main tokio runtime: parking the worker via
    // `block_in_place` and driving the shutdown on the entered runtime (a
    // fresh runtime's block_on would panic with "Cannot start a runtime from
    // within a runtime"). From non-runtime contexts (tests) fall back to a
    // scratch current-thread runtime.
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => tokio::task::block_in_place(|| {
            handle.block_on(omp_lsp::client::shutdown_all());
        }),
        Err(_) => {
            let _ = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map(|rt| rt.block_on(omp_lsp::client::shutdown_all()));
        }
    }
}

/// `shutdown_lsp` + `process::exit` (the only way main may terminate).
fn exit(code: i32) -> ! {
    shutdown_lsp();
    std::process::exit(code);
}

/// Print-mode run: extension lifecycle hook points, optional session resume
/// (`--resume`/`--continue`/`--fork`), then the single prompt.
#[allow(clippy::too_many_arguments)] // print-mode run context; mirrors omp's runPrint signature
async fn run_print(
    agent: pa_agent::agent::Agent,
    target: agent::Target,
    cwd: PathBuf,
    persist: Option<persist::Persistence>,
    resume: Option<PathBuf>,
    session_dir: Option<PathBuf>,
    prompt: String,
    images: Vec<pa_agent::types::ImageContent>,
    extensions: Option<&std::sync::Arc<ext::Extensions>>,
) -> Result<Vec<pa_agent::types::AgentMessage>> {
    let Some(persist) = persist else {
        // No session persistence: replaying a transcript is impossible; a
        // direct one-shot prompt matches the old ephemeral behavior.
        if resume.is_some() {
            eprintln!("omp-rs: resume ignored: session persistence unavailable");
        }
        if !images.is_empty() {
            eprintln!("omp-rs: image attachments require session persistence; dropped");
        }
        let prompt = match extensions {
            Some(e) => match e.before_prompt(&agent, &prompt).await {
                Ok(ext::PromptGate::Run(text)) => text,
                Ok(ext::PromptGate::Handled) => String::new(),
                Err(e) => {
                    eprintln!("omp-rs: extension input: {e:#}");
                    prompt.clone()
                }
            },
            None => prompt,
        };
        if !prompt.is_empty() {
            agent.prompt(prompt.as_str()).await?;
        }
        return Ok(agent.state().await.messages);
    };
    if let Some(e) = extensions {
        e.session_start().await;
    }
    if let Some(f) = persist.file() {
        eprintln!("session: {}", f.display());
    }
    let agent = std::sync::Arc::new(agent);
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    drop(rx);
    let rt = runtime::Runtime::start(agent.clone(), target, cwd, persist, tx).await;
    if let Some(e) = extensions {
        e.attach(rt.clone());
        crate::rpc::slash::set_extensions(e.clone());
        // Every title change re-sends the ext ctx with the new sessionName so
        // hosts reading `getSessionName()` (herdr-pane-names) see it at once.
        let ext = e.clone();
        rt.set_name_listener(Some(std::sync::Arc::new(move |name: &str| {
            let ext = ext.clone();
            let name = name.to_string();
            tokio::spawn(async move {
                let _ = ext.set_session_name(Some(name)).await;
            });
        })));
    }
    if let Some(path) = resume {
        // `--session-dir` relocates the session store: the resumed transcript
        // continues in the same dir the launch already switched to (without
        // this, load_session's start_new fell back to the default store).
        rt.load_session(&path, session_dir.as_deref()).await?;
    }
    // Slash-command dispatch in print mode (TS runs these through the same
    // dispatcher as the TUI); a matched command handles or rewrites the prompt.
    let mut prompt = prompt;
    match crate::rpc::slash::try_dispatch(&rt, &prompt).await {
        crate::rpc::slash::Outcome::Handled { .. } | crate::rpc::slash::Outcome::ExtensionHandled => {
            prompt = String::new();
        }
        crate::rpc::slash::Outcome::Prompt(expanded) => prompt = expanded,
        crate::rpc::slash::Outcome::NotSlash => {}
    }
    let prompt = match extensions {
        Some(e) => match e.before_prompt(&agent, &prompt).await {
            Ok(ext::PromptGate::Run(text)) => text,
            Ok(ext::PromptGate::Handled) => String::new(),
            Err(e) => {
                eprintln!("omp-rs: extension input: {e:#}");
                prompt.clone()
            }
        },
        None => prompt,
    };
    if !prompt.is_empty() || !images.is_empty() {
        rt.prompt(None, &prompt, None, images)
            .await
            .map_err(|e| anyhow::anyhow!(e.message))?;
    }
    rt.wait_settled().await;
    // The automatic title request rides the session lifecycle (TS ties it to
    // the session, not a single turn); let it land before the process exits.
    rt.wait_for_title().await;
    Ok(agent.state().await.messages)
}

#[cfg(test)]
mod tests {
    /// Review #2: the exit helper's shutdown leg must complete cleanly with an
    /// empty registry (a no-op, but proves the runtime dance is sound).
    #[test]
    fn shutdown_lsp_runs_without_a_runtime_client() {
        super::shutdown_lsp();
    }
}
