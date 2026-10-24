//! omp-rs: Rust port of the omp coding agent (print + rpc modes).

mod adapter;
mod agent;
mod approval;
mod cli;
mod compact;
mod context;
mod context_policy;
mod dedup;
mod events;
mod ext;
mod herdr;
mod images;
mod methodology;
mod model;
mod notices;
mod paging;
mod persist;
mod rpc;
mod runtime;
mod slash_extra;
pub mod subcmd;
mod sysprompt;
mod title;
mod tools;
mod tree;
mod tui;

use std::path::{Path, PathBuf};

use anyhow::{bail, Result};
use cli::{Args, Resume};

// Allocation strategy, measured (AllocMem, Tails reports):
// glibc malloc retains the cross-thread per-turn churn in per-thread arenas;
// mimalloc cut that (197 -> 162 MB/session). It still keeps large per-turn
// wire-context transients committed (~0.15 MB/turn growth, Tails report) — open.
#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

const HELP: &str = r"usage: omt [--mode text|json|rpc] [-p|--print] [--model provider/id[:thinking]]
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

Session control (bare --resume in a terminal opens the session picker; headless
modes need a session id prefix or path):
  -c, --continue       continue the newest previous session
  -r, --resume [ID]    resume a session (also --session)
  --fork ID            fork a session into a fresh file

Flags parsed for TS parity whose subsystems are not in omt (no-op or
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

/// Directory sessions are stored in/looked up from, honoring --session-dir.
fn sessions_dir(cwd: &Path, session_dir: Option<&Path>) -> Result<PathBuf> {
    if let Some(d) = session_dir {
        return Ok(d.to_path_buf());
    }
    persist::sessions_dir(cwd).ok_or_else(|| anyhow::anyhow!("HOME is not set"))
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
            .and_then(|v| {
                v.get("id")
                    .and_then(serde_json::Value::as_str)
                    .map(ToString::to_string)
            });
        if id.is_some_and(|id| id.starts_with(value)) {
            return Some(path);
        }
    }
    None
}

/// Resolve the session-source flags (--resume/--session/--continue/--fork) to
/// a session file to load, or None to start fresh.
///
/// `ts_dir` is TS omp's session dir for the same folder (absent with
/// `--session-dir`): ids/the picker also search it; `--continue` stays on the
/// omp-rs store.
fn resolve_resume_target(
    resume: Option<&Resume>,
    continue_prev: bool,
    fork: Option<&str>,
    dir: &Path,
    ts_dir: Option<&Path>,
) -> Result<Option<PathBuf>> {
    let resolve = |value: &str| {
        resolve_session(dir, value).or_else(|| ts_dir.and_then(|d| resolve_session(d, value)))
    };
    if let Some(fork) = fork {
        return match resolve(fork) {
            Some(p) => Ok(Some(p)),
            None => bail!(
                "Session \"{fork}\" not found.\nRun `omp --resume` without an argument to pick from recent sessions, or `omp` to start a new one."
            ),
        };
    }
    match resume {
        Some(Resume::Id(value)) => match resolve(value) {
            Some(p) => Ok(Some(p)),
            None => bail!(
                "Session \"{value}\" not found.\nRun `omp --resume` without an argument to pick from recent sessions, or `omp` to start a new one."
            ),
        },
        Some(Resume::Yes) => {
                // TS opens the session picker (apps/session-picker.ts);
                // non-TTY keeps the headless error.
                if std::io::IsTerminal::is_terminal(&std::io::stdin())
                    && std::io::IsTerminal::is_terminal(&std::io::stdout())
                {
                    use omp_tui::overlays::sessions::SessionSource;
                    let mut sources = vec![SessionSource::new(dir)];
                    if let Some(ts) = ts_dir {
                        sources.push(SessionSource::labeled(ts, "ts"));
                    }
                    match omp_tui::overlays::sessions::select_session_standalone(
                        &sources,
                        "Resume session",
                    ) {
                        Some(p) => Ok(Some(p)),
                        None => bail!("resume cancelled"),
                    }
                } else {
                    bail!(
                        "omt: bare --resume needs the interactive session picker; pass a session id prefix or path"
                    )
                }
            }
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
        eprintln!("omt: allocator-tuning re-exec failed: {error}");
    }
}

#[allow(clippy::too_many_lines)] // sequential startup guard clauses; mirrors omp's main flow
fn main() -> Result<()> {
    // The rpc/print loop is strictly sequential; a multi-thread runtime costs
    // ~20-30 ms CPU/turn in scheduler overhead (measured, RPC12 team run).
    // Run those modes on a current-thread runtime; interactive TUI keeps the
    // multi-thread default (spawns, lsp, ext-host IO overlap).
    let mut raw: Vec<std::ffi::OsString> = std::env::args_os().collect();
    raw.remove(0);
    let sequential = raw
        .iter()
        .any(|a| a == "--mode" || a.to_string_lossy().starts_with("--mode="));
    let rt = if sequential {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
    } else {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
    }
    .expect("tokio runtime");
    rt.block_on(async_main())
}

async fn async_main() -> Result<()> {
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
        println!("omt/{}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }
    // Flags of subsystems omp-rs does not have: omp's exact error where one
    // exists, otherwise an explicit unsupported message.
    if args.goal.is_some() {
        return Err(usage_err(
            "--goal requires an interactive terminal (not --print or --mode).",
        ));
    }
    if args.alias.is_some() && args.profile.is_none() && std::env::var_os("OMP_PROFILE").is_none() {
        return Err(usage_err(
            "--alias requires --profile <name> or OMP_PROFILE",
        ));
    }
    if args.profile.is_some() {
        bail!("omt: --profile is not supported (no profile subsystem)");
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
        bail!("omt: --from-claude/--from-codex are not supported (no importer subsystem)");
    }
    if let Some(mode @ ("acp" | "rpc-ui")) = args.mode.as_deref() {
        bail!("omt: --mode {mode} is not supported (no {mode} subsystem)");
    }
    if !args.config.is_empty() || !args.add_dir.is_empty() {
        eprintln!(
            "omt: --config/--add-dir parsed but their subsystems are not in omt; ignored"
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
                eprintln!(
                    "omt: started in ~; switching to a temp dir (pass --allow-home to stay)"
                );
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
    for (flag, value) in [
        ("--smol", &args.smol),
        ("--slow", &args.slow),
        ("--plan", &args.plan),
    ] {
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
        let metas: Vec<sysprompt::ToolMeta> = tools::session_builtin_tools(
            &cwd,
            tools::edit_mode_for(sp_opts.model.split('/').next().unwrap_or("")),
            &omp_settings::Settings::load(&cwd),
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
    if reference.is_some() {
        runtime::MODEL_PINNED.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    if args.thinking.is_some() {
        runtime::THINKING_PINNED.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    let mut selected = model::select(reference.as_deref()).await?;
    if !selected.fallbacks.is_empty() {
        let primary = format!("{}/{}", selected.model.provider, selected.model.id);
        let _ = runtime::LAUNCH_FALLBACKS.set((primary, std::mem::take(&mut selected.fallbacks)));
    }
    if let Some(key) = args.api_key.clone() {
        selected.api_key = Some(key);
        selected.credential_id = None;
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
            Err(e) => eprintln!("omt: extensions disabled: {e:#}"),
        }
    }
    let extra_tools = match extensions.as_ref() {
        Some(e) => e.tools().await.unwrap_or_default(),
        None => Vec::new(),
    };
    let context_strategy = context::Kind::resolve(
        args.context_strategy.as_deref(),
        &omp_settings::Settings::load(&cwd),
    )
    .map_err(|e| anyhow::anyhow!(e))?;
    context::set_active(context_strategy);
    approval::Gate::install(
        args.approval_mode.as_deref().and_then(approval::parse_mode),
        args.auto_approve,
        &cwd,
        extensions.as_ref(),
    );
    let (agent, target) = agent::build_agent(
        &cwd,
        selected,
        extra_tools,
        extensions.as_ref(),
        &sp_opts,
        context_strategy,
    )?;
    // eval's in-cell bridge: `tool.<name>` dispatches to the session's tools, and
    // completion()/judge() need the live provider target.
    crate::tools::eval_bridge::set_session_tools(agent.tools().await);
    crate::tools::eval_agents::set_runtime_hooks(target.clone());

    let session = persist::attach_with(&agent, &cwd, args.no_session).await;
    // Resolve internal URLs (local://, artifact://, agent://, history://)
    // against this session's artifacts dir.
    if let Ok(p) = &session {
        omp_urls::set_global_session_file(p.file());
    }
    if let (Some(e), Ok(p)) = (extensions.as_ref(), &session) {
        let model = agent.model().await;
        let model_ref = format!("{}/{}", model.provider, model.id);
        if let Err(err) = e
            .set_session_ctx(&p.session_id(), p.file().as_deref(), &model_ref)
            .await
        {
            eprintln!("omt: extension ctx: {err:#}");
        }
    }

    // Session-source flags: --resume/--session/--continue/--fork load a
    // previous transcript; --session-dir relocates the session store.
    let resume_target = if args.mode.as_deref() == Some("rpc") {
        None // the rpc runtime owns session lifecycle
    } else {
        let dir = sessions_dir(&cwd, args.session_dir.as_deref())?;
        std::fs::create_dir_all(&dir).ok();
        let ts_dir = if args.session_dir.is_some() {
            None
        } else {
            persist::ts_sessions_dir(&cwd)
        };
        let target = resolve_resume_target(
            args.resume.as_ref(),
            args.continue_prev,
            args.fork.as_deref(),
            &dir,
            ts_dir.as_deref(),
        )?;
        // With a resume/fork target `load_session` binds the right file itself (and drops
        // the startup file); only a fresh run relocates to the new store here.
        if args.session_dir.is_some() && target.is_none() {
            if let Ok(p) = &session {
                p.discard_pristine();
                p.start_new(&cwd, args.session_dir.as_deref())?;
            }
        }
        target
    };
    // Explicit --model/--thinking only outrank the session's own model for the
    // startup resume; without one, later /resume and switch_session restore.
    if resume_target.is_none() {
        runtime::MODEL_PINNED.store(false, std::sync::atomic::Ordering::Relaxed);
        runtime::THINKING_PINNED.store(false, std::sync::atomic::Ordering::Relaxed);
    }

    if args.mode.as_deref() == Some("rpc") {
        let persistence = session.map_err(|e| anyhow::anyhow!("session file: {e:#}"))?;
        if let Some(e) = extensions.as_ref() {
            crate::rpc::slash::set_extensions(e.clone());
        }
        // TS `hasUI`: RPC clients answer dialogs unless started with `--no-ui`.
        if !args.no_ui {
            crate::slash_extra::mark_interactive_ui();
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
        eprintln!("omt: session persistence disabled: {e:#}");
    }
    // Interactive TUI: TTY on stdin+stdout, no -p/--mode/prompt — TS enters
    // `interactive-mode.ts` on exactly this condition. `--resume`/`--continue`
    // bring their resolved session file along; `--fork` stays headless for
    // now.
    if args.mode.is_none()
        && !args.print
        && args.prompt.is_none()
        && args.fork.is_none()
        && std::io::IsTerminal::is_terminal(&std::io::stdin())
        && std::io::IsTerminal::is_terminal(&std::io::stdout())
    {
        let persistence = session.map_err(|e| anyhow::anyhow!("session file: {e:#}"))?;
        let result = tui::run(
            agent,
            target,
            cwd,
            persistence,
            resume_target,
            extensions.clone(),
        )
        .await;
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
        args.fork.is_some(),
        args.session_dir.clone(),
        prompt,
        images,
        extensions.as_ref(),
        if args.mode.as_deref() == Some("json") {
            ext::ExtMode::Json
        } else {
            ext::ExtMode::Print
        },
    );
    let messages = if let Some(secs) = args.max_time {
        if let Ok(r) = tokio::time::timeout(std::time::Duration::from_secs_f64(secs), run).await {
            r?
        } else {
            if let Some(e) = &extensions {
                e.shutdown().await;
            }
            eprintln!(
                "omt: stopped after --max-time {:?}",
                args.max_time_raw.as_deref().unwrap_or("")
            );
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
        None if PRINTED_COMMAND_OUTPUT.load(std::sync::atomic::Ordering::Relaxed) => {}
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
    use pa_agent::types::AgentMessage;
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
    // Herdr: an exit() bypasses Runtime's Drop — release the pane here so it
    // clears instead of waiting for the idle-shell safety net. Bounded, no-op
    // unless HERDR_ENV=1.
    if let Some(h) = herdr::global() {
        h.release_blocking();
    }
    shutdown_lsp();
    std::process::exit(code);
}

/// Print-mode run: extension lifecycle hook points, optional session resume
/// (`--resume`/`--continue`/`--fork`), then the single prompt.
#[allow(clippy::too_many_arguments)] // print-mode run context; mirrors omp's runPrint signature
/// Set when print mode answered a handled slash command with its output
/// (no model turn ran, so "no assistant response" is not an error).
static PRINTED_COMMAND_OUTPUT: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

async fn run_print(
    agent: pa_agent::agent::Agent,
    target: agent::Target,
    cwd: PathBuf,
    persist: Option<persist::Persistence>,
    resume: Option<PathBuf>,
    fork: bool,
    session_dir: Option<PathBuf>,
    prompt: String,
    images: Vec<pa_agent::types::ImageContent>,
    extensions: Option<&std::sync::Arc<ext::Extensions>>,
    mode: ext::ExtMode,
) -> Result<Vec<pa_agent::types::AgentMessage>> {
    let Some(persist) = persist else {
        // No session persistence: replaying a transcript is impossible; a
        // direct one-shot prompt matches the old ephemeral behavior.
        if resume.is_some() {
            eprintln!("omt: resume ignored: session persistence unavailable");
        }
        if !images.is_empty() {
            eprintln!("omt: image attachments require session persistence; dropped");
        }
        let mut injected = Vec::new();
        if let Some(e) = extensions {
            e.set_mode(mode);
            e.session_start().await;
            // No runtime here (no session file): the TS `before_agent_start` still precedes the run.
            injected = e
                .before_agent_start(&agent, &prompt, None)
                .await
                .unwrap_or_else(|err| {
                    eprintln!("omt: extension before_agent_start: {err:#}");
                    Vec::new()
                });
        }
        if !prompt.is_empty() {
            agent
                .prompt(runtime::prompt_input(
                    &prompt,
                    Vec::new(),
                    Vec::new(),
                    injected,
                ))
                .await?;
        }
        return Ok(agent.messages_snapshot().await.into_vec());
    };
    if let Some(f) = persist.file() {
        eprintln!("session: {}", f.display());
    }
    let agent = std::sync::Arc::new(agent);
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<serde_json::Value>();
    let rt = runtime::Runtime::start(agent.clone(), target, cwd, persist, tx).await;
    if let Some(path) = resume {
        // `--resume`/`--continue` keep appending to the same session file; `--fork`
        // writes a new one (in the `--session-dir` store when relocated).
        if fork {
            rt.fork_session(&path, session_dir.as_deref()).await?;
        } else {
            rt.load_session(&path, session_dir.as_deref()).await?;
        }
    }
    if let Some(e) = extensions {
        e.attach_session(rt.clone(), mode).await;
    }
    // Slash-command dispatch in print mode (TS runs these through the same
    // dispatcher as the TUI); a matched command handles or rewrites the prompt.
    let mut prompt = prompt;
    match crate::rpc::slash::try_dispatch(&rt, &prompt).await {
        crate::rpc::slash::Outcome::Handled { .. }
        | crate::rpc::slash::Outcome::ExtensionHandled => {
            prompt = String::new();
            // A handled command reports through `command_output` frames, which
            // in print mode have no UI to land in: print them, so e.g.
            // `omp-rs -p "/rlm stats"` shows its result instead of
            // "no assistant response".
            let mut printed = false;
            while let Ok(frame) = rx.try_recv() {
                if frame.get("type").and_then(serde_json::Value::as_str) == Some("command_output") {
                    if let Some(text) = frame.get("text").and_then(serde_json::Value::as_str) {
                        println!("{text}");
                        printed = true;
                    }
                }
            }
            if printed && images.is_empty() {
                rt.wait_settled().await;
                PRINTED_COMMAND_OUTPUT.store(true, std::sync::atomic::Ordering::Relaxed);
                return Ok(agent.messages_snapshot().await.into_vec());
            }
        }
        crate::rpc::slash::Outcome::Prompt(expanded) => prompt = expanded,
        crate::rpc::slash::Outcome::NotSlash => {}
    }
    // Nothing else reads the UI frame stream in print mode.
    drop(rx);
    if !prompt.is_empty() || !images.is_empty() {
        rt.prompt(None, &prompt, None, images)
            .await
            .map_err(|e| anyhow::anyhow!(e.message))?;
    }
    rt.wait_settled().await;
    // The automatic title request rides the session lifecycle (TS ties it to
    // the session, not a single turn); let it land before the process exits.
    rt.wait_for_title().await;
    Ok(agent.messages_snapshot().await.into_vec())
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
