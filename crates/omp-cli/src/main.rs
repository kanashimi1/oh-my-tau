//! omp-rs: Rust port of the omp coding agent (print + rpc modes).

mod adapter;
mod agent;
mod events;
mod model;
mod paging;
mod persist;
mod rpc;
mod runtime;
mod tools;

use std::path::PathBuf;

use anyhow::{bail, Result};

struct Args {
    prompt: Option<String>,
    model: Option<String>,
    cwd: Option<PathBuf>,
    mode: Option<String>,
    no_session: bool,
}

fn parse_args() -> Result<Args> {
    let mut a = Args { prompt: None, model: None, cwd: None, mode: None, no_session: false };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut value = |name: &str| it.next().ok_or_else(|| anyhow::anyhow!("{name} needs a value"));
        match arg.as_str() {
            "-p" | "--print" => a.prompt = Some(value("-p")?),
            "--model" => a.model = Some(value("--model")?),
            "--cwd" => a.cwd = Some(PathBuf::from(value("--cwd")?)),
            "--mode" => a.mode = Some(value("--mode")?),
            "--no-session" => a.no_session = true,
            "-h" | "--help" => {
                println!("usage: omp-rs -p <prompt> [--model provider/id[:thinking]] [--cwd DIR]\n       omp-rs --mode rpc [--model ...] [--cwd DIR]");
                std::process::exit(0);
            }
            other => bail!("unknown argument: {other}"),
        }
    }
    Ok(a)
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = parse_args()?;
    let cwd = match args.cwd {
        Some(c) => c.canonicalize()?,
        None => std::env::current_dir()?,
    };
    let selected = model::select(args.model.as_deref())?;
    let (agent, target) = agent::build_agent(&cwd, selected)?;

    let session = persist::attach_with(&agent, &cwd, args.no_session).await;
    if args.mode.as_deref() == Some("rpc") {
        let persistence = session.map_err(|e| anyhow::anyhow!("session file: {e:#}"))?;
        return rpc::run(agent, target, cwd, persistence).await;
    }
    if let Err(e) = &session {
        eprintln!("omp-rs: session persistence disabled: {e:#}");
    }
    let Some(prompt) = args.prompt else {
        bail!("nothing to do: pass -p <prompt> or --mode rpc");
    };
    let messages = match session {
        Ok(p) => {
            if let Some(f) = p.file() {
                eprintln!("session: {}", f.display());
            }
            runtime::run_print(agent, target, cwd, p, &prompt).await?
        }
        Err(_) => {
            agent.prompt(prompt.as_str()).await?;
            agent.state().await.messages
        }
    };
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
