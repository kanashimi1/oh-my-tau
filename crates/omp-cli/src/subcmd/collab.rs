//! `omp-rs collab` — local Collab host discovery and link retrieval (TS
//! `commands/collab.ts` + `cli/collab-cli.ts`, minimal port):
//! `collab list [--json]` lists live hosts via `crate::registry`; `collab link
//! <instanceId|pid> [--view] [--json]` prints one browser URL.

use std::future::Future;

use anyhow::{bail, Context as _, Result};
use omp_collab::registry::{
    list_collab_hosts, resolve_collab_host_link, CollabAccess, CollabLinkError,
    COLLAB_REGISTRY_VERSION,
};
use serde_json::json;

const USAGE: &str =
    "Usage: omt collab <list|link> ...\n  omt collab list [--json]\n  omt collab link <instanceId|pid> [--view] [--json]";

/// Registry IPC is async; this subcommand surface is sync — and `run` is
/// called from inside the CLI's runtime, so `block_on` must happen off that
/// runtime (dedicated thread, `subcmd/stats.rs` pattern).
fn block_on_registry<T>(fut: impl Future<Output = anyhow::Result<T>> + Send + 'static) -> Result<T>
where
    T: Send + 'static,
{
    std::thread::spawn(move || {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| anyhow::anyhow!("create tokio runtime: {e}"))
            .and_then(|rt| rt.block_on(fut))
    })
    .join()
    .map_err(|_| anyhow::anyhow!("collab registry thread panicked"))?
}

/// Minimal flag parse for `collab list` / `collab link`.
fn parse_list(args: &[String]) -> Result<bool> {
    let mut json_out = false;
    for a in args {
        match a.as_str() {
            "-j" | "--json" => json_out = true,
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            other => bail!("collab list accepts no {other} (usage: collab list [--json])"),
        }
    }
    Ok(json_out)
}

fn parse_link(args: &[String]) -> Result<(String, bool, bool)> {
    let mut selector = String::new();
    let mut view = false;
    let mut json_out = false;
    for a in args {
        match a.as_str() {
            "--view" => view = true,
            "-j" | "--json" => json_out = true,
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            other if other.starts_with('-') => bail!("Unknown option: {other}. {USAGE}"),
            other => {
                if selector.is_empty() {
                    selector = other.to_string();
                } else {
                    bail!("collab link takes one selector (usage: collab link <instanceId|pid> [--view])");
                }
            }
        }
    }
    if selector.is_empty() {
        bail!("collab link requires a selector (usage: collab link <instanceId|pid> [--view])");
    }
    Ok((selector, view, json_out))
}

fn print_list(json_out: bool) -> Result<()> {
    let hosts =
        block_on_registry(async { list_collab_hosts(None).await }).context("list Collab hosts")?;
    if json_out {
        println!(
            "{}",
            serde_json::to_string_pretty(
                &json!({"version": COLLAB_REGISTRY_VERSION, "hosts": hosts})
            )?
        );
        return Ok(());
    }
    if hosts.is_empty() {
        println!("No active Collab hosts.");
        return Ok(());
    }
    println!(
        "{} active Collab {}",
        hosts.len(),
        if hosts.len() == 1 { "host" } else { "hosts" }
    );
    for h in &hosts {
        let name = h.session_name.as_deref().unwrap_or("");
        let session = if name.is_empty() {
            h.session_id.clone()
        } else {
            format!("{name} ({})", h.session_id)
        };
        println!("{:<8} {session}  {}", h.instance_id, h.cwd);
        let guests = h.participants - 1;
        let mut details = vec![
            format!("pid {}", h.pid),
            format!("gen {}", h.generation),
            h.model.as_ref().map_or_else(
                || "no model".to_string(),
                |m| format!("{}/{}", m.provider, m.id),
            ),
            format!("participants {}", h.participants),
            format!("{guests} {}", if guests == 1 { "guest" } else { "guests" }),
            h.access.to_string(),
            if h.relay_connected {
                "relay connected"
            } else {
                "relay reconnecting"
            }
            .to_string(),
        ];
        if h.busy == Some(true) {
            details.push("working".to_string());
        } else if h.busy == Some(false) {
            details.push("idle".to_string());
        }
        println!("  {}", details.join(" · "));
    }
    println!("Get a link: omp collab link <instanceId|pid> [--view]");
    Ok(())
}

fn print_link(selector: &str, view: bool, json_out: bool) -> Result<i32> {
    let access = if view {
        CollabAccess::View
    } else {
        CollabAccess::Control
    };
    let selector = selector.to_string();
    match block_on_registry(async move { resolve_collab_host_link(&selector, access, None).await })
    {
        Ok(link) => {
            if json_out {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&json!({
                        "version": COLLAB_REGISTRY_VERSION,
                        "instanceId": link.instance_id,
                        "generation": link.generation,
                        "access": link.access.to_string(),
                        "url": link.url,
                    }))?
                );
            } else {
                println!("{}", link.url);
            }
            Ok(0)
        }
        Err(e) => {
            if let Some(lerr) = e.downcast_ref::<CollabLinkError>() {
                eprintln!("error: {}", lerr.message);
                Ok(1)
            } else {
                Err(e)
            }
        }
    }
}

/// Entry point: `omp-rs collab <action> [options]`.
///
/// # Errors
/// Unknown action, bad flags, or registry I/O failure.
pub fn run(args: &[String]) -> Result<i32> {
    let (action, rest) = match args.split_first() {
        Some((a, r)) => (a.as_str(), r),
        None => {
            println!("{USAGE}");
            return Ok(0);
        }
    };
    match action {
        "list" => {
            let json_out = parse_list(rest)?;
            print_list(json_out)?;
            Ok(0)
        }
        "link" => {
            let (selector, view, json_out) = parse_link(rest)?;
            print_link(&selector, view, json_out)
        }
        "-h" | "--help" => {
            println!("{USAGE}");
            Ok(0)
        }
        other => bail!("unknown collab action: {other}. {USAGE}"),
    }
}
