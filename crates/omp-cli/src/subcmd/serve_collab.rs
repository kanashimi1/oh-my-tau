//! `omp-rs serve-collab` — run the offline collab relay and serve the
//! collab-web browser client on one port (TS `local-relay.ts` +
//! `serve_collab_client`). The relay is what `/collab` hosts connect to and
//! what `http://127.0.0.1:<port>/#<link>` deep links open against.
//!
//! v1: serving the client and relay is enough. A host needs a live session,
//! so no host is started here; run `/collab` inside an interactive omp-rs
//! for that.

use std::io::Write as _;
use std::path::PathBuf;

use anyhow::{bail, Context as _, Result};

const USAGE: &str = "Usage: omt serve-collab [--port <port>] [--host <host>]";
const DEFAULT_PORT: u16 = 7466;

/// TS `parseInt(value, 10)`: leading digits are the port; trailing junk is
/// ignored (mirrors `subcmd/stats.rs::parse_port`).
fn parse_port(value: &str) -> Result<u16> {
    let digits: String = value
        .trim_start()
        .chars()
        .take_while(char::is_ascii_digit)
        .collect();
    if digits.is_empty() {
        bail!("Invalid port: {value}");
    }
    digits
        .parse::<u16>()
        .map_err(|_| anyhow::anyhow!("Invalid port: {value}"))
}

/// Parse `omp-rs serve-collab` flags (`-p/--port`, `--host`, `--port=`,
/// `--host=`).
///
/// # Errors
/// Unknown flag, missing value, or an invalid port.
pub fn parse_args(args: &[String]) -> Result<(u16, String)> {
    let mut port = DEFAULT_PORT;
    let mut host = "127.0.0.1".to_string();
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-p" | "--port" => port = parse_port(it.next().context("--port needs a value")?)?,
            "--host" => it
                .next()
                .context("--host needs a value")?
                .clone_into(&mut host),
            other if other.starts_with("--port=") => port = parse_port(&other["--port=".len()..])?,
            other if other.starts_with("--host=") => host = other["--host=".len()..].to_string(),
            other if !other.starts_with('-') || other == "-" => {}
            other => bail!("Unknown option: {other}. {USAGE}"),
        }
    }
    Ok((port, host))
}

/// Locate the collab-web static `dist` (the relay serves it so `curl /`
/// returns HTML). Prefers the vendored build; falls back to a minimal
/// placeholder page when it is absent.
fn dist_dir() -> PathBuf {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let vendored = repo_root.join("vendor-src/oh-my-pi/packages/collab-web/dist");
    if vendored.join("index.html").is_file() {
        return vendored;
    }
    // Fallback: a temp dir with a placeholder page so the port still serves
    // HTML for smoke tests.
    let dir = std::env::temp_dir().join(format!("omp-collab-web-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    if let Ok(mut f) = std::fs::File::create(dir.join("index.html")) {
        let _ = f.write_all(
            b"<!doctype html><html><body><h1>omp collab-web</h1>\
              <p>vendored dist not found; run serve-collab from the omp-rs repo.</p>\
              </body></html>",
        );
    }
    dir
}

/// Entry point: `omp-rs serve-collab [options]`.
///
/// # Errors
/// Bad flags, a non-localhost `--host`, or relay bind/start failure.
pub fn run(args: &[String]) -> Result<i32> {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("{USAGE}");
        return Ok(0);
    }
    let (port, host) = parse_args(args)?;
    if host != "127.0.0.1" && host != "localhost" {
        bail!("serve-collab binds 127.0.0.1 only (got --host {host})");
    }
    let dir = dist_dir();
    // `run` is called from inside the CLI's runtime: the relay setup and the
    // Ctrl+C wait must block on a dedicated thread (stats.rs pattern), not on
    // a runtime worker.
    std::thread::spawn(move || -> Result<i32> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .context("create tokio runtime")?;
        let relay = rt
            .block_on(async { omp_collab::serve_collab_client(port, &dir) })
            .context("failed to start the collab relay + web client")?;
        println!("Collab relay + web client on http://127.0.0.1:{port}/");
        println!("Relay socket: {}", relay.url);
        println!("Hosts connect here; open a link (http://127.0.0.1:{port}/#<collab-link>) in a browser.");
        println!("Press Ctrl+C to stop\n");
        if let Err(e) = rt.block_on(tokio::signal::ctrl_c()) {
            bail!("failed to await Ctrl+C: {e}");
        }
        relay.stop();
        println!("\nShutting down.");
        Ok(0)
    })
    .join()
    .map_err(|_| anyhow::anyhow!("collab relay thread panicked"))?
}
