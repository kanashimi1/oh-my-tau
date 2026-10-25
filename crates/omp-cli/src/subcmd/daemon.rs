//! `omt daemon` — a long-lived Unix-socket listener that hands out one
//! `omt --mode rpc` agent per connection.
//!
//! # Protocol
//!
//! The daemon speaks newline-delimited JSON on `$XDG_RUNTIME_DIR/omt.sock`
//! (falling back to `/tmp/omt-<uid>.sock`). Each accepted connection becomes
//! an independent RPC session: the daemon spawns the current binary with
//! `--mode rpc` and pipes socket bytes to the child's stdin and the child's
//! stdout lines back to the socket — exactly the frames `omt --mode rpc`
//! speaks over stdio.
//!
//! Before the first rpc frame a client MAY send one hello line
//! `{"type":"hello","cwd":"/some/dir"}`; the child is then started in that
//! directory (default: the daemon's cwd). Any other first line is forwarded
//! verbatim (the child starts in the daemon's cwd).
//!
//! Control operations do not need an agent child:
//! - `omt daemon status` — report whether the daemon is running (pidfile).
//! - `omt daemon stop` — ask the daemon to shut down via a
//!   `{"type":"daemon-stop"}` line on the socket (pidfile fallback), then
//!   remove socket and pidfile.
//!
//! The daemon writes a pidfile next to the socket (`<socket>.pid`) and
//! removes socket + pidfile on exit.

use std::path::{Path, PathBuf};
use std::process::Stdio;

use anyhow::{bail, Context as _, Result};
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tokio::process::Command;

const USAGE: &str = "Usage: omt daemon [--socket <path>]

Start the omt daemon: a Unix-socket listener that spawns one
`omt --mode rpc` agent per connection.

  omt daemon [--socket <path>]   start the daemon (foreground)
  omt daemon status [--socket <path>]
  omt daemon stop   [--socket <path>]

Socket: $XDG_RUNTIME_DIR/omt.sock, else /tmp/omt-<uid>.sock.
Protocol: newline JSON; optional first line {\"type\":\"hello\",\"cwd\":..}
selects the agent's working directory, everything else is passed through
to the rpc child.";

/// First line a client sends to ask the daemon to exit.
const STOP_FRAME: &str = "{\"type\":\"daemon-stop\"}";

struct Options {
    socket: Option<PathBuf>,
    action: Action,
}

enum Action {
    Run,
    Stop,
    Status,
}

fn parse_args(args: &[String]) -> Result<Options> {
    let mut opts = Options {
        socket: None,
        action: Action::Run,
    };
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        let (flag, inline) = match arg.split_once('=') {
            Some((f, v)) if f.starts_with("--") => (f, Some(v.to_string())),
            _ => (arg.as_str(), None),
        };
        match flag {
            "--socket" => {
                let v = match inline {
                    Some(v) => v,
                    None => it.next().cloned().context("--socket needs a value")?,
                };
                opts.socket = Some(PathBuf::from(v));
            }
            "stop" => opts.action = Action::Stop,
            "status" => opts.action = Action::Status,
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            other => bail!("unknown daemon argument: {other}\n{USAGE}"),
        }
    }
    Ok(opts)
}

/// Numeric user id of this process (`/proc/self/status` on Linux, `id -u`
/// elsewhere); `None` when it cannot be determined.
fn current_uid() -> Option<u32> {
    #[cfg(target_os = "linux")]
    {
        let status = std::fs::read_to_string("/proc/self/status").ok()?;
        for line in status.lines() {
            if let Some(rest) = line.strip_prefix("Uid:") {
                return rest.split_whitespace().next()?.parse().ok();
            }
        }
        None
    }
    #[cfg(not(target_os = "linux"))]
    {
        let out = std::process::Command::new("id").arg("-u").output().ok()?;
        String::from_utf8_lossy(&out.stdout).trim().parse().ok()
    }
}

/// Default socket path: `$XDG_RUNTIME_DIR/omt.sock`, else
/// `/tmp/omt-<uid>.sock`, else `/tmp/omt.sock`.
#[must_use]
pub fn default_socket_path() -> PathBuf {
    if let Ok(dir) = std::env::var("XDG_RUNTIME_DIR") {
        if !dir.is_empty() {
            return PathBuf::from(dir).join("omt.sock");
        }
    }
    match current_uid() {
        Some(uid) => std::env::temp_dir().join(format!("omt-{uid}.sock")),
        None => std::env::temp_dir().join("omt.sock"),
    }
}

/// Pidfile lives next to the socket.
fn pidfile_path(socket: &Path) -> PathBuf {
    let mut name = socket.file_name().map_or_else(
        || "omt.sock".to_string(),
        |n| n.to_string_lossy().into_owned(),
    );
    name.push_str(".pid");
    socket.parent().unwrap_or(Path::new("/tmp")).join(name)
}

/// Whether `pid` names a live process.
fn process_alive(pid: u32) -> bool {
    #[cfg(target_os = "linux")]
    {
        Path::new("/proc").join(pid.to_string()).exists()
    }
    #[cfg(not(target_os = "linux"))]
    {
        std::process::Command::new("kill")
            .arg("-0")
            .arg(pid.to_string())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }
}

/// Read the pid recorded in `path`, if any.
fn read_pid(path: &Path) -> Option<u32> {
    let text = std::fs::read_to_string(path).ok()?;
    text.trim().parse().ok()
}

fn say(text: &str) {
    println!("{text}");
}

/// `omt daemon [start|stop|status] [--socket PATH]`.
///
/// # Errors
/// Invalid arguments or socket/pidfile failures.
pub fn run(args: &[String]) -> Result<i32> {
    run_opts(parse_args(args)?)
}

fn run_opts(opts: Options) -> Result<i32> {
    let socket = opts.socket.unwrap_or_else(default_socket_path);
    let pidfile = pidfile_path(&socket);
    match opts.action {
        Action::Status => match read_pid(&pidfile).filter(|pid| process_alive(*pid)) {
            Some(pid) => {
                say(&format!(
                    "daemon running (pid {pid}), socket {}",
                    socket.display()
                ));
                Ok(0)
            }
            None => {
                say(&format!("daemon not running (socket {})", socket.display()));
                Ok(1)
            }
        },
        Action::Stop => {
            use std::io::Write as _;
            let Some(pid) = read_pid(&pidfile) else {
                say(&format!("daemon not running (socket {})", socket.display()));
                return Ok(0);
            };
            if !process_alive(pid) {
                let _ = std::fs::remove_file(&socket);
                let _ = std::fs::remove_file(&pidfile);
                say("daemon not running (stale pidfile removed)");
                return Ok(0);
            }
            // Preferred: a graceful shutdown frame over the socket.
            let asked = std::os::unix::net::UnixStream::connect(&socket)
                .and_then(|mut s| {
                    s.write_all(STOP_FRAME.as_bytes())?;
                    s.flush()
                })
                .is_ok();
            if !asked {
                std::process::Command::new("kill")
                    .arg(pid.to_string())
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .status()
                    .ok();
            }
            // Wait briefly for the socket to disappear (daemon cleanup).
            for _ in 0..40 {
                if !socket.exists() && !process_alive(pid) {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
            if process_alive(pid) {
                say(&format!("daemon (pid {pid}) did not stop"));
                return Ok(1);
            }
            say(&format!("daemon stopped (pid {pid})"));
            Ok(0)
        }
        Action::Run => run_daemon(&socket, &pidfile),
    }
}

fn run_daemon(socket: &Path, pidfile: &Path) -> Result<i32> {
    if read_pid(pidfile).is_some_and(process_alive) {
        bail!(
            "daemon already running (pid {}), socket {}",
            read_pid(pidfile).unwrap_or(0),
            socket.display()
        );
    }
    if let Some(parent) = socket.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    // A leftover socket file would fail bind(2) with EADDRINUSE.
    let _ = std::fs::remove_file(socket);
    std::fs::write(pidfile, format!("{}\n", std::process::id()))
        .with_context(|| format!("write pidfile {}", pidfile.display()))?;

    // `subcmd::run` is called from inside main's multi-thread runtime; starting
    // another runtime there panics ("Cannot start a runtime from within a
    // runtime"). Serve on the entered runtime via `block_in_place`, falling
    // back to a scratch current-thread runtime in non-runtime contexts (tests).
    let result = match tokio::runtime::Handle::try_current() {
        Ok(handle) => {
            if handle.runtime_flavor() == tokio::runtime::RuntimeFlavor::CurrentThread {
                // `block_in_place` panics on a current-thread runtime; park the
                // work on a dedicated thread with its own runtime instead.
                let owned = socket.to_path_buf();
                std::thread::spawn(move || {
                    tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .context("tokio runtime")?
                        .block_on(serve(&owned))
                })
                .join()
                .map_err(|_| anyhow::anyhow!("daemon thread panicked"))?
            } else {
                tokio::task::block_in_place(|| handle.block_on(serve(socket)))
            }
        }
        Err(_) => tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .context("tokio runtime")?
            .block_on(serve(socket)),
    };
    let _ = std::fs::remove_file(socket);
    let _ = std::fs::remove_file(pidfile);
    say("daemon stopped");
    result.map(|()| 0)
}

async fn serve(socket: &Path) -> Result<()> {
    let listener =
        UnixListener::bind(socket).with_context(|| format!("bind {}", socket.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        // Only the current user may talk to the daemon.
        let _ = std::fs::set_permissions(socket, std::fs::Permissions::from_mode(0o600));
    }
    say(&format!(
        "omt daemon listening on {} (pid {})",
        socket.display(),
        std::process::id()
    ));
    let (stop_tx, mut stop_rx) = tokio::sync::mpsc::channel::<()>(1);
    loop {
        tokio::select! {
            accepted = listener.accept() => {
                let (stream, _) = accepted.context("accept connection")?;
                let stop_tx = stop_tx.clone();
                // Boxed: the connection future is large (8 KiB buffers).
                tokio::spawn(Box::pin(async move {
                    if let Err(e) = handle_connection(stream, stop_tx).await {
                        eprintln!("omt daemon: connection error: {e:#}");
                    }
                }));
            }
            _ = stop_rx.recv() => return Ok(()),
        }
    }
}

/// Read the first line; either a control frame for the daemon itself, a
/// hello that selects the child cwd, or the first rpc frame to forward.
/// Returns `Ok(None)` when the daemon should shut down.
async fn first_line(
    reader: &mut BufReader<tokio::net::unix::OwnedReadHalf>,
) -> Result<Option<(Option<String>, String)>> {
    let mut line = String::new();
    let n = reader.read_line(&mut line).await?;
    if n == 0 {
        return Ok(None);
    }
    let line = line.trim_end_matches(['\n', '\r']);
    let parsed: Option<Value> = serde_json::from_str(line).ok();
    if let Some(obj) = parsed.as_ref().and_then(Value::as_object) {
        match obj.get("type").and_then(Value::as_str) {
            Some("daemon-stop") => return Ok(Some((None, STOP_FRAME.to_string()))),
            Some("hello") => {
                let cwd = obj.get("cwd").and_then(Value::as_str).map(str::to_string);
                let mut next = String::new();
                let n = reader.read_line(&mut next).await?;
                if n == 0 {
                    return Ok(None);
                }
                return Ok(Some((cwd, next)));
            }
            _ => {}
        }
    }
    Ok(Some((None, format!("{line}\n"))))
}

async fn handle_connection(
    stream: UnixStream,
    stop_tx: tokio::sync::mpsc::Sender<()>,
) -> Result<()> {
    let exe = std::env::current_exe().context("current executable")?;
    let (rx, mut tx) = stream.into_split();
    let mut reader = BufReader::new(rx);

    let Some((cwd, first)) = first_line(&mut reader).await? else {
        return Ok(());
    };
    if first.trim() == STOP_FRAME.trim() {
        let _ = stop_tx.send(()).await;
        return Ok(());
    }

    let mut cmd = Command::new(&exe);
    cmd.arg("--mode").arg("rpc");
    if let Some(dir) = cwd.as_deref() {
        cmd.arg("--cwd").arg(dir);
    }
    cmd.stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .kill_on_drop(true);
    let mut child = cmd
        .spawn()
        .with_context(|| format!("spawn {} --mode rpc", exe.display()))?;
    let mut stdin = child.stdin.take().context("rpc child stdin")?;
    let mut stdout = child.stdout.take().context("rpc child stdout")?;

    // First line goes first (it may already be an rpc frame).
    stdin.write_all(first.as_bytes()).await?;
    stdin.flush().await?;

    let to_child = async {
        let mut buf = [0u8; 8192];
        loop {
            match reader.read(&mut buf).await {
                Ok(0) | Err(_) => break,
                Ok(n) => {
                    if stdin.write_all(&buf[..n]).await.is_err() {
                        break;
                    }
                    let _ = stdin.flush().await;
                }
            }
        }
        // Client hung up: end the child's stdin so it can finish its turn.
        let _ = stdin.shutdown().await;
    };
    let to_client = async {
        let mut buf = [0u8; 8192];
        loop {
            match stdout.read(&mut buf).await {
                Ok(0) | Err(_) => break,
                Ok(n) => {
                    if tx.write_all(&buf[..n]).await.is_err() {
                        break;
                    }
                }
            }
        }
    };
    tokio::select! {
        () = to_child => { let _ = child.wait().await; }
        () = to_client => { child.kill().await.ok(); }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_socket_prefers_xdg_runtime_dir() {
        // SAFETY: single-threaded test process section; restoring afterwards.
        let saved = std::env::var("XDG_RUNTIME_DIR").ok();
        std::env::set_var("XDG_RUNTIME_DIR", "/run/user/1234");
        let path = default_socket_path();
        match saved {
            Some(v) => std::env::set_var("XDG_RUNTIME_DIR", v),
            None => std::env::remove_var("XDG_RUNTIME_DIR"),
        }
        assert_eq!(path, PathBuf::from("/run/user/1234/omt.sock"));
    }

    #[test]
    fn default_socket_falls_back_to_tmp_uid() {
        let saved = std::env::var("XDG_RUNTIME_DIR").ok();
        std::env::remove_var("XDG_RUNTIME_DIR");
        let path = default_socket_path();
        match saved {
            Some(v) => std::env::set_var("XDG_RUNTIME_DIR", v),
            None => std::env::remove_var("XDG_RUNTIME_DIR"),
        }
        let name = path.file_name().map(|n| n.to_string_lossy().into_owned());
        match current_uid() {
            Some(uid) => assert_eq!(name.as_deref(), Some(format!("omt-{uid}.sock").as_str())),
            None => assert_eq!(name.as_deref(), Some("omt.sock")),
        }
        assert_eq!(
            path.parent(),
            Some(std::path::Path::new(
                std::env::temp_dir().to_str().unwrap_or("/tmp")
            ))
        );
    }

    #[test]
    fn pidfile_sits_next_to_socket() {
        assert_eq!(
            pidfile_path(Path::new("/run/user/9/omt.sock")),
            PathBuf::from("/run/user/9/omt.sock.pid")
        );
        // No parent component: falls back next to the bare name.
        assert_eq!(
            pidfile_path(Path::new("omt.sock")),
            PathBuf::from("omt.sock.pid")
        );
    }

    #[test]
    fn parse_args_routes_actions_and_socket() {
        let opts = parse_args(&["status".to_string()]).unwrap();
        assert!(matches!(opts.action, Action::Status));
        assert!(opts.socket.is_none());

        let opts = parse_args(&["--socket=/x/y.sock".to_string(), "stop".to_string()]).unwrap();
        assert!(matches!(opts.action, Action::Stop));
        assert_eq!(opts.socket, Some(PathBuf::from("/x/y.sock")));

        let opts = parse_args(&["--socket".to_string(), "/z.sock".to_string()]).unwrap();
        assert!(matches!(opts.action, Action::Run));
        assert_eq!(opts.socket, Some(PathBuf::from("/z.sock")));

        assert!(parse_args(&["--nope".to_string()]).is_err());
    }
}
