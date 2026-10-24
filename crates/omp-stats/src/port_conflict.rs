//! Port-conflict handling for the stats dashboard (port of `port-conflict.ts`).
//!
//! Before binding, the dashboard probes the port: a live dashboard stamping the
//! identity headers with the current security version is *reused*, an older one
//! is *replaced*, and anything else holding the port is only stopped when it
//! identifies as an omp stats process. Holder discovery and the stale-process
//! kill are Linux-only (`/proc` plus `kill`); the probe is a hand-written
//! HTTP/1.1 GET over `std::net`, so no HTTP client is pulled in.

use std::io::{ErrorKind, Read, Write};
use std::net::{TcpListener, TcpStream, ToSocketAddrs};
use std::process::{Command, Stdio};
use std::sync::LazyLock;
use std::time::Duration;

use anyhow::{anyhow, bail, Result};
use regex::Regex;

/// Header stamped on every dashboard response so reuse probes can identify us.
pub const STATS_DASHBOARD_HEADER: &str = "x-omp-stats-dashboard";
/// Header recording the server's requested bind host so reuse cannot change its exposure scope.
pub const STATS_DASHBOARD_HOSTNAME_HEADER: &str = "x-omp-stats-hostname";
/// Identity-header value for dashboards enforcing an explicit bind host and same-origin access.
pub const STATS_DASHBOARD_SECURITY_VERSION: &str = "3";
/// IPv4 loopback address shared by the dashboard server and reuse probe.
pub const STATS_DASHBOARD_HOSTNAME: &str = "127.0.0.1";

/// Numeric twin of [`STATS_DASHBOARD_SECURITY_VERSION`] for probe comparisons.
const SECURITY_VERSION_NUMBER: i64 = 3;
const STATS_PROBE_TIMEOUT_MS: u64 = 500;
const PROCESS_EXIT_POLL_MS: u64 = 50;
const PROCESS_EXIT_POLLS: usize = 10;
const STATS_PROBE_PATH: &str = "/api/stats/models";
const STATS_PROBE_MAX_BYTES: usize = 64 * 1024;
const STATS_RUNTIME_IMAGES: [&str; 4] = ["bun", "node", "omp", "omp-stats"];

/// What the server should do with the port after the preflight.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PortAction {
    /// Bind (or re-bind) the port.
    Retry,
    /// A compatible dashboard already serves this port; leave it alone.
    Reuse,
}

impl PortAction {
    /// The action name the TS port returns (`"retry"` / `"reuse"`).
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Retry => "retry",
            Self::Reuse => "reuse",
        }
    }
}

/// The process holding the port, as far as `/proc` can tell us.
#[derive(Debug, Clone)]
struct PortHolder {
    pid: u32,
    image: String,
    command_line: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StatsDashboardProbe {
    Reusable,
    Replaceable,
    Occupied,
    Unreachable,
}

/// `true` when nothing is listening on `port` right now.
#[must_use]
pub fn is_port_free(port: u16, hostname: &str) -> bool {
    TcpListener::bind((normalize_host(hostname), port)).is_ok()
}

/// Reuse a secure dashboard or reclaim an insecure HTTP dashboard before binding.
/// The preflight is needed on platforms that permit wildcard and loopback-specific
/// listeners to coexist on one port.
pub fn prepare_stats_port(port: u16, hostname: &str) -> Result<PortAction> {
    if port == 0 {
        return Ok(PortAction::Retry);
    }
    match probe_stats_dashboard(port, hostname) {
        StatsDashboardProbe::Reusable => Ok(PortAction::Reuse),
        StatsDashboardProbe::Replaceable => reclaim_stats_port(port, true),
        StatsDashboardProbe::Occupied => reclaim_stats_port(port, false),
        StatsDashboardProbe::Unreachable => Ok(PortAction::Retry),
    }
}

/// [`prepare_stats_port`] on the default loopback host.
pub fn prepare_stats_port_on_loopback(port: u16) -> Result<PortAction> {
    prepare_stats_port(port, STATS_DASHBOARD_HOSTNAME)
}

/// Reuse or reclaim a listener found after the server bind reports EADDRINUSE.
pub fn recover_stats_port(port: u16, hostname: &str) -> Result<PortAction> {
    match probe_stats_dashboard(port, hostname) {
        StatsDashboardProbe::Reusable => Ok(PortAction::Reuse),
        probe => reclaim_stats_port(port, probe == StatsDashboardProbe::Replaceable),
    }
}

/// [`recover_stats_port`] on the default loopback host.
pub fn recover_stats_port_on_loopback(port: u16) -> Result<PortAction> {
    recover_stats_port(port, STATS_DASHBOARD_HOSTNAME)
}

fn probe_stats_dashboard(port: u16, hostname: &str) -> StatsDashboardProbe {
    // The identity header carries the normalized host (see `start_server`), so compare that.
    let hostname = normalize_host(hostname);
    let probe_hostname = match hostname {
        "0.0.0.0" => STATS_DASHBOARD_HOSTNAME,
        "::" => "::1",
        other => other,
    };
    // Every failure mode of the TS `fetch` (refused, timed out, junk reply) is "unreachable".
    let Some(response) = http_probe(port, probe_hostname) else {
        return StatsDashboardProbe::Unreachable;
    };
    let version_header = response.header(STATS_DASHBOARD_HEADER);
    let dashboard_version = parse_dashboard_version(version_header);
    // Never replace a newer dashboard: an older CLI must not downgrade it.
    let replaceable = response.status == 200
        && matches!(dashboard_version, Some(version) if version > 0 && version <= SECURITY_VERSION_NUMBER);
    let reusable = response.status == 200
        && version_header == Some(STATS_DASHBOARD_SECURITY_VERSION)
        && response.header(STATS_DASHBOARD_HOSTNAME_HEADER) == Some(hostname)
        && !response.has_header("access-control-allow-origin");
    if reusable {
        return StatsDashboardProbe::Reusable;
    }
    if replaceable {
        StatsDashboardProbe::Replaceable
    } else {
        StatsDashboardProbe::Occupied
    }
}

fn parse_dashboard_version(value: Option<&str>) -> Option<i64> {
    let value = value?.trim();
    if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    value.parse::<i64>().ok()
}

struct ProbeResponse {
    status: u16,
    headers: Vec<(String, String)>,
}

impl ProbeResponse {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value.as_str())
    }

    fn has_header(&self, name: &str) -> bool {
        self.headers.iter().any(|(key, _)| key == name)
    }
}

/// `GET STATS_PROBE_PATH` written by hand; `None` on any dial/IO/protocol failure.
fn http_probe(port: u16, hostname: &str) -> Option<ProbeResponse> {
    let deadline = Duration::from_millis(STATS_PROBE_TIMEOUT_MS);
    let mut stream = (normalize_host(hostname), port)
        .to_socket_addrs()
        .ok()?
        .filter_map(|address| TcpStream::connect_timeout(&address, deadline).ok())
        .next()?;
    stream.set_read_timeout(Some(deadline)).ok()?;
    stream.set_write_timeout(Some(deadline)).ok()?;

    let authority = if hostname.contains(':') {
        format!("[{hostname}]:{port}")
    } else {
        format!("{hostname}:{port}")
    };
    let request = format!(
        "GET {STATS_PROBE_PATH} HTTP/1.1\r\nHost: {authority}\r\nUser-Agent: omp-stats-probe\r\nAccept: */*\r\nConnection: close\r\n\r\n"
    );
    stream.write_all(request.as_bytes()).ok()?;
    parse_response(&read_until_headers(&mut stream)?)
}

fn read_until_headers(stream: &mut TcpStream) -> Option<Vec<u8>> {
    let mut buffer = Vec::with_capacity(1024);
    let mut chunk = [0u8; 1024];
    loop {
        match stream.read(&mut chunk) {
            Ok(0) => break,
            Ok(read) => buffer.extend_from_slice(&chunk[..read]),
            // A timeout is the probe deadline expiring; the header check below decides.
            Err(err) if matches!(err.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => break,
            Err(_) => return None,
        }
        if let Some(end) = find_header_end(&buffer) {
            buffer.truncate(end);
            return Some(buffer);
        }
        if buffer.len() >= STATS_PROBE_MAX_BYTES {
            break;
        }
    }
    let end = find_header_end(&buffer)?;
    buffer.truncate(end);
    Some(buffer)
}

fn find_header_end(buffer: &[u8]) -> Option<usize> {
    buffer
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .map(|index| index + 4)
}

fn parse_response(raw: &[u8]) -> Option<ProbeResponse> {
    let text = String::from_utf8_lossy(raw);
    let mut lines = text.split("\r\n");
    let status_line = lines.next().unwrap_or_default();
    if !status_line.starts_with("HTTP/") {
        return None;
    }
    let status = status_line.split_whitespace().nth(1)?.parse::<u16>().ok()?;
    let headers = lines
        .filter_map(|line| line.split_once(':'))
        .map(|(name, value)| (name.trim().to_ascii_lowercase(), value.trim().to_owned()))
        .collect();
    Some(ProbeResponse { status, headers })
}

/// Bare host of a `--host` value: exactly one `[`…`]` pair is removed, so
/// `[::1]` and `::1` name the same listener. Other strings pass through.
pub(crate) fn normalize_host(hostname: &str) -> &str {
    hostname
        .strip_prefix('[')
        .and_then(|host| host.strip_suffix(']'))
        .unwrap_or(hostname)
}

fn reclaim_stats_port(port: u16, has_dashboard_identity: bool) -> Result<PortAction> {
    let Some(holder) = find_port_holder(port) else {
        bail!("Port {port} is in use, but the listening process could not be identified.");
    };
    if holder.pid == std::process::id() {
        bail!(
            "Port {port} is held by the current process ({}, PID {}).",
            holder.image,
            holder.pid
        );
    }

    let normalized_image = normalize_image(&holder.image);
    let normalized_command = holder.command_line.to_lowercase().replace('\\', "/");
    let has_stats_identity = normalized_image == "omp-stats"
        || IDENTITY.omp_stats_binary.is_match(&normalized_command)
        || IDENTITY.stats_index_ts.is_match(&normalized_command)
        || (normalized_image == "omp"
            && IDENTITY.omp_stats_subcommand.is_match(&normalized_command))
        || IDENTITY.omp_binary_stats.is_match(&normalized_command);
    if !STATS_RUNTIME_IMAGES.contains(&normalized_image.as_str())
        || (!has_stats_identity && !has_dashboard_identity)
    {
        bail!(
            "Port {port} is in use by {} (PID {}), which is not identifiable as an omp stats dashboard; refusing to stop it.",
            holder.image,
            holder.pid
        );
    }

    terminate_port_holder(&holder)?;
    Ok(PortAction::Retry)
}

struct StatsIdentity {
    omp_stats_binary: Regex,
    stats_index_ts: Regex,
    omp_stats_subcommand: Regex,
    omp_binary_stats: Regex,
}

static IDENTITY: LazyLock<StatsIdentity> = LazyLock::new(|| StatsIdentity {
    omp_stats_binary: Regex::new(r#"(?:^|[/"'\s])omp-stats(?:\.exe)?(?:["'\s]|$)"#)
        .expect("omp-stats binary pattern"),
    stats_index_ts: Regex::new(r#"/packages/stats/src/index\.ts(?:["'\s]|$)"#)
        .expect("stats index pattern"),
    omp_stats_subcommand: Regex::new(r"(?:^|\s)stats(?:\s|$)").expect("stats subcommand pattern"),
    omp_binary_stats: Regex::new(r#"(?:^|/)omp(?:\.exe)?["'\s]+stats(?:["'\s]|$)"#)
        .expect("omp stats subcommand pattern"),
});

fn normalize_image(image: &str) -> String {
    let lowered = image.to_lowercase();
    let without_exe = lowered.strip_suffix(".exe").unwrap_or(&lowered);
    without_exe
        .strip_suffix(" (deleted)")
        .unwrap_or(without_exe)
        .to_owned()
}

/// SIGTERM, poll for the exit, then SIGKILL. The `kill` binary stands in for libc
/// so the crate needs no new dependency.
fn terminate_port_holder(holder: &PortHolder) -> Result<()> {
    if !process_alive(holder.pid) {
        return Ok(());
    }
    signal_pid(holder.pid, "-TERM").map_err(|err| {
        anyhow!(
            "Failed to stop {} (PID {}): {err}",
            holder.image,
            holder.pid
        )
    })?;
    for _ in 0..PROCESS_EXIT_POLLS {
        std::thread::sleep(Duration::from_millis(PROCESS_EXIT_POLL_MS));
        if !process_alive(holder.pid) {
            return Ok(());
        }
    }
    signal_pid(holder.pid, "-KILL").map_err(|err| {
        anyhow!(
            "Failed to kill {} (PID {}): {err}",
            holder.image,
            holder.pid
        )
    })?;
    std::thread::sleep(Duration::from_millis(PROCESS_EXIT_POLL_MS));
    Ok(())
}

/// The Rust stand-in for `process.kill(pid, 0)`.
fn process_alive(pid: u32) -> bool {
    #[cfg(target_os = "linux")]
    {
        std::path::Path::new(&format!("/proc/{pid}")).exists()
    }
    #[cfg(not(target_os = "linux"))]
    {
        matches!(Command::new("kill").arg("-0").arg(pid.to_string()).output(), Ok(out) if out.status.success())
    }
}

fn signal_pid(pid: u32, signal: &str) -> Result<()> {
    let status = Command::new("kill")
        .arg(signal)
        .arg(pid.to_string())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()?;
    // ESRCH (already gone) is not an error; anything else is.
    if status.success() || !process_alive(pid) {
        return Ok(());
    }
    bail!("`kill {signal} {pid}` exited with {status}");
}

fn find_port_holder(port: u16) -> Option<PortHolder> {
    #[cfg(target_os = "linux")]
    {
        find_linux_port_holder(port)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = port;
        None
    }
}

#[cfg(target_os = "linux")]
fn find_linux_port_holder(port: u16) -> Option<PortHolder> {
    use std::collections::HashSet;

    let mut inodes: HashSet<String> = HashSet::new();
    for table in ["/proc/net/tcp", "/proc/net/tcp6"] {
        let Ok(contents) = std::fs::read_to_string(table) else {
            continue;
        };
        for line in contents.lines().skip(1) {
            let fields: Vec<&str> = line.split_whitespace().collect();
            let (Some(address), Some(state), Some(inode)) =
                (fields.get(1), fields.get(3), fields.get(9))
            else {
                continue;
            };
            if *state != "0A" {
                continue;
            }
            let Some(encoded_port) = address.rsplit(':').next() else {
                continue;
            };
            if u16::from_str_radix(encoded_port, 16).ok() == Some(port) {
                inodes.insert((*inode).to_owned());
            }
        }
    }
    if inodes.is_empty() {
        return None;
    }

    for entry in std::fs::read_dir("/proc").ok()? {
        let Ok(entry) = entry else { continue };
        let file_name = entry.file_name();
        let Some(name) = file_name.to_str() else {
            continue;
        };
        if !name.bytes().all(|byte| byte.is_ascii_digit()) || !entry.path().is_dir() {
            continue;
        }
        let Ok(pid) = name.parse::<u32>() else {
            continue;
        };
        let Ok(descriptors) = std::fs::read_dir(format!("/proc/{pid}/fd")) else {
            continue;
        };

        let mut owns_socket = false;
        for descriptor in descriptors.flatten() {
            let Ok(target) = std::fs::read_link(descriptor.path()) else {
                continue;
            };
            let link = target.to_string_lossy().into_owned();
            let Some(inode) = SOCKET_LINK.captures(&link).and_then(|caps| caps.get(1)) else {
                continue;
            };
            if inodes.contains(inode.as_str()) {
                owns_socket = true;
                break;
            }
        }
        if !owns_socket {
            continue;
        }

        let command_line = std::fs::read(format!("/proc/{pid}/cmdline"))
            .map(|raw| {
                String::from_utf8_lossy(&raw)
                    .split('\0')
                    .filter(|part| !part.is_empty())
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .unwrap_or_default();
        let image = std::fs::read_link(format!("/proc/{pid}/exe"))
            .ok()
            .map(|executable| basename(&executable.to_string_lossy()))
            .or_else(|| command_line.split(' ').next().map(basename))
            .filter(|name| !name.is_empty())
            .unwrap_or_else(|| "unknown".to_owned());
        return Some(PortHolder {
            pid,
            image,
            command_line,
        });
    }
    None
}

#[cfg(target_os = "linux")]
static SOCKET_LINK: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^socket:\[(\d+)\]$").expect("socket link pattern"));

#[cfg(target_os = "linux")]
fn basename(path: &str) -> String {
    path.rsplit('/')
        .find(|part| !part.is_empty())
        .unwrap_or_default()
        .to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Binds an ephemeral loopback port and answers `response` to `connections` probes.
    fn start_server(response: &'static str, connections: usize) -> u16 {
        start_server_on(STATS_DASHBOARD_HOSTNAME, response, connections)
    }

    fn start_server_on(host: &str, response: &'static str, connections: usize) -> u16 {
        let listener = TcpListener::bind((host, 0)).expect("bind test server");
        let port = listener.local_addr().expect("local addr").port();
        std::thread::spawn(move || {
            for _ in 0..connections {
                let Ok((mut stream, _)) = listener.accept() else {
                    return;
                };
                let mut sink = [0u8; 2048];
                let _ = stream.read(&mut sink);
                let _ = stream.write_all(response.as_bytes());
                let _ = stream.flush();
            }
        });
        port
    }

    #[test]
    fn free_port_is_free_and_prepare_retries() {
        let probe = TcpListener::bind((STATS_DASHBOARD_HOSTNAME, 0)).expect("bind probe");
        let port = probe.local_addr().expect("local addr").port();
        drop(probe);

        assert!(is_port_free(port, STATS_DASHBOARD_HOSTNAME));
        assert_eq!(
            probe_stats_dashboard(port, STATS_DASHBOARD_HOSTNAME),
            StatsDashboardProbe::Unreachable
        );
        assert_eq!(
            prepare_stats_port(port, STATS_DASHBOARD_HOSTNAME).expect("prepare"),
            PortAction::Retry
        );
    }

    #[test]
    fn plain_listener_is_foreign() {
        let listener = TcpListener::bind((STATS_DASHBOARD_HOSTNAME, 0)).expect("bind listener");
        let port = listener.local_addr().expect("local addr").port();

        // Speaks no HTTP: the probe deadline expires, exactly like the TS fetch timeout.
        assert!(!is_port_free(port, STATS_DASHBOARD_HOSTNAME));
        assert_eq!(
            probe_stats_dashboard(port, STATS_DASHBOARD_HOSTNAME),
            StatsDashboardProbe::Unreachable
        );
        assert_eq!(
            prepare_stats_port(port, STATS_DASHBOARD_HOSTNAME).expect("prepare"),
            PortAction::Retry
        );
    }

    #[test]
    fn http_responder_without_dashboard_header_is_occupied() {
        let port = start_server("HTTP/1.1 200 OK\r\ncontent-length: 0\r\n\r\n", 4);
        assert_eq!(
            probe_stats_dashboard(port, STATS_DASHBOARD_HOSTNAME),
            StatsDashboardProbe::Occupied
        );
        // Reclaiming refuses: the holder is not identifiable as an omp stats dashboard.
        let error =
            prepare_stats_port(port, STATS_DASHBOARD_HOSTNAME).expect_err("must refuse to stop it");
        let message = error.to_string();
        assert!(
            message.contains(&format!("Port {port} is in use"))
                || message.contains(&format!("Port {port} is held by the current process")),
            "{message}"
        );
    }

    #[test]
    fn dashboard_header_marks_the_listener_as_ours() {
        let port = start_server(
            concat!(
                "HTTP/1.1 200 OK\r\nx-omp-stats-dashboard: 3\r\n",
                "x-omp-stats-hostname: 127.0.0.1\r\ncontent-length: 0\r\n\r\n"
            ),
            4,
        );
        assert_eq!(
            probe_stats_dashboard(port, STATS_DASHBOARD_HOSTNAME),
            StatsDashboardProbe::Reusable
        );
        assert_eq!(
            prepare_stats_port(port, STATS_DASHBOARD_HOSTNAME).expect("prepare"),
            PortAction::Reuse
        );
        assert_eq!(
            recover_stats_port(port, STATS_DASHBOARD_HOSTNAME).expect("recover"),
            PortAction::Reuse
        );
    }

    #[test]
    fn bracketed_and_bare_ipv6_hosts_share_one_identity() {
        assert_eq!(normalize_host("[::1]"), "::1");
        assert_eq!(normalize_host("::1"), "::1");
        assert_eq!(normalize_host("[[::1]]"), "[::1]", "exactly one pair");
        assert_eq!(normalize_host("[::1"), "[::1");
        assert_eq!(normalize_host("127.0.0.1"), "127.0.0.1");
        // The dashboard bound to `::1` identifies itself as `::1`; both spellings must reuse it.
        let response = "HTTP/1.1 200 OK\r\nx-omp-stats-dashboard: 3\r\nx-omp-stats-hostname: ::1\r\ncontent-length: 0\r\n\r\n";
        let port = start_server_on("::1", response, 4);
        assert_eq!(
            probe_stats_dashboard(port, "[::1]"),
            StatsDashboardProbe::Reusable
        );
        assert_eq!(
            probe_stats_dashboard(port, "::1"),
            StatsDashboardProbe::Reusable
        );
    }

    #[test]
    fn newer_dashboard_is_never_downgraded() {
        let port = start_server(
            concat!(
                "HTTP/1.1 200 OK\r\nx-omp-stats-dashboard: 4\r\n",
                "x-omp-stats-hostname: 127.0.0.1\r\ncontent-length: 0\r\n\r\n"
            ),
            2,
        );
        assert_eq!(
            probe_stats_dashboard(port, STATS_DASHBOARD_HOSTNAME),
            StatsDashboardProbe::Occupied
        );
    }

    #[test]
    fn hostname_mismatch_and_cors_downgrade_reuse_to_replaceable() {
        let port = start_server(
            concat!(
                "HTTP/1.1 200 OK\r\nx-omp-stats-dashboard: 3\r\n",
                "x-omp-stats-hostname: 0.0.0.0\r\naccess-control-allow-origin: *\r\ncontent-length: 0\r\n\r\n"
            ),
            2,
        );
        assert_eq!(
            probe_stats_dashboard(port, STATS_DASHBOARD_HOSTNAME),
            StatsDashboardProbe::Replaceable
        );
    }

    #[test]
    fn port_zero_always_retries() {
        assert_eq!(
            prepare_stats_port(0, STATS_DASHBOARD_HOSTNAME).expect("prepare"),
            PortAction::Retry
        );
        assert_eq!(PortAction::Reuse.as_str(), "reuse");
        assert_eq!(PortAction::Retry.as_str(), "retry");
    }

    #[test]
    fn identity_detection_matches_stats_processes() {
        assert!(IDENTITY
            .omp_stats_binary
            .is_match("/usr/local/bin/omp-stats --port 7777"));
        assert!(IDENTITY
            .stats_index_ts
            .is_match("bun /repo/packages/stats/src/index.ts"));
        assert!(IDENTITY
            .omp_stats_subcommand
            .is_match("omp stats --port 7777"));
        assert!(IDENTITY.omp_binary_stats.is_match("omp.exe stats"));
        assert!(!IDENTITY.omp_binary_stats.is_match("stats"));
        assert!(!IDENTITY
            .omp_stats_binary
            .is_match("python -m omp_stats_helper"));
        assert_eq!(normalize_image("OMP-Stats.EXE"), "omp-stats");
        assert_eq!(normalize_image("bun (deleted)"), "bun");
    }
}
