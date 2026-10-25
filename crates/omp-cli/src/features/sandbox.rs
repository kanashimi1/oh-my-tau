//! `/sandbox`: Landlock + seccomp confinement of tool subprocesses.
//!
//! Enforcement runs in `omt` itself (Linux-only): the launcher re-executes the
//! same binary as `omt __sandbox-exec --write-dir <cwd> [--net-deny] -- CMD`,
//! which applies Landlock — write access limited to the project cwd, `/tmp`,
//! the agent dir (`~/.omp/agent`) and the null/tty devices, read/execute
//! anywhere — and optionally a seccomp filter denying IPv4/IPv6 sockets, then
//! `execve`-replaces into the tool command. The host (`tools/bash.rs` and other
//! tool spawns) re-launches through this helper when the sandbox is enabled; it
//! never confines itself.
//!
//! Controls: `/sandbox on|off|status|net on|net off`, and the `--sandbox` CLI
//! flag / `sandbox.enabled` setting (both simply enable the toggle). `status`
//! reports the kernel's Landlock ABI; on a kernel without Landlock the sandbox
//! warns and runs unsandboxed.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::rpc::slash::Outcome;
use crate::runtime::Runtime;

/// Sandbox enabled for tool spawns (write policy + optional net deny).
static ENABLED: AtomicBool = AtomicBool::new(false);
/// Deny IPv4/IPv6 sockets inside sandboxed tool spawns.
static NET_DENY: AtomicBool = AtomicBool::new(false);
/// Cached kernel Landlock probe (probing is a syscall; only ever needed once).
#[cfg(target_os = "linux")]
static KERNEL_STATUS: std::sync::OnceLock<Option<omp_sandbox::KernelStatus>> =
    std::sync::OnceLock::new();

/// Whether tool subprocesses should be run inside the sandbox.
#[must_use]
pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// Turn the sandbox on/off.
pub fn set_enabled(on: bool) {
    ENABLED.store(on, Ordering::Relaxed);
}

/// Whether sandboxed tool spawns should deny IPv4/IPv6 sockets.
#[must_use]
pub fn net_deny() -> bool {
    NET_DENY.load(Ordering::Relaxed)
}

/// Turn network denial on/off.
pub fn set_net_deny(on: bool) {
    NET_DENY.store(on, Ordering::Relaxed);
}

/// The kernel's Landlock status, probed once and cached (Linux only; the
/// `omp-sandbox` dependency is target-gated in `Cargo.toml`).
#[cfg(target_os = "linux")]
fn kernel_status() -> Option<omp_sandbox::KernelStatus> {
    *KERNEL_STATUS.get_or_init(omp_sandbox::kernel_status)
}

/// Whether the running kernel can enforce Landlock.
#[cfg(target_os = "linux")]
fn kernel_supported() -> bool {
    matches!(
        kernel_status(),
        Some(omp_sandbox::KernelStatus::Available(_))
    )
}

/// Human-readable kernel Landlock status.
#[cfg(target_os = "linux")]
fn kernel_desc() -> String {
    match kernel_status() {
        Some(omp_sandbox::KernelStatus::Available(v)) => format!("Landlock ABI v{v}"),
        Some(omp_sandbox::KernelStatus::NotEnabled) => "Landlock not enabled at boot".to_string(),
        Some(omp_sandbox::KernelStatus::NotImplemented) => "Landlock not implemented".to_string(),
        None => "Landlock unavailable".to_string(),
    }
}

/// Non-Linux builds have no Landlock probe: never supported, `/sandbox`
/// reports the platform as unsupported.
#[cfg(not(target_os = "linux"))]
fn kernel_supported() -> bool {
    false
}

#[cfg(not(target_os = "linux"))]
fn kernel_desc() -> String {
    "unsupported on this platform".to_string()
}

/// Path to the sandbox launcher: this binary itself (`omt` re-executes itself
/// as `omt __sandbox-exec ...`, so only one binary ships). `/proc/self/exe`
/// can read `... (deleted)` after an on-disk upgrade; prefer the real path
/// when it still exists.
fn launcher_path() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    if let Some(stripped) = exe.to_str().and_then(|s| s.strip_suffix(" (deleted)")) {
        let real = PathBuf::from(stripped);
        if real.exists() {
            return Some(real);
        }
    }
    Some(exe)
}

/// The launcher argv that runs `inner` (a program plus its arguments) inside the
/// sandbox.
fn launch_argv(cwd: &Path, net_deny: bool, inner: &[String]) -> Vec<OsString> {
    let mut v: Vec<OsString> = Vec::with_capacity(inner.len() + 6);
    v.push(OsString::from("__sandbox-exec"));
    v.push(OsString::from("--write-dir"));
    v.push(cwd.as_os_str().to_owned());
    if net_deny {
        v.push(OsString::from("--net-deny"));
    }
    v.push(OsString::from("--"));
    v.extend(inner.iter().map(OsString::from));
    v
}

/// If the sandbox is active and this system can enforce it, the `(program,
/// args)` that runs `inner` confined; `None` means run unsandboxed.
pub fn wrap_inner(cwd: &Path, inner: &[String]) -> Option<(PathBuf, Vec<OsString>)> {
    wrap_inner_with(enabled(), net_deny(), cwd, inner)
}

/// Pure core of [`wrap_inner`] with explicit state (testable without the
/// process-wide toggles).
fn wrap_inner_with(
    on: bool,
    net: bool,
    cwd: &Path,
    inner: &[String],
) -> Option<(PathBuf, Vec<OsString>)> {
    if !on || !cfg!(target_os = "linux") || !kernel_supported() {
        return None;
    }
    Some((launcher_path()?, launch_argv(cwd, net, inner)))
}

/// `/sandbox` slash-command entry point.
// Signature is fixed by the slash dispatcher: it must stay `async`.
#[allow(clippy::unused_async)]
pub async fn command(rt: &Arc<Runtime>, args: &str) -> Outcome {
    let text = response_text(args.trim());
    rt.send(serde_json::json!({"type": "command_output", "text": text}));
    Outcome::Handled {
        agent_invoked: false,
    }
}

/// Compute the `/sandbox` command's reply text, applying the state change it
/// implies to the process-wide toggles.
fn response_text(args: &str) -> String {
    let (text, on, net) = response_text_with(args, enabled(), net_deny());
    set_enabled(on);
    set_net_deny(net);
    text
}

/// Pure core of [`response_text`]: given the current `(enabled, net_deny)`
/// state, return the reply text plus the post-command state. Testable
/// without touching the process-wide toggles.
fn response_text_with(args: &str, mut on: bool, net: bool) -> (String, bool, bool) {
    if !cfg!(target_os = "linux") {
        return (
            "sandbox: unsupported on this platform (Linux only)".to_string(),
            on,
            net,
        );
    }
    let (verb, rest) = match args.split_once(char::is_whitespace) {
        Some((a, b)) => (a, b.trim()),
        None => (args, ""),
    };
    match verb {
        "on" => {
            on = kernel_supported();
            let text = if on {
                "sandbox: on (Landlock active)".to_string()
            } else {
                format!("sandbox: warning — {}; left unsandboxed", kernel_desc())
            };
            (text, on, net)
        }
        "off" => ("sandbox: off".to_string(), false, net),
        "status" => (status_text_with(on, net), on, net),
        "net" => match rest {
            "on" => ("sandbox: net deny on".to_string(), on, true),
            "off" => ("sandbox: net deny off".to_string(), on, false),
            _ => ("usage: /sandbox net on|off".to_string(), on, net),
        },
        _ => (
            "usage: /sandbox on|off|status|net on|net off".to_string(),
            on,
            net,
        ),
    }
}

fn status_text_with(on: bool, net: bool) -> String {
    format!(
        "sandbox: enabled={} net-deny={} kernel={}",
        if on { "on" } else { "off" },
        if net { "on" } else { "off" },
        kernel_desc(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inner_cmd() -> Vec<String> {
        vec![
            "setsid".to_string(),
            "--wait".to_string(),
            "bash".to_string(),
            "-c".to_string(),
            "echo hi".to_string(),
        ]
    }

    #[test]
    fn launch_argv_injects_write_dir_and_net_flag() {
        let argv = launch_argv(Path::new("/proj"), true, &inner_cmd());
        let s: Vec<&str> = argv.iter().map(|o| o.to_str().unwrap()).collect();
        assert!(s
            .windows(2)
            .any(|w| w[0] == "--write-dir" && w[1] == "/proj"));
        assert!(s.contains(&"--net-deny"));
        let sep = s.iter().position(|x| *x == "--").unwrap();
        assert_eq!(
            &s[..sep],
            &["__sandbox-exec", "--write-dir", "/proj", "--net-deny"]
        );
        assert_eq!(
            &s[sep + 1..],
            &["setsid", "--wait", "bash", "-c", "echo hi"]
        );
    }

    #[test]
    fn launch_argv_omits_net_flag_when_off() {
        let argv = launch_argv(Path::new("/proj"), false, &inner_cmd());
        let s: Vec<&str> = argv.iter().map(|o| o.to_str().unwrap()).collect();
        assert!(!s.contains(&"--net-deny"));
        assert!(s.contains(&"--write-dir"));
    }

    #[test]
    fn wrap_inner_disabled_returns_none() {
        // Pure fn only: never touch the process-wide ENABLED/NET_DENY
        // globals (they would leak into other tests in this binary).
        assert!(wrap_inner_with(false, false, Path::new("/"), &inner_cmd()).is_none());
        if !kernel_supported() {
            assert!(wrap_inner_with(true, false, Path::new("/"), &inner_cmd()).is_none());
        }
    }

    #[test]
    fn response_text_handles_usage_and_toggles() {
        // Pure core with explicit state; no global mutation.
        let (text, _, _) = response_text_with("", false, false);
        assert!(text.starts_with("usage:"));
        let (status, on, net) = response_text_with("status", false, false);
        assert!(status.contains("enabled=off"));
        assert!(status.contains("net-deny=off"));
        assert!(status.contains("Landlock"));
        assert!(!on && !net);
        if kernel_supported() {
            let (text, on, net) = response_text_with("on", false, false);
            assert!(text.starts_with("sandbox: on"));
            assert!(on);
            let (text, on, net) = response_text_with("net on", on, net);
            assert!(text.contains("on"));
            assert!(net);
            let (text, on, _) = response_text_with("off", on, net);
            assert!(text.starts_with("sandbox: off"));
            assert!(!on);
        } else {
            let (text, on, _) = response_text_with("on", false, false);
            assert!(text.contains("warning"));
            assert!(!on);
        }
    }
}
