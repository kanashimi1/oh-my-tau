//! Landlock + seccomp confinement for tool subprocesses.
//!
//! Kept as its own crate (not inside `omp-cli`) because confining a spawned
//! tool requires low-level syscalls (Landlock rulesets, seccomp BPF) and an
//! `execve` so the launcher's pid becomes the confined command — all of which
//! are `unsafe`, a lint the `omp-cli` workspace sets to `forbid`. The public
//! surface here is safe; the confined `unsafe` regions live only in this crate.

use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Landlock kernel support as probed from the running kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelStatus {
    /// Landlock works; the ABI version the kernel reports.
    Available(u32),
    /// Landlock compiled in but not enabled at boot (`EOPNOTSUPP`).
    NotEnabled,
    /// Landlock not built into this kernel (`ENOSYS`).
    NotImplemented,
}

/// Flag to `landlock_create_ruleset` that returns the ABI version instead of
/// creating a ruleset.
const LANDLOCK_CREATE_RULESET_VERSION: u64 = 0x1;

/// Probe the running kernel for its Landlock support.
///
/// Returns `None` only when the syscall number is not known on the current
/// architecture (this crate only probes on Linux).
#[cfg(target_os = "linux")]
pub fn kernel_status() -> Option<KernelStatus> {
    // SAFETY: `landlock_create_ruleset` with the version flag and null pointers
    // never touches memory; it returns the ABI version (>= 1) or -1/errno.
    let rc = unsafe {
        libc::syscall(
            libc::SYS_landlock_create_ruleset,
            std::ptr::null::<u8>(),
            0usize,
            LANDLOCK_CREATE_RULESET_VERSION,
        )
    };
    if rc < 0 {
        return match std::io::Error::last_os_error().raw_os_error() {
            Some(libc::EOPNOTSUPP) => Some(KernelStatus::NotEnabled),
            _ => Some(KernelStatus::NotImplemented),
        };
    }
    Some(KernelStatus::Available(rc as u32))
}

/// Non-Linux build: no kernel probe, report "not implemented" so callers fall
/// back to running unsandboxed (the `/sandbox` command prints "unsupported").
#[cfg(not(target_os = "linux"))]
pub fn kernel_status() -> Option<KernelStatus> {
    Some(KernelStatus::NotImplemented)
}

/// Apply Landlock and (optionally) a seccomp network-deny filter to the *calling
/// process*. Write access is limited to `write_dirs`; read/execute is allowed
/// everywhere. After this returns, the process (and everything it forks/execs)
/// is confined.
pub fn apply(write_dirs: &[&Path], net_deny: bool) -> Result<(), String> {
    apply_landlock(write_dirs)?;
    if net_deny {
        apply_seccomp_net_deny()?;
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn apply_landlock(write_dirs: &[&Path]) -> Result<(), String> {
    use landlock::{
        Access, AccessFs, PathBeneath, PathFd, Ruleset, RulesetAttr, RulesetCreatedAttr,
    };

    let abi = landlock::ABI::V9;
    let mut ruleset = Ruleset::default()
        .handle_access(AccessFs::from_all(abi))
        .map_err(|e| format!("landlock handle_access: {e}"))?
        .create()
        .map_err(|e| format!("landlock create ruleset: {e}"))?;

    // Read (and execute) everywhere under `/`: Landlock grants by path, so a
    // single root rule with the read/execute set allows transparent read access
    // to the whole tree.
    ruleset = ruleset
        .add_rule(PathBeneath::new(
            PathFd::new("/").map_err(|e| format!("open /: {e}"))?,
            AccessFs::from_read(abi),
        ))
        .map_err(|e| format!("landlock root read rule: {e}"))?;

    // Write only inside the granted directories. Device nodes (`/dev/null`,
    // `/dev/tty`) are best-effort: `/dev/tty` in particular cannot be opened
    // when there is no controlling terminal, and that must not fail the whole
    // sandbox. The cwd/`/tmp`/agent-dir rules are strict.
    for d in write_dirs {
        let path = *d;
        let lenient = path.starts_with("/dev/");
        let path_fd = match PathFd::new(path) {
            Ok(fd) => fd,
            Err(_) if lenient => continue,
            Err(e) => return Err(format!("open {}: {e}", path.display())),
        };
        match (&mut ruleset).add_rule(PathBeneath::new(path_fd, AccessFs::from_write(abi))) {
            Ok(_) | Err(_) if lenient => {}
            Ok(_) => {}
            Err(e) => return Err(format!("landlock write rule {}: {e}", path.display())),
        }
    }

    let status = ruleset
        .restrict_self()
        .map_err(|e| format!("landlock restrict_self: {e}"))?;
    match status.ruleset {
        landlock::RulesetStatus::FullyEnforced | landlock::RulesetStatus::PartiallyEnforced => {
            Ok(())
        }
        landlock::RulesetStatus::NotEnforced => {
            Err("landlock ruleset not enforced by kernel".into())
        }
    }
}

/// Non-Linux: no Landlock enforcement.
#[cfg(not(target_os = "linux"))]
fn apply_landlock(_write_dirs: &[&Path]) -> Result<(), String> {
    Err("landlock unsupported on this platform".into())
}

#[cfg(target_os = "linux")]
fn apply_seccomp_net_deny() -> Result<(), String> {
    use seccompiler::{
        SeccompAction, SeccompCmpArgLen, SeccompCmpOp, SeccompCondition, SeccompFilter,
        SeccompRule, TargetArch,
    };

    // `socket(AF_INET, ...)` and `socket(AF_INET6, ...)` are denied (EACCES);
    // everything else — including `socket(AF_UNIX, ...)` — is allowed.
    let rules = vec![(
        libc::SYS_socket,
        vec![
            SeccompRule::new(vec![SeccompCondition::new(
                0,
                SeccompCmpArgLen::Dword,
                SeccompCmpOp::Eq,
                libc::AF_INET as u64,
            )
            .map_err(|e| format!("seccomp cond inet: {e}"))?])
            .map_err(|e| format!("seccomp rule inet: {e}"))?,
            SeccompRule::new(vec![SeccompCondition::new(
                0,
                SeccompCmpArgLen::Dword,
                SeccompCmpOp::Eq,
                libc::AF_INET6 as u64,
            )
            .map_err(|e| format!("seccomp cond inet6: {e}"))?])
            .map_err(|e| format!("seccomp rule inet6: {e}"))?,
        ],
    )]
    .into_iter()
    .collect();

    let arch = TargetArch::try_from(std::env::consts::ARCH)
        .map_err(|e| format!("seccomp target arch: {e}"))?;
    let filter = SeccompFilter::new(
        rules,
        SeccompAction::Allow,
        SeccompAction::Errno(libc::EACCES as u32),
        arch,
    )
    .map_err(|e| format!("seccomp filter: {e}"))?;
    let bpf: seccompiler::BpfProgram = filter
        .try_into()
        .map_err(|e| format!("seccomp compile: {e}"))?;
    seccompiler::apply_filter(&bpf).map_err(|e| format!("seccomp apply: {e}"))?;
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn apply_seccomp_net_deny() -> Result<(), String> {
    Err("seccomp unsupported on this platform".into())
}

/// Parsed `__sandbox-exec` arguments (everything after the `__sandbox-exec`
/// token itself).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SandboxArgs {
    /// Directories the confined command may write to (in addition to the
    /// fixed set: `/tmp`, the agent dir, `/dev/null`, `/dev/tty`).
    pub write_dirs: Vec<PathBuf>,
    /// Deny IPv4/IPv6 sockets via seccomp.
    pub net_deny: bool,
    /// The program to exec.
    pub program: OsString,
    /// Its arguments.
    pub args: Vec<OsString>,
}

/// Parse `--write-dir DIR` / `--net-deny` options followed by `--` and the
/// command to run. Errors on a missing separator or an empty command.
pub fn parse_sandbox_args(args: &[OsString]) -> Result<SandboxArgs, String> {
    let mut write_dirs: Vec<PathBuf> = Vec::new();
    let mut net_deny = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].to_str() {
            Some("--") => {
                if i + 1 >= args.len() {
                    return Err(
                        "usage: __sandbox-exec [--write-dir DIR]... [--net-deny] -- CMD ARGS..."
                            .into(),
                    );
                }
                return Ok(SandboxArgs {
                    write_dirs,
                    net_deny,
                    program: args[i + 1].clone(),
                    args: args[i + 2..].to_vec(),
                });
            }
            Some("--net-deny") => net_deny = true,
            Some("--write-dir") => {
                i += 1;
                if i >= args.len() {
                    return Err("--write-dir requires a directory argument".into());
                }
                write_dirs.push(args[i].clone().into());
            }
            _ => {
                return Err(format!(
                    "unexpected argument: {}",
                    args[i].to_string_lossy()
                ))
            }
        }
        i += 1;
    }
    Err("usage: __sandbox-exec [--write-dir DIR]... [--net-deny] -- CMD ARGS...".into())
}

/// Confined-exec entry point used by `omt` itself. `argv` is the full process
/// argv with `argv[1] == "__sandbox-exec"`; the remaining tokens are parsed by
/// [`parse_sandbox_args`]. Applies Landlock (write limited to the given dirs
/// plus `/tmp`, the agent dir `~/.omp/agent` and `/dev/null`, `/dev/tty`
/// best-effort; read+execute everywhere) and, with `--net-deny`, a seccomp
/// filter denying IPv4/IPv6 sockets, then execvp's the command. Never returns:
/// on success the process is replaced; on any failure `omt sandbox: <error>`
/// is printed to stderr and the process exits 126.
pub fn exec_confined(argv: &[OsString]) -> ! {
    let args: &[OsString] = if argv.len() >= 2 { &argv[2..] } else { &[] };
    let run = match parse_sandbox_args(args) {
        Ok(r) => r,
        Err(e) => sandbox_fail(&e),
    };
    let mut write_dirs = run.write_dirs;
    write_dirs.push(std::env::temp_dir());
    if let Some(home) = std::env::var_os("HOME").map(PathBuf::from) {
        write_dirs.push(home.join(".omp/agent"));
    }
    write_dirs.push(PathBuf::from("/dev/null"));
    // Best-effort: no controlling terminal must not fail the sandbox.
    write_dirs.push(PathBuf::from("/dev/tty"));

    let paths: Vec<&Path> = write_dirs.iter().map(PathBuf::as_path).collect();
    if let Err(e) = apply(&paths, run.net_deny) {
        sandbox_fail(&e);
    }
    exec_into(&run.program, &run.args)
}

/// Print the sandbox error and exit 126 (a command could not be invoked).
fn sandbox_fail(msg: &str) -> ! {
    eprintln!("omt sandbox: {msg}");
    std::process::exit(126);
}

/// Replace this process with `program args...` (same pid, so the host's
/// process management — kill_on_drop, timeout/abort kill, wait — hits the
/// tool directly).
#[cfg(unix)]
fn exec_into(program: &OsStr, args: &[OsString]) -> ! {
    use std::os::unix::process::CommandExt;
    // SAFETY: the appended argv is non-empty; `exec` only returns on error.
    let err = Command::new(program).args(args).exec();
    sandbox_fail(&format!("exec {}: {err}", program.to_string_lossy()));
}

/// Non-unix fallback (the self-exec hook never triggers there): spawn and
/// forward the exit status.
#[cfg(not(unix))]
fn exec_into(program: &OsStr, args: &[OsString]) -> ! {
    match Command::new(program).args(args).status() {
        Ok(status) => std::process::exit(status.code().unwrap_or(1)),
        Err(e) => sandbox_fail(&format!("exec {}: {e}", program.to_string_lossy())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn os(args: &[&str]) -> Vec<OsString> {
        args.iter().map(OsString::from).collect()
    }

    #[test]
    fn kernel_status_never_none_on_linux() {
        let status = kernel_status();
        assert!(status.is_some());
    }

    #[test]
    fn apply_rejects_missing_write_dir() {
        let err = apply(&[Path::new("/definitely/not/a/real/dir/xyz")], false);
        assert!(err.is_err());
    }

    #[test]
    fn parse_full_form() {
        let a = parse_sandbox_args(&os(&[
            "--write-dir",
            "/proj",
            "--write-dir",
            "/tmp2",
            "--net-deny",
            "--",
            "bash",
            "-c",
            "echo hi",
        ]))
        .unwrap();
        assert_eq!(
            a.write_dirs,
            vec![PathBuf::from("/proj"), PathBuf::from("/tmp2")]
        );
        assert!(a.net_deny);
        assert_eq!(a.program, OsString::from("bash"));
        assert_eq!(a.args, os(&["-c", "echo hi"]));
    }

    #[test]
    fn parse_minimal_form() {
        let a = parse_sandbox_args(&os(&["--", "ls"])).unwrap();
        assert!(a.write_dirs.is_empty());
        assert!(!a.net_deny);
        assert_eq!(a.program, OsString::from("ls"));
        assert!(a.args.is_empty());
    }

    #[test]
    fn parse_rejects_missing_separator() {
        assert!(parse_sandbox_args(&os(&["--net-deny"])).is_err());
    }

    #[test]
    fn parse_rejects_empty_command() {
        assert!(parse_sandbox_args(&os(&["--write-dir", "/x", "--"])).is_err());
    }

    #[test]
    fn parse_rejects_unknown_option() {
        assert!(parse_sandbox_args(&os(&["--bogus", "--", "ls"])).is_err());
    }

    #[test]
    fn exec_confined_argv_skips_two_tokens() {
        // The parser never sees `omt` or `__sandbox-exec` itself.
        let mut argv = os(&["omt", "__sandbox-exec", "--", "true"]);
        argv[0] = OsString::from("/usr/bin/omt");
        let a = parse_sandbox_args(&argv[2..]).unwrap();
        assert_eq!(a.program, OsString::from("true"));
    }
}
