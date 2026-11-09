//! Switching a live hub agent between front ends on a WSL host: raise (or
//! start) the Tau desktop window, or open a Windows Terminal tab running
//! `omt attach <id>`. Both are fire-and-forget Windows processes started
//! through WSL interop.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{bail, Context as _, Result};

const POWERSHELL: &str = "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe";
/// Windows cwd for interop children: a UNC `\\wsl.localhost` cwd makes
/// cmd/powershell warn and fall back anyway.
const WIN_CWD: &str = "/mnt/c/";

/// Raise-or-start script: restore + foreground an existing Tau window, start
/// Tau only when no Tau process exists (a tray-hidden one is left alone).
const RAISE_PS: &str = r#"
Add-Type @'
using System; using System.Runtime.InteropServices;
public class OmtWin {
  [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int c);
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
  [DllImport("user32.dll")] public static extern bool IsIconic(IntPtr h);
  [DllImport("user32.dll")] public static extern void keybd_event(byte k, byte s, uint f, UIntPtr e);
}
'@
$p = Get-Process tau -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowHandle -ne 0 -and $_.MainWindowTitle } | Select-Object -First 1
if ($p) {
  $h = $p.MainWindowHandle
  if ([OmtWin]::IsIconic($h)) { [OmtWin]::ShowWindow($h, 9) | Out-Null }
  # A synthetic Alt tap lifts the foreground lock for background callers.
  [OmtWin]::keybd_event(0x12, 0, 0, [UIntPtr]::Zero)
  [OmtWin]::keybd_event(0x12, 0, 2, [UIntPtr]::Zero)
  [OmtWin]::SetForegroundWindow($h) | Out-Null
} elseif (-not (Get-Process tau -ErrorAction SilentlyContinue)) {
  Start-Process "$env:LOCALAPPDATA\Programs\Tau\tau.exe"
}
"#;

fn detached(cmd: &mut Command) -> Result<()> {
    cmd.current_dir(WIN_CWD)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    let mut child = cmd.spawn().context("start Windows process (WSL interop)")?;
    // Reap in the background; the Windows side outlives the interop shim.
    std::thread::spawn(move || {
        let _ = child.wait();
    });
    Ok(())
}

fn require_wsl() -> Result<()> {
    if !Path::new(POWERSHELL).exists() {
        bail!("switching front ends needs a WSL host with Windows interop");
    }
    Ok(())
}

/// Bring the Tau window to the front, starting Tau when it is not running.
pub fn raise() -> Result<()> {
    require_wsl()?;
    detached(Command::new(POWERSHELL).args([
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        RAISE_PS,
    ]))
}

/// WSL distro name for `wsl.exe -d` (`\\wsl.localhost\<distro>\` → `<distro>`).
fn distro() -> Option<String> {
    if let Some(d) = std::env::var("WSL_DISTRO_NAME")
        .ok()
        .filter(|d| !d.is_empty())
    {
        return Some(d);
    }
    let out = Command::new("wslpath").args(["-w", "/"]).output().ok()?;
    let unc = String::from_utf8_lossy(&out.stdout).trim().to_owned();
    unc.trim_start_matches('\\')
        .split('\\')
        .nth(1)
        .filter(|d| !d.is_empty())
        .map(str::to_owned)
}

/// Windows Terminal's app-execution alias in some Windows profile.
fn windows_terminal() -> Option<PathBuf> {
    std::fs::read_dir("/mnt/c/Users")
        .ok()?
        .flatten()
        .find_map(|e| {
            let wt = e.path().join("AppData/Local/Microsoft/WindowsApps/wt.exe");
            wt.exists().then_some(wt)
        })
}

/// Open a Windows Terminal tab (a plain console window without wt) running
/// `<exe> attach <agent_id>` in `cwd`.
pub fn open_terminal(exe: &Path, cwd: &Path, agent_id: &str) -> Result<()> {
    require_wsl()?;
    let distro = distro().context("cannot determine the WSL distro name")?;
    let (exe, cwd) = (exe.display().to_string(), cwd.display().to_string());
    let wsl = [
        "wsl.exe", "-d", &distro, "--cd", &cwd, "-e", &exe, "attach", agent_id,
    ];
    match windows_terminal() {
        Some(wt) => detached(
            Command::new(wt)
                .args(["-w", "0", "new-tab", "--title", "omt"])
                .args(wsl),
        ),
        None => detached(
            Command::new("/mnt/c/Windows/System32/cmd.exe")
                .args(["/c", "start", "omt"])
                .args(wsl),
        ),
    }
}
