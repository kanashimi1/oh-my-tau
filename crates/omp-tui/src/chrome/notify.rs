//! Desktop notifications — port of `pi-tui/src/desktop-notify.ts` +
//! `terminal-capabilities.ts` §9.
//!
//! A notification always emits BEL (tmux monitor-bell / X11 urgency / audible
//! bell) and additionally a terminal notify protocol (OSC 9 / 99 / 777) when
//! one is configured. On Linux, when the protocol is Bell only and a desktop
//! session exists, `notify-send` (or `gdbus call`) fires fire-and-forget so
//! VTE/Alacritty users still get a toast. `PI_NO_DESKTOP_NOTIFY=1` disables
//! the desktop fallback (BEL/OSC still fire).

use std::io::IsTerminal;
use std::process::{Command, Stdio};

/// Terminal notify protocol (`TerminalInfo.notifyProtocol`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NotifyProtocol {
    /// Raw BEL only (default; most Linux terminals).
    #[default]
    Bell,
    /// ConEmu-style OSC 9 (`\x1b]9;<text>\x1b\\`). Note: on VTE terminals
    /// OSC 9 means progress, so gate on confirmed capability upstream.
    Osc9,
    /// Kitty/iTerm2-style OSC 99 (`\x1b]99;<text>\x1b\\`).
    Osc99,
    /// OSC 777 (`\x1b]777;notify;<title>;<body>\x1b\\`, rxvt-unicode style).
    Osc777,
}

/// Notification urgency (`urgency` field; out-of-range → normal).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Urgency {
    /// Low.
    Low,
    /// Normal (default).
    #[default]
    Normal,
    /// Critical.
    Critical,
}

impl Urgency {
    /// `notify-send -u` value.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Normal => "normal",
            Self::Critical => "critical",
        }
    }

    /// `gdbus` urgency byte.
    #[must_use]
    pub fn gdbus_byte(self) -> u8 {
        match self {
            Self::Low => 0,
            Self::Normal => 1,
            Self::Critical => 2,
        }
    }
}

/// Notification kind (TS `TerminalNotification.type`): picks the herdr sound.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NotifyKind {
    /// Untyped: herdr stays silent.
    #[default]
    None,
    /// A settled turn: herdr rings `done`.
    Completion,
    /// A turn that stopped with an error: herdr rings `request`.
    Error,
    /// A question waiting on the user: herdr rings `request`.
    Ask,
}

/// One desktop/terminal notification (TS `TerminalNotification`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TerminalNotification {
    /// Optional toast title; defaults to `"omp"`.
    pub title: Option<String>,
    /// Message body.
    pub body: String,
    /// Urgency; defaults to normal.
    pub urgency: Urgency,
    /// Kind (TS `type`).
    pub kind: NotifyKind,
}

impl TerminalNotification {
    /// Convenience constructor: body only, normal urgency.
    #[must_use]
    pub fn body(body: impl Into<String>) -> Self {
        Self {
            title: None,
            body: body.into(),
            urgency: Urgency::Normal,
            kind: NotifyKind::None,
        }
    }
}

/// Collapse a notification to a single text line (TS `notificationToLine`).
#[must_use]
pub fn notification_to_line(n: &TerminalNotification) -> String {
    match (&n.title, &n.body) {
        (Some(t), b) if !b.is_empty() => format!("{t}: {}", n.body),
        (Some(t), _) => t.clone(),
        (None, b) => b.clone(),
    }
}

/// Format the in-band escape sequence for `protocol` (TS `formatNotification`,
/// terminal-capabilities.ts:221). Bell → raw BEL; OSC variants are
/// ST-terminated. The structured OSC 99 metadata form is only used once the
/// terminal confirms support (`osc99_confirmed`); otherwise all OSC protocols
/// collapse to `notificationToLine` (basic OSC 99/9 still work).
#[must_use]
pub fn format_notification(
    protocol: NotifyProtocol,
    n: &TerminalNotification,
    osc99_confirmed: bool,
) -> String {
    match protocol {
        NotifyProtocol::Bell => "\x07".to_string(),
        NotifyProtocol::Osc99 if osc99_confirmed => {
            // Rich metadata form, minimal fields (TS `formatOsc99Notification`):
            // id, app name, urgency; title/body base64-encoded.
            let title = n.title.clone().unwrap_or_else(|| n.body.clone());
            let body = if n.title.is_some() {
                n.body.clone()
            } else {
                String::new()
            };
            format!(
                "\x1b]99;i=omp;f=eG1wO;u={};t={};b={}\x1b\\",
                osc99_urgency(n.urgency),
                base64_utf8(&title),
                base64_utf8(&body)
            )
        }
        NotifyProtocol::Osc9 | NotifyProtocol::Osc99 => {
            format!(
                "\x1b]{};{}\x1b\\",
                protocol_code(protocol),
                notification_to_line(n)
            )
        }
        NotifyProtocol::Osc777 => {
            let title = n.title.clone().unwrap_or_else(|| "omp".into());
            format!("\x1b]777;notify;{title};{}\x1b\\", n.body)
        }
    }
}

fn protocol_code(protocol: NotifyProtocol) -> &'static str {
    match protocol {
        NotifyProtocol::Osc9 => "9",
        NotifyProtocol::Osc99 => "99",
        _ => "9",
    }
}

/// OSC 99 urgency letter (TS `osc99Urgency`): critical `2`, low `3`, else
/// absent (normal is the default and omitted).
fn osc99_urgency(urgency: Urgency) -> &'static str {
    match urgency {
        Urgency::Low => "3",
        Urgency::Normal => "",
        Urgency::Critical => "2",
    }
}

/// UTF-8 → base64 (OSC 99 payload encoding; no external crate — hand-rolled
/// RFC 4648, padded).
pub(crate) fn base64_utf8(value: &str) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let bytes = value.as_bytes();
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = chunk.get(1).copied().unwrap_or(0) as u32;
        let b2 = chunk.get(2).copied().unwrap_or(0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(TABLE[(n >> 18) as usize & 63] as char);
        out.push(TABLE[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 {
            TABLE[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            TABLE[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

/// Whether a Linux desktop session is visible (TS `hasLinuxDesktopSession`):
/// linux + any of the usual desktop-session env vars.
#[must_use]
pub fn has_linux_desktop_session() -> bool {
    if !cfg!(target_os = "linux") {
        return false;
    }
    ["DISPLAY", "WAYLAND_DISPLAY", "DBUS_SESSION_BUS_ADDRESS"]
        .iter()
        .any(|k| std::env::var_os(k).is_some())
}

/// Send a notification: always BEL (tmux monitor-bell, X11 urgency, audible),
/// plus `notify-send`/`gdbus` on Linux when the protocol is Bell-only and a
/// desktop session exists (TS `sendDesktopNotification`). Fire-and-forget;
/// spawn failures are silent. `PI_NO_DESKTOP_NOTIFY=1` disables the desktop
/// fallback.
pub fn send_notification(protocol: NotifyProtocol, n: &TerminalNotification) {
    // TS `isNotificationSuppressed`: `PI_NOTIFICATIONS=off|0|false`.
    if matches!(
        std::env::var("PI_NOTIFICATIONS").as_deref(),
        Ok("off" | "0" | "false")
    ) {
        return;
    }
    // Innermost surface first (TS `sendNotification`): herdr, then cmux.
    let env = |k: &str| std::env::var(k).ok();
    if let Some(argv) = herdr_argv(&env, n) {
        if spawn_detached(&argv) {
            return;
        }
    }
    if let Some(argv) = cmux_argv(&env, n) {
        if spawn_detached(&argv) {
            return;
        }
    }
    if std::io::stdout().is_terminal() {
        // BEL always fires (tmux monitor-bell, X11 urgency, audible bell),
        // then the protocol sequence (rich OSC 99 collapsed unless the
        // terminal confirmed support).
        use std::io::Write;
        let mut out = std::io::stdout().lock();
        let _ = out.write_all(b"\x07");
        let _ = out.write_all(format_notification(protocol, n, false).as_bytes());
        let _ = out.flush();
    }
    if protocol == NotifyProtocol::Bell
        && has_linux_desktop_session()
        && std::env::var("PI_NO_DESKTOP_NOTIFY").as_deref() != Ok("1")
    {
        spawn_desktop_notify(n);
    }
}

/// TS `isInsideHerdr`: `HERDR_ENV=1` or any pane/tab/workspace identity var.
fn inside_herdr(env: &dyn Fn(&str) -> Option<String>) -> bool {
    let set = |k: &str| env(k).is_some_and(|v| !v.is_empty());
    env("HERDR_ENV").as_deref() == Some("1")
        || set("HERDR_PANE_ID")
        || set("HERDR_TAB_ID")
        || set("HERDR_WORKSPACE_ID")
}

fn notification_title_and_body(n: &TerminalNotification) -> (String, String) {
    let title = n
        .title
        .as_deref()
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .unwrap_or("omp")
        .to_owned();
    (title, n.body.clone())
}

/// TS `sendHerdrNotification` argv: `herdr notification show <title> --body <body> --sound <s>`
/// when the process runs inside a herdr pane (valid `HERDR_PANE_ID` required).
#[must_use]
pub fn herdr_argv(
    env: &dyn Fn(&str) -> Option<String>,
    n: &TerminalNotification,
) -> Option<Vec<String>> {
    if !inside_herdr(env) {
        return None;
    }
    let pane = env("HERDR_PANE_ID")?;
    let pane = pane.trim();
    if pane.is_empty()
        || pane.len() > 64
        || !pane
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, ':' | '_' | '-'))
    {
        return None;
    }
    let (title, body) = notification_title_and_body(n);
    // `herdr notification show` reads a bare help token in the title slot as a help request.
    let title = if matches!(title.as_str(), "help" | "--help" | "-h") {
        "omp".to_owned()
    } else {
        title
    };
    let sound = match n.kind {
        NotifyKind::Ask | NotifyKind::Error => "request",
        NotifyKind::Completion => "done",
        NotifyKind::None => "none",
    };
    Some(vec![
        "herdr".into(),
        "notification".into(),
        "show".into(),
        title,
        "--body".into(),
        body,
        "--sound".into(),
        sound.into(),
    ])
}

/// TS `sendCmuxNotification` argv (a valid `CMUX_SURFACE_ID` UUID is required).
#[must_use]
pub fn cmux_argv(
    env: &dyn Fn(&str) -> Option<String>,
    n: &TerminalNotification,
) -> Option<Vec<String>> {
    let surface = env("CMUX_SURFACE_ID")?;
    let surface = surface.trim();
    let groups: Vec<&str> = surface.split('-').collect();
    let uuid = groups.len() == 5
        && groups
            .iter()
            .zip([8, 4, 4, 4, 12])
            .all(|(g, len)| g.len() == len && g.chars().all(|c| c.is_ascii_hexdigit()));
    if !uuid {
        return None;
    }
    let (title, body) = notification_title_and_body(n);
    Some(vec![
        "cmux".into(),
        "notify".into(),
        "--surface".into(),
        surface.to_owned(),
        "--title".into(),
        title,
        "--body".into(),
        body,
    ])
}

/// Spawn detached with stdio ignored; `false` when the binary is missing (the caller falls back).
fn spawn_detached(argv: &[String]) -> bool {
    Command::new(&argv[0])
        .args(&argv[1..])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .is_ok()
}

/// Spawn `notify-send` (preferred) or `gdbus call`; failures silent.
fn spawn_desktop_notify(n: &TerminalNotification) {
    let title = n.title.clone().unwrap_or_else(|| "omp".into());
    if let Ok(path) = which("notify-send") {
        let _ = Command::new(path)
            .args(["-u", n.urgency.as_str()])
            .arg(&title)
            .arg(&n.body)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn();
        return;
    }
    let path = match which("gdbus") {
        Ok(p) => p,
        Err(()) => return,
    };
    // org.freedesktop.Notifications.Notify(app_name, replaces_id, app_icon,
    // summary, body, actions, hints, expire_timeout)
    let summary = gdbus_string(&title);
    let body = gdbus_string(&n.body);
    let arg = format!(
        "<'omp'> 0 <> {summary} {body} [] {{'urgency': <byte {0}>}} -1",
        n.urgency.gdbus_byte()
    );
    let _ = Command::new(path)
        .args([
            "call",
            "--session",
            "--dest",
            "org.freedesktop.Notifications",
            "--object-path",
            "/org/freedesktop/Notifications",
            "--method",
            "org.freedesktop.Notifications.Notify",
            &arg,
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn();
}

/// GVariant string literal (single-quoted, internal quotes escaped).
fn gdbus_string(s: &str) -> String {
    format!("'{}'", s.replace('\\', "\\\\").replace('\'', "\\'"))
}

/// Cached `$which` resolution (TS caches too); `None` when not found.
fn which(bin: &str) -> Result<std::path::PathBuf, ()> {
    use parking_lot::Mutex;
    use std::collections::HashMap;
    use std::sync::LazyLock;
    static CACHE: LazyLock<Mutex<HashMap<String, Option<std::path::PathBuf>>>> =
        LazyLock::new(|| Mutex::new(HashMap::new()));
    let cache = &*CACHE;
    if let Some(hit) = cache.lock().get(bin) {
        return hit.clone().ok_or(());
    }
    let resolved = std::env::var_os("PATH")
        .map(std::path::PathBuf::from)
        .map(|paths| {
            for dir in std::env::split_paths(&paths) {
                let candidate = dir.join(bin);
                if candidate.is_file() {
                    return Some(candidate);
                }
            }
            None
        })
        .unwrap_or(None);
    cache.lock().insert(bin.to_string(), resolved.clone());
    resolved.ok_or(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn herdr_argv_matches_ts() {
        let env = |pairs: &'static [(&'static str, &'static str)]| {
            move |k: &str| {
                pairs
                    .iter()
                    .find(|(n, _)| *n == k)
                    .map(|(_, v)| (*v).to_owned())
            }
        };
        let mut n = TerminalNotification::body("Complete");
        n.kind = NotifyKind::Completion;
        let argv = herdr_argv(&env(&[("HERDR_ENV", "1"), ("HERDR_PANE_ID", "p_1:2")]), &n).unwrap();
        assert_eq!(
            argv.join(" "),
            "herdr notification show omp --body Complete --sound done"
        );
        n.kind = NotifyKind::Error;
        n.title = Some("My session".into());
        let argv = herdr_argv(&env(&[("HERDR_PANE_ID", "p1")]), &n).unwrap();
        assert_eq!(argv[3], "My session");
        assert_eq!(argv[7], "request");
        n.kind = NotifyKind::None;
        assert_eq!(
            herdr_argv(&env(&[("HERDR_PANE_ID", "p1")]), &n).unwrap()[7],
            "none"
        );
        // Outside a pane, or without a valid pane id, herdr does not own delivery.
        assert!(herdr_argv(&env(&[]), &n).is_none());
        assert!(herdr_argv(&env(&[("HERDR_ENV", "1")]), &n).is_none());
        assert!(herdr_argv(&env(&[("HERDR_PANE_ID", "bad id")]), &n).is_none());
        n.title = Some("help".into());
        assert_eq!(
            herdr_argv(&env(&[("HERDR_PANE_ID", "p1")]), &n).unwrap()[3],
            "omp"
        );
    }

    #[test]
    fn bell_is_raw_bel() {
        let n = TerminalNotification::body("done");
        assert_eq!(format_notification(NotifyProtocol::Bell, &n, false), "\x07");
    }

    #[test]
    fn osc_lines_collapse_to_title_body() {
        let n = TerminalNotification {
            title: Some("omp".into()),
            body: "turn ended".into(),
            urgency: Urgency::Normal,
            kind: NotifyKind::None,
        };
        assert_eq!(
            format_notification(NotifyProtocol::Osc9, &n, false),
            "\x1b]9;omp: turn ended\x1b\\"
        );
        assert_eq!(
            format_notification(NotifyProtocol::Osc99, &n, false),
            "\x1b]99;omp: turn ended\x1b\\"
        );
        assert_eq!(
            format_notification(NotifyProtocol::Osc777, &n, false),
            "\x1b]777;notify;omp;turn ended\x1b\\"
        );
    }

    #[test]
    fn osc99_rich_when_confirmed() {
        let n = TerminalNotification {
            title: Some("omp".into()),
            body: "hi".into(),
            urgency: Urgency::Critical,
            kind: NotifyKind::None,
        };
        let out = format_notification(NotifyProtocol::Osc99, &n, true);
        assert!(out.starts_with("\x1b]99;i=omp;f=eG1wO;u=2;"));
        assert!(out.ends_with("\x1b\\")); // ST-terminated
    }

    #[test]
    fn base64_known_vectors() {
        assert_eq!(base64_utf8("omp"), "b21w");
        assert_eq!(base64_utf8("hi"), "aGk=");
        assert_eq!(base64_utf8("h"), "aA==");
    }

    #[test]
    fn urgency_mapping() {
        assert_eq!(Urgency::Critical.gdbus_byte(), 2);
        assert_eq!(Urgency::Low.as_str(), "low");
    }
}
