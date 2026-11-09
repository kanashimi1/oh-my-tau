//! Mechanical kill guard on shell commands run by agent sessions.
//!
//! A System-1 reflex implemented in code rather than prompt text: no model
//! cooperation is required and nothing can talk its way past it. Regression
//! for 2026-10-31, when a QA agent "cleaning zombies" took down 4 live
//! desktop sessions and 12 agents with "kill <pid>..." / "pkill -f" lists.
//!
//! Rule: a session may only signal processes inside its own subtree (the
//! target's ancestry reaches this process). The local omt fleet - hubs and
//! web servers, "--mode rpc" agents, and their children, recognized by
//! argv0/comm/flag pairs while walking /proc - is off-limits even when it
//! looks stale or renamed. Unresolvable targets ($!, %jobs, $(...)) pass, as
//! they scope to the command's own children; "kill -0" probes pass.
//! Operator override: run omt itself with `OMT_KILL_GUARD=0` (env of the omt process).

/// Marker substrings that make a pattern-based killer (pkill/killpg/killall)
/// fleet-directed. Matched against the whole command text: patterns may be
/// quoted and split across tokens.
const PATTERN_MARKERS: &[&str] = &["omt", "omp-rs", "hub serve", "mode rpc", "shard.ts", "ext-host"];

/// A /proc node: argv (empty for kernel threads), comm, parent pid.
#[derive(Clone)]
struct ProcInfo {
    argv: Vec<String>,
    comm: String,
    ppid: u32,
}

impl ProcInfo {
    fn cmdline(&self) -> String {
        if self.argv.is_empty() {
            format!("[{}]", self.comm)
        } else {
            self.argv.join(" ")
        }
    }
}

/// One signalling command site found in a shell command line.
struct Site {
    /// True for pkill/killpg/killall (match/group-directed).
    pattern: bool,
    /// True when only signal 0 (probe) or -l was given.
    probe: bool,
    /// Numeric pids (or, after "--", pgids) the site would signal.
    targets: Vec<i64>,
}

/// Split a command into shell words on whitespace and separators, so each
/// signalling keyword and its arguments become separate tokens.
fn tokens(cmd: &str) -> Vec<String> {
    cmd.split(|c: char| c.is_whitespace() || matches!(c, ';' | '&' | '|' | '(' | ')' | '<' | '>' | '\n' | '`' | '\r'))
        .map(str::to_string)
        .collect()
}

/// Basename of a token, so "/usr/bin/kill" and "kill" compare equal.
fn base(token: &str) -> &str {
    std::path::Path::new(token)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(token)
}

fn is_kill_word(token: &str) -> bool {
    matches!(base(token), "kill" | "pkill" | "killpg" | "killall")
}

/// Find every signalling site in "cmd".
fn sites(cmd: &str) -> Vec<Site> {
    let toks = tokens(cmd);
    let mut out = Vec::new();
    let mut i = 0;
    while i < toks.len() {
        if !is_kill_word(&toks[i]) {
            i += 1;
            continue;
        }
        let pattern = base(&toks[i]) != "kill";
        let mut probe = false;
        let mut explicit_signal = false;
        let mut dashdash = false;
        let mut targets: Vec<i64> = Vec::new();
        let mut j = i + 1;
        while j < toks.len() {
            let a = toks[j].as_str();
            if is_kill_word(a) {
                break;
            }
            if let Some(rest) = a.strip_prefix('-') {
                if rest == "-" {
                    dashdash = true;
                } else if let Ok(n) = rest.parse::<i64>() {
                    if n == 0 {
                        probe = true;
                    } else if dashdash {
                        // after "--" a negative-looking number is a pgid
                        targets.push(-n);
                    } else {
                        explicit_signal = true; // -9, -15: a signal spec
                    }
                } else if rest == "s" {
                    // "-s SIGTERM" / "-s 0": the next token names the signal
                    if let Some(sig) = toks.get(j + 1) {
                        if sig == "0" || sig == "SIG0" {
                            probe = true;
                        } else {
                            explicit_signal = true;
                        }
                        j += 1;
                    }
                } else if rest == "l" {
                    probe = true; // "kill -l" lists signals, sends nothing
                } else {
                    explicit_signal = true; // -TERM, --signal KILL, ...
                }
            } else if !a.is_empty() && a.chars().all(|c| c.is_ascii_digit()) {
                targets.push(a.parse().unwrap_or(0));
            } else {
                break; // first non-numeric, non-flag word ends the site
            }
            j += 1;
        }
        out.push(Site {
            pattern,
            probe: probe && !explicit_signal,
            targets,
        });
        i = j;
    }
    out
}

/// Is this /proc node part of the local omt fleet?
fn node_is_fleet(info: &ProcInfo) -> bool {
    let argv0 = info.argv.first().map(|a| base(a).to_string()).unwrap_or_default();
    if argv0 == "omt" || argv0 == "omp-rs" || info.comm == "omt" || info.comm == "omp-rs" {
        return true;
    }
    // "omt hub serve", "omt --mode rpc": flag pairs survive argv0 renames
    for w in info.argv.windows(2) {
        if (w[0] == "hub" && w[1] == "serve") || (w[0] == "--mode" && w[1] == "rpc") {
            return true;
        }
    }
    // desktop host children (bun ext-host/shard.ts)
    info.argv.iter().any(|a| a.contains("shard.ts") || a.contains("ext-host"))
}

/// Walk "pid"'s ancestry: if this process appears anywhere in the chain the
/// session owns the target (it spawned that subtree, fleet node or not);
/// otherwise the first fleet node reached protects it. Returns that node's
/// cmdline.
fn protector(pid: u32, self_pid: u32, info_of: &dyn Fn(u32) -> Option<ProcInfo>) -> Option<String> {
    let mut cur = pid;
    let mut fleet: Option<String> = None;
    for _ in 0..40 {
        if cur == self_pid {
            return None;
        }
        let info = info_of(cur)?;
        if fleet.is_none() && node_is_fleet(&info) {
            fleet = Some(info.cmdline());
        }
        if info.ppid <= 1 {
            break;
        }
        cur = info.ppid;
    }
    fleet
}

/// Real /proc reader behind "check".
#[allow(clippy::similar_names)] // pid (arg) and ppid (/proc field): the two ids /proc itself names
fn proc_info(pid: u32) -> Option<ProcInfo> {
    let raw = std::fs::read_to_string(format!("/proc/{pid}/cmdline")).ok()?;
    let argv: Vec<String> = raw.split('\0').filter(|s| !s.is_empty()).map(str::to_string).collect();
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    // comm sits in parentheses and may contain spaces; ppid follows the state
    let close = stat.rfind(')')?;
    let mut fields = stat[close + 1..].split_whitespace();
    let _state = fields.next()?;
    let ppid = fields.next()?.parse().ok()?;
    let comm = std::fs::read_to_string(format!("/proc/{pid}/comm"))
        .unwrap_or_default()
        .trim()
        .to_string();
    Some(ProcInfo { argv, comm, ppid })
}

/// Check one shell command. None = allowed; Some(reason) = refuse.
#[must_use]
pub fn check(command: &str) -> Option<String> {
    if std::env::var("OMT_KILL_GUARD").ok().as_deref() == Some("0") {
        return None;
    }
    check_with(command, std::process::id(), &proc_info)
}

/// "check" over an injected /proc reader (tests).
#[must_use]
fn check_with(command: &str, self_pid: u32, info_of: &dyn Fn(u32) -> Option<ProcInfo>) -> Option<String> {
    for site in sites(command) {
        if site.probe {
            continue;
        }
        if site.pattern {
            let lowered = command.to_lowercase();
            let hit = PATTERN_MARKERS.iter().find(|m| lowered.contains(&m.to_lowercase()));
            if let Some(marker) = hit {
                return Some(format!(
                    "kill-guard: refused - this command would signal processes matched as {marker:?}, \
                     and the local omt fleet (hubs, web servers, --mode rpc agents, their children) \
                     is off-limits to agent sessions. Report stale processes (pid + cmdline) to the \
                     user instead of reaping them. To manage a server from a session, start it under \
                     the bash tool service mode (name=...). Operator override: OMT_KILL_GUARD=0."
                ));
            }
            continue;
        }
        for t in site.targets {
            if t == 0 {
                continue; // own process group: this session's shell
            }
            if t.abs() == 1 {
                return Some(
                    "kill-guard: refused - signalling pid/pgid 1 broadcasts to every process.".to_string(),
                );
            }
            #[allow(clippy::cast_sign_loss)]
            let pid = u32::try_from(t.abs()).unwrap_or(u32::MAX);
            if pid == self_pid {
                return Some("kill-guard: refused to signal this session's own agent process.".to_string());
            }
            if let Some(cmdline) = protector(pid, self_pid, info_of) {
                return Some(format!(
                    "kill-guard: refused - pid {pid} belongs to the local omt fleet, not to this session:\n  {cmdline}\nOnly processes this session started may be signalled. Report stale processes (pid + cmdline) to the user instead of reaping them. To manage a server from a session, start it under the bash tool service mode (name=...). Operator override: OMT_KILL_GUARD=0."
                ));
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    // pid, argv, comm, ppid
    const NODES: &[(u32, &str, &str, u32)] = &[
        (100, "/x/omt web --port 7467", "omt", 99),        // this session agent
        (200, "/x/omt --mode rpc", "omt", 300),            // sibling agent
        (300, "/x/omt hub serve", "omt", 1),               // the hub
        (310, "bun --smol ext-host/shard.ts", "bun", 300), // hub host child
        (400, "/x/old/omp-rs --mode rpc", "omp-rs", 1),    // renamed zombie agent
        (500, "python3 qa.py", "python3", 1),              // unrelated process
        (150, "bash -c nohup x", "bash", 100),             // own shell child
        (151, "/x/omt web --port 7469", "omt", 150),       // server this session spawned
    ];

    fn checker() -> impl Fn(&str) -> Option<String> {
        let m: HashMap<u32, ProcInfo> = NODES
            .iter()
            .map(|(pid, argv, comm, ppid)| {
                (
                    *pid,
                    ProcInfo {
                        argv: argv.split_whitespace().map(str::to_string).collect(),
                        comm: (*comm).to_string(),
                        ppid: *ppid,
                    },
                )
            })
            .collect();
        move |cmd: &str| {
            let f = |pid: u32| m.get(&pid).cloned();
            check_with(cmd, 100, &f)
        }
    }

    #[test]
    fn blocks_incident_style_commands() {
        let c = checker();
        // 2026-10-31: "kill 265666 416671 ..." against sibling fleet agents
        assert!(c("kill 200 400 2>/dev/null").is_some());
        assert!(c("kill -TERM 300").is_some());
        assert!(c("kill 310").is_some());
        // the pkill pattern from the incident class
        assert!(c("pkill -f '/omt hub serve --dir /tmp/'").is_some());
        assert!(c("pkill -f 'mode rpc' | wc -l").is_some());
        // broadcasts and self
        assert!(c("kill -- -1").is_some());
        assert!(c("kill 1").is_some());
        assert!(c("kill 100").is_some());
    }

    #[test]
    fn allows_session_owned_and_unrelated() {
        let c = checker();
        assert!(c("kill 150 151").is_none()); // own shell child + own server
        assert!(c("kill 500").is_none()); // unrelated user process
        assert!(c("kill $!; kill %1; kill $(cat /tmp/x.pid)").is_none());
        assert!(c("kill -0 300").is_none()); // probe sends nothing
        assert!(c("kill -l").is_none());
        assert!(c("pkill -f chromium").is_none()); // outside the fleet
        assert!(c("echo kill 500 && sleep 1").is_none());
        assert!(c("skillet -9 300").is_none());
    }

    #[test]
    fn refuses_even_mid_pipeline() {
        let c = checker();
        // 736/2285 unknown here -> only hub-side pids in the fake map resolve;
        // assert the real incident shape with resolvable fleet pids:
        assert!(c("sleep 1 ; kill 200 310").is_some());
    }
}
