//! `/ssh`: SSH host manager / remote bash target.
//! TS: `slash-commands/helpers/ssh.ts` (cli/ssh-cli.ts), `ssh/config-writer.ts`,
//! `ssh/utils.ts`. Config file and format match the TS implementation:
//! project `.omp/ssh.json`, user `~/.omp/agent/ssh.json`.
//!
//! `/ssh <host>` sets a process-wide bash-tool target: every subsequent
//! `bash` tool call runs `ssh <host> -- bash -lc <script>` on the remote
//! (see `tools/bash.rs`). `/ssh off` restores local execution.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::RwLock;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::rpc::slash::Outcome;
use crate::runtime::Runtime;

/// One saved host. Same shape as TS `SSHHostConfig` (`ssh/config-writer.ts`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SshHostConfig {
    pub host: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub username: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub port: Option<u16>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compat: Option<bool>,
}

/// `ssh.json` contents: `{"hosts": {"name": {...}}}`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SshConfigFile {
    #[serde(default)]
    pub hosts: BTreeMap<String, SshHostConfig>,
}

// ---------------------------------------------------------------------------
// Config file I/O (TS: config-writer.ts)
// ---------------------------------------------------------------------------

/// Read a config file; a missing file yields an empty config, a malformed one
/// is an error (TS `readSSHConfigFile`).
pub fn read_ssh_config_file(path: &std::path::Path) -> Result<SshConfigFile> {
    match std::fs::read_to_string(path) {
        Ok(text) => serde_json::from_str(&text)
            .with_context(|| format!("failed to parse SSH config file {}", path.display())),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(SshConfigFile::default()),
        Err(e) => Err(e).with_context(|| format!("failed to read {}", path.display())),
    }
}

/// Write atomically (temp file + rename), `0700` dir / `0600` file like TS.
pub fn write_ssh_config_file(path: &std::path::Path, config: &SshConfigFile) -> Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("failed to create {}", dir.display()))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700));
        }
    }
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_string_pretty(config)?.as_bytes())?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o600));
    }
    std::fs::rename(&tmp, path).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

/// TS `validateHostName`: `None` when valid, else the error message.
#[must_use]
pub fn validate_host_name(name: &str) -> Option<String> {
    if name.is_empty() {
        return Some("Host name cannot be empty".into());
    }
    if name.len() > 100 {
        return Some("Host name is too long (max 100 characters)".into());
    }
    if !name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.'))
    {
        return Some(
            "Host name can only contain letters, numbers, dash, underscore, and dot".into(),
        );
    }
    None
}

/// TS `addSSHHost`: fails on invalid name or duplicate.
pub fn add_ssh_host(path: &std::path::Path, name: &str, cfg: SshHostConfig) -> Result<()> {
    if let Some(err) = validate_host_name(name) {
        bail!("{err}");
    }
    if cfg.host.is_empty() {
        bail!("Host address cannot be empty");
    }
    let mut file = read_ssh_config_file(path)?;
    if file.hosts.contains_key(name) {
        bail!("Host \"{name}\" already exists in {}", path.display());
    }
    file.hosts.insert(name.to_string(), cfg);
    write_ssh_config_file(path, &file)
}

/// TS `removeSSHHost`: fails when the host is unknown.
pub fn remove_ssh_host(path: &std::path::Path, name: &str) -> Result<()> {
    let mut file = read_ssh_config_file(path)?;
    if file.hosts.remove(name).is_none() {
        bail!("Host \"{name}\" not found in {}", path.display());
    }
    write_ssh_config_file(path, &file)
}

/// Project scope: `<cwd>/.omp/ssh.json`; user scope: `~/.omp/agent/ssh.json`
/// (TS `getSSHConfigPath`).
#[must_use]
pub fn ssh_config_path(scope: Scope, cwd: &std::path::Path) -> PathBuf {
    match scope {
        Scope::Project => cwd.join(".omp").join("ssh.json"),
        Scope::User => home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".omp")
            .join("agent")
            .join("ssh.json"),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scope {
    Project,
    User,
}

fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .filter(|p| !p.as_os_str().is_empty())
}

// ---------------------------------------------------------------------------
// Remote target (process-wide; consulted by tools/bash.rs)
// ---------------------------------------------------------------------------

/// Resolved ssh invocation prefix: `[user@]host` plus `-p port` / `-i key`.
static REMOTE_TARGET: RwLock<Option<Vec<String>>> = RwLock::new(None);

/// TS `utils.ts` `buildSshTarget`: reject `-`-prefixed hosts/users (argument
/// injection) and render `[user@]host`.
pub fn build_ssh_target(username: Option<&str>, host: &str) -> Result<String> {
    if host.starts_with('-') {
        bail!(
            "Invalid SSH host \"{host}\": an SSH destination must not begin with \"-\" \
             (argument-injection guard)"
        );
    }
    if let Some(user) = username {
        if user.starts_with('-') {
            bail!(
                "Invalid SSH username \"{user}\": an SSH username must not begin with \"-\" \
                 (argument-injection guard)"
            );
        }
        return Ok(format!("{user}@{host}"));
    }
    Ok(host.to_string())
}

/// TS `quotePosixPath`: single-quote for a POSIX remote shell.
#[must_use]
pub fn quote_posix(value: &str) -> String {
    if value.is_empty() {
        return "''".into();
    }
    format!("'{}'", value.replace('\'', r"'\''"))
}

/// Current remote target as ssh argv *before* the remote command, i.e. the
/// full invocation is `ssh <argv> -- bash -lc <quoted script>`.
#[must_use]
pub fn remote_target_argv() -> Option<Vec<String>> {
    REMOTE_TARGET.read().ok()?.clone()
}

/// Full argv for a bash script under the current target (`None` = run local).
#[must_use]
pub fn remote_argv(script: &str) -> Option<Vec<String>> {
    remote_argv_for(remote_target_argv().as_deref(), script)
}

/// Pure core of [`remote_argv`]: the argv for `script` under an explicit
/// target prefix (`None` = run local). Testable without global state.
#[must_use]
pub fn remote_argv_for(prefix: Option<&[String]>, script: &str) -> Option<Vec<String>> {
    let mut argv = prefix.map(<[String]>::to_vec)?;
    argv.push("--".into());
    argv.push("bash".into());
    argv.push("-lc".into());
    argv.push(quote_posix(script));
    Some(argv)
}

fn set_remote_target(prefix: Vec<String>) {
    if let Ok(mut slot) = REMOTE_TARGET.write() {
        *slot = Some(prefix);
    }
}

fn clear_remote_target() {
    if let Ok(mut slot) = REMOTE_TARGET.write() {
        *slot = None;
    }
}

fn target_prefix(cfg: &SshHostConfig) -> Result<Vec<String>> {
    let mut argv = Vec::new();
    if let Some(port) = cfg.port {
        argv.push("-p".into());
        argv.push(port.to_string());
    }
    if let Some(key) = &cfg.key_path {
        argv.push("-i".into());
        argv.push(key.clone());
    }
    argv.push(build_ssh_target(cfg.username.as_deref(), &cfg.host)?);
    Ok(argv)
}

// ---------------------------------------------------------------------------
// Command parsing (`/ssh ...`)
// ---------------------------------------------------------------------------

/// Parsed `/ssh` subcommand.
#[derive(Debug, Clone, PartialEq)]
pub enum SshSub {
    List,
    /// Raw `add` tokens (name + kwargs), parsed by the command runner.
    Add {
        tokens: Vec<String>,
    },
    Remove {
        name: String,
    },
    Use {
        target: String,
    },
    Off,
}

/// One `key=value` (or `--key value` / `-key value`) pair after the action.
#[derive(Debug, Clone, PartialEq)]
pub struct Kwargs {
    pub pairs: Vec<(String, String)>,
}

/// Parse `add`/`remove` flags in both `--key value` and `key=value` forms.
#[must_use]
pub fn parse_kwargs(tokens: &[String]) -> Kwargs {
    let mut pairs = Vec::new();
    let mut i = 0;
    while i < tokens.len() {
        let tok = tokens[i].trim_start_matches('-');
        if let Some((key, inline)) = tokens[i].split_once('=') {
            pairs.push((key.trim_start_matches('-').to_string(), inline.to_string()));
        } else if matches!(
            tok,
            "name" | "host" | "user" | "port" | "key" | "desc" | "scope"
        ) && i + 1 < tokens.len()
        {
            pairs.push((tok.to_string(), tokens[i + 1].clone()));
            i += 1;
        }
        i += 1;
    }
    Kwargs { pairs }
}

/// Build an `SshHostConfig` from `add` keyword arguments.
pub fn config_from_kwargs(kwargs: &Kwargs) -> Result<SshHostConfig> {
    let mut cfg = SshHostConfig::default();
    for (key, value) in &kwargs.pairs {
        match key.as_str() {
            "host" => cfg.host.clone_from(value),
            "user" | "username" => cfg.username = Some(value.clone()),
            "port" => {
                cfg.port = Some(
                    value
                        .parse()
                        .map_err(|_| anyhow::anyhow!("invalid port \"{value}\""))?,
                );
            }
            "key" | "keypath" => cfg.key_path = Some(value.clone()),
            "desc" | "description" => cfg.description = Some(value.clone()),
            "compat" => cfg.compat = value.parse().ok(),
            _ => bail!("unknown option \"{key}\" (host, user, port, key, desc, compat)"),
        }
    }
    Ok(cfg)
}

/// Parse the full `/ssh` argument string.
#[must_use]
pub fn parse_args(args: &str) -> SshSub {
    let tokens: Vec<String> = args.split_whitespace().map(str::to_string).collect();
    match tokens.first().map(String::as_str) {
        None | Some("list") => SshSub::List,
        Some("add") => SshSub::Add {
            tokens: tokens[1.min(tokens.len())..].to_vec(),
        },
        Some("remove" | "rm") => SshSub::Remove {
            name: tokens.get(1).cloned().unwrap_or_default(),
        },
        Some("off") => SshSub::Off,
        Some(_) => SshSub::Use {
            target: tokens[0].clone(),
        },
    }
}

/// Usage line printed when `add` arguments do not parse.
pub const ADD_USAGE: &str = "usage: /ssh add <name> <host> [user@][:port] | /ssh add <name> host=<host> [user=<user>] [port=<port>] [key=<path>] [desc=<desc>] [compat=true|false]";

/// Positional (non-kwarg) tokens of an `add` argument list, for the
/// `/ssh add <name> <host> [user@][:port]` form.
fn positional_tokens(tokens: &[String]) -> Vec<String> {
    let is_key = |tok: &str| {
        matches!(
            tok,
            "name" | "host" | "user" | "port" | "key" | "desc" | "scope"
        )
    };
    let mut pos = Vec::new();
    let mut i = 0;
    while i < tokens.len() {
        let raw = &tokens[i];
        if raw.contains('=') {
            i += 1;
            continue;
        }
        let key = raw.trim_start_matches('-');
        if is_key(key) && i + 1 < tokens.len() {
            i += 2;
            continue;
        }
        if !is_key(key) {
            pos.push(raw.clone());
        }
        i += 1;
    }
    pos
}

/// Parse `add` arguments into `(name, config)`. Accepts both the kwargs form
/// (`<name> host=<h> [user=..] [port=..] ...`) and the positional form
/// (`<name> <host> [user@][:port]`); kwargs win over positionals.
pub fn parse_add(tokens: &[String]) -> Result<(String, SshHostConfig)> {
    let kwargs = parse_kwargs(tokens);
    let mut cfg = config_from_kwargs(&kwargs)?;
    let mut name = positional_tokens(tokens)
        .first()
        .cloned()
        .unwrap_or_default();
    if name.is_empty() {
        name = kwargs
            .pairs
            .iter()
            .find(|(k, _)| k == "name")
            .map(|(_, v)| v.clone())
            .unwrap_or_default();
    }
    let pos = positional_tokens(tokens);
    if let Some(dest) = pos.get(1) {
        let (user, host, inline_port) = match dest.split_once('@') {
            Some((user, rest)) if !user.is_empty() => {
                let (host, port) = match rest.split_once(':') {
                    Some((h, p)) => (h, Some(p)),
                    None => (rest, None),
                };
                (Some(user), host, port)
            }
            _ => (None, dest.as_str(), None),
        };
        if cfg.username.is_none() {
            cfg.username = user.map(str::to_string);
        }
        if cfg.host.is_empty() {
            cfg.host = host.to_string();
        }
        if let (Some(p), true) = (inline_port, cfg.port.is_none()) {
            cfg.port = Some(
                p.parse()
                    .map_err(|_| anyhow::anyhow!("invalid port \"{p}\""))?,
            );
        }
    }
    // Remaining positionals: `user@` and/or `[:port]`.
    for extra in pos.iter().skip(2) {
        if let Some((user, rest)) = extra.split_once('@') {
            if cfg.username.is_none() && !user.is_empty() {
                cfg.username = Some(user.to_string());
            }
            if cfg.host.is_empty() && !rest.is_empty() {
                cfg.host = rest.to_string();
            }
        } else if cfg.port.is_none() {
            let digits = extra.trim_start_matches(':');
            cfg.port = Some(
                digits
                    .parse()
                    .map_err(|_| anyhow::anyhow!("invalid port \"{extra}\""))?,
            );
        }
    }
    Ok((name, cfg))
}

// ---------------------------------------------------------------------------
// Slash command entry point
// ---------------------------------------------------------------------------

/// Slash-command entry point.
// Signature is fixed by the slash dispatcher: it must stay `async`.
#[allow(clippy::unused_async)]
pub async fn command(rt: &std::sync::Arc<Runtime>, args: &str) -> Outcome {
    let out = match run(rt, args) {
        Ok(text) => text,
        Err(e) => format!("ssh: {e:#}"),
    };
    rt.send(serde_json::json!({"type": "command_output", "text": out}));
    Outcome::Handled {
        agent_invoked: false,
    }
}

fn run(rt: &std::sync::Arc<Runtime>, args: &str) -> Result<String> {
    match parse_args(args) {
        SshSub::List => Ok(list_hosts(rt)),
        SshSub::Off => {
            let was = remote_target_argv().is_some();
            clear_remote_target();
            Ok(if was {
                "ssh: remote target cleared, bash runs locally".into()
            } else {
                "ssh: no remote target set".into()
            })
        }
        SshSub::Add { tokens } => {
            let (name, cfg) = match parse_add(&tokens) {
                Ok(parsed) => parsed,
                Err(e) => return Ok(format!("ssh: {e:#}\n{ADD_USAGE}")),
            };
            if name.is_empty() || cfg.host.is_empty() {
                return Ok(format!(
                    "ssh: both a host name and an address are required\n{ADD_USAGE}"
                ));
            }
            let path = ssh_config_path(Scope::Project, &rt.cwd);
            add_ssh_host(&path, &name, cfg)?;
            Ok(format!("Added SSH host \"{name}\" to {}", path.display()))
        }
        SshSub::Remove { name } => {
            let path = ssh_config_path(Scope::Project, &rt.cwd);
            remove_ssh_host(&path, &name)?;
            Ok(format!(
                "Removed SSH host \"{name}\" from {}",
                path.display()
            ))
        }
        SshSub::Use { target } => {
            let prefix = resolve_target(rt, &target)?;
            set_remote_target(prefix);
            Ok(format!(
                "ssh: bash tool now runs on {target} (`ssh` per command; /ssh off to stop)"
            ))
        }
    }
}

/// Resolve `/ssh <name>`: a saved host first (project, then user scope),
/// otherwise a raw destination (`host` or `user@host`).
fn resolve_target(rt: &std::sync::Arc<Runtime>, target: &str) -> Result<Vec<String>> {
    for scope in [Scope::Project, Scope::User] {
        let path = ssh_config_path(scope, &rt.cwd);
        if let Ok(file) = read_ssh_config_file(&path) {
            if let Some(cfg) = file.hosts.get(target) {
                return target_prefix(cfg);
            }
        }
    }
    // Raw destination: split one optional `user@` prefix.
    match target.split_once('@') {
        Some((user, host)) => {
            let rendered = build_ssh_target(Some(user), host)?;
            Ok(vec![rendered])
        }
        None => Ok(vec![build_ssh_target(None, target)?]),
    }
}

fn list_hosts(rt: &std::sync::Arc<Runtime>) -> String {
    let mut out = String::new();
    for (scope, title) in [
        (Scope::Project, "Project SSH Hosts (.omp/ssh.json):"),
        (Scope::User, "User SSH Hosts (~/.omp/agent/ssh.json):"),
    ] {
        let path = ssh_config_path(scope, &rt.cwd);
        let Ok(file) = read_ssh_config_file(&path) else {
            continue;
        };
        if file.hosts.is_empty() {
            continue;
        }
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(title);
        out.push('\n');
        for (name, cfg) in &file.hosts {
            out.push_str(&format_host(name, cfg));
            out.push('\n');
        }
    }
    if out.is_empty() {
        "No SSH hosts configured".into()
    } else {
        // Trailing newline tidy-up from the per-host loop.
        out.trim_end().to_string()
    }
}

/// TS `printHosts` line: `name  host  user  port:N  key  - description`.
#[must_use]
fn format_host(name: &str, cfg: &SshHostConfig) -> String {
    let mut parts = vec![name.to_string(), cfg.host.clone()];
    if let Some(user) = &cfg.username {
        parts.push(user.clone());
    }
    if let Some(port) = cfg.port.filter(|p| *p != 22) {
        parts.push(format!("port:{port}"));
    }
    if let Some(key) = &cfg.key_path {
        parts.push(key.clone());
    }
    if let Some(desc) = &cfg.description {
        parts.push(format!("- {desc}"));
    }
    parts.join("  ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_host_names() {
        assert!(validate_host_name("web-1.example").is_none());
        assert_eq!(
            validate_host_name(""),
            Some("Host name cannot be empty".into())
        );
        assert!(validate_host_name("bad name").is_some());
        assert!(validate_host_name("$(cmd)").is_some());
        assert!(validate_host_name(&"x".repeat(101)).is_some());
    }

    #[test]
    fn parses_kwargs_both_forms() {
        let tokens: Vec<String> = vec![
            "--host".into(),
            "h1".into(),
            "--port".into(),
            "2222".into(),
            "key=/tmp/k".into(),
        ];
        let kw = parse_kwargs(&tokens);
        assert_eq!(
            kw.pairs,
            vec![
                ("host".into(), "h1".into()),
                ("port".into(), "2222".into()),
                ("key".into(), "/tmp/k".into()),
            ]
        );
        let inline: Vec<String> = "host=h2 user=root"
            .split_whitespace()
            .map(str::to_string)
            .collect();
        assert_eq!(
            parse_kwargs(&inline).pairs,
            vec![("host".into(), "h2".into()), ("user".into(), "root".into())]
        );
    }

    #[test]
    fn builds_config_from_kwargs() {
        let tokens: Vec<String> = "web host=10.0.0.1 user=root port=2200"
            .split_whitespace()
            .map(str::to_string)
            .collect();
        let (name, cfg) = parse_add(&tokens).unwrap();
        assert_eq!(name, "web");
        assert_eq!(
            cfg,
            SshHostConfig {
                host: "10.0.0.1".into(),
                username: Some("root".into()),
                port: Some(2200),
                ..SshHostConfig::default()
            }
        );
        assert!(parse_add(&["x".into(), "port=abc".into()]).is_err());
    }

    #[test]
    fn parses_add_positional_form() {
        let toks = |s: &str| s.split_whitespace().map(str::to_string).collect::<Vec<_>>();
        let (name, cfg) = parse_add(&toks("box localhost")).unwrap();
        assert_eq!(name, "box");
        assert_eq!(cfg.host, "localhost");
        assert_eq!(cfg.username, None);
        assert_eq!(cfg.port, None);

        // <name> <host> <user@> <port>
        let (name, cfg) = parse_add(&toks("box 10.0.0.5 root@ 2222")).unwrap();
        assert_eq!(name, "box");
        assert_eq!(cfg.host, "10.0.0.5");
        assert_eq!(cfg.username.as_deref(), Some("root"));
        assert_eq!(cfg.port, Some(2222));

        // <name> <user@host> [:port]
        let (name, cfg) = parse_add(&toks("box root@10.0.0.5 :2200")).unwrap();
        assert_eq!(name, "box");
        assert_eq!(cfg.host, "10.0.0.5");
        assert_eq!(cfg.username.as_deref(), Some("root"));
        assert_eq!(cfg.port, Some(2200));

        // kwargs win over positionals
        let (name, cfg) = parse_add(&toks("box localhost port=22")).unwrap();
        assert_eq!(name, "box");
        assert_eq!(cfg.host, "localhost");
        assert_eq!(cfg.port, Some(22));

        // bare port (no colon)
        let (_, cfg) = parse_add(&toks("box localhost 2222")).unwrap();
        assert_eq!(cfg.port, Some(2222));

        // invalid port is an error
        assert!(parse_add(&toks("box localhost notaport")).is_err());

        // missing host leaves cfg.host empty (run() prints usage)
        let (_, cfg) = parse_add(&toks("box")).unwrap();
        assert_eq!(cfg.host, "");
    }

    #[test]
    fn parses_subcommands() {
        assert_eq!(parse_args(""), SshSub::List);
        assert_eq!(parse_args("list"), SshSub::List);
        assert_eq!(parse_args("off"), SshSub::Off);
        assert_eq!(
            parse_args("remove web"),
            SshSub::Remove { name: "web".into() }
        );
        assert_eq!(
            parse_args("web.example"),
            SshSub::Use {
                target: "web.example".into()
            }
        );
        assert_eq!(
            parse_args("root@10.0.0.9"),
            SshSub::Use {
                target: "root@10.0.0.9".into()
            }
        );
        assert_eq!(
            parse_args("add web host=h1"),
            SshSub::Add {
                tokens: vec!["web".into(), "host=h1".into()]
            }
        );
    }

    #[test]
    fn guards_destination_injection() {
        assert!(build_ssh_target(None, "-oProxyCommand=evil").is_err());
        assert!(build_ssh_target(Some("-x"), "host").is_err());
        assert_eq!(build_ssh_target(Some("root"), "h").unwrap(), "root@h");
    }

    #[test]
    fn builds_remote_argv() {
        // Pure fn only: never touch the process-wide REMOTE_TARGET global
        // (it would leak into other tests in this binary).
        assert!(remote_argv_for(None, "ls").is_none());
        let prefix: Vec<String> = vec!["-p".into(), "2222".into(), "web".into()];
        assert_eq!(
            remote_argv_for(Some(&prefix), "echo hi"),
            Some(vec![
                "-p".into(),
                "2222".into(),
                "web".into(),
                "--".into(),
                "bash".into(),
                "-lc".into(),
                "'echo hi'".into(),
            ])
        );
        let prefix: Vec<String> = vec!["h".into()];
        // POSIX single-quote: every embedded `'` becomes `'\''`, the whole
        // script wrapped in `'…'`.
        assert_eq!(
            remote_argv_for(Some(&prefix), "echo 'a'"),
            Some(vec![
                "h".into(),
                "--".into(),
                "bash".into(),
                "-lc".into(),
                "'echo '\\''a'\\'''".into(),
            ])
        );
    }

    #[test]
    fn config_roundtrip_and_crud() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ssh.json");
        assert_eq!(
            read_ssh_config_file(&path).unwrap(),
            SshConfigFile::default()
        );
        add_ssh_host(
            &path,
            "web",
            SshHostConfig {
                host: "1.2.3.4".into(),
                port: Some(2200),
                ..Default::default()
            },
        )
        .unwrap();
        let dup = SshHostConfig {
            host: "x".into(),
            ..SshHostConfig::default()
        };
        assert!(add_ssh_host(&path, "web", dup).is_err());
        let file = read_ssh_config_file(&path).unwrap();
        assert_eq!(file.hosts["web"].host, "1.2.3.4");
        remove_ssh_host(&path, "web").unwrap();
        assert!(remove_ssh_host(&path, "web").is_err());
        assert!(read_ssh_config_file(&path).unwrap().hosts.is_empty());
    }
}
