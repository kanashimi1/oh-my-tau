//! `/skills`, `/marketplace`: skill and plugin catalogs. TS:
//! `slash-commands/builtin-skills.ts`, `builtin-marketplace.ts`,
//! `skillshare/` and `extensibility/plugins/marketplace/`.
//!
//! Skills registry (`skills.omp.sh`, TS `pi-wire/skillshare.ts`): scoped
//! `@scope/name` packages; `GET /api/v1/search?q=`, packument at
//! `/api/v1/skills/@scope/name`, tarball at `.../versions/<v>/tarball`.
//! Installed registry skills keep TS's files: `skills.json` + `skills.lock.json`
//! in the agent dir (global) or the project `.omp/` dir, with unpacked copies in
//! the shared store `<~/.omp>/skillshare/@scope/name/<version>/`.
//!
//! Marketplace (TS `extensibility/plugins/marketplace/`): `~/.omp/marketplaces.json`
//! lists catalog sources; catalogs are cloned/fetched into
//! `~/.omp/plugins/cache/marketplaces/<name>`; installs are copied to
//! `~/.omp/plugins/cache/plugins/<name>` and recorded in `installed_plugins.json`
//! (`version: 2`, `plugins: {id: [entry…]}`) under `~/.omp/plugins/` (user scope)
//! or `<cwd>/.omp/plugins/` (project scope).

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use base64::Engine as _;
use serde_json::{json, Map, Value};

use crate::rpc::slash::Outcome;
use crate::runtime::Runtime;

const DEFAULT_SKILLS_URL: &str = "https://skills.omp.sh";
const SKILLS_MANIFEST_FILE: &str = "skills.json";
const SKILLS_LOCK_FILE: &str = "skills.lock.json";
const SKILLS_LOCK_VERSION: u32 = 1;
const STORE_INTEGRITY_FILE: &str = ".skillshare-integrity";
const MARKETPLACES_REGISTRY: &str = "marketplaces.json";
const CATALOG_RELPATHS: &[&str] = &[
    ".omp-plugin/marketplace.json",
    ".claude-plugin/marketplace.json",
];
const USAGE: &str = "Skill registry (skills.omp.sh) commands:
  /skills                                        List local skills
  /skills search <query>                         Search the registry
  /skills install <@scope/name[@range]>… [-g]    Install into this project (-g: user-global)
  /skills installed                              List installed registry skills
  /skills remove <@scope/name> [-g]              Remove an installed skill
  /skills update [@scope/name…] [-g]             Update within the ranges in skills.json";

const MARKETPLACE_USAGE: &str = "Marketplace commands:
  /marketplace                              List configured marketplaces
  /marketplace add <source>                  Add a marketplace (e.g. owner/repo)
  /marketplace remove <name>                 Remove a marketplace
  /marketplace update [name]                 Re-fetch catalog(s)
  /marketplace list                          List configured marketplaces
  /marketplace discover [marketplace]        Browse available plugins
  /marketplace install <name@marketplace>    Install a plugin
  /marketplace uninstall <name@marketplace>  Uninstall a plugin
  /marketplace installed                     List installed plugins

Quick start:
  /marketplace add anthropics/claude-plugins-official";

fn home_dir() -> PathBuf {
    std::env::var_os("HOME").map_or_else(|| PathBuf::from("/"), PathBuf::from)
}

/// omp config root (`~/.omp` for the default profile).
fn config_root() -> PathBuf {
    home_dir().join(".omp")
}

/// omp agent dir (TS `getAgentDir()`).
fn agent_dir() -> PathBuf {
    config_root().join("agent")
}

fn output(rt: &Runtime, text: &str) {
    rt.send(json!({"type": "command_output", "text": text}));
}

fn handled() -> Outcome {
    Outcome::Handled {
        agent_invoked: false,
    }
}

/// Split a slash-command argument string into `(verb, rest)` (slash.rs `subcommand`).
fn split_sub(input: &str) -> (String, String) {
    let trimmed = input.trim();
    match trimmed.split_once(char::is_whitespace) {
        None => (trimmed.to_lowercase(), String::new()),
        Some((v, rest)) => (v.to_lowercase(), rest.trim().to_string()),
    }
}

/// Write JSON atomically with a `.tmp` + rename (TS `writeJsonAtomic`);
/// top-level `skills` keys are emitted sorted (TS `sortedRecord`).
fn write_json_sorted(path: &Path, mut doc: Map<String, Value>) -> Result<(), String> {
    if let Some(skills) = doc.get("skills").and_then(Value::as_object).cloned() {
        let mut sorted = Map::new();
        let mut keys: Vec<&String> = skills.keys().collect();
        keys.sort();
        for key in keys {
            sorted.insert(key.clone(), skills[key].clone());
        }
        doc.insert("skills".into(), Value::Object(sorted));
    }
    let text = format!("{}\n", Value::Object(doc));
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("create {}: {e}", parent.display()))?;
    }
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, text).map_err(|e| format!("write {}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, path).map_err(|e| format!("write {}: {e}", path.display()))
}

// ---------------------------------------------------------------------------
// Skill spec parsing (TS `parseSkillSpec`, `SKILL_SPEC_RE`)
// ---------------------------------------------------------------------------

/// Scope segment: `[a-z0-9][a-z0-9_]{2,31}` (TS `SKILL_SCOPE_RE`).
pub fn is_valid_scope(scope: &str) -> bool {
    let b = scope.as_bytes();
    let len_ok = (3..=32).contains(&b.len());
    let head_ok = b
        .first()
        .is_some_and(|&c| c.is_ascii_lowercase() || c.is_ascii_digit());
    len_ok
        && head_ok
        && b[1..]
            .iter()
            .all(|&c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == b'_')
}

/// Name segment: `[a-z0-9]+(?:-[a-z0-9]+)*`, max 64 chars (TS `SKILL_NAME_RE`).
pub fn is_valid_name(name: &str) -> bool {
    if name.is_empty() || name.len() > 64 {
        return false;
    }
    name.split('-').all(|seg| {
        !seg.is_empty()
            && seg
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit())
    })
}

/// A parsed `<@scope/name[@range]>` install spec.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkillSpec {
    pub scope: String,
    pub name: String,
    /// Exact version, semver range, or dist-tag; absent without an `@…` suffix.
    pub range: Option<String>,
}

/// Parse `@scope/name[@range]`; `None` when the spec is not a registry package id.
pub fn parse_skill_spec(spec: &str) -> Option<SkillSpec> {
    let spec = spec.trim();
    let rest = spec.strip_prefix('@')?;
    let (scope, name_range) = rest.split_once('/')?;
    if !is_valid_scope(scope) {
        return None;
    }
    let (name, range) = match name_range.split_once('@') {
        Some((n, r)) if !n.is_empty() && !r.is_empty() => (n, Some(r.to_string())),
        Some(_) => return None,
        None => (name_range, None),
    };
    if !is_valid_name(name) {
        return None;
    }
    Some(SkillSpec {
        scope: scope.to_string(),
        name: name.to_string(),
        range,
    })
}

/// `@scope/name` id validation (TS `SKILL_ID_RE`: ids never carry a range).
pub fn parse_skill_id(id: &str) -> Option<(String, String)> {
    let spec = parse_skill_spec(id)?;
    if spec.range.is_some() {
        return None;
    }
    Some((spec.scope, spec.name))
}

/// Exact semver: `\d+\.\d+\.\d+(-pre)?(+build)?` (TS `SEMVER_RE`).
pub fn is_exact_semver(v: &str) -> bool {
    let (v, _build) = v.split_once('+').unwrap_or((v, ""));
    let (core, pre) = v.split_once('-').unwrap_or((v, ""));
    let mut parts = core.split('.');
    for _ in 0..3 {
        match parts.next() {
            Some(p) if !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()) => {}
            _ => return false,
        }
    }
    if parts.next().is_some() {
        return false;
    }
    pre.bytes()
        .all(|b| b.is_ascii_alphanumeric() || b == b'.' || b == b'-')
}

// ---------------------------------------------------------------------------
// Semver resolution (TS `resolveSkillVersion`)
// ---------------------------------------------------------------------------

fn semver_key(v: &str) -> Option<(u64, u64, u64, String)> {
    let (v, _build) = v.split_once('+').unwrap_or((v, ""));
    let (core, pre) = v.split_once('-').map_or((v, ""), |(c, p)| (c, p));
    let mut it = core.split('.');
    let major: u64 = it.next()?.parse().ok()?;
    let minor: u64 = it.next()?.parse().ok()?;
    let patch: u64 = it.next()?.parse().ok()?;
    if it.next().is_some() {
        return None;
    }
    Some((major, minor, patch, pre.to_string()))
}

fn semver_gt(a: &str, b: &str) -> bool {
    match (semver_key(a), semver_key(b)) {
        // A prerelease sorts below its release.
        (Some(x), Some(y)) => {
            (x.0, x.1, x.2, u8::from(x.3.is_empty()), x.3.clone())
                > (y.0, y.1, y.2, u8::from(y.3.is_empty()), y.3.clone())
        }
        _ => false,
    }
}

/// Does `version` satisfy a minimal npm-style range (`^`, `~`, `>=`, `>`,
/// `<=`, `<`, exact, `*`, `||` alternates)? Partial ranges (`1`, `1.2`) are
/// padded like npm.
pub fn semver_satisfies(version: &str, range: &str) -> bool {
    let range = range.trim();
    if range == "*" || range.is_empty() {
        return !version.is_empty();
    }
    range
        .split("||")
        .any(|part| comparators_satisfy(version, part.trim()))
}

fn comparators_satisfy(version: &str, range: &str) -> bool {
    if range.is_empty() {
        return false;
    }
    range.split_whitespace().all(|comp| {
        let (op, ver) = if let Some(rest) = comp.strip_prefix(">=") {
            (">=", rest)
        } else if let Some(rest) = comp.strip_prefix("<=") {
            ("<=", rest)
        } else if let Some(rest) = comp.strip_prefix('>') {
            (">", rest)
        } else if let Some(rest) = comp.strip_prefix('<') {
            ("<", rest)
        } else if let Some(rest) = comp.strip_prefix('^') {
            ("^", rest)
        } else if let Some(rest) = comp.strip_prefix('~') {
            ("~", rest)
        } else {
            ("=", comp.trim_start_matches('='))
        };
        let nums: Vec<u64> = ver
            .split('.')
            .map(|p| p.split('-').next().unwrap_or("0"))
            .map_while(|p| p.parse().ok())
            .collect();
        let key = match nums.as_slice() {
            [maj] => (*maj, 0, 0),
            [maj, min] => (*maj, *min, 0),
            [maj, min, pat] => (*maj, *min, *pat),
            _ => return false,
        };
        let vk = match semver_key(version) {
            Some(k) => (k.0, k.1, k.2),
            None => return false,
        };
        match op {
            "^" => {
                vk >= key
                    && if key.0 > 0 {
                        vk.0 == key.0
                    } else if key.1 > 0 {
                        vk.0 == 0 && vk.1 == key.1
                    } else {
                        vk == (0, 0, key.2)
                    }
            }
            "~" => vk >= key && vk.0 == key.0 && (nums.len() == 1 || vk.1 == key.1),
            ">=" => vk >= key,
            ">" => vk > key,
            "<=" => vk <= key,
            "<" => vk < key,
            // Exact (npm treats a partial exact as a prefix range).
            _ => {
                if is_exact_semver(ver) {
                    semver_key(version) == semver_key(ver)
                } else {
                    vk.0 == key.0
                        && (nums.len() < 2 || vk.1 == key.1)
                        && (nums.len() < 3 || vk.2 == key.2)
                }
            }
        }
    })
}

/// Resolve a request (exact version, dist-tag, or range) against a packument
/// (TS `resolveSkillVersion`). Returns the picked version string.
pub fn resolve_packument_version(packument: &Value, request: &str) -> Result<String, String> {
    let id = format!(
        "@{}/{}",
        packument
            .get("scope")
            .and_then(Value::as_str)
            .unwrap_or("?"),
        packument.get("name").and_then(Value::as_str).unwrap_or("?")
    );
    let versions = packument
        .get("versions")
        .and_then(Value::as_object)
        .ok_or_else(|| format!("{id}: packument has no versions"))?;
    let dist_tags = packument
        .get("distTags")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    if is_exact_semver(request) {
        if versions.contains_key(request) {
            return Ok(request.to_string());
        }
        return Err(format!("{id}@{request} does not exist"));
    }
    if let Some(tagged) = dist_tags.get(request).and_then(Value::as_str) {
        return Ok(tagged.to_string());
    }
    let mut best: Option<String> = None;
    for candidate in versions.keys() {
        let yanked = versions
            .get(candidate)
            .and_then(|s| s.get("yanked"))
            .and_then(Value::as_bool)
            .unwrap_or(false);
        if yanked || !semver_satisfies(candidate, request) {
            continue;
        }
        if best.as_ref().is_none_or(|b| semver_gt(candidate, b)) {
            best = Some(candidate.clone());
        }
    }
    best.ok_or_else(|| format!("No non-yanked version of {id} matches \"{request}\""))
}

// ---------------------------------------------------------------------------
// Registry HTTP client (TS `SkillshareClient`)
// ---------------------------------------------------------------------------

fn registry_base(rt: &Runtime) -> String {
    let mut url = omp_settings::Settings::load(&rt.cwd)
        .get_string("skills.registryUrl")
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_SKILLS_URL.to_string());
    while url.ends_with('/') {
        url.pop();
    }
    url
}

async fn registry_get_json(base: &str, route: &str) -> Result<Value, String> {
    let url = format!("{base}{route}");
    let resp = reqwest::Client::new()
        .get(&url)
        .send()
        .await
        .map_err(|e| format!("Skill registry unreachable at {url}: {e}"))?;
    let status = resp.status();
    let body = resp.text().await.unwrap_or_default();
    if !status.is_success() {
        let reason = serde_json::from_str::<Value>(&body)
            .ok()
            .and_then(|v| v.get("error").and_then(Value::as_str).map(Into::into))
            .unwrap_or_else(|| truncate_body(&body));
        return Err(format!(
            "Skill registry error: HTTP {status} from {url}: {reason}"
        ));
    }
    serde_json::from_str(&body)
        .map_err(|e| format!("Skill registry returned invalid JSON from {url}: {e}"))
}

async fn registry_get_bytes(base: &str, route: &str) -> Result<Vec<u8>, String> {
    let url = format!("{base}{route}");
    let resp = reqwest::Client::new()
        .get(&url)
        .send()
        .await
        .map_err(|e| format!("Skill registry unreachable at {url}: {e}"))?;
    let status = resp.status();
    if !status.is_success() {
        return Err(format!(
            "Skill registry error: HTTP {status} fetching {url}"
        ));
    }
    resp.bytes()
        .await
        .map(|b| b.to_vec())
        .map_err(|e| format!("Skill registry download failed for {url}: {e}"))
}

fn truncate_body(body: &str) -> String {
    let t = body.trim();
    if t.len() <= 200 {
        t.to_string()
    } else {
        format!("{}…", &t[..200])
    }
}

/// SRI `sha512-<base64>` of a tarball (TS `computeIntegrity`).
pub fn sri_sha512(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha512};
    let digest = Sha512::digest(bytes);
    format!(
        "sha512-{}",
        base64::engine::general_purpose::STANDARD.encode(digest)
    )
}

fn urlencode(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => {
                let _ = write!(out, "%{b:02X}");
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Install locations (TS `skillshare/manifest.ts`)
// ---------------------------------------------------------------------------

struct InstallPaths {
    dir: PathBuf,
    manifest: PathBuf,
    lock: PathBuf,
}

fn global_install_paths() -> InstallPaths {
    let dir = agent_dir();
    InstallPaths {
        manifest: dir.join(SKILLS_MANIFEST_FILE),
        lock: dir.join(SKILLS_LOCK_FILE),
        dir,
    }
}

/// Nearest `.omp/` walking up from `cwd`; `None` when `cwd` is the home
/// directory or the config root itself (TS: "the home directory is not a
/// project").
fn project_install_paths(cwd: &Path) -> Option<InstallPaths> {
    for dir in cwd.ancestors() {
        if dir == config_root() || dir == home_dir() {
            return None;
        }
        let omp = dir.join(".omp");
        if omp.is_dir() {
            return Some(InstallPaths {
                manifest: omp.join(SKILLS_MANIFEST_FILE),
                lock: omp.join(SKILLS_LOCK_FILE),
                dir: omp,
            });
        }
    }
    None
}

fn install_paths(cwd: &Path, global: bool) -> Result<InstallPaths, String> {
    if global {
        return Ok(global_install_paths());
    }
    project_install_paths(cwd).ok_or_else(|| {
        "No project `.omp/` directory found from the current directory; pass -g to install for your user.".to_string()
    })
}

/// Shared store: `<~/.omp>/skillshare/@scope/name/<version>/` (TS `getSkillStorePath`).
fn skill_store_path(scope: &str, name: &str, version: &str) -> PathBuf {
    config_root()
        .join("skillshare")
        .join(format!("@{scope}"))
        .join(name)
        .join(version)
}

fn read_json_file(path: &Path) -> Result<Value, String> {
    match std::fs::read_to_string(path) {
        Ok(text) => serde_json::from_str(&text)
            .map_err(|e| format!("{}: invalid JSON: {e}", path.display())),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(json!({})),
        Err(e) => Err(format!("read {}: {e}", path.display())),
    }
}

fn manifest_of(path: &Path) -> Result<Map<String, Value>, String> {
    Ok(read_json_file(path)?
        .get("skills")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default())
}

fn lock_of(path: &Path) -> Result<Map<String, Value>, String> {
    Ok(read_json_file(path)?
        .get("skills")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default())
}

// ---------------------------------------------------------------------------
// Tarball unpack (TS `skillshare/tar.ts`; minimal ustar reader, no new deps)
// ---------------------------------------------------------------------------

/// Extract a `.tgz` into `dest`, rejecting path traversal (TS unpack guard).
/// Returns the number of regular files written.
pub fn unpack_tgz(tgz: &[u8], dest: &Path) -> Result<usize, String> {
    use std::io::Read;
    let mut reader = flate2::read::GzDecoder::new(tgz);
    let mut written = 0usize;
    loop {
        let mut header = [0u8; 512];
        match reader.read_exact(&mut header) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(format!("read tarball: {e}")),
        }
        if header.iter().all(|&b| b == 0) {
            break; // End of archive.
        }
        let name = str_field(&header[0..100]);
        let prefix = str_field(&header[345..500]);
        let size = octal_field(&header[124..136]);
        let typeflag = header[156];
        let full = if prefix.is_empty() {
            name.to_string()
        } else {
            format!("{prefix}/{name}")
        };
        if full.is_empty() {
            return Err("tarball: empty entry path".to_string());
        }
        if Path::new(&full).is_absolute() || full.split('/').any(|seg| seg == "..") {
            return Err(format!("tarball: unsafe entry path {full:?}"));
        }
        let mut content = vec![0u8; size];
        reader
            .read_exact(&mut content)
            .map_err(|e| format!("read tar entry {full:?}: {e}"))?;
        let is_regular = typeflag == b'0' || typeflag == 0;
        if !is_regular {
            continue;
        }
        let out = dest.join(&full);
        if let Some(parent) = out.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("create {}: {e}", parent.display()))?;
        }
        std::fs::write(&out, &content).map_err(|e| format!("write {}: {e}", out.display()))?;
        written += 1;
    }
    Ok(written)
}

fn str_field(bytes: &[u8]) -> &str {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    std::str::from_utf8(&bytes[..end]).unwrap_or("")
}

fn octal_field(bytes: &[u8]) -> usize {
    let s = str_field(bytes);
    usize::from_str_radix(s.trim_matches([' ', '\0']), 8).unwrap_or(0)
}

/// Unpack with integrity check + atomic-ish staging (TS `storeSkillVersion`).
/// `server_integrity` is the packument's SRI when present; it wins.
fn store_skill_version(
    scope: &str,
    name: &str,
    version: &str,
    server_integrity: Option<&str>,
    tgz: &[u8],
) -> Result<PathBuf, String> {
    let computed = sri_sha512(tgz);
    if let Some(expected) = server_integrity {
        if expected != computed {
            return Err(format!(
                "@{scope}/{name}@{version}: integrity mismatch (expected {expected}, got {computed})"
            ));
        }
    }
    let integrity = server_integrity.unwrap_or(computed.as_str());
    let store = skill_store_path(scope, name, version);
    if std::fs::read_to_string(store.join(STORE_INTEGRITY_FILE))
        .is_ok_and(|s| s.trim() == integrity)
    {
        return Ok(store); // Already unpacked from the same tarball.
    }
    let staging = store.with_extension(format!("staging-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&staging);
    std::fs::create_dir_all(&staging).map_err(|e| format!("create {}: {e}", staging.display()))?;
    unpack_tgz(tgz, &staging)?;
    std::fs::write(staging.join(STORE_INTEGRITY_FILE), format!("{integrity}\n"))
        .map_err(|e| format!("write integrity marker: {e}"))?;
    let _ = std::fs::remove_dir_all(&store);
    std::fs::rename(&staging, &store)
        .map_err(|e| format!("promote store dir {}: {e}", store.display()))?;
    Ok(store)
}

// ---------------------------------------------------------------------------
// `/skills`
// ---------------------------------------------------------------------------

/// Slash-command entry point.
pub async fn command(rt: &Arc<Runtime>, args: &str) -> Outcome {
    let (verb, rest) = split_sub(args);
    let result: Result<String, String> = match verb.as_str() {
        "" | "help" => Ok(USAGE.to_string()),
        "search" if !rest.is_empty() => search(rt, &rest).await,
        "search" => Err("Usage: /skills search <query>".to_string()),
        "install" | "update" => match parse_targets(&rest) {
            Ok((names, global)) => {
                if names.is_empty() && verb == "install" {
                    Err("Usage: /skills install <@scope/name[@range]>… [--global]".to_string())
                } else {
                    install(rt, &names, global).await
                }
            }
            Err(e) => Err(e),
        },
        "installed" | "list" => Ok(list_installed(rt)),
        "remove" | "uninstall" | "rm" => match parse_targets(&rest) {
            Ok((names, global)) if !names.is_empty() => remove(rt, &names, global),
            Ok(_) => Err("Usage: /skills remove <@scope/name> [-g]".to_string()),
            Err(e) => Err(e),
        },
        other => Err(format!("Unknown /skills subcommand: {other}\n\n{USAGE}")),
    };
    match result {
        Ok(text) => output(rt, &text),
        Err(e) => output(rt, &format!("Skills: {e}")),
    }
    handled()
}

/// Split `<name…> [--global|-g]`; unknown flags are errors (TS `parseTargets`).
pub fn parse_targets(rest: &str) -> Result<(Vec<String>, bool), String> {
    let mut names = Vec::new();
    let mut global = false;
    for token in rest.split_whitespace() {
        if token == "--global" || token == "-g" {
            global = true;
        } else if token.starts_with('-') {
            return Err(format!("Unknown option: {token}\n\n{USAGE}"));
        } else {
            names.push(token.to_string());
        }
    }
    Ok((names, global))
}

async fn search(rt: &Runtime, query: &str) -> Result<String, String> {
    let base = registry_base(rt);
    let response =
        registry_get_json(&base, &format!("/api/v1/search?q={}", urlencode(query))).await?;
    Ok(format_skill_search(&response))
}

/// TS `formatSkillSearch` (plain paint).
pub fn format_skill_search(response: &Value) -> String {
    let Some(hits) = response.get("hits").and_then(Value::as_array) else {
        return "No skills found.".to_string();
    };
    if hits.is_empty() {
        return "No skills found.".to_string();
    }
    let mut lines = Vec::new();
    for hit in hits {
        let scope = hit.get("scope").and_then(Value::as_str).unwrap_or("?");
        let name = hit.get("name").and_then(Value::as_str).unwrap_or("?");
        let version = hit.get("version").and_then(Value::as_str).unwrap_or("?");
        lines.push(format!("@{scope}/{name} {version}"));
        if let Some(desc) = hit.get("description").and_then(Value::as_str) {
            if !desc.is_empty() {
                lines.push(format!("  {desc}"));
            }
        }
        let downloads = hit
            .get("weeklyDownloads")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let publisher = hit
            .get("publisher")
            .and_then(|p| p.get("username"))
            .and_then(Value::as_str)
            .unwrap_or("?");
        let mut meta = vec![
            format!("{downloads} weekly downloads"),
            format!("by {publisher}"),
        ];
        if let Some(kws) = hit.get("keywords").and_then(Value::as_array) {
            let kws: Vec<&str> = kws.iter().filter_map(Value::as_str).collect();
            if !kws.is_empty() {
                meta.push(kws.join(", "));
            }
        }
        lines.push(format!("  {}", meta.join(" · ")));
        if let Some(dep) = hit.get("deprecated").and_then(Value::as_str) {
            lines.push(format!("  deprecated: {dep}"));
        }
    }
    let total = response.get("total").and_then(Value::as_u64).unwrap_or(0);
    let page = response.get("page").and_then(Value::as_u64).unwrap_or(1);
    let per_page = response
        .get("perPage")
        .and_then(Value::as_u64)
        .unwrap_or(hits.len() as u64);
    let shown = (page - 1).saturating_mul(per_page) + hits.len() as u64;
    if total > shown {
        lines.push(format!("… {} more (page {page})", total - shown));
    }
    lines.join("\n")
}

async fn install(rt: &Arc<Runtime>, specs: &[String], global: bool) -> Result<String, String> {
    let base = registry_base(rt);
    let paths = install_paths(&rt.cwd, global)?;
    let mut manifest = manifest_of(&paths.manifest)?;
    let mut lock = lock_of(&paths.lock)?;
    let mut changes: Vec<String> = Vec::new();
    for spec in specs {
        let parsed = parse_skill_spec(spec)
            .ok_or_else(|| format!("Invalid skill spec {spec:?} (expected @scope/name[@range])"))?;
        let request = parsed.range.clone().unwrap_or_else(|| "latest".to_string());
        let packument = registry_get_json(
            &base,
            &format!("/api/v1/skills/@{}/{}", parsed.scope, parsed.name),
        )
        .await?;
        let version = resolve_packument_version(&packument, &request)?;
        let summary = packument
            .get("versions")
            .and_then(|v| v.get(&version))
            .cloned()
            .unwrap_or_else(|| json!({}));
        if let Some(dep) = summary.get("deprecated").and_then(Value::as_str) {
            if !dep.is_empty() {
                changes.push(format!(
                    "warning: @{}/{}@{version} is deprecated: {dep}",
                    parsed.scope, parsed.name
                ));
            }
        }
        let tgz = registry_get_bytes(
            &base,
            &format!(
                "/api/v1/skills/@{}/{}/versions/{}/tarball",
                parsed.scope, parsed.name, version
            ),
        )
        .await?;
        let dir = store_skill_version(
            &parsed.scope,
            &parsed.name,
            &version,
            summary.get("integrity").and_then(Value::as_str),
            &tgz,
        )?;
        let id = format!("@{}/{}", parsed.scope, parsed.name);
        manifest.insert(id.clone(), Value::String(request));
        lock.insert(
            id.clone(),
            json!({
                "version": version,
                "integrity": sri_sha512(&tgz),
                "resolved": format!(
                    "/api/v1/skills/@{}/{}/versions/{}/tarball",
                    parsed.scope, parsed.name, version
                ),
            }),
        );
        changes.push(format!("{id}@{version} → {}", dir.display()));
    }
    let mut manifest_doc = Map::new();
    manifest_doc.insert("skills".into(), Value::Object(manifest));
    write_json_sorted(&paths.manifest, manifest_doc)?;
    let mut lock_doc = Map::new();
    lock_doc.insert("version".into(), json!(SKILLS_LOCK_VERSION));
    lock_doc.insert("skills".into(), Value::Object(lock));
    write_json_sorted(&paths.lock, lock_doc)?;
    let mut text = String::new();
    if changes.is_empty() {
        text.push_str("Registry skills are already up to date.");
    } else {
        let _ = write!(text, "Installed:\n  {}", changes.join("\n  "));
        let _ = write!(text, "\nSaved {}", paths.manifest.display());
    }
    Ok(text)
}

fn remove(rt: &Runtime, names: &[String], global: bool) -> Result<String, String> {
    let paths = install_paths(&rt.cwd, global)?;
    let mut manifest = manifest_of(&paths.manifest)?;
    let mut lock = lock_of(&paths.lock)?;
    let mut removed = Vec::new();
    for raw in names {
        let (scope, name) = if raw.starts_with('@') {
            parse_skill_id(raw).ok_or_else(|| format!("Invalid skill id {raw:?}"))?
        } else {
            manifest
                .keys()
                .find_map(|k| parse_skill_id(k).filter(|(_, n)| n == raw))
                .ok_or_else(|| {
                    format!(
                        "Skill \"{raw}\" is not installed (looked in {})",
                        paths.dir.display()
                    )
                })?
        };
        let full = format!("@{scope}/{name}");
        let version = lock
            .get(&full)
            .and_then(|e| e.get("version"))
            .and_then(Value::as_str)
            .map(str::to_string);
        manifest.shift_remove(&full);
        lock.shift_remove(&full);
        if let Some(version) = version {
            let still_referenced = manifest.contains_key(&full)
                || lock_of(&global_install_paths().lock).is_ok_and(|l| l.contains_key(&full));
            if !still_referenced {
                let _ = std::fs::remove_dir_all(skill_store_path(&scope, &name, &version));
            }
        }
        removed.push(full);
    }
    let mut manifest_doc = Map::new();
    manifest_doc.insert("skills".into(), Value::Object(manifest));
    write_json_sorted(&paths.manifest, manifest_doc)?;
    let mut lock_doc = Map::new();
    lock_doc.insert("version".into(), json!(SKILLS_LOCK_VERSION));
    lock_doc.insert("skills".into(), Value::Object(lock));
    write_json_sorted(&paths.lock, lock_doc)?;
    Ok(format!(
        "Removed:\n  {}\nSaved {}",
        removed.join("\n  "),
        paths.manifest.display()
    ))
}

fn list_installed(rt: &Runtime) -> String {
    let mut lines = Vec::new();
    // Local skills visible to the prompt (TS refreshSkillState view).
    let skills = crate::sysprompt::load_skills(&agent_dir(), &rt.cwd, true);
    if !skills.is_empty() {
        lines.push("Local skills:".to_string());
        for s in &skills {
            lines.push(format!("  {}  {}", s.name, s.description));
        }
    }
    // Registry installs (project + user scopes, TS `listInstalledSkills`).
    let mut registry_lines = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut sources: Vec<(&str, InstallPaths)> = Vec::new();
    if let Some(p) = project_install_paths(&rt.cwd) {
        sources.push(("project", p));
    }
    sources.push(("user", global_install_paths()));
    for (scope, paths) in &sources {
        let Ok(manifest) = manifest_of(&paths.manifest) else {
            continue;
        };
        let Ok(lock) = lock_of(&paths.lock) else {
            continue;
        };
        let mut ids: Vec<&String> = manifest.keys().chain(lock.keys()).collect();
        ids.sort();
        ids.dedup();
        for id in ids {
            if !seen.insert((*id).clone()) {
                continue;
            }
            let entry = lock.get(id);
            let version = entry
                .and_then(|e| e.get("version"))
                .and_then(Value::as_str)
                .unwrap_or("-");
            let range = manifest.get(id).and_then(Value::as_str).unwrap_or("-");
            let stored = entry
                .and_then(|e| e.get("version"))
                .and_then(Value::as_str)
                .is_some_and(|v| {
                    let (scope_name, name) = id.split_once('/').unwrap_or((id, id));
                    skill_store_path(scope_name.trim_start_matches('@'), name, v).is_dir()
                });
            let marker = if stored { "" } else { " [not unpacked]" };
            registry_lines.push(format!(
                "  {id} [{scope}] {version} (range {range}){marker}"
            ));
        }
    }
    if !registry_lines.is_empty() {
        lines.push("Registry skills:".to_string());
        lines.append(&mut registry_lines);
    }
    if lines.is_empty() {
        return "No local skills found. Install some with:\n  /skills search <query>\n  /skills install @scope/name"
            .to_string();
    }
    lines.join("\n")
}

// ---------------------------------------------------------------------------
// Marketplace (TS `extensibility/plugins/marketplace/`)
// ---------------------------------------------------------------------------

fn marketplaces_registry_path() -> PathBuf {
    config_root().join(MARKETPLACES_REGISTRY)
}

fn user_installed_registry_path() -> PathBuf {
    config_root().join("plugins").join("installed_plugins.json")
}

fn project_installed_registry_path(cwd: &Path) -> PathBuf {
    cwd.join(".omp")
        .join("plugins")
        .join("installed_plugins.json")
}

fn marketplaces_cache_dir() -> PathBuf {
    config_root()
        .join("plugins")
        .join("cache")
        .join("marketplaces")
}

fn plugins_cache_dir() -> PathBuf {
    config_root().join("plugins").join("cache").join("plugins")
}

fn empty_marketplaces_registry() -> Map<String, Value> {
    let mut reg = Map::new();
    reg.insert("version".into(), json!(1));
    reg.insert("marketplaces".into(), json!([]));
    reg
}

fn read_marketplaces_registry() -> Result<Map<String, Value>, String> {
    match std::fs::read_to_string(marketplaces_registry_path()) {
        Ok(content) => {
            let data: Value = serde_json::from_str(&content).map_err(|e| {
                format!(
                    "{}: invalid JSON: {e}",
                    marketplaces_registry_path().display()
                )
            })?;
            let mut reg = empty_marketplaces_registry();
            if let Some(list) = data.get("marketplaces").and_then(Value::as_array) {
                reg.insert("marketplaces".into(), Value::Array(list.clone()));
            }
            Ok(reg)
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(empty_marketplaces_registry()),
        Err(e) => Err(format!(
            "read {}: {e}",
            marketplaces_registry_path().display()
        )),
    }
}

fn write_marketplaces_registry(reg: &Map<String, Value>) -> Result<(), String> {
    write_json_sorted(&marketplaces_registry_path(), reg.clone())
}

fn read_installed_registry(path: &Path) -> Result<Map<String, Value>, String> {
    let mut reg = Map::new();
    reg.insert("version".into(), json!(2));
    reg.insert("plugins".into(), json!({}));
    match std::fs::read_to_string(path) {
        Ok(content) => {
            let data: Value = serde_json::from_str(&content)
                .map_err(|e| format!("{}: invalid JSON: {e}", path.display()))?;
            if let Some(plugins) = data.get("plugins").and_then(Value::as_object) {
                reg.insert("plugins".into(), Value::Object(plugins.clone()));
            }
            Ok(reg)
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(reg),
        Err(e) => Err(format!("read {}: {e}", path.display())),
    }
}

fn write_installed_registry(path: &Path, reg: &Map<String, Value>) -> Result<(), String> {
    write_json_sorted(path, reg.clone())
}

/// Unix-seconds stamp (TS stores ISO 8601; the value is informational here).
fn now_iso() -> String {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
        .to_string()
}

/// `/marketplace` slash-command entry point.
pub async fn marketplace_command(rt: &Arc<Runtime>, args: &str) -> Outcome {
    let (verb, rest) = split_sub(args);
    let result: Result<String, String> = match verb.as_str() {
        "" | "list" => list_marketplaces(),
        "help" => Ok(MARKETPLACE_USAGE.to_string()),
        "add" if !rest.is_empty() => add_marketplace(rest.trim()).await,
        "add" => Err("Usage: /marketplace add <source>".to_string()),
        "remove" | "rm" if !rest.is_empty() => remove_marketplace(rest.trim()),
        "remove" | "rm" => Err("Usage: /marketplace remove <name>".to_string()),
        "update" => update_marketplaces(rest.trim()).await,
        "discover" => discover(rest.trim()).await,
        "install" if !rest.is_empty() => {
            parse_marketplace_install_args(&rest).and_then(|parsed| install_plugin(rt, &parsed))
        }
        "install" => Err(
            "Interactive plugin pickers are TUI-only. Pass an explicit name@marketplace argument.".to_string(),
        ),
        "uninstall" if !rest.is_empty() => parse_plugin_scope_args(
            &rest,
            "Usage: /marketplace uninstall [--scope user|project] <name@marketplace>",
        )
        .and_then(|(plugin_id, scope)| uninstall_plugin(rt, &plugin_id, scope)),
        "uninstall" => Err("Usage: /marketplace uninstall <name@marketplace>".to_string()),
        "installed" => list_installed_plugins(rt),
        "upgrade" => Err(
            "/marketplace upgrade is not ported yet; reinstall with /marketplace install to update a plugin."
                .to_string(),
        ),
        other => Err(format!(
            "Unknown /marketplace subcommand: {other}. Use /marketplace help for available commands."
        )),
    };
    match result {
        Ok(text) => output(rt, &text),
        Err(e) => output(rt, &format!("Marketplace error: {e}")),
    }
    handled()
}

fn list_marketplaces() -> Result<String, String> {
    let reg = read_marketplaces_registry()?;
    let list = reg
        .get("marketplaces")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if list.is_empty() {
        return Ok(
            "No marketplaces configured.\n\nGet started:\n  /marketplace add anthropics/claude-plugins-official\n\nThen browse with /marketplace discover"
                .to_string(),
        );
    }
    let lines: Vec<String> = list
        .iter()
        .map(|m| {
            format!(
                "  {}  {}",
                m.get("name").and_then(Value::as_str).unwrap_or("?"),
                m.get("sourceUri").and_then(Value::as_str).unwrap_or("?")
            )
        })
        .collect();
    Ok(format!(
        "Marketplaces:\n{}\n\nUse /marketplace discover to browse plugins, or /marketplace help for all commands",
        lines.join("\n")
    ))
}

/// TS `parseMarketplaceInstallArgs`: `[--force] [--scope user|project] <name@mkt>`.
pub fn parse_marketplace_install_args(rest: &str) -> Result<MarketplaceInstall, String> {
    const INSTALL_USAGE: &str =
        "Usage: /marketplace install [--force] [--scope user|project] <name@marketplace>";
    let mut force = false;
    let mut scope = "user".to_string();
    let mut spec: Option<String> = None;
    let mut tokens = rest.split_whitespace().peekable();
    while let Some(token) = tokens.next() {
        match token {
            "--force" => force = true,
            "--scope" => {
                let value = tokens
                    .next()
                    .ok_or_else(|| format!("--scope needs a value\n\n{INSTALL_USAGE}"))?;
                if value != "user" && value != "project" {
                    return Err(format!(
                        "--scope must be user or project, got {value:?}\n\n{INSTALL_USAGE}"
                    ));
                }
                scope = value.to_string();
            }
            t if t.starts_with('-') => {
                return Err(format!("Unknown option: {t}\n\n{INSTALL_USAGE}"))
            }
            t => {
                if spec.is_some() {
                    return Err(format!("Unexpected extra argument: {t}\n\n{INSTALL_USAGE}"));
                }
                spec = Some(t.to_string());
            }
        }
    }
    let spec = spec.ok_or_else(|| INSTALL_USAGE.to_string())?;
    let (name, marketplace) = spec
        .rsplit_once('@')
        .filter(|(n, _)| !n.is_empty())
        .ok_or_else(|| format!("Install specs must be <name@marketplace>, got {spec:?}"))?;
    Ok(MarketplaceInstall {
        name: name.to_string(),
        marketplace: marketplace.to_string(),
        force,
        scope,
    })
}

pub struct MarketplaceInstall {
    pub name: String,
    pub marketplace: String,
    pub force: bool,
    pub scope: String,
}

/// Parse `[--scope user|project] <name@marketplace>` (TS `parsePluginScopeArgs`).
pub fn parse_plugin_scope_args(rest: &str, usage: &str) -> Result<(String, &'static str), String> {
    let mut scope = "user";
    let mut spec: Option<String> = None;
    let mut tokens = rest.split_whitespace().peekable();
    while let Some(token) = tokens.next() {
        if token == "--scope" {
            let value = tokens
                .next()
                .ok_or_else(|| format!("--scope needs a value\n\n{usage}"))?;
            scope = match value {
                "user" => "user",
                "project" => "project",
                other => {
                    return Err(format!(
                        "--scope must be user or project, got {other:?}\n\n{usage}"
                    ))
                }
            };
        } else {
            spec = Some(token.to_string());
        }
    }
    let spec = spec.ok_or_else(|| usage.to_string())?;
    let (name, marketplace) = spec
        .rsplit_once('@')
        .filter(|(n, _)| !n.is_empty())
        .ok_or_else(|| format!("Plugin selectors must be <name@marketplace>, got {spec:?}"))?;
    Ok((format!("{name}@{marketplace}"), scope))
}

/// Resolve a marketplace source to `(dir, catalogRelPath, catalog)` holding the
/// catalog (TS `fetchMarketplace`): local dir, https JSON URL, or a git/GitHub
/// source cloned into the cache.
async fn fetch_marketplace_source(
    source: &str,
    cache_dir: &Path,
) -> Result<(PathBuf, String, Value), String> {
    std::fs::create_dir_all(cache_dir)
        .map_err(|e| format!("create {}: {e}", cache_dir.display()))?;
    let local = PathBuf::from(source);
    if local.is_dir() {
        return read_catalog_from(&local).map(|(rel, catalog)| (local, rel, catalog));
    }
    let (url, is_git, owner_repo) = if let Some(or) = parse_owner_repo(source) {
        (
            format!("https://github.com/{}/{}.git", or.0, or.1),
            true,
            Some(or),
        )
    } else if Path::new(source)
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("git"))
        || source.starts_with("git@")
    {
        (source.to_string(), true, None)
    } else if source.starts_with("http://") || source.starts_with("https://") {
        (source.to_string(), false, None)
    } else {
        return Err(format!(
            "Unknown marketplace source {source:?}; use owner/repo, a git URL, an https JSON URL, or a local directory."
        ));
    };
    if is_git {
        let tmp = cache_dir.join(format!(".tmp-clone-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let clone = Command::new("git")
            .args(["clone", "--depth", "1", &url])
            .arg(&tmp)
            .output()
            .map_err(|e| format!("run git: {e} (is git installed?)"))?;
        if clone.status.success() {
            let result = read_catalog_from(&tmp).map(|(rel, catalog)| (tmp.clone(), rel, catalog));
            if result.is_err() {
                let _ = std::fs::remove_dir_all(&tmp);
            }
            return result;
        }
        let _ = std::fs::remove_dir_all(&tmp);
        // GitHub fallback: the anonymous codeload tarball (covers missing git
        // credentials where TS would surface the clone error directly).
        if let Some((owner, repo)) = owner_repo {
            let tgz_url = format!("https://codeload.github.com/{owner}/{repo}/tar.gz/HEAD");
            return download_tarball_catalog(&tgz_url, cache_dir).await;
        }
        return Err(format!(
            "git clone {url} failed: {}",
            String::from_utf8_lossy(&clone.stderr).trim()
        ));
    }
    fetch_json_catalog(source).await
}

fn parse_owner_repo(source: &str) -> Option<(String, String)> {
    let mut parts = source.split('/');
    let owner = parts.next()?;
    let repo = parts.next()?;
    if parts.next().is_some() || owner.is_empty() || repo.is_empty() {
        return None;
    }
    Some((owner.to_string(), repo.to_string()))
}

/// Read `.omp-plugin/marketplace.json` / `.claude-plugin/marketplace.json`.
fn read_catalog_from(dir: &Path) -> Result<(String, Value), String> {
    for rel in CATALOG_RELPATHS {
        let path = dir.join(rel);
        if let Ok(content) = std::fs::read_to_string(&path) {
            let catalog: Value = serde_json::from_str(&content).map_err(|e| {
                format!(
                    "Failed to parse marketplace catalog at {}: {e}",
                    path.display()
                )
            })?;
            if catalog.get("name").and_then(Value::as_str).is_none() {
                return Err(format!(
                    "Marketplace catalog at {} is missing the required \"name\" field",
                    path.display()
                ));
            }
            return Ok(((*rel).to_string(), catalog));
        }
    }
    Err(format!(
        "No marketplace catalog found in {} (looked for {})",
        dir.display(),
        CATALOG_RELPATHS.join(", ")
    ))
}

async fn fetch_json_catalog(url: &str) -> Result<(PathBuf, String, Value), String> {
    let resp = reqwest::Client::new()
        .get(url)
        .timeout(std::time::Duration::from_secs(60))
        .send()
        .await
        .map_err(|e| format!("Failed to fetch marketplace catalog from {url}: {e}"))?;
    let status = resp.status();
    let body = resp.text().await.unwrap_or_default();
    if !status.is_success() {
        return Err(format!(
            "Failed to fetch marketplace catalog from {url}: HTTP {status}"
        ));
    }
    let catalog: Value = serde_json::from_str(&body)
        .map_err(|e| format!("Failed to parse marketplace catalog at {url}: {e}"))?;
    let name = catalog.get("name").and_then(Value::as_str).ok_or_else(|| {
        format!("Marketplace catalog at {url} is missing the required \"name\" field")
    })?;
    let dir = marketplaces_cache_dir().join(name);
    std::fs::create_dir_all(&dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
    let path = dir.join("marketplace.json");
    std::fs::write(&path, &body).map_err(|e| format!("write {}: {e}", path.display()))?;
    Ok((dir, "marketplace.json".to_string(), catalog))
}

async fn download_tarball_catalog(
    url: &str,
    cache_dir: &Path,
) -> Result<(PathBuf, String, Value), String> {
    let resp = reqwest::Client::new()
        .get(url)
        .timeout(std::time::Duration::from_secs(120))
        .send()
        .await
        .map_err(|e| format!("Failed to fetch {url}: {e}"))?;
    let status = resp.status();
    if !status.is_success() {
        return Err(format!("Failed to fetch {url}: HTTP {status}"));
    }
    let bytes = resp
        .bytes()
        .await
        .map(|b| b.to_vec())
        .map_err(|e| format!("download {url}: {e}"))?;
    let tmp = cache_dir.join(format!(".tmp-tar-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).map_err(|e| format!("create {}: {e}", tmp.display()))?;
    unpack_tgz(&bytes, &tmp)?;
    // GitHub tarballs wrap everything in `<repo>-<sha>/`.
    let inner = std::fs::read_dir(&tmp)
        .map_err(|e| format!("read {}: {e}", tmp.display()))?
        .flatten()
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .ok_or_else(|| format!("tarball from {url} is empty"))?;
    let result = read_catalog_from(&inner);
    if result.is_err() {
        let _ = std::fs::remove_dir_all(&tmp);
    }
    let (rel, catalog) = result?;
    Ok((inner, rel, catalog))
}

async fn add_marketplace(source: &str) -> Result<String, String> {
    let mut reg = read_marketplaces_registry()?;
    let list = reg
        .get("marketplaces")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let (_dir, _rel, catalog) = fetch_marketplace_source(source, &marketplaces_cache_dir()).await?;
    let name = catalog
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| "marketplace catalog has no name".to_string())?
        .to_string();
    if list
        .iter()
        .any(|m| m.get("name").and_then(Value::as_str) == Some(name.as_str()))
    {
        return Err(format!("Marketplace \"{name}\" already exists"));
    }
    let source_type = if parse_owner_repo(source).is_some() {
        "github"
    } else if source.starts_with("http") {
        "url"
    } else {
        "path"
    };
    let entry = json!({
        "name": name,
        "sourceType": source_type,
        "sourceUri": source,
        "catalogPath": marketplaces_cache_dir().join(&name).join("marketplace.json").display().to_string(),
        "addedAt": now_iso(),
        "updatedAt": now_iso(),
    });
    let mut new_list = list;
    new_list.push(entry);
    reg.insert("marketplaces".into(), Value::Array(new_list));
    write_marketplaces_registry(&reg)?;
    Ok(format!("Added marketplace: {name}"))
}

fn remove_marketplace(name: &str) -> Result<String, String> {
    let mut reg = read_marketplaces_registry()?;
    let list = reg
        .get("marketplaces")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let filtered: Vec<Value> = list
        .iter()
        .filter(|m| m.get("name").and_then(Value::as_str) != Some(name))
        .cloned()
        .collect();
    if filtered.len() == list.len() {
        return Err(format!("Marketplace \"{name}\" not found"));
    }
    reg.insert("marketplaces".into(), Value::Array(filtered));
    write_marketplaces_registry(&reg)?;
    let _ = std::fs::remove_dir_all(marketplaces_cache_dir().join(name));
    Ok(format!("Removed marketplace: {name}"))
}

async fn update_marketplaces(name: &str) -> Result<String, String> {
    let reg = read_marketplaces_registry()?;
    let list = reg
        .get("marketplaces")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let targets: Vec<Value> = if name.is_empty() {
        list
    } else {
        list.into_iter()
            .filter(|m| m.get("name").and_then(Value::as_str) == Some(name))
            .collect()
    };
    if targets.is_empty() {
        return Err(format!("Marketplace \"{name}\" not found"));
    }
    for m in &targets {
        let source = m
            .get("sourceUri")
            .and_then(Value::as_str)
            .ok_or_else(|| "marketplace entry has no sourceUri".to_string())?;
        fetch_marketplace_source(source, &marketplaces_cache_dir()).await?;
    }
    if name.is_empty() {
        Ok(format!("Updated {} marketplace(s)", targets.len()))
    } else {
        Ok(format!("Updated marketplace: {name}"))
    }
}

async fn discover(marketplace: &str) -> Result<String, String> {
    let catalogs = load_catalogs(marketplace)?;
    if catalogs.is_empty() {
        let reg = read_marketplaces_registry()?;
        let count = reg
            .get("marketplaces")
            .and_then(Value::as_array)
            .map_or(0, |v: &Vec<Value>| v.len());
        return Ok(if count == 0 {
            "No marketplaces configured. Try:\n  /marketplace add anthropics/claude-plugins-official".to_string()
        } else {
            "No plugins available in configured marketplaces".to_string()
        });
    }
    let mut lines = vec!["Available plugins:".to_string()];
    for catalog in &catalogs {
        for plugin in catalog
            .get("plugins")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default()
        {
            let name = plugin.get("name").and_then(Value::as_str).unwrap_or("?");
            match plugin.get("version").and_then(Value::as_str) {
                Some(v) => lines.push(format!("  - {name}@{v}")),
                None => lines.push(format!("  - {name}")),
            }
            if let Some(desc) = plugin.get("description").and_then(Value::as_str) {
                lines.push(format!("      {desc}"));
            }
        }
    }
    Ok(lines.join("\n"))
}

/// Catalog JSON values for one (or all) configured marketplace(s).
fn load_catalogs(marketplace: &str) -> Result<Vec<Value>, String> {
    let reg = read_marketplaces_registry()?;
    let list = reg
        .get("marketplaces")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut out = Vec::new();
    for m in &list {
        let name = m.get("name").and_then(Value::as_str).unwrap_or("");
        if !marketplace.is_empty() && name != marketplace {
            continue;
        }
        // Prefer the cached clone's canonical file, else the recorded path.
        let cache = marketplaces_cache_dir().join(name);
        let path = if cache.join("marketplace.json").is_file() {
            cache.join("marketplace.json")
        } else {
            let rel = m.get("catalogPath").and_then(Value::as_str).unwrap_or("");
            if rel.is_empty() {
                continue;
            }
            PathBuf::from(rel)
        };
        let content = std::fs::read_to_string(&path).map_err(|_| {
            format!("Marketplace \"{name}\" has no cached catalog; run /marketplace update {name}")
        })?;
        let catalog: Value = serde_json::from_str(&content)
            .map_err(|e| format!("{}: invalid JSON: {e}", path.display()))?;
        out.push(catalog);
    }
    if !marketplace.is_empty() && out.is_empty() {
        return Err(format!("Marketplace \"{marketplace}\" not found"));
    }
    Ok(out)
}

fn install_plugin(rt: &Arc<Runtime>, parsed: &MarketplaceInstall) -> Result<String, String> {
    let catalogs = load_catalogs(&parsed.marketplace)?;
    let entry = catalogs
        .iter()
        .flat_map(|c| {
            c.get("plugins")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default()
        })
        .find(|p| p.get("name").and_then(Value::as_str) == Some(parsed.name.as_str()))
        .ok_or_else(|| {
            format!(
                "Plugin {:?} not found in marketplace {:?} (browse with /marketplace discover {})",
                parsed.name, parsed.marketplace, parsed.marketplace
            )
        })?;
    let version = entry
        .get("version")
        .and_then(Value::as_str)
        .unwrap_or("0.0.0");
    let source = entry.get("source").cloned().unwrap_or_else(|| json!("./"));
    let source_dir = resolve_plugin_source_dir(&parsed.marketplace, &parsed.name, &source)?;
    let install_path = plugins_cache_dir().join(&parsed.name);
    if install_path.exists() && !parsed.force {
        return Err(format!(
            "Plugin {:?} is already installed at {}; pass --force to overwrite",
            parsed.name,
            install_path.display()
        ));
    }
    let _ = std::fs::remove_dir_all(&install_path);
    copy_dir_recursive(&source_dir, &install_path)?;
    let reg_path = if parsed.scope == "project" {
        project_installed_registry_path(&rt.cwd)
    } else {
        user_installed_registry_path()
    };
    let mut reg = read_installed_registry(&reg_path)?;
    let plugins = reg
        .get_mut("plugins")
        .and_then(Value::as_object_mut)
        .ok_or_else(|| "installed registry: no plugins object".to_string())?;
    let id = format!("{}/{}", parsed.marketplace, parsed.name);
    let entry_json = json!({
        "scope": parsed.scope,
        "installPath": install_path.display().to_string(),
        "version": version,
        "installedAt": now_iso(),
        "lastUpdated": now_iso(),
        "enabled": true,
    });
    plugins
        .entry(id.clone())
        .and_modify(|v| {
            if let Some(arr) = v.as_array_mut() {
                arr.push(entry_json.clone());
            }
        })
        .or_insert_with(|| json!([entry_json]));
    write_installed_registry(&reg_path, &reg)?;
    Ok(format!(
        "Installed {} from {}",
        parsed.name, parsed.marketplace
    ))
}

fn copy_dir_recursive(src: &Path, dest: &Path) -> Result<(), String> {
    std::fs::create_dir_all(dest).map_err(|e| format!("create {}: {e}", dest.display()))?;
    for entry in std::fs::read_dir(src).map_err(|e| format!("read {}: {e}", src.display()))? {
        let entry = entry.map_err(|e| format!("read {}: {e}", src.display()))?;
        let from = entry.path();
        let to = dest.join(entry.file_name());
        if from.is_dir() {
            copy_dir_recursive(&from, &to)?;
        } else {
            std::fs::copy(&from, &to).map_err(|e| format!("copy {}: {e}", from.display()))?;
        }
    }
    Ok(())
}

/// Resolve a plugin `source` to an existing local directory.
fn resolve_plugin_source_dir(
    marketplace: &str,
    plugin: &str,
    source: &Value,
) -> Result<PathBuf, String> {
    if let Some(rel) = source.as_str() {
        if !rel.starts_with("./") {
            return Err(format!(
                "Relative plugin source paths must start with \"./\" — got: {rel:?}"
            ));
        }
        let resolved = marketplaces_cache_dir()
            .join(marketplace)
            .join(rel.trim_start_matches("./"));
        if !resolved.is_dir() {
            return Err(format!(
                "Plugin source directory does not exist: {} (re-run /marketplace update {marketplace})",
                resolved.display()
            ));
        }
        return Ok(resolved);
    }
    let kind = source.get("source").and_then(Value::as_str).unwrap_or("");
    match kind {
        "github" => {
            let repo = source
                .get("repo")
                .and_then(Value::as_str)
                .ok_or_else(|| "github plugin source needs a \"repo\"".to_string())?;
            clone_plugin_repo(&format!("https://github.com/{repo}.git"), plugin)
        }
        "url" => {
            let url = source
                .get("url")
                .and_then(Value::as_str)
                .ok_or_else(|| "url plugin source needs a \"url\"".to_string())?;
            clone_plugin_repo(url, plugin)
        }
        "git-subdir" => {
            let url = source
                .get("url")
                .and_then(Value::as_str)
                .ok_or_else(|| "git-subdir plugin source needs a \"url\"".to_string())?;
            let sub = source.get("path").and_then(Value::as_str).unwrap_or("");
            let root = clone_plugin_repo(url, plugin)?;
            let resolved = root.join(sub);
            if !resolved.is_dir() {
                return Err(format!(
                    "Plugin source subdirectory does not exist: {}",
                    resolved.display()
                ));
            }
            Ok(resolved)
        }
        "npm" => Err(
            "npm plugin sources are not yet supported. Use git-based sources instead.".to_string(),
        ),
        other => Err(format!("Unknown plugin source type: {other:?}")),
    }
}

fn clone_plugin_repo(url: &str, plugin: &str) -> Result<PathBuf, String> {
    let dest = plugins_cache_dir().join(format!(".src-{plugin}"));
    let _ = std::fs::remove_dir_all(&dest);
    std::fs::create_dir_all(&dest).map_err(|e| format!("create {}: {e}", dest.display()))?;
    let out = Command::new("git")
        .args(["clone", "--depth", "1", url])
        .arg(&dest)
        .output()
        .map_err(|e| format!("run git: {e} (is git installed?)"))?;
    if !out.status.success() {
        let _ = std::fs::remove_dir_all(&dest);
        return Err(format!(
            "git clone {url} failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    Ok(dest)
}

fn uninstall_plugin(rt: &Arc<Runtime>, plugin_id: &str, scope: &str) -> Result<String, String> {
    let reg_path = if scope == "project" {
        project_installed_registry_path(&rt.cwd)
    } else {
        user_installed_registry_path()
    };
    let mut reg = read_installed_registry(&reg_path)?;
    let plugins = reg
        .get_mut("plugins")
        .and_then(Value::as_object_mut)
        .ok_or_else(|| "installed registry: no plugins object".to_string())?;
    let entries = plugins
        .remove(plugin_id)
        .ok_or_else(|| format!("Plugin \"{plugin_id}\" not found in {}", reg_path.display()))?;
    write_installed_registry(&reg_path, &reg)?;
    // Drop the cached copy only when no other scope's registry still references it.
    for path in [
        user_installed_registry_path(),
        project_installed_registry_path(&rt.cwd),
    ] {
        if let Ok(other) = read_installed_registry(&path) {
            if other
                .get("plugins")
                .and_then(Value::as_object)
                .is_some_and(|p| p.contains_key(plugin_id))
            {
                return Ok(format!("Uninstalled {plugin_id}"));
            }
        }
    }
    if let Some(path) = entries
        .get(0)
        .and_then(|e| e.get("installPath"))
        .and_then(Value::as_str)
    {
        let _ = std::fs::remove_dir_all(path);
    }
    Ok(format!("Uninstalled {plugin_id}"))
}

fn list_installed_plugins(rt: &Arc<Runtime>) -> Result<String, String> {
    let mut lines = Vec::new();
    for (scope, path) in [
        ("user", user_installed_registry_path()),
        ("project", project_installed_registry_path(&rt.cwd)),
    ] {
        let reg = read_installed_registry(&path)?;
        let Some(plugins) = reg.get("plugins").and_then(Value::as_object) else {
            continue;
        };
        for (id, entries) in plugins {
            let entries = entries.as_array().cloned().unwrap_or_default();
            let entry = entries.first().cloned().unwrap_or_else(|| json!({}));
            let version = entry.get("version").and_then(Value::as_str).unwrap_or("-");
            let enabled = entry
                .get("enabled")
                .and_then(Value::as_bool)
                .unwrap_or(true);
            let state = if enabled { "" } else { " [disabled]" };
            lines.push(format!(
                "  {id} [{scope}] v{version}{state} ({} entries)",
                entries.len()
            ));
        }
    }
    if lines.is_empty() {
        return Ok("No marketplace plugins installed".to_string());
    }
    Ok(format!("Installed plugins:\n{}", lines.join("\n")))
}

// ---------------------------------------------------------------------------
// `/plugins` helpers (called from `plugins_command` in slash.rs)
// ---------------------------------------------------------------------------

/// Text for `/plugins list`: installed marketplace plugins (TS lists npm +
/// marketplace plugins; omp-rs has no npm plugin host).
pub fn plugins_list_text(rt: &Runtime) -> String {
    let mut lines = Vec::new();
    for (scope, path) in [
        ("user", user_installed_registry_path()),
        ("project", project_installed_registry_path(&rt.cwd)),
    ] {
        let Ok(reg) = read_installed_registry(&path) else {
            continue;
        };
        let Some(plugins) = reg.get("plugins").and_then(Value::as_object) else {
            continue;
        };
        for (id, entries) in plugins {
            let entries = entries.as_array().cloned().unwrap_or_default();
            let entry = entries.first().cloned().unwrap_or_else(|| json!({}));
            let version = entry.get("version").and_then(Value::as_str).unwrap_or("-");
            let enabled = entry
                .get("enabled")
                .and_then(Value::as_bool)
                .unwrap_or(true);
            let state = if enabled { "" } else { " [disabled]" };
            lines.push(format!("  {id} [{scope}] v{version}{state}"));
        }
    }
    if lines.is_empty() {
        "No marketplace plugins installed. Get started:\n  /marketplace add anthropics/claude-plugins-official"
            .to_string()
    } else {
        format!("Installed plugins:\n{}", lines.join("\n"))
    }
}

/// Set `enabled` on every entry of `plugin_id` across both registries
/// (TS `/plugins enable|disable`).
pub fn plugins_set_enabled(rt: &Runtime, plugin_id: &str, enabled: bool) -> Result<(), String> {
    let mut touched = false;
    for path in [
        user_installed_registry_path(),
        project_installed_registry_path(&rt.cwd),
    ] {
        let mut reg = read_installed_registry(&path)?;
        let Some(plugins) = reg.get_mut("plugins").and_then(Value::as_object_mut) else {
            continue;
        };
        let Some(entries) = plugins.get_mut(plugin_id).and_then(Value::as_array_mut) else {
            continue;
        };
        for entry in entries.iter_mut() {
            if let Some(obj) = entry.as_object_mut() {
                obj.insert("enabled".to_string(), json!(enabled));
                touched = true;
            }
        }
        if touched {
            write_installed_registry(&path, &reg)?;
        }
    }
    if !touched {
        return Err(format!(
            "Plugin \"{plugin_id}\" not found in any installed-plugins registry (list with /plugins list)"
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_valid_specs() {
        assert_eq!(
            parse_skill_spec("@foo/bar"),
            Some(SkillSpec {
                scope: "foo".into(),
                name: "bar".into(),
                range: None
            })
        );
        assert_eq!(
            parse_skill_spec("@foo_bar/baz@1.2.3"),
            Some(SkillSpec {
                scope: "foo_bar".into(),
                name: "baz".into(),
                range: Some("1.2.3".into())
            })
        );
        assert_eq!(
            parse_skill_spec("@scope/name@^1.0.0"),
            Some(SkillSpec {
                scope: "scope".into(),
                name: "name".into(),
                range: Some("^1.0.0".into())
            })
        );
        assert_eq!(
            parse_skill_spec("@scope/name@next"),
            Some(SkillSpec {
                scope: "scope".into(),
                name: "name".into(),
                range: Some("next".into())
            })
        );
    }

    #[test]
    fn rejects_invalid_specs() {
        assert_eq!(parse_skill_spec("foo/bar"), None);
        assert_eq!(parse_skill_spec("@a/b"), None); // scope too short
        assert_eq!(parse_skill_spec("@scope/Name"), None); // uppercase
        assert_eq!(parse_skill_spec("@scope/name@"), None); // empty range
        assert_eq!(parse_skill_spec("@scope/-name"), None); // leading dash
        assert_eq!(parse_skill_spec("@scope/na_me"), None); // underscore in name
        assert_eq!(parse_skill_id("@scope/name@1.0.0"), None); // id has no range
        assert!(parse_skill_id("@scope/name").is_some());
        assert!(!is_valid_name(&"a".repeat(65)));
        assert!(is_valid_name("a1-b2"));
        assert!(is_valid_scope("abc"));
        assert!(is_valid_scope("a1_"));
        assert!(!is_valid_scope("ab"));
        assert!(!is_valid_scope("_abc"));
        assert!(!is_valid_scope("Abc"));
    }

    #[test]
    fn semver_checks() {
        assert!(is_exact_semver("1.2.3"));
        assert!(is_exact_semver("1.2.3-beta.1"));
        assert!(is_exact_semver("1.2.3+build"));
        assert!(!is_exact_semver("1.2"));
        assert!(!is_exact_semver("latest"));
        assert!(semver_satisfies("1.4.2", "^1.0.0"));
        assert!(!semver_satisfies("2.0.0", "^1.0.0"));
        assert!(semver_satisfies("1.4.2", "~1.4.0"));
        assert!(!semver_satisfies("1.5.0", "~1.4.0"));
        assert!(semver_satisfies("2.1.0", ">=1.0.0 <3.0.0"));
        assert!(semver_satisfies("0.9.0", "*"));
        assert!(semver_satisfies("1.2.3", "1.2.3"));
        assert!(semver_satisfies("1.9.0", "^1 || ^2"));
        assert!(!semver_satisfies("3.0.0", "^1 || ^2"));
        assert!(semver_satisfies("1.2.3", "1"));
        assert!(semver_satisfies("1.2.9", "1.2"));
        assert!(!semver_satisfies("1.9.0", "1.2"));
    }

    #[test]
    fn resolves_versions_like_ts() {
        let packument = json!({
            "scope": "scope",
            "name": "name",
            "distTags": {"latest": "1.2.0", "next": "2.0.0-beta.1"},
            "versions": {
                "1.0.0": {},
                "1.2.0": {},
                "1.5.0": {"yanked": true},
                "2.0.0-beta.1": {},
            },
        });
        assert_eq!(
            resolve_packument_version(&packument, "1.2.0").unwrap(),
            "1.2.0"
        );
        assert_eq!(
            resolve_packument_version(&packument, "latest").unwrap(),
            "1.2.0"
        );
        assert_eq!(
            resolve_packument_version(&packument, "next").unwrap(),
            "2.0.0-beta.1"
        );
        // 1.5.0 is yanked, so ^1.0.0 tops out at 1.2.0.
        assert_eq!(
            resolve_packument_version(&packument, "^1.0.0").unwrap(),
            "1.2.0"
        );
        assert!(resolve_packument_version(&packument, "9.9.9").is_err());
        assert!(resolve_packument_version(&packument, "^9").is_err());
    }

    #[test]
    fn parses_marketplace_install_args() {
        let parsed = parse_marketplace_install_args("--force foo@bar").unwrap();
        assert_eq!(parsed.name, "foo");
        assert_eq!(parsed.marketplace, "bar");
        assert!(parsed.force);
        assert_eq!(parsed.scope, "user");
        let scoped = parse_marketplace_install_args("--scope project --force foo@bar").unwrap();
        assert_eq!(scoped.scope, "project");
        assert!(parse_marketplace_install_args("nonsense").is_err());
        assert!(parse_marketplace_install_args("").is_err());
        assert!(parse_marketplace_install_args("--bogus foo@bar").is_err());
    }

    #[test]
    fn parses_plugin_scope_args() {
        let (id, scope) = parse_plugin_scope_args("--scope project foo@bar", "usage").unwrap();
        assert_eq!(id, "foo@bar");
        assert_eq!(scope, "project");
        let (id, scope) = parse_plugin_scope_args("foo@bar", "usage").unwrap();
        assert_eq!(id, "foo@bar");
        assert_eq!(scope, "user");
        assert!(parse_plugin_scope_args("--scope", "usage").is_err());
        assert!(parse_plugin_scope_args("--scope nope foo@bar", "usage").is_err());
        assert!(parse_plugin_scope_args("", "usage").is_err());
    }

    #[test]
    fn untars_simple_archive() {
        let content = b"hi\n";
        let archive = ustar_archive("hello.txt", content);
        let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        std::io::Write::write_all(&mut enc, &archive).unwrap();
        let tgz = enc.finish().unwrap();

        let dir = std::env::temp_dir().join(format!("omp-skills-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        assert_eq!(unpack_tgz(&tgz, &dir).unwrap(), 1);
        assert_eq!(
            std::fs::read_to_string(dir.join("hello.txt")).unwrap(),
            "hi\n"
        );

        // Traversal rejected.
        let mut evil = archive.clone();
        evil[..9].copy_from_slice(b"../pwn\0\0\0");
        let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        std::io::Write::write_all(&mut enc, &evil).unwrap();
        assert!(unpack_tgz(&enc.finish().unwrap(), &dir).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Minimal valid ustar archive with one regular file.
    fn ustar_archive(name: &str, content: &[u8]) -> Vec<u8> {
        let mut header = [0u8; 512];
        header[..name.len()].copy_from_slice(name.as_bytes());
        header[100..107].copy_from_slice(b"0000644");
        header[108..116].copy_from_slice(b"0000000\0");
        header[116..124].copy_from_slice(b"0000000\0");
        header[124..136].copy_from_slice(format!("{:011o}\0", content.len()).as_bytes());
        header[156] = b'0';
        header[257..263].copy_from_slice(b"ustar\0");
        for b in header[148..156].iter_mut() {
            *b = b' ';
        }
        let sum: u64 = header.iter().map(|&b| u64::from(b)).sum();
        header[148..156].copy_from_slice(format!("{:06o}\0 ", sum).as_bytes());
        let mut archive = header.to_vec();
        archive.extend_from_slice(content);
        let pad = 512 - content.len() % 512;
        archive.extend(std::iter::repeat_n(0u8, pad));
        archive.extend_from_slice(&[0u8; 1024]);
        archive
    }

    #[test]
    fn project_paths_walk_up_and_stop_at_home() {
        let home = home_dir();
        assert!(project_install_paths(&home).is_none());
        assert!(project_install_paths(&config_root()).is_none());
        let base = std::env::temp_dir().join(format!("omp-skillpaths-{}", std::process::id()));
        let omp = base.join(".omp");
        let deep = omp.join("sub").join("deeper");
        std::fs::create_dir_all(&deep).unwrap();
        let found = project_install_paths(&deep).unwrap();
        assert_eq!(found.dir, omp);
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn search_formatting_matches_ts_shape() {
        let response = json!({
            "total": 3, "page": 1, "perPage": 2,
            "hits": [
                {"scope": "s", "name": "a", "version": "1.0.0", "description": "First",
                 "weeklyDownloads": 12, "publisher": {"username": "u"}, "keywords": ["x"]},
                {"scope": "s", "name": "b", "version": "0.2.0", "description": "",
                 "weeklyDownloads": 1, "publisher": {"username": "v"}, "keywords": [],
                 "deprecated": "use a"},
            ],
        });
        let text = format_skill_search(&response);
        assert!(text.contains("@s/a 1.0.0"));
        assert!(text.contains("12 weekly downloads"));
        assert!(text.contains("deprecated: use a"));
        assert!(text.contains("… 1 more (page 1)"));
        assert_eq!(
            format_skill_search(&json!({"hits": []})),
            "No skills found."
        );
        assert_eq!(format_skill_search(&json!({})), "No skills found.");
    }

    #[test]
    fn targets_parse_global_flag() {
        let (names, global) = parse_targets("@a/b@c -g").unwrap();
        assert_eq!(names, vec!["@a/b@c".to_string()]);
        assert!(global);
        assert!(parse_targets("--what").is_err());
        let (names, global) = parse_targets("x y").unwrap();
        assert_eq!(names.len(), 2);
        assert!(!global);
    }

    #[test]
    fn sri_is_sha512_base64() {
        assert_eq!(
            sri_sha512(b""),
            "sha512-z4PhNX7vuL3xVChQ1m2AB9Yg5AULVxXcg/SpIdNs6c5H0NE8XYXysP+DGNKHfuwvY7kxvUdBeoGlODJ6+SfaPg=="
        );
    }
}
