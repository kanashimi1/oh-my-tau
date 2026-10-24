//! Server discovery and config loading (omp `lsp/config.ts`).
//!
//! Sources, lowest priority applied first (later overrides win):
//! 1. Built-in `defaults.json`
//! 2. User home root (`~/lsp.json`, `~/.lsp.json`, + yaml variants)
//! 3. User config dirs (`~/.omp`, `~/.pi`, `~/.claude` — visible and hidden files)
//! 4. Project config dirs (`<cwd>/.omp`, `<cwd>/.pi`, `<cwd>/.claude`)
//! 5. Project root files (highest priority)
//!
//! Servers are then filtered to those whose root markers exist in the cwd and
//! whose binary resolves (project-local bin dirs first, then `$PATH`).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use serde_json::Value;
use tracing::warn;

use crate::types::{LspConfig, ServerConfig};

/// omp `lsp/defaults.json`, byte-identical.
const DEFAULTS_JSON: &str = include_str!("defaults.json");

const CONFIG_FILENAMES: [&str; 6] = [
    "lsp.json",
    ".lsp.json",
    "lsp.yaml",
    ".lsp.yaml",
    "lsp.yml",
    ".lsp.yml",
];

// =============================================================================
// Config File Parsing / Normalization
// =============================================================================

/// Parse a config file as YAML (`.yaml`/`.yml`) or JSON.
///
/// # Errors
///
/// Returns `Err` on unparsable content.
pub fn parse_config_content(content: &str, file_path: &Path) -> anyhow::Result<Value> {
    match file_path
        .extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .as_deref()
    {
        Some("yaml" | "yml") => Ok(serde_yaml::from_str(content)?),
        _ => Ok(serde_json::from_str(content)?),
    }
}

fn normalize_string_array(value: &Value) -> Option<Vec<String>> {
    let arr = value.as_array()?;
    let items: Vec<String> = arr
        .iter()
        .filter_map(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(ToString::to_string)
        .collect();
    if items.is_empty() {
        None
    } else {
        Some(items)
    }
}

fn normalize_extension_map(value: &Value) -> Option<Vec<String>> {
    let map = value.as_object()?;
    let exts: Vec<String> = map.keys().filter(|k| !k.is_empty()).cloned().collect();
    if exts.is_empty() {
        None
    } else {
        Some(exts)
    }
}

/// Normalize one raw server entry; `None` when required fields are missing.
fn normalize_server_config(_name: &str, config: &Value) -> Option<ServerConfig> {
    let command = config
        .get("command")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())?
        .to_string();
    let file_types = normalize_string_array(
        &config.get("fileTypes").cloned().unwrap_or(Value::Null),
    )
    .or_else(|| {
        normalize_extension_map(
            &config
                .get("extensionToLanguage")
                .cloned()
                .unwrap_or(Value::Null),
        )
    })?;
    let root_markers =
        normalize_string_array(&config.get("rootMarkers").cloned().unwrap_or(Value::Null))
            .or_else(|| {
                if config.get("extensionToLanguage").is_some() {
                    Some(vec![".".to_string()])
                } else {
                    None
                }
            })?;
    let language_id = config
        .get("languageId")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(ToString::to_string);
    let args = config.get("args").and_then(|v| {
        let arr = v.as_array()?;
        let items: Vec<String> = arr
            .iter()
            .filter_map(|x| x.as_str())
            .map(ToString::to_string)
            .collect();
        Some(items)
    });
    let init_options = config
        .get("initOptions")
        .or_else(|| config.get("initializationOptions"))
        .filter(|v| v.is_object())
        .cloned();

    let parsed = ServerConfig {
        command,
        args,
        file_types,
        language_id,
        root_markers,
        init_options,
        settings: config.get("settings").filter(|v| v.is_object()).cloned(),
        disabled: config.get("disabled").and_then(Value::as_bool),
        warmup_timeout_ms: config.get("warmupTimeoutMs").and_then(Value::as_u64),
        workspace_ready_timings: config
            .get("workspaceReadyTimings")
            .and_then(|v| serde_json::from_value(v.clone()).ok()),
        is_linter: config.get("isLinter").and_then(Value::as_bool),
        capabilities: config
            .get("capabilities")
            .and_then(|v| serde_json::from_value(v.clone()).ok()),
        resolved_command: None,
    };
    Some(parsed)
}

fn read_config_file(file_path: &Path) -> Option<(HashMap<String, Value>, Option<u64>)> {
    let content = std::fs::read_to_string(file_path).ok()?;
    let parsed = parse_config_content(&content, file_path).ok()?;
    let idle_timeout_ms = parsed.get("idleTimeoutMs").and_then(Value::as_u64);
    let servers = parsed.get("servers").and_then(Value::as_object);
    let map = match servers {
        Some(obj) => obj
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect::<HashMap<_, _>>(),
        None => parsed
            .as_object()?
            .iter()
            .filter(|(k, _)| *k != "idleTimeoutMs")
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect::<HashMap<_, _>>(),
    };
    Some((map, idle_timeout_ms))
}

fn merge_servers(
    base: &HashMap<String, ServerConfig>,
    overrides: &HashMap<String, Value>,
) -> HashMap<String, ServerConfig> {
    let mut merged = base.clone();
    for (name, raw) in overrides {
        let candidate = match merged.get(name) {
            Some(prev) => {
                // Shallow-merge the previous config with the raw overrides (TS
                // `{...prev, ...config}`), as loose JSON values.
                let mut v = serde_json::to_value(prev).unwrap_or(Value::Null);
                if let (Some(a), Some(b)) = (v.as_object_mut(), raw.as_object()) {
                    for (k, val) in b {
                        a.insert(k.clone(), val.clone());
                    }
                }
                v
            }
            None => raw.clone(),
        };
        if let Some(cfg) = normalize_server_config(name, &candidate) {
            merged.insert(name.clone(), cfg);
        } else {
            warn!("Ignoring invalid LSP overrides (keeping previous config): {name}")
        }
    }
    merged
}

// =============================================================================
// Root Markers / Binary Resolution
// =============================================================================

/// Check if any root marker exists in `cwd` (glob markers match one readdir level).
#[must_use]
pub fn has_root_markers(cwd: &Path, markers: &[String]) -> bool {
    let mut entries: Option<Vec<String>> = None;
    for marker in markers {
        if marker.contains('*') {
            if entries.is_none() {
                entries = Some(
                    std::fs::read_dir(cwd)
                        .map(|rd| {
                            rd.filter_map(std::result::Result::ok)
                                .filter_map(|e| e.file_name().into_string().ok())
                                .collect()
                        })
                        .unwrap_or_default(),
                );
            }
            if let Some(list) = &entries {
                if list.iter().any(|e| glob_match(marker, e)) {
                    return true;
                }
            }
            continue;
        }
        if cwd.join(marker).exists() {
            return true;
        }
    }
    false
}

/// Minimal glob: `*` matches any run of non-separator characters, `?` one char.
fn glob_match(pattern: &str, name: &str) -> bool {
    fn inner(p: &[u8], n: &[u8]) -> bool {
        if p.is_empty() {
            return n.is_empty();
        }
        match p[0] {
            b'*' => (0..=n.len()).any(|i| inner(&p[1..], &n[i..])),
            b'?' => !n.is_empty() && inner(&p[1..], &n[1..]),
            c => !n.is_empty() && n[0] == c && inner(&p[1..], &n[1..]),
        }
    }
    inner(pattern.as_bytes(), name.as_bytes())
}

/// Whether any ancestor directory of `file_path` is an LSP project root.
#[must_use]
pub fn has_root_marker_ancestor(file_path: &Path, markers: &[String]) -> bool {
    if markers.is_empty() {
        return false;
    }
    let mut dir = file_path.parent().map(crate::utils::normalize_path);
    while let Some(d) = dir {
        if has_root_markers(&d, markers) {
            return true;
        }
        let parent = d.parent().map(std::path::Path::to_path_buf);
        match parent {
            Some(p) if p != d => dir = Some(p),
            _ => return false,
        }
    }
    false
}

const PYTHON_ROOT_MARKERS: [&str; 9] = [
    "pyproject.toml",
    "ty.toml",
    "requirements.txt",
    "setup.py",
    "setup.cfg",
    "Pipfile",
    "pyrightconfig.json",
    "ruff.toml",
    ".ruff.toml",
];

const LOCAL_BIN_PATHS: [(&[&str], &str); 8] = [
    (
        &[
            "package.json",
            "package-lock.json",
            "yarn.lock",
            "pnpm-lock.yaml",
        ],
        "node_modules/.bin",
    ),
    (&PYTHON_ROOT_MARKERS, ".venv/bin"),
    (&PYTHON_ROOT_MARKERS, ".venv/Scripts"),
    (&PYTHON_ROOT_MARKERS, "venv/bin"),
    (&PYTHON_ROOT_MARKERS, "venv/Scripts"),
    (&PYTHON_ROOT_MARKERS, ".env/bin"),
    (&PYTHON_ROOT_MARKERS, ".env/Scripts"),
    (&["Gemfile", "Gemfile.lock"], "vendor/bundle/bin"),
];

fn resolve_command_from_local_root(command: &str, cwd: &Path) -> Option<PathBuf> {
    for (markers, bin_dir) in LOCAL_BIN_PATHS {
        if !has_root_markers(
            cwd,
            &markers.iter().map(ToString::to_string).collect::<Vec<_>>(),
        ) {
            continue;
        }
        let candidate = cwd.join(bin_dir).join(command);
        if candidate.exists() {
            return Some(candidate);
        }
        #[cfg(windows)]
        for ext in [".exe", ".cmd", ".bat"] {
            let candidate = cwd.join(bin_dir).join(format!("{command}{ext}"));
            if candidate.exists() {
                return Some(candidate);
            }
        }
    }
    // Go-style project-local bin (markers go.mod/go.sum/go.work) shares `bin/`.
    if has_root_markers(cwd, &["go.mod".into(), "go.sum".into(), "go.work".into()]) {
        let candidate = cwd.join("bin").join(command);
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

/// Resolve a command to an executable path: project-local bin dirs, then `$PATH`.
#[must_use]
pub fn resolve_command(command: &str, cwd: &Path) -> Option<String> {
    if let Some(local) = resolve_command_from_local_root(command, cwd) {
        return Some(local.to_string_lossy().into_owned());
    }
    let path_var = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path_var) {
        let candidate = dir.join(command);
        if is_executable(&candidate) {
            return Some(candidate.to_string_lossy().into_owned());
        }
        #[cfg(windows)]
        for ext in [".exe", ".cmd", ".bat"] {
            let candidate = dir.join(format!("{command}{ext}"));
            if candidate.exists() {
                return Some(candidate.to_string_lossy().into_owned());
            }
        }
    }
    None
}

#[cfg(unix)]
fn is_executable(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    std::fs::metadata(path).is_ok_and(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
}

#[cfg(not(unix))]
fn is_executable(path: &Path) -> bool {
    path.is_file()
}

// =============================================================================
// TypeScript Server Selection
// =============================================================================

fn typescript_package_dir(tsc_path: &Path) -> Option<PathBuf> {
    let real = std::fs::canonicalize(tsc_path).unwrap_or_else(|_| tsc_path.to_path_buf());
    let bin_dir = tsc_path.parent()?;
    let mut candidates: Vec<PathBuf> = Vec::new();
    if real.parent().and_then(|p| p.file_name()) == Some(std::ffi::OsStr::new("bin")) {
        if let Some(grand) = real.parent().and_then(|p| p.parent()) {
            candidates.push(grand.to_path_buf());
        }
    }
    candidates.push(bin_dir.join("..").join("typescript"));
    candidates.push(bin_dir.join("node_modules").join("typescript"));
    candidates
        .into_iter()
        .find(|dir| dir.join("package.json").exists())
}

/// TypeScript 7 dropped `lib/tsserver.js`; older releases reject `tsc --lsp`.
fn typescript_speaks_lsp(tsc_path: &Path) -> bool {
    typescript_package_dir(tsc_path)
        .is_some_and(|dir| !dir.join("lib").join("tsserver.js").exists())
}

fn select_typescript_server(servers: &mut HashMap<String, ServerConfig>) {
    let Some(native) = servers.get("typescript-native") else {
        return;
    };
    let Some(resolved) = native.resolved_command.clone() else {
        return;
    };
    let native_path = PathBuf::from(&resolved);
    if typescript_speaks_lsp(&native_path) {
        servers.remove("typescript-language-server");
    } else {
        servers.remove("typescript-native");
    }
}

// =============================================================================
// Config Sources
// =============================================================================

fn config_dirs(base: &Path) -> Vec<PathBuf> {
    [".omp", ".pi", ".claude"]
        .iter()
        .map(|d| base.join(d))
        .collect()
}

fn source_files(cwd: &Path) -> Vec<PathBuf> {
    let mut sources: Vec<PathBuf> = Vec::new();
    // Highest priority first (mirrors TS `getConfigSources` before the reverse).
    for filename in CONFIG_FILENAMES {
        sources.push(cwd.join(filename));
    }
    for dir in config_dirs(cwd) {
        for filename in CONFIG_FILENAMES {
            sources.push(dir.join(filename));
        }
    }
    if let Some(home) = home_dir() {
        for dir in config_dirs(&home) {
            for filename in CONFIG_FILENAMES {
                sources.push(dir.join(filename));
            }
        }
        for filename in CONFIG_FILENAMES {
            sources.push(home.join(filename));
        }
    }
    sources
}

fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}

/// Load and merge the LSP config for `cwd` (does not consult the cache).
#[must_use]
pub fn load_config(cwd: &Path) -> LspConfig {
    let defaults: Value = serde_json::from_str(DEFAULTS_JSON).unwrap_or(Value::Null);
    let default_servers: HashMap<String, Value> = defaults
        .as_object()
        .map(|o| o.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
        .unwrap_or_default();

    let mut merged: HashMap<String, ServerConfig> = HashMap::new();
    for (name, raw) in &default_servers {
        if let Some(cfg) = normalize_server_config(name, raw) {
            merged.insert(name.clone(), cfg);
        }
    }

    let mut idle_timeout_ms: Option<u64> = None;
    // TS reverses the source list so the lowest priority applies first.
    for file in source_files(cwd).iter().rev() {
        if let Some((servers, idle)) = read_config_file(file) {
            merged = merge_servers(&merged, &servers);
            if idle.is_some() {
                idle_timeout_ms = idle;
            }
        }
    }

    // Filter to servers whose project markers exist and whose binary resolves.
    let mut servers: HashMap<String, ServerConfig> = HashMap::new();
    for (name, mut config) in merged {
        if config.disabled == Some(true) {
            continue;
        }
        if !has_root_markers(cwd, &config.root_markers) {
            continue;
        }
        let Some(resolved) = resolve_command(&config.command, cwd) else {
            continue;
        };
        // Custom linter clients (biome/swiftlint) are not ported; drop them so
        // they never shadow a real language server.
        if matches!(name.as_str(), "biome" | "swiftlint") {
            continue;
        }
        // `$PID` token substitution (omnisharp).
        if let Some(args) = &mut config.args {
            for arg in args.iter_mut() {
                if arg == "$PID" {
                    *arg = std::process::id().to_string();
                }
            }
        }
        config.resolved_command = Some(resolved);
        servers.insert(name, config);
    }
    select_typescript_server(&mut servers);

    LspConfig {
        servers,
        idle_timeout_ms,
    }
}

// =============================================================================
// Per-cwd Cache
// =============================================================================

static CONFIG_CACHE: OnceLock<Mutex<HashMap<PathBuf, Arc<LspConfig>>>> = OnceLock::new();

fn cache() -> &'static Mutex<HashMap<PathBuf, Arc<LspConfig>>> {
    CONFIG_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Cached config for `cwd` (omp `getConfig`).
#[must_use]
pub fn get_config(cwd: &Path) -> Arc<LspConfig> {
    let key = crate::utils::normalize_path(cwd);
    if let Ok(guard) = cache().lock() {
        if let Some(cfg) = guard.get(&key) {
            return Arc::clone(cfg);
        }
    }
    let cfg = Arc::new(load_config(cwd));
    if let Ok(mut guard) = cache().lock() {
        guard.insert(key, Arc::clone(&cfg));
    }
    Arc::clone(&cfg)
}

/// Drop the cached config for `cwd` (omp `reload *`).
pub fn invalidate_config(cwd: &Path) {
    let key = crate::utils::normalize_path(cwd);
    if let Ok(mut guard) = cache().lock() {
        guard.remove(&key);
    }
}

// =============================================================================
// Server Selection
// =============================================================================

/// All servers that can handle a file based on extension, primary first.
#[must_use]
pub fn get_servers_for_file(config: &LspConfig, file_path: &Path) -> Vec<(String, ServerConfig)> {
    let ext = file_path
        .extension()
        .map(|e| format!(".{}", e.to_string_lossy().to_lowercase()))
        .unwrap_or_default();
    let ext_no_dot = ext.trim_start_matches('.').to_string();
    let file_name = file_path
        .file_name()
        .map(|n| n.to_string_lossy().to_lowercase())
        .unwrap_or_default();

    let mut matches: Vec<(String, ServerConfig)> = Vec::new();
    for (name, server_config) in &config.servers {
        let supports = server_config.file_types.iter().any(|ft| {
            let normalized = ft.to_lowercase();
            let normalized_no_dot = normalized.trim_start_matches('.').to_string();
            normalized == ext
                || normalized == file_name
                || normalized_no_dot == ext_no_dot
                || normalized_no_dot == file_name
        });
        if supports {
            matches.push((name.clone(), server_config.clone()));
        }
    }
    // Primary (non-linter) servers first.
    matches.sort_by_key(|(_, c)| i32::from(c.is_linter == Some(true)));
    matches
}

/// Primary server for a file (type-checkers preferred over linters).
#[must_use]
pub fn get_server_for_file(config: &LspConfig, file_path: &Path) -> Option<(String, ServerConfig)> {
    get_servers_for_file(config, file_path).into_iter().next()
}
