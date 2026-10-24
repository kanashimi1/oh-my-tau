//! MCP server config loading: a port of `mcp/config.ts` + `discovery/mcp-json.ts`.
//!
//! Sources, matching omp: the user config at `~/.omp/agent/mcp.json` and
//! standalone project configs (`mcp.json` / `.mcp.json` in the project root,
//! `<cwd>/.omp/mcp.json`). The user's `disabledServers`/`enabledServers`
//! lists and per-server `enabled: false` are honored exactly as omp's
//! `loadAllMCPConfigs` does.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::timeout::resolve_timeout_ms;

/// Root `mcp.json` file structure.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct McpConfigFile {
    #[serde(rename = "$schema", default)]
    pub schema: Option<String>,
    #[serde(default)]
    #[serde(rename = "mcpServers")]
    pub mcp_servers: BTreeMap<String, McpServerConfig>,
    /// Names hidden regardless of any source `enabled` flag.
    #[serde(default)]
    #[serde(rename = "disabledServers")]
    pub disabled_servers: Vec<String>,
    /// Names force-enabled when a source reports `enabled: false`.
    #[serde(default)]
    #[serde(rename = "enabledServers")]
    pub enabled_servers: Vec<String>,
}

/// One MCP server entry. Unknown fields are kept so round-tripping configs
/// (e.g. `oauth`) does not lose data.
#[derive(Debug, Clone, Deserialize)]
pub struct McpServerConfig {
    #[serde(default)]
    pub enabled: Option<bool>,
    /// Per-request/connect timeout in ms (default 30000, 0 disables).
    #[serde(default)]
    pub timeout: Option<u64>,
    #[serde(rename = "type", default)]
    pub kind: ServerKind,
    // stdio
    #[serde(default)]
    pub command: Option<String>,
    #[serde(default)]
    pub args: Option<Vec<String>>,
    #[serde(default)]
    pub env: Option<BTreeMap<String, String>>,
    #[serde(default)]
    pub cwd: Option<String>,
    // http/sse
    #[serde(default)]
    pub url: Option<String>,
    #[serde(default)]
    pub headers: Option<BTreeMap<String, String>>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

/// Transport kind; missing `type` means stdio, as in omp.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ServerKind {
    #[default]
    Stdio,
    Http,
    Sse,
}

impl<'de> Deserialize<'de> for ServerKind {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        match Option::<String>::deserialize(d)?.as_deref() {
            None | Some("stdio") => Ok(Self::Stdio),
            Some("http") => Ok(Self::Http),
            Some("sse") => Ok(Self::Sse),
            Some(other) => Err(serde::de::Error::custom(format!(
                "unknown server type: {other}"
            ))),
        }
    }
}

impl McpServerConfig {
    /// Effective request timeout in ms (env override > config > default).
    #[must_use]
    pub fn timeout_ms(&self) -> u64 {
        resolve_timeout_ms(self.timeout)
    }
}

/// Standard user config path: `~/.omp/agent/mcp.json`.
#[must_use]
pub fn user_config_path() -> Option<PathBuf> {
    let home = std::env::var_os("OMP_CONFIG_DIR")
        .map(PathBuf::from)
        .or_else(dirs_home)?;
    Some(home.join(".omp").join("agent").join("mcp.json"))
}

fn dirs_home() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .filter(|h| !h.is_empty())
        .map(PathBuf::from)
}

/// Project config candidates for `cwd`, in omp's discovery order
/// (standalone `mcp.json` first, dotfile second, config dir last; later
/// sources override earlier ones per server name).
#[must_use]
pub fn project_config_paths(cwd: &Path) -> Vec<PathBuf> {
    [
        cwd.join("mcp.json"),
        cwd.join(".mcp.json"),
        cwd.join(".omp").join("mcp.json"),
    ]
    .to_vec()
}

/// Read one config file; a missing file yields an empty config, an unreadable
/// or malformed one is ignored with a warning (exactly like omp, which logs
/// "Ignoring unreadable user MCP config for server lists").
fn read_config(path: &Path) -> McpConfigFile {
    if !path.is_file() {
        return McpConfigFile::default();
    }
    match std::fs::read_to_string(path)
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
    {
        Some(cfg) => cfg,
        None => {
            tracing::warn!("ignoring unreadable MCP config {}", path.display());
            McpConfigFile::default()
        }
    }
}

/// Load merged server configs for `cwd`: user config plus project configs.
/// Per omp, `disabledServers` hides unconditionally, `enabled: false` hides
/// unless listed in `enabledServers`, and project entries override user
/// entries with the same name.
#[must_use]
pub fn load_configs(cwd: &Path) -> BTreeMap<String, McpServerConfig> {
    let user = user_config_path()
        .map(|p| read_config(&p))
        .unwrap_or_default();
    let mut merged = user.mcp_servers.clone();
    for path in project_config_paths(cwd) {
        for (name, cfg) in read_config(&path).mcp_servers {
            merged.insert(name, cfg);
        }
    }

    let disabled: std::collections::HashSet<&str> =
        user.disabled_servers.iter().map(String::as_str).collect();
    let forced: std::collections::HashSet<&str> =
        user.enabled_servers.iter().map(String::as_str).collect();
    merged
        .into_iter()
        .filter(|(name, _)| !disabled.contains(name.as_str()))
        .filter(|(name, cfg)| cfg.enabled != Some(false) || forced.contains(name.as_str()))
        .collect()
}
