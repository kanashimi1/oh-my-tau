//! Per-server manager with lazy connect and failure isolation, a port of the
//! agent-facing half of `manager.ts` + the `DeferredMCPTool` half of
//! `tool-bridge.ts`.
//!
//! Each configured server is an independent state machine:
//! `NotConnected → Connected | Failed`. `tools()` connects every server that
//! is not yet resolved (failures are recorded, never propagated — one dead
//! server must not hide another server's tools). `call()` connects lazily on
//! first use, and retries once after reconnect when the error looks like a
//! stale/dead connection (omp `isRetriableConnectionError`).

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use serde_json::Value;
use tokio::sync::Mutex;

use crate::client::{call_tool, connect_to_server, list_tools, Connection};
use crate::config::{load_configs, McpServerConfig};
use crate::errors::McpError;
use crate::names::create_tool_name;
use crate::{ToolCallResult, ToolDefinition};

/// One tool discovered on a server, with its omp registry name pre-minted.
#[derive(Debug, Clone)]
pub struct DiscoveredTool {
    pub server: String,
    /// Registry key: `mcp__<server>_<tool>`, minted by [`create_tool_name`].
    pub registry_name: String,
    pub definition: ToolDefinition,
}

/// Per-server health, for diagnostics and UIs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ServerStatus {
    /// Never attempted / not yet resolved.
    Pending,
    Connected,
    /// Connection or tool listing failed; the message is user-safe.
    Failed(String),
}

enum ServerState {
    NotConnected,
    Connecting,
    Connected(Box<Connection>),
    Failed(String),
}

struct ServerEntry {
    config: McpServerConfig,
    state: ServerState,
    status: ServerStatus,
}

/// MCP server manager: owns configs, connections and tool caches.
pub struct McpManager {
    cwd: Arc<PathBuf>,
    servers: Mutex<BTreeMap<String, ServerEntry>>,
}

impl std::fmt::Debug for McpManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("McpManager")
            .field("cwd", &self.cwd)
            .finish_non_exhaustive()
    }
}

impl McpManager {
    /// Build a manager from explicit server configs (tests and embedders that
    /// resolve config themselves).
    pub fn new(cwd: Arc<PathBuf>, configs: BTreeMap<String, McpServerConfig>) -> Self {
        let servers = configs
            .into_iter()
            .map(|(name, config)| {
                (
                    name,
                    ServerEntry {
                        config,
                        state: ServerState::NotConnected,
                        status: ServerStatus::Pending,
                    },
                )
            })
            .collect();
        Self {
            cwd,
            servers: Mutex::new(servers),
        }
    }

    /// Build a manager from omp's standard config locations for `cwd`.
    #[must_use]
    pub fn from_default_configs(cwd: &std::path::Path) -> Self {
        Self::new(Arc::new(cwd.to_path_buf()), load_configs(cwd))
    }

    /// Configured server names.
    pub async fn server_names(&self) -> Vec<String> {
        self.servers.lock().await.keys().cloned().collect()
    }

    /// Health snapshot per server.
    pub async fn statuses(&self) -> BTreeMap<String, ServerStatus> {
        self.servers
            .lock()
            .await
            .iter()
            .map(|(name, entry)| (name.clone(), entry.status.clone()))
            .collect()
    }

    /// Failure message for a server, if it failed to connect/list.
    pub async fn failure(&self, name: &str) -> Option<String> {
        match &self.servers.lock().await.get(name)?.status {
            ServerStatus::Failed(message) => Some(message.clone()),
            _ => None,
        }
    }
}

impl McpManager {
    /// Ensure `name` is resolved (connected with tools cached, or failed).
    /// Failures are recorded and returned, never panicking nor affecting
    /// other servers.
    async fn resolve(&self, name: &str) -> Result<(), McpError> {
        let config: Option<McpServerConfig> = {
            let mut guard = self.servers.lock().await;
            match guard.get_mut(name) {
                None => {
                    return Err(McpError::Config(format!("unknown MCP server: {name}")));
                }
                Some(entry) => match &mut entry.state {
                    ServerState::Connected(_) | ServerState::Connecting => None,
                    ServerState::Failed(message) => {
                        return Err(McpError::Config(format!(
                            "MCP server \"{name}\": {message}"
                        )));
                    }
                    ServerState::NotConnected => {
                        entry.state = ServerState::Connecting;
                        Some(entry.config.clone())
                    }
                },
            }
        };
        let Some(config) = config else {
            return Ok(());
        };
        let result: Result<Connection, McpError> = async {
            let mut connection = connect_to_server(name, &config, self.cwd.clone()).await?;
            list_tools(&mut connection).await?;
            Ok(connection)
        }
        .await;
        let mut guard = self.servers.lock().await;
        let entry = guard
            .get_mut(name)
            .expect("server entry vanished while connecting");
        match result {
            Ok(connection) => {
                entry.state = ServerState::Connected(Box::new(connection));
                entry.status = ServerStatus::Connected;
                Ok(())
            }
            Err(error) => {
                let message = error.to_string();
                entry.state = ServerState::Failed(message.clone());
                entry.status = ServerStatus::Failed(message.clone());
                Err(McpError::Config(format!(
                    "MCP server \"{name}\": {message}"
                )))
            }
        }
    }

    /// Connect every configured server once and return all discovered tools
    /// with their omp registry names (`mcp__<server>_<tool>`). Per-server
    /// failures are isolated: they are recorded in [`statuses`] and skipped.
    pub async fn tools(&self) -> Vec<DiscoveredTool> {
        let names = self.server_names().await;
        for name in &names {
            if let Err(error) = self.resolve(name).await {
                tracing::warn!("MCP server \"{name}\" unavailable: {error}");
            }
        }
        let guard = self.servers.lock().await;
        let mut out = Vec::new();
        for (server, entry) in guard.iter() {
            if let ServerState::Connected(connection) = &entry.state {
                if let Some(tools) = &connection.tools {
                    for definition in tools {
                        out.push(DiscoveredTool {
                            server: server.clone(),
                            registry_name: create_tool_name(server, &definition.name),
                            definition: definition.clone(),
                        });
                    }
                }
            }
        }
        out
    }

    /// Call `tool` on `server`, connecting lazily on first use. A retriable
    /// connection error triggers one reconnect + retry, like omp's
    /// `DeferredMCPTool.execute`.
    pub async fn call(
        &self,
        server: &str,
        tool: &str,
        args: Value,
    ) -> Result<ToolCallResult, McpError> {
        self.resolve(server).await?;
        let first = {
            let mut guard = self.servers.lock().await;
            let entry = guard.get_mut(server).expect("resolved server vanished");
            match &mut entry.state {
                ServerState::Connected(connection) => {
                    Some(call_tool(connection, tool, args.clone()).await)
                }
                _ => None,
            }
        };
        let Some(first) = first else {
            return Err(McpError::Config(format!(
                "MCP server \"{server}\" is not connected"
            )));
        };
        match first {
            Ok(result) => Ok(result),
            Err(error) if is_retriable(&error) => {
                tracing::info!("MCP server \"{server}\" connection stale ({error}); reconnecting");
                {
                    let mut guard = self.servers.lock().await;
                    if let Some(entry) = guard.get_mut(server) {
                        if let ServerState::Connected(mut connection) =
                            std::mem::replace(&mut entry.state, ServerState::NotConnected)
                        {
                            connection.transport.close().await;
                        }
                        entry.state = ServerState::NotConnected;
                    }
                }
                self.resolve(server).await?;
                let mut guard = self.servers.lock().await;
                let entry = guard.get_mut(server).expect("resolved server vanished");
                match &mut entry.state {
                    ServerState::Connected(connection) => call_tool(connection, tool, args).await,
                    _ => Err(McpError::Config(format!(
                        "MCP server \"{server}\" is not connected"
                    ))),
                }
            }
            Err(error) => Err(error),
        }
    }
}

/// Port of `isRetriableConnectionError`: network-level and stale-session
/// errors warrant a reconnect + single retry.
fn is_retriable(error: &McpError) -> bool {
    let message = error.to_string().to_lowercase();
    if message.starts_with("http 404:")
        || message.starts_with("http 502:")
        || message.starts_with("http 503:")
    {
        return true;
    }
    [
        "econnrefused",
        "econnreset",
        "epipe",
        "enetunreach",
        "ehostunreach",
        "fetch failed",
        "transport not connected",
        "transport closed",
        "network error",
        "closed stdout",
    ]
    .iter()
    .any(|pattern| message.contains(pattern))
}
