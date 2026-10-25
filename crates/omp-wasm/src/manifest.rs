//! Plugin manifest types (`omt_manifest` JSON contract).

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// A plugin tool definition from the manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDef {
    /// Tool name (registered as `<plugin>_<tool>`).
    pub name: String,
    /// Human-readable description (empty when omitted).
    #[serde(default)]
    pub description: String,
    /// JSON Schema describing the accepted arguments (`{}` when omitted).
    #[serde(default = "empty_object")]
    pub parameters: Value,
}

fn empty_object() -> Value {
    Value::Object(Default::default())
}

/// A plugin slash command from the manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandDef {
    /// Command name (used as `/<plugin>:<command>`).
    pub name: String,
    /// Human-readable description (empty when omitted).
    #[serde(default)]
    pub description: String,
}

/// The manifest returned by `omt_manifest` as JSON over linear memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    /// Plugin name; also the tool/command prefix.
    pub name: String,
    /// Version string (empty when omitted).
    #[serde(default)]
    pub version: String,
    /// Declared tools.
    #[serde(default)]
    pub tools: Vec<ToolDef>,
    /// Declared commands.
    #[serde(default)]
    pub commands: Vec<CommandDef>,
}
