//! omp-wasm: host runtime for omt WASM plugins.
//!
//! A plugin is a `.wasm` module implementing a JSON-over-linear-memory ABI:
//! it exports `memory`, `omt_alloc`, `omt_manifest`, and `omt_call`, and
//! imports `omt.log` from the host. See `crates/omp-cli` for the agent-tool
//! and slash-command integration, and `docs/wasm-plugins.md` for the ABI.

pub mod host;
pub mod manifest;
pub mod registry;

pub use host::{
    CallKind, CallOutput, HostLimits, WasmModule, DEFAULT_FUEL, DEFAULT_MEMORY_LIMIT,
    DEFAULT_WALL_TIMEOUT,
};
pub use manifest::{CommandDef, Manifest, ToolDef};
pub use registry::{discover, Discovery, PluginEntry};
