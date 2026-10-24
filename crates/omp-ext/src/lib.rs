//! Rust client for the omp extension host (`ext-host/host.ts`, Bun).
//!
//! The host loads unmodified omp `ExtensionAPI` extensions and speaks newline-delimited JSON-RPC 2.0
//! on stdio. [`ExtHost`] is the transport; [`hooks`] maps the host's event aggregation onto the
//! `pa-agent` hook points (`transform_context`, `before_tool_call`, `after_tool_call`).

pub mod hooks;
// `emit_auto_compaction_end` mirrors omp's six-field event as one argument per field.
mod host;
#[allow(clippy::too_many_arguments)]
pub mod lifecycle;

pub use host::{
    CommandInfo, EmitResult, ExtHost, HandlerRecord, HostConfig, HostMessage, LoadResult,
    LoadedExtension, ToolInfo,
};
