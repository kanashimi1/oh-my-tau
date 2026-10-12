//! LSP protocol types (omp `lsp/types.ts`).
//!
//! Shapes mirror the TS interfaces exactly where they cross a wire or a
//! config file; loose server payloads (capabilities, hover contents) stay
//! `serde_json::Value`.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

// =============================================================================
// Core LSP Protocol Types
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Position {
    pub line: u32,
    pub character: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Range {
    pub start: Position,
    pub end: Position,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Location {
    pub uri: String,
    pub range: Range,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocationLink {
    #[serde(rename = "originSelectionRange")]
    pub origin_selection_range: Option<Range>,
    #[serde(rename = "targetUri")]
    pub target_uri: String,
    #[serde(rename = "targetRange")]
    pub target_range: Range,
    #[serde(rename = "targetSelectionRange")]
    pub target_selection_range: Option<Range>,
}

// =============================================================================
// Diagnostics
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Diagnostic {
    pub range: Range,
    pub severity: Option<u32>,
    pub code: Option<Value>,
    #[serde(rename = "codeDescription")]
    pub code_description: Option<Value>,
    pub source: Option<String>,
    pub message: String,
    pub tags: Option<Vec<u32>>,
    #[serde(rename = "relatedInformation")]
    pub related_information: Option<Value>,
    pub data: Option<Value>,
}

#[derive(Debug, Clone, Default)]
pub struct PublishedDiagnostics {
    pub diagnostics: Vec<Diagnostic>,
    pub version: Option<i64>,
}

// =============================================================================
// Text Edits
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TextEdit {
    pub range: Range,
    pub new_text: String,
    #[serde(rename = "insertTextFormat")]
    pub insert_text_format: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextDocumentEdit {
    #[serde(rename = "textDocument")]
    pub text_document: VersionedTextDocumentIdentifier,
    pub edits: Vec<TextEdit>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionedTextDocumentIdentifier {
    pub uri: String,
    pub version: Option<i64>,
}

// =============================================================================
// Resource Operations
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateFile {
    pub kind: String, // "create"
    pub uri: String,
    pub options: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenameFile {
    pub kind: String, // "rename"
    #[serde(rename = "oldUri")]
    pub old_uri: String,
    #[serde(rename = "newUri")]
    pub new_uri: String,
    pub options: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeleteFile {
    pub kind: String, // "delete"
    pub uri: String,
    pub options: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum DocumentChange {
    TextDocumentEdit(TextDocumentEdit),
    CreateFile(CreateFile),
    RenameFile(RenameFile),
    DeleteFile(DeleteFile),
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WorkspaceEdit {
    pub changes: Option<std::collections::HashMap<String, Value>>,
    #[serde(rename = "documentChanges")]
    pub document_changes: Option<Vec<DocumentChange>>,
    #[serde(rename = "changeAnnotations")]
    pub change_annotations: Option<Value>,
}

// =============================================================================
// Code Actions
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Command {
    pub title: String,
    pub command: String,
    pub arguments: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeAction {
    pub title: String,
    pub kind: Option<String>,
    pub diagnostics: Option<Vec<Diagnostic>>,
    #[serde(rename = "isPreferred")]
    pub is_preferred: Option<bool>,
    pub disabled: Option<Value>,
    pub edit: Option<WorkspaceEdit>,
    pub command: Option<Command>,
    pub data: Option<Value>,
}

// =============================================================================
// Symbols
// =============================================================================

pub const SYMBOL_KIND_NAMES: [&str; 27] = [
    "Unknown",
    "File",
    "Module",
    "Namespace",
    "Package",
    "Class",
    "Method",
    "Property",
    "Field",
    "Constructor",
    "Enum",
    "Interface",
    "Function",
    "Variable",
    "Constant",
    "String",
    "Number",
    "Boolean",
    "Array",
    "Object",
    "Key",
    "Null",
    "EnumMember",
    "Struct",
    "Event",
    "Operator",
    "TypeParameter",
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentSymbol {
    pub name: String,
    pub detail: Option<String>,
    pub kind: u32,
    pub tags: Option<Vec<u32>>,
    pub deprecated: Option<bool>,
    pub range: Range,
    #[serde(rename = "selectionRange")]
    pub selection_range: Range,
    pub children: Option<Vec<DocumentSymbol>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolInformation {
    pub name: String,
    pub kind: u32,
    pub tags: Option<Vec<u32>>,
    pub deprecated: Option<bool>,
    pub location: Location,
    #[serde(rename = "containerName")]
    pub container_name: Option<String>,
}

// =============================================================================
// Hover
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hover {
    pub contents: Value,
    pub range: Option<Range>,
}

// =============================================================================
// Server Configuration
// =============================================================================

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ServerCapabilities {
    pub flycheck: Option<bool>,
    pub ssr: Option<bool>,
    #[serde(rename = "expandMacro")]
    pub expand_macro: Option<bool>,
    pub runnables: Option<bool>,
    #[serde(rename = "relatedTests")]
    pub related_tests: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceReadyTimings {
    #[serde(rename = "timeoutMs")]
    pub timeout_ms: Option<u64>,
    #[serde(rename = "pollMs")]
    pub poll_ms: Option<u64>,
    #[serde(rename = "settleMs")]
    pub settle_ms: Option<u64>,
    #[serde(rename = "statusRequestTimeoutMs")]
    pub status_request_timeout_ms: Option<u64>,
}

/// One language-server entry from `defaults.json` / user config files.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub command: String,
    pub args: Option<Vec<String>>,
    #[serde(rename = "fileTypes", alias = "extensionToLanguage")]
    pub file_types: Vec<String>,
    /// LSP language identifier sent in didOpen; inferred from the file path when omitted.
    #[serde(rename = "languageId")]
    pub language_id: Option<String>,
    #[serde(rename = "rootMarkers")]
    pub root_markers: Vec<String>,
    #[serde(rename = "initOptions", alias = "initializationOptions")]
    pub init_options: Option<Value>,
    pub settings: Option<Value>,
    pub disabled: Option<bool>,
    #[serde(rename = "warmupTimeoutMs")]
    pub warmup_timeout_ms: Option<u64>,
    #[serde(rename = "workspaceReadyTimings")]
    pub workspace_ready_timings: Option<WorkspaceReadyTimings>,
    #[serde(rename = "isLinter")]
    pub is_linter: Option<bool>,
    pub capabilities: Option<ServerCapabilities>,
    /// Absolute path after `command` was resolved against the cwd/PATH (runtime-only).
    #[serde(rename = "resolvedCommand", skip_deserializing)]
    pub resolved_command: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct LspConfig {
    pub servers: HashMap<String, ServerConfig>,
    /// Idle timeout in milliseconds; clients shut down after this period of inactivity.
    pub idle_timeout_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublishDiagnosticsParams {
    pub uri: String,
    pub diagnostics: Vec<Diagnostic>,
    pub version: Option<i64>,
}
