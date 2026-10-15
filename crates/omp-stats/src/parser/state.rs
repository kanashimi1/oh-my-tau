//! Parser ↔ ingest contract from `parser.ts` (`SessionParserState`,
//! `ParseSessionResult`). Integrator-owned: `db::ingest` consumes these.

use serde::{Deserialize, Serialize};

use crate::types::{
    MessageStatsInput, ServiceTierByFamily, ToolCallStats, ToolResultLink, UserMessageLink,
    UserMessageStats,
};

/// Resume state persisted per session file (`file_offsets.parser_state` JSON).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionParserState {
    /// Always 1 (TS literal type).
    pub version: u32,
    pub offset: i64,
    pub dev: f64,
    pub ino: f64,
    pub birthtime_ms: f64,
    pub size: i64,
    pub mtime_ms: f64,
    /// `Bun.hash(last ≤256 bytes before offset).toString(16)` (wyhash, seed 0); detects rewrites.
    pub checkpoint: String,
    pub service_tier: Option<ServiceTierByFamily>,
}

/// Rows parsed from one session file since its last offset.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ParseSessionResult {
    pub stats: Vec<MessageStatsInput>,
    pub user_stats: Vec<UserMessageStats>,
    pub user_links: Vec<UserMessageLink>,
    pub tool_calls: Vec<ToolCallStats>,
    pub tool_results: Vec<ToolResultLink>,
    pub new_offset: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parser_state: Option<SessionParserState>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reset: Option<bool>,
}
