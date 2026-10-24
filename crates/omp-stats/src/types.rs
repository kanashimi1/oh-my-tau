//! Port of `stats/src/shared-types.ts` + `stats/src/types.ts`.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Minimal pi-ai `ServiceTier` (wire strings).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ServiceTier {
    Auto,
    Default,
    Flex,
    Scale,
    Priority,
    Ultrafast,
}

/// Minimal pi-ai `ServiceTierByFamily`.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ServiceTierByFamily {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub openai: Option<ServiceTier>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anthropic: Option<ServiceTier>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub google: Option<ServiceTier>,
}

/// Minimal pi-ai `StopReason`. A wire string this port does not know (written
/// by a newer producer) is kept verbatim in `Other` and round-trips unchanged.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum StopReason {
    #[default]
    Stop,
    Length,
    ToolUse,
    Error,
    Aborted,
    Other(String),
}

impl StopReason {
    /// The wire string (`toolUse`, not `tool_calls`); `Other` is returned as stored.
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::Stop => "stop",
            Self::Length => "length",
            Self::ToolUse => "toolUse",
            Self::Error => "error",
            Self::Aborted => "aborted",
            Self::Other(raw) => raw,
        }
    }

    /// Parse a wire string. `tool_calls` is the legacy alias of `toolUse`; any
    /// other unknown string becomes `Other`.
    #[must_use]
    pub fn from_wire(raw: &str) -> Self {
        match raw {
            "stop" => Self::Stop,
            "length" => Self::Length,
            "toolUse" | "tool_calls" => Self::ToolUse,
            "error" => Self::Error,
            "aborted" => Self::Aborted,
            other => Self::Other(other.to_string()),
        }
    }
}

impl Serialize for StopReason {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for StopReason {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = String::deserialize(deserializer)?;
        Ok(Self::from_wire(&raw))
    }
}

/// Minimal pi-ai money/cost bucket.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UsageCost {
    pub input: f64,
    pub output: f64,
    #[serde(rename = "cacheRead")]
    pub cache_read: f64,
    #[serde(rename = "cacheWrite")]
    pub cache_write: f64,
    pub total: f64,
}

/// Minimal pi-ai `Usage`.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Usage {
    pub input: i64,
    pub output: i64,
    #[serde(rename = "cacheRead")]
    pub cache_read: i64,
    #[serde(rename = "cacheWrite")]
    pub cache_write: i64,
    #[serde(rename = "totalTokens")]
    pub total_tokens: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub premium_requests: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<i64>,
    pub cost: UsageCost,
}

/// Minimal pi-ai `Usage` with possibly-partial cost (ingest input).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UsageInputCost {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<f64>,
    #[serde(rename = "cacheRead", default, skip_serializing_if = "Option::is_none")]
    pub cache_read: Option<f64>,
    #[serde(
        rename = "cacheWrite",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub cache_write: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UsageInput {
    #[serde(default)]
    pub input: i64,
    #[serde(default)]
    pub output: i64,
    #[serde(rename = "cacheRead", default)]
    pub cache_read: i64,
    #[serde(rename = "cacheWrite", default)]
    pub cache_write: i64,
    #[serde(rename = "totalTokens", default)]
    pub total_tokens: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub premium_requests: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost: Option<UsageInputCost>,
    /// Provider-side orchestration tokens (pi-ai `Usage.orchestration`):
    /// billed on top of the conversation buckets (e.g. multi-agent models).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub orchestration: Option<UsageOrchestration>,
    /// Cache-write TTL split (pi-ai `Usage.cttl`): prices 5m vs 1h writes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cttl: Option<UsageCttl>,
}

/// pi-ai `Usage.orchestration`.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UsageOrchestration {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_read: Option<i64>,
}

/// pi-ai `Usage.cttl`.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct UsageCttl {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ephemeral5m: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ephemeral1h: Option<i64>,
}

/// Minimal pi-ai `AssistantMessage` (fields stats reads).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssistantMessage {
    #[serde(default)]
    pub role: String,
    #[serde(default)]
    pub content: serde_json::Value,
    #[serde(default)]
    pub api: String,
    #[serde(default)]
    pub provider: String,
    #[serde(default)]
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<ServiceTier>,
    #[serde(default)]
    pub usage: UsageInput,
    #[serde(default)]
    pub stop_reason: StopReason,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
    #[serde(default)]
    pub timestamp: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttft: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub is_error: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<i64>,
    #[serde(flatten)]
    pub rest: BTreeMap<String, serde_json::Value>,
}

// ---------- shared-types.ts ----------

/// Aggregated stats for a model or folder.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AggregatedStats {
    pub total_requests: i64,
    pub successful_requests: i64,
    pub failed_requests: i64,
    pub error_rate: f64,
    pub total_input_tokens: i64,
    pub total_output_tokens: i64,
    pub total_cache_read_tokens: i64,
    pub total_cache_write_tokens: i64,
    pub cache_rate: f64,
    pub cache_savings: f64,
    pub total_cost: f64,
    pub unpriced_requests: i64,
    pub total_premium_requests: f64,
    #[serde(default)]
    pub avg_duration: Option<f64>,
    #[serde(default)]
    pub avg_ttft: Option<f64>,
    #[serde(default)]
    pub avg_tokens_per_second: Option<f64>,
    pub first_timestamp: i64,
    pub last_timestamp: i64,
}

/// Stats grouped by model.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelStats {
    #[serde(flatten)]
    pub base: AggregatedStats,
    pub model: String,
    pub provider: String,
}

/// Stats grouped by folder.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FolderStats {
    #[serde(flatten)]
    pub base: AggregatedStats,
    pub folder: String,
}

/// Time series data point.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TimeSeriesPoint {
    pub timestamp: i64,
    pub requests: i64,
    pub errors: i64,
    pub tokens: i64,
    pub cost: f64,
}

/// Model usage time series data point (daily buckets).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelTimeSeriesPoint {
    pub timestamp: i64,
    pub model: String,
    pub provider: String,
    pub requests: i64,
}

/// Model performance time series data point (daily buckets).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelPerformancePoint {
    pub timestamp: i64,
    pub model: String,
    pub provider: String,
    pub requests: i64,
    #[serde(default)]
    pub avg_ttft: Option<f64>,
    #[serde(default)]
    pub avg_tokens_per_second: Option<f64>,
}

/// Cost time series data point (daily buckets).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CostTimeSeriesPoint {
    pub timestamp: i64,
    pub model: String,
    pub provider: String,
    pub cost: f64,
    pub unpriced_requests: i64,
    pub cost_input: f64,
    pub cost_output: f64,
    pub cost_cache_read: f64,
    pub cost_cache_write: f64,
    pub requests: i64,
}

/// One local calendar day of aggregate request activity.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DailyActivityPoint {
    pub day: String,
    pub cost: f64,
    pub requests: i64,
    pub total_tokens: i64,
}

/// Overall dashboard stats.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DashboardStats {
    pub overall: AggregatedStats,
    pub by_model: Vec<ModelStats>,
    pub by_folder: Vec<FolderStats>,
    pub by_agent_type: Vec<AgentTypeStats>,
    pub time_series: Vec<TimeSeriesPoint>,
    pub model_series: Vec<ModelTimeSeriesPoint>,
    pub model_performance_series: Vec<ModelPerformancePoint>,
    pub cost_series: Vec<CostTimeSeriesPoint>,
}

/// Which agent produced a message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AgentType {
    #[default]
    Main,
    Subagent,
    Advisor,
}

/// Token usage aggregated by {@link AgentType}.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentTypeStats {
    pub agent_type: AgentType,
    pub total_requests: i64,
    pub total_input_tokens: i64,
    pub total_output_tokens: i64,
    pub total_cache_read_tokens: i64,
    pub total_cache_write_tokens: i64,
    pub total_cost: f64,
}

/// Frustration tallies over a set of user messages.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FrustrationCounts {
    pub messages: i64,
    pub judged: i64,
    pub annoyed: i64,
    pub at_assistant: i64,
    pub angry: i64,
}

/// Frustration tallies for one model version.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FrustrationModelStats {
    #[serde(flatten)]
    pub counts: FrustrationCounts,
    pub key: String,
    pub label: String,
    pub model_class: String,
    pub family: Option<String>,
    pub revision: Option<String>,
    pub models: Vec<String>,
    pub first_seen: i64,
}

/// Lifecycle of the dashboard-host judge run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FrustrationJobState {
    #[default]
    Idle,
    Running,
    Done,
    Cancelled,
    Failed,
}

/// Progress of the frustration judge run.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FrustrationJobStatus {
    pub state: FrustrationJobState,
    pub total: i64,
    pub done: i64,
    pub failed: i64,
    pub cost: f64,
    #[serde(default)]
    pub judge: Option<String>,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub started_at: Option<i64>,
    #[serde(default)]
    pub finished_at: Option<i64>,
    pub concurrency: i64,
}

/// Payload of GET /api/stats/frustration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FrustrationDashboardStats {
    pub overall: FrustrationCounts,
    pub by_model: Vec<FrustrationModelStats>,
    pub judge_available: bool,
    pub job: FrustrationJobStatus,
}

/// Payload of GET /api/frustration/estimate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum FrustrationEstimate {
    #[serde(rename_all = "camelCase")]
    Available {
        available: bool,
        messages: i64,
        chars: i64,
        input_tokens: i64,
        cost: f64,
        judge: String,
    },
    #[serde(rename_all = "camelCase")]
    Unavailable { available: bool, reason: String },
}

/// Token savings from a single source type.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GainSourceTotals {
    pub saved_tokens: i64,
    pub saved_bytes: i64,
    pub hits: i64,
    pub output_bytes: i64,
    pub original_bytes: i64,
    #[serde(default)]
    pub reduction_percent: Option<f64>,
}

/// Per-source breakdown identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GainSource {
    #[serde(rename = "snapcompact")]
    Snapcompact,
}

/// Time-series point for gain (daily bucket).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GainTimeSeriesPoint {
    pub date: String,
    pub snapcompact: f64,
    pub total: f64,
}

/// Complete gain dashboard payload.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GainDashboardStats {
    pub overall: GainSourceTotals,
    pub by_source: BTreeMap<GainSource, GainSourceTotals>,
    pub time_series: Vec<GainTimeSeriesPoint>,
    #[serde(default)]
    pub project: Option<String>,
    pub projects: Vec<String>,
}

/// Aggregated usage for a single tool.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolUsageStats {
    pub tool: String,
    pub calls: i64,
    pub errors: i64,
    pub args_chars: i64,
    pub result_chars: i64,
    pub total_tokens_share: f64,
    pub output_tokens_share: f64,
    pub cost_share: f64,
    pub unpriced_requests_share: f64,
    pub last_used: i64,
}

/// Per-(tool, model) breakdown.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolModelStats {
    #[serde(flatten)]
    pub base: ToolUsageStats,
    pub model: String,
    pub provider: String,
}

/// Tool-call time-series point.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolTimeSeriesPoint {
    pub timestamp: i64,
    pub tool: String,
    pub calls: i64,
    pub errors: i64,
}

/// Complete tools dashboard payload.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolDashboardStats {
    pub by_tool: Vec<ToolUsageStats>,
    pub by_tool_model: Vec<ToolModelStats>,
    pub series: Vec<ToolTimeSeriesPoint>,
}

/// Aggregated request/token/cost totals for one provider.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderAggregate {
    pub provider: String,
    pub total_requests: i64,
    pub failed_requests: i64,
    pub models: i64,
    pub total_input_tokens: i64,
    pub total_output_tokens: i64,
    pub total_cache_read_tokens: i64,
    pub total_cache_write_tokens: i64,
    pub total_tokens: i64,
    pub total_cost: f64,
    pub unpriced_requests: i64,
    pub total_premium_requests: f64,
    #[serde(default)]
    pub avg_tokens_per_second: Option<f64>,
}

/// Token burn attributed to one local hour-of-day (0-23) for one provider.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderHourlyPoint {
    pub provider: String,
    pub hour: i64,
    pub total_tokens: i64,
    pub output_tokens: i64,
    pub requests: i64,
}

/// Provider token/cost time-series point.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderTimeSeriesPoint {
    pub timestamp: i64,
    pub provider: String,
    pub total_tokens: i64,
    pub cost: f64,
    pub unpriced_requests: i64,
    pub requests: i64,
}

/// One recorded usage-limit snapshot.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UsageWindowPoint {
    pub timestamp: i64,
    #[serde(default)]
    pub used_fraction: Option<f64>,
    pub exhausted: bool,
}

/// Utilization history for one (account, limit window) pair.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UsageWindowSeries {
    pub provider: String,
    pub account_key: String,
    pub account_label: String,
    pub window_key: String,
    pub window_label: String,
    pub points: Vec<UsageWindowPoint>,
}

/// Derived subscription insight.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderWindowInsight {
    pub provider: String,
    pub window_key: String,
    pub window_label: String,
    pub accounts: i64,
    pub cycles: i64,
    pub fraction_consumed: f64,
    #[serde(default)]
    pub est_tokens_per_window: Option<f64>,
    pub peak_concurrent_fraction: f64,
    pub ideal_accounts: i64,
    pub exhausted_events: i64,
}

/// Ingest state of background session sync.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LiveSyncStatus {
    pub phase: LiveSyncPhase,
    pub current: i64,
    pub total: i64,
    pub processed: i64,
    #[serde(default)]
    pub last_synced_at: Option<i64>,
    #[serde(default)]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LiveSyncPhase {
    #[default]
    Idle,
    Syncing,
    Error,
}

/// Payload of GET /api/events and /api/status.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LiveStatus {
    pub version: i64,
    pub sync: LiveSyncStatus,
    pub indexing_hours: i64,
}

/// Providers dashboard payload.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderDashboardStats {
    pub providers: Vec<ProviderAggregate>,
    pub hourly: Vec<ProviderHourlyPoint>,
    pub series: Vec<ProviderTimeSeriesPoint>,
}

/// Subscription-window payload.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderWindowStats {
    pub window_insights: Vec<ProviderWindowInsight>,
    pub usage_series: Vec<UsageWindowSeries>,
}

/// One row of the Traces session list.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionSummary {
    pub file: String,
    pub folder: String,
    #[serde(default)]
    pub title: Option<String>,
    pub started_at: i64,
    pub ended_at: f64,
    pub requests: i64,
    pub tool_calls: i64,
    pub subagents: i64,
    pub total_tokens: i64,
    pub cost_total: f64,
    pub unpriced_requests: i64,
    pub models: Vec<String>,
}

/// Trace span kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TraceSpanKind {
    #[default]
    Turn,
    Model,
    Tool,
    Subagent,
    Background,
}

/// One rendered block on a trace track lane.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TraceSpan {
    pub id: String,
    pub kind: TraceSpanKind,
    pub start: i64,
    pub end: i64,
    pub label: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entry_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokens: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttft: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub is_error: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unterminated: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub child_track_id: Option<String>,
}

/// Point event drawn on a track header row.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TraceMarker {
    pub time: i64,
    pub kind: TraceMarkerKind,
    pub label: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TraceMarkerKind {
    #[default]
    #[serde(rename = "compaction")]
    Compaction,
    #[serde(rename = "model_change")]
    ModelChange,
    #[serde(rename = "mode_change")]
    ModeChange,
    #[serde(rename = "reset")]
    Reset,
    #[serde(rename = "session_exit")]
    SessionExit,
}

/// One transcript in a trace.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TraceTrack {
    pub id: String,
    pub parent_id: Option<String>,
    pub label: String,
    #[serde(default)]
    pub agent: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    pub file: String,
    pub spans: Vec<TraceSpan>,
    pub markers: Vec<TraceMarker>,
}

/// Per-tool duration aggregate across all tracks of one trace.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TraceToolStat {
    pub tool: String,
    pub calls: i64,
    pub errors: i64,
    pub total_ms: i64,
    pub max_ms: i64,
}

/// Headline aggregates for one trace.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TraceSummary {
    pub wall_ms: i64,
    pub model_ms: i64,
    pub tool_ms: i64,
    pub idle_ms: i64,
    pub turns: i64,
    pub requests: i64,
    pub tool_calls: i64,
    pub subagents: i64,
    pub total_tokens: i64,
    pub cost_total: f64,
    pub unpriced_requests: i64,
    pub tool_stats: Vec<TraceToolStat>,
}

/// Complete span tree for one session.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionTrace {
    pub file: String,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub cwd: Option<String>,
    pub started_at: i64,
    pub ended_at: i64,
    pub mtime_ms: f64,
    pub etag: String,
    pub tracks: Vec<TraceTrack>,
    pub summary: TraceSummary,
}

// ---------- types.ts ----------

/// Extracted stats from an assistant message.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MessageStats {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<i64>,
    pub session_file: String,
    pub entry_id: String,
    pub folder: String,
    pub model: String,
    pub provider: String,
    pub api: String,
    pub timestamp: i64,
    #[serde(default)]
    pub duration: Option<f64>,
    #[serde(default)]
    pub ttft: Option<f64>,
    pub stop_reason: StopReason,
    #[serde(default)]
    pub error_message: Option<String>,
    pub usage: Usage,
    pub agent_type: AgentType,
    #[serde(default)]
    pub service_tier: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost_unpriced: Option<bool>,
}

/// Session-recorded token usage before pricing.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MessageStatsInput {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<i64>,
    pub session_file: String,
    pub entry_id: String,
    pub folder: String,
    pub model: String,
    pub provider: String,
    pub api: String,
    pub timestamp: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttft: Option<f64>,
    pub stop_reason: StopReason,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
    pub usage: UsageInput,
    pub agent_type: AgentType,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<ServiceTier>,
}

/// Full details of a request, including content.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RequestDetails {
    #[serde(flatten)]
    pub base: MessageStats,
    pub messages: Vec<serde_json::Value>,
    pub output: serde_json::Value,
}

/// Session log entry: session header.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionHeader {
    #[serde(rename = "type")]
    pub entry_type: String,
    pub version: i64,
    pub id: String,
    pub timestamp: String,
    pub cwd: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
}

/// Session log entry: message.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionMessageEntry {
    #[serde(rename = "type")]
    pub entry_type: String,
    pub id: String,
    pub parent_id: Option<String>,
    pub timestamp: String,
    pub message: serde_json::Value,
}

/// Session log entry: service tier change.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionServiceTierChangeEntry {
    #[serde(rename = "type")]
    pub entry_type: String,
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<Option<String>>,
    pub timestamp: String,
    #[serde(default)]
    pub service_tier: Option<serde_json::Value>,
}

/// Session log entry: model usage.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionModelUsageEntry {
    #[serde(rename = "type")]
    pub entry_type: String,
    pub id: String,
    pub parent_id: Option<String>,
    pub timestamp: String,
    pub purpose: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    pub api: String,
    pub provider: String,
    pub model: String,
    pub usage: Usage,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<StopReason>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
}

/// Session log entry: custom.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionCustomEntry {
    #[serde(rename = "type")]
    pub entry_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<Option<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<String>,
    pub custom_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<BTreeMap<String, serde_json::Value>>,
}

/// Structural variants the trace builder matches on beyond messages.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionTypedEntry {
    #[serde(rename = "type")]
    pub entry_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<Option<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<String>,
    #[serde(flatten)]
    pub rest: BTreeMap<String, serde_json::Value>,
}

/// Union of session log entries.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SessionEntry {
    Header(SessionHeader),
    Message(SessionMessageEntry),
    ServiceTierChange(SessionServiceTierChangeEntry),
    ModelUsage(SessionModelUsageEntry),
    Custom(SessionCustomEntry),
    Typed(SessionTypedEntry),
    Other(serde_json::Value),
}

/// Behavioral stats extracted from a single user message.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UserMessageStats {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<i64>,
    pub session_file: String,
    pub entry_id: String,
    pub folder: String,
    pub timestamp: i64,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub provider: Option<String>,
    pub chars: i64,
    pub words: i64,
    pub yelling: i64,
    pub profanity: i64,
    pub anguish: i64,
    pub negation: i64,
    pub repetition: i64,
    pub blame: i64,
    pub prose: String,
    pub prose_hash: String,
}

/// Pair emitted by the parser when linking user messages to assistant models.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UserMessageLink {
    pub session_file: String,
    pub entry_id: String,
    pub model: String,
    pub provider: String,
}

/// One tool call extracted from an assistant message.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCallStats {
    pub session_file: String,
    pub entry_id: String,
    pub tool_call_id: String,
    pub folder: String,
    pub tool_name: String,
    pub model: String,
    pub provider: String,
    pub timestamp: i64,
    pub agent_type: AgentType,
    pub calls_in_turn: i64,
    pub args_chars: i64,
}

/// Result linkage emitted when parser sees a toolResult entry.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolResultLink {
    pub session_file: String,
    pub tool_call_id: String,
    pub result_chars: i64,
    pub is_error: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optional_absent_omitted() {
        let span = TraceSpan {
            id: "x".into(),
            kind: TraceSpanKind::Turn,
            start: 1,
            end: 2,
            label: "l".into(),
            detail: None,
            entry_id: None,
            tool_call_id: None,
            model: None,
            tokens: None,
            cost: None,
            ttft: None,
            is_error: None,
            unterminated: None,
            child_track_id: None,
        };
        let v = serde_json::to_value(&span).unwrap();
        assert!(v.get("detail").is_none());
    }

    #[test]
    fn nullable_null_emitted() {
        let s = SessionSummary {
            file: "f".into(),
            folder: "p".into(),
            title: None,
            started_at: 1,
            ended_at: 2.0,
            requests: 0,
            tool_calls: 0,
            subagents: 0,
            total_tokens: 0,
            cost_total: 0.0,
            unpriced_requests: 0,
            models: vec![],
        };
        let v = serde_json::to_value(&s).unwrap();
        assert_eq!(v["title"], serde_json::Value::Null);
    }

    #[test]
    fn enum_roundtrip() {
        let a = AgentType::Subagent;
        let s = serde_json::to_string(&a).unwrap();
        assert_eq!(s, "\"subagent\"");
        let b: AgentType = serde_json::from_str(&s).unwrap();
        assert_eq!(a, b);
    }
}
