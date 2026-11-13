//! TypeScript codegen for ext-host/host.ts.
//!
//! Run `cargo test -p omp-cli export_ts_bindings` to regenerate
//! `ext-host/bindings/events.ts` from the Rust wire types.

use ts_rs::TS;

/// Re-export the wire types we want in TS.
pub use pa_agent::types::{AssistantContent, AssistantMessage, TextContent, ThinkingContent};

/// One exported frame type per AssistantMessageEvent variant.
#[derive(TS)]
#[ts(export, export_to = "../../ext-host/bindings/events.ts")]
pub struct Start {
    pub r#type: &'static str,
}

#[derive(TS)]
#[ts(export, export_to = "../../ext-host/bindings/events.ts")]
pub struct TextStart {
    pub r#type: &'static str,
    pub content_index: usize,
}

#[derive(TS)]
#[ts(export, export_to = "../../ext-host/bindings/events.ts")]
pub struct TextDelta {
    pub r#type: &'static str,
    pub content_index: usize,
    pub delta: String,
}

#[derive(TS)]
#[ts(export, export_to = "../../ext-host/bindings/events.ts")]
pub struct TextEnd {
    pub r#type: &'static str,
    pub content_index: usize,
    pub content: String,
}

#[derive(TS)]
#[ts(export, export_to = "../../ext-host/bindings/events.ts")]
pub struct ThinkingStart {
    pub r#type: &'static str,
    pub content_index: usize,
}

#[derive(TS)]
#[ts(export, export_to = "../../ext-host/bindings/events.ts")]
pub struct ThinkingDelta {
    pub r#type: &'static str,
    pub content_index: usize,
    pub delta: String,
}

#[derive(TS)]
#[ts(export, export_to = "../../ext-host/bindings/events.ts")]
pub struct ThinkingEnd {
    pub r#type: &'static str,
    pub content_index: usize,
}

#[derive(TS)]
#[ts(export, export_to = "../../ext-host/bindings/events.ts")]
pub struct ToolCallStart {
    pub r#type: &'static str,
    pub content_index: usize,
}

#[derive(TS)]
#[ts(export, export_to = "../../ext-host/bindings/events.ts")]
pub struct ToolCallDelta {
    pub r#type: &'static str,
    pub content_index: usize,
    pub delta: String,
}

#[derive(TS)]
#[ts(export, export_to = "../../ext-host/bindings/events.ts")]
pub struct ToolCallEnd {
    pub r#type: &'static str,
    pub content_index: usize,
    pub tool_call: ToolCallWire,
}

/// TS-facing form of `pa_agent::types::ToolCall` (ts-rs can't derive for
/// vendor types without a feature flag; mirror the wire shape manually).
#[derive(TS)]
#[ts(export, export_to = "../../ext-host/bindings/events.ts")]
pub struct ToolCallWire {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
    #[ts(optional)]
    pub thought_signature: Option<String>,
}

impl From<&pa_agent::types::ToolCall> for ToolCallWire {
    fn from(tc: &pa_agent::types::ToolCall) -> Self {
        Self {
            id: tc.id.clone(),
            name: tc.name.clone(),
            arguments: tc.arguments.clone(),
            thought_signature: tc.thought_signature.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn export_ts_bindings() {
        // ts-rs needs explicit export_all_to; export_to attribute alone
        // only sets the filename inside TS_RS_EXPORT_DIR.
        Start::export_all_to("../../ext-host/bindings").expect("export Start");
        TextStart::export_all_to("../../ext-host/bindings").expect("export TextStart");
        TextDelta::export_all_to("../../ext-host/bindings").expect("export TextDelta");
        TextEnd::export_all_to("../../ext-host/bindings").expect("export TextEnd");
        ThinkingStart::export_all_to("../../ext-host/bindings").expect("export ThinkingStart");
        ThinkingDelta::export_all_to("../../ext-host/bindings").expect("export ThinkingDelta");
        ThinkingEnd::export_all_to("../../ext-host/bindings").expect("export ThinkingEnd");
        ToolCallStart::export_all_to("../../ext-host/bindings").expect("export ToolCallStart");
        ToolCallDelta::export_all_to("../../ext-host/bindings").expect("export ToolCallDelta");
        ToolCallEnd::export_all_to("../../ext-host/bindings").expect("export ToolCallEnd");
    }
}
