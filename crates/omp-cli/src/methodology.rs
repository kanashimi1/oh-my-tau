//! Reason-style provider tool projection and opt-in methodology audit.
//! The agent keeps every tool for eval's `tool.*` bridge; only the provider-facing
//! `LlmContext` is filtered, so deferred tools remain executable without changing
//! the registered tool set or schemas between requests.

use std::collections::BTreeMap;
use std::fs::OpenOptions;
use std::io::Write;
use std::sync::{Arc, Mutex};

use pa_agent::stream::{LlmContext, StreamFn, ToolDefinition};
use pa_agent::types::Model;
use serde_json::json;
use sha2::{Digest, Sha256};

const CORE_TOOLS: &[&str] = &[
    "read", "bash", "edit", "write", "glob", "grep", "eval", "wait",
];

#[derive(Default)]
struct AuditState {
    requests: u64,
    previous_messages: Vec<String>,
    previous_system: String,
    previous_tools: String,
}

static AUDIT: Mutex<Option<AuditState>> = Mutex::new(None);

#[must_use]
pub fn core_profile_requested() -> bool {
    std::env::var("OMP_TOOLS_PROFILE").is_ok_and(|v| v == "core")
}

#[must_use]
pub fn raw_tool_output_requested() -> bool {
    std::env::var("SOLPI_RAW_TOOL_OUTPUT").is_ok_and(|v| v == "1")
}

#[must_use]
pub fn is_core_tool(name: &str) -> bool {
    CORE_TOOLS.contains(&name)
}

fn project_core(tools: &mut Vec<ToolDefinition>) {
    tools.retain(|tool| is_core_tool(&tool.name));
}

fn hash(value: &str) -> String {
    format!("{:x}", Sha256::digest(value.as_bytes()))
}

fn audit(context: &LlmContext) {
    let Ok(path) = std::env::var("METHOD_AUDIT_LOG") else {
        return;
    };
    let mut guard = AUDIT
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let state = guard.get_or_insert_with(AuditState::default);
    let messages: Vec<String> = context
        .messages
        .iter()
        .map(|m| serde_json::to_string(m).unwrap_or_default())
        .collect();
    let system = serde_json::to_string(&context.system_prompt).unwrap_or_default();
    let tools = serde_json::to_string(&context.tools).unwrap_or_default();
    let mut divergence = None;
    for (i, previous) in state.previous_messages.iter().enumerate() {
        if messages.get(i) != Some(previous) {
            divergence = Some(i);
            break;
        }
    }
    if divergence.is_none() && state.previous_messages.len() > messages.len() {
        divergence = Some(messages.len());
    }
    state.requests += 1;
    let mut extra: BTreeMap<String, serde_json::Value> = BTreeMap::new();
    if state.requests > 1 {
        extra.insert(
            "system_unchanged".into(),
            json!(state.previous_system == system),
        );
        extra.insert(
            "tools_unchanged".into(),
            json!(state.previous_tools == tools),
        );
    }
    let record = json!({
        "event": "provider_context",
        "request": state.requests,
        "messages": messages.len(),
        "tools_count": context.tools.len(),
        "append_only": divergence.is_none(),
        "divergence": divergence,
        "system_hash": hash(&system),
        "tools_hash": hash(&tools),
        "extra": extra,
    });
    state.previous_messages = messages;
    state.previous_system = system;
    state.previous_tools = tools;
    drop(guard);
    if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(file, "{record}");
    }
}

/// Apply the provider-only core projection and then audit the exact LLM context.
#[must_use]
pub fn wrap_stream(
    base: StreamFn,
    core: bool,
    extensions: Option<Arc<crate::ext::Extensions>>,
) -> StreamFn {
    Arc::new(move |model: Model, mut context: LlmContext, options| {
        let extensions = extensions.clone();
        let base = base.clone();
        Box::pin(async move {
            if core {
                project_core(&mut context.tools);
            }
            audit(&context);
            if std::env::var("METHOD_PROVIDER_EVENTS").is_ok_and(|v| v == "1") {
                if let Some(ext) = extensions {
                    if let Err(error) = ext.before_provider_request().await {
                        eprintln!("omt: before_provider_request: {error:#}");
                    }
                }
            }
            base(model, context, options).await
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn core_projection_keeps_general_tools_and_defers_platform_tools() {
        let mut tools = ["read", "eval", "task", "web_search", "wait"]
            .into_iter()
            .map(|name| ToolDefinition {
                name: name.into(),
                description: String::new(),
                parameters: json!({"type":"object"}),
            })
            .collect();
        project_core(&mut tools);
        assert_eq!(
            tools.iter().map(|t| t.name.as_str()).collect::<Vec<_>>(),
            ["read", "eval", "wait"]
        );
    }
}
