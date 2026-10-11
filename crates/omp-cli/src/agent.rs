//! Agent assembly: system prompt, tools, and the pa-ai stream adapter wired
//! into pa-agent's `Agent` (pattern derived from prime-agent's pa-cli, MIT).

use std::path::Path;

use anyhow::{Context, Result};
use pa_agent::agent::{Agent, AgentInitialState, AgentOptions};
use pa_agent::types::AgentMessage;

use std::sync::{Arc, RwLock};

use crate::adapter::{json_round_trip, switchable_stream_fn, ProviderTarget};
use crate::model::Selected;
use crate::tools::builtin_tools;

const BASE_PROMPT: &str = "You are a coding agent running in a terminal. Use the available tools (bash, read, write, edit, grep, glob) to complete the user's task; prefer reading before editing and verify results by running them. Be concise.";

/// Days since 1970-01-01 to (year, month, day), civil calendar.
fn civil(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = u32::try_from(doy - (153 * mp + 2) / 5 + 1).unwrap_or(1);
    let m = u32::try_from(if mp < 10 { mp + 3 } else { mp - 9 }).unwrap_or(1);
    (if m <= 2 { y + 1 } else { y }, m, d)
}

fn today() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_secs()).unwrap_or(0));
    let (y, m, d) = civil(secs.div_euclid(86_400));
    format!("{y:04}-{m:02}-{d:02}")
}

#[must_use]
pub fn system_prompt(cwd: &Path) -> String {
    // Read-only: a user-level SYSTEM.md replaces the short built-in prompt.
    let base = std::env::var_os("HOME")
        .map(|h| Path::new(&h).join(".omp/agent/SYSTEM.md"))
        .and_then(|p| std::fs::read_to_string(p).ok())
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| BASE_PROMPT.to_string());
    format!(
        "{}\n\nCurrent working directory: {}\nCurrent date: {}",
        base.trim_end(),
        cwd.display(),
        today()
    )
}

/// Build an agent for `cwd` using the selected model.
///
/// # Errors
/// Fails when the model cannot cross the pa-ai/pa-agent boundary.
pub fn build_agent(cwd: &Path, selected: Selected) -> Result<Agent> {
    let agent_model: pa_agent::types::Model = json_round_trip(&selected.model)
        .context("model does not convert to the agent model shape")?;
    Ok(Agent::new(AgentOptions {
        initial_state: AgentInitialState {
            system_prompt: Some(system_prompt(cwd)),
            model: Some(agent_model),
            thinking_level: Some(selected.thinking),
            tools: Some(builtin_tools(cwd)),
            messages: None,
        },
        stream_fn: Some(switchable_stream_fn(Arc::new(RwLock::new(Some(ProviderTarget {
            api_key: selected.api_key,
            model: selected.model,
            service_tier: None,
            headers: selected.headers,
        }))))),
        ..AgentOptions::default()
    }))
}

/// Concatenated text of the last assistant message, plus its error (if any).
#[must_use]
pub fn last_assistant(messages: &[AgentMessage]) -> Option<(String, Option<String>)> {
    use pa_agent::types::{AssistantContent, Message};
    messages.iter().rev().find_map(|m| match m {
        AgentMessage::Standard(Message::Assistant(a)) => {
            let text = a
                .content
                .iter()
                .filter_map(|c| match c {
                    AssistantContent::Text(t) => Some(t.text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n");
            Some((text, a.error_message.clone()))
        }
        _ => None,
    })
}
