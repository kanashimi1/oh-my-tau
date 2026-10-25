//! Agent assembly: system prompt, tools, and the pa-ai stream adapter wired
//! into pa-agent's `Agent` (pattern derived from prime-agent's pa-cli, MIT).

use std::path::Path;

use anyhow::{Context, Result};
use pa_agent::agent::{Agent, AgentInitialState, AgentOptions};
use pa_agent::agent_loop::ConvertToLlmFn;
use pa_agent::types::{AgentMessage, AgentTool, DeveloperMessage, Message, UserContent, UserPart};

use std::sync::{Arc, RwLock};

use crate::adapter::{json_round_trip, switchable_stream_fn, ProviderTarget};
use crate::model::Selected;
use crate::tools::edit_mode_for;

/// TS `convertMessageToLlm` for `custom` / `hookMessage` / `developer` rows: a provider-facing
/// `developer` message. A custom message's string content becomes one text part; a developer
/// message keeps its content shape; attribution defaults to the message's own (`agent` for
/// developer rows).
fn developer_message(custom: &pa_agent::types::CustomAgentMessage) -> Option<DeveloperMessage> {
    let is_custom = custom.role == "custom";
    let content = match &custom.payload["content"] {
        serde_json::Value::String(text) if is_custom => {
            UserContent::Parts(vec![UserPart::Text(pa_agent::types::TextContent {
                text: text.clone(),
                text_signature: None,
            })])
        }
        serde_json::Value::String(text) => UserContent::Text(text.clone()),
        serde_json::Value::Array(parts) => UserContent::Parts(
            parts
                .iter()
                .filter_map(|p| serde_json::from_value::<UserPart>(p.clone()).ok())
                .collect(),
        ),
        _ => return None,
    };
    let attribution = custom.payload["attribution"]
        .as_str()
        .map(str::to_owned)
        .or_else(|| (!is_custom).then(|| "agent".to_owned()));
    Some(DeveloperMessage {
        content,
        timestamp: custom.payload["timestamp"].as_i64().unwrap_or_default(),
        attribution,
    })
}

/// Custom-role messages extension hooks inject: omp's `developer`
/// (`additionalContext`) reaches the model as user text; other custom roles
/// are dropped, matching `pa-agent`'s default converter (kept explicit here
/// because the CLI pins its own converter when extensions are on).
#[must_use]
pub fn convert_to_llm() -> ConvertToLlmFn {
    Arc::new(|messages: Vec<AgentMessage>| {
        Box::pin(async move {
            Ok(messages
                .into_iter()
                .filter_map(|m| match m {
                    AgentMessage::Standard(message) => Some(message),
                    AgentMessage::Custom(custom)
                        if custom.role == "developer" || custom.role == "custom" =>
                    {
                        developer_message(&custom).map(Message::Developer)
                    }
                    AgentMessage::Custom(_) => None,
                })
                .collect())
        })
    })
}

/// Shared slot the provider stream reads its target from; `set_model` swaps it.
pub type Target = Arc<RwLock<Option<ProviderTarget>>>;

/// Days since 1970-01-01 to (year, month, day), civil calendar.
pub fn civil(days: i64) -> (i64, u32, u32) {
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

/// Assemble the provider-facing system prompt: the ported omp prompt
/// (SYSTEM.md override / flags / context files / skills / tools / env block)
/// as one concatenated string of `\n\n`-joined blocks.
#[must_use]
pub fn system_prompt(
    cwd: &Path,
    opts: &crate::sysprompt::Options,
    active_hidden: &[Arc<dyn AgentTool>],
) -> String {
    // Edit mode does not affect the prompt; the tool surface (names) does.
    let settings = omp_settings::Settings::load(cwd);
    let mut metas: Vec<crate::sysprompt::ToolMeta> =
        crate::tools::session_builtin_tools(cwd, crate::tools::edit_mode_for(""), &settings)
            .iter()
            .map(|t| (t.name().to_string(), t.name().to_string()))
            .collect();
    // Active hidden tools join the prompt's tool list/refs (TS lists active
    // tools; `{{toolRefs.think}}` resolves only while the tool is active).
    metas.extend(
        active_hidden
            .iter()
            .map(|t| (t.name().to_string(), t.name().to_string())),
    );
    crate::sysprompt::build(cwd, opts, &metas)
        .blocks
        .join("\n\n")
}

/// One-line pointer to the deferred tools (`tools.profile: core`): names
/// only; schemas come from `tool.describe` on demand.
#[must_use]
pub fn deferred_tools_note(deferred: &[String]) -> String {
    format!(
        "Deferred tools (not in the tool list; call from an eval cell: `await tool.<name>(args)`, schema via `tool.describe({{name}})`, all via `tool.list()`): {}.",
        deferred.join(", ")
    )
}

/// Build an agent for `cwd` using the selected model, plus the provider
/// target slot a live `set_model` swaps. `extra_tools` (extension tools) are
/// appended to the built-ins; `ext_hooks`, when present, wires the extension
/// host into context transformation and tool call/result interception.
///
/// Returns the agent, the target and the *deferred* tools: under
/// `tools.profile: core` every enabled tool outside [`crate::tools::CORE_TOOLS`]
/// (built-ins, MCP, extension tools) stays off the model's tool list and is
/// reachable only through the eval bridge; the caller registers them there.
/// An explicit `--tools <csv>` list wins over the profile (nothing deferred).
///
/// # Errors
/// Fails when the model cannot cross the pa-ai/pa-agent boundary.
pub fn build_agent(
    cwd: &Path,
    selected: Selected,
    extra_tools: Vec<Arc<dyn AgentTool>>,
    ext_hooks: Option<&Arc<crate::ext::Extensions>>,
    sysprompt_opts: &crate::sysprompt::Options,
    context_strategy: crate::context::Kind,
) -> Result<(Agent, Target)> {
    let agent_model: pa_agent::types::Model = json_round_trip(&selected.model)
        .context("model does not convert to the agent model shape")?;
    let settings = omp_settings::Settings::load(cwd);
    let mut tools =
        crate::tools::session_builtin_tools(cwd, edit_mode_for(&selected.model.id), &settings);
    let session_jobs = crate::tools::take_last_jobs();
    tools.extend(crate::tools::mcp::mcp_tools(cwd));
    tools.extend(crate::tools::wasm::wasm_tools(cwd));
    // Hidden tools (TS `HIDDEN_TOOLS`): the active legs (think) are appended
    // after the built-ins; a `--tools <csv>` list admits hidden names too
    // (subagent runs carry `yield` in their csv), like TS's requestedTools
    // filter over BUILTIN_TOOLS + HIDDEN_TOOLS.
    let hidden = crate::tools::hidden_tools(&settings, Some(&selected.model), false, false);
    if let Some(csv) = sysprompt_opts.tools_csv.as_deref() {
        let keep: Vec<String> = csv
            .split(',')
            .map(str::trim)
            .filter(|n| !n.is_empty())
            .map(ToString::to_string)
            .collect();
        tools.retain(|t| keep.iter().any(|k| k == t.name()));
        for name in &keep {
            if hidden.iter().all(|t| t.name() != name) {
                if let Some(t) = crate::tools::hidden_tool_by_name(name) {
                    tools.push(t);
                }
            }
        }
    }
    let replacement_names: std::collections::HashSet<String> = extra_tools
        .iter()
        .map(|tool| tool.name().to_string())
        .collect();
    tools.retain(|tool| !replacement_names.contains(tool.name()));
    tools.extend(extra_tools);
    let mut prompt = system_prompt(cwd, sysprompt_opts, &hidden);
    crate::tools::set_session_jobs(session_jobs);
    tools.extend(hidden);
    let core_profile =
        sysprompt_opts.tools_csv.is_none() && crate::methodology::core_profile_requested();
    if core_profile {
        let deferred: Vec<String> = tools
            .iter()
            .filter(|tool| !crate::methodology::is_core_tool(tool.name()))
            .map(|tool| tool.name().to_string())
            .collect();
        if !deferred.is_empty() {
            prompt.push_str("\n\n");
            prompt.push_str(&deferred_tools_note(&deferred));
        }
    }
    let ctx = (context_strategy != crate::context::Kind::Append).then(|| {
        let policy = crate::context_policy::Policy::resolve(&settings, &|k| std::env::var(k).ok())
            .unwrap_or_else(|e| {
                tracing::warn!("{e}; context policy disabled");
                None
            });
        crate::context::Shared::with_policy(
            context_strategy,
            crate::context::Limits::from_settings(&settings),
            policy,
        )
    });
    if let Some(ctx) = &ctx {
        tools.push(crate::context::tool(ctx.clone()));
    }
    // TS `ExtensionToolWrapper`: every tool passes the approval gate before it executes.
    let tools = match crate::approval::Gate::get() {
        Some(gate) => gate.wrap_tools(tools),
        None => tools,
    };
    let target: Target = Arc::new(RwLock::new(Some(ProviderTarget {
        api_key: selected.api_key,
        model: selected.model,
        service_tier: None,
        headers: selected.headers,
        credential_id: selected.credential_id,
    })));
    let (transform_context, before_tool_call, after_tool_call) = match ext_hooks.as_ref() {
        Some(e) => {
            let (tc, btc, atc) = e.hook_fns();
            (Some(tc), Some(btc), Some(atc))
        }
        None => (None, None, None),
    };
    // Eval cells' `tool.<name>`/`read`/`write` pass the same tool-call policy.
    crate::tools::eval_bridge::set_session_hooks(before_tool_call.clone(), after_tool_call.clone());
    let agent = Agent::new(AgentOptions {
        initial_state: AgentInitialState {
            system_prompt: Some(prompt),
            model: Some(agent_model),
            thinking_level: Some(selected.thinking),
            tools: Some(tools),
            messages: None,
        },
        convert_to_llm: Some({
            // The context strategy is read per request: an ultra mode can
            // switch `clm-host` on and off mid-session (`context::set_live`).
            // With none live (`append`) the messages pass through untouched.
            crate::context::set_live(ctx, false);
            let base = convert_to_llm();
            Arc::new(move |messages: Vec<AgentMessage>| {
                let base = Arc::clone(&base);
                Box::pin(async move {
                    let messages = match crate::context::live() {
                        Some(c) => c.wire(messages),
                        None => messages,
                    };
                    // Wire dedup: exact-duplicate tool results collapse to
                    // pointer stubs. Default-on in non-interactive sessions
                    // (rpc/print): the loop re-carries every past tool result
                    // through every request, and measured ~65% of billed input
                    // bytes were re-occurrences (COST-CACHE-HYGIENE-20261014).
                    // Env override: OMP_RS_DEDUP_TOOL_RESULTS=0 disables.
                    let messages = if crate::dedup::dedup_active() {
                        crate::dedup::dedup_tool_results(messages)
                    } else {
                        messages
                    };
                    base(messages).await
                })
            })
        }),
        stream_fn: Some(crate::methodology::wrap_stream(
            switchable_stream_fn(target.clone()),
            core_profile,
            ext_hooks.cloned(),
        )),
        transform_context,
        before_tool_call: Some(crate::features::plan::gate(
            crate::tools::relevance::wrap_before(before_tool_call),
        )),
        after_tool_call,
        ..AgentOptions::default()
    });
    Ok((agent, target))
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
