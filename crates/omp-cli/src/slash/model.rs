//! Model, preset and fast/slow-mode slash commands.

use super::{json, output, usage, Handled, Outcome, Runtime, Value};

// =============================================================================
// /modelpreset (config/model-presets.ts `runPresetsCommand`)
// =============================================================================

/// `formatModelString` equivalent: `provider/id`.
pub(super) fn model_ref(info: &serde_json::Value) -> String {
    format!(
        "{}/{}",
        info.get("provider").and_then(|v| v.as_str()).unwrap_or("?"),
        info.get("id").and_then(|v| v.as_str()).unwrap_or("?")
    )
}

/// Resolve a `/model <selector>` argument: `provider/id`, or a unique model-id
/// prefix match over the models with configured auth (fuzzy `@role` selectors
/// and `:level` suffixes are TS-selector features; `:level` is honored here).
pub(super) async fn resolve_model(
    _rt: &std::sync::Arc<Runtime>,
    selector: &str,
) -> Option<(String, String, Option<pa_agent::types::ThinkingLevel>)> {
    let (sel, level) = match selector.rsplit_once(':') {
        Some((s, l)) if crate::runtime::parse_level(l).is_some() => {
            (s, crate::runtime::parse_level(l))
        }
        _ => (selector, None),
    };
    if sel.is_empty() {
        return None;
    }
    if let Some((provider, id)) = sel.split_once('/') {
        return Some((provider.to_string(), id.to_string(), level));
    }
    let models = crate::runtime::available_models(None).await;
    let mut hits = models
        .iter()
        .filter(|m| m.id == sel || m.id.starts_with(sel))
        .collect::<Vec<_>>();
    if hits.len() != 1 {
        hits = models
            .iter()
            .filter(|m| format!("{}/{}", m.provider, m.id).contains(sel))
            .collect();
    }
    hits.into_iter()
        .next()
        .map(|m| (m.provider.clone(), m.id.clone(), level))
}

const FAST_USAGE: &str = "Usage: /fast [on|ultra|off|status]";
const SLOW_USAGE: &str = "Usage: /slow [on|off|status]";
const SLOW_UNSUPPORTED: &str = "The current model has no slow mode: /slow uses the flex tier on OpenAI/Google models and low priority on Anthropic subscriptions.";

fn tier_label(tier: Option<pa_types::ai::ServiceTier>) -> &'static str {
    match tier {
        Some(pa_types::ai::ServiceTier::Priority) => "priority tier",
        Some(pa_types::ai::ServiceTier::Flex) => "flex tier",
        Some(pa_types::ai::ServiceTier::Scale) => "scale tier",
        Some(pa_types::ai::ServiceTier::Auto) => "auto tier",
        Some(pa_types::ai::ServiceTier::Default) | None => "default tier",
    }
}

/// `/fast [on|ultra|off|status]` — TS maps fast mode to the provider service
/// tier (priority/ultrafast). omp-rs drives `ProviderTarget.service_tier`,
/// so `on` selects `priority` and `off` restores the default tier.
pub(super) fn fast_command(rt: &Runtime, args: &str) -> Outcome {
    match args.trim().to_lowercase().as_str() {
        "" | "toggle" => {
            let next = rt.service_tier() != Some(pa_types::ai::ServiceTier::Priority);
            rt.set_service_tier(next.then_some(pa_types::ai::ServiceTier::Priority));
            output(
                rt,
                &format!("Fast mode {}.", if next { "enabled" } else { "disabled" }),
            );
            Handled {
                agent_invoked: false,
            }
        }
        "on" => {
            rt.set_service_tier(Some(pa_types::ai::ServiceTier::Priority));
            output(rt, "Fast mode enabled.");
            Handled {
                agent_invoked: false,
            }
        }
        "ultra" | "ultrafast" => usage(
            rt,
            "Ultrafast is unavailable in omt: the ultrafast service tier is not exposed by the provider adapter.",
        ),
        "off" => {
            rt.set_service_tier(None);
            output(rt, "Fast mode disabled.");
            Handled {
                agent_invoked: false,
            }
        }
        "status" => {
            output(
                rt,
                &if rt.service_tier() == Some(pa_types::ai::ServiceTier::Priority) {
                    "Fast mode is on (priority tier).".to_string()
                } else {
                    format!("Fast mode is off ({}).", tier_label(rt.service_tier()))
                },
            );
            Handled {
                agent_invoked: false,
            }
        }
        _ => usage(rt, FAST_USAGE),
    }
}

/// `/slow [on|off|status]` — TS maps slow mode to the flex tier on
/// OpenAI/Google (Anthropic subscription low-priority needs oauth lanes, which
/// omp-rs lacks; see `SLOW_UNSUPPORTED`).
pub(super) fn slow_command(rt: &Runtime, args: &str) -> Outcome {
    let provider = rt
        .model_info()
        .get("provider")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let anthropic = provider == "anthropic";
    match args.trim().to_lowercase().as_str() {
        "status" => {
            let on = rt.service_tier() == Some(pa_types::ai::ServiceTier::Flex);
            output(
                rt,
                &if on {
                    if anthropic {
                        "Slow mode is on (low priority at the Claude session limit).".to_string()
                    } else {
                        "Slow mode is on (flex tier).".to_string()
                    }
                } else {
                    "Slow mode is off.".to_string()
                },
            );
            Handled {
                agent_invoked: false,
            }
        }
        arg @ ("" | "toggle" | "on" | "off") => {
            let enabled = arg == "on"
                || (arg != "off" && rt.service_tier() != Some(pa_types::ai::ServiceTier::Flex));
            if anthropic {
                return usage(rt, SLOW_UNSUPPORTED);
            }
            rt.set_service_tier(enabled.then_some(pa_types::ai::ServiceTier::Flex));
            output(
                rt,
                if enabled {
                    "Slow mode on: requests use the flex tier (lower cost, higher latency)."
                } else {
                    "Slow mode off."
                },
            );
            Handled {
                agent_invoked: false,
            }
        }
        _ => usage(rt, SLOW_USAGE),
    }
}

// =============================================================================
// /todo (helpers/todo.ts `handleTodoAcp`)
// =============================================================================

fn preset_name_valid(name: &str) -> bool {
    let mut chars = name.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_alphabetic())
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

/// The merged `modelPresets` record (TS reads the highest layer owning each
/// name; `write_global` updates the global file in place, preserving keys).
fn read_model_presets(settings: &omp_settings::Settings) -> serde_json::Map<String, Value> {
    settings
        .get("modelPresets")
        .as_object()
        .cloned()
        .unwrap_or_default()
}

const PRESETS_USAGE: &str =
    "Usage: /modelpreset [list | save <name> | switch <name> | delete <name>]";
const NO_PRESETS_MESSAGE: &str =
    "No model presets saved. Use /modelpreset save <name> to create one.";

#[allow(clippy::too_many_lines)] // preset verbs share one settings snapshot; splitting spreads the writes
pub(super) async fn modelpreset_command(rt: &Runtime, args: &str) -> Outcome {
    let settings = omp_settings::Settings::load(&rt.cwd);
    // TS `handleTui`: bare `/modelpreset` picks a preset to switch to.
    let picked;
    let args = if args.trim().is_empty() && crate::slash_extra::dialog::has_ui() {
        let names: Vec<String> = settings.model_presets().keys().cloned().collect();
        if names.is_empty() {
            output(rt, NO_PRESETS_MESSAGE);
            return Handled {
                agent_invoked: false,
            };
        }
        let Some(index) =
            crate::slash_extra::dialog::select(rt, "Switch to model preset", &names).await
        else {
            return Handled {
                agent_invoked: false,
            };
        };
        picked = format!("switch {}", names[index]);
        picked.as_str()
    } else {
        args
    };
    let tokens: Vec<&str> = args.split_whitespace().collect();
    let sub = tokens.first().copied().unwrap_or("list");
    let name = tokens.get(1..).unwrap_or_default().join(" ");
    match sub {
        "list" => {
            let presets = settings.model_presets();
            output(
                rt,
                &if presets.is_empty() {
                    NO_PRESETS_MESSAGE.to_string()
                } else {
                    format!(
                        "Model presets: {}",
                        presets.keys().cloned().collect::<Vec<_>>().join(", ")
                    )
                },
            );
            Handled {
                agent_invoked: false,
            }
        }
        "save" => {
            if name.is_empty() {
                return usage(rt, "Usage: /modelpreset save <name>");
            }
            if !preset_name_valid(&name) {
                return usage(
                    rt,
                    &format!(
                        "Invalid preset name \"{name}\": use a letter, then letters, digits, - or _"
                    ),
                );
            }
            let roles = settings.get("modelRoles");
            let level = settings.get("defaultThinkingLevel");
            let mut presets = read_model_presets(&settings);
            presets.insert(
                name.clone(),
                json!({"modelRoles": roles, "defaultThinkingLevel": level}),
            );
            match omp_settings::write_global("modelPresets", &Value::Object(presets)) {
                Ok(path) => {
                    output(
                        rt,
                        &format!("Saved model preset \"{name}\" to {}.", path.display()),
                    );
                    Handled {
                        agent_invoked: false,
                    }
                }
                Err(e) => usage(rt, &format!("Failed to save preset: {e}")),
            }
        }
        "switch" => {
            if name.is_empty() {
                return usage(rt, "Usage: /modelpreset switch <name>");
            }
            let presets = settings.model_presets();
            let Some(preset) = presets.get(&name) else {
                return usage(rt, &format!("Preset not found: {name}"));
            };
            // TS applies role assignments hub-style and the default thinking
            // level; omp-rs runs a single active model, so the roles are
            // persisted and the thinking level applies now.
            let level = preset
                .default_thinking_level
                .as_deref()
                .and_then(crate::runtime::parse_level);
            if let Some(level) = level {
                rt.set_thinking_level(level).await;
            }
            output(
                rt,
                &format!(
                    "Switched to model preset \"{name}\".{}",
                    if level.is_some() {
                        ""
                    } else {
                        " (role model assignments are stored for multi-role hosts; this runtime drives one active model)"
                    }
                ),
            );
            Handled {
                agent_invoked: false,
            }
        }
        "delete" => {
            if name.is_empty() {
                return usage(rt, "Usage: /modelpreset delete <name>");
            }
            let mut presets = read_model_presets(&settings);
            if presets.remove(&name).is_none() {
                return usage(rt, &format!("Preset not found: {name}"));
            }
            match omp_settings::write_global("modelPresets", &Value::Object(presets)) {
                Ok(_) => {
                    output(rt, &format!("Deleted model preset \"{name}\""));
                    Handled {
                        agent_invoked: false,
                    }
                }
                Err(e) => usage(rt, &format!("Failed to delete preset: {e}")),
            }
        }
        _ => usage(rt, PRESETS_USAGE),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preset_name_validation_matches_ts() {
        assert!(preset_name_valid("fast-1_x"));
        assert!(!preset_name_valid("1fast"));
        assert!(!preset_name_valid("has space"));
        assert!(!preset_name_valid(""));
    }
}
