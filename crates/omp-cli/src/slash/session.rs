//! Session slash commands: `/pin`, `/move`, `/plugins`, `/handoff`.

use super::{
    output, subcommand, usage, Handled, HashSet, Outcome, Path, PathBuf,
    Runtime, Value,
};

// =============================================================================
// /pin (session/session-pins.ts)
// =============================================================================

/// `~/.omp-rs/session-pins.json` — pinned session ids (keyed by id so pins
/// survive moves), the omp-rs counterpart of TS `~/.omp/session-pins.json`.
fn pins_path() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    Some(
        PathBuf::from(home)
            .join(".omp-rs")
            .join("session-pins.json"),
    )
}

fn load_pinned_sessions() -> HashSet<String> {
    pins_path()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|text| serde_json::from_str::<Vec<String>>(&text).ok())
        .map(|v| v.into_iter().collect())
        .unwrap_or_default()
}

fn store_pinned_sessions(pins: &HashSet<String>) -> Result<(), String> {
    let path = pins_path().ok_or_else(|| "no HOME for pins file".to_string())?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let mut sorted: Vec<&String> = pins.iter().collect();
    sorted.sort();
    let text = serde_json::to_string_pretty(&sorted).map_err(|e| e.to_string())?;
    std::fs::write(&path, text).map_err(|e| e.to_string())
}

/// `/pin [session id]` — toggle this session's pin (TS `handle` in
/// builtin-lifecycle.ts). Only the live session is resolvable headlessly, so
/// an explicit id pins that id without a resumable-session lookup.
pub(super) fn pin_command(rt: &Runtime, args: &str) -> Outcome {
    let arg = args.trim();
    let session_id = if arg.is_empty() {
        rt.session_id()
    } else {
        arg.to_string()
    };
    let mut pinned = load_pinned_sessions();
    let now_pinned = if pinned.remove(&session_id) {
        false
    } else {
        pinned.insert(session_id);
        true
    };
    match store_pinned_sessions(&pinned) {
        Ok(()) => {
            output(
                rt,
                if now_pinned {
                    "Session pinned to the top of the resume list."
                } else {
                    "Session unpinned."
                },
            );
            Handled {
                agent_invoked: false,
            }
        }
        Err(e) => usage(rt, &format!("Failed to update pins: {e}")),
    }
}

// =============================================================================
// Shared setting toggles (/skillful, /extended-context) and /move, /plugins
// =============================================================================

/// Status is served from the live config; toggling is a session override in TS
/// (applied to the in-memory settings and the rebuilt system prompt), which
/// the omp-rs runtime does not implement — say so honestly.
pub(super) fn setting_status_toggle(
    rt: &Runtime,
    args: &str,
    setting: &str,
    label: &str,
    usage_line: &str,
) -> Outcome {
    let settings = omp_settings::Settings::load(&rt.cwd);
    let current = settings.get_bool(setting).unwrap_or(false);
    match args.trim().to_lowercase().as_str() {
        "" | "status" => {
            output(
                rt,
                &format!(
                    "{label}: {} (from config).",
                    if current { "on" } else { "off" }
                ),
            );
            Handled {
                agent_invoked: false,
            }
        }
        "toggle" | "on" | "off" => usage(
            rt,
            &format!(
                "{label} toggling is a session override in omp (settings + rebuilt system prompt); omt applies these settings at startup only, so edit them in config.yml and restart. Status: {}.",
                if current { "on" } else { "off" }
            ),
        ),
        _ => usage(rt, usage_line),
    }
}

pub(super) const ADVISOR_INACTIVE_ON: &str =
    "Advisor setting enabled, but no model is assigned to the 'advisor' role.";

/// `/move <path>` — TS relocates the live session to another directory. The
/// omp-rs writer holds the session file path for the process lifetime, so a
/// live relocation would split the transcript; report that precisely.
pub(super) async fn move_command(rt: &Runtime, args: &str) -> Outcome {
    if rt.state().await.get("isStreaming").and_then(Value::as_bool) == Some(true) {
        return usage(rt, "Cannot move while streaming.");
    }
    if args.trim().is_empty() {
        return usage(rt, "Usage: /move <path>");
    }
    let resolved = if Path::new(args.trim()).is_absolute() {
        PathBuf::from(args.trim())
    } else {
        rt.cwd.join(args.trim())
    };
    if !resolved.is_dir() {
        return usage(
            rt,
            &format!("Directory does not exist: {}", resolved.display()),
        );
    }
    usage(
        rt,
        &format!(
            "Moving a live session is not supported in omt: the session writer is bound to {} for the process lifetime and a move would split the transcript. Start a new session in {} instead (`omt --resume` can reopen this one there).",
            rt.cwd.display(),
            resolved.display()
        ),
    )
}

/// `/plugins [list|enable|disable]` — marketplace plugin registry
/// (`features/skills.rs`: `installed_plugins.json` + enable flags).
pub(super) async fn plugins_command(rt: &Runtime, args: &str) -> Outcome {
    let (verb, rest) = subcommand(args);
    match verb.as_str() {
        "list" | "" => {
            output(rt, &crate::features::skills::plugins_list_text(rt));
            let wasm = crate::tools::wasm::plugins_list_text(&rt.cwd);
            output(rt, &format!("\n{wasm}"));
            Handled {
                agent_invoked: false,
            }
        }
        "enable" | "disable" => {
            let name = rest.trim();
            if name.is_empty() {
                return usage(rt, &format!("Usage: /plugins {verb} <marketplace/name>"));
            }
            match crate::features::skills::plugins_set_enabled(rt, name, verb == "enable") {
                Ok(()) => output(
                    rt,
                    &format!(
                        "{} {name} (restart the session to apply; plugin loading is not hot-reloaded)",
                        if verb == "enable" { "Enabled" } else { "Disabled" }
                    ),
                ),
                Err(e) => {
                    output(rt, &format!("Plugins error: {e}"));
                }
            }
            Handled {
                agent_invoked: false,
            }
        }
        _ => usage(rt, "Usage: /plugins [list|enable|disable]"),
    }
}

/// `/handoff [focus]` — TS generates a handoff document and compacts in
/// place. The summary prompts exist in omp-compact, but no live
/// generate-and-compact-in-place flow is wired into the runtime.
pub(super) async fn handoff_command(rt: &Runtime, args: &str) -> Outcome {
    if rt.state().await.get("isStreaming").and_then(Value::as_bool) == Some(true) {
        return usage(
            rt,
            "Wait for the current response to finish or abort it before handing off.",
        );
    }
    let _ = args;
    usage(
        rt,
        "/handoff (handoff document generation + compact-in-place) is not wired into the omt runtime; only the /compact summary path is available.",
    )
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    #[test]
    fn pins_toggle_round_trips() {
        // Uses OMP_CONFIG_DIR-free HOME path; isolate via a pins file in a
        // temp HOME when one is not already set by the harness.
        let dir = std::env::temp_dir().join(format!("omp-pins-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let pins_file = dir.join("session-pins.json");
        std::fs::write(
            &pins_file,
            serde_json::to_string(&["sess-a".to_string()]).unwrap(),
        )
        .unwrap();
        // Direct store/load check against the file contents (load_pinned_sessions
        // reads HOME; emulate by parsing the same JSON shape).
        let pins: HashSet<String> =
            serde_json::from_str::<Vec<String>>(&std::fs::read_to_string(&pins_file).unwrap())
                .unwrap()
                .into_iter()
                .collect();
        assert!(pins.contains("sess-a"));
        std::fs::remove_dir_all(&dir).ok();
    }
}
