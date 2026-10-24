//! Integration test: parse every real user agent definition under
//! `~/.omp/agent/agents/*.md` (read-only). No writes, no subprocesses.

use omp_task::agent::{AgentSource, ThinkingLevel};
use omp_task::discovery::discover_user_agents;

/// The known user agents in the fleet workspace as of 2026-10-12. If the
/// user adds agents this list should be extended, not shrunk.
const KNOWN_AGENTS: &[&str] = &[
    "asymmetric-reviewer",
    "atria-worker",
    "gemini-worker",
    "glm-flash-coder",
    "herdr-ops",
    "kimi-coder",
    "laguna-worker",
    "mercury",
    "pixel-canary",
    "scout-codex",
    "scout-glm",
    "scout-grok",
    "scout-kimi",
    "sonnet-coder",
    "space-bunny-free",
    "space-bunny-high",
];

#[test]
fn parses_all_real_user_agent_defs() {
    let home = std::path::PathBuf::from(std::env::var("HOME").expect("HOME must be set"));
    let agents_dir = home.join(".omp/agent/agents");
    if !agents_dir.is_dir() {
        eprintln!("skipping: {} not present", agents_dir.display());
        return;
    }

    let result = discover_user_agents(&home);
    eprintln!(
        "parsed {} agents, {} failures",
        result.agent_count(),
        result.failure_count()
    );
    for failure in &result.failures {
        eprintln!(
            "FAILURE {}: {}",
            failure.file_path.display(),
            failure.message
        );
    }

    assert_eq!(result.failure_count(), 0, "all real agent defs must parse");
    assert!(
        result.agent_count() >= KNOWN_AGENTS.len(),
        "expected at least {} agents, got {}",
        KNOWN_AGENTS.len(),
        result.agent_count()
    );

    // Every known agent resolves with the invariants the TS contract enforces.
    for name in KNOWN_AGENTS {
        let def = result
            .get_agent(name)
            .unwrap_or_else(|| panic!("agent {name} must be discoverable"));
        assert_eq!(def.source, AgentSource::User);
        assert!(
            !def.fields.description.is_empty(),
            "{name}: description required"
        );
        assert!(
            !def.system_prompt.trim().is_empty(),
            "{name}: system prompt body required"
        );
        if let Some(tools) = &def.fields.tools {
            assert!(
                tools.contains(&"yield".to_string()),
                "{name}: explicit tool lists always include yield (got {tools:?})"
            );
            let lower: Vec<&str> = tools.iter().map(String::as_str).collect();
            let mut sorted = lower.clone();
            sorted.sort_unstable();
            let mut dedup = sorted.clone();
            dedup.dedup();
            assert_eq!(
                sorted, dedup,
                "{name}: normalized tool names are dedup-able"
            );
        }
        // file_path points at the real file.
        assert!(
            def.file_path.is_file(),
            "{}: {}",
            name,
            def.file_path.display()
        );
    }
}

#[test]
fn known_thinking_levels_and_models_parse() {
    let home = std::path::PathBuf::from(std::env::var("HOME").expect("HOME must be set"));
    let result = discover_user_agents(&home);
    if result.agents.is_empty() {
        eprintln!("skipping: no user agents");
        return;
    }
    for def in &result.agents {
        if let Some(level) = &def.fields.thinking_level {
            assert!(
                matches!(
                    level,
                    ThinkingLevel::Low
                        | ThinkingLevel::Medium
                        | ThinkingLevel::High
                        | ThinkingLevel::Max
                        | ThinkingLevel::Other(_)
                ),
                "{}: unexpected thinking level {level:?}",
                def.fields.name
            );
            assert!(!level.effort().is_empty());
        }
        if let Some(models) = &def.fields.model {
            assert!(!models.is_empty(), "{}: empty model list", def.fields.name);
        }
    }
    // Spot-check specific fleet agents observed in the TS-format files.
    let glm = result
        .get_agent("glm-flash-coder")
        .expect("glm-flash-coder");
    assert_eq!(glm.fields.thinking_level, Some(ThinkingLevel::Low));
    assert_eq!(
        glm.fields.model.as_deref(),
        Some(&["zai/glm-5.3-flash".to_string()][..])
    );

    let atria = result.get_agent("atria-worker").expect("atria-worker");
    assert_eq!(atria.fields.thinking_level, Some(ThinkingLevel::Medium));
    assert_eq!(
        atria.fields.model.as_deref(),
        Some(&["atria/Atria-Dawn-Preview".to_string()][..])
    );
}
