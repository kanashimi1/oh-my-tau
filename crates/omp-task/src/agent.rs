//! Agent-definition parsing, ported from `task/agents.ts` and
//! `discovery/helpers.ts::parseAgentFields` in the TS source.
//!
//! Agent files are markdown with YAML frontmatter:
//!
//! ```markdown
//! ---
//! name: my-agent
//! description: "Does a thing"
//! tools:
//!   - read
//!   - grep
//! model:
//!   - "zai/glm-5.3-flash"
//! thinkingLevel: high
//! ---
//!
//! System prompt body...
//! ```

use serde_json::Value;
use thiserror::Error;

/// Sentinel `agentName` values reserved for the top-level session ("main")
/// and the unnamed subagent session ("sub"). A custom agent definition
/// sharing either name would hijack rules scoped to those session kinds, so
/// `parseAgentFields` rejects them (see `MAIN_AGENT_RULE_NAME` /
/// `SUB_AGENT_RULE_NAME` in the TS source).
const RESERVED_AGENT_NAMES: [&str; 2] = ["main", "sub"];

/// Coarse thinking effort levels accepted by the TS `parseConfiguredThinkingLevel`.
///
/// TS stores the raw configured string; the per-spawn effort rule maps to
/// `"lo" | "med" | "hi"`. We keep the configured value verbatim and validate
/// it against the known set.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ThinkingLevel {
    Low,
    Medium,
    High,
    Max,
    /// Any other configured string, preserved verbatim.
    Other(String),
}

impl ThinkingLevel {
    /// Parse from the frontmatter `thinkingLevel` / `thinking` string.
    #[must_use]
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_lowercase().as_str() {
            "low" | "lo" => Some(Self::Low),
            "medium" | "med" => Some(Self::Medium),
            "high" | "hi" => Some(Self::High),
            "max" => Some(Self::Max),
            other if !other.is_empty() => Some(Self::Other(other.to_string())),
            _ => None,
        }
    }

    /// Coarse effort bucket used by the task tool's per-spawn effort rule.
    #[must_use]
    pub fn effort(&self) -> &'static str {
        match self {
            Self::Low => "lo",
            Self::High | Self::Max => "hi",
            Self::Medium | Self::Other(_) => "med",
        }
    }
}

/// Agent fields parsed from frontmatter (everything except body/source/path).
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct ParsedAgentFields {
    pub name: String,
    pub description: String,
    /// Tool allow-list; `Some(Vec::new())` means an explicitly empty list.
    /// Subagents with an explicit tool list always get `yield` appended
    /// (they must be able to return results).
    pub tools: Option<Vec<String>>,
    /// Agents this agent may spawn: `*` = unrestricted.
    pub spawns: Option<Spawns>,
    /// Prioritized model list (string or array in frontmatter).
    pub model: Option<Vec<String>>,
    /// Declared structured-output schema (`output:` frontmatter field), kept
    /// verbatim as JSON.
    pub output: Option<Value>,
    pub thinking_level: Option<ThinkingLevel>,
    pub blocking: Option<bool>,
    pub autoload_skills: Option<Vec<String>>,
    pub read_summarize: Option<bool>,
    /// `true` = prewalk into the default target; string = that model pattern.
    pub prewalk: Option<PrewalkAdvisor>,
    /// `true` = advise with the default advisor-role model; string = custom.
    pub advisor: Option<PrewalkAdvisor>,
    /// `listed: false` hides the agent from the `task` tool's summary list
    /// (it is only named in a compact `Also:` line); it stays spawnable.
    /// Defaults to `true`.
    pub listed: bool,
}

/// Spawn-permission field: unrestricted (`*`) or an explicit agent list.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub enum Spawns {
    All,
    Agents(Vec<String>),
}

/// `prewalk`/`advisor` boolean-or-string frontmatter fields.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub enum PrewalkAdvisor {
    Enabled,
    Target(String),
}

/// A parsed agent definition: frontmatter fields plus the markdown body as
/// the system prompt and provenance.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct AgentDefinition {
    #[serde(flatten)]
    pub fields: ParsedAgentFields,
    /// Markdown body below the frontmatter; becomes the system prompt.
    pub system_prompt: String,
    /// Where the definition came from (user/project/extension/bundled).
    pub source: AgentSource,
    /// Absolute or discovered path of the source file.
    pub file_path: std::path::PathBuf,
}

/// Provenance of an agent definition, mirroring the TS `AgentSource`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AgentSource {
    User,
    Project,
    Extension,
    Bundled,
}

/// Failure to parse one agent definition.
#[derive(Debug, Error)]
#[error("failed to parse agent {file_path}: {message}")]
pub struct AgentParseError {
    pub file_path: std::path::PathBuf,
    pub message: String,
}

/// Split markdown into YAML frontmatter and body.
///
/// Mirrors `parseFrontmatter` in `pi-utils`: the file must start with a
/// `---` line; the frontmatter runs to the next `---` line; the rest is the
/// body (with leading blank lines trimmed).
fn split_frontmatter(content: &str) -> Result<(Value, String), String> {
    let mut lines = content.lines();
    if lines.next().map(str::trim) != Some("---") {
        return Err("missing opening '---' frontmatter delimiter".to_string());
    }
    let mut yaml_lines: Vec<&str> = Vec::new();
    let mut closed = false;
    for line in lines.by_ref() {
        if line.trim() == "---" {
            closed = true;
            break;
        }
        yaml_lines.push(line);
    }
    if !closed {
        return Err("unterminated frontmatter: no closing '---'".to_string());
    }
    let yaml_src = yaml_lines.join("\n");
    let frontmatter: Value =
        serde_yaml_ng::from_str(&yaml_src).map_err(|e| format!("invalid YAML frontmatter: {e}"))?;
    let mut body = lines.collect::<Vec<_>>().join("\n");
    while body.starts_with('\n') {
        body.remove(0);
    }
    Ok((frontmatter, body))
}

/// Parse an array-or-CSV frontmatter value into a trimmed, non-empty list.
/// Mirrors `parseArrayOrCSV`.
fn parse_array_or_csv(value: Option<&Value>) -> Option<Vec<String>> {
    let value = value?;
    match value {
        Value::Array(items) => {
            let out: Vec<String> = items
                .iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect();
            if out.is_empty() {
                None
            } else {
                Some(out)
            }
        }
        Value::String(s) => {
            let out: Vec<String> = s
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect();
            if out.is_empty() {
                None
            } else {
                Some(out)
            }
        }
        _ => None,
    }
}

/// Normalize tool names (trim + lowercase), mirroring `normalizeToolNames`.
fn normalize_tool_names(tools: &[String]) -> Vec<String> {
    tools.iter().map(|t| t.trim().to_lowercase()).collect()
}

fn parse_bool(value: Option<&Value>) -> Option<bool> {
    value?.as_bool()
}

/// Parse agent fields from frontmatter. Returns `None` if required fields
/// (`name`, `description`) are missing or the name is a reserved sentinel.
/// Mirrors `discovery/helpers.ts::parseAgentFields`.
fn parse_agent_fields(frontmatter: &Value) -> Option<ParsedAgentFields> {
    let obj = frontmatter.as_object()?;
    let name = obj.get("name")?.as_str()?.to_string();
    let description = obj.get("description")?.as_str()?.to_string();

    let normalized_name = name.trim().to_lowercase();
    if RESERVED_AGENT_NAMES.contains(&normalized_name.as_str()) {
        return None;
    }

    let mut tools = match obj.get("tools") {
        Some(Value::Array(items)) if items.is_empty() => Some(Vec::new()),
        other => parse_array_or_csv(other).map(|t| normalize_tool_names(&t)),
    };
    if let Some(t) = tools.as_mut() {
        // Subagents with explicit tool lists always need yield.
        if !t.iter().any(|tool| tool == "yield") {
            t.push("yield".to_string());
        }
    }

    // spawns: array, "*", or CSV.
    let mut spawns: Option<Spawns> = match obj.get("spawns") {
        Some(Value::String(s)) if s.trim() == "*" => Some(Spawns::All),
        Some(Value::String(_) | Value::Array(_)) | None => {
            parse_array_or_csv(obj.get("spawns")).map(Spawns::Agents)
        }
        Some(_) => None,
    };
    if spawns.is_none()
        && tools
            .as_ref()
            .is_some_and(|t| t.contains(&"task".to_string()))
    {
        // Backward compat: tools includes "task" implies unrestricted spawns.
        spawns = Some(Spawns::All);
    }

    let raw_thinking = obj
        .get("thinkingLevel")
        .and_then(Value::as_str)
        .or_else(|| obj.get("thinking").and_then(Value::as_str));
    let thinking_level = raw_thinking.and_then(ThinkingLevel::parse);

    let model = parse_array_or_csv(obj.get("model")).map(|mut m| {
        m.retain(|s| !s.trim().is_empty());
        m
    });

    // `prewalk: false` disables the feature; only `true` (or a non-empty
    // target string) yields a value. Same for `advisor`.
    let prewalk = match obj.get("prewalk") {
        Some(Value::Bool(true)) => Some(PrewalkAdvisor::Enabled),
        Some(Value::String(s)) if !s.trim().is_empty() => {
            Some(PrewalkAdvisor::Target(s.trim().to_string()))
        }
        _ => None,
    };
    let advisor = match obj.get("advisor") {
        Some(Value::Bool(true)) => Some(PrewalkAdvisor::Enabled),
        Some(Value::String(s)) if !s.trim().is_empty() => {
            Some(PrewalkAdvisor::Target(s.trim().to_string()))
        }
        _ => None,
    };

    Some(ParsedAgentFields {
        name,
        description,
        tools,
        spawns,
        model,
        output: obj.get("output").cloned(),
        thinking_level,
        blocking: parse_bool(obj.get("blocking")),
        autoload_skills: parse_array_or_csv(obj.get("autoloadSkills")).map(|mut s| {
            s.retain(|x| !x.trim().is_empty());
            s
        }),
        read_summarize: parse_bool(obj.get("readSummarize")),
        prewalk,
        listed: parse_bool(obj.get("listed")).unwrap_or(true),
        advisor,
    })
}

/// Parse an agent from file content, mirroring `task/agents.ts::parseAgent`.
///
/// `level` is accepted for signature parity with the TS source; malformed
/// frontmatter here always surfaces as an [`AgentParseError`] and the caller
/// (discovery) decides whether that is fatal.
///
/// # Errors
/// Returns an [`AgentParseError`] when the frontmatter is missing, has
/// invalid YAML, or lacks the required `name`/`description` fields (or uses
/// a reserved sentinel name).
pub fn parse_agent(
    file_path: impl Into<std::path::PathBuf>,
    content: &str,
    source: AgentSource,
) -> Result<AgentDefinition, AgentParseError> {
    let file_path = file_path.into();
    let (frontmatter, body) = split_frontmatter(content).map_err(|message| AgentParseError {
        file_path: file_path.clone(),
        message,
    })?;
    let fields = parse_agent_fields(&frontmatter).ok_or_else(|| AgentParseError {
        file_path: file_path.clone(),
        message: "invalid agent fields: `name` and `description` are required, and the \
                  names `main`/`sub` are reserved"
            .to_string(),
    })?;
    Ok(AgentDefinition {
        fields,
        system_prompt: body,
        source,
        file_path,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"---
name: glm-flash-coder
description: "Implementation agent on the direct zai subscription."
tools:
  - read
  - grep
model:
  - "zai/glm-5.3-flash"  # inline comment
thinkingLevel: low
---

You are an implementation agent.
"#;

    #[test]
    fn parses_full_frontmatter() {
        let def = parse_agent("glm.md", SAMPLE, AgentSource::User).unwrap();
        assert_eq!(def.fields.name, "glm-flash-coder");
        assert_eq!(
            def.fields.description,
            "Implementation agent on the direct zai subscription."
        );
        assert_eq!(
            def.fields.tools,
            Some(vec![
                "read".to_string(),
                "grep".to_string(),
                "yield".to_string()
            ])
        );
        assert_eq!(
            def.fields.model,
            Some(vec!["zai/glm-5.3-flash".to_string()])
        );
        assert_eq!(def.fields.thinking_level, Some(ThinkingLevel::Low));
        assert_eq!(def.fields.thinking_level.unwrap().effort(), "lo");
        assert_eq!(def.system_prompt, "You are an implementation agent.");
    }

    #[test]
    fn yield_appended_to_explicit_tool_list() {
        let def = parse_agent(
            "a.md",
            "---\nname: a\ndescription: d\ntools: [read]\n---\nbody\n",
            AgentSource::User,
        )
        .unwrap();
        assert_eq!(
            def.fields.tools,
            Some(vec!["read".to_string(), "yield".to_string()])
        );
    }

    #[test]
    fn empty_tool_list_still_gets_yield() {
        // TS semantics: `tools: []` is truthy, so the mandatory-yield rule
        // still appends yield even to an explicitly empty list.
        let def = parse_agent(
            "a.md",
            "---\nname: a\ndescription: d\ntools: []\n---\nbody\n",
            AgentSource::User,
        )
        .unwrap();
        assert_eq!(def.fields.tools, Some(vec!["yield".to_string()]));
    }

    #[test]
    fn csv_model_and_tools_strings() {
        let def = parse_agent(
            "a.md",
            "---\nname: a\ndescription: d\ntools: read, grep\nmodel: zai/glm-5.3-flash, kimi/k2\n---\n",
            AgentSource::User,
        )
        .unwrap();
        assert_eq!(
            def.fields.tools,
            Some(vec![
                "read".to_string(),
                "grep".to_string(),
                "yield".to_string()
            ])
        );
        assert_eq!(
            def.fields.model,
            Some(vec!["zai/glm-5.3-flash".to_string(), "kimi/k2".to_string()])
        );
    }

    #[test]
    fn task_tool_implies_unrestricted_spawns() {
        let def = parse_agent(
            "a.md",
            "---\nname: a\ndescription: d\ntools: [task]\n---\n",
            AgentSource::User,
        )
        .unwrap();
        assert_eq!(def.fields.spawns, Some(Spawns::All));
    }

    #[test]
    fn explicit_spawns_star_and_list() {
        let all = parse_agent(
            "a.md",
            "---\nname: a\ndescription: d\nspawns: \"*\"\n---\n",
            AgentSource::User,
        )
        .unwrap();
        assert_eq!(all.fields.spawns, Some(Spawns::All));
        let list = parse_agent(
            "a.md",
            "---\nname: a\ndescription: d\nspawns: [scout-glm, scout-kimi]\n---\n",
            AgentSource::User,
        )
        .unwrap();
        assert_eq!(
            list.fields.spawns,
            Some(Spawns::Agents(vec![
                "scout-glm".to_string(),
                "scout-kimi".to_string()
            ]))
        );
    }

    #[test]
    fn output_schema_preserved_verbatim() {
        let def = parse_agent(
            "a.md",
            "---\nname: a\ndescription: d\noutput:\n  type: object\n  properties:\n    ok:\n      type: boolean\n---\n",
            AgentSource::User,
        )
        .unwrap();
        let out = def.fields.output.unwrap();
        assert_eq!(out["type"], "object");
        assert_eq!(out["properties"]["ok"]["type"], "boolean");
    }

    #[test]
    fn reserved_names_rejected() {
        for name in ["main", "sub", "Main", " SUB "] {
            let content = format!("---\nname: {name}\ndescription: d\n---\n");
            assert!(
                parse_agent("a.md", &content, AgentSource::User).is_err(),
                "name {name:?} must be rejected"
            );
        }
    }

    #[test]
    fn missing_required_fields_rejected() {
        assert!(parse_agent("a.md", "---\nname: a\n---\n", AgentSource::User).is_err());
        assert!(parse_agent("a.md", "---\ndescription: d\n---\n", AgentSource::User).is_err());
        assert!(parse_agent("a.md", "no frontmatter at all\n", AgentSource::User).is_err());
        assert!(parse_agent(
            "a.md",
            "---\nname: [broken\ndescription: d\n",
            AgentSource::User
        )
        .is_err());
    }

    #[test]
    fn thinking_fallback_and_prewalk() {
        let def = parse_agent(
            "a.md",
            "---\nname: a\ndescription: d\nthinking: max\nprewalk: true\nadvisor: zai/glm-5.3-flash\nblocking: true\n---\n",
            AgentSource::User,
        )
        .unwrap();
        assert_eq!(def.fields.thinking_level, Some(ThinkingLevel::Max));
        assert_eq!(def.fields.prewalk, Some(PrewalkAdvisor::Enabled));
        assert_eq!(
            def.fields.advisor,
            Some(PrewalkAdvisor::Target("zai/glm-5.3-flash".to_string()))
        );
        assert_eq!(def.fields.blocking, Some(true));
    }
}
