//! Agent discovery from filesystem, ported from `task/discovery.ts`.
//!
//! Discovery roots (OMP-native task-agent convention):
//! - `~/.omp/agent/agents/*.md` (user-level)
//! - `.omp/agents/*.md` (project-level)
//!
//! Like the TS loader, a malformed file never aborts discovery: failures are
//! collected per-file and reported alongside the successfully parsed agents.

use std::path::{Path, PathBuf};

use tracing::warn;

use crate::agent::{parse_agent, AgentDefinition, AgentParseError, AgentSource};

/// A file that failed to load during discovery.
pub type AgentLoadError = AgentParseError;

/// Result of agent discovery: parsed agents plus per-file failures.
#[derive(Debug, Default)]
pub struct DiscoveryResult {
    pub agents: Vec<AgentDefinition>,
    pub failures: Vec<AgentLoadError>,
}

impl DiscoveryResult {
    /// Number of successfully parsed agents.
    #[must_use]
    pub fn agent_count(&self) -> usize {
        self.agents.len()
    }

    /// Number of files that failed to parse.
    #[must_use]
    pub fn failure_count(&self) -> usize {
        self.failures.len()
    }

    /// Look up an agent by name (mirrors `getAgent`).
    #[must_use]
    pub fn get_agent(&self, name: &str) -> Option<&AgentDefinition> {
        self.agents.iter().find(|a| a.fields.name == name)
    }
}

/// Load all `*.md` agent definitions from a directory.
///
/// Non-directories and unreadable directories yield an empty result (a
/// missing agents dir is normal, not an error); individual malformed files
/// land in [`DiscoveryResult::failures`].
pub fn discover_agents_in_dir(dir: &Path, source: AgentSource) -> DiscoveryResult {
    let mut result = DiscoveryResult::default();
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return result,
        Err(e) => {
            warn!(dir = %dir.display(), error = %e, "failed to read agent directory");
            return result;
        }
    };
    let mut paths: Vec<PathBuf> = entries
        .filter_map(std::result::Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "md"))
        .collect();
    // Deterministic order: sorted by file name.
    paths.sort();
    for path in paths {
        match std::fs::read_to_string(&path) {
            Ok(content) => match parse_agent(&path, &content, source) {
                Ok(agent) => result.agents.push(agent),
                Err(err) => {
                    warn!(file = %path.display(), error = %err.message, "skipping malformed agent definition");
                    result.failures.push(err);
                }
            },
            Err(e) => result.failures.push(AgentLoadError {
                file_path: path.clone(),
                message: format!("failed to read file: {e}"),
            }),
        }
    }
    result
}

/// Discover user-level agents from `~/.omp/agent/agents/*.md`.
#[must_use]
pub fn discover_user_agents(home: &Path) -> DiscoveryResult {
    discover_agents_in_dir(&home.join(".omp/agent/agents"), AgentSource::User)
}

/// Discover project-level agents from `<cwd>/.omp/agents/*.md`.
#[must_use]
pub fn discover_project_agents(cwd: &Path) -> DiscoveryResult {
    discover_agents_in_dir(&cwd.join(".omp/agents"), AgentSource::Project)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn missing_dir_is_not_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let result = discover_agents_in_dir(&dir.path().join("nope"), AgentSource::User);
        assert_eq!(result.agent_count(), 0);
        assert_eq!(result.failure_count(), 0);
    }

    #[test]
    fn mixed_good_and_bad_files() {
        let dir = tempfile::tempdir().unwrap();
        let agents = dir.path();
        let mut good = std::fs::File::create(agents.join("good.md")).unwrap();
        writeln!(good, "---\nname: good\ndescription: d\n---\nbody").unwrap();
        std::fs::write(agents.join("bad.md"), "not frontmatter").unwrap();
        std::fs::write(agents.join("not-an-agent.txt"), "ignored").unwrap();

        let result = discover_agents_in_dir(agents, AgentSource::Project);
        assert_eq!(result.agent_count(), 1);
        assert_eq!(result.failure_count(), 1);
        assert!(result.get_agent("good").is_some());
        assert_eq!(
            result.get_agent("good").unwrap().source,
            AgentSource::Project
        );
    }
}
