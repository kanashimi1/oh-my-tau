//! Session-file bookkeeping: entry access, branch appends, and resume heads.

use anyhow::Result;
use pa_agent::types::AgentMessage;
use serde_json::Value;
use std::path::{Path, PathBuf};

use super::Runtime;

impl Runtime {
    /// Publish the live session file to `omp-urls` (URL resolution cache
    /// scoping); `None` under `--no-session`. Call at every session-file
    /// attach point (start, `new_session`, `load_session`, fork/branch).
    pub fn publish_session_file(&self) {
        omp_urls::set_global_session_file(self.persist.file());
    }

    pub fn session_id(&self) -> String {
        self.persist.session_id()
    }

    /// The live session file; `None` under `--no-session`.
    pub fn live_session_file(&self) -> Option<PathBuf> {
        self.persist.file()
    }

    pub fn persist_file(&self) -> Option<PathBuf> {
        self.persist.file()
    }

    pub fn session_file(&self) -> PathBuf {
        self.persist.path()
    }

    /// Raw JSON entries of the live session file (header excluded), append order.
    pub fn entries(&self) -> Result<Vec<Value>> {
        read_entries(&self.persist.path())
    }

    /// Root-to-leaf entries of the active branch (TS `sessionManager.getBranch()`): the writer's
    /// live append point after a `/tree` move, else the last journaled entry.
    pub fn branch_entries(&self) -> Vec<Value> {
        let Ok(entries) = self.entries() else {
            return Vec::new();
        };
        let leaf = match self.persist.leaf_divergence() {
            Some(Some(id)) => Some(id),
            Some(None) => None,
            None => omp_session::tree::leaf_id(&entries).map(str::to_owned),
        };
        leaf.map(|leaf| {
            omp_session::tree::branch_path(&entries, &leaf)
                .into_iter()
                .cloned()
                .collect()
        })
        .unwrap_or_default()
    }

    /// Run `f` against the live session writer (tree navigation: leaf moves,
    /// `branch_summary` appends).
    #[allow(dead_code)] // host-request API used by tree navigation, wired with the rpc surface
    pub fn with_writer<R>(&self, f: impl FnOnce(&mut omp_session::SessionWriter) -> R) -> R {
        self.persist.with_writer(f)
    }

    /// Move the append point to an earlier entry (`/tree` navigation, omp
    /// `SessionManager.branch`); tracked by the compaction guard's navigation
    /// mirror (bug-hunt #4: compact must target the selected branch).
    ///
    /// # Errors
    /// `id` is not an entry of the live session file.
    pub fn branch_to(&self, id: Option<String>) -> Result<()> {
        self.persist.branch_to(id)
    }

    /// Append a `branch_summary` entry parented at `branch_from` and land the
    /// writer on it (`/tree` navigation that abandons the branch durably).
    ///
    /// # Errors
    /// `branch_from` is not an entry of the live session file, or the write failed.
    pub fn append_branch_summary(
        &self,
        branch_from: Option<&str>,
        summary: &str,
        details: Option<Value>,
        from_extension: Option<bool>,
    ) -> Result<String> {
        self.persist.append_branch_summary(
            branch_from,
            summary,
            details,
            from_extension,
            &crate::persist::now_iso(),
        )
    }

    /// Append a non-message entry (`custom`, extension `append_entry`) to the
    /// live session file; a no-op write-wise for `--no-session` scratch files.
    ///
    /// # Errors
    /// Session-writer serialization/I/O failure.
    pub fn append_entry(&self, kind: &str, payload: Value) -> Result<()> {
        self.persist
            .append_entry(kind, payload)
            .map_err(|e| anyhow::anyhow!(e))
    }
}

/// Borrowed view of a `message` entry line: deserializing the envelope straight
/// from the raw journal text yields the entry's `AgentMessage` without building
/// the entry's `Value` DOM (lazy resume).
#[derive(serde::Deserialize)]
pub(super) struct MessageEnvelope {
    pub(super) message: AgentMessage,
}

/// Model/thinking bookkeeping of a session branch (TS `buildSessionContext`
/// `models` + `getLastModelChangeRole` + thinking fields).
#[derive(Debug, Default, PartialEq)]
pub(super) struct SessionModelState {
    pub(super) models: std::collections::BTreeMap<String, String>,
    pub(super) last_role: Option<String>,
    /// `(thinkingLevel, configured)` of the last `thinking_level_change`.
    pub(super) thinking: Option<(String, Option<String>)>,
    /// `ultra` of that same entry (omp-rs ultra mode; absent in TS files).
    pub(super) ultra: Option<String>,
}

#[derive(serde::Deserialize)]
struct BookkeepingHead {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    role: Option<String>,
    #[serde(rename = "thinkingLevel", default)]
    thinking_level: Option<String>,
    #[serde(default)]
    configured: Option<String>,
    #[serde(default)]
    ultra: Option<String>,
}

/// Read only the branch's `model_change` / `thinking_level_change` entries
/// and, when no explicit default model exists, the last assistant message's
/// `provider/model` (TS infers `models.default` from assistant messages
/// until an explicit default is recorded).
pub(super) fn session_model_state(
    index: &omp_session::index::SessionIndex,
    path: &Path,
    branch: &[u32],
) -> Result<SessionModelState> {
    use omp_session::index::EntryKind;
    let picks: Vec<u32> = branch
        .iter()
        .copied()
        .filter(|&i| {
            matches!(
                index.refs[i as usize].kind,
                EntryKind::ModelChange | EntryKind::ThinkingLevelChange
            )
        })
        .collect();
    let mut st = SessionModelState::default();
    index.for_each_line(path, &picks, |r, raw| {
        let Ok(h) = serde_json::from_str::<BookkeepingHead>(raw) else {
            return Ok(());
        };
        if r.kind == EntryKind::ModelChange {
            let role = h.role.unwrap_or_else(|| "default".to_owned());
            if let Some(m) = h.model.filter(|m| !m.is_empty()) {
                st.models.insert(role.clone(), m);
            }
            st.last_role = Some(role);
        } else {
            st.thinking = Some((
                h.thinking_level.unwrap_or_else(|| "off".to_owned()),
                h.configured,
            ));
            st.ultra = h.ultra;
        }
        Ok(())
    })?;
    if !st.models.contains_key("default") {
        for &i in branch.iter().rev() {
            if index.refs[i as usize].kind != EntryKind::Message {
                continue;
            }
            let mut found = None;
            index.for_each_line(path, &[i], |_, raw| {
                if let Ok(MessageHead { message: Some(m) }) =
                    serde_json::from_str::<MessageHead>(raw)
                {
                    if m.role.as_deref() == Some("assistant") {
                        found = Some(format!(
                            "{}/{}",
                            m.provider.as_deref().unwrap_or("undefined"),
                            m.model.as_deref().unwrap_or("undefined")
                        ));
                    }
                }
                Ok(())
            })?;
            if let Some(m) = found {
                st.models.insert("default".to_owned(), m);
                break;
            }
        }
    }
    Ok(st)
}

#[derive(serde::Deserialize)]
struct MessageHead {
    #[serde(default)]
    message: Option<MessageHeadFields>,
}

#[derive(serde::Deserialize)]
struct MessageHeadFields {
    #[serde(default)]
    role: Option<String>,
    #[serde(default)]
    provider: Option<String>,
    #[serde(default)]
    model: Option<String>,
}

#[derive(serde::Deserialize)]
struct CustomHead {
    #[serde(rename = "customType", default)]
    custom_type: Option<String>,
}

/// Bookkeeping entry kinds whose full payload the context-state accumulation
/// reads from anywhere on the branch (all small).
pub(super) fn is_bookkeeping(kind: omp_session::index::EntryKind) -> bool {
    matches!(
        kind,
        omp_session::index::EntryKind::ThinkingLevelChange
            | omp_session::index::EntryKind::ModelChange
            | omp_session::index::EntryKind::ServiceTierChange
            | omp_session::index::EntryKind::TtsrInjection
            | omp_session::index::EntryKind::ModeChange
    )
}

/// Link-only stand-in for a branch entry the context rebuild never reads
/// beyond its `type`/`id`/`timestamp` (pre-history messages, old compactions,
/// labels, ...). Keeps the reduced branch layout-identical to the journal.
pub(super) fn link_shim(r: &omp_session::index::EntryRef) -> Value {
    let mut o = serde_json::Map::with_capacity(3);
    o.insert("type".into(), r.type_name.to_string().into());
    if !r.id.is_empty() {
        o.insert("id".into(), r.id.to_string().into());
    }
    if !r.timestamp.is_empty() {
        o.insert("timestamp".into(), r.timestamp.to_string().into());
    }
    Value::Object(o)
}

/// A message shim carrying just the fields the rebuild's state accumulation
/// looks at (`message.role`/`provider`/`model` for the default-model inference).
pub(super) fn message_head_shim(r: &omp_session::index::EntryRef, raw: &str) -> Value {
    let mut shim = link_shim(r);
    let Ok(head) = serde_json::from_str::<MessageHead>(raw) else {
        return shim;
    };
    if let Some(m) = head.message {
        let mut msg = serde_json::Map::with_capacity(3);
        if let Some(role) = m.role {
            msg.insert("role".into(), role.into());
        }
        if let Some(provider) = m.provider {
            msg.insert("provider".into(), provider.into());
        }
        if let Some(model) = m.model {
            msg.insert("model".into(), model.into());
        }
        shim["message"] = Value::Object(msg);
    }
    shim
}

/// The `customType` of a `custom` entry line, when it parses as one.
pub(super) fn custom_type(raw: &str) -> Option<String> {
    serde_json::from_str::<CustomHead>(raw).ok()?.custom_type
}

pub fn read_entries(path: &Path) -> Result<Vec<Value>> {
    let lines = omp_session::read(path)?;
    let mut out = Vec::new();
    for l in lines {
        let v: Value = serde_json::from_str(&l.raw)?;
        if v.get("type").and_then(Value::as_str) != Some("session") {
            out.push(v);
        }
    }
    Ok(out)
}

/// Newest non-empty `*.jsonl` session in `dir` (has at least one message entry).
pub fn newest_session(dir: &Path) -> Option<PathBuf> {
    let mut files: Vec<(std::time::SystemTime, PathBuf)> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(Result::ok)
        .filter(|e| e.path().extension().is_some_and(|x| x == "jsonl"))
        .filter_map(|e| Some((e.metadata().ok()?.modified().ok()?, e.path())))
        .collect();
    files.sort_by_key(|b| std::cmp::Reverse(b.0));
    files.into_iter().map(|(_, p)| p).find(|p| {
        std::fs::read_to_string(p)
            .is_ok_and(|t| t.lines().any(|l| l.contains("\"type\":\"message\"")))
    })
}

#[cfg(test)]
#[allow(clippy::disallowed_methods)] // tests: throwaway runtimes drive async code
mod session_model_state_tests {
    use super::*;

    fn state_of(lines: &[&str]) -> SessionModelState {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.jsonl");
        std::fs::write(&path, lines.join("\n") + "\n").unwrap();
        let index = omp_session::index::SessionIndex::load(&path).expect("index");
        session_model_state(&index, &path, &index.branch()).expect("state")
    }

    const HEAD: &str = r#"{"type":"session","version":3,"id":"s","timestamp":"t","cwd":"/tmp"}"#;

    /// Regression: resume must see the model the session last switched to
    /// (here kimi after a glm start), not fall back to the config default.
    #[test]
    fn last_model_change_and_thinking_are_read() {
        let st = state_of(&[
            HEAD,
            r#"{"type":"model_change","id":"a","parentId":null,"timestamp":"t","model":"zai/glm-5.3-flash"}"#,
            r#"{"type":"thinking_level_change","id":"b","parentId":"a","timestamp":"t","thinkingLevel":"high","configured":"high"}"#,
            r#"{"type":"message","id":"c","parentId":"b","timestamp":"t","message":{"role":"assistant","provider":"zai","model":"glm-5.3-flash","content":[]}}"#,
            r#"{"type":"model_change","id":"d","parentId":"c","timestamp":"t","model":"kimi-code/kimi-for-coding","role":"default"}"#,
            r#"{"type":"thinking_level_change","id":"e","parentId":"d","timestamp":"t","thinkingLevel":"medium","configured":"max"}"#,
        ]);
        assert_eq!(
            st.models.get("default").map(String::as_str),
            Some("kimi-code/kimi-for-coding")
        );
        assert_eq!(st.last_role.as_deref(), Some("default"));
        assert_eq!(st.thinking, Some(("medium".into(), Some("max".into()))));
    }

    /// No `model_change` at all (old journals): the default is inferred from the
    /// last assistant message, as TS `buildSessionContext` does.
    #[test]
    fn default_inferred_from_last_assistant() {
        let st = state_of(&[
            HEAD,
            r#"{"type":"message","id":"a","parentId":null,"timestamp":"t","message":{"role":"assistant","provider":"zai","model":"glm-5.3-flash","content":[]}}"#,
            r#"{"type":"message","id":"b","parentId":"a","timestamp":"t","message":{"role":"assistant","provider":"anthropic","model":"claude-opus-5-5","content":[]}}"#,
            r#"{"type":"message","id":"c","parentId":"b","timestamp":"t","message":{"role":"user","content":"hi"}}"#,
        ]);
        assert_eq!(
            st.models.get("default").map(String::as_str),
            Some("anthropic/claude-opus-5-5")
        );
        assert_eq!(st.last_role, None);
        assert_eq!(st.thinking, None);
    }
}
