//! Loopback Herdr integration: roster, journals, user prompts and activity.
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant};

use omp_session::index::{EntryKind, SessionIndex};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::io::AsyncReadExt as _;
use tokio::process::Command;
use tokio::sync::Mutex;

const MAX_OUTPUT: u64 = 4 * 1024 * 1024;
const TTL: Duration = Duration::from_secs(4);

#[derive(Default)]
pub(super) struct Roster {
    cache: Mutex<Option<(Instant, Snapshot)>>,
    // Indexing journals uses blocking I/O. Limit it independently of roster polls.
    history: Mutex<Option<(PathBuf, u64, Option<std::time::SystemTime>, HistoryPage)>>,
    // Restored Herdr processes initially omit tokens.model. Remember the
    // journal-derived model until that journal changes instead of reindexing
    // every restored session on each five-second sidebar refresh.
    models: Mutex<HashMap<PathBuf, (u64, Option<std::time::SystemTime>, Option<String>)>>,
}

#[derive(Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct Snapshot {
    agents: Vec<Row>,
    names_available: bool,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct Row {
    pane_id: String,
    workspace: String,
    cwd: String,
    title: String,
    model: String,
    status: String,
    state_change_seq: u64,
    address: Option<String>,
    session_file: Option<String>,
}

#[derive(Deserialize)]
struct Pane {
    pane_id: String,
    workspace_id: String,
    #[serde(default)]
    cwd: String,
    agent: Option<String>,
    agent_status: String,
    #[serde(default)]
    state_change_seq: u64,
    terminal_title_stripped: Option<String>,
    agent_session: Option<Session>,
    #[serde(default)]
    tokens: HashMap<String, String>,
}

#[derive(Deserialize)]
struct Session {
    kind: String,
    value: String,
}

#[derive(Deserialize)]
struct Space {
    workspace_id: String,
    label: String,
}

#[derive(Deserialize)]
struct NamedSpace {
    addresses: Vec<Address>,
}

#[derive(Deserialize)]
struct Address {
    pane_id: Option<String>,
    alias: String,
    pane_type: String,
    retired: bool,
}

impl Roster {
    pub(super) async fn snapshot(&self) -> Snapshot {
        let mut cache = self.cache.lock().await;
        if let Some((_, value)) = cache.as_ref().filter(|(time, _)| time.elapsed() < TTL) {
            return value.clone();
        }
        let mut value = collect().await;
        self.restore_models(&mut value).await;
        *cache = Some((Instant::now(), value.clone()));
        value
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) enum HistoryError {
    Unavailable,
    TooLarge,
    Invalid,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct HistoryPage {
    messages: Vec<Value>,
    before: Option<u64>,
    omitted: bool,
    offsets: Vec<u64>,
    tokens: JournalTokens,
    model: Option<String>,
    thinking_level: Option<String>,
    ultra: Option<String>,
}

impl Roster {
    async fn restore_models(&self, snapshot: &mut Snapshot) {
        let live: HashSet<PathBuf> = snapshot
            .agents
            .iter()
            .filter_map(|row| row.session_file.as_deref().map(PathBuf::from))
            .collect();
        let mut cache = self.models.lock().await;
        cache.retain(|path, _| live.contains(path));
        for row in &mut snapshot.agents {
            if !matches!(row.model.as_str(), "" | "omp" | "omt") {
                continue;
            }
            let Some(path) = row.session_file.as_deref().map(PathBuf::from) else {
                continue;
            };
            let Ok(meta) = tokio::fs::metadata(&path).await else {
                continue;
            };
            let stamp = (meta.len(), meta.modified().ok());
            if let Some((_, _, model)) = cache
                .get(&path)
                .filter(|(len, mtime, _)| (*len, *mtime) == stamp)
            {
                if let Some(model) = model {
                    row.model = model.clone();
                }
                continue;
            }
            let job_path = path.clone();
            let model = tokio::task::spawn_blocking(move || read_history(&job_path, None))
                .await
                .ok()
                .and_then(Result::ok)
                .and_then(|page| page.model);
            if let Some(model) = &model {
                row.model = model.clone();
            }
            cache.insert(path, (stamp.0, stamp.1, model));
        }
    }

    pub(super) async fn history(
        &self,
        pane: &str,
        before: Option<u64>,
    ) -> Result<HistoryPage, HistoryError> {
        // Only resolve pane IDs from Herdr; never accept a file path from the browser.
        let path = self
            .snapshot()
            .await
            .agents
            .into_iter()
            .find(|row| row.pane_id == pane)
            .and_then(|row| row.session_file)
            .map(PathBuf::from)
            .ok_or(HistoryError::Unavailable)?;
        let mut cache = self.history.lock().await;
        let meta = tokio::fs::metadata(&path)
            .await
            .map_err(|_| HistoryError::Unavailable)?;
        let stamp = (meta.len(), meta.modified().ok());
        if before.is_none() {
            if let Some((_, _, _, page)) = cache
                .as_ref()
                .filter(|(p, len, mtime, _)| p == &path && (*len, *mtime) == stamp)
            {
                return Ok(page.clone());
            }
        }
        let job_path = path.clone();
        let page = tokio::task::spawn_blocking(move || read_history(&job_path, before))
            .await
            .map_err(|_| HistoryError::Invalid)??;
        if before.is_none() {
            *cache = Some((path, stamp.0, stamp.1, page.clone()));
        }
        Ok(page)
    }
}

#[derive(Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
struct JournalTokens {
    input: u64,
    output: u64,
    cache_read: u64,
    cache_write: u64,
    reported: bool,
}

fn read_history(path: &Path, before: Option<u64>) -> Result<HistoryPage, HistoryError> {
    const MAX_JOURNAL: u64 = 128 * 1024 * 1024;
    const PAGE_BYTES: u64 = 4 * 1024 * 1024;
    const PAGE_MESSAGES: usize = 96;
    if !path.is_absolute() || !path.extension().is_some_and(|ext| ext == "jsonl") {
        return Err(HistoryError::Unavailable);
    }
    let metadata = std::fs::metadata(path).map_err(|_| HistoryError::Unavailable)?;
    if !metadata.is_file() {
        return Err(HistoryError::Unavailable);
    }
    if metadata.len() > MAX_JOURNAL {
        return Err(HistoryError::TooLarge);
    }
    let index = SessionIndex::load(path).map_err(|_| HistoryError::Invalid)?;
    if index.invalid_header {
        return Err(HistoryError::Invalid);
    }
    let mut page = Vec::new();
    let mut bytes = 0;
    let mut omitted = false;
    let mut cursor = None;
    let mut has_more = false;
    let branch = index.branch();
    let bookkeeping: Vec<u32> = branch
        .iter()
        .copied()
        .filter(|&i| {
            matches!(
                index.refs[i as usize].kind,
                EntryKind::ModelChange | EntryKind::ThinkingLevelChange
            )
        })
        .collect();
    let mut model = None;
    let mut thinking_level = None;
    let mut ultra = None;
    index
        .for_each_line(path, &bookkeeping, |entry, raw| {
            let Ok(value) = serde_json::from_str::<Value>(raw) else {
                return Ok(());
            };
            if entry.kind == EntryKind::ModelChange {
                if value
                    .get("role")
                    .and_then(Value::as_str)
                    .unwrap_or("default")
                    == "default"
                {
                    model = value
                        .get("model")
                        .and_then(Value::as_str)
                        .map(str::to_owned);
                }
            } else {
                thinking_level = value
                    .get("thinkingLevel")
                    .and_then(Value::as_str)
                    .map(str::to_owned);
                ultra = value
                    .get("ultra")
                    .and_then(Value::as_str)
                    .map(str::to_owned);
            }
            Ok(())
        })
        .map_err(|_| HistoryError::Invalid)?;
    for &i in branch.iter().rev() {
        let entry = &index.refs[i as usize];
        if entry.kind != EntryKind::Message || before.is_some_and(|end| entry.offset >= end) {
            continue;
        }
        let len = u64::from(entry.len);
        if page.len() == PAGE_MESSAGES || (len <= PAGE_BYTES && bytes + len > PAGE_BYTES) {
            has_more = true;
            break;
        }
        cursor = Some(entry.offset);
        if len > PAGE_BYTES {
            omitted = true;
            continue;
        }
        bytes += len;
        page.push(i);
    }
    page.reverse();
    let mut offsets = Vec::new();
    let messages = index
        .load_branch(path, &page, true)
        .map_err(|_| HistoryError::Invalid)?
        .into_iter()
        .zip(&page)
        .filter_map(|(line, i)| {
            let message = line.value?.get_mut("message").map(Value::take)?;
            offsets.push(index.refs[*i as usize].offset);
            Some(message)
        })
        .collect();
    let mut tokens = JournalTokens::default();
    if before.is_none() {
        // Read only message entries; compute totals independently of pagination.
        for batch in branch
            .iter()
            .copied()
            .filter(|i| {
                index.refs[*i as usize].kind == EntryKind::Message
                    && u64::from(index.refs[*i as usize].len) <= PAGE_BYTES
            })
            .collect::<Vec<_>>()
            .chunks(16)
        {
            let entries = index
                .load_branch(path, batch, true)
                .map_err(|_| HistoryError::Invalid)?;
            for message in entries
                .iter()
                .filter_map(|line| line.value.as_ref()?.get("message"))
            {
                if message["role"] != "assistant" {
                    continue;
                }
                let Some(usage) = message.get("usage").filter(|v| v.is_object()) else {
                    continue;
                };
                let get = |key: &str| usage[key].as_u64().unwrap_or(0);
                tokens.reported = true;
                tokens.input = tokens.input.saturating_add(get("input"));
                tokens.output = tokens.output.saturating_add(get("output"));
                tokens.cache_read = tokens.cache_read.saturating_add(get("cacheRead"));
                tokens.cache_write = tokens.cache_write.saturating_add(get("cacheWrite"));
            }
        }
    }
    Ok(HistoryPage {
        messages,
        before: cursor.filter(|_| has_more),
        omitted,
        offsets,
        tokens,
        model,
        thinking_level,
        ultra,
    })
}

#[derive(Debug)]
pub(super) enum LiveError {
    Unavailable,
    Rejected,
    Unknown,
}

impl Roster {
    async fn live_bin(&self, pane: &str) -> Result<PathBuf, LiveError> {
        if !self
            .snapshot()
            .await
            .agents
            .iter()
            .any(|row| row.pane_id == pane)
        {
            return Err(LiveError::Unavailable);
        }
        herdr_path().ok_or(LiveError::Unavailable)
    }

    pub(super) async fn prompt(&self, pane: &str, text: &str) -> Result<(), LiveError> {
        let bin = self.live_bin(pane).await?;
        // Explicit user chat, not an automated dispatch. Herdr validates the
        // foreground process as an agent before submitting (never a shell).
        let result = tokio::time::timeout(Duration::from_secs(10), async {
            Command::new(&bin)
                .args(["agent", "prompt", pane, text])
                .stdin(Stdio::null())
                .kill_on_drop(true)
                .output()
                .await
        })
        .await
        .map_err(|_| LiveError::Unknown)?
        .map_err(|_| LiveError::Unavailable)?;
        if result.status.success() {
            return Ok(());
        }
        let rejected = [&result.stdout, &result.stderr].iter().any(|bytes| {
            serde_json::from_slice::<Value>(bytes)
                .ok()
                .is_some_and(|v| {
                    matches!(
                        v.pointer("/error/code").and_then(Value::as_str),
                        Some("agent_blocked" | "agent_not_ready" | "agent_not_found")
                    )
                })
        });
        Err(if rejected {
            LiveError::Rejected
        } else {
            LiveError::Unknown
        })
    }

    pub(super) async fn activity(&self, pane: &str) -> Result<String, LiveError> {
        let bin = self.live_bin(pane).await?;
        let text = command_output(
            &bin,
            &[
                "agent", "read", pane, "--source", "recent", "--lines", "160", "--format", "text",
            ],
            &bin,
        )
        .await
        .ok_or(LiveError::Unavailable)?;
        Ok(String::from_utf8_lossy(&text).into_owned())
    }
}

fn herdr_path() -> Option<PathBuf> {
    crate::features::browser::find_on_path("herdr").or_else(|| {
        std::env::var_os("HOME")
            .map(PathBuf::from)
            .map(|path| path.join(".local/bin/herdr"))
            .filter(|path| path.is_file())
    })
}

async fn command(bin: &Path, args: &[&str], herdr: &Path) -> Option<Value> {
    serde_json::from_slice(&command_output(bin, args, herdr).await?).ok()
}

async fn command_output(bin: &Path, args: &[&str], herdr: &Path) -> Option<Vec<u8>> {
    tokio::time::timeout(Duration::from_secs(3), async {
        let mut child = Command::new(bin)
            .args(args)
            .env("HERDR_BIN_PATH", herdr)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .spawn()
            .ok()?;
        let mut bytes = Vec::new();
        child
            .stdout
            .take()?
            .take(MAX_OUTPUT + 1)
            .read_to_end(&mut bytes)
            .await
            .ok()?;
        if bytes.len() as u64 > MAX_OUTPUT || !child.wait().await.ok()?.success() {
            return None;
        }
        Some(bytes)
    })
    .await
    .ok()
    .flatten()
}

fn take_rows<T: serde::de::DeserializeOwned>(
    response: Option<Value>,
    pointers: &str,
) -> Option<Vec<T>> {
    serde_json::from_value(response?.pointer_mut(pointers)?.take()).ok()
}

async fn collect() -> Snapshot {
    let home = std::env::var_os("HOME").map(PathBuf::from);
    let Some(herdr) = herdr_path() else {
        return Snapshot::default();
    };
    let (panes, spaces) = tokio::join!(
        command(&herdr, &["agent", "list"], &herdr),
        command(&herdr, &["workspace", "list"], &herdr),
    );
    let panes = take_rows::<Pane>(panes, "/result/agents").unwrap_or_default();
    let Some(first) = panes.first() else {
        return Snapshot::default();
    };
    let spaces = take_rows::<Space>(spaces, "/result/workspaces").unwrap_or_default();
    let state = std::env::var_os("XDG_STATE_HOME")
        .map(PathBuf::from)
        .or_else(|| home.map(|path| path.join(".local/state")))
        .map(|path| path.join("herdr/plugins/namedrop"));
    let names = if let Some(state) = state.filter(|path| path.join("runtime/namedrop").is_file()) {
        if let Some(state_arg) = state.to_str() {
            take_rows::<NamedSpace>(
                command(
                    &state.join("runtime/namedrop"),
                    &[
                        "--plugin-state",
                        state_arg,
                        "address",
                        "list",
                        "--input-json",
                        r#"{"workspaces":"all"}"#,
                        "--herdr-workspace",
                        &first.workspace_id,
                    ],
                    &herdr,
                )
                .await,
                "/data/workspaces",
            )
        } else {
            None
        }
    } else {
        None
    };
    merge(panes, spaces, names)
}

fn merge(panes: Vec<Pane>, spaces: Vec<Space>, names: Option<Vec<NamedSpace>>) -> Snapshot {
    let names_available = names.is_some();
    let mut aliases: HashMap<String, String> = HashMap::new();
    for address in names
        .into_iter()
        .flatten()
        .flat_map(|space| space.addresses)
    {
        if !address.retired && address.pane_type == "agent" && !address.alias.is_empty() {
            if let Some(id) = address.pane_id {
                aliases.insert(id, format!("@{}", address.alias));
            }
        }
    }
    let spaces: HashMap<_, _> = spaces
        .into_iter()
        .map(|space| (space.workspace_id, space.label))
        .collect();
    let agents = panes.into_iter().map(|pane| {
        // Herdr commonly prefixes the model with a Nerd Font provider glyph.
        let model = pane.tokens.get("model").map(|model| {
            model.trim_start_matches(|c: char| c.is_whitespace() || matches!(c as u32, 0xe000..=0xf8ff | 0xf0000..=0xffffd | 0x100000..=0x10fffd)).to_owned()
        }).filter(|model| !model.is_empty()).or(pane.agent).unwrap_or_default();
        Row {
            address: aliases.remove(&pane.pane_id),
            title: pane.terminal_title_stripped.unwrap_or_default().trim_start_matches("τ > ").to_owned(),
            workspace: spaces.get(&pane.workspace_id).cloned().unwrap_or(pane.workspace_id),
            cwd: pane.cwd,
            pane_id: pane.pane_id,
            model,
            status: pane.agent_status,
            state_change_seq: pane.state_change_seq,
            session_file: pane.agent_session.filter(|session| session.kind == "path").map(|session| session.value),
        }
    }).collect();
    Snapshot {
        agents,
        names_available,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn history_pages_the_active_branch_without_writing_to_the_file() {
        use std::io::Write as _;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("session.jsonl");
        let mut file = std::fs::File::create(&path).unwrap();
        writeln!(file, "{}", json!({"type":"session","id":"session"})).unwrap();
        writeln!(
            file,
            "{}",
            json!({"type":"message","id":"abandoned","message":{"content":"skip"}})
        )
        .unwrap();
        writeln!(
            file,
            "{}",
            json!({"type":"model_change","id":"model","model":"kimi-code/k3.1-pro-highspeed"})
        )
        .unwrap();
        writeln!(file, "{}", json!({"type":"thinking_level_change","id":"thinking","parentId":"model","thinkingLevel":"high","configured":"high"})).unwrap();
        for i in 0..100 {
            let parent = Some(if i > 0 {
                (i - 1).to_string()
            } else {
                "thinking".to_owned()
            });
            writeln!(file, "{}", json!({"type":"message","id":i.to_string(),"parentId":parent,"message":{"role":"user","content":i}})).unwrap();
        }
        drop(file);
        let original = std::fs::read(&path).unwrap();
        let latest = read_history(&path, None).unwrap();
        assert_eq!(latest.messages.len(), 96);
        assert_eq!(
            latest.model.as_deref(),
            Some("kimi-code/k3.1-pro-highspeed")
        );
        assert_eq!(latest.thinking_level.as_deref(), Some("high"));
        assert_eq!(latest.messages[0]["content"], 4);
        let older = read_history(&path, Some(latest.before.unwrap())).unwrap();
        assert_eq!(
            older
                .messages
                .iter()
                .map(|m| m["content"].as_u64().unwrap())
                .collect::<Vec<_>>(),
            vec![0, 1, 2, 3]
        );
        assert!(older.before.is_none());
        assert_eq!(std::fs::read(&path).unwrap(), original);
    }

    #[tokio::test]
    async fn restores_missing_roster_models_from_session_journals() {
        use std::io::Write as _;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("session.jsonl");
        let mut file = std::fs::File::create(&path).unwrap();
        writeln!(file, "{}", json!({"type":"session","id":"session"})).unwrap();
        writeln!(
            file,
            "{}",
            json!({"type":"model_change","id":"model","model":"zai/glm-5.3"})
        )
        .unwrap();
        drop(file);
        let mut snapshot = Snapshot {
            agents: vec![Row {
                pane_id: "w5:p1".into(),
                workspace: "w5".into(),
                cwd: "/tmp".into(),
                title: "agent".into(),
                model: "omp".into(),
                status: "idle".into(),
                state_change_seq: 1,
                address: None,
                session_file: Some(path.to_string_lossy().into_owned()),
            }],
            names_available: false,
        };
        let roster = Roster::default();
        roster.restore_models(&mut snapshot).await;
        assert_eq!(snapshot.agents[0].model, "zai/glm-5.3");
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap();
        writeln!(
            file,
            "{}",
            json!({"type":"model_change","id":"model2","parentId":"model","model":"kimi-code/k3.1-pro"})
        )
        .unwrap();
        drop(file);
        // A fresh Herdr poll again lacks tokens.model; this is not a
        // continuation of the previously enriched Snapshot.
        snapshot.agents[0].model = "omp".into();
        roster.restore_models(&mut snapshot).await;
        assert_eq!(snapshot.agents[0].model, "kimi-code/k3.1-pro");
    }

    #[test]
    fn joins_only_attached_agent_addresses_by_full_pane_id() {
        let panes = serde_json::from_value(json!([
            { "pane_id": "w5:p1", "workspace_id": "w5", "agent_status": "working", "tokens": {"model": "\u{f5e0} k3.1-pro"} },
            { "pane_id": "w6:p1", "workspace_id": "w6", "agent_status": "blocked" }
        ])).unwrap();
        let names = serde_json::from_value(json!([{"addresses": [
            {"pane_id": "w5:p1", "alias": "actor", "pane_type": "agent", "retired": false},
            {"pane_id": "w6:p1", "alias": "shell", "pane_type": "terminal", "retired": false},
            {"pane_id": "w6:p1", "alias": "old", "pane_type": "agent", "retired": true},
            {"pane_id": null, "alias": "detached", "pane_type": "agent", "retired": false}
        ]}]))
        .unwrap();
        let result = merge(panes, vec![], Some(names));
        assert_eq!(result.agents[0].address.as_deref(), Some("@actor"));
        assert_eq!(result.agents[0].model, "k3.1-pro");
        assert_eq!(result.agents[1].address, None);
        assert_eq!(result.agents[1].status, "blocked");
    }
}
