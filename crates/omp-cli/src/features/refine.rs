//! `/refine`: continual-harness refinement, ported from Prime Agent's
//! `/refine` (PrimeIntellect prime-agent, pa-core `refinement/`) and adapted
//! to omt's desktop surface.
//!
//! The planner reviews the current trajectory and emits small, evidence-backed
//! edits (`create|update|delete` of `prompt` notes or `memory` entries) to a
//! persistent harness store. Applied entries ride every future system prompt
//! (see [`harness_block`]) and, when applied mid-session, a hidden notice so
//! the live conversation sees them on the next turn. Every applied refinement
//! is recorded with the prior store snapshot, so [`rollback`] restores it.
//!
//! Store layout (mirrors the `.omp/` project convention, PA names the files):
//! - project (default): `<cwd>/.omp/harness/harness.json` + `refinement_history.jsonl`
//! - global (`/refine global …`): `~/.omp/agent/harness/` (same files)
//!
//! Usage: `/refine [instructions]`, `/refine list`, `/refine show <id>`,
//! `/refine history`, `/refine rollback [id]`.

use std::path::PathBuf;
use std::sync::Arc;

use serde_json::{json, Value};

use crate::rpc::slash::Outcome;
use crate::runtime::Runtime;

/// Planner answer ceiling (PA: 32k; this harness has two kinds, smaller).
const REFINE_MAX_TOKENS: u64 = 16_384;
/// Trajectory tail kept for the planner (chars; PA budgets tokens).
const TRAJECTORY_CAP_CHARS: usize = 80_000;
/// Per-entry content cap inside the system-prompt block.
const PROMPT_BLOCK_CONTENT_CAP: usize = 400;
/// Most recent refinement records shown to the planner.
const HISTORY_FOR_PROMPT: usize = 5;

/// Adapted from PA `REFINEMENT_SYSTEM_PROMPT` (kinds trimmed to this fork's
/// harness: prompt notes + memory entries; scope = project store by default).
const REFINE_SYSTEM_PROMPT: &str = "You are omt's /refine continual-harness subsystem.\n\nImprove the editable continual harness from the current trajectory. Instead of summarizing the conversation, emit precise create, update, or delete edits to reusable state — like compaction, but for durable notes instead of the token history.\n\nHarness kinds (this installation has exactly two):\n- prompt: supplemental prompt notes — narrow behavioral policy addendums. The base system prompt is immutable and MUST NOT be rewritten.\n- memory: durable facts, decisions, failures, preferences, and outcomes.\n\nScope policy:\n- The default store is project-local and persists across sessions in this workspace. Use it for project facts, conventions, tool/environment lessons, and active-task coordination notes.\n- Propose global edits only when the run instructions say the scope is global: stable cross-session user preferences or environment facts that clearly transcend this project.\n- During a project run, global entries are read-only context: never update or delete them; create a project entry instead when an override is needed.\n- Prefer memory for declarative facts, prompt for narrow behavioral policy.\n\nCreate or update the smallest relevant component. Prefer small evidence-backed edits; reject one-off noise, transient tool outputs, and unsupported hypotheses. If a prior refinement proved wrong, update or delete it instead of adding a contradicting one. Never propose editing source files directly.\n\nUse the trajectory, the current harness state, and the recent refinement history. Output JSON only with this exact shape:\n\n{\n  \"summary\": \"one sentence\",\n  \"rationale\": \"why these edits are justified by trajectory evidence\",\n  \"expectedOutcome\": \"what should improve and how to validate it\",\n  \"edits\": [\n    {\n      \"action\": \"create|update|delete\",\n      \"kind\": \"prompt|memory\",\n      \"id\": \"stable id for update/delete, optional for create\",\n      \"title\": \"required for create/update\",\n      \"content\": \"required for create/update\",\n      \"reason\": \"why this edit is useful\"\n    }\n  ]\n}";

// ────────────────────────── store ──────────────────────────

/// Which store a refinement touches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scope {
    /// `<cwd>/.omp/harness/` — default.
    Project,
    /// `~/.omp/agent/harness/` — explicit (`/refine global`).
    Global,
}

impl Scope {
    fn dir(self, cwd: &std::path::Path) -> PathBuf {
        match self {
            Self::Project => cwd.join(".omp/harness"),
            Self::Global => crate::sysprompt::agent_dir_pub().join("harness"),
        }
    }
    fn label(self) -> &'static str {
        match self {
            Self::Project => "project (.omp/harness)",
            Self::Global => "global (~/.omp/agent/harness)",
        }
    }
}

/// One harness entry (PA `HarnessEntry`, kinds trimmed).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    pub id: String,
    pub kind: String,
    pub title: String,
    pub content: String,
    pub updated_ms: u64,
}

/// One applied refinement record: enough to render history and roll back.
#[derive(Debug, Clone)]
struct HistoryRecord {
    id: String,
    ts_ms: u64,
    scope: Scope,
    summary: String,
    lines: Vec<String>,
    /// Store snapshot before this refinement applied (rollback target).
    prior: Vec<Entry>,
}

fn harness_file(dir: &std::path::Path) -> PathBuf {
    dir.join("harness.json")
}

fn history_file(dir: &std::path::Path) -> PathBuf {
    dir.join("refinement_history.jsonl")
}

fn load_entries(dir: &std::path::Path) -> Vec<Entry> {
    let Ok(text) = std::fs::read_to_string(harness_file(dir)) else {
        return Vec::new();
    };
    let Ok(Value::Array(rows)) = serde_json::from_str::<Value>(&text) else {
        return Vec::new();
    };
    rows.iter()
        .filter_map(|r| {
            let g = |k: &str| r.get(k).and_then(Value::as_str).map(str::to_owned);
            Some(Entry {
                id: g("id")?,
                kind: g("kind").unwrap_or_else(|| "memory".into()),
                title: g("title").unwrap_or_default(),
                content: g("content").unwrap_or_default(),
                updated_ms: r.get("updated_ms").and_then(Value::as_u64).unwrap_or(0),
            })
        })
        .collect()
}

fn save_entries(dir: &std::path::Path, entries: &[Entry]) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let rows: Vec<Value> = entries
        .iter()
        .map(|e| {
            json!({
                "id": e.id, "kind": e.kind, "title": e.title,
                "content": e.content, "updated_ms": e.updated_ms,
            })
        })
        .collect();
    std::fs::write(
        harness_file(dir),
        serde_json::to_string_pretty(&rows).unwrap_or_default(),
    )
}

fn load_history(dir: &std::path::Path) -> Vec<HistoryRecord> {
    let Ok(text) = std::fs::read_to_string(history_file(dir)) else {
        return Vec::new();
    };
    text.lines()
        .filter_map(|line| {
            let v: Value = serde_json::from_str(line).ok()?;
            let scope = if v.get("scope").and_then(Value::as_str) == Some("global") {
                Scope::Global
            } else {
                Scope::Project
            };
            let prior = v
                .get("prior")?
                .as_array()?
                .iter()
                .filter_map(|r| {
                    let g = |k: &str| r.get(k).and_then(Value::as_str).map(str::to_owned);
                    Some(Entry {
                        id: g("id")?,
                        kind: g("kind").unwrap_or_else(|| "memory".into()),
                        title: g("title").unwrap_or_default(),
                        content: g("content").unwrap_or_default(),
                        updated_ms: r.get("updated_ms").and_then(Value::as_u64).unwrap_or(0),
                    })
                })
                .collect::<Vec<_>>();
            Some(HistoryRecord {
                id: v.get("id").and_then(Value::as_str)?.to_owned(),
                ts_ms: v.get("ts_ms").and_then(Value::as_u64).unwrap_or(0),
                scope,
                summary: v
                    .get("summary")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
                lines: v
                    .get("lines")
                    .and_then(Value::as_array)
                    .map(|a| {
                        a.iter()
                            .filter_map(Value::as_str)
                            .map(str::to_owned)
                            .collect()
                    })
                    .unwrap_or_default(),
                prior,
            })
        })
        .collect()
}

fn append_history(dir: &std::path::Path, rec: &HistoryRecord) -> std::io::Result<()> {
    use std::fmt::Write as _;
    std::fs::create_dir_all(dir)?;
    let prior: Vec<Value> = rec
        .prior
        .iter()
        .map(|e| {
            json!({"id": e.id, "kind": e.kind, "title": e.title, "content": e.content, "updated_ms": e.updated_ms})
        })
        .collect();
    let mut line = serde_json::to_string(&json!({
        "id": rec.id,
        "ts_ms": rec.ts_ms,
        "scope": if rec.scope == Scope::Global { "global" } else { "project" },
        "summary": rec.summary,
        "lines": rec.lines,
        "prior": prior,
    }))
    .unwrap_or_default();
    line.push('\n');
    // invariant: writing to a String then the file; the record is one line.
    let _ = line.len();
    let mut text = std::fs::read_to_string(history_file(dir)).unwrap_or_default();
    let _ = write!(text, "{line}");
    std::fs::write(history_file(dir), text)
}

// ────────────────────────── planner ──────────────────────────

/// One proposed edit (subset of PA `RefinementEdit`).
#[derive(Debug, Clone)]
struct Edit {
    action: String,
    kind: String,
    id: String,
    title: String,
    content: String,
    reason: String,
}

#[derive(Debug, Clone)]
struct Proposal {
    summary: String,
    rationale: String,
    expected_outcome: String,
    edits: Vec<Edit>,
}

/// PA `extract_json_object`: first balanced `{...}` run, respecting strings.
fn extract_json_object(text: &str) -> Option<&str> {
    let bytes = text.as_bytes();
    let start = bytes.iter().position(|&b| b == b'{')?;
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for (i, &b) in bytes.iter().enumerate().skip(start) {
        if in_string {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_string = false;
            }
            continue;
        }
        match b {
            b'"' => in_string = true,
            b'{' => depth += 1,
            b'}' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return text.get(start..=i);
                }
            }
            _ => {}
        }
    }
    None
}

fn parse_proposal(text: &str) -> Result<Proposal, String> {
    let raw = extract_json_object(text).ok_or_else(|| {
        "the planner did not return a JSON object (truncated output? retry)".to_owned()
    })?;
    let v: Value = serde_json::from_str(raw).map_err(|e| format!("invalid JSON: {e}"))?;
    let s = |k: &str| {
        v.get(k)
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned()
    };
    let mut edits = Vec::new();
    if let Some(arr) = v.get("edits").and_then(Value::as_array) {
        for e in arr {
            let g = |k: &str| {
                e.get(k)
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned()
            };
            edits.push(Edit {
                action: g("action"),
                kind: g("kind"),
                id: g("id"),
                title: g("title"),
                content: g("content"),
                reason: g("reason"),
            });
        }
    }
    Ok(Proposal {
        summary: s("summary"),
        rationale: s("rationale"),
        expected_outcome: s("expectedOutcome"),
        edits,
    })
}

/// Apply a proposal's edits to `entries` (deletes → updates → creates).
/// Returns the rendered `+ ~ -` lines; skipped edits are noted inline.
fn apply_edits(entries: &mut Vec<Entry>, edits: &[Edit], now_ms: u64) -> Vec<String> {
    let mut lines = Vec::new();
    let next_id = |entries: &[Entry], kind: &str| -> String {
        let prefix = if kind == "prompt" { "p" } else { "m" };
        let mut n = entries
            .iter()
            .filter(|e| e.kind == kind)
            .filter_map(|e| e.id.trim_start_matches(prefix).parse::<u32>().ok())
            .max()
            .unwrap_or(0)
            + 1;
        while entries.iter().any(|e| e.id == format!("{prefix}{n}")) {
            n += 1;
        }
        format!("{prefix}{n}")
    };
    let mut order: Vec<usize> = (0..edits.len()).collect();
    order.sort_by_key(|&i| match edits[i].action.as_str() {
        "delete" => 0,
        "update" => 1,
        _ => 2,
    });
    for i in order {
        let e = &edits[i];
        let kind_ok = matches!(e.kind.as_str(), "prompt" | "memory");
        let action_ok = matches!(e.action.as_str(), "create" | "update" | "delete");
        if !kind_ok || !action_ok {
            lines.push(format!(
                "  ? skipped [{}] {} — unsupported action/kind",
                e.kind, e.title
            ));
            continue;
        }
        match e.action.as_str() {
            "delete" => {
                if let Some(pos) = entries.iter().position(|x| x.id == e.id) {
                    lines.push(format!(
                        "  - [{}] #{} «{}» — {}",
                        entries[pos].kind, e.id, entries[pos].title, e.reason
                    ));
                    entries.remove(pos);
                } else {
                    lines.push(format!("  ? skipped delete #{} — no such id", e.id));
                }
            }
            "update" => {
                if let Some(pos) = entries.iter().position(|x| x.id == e.id) {
                    let entry = &mut entries[pos];
                    if !e.title.is_empty() {
                        entry.title = e.title.clone();
                    }
                    if !e.content.is_empty() {
                        entry.content = e.content.clone();
                    }
                    entry.updated_ms = now_ms;
                    lines.push(format!(
                        "  ~ [{}] #{} «{}» — {}",
                        entry.kind, entry.id, entry.title, e.reason
                    ));
                } else {
                    lines.push(format!("  ? skipped update #{} — no such id", e.id));
                }
            }
            _ => {
                if e.title.trim().is_empty() || e.content.trim().is_empty() {
                    lines.push(format!(
                        "  ? skipped create [{}] — empty title/content",
                        e.kind
                    ));
                    continue;
                }
                let id = if e.id.is_empty() {
                    next_id(entries, &e.kind)
                } else {
                    e.id.clone()
                };
                entries.push(Entry {
                    id: id.clone(),
                    kind: e.kind.clone(),
                    title: e.title.clone(),
                    content: e.content.clone(),
                    updated_ms: now_ms,
                });
                lines.push(format!(
                    "  + [{}] #{} «{}» — {}",
                    e.kind, id, e.title, e.reason
                ));
            }
        }
    }
    lines
}

// ─────────────────── system-prompt injection ───────────────────

fn cap(text: &str, max: usize) -> String {
    let mut out: String = text.chars().take(max).collect();
    if text.chars().count() > max {
        out.push('…');
    }
    out
}

/// The supplemental-harness block for the system prompt: project entries then
/// global ones (both stores, whatever exists). `None` when both are empty.
pub fn harness_block(cwd: &std::path::Path, agent: &std::path::Path) -> Option<String> {
    let project = load_entries(&Scope::Project.dir(cwd));
    let global = load_entries(&agent.join("harness"));
    if project.is_empty() && global.is_empty() {
        return None;
    }
    let mut block =
        String::from("## Supplemental harness (managed by /refine; do not edit by hand)\n");
    for (label, entries) in [("project", &project), ("global", &global)] {
        if entries.is_empty() {
            continue;
        }
        block.push_str(&format!("### {label}\n"));
        for e in entries {
            block.push_str(&format!(
                "- [{}] {} — {}\n",
                e.kind,
                e.title,
                cap(&e.content, PROMPT_BLOCK_CONTENT_CAP)
            ));
        }
    }
    Some(block.trim_end().to_owned())
}

fn overview(entries: &[Entry]) -> String {
    if entries.is_empty() {
        return "(empty)".to_owned();
    }
    entries
        .iter()
        .map(|e| {
            format!(
                "- #{} [{}] «{}»: {}",
                e.id,
                e.kind,
                e.title,
                cap(&e.content, 200)
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

// ────────────────────────── command ──────────────────────────

fn output(rt: &Runtime, text: &str) {
    rt.send(json!({"type": "command_output", "text": text}));
}

fn usage() -> String {
    "usage: /refine [instructions]        review trajectory, apply harness edits (confirm dialog)\n       /refine global [instructions]  same, into the global store\n       /refine list | show <id>       browse entries (interactive picker)\n       /refine history                applied refinements\n       /refine rollback [id]          undo a refinement (default: last)".to_owned()
}

/// Slash-command entry point.
pub async fn command(rt: &Arc<Runtime>, args: &str) -> Outcome {
    let mut parts = args.split_whitespace();
    let head = parts.next().unwrap_or("");
    match head {
        "" => run(rt, None, Scope::Project).await,
        "global" => {
            let rest = args.trim_start().trim_start_matches("global").trim();
            if rest.is_empty() {
                run(rt, None, Scope::Global).await
            } else {
                run(rt, Some(rest), Scope::Global).await
            }
        }
        "list" => list_cmd(rt).await,
        "show" => {
            let Some(id) = parts.next() else {
                output(rt, &usage());
                return handled();
            };
            show_cmd(rt, id)
        }
        "history" => {
            history_cmd(rt);
            handled()
        }
        "rollback" => rollback_cmd(rt, parts.next()),
        "help" => {
            output(rt, &usage());
            handled()
        }
        _ => {
            let instructions = args.trim();
            run(
                rt,
                (!instructions.is_empty()).then_some(instructions),
                Scope::Project,
            )
            .await
        }
    }
}

fn handled() -> Outcome {
    Outcome::Handled {
        agent_invoked: false,
    }
}

/// One refinement run: snapshot trajectory → planner → (confirm) → apply.
async fn run(rt: &Arc<Runtime>, instructions: Option<&str>, scope: Scope) -> Outcome {
    rt.wait_settled().await;
    let dir = scope.dir(&rt.cwd);

    let messages = rt.messages().await;
    let mut trajectory = crate::slash_extra::session_transcript_for_prompt(&messages);
    if trajectory.chars().count() > TRAJECTORY_CAP_CHARS {
        let tail: String = trajectory
            .chars()
            .skip(trajectory.chars().count() - TRAJECTORY_CAP_CHARS)
            .collect();
        trajectory = format!("(older turns trimmed)\n{tail}");
    }
    if trajectory.trim().is_empty() {
        output(
            rt,
            "refine: this session has no conversation to review yet.",
        );
        return handled();
    }

    let project_entries = load_entries(&Scope::Project.dir(&rt.cwd));
    let global_entries = load_entries(&Scope::Global.dir(&rt.cwd));
    let (own, readonly) = match scope {
        Scope::Project => (&project_entries, &global_entries),
        Scope::Global => (&global_entries, &project_entries),
    };
    let history = load_history(&dir);
    let history_text = if history.is_empty() {
        "(none)".to_owned()
    } else {
        history
            .iter()
            .rev()
            .take(HISTORY_FOR_PROMPT)
            .map(|r| format!("- {}: {}", r.id, r.summary))
            .collect::<Vec<_>>()
            .join("\n")
    };
    let scope_note = match scope {
        Scope::Project => "project (the default store for this run)".to_owned(),
        Scope::Global => "GLOBAL — the run's edits go to the global store".to_owned(),
    };
    let user_message = format!(
        "[scope]\n{scope_note}\n\n[trajectory]\n{trajectory}\n\n[harness — this run's store ({})]\n{}\n\n[harness — other store (read-only for this run)]\n{}\n\n[recent refinements]\n{history_text}\n\n[instructions]\n{}",
        scope.label(),
        overview(own),
        overview(readonly),
        instructions.unwrap_or("(none)"),
    );

    output(
        rt,
        &format!(
            "refine: reviewing the trajectory ({} messages, {} chars) with the planner…",
            messages.len(),
            user_message.chars().count()
        ),
    );

    let target = rt.provider_target();
    let live = match crate::compact::LiveCompleter::from_target(&target) {
        Ok(live) => live,
        Err(e) => {
            output(rt, &format!("refine: no model available: {e}"));
            return handled();
        }
    };
    let (api_key, _) = crate::adapter::fresh_stored_key(
        &live.model.provider,
        live.credential_id,
        live.api_key.clone(),
    )
    .await;
    let options = pa_ai::types::SimpleStreamOptions {
        base: pa_ai::types::StreamOptions {
            max_tokens: Some(REFINE_MAX_TOKENS),
            api_key,
            headers: live.headers.clone().map(|h| h.into_iter().collect()),
            ..pa_ai::types::StreamOptions::default()
        },
        reasoning: live
            .model
            .reasoning
            .then_some(pa_types::ai::ModelThinkingLevel::Medium),
        thinking_budgets: None,
        disable_reasoning: false,
        tool_choice: None,
    };
    let context = pa_types::ai::Context {
        system_prompt: Some(REFINE_SYSTEM_PROMPT.to_owned()),
        messages: vec![pa_types::ai::Message::User(pa_types::ai::UserMessage {
            content: pa_types::ai::UserContent::Text(user_message),
            timestamp: pa_agent::now_ms().cast_unsigned(),
            rest: serde_json::Map::default(),
        })],
        tools: None,
    };
    let planner_text = match pa_ai::complete_simple(&live.model, &context, Some(options)).await {
        Ok(response) => response
            .content
            .iter()
            .filter_map(|b| match b {
                pa_types::ai::AssistantContentBlock::Text(t) => Some(t.text.as_str().to_owned()),
                _ => None,
            })
            .collect::<String>(),
        Err(e) => {
            output(rt, &format!("refine: planner request failed: {e:?}"));
            return handled();
        }
    };
    let proposal = match parse_proposal(&planner_text) {
        Ok(p) => p,
        Err(e) => {
            output(
                rt,
                &format!("refine: the planner answer was unusable — {e}.\nRaw answer (first 500 chars): {}", cap(&planner_text, 500)),
            );
            return handled();
        }
    };
    if proposal.edits.is_empty() {
        output(
            rt,
            &format!(
                "refine: no edits proposed.\nsummary: {}\nrationale: {}",
                proposal.summary, proposal.rationale
            ),
        );
        return handled();
    }

    // Interactive gate (desktop/web + TUI): show the proposal, apply on confirm.
    if crate::slash_extra::dialog::has_ui() {
        let edit_lines = proposal
            .edits
            .iter()
            .map(|e| format!("[{}] {} «{}» — {}", e.action, e.kind, e.title, e.reason))
            .collect::<Vec<_>>()
            .join("\n");
        let message = format!(
            "{}\n\n{}\n\n(expected: {})",
            proposal.summary, edit_lines, proposal.expected_outcome
        );
        match crate::slash_extra::dialog::confirm(
            rt,
            &format!(
                "Apply {} refinement edits ({})?",
                proposal.edits.len(),
                scope.label()
            ),
            &message,
        )
        .await
        {
            Some(true) => {}
            _ => {
                output(rt, "refine: proposal discarded.");
                return handled();
            }
        }
    }

    let now = pa_agent::now_ms().cast_unsigned();
    let rec_id = format!("r{:x}", now);
    let mut entries = own.clone();
    let lines = apply_edits(&mut entries, &proposal.edits, now);
    if let Err(e) = save_entries(&dir, &entries) {
        output(rt, &format!("refine: cannot write the harness store: {e}"));
        return handled();
    }
    let rec = HistoryRecord {
        id: rec_id.clone(),
        ts_ms: now,
        scope,
        summary: proposal.summary.clone(),
        lines: lines.clone(),
        prior: own.clone(),
    };
    if let Err(e) = append_history(&dir, &rec) {
        output(
            rt,
            &format!("refine: applied, but the history record failed: {e}"),
        );
    }

    // Mid-session effect: a hidden notice so the next turn sees the entries.
    let notice = format!(
        "Harness refinement {} applied ({}). Active supplemental harness entries — follow them where they apply:\n{}",
        rec_id,
        scope.label(),
        overview(&entries),
    );
    let notice = crate::notices::message("refinement-notice", &notice, pa_agent::now_ms());
    rt.agent.append_messages(vec![notice]).await;

    output(
        rt,
        &format!(
            "refine {rec_id} — {}\nrationale: {}\nexpected: {}\napplied to {}:\n{}\nrollback: /refine rollback {rec_id}",
            proposal.summary,
            proposal.rationale,
            proposal.expected_outcome,
            scope.label(),
            lines.join("\n"),
        ),
    );
    handled()
}

/// `/refine list`: interactive picker over both stores (text fallback).
async fn list_cmd(rt: &Arc<Runtime>) -> Outcome {
    let project = load_entries(&Scope::Project.dir(&rt.cwd));
    let global = load_entries(&Scope::Global.dir(&rt.cwd));
    if project.is_empty() && global.is_empty() {
        output(
            rt,
            "refine: the harness is empty — run /refine after some work to let the planner distill lessons.",
        );
        return handled();
    }
    if !crate::slash_extra::dialog::has_ui() {
        let mut text = String::from("harness entries:\n");
        for (label, entries) in [("project", &project), ("global", &global)] {
            for e in entries {
                text.push_str(&format!(
                    "[{label}] #{} [{}] «{}»: {}\n",
                    e.id,
                    e.kind,
                    e.title,
                    cap(&e.content, 120)
                ));
            }
        }
        output(rt, text.trim_end());
        return handled();
    }
    // Both stores in one picker; the entry is identified by (scope, id).
    let mut choices: Vec<(Scope, String)> = Vec::new();
    let mut labels: Vec<crate::slash_extra::dialog::Choice> = Vec::new();
    for (scope, entries) in [(Scope::Project, &project), (Scope::Global, &global)] {
        for e in entries {
            choices.push((scope, e.id.clone()));
            labels.push(crate::slash_extra::dialog::Choice::with(
                format!("#{} «{}»", e.id, e.title),
                format!(
                    "[{}] [{}] {}",
                    if scope == Scope::Project {
                        "project"
                    } else {
                        "global"
                    },
                    e.kind,
                    cap(&e.content, 90)
                ),
            ));
        }
    }
    let Some(index) = crate::slash_extra::dialog::select_with(
        rt,
        "Harness entries — /refine",
        &labels,
        crate::slash_extra::dialog::SelectStyle {
            searchable: true,
            ..Default::default()
        },
    )
    .await
    else {
        return handled();
    };
    let (scope, id) = choices[index].clone();
    let actions = ["Show content", "Delete entry", "Cancel"];
    let action_labels: Vec<_> = actions
        .iter()
        .map(|a| crate::slash_extra::dialog::Choice::new(*a))
        .collect();
    let Some(action) = crate::slash_extra::dialog::select_with(
        rt,
        &format!("#{id}"),
        &action_labels,
        crate::slash_extra::dialog::SelectStyle::default(),
    )
    .await
    else {
        return handled();
    };
    match action {
        0 => show_cmd(rt, &id),
        1 => {
            let dir = scope.dir(&rt.cwd);
            let mut entries = load_entries(&dir);
            let Some(pos) = entries.iter().position(|e| e.id == id) else {
                output(rt, &format!("refine: #{id} is gone"));
                return handled();
            };
            let title = entries[pos].title.clone();
            if !crate::slash_extra::dialog::has_ui()
                || crate::slash_extra::dialog::confirm(
                    rt,
                    &format!("Delete #{id} «{title}»?"),
                    "The entry is removed from the harness; rollback via /refine rollback.",
                )
                .await
                    == Some(true)
            {
                let prior = entries.clone();
                entries.remove(pos);
                if let Err(e) = save_entries(&dir, &entries) {
                    output(rt, &format!("refine: cannot write the harness store: {e}"));
                    return handled();
                }
                let now = pa_agent::now_ms().cast_unsigned();
                let rec = HistoryRecord {
                    id: format!("r{:x}", now),
                    ts_ms: now,
                    scope,
                    summary: format!("manual delete #{id} «{title}»"),
                    lines: vec![format!("  - #{id} «{title}» — manual delete")],
                    prior,
                };
                let _ = append_history(&dir, &rec);
                output(
                    rt,
                    &format!("refine: deleted #{id} «{title}» from {}", scope.label()),
                );
            }
            handled()
        }
        _ => handled(),
    }
}

fn show_cmd(rt: &Arc<Runtime>, id: &str) -> Outcome {
    for scope in [Scope::Project, Scope::Global] {
        for e in load_entries(&scope.dir(&rt.cwd)) {
            if e.id == id {
                output(
                    rt,
                    &format!(
                        "#{id} [{}] «{}» ({})\n\n{}",
                        e.kind,
                        e.title,
                        scope.label(),
                        e.content
                    ),
                );
                return handled();
            }
        }
    }
    output(rt, &format!("refine: no harness entry #{id}"));
    handled()
}

fn history_cmd(rt: &Runtime) {
    let mut text = String::new();
    for scope in [Scope::Project, Scope::Global] {
        let history = load_history(&scope.dir(&rt.cwd));
        if history.is_empty() {
            continue;
        }
        text.push_str(&format!("{}:\n", scope.label()));
        for r in history.iter().rev() {
            text.push_str(&format!("  {} — {}\n", r.id, r.summary));
            for line in &r.lines {
                text.push_str(&format!("    {line}\n"));
            }
        }
    }
    if text.is_empty() {
        output(rt, "refine: no refinement history yet.");
    } else {
        output(rt, text.trim_end());
    }
}

fn rollback_cmd(rt: &Runtime, id: Option<&str>) -> Outcome {
    for scope in [Scope::Project, Scope::Global] {
        let dir = scope.dir(&rt.cwd);
        let history = load_history(&dir);
        if history.is_empty() {
            continue;
        }
        let record = match id {
            Some(id) => history.iter().rev().find(|r| r.id == id),
            None => history.last(),
        };
        let Some(record) = record else {
            continue;
        };
        let current = load_entries(&dir);
        if let Err(e) = save_entries(&dir, &record.prior) {
            output(rt, &format!("refine: cannot write the harness store: {e}"));
            return handled();
        }
        let now = pa_agent::now_ms().cast_unsigned();
        let _ = append_history(
            &dir,
            &HistoryRecord {
                id: format!("r{:x}", now),
                ts_ms: now,
                scope,
                summary: format!("rollback of {} ({})", record.id, record.summary),
                lines: vec![format!(
                    "  ↺ restored {} entries from before {}",
                    record.prior.len(),
                    record.id
                )],
                prior: current,
            },
        );
        output(
            rt,
            &format!(
                "refine: rolled back {} — store {} restored to {} entries.",
                record.id,
                scope.label(),
                record.prior.len()
            ),
        );
        return handled();
    }
    output(
        rt,
        &match id {
            Some(id) => format!("refine: no refinement {id} in either store"),
            None => "refine: nothing to roll back".to_owned(),
        },
    );
    handled()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edit(action: &str, kind: &str, id: &str, title: &str, content: &str) -> Edit {
        Edit {
            action: action.into(),
            kind: kind.into(),
            id: id.into(),
            title: title.into(),
            content: content.into(),
            reason: "test".into(),
        }
    }

    #[test]
    fn extract_json_object_finds_balanced_run() {
        assert_eq!(
            extract_json_object(r#"noise before {"a":{"b":"}"},"c":1} after"#),
            Some(r#"{"a":{"b":"}"},"c":1}"#)
        );
        assert_eq!(extract_json_object("no object"), None);
        assert_eq!(extract_json_object(r#"{"unterminated": "#), None);
    }

    #[test]
    fn parse_proposal_reads_edits() {
        let p = parse_proposal(
            r#"prefix {"summary":"s","rationale":"r","expectedOutcome":"e","edits":[
                {"action":"create","kind":"memory","title":"T","content":"C","reason":"why"}]}"#,
        )
        .expect("parses");
        assert_eq!(p.summary, "s");
        assert_eq!(p.edits.len(), 1);
        assert_eq!(p.edits[0].kind, "memory");
        assert!(parse_proposal("not json at all").is_err());
    }

    #[test]
    fn apply_edits_deletes_updates_creates_in_order() {
        let mut entries = vec![Entry {
            id: "m1".into(),
            kind: "memory".into(),
            title: "old".into(),
            content: "old body".into(),
            updated_ms: 1,
        }];
        let edits = vec![
            edit("create", "prompt", "", "P1", "note body"),
            edit("update", "memory", "m1", "new", "new body"),
            edit("create", "bogus", "", "x", "y"),
        ];
        let lines = apply_edits(&mut entries, &edits, 42);
        assert_eq!(entries.len(), 2, "{entries:?}");
        assert!(entries.iter().any(|e| e.id == "p1" && e.title == "P1"));
        assert!(entries.iter().any(|e| e.id == "m1" && e.title == "new"));
        assert!(lines.iter().any(|l| l.contains("? skipped")));
        // Delete by id after the create minted a fresh id.
        let lines = apply_edits(&mut entries, &[edit("delete", "memory", "m1", "", "")], 43);
        assert_eq!(entries.len(), 1);
        assert!(lines[0].contains("- [memory] #m1"));
    }

    #[test]
    fn harness_store_round_trip_and_rollback_snapshot() {
        let dir = std::env::temp_dir().join(format!("omp-refine-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let entries = vec![Entry {
            id: "m1".into(),
            kind: "memory".into(),
            title: "t".into(),
            content: "c".into(),
            updated_ms: 7,
        }];
        save_entries(&dir, &entries).expect("save");
        assert_eq!(load_entries(&dir), entries);
        append_history(
            &dir,
            &HistoryRecord {
                id: "r1".into(),
                ts_ms: 8,
                scope: Scope::Project,
                summary: "first".into(),
                lines: vec!["+ m1".into()],
                prior: Vec::new(),
            },
        )
        .expect("history");
        let history = load_history(&dir);
        assert_eq!(history.len(), 1);
        assert!(history[0].prior.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
