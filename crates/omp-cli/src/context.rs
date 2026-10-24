//! Context strategies (`context.strategy` / `--context-strategy`, design §3.1).
//!
//! - `append` (default): the wire carries the whole history; this module is inert and
//!   request bodies are byte-identical to a build without it.
//! - `clm`: the model maintains its own working memory through the typed tool
//!   `context_edit{ops:[{op:replace|append|evict, slot, text?}]}`. Raw history before the
//!   latest accepted edit leaves the wire; what the model wants to keep lives in *slots*.
//! - `clm-host`: `clm` plus a host enforcer — when the slots exceed the char budget the host
//!   asks the model once to compact, then mechanically evicts the oldest slots.
//!
//! Wire layout (cache-aware): `[system + tools] [pinned: leading user messages] [slots] [tail]`.
//! The slot table is a *pure fold over the history* (accepted `context_edit` calls in order), so
//! it is deterministic, resume-safe, and needs no side state: between edits the wire only grows
//! at the end (prefix-cache friendly); an edit changes bytes only from the first changed slot
//! on and never touches the pinned prefix. Edits take effect at the next request (turn
//! boundary): the tool only validates and acknowledges.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;

use anyhow::Result;
use pa_agent::abort::AbortSignal;
use pa_agent::types::{
    AgentMessage, AgentTool, AgentToolResult, AgentToolUpdateCallback, AssistantContent, Message,
    ToolResultContent, UserContent, UserMessage,
};
use pa_agent::BoxFut;
use serde_json::{json, Value};

/// First bytes of the slots message; `pa-ai` places an Anthropic cache breakpoint on it.
pub const SLOTS_MARKER: &str = "<context-slots>";
pub const TOOL_NAME: &str = "context_edit";

static ACTIVE: std::sync::OnceLock<Kind> = std::sync::OnceLock::new();

/// Process-wide active strategy (set once at startup; read by the session writer so every
/// session file — including `new_session` ones — records it).
pub fn set_active(k: Kind) {
    let _ = ACTIVE.set(k);
}

#[must_use]
pub fn active() -> Kind {
    ACTIVE.get().copied().unwrap_or_default()
}

/// The strategy `convert_to_llm` routes through right now: the startup one
/// (`--context-strategy` / `context.strategy`) or, while an ultra mode with
/// CLM runs, the one it switched on. `None` = `append` (wire untouched).
struct Live {
    shared: Arc<Shared>,
    /// Switched on by an ultra mode (`ultralong` / `hypercode`), not at startup.
    by_ultra: bool,
}

static LIVE: std::sync::RwLock<Option<Live>> = std::sync::RwLock::new(None);

/// Install (or clear) the live strategy `convert_to_llm` reads per request.
pub fn set_live(shared: Option<Arc<Shared>>, by_ultra: bool) {
    *LIVE
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner) =
        shared.map(|shared| Live { shared, by_ultra });
}

/// The live strategy, `None` for `append`.
#[must_use]
pub fn live() -> Option<Arc<Shared>> {
    LIVE.read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .as_ref()
        .map(|l| l.shared.clone())
}

/// CLM is on because an ultra mode switched it on: extensions that slim the
/// context themselves (`context-slim`) must stand aside.
#[must_use]
pub fn ultra_owned() -> bool {
    LIVE.read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .as_ref()
        .is_some_and(|l| l.by_ultra)
}

/// Extensions skipped on the core's `context` event while [`ultra_owned`].
#[must_use]
pub fn skip_extensions() -> Vec<String> {
    if ultra_owned() {
        vec!["context-slim".to_owned()]
    } else {
        Vec::new()
    }
}

/// The `clm-host` strategy an ultra mode switches on: `rule` policy with the
/// `context.policy.*` thresholds (or their defaults), whatever policy kind the
/// settings name.
#[must_use]
pub fn ultra_shared(settings: &omp_settings::Settings) -> Arc<Shared> {
    let env = |k: &str| {
        if k == "OMP_CONTEXT_POLICY" {
            Some("rule".to_owned())
        } else {
            std::env::var(k).ok()
        }
    };
    let policy = crate::context_policy::Policy::resolve(settings, &env)
        .ok()
        .flatten();
    Shared::with_policy(Kind::ClmHost, Limits::from_settings(settings), policy)
}

/// Session-file entry recording a non-default strategy (`None` for `append`, so default
/// session files stay byte-identical).
#[must_use]
pub fn session_entry() -> Option<Value> {
    let k = active();
    (k != Kind::Append)
        .then(|| json!({"customType": "context-strategy", "data": {"strategy": k.as_str()}}))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Kind {
    #[default]
    Append,
    Clm,
    ClmHost,
}

impl Kind {
    /// # Errors
    /// Unknown strategy name.
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.trim() {
            "append" => Ok(Self::Append),
            "clm" => Ok(Self::Clm),
            "clm-host" => Ok(Self::ClmHost),
            other => Err(format!(
                "unknown context strategy {other:?} (expected append, clm or clm-host)"
            )),
        }
    }

    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Append => "append",
            Self::Clm => "clm",
            Self::ClmHost => "clm-host",
        }
    }

    /// Flag wins over the `context.strategy` setting; default `append`.
    ///
    /// # Errors
    /// Unknown strategy name in the flag or the setting.
    pub fn resolve(flag: Option<&str>, settings: &omp_settings::Settings) -> Result<Self, String> {
        match flag {
            Some(f) => Self::parse(f),
            None => settings
                .get_string("context.strategy")
                .map_or(Ok(Self::Append), |s| Self::parse(&s)),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Limits {
    pub max_slots: usize,
    pub max_slot_chars: usize,
    /// Host-enforcer budget over all slots (`clm-host`).
    pub budget_chars: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_slots: 16,
            max_slot_chars: 8_000,
            budget_chars: 6_000,
        }
    }
}

impl Limits {
    #[must_use]
    pub fn from_settings(s: &omp_settings::Settings) -> Self {
        let d = Self::default();
        let num = |id: &str, dv: usize| {
            s.get_number(id)
                .filter(|v| *v >= 1.0)
                .map_or(dv, |v| v as usize)
        };
        Self {
            max_slots: num("context.clm.maxSlots", d.max_slots),
            max_slot_chars: num("context.clm.maxSlotChars", d.max_slot_chars),
            budget_chars: num("context.clm.slotBudgetChars", d.budget_chars),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    Replace { slot: String, text: String },
    Append { slot: String, text: String },
    Evict { slot: String },
}

/// Ordered slot table: creation order is the wire order, so editing slot `k` leaves slots
/// `< k` byte-identical.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SlotTable {
    pub slots: Vec<(String, String)>,
}

impl SlotTable {
    #[must_use]
    pub fn chars(&self) -> usize {
        self.slots.iter().map(|(n, t)| n.len() + t.len()).sum()
    }

    fn pos(&self, name: &str) -> Option<usize> {
        self.slots.iter().position(|(n, _)| n == name)
    }

    /// Render the slots message body (`SLOTS_MARKER` first).
    #[must_use]
    pub fn render(&self, host_evicted: &[String]) -> String {
        let mut out = String::from(SLOTS_MARKER);
        out.push_str(
            "\nYour working memory, maintained with the context_edit tool. Raw conversation \
             before your latest context_edit call is no longer shown; only the slots below \
             persist.\n",
        );
        if self.slots.is_empty() {
            out.push_str("(no slots)\n");
        }
        for (name, text) in &self.slots {
            out.push_str(&format!("<slot id=\"{name}\">\n{text}\n</slot>\n"));
        }
        if !host_evicted.is_empty() {
            out.push_str(&format!(
                "(host evicted over-budget slots: {})\n",
                host_evicted.join(", ")
            ));
        }
        out.push_str("</context-slots>");
        out
    }
}

fn valid_slot_name(n: &str) -> bool {
    !n.is_empty()
        && n.len() <= 32
        && n.bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

/// Parse the tool arguments into ops (shape validation only).
///
/// # Errors
/// A message naming the offending op.
pub fn parse_ops(args: &Value) -> Result<Vec<Op>, String> {
    let ops = args
        .get("ops")
        .and_then(Value::as_array)
        .ok_or("`ops` must be an array of {op, slot, text?}")?;
    if ops.is_empty() {
        return Err("`ops` is empty".into());
    }
    let mut out = Vec::with_capacity(ops.len());
    for (i, o) in ops.iter().enumerate() {
        let kind = o.get("op").and_then(Value::as_str).unwrap_or("");
        let slot = o
            .get("slot")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("ops[{i}]: `slot` is required"))?
            .to_owned();
        let text = o.get("text").and_then(Value::as_str).map(str::to_owned);
        out.push(match (kind, text) {
            ("replace", Some(text)) => Op::Replace { slot, text },
            ("append", Some(text)) => Op::Append { slot, text },
            ("replace" | "append", None) => {
                return Err(format!("ops[{i}]: `{kind}` requires `text`"))
            }
            ("evict", None) => Op::Evict { slot },
            ("evict", Some(_)) => return Err(format!("ops[{i}]: `evict` takes no `text`")),
            (other, _) => {
                return Err(format!(
                    "ops[{i}]: unknown op {other:?} (replace|append|evict)"
                ))
            }
        });
    }
    Ok(out)
}

/// Validate and apply `ops` atomically: on error the table is untouched.
///
/// # Errors
/// A message naming the offending op (unknown slot, bad name, size limits).
pub fn apply_ops(table: &mut SlotTable, ops: &[Op], lim: &Limits) -> Result<(), String> {
    let mut t = table.clone();
    for (i, op) in ops.iter().enumerate() {
        match op {
            Op::Replace { slot, text } => {
                let p = t.pos(slot).ok_or_else(|| {
                    format!(
                        "ops[{i}]: unknown slot {slot:?}; create it with `append` (slots: {})",
                        names(&t)
                    )
                })?;
                t.slots[p].1 = text.clone();
            }
            Op::Append { slot, text } => {
                if let Some(p) = t.pos(slot) {
                    t.slots[p].1.push('\n');
                    t.slots[p].1.push_str(text);
                } else {
                    if !valid_slot_name(slot) {
                        return Err(format!(
                            "ops[{i}]: invalid slot name {slot:?} ([A-Za-z0-9_-], 1-32 chars)"
                        ));
                    }
                    if t.slots.len() >= lim.max_slots {
                        return Err(format!(
                            "ops[{i}]: slot table full ({} slots); evict one first",
                            lim.max_slots
                        ));
                    }
                    t.slots.push((slot.clone(), text.clone()));
                }
            }
            Op::Evict { slot } => {
                let p = t.pos(slot).ok_or_else(|| {
                    format!("ops[{i}]: unknown slot {slot:?} (slots: {})", names(&t))
                })?;
                t.slots.remove(p);
            }
        }
        if let Some((n, s)) = t.slots.iter().find(|(_, s)| s.len() > lim.max_slot_chars) {
            return Err(format!(
                "ops[{i}]: slot {n:?} is {} chars, limit {}",
                s.len(),
                lim.max_slot_chars
            ));
        }
    }
    *table = t;
    Ok(())
}

fn names(t: &SlotTable) -> String {
    if t.slots.is_empty() {
        "none".into()
    } else {
        t.slots
            .iter()
            .map(|(n, _)| n.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// What the wire looks like for one request.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Plan {
    /// Messages `[..pinned_end]` (everything before the first assistant turn) stay verbatim.
    pub pinned_end: usize,
    /// First raw message kept after the slots; `None` while no edit was accepted.
    pub tail_start: Option<usize>,
    /// Slot table after every accepted edit (what `context_edit` validates against).
    pub table: SlotTable,
    /// Slot table as rendered on the wire: only edits already folded (user-turn boundary).
    pub shown: SlotTable,
    pub host_evicted: Vec<String>,
    /// Host-enforcer ask to append to this request (`clm-host` only).
    pub notice: Option<String>,
}

fn is_edit_call(c: &AssistantContent) -> Option<(&str, &Value)> {
    match c {
        AssistantContent::ToolCall(tc) if tc.name == TOOL_NAME => Some((&tc.id, &tc.arguments)),
        _ => None,
    }
}

/// Fold the history into the slot table and wire plan (pure).
#[must_use]
pub fn plan(messages: &[AgentMessage], kind: Kind, lim: &Limits) -> Plan {
    let pinned_end = messages
        .iter()
        .position(|m| matches!(m, AgentMessage::Standard(Message::Assistant(_))))
        .unwrap_or(messages.len());
    let mut p = Plan {
        pinned_end,
        ..Plan::default()
    };
    let ok_result = |id: &str| {
        messages.iter().any(|m| {
            matches!(m, AgentMessage::Standard(Message::ToolResult(tr))
                if tr.tool_call_id == id && !tr.is_error)
        })
    };
    // An accepted edit takes effect at the next *user-turn* boundary: the requests of the run
    // that carries it keep the raw history byte-for-byte (cache hit on the whole prefix); the
    // fold happens once, at the first request after the next user message.
    let last_user = messages
        .iter()
        .rposition(|m| matches!(m, AgentMessage::Standard(Message::User(_))))
        .unwrap_or(0);
    let mut ask_pending = false;
    let mut evicted_all: Vec<String> = Vec::new();
    for (i, m) in messages.iter().enumerate().skip(pinned_end) {
        let AgentMessage::Standard(Message::Assistant(a)) = m else {
            continue;
        };
        for c in &a.content {
            let Some((id, args)) = is_edit_call(c) else {
                continue;
            };
            if !ok_result(id) {
                continue;
            }
            let Ok(ops) = parse_ops(args) else { continue };
            if apply_ops(&mut p.table, &ops, lim).is_ok() && i < last_user {
                p.tail_start = Some(i);
            }
        }
        if kind == Kind::ClmHost && i < last_user && !p.table.slots.is_empty() {
            let over = p.table.chars() > lim.budget_chars;
            if ask_pending {
                if over {
                    while p.table.chars() > lim.budget_chars && !p.table.slots.is_empty() {
                        evicted_all.push(p.table.slots.remove(0).0);
                    }
                }
                ask_pending = false;
            } else if over {
                ask_pending = true;
            }
        }
        if p.tail_start.is_some() && i < last_user {
            p.shown = p.table.clone();
            p.host_evicted = evicted_all.clone();
        }
    }
    if ask_pending {
        p.notice = Some(format!(
            "HOST ENFORCER: your context slots total {} chars, budget {}. Compact them now with \
             context_edit (replace with shorter text or evict stale slots), keeping every \
             task-critical fact. If still over budget after your next turn the host will evict \
             the oldest slots.",
            p.table.chars(),
            lim.budget_chars
        ));
    }
    p
}

fn user_msg(text: String) -> AgentMessage {
    AgentMessage::Standard(Message::User(UserMessage {
        content: UserContent::Text(text),
        timestamp: 0,
    }))
}

/// Wire history for the strategy; `Append` returns the input untouched.
#[must_use]
pub fn build_wire(
    messages: Vec<AgentMessage>,
    kind: Kind,
    lim: &Limits,
) -> (Vec<AgentMessage>, Plan) {
    if kind == Kind::Append {
        return (messages, Plan::default());
    }
    let p = plan(&messages, kind, lim);
    let Some(tail) = p.tail_start else {
        return (messages, p);
    };
    let mut out = Vec::with_capacity(messages.len() - tail + p.pinned_end + 2);
    out.extend(messages[..p.pinned_end].iter().cloned());
    out.push(user_msg(p.shown.render(&p.host_evicted)));
    out.extend(messages[tail..].iter().cloned());
    if let Some(n) = &p.notice {
        out.push(user_msg(n.clone()));
    }
    (out, p)
}

/// Shared between the tool (validates against what the model last saw) and the wire builder.
pub struct Shared {
    pub kind: Kind,
    pub limits: Limits,
    seen: Mutex<SlotTable>,
    /// Opt-in `context.policy` (None = legacy: no host notice, edits always accepted).
    policy: Option<crate::context_policy::Policy>,
    /// Policy verdict for the current user turn; `context_edit` is rejected while `false`.
    allowed: AtomicBool,
    /// Offloaded blocks: message index -> the reference that replaces its text on every request.
    offloaded: Mutex<Vec<(usize, String)>>,
}

impl Shared {
    #[cfg(test)]
    #[must_use]
    pub fn new(kind: Kind, limits: Limits) -> Arc<Self> {
        Self::with_policy(kind, limits, None)
    }

    #[must_use]
    pub fn with_policy(
        kind: Kind,
        limits: Limits,
        policy: Option<crate::context_policy::Policy>,
    ) -> Arc<Self> {
        Arc::new(Self {
            kind,
            limits,
            seen: Mutex::new(SlotTable::default()),
            policy,
            allowed: AtomicBool::new(true),
            offloaded: Mutex::new(Vec::new()),
        })
    }

    /// `convert_to_llm` stage: rewrite history per strategy, remember the slot table.
    #[must_use]
    pub fn wire(&self, mut messages: Vec<AgentMessage>) -> Vec<AgentMessage> {
        let Some(pol) = self.policy.as_ref() else {
            let (out, p) = build_wire(messages, self.kind, &self.limits);
            *self.seen.lock() = p.table;
            return out;
        };
        let mut off = self.offloaded.lock();
        let table = plan(&messages, self.kind, &self.limits).table;
        let mut st = policy_state(&messages, &table);
        st.offloads_total = off.len();
        if let Some((i, n)) = largest_old_block(&messages, &off) {
            st.largest_block_chars = n;
            let f = block_features(&messages, i);
            st.block_age_turns = f.0;
            st.block_restated_pct = f.1;
            st.block_since_ref_turns = f.2;
        }
        st.offload_reads = offload_reads(&messages, &off);
        let (action, fresh) = pol.decide(&st);
        if action == crate::context_policy::Action::RlmOffload && fresh {
            if let Some((i, _)) = largest_old_block(&messages, &off) {
                if let Some(reference) = offload_block(&messages[i], i) {
                    off.push((i, reference));
                }
            }
        }
        for (i, reference) in off.iter() {
            if let Some(AgentMessage::Standard(m)) = messages.get_mut(*i) {
                replace_block_text(m, reference);
            }
        }
        drop(off);
        let edit = action == crate::context_policy::Action::ClmEdit;
        let done = edited_this_turn(&messages);
        let (mut out, p) = build_wire(messages, self.kind, &self.limits);
        *self.seen.lock() = p.table;
        self.allowed.store(edit, Ordering::Relaxed);
        if edit && !done {
            out.push(user_msg(crate::context_policy::MAINT_NOTICE.to_owned()));
        }
        out
    }
}

/// Text of a user message / tool result (what an offload would move out of the wire).
fn block_text(m: &AgentMessage) -> Option<String> {
    match m {
        AgentMessage::Standard(Message::User(u)) => Some(user_text(&u.content)),
        AgentMessage::Standard(Message::ToolResult(tr)) => Some(result_text(&tr.content)),
        _ => None,
    }
}

/// Largest not-yet-offloaded user/tool-result block before the newest user message, outside
/// the pinned prefix, that is big enough to be worth a reference. `(index, chars)`.
fn largest_old_block(messages: &[AgentMessage], off: &[(usize, String)]) -> Option<(usize, usize)> {
    let pinned_end = messages
        .iter()
        .position(|m| matches!(m, AgentMessage::Standard(Message::Assistant(_))))
        .unwrap_or(messages.len());
    let last_user = user_turns(messages).last().copied().unwrap_or(0);
    messages
        .iter()
        .enumerate()
        .filter(|(i, _)| *i >= pinned_end && *i < last_user && off.iter().all(|(j, _)| j != i))
        .filter_map(|(i, m)| block_text(m).map(|t| (i, t.chars().count())))
        .filter(|(_, n)| *n >= crate::context_policy::OFFLOAD_MIN_CHARS)
        .max_by_key(|(_, n)| *n)
}

/// Distinctive tokens of a block: identifiers/numbers (>= 6 chars with a digit), at most 400.
fn distinctive_tokens(text: &str) -> std::collections::HashSet<String> {
    let mut out = std::collections::HashSet::new();
    for t in text.split(|c: char| !(c.is_alphanumeric() || c == '-' || c == '_' || c == '.')) {
        let t = t.trim_matches(|c| c == '.' || c == '-');
        if t.len() >= 6 && t.chars().any(|c| c.is_ascii_digit()) {
            out.insert(t.to_owned());
            if out.len() >= 400 {
                break;
            }
        }
    }
    out
}

/// Text the model itself wrote in an assistant message (text blocks + tool-call arguments).
fn assistant_written(m: &AgentMessage) -> Option<String> {
    let AgentMessage::Standard(Message::Assistant(a)) = m else {
        return None;
    };
    let mut s = String::new();
    for c in &a.content {
        match c {
            AssistantContent::Text(t) => s.push_str(&t.text),
            AssistantContent::ToolCall(tc) => s.push_str(&tc.arguments.to_string()),
            AssistantContent::Thinking(_) => {}
        }
        s.push('\n');
    }
    Some(s)
}

/// `(age_turns, restated_pct, since_ref_turns)` of the block at message index `idx`.
fn block_features(messages: &[AgentMessage], idx: usize) -> (usize, usize, usize) {
    let turns = user_turns(messages);
    let turn_of = |i: usize| turns.iter().filter(|t| **t <= i).count();
    let now = turns.len();
    let born = turn_of(idx);
    let age = now.saturating_sub(born);
    let Some(text) = block_text(&messages[idx]) else {
        return (age, 0, age);
    };
    let d = distinctive_tokens(&text);
    if d.is_empty() {
        return (age, 0, age);
    }
    let mut seen = std::collections::HashSet::new();
    let mut last_ref_turn = None;
    for (i, m) in messages.iter().enumerate().skip(idx + 1) {
        let Some(w) = assistant_written(m) else {
            continue;
        };
        let hits: std::collections::HashSet<&String> =
            d.iter().filter(|t| w.contains(t.as_str())).collect();
        if hits.len() >= 2 {
            last_ref_turn = Some(turn_of(i));
        }
        seen.extend(hits.into_iter().cloned());
    }
    let since = last_ref_turn.map_or(age, |t| now.saturating_sub(t));
    (age, 100 * seen.len() / d.len(), since)
}

/// Tool calls whose arguments mention an offloaded block file.
fn offload_reads(messages: &[AgentMessage], off: &[(usize, String)]) -> usize {
    if off.is_empty() {
        return 0;
    }
    let names: Vec<String> = off.iter().map(|(i, _)| format!("block-{i}.txt")).collect();
    messages
        .iter()
        .filter_map(|m| match m {
            AgentMessage::Standard(Message::Assistant(a)) => Some(a),
            _ => None,
        })
        .flat_map(|a| a.content.iter())
        .filter(|c| match c {
            AssistantContent::ToolCall(tc) => {
                let args = tc.arguments.to_string();
                names.iter().any(|n| args.contains(n.as_str()))
            }
            _ => false,
        })
        .count()
}

/// Write the block to the offload dir and return the one-line reference that replaces it.
fn offload_block(m: &AgentMessage, idx: usize) -> Option<String> {
    let text = block_text(m)?;
    let dir = std::env::var_os("OMP_CONTEXT_OFFLOAD_DIR").map_or_else(
        || {
            std::env::current_dir()
                .unwrap_or_default()
                .join(".omp-offload")
        },
        std::path::PathBuf::from,
    );
    std::fs::create_dir_all(&dir).ok()?;
    let path = dir.join(format!("block-{idx}.txt"));
    std::fs::write(&path, &text).ok()?;
    let summary = std::env::var("OMP_CONTEXT_OFFLOAD_SUMMARY")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|n| *n > 0)
        .map(|n| extractive_summary(&text, n))
        .unwrap_or_default();
    Some(format!(
        "[offloaded: {} chars of earlier content are NOT in your context; they are in the file {}. \
         Read them on demand with the eval tool, e.g. `ctx = open({:?}).read()` then \
         `llm_query(question, context=ctx, model=\"{}\")` or slice/grep `ctx` directly.{}]",
        text.chars().count(),
        path.display(),
        path.display().to_string(),
        std::env::var("OMP_CONTEXT_OFFLOAD_MODEL").unwrap_or_else(|_| "smol".to_owned()),
        summary,
    ))
}

/// Deterministic extractive summary (LCM-lite): the head and tail of the block, whitespace-flattened,
/// at most `n` chars in total.
fn extractive_summary(text: &str, n: usize) -> String {
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let chars: Vec<char> = flat.chars().collect();
    if chars.len() <= n {
        return format!(" Content: {flat}");
    }
    let head = n * 2 / 3;
    let tail = n - head;
    let h: String = chars[..head].iter().collect();
    let t: String = chars[chars.len() - tail..].iter().collect();
    format!(" Summary (head...tail): {h} ... {t}")
}

fn replace_block_text(m: &mut Message, reference: &str) {
    match m {
        Message::User(u) => u.content = UserContent::Text(reference.to_owned()),
        Message::ToolResult(tr) => tr.content = vec![ToolResultContent::text(reference)],
        Message::Assistant(_) | Message::Developer(_) => {}
    }
}

fn user_text(c: &UserContent) -> String {
    match c {
        UserContent::Text(t) => t.clone(),
        UserContent::Parts(ps) => ps
            .iter()
            .filter_map(|p| match p {
                pa_agent::types::UserPart::Text(t) => Some(t.text.as_str()),
                pa_agent::types::UserPart::Image(_) => None,
            })
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

fn user_turns(messages: &[AgentMessage]) -> Vec<usize> {
    messages
        .iter()
        .enumerate()
        .filter(|(_, m)| matches!(m, AgentMessage::Standard(Message::User(_))))
        .map(|(i, _)| i)
        .collect()
}

/// Accepted `context_edit` calls in `messages`, as indices of the assistant messages carrying them.
fn accepted_edits(messages: &[AgentMessage]) -> Vec<usize> {
    let ok = |id: &str| {
        messages.iter().any(|m| {
            matches!(m, AgentMessage::Standard(Message::ToolResult(tr)) if tr.tool_call_id == id && !tr.is_error)
        })
    };
    messages
        .iter()
        .enumerate()
        .filter_map(|(i, m)| match m {
            AgentMessage::Standard(Message::Assistant(a)) => a
                .content
                .iter()
                .any(|c| {
                    is_edit_call(c).is_some_and(|(id, args)| ok(id) && parse_ops(args).is_ok())
                })
                .then_some(i),
            _ => None,
        })
        .collect()
}

/// Has the current user turn already produced an accepted edit (no second notice needed)?
fn edited_this_turn(messages: &[AgentMessage]) -> bool {
    let last_user = user_turns(messages).last().copied().unwrap_or(0);
    accepted_edits(messages).iter().any(|i| *i > last_user)
}

/// Controller state at the boundary before the newest user message is answered.
fn policy_state(messages: &[AgentMessage], table: &SlotTable) -> crate::context_policy::State {
    let turns = user_turns(messages);
    let step = turns.len().saturating_sub(1);
    let edits = accepted_edits(messages);
    let turn_of = |msg_idx: usize| {
        turns
            .iter()
            .filter(|t| **t <= msg_idx)
            .count()
            .saturating_sub(1)
    };
    let steps_since_edit = edits.last().map(|i| step.saturating_sub(turn_of(*i)));
    let (mut input, mut cached, mut last_usage, mut last_idx) = (0u64, 0u64, 0u64, None);
    for (i, m) in messages.iter().enumerate() {
        if let AgentMessage::Standard(Message::Assistant(a)) = m {
            let u = &a.usage;
            input += u.input + u.cache_read + u.cache_write;
            cached += u.cache_read;
            last_usage = u.input + u.cache_read + u.cache_write + u.output;
            last_idx = Some(i);
        }
    }
    let new_chars: usize = messages
        .iter()
        .skip(last_idx.map_or(0, |i| i + 1))
        .map(|m| match m {
            AgentMessage::Standard(Message::User(u)) => user_text(&u.content).chars().count(),
            AgentMessage::Standard(Message::ToolResult(tr)) => {
                result_text(&tr.content).chars().count()
            }
            _ => 0,
        })
        .sum();
    let excerpt = turns
        .last()
        .and_then(|i| match &messages[*i] {
            AgentMessage::Standard(Message::User(u)) => {
                Some(crate::context_policy::excerpt_of(&user_text(&u.content)))
            }
            _ => None,
        })
        .unwrap_or_default();
    crate::context_policy::State {
        step,
        history_tokens: last_usage as usize + new_chars / 4,
        slots_chars: table.chars(),
        slots_count: table.slots.len(),
        steps_since_edit,
        edits_total: edits.len(),
        largest_block_chars: 0,
        block_age_turns: 0,
        block_restated_pct: 0,
        block_since_ref_turns: 0,
        offloads_total: 0,
        offload_reads: 0,
        cache_hit_pct: if input == 0 {
            0
        } else {
            (100 * cached / input) as usize
        },
        excerpt,
    }
}

pub struct ContextEditTool {
    shared: Arc<Shared>,
    schema: Value,
    description: String,
}

#[must_use]
pub fn tool(shared: Arc<Shared>) -> Arc<dyn AgentTool> {
    let lim = shared.limits;
    Arc::new(ContextEditTool {
        description: format!(
            "Edit your working-memory slots (your context is managed). Slots are named notes \
             shown to you in the <context-slots> message; after you call this tool, raw conversation before this \
             call is dropped from your context at your next user turn, so \
             first copy everything you must keep (facts, codes, values, instructions, \
             operator notices) into slots. Ops are applied in order and atomically: \
             `append` adds text to a slot (creating it if new; name [A-Za-z0-9_-], <=32 chars; \
             max {} slots, {} chars each), `replace` overwrites an existing slot, `evict` \
             deletes one. An invalid op rejects the whole call with an error and changes \
             nothing. Edits take effect on your next turn.{}",
            lim.max_slots,
            lim.max_slot_chars,
            if shared.policy.is_some() {
                " Call it ONLY when the current request ends with a `CONTEXT MAINTENANCE` notice; \
                 otherwise do not call it (the host rejects it and the call is wasted)."
            } else {
                ""
            }
        ),
        shared,
        schema: json!({
            "type": "object",
            "properties": {
                "ops": {
                    "type": "array",
                    "description": "Edit operations, applied in order.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "op": {"type": "string", "enum": ["replace", "append", "evict"]},
                            "slot": {"type": "string", "description": "Slot name"},
                            "text": {"type": "string", "description": "Text (replace/append only)"}
                        },
                        "required": ["op", "slot"],
                        "additionalProperties": false
                    }
                }
            },
            "required": ["ops"],
            "additionalProperties": false
        }),
    })
}

impl AgentTool for ContextEditTool {
    fn name(&self) -> &'static str {
        TOOL_NAME
    }
    fn description(&self) -> &str {
        &self.description
    }
    fn parameters(&self) -> &Value {
        &self.schema
    }
    fn label(&self) -> &'static str {
        "Context edit"
    }
    fn execute(
        self: Arc<Self>,
        _tool_call_id: String,
        params: Value,
        _signal: AbortSignal,
        _on_update: AgentToolUpdateCallback,
    ) -> BoxFut<'static, Result<AgentToolResult>> {
        Box::pin(async move {
            if !self.shared.allowed.load(Ordering::Relaxed) {
                return Ok(AgentToolResult::error(
                    "context_edit rejected, nothing changed: no context maintenance is due this turn \
                     (context policy); continue the task.",
                ));
            }
            let mut table = self.shared.seen.lock().clone();
            let verdict = parse_ops(&params).and_then(|ops| {
                apply_ops(&mut table, &ops, &self.shared.limits).map(|()| ops.len())
            });
            Ok(match verdict {
                Ok(n) => AgentToolResult::text(format!(
                    "ok: {n} op(s) accepted; slots after edit: {}. Takes effect at your next user turn (this turn's raw history stays visible).",
                    names(&table)
                )),
                Err(e) => AgentToolResult::error(format!("context_edit rejected, nothing changed: {e}")),
            })
        })
    }
}

/// `true` when a tool result of this call carries an error (test helper symmetry).
#[allow(dead_code)]
fn result_text(c: &[ToolResultContent]) -> String {
    c.iter()
        .filter_map(|c| match c {
            ToolResultContent::Text(t) => Some(t.text.as_str()),
            ToolResultContent::Image(_) => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use pa_agent::types::ToolResultMessage;

    fn user(t: &str) -> AgentMessage {
        user_msg(t.to_owned())
    }

    fn asst(content: Value, stop: &str) -> AgentMessage {
        let v = json!({"role":"assistant","content":content,"api":"x","provider":"x","model":"x",
            "usage":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"totalTokens":0,
                     "cost":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"total":0}},
            "stopReason":stop,"timestamp":0});
        AgentMessage::Standard(serde_json::from_value::<Message>(v).expect("assistant"))
    }

    fn assistant_edit(id: &str, ops: Value) -> AgentMessage {
        asst(
            json!([{"type":"toolCall","id":id,"name":TOOL_NAME,"arguments":{"ops":ops}}]),
            "toolUse",
        )
    }

    fn assistant_text(t: &str) -> AgentMessage {
        asst(json!([{"type":"text","text":t}]), "stop")
    }

    fn result(id: &str, err: bool) -> AgentMessage {
        AgentMessage::Standard(Message::ToolResult(ToolResultMessage {
            tool_call_id: id.into(),
            tool_name: TOOL_NAME.into(),
            content: vec![ToolResultContent::text("r")],
            details: None,
            is_error: err,
            timestamp: 0,
        }))
    }

    fn wire_json(m: &[AgentMessage]) -> String {
        serde_json::to_string(
            &m.iter()
                .map(|x| serde_json::to_value(x).unwrap())
                .collect::<Vec<_>>(),
        )
        .unwrap()
    }

    fn lim() -> Limits {
        Limits {
            max_slots: 3,
            max_slot_chars: 100,
            budget_chars: 60,
        }
    }

    #[test]
    fn append_is_identity() {
        let h = vec![
            user("task"),
            assistant_edit("c1", json!([{"op":"append","slot":"a","text":"x"}])),
            result("c1", false),
        ];
        let before = wire_json(&h);
        let (out, _) = build_wire(h, Kind::Append, &lim());
        assert_eq!(wire_json(&out), before);
    }

    #[test]
    fn no_edit_means_full_history() {
        let h = vec![user("task"), assistant_text("hi"), user("more")];
        let before = wire_json(&h);
        let (out, _) = build_wire(h, Kind::Clm, &lim());
        assert_eq!(wire_json(&out), before);
    }

    #[test]
    fn edits_drop_raw_history_and_render_slots() {
        let h = vec![
            user("task"),
            assistant_text("old work"),
            user("steer"),
            assistant_edit("c1", json!([{"op":"append","slot":"facts","text":"A=1"}])),
            result("c1", false),
            assistant_text("noted"),
            user("next"),
        ];
        let (out, p) = build_wire(h, Kind::Clm, &lim());
        assert_eq!(p.tail_start, Some(3));
        assert_eq!(out.len(), 6); // pinned, slots, edit call, result, reply, next user
        let s = wire_json(&out);
        assert!(
            s.contains(SLOTS_MARKER)
                && s.contains("A=1")
                && !s.contains("old work")
                && !s.contains("steer")
        );
    }

    #[test]
    fn prefix_bytes_stable_across_turns_and_edits() {
        let l = lim();
        let mut h = vec![
            user("task"),
            assistant_edit(
                "c1",
                json!([{"op":"append","slot":"a","text":"alpha"},{"op":"append","slot":"b","text":"beta"}]),
            ),
            result("c1", false),
            assistant_text("ok"),
            user("u1"),
        ];
        let (w1, _) = build_wire(h.clone(), Kind::Clm, &l);
        // plain turn: wire only grows at the end
        h.push(assistant_text("working"));
        h.push(user("next"));
        let (w2, _) = build_wire(h.clone(), Kind::Clm, &l);
        assert!(wire_json(&w2).starts_with(wire_json(&w1).trim_end_matches(']')));
        // edit of slot b: pinned prefix and slot a bytes unchanged
        h.push(assistant_edit(
            "c2",
            json!([{"op":"replace","slot":"b","text":"BETA2"}]),
        ));
        h.push(result("c2", false));
        h.push(assistant_text("ok2"));
        h.push(user("u3"));
        let (w3, p3) = build_wire(h, Kind::Clm, &l);
        assert_eq!(
            wire_json(&w3[..1]),
            wire_json(&w1[..1]),
            "pinned prefix identical"
        );
        let (s1, s3) = (wire_json(&w1[1..2]), wire_json(&w3[1..2]));
        let common = s1.find("beta").unwrap();
        assert_eq!(
            &s1[..common],
            &s3[..common],
            "slots before the edited one identical"
        );
        assert_eq!(p3.table.slots[1].1, "BETA2");
    }

    #[test]
    fn edit_applies_at_next_user_turn_not_mid_run() {
        let l = lim();
        let mut h = vec![user("task"), assistant_text("work"), user("step")];
        let (w0, _) = build_wire(h.clone(), Kind::Clm, &l);
        h.push(assistant_edit(
            "c1",
            json!([{"op":"append","slot":"a","text":"x"}]),
        ));
        h.push(result("c1", false));
        // same run (no user message after the edit): wire is the unchanged raw history + the edit
        let (w1, p1) = build_wire(h.clone(), Kind::Clm, &l);
        assert!(p1.tail_start.is_none() && p1.shown.slots.is_empty());
        assert_eq!(wire_json(&w1), wire_json(&h));
        assert!(
            wire_json(&w1).starts_with(wire_json(&w0).trim_end_matches(']')),
            "append-only mid-run"
        );
        // validation table already includes the pending edit
        assert_eq!(p1.table.slots.len(), 1);
        // next user turn folds it
        h.push(assistant_text("done"));
        h.push(user("next"));
        let (_, p2) = build_wire(h, Kind::Clm, &l);
        assert_eq!(p2.shown.slots.len(), 1);
        assert_eq!(p2.tail_start, Some(3));
    }

    #[test]
    fn invalid_ops_rejected_atomically() {
        let l = lim();
        let mut t = SlotTable::default();
        let bad = |t: &mut SlotTable, ops: Value| {
            parse_ops(&json!({ "ops": ops })).and_then(|o| apply_ops(t, &o, &l))
        };
        assert!(bad(&mut t, json!([{"op":"replace","slot":"nope","text":"x"}])).is_err());
        assert!(bad(&mut t, json!([{"op":"evict","slot":"nope"}])).is_err());
        assert!(bad(
            &mut t,
            json!([{"op":"append","slot":"bad name!","text":"x"}])
        )
        .is_err());
        assert!(bad(&mut t, json!([{"op":"append","slot":"a"}])).is_err());
        assert!(bad(&mut t, json!([{"op":"frob","slot":"a","text":"x"}])).is_err());
        assert!(bad(
            &mut t,
            json!([{"op":"append","slot":"a","text":"x".repeat(101)}])
        )
        .is_err());
        assert!(parse_ops(&json!({"ops": []})).is_err());
        // atomic: first op fine, second invalid -> nothing applied
        assert!(bad(
            &mut t,
            json!([{"op":"append","slot":"a","text":"x"},{"op":"evict","slot":"zz"}])
        )
        .is_err());
        assert!(t.slots.is_empty());
        for s in ["a", "b", "c"] {
            bad(&mut t, json!([{"op":"append","slot":s,"text":"x"}])).unwrap();
        }
        assert!(
            bad(&mut t, json!([{"op":"append","slot":"d","text":"x"}])).is_err(),
            "max slots"
        );
    }

    #[test]
    fn rejected_call_does_not_change_wire() {
        let h = vec![
            user("task"),
            assistant_text("work"),
            assistant_edit("c1", json!([{"op":"replace","slot":"nope","text":"x"}])),
            result("c1", true),
        ];
        let before = wire_json(&h);
        let (out, p) = build_wire(h, Kind::Clm, &lim());
        assert!(p.tail_start.is_none());
        assert_eq!(wire_json(&out), before);
    }

    #[test]
    fn host_enforcer_asks_once_then_trims() {
        let l = lim(); // budget 60
        let big = "z".repeat(50);
        let mut h = vec![
            user("task"),
            assistant_edit(
                "c1",
                json!([{"op":"append","slot":"old","text":big},{"op":"append","slot":"new","text":big}]),
            ),
            result("c1", false),
            user("u"),
        ];
        let (w, p) = build_wire(h.clone(), Kind::ClmHost, &l);
        assert!(p.notice.is_some() && wire_json(&w).contains("HOST ENFORCER"));
        // plain `clm` never asks
        assert!(build_wire(h.clone(), Kind::Clm, &l).1.notice.is_none());
        // model ignores the ask -> mechanical trim of the oldest slot
        h.push(assistant_text("ignored"));
        h.push(user("go on"));
        let (w, p) = build_wire(h, Kind::ClmHost, &l);
        assert!(p.notice.is_none());
        assert_eq!(p.host_evicted, vec!["old".to_owned()]);
        assert!(wire_json(&w).contains("host evicted"));
    }

    #[tokio::test]
    async fn tool_reports_validation_to_the_model() {
        let sh = Shared::new(Kind::Clm, lim());
        let t = tool(sh.clone());
        let sig = AbortSignal::default();
        let cb: AgentToolUpdateCallback = Arc::new(|_| {});
        let r = t
            .clone()
            .execute(
                "1".into(),
                json!({"ops":[{"op":"evict","slot":"x"}]}),
                sig.clone(),
                cb.clone(),
            )
            .await
            .unwrap();
        assert_eq!(r.is_error, Some(true));
        assert!(result_text(&r.content).contains("unknown slot"));
        let r = t
            .execute(
                "2".into(),
                json!({"ops":[{"op":"append","slot":"x","text":"1"}]}),
                sig,
                cb,
            )
            .await
            .unwrap();
        assert_ne!(r.is_error, Some(true));
    }

    fn sched(script: &[u8]) -> crate::context_policy::Policy {
        let mut p = crate::context_policy::Policy::new(crate::context_policy::Kind::Schedule);
        p.schedule = script.to_vec();
        p
    }

    #[tokio::test]
    async fn policy_noop_keeps_wire_and_rejects_edits_edit_adds_one_notice() {
        let sh = Shared::with_policy(Kind::Clm, lim(), Some(sched(&[0, 1])));
        let h0 = vec![user("task")];
        // turn 0 = noop: wire is byte-identical to the raw history, edits rejected
        let w0 = sh.wire(h0.clone());
        assert_eq!(wire_json(&w0), wire_json(&h0));
        let t = tool(sh.clone());
        let cb: AgentToolUpdateCallback = Arc::new(|_| {});
        let r = t
            .clone()
            .execute(
                "1".into(),
                json!({"ops":[{"op":"append","slot":"a","text":"x"}]}),
                AbortSignal::default(),
                cb.clone(),
            )
            .await
            .unwrap();
        assert_eq!(r.is_error, Some(true));
        assert!(result_text(&r.content).contains("no context maintenance is due"));
        // turn 1 = edit: one trailing notice, edits accepted
        let h1 = vec![user("task"), assistant_text("a"), user("step 1")];
        let w1 = sh.wire(h1.clone());
        assert_eq!(w1.len(), h1.len() + 1);
        assert!(wire_json(&w1[h1.len()..]).contains("CONTEXT MAINTENANCE"));
        assert!(
            wire_json(&w1[..h1.len()]) == wire_json(&h1),
            "notice only at the tail: prefix bytes untouched"
        );
        let r = t
            .execute(
                "2".into(),
                json!({"ops":[{"op":"append","slot":"a","text":"x"}]}),
                AbortSignal::default(),
                cb,
            )
            .await
            .unwrap();
        assert_ne!(r.is_error, Some(true));
        // after the accepted edit, the same turn's next request carries no second notice
        let mut h2 = h1;
        h2.push(assistant_edit(
            "c9",
            json!([{"op":"append","slot":"a","text":"x"}]),
        ));
        h2.push(result("c9", false));
        let w2 = sh.wire(h2.clone());
        assert!(!wire_json(&w2).contains("CONTEXT MAINTENANCE"));
    }

    #[test]
    fn policy_state_counts_turns_edits_and_cache() {
        let usage = |i: u64, c: u64, o: u64| {
            let v = json!({"role":"assistant","content":[{"type":"text","text":"r"}],"api":"x","provider":"x",
                "model":"x","usage":{"input":i,"output":o,"cacheRead":c,"cacheWrite":0,"totalTokens":0,
                "cost":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"total":0}},
                "stopReason":"stop","timestamp":0});
            AgentMessage::Standard(serde_json::from_value::<Message>(v).unwrap())
        };
        let h = vec![
            user("brief"),
            usage(100, 0, 5),
            user("step 1"),
            assistant_edit("c1", json!([{"op":"append","slot":"a","text":"abc"}])),
            result("c1", false),
            usage(20, 180, 10),
            user(&format!("{} END", "w ".repeat(400))),
        ];
        let (_, p) = build_wire(h.clone(), Kind::Clm, &lim());
        let st = policy_state(&h, &p.table);
        assert_eq!(st.step, 2);
        assert_eq!(st.edits_total, 1);
        assert_eq!(st.steps_since_edit, Some(1));
        assert_eq!(st.slots_count, 1);
        assert_eq!(st.cache_hit_pct, 100 * 180 / 300);
        assert_eq!(
            st.history_tokens,
            20 + 180 + 10 + format!("{} END", "w ".repeat(400)).chars().count() / 4
        );
        assert!(st.excerpt.ends_with("END") && st.excerpt.chars().count() <= 300);
    }

    #[test]
    fn policy_offload_replaces_largest_old_block_stickily() {
        let dir = tempfile::tempdir().unwrap();
        let sh = Shared::with_policy(Kind::Clm, lim(), Some(sched(&[0, 0, 0, 2])));
        let big = "z".repeat(5000);
        let mut h = vec![
            user("brief"),
            assistant_text("ACK"),
            user(&big),
            assistant_text("a"),
        ];
        for step in 1..=3 {
            h.push(user(&format!("step {step}")));
            std::env::set_var("OMP_CONTEXT_OFFLOAD_DIR", dir.path());
            let w = sh.wire(h.clone());
            let j = wire_json(&w);
            if step < 2 {
                assert!(
                    j.contains(&big),
                    "before the offload the block is on the wire"
                );
            } else {
                assert!(
                    !j.contains(&big) && j.contains("[offloaded: 5000 chars"),
                    "offloaded and sticky at step {step}"
                );
            }
            h.push(assistant_text("ok"));
        }
        assert_eq!(
            std::fs::read_to_string(dir.path().join("block-2.txt")).unwrap(),
            big
        );
    }

    #[test]
    fn block_features_age_restatement_and_offload_reads() {
        let block = format!(
            "{} ID-1234567 ID-7654321 ID-1111111 ID-2222222",
            "filler ".repeat(400)
        );
        let h = vec![
            user("brief"),
            assistant_text("ACK"),
            user(&block),
            assistant_text("noted ID-1234567 and ID-7654321"),
            user("s2"),
            assistant_text("nothing"),
            user("s3"),
        ];
        let (age, pct, since) = block_features(&h, 2);
        assert_eq!(age, 2, "two user turns after the block's turn");
        assert_eq!(pct, 50, "2 of 4 distinct ids restated");
        assert_eq!(
            since, 2,
            "last restatement was in the turn right after the block"
        );
        let mut h2 = h.clone();
        h2.insert(5, asst(json!([{"type":"toolCall","id":"e1","name":"eval","arguments":{"code":"open('/x/block-2.txt').read()"}}]), "toolUse"));
        assert_eq!(offload_reads(&h2, &[(2, "ref".into())]), 1);
        assert_eq!(offload_reads(&h2, &[]), 0);
    }

    #[test]
    fn kind_parse() {
        assert_eq!(Kind::parse("clm-host").unwrap(), Kind::ClmHost);
        assert!(Kind::parse("rlm").is_err());
        assert_eq!(Kind::default(), Kind::Append);
    }
}
