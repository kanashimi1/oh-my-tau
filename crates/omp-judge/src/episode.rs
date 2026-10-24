//! Port of the Python `build_traj_outcome.py`: session entries -> episodes -> flat
//! `state` string + objective counters. One episode = one real human user message and all
//! agent steps until the next real human message (a subagent session is one episode).

use std::collections::HashMap;
use std::sync::OnceLock;

use regex::Regex;
use serde_json::{Map, Value};

use crate::pyfmt::{clen, collapse_ws, dumps, py_str, take_chars, truthy};

const MAX_STATE: usize = 24000;
const REQ_MAX: usize = 1500;
const FINAL_MAX: usize = 1500;

const TAG_NAMES: [&str; 5] = [
    "system-reminder",
    "system-notice",
    "memory-recall",
    "skill-state",
    "environment_context",
];
const PSEUDO_PREFIX: [&str; 9] = [
    "[fleet-envelope]",
    "[budget notice]",
    "[wait interrupted",
    "current interruptible wait",
    "[s1 ",
    "extension \"",
    "<system-notice>",
    "<system-reminder>",
    "[request interrupted",
];
const DROP_KEYS: [&str; 5] = ["i", "_i", "intent", "timeout", "agent__intent"];

fn re(cell: &'static OnceLock<Regex>, pat: &str) -> &'static Regex {
    cell.get_or_init(|| Regex::new(pat).expect("static regex"))
}

/// One tool call of an episode.
#[derive(Debug, Clone)]
pub struct Step {
    pub tool: String,
    pub args: Value,
    pub id: String,
    pub intent: String,
    pub err: bool,
    pub exit: i64,
    pub text: String,
    pub done: bool,
}

/// One real user request and what the agent did for it.
#[derive(Debug, Clone, Default)]
pub struct Episode {
    pub request: String,
    pub steps: Vec<Step>,
    pub final_text: String,
    pub last_stop: Option<String>,
    /// `(model, count)` in first-seen order (Python `Counter.most_common(1)` tie-break).
    pub models: Vec<(String, usize)>,
    pub ts: Option<String>,
    pub yield_args: Option<Value>,
}

/// Objective counters computed from the steps only (never the next user message).
#[derive(Debug, Clone, PartialEq)]
pub struct Objective {
    pub tests_run: bool,
    /// Python truthiness of `tests_final_pass` (the last test call's exit code, as the
    /// training data stores it).
    pub tests_final_pass: bool,
    pub ended_with_error: bool,
    pub identical_fail_run_max: i64,
    pub n_steps: i64,
    pub interrupted: bool,
    pub claimed_done: bool,
    pub is_subagent: bool,
    pub model: Option<String>,
}

impl Objective {
    /// Read the `objective` object of a `traj_outcome.jsonl` record.
    #[must_use]
    pub fn from_value(o: &Value) -> Self {
        let b = |k: &str| o.get(k).is_some_and(truthy);
        Self {
            tests_run: b("tests_run"),
            tests_final_pass: b("tests_final_pass"),
            ended_with_error: b("ended_with_error"),
            identical_fail_run_max: o
                .get("identical_fail_run_max")
                .and_then(Value::as_i64)
                .unwrap_or(0),
            n_steps: o.get("n_steps").and_then(Value::as_i64).unwrap_or(0),
            interrupted: b("interrupted"),
            claimed_done: b("claimed_done"),
            is_subagent: b("is_subagent"),
            model: o.get("model").and_then(Value::as_str).map(str::to_owned),
        }
    }
}

fn text_of(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(a)) => a
            .iter()
            .filter(|b| b.is_object() && b.get("type").and_then(Value::as_str) == Some("text"))
            .map(|b| b.get("text").and_then(Value::as_str).unwrap_or(""))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

/// `_TAGS.sub("", s)`: `<(system-reminder|...|s1-[a-z]+|environment_context)[^>]*>.*?</\1>` (DOTALL).
/// The regex crate has no backreferences, so the closing tag is searched for explicitly.
fn strip_tags(s: &str) -> String {
    static OPEN: OnceLock<Regex> = OnceLock::new();
    let open = re(
        &OPEN,
        &format!("<({}|s1-[a-z]+)[^>]*>", TAG_NAMES.join("|")),
    );
    let mut out = String::with_capacity(s.len());
    let mut pos = 0;
    let mut scan = 0;
    while let Some(c) = open.captures_at(s, scan) {
        let whole = c.get(0).expect("group 0");
        let name = c.get(1).expect("group 1").as_str();
        let body_start = whole.end();
        // Backtracking in Python tries shorter `s1-[a-z]+` names when the full one has no closer.
        let mut candidates = vec![name.to_owned()];
        if name.starts_with("s1-") {
            for cut in (4..name.len()).rev() {
                candidates.push(name[..cut].to_owned());
            }
        }
        let close = candidates.iter().find_map(|n| {
            s[body_start..]
                .find(&format!("</{n}>"))
                .map(|i| body_start + i + n.len() + 3)
        });
        if let Some(end) = close {
            out.push_str(&s[pos..whole.start()]);
            pos = end;
            scan = end;
        } else {
            // advance one char past the failed start
            scan = whole.start() + s[whole.start()..].chars().next().map_or(1, char::len_utf8);
        }
        if scan > s.len() {
            break;
        }
    }
    out.push_str(&s[pos..]);
    out
}

/// Cleaned text if this user message is a real human turn (or the subagent task), else `None`.
fn human_text(m: &Value, is_sub: bool) -> Option<String> {
    if !is_sub && m.get("attribution").and_then(Value::as_str) != Some("user") {
        return None;
    }
    let t = strip_tags(&text_of(m.get("content"))).trim().to_owned();
    if t.is_empty() {
        return None;
    }
    let lower = t.to_lowercase();
    if PSEUDO_PREFIX.iter().any(|p| lower.starts_with(p)) {
        return None;
    }
    Some(t)
}

fn is_test_cmd(cmd: &str) -> bool {
    static SEG: OnceLock<Regex> = OnceLock::new();
    static PREFIX: OnceLock<Regex> = OnceLock::new();
    static TEST: OnceLock<Regex> = OnceLock::new();
    let seg = re(&SEG, r"&&|\|\||;|\||\n");
    let prefix = re(
        &PREFIX,
        r"^(?:[A-Za-z_][A-Za-z0-9_]*=\S*\s+|cd\s+\S+\s*|timeout\s+\d+\S*\s+|uv\s+run\s+|npx\s+|poetry\s+run\s+|python3?\s+-m\s+|time\s+|sudo\s+|env\s+|\./)+",
    );
    let test = re(
        &TEST,
        r"^(pytest|py\.test|unittest|cargo\s+(test|nextest)|go\s+test|npm\s+(run\s+)?test|npm\s+t\b|yarn\s+test|pnpm\s+(run\s+)?test|(\S*/)?gradlew?\s+\S*(test|build)|(\S*/)?mvnw?\b|tsc\b|ruff\b|mypy\b|eslint\b)",
    );
    seg.split(cmd)
        .any(|s| test.is_match(&prefix.replace(s.trim(), "")))
}

fn done_re() -> &'static Regex {
    static DONE: OnceLock<Regex> = OnceLock::new();
    re(
        &DONE,
        r"(?i)\b(done|complete[d]?|finished|fixed|ready|implemented|works|working|passes|passing|resolved|all set|applied|added|created|updated|verified)\b|готов|исправил|сделал|выполнил|реализовал|добавил|починил|завершен|завершил|закончил|работает|прошл[иа]|обновил|создал|внёс|внес",
    )
}

fn as_pyint(v: &Value) -> Option<i64> {
    match v {
        Value::Bool(b) => Some(i64::from(*b)),
        other => other.as_i64(),
    }
}

/// Port of `parse_file` over already-decoded session entries (`{"type":"message",...}` lines).
/// `is_sub`: the entries belong to a subagent session (the first user message is the task).
pub fn parse_entries<'a>(
    entries: impl IntoIterator<Item = &'a Value>,
    is_sub: bool,
) -> Vec<Episode> {
    let mut intents: HashMap<String, String> = HashMap::new();
    let mut eps: Vec<Episode> = Vec::new();
    let mut pend: HashMap<String, (usize, usize)> = HashMap::new();
    let mut seen_task = false;
    let empty = Map::new();
    for e in entries {
        let ty = e.get("type").and_then(Value::as_str);
        if ty == Some("custom")
            && e.get("customType").and_then(Value::as_str) == Some("tool_execution_start")
        {
            let d = e.get("data").and_then(Value::as_object).unwrap_or(&empty);
            if let Some(i) = d.get("intent").filter(|v| truthy(v)) {
                let id = d.get("toolCallId").and_then(Value::as_str).unwrap_or("");
                intents.insert(id.to_owned(), py_str(i));
            }
            continue;
        }
        if ty != Some("message") {
            continue;
        }
        let Some(m) = e.get("message").filter(|m| m.is_object()) else {
            continue;
        };
        let role = m.get("role").and_then(Value::as_str);
        if role == Some("user") {
            if is_sub && seen_task {
                continue;
            }
            let Some(t) = human_text(m, is_sub) else {
                continue;
            };
            seen_task = true;
            eps.push(Episode {
                request: t,
                ts: e
                    .get("timestamp")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                ..Episode::default()
            });
            continue;
        }
        let Some(ei) = eps.len().checked_sub(1) else {
            continue;
        };
        let cur = &mut eps[ei];
        if role == Some("assistant") {
            let content = m.get("content").filter(|v| truthy(v));
            if let Some(model) = m.get("model").filter(|v| truthy(v)).map(py_str) {
                if let Some(slot) = cur.models.iter_mut().find(|(k, _)| *k == model) {
                    slot.1 += 1;
                } else {
                    cur.models.push((model, 1));
                }
            }
            cur.last_stop = m
                .get("stopReason")
                .and_then(Value::as_str)
                .map(str::to_owned);
            cur.final_text = text_of(content).trim().to_owned();
            cur.yield_args = None;
            if let Some(Value::Array(blocks)) = content {
                for b in blocks {
                    if !b.is_object() || b.get("type").and_then(Value::as_str) != Some("toolCall") {
                        continue;
                    }
                    let args = b
                        .get("arguments")
                        .filter(|v| truthy(v))
                        .cloned()
                        .unwrap_or_else(|| Value::Object(Map::new()));
                    let name = b.get("name").and_then(Value::as_str).unwrap_or("");
                    if name == "yield" || name == "submit_result" {
                        cur.yield_args = Some(args);
                        continue;
                    }
                    let arg_str = |k: &str| args.get(k).filter(|v| truthy(v)).map(py_str);
                    let intent = b
                        .get("intent")
                        .filter(|v| truthy(v))
                        .map(py_str)
                        .or_else(|| arg_str("i"))
                        .or_else(|| arg_str("_i"))
                        .unwrap_or_default();
                    let id = b.get("id").and_then(Value::as_str).unwrap_or("").to_owned();
                    pend.insert(id.clone(), (ei, cur.steps.len()));
                    cur.steps.push(Step {
                        tool: name.to_owned(),
                        args,
                        id,
                        intent,
                        err: false,
                        exit: 0,
                        text: String::new(),
                        done: false,
                    });
                }
            }
        } else if role == Some("toolResult") {
            let id = m.get("toolCallId").and_then(Value::as_str).unwrap_or("");
            let Some((ei, si)) = pend.remove(id) else {
                continue;
            };
            let st = &mut eps[ei].steps[si];
            let ec = m
                .get("details")
                .and_then(Value::as_object)
                .and_then(|d| d.get("exitCode"))
                .and_then(as_pyint);
            st.err = m.get("isError").is_some_and(truthy) || ec.is_some_and(|c| c != 0);
            st.exit = ec.unwrap_or_else(|| i64::from(st.err));
            st.text = take_chars(text_of(m.get("content")).trim(), 600).to_owned();
            st.done = true;
            if st.intent.is_empty() {
                if let Some(i) = intents.get(&st.id) {
                    st.intent.clone_from(i);
                }
            }
        }
    }
    eps
}

fn key_args(st: &Step, n: usize) -> String {
    let a: Map<String, Value> = match &st.args {
        Value::Object(o) => o
            .iter()
            .filter(|(k, _)| !DROP_KEYS.contains(&k.as_str()))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect(),
        _ => Map::new(),
    };
    let t = st.tool.as_str();
    let s = if t == "bash" && a.contains_key("command") {
        py_str(&a["command"])
    } else if t == "write" && a.contains_key("path") {
        py_str(&a["path"])
    } else if t == "edit" && a.contains_key("input") {
        py_str(&a["input"])
    } else if a.contains_key("path") && a.len() <= 2 {
        a.values().map(py_str).collect::<Vec<_>>().join(" ")
    } else {
        take_chars(&dumps(&Value::Object(a)), n).to_owned()
    };
    take_chars(&collapse_ws(&s), n).to_owned()
}

fn one_line(s: &str, n: usize) -> String {
    take_chars(&collapse_ws(s), n).to_owned()
}

fn step_line(i: usize, st: &Step, na: usize, no: usize, ni: usize) -> String {
    let out = if st.done {
        let tag = if st.err {
            format!("[ERR exit={}]", st.exit)
        } else {
            "[ok]".to_owned()
        };
        format!("{tag} {}", one_line(&st.text, no))
    } else {
        "[no result]".to_owned()
    };
    let it = one_line(&st.intent, ni);
    let head = if it.is_empty() {
        format!("{}: {}", st.tool, key_args(st, na))
    } else {
        format!("{}: {it} | {}", st.tool, key_args(st, na))
    };
    format!("{i}. {head} -> {out}")
}

fn render_steps(steps: &[Step], budget: usize) -> String {
    let n = steps.len();
    let (keep, omitted): (Vec<usize>, usize) = if n > 60 {
        ((0..10).chain(n - 50..n).collect(), n - 60)
    } else {
        ((0..n).collect(), 0)
    };
    let mut s = String::new();
    for (na, no, ni) in [
        (160, 200, 120),
        (120, 150, 80),
        (90, 110, 60),
        (60, 70, 40),
        (40, 40, 30),
    ] {
        let mut lines = Vec::with_capacity(keep.len() + 1);
        for (k, &i) in keep.iter().enumerate() {
            if omitted > 0 && k == 10 {
                lines.push(format!("... {omitted} steps omitted ..."));
            }
            lines.push(step_line(i + 1, &steps[i], na, no, ni));
        }
        s = lines.join("\n");
        if clen(&s) <= budget {
            return s;
        }
    }
    take_chars(&s, budget).to_owned()
}

impl Episode {
    /// Final reply text; falls back to the `yield` arguments (`final_text` in the builder).
    #[must_use]
    pub fn final_reply(&self) -> String {
        let mut f = self.final_text.clone();
        if f.is_empty() {
            if let Some(y) = self.yield_args.as_ref().filter(|y| truthy(y)) {
                f = dumps(y);
            }
        }
        f.trim().to_owned()
    }

    /// Flat `USER REQUEST / STEPS / FINAL REPLY` state (`render_state`).
    #[must_use]
    pub fn render_state(&self) -> String {
        let req = take_chars(&self.request, REQ_MAX);
        let fin_full = self.final_reply();
        let fin = take_chars(&fin_full, FINAL_MAX);
        let fin = if fin.is_empty() { "(none)" } else { fin };
        let head = format!("USER REQUEST:\n{req}\n\nSTEPS (action -> outcome):\n");
        let tail = format!("\n\nFINAL REPLY:\n{fin}");
        let budget = MAX_STATE.saturating_sub(clen(&head) + clen(&tail) + 50);
        format!("{head}{}{tail}", render_steps(&self.steps, budget))
    }

    /// Objective counters (`objective`).
    #[must_use]
    pub fn objective(&self, is_sub: bool) -> Objective {
        let steps = &self.steps;
        let tests: Vec<&Step> = steps
            .iter()
            .filter(|s| {
                s.tool == "bash"
                    && s.done
                    && s.args.is_object()
                    && is_test_cmd(&s.args.get("command").map_or_else(String::new, |c| {
                        if c.is_null() {
                            String::new()
                        } else {
                            py_str(c)
                        }
                    }))
            })
            .collect();
        let fin = self.final_reply();
        let (mut run, mut best) = (0_i64, 0_i64);
        let mut prev: Option<(String, String)> = None;
        for s in steps {
            if s.done && s.err {
                let sig = (s.tool.clone(), take_chars(&s.text, 60).to_owned());
                run = if prev.as_ref() == Some(&sig) {
                    run + 1
                } else {
                    1
                };
                prev = Some(sig);
                best = best.max(run);
            } else {
                run = 0;
                prev = None;
            }
        }
        let last_done = steps.iter().rev().find(|s| s.done);
        // Counter.most_common(1): highest count, first seen wins ties.
        let mut model: Option<&(String, usize)> = None;
        for m in &self.models {
            if model.is_none_or(|b| m.1 > b.1) {
                model = Some(m);
            }
        }
        let yielded = self.yield_args.as_ref().is_some_and(truthy);
        Objective {
            tests_run: !tests.is_empty(),
            tests_final_pass: tests.last().is_some_and(|t| t.exit != 0),
            ended_with_error: last_done.is_some_and(|s| s.err),
            identical_fail_run_max: best,
            n_steps: i64::try_from(steps.len()).unwrap_or(i64::MAX),
            interrupted: matches!(self.last_stop.as_deref(), Some("aborted" | "error"))
                || fin.is_empty()
                || (self.last_stop.as_deref() == Some("toolUse") && !yielded),
            claimed_done: done_re().is_match(&fin),
            is_subagent: is_sub,
            model: model.map(|m| m.0.clone()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn tags_are_stripped() {
        assert_eq!(
            strip_tags("a<system-reminder x=1>zzz\nq</system-reminder>b"),
            "ab"
        );
        assert_eq!(strip_tags("a<s1-ab>q</s1-ab>b"), "ab");
        assert_eq!(
            strip_tags("a<skill-state>never closed"),
            "a<skill-state>never closed"
        );
    }

    #[test]
    fn episode_from_entries() {
        let entries = vec![
            json!({"type":"message","timestamp":"t0","message":{"role":"user","attribution":"user","content":"fix the bug in foo.rs"}}),
            json!({"type":"message","message":{"role":"assistant","model":"m","stopReason":"toolUse","content":[
                {"type":"toolCall","id":"c1","name":"bash","arguments":{"command":"cargo test","i":"run tests"}}]}}),
            json!({"type":"message","message":{"role":"toolResult","toolCallId":"c1","isError":false,"details":{"exitCode":0},"content":[{"type":"text","text":"ok 3 passed"}]}}),
            json!({"type":"message","message":{"role":"assistant","model":"m","stopReason":"stop","content":[{"type":"text","text":"All done, tests pass."}]}}),
        ];
        let eps = parse_entries(&entries, false);
        assert_eq!(eps.len(), 1);
        let st = eps[0].render_state();
        assert!(
            st.contains("1. bash: run tests | cargo test -> [ok] ok 3 passed"),
            "{st}"
        );
        assert!(st.ends_with("FINAL REPLY:\nAll done, tests pass."));
        let o = eps[0].objective(false);
        assert!(o.tests_run && !o.tests_final_pass && !o.interrupted && o.claimed_done);
        assert_eq!(o.model.as_deref(), Some("m"));
    }

    #[test]
    fn non_human_user_messages_do_not_open_episodes() {
        let entries = vec![json!({"type":"message","message":{"role":"user","content":"hello"}})];
        assert!(parse_entries(&entries, false).is_empty());
        assert_eq!(parse_entries(&entries, true).len(), 1);
    }
}
