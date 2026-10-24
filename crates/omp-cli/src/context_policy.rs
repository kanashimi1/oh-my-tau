//! Opt-in context policy (`context.policy` / `OMP_CONTEXT_POLICY`): decides, once per user turn,
//! whether `clm` / `clm-host` should run a maintenance step this turn.
//!
//! Unset (default) = legacy behaviour: no host notice, `context_edit` always accepted.
//! With a policy the host owns the maintenance instruction:
//! - decision *edit*: the request carries [`MAINT_NOTICE`] and `context_edit` is accepted;
//! - decision *noop*: no notice, `context_edit` is rejected, the wire only grows at the end, so
//!   the provider prefix cache stays intact and the turn costs nothing extra.
//!
//! Kinds: `always` (edit every turn = the old per-step behaviour, host-driven), `rule`
//! (`history_tokens >= ruleTokens` or `ruleEvery` turns since the last edit), `laya` (the Laya
//! `action` head over [`State::render`]), `schedule` (bench replay: `OMP_CONTEXT_POLICY_SCRIPT=0,1,..`
//! per user turn). The controller state of every decision is appended to
//! `OMP_CONTEXT_POLICY_LOG` (jsonl) when set - the same text the Laya head consumes.
//!
//! The host cannot offload (RLM) by itself: a Laya `rlm_offload` decision is executed as `noop`
//! (the original action is still logged).

use std::collections::HashMap;
use std::io::Write as _;
use std::path::PathBuf;

use parking_lot::Mutex;
use serde_json::json;

/// Smallest block `rlm_offload` will move (shorter blocks are not worth a reference line).
pub const OFFLOAD_MIN_CHARS: usize = 2_000;

/// Per-turn maintenance instruction sent by the host (same wording the bench used for `clm`).
pub const MAINT_NOTICE: &str =
    "CONTEXT MAINTENANCE: your context is managed. Raw conversation before your \
`context_edit` call is dropped from your context on the next turn, so first copy every \
task-critical fact (all `ANCHOR-<n> code=<c> value=<v>` lines, counts, PINs, incoming \
notices and instructions) into slots, then keep the slots compact. Then give your \
one-line log.";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Always,
    Rule,
    Laya,
    Schedule,
}

impl Kind {
    /// # Errors
    /// Unknown policy name.
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.trim() {
            "always" => Ok(Self::Always),
            "rule" => Ok(Self::Rule),
            "laya" => Ok(Self::Laya),
            "schedule" => Ok(Self::Schedule),
            other => Err(format!(
                "unknown context policy {other:?} (expected always, rule, laya or schedule)"
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Noop,
    ClmEdit,
    RlmOffload,
}

impl Action {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Noop => "noop",
            Self::ClmEdit => "clm_edit",
            Self::RlmOffload => "rlm_offload",
        }
    }
}

/// Controller state at a user-turn boundary.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct State {
    /// 0-based user turn index.
    pub step: usize,
    /// Estimated size of the next request: last assistant usage + new text / 4.
    pub history_tokens: usize,
    pub slots_chars: usize,
    pub slots_count: usize,
    pub steps_since_edit: Option<usize>,
    pub edits_total: usize,
    /// Chars of the largest not-yet-offloaded old user/tool-result block (what `rlm_offload` would move).
    pub largest_block_chars: usize,
    /// User turns since that block was added.
    pub block_age_turns: usize,
    /// % of the block's distinctive tokens (ids/numbers) the model itself has written since (text or tool args).
    pub block_restated_pct: usize,
    /// User turns since the model last wrote >= 2 of the block's distinctive tokens (age when never).
    pub block_since_ref_turns: usize,
    pub offloads_total: usize,
    /// Tool calls whose arguments open an offloaded block file.
    pub offload_reads: usize,
    pub cache_hit_pct: usize,
    /// Last 300 chars of the newest user message, whitespace-collapsed.
    pub excerpt: String,
}

impl State {
    /// The Laya input text (mirrored byte-for-byte by `bench/context/ctxp.py` only through the
    /// Rust log, never re-rendered there).
    #[must_use]
    pub fn render(&self) -> String {
        let since = self
            .steps_since_edit
            .map_or_else(|| "none".to_owned(), |n| n.to_string());
        format!(
            "context controller\nstep: {}\nhistory_tokens: {}\nslots_chars: {} ({} slots)\nsteps_since_last_edit: {}\n\
             edits_so_far: {}\nlargest_old_block_chars: {}\nblock_age_turns: {}\nblock_restated_pct: {}\n\
             block_since_ref_turns: {}\noffloads_so_far: {}\noffload_reads: {}\ncache_hit_pct: {}\nnext_instruction: {}",
            self.step, self.history_tokens, self.slots_chars, self.slots_count, since, self.edits_total,
            self.largest_block_chars, self.block_age_turns, self.block_restated_pct, self.block_since_ref_turns,
            self.offloads_total, self.offload_reads, self.cache_hit_pct, self.excerpt
        )
    }
}

/// Last 300 chars of `text`, whitespace collapsed to single spaces.
#[must_use]
pub fn excerpt_of(text: &str) -> String {
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let n = flat.chars().count();
    flat.chars().skip(n.saturating_sub(300)).collect()
}

#[derive(Debug, Clone)]
pub struct LayaPaths {
    pub ckpt: PathBuf,
    pub onnx: PathBuf,
    pub threads: usize,
}

pub struct Policy {
    pub kind: Kind,
    pub rule_tokens: usize,
    pub rule_every: usize,
    /// `rule`: offload the largest old block once it is at least this many chars (0 = never).
    pub rule_offload_chars: usize,
    /// `rule`: do not offload a block younger than this many user turns.
    pub rule_offload_age: usize,
    /// `rule`: do not offload before the model restated at least this % of the block's ids.
    pub rule_offload_restated: usize,
    /// `rule`: do not offload while the whole history is below this many tokens (0 = no gate).
    pub rule_offload_hist: usize,
    pub schedule: Vec<u8>,
    laya_paths: Option<LayaPaths>,
    laya: Mutex<Option<Result<omp_judge::policy::PolicyScorer, ()>>>,
    log: Option<PathBuf>,
    memo: Mutex<HashMap<usize, (Action, bool)>>,
}

impl std::fmt::Debug for Policy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Policy")
            .field("kind", &self.kind)
            .finish_non_exhaustive()
    }
}

impl Policy {
    #[must_use]
    pub fn new(kind: Kind) -> Self {
        Self {
            kind,
            rule_tokens: 8_000,
            rule_every: 0,
            rule_offload_chars: 0,
            rule_offload_age: 0,
            rule_offload_restated: 0,
            rule_offload_hist: 0,
            schedule: Vec::new(),
            laya_paths: None,
            laya: Mutex::new(None),
            log: None,
            memo: Mutex::new(HashMap::new()),
        }
    }

    #[cfg(test)]
    #[must_use]
    pub fn with_log(mut self, p: PathBuf) -> Self {
        self.log = Some(p);
        self
    }

    /// Env wins over settings (`context.policy*`, `OMP_CONTEXT_POLICY*`). `None` = no policy.
    ///
    /// # Errors
    /// Unknown policy name.
    pub fn resolve(
        settings: &omp_settings::Settings,
        env: &dyn Fn(&str) -> Option<String>,
    ) -> Result<Option<Self>, String> {
        let name = env("OMP_CONTEXT_POLICY").or_else(|| settings.get_string("context.policy"));
        let Some(name) = name.filter(|n| !n.trim().is_empty() && n.trim() != "none") else {
            return Ok(None);
        };
        let mut p = Self::new(Kind::parse(&name)?);
        let num = |envk: &str, id: &str, d: usize| {
            env(envk)
                .and_then(|v| v.parse().ok())
                .or_else(|| settings.get_number(id).map(|v| v as usize))
                .unwrap_or(d)
        };
        p.rule_tokens = num(
            "OMP_CONTEXT_POLICY_RULE_TOKENS",
            "context.policy.ruleTokens",
            p.rule_tokens,
        );
        p.rule_every = num(
            "OMP_CONTEXT_POLICY_RULE_EVERY",
            "context.policy.ruleEvery",
            p.rule_every,
        );
        p.rule_offload_chars = num(
            "OMP_CONTEXT_POLICY_RULE_OFFLOAD",
            "context.policy.ruleOffloadChars",
            p.rule_offload_chars,
        );
        p.rule_offload_age = num(
            "OMP_CONTEXT_POLICY_RULE_AGE",
            "context.policy.ruleOffloadAge",
            p.rule_offload_age,
        );
        p.rule_offload_restated = num(
            "OMP_CONTEXT_POLICY_RULE_RESTATED",
            "context.policy.ruleOffloadRestated",
            p.rule_offload_restated,
        );
        p.rule_offload_hist = num(
            "OMP_CONTEXT_POLICY_RULE_OFFLOAD_HIST",
            "context.policy.ruleOffloadHist",
            p.rule_offload_hist,
        );
        p.schedule = env("OMP_CONTEXT_POLICY_SCRIPT")
            .map(|s| s.split(',').filter_map(|t| t.trim().parse().ok()).collect())
            .unwrap_or_default();
        p.log = env("OMP_CONTEXT_POLICY_LOG").map(PathBuf::from);
        if p.kind == Kind::Laya {
            let ckpt = env("OMP_CONTEXT_LAYA_CKPT")
                .or_else(|| settings.get_string("context.policy.layaCkpt"));
            let onnx = env("OMP_CONTEXT_LAYA_ONNX")
                .or_else(|| settings.get_string("context.policy.layaOnnx"));
            let (Some(ckpt), Some(onnx)) = (ckpt, onnx) else {
                return Err("context.policy=laya needs context.policy.layaCkpt and context.policy.layaOnnx \
                            (or OMP_CONTEXT_LAYA_CKPT / OMP_CONTEXT_LAYA_ONNX)"
                    .to_owned());
            };
            p.laya_paths = Some(LayaPaths {
                ckpt: ckpt.into(),
                onnx: onnx.into(),
                threads: num("OMP_CONTEXT_LAYA_THREADS", "context.policy.layaThreads", 2),
            });
        }
        Ok(Some(p))
    }

    /// Rule decision (pure).
    #[must_use]
    pub fn rule(&self, st: &State) -> bool {
        let big = self.rule_tokens > 0 && st.history_tokens >= self.rule_tokens;
        let due =
            self.rule_every > 0 && st.steps_since_edit.unwrap_or(st.step + 1) >= self.rule_every;
        big || due
    }

    fn laya_action(&self, st: &State) -> Action {
        let Some(paths) = &self.laya_paths else {
            return Action::ClmEdit;
        };
        let mut slot = self.laya.lock();
        let scorer = slot.get_or_insert_with(|| {
            omp_judge::policy::PolicyScorer::load(&omp_judge::scorer::ModelPaths {
                ckpt: paths.ckpt.clone(),
                onnx: paths.onnx.clone(),
                threads: paths.threads,
            })
            .map_err(|e| tracing::warn!("context.policy=laya: {e:#}; falling back to always"))
        });
        let Ok(scorer) = scorer else {
            return Action::ClmEdit;
        };
        match scorer.probs(&st.render()) {
            Ok(probs) => {
                let best = probs
                    .iter()
                    .max_by(|a, b| a.1.total_cmp(&b.1))
                    .map(|p| p.0.as_str());
                match best {
                    Some("noop") => Action::Noop,
                    Some("rlm_offload") => Action::RlmOffload,
                    _ => Action::ClmEdit,
                }
            }
            Err(e) => {
                tracing::warn!("context.policy=laya inference failed: {e:#}; edit");
                Action::ClmEdit
            }
        }
    }

    /// Decide for the user turn `st.step` (memoized: later requests of the same turn - tool
    /// loops - get the same answer and log nothing). Returns the action and whether it is the
    /// first request of this turn (`fresh`, so one-shot actions like an offload run once).
    pub fn decide(&self, st: &State) -> (Action, bool) {
        if let Some((a, _)) = self.memo.lock().get(&st.step) {
            return (*a, false);
        }
        let action = match self.kind {
            Kind::Always => Action::ClmEdit,
            Kind::Rule => {
                if self.rule_offload_chars > 0
                    && st.largest_block_chars >= self.rule_offload_chars.max(OFFLOAD_MIN_CHARS)
                    && st.block_age_turns >= self.rule_offload_age
                    && st.block_restated_pct >= self.rule_offload_restated
                    && st.history_tokens >= self.rule_offload_hist
                {
                    Action::RlmOffload
                } else if self.rule(st) {
                    Action::ClmEdit
                } else {
                    Action::Noop
                }
            }
            Kind::Laya => {
                let a = self.laya_action(st);
                // nothing to offload: the host cannot do it, so it is a noop
                if a == Action::RlmOffload && st.largest_block_chars < OFFLOAD_MIN_CHARS {
                    Action::Noop
                } else {
                    a
                }
            }
            Kind::Schedule => match self.schedule.get(st.step).copied().unwrap_or(0) {
                1 => Action::ClmEdit,
                2 => Action::RlmOffload,
                _ => Action::Noop,
            },
        };
        let edit = action == Action::ClmEdit;
        self.memo.lock().insert(st.step, (action, edit));
        if let Some(path) = &self.log {
            let line = json!({"step": st.step, "policy": format!("{:?}", self.kind).to_lowercase(),
                              "action": action.as_str(), "edit": edit, "state": st.render()});
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
            {
                let _ = writeln!(f, "{line}");
            }
        }
        (action, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn st(step: usize, hist: usize, since: Option<usize>) -> State {
        State {
            step,
            history_tokens: hist,
            steps_since_edit: since,
            ..State::default()
        }
    }

    #[test]
    fn rule_fires_on_size_or_age() {
        let mut p = Policy::new(Kind::Rule);
        p.rule_tokens = 1000;
        p.rule_every = 4;
        assert!(!p.rule(&st(2, 999, Some(1))));
        assert!(p.rule(&st(2, 1000, Some(1))));
        assert!(p.rule(&st(6, 10, Some(4))));
        assert!(
            !p.rule(&st(2, 10, None)),
            "no edit yet: age counts from the start (step+1 = 3 < 4)"
        );
        assert!(p.rule(&st(3, 10, None)));
        p.rule_every = 0;
        assert!(
            !p.rule(&st(99, 10, None)),
            "ruleEvery=0 disables the age trigger"
        );
    }

    #[test]
    fn rule_offloads_big_old_block_before_editing() {
        let mut p = Policy::new(Kind::Rule);
        p.rule_tokens = 1000;
        p.rule_offload_chars = 5000;
        let mut s = st(2, 5000, None);
        s.largest_block_chars = 4999;
        assert_eq!(
            p.decide(&s).0,
            Action::ClmEdit,
            "block below the threshold: edit rule applies"
        );
        s.step = 3;
        s.largest_block_chars = 5000;
        assert_eq!(p.decide(&s).0, Action::RlmOffload);
        p.rule_offload_chars = 0;
        s.step = 4;
        assert_eq!(
            p.decide(&s).0,
            Action::ClmEdit,
            "offload disabled by default"
        );
    }

    #[test]
    fn rule_offload_waits_for_age_and_restatement() {
        let mut p = Policy::new(Kind::Rule);
        p.rule_offload_chars = 5000;
        p.rule_offload_age = 3;
        p.rule_offload_restated = 60;
        let mut s = st(5, 100, None);
        s.largest_block_chars = 9000;
        s.block_age_turns = 2;
        s.block_restated_pct = 90;
        assert_eq!(p.decide(&s).0, Action::Noop, "too young");
        s.step = 6;
        s.block_age_turns = 3;
        s.block_restated_pct = 59;
        assert_eq!(p.decide(&s).0, Action::Noop, "not restated enough");
        s.step = 7;
        s.block_restated_pct = 60;
        assert_eq!(p.decide(&s).0, Action::RlmOffload);
    }

    #[test]
    fn decide_is_memoized_per_turn_and_schedule_replays() {
        let mut p = Policy::new(Kind::Schedule);
        p.schedule = vec![0, 1, 2];
        assert_eq!(p.decide(&st(0, 0, None)), (Action::Noop, true));
        assert_eq!(p.decide(&st(1, 0, None)), (Action::ClmEdit, true));
        assert_eq!(
            p.decide(&st(1, 0, None)),
            (Action::ClmEdit, false),
            "same turn: memoized, not fresh"
        );
        assert_eq!(p.decide(&st(2, 0, None)).0, Action::RlmOffload);
        assert_eq!(
            p.decide(&st(7, 0, None)).0,
            Action::Noop,
            "past the script = noop"
        );
        p.schedule = vec![1; 8]; // changing the script must not flip an already-decided turn
        assert_eq!(p.decide(&st(0, 0, None)).0, Action::Noop);
    }

    #[test]
    fn always_edits_every_turn_and_log_has_state() {
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("p.jsonl");
        let p = Policy::new(Kind::Always).with_log(log.clone());
        assert_eq!(p.decide(&st(0, 5, None)).0, Action::ClmEdit);
        assert_eq!(p.decide(&st(1, 5, Some(1))).0, Action::ClmEdit);
        let l: Vec<serde_json::Value> = std::fs::read_to_string(&log)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(l.len(), 2);
        assert!(l[1]["state"]
            .as_str()
            .unwrap()
            .contains("steps_since_last_edit: 1"));
    }

    #[test]
    fn render_is_stable_and_excerpt_is_tail() {
        let s = State {
            step: 3,
            history_tokens: 1234,
            slots_chars: 56,
            slots_count: 2,
            steps_since_edit: None,
            edits_total: 0,
            largest_block_chars: 7,
            block_age_turns: 4,
            block_restated_pct: 50,
            block_since_ref_turns: 2,
            offloads_total: 0,
            offload_reads: 1,
            cache_hit_pct: 91,
            excerpt: excerpt_of("a\n  b"),
        };
        assert_eq!(
            s.render(),
            "context controller\nstep: 3\nhistory_tokens: 1234\nslots_chars: 56 (2 slots)\nsteps_since_last_edit: none\n\
             edits_so_far: 0\nlargest_old_block_chars: 7\nblock_age_turns: 4\nblock_restated_pct: 50\n\
             block_since_ref_turns: 2\noffloads_so_far: 0\noffload_reads: 1\ncache_hit_pct: 91\nnext_instruction: a b"
        );
        let long = "x".repeat(500) + "END";
        let e = excerpt_of(&long);
        assert_eq!(e.chars().count(), 300);
        assert!(e.ends_with("END"));
    }

    #[test]
    fn resolve_names() {
        let d = tempfile::tempdir().unwrap();
        let s = omp_settings::Settings::load_from(None, d.path());
        let none = |_: &str| None;
        assert!(Policy::resolve(&s, &none).unwrap().is_none());
        let env = |k: &str| (k == "OMP_CONTEXT_POLICY").then(|| "rule".to_owned());
        assert_eq!(Policy::resolve(&s, &env).unwrap().unwrap().kind, Kind::Rule);
        let bad = |k: &str| (k == "OMP_CONTEXT_POLICY").then(|| "nope".to_owned());
        assert!(Policy::resolve(&s, &bad).is_err());
        let laya = |k: &str| (k == "OMP_CONTEXT_POLICY").then(|| "laya".to_owned());
        assert!(
            Policy::resolve(&s, &laya).is_err(),
            "laya without model paths is a config error"
        );
    }
}
