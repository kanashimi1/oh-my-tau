//! Port of the Python `judge_view.py::view_sel` (+ `parts`, `facts`): the exact judge
//! input `REQUEST / FACTS / FINAL REPLY / SELECTED STEPS`. Facts come from the steps only.

use std::collections::HashSet;
use std::sync::OnceLock;

use regex::Regex;

use crate::episode::{Episode, Objective};
use crate::pyfmt::{clen, take_chars};

/// Char budget of the selected-steps block.
pub const DEFAULT_BUDGET: usize = 2600;

fn step_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"(?m)^\d+\. ").expect("static regex"))
}

fn test_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(r"\b(pytest|unittest|cargo (test|build|check)|go test|npm (run )?test|gradlew?|mvn|tsc|ruff|mypy|eslint|test_\w+\.py)\b")
            .expect("static regex")
    })
}

/// Python's `\w` is letters, digits and `_` (no combining marks, unlike the regex crate's `\w`).
fn word_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"[A-Za-zА-Яа-я_][\p{L}\p{N}_./-]{3,}").expect("static regex"))
}

const STEPS_MARK: &str = "STEPS (action -> outcome):";

/// `parts(state)`: (request, steps, final reply).
#[must_use]
pub fn parts(state: &str) -> (String, Vec<String>, String) {
    let req = state
        .split(STEPS_MARK)
        .next()
        .unwrap_or("")
        .replace("USER REQUEST:", "")
        .trim()
        .to_owned();
    let mid = state.split_once(STEPS_MARK).map_or("", |(_, m)| m);
    let (steps_txt, fin) = match mid.split_once("FINAL REPLY:") {
        Some((a, b)) => (a, b),
        None => (mid, ""),
    };
    let steps = step_re()
        .split(steps_txt)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .collect();
    (req, steps, fin.trim().to_owned())
}

fn yes(b: bool) -> &'static str {
    if b {
        "yes"
    } else {
        "no"
    }
}

/// `facts(o, steps)`.
#[must_use]
pub fn facts(o: &Objective, steps: &[String]) -> String {
    let errs = steps.iter().filter(|s| s.contains("[ERR")).count();
    let mut f = vec![
        format!("steps: {}", o.n_steps),
        format!("failed steps: {errs}"),
        format!("last step failed: {}", yes(o.ended_with_error)),
        format!(
            "longest run of identical failures: {}",
            o.identical_fail_run_max
        ),
        format!("tests/linters run: {}", yes(o.tests_run)),
    ];
    if o.tests_run {
        f.push(format!(
            "last test/lint call passed: {}",
            yes(o.tests_final_pass)
        ));
    }
    f.push(format!(
        "agent ended without a final reply / interrupted: {}",
        yes(o.interrupted)
    ));
    f.push(format!("final reply claims done: {}", yes(o.claimed_done)));
    f.push(format!("subagent task: {}", yes(o.is_subagent)));
    f.join("; ")
}

/// `view_sel(r, budget)` for a flat `state` and its `objective`.
#[must_use]
pub fn view_sel(state: &str, objective: &Objective, budget: usize) -> String {
    let (req, steps, fin) = parts(state);
    let rw: HashSet<String> = word_re()
        .find_iter(&req)
        .map(|m| m.as_str().to_lowercase())
        .collect();
    let n = steps.len();
    let mut sc: Vec<f64> = Vec::with_capacity(n);
    for (i, s) in steps.iter().enumerate() {
        let head = s.split("->").next().unwrap_or("");
        let b = |c: bool| if c { 1.0 } else { 0.0 };
        let mut v = 3.0 * b(i + 4 >= n)
            + 2.0 * b(s.contains("[ERR"))
            + 1.0 * b(i > 0 && steps[i - 1].contains("[ERR") && !s.contains("[ERR"));
        v += 2.0 * b(test_re().is_match(head))
            + 1.0
                * b(head.trim_start().starts_with("edit") || head.trim_start().starts_with("write"));
        let sw: HashSet<String> = word_re()
            .find_iter(head)
            .map(|m| m.as_str().to_lowercase())
            .collect();
        #[allow(clippy::cast_precision_loss)]
        {
            v += 3.0 * (rw.intersection(&sw).count() as f64 / sw.len().max(1) as f64)
                + 0.001 * i as f64;
        }
        sc.push(v);
    }
    let line = |i: usize| format!("#{} {}", i + 1, take_chars(&steps[i], 240));
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        sc[b]
            .partial_cmp(&sc[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut keep: Vec<usize> = Vec::new();
    let mut used = 0;
    for i in order {
        let l = clen(&line(i));
        if used + l > budget {
            continue;
        }
        keep.push(i);
        used += l;
    }
    keep.sort_unstable();
    let mut lines: Vec<String> = Vec::new();
    let mut prev: isize = -1;
    for &i in &keep {
        #[allow(clippy::cast_possible_wrap)]
        let ii = i as isize;
        if ii - prev > 1 {
            lines.push(format!("... {} steps ...", ii - prev - 1));
        }
        lines.push(line(i));
        prev = ii;
    }
    #[allow(clippy::cast_possible_wrap)]
    let tail = n as isize - 1 - prev;
    if tail > 0 {
        lines.push(format!("... {tail} steps ..."));
    }
    let fin = if fin.is_empty() {
        "(none)"
    } else {
        take_chars(&fin, 1200)
    };
    let mut out = vec![
        format!("REQUEST: {}", take_chars(&req, 600)),
        format!("FACTS: {}", facts(objective, &steps)),
        format!("FINAL REPLY: {fin}"),
        "SELECTED STEPS:".to_owned(),
    ];
    out.extend(lines);
    out.join("\n")
}

/// Judge view of an episode: render the flat state + objective, then `view_sel`.
#[must_use]
pub fn episode_view(ep: &Episode, is_sub: bool) -> String {
    view_sel(&ep.render_state(), &ep.objective(is_sub), DEFAULT_BUDGET)
}
