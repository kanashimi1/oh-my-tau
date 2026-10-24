//! Golden test: the Rust `view_sel` must reproduce the Python `judge_view.py::view_sel`
//! byte for byte on 50 synthetic `traj_outcome`-shaped episodes (see `fixtures/gen_synthetic.py`).

use omp_judge::episode::Objective;
use omp_judge::view::{view_sel, DEFAULT_BUDGET};
use serde_json::Value;

#[test]
fn view_sel_matches_python_on_50_episodes() {
    let text = include_str!("fixtures/golden_view_sel.jsonl");
    let mut n = 0;
    for line in text.lines() {
        let r: Value = serde_json::from_str(line).unwrap();
        let got = view_sel(
            r["state"].as_str().unwrap(),
            &Objective::from_value(&r["objective"]),
            DEFAULT_BUDGET,
        );
        assert_eq!(
            got,
            r["expected"].as_str().unwrap(),
            "view mismatch for {}",
            r["id"]
        );
        n += 1;
    }
    assert_eq!(n, 50);
}

/// Whole-family check: `OMP_JUDGE_GOLDEN_ALL=/path/all.jsonl` (same record shape as the fixture).
#[test]
#[ignore = "needs OMP_JUDGE_GOLDEN_ALL pointing at a Python-generated dump of all traj_outcome episodes"]
fn view_sel_matches_python_on_all_episodes() {
    let path = std::env::var("OMP_JUDGE_GOLDEN_ALL").expect("OMP_JUDGE_GOLDEN_ALL");
    let text = std::fs::read_to_string(path).unwrap();
    let mut bad = 0;
    let mut n = 0;
    for line in text.lines() {
        let r: Value = serde_json::from_str(line).unwrap();
        let got = view_sel(
            r["state"].as_str().unwrap(),
            &Objective::from_value(&r["objective"]),
            DEFAULT_BUDGET,
        );
        n += 1;
        if got != r["expected"].as_str().unwrap() {
            bad += 1;
            let e = r["expected"].as_str().unwrap();
            let p = got.chars().zip(e.chars()).position(|(a, b)| a != b);
            eprintln!("MISMATCH {} at {p:?}\nGOT:\n{got}\nEXP:\n{e}", r["id"]);
        }
    }
    assert_eq!(bad, 0, "{bad}/{n} views differ");
}
