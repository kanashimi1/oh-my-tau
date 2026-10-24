//! laya-judge: JSONL in -> per-question probabilities out.
//!
//! Usage:
//!   laya-judge IN.jsonl OUT.jsonl [--model CKPT_DIR] [--onnx FILE] [--max-len N]
//!                                 [--threads N] [--dump-tokens FILE]
//!
//! OUT lines: {"id", "qid", "options": [...], "probs": [...]} — same schema as
//! the Python `laya_predict.py` reference. --dump-tokens writes the built token ids per
//! question ({"id", "qid", "ids", "markers"}) for parity checking.
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use laya_ort::{Judge, LayaTokenizer};
use serde_json::{json, Value};

fn arg_value(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

/// `$OMP_JUDGE_CKPT`, else `$HOME/.omp-rs/judge/tj/ckpt`.
fn default_ckpt() -> String {
    if let Ok(p) = std::env::var("OMP_JUDGE_CKPT") {
        return p;
    }
    let home = std::env::var("HOME").unwrap_or_default();
    format!("{home}/.omp-rs/judge/tj/ckpt")
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!("usage: laya-judge IN.jsonl OUT.jsonl [--model DIR] [--onnx FILE] [--threads N] [--dump-tokens FILE]");
        std::process::exit(2);
    }
    let src = &args[0];
    let dst = &args[1];
    let model_dir: PathBuf = arg_value(&args, "--model")
        .unwrap_or_else(default_ckpt)
        .into();
    let onnx: PathBuf = arg_value(&args, "--onnx")
        .unwrap_or_else(|| {
            let home = std::env::var("HOME").unwrap_or_default();
            format!("{home}/.omp-rs/judge/laya_judge.onnx")
        })
        .into();
    let max_len: usize = arg_value(&args, "--max-len").map(|s| s.parse().unwrap()).unwrap_or(1024);
    let threads: usize = arg_value(&args, "--threads").map(|s| s.parse().unwrap()).unwrap_or(1);
    let dump_tokens = arg_value(&args, "--dump-tokens").map(PathBuf::from);

    if threads > 1 {
        rayon::ThreadPoolBuilder::new().num_threads(threads).build_global().ok();
    }
    // Limit ONNX Runtime intra-op threads to `threads`.
    std::env::set_var("OMP_NUM_THREADS", threads.to_string());

    let records: Vec<Value> = std::fs::read_to_string(src)?
        .lines()
        .map(|l| serde_json::from_str(l).context("parse record"))
        .collect::<Result<_>>()?;

    if let Some(dump) = dump_tokens {
        // Token-id parity mode: dump built sequences without running the model.
        let cfg: Value = serde_json::from_str(&std::fs::read_to_string(model_dir.join("rl_agent_config.json"))?)?;
        let head_max_len = cfg["head_max_len"].as_u64().unwrap_or(192) as usize;
        let tok = LayaTokenizer::from_dir(&model_dir.join("tokenizer"))?;
        let mut w = BufWriter::new(std::fs::File::create(&dump)?);
        let mut n = 0usize;
        for r in &records {
            let state = r["state"].as_str().unwrap_or("");
            for (qid, q) in r["questions"].as_object().cloned().unwrap_or_default() {
                let t: Value = if q["type"].as_str() == Some("bool") {
                    Value::from("noul")
                } else {
                    q["type"].clone()
                };
                let internal = json!({
                    "t": t,
                    "ins": q["instructions"],
                    "crit": q.get("criteria").unwrap_or(&Value::Null),
                });
                let seq = tok.build_sequence(state, &internal, max_len, head_max_len, false);
                w.write_all(
                    json!({"id": r["id"], "qid": qid, "ids": seq.ids, "markers": seq.markers}).to_string().as_bytes(),
                )?;
                w.write_all(b"\n")?;
                n += 1;
            }
        }
        w.flush()?;
        eprintln!("{n} questions, token dump -> {}", dump.display());
        return Ok(());
    }

    let mut judge = Judge::load(&model_dir, &onnx, threads)?;
    let mut w = BufWriter::new(std::fs::File::create(dst)?);
    let t0 = Instant::now();
    let mut n = 0usize;
    for r in &records {
        let state = r["state"].as_str().unwrap_or("");
        for qp in judge.judge(state, &r["questions"])? {
            w.write_all(
                json!({
                    "id": r["id"],
                    "qid": qp.qid,
                    "options": qp.options,
                    "probs": qp.probs.iter().map(|p| ((*p as f64) * 1e5).round() / 1e5).collect::<Vec<f64>>(),
                })
                .to_string()
                .as_bytes(),
            )?;
            w.write_all(b"\n")?;
            n += 1;
        }
    }
    w.flush()?;
    let dt = t0.elapsed().as_secs_f64();
    eprintln!("{n} questions, {:.1} ms/q -> {}", 1000.0 * dt / n.max(1) as f64, dst);
    Ok(())
}
