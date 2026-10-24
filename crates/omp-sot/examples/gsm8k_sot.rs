//! GSM8K smoke bench for the SoT limited-access controllers.
//!
//! Compares three inference recipes on a deterministic stride sample of the
//! GSM8K test split against a local Ollama server:
//!
//! * `vanilla` — one direct-answer call.
//! * `cot` — one call, short sentence-by-sentence reasoning, single pass.
//! * `sot` — the closed loop from [`omp_sot::sot::run`] with the
//!   training-free operators and the SoT-Embed embedding tracker.
//!
//! The recipe is the paper's (Algorithm 1: active-evidence context, one
//! sentence per step, explicit final answer), not a reproduction of the
//! paper's trained-controller numbers — §4.1 / Appendix D report the full
//! training-free / embedding-only variants only as exploratory boundary
//! probes, and the deployment prompts there include the paper's own
//! scaffolding (sentence-level policy, calibrated thresholds) that this
//! harness approximates with exposed CLI flags.
//!
//! Data: `--test` / `--train` point at the openai/grade-school-math JSONL
//! splits. No dataset files are shipped; download both splits first.
//!
//! ```sh
//! cargo build -p omp-sot --example gsm8k_sot
//! ./target/debug/examples/gsm8k_sot \
//!     --test test.jsonl --train train.jsonl --out full60.jsonl \
//!     --n 60 --methods vanilla,cot,sot
//! ```

use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Instant;

use omp_sot::controller::{TrainingFreeEvidence, TrainingFreeStop};
use omp_sot::embed::{NearestPool, Pca, SentenceEncoder};
use omp_sot::segment;
use omp_sot::sot::{run, Backend, EmbeddingTracker, Outcome};
use omp_sot::state::{Normalizer, RawState};
use serde_json::{json, Value};

const HOST: &str = "127.0.0.1:11434";
const CHAT_MODEL: &str = "qwen2.5:7b";
const EMBED_MODEL: &str = "bge-m3";
const HTTP_TIMEOUT_S: u64 = 600;

// ------------------------------------------------------------------ http ----

fn http_post(path: &str, payload: &str) -> Result<String, String> {
    let body = payload.as_bytes();
    let request = format!(
        "POST {path} HTTP/1.1\r\nHost: {HOST}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    let mut stream = TcpStream::connect(HOST).map_err(|e| format!("connect {HOST}: {e}"))?;
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(HTTP_TIMEOUT_S)))
        .map_err(|e| e.to_string())?;
    stream
        .write_all(request.as_bytes())
        .map_err(|e| e.to_string())?;
    stream.write_all(body).map_err(|e| e.to_string())?;
    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).map_err(|e| e.to_string())?;
    let text = String::from_utf8_lossy(&raw);
    let (head, body) = text
        .split_once("\r\n\r\n")
        .ok_or_else(|| "malformed HTTP response".to_owned())?;
    let status = head
        .lines()
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|c| c.parse::<u16>().ok())
        .unwrap_or(0);
    if !(200..300).contains(&status) {
        return Err(format!("HTTP {status}: {}", &body[..body.len().min(300)]));
    }
    // Chunked encoding appears on streaming endpoints; /v1/chat/completions
    // without `stream` responds with Content-Length, but decode defensively.
    if head
        .lines()
        .any(|l| l.eq_ignore_ascii_case("transfer-encoding: chunked"))
    {
        let mut decoded = String::new();
        let mut rest = body;
        while let Some((size_str, after)) = rest.split_once("\r\n") {
            let size = usize::from_str_radix(size_str.trim(), 16).unwrap_or(0);
            if size == 0 {
                break;
            }
            if after.len() < size {
                break;
            }
            decoded.push_str(&after[..size]);
            rest = &after[size..];
        }
        Ok(decoded)
    } else {
        Ok(body.to_owned())
    }
}

fn post_json(path: &str, payload: &Value) -> Result<Value, String> {
    let text = http_post(
        path,
        &serde_json::to_string(payload).map_err(|e| e.to_string())?,
    )?;
    serde_json::from_str(&text)
        .map_err(|e| format!("JSON: {e}; body: {}", &text[..text.len().min(200)]))
}

/// One chat completion; returns (text, prompt_tokens, completion_tokens).
fn chat(prompt: &str, max_tokens: usize) -> Result<(String, u64, u64), String> {
    let resp = post_json(
        "/v1/chat/completions",
        &json!({
            "model": CHAT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": false,
        }),
    )?;
    let text = resp["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_default()
        .trim()
        .to_owned();
    let usage = &resp["usage"];
    Ok((
        text,
        usage["prompt_tokens"].as_u64().unwrap_or(0),
        usage["completion_tokens"].as_u64().unwrap_or(0),
    ))
}

// ------------------------------------------------------------------ data ----

struct Problem {
    question: String,
    reference: f64,
}

fn load_jsonl(path: &str) -> Result<Vec<Problem>, String> {
    let data = std::fs::read_to_string(path).map_err(|e| format!("read {path}: {e}"))?;
    let mut out = Vec::new();
    for line in data.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let row: Value = serde_json::from_str(line).map_err(|e| e.to_string())?;
        let question = row["question"]
            .as_str()
            .ok_or_else(|| "missing question".to_owned())?
            .replace("<<", "")
            .replace(">>", "");
        let answer = row["answer"]
            .as_str()
            .and_then(|a| a.split("####").nth(1))
            .map(str::trim)
            .and_then(|a| a.replace(',', "").parse::<f64>().ok())
            .ok_or_else(|| "no reference answer".to_owned())?;
        out.push(Problem {
            question,
            reference: answer,
        });
    }
    Ok(out)
}

/// Last monetary/number token in the text — robust to "The answer is $28.00."
/// and bare numbers. Strips `$`, thousands separators, and trailing
/// sentence periods ("28.00." -> 28.0).
fn extract_number(text: &str) -> Option<f64> {
    let mut best: Option<f64> = None;
    for tok in
        text.split(|c: char| !(c.is_ascii_digit() || c == '.' || c == ',' || c == '-' || c == '$'))
    {
        let clean = tok.trim_start_matches('$').replace(',', "");
        let clean = clean.trim_end_matches('.');
        if !clean.is_empty() && clean.chars().any(|c| c.is_ascii_digit()) {
            if let Ok(v) = clean.parse::<f64>() {
                best = Some(v);
            }
        }
    }
    best
}

// --------------------------------------------------------------- backend ----

/// Chat backend used by the SoT loop. Counts every call's usage so the
/// harness can report generated and re-sent prompt tokens separately.
struct OllamaBackend {
    prompt_tokens: u64,
    completion_tokens: u64,
    step_calls: u64,
    answer_calls: u64,
}

impl OllamaBackend {
    fn new() -> Self {
        Self {
            prompt_tokens: 0,
            completion_tokens: 0,
            step_calls: 0,
            answer_calls: 0,
        }
    }

    fn evidence_block(evidence: &[&str]) -> String {
        if evidence.is_empty() {
            String::new()
        } else {
            let list: String = evidence
                .iter()
                .enumerate()
                .map(|(i, e)| format!("{}. {e}", i + 1))
                .collect::<Vec<_>>()
                .join("\n");
            format!("\n\nReasoning so far (active evidence, most recent step last):\n{list}")
        }
    }
}

impl Backend for OllamaBackend {
    fn generate_step(&mut self, problem: &str, evidence: &[&str]) -> String {
        self.step_calls += 1;
        let prompt = format!(
            "{problem}{}\n\nContinue the reasoning with exactly one short sentence that \
             advances toward the final numeric answer. Do not repeat earlier steps. Do \
             not state the final answer yet.",
            Self::evidence_block(evidence),
        );
        let (text, p, c) = chat(&prompt, 64).expect("step generation");
        self.prompt_tokens += p;
        self.completion_tokens += c;
        text
    }

    fn generate_answer(&mut self, problem: &str, evidence: &[&str], last: &str) -> String {
        self.answer_calls += 1;
        let prompt = format!(
            "{problem}{}\n\nMost recent step: {last}\n\nState the final result as one line \
             in the exact form \"The answer is <number>\".",
            Self::evidence_block(evidence),
        );
        let (text, p, c) = chat(&prompt, 24).expect("answer generation");
        self.prompt_tokens += p;
        self.completion_tokens += c;
        text
    }
}

/// Fixed encoder backed by the local bge-m3 model (Eq. 28).
struct BgeEncoder;

impl SentenceEncoder for BgeEncoder {
    fn encode(&mut self, sentence: &str) -> Vec<f64> {
        let resp = post_json(
            "/api/embed",
            &json!({ "model": EMBED_MODEL, "input": [sentence] }),
        )
        .expect("bge-m3 embed");
        resp["embeddings"][0]
            .as_array()
            .map(|a| a.iter().filter_map(Value::as_f64).collect())
            .unwrap_or_default()
    }
}

// ------------------------------------------------------------------- cli ----

struct Args {
    test: String,
    train: String,
    out: String,
    n: usize,
    methods: Vec<String>,
    budget: usize,
    max_steps: usize,
    r_stop: usize,
    h_max: f64,
    v_max: f64,
    c_min: f64,
    norm: Option<String>,
    calibrate: Option<usize>,
    cot_sentences: usize,
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        test: "test.jsonl".into(),
        train: "train.jsonl".into(),
        out: "results.jsonl".into(),
        n: 60,
        methods: vec!["vanilla".into(), "cot".into(), "sot".into()],
        budget: 4,
        max_steps: 8,
        r_stop: 3,
        h_max: 1.0,
        v_max: 1.0,
        c_min: 0.0,
        norm: None,
        calibrate: None,
        cot_sentences: 6,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let value = it
            .next()
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--test" => args.test = value,
            "--train" => args.train = value,
            "--out" => args.out = value,
            "--norm" => args.norm = Some(value),
            "--calibrate" => args.calibrate = Some(value.parse().map_err(|_| "bad --calibrate")?),
            "--cot-sentences" => {
                args.cot_sentences = value.parse().map_err(|_| "bad --cot-sentences")?
            }
            "--n" => args.n = value.parse().map_err(|_| "bad --n")?,
            "--methods" => args.methods = value.split(',').map(str::to_owned).collect(),
            "--budget" => args.budget = value.parse().map_err(|_| "bad --budget")?,
            "--max-steps" => args.max_steps = value.parse().map_err(|_| "bad --max-steps")?,
            "--r-stop" => args.r_stop = value.parse().map_err(|_| "bad --r-stop")?,
            "--h-max" => args.h_max = value.parse().map_err(|_| "bad --h-max")?,
            "--v-max" => args.v_max = value.parse().map_err(|_| "bad --v-max")?,
            "--c-min" => args.c_min = value.parse().map_err(|_| "bad --c-min")?,
            _ => return Err(format!("unknown flag {flag}")),
        }
    }
    Ok(args)
}

// ------------------------------------------------------------------ main ----

fn main() -> Result<(), String> {
    let args = parse_args()?;
    let test = load_jsonl(&args.test)?;
    if test.is_empty() {
        return Err("empty test split".into());
    }
    // Deterministic stride sample across the full test split.
    let picked: Vec<usize> = (0..args.n).map(|i| i * test.len() / args.n).collect();
    eprintln!(
        "problems: {} (using {}), methods: {}",
        test.len(),
        picked.len(),
        args.methods.join(",")
    );

    // Fit the SoT-Embed subspace + pool on train-split reasoning sentences.
    let train = load_jsonl(&args.train)?;
    let mut sentences: Vec<String> = Vec::new();
    for p in train.iter().take(150) {
        sentences.extend(segment::sentences(&p.question));
    }
    eprintln!("fitting PCA/pool on {} train sentences", sentences.len());
    let mut embeddings = Vec::with_capacity(sentences.len());
    for chunk in sentences.chunks(32) {
        let resp = post_json(
            "/api/embed",
            &json!({ "model": EMBED_MODEL, "input": chunk }),
        )?;
        for e in resp["embeddings"].as_array().cloned().unwrap_or_default() {
            embeddings.push(
                e.as_array()
                    .map_or_else(Vec::new, |a| a.iter().filter_map(Value::as_f64).collect()),
            );
        }
    }
    let pca = Pca::fit(&embeddings, 8).ok_or_else(|| "PCA fit failed".to_owned())?;
    let pool = NearestPool::new(embeddings, 5);
    eprintln!("PCA k={} fitted", pca.k());

    // Optional calibration pass: run the loop on training problems with a
    // never-firing stop, fit the componentwise normalizer (Eq. 18) on the
    // collected raw states, and write it to `norm.json`.
    if let Some(cal_n) = args.calibrate {
        eprintln!("calibration pass on {cal_n} training problems");
        let mut raws: Vec<RawState> = Vec::new();
        for p in train.iter().take(cal_n) {
            let mut backend = OllamaBackend::new();
            let mut tracker = EmbeddingTracker::new(BgeEncoder, Some(pca.clone()), pool.clone());
            let mut never = TrainingFreeStop::new();
            never.r_stop = 0; // disable operator-driven stopping
            let outcome: Outcome = run(
                &mut backend,
                &mut tracker,
                &TrainingFreeEvidence::new(),
                &never,
                &Normalizer::identity(),
                &p.question,
                args.budget,
                args.max_steps,
            );
            raws.extend(outcome.trajectory.raw_states().iter().copied());
            eprintln!(
                "calibrate problem: steps={} correct={}",
                outcome.steps, "n/a"
            );
        }
        let norm = Normalizer::fit(&raws);
        let payload = json!({
            "mean": norm.mean().to_vec(),
            "std": norm.std().to_vec(),
        });
        std::fs::write(
            "norm.json",
            serde_json::to_string_pretty(&payload).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        eprintln!(
            "calibrated on {} raw states from {} problems -> norm.json (mean={:?} std={:?})",
            raws.len(),
            cal_n,
            norm.mean(),
            norm.std()
        );
        return Ok(());
    }

    let normalizer = match &args.norm {
        Some(path) => {
            let payload: Value =
                serde_json::from_str(&std::fs::read_to_string(path).map_err(|e| e.to_string())?)
                    .map_err(|e| e.to_string())?;
            let arr = |key: &str| -> Result<[f64; 4], String> {
                let a = payload[key]
                    .as_array()
                    .ok_or_else(|| format!("norm: missing {key}"))?;
                Ok([
                    a[0].as_f64().unwrap_or(0.0),
                    a[1].as_f64().unwrap_or(0.0),
                    a[2].as_f64().unwrap_or(0.0),
                    a[3].as_f64().unwrap_or(0.0),
                ])
            };
            let n = Normalizer::from_mean_std(arr("mean")?, arr("std")?);
            eprintln!("loaded normalizer from {path}");
            n
        }
        None => Normalizer::identity(),
    };

    let mut out =
        std::fs::File::create(&args.out).map_err(|e| format!("create {}: {e}", args.out))?;
    let mut records: Vec<Value> = Vec::new();

    for (k, &pi) in picked.iter().enumerate() {
        let problem = &test[pi];
        for method in &args.methods {
            let started = Instant::now();
            let record = match method.as_str() {
                "vanilla" => {
                    let prompt = format!(
                        "{}\n\nOutput the final answer immediately as one line in the exact \
                         form \"The answer is <number>\".",
                        problem.question
                    );
                    let (text, p, c) = chat(&prompt, 24)?;
                    json!({
                        "method": "vanilla", "problem": pi,
                        "correct": extract_number(&text) == Some(problem.reference),
                        "completion_tokens": c, "prompt_tokens": p,
                        "steps": 0, "wall_ms": started.elapsed().as_millis(),
                    })
                }
                "cot" => {
                    let prompt = format!(
                        "{}\n\nSolve the problem step by step with short sentences (at most \
                         {}). Then output the final line in the exact form \
                         \"The answer is <number>\".",
                        problem.question, args.cot_sentences
                    );
                    let (text, p, c) = chat(&prompt, 640)?;
                    json!({
                        "method": "cot", "problem": pi,
                        "correct": extract_number(&text) == Some(problem.reference),
                        "completion_tokens": c, "prompt_tokens": p,
                        "answer_text": text,
                        "steps": 1, "wall_ms": started.elapsed().as_millis(),
                    })
                }
                "sot" => {
                    let mut backend = OllamaBackend::new();
                    let mut tracker =
                        EmbeddingTracker::new(BgeEncoder, Some(pca.clone()), pool.clone());
                    let mut stop = TrainingFreeStop::new();
                    stop.r_stop = args.r_stop;
                    stop.h_max = args.h_max;
                    stop.v_max = args.v_max;
                    stop.c_min = args.c_min;
                    let outcome: Outcome = run(
                        &mut backend,
                        &mut tracker,
                        &TrainingFreeEvidence::new(),
                        &stop,
                        &normalizer,
                        &problem.question,
                        args.budget,
                        args.max_steps,
                    );
                    let states: Vec<Value> = outcome
                        .trajectory
                        .raw_states()
                        .iter()
                        .map(|s| json!({"delta": s.delta, "v": s.v, "c": s.c, "h": s.h}))
                        .collect();
                    let units: Vec<&str> = outcome
                        .trajectory
                        .units()
                        .iter()
                        .map(String::as_str)
                        .collect();
                    json!({
                        "method": "sot", "problem": pi,
                        "correct": extract_number(&outcome.answer) == Some(problem.reference),
                        "completion_tokens": backend.completion_tokens,
                        "prompt_tokens": backend.prompt_tokens,
                        "steps": outcome.steps,
                        "stop_reason": format!("{:?}", outcome.reason),
                        "answer": outcome.answer,
                        "states": states,
                        "units": units,
                        "wall_ms": started.elapsed().as_millis(),
                    })
                }
                other => return Err(format!("unknown method {other}")),
            };
            let correct = record["correct"].as_bool().unwrap_or(false);
            eprintln!(
                "[{}/{}] {method} problem={pi} correct={correct} steps={} completion_tok={}",
                k + 1,
                args.n,
                record["steps"],
                record["completion_tokens"],
            );
            records.push(record);
        }
    }

    // Summary table.
    println!(
        "\n=== GSM8K SoT smoke bench (n={} problems) ===",
        picked.len()
    );
    println!(
        "{:<10} {:>6} {:>10} {:>11} {:>7} {:>8}",
        "method", "acc%", "compl_tok", "prompt_tok", "steps", "wall_s"
    );
    for method in &args.methods {
        let rs: Vec<&Value> = records
            .iter()
            .filter(|r| r["method"].as_str() == Some(method.as_str()))
            .collect();
        if rs.is_empty() {
            continue;
        }
        let acc = 100.0
            * rs.iter()
                .filter(|r| r["correct"].as_bool().unwrap_or(false))
                .count() as f64
            / rs.len() as f64;
        let mean = |key: &str| -> f64 {
            rs.iter().filter_map(|r| r[key].as_f64()).sum::<f64>() / rs.len() as f64
        };
        println!(
            "{:<10} {:>6.1} {:>10.1} {:>11.1} {:>7.1} {:>8.1}",
            method,
            acc,
            mean("completion_tokens"),
            mean("prompt_tokens"),
            mean("steps"),
            mean("wall_ms") / 1000.0,
        );
    }

    for r in &records {
        let mut line = serde_json::to_string(r).map_err(|e| e.to_string())?;
        line.push('\n');
        out.write_all(line.as_bytes()).map_err(|e| e.to_string())?;
    }
    Ok(())
}
