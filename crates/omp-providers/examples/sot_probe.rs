#![allow(clippy::disallowed_methods)] // tests/entry points: throwaway runtimes drive async code
//! SoT (State-of-Thought) streaming early-exit probe.
//!
//! `cargo run --release -p omp-providers --example sot_probe -- \
//!    <model_ref> <base|sot> <qid> <question-file>`
//!
//! Streams one question; in `sot` mode sentence-chunks the thinking stream,
//! embeds each chunk with local bge-m3 (ollama), computes v_t / c_t / doubt
//! density, fires a pre-registered stagnation rule, aborts the stream, then
//! finalizes with a short no-reasoning call. Emits one JSON line on stdout.
use futures::StreamExt;
use pa_ai::types::{AssistantMessageEvent, Context, Message, UserMessage, UserMessageContent};
use serde_json::{json, Value};
use std::time::Instant;

// ---- pre-registered controller constants (fixed before any run) ----
const WARMUP_CHUNKS: usize = 8; // no abort before this many chunks
const STAG_V_FRAC: f64 = 0.30; // mean(v last3) < 30% of running median
const STAG_C_MIN: f64 = 0.55; // and mean(c last3) > 0.55
const DOUBT_WIN: usize = 4; // doubt-loop window
const DOUBT_DENS: f64 = 0.10; // each chunk >10% doubt-marker words
const DOUBT_V_FRAC: f64 = 0.50; // and v suppressed vs median
const MIN_WORDS: usize = 25; // chunk floor to be scored
const CUT_WORDS: usize = 30; // sentence cut target
const FORCE_CUT_WORDS: usize = 80;

const DOUBT_MARKERS: &[&str] = &[
    "wait",
    "but ",
    "however",
    "hmm",
    "actually",
    "let me",
    "double-check",
    "recheck",
    "alternatively",
    "hold on",
    "on the other hand",
    "но ",
    "однако",
    "хотя",
    "стоп",
    "проверим",
    "подожди",
    "возможно",
    "вернём",
    "вернем",
    "а если",
    "впрочем",
];

fn doubt_density(text: &str) -> f64 {
    let low = text.to_lowercase();
    let w = low.split_whitespace().count().max(1);
    let hits: usize = DOUBT_MARKERS
        .iter()
        .map(|m| low.matches(m.trim()).count())
        .sum();
    hits as f64 / w as f64
}

fn words(s: &str) -> Vec<String> {
    s.split_whitespace()
        .map(|w| {
            w.trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase()
        })
        .collect()
}

struct Chunker {
    buf: String,
    out: Vec<String>,
    n_words: usize,
}
impl Chunker {
    fn new() -> Self {
        Self {
            buf: String::new(),
            out: Vec::new(),
            n_words: 0,
        }
    }
    fn push(&mut self, delta: &str) {
        self.buf.push_str(delta);
        self.n_words = self.buf.split_whitespace().count();
        let ends_sentence = self.buf.ends_with(['.', '!', '?', '\n']);
        if (ends_sentence && self.n_words >= CUT_WORDS) || self.n_words >= FORCE_CUT_WORDS {
            self.cut();
        }
    }
    fn cut(&mut self) {
        let t = std::mem::take(&mut self.buf).trim().to_string();
        self.n_words = 0;
        if !t.is_empty() {
            self.out.push(t);
        }
    }
}

async fn embed(client: &reqwest::Client, text: &str) -> Option<Vec<f64>> {
    let r = client
        .post("http://127.0.0.1:11434/api/embed")
        .json(&json!({"model": "bge-m3", "input": [text]}))
        .send()
        .await
        .ok()?;
    let v: Value = r.json().await.ok()?;
    v["embeddings"][0]
        .as_array()
        .map(|a| a.iter().filter_map(|x| x.as_f64()).collect())
}

fn dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}
fn cos(a: &[f64], b: &[f64]) -> f64 {
    let (mut d, mut na, mut nb) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        d += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        d / (na.sqrt() * nb.sqrt())
    }
}
fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    // NaN probe values sort equal rather than crashing the probe.
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = s.len();
    if n == 0 {
        0.0
    } else if n % 2 == 1 {
        s[n / 2]
    } else {
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    }
}

// invariant: the json! literal is a well-formed user message shape.
#[allow(clippy::unwrap_used, clippy::expect_used)]
fn user_msg(text: &str) -> Message {
    Message::User(UserMessage {
        content: UserMessageContent::Text(text.into()),
        ..serde_json::from_value(json!({"role":"user","content":"","timestamp":0}))
            .expect("literal user message shape")
    })
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let model_ref = args.remove(0);
    let mode = args.remove(0); // base | sot
    let qid = args.remove(0);
    let qfile = args.remove(0);
    let question = std::fs::read_to_string(&qfile)?.trim().to_string();

    let m = omp_providers::resolve(&model_ref)?;
    let http = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(20))
        .build()?;

    let t0 = Instant::now();
    // ---- pass 1: stream the question ----
    let ctx = Context {
        system_prompt: None,
        messages: vec![user_msg(&question)],
        tools: None,
    };
    let mut o = omp_providers::SimpleStreamOptions::default();
    o.base.api_key.clone_from(&m.api_key);
    o.reasoning = m.thinking;
    // abort-by-drop: leaving the loop drops the stream, closing the connection
    let mut s = omp_providers::stream(&m, ctx, Some(o)).await?;

    let mut chunker = Chunker::new();
    let mut thinking = String::new();
    let mut answer1 = String::new();
    let mut usage1: Option<Value> = None;
    let mut aborted = false;
    let mut abort_at_ms: u64 = 0;
    let mut hs: Vec<Vec<f64>> = Vec::new();
    let mut vs: Vec<f64> = Vec::new();
    let mut cs: Vec<f64> = Vec::new();
    let mut doubts: Vec<f64> = Vec::new();
    let mut fired: Option<String> = None;

    while let Some(ev) = s.next().await {
        match ev {
            AssistantMessageEvent::ThinkingDelta { delta, .. } => {
                thinking.push_str(&delta);
                chunker.push(&delta);
                if mode == "sot" {
                    while chunker.out.len() > hs.len() + 1 {
                        let _ = chunker.out.remove(0); // not expected; safety
                    }
                    // score every completed chunk
                    while hs.len() < chunker.out.len() {
                        let i = hs.len();
                        let text = chunker.out[i].clone();
                        let w = words(&text);
                        if w.len() < MIN_WORDS {
                            hs.push(Vec::new());
                            vs.push(f64::NAN);
                            cs.push(f64::NAN);
                            doubts.push(f64::NAN);
                            continue;
                        }
                        let dens = doubt_density(&text);
                        let h = embed(&http, &text).await.unwrap_or_default();
                        hs.push(h.clone());
                        let n = vs.len();
                        if n >= 1 {
                            let v = if hs[n].is_empty() || hs[n - 1].is_empty() {
                                f64::NAN
                            } else {
                                dist(&hs[n], &hs[n - 1])
                            };
                            vs.push(v);
                        } else {
                            vs.push(f64::NAN);
                        }
                        if n >= 2 {
                            let c =
                                if hs[n].is_empty() || hs[n - 1].is_empty() || hs[n - 2].is_empty()
                                {
                                    f64::NAN
                                } else {
                                    cos(&sub(&hs[n], &hs[n - 1]), &sub(&hs[n - 1], &hs[n - 2]))
                                };
                            cs.push(c);
                        } else {
                            cs.push(f64::NAN);
                        }
                        doubts.push(dens);

                        // ---- forced-abort override (analysis only): OMT_SOT_FIRE_AT=<chunk> ----
                        if fired.is_none() {
                            if let Ok(at) = std::env::var("OMT_SOT_FIRE_AT") {
                                if let Ok(at) = at.trim().parse::<usize>() {
                                    if vs.len() >= at {
                                        fired = Some("forced".into());
                                    }
                                }
                            }
                        }
                        // ---- pre-registered stop rule ----
                        if fired.is_none() && vs.len() >= WARMUP_CHUNKS {
                            let med = median(
                                &vs[..vs.len() - 1]
                                    .iter()
                                    .cloned()
                                    .filter(|x| !x.is_nan())
                                    .collect::<Vec<_>>(),
                            );
                            if med > 1e-9 {
                                let l3: Vec<f64> = vs[vs.len() - 3..]
                                    .iter()
                                    .cloned()
                                    .filter(|x| !x.is_nan())
                                    .collect();
                                let c3: Vec<f64> = cs[cs.len() - 3..]
                                    .iter()
                                    .cloned()
                                    .filter(|x| !x.is_nan())
                                    .collect();
                                let mv = l3.iter().sum::<f64>() / l3.len().max(1) as f64;
                                let mc = c3.iter().sum::<f64>() / c3.len().max(1) as f64;
                                if !l3.is_empty() && mv < STAG_V_FRAC * med && mc > STAG_C_MIN {
                                    fired = Some("stagnation".into());
                                }
                                let l4d = &doubts[doubts.len() - DOUBT_WIN..];
                                let l4v: Vec<f64> = vs[vs.len() - DOUBT_WIN..]
                                    .iter()
                                    .cloned()
                                    .filter(|x| !x.is_nan())
                                    .collect();
                                let mv4 = l4v.iter().sum::<f64>() / l4v.len().max(1) as f64;
                                if l4d.iter().all(|d| *d > DOUBT_DENS)
                                    && !l4v.is_empty()
                                    && mv4 < DOUBT_V_FRAC * med
                                {
                                    fired = Some("doubt_loop".into());
                                }
                            }
                        }
                        if fired.is_some() {
                            break;
                        }
                    }
                }
                if fired.is_some() {
                    break;
                }
            }
            AssistantMessageEvent::TextDelta { delta, .. } => answer1.push_str(&delta),
            AssistantMessageEvent::Done { message, .. } => {
                usage1 = serde_json::to_value(message.usage)
                    .ok()
                    .filter(|v| !v.is_null());
            }
            AssistantMessageEvent::Error { .. } => break,
            _ => {}
        }
        if fired.is_some() {
            break;
        }
    }
    if fired.is_some() {
        aborted = true;
        abort_at_ms = t0.elapsed().as_millis() as u64;
        drop(s); // close the connection: generated-token billing stops here
        chunker.cut();
    } else {
        chunker.cut();
    }

    // in base mode: score all chunks post-hoc for trace analysis
    if mode == "base" {
        let mut ch = Chunker::new();
        for line in thinking.split_inclusive(['.', '!', '?', '\n']) {
            ch.push(line);
        }
        ch.cut();
        chunker.out = ch.out;
        for text in chunker.out.clone() {
            let w = words(&text);
            if w.len() < MIN_WORDS {
                hs.push(Vec::new());
                vs.push(f64::NAN);
                cs.push(f64::NAN);
                doubts.push(f64::NAN);
                continue;
            }
            let dens = doubt_density(&text);
            let h = embed(&http, &text).await.unwrap_or_default();
            hs.push(h.clone());
            let n = vs.len();
            if n >= 1 {
                vs.push(if hs[n].is_empty() || hs[n - 1].is_empty() {
                    f64::NAN
                } else {
                    dist(&hs[n], &hs[n - 1])
                });
            } else {
                vs.push(f64::NAN);
            }
            if n >= 2 {
                cs.push(
                    if hs[n].is_empty() || hs[n - 1].is_empty() || hs[n - 2].is_empty() {
                        f64::NAN
                    } else {
                        cos(&sub(&hs[n], &hs[n - 1]), &sub(&hs[n - 1], &hs[n - 2]))
                    },
                );
            } else {
                cs.push(f64::NAN);
            }
            doubts.push(dens);
        }
    }

    // ---- pass 2 (sot finalize): short no-reasoning call ----
    let (answer2, usage2) = if aborted {
        let prompt = format!(
            "{question}\n\n[Промежуточные выводы моего рассуждения]\n{thinking}\n\nНа основе приведенных выше выводов немедленно дай окончательный ответ. Формат: ровно одна строка «Ответ: <значение>». Не рассуждай заново.",
        );
        let ctx2 = Context {
            system_prompt: None,
            messages: vec![user_msg(&prompt)],
            tools: None,
        };
        let mut o = omp_providers::SimpleStreamOptions::default();
        o.base.api_key.clone_from(&m.api_key);
        o.disable_reasoning = true;
        o.base.max_tokens = Some(300);
        match omp_providers::stream(&m, ctx2, Some(o)).await {
            Ok(mut s2) => {
                let mut txt = String::new();
                let mut u2 = None;
                while let Some(ev) = s2.next().await {
                    match ev {
                        AssistantMessageEvent::TextDelta { delta, .. } => txt.push_str(&delta),
                        AssistantMessageEvent::Done { message, .. } => {
                            u2 = serde_json::to_value(message.usage)
                                .ok()
                                .filter(|v| !v.is_null());
                        }
                        AssistantMessageEvent::Error { .. } => break,
                        _ => {}
                    }
                }
                (txt, u2)
            }
            Err(_) => (String::new(), None),
        }
    } else {
        (String::new(), None)
    };

    let round = |v: &[f64]| -> Vec<Value> {
        v.iter()
            .map(|x| {
                if x.is_nan() {
                    Value::Null
                } else {
                    json!((x * 10_000.0).round() / 10_000.0)
                }
            })
            .collect()
    };
    println!(
        "{}",
        json!({
            "qid": qid, "model": model_ref, "mode": mode,
            "aborted": aborted, "fired": fired, "abort_at_ms": abort_at_ms,
            "wall_ms": t0.elapsed().as_millis() as u64,
            "thinking_chars": thinking.chars().count(),
            "answer1_chars": answer1.chars().count(),
            "answer2_chars": answer2.chars().count(),
            "answer1_tail": answer1.chars().rev().take(400).collect::<String>().chars().rev().collect::<String>(),
            "answer2_tail": answer2.chars().rev().take(400).collect::<String>().chars().rev().collect::<String>(),
            "chunks": chunker.out.len(),
            "v": round(&vs), "c": round(&cs), "doubt": round(&doubts),
            "usage1": usage1, "usage2": usage2,
        })
    );
    Ok(())
}

fn sub(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x - y).collect()
}
