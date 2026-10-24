//! Laya judge inference via ONNX Runtime (CPU).
//!
//! Reimplements laya.common.build_sequence / render_options with the `tokenizers`
//! crate so token ids match the Python fast tokenizer exactly, and applies the
//! per-bucket temperature from rl_agent_config.json (same as laya-train-v2/train.py).
#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use ort::session::Session;
use serde_json::Value;
use tokenizers::Tokenizer;

pub const QTYPES: [(&str, i64); 3] = [("choice", 0), ("score", 1), ("noul", 2)];

/// Python json.dumps(value, ensure_ascii=False, separators=(", ", ": "))
/// for a serde_json::Value (objects/arrays keep insertion order).
pub fn render_criterion(value: &Value) -> String {
    if let Some(s) = value.as_str() {
        return s.to_string();
    }
    fn esc(s: &str) -> String {
        serde_json::to_string(s).unwrap_or_else(|_| format!("\"{s}\""))
    }
    match value {
        Value::Null => "null".into(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => esc(s),
        Value::Array(a) => format!("[{}]", a.iter().map(py_json).collect::<Vec<_>>().join(", ")),
        Value::Object(o) => format!(
            "{{{}}}",
            o.iter()
                .map(|(k, v)| format!("{}: {}", esc(k), py_json(v)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn py_json(v: &Value) -> String {
    render_criterion(v)
}

/// Render option texts in label-index order (laya.common.render_options).
pub fn render_options(q: &Value) -> Vec<String> {
    let t = q["t"].as_str().unwrap_or("");
    let crit = q.get("crit").cloned().unwrap_or(Value::Null);
    if t == "choice" {
        let obj = crit.as_object().cloned().unwrap_or_default();
        return obj
            .iter()
            .map(|(k, v)| {
                if v.is_null() || (v.as_str() == Some("")) {
                    k.clone()
                } else {
                    format!("{}: {}", k, render_criterion(v))
                }
            })
            .collect();
    }
    if t == "score" {
        let arr = crit.as_array().cloned().unwrap_or_default();
        return arr
            .iter()
            .enumerate()
            .map(|(i, c)| format!("level {}: {}", i, render_criterion(c)))
            .collect();
    }
    let obj = crit.as_object().cloned().unwrap_or_default();
    let def_false = "no, the statement does not hold".to_string();
    let def_true = "yes, the statement holds".to_string();
    let false_crit = obj.get("false").cloned().unwrap_or(Value::Null);
    let true_crit = obj.get("true").cloned().unwrap_or(Value::Null);
    let f = if false_crit.is_null() || (false_crit.as_str() == Some("")) {
        def_false
    } else {
        render_criterion(&false_crit)
    };
    let tr = if true_crit.is_null() || (true_crit.as_str() == Some("")) {
        def_true
    } else {
        render_criterion(&true_crit)
    };
    vec![format!("false: {}", f), format!("true: {}", tr)]
}

/// One built sequence (mirrors laya.common.build_sequence output).
#[derive(Debug, Clone)]
pub struct Sequence {
    pub ids: Vec<i64>,
    pub markers: Vec<usize>,
}

/// Tokenizer + special token ids loaded from a checkpoint dir (tokenizer/ + tokenizer_config.json).
pub struct LayaTokenizer {
    tok: Tokenizer,
    pub cls_id: i64,
    pub sep_id: i64,
    pub mask_id: i64,
    pub pad_id: i64,
    pub mask_tok: String,
}

impl LayaTokenizer {
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let tok = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow::anyhow!("tokenizer load: {e}"))?;
        let cfg: Value = serde_json::from_str(
            &std::fs::read_to_string(dir.join("tokenizer_config.json")).context("tokenizer_config.json")?,
        )?;
        let s = |k: &str| -> Result<String> {
            cfg[k]
                .as_str()
                .map(str::to_string)
                .with_context(|| format!("tokenizer_config.json missing {k}"))
        };
        let id_of = |tok_str: &str| -> Result<i64> {
            tok.token_to_id(tok_str)
                .map(|u| u as i64)
                .with_context(|| format!("token {tok_str:?} not in vocab"))
        };
        let mask_tok = s("mask_token")?;
        Ok(Self {
            cls_id: id_of(&s("cls_token")?)?,
            sep_id: id_of(&s("sep_token")?)?,
            mask_id: id_of(&mask_tok)?,
            pad_id: id_of(&s("pad_token")?)?,
            mask_tok,
            tok,
        })
    }

    /// tok(text, add_special_tokens=False)["input_ids"]
    pub fn encode_no_special(&self, text: &str) -> Vec<i64> {
        self.tok
            .encode(text, false)
            .map(|e| e.get_ids().iter().map(|&u| u as i64).collect())
            .unwrap_or_default()
    }

    /// Exact port of laya.common.build_sequence. Returns (ids, markers).
    pub fn build_sequence(
        &self,
        state: &str,
        q: &Value,
        max_len: usize,
        head_max_len: usize,
        truncate_left: bool,
    ) -> Sequence {
        let mask_tok = self.mask_tok.as_str();
        let opts = render_options(q);
        let ins = q["ins"].as_str().unwrap_or("").replace(mask_tok, " ");
        let head_text = format!("{} question: {}", q["t"].as_str().unwrap_or(""), ins);
        let mut head_ids = self.encode_no_special(&head_text);
        let mut opt_ids: Vec<Vec<i64>> = opts
            .iter()
            .map(|o| {
                let mut v = vec![self.mask_id];
                v.extend(self.encode_no_special(&format!(" {}", o.replace(mask_tok, " "))).iter().take(48));
                v
            })
            .collect();
        let mut opt_budget = head_max_len as i64 - opt_ids.iter().map(|o| o.len() as i64).sum::<i64>();
        if opt_budget < 16 {
            let per = ((head_max_len as i64 - 16) / opt_ids.len().max(1) as i64).max(4) as usize;
            opt_ids = opt_ids.iter().map(|o| o[..per.min(o.len())].to_vec()).collect();
            opt_budget = head_max_len as i64 - opt_ids.iter().map(|o| o.len() as i64).sum::<i64>();
        }
        let head_keep = (opt_budget.max(8)) as usize;
        head_ids.truncate(head_keep);
        let mut ids = vec![self.cls_id];
        ids.extend(&head_ids);
        ids.push(self.sep_id);
        let mut markers = Vec::new();
        for o in &opt_ids {
            markers.push(ids.len());
            ids.extend(o);
        }
        ids.push(self.sep_id);
        let room = max_len.saturating_sub(ids.len() + 1);
        let mut st = self.encode_no_special(&state.replace(mask_tok, " "));
        if truncate_left {
            let skip = st.len().saturating_sub(room);
            st = st[skip..].to_vec();
        } else {
            st.truncate(room);
        }
        ids.extend(st);
        ids.push(self.sep_id);
        ids.truncate(max_len);
        let markers: Vec<usize> = markers.into_iter().filter(|&m| m < max_len).collect();
        Sequence { ids, markers }
    }
}

/// temp_bucket from laya.common.
pub fn temp_bucket(qtype: i64, k: usize) -> String {
    let name = match qtype {
        0 => "choice",
        1 => "score",
        _ => "noul",
    };
    let size = if k <= 2 {
        "2"
    } else if k <= 5 {
        "3-5"
    } else if k <= 10 {
        "6-10"
    } else {
        "11+"
    };
    format!("{}:{}", name, size)
}

/// train.py softmax: exp((x - max) / t) normalized.
pub fn softmax_with_temp(z: &[f32], t: f32) -> Vec<f32> {
    let m = z.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = z.iter().map(|x| ((x - m) / t).exp()).collect();
    let s: f32 = e.iter().sum();
    e.iter().map(|x| x / s).collect()
}

/// Per-bucket temperatures from rl_agent_config.json (temperature_by_options).
pub struct Temperatures {
    pub by_options: HashMap<String, f64>,
}

impl Temperatures {
    pub fn from_config(cfg: &Value) -> Self {
        let mut by_options = HashMap::new();
        if let Some(m) = cfg.get("temperature_by_options").and_then(|v| v.as_object()) {
            for (k, v) in m {
                if let Some(t) = v.as_f64() {
                    by_options.insert(k.clone(), t);
                }
            }
        }
        Self { by_options }
    }
    pub fn get(&self, qtype: i64, k: usize) -> f32 {
        self.by_options
            .get(&temp_bucket(qtype, k))
            .copied()
            .unwrap_or(1.0) as f32
    }
}

/// Result for one question.
#[derive(Debug)]
pub struct QuestionProbs {
    pub qid: String,
    pub options: Vec<String>,
    pub probs: Vec<f32>,
}

fn qtype_of(t: &str) -> i64 {
    QTYPES.iter().find(|(n, _)| *n == t).map(|(_, v)| *v).unwrap_or(0)
}

/// Load rl_agent_config.json + tokenizer.json + model.onnx from a checkpoint dir.
pub struct Judge {
    pub env: ort::environment::Environment,
    pub session: Session,
    pub tok: LayaTokenizer,
    pub temps: Temperatures,
    pub max_len: usize,
    pub head_max_len: usize,
}

impl Judge {
    pub fn load(ckpt: &Path, onnx_path: &Path, threads: usize) -> Result<Self> {
        let cfg: Value = serde_json::from_str(
            &std::fs::read_to_string(ckpt.join("rl_agent_config.json")).context("rl_agent_config.json")?,
        )?;
        let max_len = cfg["max_len"].as_u64().unwrap_or(1024) as usize;
        let head_max_len = cfg["head_max_len"].as_u64().unwrap_or(192) as usize;
        let env = ort::init().with_name("laya-ort").build().map_err(ort_err)?;
        let session = Session::builder(&env)
            .map_err(ort_err)?
            .with_intra_threads(threads)
            .map_err(ort_err)?
            .commit_from_file(onnx_path)
            .map_err(ort_err)?;
        Ok(Self {
            env,
            session,
            tok: LayaTokenizer::from_dir(&ckpt.join("tokenizer"))?,
            temps: Temperatures::from_config(&cfg),
            max_len,
            head_max_len,
        })
    }

    /// Judge every question of one record. `questions` is the record's questions object.
    pub fn judge(&mut self, state: &str, questions: &Value) -> Result<Vec<QuestionProbs>> {
        let mut out = Vec::new();
        let obj = questions.as_object().context("questions not an object")?;
        // Batches of sequences padded like laya.common.collate_items.
        let mut items: Vec<(String, Vec<String>, i64, Sequence, usize)> = Vec::new();
        for (qid, q) in obj {
            let t: Value = if q["type"].as_str() == Some("bool") {
                Value::from("noul")
            } else {
                q["type"].clone()
            };
            let internal = serde_json::json!({
                "t": t,
                "ins": q["instructions"],
                "crit": q.get("criteria").unwrap_or(&Value::Null),
            });
            let options: Vec<String> = match q["type"].as_str().unwrap_or("") {
                "choice" => q["criteria"]
                    .as_object()
                    .map(|m| m.keys().cloned().collect())
                    .unwrap_or_default(),
                "bool" => vec!["false".to_string(), "true".to_string()],
                _ => q["criteria"]
                    .as_array()
                    .map(|a| (0..a.len()).map(|i| i.to_string()).collect())
                    .unwrap_or_default(),
            };
            let seq = self.tok.build_sequence(state, &internal, self.max_len, self.head_max_len, false);
            if seq.markers.len() != options.len() {
                eprintln!("skip {qid}: {} markers vs {} options", seq.markers.len(), options.len());
                continue;
            }
            let qt = qtype_of(internal["t"].as_str().unwrap_or(""));
            items.push((qid.clone(), options, qt, seq, options_len(&internal)));
        }
        // Simple batching: process in chunks with a token budget, like train.py batches().
        let mut start = 0;
        let mut results: Vec<Option<(String, Vec<String>, Vec<f32>)>> = vec![None; items.len()];
        let mut order: Vec<usize> = (0..items.len()).collect();
        order.sort_by_key(|&i| items[i].3.ids.len());
        while start < order.len() {
            let mut end = start;
            let mut longest = 0usize;
            while end < order.len() {
                let l = items[order[end]].3.ids.len();
                if end > start && (longest.max(l)) * (end - start + 1) > 8192 {
                    break;
                }
                longest = longest.max(l);
                end += 1;
            }
            let batch = &order[start..end];
            start = end;
            let bsz = batch.len();
            let l = batch.iter().map(|&i| items[i].3.ids.len()).max().unwrap();
            let m = batch.iter().map(|&i| items[i].3.markers.len()).max().unwrap();
            let mut ids = vec![self.tok.pad_id; bsz * l];
            let mut attn = vec![0i64; bsz * l];
            let mut mpos = vec![0i64; bsz * m];
            let mut mmask = vec![false; bsz * m];
            let mut qtype = vec![0i64; bsz];
            for (r, &i) in batch.iter().enumerate() {
                let (_, _, qt, seq, _) = &items[i];
                ids[r * l..r * l + seq.ids.len()].copy_from_slice(&seq.ids);
                // only real tokens are attended (laya.common.collate_items); padding stays 0
                for a in attn[r * l..r * l + seq.ids.len()].iter_mut() {
                    *a = 1;
                }
                for (c, &mk) in seq.markers.iter().enumerate() {
                    mpos[r * m + c] = mk as i64;
                    mmask[r * m + c] = true;
                }
                qtype[r] = *qt;
            }
            let logits = run_batch(&mut self.session, &ids, &attn, &mpos, &mmask, &qtype, bsz, l, m)?;
            for (r, &i) in batch.iter().enumerate() {
                let (qid, options, qt, _, k) = &items[i];
                let z: Vec<f32> = logits[r * m..r * m + *k].to_vec();
                let t = self.temps.get(*qt, *k);
                let probs = softmax_with_temp(&z, t);
                results[i] = Some((qid.clone(), options.clone(), probs));
            }
        }
        for r in results.into_iter().flatten() {
            out.push(QuestionProbs { qid: r.0, options: r.1, probs: r.2 });
        }
        Ok(out)
    }
}

fn options_len(internal: &Value) -> usize {
    render_options(internal).len()
}

fn ort_err(e: impl std::fmt::Debug) -> anyhow::Error {
    anyhow::anyhow!("ort error: {e:?}")
}

fn run_batch(
    session: &mut Session,
    ids: &[i64],
    attn: &[i64],
    mpos: &[i64],
    mmask: &[bool],
    qtype: &[i64],
    bsz: usize,
    l: usize,
    m: usize,
) -> Result<Vec<f32>> {
    use ort::value::Tensor;
    let logits = session.run(ort::inputs![
        "input_ids" => Tensor::from_array((vec![bsz as i64, l as i64], ids.to_vec()))?,
        "attention_mask" => Tensor::from_array((vec![bsz as i64, l as i64], attn.to_vec()))?,
        "marker_pos" => Tensor::from_array((vec![bsz as i64, m as i64], mpos.to_vec()))?,
        "marker_mask" => Tensor::from_array((vec![bsz as i64, m as i64], mmask.to_vec()))?,
        "qtype" => Tensor::from_array((vec![bsz as i64], qtype.to_vec()))?,
    ])?;
    let (_, logits) = logits["logits"].try_extract_tensor::<f32>()?;
    Ok(logits.to_vec())
}
