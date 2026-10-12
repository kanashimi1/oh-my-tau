//! `omp-rs stats` — one-shot usage summary computed from omp session files,
//! the offline counterpart of TS `omp stats --summary` (the TS dashboard and
//! its sqlite rollups are not ported; the numbers below come straight from
//! the `model_usage` / assistant `custom_message` entries in each JSONL).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result};
use serde_json::{json, Value};

/// Aggregate token usage for one model selector.
#[derive(Default)]
struct ModelTotals {
    calls: u64,
    input: u64,
    output: u64,
    cache_read: u64,
    cache_write: u64,
    cost: f64,
}

/// Entry point: `omp-rs stats [--json] [--dir <sessions-root>]`.
///
/// # Errors
/// I/O failure walking the sessions directory.
pub fn run(args: &[String]) -> Result<i32> {
    let mut json = false;
    let mut dir: Option<PathBuf> = None;
    let mut it = args.iter().peekable();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--json" => json = true,
            "--dir" => {
                let value = it.next().context("--dir needs a value")?;
                dir = Some(PathBuf::from(value));
            }
            other if other.starts_with("--dir=") => dir = Some(PathBuf::from(&other[6..])),
            other if other.starts_with('-') => anyhow::bail!("unknown flag: {other}"),
            _ => anyhow::bail!("stats takes no positional arguments"),
        }
    }
    let root = dir.unwrap_or_else(default_sessions_dir);
    let summary = collect(&root);

    if json {
        println!("{}", serde_json::to_string_pretty(&summary)?);
    } else {
        print_text(&root, &summary);
    }
    Ok(0)
}

/// TS keeps sessions under `<agentDir>/sessions/<project-slug>/`; omp-rs reads
/// the same tree (its own live sessions live under `~/.omp-rs/sessions`, also
/// scanned when present via `--dir`).
fn default_sessions_dir() -> PathBuf {
    let dir = omp_providers::AgentDir::default();
    dir.0.join("sessions")
}

fn collect(root: &Path) -> Value {
    let mut files = 0u64;
    let mut entries_total = 0u64;
    let mut messages = 0u64;
    let mut by_model: BTreeMap<String, ModelTotals> = BTreeMap::new();
    let mut projects = 0u64;

    if root.is_dir() {
        projects = std::fs::read_dir(root)
            .map_or(0, |rd| rd.filter_map(std::result::Result::ok).filter(|e| e.file_type().is_ok_and(|t| t.is_dir())).count() as u64);
        let mut stack: Vec<PathBuf> = vec![root.to_path_buf()];
        while let Some(dir) = stack.pop() {
            let Ok(rd) = std::fs::read_dir(&dir) else {
                continue;
            };
            for entry in rd.filter_map(std::result::Result::ok) {
                let path = entry.path();
                let Ok(file_type) = entry.file_type() else { continue };
                if file_type.is_dir() {
                    stack.push(path);
                } else if path.extension().is_some_and(|e| e == "jsonl") {
                    files += 1;
                    scan_file(&path, &mut entries_total, &mut messages, &mut by_model);
                }
            }
        }
    }

    let totals = ModelTotals {
        calls: by_model.values().map(|m| m.calls).sum(),
        input: by_model.values().map(|m| m.input).sum(),
        output: by_model.values().map(|m| m.output).sum(),
        cache_read: by_model.values().map(|m| m.cache_read).sum(),
        cache_write: by_model.values().map(|m| m.cache_write).sum(),
        cost: by_model.values().map(|m| m.cost).sum(),
    };

    let models: Vec<Value> = by_model
        .iter()
        .map(|(model, t)| {
            json!({
                "model": model,
                "calls": t.calls,
                "input": t.input,
                "output": t.output,
                "cacheRead": t.cache_read,
                "cacheWrite": t.cache_write,
                "cost": round2(t.cost),
            })
        })
        .collect();

    json!({
        "sessionsDir": root,
        "projects": projects,
        "sessionFiles": files,
        "entries": entries_total,
        "messages": messages,
        "totals": {
            "calls": totals.calls,
            "input": totals.input,
            "output": totals.output,
            "cacheRead": totals.cache_read,
            "cacheWrite": totals.cache_write,
            "cost": round2(totals.cost),
        },
        "models": models,
    })
}

fn round2(n: f64) -> f64 {
    (n * 100.0).round() / 100.0
}

fn scan_file(path: &Path, entries: &mut u64, messages: &mut u64, by_model: &mut BTreeMap<String, ModelTotals>) {
    let Ok(text) = std::fs::read_to_string(path) else { return };
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let Ok(Value::Object(obj)) = serde_json::from_str::<Value>(line) else { continue };
        *entries += 1;
        // Work on the raw values (not `OmpEntry`) so foreign session files with
        // unmodeled shapes still contribute their usage.
        let kind = obj.get("type").and_then(Value::as_str);
        if kind == Some("model_usage") {
            if let Some(usage) = obj.get("usage") {
                let model = obj.get("model").and_then(Value::as_str).unwrap_or("unknown");
                add_usage(by_model, model, usage);
            }
            continue;
        }
        if kind == Some("custom_message") && obj.get("customType").and_then(Value::as_str) == Some("message") {
            *messages += 1;
            let content = obj.get("content").cloned().unwrap_or(Value::Null);
            if let Some(usage) = find_usage(&content) {
                let model = obj
                    .get("model")
                    .and_then(Value::as_str)
                    .or_else(|| content.get("model").and_then(Value::as_str))
                    .unwrap_or("unknown");
                add_usage(by_model, model, usage);
            }
        }
    }
}

/// Pull a `usage` object out of a custom message content value (any nesting).
fn find_usage(content: &Value) -> Option<&Value> {
    content.get("usage").or_else(|| content.pointer("/message/usage"))
}

fn add_usage(by_model: &mut BTreeMap<String, ModelTotals>, model: &str, usage: &Value) {
    let num = |key: &str, alt: &str| usage.get(key).or_else(|| usage.get(alt)).and_then(Value::as_u64).unwrap_or(0);
    let cost = usage
        .get("cost")
        .and_then(|c| c.get("total").and_then(Value::as_f64))
        .unwrap_or(0.0);
    let totals = by_model.entry(model.to_string()).or_default();
    totals.calls += 1;
    totals.input += num("input", "inputTokens");
    totals.output += num("output", "outputTokens");
    totals.cache_read += num("cacheRead", "cacheReadTokens");
    totals.cache_write += num("cacheWrite", "cacheWriteTokens");
    totals.cost += cost;
}

fn print_text(root: &Path, summary: &Value) {
    println!("Sessions:  {}", root.display());
    println!("Projects:  {}", summary["projects"]);
    println!("Files:     {}", summary["sessionFiles"]);
    println!("Entries:   {}", summary["entries"]);
    println!("Messages:  {}", summary["messages"]);
    let totals = &summary["totals"];
    println!(
        "Tokens:    {} in / {} out / {} cache-read / {} cache-write",
        totals["input"], totals["output"], totals["cacheRead"], totals["cacheWrite"]
    );
    println!("Cost:      ${}", totals["cost"]);
    let models = summary["models"].as_array().cloned().unwrap_or_default();
    if !models.is_empty() {
        println!("\nPer model:");
        for m in models {
            println!(
                "  {:<44} {:>6} calls  {:>12} in  {:>12} out  ${}",
                m["model"], m["calls"], m["input"], m["output"], m["cost"]
            );
        }
    }
}
