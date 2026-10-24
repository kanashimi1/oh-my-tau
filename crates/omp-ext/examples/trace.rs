//! Load every user extension into the host, emit a scripted omp event sequence and compare the
//! per-handler trace with a TS-omp reference (`hook_trace.jsonl` rows of one `omp -p` run).
//!
//! `cargo run -p omp-ext --example trace -- [trace_out.jsonl] [ts_reference.jsonl] [extensions_dir]`

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::Result;
use omp_ext::{ExtHost, HostConfig};
use serde_json::{json, Value};

type Row = (String, String, String);

fn rows_of_ts(path: &str) -> Result<Vec<Row>> {
    let mut rows = Vec::new();
    for line in std::fs::read_to_string(path)?.lines() {
        let v: Value = serde_json::from_str(line)?;
        let s = |k: &str| v[k].as_str().unwrap_or("").to_owned();
        rows.push((s("ext"), s("event"), s("tool")));
    }
    Ok(rows)
}

fn rows_of_rust(path: &str) -> Result<Vec<Row>> {
    let mut rows = Vec::new();
    for line in std::fs::read_to_string(path)?.lines() {
        let v: Value = serde_json::from_str(line)?;
        for h in v["handlers"].as_array().into_iter().flatten() {
            let s = |k: &str| h[k].as_str().unwrap_or("").to_owned();
            rows.push((s("ext"), s("event"), s("tool")));
        }
    }
    Ok(rows)
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let trace = args.next().unwrap_or_else(|| "hook_trace.jsonl".into());
    let reference = args.next();
    let dir = args.next();

    let _ = std::fs::remove_file(&trace);
    let mut cfg =
        HostConfig::new(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../ext-host/host.ts"));
    cfg.trace = Some(PathBuf::from(&trace));
    cfg.handler_timeout_ms = Some(30_000);
    if std::env::var_os("OMP_EXT_TEST_SHARED").is_some_and(|v| v != "0") {
        let dir = std::env::temp_dir().join(format!("omp-ext-shared-trace-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        cfg.shared_socket = Some(dir.join("ext.sock"));
    }
    let host = ExtHost::spawn(&cfg).await?;
    let mut incoming = host.take_incoming().expect("incoming");
    let notes = tokio::spawn(async move {
        let mut counts: HashMap<String, u32> = HashMap::new();
        while let Some(m) = incoming.recv().await {
            if let omp_ext::HostMessage::Notification { method, .. } = m {
                *counts.entry(method).or_default() += 1;
            }
        }
        counts
    });

    let t = std::time::Instant::now();
    let r = host.load(None, dir.as_deref(), "/tmp", None).await?;
    println!(
        "loaded {} extensions in {:?}, {} failed",
        r.loaded.len(),
        t.elapsed(),
        r.errors.len()
    );
    for e in &r.errors {
        println!("  FAILED {}: {}", e.path, e.error);
    }
    let handlers: u64 = r.loaded.iter().flat_map(|e| e.handlers.values()).sum();
    let tools: Vec<_> = r.loaded.iter().flat_map(|e| e.tools.clone()).collect();
    let cmds: usize = r.loaded.iter().map(|e| e.commands.len()).sum();
    println!("handlers={handlers} tools={tools:?} commands={cmds}");

    let ctx = json!({"cwd": "/tmp", "hasUI": false, "agentKind": "main", "sessionId": "trace-session", "branch": [], "idle": true});
    let user = json!({"role": "user", "content": [{"type": "text", "text": "run echo hi"}], "timestamp": 1});
    let steps = [
        json!({"type": "session_start"}),
        json!({"type": "before_agent_start", "prompt": "run echo hi", "systemPrompt": ["You are omp."]}),
        json!({"type": "context", "messages": [user]}),
        json!({"type": "tool_call", "toolName": "bash", "toolCallId": "t1", "input": {"command": "echo hi"}}),
        json!({"type": "tool_result", "toolName": "bash", "toolCallId": "t1", "input": {"command": "echo hi"}, "content": [{"type": "text", "text": "hi\n"}], "isError": false}),
        json!({"type": "context", "messages": [user]}),
        json!({"type": "session_stop", "messages": [user], "turn_id": 1, "session_id": "trace-session", "stop_hook_active": false}),
        json!({"type": "agent_end", "messages": [user]}),
        json!({"type": "session_shutdown"}),
    ];
    for (i, ev) in steps.into_iter().enumerate() {
        let name = ev["type"].as_str().unwrap_or("").to_owned();
        let t = std::time::Instant::now();
        let r = host
            .emit(ev, if i == 0 { Some(ctx.clone()) } else { None })
            .await?;
        let acted: Vec<_> = r
            .handlers
            .iter()
            .filter(|h| h.action != "none")
            .map(|h| format!("{}:{}", h.ext, h.action))
            .collect();
        println!(
            "{name:<20} {:>3} handlers {:>7.1?} non-none={acted:?} errors={}",
            r.handlers.len(),
            t.elapsed(),
            r.errors.len()
        );
    }
    host.shutdown().await?;
    // Close the incoming channel so the notes task can finish.
    drop(host);
    let counts = notes.await?;
    println!("host->core notifications: {counts:?}");

    if let Some(reference) = reference {
        compare(&reference, &trace)?;
    }
    Ok(())
}

fn compare(reference: &str, trace: &str) -> Result<()> {
    let ts = rows_of_ts(reference)?;
    let rs = rows_of_rust(trace)?;
    // The TS reference only has rows for handlers wrapped by lib/hook-trace.ts (traceExt/traceArgus),
    // and gates.ts reports its inner gates under their own names while the host sees one `gates`
    // handler. So compare, in order, the (ext, event) rows of non-tool_call events restricted to the
    // (ext, event) pairs the TS trace knows, and check the tool_call chain by presence.
    let traced: std::collections::HashSet<(&str, &str)> =
        ts.iter().map(|r| (r.0.as_str(), r.1.as_str())).collect();
    let pick = |rows: &[Row]| -> Vec<(String, String)> {
        rows.iter()
            .filter(|r| r.1 != "tool_call" && traced.contains(&(r.0.as_str(), r.1.as_str())))
            .map(|r| (r.0.clone(), r.1.clone()))
            .collect()
    };
    let (ts_cmp, rs_cmp) = (pick(&ts), pick(&rs));
    // The scripted run emits `context` twice around the tool; TS did too (tool turn + final turn).
    println!(
        "TS rows={} (non-tool_call, traced) Rust rows={}",
        ts_cmp.len(),
        rs_cmp.len()
    );
    let mut diffs = 0;
    for i in 0..ts_cmp.len().max(rs_cmp.len()) {
        if ts_cmp.get(i) != rs_cmp.get(i) {
            diffs += 1;
            println!(
                "  diff @{i}: TS={:?} Rust={:?}",
                ts_cmp.get(i),
                rs_cmp.get(i)
            );
        }
    }
    println!("order/handler diffs: {diffs}");
    let ts_tc = ts.iter().filter(|r| r.1 == "tool_call").count();
    let rs_tc: Vec<_> = rs
        .iter()
        .filter(|r| r.1 == "tool_call")
        .map(|r| r.0.as_str())
        .collect();
    println!("tool_call: TS {ts_tc} gate rows (inner gates of gates.ts + sub-chains), Rust handlers {rs_tc:?}");
    Ok(())
}
