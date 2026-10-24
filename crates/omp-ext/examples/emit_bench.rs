//! Emit-latency micro-bench: load the user's extensions, then time N repeated
//! `tool_result` emits and report the median. Used to compare per-event cost
//! between the stdio host (default) and the shared daemon (`OMP_EXT_TEST_SHARED=1`).
//!
//! `cargo run -p omp-ext --example emit_bench -- [extensions_dir] [n]`

use std::time::Instant;

use anyhow::Result;
use omp_ext::{ExtHost, HostConfig};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let dir = args.next();
    let n: u32 = args.next().map_or(50, |s| s.parse().unwrap_or(50));

    let mut cfg = HostConfig::new(
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../ext-host/host.ts"),
    );
    cfg.handler_timeout_ms = Some(30_000);
    if std::env::var_os("OMP_EXT_TEST_SHARED").is_some_and(|v| v != "0") {
        let d = std::env::temp_dir().join(format!("omp-ext-shared-bench-{}", std::process::id()));
        std::fs::create_dir_all(&d)?;
        cfg.shared_socket = Some(d.join("ext.sock"));
    }
    let shared = cfg.shared_socket.is_some();
    let host = ExtHost::spawn(&cfg).await?;
    let t = Instant::now();
    let r = host.load(None, dir.as_deref(), "/tmp", None).await?;
    println!(
        "shared={shared} loaded {} extensions in {:?} ({} failed)",
        r.loaded.len(),
        t.elapsed(),
        r.errors.len()
    );

    // warmup
    host.emit(
        json!({"type":"tool_result","toolName":"bash","toolCallId":"w","input":{},"content":[{"type":"text","text":"hi\n"}],"isError":false}),
        Some(json!({"cwd": "/tmp", "hasUI": false, "branch": []})),
    )
    .await?;

    let mut samples = Vec::with_capacity(n as usize);
    for i in 0..n {
        let t = Instant::now();
        host.emit(
            json!({"type":"tool_result","toolName":"bash","toolCallId":i.to_string(),"input":{},"content":[{"type":"text","text":"hello world\n"}],"isError":false}),
            None,
        )
        .await?;
        samples.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    samples.sort_by(f64::total_cmp);
    let med = samples[n as usize / 2];
    let p90 = samples[(n as usize * 9) / 10];
    println!(
        "shared={shared} tool_result emit: n={n} median {med:.2} ms p90 {p90:.2} ms min {:.2} ms",
        samples[0]
    );
    host.shutdown().await?;
    Ok(())
}
