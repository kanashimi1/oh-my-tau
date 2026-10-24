//! `mux_session`: single-session driver used by the LSP-sharing measurement
//! (report `LspShare.md`). Opens one LSP client against the `fake_lsp`
//! fixture in `--root`, runs a didOpen → diagnostics → definition turn, holds
//! the session open until stdin closes, then shuts down.
//!
//! Usage: `mux_session --root DIR --command FAKE_LSP_BIN [--shared]`
//! Prints one JSON status line on success.

use std::path::PathBuf;

use omp_lsp::client::{get_or_create_client, ClientStatus};
use omp_lsp::mux::set_shared_lsp_enabled;
use omp_lsp::types::ServerConfig;

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let mut root = None;
    let mut command = None;
    let mut shared = false;
    let mut args_iter = std::env::args().skip(1);
    while let Some(arg) = args_iter.next() {
        match arg.as_str() {
            "--root" => root = args_iter.next().map(PathBuf::from),
            "--command" => command = args_iter.next().map(PathBuf::from),
            "--shared" => shared = true,
            other => {
                eprintln!("unknown argument {other}");
                std::process::exit(2);
            }
        }
    }
    let (Some(root), Some(command)) = (root, command) else {
        eprintln!("usage: mux_session --root DIR --command BIN [--shared]");
        std::process::exit(2);
    };
    set_shared_lsp_enabled(shared);

    let started = std::time::Instant::now();
    let file_path = root.join("src").join("main.fake");
    let config = ServerConfig {
        command: command.to_string_lossy().into_owned(),
        args: None,
        file_types: vec![".fake".to_string()],
        language_id: Some("fake".to_string()),
        root_markers: vec![".".to_string()],
        init_options: None,
        settings: None,
        disabled: None,
        warmup_timeout_ms: None,
        workspace_ready_timings: None,
        is_linter: None,
        capabilities: None,
        resolved_command: Some(command.to_string_lossy().into_owned()),
    };

    let client = match get_or_create_client(
        &config,
        &root,
        Some(std::time::Duration::from_secs(10)),
        None,
    )
    .await
    {
        Ok(client) => client,
        Err(e) => {
            eprintln!("initialize failed: {e}");
            std::process::exit(1);
        }
    };
    assert_eq!(client.status(), ClientStatus::Ready);
    let ready_ms = started.elapsed().as_millis();

    client.refresh_file(&file_path).await;
    let uri = omp_lsp::utils::file_to_uri(&file_path);
    let diag_secs: u64 = std::env::var("MUX_SESSION_DIAG_SECS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(30);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(diag_secs);
    let published = loop {
        if let Some(p) = client.published_diagnostics(&uri) {
            if !p.diagnostics.is_empty() {
                break p;
            }
        }
        assert!(std::time::Instant::now() < deadline, "no diagnostics");
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    };
    let _ = client
        .send_request(
            "textDocument/definition",
            serde_json::json!({
                "textDocument": { "uri": uri },
                "position": { "line": 0, "character": 0 }
            }),
            None,
            Some(std::time::Duration::from_secs(5)),
        )
        .await
        .expect("definition");

    // Hold the session open until the harness closes our stdin.
    let mut sink = Vec::new();
    let _ = tokio::io::stdin().read_to_end(&mut sink).await;

    let shared_mux = client.is_shared_mux();
    omp_lsp::client::shutdown_all().await;
    println!(
        "{}",
        serde_json::json!({
            "ok": true, "sharedMux": shared_mux, "readyMs": ready_ms,
            "diagnostics": published.diagnostics.len(),
        })
    );
}

use tokio::io::AsyncReadExt;
