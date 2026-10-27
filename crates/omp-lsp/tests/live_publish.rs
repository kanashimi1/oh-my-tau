#![allow(clippy::disallowed_methods)] // tests/entry points: throwaway runtimes drive async code
//! Live check: does a locally spawned rust-analyzer publish diagnostics to an
//! omp-lsp client after `refresh_file` (didOpen)? Run with --nocapture:
//! `cargo test -p omp-lsp --test live_publish -- --nocapture --ignored`

use std::time::{Duration, Instant};

#[tokio::test]
#[ignore = "spawns a real rust-analyzer; run explicitly"]
async fn rust_analyzer_publishes_after_refresh_file() {
    let dir = std::env::temp_dir().join("omp-lsp-live-pub");
    let src = dir.join("src");
    std::fs::create_dir_all(&src).unwrap();
    std::fs::write(
        dir.join("Cargo.toml"),
        "[package]\nname=\"livepub\"\nversion=\"0.1.0\"\nedition=\"2021\"\n",
    )
    .unwrap();
    std::fs::write(src.join("lib.rs"), "pub fn f() -> u32 { 1 }\n").unwrap();

    let config = omp_lsp::config::get_config(&dir);
    let (name, mut server) =
        omp_lsp::servers::get_lsp_server_for_file(&config, &src.join("lib.rs"))
            .expect("rust-analyzer configured");
    println!("server: {name}");
    if std::env::var_os("LIVE_PUB_PROXY").is_some() {
        server.resolved_command = Some("/tmp/prewarm-check/ra-proxy".to_string());
        server.args = Some(Vec::new());
        println!("proxy: on");
    }
    let client =
        omp_lsp::client::get_or_create_client(&server, &dir, Some(Duration::from_secs(10)), None)
            .await
            .expect("client init");

    let file = src.join("lib.rs");
    if std::env::var_os("LIVE_PUB_WAIT").is_some() {
        println!("waiting for project load…");
        tokio::time::sleep(Duration::from_secs(8)).await;
    }
    client.refresh_file(&file).await;
    let uri = omp_lsp::utils::file_to_uri(&file);
    println!("open: {}", client.is_open(&uri));

    // Wait for the first publish (clean file may publish an empty list).
    let deadline = Instant::now() + Duration::from_secs(15);
    while Instant::now() < deadline {
        if client.published_diagnostics(&uri).is_some() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    let clean = client.published_diagnostics(&uri);
    println!("clean publish: {clean:?}");

    // Break the file on disk, then refresh (didChange) — the publish must update.
    std::fs::write(&file, "pub fn f() -> u32 { \"x\" }\n").unwrap();
    client.refresh_file(&file).await;
    let deadline = Instant::now() + Duration::from_secs(15);
    let mut broken = client.published_diagnostics(&uri);
    while Instant::now() < deadline {
        broken = client.published_diagnostics(&uri);
        if broken.as_ref().is_some_and(|p| !p.diagnostics.is_empty()) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    let count = broken.as_ref().map_or(0, |p| p.diagnostics.len());
    println!("broken publish: {count} diagnostics");
    assert!(
        count > 0,
        "no diagnostics published after introducing a type error"
    );
}
