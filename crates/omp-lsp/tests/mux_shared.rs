//! Integration test for the broker-shared LSP mux (omp `src/lsp/mux`):
//! two sessions attach to one shared `fake_lsp` process behind the
//! `omp-lsp-mux` daemon, the full client path works over the socket, and the
//! daemon (plus its server child) idles out after the links drop.

use std::path::PathBuf;
use std::time::Duration;

use omp_lsp::client::{get_or_create_client, ClientStatus};
use omp_lsp::config::{get_config, get_server_for_file, invalidate_config};
use omp_lsp::diagnostics::{normalize_location_result, wait_for_diagnostics};
use omp_lsp::mux::{
    connect_shared_lsp_transport, protocol::lsp_mux_endpoint, set_shared_lsp_enabled,
};
use omp_lsp::utils::file_to_uri;
use serde_json::json;

fn unique_temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "omp-lsp-mux-shared-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos()
    ));
    std::fs::create_dir_all(dir.join("src")).expect("create temp project");
    dir
}

#[tokio::test]
async fn mux_shares_one_server_across_sessions() {
    let cwd = unique_temp_dir("shared");
    std::env::set_var("OMP_LSP_MUX_BIN", env!("CARGO_BIN_EXE_omp-lsp-mux"));
    // Short daemon idle + server linger so the test can observe teardown.
    std::env::set_var("OMP_LSP_MUX_IDLE_MS", "1000");
    set_shared_lsp_enabled(true);

    let server_bin = env!("CARGO_BIN_EXE_fake_lsp");
    let lsp_json = json!({
        "servers": {
            "fakeserver": {
                "command": server_bin,
                "fileTypes": [".fake"],
                "rootMarkers": ["."],
                "languageId": "fake"
            }
        }
    });
    std::fs::write(
        cwd.join("lsp.json"),
        serde_json::to_string_pretty(&lsp_json).unwrap(),
    )
    .expect("write lsp.json");
    let file_path = cwd.join("src").join("main.fake");
    std::fs::write(&file_path, "fake document body\n").expect("write source file");

    invalidate_config(&cwd);
    let config = get_config(&cwd);
    let (_name, server) = get_server_for_file(&config, &file_path)
        .unwrap_or_else(|| panic!("fakeserver should match {}", file_path.display()));
    let command = server
        .resolved_command
        .clone()
        .unwrap_or_else(|| server.command.clone());
    let args = server.args.clone().unwrap_or_default();

    // Session 1 spawns the shared server; session 2 must reuse its pid.
    let t1 = connect_shared_lsp_transport(&command, &args, &cwd, None)
        .await
        .expect("first shared transport");
    let pid1 = t1.pid.expect("spawned shared server reports a pid");
    let t2 = connect_shared_lsp_transport(&command, &args, &cwd, None)
        .await
        .expect("second shared transport");
    assert_eq!(
        t2.pid,
        Some(pid1),
        "second session must reuse the same server process"
    );
    drop(t1);
    drop(t2);

    // End-to-end: a full LSP client session over the mux socket.
    let client = get_or_create_client(&server, &cwd, Some(Duration::from_secs(10)), None)
        .await
        .expect("initialize shared fake_lsp");
    assert!(client.is_shared_mux(), "client must ride the mux transport");
    assert_eq!(client.status(), ClientStatus::Ready);
    assert_eq!(client.mux_server_pid(), Some(pid1));

    client.refresh_file(&file_path).await;
    let uri = file_to_uri(&file_path);
    let diags = wait_for_diagnostics(&client, &uri, 3000, None, None, Some(1))
        .await
        .expect("diagnostics over mux");
    assert_eq!(diags.len(), 1);
    assert_eq!(diags[0].message, "fake error for tests");

    let definition = client
        .send_request(
            "textDocument/definition",
            json!({
                "textDocument": { "uri": uri },
                "position": { "line": 0, "character": 0 }
            }),
            None,
            Some(Duration::from_secs(5)),
        )
        .await
        .expect("definition over mux");
    let locations = normalize_location_result(&definition);
    assert_eq!(locations.len(), 1);
    assert_eq!(locations[0].uri, uri);

    omp_lsp::client::shutdown_all().await;

    // The mux daemon (and its fake_lsp child) idle out and clean the socket.
    let endpoint = lsp_mux_endpoint(&omp_lsp::utils::normalize_path(&cwd))
        .expect("endpoint for project scope");
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    loop {
        if !endpoint.exists() {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "mux daemon did not idle out; socket still at {}",
            endpoint.display()
        );
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
}
