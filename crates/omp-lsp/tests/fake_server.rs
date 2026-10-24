//! Integration test driving `omp-lsp` against the `fake_lsp` fixture binary
//! (a minimal stdio JSON-RPC server built via `CARGO_BIN_EXE_fake_lsp`).
//!
//! Covers the full acceptance path: config discovery, initialize handshake,
//! didOpen → publishDiagnostics (exactly one diagnostic echoing the document
//! version), definition, hover, and clean termination via `shutdown`/`exit`.

use std::path::PathBuf;
use std::time::Duration;

use omp_lsp::client::{get_or_create_client, ClientStatus};
use omp_lsp::config::{get_config, get_server_for_file, invalidate_config};
use omp_lsp::diagnostics::{normalize_location_result, wait_for_diagnostics};
use omp_lsp::utils::{extract_hover_text, file_to_uri};
use serde_json::json;

fn unique_temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "omp-lsp-fake-server-{tag}-{}-{}",
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
async fn fake_server_full_session() {
    let cwd = unique_temp_dir("session");

    // A temp project wired to the fake server binary.
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

    // 1. Config discovery: the project config is found, the server matches
    //    the `.fake` file, and the binary resolves to the fixture.
    invalidate_config(&cwd);
    let config = get_config(&cwd);
    let (name, server) = get_server_for_file(&config, &file_path)
        .unwrap_or_else(|| panic!("fakeserver should match {}", file_path.display()));
    assert_eq!(name, "fakeserver");
    let resolved = server
        .resolved_command
        .as_deref()
        .unwrap_or_else(|| panic!("command should resolve: {}", server.command));
    assert!(
        resolved.ends_with("fake_lsp"),
        "unexpected resolved command: {resolved}"
    );

    // 2. Initialize handshake.
    let client = get_or_create_client(&server, &cwd, Some(Duration::from_secs(10)), None)
        .await
        .expect("initialize fake_lsp");
    assert_eq!(client.status(), ClientStatus::Ready);
    let caps = client.server_capabilities();
    assert_eq!(caps["definitionProvider"], json!(true));
    assert_eq!(caps["hoverProvider"], json!(true));
    assert_eq!(caps["textDocumentSync"]["change"], json!(1));

    // 3. didOpen → publishDiagnostics: exactly one diagnostic, version echoed.
    client.refresh_file(&file_path).await;
    let uri = file_to_uri(&file_path);
    assert_eq!(client.open_file_version(&uri), Some(1));
    let diags = wait_for_diagnostics(&client, &uri, 3000, None, None, Some(1))
        .await
        .expect("diagnostics");
    assert_eq!(diags.len(), 1);
    assert_eq!(diags[0].message, "fake error for tests");
    assert_eq!(diags[0].source.as_deref(), Some("fake-lsp"));
    let published = client
        .published_diagnostics(&uri)
        .expect("published diagnostics recorded");
    assert_eq!(published.diagnostics.len(), 1);
    assert_eq!(published.version, Some(1));

    // 4. Definition → the fixture's fixed location.
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
        .expect("definition request");
    let locations = normalize_location_result(&definition);
    assert_eq!(locations.len(), 1);
    assert_eq!(locations[0].uri, uri);
    assert_eq!(locations[0].range.start.line, 0);
    assert_eq!(locations[0].range.end.character, 5);

    // 5. Hover → the fixture's fixed markdown.
    let hover = client
        .send_request(
            "textDocument/hover",
            json!({
                "textDocument": { "uri": uri },
                "position": { "line": 0, "character": 0 }
            }),
            None,
            Some(Duration::from_secs(5)),
        )
        .await
        .expect("hover request");
    assert_eq!(extract_hover_text(&hover["contents"]), "fake hover");

    // 6. Clean termination: graceful stop handshake succeeds.
    assert!(client.shutdown_instance().await);

    std::fs::remove_dir_all(&cwd).ok();
}
