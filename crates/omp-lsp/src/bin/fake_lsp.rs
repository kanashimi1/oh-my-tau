//! Fake language server for `omp-lsp` integration tests.
//!
//! Speaks JSON-RPC over stdio with Content-Length framing and answers a fixed
//! script:
//! - `initialize` → capabilities: definition, hover, full text-document sync
//! - `textDocument/didOpen` → publishes exactly one diagnostic, echoing the
//!   document version from the notification
//! - `textDocument/definition` → a fixed location at 0:0-0:5 of the same file
//! - `textDocument/hover` → fixed markdown
//! - `shutdown` → null; then EOF ends the loop (clean termination)
//! - everything else → standard method-not-found error

use std::io::{BufRead, Write};

fn read_message(reader: &mut impl BufRead) -> Option<serde_json::Value> {
    let mut content_length: Option<usize> = None;
    let mut line = String::new();
    loop {
        line.clear();
        let n = reader.read_line(&mut line).ok()?;
        if n == 0 {
            return None;
        }
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed.is_empty() {
            break;
        }
        if let Some((name, value)) = trimmed.split_once(':') {
            if name.trim().eq_ignore_ascii_case("content-length") {
                content_length = value.trim().parse().ok();
            }
        }
    }
    let len = content_length?;
    let mut body = vec![0u8; len];
    reader.read_exact(&mut body).ok()?;
    serde_json::from_slice(&body).ok()
}

fn write_message(value: &serde_json::Value) {
    let body = serde_json::to_string(value).expect("serialize response");
    let mut out = std::io::stdout().lock();
    write!(out, "Content-Length: {}\r\n\r\n{body}", body.len()).expect("write response");
    out.flush().expect("flush response");
}

fn main() {
    let stdin = std::io::stdin();
    let mut reader = stdin.lock();
    while let Some(msg) = read_message(&mut reader) {
        let method = msg.get("method").and_then(serde_json::Value::as_str);
        let id = msg.get("id").cloned();
        match (method, id) {
            (Some("initialize"), Some(id)) => {
                write_message(&serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": {
                        "capabilities": {
                            "definitionProvider": true,
                            "hoverProvider": true,
                            "textDocumentSync": {
                                "openClose": true,
                                "change": 1
                            }
                        }
                    }
                }));
            }
            (Some("textDocument/didOpen"), None) => {
                let uri = msg
                    .pointer("/params/textDocument/uri")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("file:///unknown")
                    .to_string();
                let version = msg
                    .pointer("/params/textDocument/version")
                    .cloned()
                    .unwrap_or(serde_json::json!(1));
                write_message(&serde_json::json!({
                    "jsonrpc": "2.0",
                    "method": "textDocument/publishDiagnostics",
                    "params": {
                        "uri": uri,
                        "version": version,
                        "diagnostics": [{
                            "range": {
                                "start": { "line": 0, "character": 0 },
                                "end": { "line": 0, "character": 1 }
                            },
                            "severity": 1,
                            "source": "fake-lsp",
                            "message": "fake error for tests"
                        }]
                    }
                }));
            }
            (Some("textDocument/definition"), Some(id)) => {
                let uri = msg
                    .pointer("/params/textDocument/uri")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("file:///unknown")
                    .to_string();
                write_message(&serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": [{
                        "uri": uri,
                        "range": {
                            "start": { "line": 0, "character": 0 },
                            "end": { "line": 0, "character": 5 }
                        }
                    }]
                }));
            }
            (Some("textDocument/hover"), Some(id)) => {
                write_message(&serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": { "contents": { "kind": "markdown", "value": "fake hover" } }
                }));
            }
            (Some("shutdown"), Some(id)) => {
                write_message(&serde_json::json!({ "jsonrpc": "2.0", "id": id, "result": null }));
            }
            (_, Some(id)) => {
                write_message(&serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "error": { "code": -32601, "message": "Method not found" }
                }));
            }
            _ => {}
        }
    }
}
