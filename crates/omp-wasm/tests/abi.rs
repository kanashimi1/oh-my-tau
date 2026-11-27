//! ABI tests: a WAT module implementing the omp-wasm ABI exercises manifest
//! parsing, a tool-call round trip (including args pass-through via the host
//! `omt.log` import), fuel exhaustion, a trap, and a missing export.

use std::path::{Path, PathBuf};

use omp_wasm::{CallKind, HostLimits, WasmModule};
use serde_json::Value;

const MANIFEST_JSON: &str = r#"{"name":"demo","version":"1.0.0","tools":[{"name":"echo","description":"Echo text","parameters":{"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}}],"commands":[{"name":"hi","description":"Say hi"}]}"#;
const RESPONSE_JSON: &str = r#"{"ok":true,"text":"hello-from-wasm"}"#;

/// Encode bytes as a WAT string literal (`\XX` hex escapes).
fn wat_string(bytes: &[u8]) -> String {
    let mut out = String::from("\"");
    for b in bytes {
        out.push_str(&format!("\\{b:02X}"));
    }
    out.push('"');
    out
}

/// Build a WAT module with the core ABI skeleton and a custom `omt_call`
/// body. `manifest` is placed at address 0, `response` (optional) at 8192.
fn module_wat(manifest: &[u8], call_body: &str, response: Option<&[u8]>) -> String {
    let mut wat = format!(
        r#"(module
  (import "omt" "log" (func $log (param i32 i32)))
  (memory (export "memory") 2)
  (global $heap (mut i32) (i32.const 4096))
  (func (export "omt_alloc") (param $len i32) (result i32)
    (local $p i32)
    (local.set $p (global.get $heap))
    (global.set $heap (i32.add (i32.add (global.get $heap) (local.get $len)) (i32.const 7)))
    (local.get $p))
  (func (export "omt_manifest") (result i64)
    (i64.or (i64.shl (i64.extend_i32_u (i32.const 0)) (i64.const 32)) (i64.extend_i32_u (i32.const {}))))
  (func (export "omt_call") (param $ptr i32) (param $len i32) (result i64)
    {}
  )
  (data (i32.const 0) {})"#,
        manifest.len(),
        call_body,
        wat_string(manifest),
    );
    if let Some(response) = response {
        wat.push_str(&format!(
            "\n  (data (i32.const 8192) {})",
            wat_string(response)
        ));
    }
    wat.push_str("\n)\n");
    wat
}

fn compile(wat: &str) -> Vec<u8> {
    wat::parse_str(wat).expect("valid WAT")
}

fn write_plugin(dir: &Path, bytes: &[u8]) -> PathBuf {
    let path = dir.join("test.wasm");
    std::fs::write(&path, bytes).expect("write plugin");
    path
}

fn load_plugin(dir: &Path, call_body: &str, response: Option<&[u8]>) -> WasmModule {
    let manifest = MANIFEST_JSON.as_bytes();
    let wat = module_wat(manifest, call_body, response);
    let path = write_plugin(dir, &compile(&wat));
    WasmModule::load(&path).expect("load plugin")
}

#[test]
fn loads_and_parses_manifest() {
    let dir = tempfile::tempdir().expect("tempdir");
    let plugin = load_plugin(dir.path(), "(unreachable)", None);
    let manifest = plugin.manifest();
    assert_eq!(manifest.name, "demo");
    assert_eq!(manifest.version, "1.0.0");
    assert_eq!(manifest.tools.len(), 1);
    assert_eq!(manifest.tools[0].name, "echo");
    assert_eq!(manifest.tools[0].description, "Echo text");
    assert_eq!(
        manifest.tools[0].parameters["type"],
        Value::String("object".into())
    );
    assert_eq!(manifest.commands.len(), 1);
    assert_eq!(manifest.commands[0].name, "hi");
    assert_eq!(plugin.name(), "demo");
}

#[test]
fn tool_call_round_trip_echoes_and_passes_args() {
    let dir = tempfile::tempdir().expect("tempdir");
    // `omt_call` logs the incoming request and returns a fixed response, so
    // the test can assert the args we wrote actually reached the plugin.
    let body = "(call $log (local.get $ptr) (local.get $len))\n    (i64.or (i64.shl (i64.extend_i32_u (i32.const 8192)) (i64.const 32)) (i64.extend_i32_u (i32.const 36)))";
    let plugin = load_plugin(dir.path(), body, Some(RESPONSE_JSON.as_bytes()));

    let (out, logs) = plugin
        .call_with_logs(CallKind::Tool, "echo", &serde_json::json!({"text": "hi"}))
        .expect("call");
    assert!(out.ok);
    assert_eq!(out.text, "hello-from-wasm");
    assert!(out.error.is_none());
    assert_eq!(logs.len(), 1, "plugin logged the request once");
    assert!(logs[0].contains("\"kind\":\"tool\""), "log: {}", logs[0]);
    assert!(logs[0].contains("\"name\":\"echo\""), "log: {}", logs[0]);
    assert!(logs[0].contains("\"text\":\"hi\""), "log: {}", logs[0]);
}

#[test]
fn command_call_round_trip() {
    let dir = tempfile::tempdir().expect("tempdir");
    let body = "(i64.or (i64.shl (i64.extend_i32_u (i32.const 8192)) (i64.const 32)) (i64.extend_i32_u (i32.const 36)))";
    let plugin = load_plugin(dir.path(), body, Some(RESPONSE_JSON.as_bytes()));
    let out = plugin
        .call(CallKind::Command, "hi", &serde_json::json!("there"))
        .expect("call");
    assert!(out.ok);
    assert_eq!(out.text, "hello-from-wasm");
}

#[test]
fn fuel_exhaustion_is_a_clean_error() {
    let dir = tempfile::tempdir().expect("tempdir");
    // Infinite loop consuming fuel; `unreachable` makes the function
    // type-check (the loop never falls through to it).
    let body = "(loop $l (br $l))\n    (unreachable)\n    (i64.const 0)";
    let plugin = load_plugin(dir.path(), body, None);
    let err = plugin
        .call_with_limits(
            CallKind::Tool,
            "echo",
            &serde_json::json!({}),
            HostLimits {
                fuel: 1000,
                ..HostLimits::default()
            },
        )
        .expect_err("should exhaust fuel");
    assert!(err.to_string().contains("fuel exhausted"), "error: {err:#}");
}

#[test]
fn trap_is_a_clean_error() {
    let dir = tempfile::tempdir().expect("tempdir");
    let plugin = load_plugin(dir.path(), "(unreachable)", None);
    let err = plugin
        .call(CallKind::Tool, "echo", &serde_json::json!({}))
        .expect_err("should trap");
    assert!(err.to_string().contains("trapped"), "error: {err:#}");
}

#[test]
fn missing_export_is_a_clean_error() {
    let dir = tempfile::tempdir().expect("tempdir");
    // A module with the manifest + alloc exports but no `omt_call`.
    let manifest = MANIFEST_JSON.as_bytes();
    let wat = format!(
        r#"(module
  (import "omt" "log" (func $log (param i32 i32)))
  (memory (export "memory") 2)
  (global $heap (mut i32) (i32.const 4096))
  (func (export "omt_alloc") (param $len i32) (result i32)
    (local $p i32)
    (local.set $p (global.get $heap))
    (global.set $heap (i32.add (i32.add (global.get $heap) (local.get $len)) (i32.const 7)))
    (local.get $p))
  (func (export "omt_manifest") (result i64)
    (i64.or (i64.shl (i64.extend_i32_u (i32.const 0)) (i64.const 32)) (i64.extend_i32_u (i32.const {}))))
  (data (i32.const 0) {})
)
"#,
        manifest.len(),
        wat_string(manifest),
    );
    let path = write_plugin(dir.path(), &compile(&wat));
    let plugin = WasmModule::load(&path).expect("load plugin with manifest");

    let err = plugin
        .call(CallKind::Tool, "echo", &serde_json::json!({}))
        .expect_err("missing omt_call should fail");
    assert!(err.to_string().contains("ABI export"), "error: {err:#}");
}

#[test]
fn invalid_wasm_fails_load() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = write_plugin(dir.path(), b"this is not wasm");
    let err = WasmModule::load(&path).expect_err("invalid wasm should fail to load");
    assert!(
        err.to_string().contains("not a valid WASM module"),
        "error: {err:#}"
    );
}

#[test]
fn discovery_lists_failures_and_plugins() {
    let dir = tempfile::tempdir().expect("tempdir");
    let plugins_dir = dir.path().join(".omp/plugins");
    std::fs::create_dir_all(&plugins_dir).unwrap();
    // Valid plugin.
    let manifest = MANIFEST_JSON.as_bytes();
    let wat = module_wat(manifest, "(unreachable)", None);
    #[allow(clippy::needless_borrows_for_generic_args)] // `write_plugin` takes &[u8]
    std::fs::write(plugins_dir.join("good.wasm"), &compile(&wat)).unwrap();
    // Broken plugin.
    std::fs::write(plugins_dir.join("bad.wasm"), b"junk").unwrap();

    // Point HOME at the tempdir so a real `~/.omp/agent/plugins` directory
    // cannot leak into the count.
    let home = dir.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    let old_home = std::env::var_os("HOME");
    std::env::set_var("HOME", &home);
    let discovery = omp_wasm::discover(dir.path());
    match old_home {
        Some(home) => std::env::set_var("HOME", home),
        None => std::env::remove_var("HOME"),
    }
    assert_eq!(discovery.plugins.len(), 1);
    assert_eq!(discovery.plugins[0].module.name(), "demo");
    assert_eq!(discovery.failures.len(), 1);
    assert!(discovery.failures[0].0.ends_with("bad.wasm"));
}
