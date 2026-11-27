//! Hello-world WASM plugin for omt.
//!
//! Implements the omp-wasm ABI (JSON over linear memory):
//! - imports `omt.log(ptr, len)` from the host;
//! - exports `memory`, `omt_alloc`, `omt_manifest`, and `omt_call`.
//!
//! This is a `#![no_std]` crate so it links cleanly for
//! `wasm32-unknown-unknown` with no OS dependencies. It demonstrates a
//! single `greet` tool (`omt_call` with `kind:"tool", name:"greet"`) that
//! reads the JSON-encoded `name` argument from linear memory and answers with
//! a `{"ok":true,"text":"Hello, <name> from WASM!"}` response, and a single
//! `ping` command.

#![no_std]
// Single-threaded ABI surface: the host calls one entry point at a time, so
// direct access to the fixed request/response buffers is sound.
#![allow(static_mut_refs)]

// Host `omt.log(ptr, len)` import.
#[link(wasm_import_module = "omt")]
extern "C" {
    fn log(ptr: i32, len: i32);
}

/// The manifest returned by `omt_manifest`, as UTF-8 bytes in linear memory.
static MANIFEST: &[u8] = br#"{"name":"hello","version":"1.0.0","tools":[{"name":"greet","description":"Greet a person by name","parameters":{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}}],"commands":[{"name":"ping","description":"Reply pong"}]}"#;

/// Where the host writes the request JSON before calling `omt_call`
/// (single-call-at-a-time: a fixed slot is fine for a demo plugin).
static mut REQUEST_BUF: [u8; 8192] = [0; 8192];

/// Where the plugin writes the response JSON for `omt_call`.
static mut RESPONSE_BUF: [u8; 8192] = [0; 8192];

const RESPONSE_PREFIX: &[u8] = br#"{"ok":true,"text":"Hello, "#;
const RESPONSE_SUFFIX: &[u8] = br#" from WASM!"}"#;

/// The bump slot is fixed; the ABI only requires a writable buffer that can
/// hold `len` bytes, which `REQUEST_BUF` satisfies for reasonable requests.
#[no_mangle]
pub extern "C" fn omt_alloc(_len: i32) -> i32 {
    unsafe { REQUEST_BUF.as_mut_ptr() as i32 }
}

/// Return `(ptr << 32) | len` of the manifest JSON in linear memory.
#[no_mangle]
pub extern "C" fn omt_manifest() -> i64 {
    let ptr = MANIFEST.as_ptr() as i64;
    (ptr << 32) | (MANIFEST.len() as i64)
}

/// Handle a tool/command call. `ptr`/`len` point at the request JSON
/// `{"kind":...,"name":...,"args":...}` written by the host.
#[no_mangle]
pub extern "C" fn omt_call(ptr: i32, len: i32) -> i64 {
    // Echo the incoming request to the host log (visible in `call_with_logs`
    // and useful for debugging).
    unsafe { log(ptr, len) };

    let request = unsafe { core::slice::from_raw_parts(ptr as *const u8, len as usize) };

    // Dispatch on the top-level request `name` (`greet` tool vs anything
    // else, e.g. the `ping` command). Note the host serializes with
    // `preserve_order` in insertion order: kind, name, args.
    let mut name_buf = [0u8; 128];
    let is_greet = top_level_name(request, &mut name_buf) == b"greet";
    let text = if is_greet {
        // `greet` tool: read `args.name` — "Hello, <name> from WASM!"
        let mut who_buf = [0u8; 128];
        let who_len = args_string_param(request, b"name", &mut who_buf);
        let out = unsafe { &mut RESPONSE_BUF };
        let mut o = copy_bytes(out, 0, RESPONSE_PREFIX);
        o += copy_bytes(out, o, &who_buf[..who_len]);
        o += copy_bytes(out, o, RESPONSE_SUFFIX);
        o
    } else {
        // `ping` command (or unknown calls): reply pong.
        let out = unsafe { &mut RESPONSE_BUF };
        let pong: &[u8] = br#"{"ok":true,"text":"pong"}"#;
        copy_bytes(out, 0, pong)
    };

    unsafe { (RESPONSE_BUF.as_ptr() as i64) << 32 | (text as i64) }
}

/// Copy the top-level `"name":"<value>"` (before `"args":`) into `out`.
/// Returns the number of bytes written (0 when absent).
fn top_level_name<'a>(req: &[u8], out: &'a mut [u8]) -> &'a [u8] {
    // The top-level name appears before the args object; restrict the search
    // to the prefix ending at `"args":` so `"name"` keys inside args (or the
    // manifest-style nested objects) can never shadow it.
    let scope = find_subslice(req, b"\"args\":").map_or(req.len(), |p| p);
    let written = quoted_value_after_key(&req[..scope], b"name", out);
    &out[..written]
}

/// Copy `"name":"<value>"` from inside the `"args"` object into `out`.
/// Returns the number of bytes written (0 when absent).
fn args_string_param(req: &[u8], key: &[u8], out: &mut [u8]) -> usize {
    // Search only inside the args object: from `"args":{` to the request end
    // (the request has no sibling top-level keys after args in this demo).
    match find_subslice(req, b"\"args\":") {
        Some(start) => {
            let inner = &req[start + b"\"args\":".len()..];
            quoted_value_after_key(inner, key, out)
        }
        None => 0,
    }
}

/// Find `"<key>":"` in `req` and copy the quoted value's **contents** (after
/// the opening quote, up to the closing quote) into `out`. Returns the number
/// of bytes written (0 when the key or a non-empty quoted value is absent).
fn quoted_value_after_key(req: &[u8], key: &[u8], out: &mut [u8]) -> usize {
    let mut needle = [0u8; 64];
    let nl = 1 + key.len() + 3; // `"key":"`
    if nl > needle.len() {
        return 0;
    }
    needle[0] = b'"';
    needle[1..1 + key.len()].copy_from_slice(key);
    needle[1 + key.len()] = b'"';
    needle[1 + key.len() + 1] = b':';
    needle[1 + key.len() + 2] = b'"';
    if let Some(value_start) = find_subslice(req, &needle[..nl]) {
        let value_start = value_start + nl;
        let mut end = value_start;
        while end < req.len() && req[end] != b'"' {
            end += 1;
        }
        let len = end.saturating_sub(value_start).min(out.len());
        if len > 0 {
            out[..len].copy_from_slice(&req[value_start..value_start + len]);
            return len;
        }
    }
    0
}

/// Copy `src` into `dst[offset..]`, stopping at `dst`'s end.
fn copy_bytes(dst: &mut [u8], offset: usize, src: &[u8]) -> usize {
    if offset >= dst.len() {
        return 0;
    }
    let n = (dst.len() - offset).min(src.len());
    dst[offset..offset + n].copy_from_slice(&src[..n]);
    n
}

/// Naive subslice search.
fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || needle.len() > haystack.len() {
        return None;
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}
