//! The WASM plugin host runtime.
//!
//! Loads a plugin `.wasm` module (compiled once with `wasmi`), reads its
//! manifest via the `omt_manifest` ABI export, and dispatches tool/command
//! calls through the `omt_alloc`/`omt_call` ABI exports. Calls run in a
//! dedicated OS thread with a wall-clock timeout so a stuck plugin never
//! stalls omt's single-threaded runtime, and are metered by `wasmi` fuel
//! plus a 256 MiB memory limit.

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use serde_json::Value;
use wasmi::{
    Caller, Config, Engine, Linker, Module, Store, StoreLimits, StoreLimitsBuilder, TrapCode,
};

use crate::manifest::Manifest;

/// Default fuel budget per call (10^10).
pub const DEFAULT_FUEL: u64 = 10_000_000_000;
/// Default maximum linear-memory size per call.
pub const DEFAULT_MEMORY_LIMIT: usize = 256 * 1024 * 1024;
/// Default wall-clock timeout per call.
pub const DEFAULT_WALL_TIMEOUT: Duration = Duration::from_secs(30);

/// Which ABI entry point to invoke.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallKind {
    /// A tool call (`omt_call` with `kind: "tool"`).
    Tool,
    /// A slash-command call (`omt_call` with `kind: "command"`).
    Command,
}

impl CallKind {
    fn as_str(self) -> &'static str {
        match self {
            CallKind::Tool => "tool",
            CallKind::Command => "command",
        }
    }
}

/// The parsed result of an ABI call.
#[derive(Debug, Clone)]
pub struct CallOutput {
    /// `true` when the plugin reported success.
    pub ok: bool,
    /// The output text.
    pub text: String,
    /// The error message when `ok` is false.
    pub error: Option<String>,
}

/// Tunable limits for a single plugin call.
#[derive(Debug, Clone, Copy)]
pub struct HostLimits {
    /// Fuel budget for the call.
    pub fuel: u64,
    /// Maximum linear-memory size for the call.
    pub memory_limit: usize,
    /// Wall-clock timeout for the call.
    pub wall_timeout: Duration,
}

impl Default for HostLimits {
    fn default() -> Self {
        Self {
            fuel: DEFAULT_FUEL,
            memory_limit: DEFAULT_MEMORY_LIMIT,
            wall_timeout: DEFAULT_WALL_TIMEOUT,
        }
    }
}

/// Per-call store state: captured host `omt.log` lines and the resource
/// limiter (kept in the store so the limiter closure can hand out a mutable
/// reference to it).
struct PluginState {
    log_lines: Vec<String>,
    limits: StoreLimits,
}

impl Default for PluginState {
    fn default() -> Self {
        Self::new(StoreLimits::default())
    }
}

impl PluginState {
    fn new(limits: StoreLimits) -> Self {
        Self {
            log_lines: Vec::new(),
            limits,
        }
    }
}

/// A loaded WASM plugin: the compiled module plus its parsed manifest.
///
/// `Module` is compiled once at load; each call instantiates a fresh store so
/// calls cannot share mutable state and never need to be `Send` internally.
#[derive(Debug, Clone)]
pub struct WasmModule {
    path: PathBuf,
    module: Module,
    manifest: Manifest,
}

impl WasmModule {
    /// Compile and validate `path`, read the manifest, and return a ready plugin.
    ///
    /// # Errors
    ///
    /// Fails when the file is not a valid WASM module, the `omt_manifest`
    /// export is missing, or its JSON does not parse.
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
        let engine = engine();
        let module =
            Module::new(&engine, &bytes).map_err(|e| anyhow!("not a valid WASM module: {e}"))?;
        let manifest = read_manifest(&engine, &module)?;
        Ok(Self {
            path: path.to_path_buf(),
            module,
            manifest,
        })
    }

    /// Absolute path of the `.wasm` file this plugin was loaded from.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The parsed manifest.
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    /// The plugin name (manifest `name`).
    pub fn name(&self) -> &str {
        &self.manifest.name
    }

    /// Invoke `kind`/`name` with JSON `args` under [`HostLimits::default`].
    pub fn call(&self, kind: CallKind, name: &str, args: &Value) -> Result<CallOutput> {
        self.call_with_limits(kind, name, args, HostLimits::default())
    }

    /// [`WasmModule::call`] but also returns `omt.log` lines captured by the
    /// host during the call (used by tests/diagnostics).
    pub fn call_with_logs(
        &self,
        kind: CallKind,
        name: &str,
        args: &Value,
    ) -> Result<(CallOutput, Vec<String>)> {
        self.call_with_limits_and_logs(kind, name, args, HostLimits::default())
    }

    /// Invoke `kind`/`name` with JSON `args` under custom `limits`.
    pub fn call_with_limits(
        &self,
        kind: CallKind,
        name: &str,
        args: &Value,
        limits: HostLimits,
    ) -> Result<CallOutput> {
        self.call_with_limits_and_logs(kind, name, args, limits)
            .map(|(out, _)| out)
    }

    fn call_with_limits_and_logs(
        &self,
        kind: CallKind,
        name: &str,
        args: &Value,
        limits: HostLimits,
    ) -> Result<(CallOutput, Vec<String>)> {
        let request = serde_json::json!({
            "kind": kind.as_str(),
            "name": name,
            "args": args,
        });
        let request_bytes = request.to_string().into_bytes();
        let engine = self.module.engine().clone();
        let module = self.module.clone();
        // Run the call on a dedicated OS thread so a stuck plugin never blocks
        // omt's single-threaded runtime; fuel metering is the hard bound and
        // the wall clock here is the user-visible guarantee.
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let outcome = run_call(&engine, &module, &request_bytes, limits);
            let _ = tx.send(outcome);
        });
        match rx.recv_timeout(limits.wall_timeout) {
            Ok(outcome) => outcome,
            Err(_) => Err(anyhow!(
                "plugin call exceeded the {}s wall timeout",
                limits.wall_timeout.as_secs(),
            )),
        }
    }
}

/// Build the interpreter engine with fuel metering enabled.
fn engine() -> Engine {
    let mut config = Config::default();
    config.consume_fuel(true);
    Engine::new(&config)
}

/// Receive a manifest by instantiating `module` and calling `omt_manifest`.
fn read_manifest(engine: &Engine, module: &Module) -> Result<Manifest> {
    let mut store = Store::new(
        engine,
        PluginState::new(
            StoreLimitsBuilder::new()
                .memory_size(DEFAULT_MEMORY_LIMIT)
                .build(),
        ),
    );
    store.limiter(|state| &mut state.limits);
    store.set_fuel(DEFAULT_FUEL).map_err(map_wasm_error)?;
    let mut linker = Linker::new(engine);
    linker
        .func_wrap("omt", "log", log_func)
        .map_err(|e| anyhow!("install omt.log import: {e}"))?;
    let instance = linker
        .instantiate_and_start(&mut store, module)
        .map_err(map_wasm_error)?;
    let manifest_fn = instance
        .get_typed_func::<(), i64>(&store, "omt_manifest")
        .map_err(|e| anyhow!("plugin is missing a required ABI export `omt_manifest`: {e}"))?;
    let raw = manifest_fn.call(&mut store, ()).map_err(map_wasm_error)?;
    let memory = instance
        .get_memory(&store, "memory")
        .ok_or_else(|| anyhow!("plugin does not export `memory`"))?;
    let (ptr, len) = decode_ptr_len(raw);
    let data = memory.data(&store);
    let slice =
        checked_slice(data, ptr, len).ok_or_else(|| anyhow!("manifest buffer out of bounds"))?;
    let manifest: Manifest =
        serde_json::from_slice(slice).with_context(|| "plugin manifest is not valid JSON")?;
    Ok(manifest)
}

/// Execute one ABI call against a freshly instantiated store.
fn run_call(
    engine: &Engine,
    module: &Module,
    request: &[u8],
    limits: HostLimits,
) -> Result<(CallOutput, Vec<String>)> {
    let mut store = Store::new(
        engine,
        PluginState::new(
            StoreLimitsBuilder::new()
                .memory_size(limits.memory_limit)
                .trap_on_grow_failure(true)
                .build(),
        ),
    );
    store.set_fuel(limits.fuel).map_err(map_wasm_error)?;
    store.limiter(|state| &mut state.limits);

    let mut linker = Linker::new(engine);
    linker
        .func_wrap("omt", "log", log_func)
        .map_err(|e| anyhow!("install omt.log import: {e}"))?;
    let instance = linker
        .instantiate_and_start(&mut store, module)
        .map_err(map_wasm_error)?;
    let memory = instance
        .get_memory(&store, "memory")
        .ok_or_else(|| anyhow!("plugin does not export `memory`"))?;

    // 1. allocate a buffer and copy the request JSON into it.
    let alloc = instance
        .get_typed_func::<i32, i32>(&store, "omt_alloc")
        .map_err(|e| anyhow!("plugin is missing a required ABI export `omt_alloc`: {e}"))?;
    let request_len = i32::try_from(request.len())
        .map_err(|_| anyhow!("request is too large ({} bytes)", request.len()))?;
    let ptr = alloc
        .call(&mut store, request_len)
        .map_err(map_wasm_error)?;
    {
        let buf = memory.data_mut(&mut store);
        let start = usize::try_from(ptr).map_err(|_| anyhow!("negative allocation pointer"))?;
        let end = start
            .checked_add(request.len())
            .filter(|e| *e <= buf.len())
            .ok_or_else(|| anyhow!("allocated buffer out of bounds"))?;
        buf[start..end].copy_from_slice(request);
    }

    // 2. invoke the plugin.
    let call_fn = instance
        .get_typed_func::<(i32, i32), i64>(&store, "omt_call")
        .map_err(|e| anyhow!("plugin is missing a required ABI export `omt_call`: {e}"))?;
    let ret = call_fn
        .call(&mut store, (ptr, request_len))
        .map_err(map_wasm_error)?;

    // 3. decode and parse the response buffer.
    let (resp_ptr, resp_len) = decode_ptr_len(ret);
    let data = memory.data(&store);
    let slice = checked_slice(data, resp_ptr, resp_len)
        .ok_or_else(|| anyhow!("response buffer out of bounds"))?;
    let value: Value =
        serde_json::from_slice(slice).with_context(|| "plugin returned invalid JSON response")?;

    let state = store.into_data();
    Ok((parse_response(value), state.log_lines))
}

/// Decode an `i64` ABI result into `(ptr, len)` — the low 32 bits hold the
/// length, the high 32 bits the pointer.
fn decode_ptr_len(raw: i64) -> (usize, usize) {
    let raw = raw as u64;
    let ptr = (raw >> 32) as usize;
    let len = (raw & 0xffff_ffff) as usize;
    (ptr, len)
}

/// Return `data[ptr..ptr + len]` if it is in bounds.
fn checked_slice(data: &[u8], ptr: usize, len: usize) -> Option<&[u8]> {
    let end = ptr.checked_add(len)?;
    data.get(ptr..end)
}

fn parse_response(value: Value) -> CallOutput {
    let ok = value.get("ok").and_then(Value::as_bool).unwrap_or(false);
    let text = value
        .get("text")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let error = value
        .get("error")
        .and_then(Value::as_str)
        .map(str::to_string);
    CallOutput { ok, text, error }
}

/// Host `omt.log(ptr, len)` import: append the UTF-8 slice to state logs.
fn log_func(mut caller: Caller<'_, PluginState>, ptr: i32, len: i32) -> Result<(), wasmi::Error> {
    let memory = caller
        .get_export("memory")
        .and_then(|e| e.into_memory())
        .ok_or_else(|| wasmi::Error::new("plugin log import requires an exported `memory`"))?;
    let data = memory.data(&caller);
    let (start, end) = if ptr < 0 || len < 0 {
        return Err(wasmi::Error::new("negative log range"));
    } else {
        let start = ptr as usize;
        let end = start
            .checked_add(len as usize)
            .filter(|e| *e <= data.len())
            .ok_or_else(|| wasmi::Error::new("log buffer out of bounds"))?;
        (start, end)
    };
    let line = String::from_utf8_lossy(&data[start..end]).into_owned();
    caller.data_mut().log_lines.push(line);
    Ok(())
}

/// Turn a `wasmi` error into a clean user/command-visible message.
fn map_wasm_error(error: wasmi::Error) -> anyhow::Error {
    if error.as_trap_code() == Some(TrapCode::OutOfFuel) {
        anyhow!("plugin fuel exhausted")
    } else if let Some(code) = error.as_trap_code() {
        anyhow!("plugin trapped: {code}")
    } else if error.to_string().contains("exported function not found") {
        anyhow!("plugin is missing a required ABI export: {error}")
    } else {
        anyhow!("wasm error: {error}")
    }
}
