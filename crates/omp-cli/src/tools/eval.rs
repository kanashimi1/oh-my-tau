//! `eval`: persistent Python/JS kernels per cwd (omp `EvalTool`, `tools/eval.ts`).
//!
//! Python cells run through [`omp_eval`]'s [`PythonKernel`], JavaScript cells
//! through its [`JsKernel`] — long-lived subprocesses speaking omp's NDJSON
//! protocol; state persists across calls. `language` is a required field in
//! omp's schema, so a missing one is rejected like TS input validation.
//!
//! Cells run with the TS eval tool's host bridge ([`EvalBridge`]) so prelude
//! helpers (`tool.*`, `read`, `write`, `agent`, `judge`, …) reach the session,
//! with omp's idle-style timeout (the deadline pauses while the cell waits on
//! a host call, `eval/idle-timeout.ts`) and Esc/abort cooperative cancel.
//! Over-long output spills to the session artifacts dir and is referenced as
//! `artifact://<id>` (TS `OutputSink` recovery pointer). Kernel loss is
//! journaled per session (`eval_journal`) so `%replay` can restore setup.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, LazyLock, Mutex};
use std::time::Duration;

use anyhow::{bail, Result};
use omp_eval::js::JsKernel;
use omp_eval::{CancelToken, CellResult, CellStatus, ExecOptions, PythonKernel};
use pa_agent::abort::AbortSignal;
use pa_agent::types::{AgentTool, AgentToolResult, AgentToolUpdateCallback};
use pa_agent::BoxFut;
use serde_json::{json, Value};
use tokio::sync::Mutex as AsyncMutex;

use super::eval_bridge::{EvalBridge, OpCtx};
use super::eval_journal::{self, Lang};
use super::util::{notice_with_artifact, save_artifact, sink_window, spill_window, SpillConfig};
use super::Env;

#[cfg(test)]
#[path = "eval_tests.rs"]
mod eval_tests;

/// omp `TOOL_TIMEOUTS.eval` (`tools/tool-timeouts.ts`): default 30 s, clamped
/// to 1–3600 s when the agent passes a value; `0` disables the deadline.
const DEFAULT_TIMEOUT_SECS: f64 = 30.0;
const MIN_TIMEOUT_SECS: f64 = 1.0;
const MAX_TIMEOUT_SECS: f64 = 3600.0;

/// Static model-facing description: the persistence/await/rollback contract
/// and the prelude signatures inline; namespace details stay behind
/// `xd://eval/*` docs (read on first use).
pub const DESCRIPTION: &str = r"One cell per call; top-level state persists across calls and compaction; subagents get separate kernels.
Python (IPython): top-level `await` works, `asyncio.run` fails. JS (Bun): top-level `await`/`return` work.
On error, names (re)bound by the failed cell roll back (side effects and in-place mutations stay): re-run the whole fixed cell.
Prelude (py: sync, kwargs; js: async, one trailing options object): `display(v)` `print` `log(msg)` `phase(title)` `read(path, offset?, limit?)` `write(path, content)` `env(key?, value?)` `output(*ids, format?, query?, offset?, limit?)` `await tool.<name>(args)` `wait(handles, timeout?, raise_errors?=True)` (ordered results).
Read the doc before first use: `judge`/`judge_batch`/`completion` → `xd://eval/judge`; `%load`/`%pip`/`%bun add`/`budget`/`@tool` → `xd://eval/helpers`; `agent`/`workpool` (2+ independent items) → `xd://eval/agents`.
Never repeat successful setup; after a kernel-loss notice, reload it.";

/// Wire schema, byte-matching the TS `evalSchema` (`py` + `js` language
/// union, `language` + `code` required).
#[must_use]
pub fn schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "language": {"enum": ["py", "js"], "type": "string", "description": "\"py\": IPython; \"js\": Bun"},
            "code": {"type": "string", "description": "Code or standalone % command; top-level await works."},
            "title": {"type": "string", "description": "Short transcript label."},
            "timeout": {"type": "number", "description": "Cell deadline in seconds; 0 disables it."},
            "reset": {"type": "boolean", "description": "Wipe only this kernel."}
        },
        "required": ["language", "code"]
    })
}

/// The `eval` tool proper: receives the call's abort signal and update
/// callback (the old closure registration dropped both, leaving cells
/// unabortable and the host bridge blind).
pub struct EvalTool {
    env: Arc<Env>,
}

impl EvalTool {
    #[must_use]
    pub fn new(env: Arc<Env>) -> Self {
        Self { env }
    }
}

impl AgentTool for EvalTool {
    fn name(&self) -> &str {
        "eval"
    }
    fn description(&self) -> &str {
        DESCRIPTION
    }
    fn parameters(&self) -> &Value {
        static SCHEMA: LazyLock<Value> = LazyLock::new(schema);
        &SCHEMA
    }
    fn execute(
        self: Arc<Self>,
        tool_call_id: String,
        params: Value,
        signal: AbortSignal,
        on_update: AgentToolUpdateCallback,
    ) -> BoxFut<'static, Result<AgentToolResult>> {
        Box::pin(async move {
            eval(
                self.env.clone(),
                params,
                CallCtx {
                    tool_call_id,
                    signal,
                    on_update: Some(on_update),
                },
            )
            .await
        })
    }
}

pub(crate) struct CallCtx {
    pub tool_call_id: String,
    pub signal: AbortSignal,
    pub on_update: Option<AgentToolUpdateCallback>,
}

/// Per-cwd kernel state. The async mutex serializes all kernel ops (cell
/// execution, reset, replacement) so a `reset: true` or kernel-kill removal
/// can never drop a kernel while another cell for the same cwd is in flight
/// (omp marks the tool `concurrency = "exclusive"`).
#[derive(Default)]
struct KernelState {
    py: Option<Arc<PythonKernel>>,
    js: Option<Arc<JsKernel>>,
}

/// Retained kernels keyed by cwd (omp keeps them per session/interpreter;
/// omp-rs has one cwd per run).
static SLOTS: LazyLock<Mutex<HashMap<PathBuf, Arc<AsyncMutex<KernelState>>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

fn slot(cwd: &std::path::Path) -> Arc<AsyncMutex<KernelState>> {
    let mut slots = SLOTS.lock().expect("kernel registry");
    Arc::clone(
        slots
            .entry(cwd.to_path_buf())
            .or_insert_with(|| Arc::new(AsyncMutex::new(KernelState::default()))),
    )
}

/// omp `clampTimeout("eval", …)`: resolve the cell deadline — TS default 30 s
/// when the agent omits `timeout`, `0` disables, otherwise clamped 1–3600 s.
fn resolve_timeout(args: &Value) -> Option<Duration> {
    match args.get("timeout").and_then(Value::as_f64) {
        None => Some(Duration::from_secs_f64(DEFAULT_TIMEOUT_SECS)),
        Some(0.0) => None,
        Some(secs) => Some(Duration::from_secs_f64(
            secs.clamp(MIN_TIMEOUT_SECS, MAX_TIMEOUT_SECS),
        )),
    }
}

/// Runner environment mirroring the TS session (parent `irc`): the preludes
/// resolve `artifact://` and spill outputs through `PI_EVAL_LOCAL_ROOTS` /
/// `PI_ARTIFACTS_DIR` / `PI_SESSION_FILE`. Empty when no session file is
/// installed (tests, `--no-session`) — kernels then run without extra vars.
/// The cell-set `env` overlay ([`super::eval_bridge::env_overlay`]) rides
/// along; the session vars are applied last so a cell cannot repoint them.
fn kernel_env() -> Vec<(String, String)> {
    let mut vars = super::eval_bridge::env_overlay();
    let Some(session) = omp_urls::global_session_file() else {
        return vars;
    };
    vars.push((
        "PI_SESSION_FILE".to_owned(),
        session.to_string_lossy().into_owned(),
    ));
    if let Some(dir) = omp_urls::artifacts_directory_for(&session) {
        vars.push((
            "PI_ARTIFACTS_DIR".to_owned(),
            dir.to_string_lossy().into_owned(),
        ));
        vars.push((
            "PI_EVAL_LOCAL_ROOTS".to_owned(),
            json!({"artifact": dir.to_string_lossy()}).to_string(),
        ));
    }
    vars
}

/// Python kernels also carry TS's `PI_TOOL_BRIDGE_SESSION` (`python:<namespace>`,
/// `eval/py/index.ts` `PYTHON_SESSION_PREFIX`). User kernel customizations key off
/// it (e.g. a `usercustomize.py` that installs a REPL prelude only inside omp
/// python kernels), so without it they silently never load under omp-rs.
fn python_kernel_env(cwd: &std::path::Path) -> Vec<(String, String)> {
    let mut vars = kernel_env();
    let namespace = omp_urls::global_session_file().map_or_else(
        || format!("session:{}", cwd.display()),
        |f| f.to_string_lossy().into_owned(),
    );
    vars.push((
        "PI_TOOL_BRIDGE_SESSION".to_owned(),
        format!("python:{namespace}"),
    ));
    vars
}

/// Cell options: the bridge lives in the blocking task next to the cell it
/// serves; TS idle-timeout semantics (host calls pause the deadline).
fn run_opts(
    bridge: &EvalBridge,
    timeout: Option<Duration>,
    cancel: CancelToken,
) -> ExecOptions<'_> {
    ExecOptions {
        timeout,
        idle_timeout: true,
        bridge: Some(bridge),
        cancel: Some(cancel),
    }
}

/// Bound the rendered output the way omp's eval does. The `OutputSink`
/// stream window (middle elision past `DEFAULT_MAX_BYTES`, head per
/// `tools.artifactHeadBytes`) always applies; output that fits it but tops
/// `tools.artifactSpillThreshold` gets the session-level spill
/// (`spillLargeResultToArtifact`: `artifactHeadBytes` + `artifactTailBytes`
/// within `2 × artifactTailLines` lines). The full text goes to an artifact;
/// the notice links it only when the save succeeded.
fn cap_output_with(text: &str, cfg: &SpillConfig) -> String {
    if crate::methodology::raw_tool_output_requested() {
        return text.to_owned();
    }
    let elided = match sink_window(text, cfg) {
        Some(e) => e,
        // The session-level spill needs a session to hold the artifact.
        None if omp_urls::global_session_file().is_none() => return text.to_owned(),
        None => match spill_window(text, cfg) {
            Some(e) => e,
            None => return text.to_owned(),
        },
    };
    let url = save_artifact(text);
    format!(
        "{}\n\n[{}]",
        elided.content,
        notice_with_artifact(&elided.notice, url.as_deref())
    )
}

/// [`cap_output_with`] at the schema-default limits.
#[cfg(test)]
fn cap_output(text: &str) -> String {
    cap_output_with(text, &SpillConfig::default())
}

/// True per TS eval error semantics: raised cells, timeouts (interrupted or
/// kernel-killed) and aborts all render `.error()` (omp `isError: true`).
fn cell_is_error(result: &CellResult) -> bool {
    result.cancelled
        || matches!(
            result.status,
            CellStatus::Error | CellStatus::Interrupted | CellStatus::KernelKilled
        )
}

/// Execute one `eval` cell (the tool entry point).
pub async fn eval(env: Arc<Env>, args: Value, ctx: CallCtx) -> Result<AgentToolResult> {
    use std::fmt::Write as _;
    // omp's schema requires `language`; a missing one fails TS input
    // validation before execute, so reject it here too (no silent default).
    let language = args
        .get("language")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow::anyhow!("missing string argument `language`"))?;
    let code = args
        .get("code")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow::anyhow!("missing string argument `code`"))?
        .to_string();
    let is_js = match language {
        "py" | "python" => false,
        "js" | "javascript" => true,
        other => bail!("language must be \"py\" or \"js\" (was {other:?})"),
    };
    let lang = if is_js { Lang::Js } else { Lang::Py };

    let timeout = resolve_timeout(&args);
    let reset = args.get("reset").and_then(Value::as_bool).unwrap_or(false);

    let cwd = env.cwd.clone();
    // Exclusive per-cwd section: reset/kill cannot interleave with a cell.
    // The wait is raced against the call's abort so a queued cell can never
    // hang past Esc (nested `eval` from a cell is refused by the bridge).
    let slot = slot(&cwd);
    let mut state = tokio::select! {
        state = slot.lock() => state,
        () = ctx.signal.aborted() => bail!("eval aborted while waiting for the kernel"),
    };

    if reset {
        // omp `reset?` wipes only the requested language's kernel.
        if is_js {
            state.js = None;
        } else {
            state.py = None;
        }
    }

    // Cooperative cancel: Esc aborts the tool call → the running cell is
    // interrupted (SIGINT for Python, kill + fresh kernel for JS).
    let cancel = CancelToken::new();
    {
        let signal = ctx.signal.clone();
        let cancel = cancel.clone();
        tokio::spawn(async move {
            signal.aborted().await;
            cancel.cancel();
        });
    }

    // `%replay`: re-run the journaled setup cells in order (omp reloads
    // prelude setup after a kernel-loss notice instead of retyping it).
    let replayed = code.trim() == "%replay";
    let codes: Vec<String> = if replayed {
        eval_journal::replay_setup(&cwd, lang)
    } else {
        vec![code.clone()]
    };
    if replayed && codes.is_empty() {
        return Ok(AgentToolResult {
            content: vec![pa_agent::types::ToolResultContent::text(
                "No setup cells recorded for this kernel.",
            )],
            details: json!({"replayed": 0}),
            terminate: None,
            is_error: None,
        });
    }

    // The bridge drives async tool work through the ambient runtime from the
    // kernel's blocking thread; fresh per cell, like omp's per-request ctx.
    let rt = tokio::runtime::Handle::current();
    let make_bridge = || {
        EvalBridge::new(
            Arc::clone(&env),
            OpCtx {
                cwd: cwd.clone(),
                signal: ctx.signal.clone(),
                on_update: ctx.on_update.clone(),
                tool_call_id: ctx.tool_call_id.clone(),
            },
            rt.clone(),
        )
    };

    let mut kernel_loss_notice: Option<String> = None;
    let mut results: Vec<CellResult> = Vec::with_capacity(codes.len());
    if is_js {
        let kernel = if let Some(existing) = state.js.as_ref().filter(|k| k.is_alive()) {
            Arc::clone(existing)
        } else {
            // JsKernel has no env-var constructor (no in-process interrupt
            // path needs the session roots there); see port report.
            let fresh = Arc::new(JsKernel::start(omp_eval::js::default_runtime(), &cwd)?);
            state.js = Some(Arc::clone(&fresh));
            kernel_loss_notice = eval_journal::kernel_started(&cwd, lang);
            fresh
        };
        for cell_code in &codes {
            let code_c = cell_code.clone();
            let k = Arc::clone(&kernel);
            let bridge = make_bridge();
            let cell_cancel = cancel.clone();
            let result = tokio::task::spawn_blocking(move || {
                k.execute_with(&code_c, run_opts(&bridge, timeout, cell_cancel))
            })
            .await
            .map_err(|e| anyhow::anyhow!("eval kernel task failed: {e}"))??;
            eval_journal::record_cell(&cwd, lang, cell_code, &result);
            let failed = result.status != CellStatus::Ok;
            results.push(result);
            if failed {
                break;
            }
        }
    } else {
        let kernel = if let Some(existing) = state.py.as_ref().filter(|k| k.is_alive()) {
            Arc::clone(existing)
        } else {
            let fresh = Arc::new(PythonKernel::start_with_env(
                omp_eval::default_interpreter(),
                &cwd,
                &python_kernel_env(&cwd),
            )?);
            state.py = Some(Arc::clone(&fresh));
            kernel_loss_notice = eval_journal::kernel_started(&cwd, lang);
            fresh
        };
        for cell_code in &codes {
            let code_c = cell_code.clone();
            let k = Arc::clone(&kernel);
            let bridge = make_bridge();
            let cell_cancel = cancel.clone();
            let result = tokio::task::spawn_blocking(move || {
                k.execute_with(&code_c, run_opts(&bridge, timeout, cell_cancel))
            })
            .await
            .map_err(|e| anyhow::anyhow!("eval kernel task failed: {e}"))??;
            eval_journal::record_cell(&cwd, lang, cell_code, &result);
            let failed = result.status != CellStatus::Ok;
            results.push(result);
            if failed {
                break;
            }
        }
    };

    let result = results.last().expect("at least one cell result");
    let mut text = String::new();
    if let Some(notice) = kernel_loss_notice {
        // Prepend so the model reads the state-loss warning first (TS
        // `eval-prelude-notice`).
        text.push_str(&notice);
        text.push('\n');
    }
    if replayed {
        let _ = write!(
            text,
            "Replayed {} setup cell{}.\n",
            results.len(),
            if results.len() == 1 { "" } else { "s" }
        );
    }
    text.push_str(&result.output);
    if is_js {
        if result.timed_out {
            let _ = write!(text, "\n[{}]", CellResult::js_timeout_annotation(timeout));
        }
    } else if let Some(annotation) = result.kernel_timeout_annotation() {
        let _ = write!(text, "\n[{annotation}]");
    }
    if result.status == CellStatus::KernelKilled {
        // Kernel state is gone; drop it so the next call starts fresh.
        if is_js {
            state.js = None;
        } else {
            state.py = None;
        }
    }
    let text = cap_output_with(&text, &SpillConfig::load(&cwd));
    Ok(AgentToolResult {
        content: vec![pa_agent::types::ToolResultContent::text(text)],
        details: cell_details(result, replayed),
        terminate: None,
        is_error: cell_is_error(result).then_some(true),
    })
}

/// Shut down the retained kernel for `cwd` (test helper). Leaked runner
/// children hold inherited stdio fds open, which keeps `cargo test`'s output
/// pipe from closing; tests must call this before returning. Per-slot, not
/// global: parallel eval tests each own a distinct cwd, and clearing the
/// whole registry would drop a sibling test's live kernel mid-run (its next
/// cell then loses state or fails on broken kernel I/O).
#[cfg(test)]
pub(crate) fn shutdown_slot(cwd: &std::path::Path) {
    // Dropping the slot drops KernelState, whose kernel Arcs run Drop and
    // settle the runner children.
    SLOTS.lock().expect("kernel registry").remove(cwd);
}

/// `details` payload: execution counter plus timeout annotations, omp-style.
fn cell_details(result: &CellResult, replayed: bool) -> Value {
    json!({
        "executionCount": result.execution_count,
        "status": format!("{:?}", result.status),
        "timedOut": result.timed_out,
        "displays": result.displays,
        "isError": cell_is_error(result),
        "replayed": replayed.then_some(true),
    })
}

#[cfg(test)]
mod bridge_session_env_tests {
    /// User `usercustomize.py` preludes (`rlm()` / `P()` / `ask_session()`) load only
    /// when `PI_TOOL_BRIDGE_SESSION` starts with `python:` (TS kernel contract).
    #[test]
    fn python_kernels_carry_the_ts_bridge_session_prefix() {
        let vars = super::python_kernel_env(std::path::Path::new("/tmp/x"));
        let v = vars
            .iter()
            .find(|(k, _)| k == "PI_TOOL_BRIDGE_SESSION")
            .map(|(_, v)| v.clone())
            .expect("PI_TOOL_BRIDGE_SESSION set");
        assert!(v.starts_with("python:"), "{v}");
    }
}
