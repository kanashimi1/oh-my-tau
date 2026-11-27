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

use parking_lot::Mutex;
use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::sync::{Arc, LazyLock};
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
use super::turn_arms;
use super::util::{notice_with_artifact, save_artifact, sink_window, spill_window, SpillConfig};
use super::Env;
#[path = "eval_output.rs"]
mod eval_output;

#[cfg(test)]
#[allow(clippy::disallowed_methods)] // tests: throwaway runtimes drive async code
#[path = "eval_tests.rs"]
mod eval_tests;

/// Optional semantic observation packs + same-session output dedup
/// (`OMP_EVAL_OBSERVATION_PACK=1`); registered only here.
#[path = "eval_observation.rs"]
mod eval_observation;

/// omp `TOOL_TIMEOUTS.eval` (`tools/tool-timeouts.ts`): default 30 s (180 s
/// under the `OMP_TURN_HARNESS` arm), clamped to 1–3600 s when the agent
/// passes a value; `0` disables the deadline.
const MIN_TIMEOUT_SECS: f64 = 1.0;
const MAX_TIMEOUT_SECS: f64 = 3600.0;

/// Static model-facing description: the persistence/await/rollback contract
/// and the prelude signatures inline; namespace details stay behind
/// `xd://eval/*` docs (read on first use).
pub(crate) fn set_hypercode(active: bool) {
    eval_observation::set_hypercode(active);
}

pub const DESCRIPTION: &str = r"One cell per call; top-level state persists across calls and compaction; subagents get separate kernels.
Python (IPython): top-level `await` works, `asyncio.run` fails. JS (Bun): top-level `await`/`return` work.
On error, names (re)bound by the failed cell roll back (side effects and in-place mutations stay): re-run the whole fixed cell.
Prelude (py: sync, kwargs; js: async, one trailing options object): `display(v)` `print` `log(msg)` `phase(title)` `read(path, offset?, limit?)` `write(path, content)` `env(key?, value?)` `output(*ids, format?, query?, offset?, limit?)` `await tool.<name>(args)` `wait(handles, timeout?, raise_errors?=True)` (ordered results).
Read the doc before first use: `judge`/`judge_batch`/`completion` → `xd://eval/judge`; `%load`/`%pip`/`%bun add`/`budget`/`@tool` → `xd://eval/helpers`; `agent`/`workpool` (2+ independent items) → `xd://eval/agents`.
Never repeat successful setup; after a kernel-loss notice, reload it.
Multi-step mechanical work belongs in ONE cell: when a change needs read → decide → patch → validate, write one cell that does every step (read the file, check the condition, apply the patch, run the validator) and return ONLY the final diff + pass/fail status.
NEVER burn model turns on chains of single tool calls (read A → grep → edit B → bash validator) when a single eval cell can complete the transaction; spilled output goes to `artifact://<id>` automatically.";

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
    fn name(&self) -> &'static str {
        "eval"
    }
    fn description(&self) -> &str {
        turn_arms::description()
    }
    fn parameters(&self) -> &Value {
        static SCHEMA: LazyLock<Value> = LazyLock::new(schema);
        &SCHEMA
    }
    fn set_a2a_owner(&self, owner: &Arc<str>) {
        // eval cells (`agent()`, `workpool()`) spawn children through this
        // Env's owner.
        self.env.set_a2a_owner(owner.clone());
    }
    fn a2a_owner(&self) -> Option<Arc<str>> {
        self.env.a2a_owner()
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
    /// Last non-error output identity (opt-in `OMP_EVAL_OBSERVATION_PACK=1`
    /// same-session dedup; see [`eval_observation`]).
    last_observation: Option<eval_observation::ObservationKey>,
}

/// Retained kernels keyed by cwd (omp keeps them per session/interpreter;
/// omp-rs has one cwd per run).
static SLOTS: LazyLock<Mutex<HashMap<PathBuf, Arc<AsyncMutex<KernelState>>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

fn slot(cwd: &std::path::Path) -> Arc<AsyncMutex<KernelState>> {
    let mut slots = SLOTS.lock();
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
        None => Some(Duration::from_secs_f64(turn_arms::eval_default_timeout_secs())),
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
fn run_opts<'a>(
    bridge: &'a EvalBridge,
    timeout: Option<Duration>,
    cancel: CancelToken,
    output_sink: Option<&'a dyn Fn(&str)>,
) -> ExecOptions<'a> {
    ExecOptions {
        timeout,
        idle_timeout: true,
        bridge: Some(bridge),
        cancel: Some(cancel),
        output_sink,
    }
}

fn output_prefix(notice: Option<&str>, replayed: bool, cells: usize) -> String {
    let mut text = String::new();
    if let Some(notice) = notice {
        text.push_str(notice);
        text.push('\n');
    }
    if replayed {
        let _ = write!(
            text,
            "Replayed {cells} setup cell{}.",
            if cells == 1 { "" } else { "s" }
        );
    }
    text
}

fn push_timeout_annotation(
    output: &mut eval_output::StreamingOutput,
    result: &CellResult,
    is_js: bool,
    timeout: Option<Duration>,
) {
    if is_js {
        if result.timed_out {
            output.push(&format!(
                "\n[{}]",
                CellResult::js_timeout_annotation(timeout)
            ));
        }
    } else if let Some(annotation) = result.kernel_timeout_annotation() {
        output.push(&format!("\n[{annotation}]"));
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
#[allow(clippy::too_many_lines)]
// cell loop: language select, kernel start/reuse, run, journal, reply
// invariant: the execution loop above always pushes at least one cell result.
#[allow(clippy::expect_used)]
pub async fn eval(env: Arc<Env>, args: Value, ctx: CallCtx) -> Result<AgentToolResult> {
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
    // `#!bash` cell (harness arm): the script below the shebang is one `tool.bash` call.
    let shell_cell = turn_arms::shell_cell(&code);
    let code = shell_cell.clone().unwrap_or(code);
    let is_js = if shell_cell.is_some() {
        true
    } else {
        match language {
        "py" | "python" => false,
        "js" | "javascript" => true,
            other => bail!("language must be \"py\" or \"js\" (was {other:?})"),
        }
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

    // The bridge drives async tool work through the ambient runtime (via
    // `crate::sync_bridge`) from the kernel's blocking thread; fresh per
    // cell, like omp's per-request ctx.
    let make_bridge = || {
        EvalBridge::new(
            Arc::clone(&env),
            OpCtx {
                cwd: cwd.clone(),
                signal: ctx.signal.clone(),
                on_update: ctx.on_update.clone(),
                tool_call_id: ctx.tool_call_id.clone(),
            },
        )
    };

    let raw_output = crate::methodology::raw_tool_output_requested();
    let pack_output = eval_observation::pack_requested();
    let dedup_only = eval_observation::dedup_only_requested();
    // Pack classification needs the complete text; the bounded streaming
    // spool composes with the default and raw paths, not with semantic packs.
    let stream_output = !raw_output && !pack_output;
    let output_cfg = SpillConfig::load(&cwd);
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
        for (cell_index, cell_code) in codes.iter().enumerate() {
            let code_c = cell_code.clone();
            let k = Arc::clone(&kernel);
            let bridge = make_bridge();
            let cell_cancel = cancel.clone();
            let buffered = stream_output.then(|| {
                let mut output = eval_output::StreamingOutput::new(output_cfg);
                output.push(&output_prefix(
                    kernel_loss_notice.as_deref(),
                    replayed,
                    cell_index + 1,
                ));
                Arc::new(Mutex::new(Some(output)))
            });
            let task_buffered = buffered.clone();
            let (result, task_buffered) = tokio::task::spawn_blocking(move || {
                let sink = |text: &str| {
                    if let Some(buffer) = &task_buffered {
                        if let Some(output) = buffer.lock().as_mut() {
                            output.push(text);
                        }
                    }
                };
                let output_sink = task_buffered.as_ref().map(|_| &sink as &dyn Fn(&str));
                let result = k.execute_with(
                    &code_c,
                    run_opts(&bridge, timeout, cell_cancel, output_sink),
                );
                (result, task_buffered)
            })
            .await
            .map_err(|e| anyhow::anyhow!("eval kernel task failed: {e}"))?;
            let (mut result, buffered) = (result?, task_buffered);
            if let Some(buffer) = buffered {
                let mut output = buffer.lock().take().expect("cell output buffer");
                push_timeout_annotation(&mut output, &result, true, timeout);
                result.output = output.finish();
            }
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
        for (cell_index, cell_code) in codes.iter().enumerate() {
            let code_c = cell_code.clone();
            let k = Arc::clone(&kernel);
            let bridge = make_bridge();
            let cell_cancel = cancel.clone();
            let buffered = stream_output.then(|| {
                let mut output = eval_output::StreamingOutput::new(output_cfg);
                output.push(&output_prefix(
                    kernel_loss_notice.as_deref(),
                    replayed,
                    cell_index + 1,
                ));
                Arc::new(Mutex::new(Some(output)))
            });
            let task_buffered = buffered.clone();
            let (result, task_buffered) = tokio::task::spawn_blocking(move || {
                let sink = |text: &str| {
                    if let Some(buffer) = &task_buffered {
                        if let Some(output) = buffer.lock().as_mut() {
                            output.push(text);
                        }
                    }
                };
                let output_sink = task_buffered.as_ref().map(|_| &sink as &dyn Fn(&str));
                let result = k.execute_with(
                    &code_c,
                    run_opts(&bridge, timeout, cell_cancel, output_sink),
                );
                (result, task_buffered)
            })
            .await
            .map_err(|e| anyhow::anyhow!("eval kernel task failed: {e}"))?;
            let (mut result, buffered) = (result?, task_buffered);
            if let Some(buffer) = buffered {
                let mut output = buffer.lock().take().expect("cell output buffer");
                push_timeout_annotation(&mut output, &result, false, timeout);
                result.output = output.finish();
            }
            eval_journal::record_cell(&cwd, lang, cell_code, &result);
            let failed = result.status != CellStatus::Ok;
            results.push(result);
            if failed {
                break;
            }
        }
    }

    let result = results.last().expect("at least one cell result");
    if result.status == CellStatus::KernelKilled {
        // Kernel state is gone; drop it so the next call starts fresh.
        if is_js {
            state.js = None;
        } else {
            state.py = None;
        }
    }
    let text = if stream_output {
        result.output.clone()
    } else {
        let mut text = output_prefix(kernel_loss_notice.as_deref(), replayed, results.len());
        text.push_str(&result.output);
        if is_js {
            if result.timed_out {
                let _ = write!(text, "\n[{}]", CellResult::js_timeout_annotation(timeout));
            }
        } else if let Some(annotation) = result.kernel_timeout_annotation() {
            let _ = write!(text, "\n[{annotation}]");
        }
        if pack_output {
            let (packed, observed) = eval_observation::render(&eval_observation::PackInput {
                text: &text,
                cell_error: cell_is_error(result),
                raw: raw_output,
                enabled: true,
                dedup_only,
                prev: state.last_observation.clone(),
            });
            // An error/raw result breaks the consecutive-output chain.
            state.last_observation = observed;
            packed.unwrap_or_else(|| cap_output_with(&text, &output_cfg))
        } else {
            cap_output_with(&text, &output_cfg)
        }
    };
    // Images a cell displayed (PIL/_repr_png_ values, bridged `read` of an image file, image bundles)
    // reach the model as image content; the cell text alone cannot carry pixels.
    let images = display_images(&result.displays);
    let mut content = vec![pa_agent::types::ToolResultContent::text(text)];
    content.extend(
        images
            .into_iter()
            .map(pa_agent::types::ToolResultContent::Image),
    );
    Ok(AgentToolResult {
        content,
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
    SLOTS.lock().remove(cwd);
}

/// At most this many displayed images ride one eval result (each also capped by size).
const MAX_DISPLAY_IMAGES: usize = 8;
const MAX_DISPLAY_IMAGE_B64: usize = 6 * 1024 * 1024;
const IMAGE_MIMES: &[&str] = &["image/png", "image/jpeg", "image/webp", "image/gif"];

fn image_of(v: &Value) -> Option<pa_agent::types::ImageContent> {
    let data = v.get("data")?.as_str()?;
    let mime = v.get("mimeType").or_else(|| v.get("mime_type"))?.as_str()?;
    (IMAGE_MIMES.contains(&mime) && !data.is_empty() && data.len() <= MAX_DISPLAY_IMAGE_B64).then(
        || pa_agent::types::ImageContent {
            data: data.to_string(),
            mime_type: mime.to_string(),
        },
    )
}

/// Image content from a cell's display bundles: `application/x-omp-image` ({data, mimeType}),
/// raw `image/<fmt>` base64 strings (`_repr_png_` values), and JSON values shaped like a tool
/// result with `images: [{data, mimeType}]` or a single `{type: "image", data, mimeType}`.
fn display_images(displays: &[Value]) -> Vec<pa_agent::types::ImageContent> {
    let mut out = Vec::new();
    for bundle in displays {
        if out.len() >= MAX_DISPLAY_IMAGES {
            break;
        }
        if let Some(img) = bundle.get("application/x-omp-image").and_then(image_of) {
            out.push(img);
            continue;
        }
        if let Some((mime, data)) = IMAGE_MIMES
            .iter()
            .find_map(|m| bundle.get(*m).and_then(Value::as_str).map(|d| (*m, d)))
        {
            if !data.is_empty() && data.len() <= MAX_DISPLAY_IMAGE_B64 {
                out.push(pa_agent::types::ImageContent {
                    data: data.to_string(),
                    mime_type: mime.to_string(),
                });
            }
            continue;
        }
        if let Some(json) = bundle.get("application/json") {
            if let Some(img) =
                image_of(json).filter(|_| json.get("type").and_then(Value::as_str) == Some("image"))
            {
                out.push(img);
            } else if let Some(list) = json.get("images").and_then(Value::as_array) {
                out.extend(
                    list.iter()
                        .filter_map(image_of)
                        .take(MAX_DISPLAY_IMAGES - out.len()),
                );
            }
        }
    }
    out.truncate(MAX_DISPLAY_IMAGES);
    out
}

/// Display bundles for `details` with image payloads elided (the pixels travel as image content;
/// keeping base64 here would bloat every session file and text-plain previews).
fn displays_for_details(displays: &[Value]) -> Vec<Value> {
    fn elide(v: &Value) -> Value {
        match v {
            Value::Object(m) => Value::Object(
                m.iter()
                    .map(|(k, x)| {
                        let big = matches!(x, Value::String(s) if s.len() > 4096);
                        let keep = !(big
                            && (k == "data"
                                || k == "text/plain"
                                || IMAGE_MIMES.contains(&k.as_str())));
                        (
                            k.clone(),
                            if keep {
                                elide(x)
                            } else {
                                Value::String(format!(
                                    "<{} bytes elided>",
                                    x.as_str().map_or(0, str::len)
                                ))
                            },
                        )
                    })
                    .collect(),
            ),
            Value::Array(a) => Value::Array(a.iter().map(elide).collect()),
            _ => v.clone(),
        }
    }
    displays.iter().map(elide).collect()
}

/// `details` payload: execution counter plus timeout annotations, omp-style.
fn cell_details(result: &CellResult, replayed: bool) -> Value {
    json!({
        "executionCount": result.execution_count,
        "status": format!("{:?}", result.status),
        "timedOut": result.timed_out,
        "displays": displays_for_details(&result.displays),
        "isError": cell_is_error(result),
        "replayed": replayed.then_some(true),
    })
}

#[cfg(test)]
mod display_image_tests {
    use super::display_images;
    use serde_json::json;

    #[test]
    fn displayed_images_become_image_content() {
        let d = vec![
            json!({"text/plain": "x"}),
            json!({"application/x-omp-image": {"data": "QUJD", "mimeType": "image/png"}}),
            json!({"text/plain": "<PIL>", "image/jpeg": "REVG"}),
            json!({"application/json": {"text": "Read image file", "images": [{"data": "R0hJ", "mimeType": "image/webp"}]}}),
            json!({"application/json": {"type": "image", "data": "SktM", "mimeType": "image/gif"}}),
            json!({"application/json": {"images": [{"data": "QUJD", "mimeType": "text/html"}]}}),
        ];
        let got: Vec<(String, String)> = display_images(&d)
            .into_iter()
            .map(|i| (i.mime_type, i.data))
            .collect();
        assert_eq!(
            got,
            vec![
                ("image/png".into(), "QUJD".into()),
                ("image/jpeg".into(), "REVG".into()),
                ("image/webp".into(), "R0hJ".into()),
                ("image/gif".into(), "SktM".into()),
            ]
        );
    }

    #[test]
    fn image_count_is_capped() {
        let d: Vec<_> = (0..20).map(|_| json!({"image/png": "QUJD"})).collect();
        assert_eq!(display_images(&d).len(), super::MAX_DISPLAY_IMAGES);
    }
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
