//! `eval`: persistent Python/JS kernels per cwd (omp `EvalTool`, `tools/eval.ts`).
//!
//! Python cells run through [`omp_eval`]'s [`PythonKernel`], JavaScript cells
//! through its [`JsKernel`] — long-lived subprocesses speaking omp's NDJSON
//! protocol; state persists across calls. `language` is a required field in
//! omp's schema, so a missing one is rejected like TS input validation.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, LazyLock, Mutex};
use std::time::Duration;

use anyhow::{bail, Result};
use omp_eval::js::JsKernel;
use omp_eval::{CellResult, CellStatus, PythonKernel};
use pa_agent::types::AgentToolResult;
use serde_json::{json, Value};
use tokio::sync::Mutex as AsyncMutex;

use super::Env;

/// Static model-facing description (omp `getEvalToolDescription` under the
/// protocol defaults: py+js advertised, prelude topics linked).
pub const DESCRIPTION: &str = r"One cell per call; top-level state persists, including across compaction. Subagents have separate kernels.
For 2+ independent items, use a named `workpool()`; results auto-deliver. If blocked, leave `eval` and call `wait`.
Python: top-level `await` works; `asyncio.run(…)` fails.
JS: Bun (`Bun.file`, `Bun.write`, `Bun.$`); top-level `await`/`return` work.
On error, retry only the failed step; earlier steps may have taken effect.

<prelude>
Python helpers: sync, kwargs; JS helpers: async, ONE trailing options object.
```
display(value)  print(value, ...)  log(message)  phase(title)
read(path, offset?, limit?)  write(path, content)  env(key?, value?)  output(*ids, format?, query?, offset?, limit?)
await tool.<name>(args) — session tool; `args` is its parameter object
wait(handles, timeout?=None, raise_errors?=True) — agent/completion barrier, ordered results; JS: wait(handles, { timeout, raiseErrors }); `raise_errors=False` retains failures.
```
</prelude>

<namespaces>
More globals; `read` the linked docs before first use:
- `judge`, `judge_batch`, `completion`: classification, bulk judgment, model calls → `xd://eval/judge`
- `%load`, `%pip`, `%bun add`, `budget`, `@tool`/`tool(fn)`: setup, installs, utilities → `xd://eval/helpers`
- `agent`, `workpool`: background subagents, DAG waves → `xd://eval/agents`
</namespaces>

<critical>
NEVER repeat successful setup. Kernel-loss notice means reload setup.
</critical>";

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

/// Execute one `eval` cell.
pub async fn eval(env: Arc<Env>, args: Value) -> Result<AgentToolResult> {
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

    let timeout = match args.get("timeout").and_then(Value::as_f64) {
        None | Some(0.0) => None,
        Some(secs) => Some(Duration::from_secs_f64(secs.max(1.0))),
    };
    let reset = args.get("reset").and_then(Value::as_bool).unwrap_or(false);

    let cwd = env.cwd.clone();
    // Exclusive per-cwd section: reset/kill cannot interleave with a cell.
    let slot = slot(&cwd);
    let mut state = slot.lock().await;

    if reset {
        // Drop happens here, inside the exclusive section.
        state.py = None;
        state.js = None;
    }

    let (result, is_js) = if is_js {
        let kernel = match state.js.as_ref().filter(|k| k.is_alive()) {
            Some(existing) => Arc::clone(existing),
            None => {
                let fresh = Arc::new(JsKernel::start(omp_eval::js::default_runtime(), &cwd)?);
                state.js = Some(Arc::clone(&fresh));
                fresh
            }
        };
        let result =
            tokio::task::spawn_blocking(move || kernel.execute(&code, timeout))
                .await
                .map_err(|e| anyhow::anyhow!("eval kernel task failed: {e}"))??;
        (result, true)
    } else {
        let kernel = match state.py.as_ref().filter(|k| k.is_alive()) {
            Some(existing) => Arc::clone(existing),
            None => {
                let fresh = Arc::new(PythonKernel::start(omp_eval::default_interpreter(), &cwd)?);
                state.py = Some(Arc::clone(&fresh));
                fresh
            }
        };
        let result =
            tokio::task::spawn_blocking(move || kernel.execute(&code, timeout))
                .await
                .map_err(|e| anyhow::anyhow!("eval kernel task failed: {e}"))??;
        (result, false)
    };

    let mut text = result.output.clone();
    use std::fmt::Write as _;
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
    Ok(AgentToolResult {
        content: vec![pa_agent::types::ToolResultContent::text(text)],
        details: cell_details(&result),
        terminate: None,
    })
}

/// Shut down every retained kernel (test helper). Leaked runner children hold
/// inherited stdio fds open, which keeps `cargo test`'s output pipe from
/// closing; tests must call this before returning.
#[cfg(test)]
fn shutdown_all() {
    // Dropping the slot drops KernelState, whose kernel Arcs run Drop and
    // settle the runner children.
    SLOTS.lock().expect("kernel registry").clear();
}

/// `details` payload: execution counter plus timeout annotations, omp-style.
fn cell_details(result: &CellResult) -> Value {
    json!({
        "executionCount": result.execution_count,
        "status": format!("{:?}", result.status),
        "timedOut": result.timed_out,
        "displays": result.displays,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unique cwd per test: parallel tests must not share a kernel slot (a
    /// JS timeout kill is global to the slot's kernel).
    fn test_env(tag: &str) -> Arc<Env> {
        let dir = std::env::temp_dir().join(format!("omp-rs-eval-test-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        Arc::new(Env::new(&dir, false))
    }

    #[test]
    fn schema_matches_ts_reference() {
        let s = schema();
        let props = s["properties"].as_object().unwrap();
        assert_eq!(props["language"]["enum"], json!(["py", "js"]));
        assert_eq!(s["required"], json!(["language", "code"]));
        assert_eq!(props.len(), 5);
    }

    #[tokio::test]
    async fn missing_language_is_rejected_like_ts_validation() {
        let env = test_env("lang");
        let err = eval(env.clone(), json!({"code": "1"})).await.unwrap_err();
        assert!(err.to_string().contains("language"), "got: {err}");
    }

    #[tokio::test]
    async fn python_state_persists() {
        let env = test_env("py");
        eval(env.clone(), json!({"language": "py", "code": "x = 40 + 2"}))
            .await
            .unwrap();
        let r = eval(env.clone(), json!({"language": "py", "code": "x"})).await.unwrap();
        let pa_agent::types::ToolResultContent::Text(content) = &r.content[0] else {
            panic!("unexpected content");
        };
        let text = content.text.clone();
        assert!(text.contains("42"), "got: {text}");
        // reset wipes the kernel
        eval(env.clone(), json!({"language": "py", "code": "x", "reset": true}))
            .await
            .unwrap();
        shutdown_all();
    }

    #[tokio::test]
    async fn js_state_persists_and_top_level_await_works() {
        let env = test_env("js-state");
        eval(
            env.clone(),
            json!({"language": "js", "code": "let acc = await Promise.resolve(40)"}),
        )
        .await
        .unwrap();
        let r = eval(env.clone(), json!({"language": "js", "code": "acc + 2"}))
            .await
            .unwrap();
        let pa_agent::types::ToolResultContent::Text(content) = &r.content[0] else {
            panic!("unexpected content");
        };
        assert!(content.text.contains("42"), "got: {}", content.text);
        // js reset wipes the kernel (fresh interpreter on next call)
        eval(
            env.clone(),
            json!({"language": "js", "code": "acc", "reset": true}),
        )
        .await
        .unwrap();
        let r = eval(env.clone(), json!({"language": "js", "code": "typeof acc"}))
            .await
            .unwrap();
        let pa_agent::types::ToolResultContent::Text(content) = &r.content[0] else {
            panic!("unexpected content");
        };
        assert!(
            content.text.contains("\"undefined\""),
            "state wiped: {}",
            content.text
        );
        shutdown_all();
    }

    #[tokio::test]
    async fn js_error_and_timeout_annotations_match_omp() {
        let env = test_env("js-timeout");
        let r = eval(
            env.clone(),
            json!({"language": "js", "code": "throw new Error('boom')"}),
        )
        .await
        .unwrap();
        let pa_agent::types::ToolResultContent::Text(content) = &r.content[0] else {
            panic!("unexpected content");
        };
        assert!(content.text.contains("Error: boom"), "got: {}", content.text);

        let r = eval(
            env.clone(),
            json!({"language": "js", "code": "while (true) {}", "timeout": 1}),
        )
        .await
        .unwrap();
        let pa_agent::types::ToolResultContent::Text(content) = &r.content[0] else {
            panic!("unexpected content");
        };
        assert!(
            content
                .text
                .contains("The JS worker was force-killed and its VM state was reset"),
            "got: {}",
            content.text
        );
        // kernel recreated transparently
        let r = eval(env.clone(), json!({"language": "js", "code": "'alive'"}))
            .await
            .unwrap();
        let pa_agent::types::ToolResultContent::Text(content) = &r.content[0] else {
            panic!("unexpected content");
        };
        assert!(content.text.contains("alive"), "got: {}", content.text);
        shutdown_all();
    }
}
