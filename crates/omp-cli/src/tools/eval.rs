//! `eval`: persistent Python kernel per cwd (omp `EvalTool`, `tools/eval.ts`).
//!
//! Cells run through [`omp_eval`]'s [`PythonKernel`], a long-lived `python3`
//! subprocess speaking omp's NDJSON protocol; state persists across calls.
//! Only Python is ported: `language: "js"` returns omp's disabled-backend
//! error text (Bun backend not implemented).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use anyhow::{bail, Result};
use omp_eval::{CellStatus, CellResult, PythonKernel};
use pa_agent::types::AgentToolResult;
use serde_json::{json, Value};

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

/// Retained kernels keyed by cwd (omp keeps them per session/interpreter;
/// omp-rs has one cwd per run).
fn kernels() -> &'static Mutex<HashMap<PathBuf, Arc<PythonKernel>>> {
    static KERNELS: OnceLock<Mutex<HashMap<PathBuf, Arc<PythonKernel>>>> = OnceLock::new();
    KERNELS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// omp's unsupported/omitted JS backend surfaces as the disabled-backend
/// error (`tools/eval.ts::resolveBackend`).
pub const JS_UNSUPPORTED: &str = "JavaScript backend is disabled (PI_JS=0 or eval.js = false).";

/// Execute one `eval` cell.
pub async fn eval(env: Arc<Env>, args: Value) -> Result<AgentToolResult> {
    let language = args.get("language").and_then(Value::as_str).unwrap_or("js");
    let code = args
        .get("code")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow::anyhow!("missing string argument `code`"))?
        .to_string();
    match language {
        "py" | "python" => {}
        "js" | "javascript" => bail!("{JS_UNSUPPORTED}"),
        other => bail!("language must be \"py\" or \"js\" (was {other:?})"),
    }

    let timeout = match args.get("timeout").and_then(Value::as_f64) {
        None | Some(0.0) => None,
        Some(secs) => Some(Duration::from_secs_f64(secs.max(1.0))),
    };
    let reset = args.get("reset").and_then(Value::as_bool).unwrap_or(false);

    let cwd = env.cwd.clone();
    if reset {
        kernels().lock().expect("kernel registry").remove(&cwd);
    }
    let kernel = {
        let mut kernels = kernels().lock().expect("kernel registry");
        if let Some(existing) = kernels.get(&cwd) {
            if existing.is_alive() {
                Arc::clone(existing)
            } else {
                let fresh = Arc::new(PythonKernel::start(omp_eval::default_interpreter(), &cwd)?);
                kernels.insert(cwd.clone(), Arc::clone(&fresh));
                fresh
            }
        } else {
            let fresh = Arc::new(PythonKernel::start(omp_eval::default_interpreter(), &cwd)?);
            kernels.insert(cwd.clone(), Arc::clone(&fresh));
            fresh
        }
    };

    let code = code.clone();
    let result = tokio::task::spawn_blocking(move || kernel.execute(&code, timeout))
        .await
        .map_err(|e| anyhow::anyhow!("eval kernel task failed: {e}"))??;

    let mut text = result.output.clone();
    if let Some(annotation) = result.kernel_timeout_annotation() {
        use std::fmt::Write as _;
        let _ = write!(text, "\n[{annotation}]");
    }
    if result.status == CellStatus::KernelKilled {
        // Kernel state is gone; drop it so the next call starts fresh.
        kernels().lock().expect("kernel registry").remove(&cwd);
    }
    Ok(AgentToolResult {
        content: vec![pa_agent::types::ToolResultContent::text(text)],
        details: cell_details(&result),
        terminate: None,
    })
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
    use std::path::Path;

    #[test]
    fn schema_matches_ts_reference() {
        let s = schema();
        let props = s["properties"].as_object().unwrap();
        assert_eq!(props["language"]["enum"], json!(["py", "js"]));
        assert_eq!(s["required"], json!(["language", "code"]));
        assert_eq!(props.len(), 5);
    }

    #[tokio::test]
    async fn js_reports_unsupported() {
        let env = Arc::new(Env::new(Path::new("/tmp"), false));
        let err = eval(env.clone(), json!({"language": "js", "code": "1"}))
            .await
            .unwrap_err();
        assert!(err.to_string().contains(JS_UNSUPPORTED));
    }

    #[tokio::test]
    async fn python_state_persists() {
        let env = Arc::new(Env::new(Path::new("/tmp"), false));
        eval(env.clone(), json!({"language": "py", "code": "x = 40 + 2"})).await.unwrap();
        let r = eval(env.clone(), json!({"language": "py", "code": "x"})).await.unwrap();
        let pa_agent::types::ToolResultContent::Text(content) = &r.content[0] else {
            panic!("unexpected content");
        };
        let text = content.text.clone();
        assert!(text.contains("42"), "got: {text}");
        // reset wipes the kernel
        eval(env.clone(), json!({"language": "py", "code": "x", "reset": true})).await.unwrap();
    }
}
