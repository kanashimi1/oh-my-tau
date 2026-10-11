# EvalKernelSpike - step 5: omp-eval Python kernel core

## Delivered
New crate crates/omp-eval/ (workspace glob crates/* already covers it; no workspace edit needed).

- src/runner.py: self-contained stdlib-only Python runner, wire-compatible with omp
  pi-coding-agent/src/eval/py/runner.py NDJSON protocol. Host sends {id, code, silent?, storeHistory?,
  timeoutMs?} and {"type":"exit"}; runner emits started/stdout/stderr/display/result/error/done frames
  (executionCount camelCase keys, like omp). Protocol channel owns fd 1; user stdout/stderr go through
  line-buffered proxies so interleaving is preserved. Error framing matches omp _emit_error: SyntaxError
  in the cell -> caret-only REPL-style lines; other exceptions -> TracebackException with runner frames
  ("<string>") filtered; lines rstripped. Exec core: tail-expression split (repr -> result frame) +
  PyCF_ALLOW_TOP_LEVEL_AWAIT (top-level await cells via the IPython eval trick to obtain the module
  coroutine); state persists across cells in _USER_NS. Prelude: display(obj) helper (repr + JSON mime
  when safe), __omp_prelude_loaded__ guard. storeHistory:false cells (startup) do not advance
  executionCount, like omp. NOT ported (minimal scope): cell magics, rich _repr_*_ fallbacks,
  shadow-plan/speculation, tool bridge (protocol study only - omp bridge is host-side
  {type:"tool", op:"describe|call"} frames over the same channel).
- src/lib.rs: PythonKernel (spawn python3 -u -c RUNNER, init + prelude within 10s startup budget,
