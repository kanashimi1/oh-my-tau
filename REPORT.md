# Token/output containment recovery report

## Result

- Code SHA: `179e076bcc39c08e753f54fa10dad13569cbd4c9`
- Branch: `agent/omt-containment-v2`
- Pushed: `origin/agent/omt-containment-v2`
- Base in this worktree: `846be10`
- Temporary commit `1f62c04` was unavailable; this slice was rebuilt against the current APIs. The older `9f6f62c` sketch was inspected but not cherry-picked because it did not compile against this base.

## Implementation

- Added `crates/omp-cli/src/tools/structured.rs`.
  - Preserves `false`, `0`, `null`, empty strings, and all other JSON values by default.
  - Omits defaults only when both a JSON Schema and `omit_schema_defaults` are explicitly supplied.
  - Encodes non-empty arrays of same-shape flat objects with uniform scalar column types as `# omt.table.v1` typed TSV.
  - Escapes backslashes, tabs, LF, and CR in column names and string cells; nested, mixed-type, empty, and shape-mismatched rows fall back to compact JSON.
  - Above the configured artifact threshold, stores the complete encoded output through existing `save_artifact` /`artifact://N` handling and returns JSON head/tail, omitted/retained byte and line counts, full size, and the handle.
- Wired the encoder into the existing Code Mode `output(..., query=...)` bridge in `eval_bridge.rs`, using the existing `SpillConfig` for the call cwd.
- Existing Code Mode session variables were reused: `PI_SESSION_FILE`, `PI_ARTIFACTS_DIR`, and `PI_EVAL_LOCAL_ROOTS`; no new environment variable or artifact store was added.

## Verification

All commands used `CARGO_TARGET_DIR=/home/kaneki/fleet-split/.cargo-target/omt/blaze-v2-release` and `-p omp-cli` only.

- `cargo test -p omp-cli tools::structured::tests -- --nocapture`: **4 passed, 0 failed**.
  - Typed TSV round trip, including false/zero/null and escaping.
  - JSON fallback cases.
  - Schema-only, explicit default omission.
  - Oversized artifact round trip and measurement fixture.
- `cargo test -p omp-cli output_query_uses_lossless_typed_tsv -- --nocapture`: **1 passed, 0 failed** on the real output bridge.
- `cargo run -p omp-cli -- --version`: **omt/0.10.0**.
- `git diff --check`: **passed** before commit.
- Existing unrelated warnings remain in `rpc.rs` for `LEGACY_MAX_GET_MESSAGES_PAGE_SIZE` and `legacy_capabilities_payload`.

## Measurement

Representative input: 3,000 homogeneous rows with `id`, `ok`, `zero`, `nil`, and escaped `text`. Baseline is the previous path's pretty JSON. The deterministic token proxy is `ceil(bytes / 4)`, not a model tokenizer claim.

| Surface | Bytes | Token proxy |
|---|---:|---:|
| Baseline pretty JSON | 295,892 | 73,973 |
| Full lossless typed TSV artifact | 85,958 | 21,490 |
| Inline head/tail envelope | 37,521 | 9,381 |

Full encoding is 70.95% smaller than baseline; inline output is 87.32% smaller. The inline envelope is intentionally lossy as a view; the artifact round trip was asserted lossless.

## Gaps / integration remaining

- Final-expression and `display()` rendering still use the existing JS `inspect` / Python `repr` runner path and existing eval stream truncation. They were not changed because their structured values cross the runner/host boundary before the CLI artifact policy is applied.
- Non-query `output()` raw text, line slicing, and `format=json` API shape remain unchanged except that query-selected content now uses the structured encoder.
- TSV intentionally falls back to JSON for mixed types within a column; this avoids inventing a lossy column coercion.
- Artifact creation still requires an active session and retains the existing 64 MiB artifact-body cap.
- The supplied main SHA `6b27f77` was not present in this clone; no rebase, fetch, branch cleanup, or unrelated integration was attempted.
