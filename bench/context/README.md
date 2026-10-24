# bench/context — context-strategy bench (append vs clm vs clm-host)

Scripted runs through the real runtime (`omp-rs --mode rpc`, temp HOME with copied provider
credentials, no tools but `context_edit`) against a cheap real model. Ported from
the clm-blackbox harness (t1 anchors, t2 token counts, t3 hostile steer + PIN, 30-step t1 horizon);
seeds, answers and envelope schedule are unchanged, so results compare with the earlier clm-blackbox reports.

```
cargo build --release -p omp-cli            # CARGO_TARGET_DIR=target/shared
python3 bench/context/run.py --tasks t1,t2,t3 --strategies append,clm --model zai/glm-5.3-flash
python3 bench/context/run.py --horizon --strategies append,clm --model zai/glm-5.3-flash
python3 bench/context/report.py bench/context/results/<run-id>
```

Per cell: prompt uncached (`usage.input`), prompt cached (`cacheRead`), cache write, completion
tokens, cache-hit % = cached / (uncached + cached + cache write), wall latency, LLM calls,
accepted/rejected `context_edit` calls, task score. `report.py` adds the acceptance comparison:
CLM cache-hit within 5 pp of `append` on the same task/model/steps.

Fairness: every strategy sees the same message script (handshake, seed at step 1, envelopes at
the clm-blackbox schedule); CLM arms additionally get a per-step maintenance instruction. Step 0
is a short handshake so the seed is *not* in the pinned prefix and CLM can fold it into slots.

## Context-policy experiment (`ctxp.py`, `ctxp_analysis.py`)

Does a controller decide *when* to edit (`clm_edit`), offload a block to the kernel (`rlm_offload`) or do nothing
(`noop`)? Opt-in host hook: `context.policy = always|rule|laya|schedule` (env `OMP_CONTEXT_POLICY`, only with
`--context-strategy clm|clm-host`; default unset = unchanged behaviour). See `crates/omp-cli/src/context_policy.rs`;
rule knobs `OMP_CONTEXT_POLICY_RULE_{TOKENS,EVERY,OFFLOAD}`, laya `OMP_CONTEXT_LAYA_{CKPT,ONNX,THREADS}`,
decision log `OMP_CONTEXT_POLICY_LOG`, offload dir `OMP_CONTEXT_OFFLOAD_DIR`. Results and verdict: see `RESULTS-glm-5.3-flash-v2.md`.

```
python3 bench/context/ctxp.py cf --tasks t1 --seeds 0,1,2,3,4,5,7     # counterfactual rollouts (7 per source)
python3 bench/context/ctxp_analysis.py labels|tune|data|oracle-scripts
python3 bench/context/ctxp.py run --policy append|clm|clm-host|rule|laya|oracle --sources t1:0,t1:1 [--rule 0,0,2000] [--laya CKPT,ONNX]
python3 bench/context/ctxp_analysis.py report
```
