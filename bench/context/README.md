# bench/context — context-strategy bench (append vs clm vs clm-host)

Scripted runs through the real runtime (`omp-rs --mode rpc`, temp HOME with copied provider
credentials, no tools but `context_edit`) against a cheap real model. Ported from
`runs/clm-blackbox` (t1 anchors, t2 token counts, t3 hostile steer + PIN, 30-step t1 horizon);
seeds, answers and envelope schedule are unchanged, so results compare with REPORT-m4/horizon.

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
