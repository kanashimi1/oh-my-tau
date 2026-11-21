# Harness² pilot (paper 2610.14224) — omt rpc, executor+editor = zai/glm-5.3-flash

## Setup
- Suite: `h2tasks.py`, 8 sandbox tasks with hidden checkers (0..1). STREAM (evolve): d_reconcile, c_pricing, d_logs, c_crlf. HELDOUT: d_budget, c_rename, d_sessions, c_duration.
- `h2.py evolve --k 2`: per stream task: base run → editor proposes 2 scoped lessons (invariant / domain:data / domain:code) → probe rerun with each lesson → judge accepts only on a behavioral contrast → COMPOSE into `lib/EXPERIENCE.md`. Arm `naive` = every proposed lesson, no probe (BTM style) → `lib/NAIVE.md`.
- `h2.py eval --libs none=,naive=…,h2=… --reps 2` (48 sessions). Raw data: `results/eval-pilot.{log,json}`, `results/evolve.jsonl`.

## Evolve
| task | base | probe scores | accepted |
|---|---|---|---|
| d_reconcile | 0.75 | 0.75, 1.0 | 1/2 |
| c_pricing | 1.0 | 1.0, 1.0 | 1/2 |
| d_logs | 0.333 | 0.333, 1.0 | 2/2 |
| c_crlf | 1.0 | 1.0, 1.0 | 0/2 |
Library: h2 = 1013 chars (4 data + 1 invariant, after compose), naive = 2276 chars (8 lessons).

## Eval (mean score / solved / avg tokens per run)
| arm | STREAM | HELDOUT | ALL | tok/run | tool errors |
|---|---|---|---|---|---|
| none | 0.808, 5/8 | 0.925, 7/8 | 0.867, 12/16 | 13.6k | 1 |
| naive | 0.833, 6/8 | 1.000, 8/8 | 0.917, 14/16 | 25.7k | 1 |
| h2 | 0.917, 7/8 | 1.000, 8/8 | 0.958, 15/16 | 25.2k | 4 |

Failures: d_logs (none 0/2, naive 0/2, h2 1/2 — counts+order); c_crlf none 0.8 (1 rep); d_budget none 0.4 (1 rep).

## Conclusions
- Direction matches the paper (h2 > naive > none), but n=2 per cell: the h2-vs-none gap is 3 runs out of 16 and lies within agent noise (calib runs: d_budget 1.0/0.0, c_crlf 0.8/1.0). Not statistically significant.
- The only targeted, reproducible effect: d_logs (none failed 0/2 in calib2 + 0/2 here; h2 1/2). Held-out gain = the single d_budget miss of `none` — weak evidence of transfer.
- Cost: both library arms ~1.9x tokens per run (more inspection/verification steps; the library itself is only ~250 tokens). The probe gate halves library size vs naive at equal or better score.
- Evolve cost: per stream task 1 base + k probes + editor/judge calls ≈ 3 executor runs + 3–4 editor calls.

## Probe gate in ~/bin/btm_promote.py
- New `probe <tool> [--reps 2]`: editor generates a sandbox task that should trigger the failure (cached in `~/.omc/state/memory/btm-probe-tasks/`), replays control vs clause-in-system-prompt, counts failure-mode hits; status supported|rejected|inconclusive → `~/.omc/state/memory/btm-probes.jsonl` (key = hash of tool|mode|clause).
- `propose` prints `probe=<status>` for each candidate; `apply` refuses (rc=3) unless supported, `--force` overrides. Backup: `~/bin/btm_promote.py.bak-20261107-probe`.
- First real probes: top eval (asyncio transient traceback, pairs=45) → inconclusive (not reproduced, 0 hits); top bash (exit code/wall time, pairs=38) → rejected (control 2 hits, clause 4). The two candidates with the most pairs would have been promoted under the old rule; both are now blocked.

## Follow-up (next steps 1–5)
Bootstrap = 95% CI of the per-task-mean difference (10k resamples, stratified by task).

1. **Significance** (flash, d_logs/d_budget/d_reconcile/c_crlf, pooled with the pilot, n=8 per cell):
   none 0.745 (16/32) · naive 0.817 (21/32, 23.6k tok) · h2 0.830 (22/32, 20.7k tok).
   h2−none +0.085 [−0.013, +0.184]; naive−none +0.072 [−0.016, +0.164]; h2−naive +0.013 [−0.084, +0.111].
   → the direction is consistent, but **not significant at 95%**; h2 ≈ naive on score and costs ~12% fewer tokens.
2. **Harder held-out** (new tasks d_metrics, d_ledger with an EU ';'/decimal-comma/trailing-minus export, c_cache, c_slug; flash, n=4):
   none 0.802 · naive 0.854 · h2 0.877; h2−none +0.074 [−0.112, +0.252]. Variance is large (d_ledger naive 0.42, d_metrics naive 1.00).
   The calibration target (none ≤0.6) was met only by d_metrics (0.50); c_cache/c_slug are too easy for flash (0.96/1.00).
3. **Scoped injection** (only lessons whose category matches the task, `lib@scoped`; code tasks, n=3):
   h2 0.978 (19.2k tok) · h2s 0.967 (17.4k, lib 1013→301 chars) · naive 1.000 (30.0k) · naives 0.981 (22.7k).
   Scoping cuts tokens 9–24% and wall time 23–38% at −0.01…−0.02 score (CI includes 0). Code tasks saturate (~1.0), so the score signal is weak.
4. **BTM probe**: the probe task-generation retry is in place (malformed editor JSON); `probe-top --top 10`: 2 rejected, 7 inconclusive, 1 generation failure, 0 supported
   → nothing gets promoted, and the gate blocks unproven candidates as designed. Most inconclusive cases are probes where the control arm does not reproduce the failure (eval/kernel errors).
   `~/bin/dream_cycle.sh` step 3 now runs `probe-top --top ${BTM_PROBE_TOP:-2}` (timeout 1800s; `BTM_PROBE_TOP=0` turns it off).
5. **Cross-model, executor zai/glm-5.3** (library evolved on flash, 8 tasks, n=3):
   none 0.734 · naive 0.713 (54.7k tok) · h2 0.816 (44.6k tok). h2−none +0.082 [−0.083, +0.240]; h2−naive +0.102 [−0.039, +0.241].
   The naive library *hurts* the stronger model (d_metrics 0.00, 223k tok); h2 transfers better. Executor **kimi-code/kimi-for-coding** (openrouter/kimi-k3.1-pro returned 401/402; the kimi-code provider, as in omp, works), same 8 tasks, n=3:
   none 0.763 · naive 0.797 (30.1k tok) · h2 0.818 (31.8k tok); h2−none +0.055 [−0.099, +0.198]; h2−naive +0.021 [−0.115, +0.150]. Here naive does not degrade, but h2 is still the best arm.

**Summary:** across all four samples (flash ×2, glm-5.3, kimi) h2 ≥ none (+0.06…+0.09), but no single CI excludes 0; the main measurable win is the
token/robustness edge over naive (naive degrades on another model). Significance at this effect size needs n≈25–30 per cell.

## Next steps (not done)
- n≈25/cell on the 8 tasks (or a paired design with fixed seeds) to get past the CI.
- Harder code tasks (c_cache/c_slug saturated).
