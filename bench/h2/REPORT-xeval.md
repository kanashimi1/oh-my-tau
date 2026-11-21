# Follow-up evals: CarryOn / HCU / PoS ideas on omt (2026-11-07)
Executor zai/glm-5.3-flash (thinking low), omt rpc, temp HOME per run. Bootstrap = 95% CI of the per-task-mean difference (10k resamples, stratified by task).
Code: `xeval.py` (compact | loop | gui), `xgui.py` (GUI simulator), `xtasks.py` (trap tasks), `ext/pos-guard.ts`, `ext/pos-replay.ts`. Rows: `results/x-*-v1.log`, `x-loop-v2.log`, GUI traces in `results/x-gui-traces/`.

## 1. Compaction prompt (CarryOn-guided summary) — `x-compact-v1.log`
Two-phase task: phase 1 = investigate only (no writes); then rpc `compact` (keepRecentTokens 0, summarizer glm-5.3-flash:low); phase 2 = "complete the original task".
Arms: none (no compaction), default (stock prompt), cgs (stock prompt + CGS rules via `customInstructions`, i.e. appended as "Additional focus").
8 tasks x 3 reps; 4 runs excluded (summarizer 429 → no compaction).
| arm | score | solved | phase-2 tokens | phase-2 calls |
|---|---|---|---|---|
| none | 0.792 | 16/24 | 65.3k | 6.6 |
| default | 0.898 | 18/22 | 44.1k | 6.6 |
| cgs | 0.903 | 17/22 | 28.3k | 4.9 |
cgs−default: score +0.044 [−0.044, +0.125]; phase-2 tokens −19.3k [−41.9k, +3.3k]; calls −2.0 [−4.1, +0.2].
cgs−none: score +0.119 [−0.008, +0.244]; phase-2 tokens −34.5k [−53.4k, −17.2k] (significant).
Biggest single task: d_ledger default 0.17 vs cgs 1.00 (n=2 each).

## 2. HCU-style GUI tools — `x-gui-v1.log`
Simulated 1.20.1 menus behind the real bridge JSON contract, driven through the real `mcp-mc/server/mcp_mc.py` (MCP).
Dynamics: delayed screens (resource reload 2.5 s, world load 3 s) and a pop-up that replaces Video Settings 0.8 s after choosing Fabulous! (its Cancel sits where Clouds was: designed stale-click trap).
Arms: base = mc_screen/mc_click/mc_key/mc_state (current tools); hcu = new mc_ui (compact view, stable refs, frame id) + mc_act (batched ref clicks/keys/waits, live target resolution, frame guard, stop on screen change, diff feedback) + mc_state.
3 tasks x 5 reps. mc_act fixes made after 4 smoke runs (click+wait in one step, wait-only batches skip the frame guard, `moved` in diff); the eval runs are fresh.
| arm | success | tool calls | tokens | dead clicks | pop-up Cancel hits |
|---|---|---|---|---|---|
| base | 14/15 | 27.5 | 130k | 1.2/run | 1 |
| hcu | 14/15 | 13.5 | 57k | 0 | 0 |
hcu−base: calls −14.1 [−19.7, −8.8]; tokens −72.9k [−115k, −35k]; score 0.0 [−0.07, +0.07] (both at ceiling). Real game not tested.

## 3. PoS-style stagnation detector — `x-loop-v1.log`, `x-loop-v2.log`, offline replay
Health over the last K=5 tool results: S = steps with an error or an already-seen (call, result) pair, R = recurrence at lag 1..3, H = 1 − max(S, R); fire at H ≤ 0.34 (cooldown 4).
Pattern (static/cycle/drift) + gap type (epistemic/achievement) select the recovery text; `generic` uses the same trigger with a generic text; `none` = shadow mode (logs only).
Online: 0 fires in 93 runs (h2 data tasks 4x4x3; trap tasks t_shadow/t_utf16/t_nbsp 3x5x3; flash solved the trap tasks in ~6 calls). No effect measurable.
A/A noise: in v1 (0 fires, all arms identical in practice) generic−none = +0.135 [−0.058, +0.335] — the noise floor at n=16/arm.
Offline replay, 448 historic h2 episodes: fires in 11 (2.5%); their mean score 0.27 vs 0.86 without a fire.
Offline replay, 60 recent real omp sessions (8,496 tool calls): first version fired 19x, 10 of them false (wait <-> bash/todo polling read as a cycle) → polling tools excluded; now 7 fires (0.08/100 calls), all windows with 4-5 failing calls out of 5.

## Verdicts
1. Compaction: adopt CGS-style rules — phase-2 cost clearly lower, quality not worse (likely better); needs a default-prompt change in omp-compact (not done).
2. GUI: keep mc_ui/mc_act (added to mcp_mc.py, old tools unchanged; 13 offline tests pass) — same success with half the calls/tokens in the simulator; needs a real-game check.
3. Detector: do not enable injection yet; it is calibrated (polling fix) but trapping is too rare in short tasks to measure an effect. Shadow mode in real sessions is the next step.

## 4. Blender scene diff (HCU-style feedback on a code channel) — `x-blend-v1.log`
Bridge change (subagent glm-5.3-flash, verified by parent): `blender-ai-toolkit/scene_diff.py` + `ai_cad_bridge.py` return `rev`, `diff`, `external_changes`, refuse `expect_rev` mismatches (`scene_stale`); tests: bridge_benchmark 12/12, diff_test 17/17 on headless Blender 4.2.3; snapshot ~19 ms median for 200 cubes.
Eval: headless Blender per run, agent uses `./bl.py` (bash) only; base = diff:false, diff = rev/diff/external_changes shown. Tasks: b_table (resize + teammate moves Leg3/Lamp after the agent's 2nd call), b_mecha (repair fixture), b_posts (8 posts + 8 rails). 3 tasks x 5 reps; 1 harness error (Blender start).
| arm | success | bridge calls | tokens |
|---|---|---|---|
| base | 10/14 | 6.0 | 40k |
| diff | 10/15 | 5.3 | 45k |
diff−base: score −0.02 [−0.11, +0.07]; bridge calls −0.8 [−3.5, +1.1]; tokens +4k [−11k, +16k]. No measurable effect.
b_table failed 3/5 in BOTH arms on the leg check: in the diff arm the teammate's move was reported in external_changes and still not repaired (inferred from the outcome; traces not saved).
Reading: with a code channel the agent already batches work into 2–8 calls and writes its own verification, so diff feedback has little to remove; the GUI-click setting (section 2) is where HCU-style feedback paid off.

## 5. Blockbench bridge v2 plugin + model diff (subagents glm-5.3-flash, 6 runs: 4 aborted mid-way, work resumed from PROGRESS.log)
`blockbench-mcp-v2/plugin/blockbench_mcp.js` (80 → ~1100 lines): SPEC 1-8 (FIFO queue, per-call project, leases, textures, uv, animations, camera preview, export with codec detection, atomic batch) + `rev`/`diff` on every mutating reply, `expect_rev` → `model_stale`.
Verified by parent: `node --test plugin/test` 99/99; e2e through the real Rust binary (`BB_PORT` added to src/main.rs, default 43118) against `plugin/test/sim_server.js` (Node mocks): batch diff, atomic rollback keeps pre-batch edits (bug found by parent, root cause in the mock Undo, regression tests added), inspect.
Not verified: real Blockbench (22 UNVERIFIED API entries in plugin/NOTES.md). Not deployed: installed plugin and mcp.json (server disabled, points to v1) untouched.

## 6. Deploy + GUI items 1-6 (2026-11-08/09; subagents glm-5.3-flash, 11 runs, 5 aborted by infra and resumed from PROGRESS logs)
Deployed: Blockbench plugin v2 → %APPDATA%/blockbench/plugins/blockbench_mcp.js (v1 + old mcp.json in ~/fleet-split/backups/blockbench-v1-20261108/); mcp.json `blockbench` → v2 binary, enabled (29 tools listed over MCP). Blender bridge → extension `ai_cad_bridge_ext` 0.2.0 in Windows Blender 5.2.2 (`blender-ai-toolkit/deploy_windows.py --check`: /health ok, diff on a real 5.2 instance; port 8766 because 8765 is taken by hound-http in WSL; same token on both sides). New MCP server `uia` registered (UIA_ALLOW=Blockbench).
1 mc-gui skill (~/.agents/skills/mc-gui → mcp-mc/skills/mc-gui): hcu 24/25 vs hcu+skill 25/25, success +0.04 [0.00, +0.12], calls −0.4 [−1.1, +0.5], tokens +4.3k; 0 malformed calls in both arms.
2 mc_wait (8 conditions, existing routes only): 23 offline tests; live smoke ok on the dev client.
3 Blender strict_external (refuse until external changes are acknowledged): 22/22 tests; eval diff vs strict 7/12 vs 7/12, repair given a teammate edit 0.81 vs 0.83 — no effect.
4 mc_shot_marked (numbered boxes, zoom by ref): 9 offline tests; live smoke on the pause menu: legend of 10 widgets = mc_ui refs.
5 uia-mcp (Windows UI Automation, persistent PowerShell): 30 offline + 7 live notepad tests (1 skip: Win11 notepad dies on repeated menu cycles); found and fixed a path-resolution bug the fake hid.
6 real client: mc_ui/mc_act fixes (container slots, click_xy, key+wait, screen-name wait); bridge /screen now walks nested widgets (OptionsList rows, pack/world/language lists, KeyBinds) + /scroll; parent fixed refs for nested views (containers dropped, "[list]", "Jump > Reset", "L2 > Default") and added mc_act scroll step; live: Video Settings "Graphics" cycle via ref, KeyBinds scroll diff. mcp-mc suite 64 passed.
