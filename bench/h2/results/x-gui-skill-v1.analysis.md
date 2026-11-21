# skill-v1 analysis — mc-gui SKILL.md eval (item: HCU arXiv 2610.41432 §3.3 follow-up)

## Run
- `python3 xeval.py gui --arms hcu,hcu_skill --tasks g_video,g_packs,g_fov,g_keys,g_lang --reps 5 --par 2 --label skill-v1`
- 50/50 episodes completed (25 per arm), model zai/glm-5.3-flash (same as v1 baseline), par=2, rows in results/x-gui-skill-v1.log, traces in results/x-gui-traces/skill-v1-*.json.
- hcu_skill = hcu tools (mc_ui/mc_act/mc_state) + mc-gui SKILL.md body via `--append-system-prompt` (plumbing smoke-tested first: results/x-gui-skill-v1-smoke.log, 1 episode, success).
- g_keys / g_lang are NEW held-out tasks added with this change (Controls->Key Binds rebinding with listen/esc-cancel dynamics; Language selection with pending->apply-on-Done). Not present in any v1 trace.

## Headline (per-episode means)
| arm | success | calls | tokens | tool errors | dead clicks |
|---|---|---|---|---|---|
| hcu | 24/25 (0.96) | 10.4 | 41,908 | 0 | 0 |
| hcu_skill | 25/25 (1.00) | 10.0 | 46,247 | 0 | 0 |

Per-task success (hcu -> hcu_skill): g_video 5/5->5/5, g_packs 5/5->5/5, g_fov 5/5->5/5, g_keys 4/5->5/5, g_lang 5/5->5/5.
Per-task mean calls: g_video 10.4->12.4, g_packs 17.4->12.6, g_fov 10.2->10.4, g_keys 8.2->7.8, g_lang 6.0->7.0.

## Bootstrap 95% CI (10,000 resamples, reps resampled within task x arm, percentile)
- per-task-mean success diff (hcu_skill - hcu): +0.040, CI (+0.000, +0.120).
- per-task-mean calls diff: -0.4, CI (-1.1, +0.5)  (not significant; g_packs alone -4.8).
- tokens: +4,339/episode mean (skill text adds prompt tokens; no CI run).

## Schema / argument errors (failed mc_act parses)
- skill-v1: hcu 0/25 files, hcu_skill 0/25 files (v1 baseline had 1 in 15 hcu files). 0 malformed-actions errors in this run.

## Mistake counts, traces
| signal | v1 hcu (15 runs) | skill-v1 hcu (25) | skill-v1 hcu_skill (25) |
|---|---|---|---|
| wait_timeout (total) | 15 | 12 (fov 3, packs 4, video 2, keys 3) | 18 (fov 1, packs 8, video 6, keys 2, lang 1) |
| frame_stale | 4 | 7 | 5 |
| no_target (invented ref) | 1 | 0 | 0 |
| malformed actions JSON | 1 | 0 | 0 |
| ends without any read/act-wait verification | 10/15 | 20/25* | 18/25* |
*most episodes end on mc_act whose trailing wait succeeded (act_wait_end), which the skill encourages; read_end hcu 5, skill 6+1.

## Trace mining that produced the skill (v1, 30 files)
- base arm: 27.5 steps/file avg; screen reads 177 (11.8/file) + clicks 168 (11.2/file) = read-per-click pattern; 18 dead clicks (1.2/file); 9 consecutive redundant screen reads; 5 same-widget-3plus.
- hcu arm: 13.5 steps/file avg; wait_timeout 15 (6 = waiting for a final value after one click of a multi-click cycle/slider, 5 = waiting for unreachable widgets/text, 3 = wait after a no-effect click, 1 = popup screen_changed mid-batch); frame_stale 4; invented ref "> High Contrast" 1; malformed actions JSON 1; partial-batch-then-stop 15 (completed steps already applied); skipped final verification 10/15.

## The single failure (hcu g_keys rep 0)
checks {"bound": true, "title": false}: agent rebound Jump->B, verified by re-entering Key Binds, but finished with the Controls screen still open instead of returning to title. No failed tool calls.

## Caveats / NOT verified
- One model (zai/glm-5.3-flash), 5 tasks x 5 reps; success diff CI touches 0 — 25/25 vs 24/25 is weak evidence of a success lift; calls/tokens differences are not significant overall.
- g_keys/g_lang exercise a simulator (Sim screens added here), not real Minecraft; real-MC behavior beyond the modeled dynamics is untested.
- The skill's effect on wait_timeout counts is NOT a reduction in this run (18 vs 12; on common tasks 15 vs 9) — agents using the skill assert more intermediate states via waits; timeouts are recovered from diffs and did not break success.
- HCU paper's headline (malformed calls 17%->1.4%) not reproducible here: both arms already at 0 parse failures in this run (baseline v1 had 1/15 files).
- base arm not re-run under skill-v1; base numbers cited from v1 for context only.
