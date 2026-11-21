# Item 1: mc-gui skill distilled from real failures (HCU arXiv 2610.41432, section 3.3)
Context: ~/fleet-split/mcp-mc/server/mcp_mc.py has mc_ui (compact screen view: refs, frame) and mc_act (batched guarded GUI steps, waits, diff). In HCU, a short usage skill distilled from recurring mistakes cut malformed calls 17% -> 1.4% and steps -15%.
Eval harness: ~/fleet-split/omt/bench/h2/xgui.py (Minecraft-menu simulator served through the REAL mcp_mc.py) driven by `python3 xeval.py gui --arms ... --tasks ... --reps N --par 2 --label L` (cwd ~/fleet-split/omt/bench/h2). Traces of earlier runs: results/x-gui-traces/v1-*.json (steps with tool args/outputs, sim clicks). Earlier result: base 27.5 calls vs hcu 13.5, both 14/15 success.

## Do
1. Mine the traces (all v1-* files, both arms): list recurring mistakes with counts (malformed actions JSON, refs that do not exist, ignored stopped/frame_stale, redundant mc_ui reads, unnecessary waits, missing final verification, etc.).
2. Write ~/fleet-split/mcp-mc/skills/mc-gui/SKILL.md (front matter `name: mc-gui`, `description:` one line; body <= 2500 chars): channel choice (mc_ui/mc_act vs mc_screen/mc_click), how to batch, how to read stopped/frame_stale/diff, verification, recovery rules. No task-specific solutions.
3. In xgui.py add arm "hcu_skill" = the hcu tools + the SKILL body passed with `--append-system-prompt` (extend gui_one; do not edit xeval.py), and add 2 NEW held-out tasks to the simulator that were not in the traces (e.g. key binding change in Controls, language or difficulty change; extend Sim screens accordingly, deterministic checks).
4. Run `python3 xeval.py gui --arms hcu,hcu_skill --tasks g_video,g_packs,g_fov,<new1>,<new2> --reps 5 --par 2 --label skill-v1` and report per-arm success, tool calls, tokens, schema/argument errors (count failed mc_act parses), with a bootstrap 95% CI of the per-task-mean difference.
## You own
~/fleet-split/mcp-mc/skills/mc-gui/** , ~/fleet-split/omt/bench/h2/xgui.py , ~/fleet-split/omt/bench/h2/results/x-gui-skill-v1.* and results/x-gui-traces/skill-v1-* , PROGRESS: ~/fleet-split/mcp-mc/skills/mc-gui/PROGRESS.log
If a tool bug in mc_ui/mc_act blocks you, describe it in your reply (another agent owns mcp_mc.py).

## Rules (all items)
- You may create/modify ONLY the files listed under "You own". Everything else is read-only for you (other agents are editing other files in parallel right now).
- Work in small verified steps; after each step append one line to your PROGRESS file (step, test result). Never leave a file you own broken between steps.
- Model calls are rate-limited (429s happen): any eval you run uses --par 2 at most.
- Leave no background process running at the end (Blender, Minecraft, notepad, sim servers); remove temp files.
- Final reply must not be empty: files changed, commands run with pass/fail counts / numbers, what is NOT verified.
