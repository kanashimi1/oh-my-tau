// @ts-nocheck
// Conformance probe (test-only, copied into the temp HOME's extensions dir): records every event the host
// delivers (type + sorted payload keys + a few ctx facts) so omp and omp-rs emissions can be diffed.
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
const EVENTS = ["resources_discover","session_start","session_before_switch","session_switch","session_before_branch","session_branch",
  "session_before_compact","session_compact","session_shutdown","session_before_tree","session_tree","context","before_agent_start",
  "agent_start","agent_end","session_stop","turn_start","turn_end","message_start","message_end","tool_execution_start",
  "tool_execution_end","auto_compaction_start","auto_compaction_end","input","tool_approval_requested","tool_approval_resolved",
  "tool_call","tool_result","user_bash","user_python","auto_retry_start","auto_retry_end"];
export default function (pi) {
  // `/probe-tree`: navigate to the first message of the branch (session_before_tree / session_tree).
  pi.registerCommand("probe-tree", {
    description: "conformance: navigate the session tree",
    handler: async (_args, ctx) => {
      const target = (ctx.sessionManager?.getBranch?.() ?? []).find((e) => e.type === "message")?.id;
      if (target) await ctx.navigateTree(target, { summarize: false });
    },
  });
  const file = path.join(os.homedir(), "probe.jsonl");
  const w = (o) => { try { fs.appendFileSync(file, JSON.stringify(o) + "\n"); } catch {} };
  for (const ev of EVENTS) {
    pi.on(ev, async (e, ctx) => {
      let usage = "n/a", branch = "n/a", sid = "n/a", sfile = "n/a";
      try { usage = ctx.getContextUsage?.() ? "set" : "unset"; } catch {}
      try { branch = String((ctx.sessionManager?.getBranch?.() ?? []).length > 0); } catch {}
      try { sid = ctx.sessionManager?.getSessionId?.() ? "set" : "unset"; } catch {}
      try { sfile = ctx.sessionManager?.getSessionFile?.() ? "set" : "unset"; } catch {}
      let jobs = "n/a";
      try { const j = ctx.getAsyncJobSnapshot?.(); jobs = j ? Object.keys(j).sort().join(",") + ` running=${j.running?.length} recent=${j.recent?.length}` : "null"; } catch {}
      w({ ev, role: e?.message?.role, keys: Object.keys(e ?? {}).filter((k) => e[k] !== undefined).sort(), tool: e?.toolName, reason: e?.reason, source: e?.source,
        ctx: { hasUI: ctx?.hasUI, kind: ctx?.agent?.kind, usage, branch, sid, sfile, idle: ctx?.isIdle?.(), jobs, cwd: !!ctx?.cwd, mode: ctx?.mode, model: ctx?.model?.contextWindow ? "obj" : typeof ctx?.model } });
    });
  }
}
