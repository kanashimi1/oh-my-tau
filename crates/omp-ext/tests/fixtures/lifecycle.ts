// Records the exact JSON of every lifecycle event it receives (as `seen:<type>` entries) and answers like omp extensions do.
const OBSERVED = [
  "session_start", "agent_start", "agent_end", "turn_start", "turn_end", "message_start", "message_update", "message_end",
  "tool_execution_start", "tool_execution_update", "tool_execution_end", "session_compact", "session_switch",
  "session_branch", "session_tree", "session_shutdown",
  "assistant_message", "auto_compaction_start", "auto_compaction_end", "auto_retry_start", "auto_retry_end",
  "retry_fallback_applied", "retry_fallback_succeeded", "ttsr_triggered", "todo_reminder", "goal_updated",
  "tool_approval_requested", "tool_approval_resolved",
];
const RESULTS: Record<string, unknown> = {
  before_agent_start: { message: { customType: "fx", content: "hello", display: false }, systemPrompt: ["sys-a", "sys-b"] },
  input: { text: "rewritten", images: [] },
  session_before_compact: { cancel: false, compaction: { summary: "ext summary" } },
  session_before_switch: { cancel: true },
  session_before_branch: { cancel: false },
  session_before_tree: { cancel: true },
  session_stop: { continue: true, additionalContext: "keep going" },
  user_bash: { result: { output: "handled", exitCode: 0 } },
  user_python: { result: { output: "py", exitCode: 0 } },
};

export default function lifecycle(pi: any): void {
  for (const type of [...OBSERVED, ...Object.keys(RESULTS)]) {
    pi.on(type, async (e: any) => {
      pi.appendEntry(`seen:${type}`, JSON.stringify(e));
      if (type === "assistant_message")
        return { content: e.message.content.map((b: any) => (b.type === "text" ? { ...b, text: b.text.toUpperCase() } : b)) };
      return RESULTS[type];
    });
  }
  // host -> core requests, driven from a handler
  pi.on("probe_requests", async (_e: any, ctx: any) => {
    const confirm = await ctx.ui.confirm("title", "message");
    await ctx.compact({ customInstructions: "short" });
    const tree = await ctx.navigateTree("entry-1", { summarize: true });
    const model = await pi.setModel({ id: "m" });
    await pi.setSessionName("renamed");
    pi.appendEntry("probe", JSON.stringify({ confirm, tree, model }));
  });
}
