// Fixture extension written against the omp ExtensionAPI surface the user's extensions use.
// Module-level state: `echoCalls` must be per-session — in the shared host each
// session runs this module in its own Worker, so counters never leak across
// sessions (asserted by the shared_daemon_sessions_are_isolated test).
let echoCalls = 0;
export default function fixture(pi: any): void {
  pi.on("session_start", async (_e: any, ctx: any) => {
    pi.appendEntry("fixture-started", { cwd: ctx.cwd, sid: ctx.sessionManager.getSessionId() });
  });
  pi.on("tool_call", async (e: any) => {
    if (e.toolName === "bash" && String(e.input.command).includes("rm -rf")) return { block: true, reason: "no rm" };
    if (e.toolName === "bash" && e.input.command === "echo hi") return { input: { command: "echo rewritten" }, additionalContext: "ctx-a" };
    if (e.toolName === "boom") throw new Error("handler exploded");
  });
  pi.on("tool_call", async (e: any) => (e.toolName === "bash" ? { additionalContext: "ctx-b" } : undefined));
  pi.on("tool_result", async (e: any) => {
    if (e.toolName === "bash" && !e.isError) return { content: [{ type: "text", text: `[redacted] ${e.content[0].text.length}` }] };
  });
  pi.on("context", async (e: any) => ({ messages: e.messages.filter((m: any) => m.role !== "drop") }));
  pi.on("before_agent_start", async (e: any) => ({ message: { customType: "fx", content: `prompt:${e.prompt}`, display: false }, systemPrompt: [...e.systemPrompt, "extra"] }));
  pi.on("hang", () => new Promise(() => {}));
  pi.registerTool({
    name: "fx_echo",
    label: "Echo",
    description: "echo",
    parameters: pi.typebox ? pi.typebox.Type.Object({ text: pi.typebox.Type.String() }) : {},
    async execute(_id: string, params: any) {
      pi.sendMessage({ customType: "fx-msg", content: "sent", display: true });
      return { content: [{ type: "text", text: `echo:${params.text}` }], details: { n: ++echoCalls } };
    },
  });
  pi.registerCommand("fx-cmd", { description: "cmd", async handler(args: string) { return `cmd:${args}`; } });
}
