// Fire-and-forget handler that blocks on a core request (ctx.compact()) a
// moment after it started, so awaited emits are already parked behind it.
interface Ctx {
  compact(): Promise<void>;
}
interface Api {
  on(event: "agent_start" | "turn_start", handler: (event: unknown, ctx: Ctx) => unknown): void;
}

export default function (pi: Api) {
  pi.on("agent_start", async (_event, ctx) => {
    await new Promise((r) => setTimeout(r, 300));
    await ctx.compact();
  });
  pi.on("turn_start", () => undefined);
}
