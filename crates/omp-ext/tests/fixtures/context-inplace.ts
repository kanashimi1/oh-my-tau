// Pi-style context handler that edits the conversation clone in place and
// returns nothing (as SoL-Pi ObservationPack does).
export default function contextInplace(pi: any): void {
  pi.on("context", async (e: any) => {
    for (const m of e.messages) if (m.role === "user") m.content = [{ type: "text", text: "packed" }];
  });
}
