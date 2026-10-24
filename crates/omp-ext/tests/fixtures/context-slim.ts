// Stand-in for the user's `context-slim` extension: its name (file stem) is
// what the core lists in `skipExtensions` while it runs its own CLM strategy.
export default function contextSlim(pi: any): void {
  pi.on("context", async (e: any) => ({ messages: [...e.messages, { role: "user", content: "slimmed", timestamp: 1 }] }));
}
