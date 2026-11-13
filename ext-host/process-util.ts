// process-util.ts — child process helper (mechanical split from host.ts).
import { spawn } from "node:child_process";
import type { Json } from "./types.ts";

export function execCommand(command: string, args: string[], cwd: string, opts: Json = {}) {
  return new Promise<Json>((res) => {
    const child = spawn(command, args, { cwd, stdio: ["ignore", "pipe", "pipe"], signal: opts.signal as AbortSignal | undefined });
    let stdout = "";
    let stderr = "";
    let killed = false;
    const t = typeof opts.timeout === "number" ? setTimeout(() => ((killed = true), child.kill("SIGKILL")), opts.timeout) : undefined;
    child.stdout.on("data", (d) => (stdout += d));
    child.stderr.on("data", (d) => (stderr += d));
    child.on("error", (e) => {
      clearTimeout(t);
      res({ stdout, stderr: stderr || String(e), code: -1, killed: killed || (e as Error).name === "AbortError" });
    });
    child.on("close", (code) => {
      clearTimeout(t);
      res({ stdout, stderr, code: code ?? -1, killed });
    });
  });
}
