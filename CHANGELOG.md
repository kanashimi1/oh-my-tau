# Changelog

## 0.13.0

Everything since v0.12.0 (2026-10-25): the local development line from 2026-10-24 to
2026-11-09 plus the upstream v0.10.0-v0.12.0 changes, merged into one tree.

### Browser UI and desktop
- Session explorer, tmux-style panes, review queue, worktree spawn, pin flow,
  activity-glyph picker, stored titles for resumed agents.
- File preview panel: HTML (sandboxed, relative assets), images, video/audio (Range),
  PDF, Markdown/text, Office via PowerPoint/Word/Excel -> PDF; live reload on file
  change; pins on any preview; inline image/video/audio previews in chat replies.
- Composer: dictation (mic button -> local faster-whisper via `POST /api/transcribe`),
  any-file attachments (`POST /api/uploads`), inline completion from prompt history.
- Redesigned themes and photo wallpapers, graphite session-dot marks, Maps tab with code
  hypergraph views, agents panel with roles/log tail, sessions parked on background
  subagents are flagged.
- **Tau desktop app** (`desktop/`, Tauri 2, Windows): tray, native notifications, single
  instance, re-bindable global capture shortcut, region capture for preview pins,
  dictation settings (enabled, hold time, language), F5 / Ctrl+R reload.
- `omt attach`: one live session across desktop, web and terminal.
- Tau mascot sprite (web app, terminal welcome, favicon, desktop icons).

### Agent, tools and protocol
- New slash commands and features: `/plan`, `/wt`, `/btw`, `/loop`, `/shake`, `/restart`,
  `/skills`, `/marketplace`, `/plugins`, `/share`, `/ssh`, `/browser`, `/budget` (stops the
  agent when a cost/token budget is spent), `/sandbox` (Landlock + seccomp confinement of
  tool subprocesses on Linux; `--sandbox`, `sandbox.enabled`), `omt daemon`.
- Compact system prompt (`--compact-prompt`, `systemPrompt.compact`).
- Live per-turn token usage and cost in the TUI loader row.
- **WASM plugins** (`crates/omp-wasm`, `docs/wasm-plugins.md`, `examples/wasm-plugin-hello`):
  sandboxed `.wasm` tools and slash commands, run by the pure-Rust `wasmi` interpreter.
- Parallel zero-copy session scan, parallel `grep`/`find`/`ast` with cooperative
  cancellation, subagent concurrency defaults to `min(CPUs, 8)` (`task.maxConcurrency`).
- Read-only tools start as soon as their streamed arguments complete; a panicking tool
  fails its own call, not the turn; kill guard for agent shell commands; `codeMode`
  projection; `save_artifact` output capped at 64 MiB.
- RPC: versioned `get_capabilities` with backward message paging, deltas for
  `message_update` in detached mode, producer heap released at turn boundaries (memory fix
  for long RPC sessions), `OMP_RPC_SOCKET` no longer leaks into task children, hidden
  steer notice so a mid-run message continues the task.
- MCP: call lifecycle from upstream 18.12.4 (cancel, no double run, progress), unique
  JSON-RPC ids for concurrent HTTP requests.
- Rubric JSONL log: one write per record (no torn lines under concurrency).

### Fleet / a2a
- `omt hub`: durable SQLite-WAL envelope store, Unix-socket JSON-RPC server, claims with
  fencing tokens, pane delivery, stdio MCP server (`omt hub mcp`); the hub exits when
  orphaned or idle.
- Task children run in git worktrees with a merge review flow (`omt agent-merge`);
  non-blocking `task`, per-agent `approvalMode`.

### Context policy and judge (opt-in)
- `context.policy` / `OMP_CONTEXT_POLICY` (`always`, `rule`, `laya`, `schedule`),
  cache-aligned batch eviction, recall-on-miss, offload-match signal.
- One shared Laya judge daemon per model (`omt laya-serve`) instead of a copy per `omt`
  process. The ONNX weights are not part of the repository; without them the judge and
  the `laya` policy kind are unavailable and everything else behaves as before.
- The ONNX runtime is a Cargo feature (`onnx`, default on); `--no-default-features` builds
  without it (used by the static musl build).

### Providers and login
- 14 more `omt login` flows: gitlab-duo, gitlab-duo-agent, cloudflare-ai-gateway,
  factory-droid, kilo, xiaomi, cursor, muse-code, perplexity, google-gemini-cli,
  google-antigravity, openai-codex-device, devin, stencil; custom endpoint prompts.
- opencode-go session header; a mid-stream error frame now fails the turn instead of
  ending it as a success.
- A 429 with a fallback chain switches to the fallback immediately; `retry.maxRetries`
  default is 4 (was 10). Retry-After is honoured, body-read failures after a 2xx are
  retried, `--model a,b,c` chains become the runtime fallback chain.
- `web_search` Anthropic OAuth credential path.

### Build, release and quality
- Static musl builds (`scripts/build-musl.sh`), Windows and macOS portability (cfg-gates),
  arm64 Linux. Release workflow builds `omt-linux-{x64,arm64}` (glibc) and
  `-musl`, `omt-macos-{x64,arm64}`, `omt-windows-x64.exe` with `SHA256SUMS.txt`;
  `install.sh` picks the right asset (Linux/macOS).
- `dist` Cargo profile (release minus DWARF line tables).
- Workspace lints: `unwrap_used`/`expect_used` denied in library code, pedantic clippy,
  sync-to-async bridges (`block_on`) disallowed; `runtime.rs`, `slash.rs`,
  `eval_agents.rs` and the TUI `app.rs` split into modules.
- CI: fmt, clippy and tests (fmt/clippy are advisory while the remaining pre-existing lint
  debt is paid down), build matrix on macOS, arm64 Linux and Windows.

## 0.12.0 and earlier
See the GitHub releases (v0.10.0 initial public release, v0.11.0, v0.12.0).
