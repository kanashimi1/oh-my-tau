<p align="center"><img src="docs/brand/banner.svg" alt="oh my tau (omt)" width="720"></p>

# τ oh my tau (omt)

**oh my tau** (**omt**) is a Rust port of [oh-my-pi](https://github.com/can1357/oh-my-pi)
("omp"), the TypeScript coding agent. It re-implements the parts of omp that
matter day to day: the interactive TUI, headless print mode, an RPC mode, a
browser UI, and the model/provider/session machinery — reading and writing the
same config and session files as omp, so the two can be used interchangeably
on the same agent directory (`~/.omp/agent`).

The release binary is `omt`. Highlights:

- **TUI** — interactive terminal session with a `/model` browser, autocomplete,
  themes and magic-keyword gradients.
- **Print mode** — `omt -p "prompt"` for one-shot, scriptable runs.
- **RPC mode** — `omt --mode rpc`: newline-delimited JSON-RPC on stdio (each
  browser session below runs as one of these).
- **Browser UI** — `omt web` serves a local web UI (default
  http://127.0.0.1:7467) with themes, wallpapers and per-session agents.
- **Extensions** — runs unmodified omp `ExtensionAPI` extensions through a
  Bun-based extension host (`ext-host/`).
- **MCP** — Model Context Protocol client (stdio and Streamable-HTTP/SSE).
- **LSP** — language-server integration with diagnostics and workspace edits.
- **Sessions** — omp session JSONL (format v3) read/write; `-c`, `-r` and
  `--fork` work on the same sessions omp writes.

## Screenshots

The `omt web` browser UI:

| Session | Models | Appearance |
| --- | --- | --- |
| ![Session with sakura theme](docs/screenshots/web-sakura.png) | ![Model picker](docs/screenshots/web-models.png) | ![Appearance settings](docs/screenshots/web-appearance.png) |

## Build from source

Requirements:

- A Rust toolchain. The workspace pins no MSRV, but vendored crates use
  edition 2024, so **Rust 1.85+** is required (a recent stable is
  recommended). Check `rustc --version`.
- Optional: [bun](https://bun.sh), needed only to run omp extensions (the
  extension host is a Bun script) and for the golden tests.

```sh
cargo build --release -p omp-cli
```

The binary lands at `target/release/omt`.

## Quick start

omt resolves providers, models and credentials from omp's agent directory
(`$OMP_AGENT_DIR`, default `~/.omp/agent`). Pick whichever of these applies:

1. **Environment API key** — standard per-provider variables are picked up
   automatically, e.g. `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
   `GEMINI_API_KEY`, `DEEPSEEK_API_KEY`.

2. **OAuth login** — for providers with an OAuth flow:

   ```sh
   omt login            # pick a provider interactively
   omt login anthropic  # or name one
   ```

   Credentials are stored in `agent.db` (the omp credential store); refresh
   tokens are renewed automatically. `omt logout <provider>` removes them.

3. **models.yml** — custom providers and base-URL overrides in
   `~/.omp/agent/models.yml`:

   ```yaml
   providers:
     my-proxy:
       baseUrl: https://my-proxy.example.com/v1
       apiKey: sk-...        # literal, an env var name, or `!command` output
       api: openai-completions
   ```

Then start a session:

```sh
omt                      # interactive TUI
omt -p "summarize this"  # print mode, exits when done
omt --mode rpc           # JSON-RPC on stdio
omt web                  # browser UI on http://127.0.0.1:7467
```

Useful subcommands (see `omt <sub> --help`): `models`, `stats`, `config`,
`gc`, `export`, `judge-sessions`, `login`/`logout`, `collab`, `serve-collab`,
`web`.

## Key features

- **Slash commands** — `/help`, `/model`, `/effort`, `/compact`, `/session`,
  `/tools`, `/todo`, `/mcp`, `/usage`, `/stats`, `/ultra` and more; extensions
  and file templates can add their own.
- **Ultra modes** — `/ultra code|deep|long|hyper|off` raises the effort level
  past `max`: `ultracode` (workflow workpool on every substantial task),
  `ultradeep` (+ RLM retrieval for huge inputs), `ultralong` (+ CLM context
  strategy for long sessions), `hypercode` (both).
- **Magic keywords** — writing `ultrathink`, `orchestrate`, `workflowz`,
  `jevify` or `ultracode` as a standalone word in a prompt (never inside code
  spans or fences) fires a hidden per-message notice that steers the agent;
  the editor paints each keyword with its gradient.
- **MCP servers** — configured from `~/.omp/agent/mcp.json` and project
  `mcp.json`/`.mcp.json`; tools surface as `mcp__<server>_<tool>`. Inspect
  with `/mcp`.
- **LSP** — language servers from `defaults.json` + user/project
  `lsp.json`/`lsp.yaml`; diagnostics feed the model, edits round-trip.
- **Sessions** — resumable JSONL v3 sessions under `~/.omp/agent/sessions`,
  compatible with omp (`-c` continue newest, `-r [ID]` resume with picker,
  `--fork ID` to branch, `omt export` to HTML).

## Adding new models

New models reach omt through three layers; each is optional and later layers
win:

1. **Live refresh** — `omt models --refresh [provider]` (or `F5` in the
   `/model` browser) ports TS `ModelRegistry.refresh`: for every provider
   with a base URL + credential (models.yml `apiKey`/`baseUrl`, env API key,
   or an `agent.db` credential), it fetches `/models` (query string
   preserved), merges with the existing cache, and writes a `models.db`
   `model_cache` row in the exact TS schema — including the
   `materialization_policy` string TS readers gate on (`app-<version>:
   builder-1:rules-<n>-<wyhash of rules.json>`, computed in
   `crates/omp-providers/build.rs` from the vendored TS checkout). Fresh
   rows skip the network for 2 h (TS `BUILT_IN_DISCOVERY_CACHE_TTL_MS`).
   Verify TS still reads Rust-written rows with
   `bun scripts/verify-models-db.ts <models.db> [provider]`.
2. **models.yml** — declare the provider (`baseUrl`, `apiKey`) or pin
   individual models under `providers.<name>.models[]`; `modelOverrides`
   patches apply on top of catalog rows.
3. **Bundled catalog** — `scripts/sync-catalog.sh` regenerates the embedded
   `crates/omp-cli/assets/tui-model-catalog.json` and the pa-ai compat data
   (`vendor/pa-ai/data/compat/{rules,axes}.json`) from a newer
   `vendor-src` checkout (requires bun). Run it after every vendor-src bump
   and rebuild — the rules hash feeds the materialization policy above, so
   stale rules make TS discard the cache.

## Development

```sh
cargo test --release --workspace
```

The golden tests in `tests-golden/` compare Rust behaviour against the
TypeScript implementation; they additionally need bun and a local oh-my-pi
checkout at `vendor-src/oh-my-pi/` (gitignored), and are optional.

omt is a workspace of crates: `omp-cli` (the `omt` binary: TUI, RPC loop
and the `web` browser UI) plus `omp-providers` (model/credential resolution),
`omp-session`, `omp-tui`, `omp-ext`, `omp-mcp`, `omp-lsp` and the other
`omp-*` support crates — layered over the vendored `pa-*`/`pi-*` crates (see
[`NOTICE.md`](NOTICE.md)).

## License

omt is MIT-licensed — see [`LICENSE`](LICENSE). Third-party vendored code is
attributed in [`NOTICE.md`](NOTICE.md).
