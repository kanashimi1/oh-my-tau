# compat oracle

TS ground truth for the Rust policy resolver (`vendor/pa-ai/src/compat`) and the chat-completions
request builder (`vendor/pa-ai/src/providers/openai_completions`). Run from the repo root with
[bun](https://bun.sh) (needs an oh-my-pi checkout, see below):

    bun scripts/compat-oracle/gen-data.ts   # refresh vendor/pa-ai/data/compat/{rules,axes}.json
    bun scripts/compat-oracle/oracle.ts     # -> <out>/dump.jsonl  (resolveModelPolicy per row)
    bun scripts/compat-oracle/bodies.ts     # -> <out>/bodies.jsonl        (streamOpenAICompletions bodies)
                                           #  + <out>/bodies-transport.jsonl (streamSimple bodies)

The oh-my-pi sources are loaded from `OMP_TS_SRC` (root of an oh-my-pi checkout) if set, else
`<repo>/vendor-src/oh-my-pi` — Bun must be able to resolve that tree's imports (run
`bun install` there once). Fixtures are written to `OMP_COMPAT_ORACLE_DIR` if set, else
`<repo>/target/compat-oracle/` (created); an explicit path argument overrides a single output.
The Rust tests look the fixtures up in the same places (env, then `target/compat-oracle/`, then
the legacy `runs/omp-rust/compat-oracle/`) and skip with a pointer to this README when no fixture
is found.

- `oracle.ts` copies `~/.omp/agent/models.db` to a temp dir (the real DB is never written; when
  absent only the bundled catalog and the synthetic matrix are dumped) and runs
  `resolveModelPolicy` for every cached row, every bundled `models.json` row, and a seeded
  synthetic host × id × api × compat-override matrix; each row in a *sparse* (`compatConfig`) and
  a *full* (resolved record as overrides + explicit thinking) variant.
- Rust consumers: `compat::oracle` (resolution equality) and
  `openai_completions::body_parity_tests` (request-body equality).
- The three Rust tests run in `cargo test -p pa-ai -- compat::oracle body_parity transport_parity`.

## transport bodies (`bodies-transport.jsonl`)

The second section of `bodies.ts` drives the public `streamSimple` facade (kimi-code routing,
`PI_OPENROUTER_RESPONSES`, per-api option mapping) for anthropic-messages (incl. OAuth tokens and
proxy hosts), kimi-code (Anthropic format), openrouter (Responses default, chat with
`PI_OPENROUTER_RESPONSES=0`), openai-responses and azure-openai-responses. Each model runs the 7
contexts × 6 option sets (default, high+max, medium+temp, off, session, forced). The request is
captured in `onPayload` (then aborted — nothing is sent); cases the TS facade rejects before a
request exists (unsupported effort) are recorded with `error` and skipped by the Rust test.
`T_ONLY=1` skips the completions section. Consumer: `pa-ai` `transport_parity_tests`
(`TRANSPORT_FILTER=<substr>` narrows the cases).
UUID-shaped values (the facade mints a session id per inference) are masked when the case sets no
`sessionId`.
