# omp-rs

Rust port of omp (oh-my-pi). Plan: ~/fleet-split/runs/omp-rust/PLAN.md.

- `vendor/pa-*`: crates from PrimeIntellect-ai/prime-agent @ 2ee7e62 (MIT, see vendor/LICENSE-prime-agent), unmodified.
- `crates/omp-session`: omp session JSONL v3 read/write (round-trip tested on real sessions).
- `crates/omp-providers`: model resolution from omp config (models.yml, config.yml, agent.db credentials) over pa-ai.
- `crates/omp-cli`: `omp-rs` binary: `-p` print mode and `--mode rpc`.
- `.ref/prime-agent`: full upstream checkout for reference (gitignored).

## Adding new models

New models reach omp-rs through three layers; each is optional and later
layers win:

1. **Live refresh** — `omp-rs models --refresh [provider]` (or `F5` in the
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
   `cd vendor-src/oh-my-pi && bun ../../../scripts/verify-models-db.ts <models.db> <provider>`.
2. **models.yml** — declare the provider (`baseUrl`, `apiKey`) or pin
   individual models under `providers.<name>.models[]`; `modelOverrides`
   patches apply on top of catalog rows.
3. **Bundled catalog** — `scripts/sync-catalog.sh` regenerates the embedded
   `crates/omp-cli/assets/tui-model-catalog.json` and the pa-ai compat data
   (`vendor/pa-ai/data/compat/{rules,axes}.json`) from a newer
   `vendor-src` checkout (reuses `runs/omp-rust/compat-oracle/gen-data.ts`).
   Run it after every vendor-src bump and rebuild — the rules hash feeds the
   materialization policy above, so stale rules make TS discard the cache.
