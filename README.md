# omp-rs

Rust port of omp (oh-my-pi). Plan: ~/fleet-split/runs/omp-rust/PLAN.md.

- `vendor/pa-*`: crates from PrimeIntellect-ai/prime-agent @ 2ee7e62 (MIT, see vendor/LICENSE-prime-agent), unmodified.
- `crates/omp-session`: omp session JSONL v3 read/write (round-trip tested on real sessions).
- `crates/omp-providers`: model resolution from omp config (models.yml, config.yml, agent.db credentials) over pa-ai.
- `crates/omp-cli`: `omp-rs` binary: `-p` print mode and `--mode rpc`.
- `.ref/prime-agent`: full upstream checkout for reference (gitignored).
