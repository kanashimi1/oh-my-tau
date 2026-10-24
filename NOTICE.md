# Notices

This file covers third-party code vendored into this repository. oh my tau (omt) itself
is MIT-licensed — see [`LICENSE`](LICENSE).

## oh-my-pi (omp)

omt is a Rust port of [oh-my-pi](https://github.com/can1357/oh-my-pi) ("omp"),
a TypeScript coding agent by Mario Zechner, Can Bölük and Stencil Labs, Inc.
The upstream TypeScript source used as the porting reference is checked out at
`vendor-src/oh-my-pi/` (gitignored; not distributed). Its license text is
vendored at [`vendor/LICENSE-oh-my-pi`](vendor/LICENSE-oh-my-pi) (MIT,
Copyright (c) 2025 Mario Zechner, Copyright (c) 2025-2026 Can Bölük,
Copyright (c) 2026 Stencil Labs, Inc.).

## Vendored crates (`vendor/`)

| Crate | Origin | License |
| --- | --- | --- |
| `pa-agent`, `pa-ai`, `pa-models`, `pa-types` | [PrimeIntellect-ai/prime-agent](https://github.com/PrimeIntellect-ai/prime-agent) — **modified** in this repo | MIT, see [`vendor/LICENSE-prime-agent`](vendor/LICENSE-prime-agent) |
| `pi-ast`, `pi-diff`, `pi-edit`, `pi-predict`, `pi-vfs`, `pi-walker` | oh-my-pi (can1357/oh-my-pi), vendored verbatim per each crate's `Cargo.toml` | MIT, see [`vendor/LICENSE-oh-my-pi`](vendor/LICENSE-oh-my-pi) |
| `crossterm` | fork of [crossterm](https://github.com/crossterm-rs/crossterm) 0.28 | MIT, see [`vendor/crossterm/LICENSE`](vendor/crossterm/LICENSE) |
| `laya-ort` | part of this project (ONNX Runtime inference for the Laya judge) | MIT |

Notes:

- The `pa-*` crates are a modified fork of the prime-agent crates; the prime-agent
  license file additionally credits the original product: "a port of the Prime
  Agent TypeScript product, originally Copyright (c) 2025 Mario Zechner".
- `pi-predict` vendors only the n-gram word-completion engine of oh-my-pi; the
  smollm and apple engines are not included.
