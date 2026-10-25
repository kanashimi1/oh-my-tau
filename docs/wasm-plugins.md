# WASM plugins

omt runs plugins as WebAssembly instead of requiring Bun. A WASM plugin is a
single `.wasm` file that is sandboxed (no filesystem/network access), cannot
crash omt, and runs in the static binary on every OS.

## ABI

A plugin is a WebAssembly module implementing a JSON-over-linear-memory
contract.

Exports:

- `memory` — the module's linear memory.
- `omt_alloc(len: i32) -> i32` — allocate `len` bytes and return a pointer.
- `omt_manifest() -> i64` — return `(ptr << 32) | len` of a UTF-8 JSON
  manifest buffer in plugin memory.
- `omt_call(ptr: i32, len: i32) -> i64` — handle a call; `ptr`/`len` point
  at the request JSON, and the result is returned as `(ptr << 32) | len` of
  a response JSON buffer.

Imports (module `omt`):

- `log(ptr: i32, len: i32)` — the only host capability; writes a UTF-8 log
  line. There is no filesystem or network access.

Manifest JSON:

```json
{
  "name": "hello",
  "version": "1.0.0",
  "tools": [
    {
      "name": "greet",
      "description": "Greet a person by name",
      "parameters": {
        "type": "object",
        "properties": { "name": { "type": "string" } },
        "required": ["name"]
      }
    }
  ],
  "commands": [
    { "name": "ping", "description": "Reply pong" }
  ]
}
```

Call request (`omt_call` input):

```json
{ "kind": "tool" | "command", "name": "greet", "args": { "name": "Mia" } }
```

Response (`omt_call` output):

```json
{ "ok": true, "text": "Hello, Mia!" }
```

or on failure:

```json
{ "ok": false, "error": "something went wrong" }
```

## Discovery

Plugins are loaded from `~/.omp/agent/plugins/*.wasm` and
`<cwd>/.omp/plugins/*.wasm`. Loading is lazy at startup; a broken plugin is
skipped with a notice and never fails omt.

## Integration

- Each manifest tool becomes an agent tool named `<plugin>_<tool>`
  (e.g. `hello_greet`).
- Each manifest command becomes a slash command `/<plugin>:<command>`
  (e.g. `/hello:ping`).
- `/plugins` lists WASM plugins (name, version, tools, commands, path, load
  errors).

## Limits

- Fuel metering on by default (10^10 fuel units per call; the interpreter is
  `wasmi`, pure Rust).
- Memory limit 256 MiB per call.
- Per-call wall timeout 30 s (calls run on a dedicated OS thread so a stuck
  plugin never stalls the single-threaded runtime).

## Example

Build the hello plugin:

```sh
cd examples/wasm-plugin-hello
cargo build --target wasm32-unknown-unknown --release
```

Copy the artifact (`target/wasm32-unknown-unknown/release/wasm_plugin_hello.wasm`)
into `.omp/plugins/` (or `~/.omp/agent/plugins/`), then:

- `/plugins` shows the `hello` plugin.
- ask the model to call the `hello_greet` tool, or run `/hello:ping`.

## Implementing a plugin

The `examples/wasm-plugin-hello` crate is a complete `#![no_std]` reference:
it exports `memory`, `omt_alloc`, `omt_manifest`, `omt_call` and imports
`omt.log`. The plugin owns all request/response buffers; it must tolerate the
host writing the request into the buffer returned by `omt_alloc` and then
reading the buffer returned by `omt_call` (single call at a time).
