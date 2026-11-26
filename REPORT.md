# Transactional apply_patch recovery report

## Implementation

- Commit: `0ee1d12294332da0519739a74cf09553404237a8`
- Branch: `agent/omt-tx-oracles-v2` (pushed)
- Added optional per-file `*** Expected BLAKE3: <64-hex>` anchors for exact original bytes. If any section is anchored, every changed pre-existing file in that patch must be anchored; creates are exempt.
- Stale, malformed, duplicate, and missing anchors reject staging before any write.
- A later apply-patch writer failure restores earlier writes in reverse order through the existing writer interface. Original text bytes, including BOM/line endings, are restored.
- No bash overlay was added.

## Verification

- `CARGO_TARGET_DIR=/home/kaneki/fleet-split/.cargo-target/omt/blaze-v2-release cargo test -p omp-cli apply_patch`: **3 passed, 0 failed**.
  - Real edit-tool two-file anchored success.
  - Stale-anchor and incomplete-anchor rejection with both files unchanged.
  - Injected second-write failure with byte-identical CRLF rollback of the first file and an unchanged second file.
- `git diff --check`: passed before the implementation commit.

## Remaining / limitations

- Rollback is best-effort, not crash atomicity. A process crash, a writer that partially modifies the failing file before returning an error, or a failed rollback write can leave changes behind; rollback failures are included in the returned error.
- Rollback requires retained original text bytes. An existing non-UTF-8 file deleted without decodable original bytes cannot be restored by this path and is reported as an incomplete rollback.
- A compact post-edit oracle summary was not added. Existing parse-regression and LSP-diagnostics signals are separate rather than an isolated summary hook, so this remains optional follow-up work.
