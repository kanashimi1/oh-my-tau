#!/usr/bin/env bash
# Regenerate the bundled model-catalog data omp-rs embeds from a newer
# vendor-src checkout:
#   vendor/pa-ai/data/compat/rules.json   (TS compat rule tree, verbatim)
#   vendor/pa-ai/data/compat/axes.json    (AXES wire/request + API_COMPAT_RECORDS)
#   crates/omp-cli/assets/tui-model-catalog.json (slimmed catalog for the /model browser)
#
# The rules hash feeds the models.db `materialization_policy` (omp-providers
# build.rs), so re-run this after every vendor-src bump and rebuild.
#
# The TUI catalog is a UNION: providers/models the installed TS omp still
# ships (the current committed asset — including hand-seeded providers like
# factory-droid that are gone from vendor-src models.json) are kept; the
# vendor pass only adds/updates rows and a diff summary is printed. Rows for
# providers that no longer exist anywhere are reported, never dropped
# silently.
#
# Run: scripts/sync-catalog.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# 1. pa-ai compat data (rules.json verbatim + axes/API records).
cd "$ROOT/vendor-src/oh-my-pi"
bun "$ROOT/runs/omp-rust/compat-oracle/gen-data.ts"

# 2. Slimmed TUI catalog: TS `catalog/src/models.json` → one row per model:
#    [id, name, reasoning, vision, ctx, maxTokens, $in, $out, $cache, kind, priority]
#    merged over the committed asset (existing providers/rows win; the
#    committed asset may carry hand-seeded providers the vendor source dropped).
bun -e '
const [assetPath, srcPath] = process.argv.slice(1);
const asset = JSON.parse(await Bun.file(assetPath).text());
const src = JSON.parse(await Bun.file(srcPath).text());

const slim = m => [
  m.id,
  m.name ?? m.id,
  m.reasoning ? 1 : 0,
  (m.input ?? []).includes("image") ? 1 : 0,
  m.contextWindow ?? 0,
  m.maxTokens ?? 0,
  m.cost?.input ?? 0,
  m.cost?.output ?? 0,
  m.cost?.cacheRead ?? 0,
  m.kind ?? "",
  m.priority ?? null,
];
const rowEq = (a, b) => JSON.stringify(a) === JSON.stringify(b);

let added = 0, updated = 0, addedProviders = [];
const report = [];
for (const [provider, models] of Object.entries(src)) {
  const rows = Object.values(models).map(slim);
  if (!rows.length) continue;
  if (!asset[provider]) { addedProviders.push(provider); asset[provider] = rows; continue; }
  const byId = new Map(asset[provider].map(r => [r[0], r]));
  for (const row of rows) {
    const old = byId.get(row[0]);
    if (!old) { added++; asset[provider].push(row); }
    else if (!rowEq(old, row)) { updated++; Object.assign(old, row); }
  }
}
for (const provider of Object.keys(asset).sort()) {
  if (!src[provider]) report.push(`kept (not in vendor models.json): ${provider} (${asset[provider].length} rows)`);
}
for (const p of addedProviders) report.push(`added provider: ${p} (${asset[p].length} rows)`);
await Bun.write(assetPath, JSON.stringify(asset));
console.log(`tui catalog: +${added} rows, ~${updated} updated rows, ${addedProviders.length} new providers`);
for (const line of report) console.log("  " + line);
' "$ROOT/crates/omp-cli/assets/tui-model-catalog.json" \
  "$ROOT/vendor-src/oh-my-pi/packages/catalog/src/models.json"

echo "synced:"
echo "  vendor/pa-ai/data/compat/{rules,axes}.json"
echo "  crates/omp-cli/assets/tui-model-catalog.json (union with committed asset)"
