Patches files with diff hunks. Primary tool for editing existing files.

<instruction>
Hunk headers:
- `@@`: bare, when context lines are unique
- `@@ $ANCHOR`: anchor copied verbatim from the file (full line or unique substring)
Anchors: prefer bare `@@`; otherwise pick a highly specific one: full function signature, class declaration, unique string literal/error message, uncommon config key. On "Found multiple matches": add context lines, split into hunks with separate anchors, or lengthen the anchor.
Context: enough ` `-prefixed lines to make the match unique (usually 2–8). In structured blocks (nested braces, tags, indented regions) include the opening and closing lines so the edit stays inside the block.
</instruction>

<parameters>
```ts
// Input: { path: string, edits: Entry[] }; required `path` applies to every entry.
type Entry =
   // One or more hunks for `path`; each starts with "@@" (anchor optional),
   // has only ' ' | '+' | '-' lines, and at least one + or - line.
   | { op: "update", diff: string }
   // Full file content, no prefixes.
   | { op: "create", diff: string }
   | { op: "delete" }
   // Update and move `path` to `rename`.
   | { op: "update", rename: string, diff: string }
```
</parameters>

<output>
Success or failure; errors:
- "Found multiple matches": anchor/context not unique
- "No match found": context not in file (wrong content or stale read)
- diff syntax errors
</output>

<critical>
- MUST read the target file before editing
- MUST copy anchors and context lines verbatim (including whitespace)
- NEVER use anchors as comments (line numbers, location labels, placeholders like `@@ @@`)
- NEVER place new lines outside the intended block
- If an edit fails or breaks structure, MUST re-read the file and build a new patch from current content; NEVER retry the same diff
- NEVER edit to fix indentation, whitespace or formatting; run the formatter once at the end (`bun fmt`, `cargo fmt`, `prettier --write`, …) and leave inconsistent indentation for it
</critical>

<avoid>
- Generic anchors: `import`, `export`, `describe`, `function`, `const`
- The same addition in multiple hunks (duplicate blocks)
- Full-file overwrites for minor changes (fine for major restructures or short files)
</avoid>
