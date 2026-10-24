Anchored edit patch: Find quotes existing text; Replace replaces it; Insert Before/After add lines, keeping it.

<ops>
- `*** Edit File: path` opens a file; bare `*** Edit File:` continues it. Repeat for more files; all edits apply atomically. Append ` all` to either opener to change every match. JSON-quote ambiguous paths.
- `*** Find` MUST match once unless `all`. Copy exact text/indentation from the latest verbatim file read, not Markdown, diffs or summaries. Use the smallest unique anchor; if ambiguous add parent context, NEVER retry the bare line.
- Follow Find with `*** Replace` (complete final text; empty deletes), `*** Insert Before` (before the first matched line) or `*** Insert After` (after the last matched line). Replacement MUST quote part of the changed line; insertion MUST quote an adjacent anchor and contain new lines. Changing an anchor requires Replace. Omitting an action does not delete.
- Headers MUST stand alone; bodies are raw lines until the next header or EOF (no closing delimiter or `*** End …` line). NEVER use diff prefixes `+`/`-`/space or `@@` hunks. Edits address the original file, not positions shifted by earlier edits.
- Find `…` captures omitted text: mid-line gaps stay on that line; line-end gaps span lines. Each Replace `…` re-emits the next capture. A whole `…` line without a capture errors: type those lines out. Insert `…` is literal. Avoid retyping unchanged lines: use Insert or captures.
- Replace/Insert text is verbatim: indentation, whitespace, operators, delimiters are NEVER repaired. A failure applies nothing: resend the complete corrected payload. "No change" means Replace already equals the file text; look elsewhere. File contains a standalone edit header? Use `write`.
</ops>

<example>
Move code (empty Replace deletes; insert ahead of an unchanged anchor):
```text
*** Edit File: src/util.ts
*** Find
const helper = () => 1;
*** Replace
*** Find
run(target);
*** Insert Before
const helper = () => 1;
```

Keep skipped lines and part of a line:
```text
*** Edit File: src/users.ts
*** Find
function load(…){
…
return old(…);
*** Replace
function load(…){
…
return fresh(…);
```
</example>
