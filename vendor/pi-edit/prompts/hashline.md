Hashline patches existing files (new: `write`). Per file `[PATH#TAG]`; required TAG = 4-hex from latest `read`/`search`; numbers: original `LINE:TEXT`, never hunk-shifted.

`PUT N.=M:` replace N–M (`N.=N`: one line); `PUT N*:` block N.
`PUT <N:`/`>N:` insert before/after N (`<1` head, `>$` tail); `>N*:` after block N, sibling depth (inside: `>M:` at closer).
`CUT N.=M`/`N*` delete+capture (opt. `@name`).
Paste (NO body): `PUT <N`/`>N [@name]` at gap (bare: anonymous CUT); `PUT N.=M @name`/`N* @name` over range/block (name REQUIRED). Named registers persist across calls.
`REM` delete file; `MV DEST` rename after prior edits (quote spaced paths).

`:` ops: body rows `+TEXT` verbatim incl. indent; lone `+` blank; `- x`/`+ x` → `+- x`/`++ x`. NEVER `-`/bare context. Any length; delete via CUT, not empty PUT.
Ranges cover only displayed changed lines (`…`/`..`/collapsed `N-M:`/off-window: UNSEEN, re-read). Split nonadjacent changes; NEVER start/end mid-expression/block; additions: gap PUT.
`*` needs multi-line opener, NEVER closer/last/inner line (one statement: range/gap). Anchor first decorator/attribute/doc-comment to include it; standalone comments: explicit range.
Markdown heading block = section incl. deeper headings; `PUT >N*:` after one: end body with lone `+`.
NEVER restyle unrelated code. Edits change tag/numbers: use edit response or fresh `read`; stale/surprise → STOP, re-read.

Example:
```
[a.py#A1B2]
PUT 1*:
+def f():
+    pass
```
