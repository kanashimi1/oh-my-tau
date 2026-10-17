Single-file string replacement; fuzzy whitespace matching.

<instruction>
- MUST use the smallest `old_string` uniquely identifying the change.
- Nonunique `old_string` → MUST add context, or `replace_all: true` to change every occurrence (also for renames across the file).
- SHOULD edit existing files, not create new ones.
</instruction>

<output>
Success: file modified in place. Failure (e.g. `old_string` absent, or multiple matches without `replace_all`): error describes the issue.
</output>

<critical>
MUST read the file at least once in the conversation before editing; otherwise the tool errors.
</critical>

<bash-alternatives>
Replace is content-addressed (locate by text). Pattern-addressed bulk changes: bash is cheaper, e.g. `sed -E -i 's/pattern/replacement/g' a.ts b.ts`. Structure-aware codemods: `ast_edit`. In-place edits: prefer Replace or `write` (diff preview, fuzzy matching).
</bash-alternatives>
