## invariant
- Inspect output of each non-trivial text-processing pipeline (awk/sed splits, regexes) before building on it or writing to the deliverable; visibly malformed intermediates (e.g., empty fields) mean the extraction logic is wrong. NEVER overwrite the target file with an unverified draft.

## domain:data
- Emit exactly the specified value format: follow the spec's example/field definitions literally; NEVER add unrequested decoration (e.g., '+' on numeric strings).
- Ambiguous sign/format: resolve against the given field definition, don't invent.
- When a spec allows int, float, or str money amounts, convert every input through one exact path (e.g., Decimal(str(value))) at the entry point; NEVER let floats reach arithmetic before conversion.
- Before counting records, sample the actual log format and count on the precise record pattern (e.g., 'level=ERROR'), not a loose substring like bare 'ERROR' that can match message text; derive case-sensitivity from observed level values in the data.