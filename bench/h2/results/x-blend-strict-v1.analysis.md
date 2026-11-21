# x-blend strict-v1 analysis

rows completed: 24 (excluded 0), arms diff/strict, tasks b_table,b_sign

| metric | diff | strict | strict-diff [95% CI] |
|---|---|---|---|
| success n | 7 | 7 | -0.0 [-0.333, 0.333] |
| score | 0.861 | 0.861 | -0.0 [-0.111, 0.111] |
| affected checks repaired | 0.833 | 0.792 | -0.042 [-0.25, 0.125] |
| agent calls | 5.58 | 6.0 | 0.417 [-1.167, 2.083] |
| bridge calls | 4.58 | 5.08 | 0.5 [-1.167, 2.167] |
| tokens (k) | 34.5 | 40.5 | 6.0 [-6.7, 18.9] |
| tool errors | 0.92 | 1.0 | None [None, None] |

strict traces: {'refusals': 9, 'runs_with_refusal': 9, 'acks': 24, 'teammate_edits': 9, 'runs': 12}

per-task repair:
{"b_sign": {"diff": 1.0, "strict": 0.833}, "b_table": {"diff": 0.667, "strict": 0.75}}

repair GIVEN teammate edit fired (fair comparison):
{"diff": {"runs_edited": 8, "repair_mean": 0.812, "fully_repaired": 5, "success": 4}, "strict": {"runs_edited": 9, "repair_mean": 0.833, "fully_repaired": 6, "success": 6}}
