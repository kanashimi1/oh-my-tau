<rlm>
**{{mode}}** adds the RLM toolkit to the workflow:
- Inputs over ~100k chars (logs, dumps, whole transcripts, big files): do NOT spawn agents to read them. Load the text in the kernel and read it with `llm_query(prompt, context)` and the λ-RLM combinators: `lam_rlm(P, query, task="auto")` (tasks: search | classify | aggregate | summarise | multi_hop) or hand-built plans from `lam_split`, `lam_peek`, `lam_map`, `lam_filter`, `lam_reduce`, `lam_concat`, `lam_cross` (JS: same names, `await` the calls; `llm_query(prompt, { rubric, verify, n })`).
- SERA: before creating the pool, write the rubric (`RUBRIC = [{"criterion": ..., "weight": ...}]`) and freeze it — never edit it after the first result. Score every result against it with `llm_query(prompt, context, rubric=RUBRIC, verify=fn)` (frozen host-side before the child runs; the answer carries `.score` / `.record`; `n>1` keeps the best candidate) and rank verified outcomes with `rubric_rank(records)`.
- Limits per run: depth 2, 64 model calls, 2M tokens (`OMP_RLM_MAX_*`). A refusal `lam: plan needs up to N model calls … only M remain` means raise the call budget with `/rlm --max-calls=N`, not the depth.
</rlm>
