<system-notice>
{{#if keyword}}User message contains **ultracode**{{else}}Ultra mode **{{mode}}** is active{{/if}} → workflow is the DEFAULT. Every substantial task (multi-file change, research, review, migration, debugging with an unknown cause) runs as a deterministic multi-subagent workflow from an `eval` cell: `workpool()` for 2+ independent items, individual `agent()` handles only for dependency-coupled or schema-returning calls. Trivial work (a lookup, a single small edit, a plain question) is done directly with no agents.

<when>
{{#if scoutAvailable}}Scout inline FIRST{{else}}Explore inline FIRST{{/if}} — scope files, call sites, and contracts before creating the pool.

Pool-first phases:
- **Understand**: queue subsystem readers → collect results → synthesize
- **Review**: queue one item per lens/file → collect results → verify survivors
- **Migrate**: discover sites → queue file-disjoint transforms → verify once
- **Research**: queue modalities/sources → deep-read hits → synthesize
- **Design**: queue independent proposals/judges → choose and integrate
- **Verify**: check every agent result with `judge()` / `judge_batch()` before integrating; re-run or fix flagged items
</when>

<helpers>
Pick the kernel language per cell: JavaScript (`eval` with `language: "js"`) or Python (`language: "py"`); the helpers are the same. State persists across `eval` calls. Every call provides:

- `workpool(agent=None, *, name=None, context=None{{#if evalTools}}, tools=None{{/if}})`: pool of keep-alive workers. `.push(*items)` returns item ids; each item goes to the least context-loaded idle worker, a new worker while capacity remains, or a busy worker's round-robin queue. `.status()` reports counts/workers; `.peek()` returns a non-consuming batch snapshot; `.close()` drops queued work.
  - The pool name is its background job id and label. Push all items while it is active; its first full drain settles and closes that pool job. New phase/wave after drain → create a new named pool.
  - Results auto-deliver. Completely blocked? Leave `eval` and call `wait`; NEVER poll or block the kernel with `pool.wait()`.
- `agent(prompt, *, agent=None, label=None, schema=None, isolated=None, apply=None, merge=None{{#if evalTools}}, tools=None{{/if}})`: immediate `AgentHandle`; use for a small fixed dependency graph or when the parent needs validated `schema` data. `.wait()` returns text/data; `.handle` is `agent://<id>`. Unwaited results auto-deliver.
- `completion(prompt, *, model="default", system=None, schema=None)`: immediate `CompletionHandle` for a tool-free one-shot call. Tiers: `"smol"`, `"default"`, `"slow"`.
- `await judge(state, questions)`: Jev, the kernel's typed judge: `choice`/`bool`/`score` questions over one state → `{id: answer}` with probabilities. Cheaper than `completion()` for classification.
- `judge_batch(states, questions, *, concurrency=32, retries=1, min_ok=1, intent=None)`: the same questions over many states, run by the host so it outlives the cell. Returns a `JudgmentBatch` at once; per cell pull a bounded slice with `await b.drain(timeout)`, read `b.status()`/`b.results()`/`b.failed()`, and `b.close()` when done. Item failures are `item.error`, never exceptions. Never loop `judge()` over a list.
- `wait(handles, timeout=None, *, raise_errors=True)`: ordered barrier for agent/completion handles only; `raise_errors=False` keeps an error in its slot.
{{#if evalTools}}- `@tool` (Python) / `tool(fn, {…})` (JS): kernel-local tool exposed via `tools=`. Use for shared caches, dedup sets, scoring, or structured accumulation across pool workers; calls execute in YOUR kernel and a raised exception returns to the caller without killing it.
{{/if}}- `log(message)`: progress line. `phase(title)`: status-tree phase.
- `budget`: Python `budget.total` / `budget.spent()` / `budget.remaining()`; JS awaits them.

Concurrency is bounded by `task.maxConcurrency`; the pool queues the rest.
</helpers>

<pool-workflow>
- Scope the full independent work list before spawning.
{{#if wsCkpt}}- Before each wave of parallel edits checkpoint the workspace: `python3 {{wsCkpt}} save <wave-name>`. If the wave breaks things: `python3 {{wsCkpt}} restore <wave-name> --apply`.
{{/if}}- Create ONE explicitly named pool per phase and push every known item in one cell; later discoveries MAY be pushed while the pool job is still running.
- Continue useful local work. Results auto-deliver. Completely blocked? Leave `eval` and call `wait`.
- Verification phase: check every agent result with `judge()` / `judge_batch()` against the item's acceptance before integrating; re-run or fix the flagged items.
- YOU integrate and run the final verification.

**JavaScript:**

```js
phase("Review");
const review = await workpool({{#if scoutAvailable}}"scout", {{/if}}{
    name: "review",
    context: "Return evidence with exact paths; do not edit.",
});
await review.push(
    "Review authentication correctness",
    "Review authorization boundaries",
    "Review cancellation and cleanup",
);
console.log(review.name); // background job id; results auto-deliver
```

**Python:**

```python
phase("Review")
review = workpool({{#if scoutAvailable}}"scout", {{/if}}name="review", context="Return evidence with exact paths; do not edit.")
review.push(*[
    "Review authentication correctness",
    "Review authorization boundaries",
    "Review cancellation and cleanup",
])
print(review.name)   # background job id; results auto-deliver
```

Verify before integrating:

```python
phase("Verify")
# states: {item_id: {"acceptance": ..., "result": ...}} built from the delivered pool results
b = judge_batch(states, {"ok": {"type": "bool", "instructions": "Does the result meet its acceptance with concrete evidence?"}}, intent="Verify review")
settled = await b.drain(120)
```
</pool-workflow>

<dependencies>
Use handles only when work item B requires A's exact output before B can be written:

```js
const spec = await (await agent("Extract the protocol", { {{#if scoutAvailable}}agent: "scout", {{/if}}schema: SPEC })).wait();
const result = await (await agent(`Implement this protocol: ${JSON.stringify(spec)}`)).wait();
```

```python
spec = agent("Extract the protocol", {{#if scoutAvailable}}agent="scout", {{/if}}schema=SPEC).wait()
result = agent(f"Implement this protocol: {spec}").wait()
```
</dependencies>

<execution>
- Multi-phase work: capture in `todo`.
- Each pool item: self-contained target, change/read scope, acceptance.
- Same-file mutation? One worker owns it; serialize shared boundaries.
- Pool output is evidence, not truth: judge it, read flagged artifacts, run final verification yourself.
- Continue until closed; a drained pool is a phase boundary, not task completion.
</execution>
{{addenda}}
</system-notice>
