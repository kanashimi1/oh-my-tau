<clm>
**{{mode}}** runs a lazy managed context: below the host pressure threshold, keep the raw append history and do not perform routine `context_edit` maintenance. When the host asks for maintenance, use ONE batched edit: durable exact facts (paths, decisions, open items, ids) go in stable slots; transient state goes in `scratch`/`scratch_*`. Stable edits consolidate old history; scratch updates leave that older history cache-stable. Keep slots short and never copy conversational filler. You may update `scratch` on demand for facts that are expensive to recompute.
</clm>
