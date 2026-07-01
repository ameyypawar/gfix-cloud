# ADR 0008: Real mined golden set, leave-one-out, per-regime honest framing

## Context

Evaluating a RAG system on synthetic or hand-written conflicts risks
measuring the eval author's assumptions rather than real retrieval/
generation behavior. It's also easy to produce a misleadingly high or low
single aggregate number that hides where RAG actually helps.

## Decision

`eval/golden_set.jsonl` contains 50 **real** (conflict → resolution) pairs
mined non-destructively from three OSS repos' merge history (gitbutler,
gitoxide, mio) via `git merge-tree --write-tree --name-only`, with ground
truth taken as the maintainers' actual merge commit content
(`git show <merge_sha>:<file>`) — not a model's or the author's guess at the
"right" resolution. `eval/run_eval.py` runs **leave-one-out** retrieval
(`exclude_id=<self>` so a row never retrieves its own answer) and reports
**per-regime** results, bucketing rows into `recurring` (file basename
appears ≥3 times in the golden set — e.g. `Cargo.toml`, `mod.rs`) versus
`one_off`, plus an `overall` rollup.

## Consequences

- Ground truth is externally verifiable (a real commit SHA in a real repo),
  not an eval author's judgment call — this is stated explicitly in
  `eval/README.md`'s provenance section.
- Per-regime reporting surfaces the actual, expected shape of a
  retrieval-over-history system: it should help far more on recurring
  conflict families (the corpus has real precedent) than on genuinely novel
  one-off conflicts. Reporting only an aggregate number would hide this and
  invite an apples-to-oranges "RAG doesn't help much" reading.
- Leave-one-out is required for retrieval eval to be meaningful at all —
  without excluding self, every row would trivially retrieve its own
  answer at rank 1.
- Precision is expected to **grow with corpus size**: at 50 rows, recurring
  families like `Cargo.toml` (21 entries) have real neighbors to find;
  most one-off families have zero same-family neighbors by construction.
  This is disclosed rather than presented as a ceiling.
