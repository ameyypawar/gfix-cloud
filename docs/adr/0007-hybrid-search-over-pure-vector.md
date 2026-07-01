# ADR 0007: Hybrid search (vector + BM25, RRF) over pure vector

## Context

Merge conflicts frequently hinge on exact identifiers — a dependency name in
`Cargo.toml`, a specific function or field name, a specific version string.
Dense embedding similarity captures semantic/structural resemblance well but
can under-rank a neighbor that shares the *exact* identifier at the center
of the conflict but differs in surrounding context. Pure BM25/keyword search
has the opposite failure mode: it misses semantically similar conflicts that
don't share vocabulary.

## Decision

Retrieval fuses two arms in one SQL CTE (`retrieval.py`): an HNSW
inner-product vector arm (`<#>`) and a BM25 arm (`ts_rank` over
`to_tsvector('english', conflict_text)` via `plainto_tsquery`), combined
with **Reciprocal Rank Fusion** (`k=60`, weights ~0.6 vector / 0.4 BM25).
Both arms apply the same language hard-filter before fusion.

## Consequences

- Exact-identifier recall (a renamed dependency, a specific config key)
  benefits from the BM25 arm even when the surrounding diff context differs
  enough to weaken pure vector similarity.
- RRF avoids needing to calibrate raw score scales between cosine/IP
  similarity and BM25's `ts_rank` — fusion operates on rank position, not
  raw scores, which is robust to the two arms having incomparable score
  distributions.
- Extra SQL complexity (two CTEs + a fusion `GROUP BY`) versus a single
  `ORDER BY embedding <#> $1 LIMIT k` query — accepted because retrieval
  quality is the entire point of the RAG layer being evaluated.
- The language hard-filter runs on *both* arms before fusion, so a
  Rust conflict never retrieves a Python neighbor regardless of either
  arm's score.
