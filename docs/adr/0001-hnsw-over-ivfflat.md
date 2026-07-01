# ADR 0001: HNSW over IVFFlat for the vector index

## Context

`past_resolutions.embedding` needs an approximate-nearest-neighbor index for
the retrieval arm of hybrid search. pgvector offers two index types: IVFFlat
and HNSW. IVFFlat's clustering (`lists`) is built from the data present at
`CREATE INDEX` time — it needs a representative sample to produce good
cluster centroids. gfix-cloud's corpus starts empty (a fresh `docker compose
up` has zero rows) and grows one resolution at a time as `/resolve` is
called and the demo seed loads.

## Decision

Use an HNSW index (`vector_ip_ops`) on `embedding`, built with the
migration (`m=16`, `ef_construction=128`), not IVFFlat.

## Consequences

- HNSW builds and refines incrementally — it performs reasonably even on an
  empty-then-growing table, unlike IVFFlat, which produces poor clusters
  (or requires a rebuild) when trained on too little data.
- Query-time recall/latency is tunable via `hnsw.ef_search` (set per-session
  from `HNSW_EF_SEARCH`, default 40) without rebuilding the index.
- HNSW index build/insert cost is higher than IVFFlat per-row, which is
  acceptable at this corpus size (tens to low thousands of rows in the
  portfolio/demo context).
