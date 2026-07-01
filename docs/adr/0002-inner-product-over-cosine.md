# ADR 0002: Inner-product over cosine distance

## Context

pgvector supports L2, cosine, and inner-product (`<#>`) distance operators,
each requiring a matching index opclass (`vector_l2_ops`, `vector_cosine_ops`,
`vector_ip_ops`). Embeddings come from `sentence-transformers` with
`normalize_embeddings=True` — every vector is already L2-normalized before
it is written to `pgvector`.

## Decision

Use the inner-product operator (`<#>`) with the `vector_ip_ops` opclass on
the HNSW index, not cosine distance.

## Consequences

- For L2-normalized vectors, inner product and cosine similarity produce the
  same ranking — inner product is cheaper to compute (no norm division at
  query time), so there is no accuracy cost to preferring it.
- The index opclass (`vector_ip_ops`) and the query operator (`<#>`) must
  match, or pgvector will refuse to use the index for that query. This is
  enforced consistently across `retrieval.py` (hybrid search), `persistence.py`
  (`find_similar`), and the migration's `CREATE INDEX`.
- `<#>` returns *negative* inner product, so smaller (more negative) values
  mean more similar — `ORDER BY embedding <#> $1 ASC` gives nearest-first.
  This is a one-time gotcha documented at each call site, not a runtime cost.
- If a future embedding model does not normalize its output, this decision
  must be revisited (inner product without normalization is not a valid
  similarity metric).
