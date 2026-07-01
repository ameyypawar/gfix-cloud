# ADR 0005: Local sentence-transformers bge-small embeddings, dim locked

## Context

Retrieval needs a conflict → vector representation. Options were a hosted
embedding API (OpenAI, Voyage, Gemini embeddings) or a local model run
in-process. gfix-cloud is meant to work fully **keyless** — `docker compose
up` with no API keys should still produce useful retrieval, since retrieval
is independent of the (optional) generation key.

## Decision

Use `sentence-transformers` with `BAAI/bge-small-en-v1.5` (384-dim),
loaded once as a module-level singleton (`embeddings.py`) and run
in-process. The model is pre-downloaded at Docker build time
(`api/Dockerfile` runs a one-line `SentenceTransformer(...)` load during
the build) so the runtime container never needs network access to fetch it.

## Consequences

- Retrieval works with zero API keys and zero external network calls at
  runtime — only the generation step (Gemini) needs a key. This is the
  basis for the "keyless-graceful" behavior documented in
  `docs/architecture.md`.
- 384 is now a load-bearing constant: `EMBED_DIM=384`, the `vector(384)`
  column type, and the HNSW index are all coupled to this specific model.
  Changing the embedding model requires a migration (new column/index, and
  re-embedding every existing row) — this is not a config toggle.
- Always the *conflict* is embedded (`build_conflict_text`: file path,
  language, kind, base/ours/theirs), never the resolution — retrieval
  matches on "what problem looked like this," not "what answer looked like
  this."
- Loading the model adds real memory and cold-start cost (`torch` +
  `sentence-transformers`), pre-warmed once in FastAPI's `lifespan` so the
  first `/resolve` isn't slow. Image size implications are tracked
  separately (see the image-size note).
