# gfix-cloud

A service that wraps the [gfix](https://github.com/ameyypawar/gfix) merge-conflict engine behind an API, retrieves similar past resolutions from a pgvector store to augment AI conflict suggestions, and exposes a Next.js dashboard for interactive use and eval results. The stack is FastAPI, Postgres/pgvector, sentence-transformers (local, keyless), the MCP Python SDK (to talk to `gfix mcp`), Anthropic's API (optional; enables generation), and Docker for one-command local setup. Each resolved conflict is embedded and stored, so the corpus grows with use.

## Architecture

_Diagram in Phase 6._

```
POST /resolve {base,ours,theirs,file_path,rag}
  1  materialize scratch git repo in /tmp
  2  spawn gfix mcp (MCP stdio) → merge_preview → conflict_get
  3  embed conflict (bge-small-en-v1.5, 384-dim, in-process)
  4  hybrid retrieve top-k (HNSW <#> + BM25 tsvector, RRF)
  5  few-shot → claude-haiku-4-5 (Anthropic SDK)
  6  gfix conflict_resolve {Manual} → merge_apply → audit ref
  7  persist (conflict, resolution, embedding) → Neon/pgvector
```

## Eval

_RAG-vs-no-RAG numbers pinned here in Phase 4._

## Quickstart

```bash
docker compose up   # Phase 6
```

## Stack

- **FastAPI** — async Python API; direct asyncio subprocess for gfix; no framework overhead
- **pgvector** — HNSW index on 384-dim embeddings; inner-product ops on normalized vectors
- **sentence-transformers / bge-small-en-v1.5** — local, keyless, in-process embeddings
- **MCP Python SDK** — structured stdio bridge to `gfix mcp`; typed JSON-RPC, no hand-rolled protocol
- **anthropic SDK / claude-haiku-4-5** — generation LLM; optional (deterministic floor works without it)
- **asyncpg** — async Postgres driver; no ORM overhead
- **Next.js (App Router)** — dashboard; deployed to Vercel; reads from `/resolve` and `/similar`
- **Docker / Fly / Neon** — 3-stage image; scale-to-0 on Fly; Neon free-tier Postgres (pgvector included)

## ADRs

_Index in `docs/adr/` — Phase 6._

| # | Decision |
|---|---|
| 1 | HNSW over IVFFlat (empty-then-grows; IVFFlat needs data at create time) |
| 2 | Inner-product over cosine (normalized vectors; opclass must match operator) |
| 3 | Official `mcp` SDK as MCP client (protocol lib; typed; shows MCP fluency) |
| 4 | mergiraf runtime-optional (GPL-3.0 subprocess; gfix degrades gracefully) |
| 5 | dim=384 locked (model change requires migration) |
| 6 | claude-haiku-4-5 for generation (fits 60s proxy timeout; no thinking overhead) |
| 7 | Monorepo (api/ web/ eval/ docs/ share one deploy context) |
| 8 | gfix binary from ameyypawar/gfix (verified release asset) |
| 9 | Keyless embeddings (sentence-transformers local; only generation needs a key) |
| 10 | Sync /resolve (Haiku fits 60s; poll path documented for Ollama/batch) |
