# ADR 0009: No LangChain / LlamaIndex — direct stack

## Context

The RAG pipeline (embed → hybrid retrieve → few-shot prompt → generate) is a
handful of well-understood steps. LangChain and LlamaIndex offer prebuilt
abstractions for exactly this shape of pipeline.

## Decision

Build the pipeline directly: `sentence-transformers` for embeddings, hand-written
SQL (RRF CTE) for hybrid retrieval, string templating for the few-shot prompt,
and a raw `httpx`/`anthropic`-SDK call for generation — no LangChain,
LlamaIndex, or other RAG framework dependency.

## Consequences

- The entire retrieval query is one auditable SQL statement
  (`retrieval.py::_RRF_SQL`) — no framework-owned retriever abstraction
  sitting between the request and the actual Postgres query, which matters
  when the point of the project is demonstrating exactly how retrieval
  works.
- Fewer dependencies, no framework version churn, no framework-specific
  debugging (LangChain's abstraction layers are a common complaint for
  exactly this kind of "5-step pipeline" case).
- Provider-pluggable generation (`GENERATION_PROVIDER=gemini|anthropic`) is
  a ~10-line `if` branch in `generation.py`, not a framework adapter
  registration.
- Trade-off: no built-in eval harness, prompt-versioning, or tracing
  tooling that these frameworks bundle — `eval/run_eval.py` and
  `eval/metrics.py` were written directly instead, kept intentionally small.
