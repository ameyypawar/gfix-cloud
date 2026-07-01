# ADR 0006: Generation via Gemini `gemini-2.5-flash`, provider-pluggable

## Context

RAG-augmented resolution needs a generation LLM for conflicts the
deterministic floor (git automerge / mergiraf) can't resolve. The available
key for this project is a Gemini key, not an Anthropic key. gfix itself has
its own BYOK convention (raw HTTP against whichever provider key is
configured: Anthropic → OpenAI → Ollama precedence) — gfix-cloud's own
generation call is a separate, service-owned step (see
`docs/architecture.md`), not a call through gfix.

## Decision

Default generation provider is **Gemini `gemini-2.5-flash`**, called via a
raw `httpx` POST to the `v1beta generateContent` endpoint — no Google SDK
dependency. The provider is selected by `GENERATION_PROVIDER` (default
`gemini`); `anthropic` (via the `anthropic` SDK, `AsyncAnthropic`) is coded
as an alternate path in `generation.py`. `ollama` is referenced as a future
coded alternate but not implemented in this phase.

## Consequences

- Raw `httpx` avoids adding a Google Generative AI SDK dependency for a
  single endpoint call, and mirrors gfix's own raw-HTTP BYOK pattern —
  consistent style across the stack.
- Switching providers is a config change (`GENERATION_PROVIDER`,
  `GENERATION_MODEL`, and the matching `*_API_KEY`), not a code change, for
  the two implemented paths (gemini, anthropic).
- The system prompt fences all untrusted content (conflict bodies, few-shot
  example code) with unique delimiters and instructs the model never to
  treat fenced content as instructions — a prompt-injection guard, since
  conflict text originates from arbitrary source repositories.
- No key configured → `generate_resolution` raises a `RuntimeError` the
  bridge catches and turns into a keyless-graceful partial response
  (`ai_unavailable=true`), not a 500.
