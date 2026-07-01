# ADR 0004: mergiraf bundled in the image, subprocess-only, never linked

## Context

[mergiraf](https://codeberg.org/mergiraf/mergiraf) provides a syntax-aware,
AST-based merge that resolves a meaningfully wider class of conflicts than
plain git automerge — gfix shells out to it as part of its deterministic
resolution floor. mergiraf is licensed **GPL-3.0**. gfix-cloud is MIT and is
a public portfolio repo; the license boundary between an MIT project and a
bundled GPL-3.0 binary needs to be explicit and correct.

## Decision

Fetch the mergiraf binary at image build time (`api/Dockerfile`,
architecture-matched via `TARGETARCH`) and invoke it exclusively as an
**unmodified subprocess** through gfix's own `gitfix_conflict_resolve
{kind: mergiraf}` tool call. gfix-cloud never links against mergiraf as a
library, never vendors or modifies its source, and never redistributes
mergiraf outside the container image alongside its own MIT-licensed code as
a combined work.

## Consequences

- Subprocess invocation of a GPL-3.0 binary does not create a derivative
  work under the GPL — the license boundary is the process boundary, not
  the container. This is the same boundary git itself relies on for
  external diff/merge tools.
- Attribution and source location (codeberg.org/mergiraf/mergiraf) are
  recorded in `NOTICE`, not silently omitted.
- gfix-cloud gets a richer *keyless* floor: conflicts mergiraf can resolve
  never reach the LLM path at all, which matters for the honest "how much
  of this needs AI" framing in the README and eval.
- If mergiraf's binary is ever unavailable at runtime (missing in a stage,
  wrong architecture), the bridge's `except (McpError, ValueError)` fallback
  routes straight to RAG-augmented generation — degradation is graceful, not
  a hard failure.
