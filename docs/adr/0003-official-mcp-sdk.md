# ADR 0003: Official `mcp` Python SDK for the gfix bridge

## Context

`gfix mcp` speaks the Model Context Protocol over stdio (JSON-RPC framed
messages, capability negotiation, typed tool calls). The bridge
(`api/app/gfix_bridge.py`) needs to spawn it, initialize a session, and call
tools like `gitfix_merge_preview`, `gitfix_conflict_get`,
`gitfix_conflict_resolve`, and `gitfix_merge_apply`.

The bridge could hand-roll the JSON-RPC framing and process management, or
depend on the official `mcp` Python SDK (`ClientSession`, `stdio_client`,
`StdioServerParameters`).

## Decision

Use the official `mcp` Python SDK as the client library, not a hand-rolled
protocol implementation.

## Consequences

- Protocol correctness (initialization handshake, message framing, error
  types like `McpError`) is delegated to a maintained library instead of
  reimplemented and re-debugged in this service.
- The SDK is a thin protocol client, not an agent framework — it does not
  impose orchestration patterns on top of gfix-cloud's own bridge logic in
  `gfix_bridge.py` and `rag.py`.
- Using the canonical SDK for an MCP-native engine is itself a signal of MCP
  literacy: gfix-cloud demonstrates driving a real MCP server end-to-end
  (`initialize` → `list_tools` → `call_tool` sequences), not just calling a
  REST-shaped wrapper.
- Coupled to the SDK's async API surface (`async with stdio_client(...)`
  context managers) throughout the bridge.
