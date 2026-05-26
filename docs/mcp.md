# MCP Tools

The researcher reaches external capabilities exclusively through **MCP servers**,
never through direct API wrappers. Each server is a separate process spoken to
over stdio; `mcp_client.py` manages the sessions and exposes a single
`call_tool(server, tool, args)`.

```
researcher ──> MCPResearchTools ──> MCPClient ──stdio──> web_search server ──> Tavily
                                              └─stdio──> filesystem server ─> ./workspace
```

## Servers

| Server | Module | Tools |
|---|---|---|
| `web_search` | `research_assistant.mcp_servers.web_search` | `search(query, max_results)` |
| `filesystem` | `research_assistant.mcp_servers.filesystem` | `write_file`, `read_file`, `list_dir` |

Run one standalone for inspection: `just mcp-web` / `just mcp-fs` (e.g. under the
[MCP Inspector](https://github.com/modelcontextprotocol/inspector)).

## Capability scoping

**Filesystem — sandboxed to `RESEARCH_WORKSPACE` (`./workspace/`).** Every path is
resolved and verified to live inside the workspace root; absolute paths, `..`
traversal, and symlink escapes resolve outside the root and are rejected with an
error. The server only ever reads/writes within that one directory, so a
misbehaving (or prompt-injected) agent cannot touch the wider disk.

**Web search — read-only and bounded.** The `search` tool performs outbound
read-only queries only. `max_results` is clamped server-side (`MAX_RESULTS_CAP`),
so a single research step cannot fan out into an unbounded number of backend
calls, and the researcher itself caps queries-per-step. Tavily's own account-level
rate limits apply on top.

## Citations are retrieval-grounded

The researcher passes the LLM a *numbered* list of retrieved sources and the model
returns the indices it used; code maps those indices back to real `Citation`
objects. A cited URL is therefore always one the search tool actually returned —
the model cannot invent sources.
