"""Wrapper letting agents call MCP tools without reimplementing the protocol.

``MCPClient`` connects to one or more MCP servers over stdio (each spawned as a
``python -m research_assistant.mcp_servers.<name>`` subprocess), manages their
sessions with a single ``AsyncExitStack``, and exposes one ``call_tool`` entry
point. ``MCPResearchTools`` is the thin, researcher-facing convenience over that
(``search`` / ``write_file``), so the agent code stays oblivious to MCP wiring.

Lifecycle: open ``build_research_tools`` once per research run (it owns the
subprocesses) and inject the yielded tools into the researcher.
"""

from __future__ import annotations

import json
import sys
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any, AsyncIterator, Protocol, runtime_checkable

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import get_default_environment, stdio_client
from mcp.types import CallToolResult, TextContent

from research_assistant.config import Settings, get_settings
from research_assistant.errors import MCPToolError, RetrievalError
from research_assistant.messages import SearchResult
from research_assistant.observability import get_logger

_log = get_logger("research_assistant.mcp")

#: Logical server name -> module run over stdio.
SERVER_MODULES: dict[str, str] = {
    "web_search": "research_assistant.mcp_servers.web_search",
    "filesystem": "research_assistant.mcp_servers.filesystem",
}


def _extract_text(result: CallToolResult) -> str:
    return "".join(c.text for c in result.content if isinstance(c, TextContent))


def _parse(result: CallToolResult) -> Any:
    if result.isError:
        raise MCPToolError(_extract_text(result) or "unknown MCP tool error")
    if result.structuredContent is not None:
        return result.structuredContent
    text = _extract_text(result)
    return json.loads(text) if text else None


class MCPClient:
    """Connects to a set of stdio MCP servers and dispatches tool calls."""

    def __init__(
        self, servers: dict[str, str], env: dict[str, str] | None = None
    ) -> None:
        self._servers = servers
        self._env = env or get_default_environment()
        self._stack: AsyncExitStack | None = None
        self.sessions: dict[str, ClientSession] = {}

    async def __aenter__(self) -> "MCPClient":
        self._stack = AsyncExitStack()
        for name, module in self._servers.items():
            params = StdioServerParameters(
                command=sys.executable,
                args=["-m", module],
                env=self._env,
            )
            read, write = await self._stack.enter_async_context(stdio_client(params))
            session = await self._stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self.sessions[name] = session
            _log.info("MCP server connected: %s", name)
        return self

    async def __aexit__(self, *exc: object) -> None:
        assert self._stack is not None
        await self._stack.aclose()
        self.sessions.clear()

    async def call_tool(
        self, server: str, tool: str, arguments: dict[str, Any]
    ) -> Any:
        if server not in self.sessions:
            raise MCPToolError(f"MCP server not connected: {server!r}")
        result = await self.sessions[server].call_tool(tool, arguments)
        return _parse(result)


@runtime_checkable
class ResearchTools(Protocol):
    """The tool surface the researcher depends on (injectable for tests)."""

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]: ...

    async def write_file(self, path: str, content: str) -> None: ...


class MCPResearchTools:
    """MCP-backed implementation of ``ResearchTools``."""

    def __init__(self, client: MCPClient, max_results: int = 5) -> None:
        self._client = client
        self._max_results = max_results

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        try:
            data = await self._client.call_tool(
                "web_search", "search", {"query": query, "max_results": max_results}
            )
        except MCPToolError as e:
            raise RetrievalError(f"web search failed for {query!r}: {e}") from e
        return [SearchResult.model_validate(r) for r in (data or {}).get("results", [])]

    async def write_file(self, path: str, content: str) -> None:
        # Best-effort: intermediate persistence must never fail the research run.
        try:
            await self._client.call_tool(
                "filesystem", "write_file", {"path": path, "content": content}
            )
        except MCPToolError as e:
            _log.warning("write_file(%s) failed, continuing: %s", path, e)


def _server_env(settings: Settings, overrides: dict[str, str] | None) -> dict[str, str]:
    env = dict(get_default_environment())
    # Only inject keys when present, so we never override .env with empty strings.
    if settings.tavily_api_key:
        env["TAVILY_API_KEY"] = settings.tavily_api_key
    env["RESEARCH_WORKSPACE"] = settings.research_workspace
    if overrides:
        env.update(overrides)
    return env


@asynccontextmanager
async def build_research_tools(
    settings: Settings | None = None, *, env_overrides: dict[str, str] | None = None
) -> AsyncIterator[MCPResearchTools]:
    """Open the web-search + filesystem MCP servers and yield researcher tools."""
    settings = settings or get_settings()
    env = _server_env(settings, env_overrides)
    async with MCPClient(SERVER_MODULES, env=env) as client:
        yield MCPResearchTools(client)
