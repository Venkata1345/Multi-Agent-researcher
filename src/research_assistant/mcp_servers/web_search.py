"""MCP server exposing a web ``search`` tool, backed by Tavily.

Run standalone over stdio with ``python -m research_assistant.mcp_servers.web_search``
(this is how ``mcp_client`` spawns it). Tavily is purpose-built for LLM agents:
it returns clean, snippet-rich results, so the researcher gets citation-ready
content without scraping.

Capability scoping: read-only outbound search. ``max_results`` is clamped so a
single step can't fan out into an unbounded number of backend calls.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP
from tavily import AsyncTavilyClient

from research_assistant.config import get_settings

mcp = FastMCP("web-search", log_level="WARNING")

MAX_RESULTS_CAP = 10


@mcp.tool()
async def search(query: str, max_results: int = 5) -> dict:
    """Search the web and return ranked results.

    Returns ``{"results": [{"title", "url", "content", "score"}, ...]}``.
    Raises if no Tavily API key is configured.
    """
    settings = get_settings()
    if not settings.tavily_api_key:
        raise ValueError(
            "TAVILY_API_KEY is not set; cannot run web search. Add it to .env."
        )

    k = max(1, min(max_results, MAX_RESULTS_CAP))
    client = AsyncTavilyClient(settings.tavily_api_key)
    response = await client.search(query=query, max_results=k)

    results = [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", ""),
            "score": r.get("score"),
        }
        for r in response.get("results", [])
    ]
    return {"results": results}


if __name__ == "__main__":
    mcp.run()
