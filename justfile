# Multi-Agent Research Assistant — task recipes (https://github.com/casey/just)
# `just` is optional; each recipe is a one-liner you can also run directly.

# List available recipes
default:
    @just --list

# Install/sync dependencies (incl. dev group)
sync:
    uv sync

# Run the full test suite (no network)
test:
    uv run pytest

# Run the assistant end-to-end (needs OPENAI_API_KEY + TAVILY_API_KEY in .env)
run question:
    uv run research "{{question}}"

# --- MCP servers (normally spawned automatically by the client over stdio) --- #
# Run a server standalone for manual inspection with the MCP Inspector, e.g.:
#   npx @modelcontextprotocol/inspector just mcp-web

# Start the Tavily-backed web-search MCP server (stdio)
mcp-web:
    uv run python -m research_assistant.mcp_servers.web_search

# Start the sandboxed filesystem MCP server (stdio)
mcp-fs:
    uv run python -m research_assistant.mcp_servers.filesystem
