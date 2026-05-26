"""Integration test for the MCP stack against the real (key-free) filesystem
server, spawned over stdio. No network, no API keys."""

from __future__ import annotations

import pytest
from mcp.client.stdio import get_default_environment

from research_assistant.errors import MCPToolError
from research_assistant.mcp_client import SERVER_MODULES, MCPClient


def _fs_client(workspace) -> MCPClient:
    env = {**get_default_environment(), "RESEARCH_WORKSPACE": str(workspace)}
    return MCPClient({"filesystem": SERVER_MODULES["filesystem"]}, env=env)


async def test_filesystem_write_read_list_roundtrip(tmp_path):
    async with _fs_client(tmp_path) as client:
        write = await client.call_tool(
            "filesystem", "write_file", {"path": "sub/a.txt", "content": "hello mcp"}
        )
        assert write["bytes"] == len("hello mcp")

        read = await client.call_tool(
            "filesystem", "read_file", {"path": "sub/a.txt"}
        )
        assert read["content"] == "hello mcp"

        listing = await client.call_tool("filesystem", "list_dir", {"path": "sub"})
        assert "a.txt" in listing["entries"]


async def test_filesystem_sandbox_rejects_traversal(tmp_path):
    async with _fs_client(tmp_path) as client:
        with pytest.raises(MCPToolError):
            await client.call_tool(
                "filesystem", "read_file", {"path": "../escape.txt"}
            )


async def test_unknown_server_raises(tmp_path):
    async with _fs_client(tmp_path) as client:
        with pytest.raises(MCPToolError):
            await client.call_tool("web_search", "search", {"query": "x"})
