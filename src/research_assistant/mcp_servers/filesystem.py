"""MCP server exposing sandboxed ``read_file`` / ``write_file`` / ``list_dir`` tools.

Run standalone over stdio with ``python -m research_assistant.mcp_servers.filesystem``.

Capability scoping (the whole point of this server): every path is resolved and
checked to live *inside* ``RESEARCH_WORKSPACE``. Absolute paths, ``..`` traversal,
and symlink escapes all resolve outside the root and are rejected. The researcher
uses this to persist intermediate findings without ever touching the wider disk.
"""

from __future__ import annotations

from pathlib import Path

from mcp.server.fastmcp import FastMCP

from research_assistant.config import get_settings

mcp = FastMCP("filesystem", log_level="WARNING")


def _root() -> Path:
    root = Path(get_settings().research_workspace).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _resolve(rel_path: str) -> Path:
    """Resolve ``rel_path`` against the workspace root, rejecting any escape."""
    root = _root()
    candidate = (root / rel_path).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"path {rel_path!r} escapes the workspace sandbox")
    return candidate


@mcp.tool()
async def write_file(path: str, content: str) -> dict:
    """Write ``content`` to ``path`` (relative to the workspace). Creates parents."""
    target = _resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return {"path": str(target.relative_to(_root())), "bytes": len(content.encode())}


@mcp.tool()
async def read_file(path: str) -> dict:
    """Read and return the text content of ``path`` (relative to the workspace)."""
    target = _resolve(path)
    if not target.is_file():
        raise ValueError(f"file not found: {path!r}")
    return {"path": path, "content": target.read_text(encoding="utf-8")}


@mcp.tool()
async def list_dir(path: str = ".") -> dict:
    """List entry names directly under ``path`` (relative to the workspace)."""
    target = _resolve(path)
    if not target.is_dir():
        raise ValueError(f"not a directory: {path!r}")
    return {"entries": sorted(p.name for p in target.iterdir())}


if __name__ == "__main__":
    mcp.run()
