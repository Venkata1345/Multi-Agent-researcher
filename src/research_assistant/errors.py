"""Typed errors. Agents raise these rather than swallowing failures."""

from __future__ import annotations


class AgentError(Exception):
    """Base class for all agent/pipeline failures."""


class PlanningError(AgentError):
    """The planner could not produce a usable plan."""


class RetrievalError(AgentError):
    """A research/tool retrieval step failed (e.g. web search unavailable)."""


class MCPToolError(AgentError):
    """An MCP tool call returned an error result."""
