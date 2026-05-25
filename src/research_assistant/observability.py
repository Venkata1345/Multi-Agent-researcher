"""Observability: logging + LangSmith tracing setup, custom per-agent metrics.

Tracing is *auto*: LangChain/LangGraph runnables export to LangSmith whenever the
``LANGSMITH_*`` environment is set, so every agent LLM call and graph node shows
up as a nested run with no per-call instrumentation. ``configure_tracing`` just
translates our typed ``Settings`` into that environment. (Phase 4 adds custom
per-agent latency / token / cost metadata on top.)
"""

from __future__ import annotations

import logging
import os

from research_assistant.config import Settings, get_settings

_CONFIGURED = False
_log = logging.getLogger("research_assistant.observability")


def configure_logging(level: int = logging.INFO) -> None:
    """Idempotently configure root logging for the CLI."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
    )
    _CONFIGURED = True


def configure_tracing(settings: Settings | None = None) -> bool:
    """Enable LangSmith auto-tracing from typed settings. Returns whether it's on.

    No-op (and returns False) unless both tracing is requested and an API key is
    present -- so tests and key-less runs never attempt to reach LangSmith.
    """
    settings = settings or get_settings()
    if not (settings.langsmith_tracing and settings.langsmith_api_key):
        _log.info("LangSmith tracing disabled.")
        return False

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"  # back-compat env name
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ.setdefault("LANGSMITH_PROJECT", settings.langsmith_project)
    _log.info("LangSmith tracing enabled (project=%s).", settings.langsmith_project)
    return True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
