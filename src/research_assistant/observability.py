"""Observability: logging + LangSmith tracing setup, custom per-agent metrics.

Phase 1 provides only logging configuration so agents/graph can emit structured
status without ``print``. Phase 2 wires LangSmith auto-tracing here; Phase 4 adds
custom per-agent latency / token / cost metadata.
"""

from __future__ import annotations

import logging

_CONFIGURED = False


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


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
