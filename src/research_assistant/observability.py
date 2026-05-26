"""Observability: logging, LangSmith tracing, and per-agent cost/latency metrics.

Two layers:

1. **Tracing** is automatic -- LangChain/LangGraph runnables export to LangSmith
   whenever the ``LANGSMITH_*`` environment is set, so every agent LLM call and
   graph node shows up as a nested run. ``configure_tracing`` just translates our
   typed ``Settings`` into that environment.

2. **Metrics** are explicit. ``record_llm_call`` captures latency + token usage
   for each agent's LLM call, estimates cost from a per-model pricing table, and
   (a) collects it into a ``metrics_session`` for an end-of-run summary table and
   (b) best-effort attaches it as metadata on the active LangSmith run.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator

from research_assistant.config import Settings, get_settings

_CONFIGURED = False
_log = logging.getLogger("research_assistant.observability")


# --------------------------------------------------------------------------- #
# Logging + tracing setup
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Cost / latency metrics
# --------------------------------------------------------------------------- #
#: USD per 1M tokens, (input, output). Update as pricing changes.
PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
}


@dataclass(slots=True)
class LLMCallMetric:
    agent: str
    model: str
    latency_s: float
    input_tokens: int
    output_tokens: int
    cost_usd: float


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate USD cost for a call. Returns 0.0 for models not in PRICING."""
    if model not in PRICING:
        _log.debug("No pricing for model %r; cost estimated as 0.", model)
        return 0.0
    in_price, out_price = PRICING[model]
    return (input_tokens / 1e6) * in_price + (output_tokens / 1e6) * out_price


# Collector for the current research run. A mutable list shared across async
# tasks (we append, never reassign), so it survives context copying in LangGraph.
_METRICS: ContextVar[list[LLMCallMetric] | None] = ContextVar("_metrics", default=None)


@contextmanager
def metrics_session() -> Iterator[list[LLMCallMetric]]:
    """Collect every ``record_llm_call`` within the block into one list."""
    buffer: list[LLMCallMetric] = []
    token = _METRICS.set(buffer)
    try:
        yield buffer
    finally:
        _METRICS.reset(token)


def record_llm_call(
    *, agent: str, model: str, latency_s: float, usage: dict | None
) -> LLMCallMetric:
    """Record one agent LLM call: collect it, log it, and tag the LangSmith run."""
    usage = usage or {}
    in_tok = int(usage.get("input_tokens", 0) or 0)
    out_tok = int(usage.get("output_tokens", 0) or 0)
    metric = LLMCallMetric(
        agent=agent,
        model=model,
        latency_s=latency_s,
        input_tokens=in_tok,
        output_tokens=out_tok,
        cost_usd=estimate_cost(model, in_tok, out_tok),
    )

    buffer = _METRICS.get()
    if buffer is not None:
        buffer.append(metric)

    _log.info(
        "agent=%-10s model=%-12s latency=%5.2fs tokens=%d/%d cost=$%.5f",
        metric.agent, metric.model, metric.latency_s,
        metric.input_tokens, metric.output_tokens, metric.cost_usd,
    )
    _attach_to_langsmith_run(metric)
    return metric


def _attach_to_langsmith_run(metric: LLMCallMetric) -> None:
    """Best-effort: stamp the active LangSmith run with this metric's metadata."""
    try:
        from langsmith.run_helpers import get_current_run_tree

        run = get_current_run_tree()
        if run is not None:
            run.metadata.update(
                {
                    "agent": metric.agent,
                    "latency_s": round(metric.latency_s, 3),
                    "input_tokens": metric.input_tokens,
                    "output_tokens": metric.output_tokens,
                    "cost_usd": round(metric.cost_usd, 6),
                }
            )
    except Exception:  # pragma: no cover - tracing is optional, never fatal
        pass


def summarize_metrics(metrics: list[LLMCallMetric]) -> list[dict]:
    """Aggregate metrics per agent, with a final ``total`` row.

    Shared by the CLI markdown table and the Streamlit demo so they never drift.
    """
    by_agent: dict[str, dict[str, float]] = {}
    for m in metrics:
        a = by_agent.setdefault(
            m.agent,
            {"calls": 0, "latency_s": 0.0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
        )
        a["calls"] += 1
        a["latency_s"] += m.latency_s
        a["input_tokens"] += m.input_tokens
        a["output_tokens"] += m.output_tokens
        a["cost_usd"] += m.cost_usd

    rows = [{"agent": agent, **vals} for agent, vals in by_agent.items()]
    if metrics:
        rows.append(
            {
                "agent": "total",
                "calls": len(metrics),
                "latency_s": sum(m.latency_s for m in metrics),
                "input_tokens": sum(m.input_tokens for m in metrics),
                "output_tokens": sum(m.output_tokens for m in metrics),
                "cost_usd": sum(m.cost_usd for m in metrics),
            }
        )
    return rows


def render_metrics_table(metrics: list[LLMCallMetric]) -> str:
    """Render per-agent aggregated metrics as a markdown table + totals row."""
    if not metrics:
        return "(no LLM calls recorded)"

    rows = [
        "| Agent | Calls | Latency (s) | In tok | Out tok | Cost (USD) |",
        "|-------|------:|------------:|-------:|--------:|-----------:|",
    ]
    for r in summarize_metrics(metrics):
        bold = "**" if r["agent"] == "total" else ""
        rows.append(
            f"| {bold}{r['agent']}{bold} | {bold}{int(r['calls'])}{bold} | "
            f"{bold}{r['latency_s']:.2f}{bold} | {bold}{int(r['input_tokens'])}{bold} | "
            f"{bold}{int(r['output_tokens'])}{bold} | {bold}${r['cost_usd']:.5f}{bold} |"
        )
    return "\n".join(rows)
