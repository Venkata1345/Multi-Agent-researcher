"""Tests for cost/latency metrics (no network)."""

from __future__ import annotations

from research_assistant.observability import (
    LLMCallMetric,
    estimate_cost,
    metrics_session,
    record_llm_call,
    render_metrics_table,
    summarize_metrics,
)


def test_estimate_cost_known_model():
    # 1M input @ $0.15 + 1M output @ $0.60 = $0.75
    assert estimate_cost("gpt-4o-mini", 1_000_000, 1_000_000) == 0.75


def test_estimate_cost_unknown_model_is_zero():
    assert estimate_cost("some-unknown-model", 1000, 1000) == 0.0


def test_metrics_session_collects_calls():
    with metrics_session() as metrics:
        record_llm_call(
            agent="planner",
            model="gpt-4o-mini",
            latency_s=1.2,
            usage={"input_tokens": 100, "output_tokens": 50},
        )
        record_llm_call(
            agent="planner",
            model="gpt-4o-mini",
            latency_s=0.8,
            usage={"input_tokens": 200, "output_tokens": 60},
        )
    assert len(metrics) == 2
    assert all(isinstance(m, LLMCallMetric) for m in metrics)
    assert metrics[0].cost_usd > 0


def test_record_outside_session_does_not_error():
    # No active session -> no collection, but must not raise.
    metric = record_llm_call(
        agent="critic", model="gpt-4o-mini", latency_s=0.1, usage=None
    )
    assert metric.input_tokens == 0


def test_render_metrics_table_aggregates_by_agent():
    metrics = [
        LLMCallMetric("planner", "gpt-4o-mini", 1.0, 100, 50, 0.0001),
        LLMCallMetric("planner", "gpt-4o-mini", 1.0, 100, 50, 0.0001),
        LLMCallMetric("writer", "gpt-4o-mini", 2.0, 300, 200, 0.0005),
    ]
    table = render_metrics_table(metrics)
    assert "planner" in table and "writer" in table
    assert "**total**" in table


def test_render_metrics_table_empty():
    assert "no LLM calls" in render_metrics_table([])


def test_summarize_metrics_has_total_row():
    metrics = [
        LLMCallMetric("planner", "gpt-4o-mini", 1.0, 100, 50, 0.0001),
        LLMCallMetric("writer", "gpt-4o-mini", 2.0, 300, 200, 0.0005),
    ]
    rows = summarize_metrics(metrics)
    assert rows[-1]["agent"] == "total"
    assert rows[-1]["calls"] == 2
    assert abs(rows[-1]["cost_usd"] - 0.0006) < 1e-9


def test_summarize_metrics_empty_is_empty():
    assert summarize_metrics([]) == []
