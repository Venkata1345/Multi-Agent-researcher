"""Tests for the eval harness's offline parts (no pipeline run, no network)."""

from __future__ import annotations

from evals.run_eval import (
    aggregate,
    composite_score,
    load_golden_set,
    render_markdown,
)


def test_golden_set_loads_and_is_well_formed():
    items = load_golden_set()
    assert len(items) == 10
    for item in items:
        assert item["id"] and item["question"]
        assert isinstance(item["expected_points"], list) and item["expected_points"]


def test_composite_score_rewards_grounded_high_scores():
    perfect = {
        "relevance": 1.0, "coverage": 1.0, "citation_quality": 1.0, "hallucination": 0.0
    }
    poor = {
        "relevance": 0.2, "coverage": 0.2, "citation_quality": 0.0, "hallucination": 1.0
    }
    assert composite_score(perfect) == 1.0
    assert composite_score(poor) < 0.3


def test_aggregate_means_over_records():
    records = [
        {"id": "a", "scores": {"relevance": 1.0, "coverage": 0.5,
                               "citation_quality": 0.5, "hallucination": 0.0}},
        {"id": "b", "scores": {"relevance": 0.0, "coverage": 0.5,
                               "citation_quality": 0.5, "hallucination": 0.2}},
        {"id": "c", "error": "boom"},  # errored records are excluded
    ]
    agg = aggregate(records)
    assert agg["relevance"] == 0.5
    assert "composite" in agg


def test_render_markdown_includes_rows_and_mean():
    records = [
        {"id": "q1", "cost_usd": 0.01,
         "scores": {"relevance": 1.0, "coverage": 1.0,
                    "citation_quality": 1.0, "hallucination": 0.0}},
        {"id": "q2", "cost_usd": 0.0, "error": "boom"},
    ]
    agg = aggregate(records)
    md = render_markdown(records, agg)
    assert "q1" in md and "q2" in md
    assert "_error_" in md  # errored row rendered
    assert "**mean**" in md
