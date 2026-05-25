"""Schema contract tests for the A2A messages."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from research_assistant.messages import (
    Citation,
    CritiqueResult,
    Gap,
    Report,
    ResearchPlan,
    ResearchStep,
    Severity,
)


def _valid_step() -> ResearchStep:
    return ResearchStep(id=1, question="What is X?", rationale="It grounds the topic.")


def test_research_plan_requires_at_least_one_step():
    with pytest.raises(ValidationError):
        ResearchPlan(question="Q", objective="O", steps=[])


def test_research_plan_happy_path():
    plan = ResearchPlan(question="Q", objective="O", steps=[_valid_step()])
    assert plan.steps[0].id == 1
    assert plan.created_at.tzinfo is not None  # tz-aware


def test_extra_fields_forbidden():
    # extra="forbid" makes the structured-output contract strict.
    with pytest.raises(ValidationError):
        ResearchStep(id=1, question="Q", rationale="R", bogus="nope")


def test_citation_rejects_non_url():
    with pytest.raises(ValidationError):
        Citation(url="not-a-url", title="T")


def test_critique_has_gaps_property():
    no_gaps = CritiqueResult(
        coverage_score=0.9, citation_quality=0.5, assessment="ok"
    )
    assert no_gaps.has_gaps is False

    with_gaps = CritiqueResult(
        coverage_score=0.5,
        citation_quality=0.2,
        assessment="thin",
        gaps=[Gap(description="missing detail", severity=Severity.HIGH)],
    )
    assert with_gaps.has_gaps is True


def test_scores_are_bounded():
    with pytest.raises(ValidationError):
        CritiqueResult(coverage_score=1.5, citation_quality=0.5, assessment="x")


def test_report_word_count():
    report = Report(
        title="T",
        question="Q",
        summary="S",
        body_markdown="one two three four",
    )
    assert report.word_count == 4
