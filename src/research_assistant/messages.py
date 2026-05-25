"""A2A (Agent-to-Agent) message schemas.

These Pydantic v2 models are the *contract* between agents. Every transition in
the graph passes one of these typed objects to the next agent -- no untyped dicts
ever cross an agent boundary. Agent *outputs* (``ResearchPlan``, ``ResearchFinding``,
``CritiqueResult``, ``Report``) are the A2A messages; the small ``*Input`` wrappers
exist so each agent's ``run`` method has a single typed argument.

In Phase 2 these same models become the ``response_format`` schemas handed to the
LLM for structured-output enforcement, so they are deliberately
LLM-generation-friendly: flat where possible, every field documented via
``Field(description=...)`` (the description is sent to the model).
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# Reusable constrained string aliases keep field declarations readable.
NonEmptyStr = Annotated[str, Field(min_length=1)]


class A2AMessage(BaseModel):
    """Base class for every inter-agent message.

    ``extra="forbid"`` makes the structured-output contract strict: if an LLM
    (Phase 2) emits a field we didn't declare, validation fails loudly rather
    than silently dropping data.
    """

    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- #
# Planner output
# --------------------------------------------------------------------------- #
class ResearchStep(A2AMessage):
    """A single, independently-investigable sub-question in the research plan."""

    id: int = Field(description="1-based ordinal of this step within the plan.")
    question: NonEmptyStr = Field(
        description="The specific sub-question this step answers."
    )
    rationale: NonEmptyStr = Field(
        description="Why this step matters to the overall research question."
    )
    search_queries: list[NonEmptyStr] = Field(
        default_factory=list,
        description="Suggested search queries the researcher can run for this step.",
    )


class ResearchPlan(A2AMessage):
    """Planner output: a decomposition of the user question into ordered steps."""

    question: NonEmptyStr = Field(description="The original user research question.")
    objective: NonEmptyStr = Field(
        description="One-sentence statement of what a complete answer must cover."
    )
    steps: list[ResearchStep] = Field(
        min_length=1, description="Ordered research steps to execute."
    )
    created_at: datetime = Field(default_factory=_utcnow)


# --------------------------------------------------------------------------- #
# Researcher output
# --------------------------------------------------------------------------- #
class Citation(A2AMessage):
    """A single cited source backing a finding."""

    url: HttpUrl = Field(description="Canonical source URL.")
    title: NonEmptyStr = Field(description="Title of the source page/document.")
    snippet: str | None = Field(
        default=None,
        description="Short quoted passage supporting the claim, if available.",
    )


class ResearchFinding(A2AMessage):
    """Researcher output for one step: a synthesized answer with its sources."""

    step_id: int = Field(description="``ResearchStep.id`` this finding answers.")
    summary: NonEmptyStr = Field(
        description="Concise synthesized answer to the step's sub-question."
    )
    citations: list[Citation] = Field(
        default_factory=list,
        description="Sources supporting the summary. May be empty until Phase 3.",
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Researcher's confidence the summary is well-supported.",
    )


class ResearchFindings(A2AMessage):
    """Researcher output: the full accumulated set of findings.

    A collection wrapper (rather than a bare ``list``) so the researcher's output
    is itself a typed A2A message satisfying ``BaseAgent``'s ``OutputT`` bound.
    The researcher owns accumulation across loop iterations, so this carries the
    *complete* findings set each time, not just the latest delta.
    """

    findings: list[ResearchFinding] = Field(
        default_factory=list, description="All findings gathered so far."
    )


# --------------------------------------------------------------------------- #
# Critic output
# --------------------------------------------------------------------------- #
class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Gap(A2AMessage):
    """A coverage/citation gap the critic wants the researcher to close."""

    description: NonEmptyStr = Field(description="What is missing or under-supported.")
    related_step_id: int | None = Field(
        default=None,
        description="Step this gap maps to, or None if it spans the whole plan.",
    )
    suggested_query: str | None = Field(
        default=None,
        description="A search query likely to close the gap.",
    )
    severity: Severity = Field(
        default=Severity.MEDIUM, description="How important closing this gap is."
    )


class CritiqueResult(A2AMessage):
    """Critic output: an assessment plus any gaps that should trigger a re-run."""

    coverage_score: float = Field(
        ge=0.0, le=1.0, description="How fully the findings answer the objective."
    )
    citation_quality: float = Field(
        ge=0.0, le=1.0, description="How well-sourced the findings are."
    )
    assessment: NonEmptyStr = Field(
        description="Prose summary of strengths and weaknesses of the findings."
    )
    gaps: list[Gap] = Field(
        default_factory=list,
        description="Outstanding gaps. Non-empty => graph loops back to researcher.",
    )

    @property
    def has_gaps(self) -> bool:
        return len(self.gaps) > 0


# --------------------------------------------------------------------------- #
# Writer output
# --------------------------------------------------------------------------- #
class Report(A2AMessage):
    """Writer output: the final cited markdown report."""

    title: NonEmptyStr = Field(description="Report title.")
    question: NonEmptyStr = Field(description="The original user research question.")
    summary: NonEmptyStr = Field(description="Executive summary / abstract.")
    body_markdown: NonEmptyStr = Field(
        description="Full report body in markdown, including inline citations."
    )
    citations: list[Citation] = Field(
        default_factory=list, description="De-duplicated list of all cited sources."
    )
    created_at: datetime = Field(default_factory=_utcnow)

    @property
    def word_count(self) -> int:
        return len(self.body_markdown.split())


# --------------------------------------------------------------------------- #
# Typed agent inputs (composed from the messages above)
# --------------------------------------------------------------------------- #
class PlannerInput(A2AMessage):
    question: NonEmptyStr = Field(description="User research question to plan for.")
    max_steps: int = Field(
        default=5, ge=1, le=10, description="Upper bound on plan steps."
    )


class ResearcherInput(A2AMessage):
    plan: ResearchPlan
    prior_findings: list[ResearchFinding] = Field(
        default_factory=list,
        description="Findings already gathered in earlier loop iterations.",
    )
    gaps: list[Gap] = Field(
        default_factory=list,
        description="Gaps from the critic to prioritize on a re-run. Empty on first pass.",
    )


class CriticInput(A2AMessage):
    plan: ResearchPlan
    findings: list[ResearchFinding]


class WriterInput(A2AMessage):
    plan: ResearchPlan
    findings: list[ResearchFinding]
    critique: CritiqueResult | None = Field(
        default=None, description="Final critique, for the writer's awareness."
    )


__all__ = [
    "A2AMessage",
    "ResearchStep",
    "ResearchPlan",
    "Citation",
    "ResearchFinding",
    "ResearchFindings",
    "Severity",
    "Gap",
    "CritiqueResult",
    "Report",
    "PlannerInput",
    "ResearcherInput",
    "CriticInput",
    "WriterInput",
]
