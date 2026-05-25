"""End-to-end graph tests (Phase 1: stub agents, no network/LLM)."""

from __future__ import annotations

from research_assistant.agents.base import BaseAgent
from research_assistant.graph import build_graph, run_research
from research_assistant.messages import CriticInput, CritiqueResult, Gap, Report, Severity


async def test_end_to_end_returns_report():
    report = await run_research("What is retrieval-augmented generation?")
    assert isinstance(report, Report)
    assert report.question == "What is retrieval-augmented generation?"
    assert report.body_markdown.startswith("# ")


async def test_critic_loop_fires_exactly_once_with_stub():
    # Stub critic raises one gap on the first pass, then signs off -> one loop.
    graph = build_graph()
    state = await graph.ainvoke(
        {"question": "Q", "max_iterations": 3, "iteration": 0}
    )
    assert state["iteration"] == 2  # critic ran twice
    # planner makes 3 steps -> 3 findings, then +1 gap-closing finding.
    assert len(state["findings"]) == 4
    assert state["critique"].has_gaps is False


class _AlwaysGapCritic(BaseAgent[CriticInput, CritiqueResult]):
    name = "always-gap-critic"

    async def run(self, payload: CriticInput) -> CritiqueResult:
        return CritiqueResult(
            coverage_score=0.5,
            citation_quality=0.1,
            assessment="never satisfied",
            gaps=[Gap(description="always more", severity=Severity.LOW)],
        )


async def test_max_iterations_cap_is_enforced():
    graph = build_graph(critic=_AlwaysGapCritic())
    state = await graph.ainvoke(
        {"question": "Q", "max_iterations": 2, "iteration": 0}
    )
    # Loop must stop at the cap and still produce a report.
    assert state["iteration"] == 2
    assert isinstance(state["report"], Report)


class _NoGapCritic(BaseAgent[CriticInput, CritiqueResult]):
    name = "no-gap-critic"

    async def run(self, payload: CriticInput) -> CritiqueResult:
        return CritiqueResult(
            coverage_score=1.0, citation_quality=0.5, assessment="great", gaps=[]
        )


async def test_no_gaps_skips_straight_to_writer():
    graph = build_graph(critic=_NoGapCritic())
    state = await graph.ainvoke(
        {"question": "Q", "max_iterations": 3, "iteration": 0}
    )
    assert state["iteration"] == 1  # critic ran once, no loop-back
    assert isinstance(state["report"], Report)
