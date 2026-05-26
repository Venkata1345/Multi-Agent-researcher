"""End-to-end graph tests: real agents driven by a mocked LLM (no network)."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from research_assistant.agents import (
    CriticAgent,
    PlannerAgent,
    ResearcherAgent,
    WriterAgent,
)
from research_assistant.agents.researcher import FindingDraft
from research_assistant.graph import build_graph
from research_assistant.messages import CritiqueResult, Report, ResearchPlan
from tests.fakes import (
    FakeChatModel,
    FakeResearchTools,
    make_critique,
    make_draft,
    make_plan,
    make_report,
)


def _graph_with(model: BaseChatModel):
    """Build the graph with the real agents, all sharing one (fake) model and
    fake MCP tools."""
    return build_graph(
        planner=PlannerAgent(model=model),
        researcher=ResearcherAgent(tools=FakeResearchTools(), model=model),
        critic=CriticAgent(model=model),
        writer=WriterAgent(model=model),
    )


async def test_end_to_end_no_gaps_goes_straight_to_writer():
    model = FakeChatModel(
        {
            ResearchPlan: make_plan(2, "What is RAG?"),
            FindingDraft: make_draft(),
            CritiqueResult: make_critique(with_gaps=False),
            Report: make_report("What is RAG?"),
        }
    )
    state = await _graph_with(model).ainvoke(
        {"question": "What is RAG?", "max_iterations": 3, "iteration": 0}
    )

    assert isinstance(state["report"], Report)
    assert state["iteration"] == 1  # critic ran once, no loop-back
    assert state["critique"].has_gaps is False
    # 2 plan steps -> 2 findings, each citing the 2 retrieved sources
    assert len(state["findings"]) == 2
    assert all(f.citations for f in state["findings"])


async def test_critic_loop_fires_once_then_writes():
    model = FakeChatModel(
        {
            ResearchPlan: make_plan(2),
            FindingDraft: make_draft(),
            CritiqueResult: [make_critique(with_gaps=True), make_critique(with_gaps=False)],
            Report: make_report(),
        }
    )
    state = await _graph_with(model).ainvoke(
        {"question": "Q", "max_iterations": 3, "iteration": 0}
    )

    assert state["iteration"] == 2  # looped back once
    # pass-1 findings (2 steps) + gap-closing finding (1), merged by the researcher
    assert len(state["findings"]) == 3
    assert isinstance(state["report"], Report)


async def test_max_iterations_cap_is_enforced():
    model = FakeChatModel(
        {
            ResearchPlan: make_plan(2),
            FindingDraft: make_draft(),
            CritiqueResult: make_critique(with_gaps=True),  # never satisfied
            Report: make_report(),
        }
    )
    state = await _graph_with(model).ainvoke(
        {"question": "Q", "max_iterations": 2, "iteration": 0}
    )

    assert state["iteration"] == 2  # stopped at the cap despite open gaps
    assert isinstance(state["report"], Report)
