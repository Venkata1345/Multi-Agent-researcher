"""Per-agent unit tests using a mocked LLM (no network)."""

from __future__ import annotations

import pytest

from research_assistant.agents import (
    CriticAgent,
    PlannerAgent,
    ResearcherAgent,
    WriterAgent,
)
from research_assistant.agents.base import load_prompt
from research_assistant.agents.researcher import FindingDraft
from research_assistant.messages import (
    CriticInput,
    CritiqueResult,
    Gap,
    PlannerInput,
    Report,
    ResearchFinding,
    ResearchPlan,
    ResearcherInput,
    Severity,
    WriterInput,
)
from tests.fakes import (
    FakeChatModel,
    FakeResearchTools,
    make_critique,
    make_draft,
    make_findings,
    make_plan,
    make_report,
    make_search_results,
)


@pytest.mark.parametrize("name", ["planner", "researcher", "critic", "writer"])
def test_prompt_files_load(name: str):
    text = load_prompt(name)
    assert text.strip()  # non-empty, packaged correctly


async def test_planner_sends_question_and_returns_plan():
    plan = make_plan(n_steps=3, question="What is X?")
    model = FakeChatModel({ResearchPlan: plan})
    agent = PlannerAgent(model=model)

    result = await agent.run(PlannerInput(question="What is X?", max_steps=3))

    assert result is plan
    assert model.calls[0][0] is ResearchPlan  # asked for the right schema
    assert "What is X?" in model.last_user_text()


async def test_researcher_grounds_citations_in_retrieved_sources():
    tools = FakeResearchTools(make_search_results(2))
    model = FakeChatModel({FindingDraft: make_draft(used=(0, 1))})
    agent = ResearcherAgent(tools=tools, model=model)

    result = await agent.run(ResearcherInput(plan=make_plan(1)))

    finding = result.findings[0]
    assert [str(c.url) for c in finding.citations] == [
        "https://example.com/0",
        "https://example.com/1",
    ]
    assert tools.searched  # the search tool was actually called
    assert "findings/step_1.md" in tools.written  # persisted via filesystem tool


async def test_researcher_skips_out_of_range_source_indices():
    tools = FakeResearchTools(make_search_results(2))
    model = FakeChatModel({FindingDraft: make_draft(used=(0, 99))})  # 99 invalid
    agent = ResearcherAgent(tools=tools, model=model)

    result = await agent.run(ResearcherInput(plan=make_plan(1)))

    assert len(result.findings[0].citations) == 1


async def test_researcher_merges_prior_findings():
    prior = [ResearchFinding(step_id=9, summary="prior")]
    tools = FakeResearchTools()
    model = FakeChatModel({FindingDraft: make_draft()})
    agent = ResearcherAgent(tools=tools, model=model)

    result = await agent.run(
        ResearcherInput(plan=make_plan(1), prior_findings=prior)
    )

    assert result.findings[0].step_id == 9  # prior preserved first
    assert len(result.findings) == 2


async def test_researcher_uses_gap_suggested_query_on_rerun():
    tools = FakeResearchTools()
    model = FakeChatModel({FindingDraft: make_draft()})
    agent = ResearcherAgent(tools=tools, model=model)
    gap = Gap(
        description="missing", related_step_id=2,
        suggested_query="specific gap query", severity=Severity.HIGH,
    )

    await agent.run(ResearcherInput(plan=make_plan(2), gaps=[gap]))

    assert "specific gap query" in tools.searched


async def test_critic_returns_structured_result():
    model = FakeChatModel({CritiqueResult: make_critique(with_gaps=True)})
    agent = CriticAgent(model=model)

    result = await agent.run(
        CriticInput(plan=make_plan(2), findings=make_findings(1, 2).findings)
    )

    assert isinstance(result, CritiqueResult)
    assert result.has_gaps is True


async def test_writer_returns_report():
    model = FakeChatModel({Report: make_report("What is X?")})
    agent = WriterAgent(model=model)

    result = await agent.run(
        WriterInput(plan=make_plan(2, "What is X?"), findings=make_findings(1, 2).findings)
    )

    assert isinstance(result, Report)
    assert result.question == "What is X?"
