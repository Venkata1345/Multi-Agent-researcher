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
from research_assistant.messages import (
    CriticInput,
    CritiqueResult,
    PlannerInput,
    Report,
    ResearchFinding,
    ResearchFindings,
    ResearchPlan,
    ResearcherInput,
    WriterInput,
)
from tests.fakes import (
    FakeChatModel,
    make_critique,
    make_findings,
    make_plan,
    make_report,
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


async def test_researcher_merges_prior_findings():
    prior = [ResearchFinding(step_id=1, summary="prior")]
    model = FakeChatModel({ResearchFindings: make_findings(2)})
    agent = ResearcherAgent(model=model)

    result = await agent.run(
        ResearcherInput(plan=make_plan(2), prior_findings=prior)
    )

    # prior preserved, new appended (researcher owns accumulation)
    assert [f.step_id for f in result.findings] == [1, 2]


async def test_researcher_focuses_on_gaps_when_present():
    model = FakeChatModel({ResearchFindings: make_findings(1)})
    agent = ResearcherAgent(model=model)
    critique = make_critique(with_gaps=True)

    await agent.run(
        ResearcherInput(plan=make_plan(2), gaps=critique.gaps)
    )

    assert "Close these gaps" in model.last_user_text()


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
