r"""LangGraph orchestration: planner -> researcher -> critic -(loop)-> writer.

The graph's shared ``GraphState`` carries the typed A2A messages between nodes.
Each node instantiates nothing -- it calls into an injected agent -- which keeps
the graph a thin coordination layer and lets tests swap in mock agents.

Control flow:

    START -> planner -> researcher -> critic --(gaps & under cap)--> researcher
                                            \--(no gaps | cap hit)--> writer -> END

The critic loop is the reason this needs real state: ``findings`` accumulate
across iterations and ``iteration`` enforces a hard cap. The loop runs in-memory;
the researcher node calls live MCP tools (web search + filesystem).
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from research_assistant.agents import CriticAgent, PlannerAgent, ResearcherAgent, WriterAgent
from research_assistant.agents.base import BaseAgent
from research_assistant.config import Settings, get_settings
from research_assistant.mcp_client import build_research_tools
from research_assistant.messages import (
    CriticInput,
    CritiqueResult,
    PlannerInput,
    Report,
    ResearchFinding,
    ResearchPlan,
    ResearcherInput,
    WriterInput,
)

#: Default cap on critic<->researcher cycles. The brief specifies max 3.
DEFAULT_MAX_ITERATIONS = 3


class GraphState(TypedDict, total=False):
    """Shared state threaded through the graph.

    ``total=False`` because nodes populate fields incrementally; only
    ``question`` and ``max_iterations`` are guaranteed present at START.
    """

    question: str
    max_iterations: int
    plan: ResearchPlan
    findings: list[ResearchFinding]
    critique: CritiqueResult
    iteration: int
    report: Report


def build_graph(
    *,
    planner: BaseAgent | None = None,
    researcher: BaseAgent | None = None,
    critic: BaseAgent | None = None,
    writer: BaseAgent | None = None,
):
    """Build and compile the research graph.

    Agents are injectable so tests can supply mocks; defaults are the real
    (Phase 1: stub) agents.
    """

    if researcher is None:
        raise ValueError(
            "researcher must be provided: it requires live MCP tools. "
            "Use run_research(), or build_research_tools() to construct one."
        )
    planner = planner or PlannerAgent()
    critic = critic or CriticAgent()
    writer = writer or WriterAgent()

    async def plan_node(state: GraphState) -> dict:
        plan = await planner.run(PlannerInput(question=state["question"]))
        return {"plan": plan}

    async def research_node(state: GraphState) -> dict:
        critique = state.get("critique")
        gaps = critique.gaps if critique else []
        result = await researcher.run(
            ResearcherInput(
                plan=state["plan"],
                prior_findings=state.get("findings", []),
                gaps=gaps,
            )
        )
        return {"findings": result.findings}

    async def critic_node(state: GraphState) -> dict:
        critique = await critic.run(
            CriticInput(plan=state["plan"], findings=state["findings"])
        )
        return {"critique": critique, "iteration": state.get("iteration", 0) + 1}

    async def write_node(state: GraphState) -> dict:
        report = await writer.run(
            WriterInput(
                plan=state["plan"],
                findings=state["findings"],
                critique=state.get("critique"),
            )
        )
        return {"report": report}

    def route_after_critic(state: GraphState) -> str:
        """Loop back to the researcher only if there are gaps and budget remains."""
        critique = state["critique"]
        max_iter = state.get("max_iterations", DEFAULT_MAX_ITERATIONS)
        if critique.has_gaps and state.get("iteration", 0) < max_iter:
            return "researcher"
        return "writer"

    builder = StateGraph(GraphState)
    builder.add_node("planner", plan_node)
    builder.add_node("researcher", research_node)
    builder.add_node("critic", critic_node)
    builder.add_node("writer", write_node)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "researcher")
    builder.add_edge("researcher", "critic")
    builder.add_conditional_edges(
        "critic",
        route_after_critic,
        {"researcher": "researcher", "writer": "writer"},
    )
    builder.add_edge("writer", END)

    return builder.compile()


async def run_research_state(
    question: str,
    *,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    settings: Settings | None = None,
) -> GraphState:
    """Run the full graph and return the complete final state.

    Owns the MCP tool lifecycle -- the web-search + filesystem servers are spawned
    for the duration of the run and torn down on exit. Callers that only need the
    report can use ``run_research``; the demo UI uses the full state (plan,
    findings, critique, iteration count, report).
    """
    settings = settings or get_settings()
    async with build_research_tools(settings) as tools:
        graph = build_graph(
            researcher=ResearcherAgent(tools=tools, settings=settings)
        )
        final_state = await graph.ainvoke(
            {"question": question, "max_iterations": max_iterations, "iteration": 0}
        )
    return final_state


async def run_research(
    question: str,
    *,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    settings: Settings | None = None,
) -> Report:
    """Convenience entry point: run the full graph and return the final report."""
    state = await run_research_state(
        question, max_iterations=max_iterations, settings=settings
    )
    return state["report"]
