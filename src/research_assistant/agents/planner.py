"""Planner agent.

Phase 1: returns a hardcoded but schema-valid ``ResearchPlan`` derived from the
question text so the rest of the graph has real structure to operate on.
Phase 2 replaces the body of ``run`` with a structured-output LLM call.
"""

from __future__ import annotations

from research_assistant.agents.base import BaseAgent
from research_assistant.messages import PlannerInput, ResearchPlan, ResearchStep


class PlannerAgent(BaseAgent[PlannerInput, ResearchPlan]):
    name = "planner"

    async def run(self, payload: PlannerInput) -> ResearchPlan:
        q = payload.question.strip()
        # Stub decomposition: a generic background / current-state / implications
        # breakdown. Just enough structure to drive the downstream agents.
        templates = [
            ("Establish background and key definitions for the question.",
             "Grounds the rest of the research in shared terminology."),
            ("Gather the current state of the art / most recent developments.",
             "The freshest, most decision-relevant evidence usually lives here."),
            ("Identify implications, tradeoffs, and open questions.",
             "Turns raw facts into the synthesis a reader actually needs."),
        ]
        steps = [
            ResearchStep(
                id=i + 1,
                question=f"{desc} (re: {q})",
                rationale=why,
                search_queries=[q, f"{q} {kw}"],
            )
            for i, ((desc, why), kw) in enumerate(
                zip(templates, ["overview", "latest", "tradeoffs"])
            )
        ]
        return ResearchPlan(
            question=q,
            objective=f"Produce a well-cited synthesis answering: {q}",
            steps=steps[: payload.max_steps],
        )
