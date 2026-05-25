"""Planner agent: decomposes the question into a typed ``ResearchPlan`` via LLM."""

from __future__ import annotations

from research_assistant.agents.base import BaseAgent
from research_assistant.messages import PlannerInput, ResearchPlan


class PlannerAgent(BaseAgent[PlannerInput, ResearchPlan]):
    name = "planner"
    prompt_name = "planner"

    async def run(self, payload: PlannerInput) -> ResearchPlan:
        user = (
            f"Research question:\n{payload.question}\n\n"
            f"Produce a research plan with at most {payload.max_steps} steps. "
            f"Set `question` to the exact question above."
        )
        return await self._complete(user=user, schema=ResearchPlan)
