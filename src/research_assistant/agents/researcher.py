"""Researcher agent.

Phase 2: generates findings via a structured-output LLM call (no tools yet, so
findings are uncited -- real citations arrive with the MCP tools in Phase 3).

The researcher *owns accumulation*: the LLM is asked only for findings on the
current steps/gaps, and the agent merges them with prior findings so the graph
node can simply replace state. On a re-run it focuses the LLM on the critic's
gaps rather than re-researching everything.
"""

from __future__ import annotations

import json

from research_assistant.agents.base import BaseAgent
from research_assistant.messages import ResearcherInput, ResearchFindings


class ResearcherAgent(BaseAgent[ResearcherInput, ResearchFindings]):
    name = "researcher"
    prompt_name = "researcher"

    async def run(self, payload: ResearcherInput) -> ResearchFindings:
        if payload.gaps:
            task = (
                "Close these gaps from the critic (research only these):\n"
                + json.dumps([g.model_dump(mode="json") for g in payload.gaps], indent=2)
            )
        else:
            task = (
                "Produce one finding per research step below:\n"
                + json.dumps(
                    [s.model_dump(mode="json") for s in payload.plan.steps], indent=2
                )
            )

        prior = (
            json.dumps(
                [f.model_dump(mode="json") for f in payload.prior_findings], indent=2
            )
            if payload.prior_findings
            else "(none yet)"
        )

        user = (
            f"Research objective: {payload.plan.objective}\n\n"
            f"{task}\n\n"
            f"Findings already gathered (for context, do NOT repeat):\n{prior}"
        )

        new = await self._complete(user=user, schema=ResearchFindings)
        return ResearchFindings(findings=[*payload.prior_findings, *new.findings])
