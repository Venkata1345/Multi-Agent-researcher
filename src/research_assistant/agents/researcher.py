"""Researcher agent.

Phase 1: synthesizes a placeholder ``ResearchFinding`` per plan step (or per gap
on a re-run) with no real citations. The researcher *owns accumulation*: it
returns the complete findings set (prior + new) each call, so the graph node can
simply replace state.

Phase 3 is where this agent grows teeth: it will call the web-search and
filesystem MCP tools, and citations will come from real URLs.
"""

from __future__ import annotations

from research_assistant.agents.base import BaseAgent
from research_assistant.messages import ResearcherInput, ResearchFinding, ResearchFindings


class ResearcherAgent(BaseAgent[ResearcherInput, ResearchFindings]):
    name = "researcher"

    async def run(self, payload: ResearcherInput) -> ResearchFindings:
        new: list[ResearchFinding] = []

        if payload.gaps:
            # Re-run: address the critic's gaps specifically.
            for gap in payload.gaps:
                new.append(
                    ResearchFinding(
                        step_id=gap.related_step_id or 0,
                        summary=(
                            f"[stub] Additional research closing the gap: "
                            f"{gap.description}"
                        ),
                        confidence=0.7,
                    )
                )
        else:
            # First pass: one finding per plan step.
            for step in payload.plan.steps:
                new.append(
                    ResearchFinding(
                        step_id=step.id,
                        summary=f"[stub] Findings for step {step.id}: {step.question}",
                        confidence=0.5,
                    )
                )

        return ResearchFindings(findings=[*payload.prior_findings, *new])
