"""Critic agent.

Phase 1: a deterministic stub that exercises the critic->researcher loop exactly
once. It reports a gap on the *first* pass (when the number of findings equals
the number of plan steps) and signs off on the second pass (once the researcher
has appended a gap-closing finding). This is driven by the actual data flow, not
by a hidden counter, so the loop is visible end-to-end.

Phase 2 replaces this with a real LLM judgment over coverage and citations.
"""

from __future__ import annotations

from research_assistant.agents.base import BaseAgent
from research_assistant.messages import CriticInput, CritiqueResult, Gap, Severity


class CriticAgent(BaseAgent[CriticInput, CritiqueResult]):
    name = "critic"

    async def run(self, payload: CriticInput) -> CritiqueResult:
        n_steps = len(payload.plan.steps)
        n_findings = len(payload.findings)
        first_pass = n_findings <= n_steps

        if first_pass:
            gap = Gap(
                description=(
                    "[stub] Implications step lacks supporting detail and citations."
                ),
                related_step_id=payload.plan.steps[-1].id,
                suggested_query=f"{payload.plan.question} risks and tradeoffs",
                severity=Severity.MEDIUM,
            )
            return CritiqueResult(
                coverage_score=0.6,
                citation_quality=0.2,
                assessment="[stub] Decent breadth but the synthesis step is thin.",
                gaps=[gap],
            )

        return CritiqueResult(
            coverage_score=0.9,
            citation_quality=0.4,
            assessment="[stub] Gaps addressed; findings now cover the objective.",
            gaps=[],
        )
