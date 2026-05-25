"""Critic agent: LLM judgment over coverage + citations, emitting gaps.

A non-empty ``gaps`` list makes the graph loop back to the researcher (capped at
``max_iterations``). The prompt explicitly discourages inventing gaps, since each
one triggers an expensive re-research cycle.
"""

from __future__ import annotations

import json

from research_assistant.agents.base import BaseAgent
from research_assistant.messages import CriticInput, CritiqueResult


class CriticAgent(BaseAgent[CriticInput, CritiqueResult]):
    name = "critic"
    prompt_name = "critic"

    async def run(self, payload: CriticInput) -> CritiqueResult:
        findings = json.dumps(
            [f.model_dump(mode="json") for f in payload.findings], indent=2
        )
        user = (
            f"PLAN:\n{payload.plan.model_dump_json(indent=2)}\n\n"
            f"FINDINGS:\n{findings}\n\n"
            "Judge coverage and citation quality, and list any genuine gaps."
        )
        return await self._complete(user=user, schema=CritiqueResult)
