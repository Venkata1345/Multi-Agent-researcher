"""Writer agent: synthesizes the plan + findings into a final markdown ``Report``."""

from __future__ import annotations

import json

from research_assistant.agents.base import BaseAgent
from research_assistant.messages import Report, WriterInput


class WriterAgent(BaseAgent[WriterInput, Report]):
    name = "writer"
    prompt_name = "writer"

    async def run(self, payload: WriterInput) -> Report:
        findings = json.dumps(
            [f.model_dump(mode="json") for f in payload.findings], indent=2
        )
        critique = (
            payload.critique.model_dump_json(indent=2)
            if payload.critique is not None
            else "(none)"
        )
        user = (
            f"PLAN:\n{payload.plan.model_dump_json(indent=2)}\n\n"
            f"FINDINGS:\n{findings}\n\n"
            f"CRITIC ASSESSMENT (for your awareness):\n{critique}\n\n"
            "Write the final report. Set `question` to the plan's question."
        )
        return await self._complete(user=user, schema=Report)
