"""Writer agent.

Phase 1: assembles a schema-valid markdown ``Report`` from the plan and findings.
Phase 2 replaces the body with a structured-output LLM call that writes real
prose and inline citations.
"""

from __future__ import annotations

from research_assistant.agents.base import BaseAgent
from research_assistant.messages import Citation, Report, WriterInput


class WriterAgent(BaseAgent[WriterInput, Report]):
    name = "writer"

    async def run(self, payload: WriterInput) -> Report:
        plan = payload.plan

        # De-duplicate citations across all findings by URL.
        seen: dict[str, Citation] = {}
        for finding in payload.findings:
            for c in finding.citations:
                seen.setdefault(str(c.url), c)
        citations = list(seen.values())

        sections = []
        for step in plan.steps:
            step_findings = [f for f in payload.findings if f.step_id == step.id]
            body = (
                "\n".join(f"- {f.summary}" for f in step_findings)
                if step_findings
                else "- _(no findings)_"
            )
            sections.append(f"## {step.question}\n\n{body}")

        body_markdown = (
            f"# {plan.question}\n\n"
            f"_{plan.objective}_\n\n"
            + "\n\n".join(sections)
        )

        return Report(
            title=f"Research Report: {plan.question}",
            question=plan.question,
            summary=f"[stub] Synthesized answer to: {plan.question}",
            body_markdown=body_markdown,
            citations=citations,
        )
