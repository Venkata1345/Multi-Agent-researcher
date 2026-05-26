"""Researcher agent (Phase 3): retrieval-grounded findings via MCP tools.

For each step (or each critic gap on a re-run) the researcher:
  1. calls the MCP ``search`` tool for the step's queries and de-duplicates hits;
  2. hands the LLM the *numbered* sources and gets back a summary plus the source
     indices it actually used;
  3. maps those indices to real ``Citation`` objects -- so every cited URL is one
     the search tool returned, never one the model invented;
  4. persists the finding to the sandboxed filesystem via the MCP ``write_file``
     tool (best-effort).

Tools are injected (``ResearchTools``) so the integration test can run against a
fake without any live web call. The researcher still owns accumulation: it
returns prior + new findings each call.
"""

from __future__ import annotations

import json

from pydantic import Field, ValidationError

from research_assistant.agents.base import BaseAgent
from research_assistant.mcp_client import ResearchTools
from research_assistant.messages import (
    A2AMessage,
    Citation,
    ResearcherInput,
    ResearchFinding,
    ResearchFindings,
    SearchResult,
)


class FindingDraft(A2AMessage):
    """LLM-generated part of a finding. Citations are attached in code from the
    retrieved sources (by index), not generated here."""

    summary: str = Field(description="2-4 sentence synthesized answer to the step.")
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Confidence the summary is supported."
    )
    used_source_indices: list[int] = Field(
        default_factory=list,
        description="Indices (from the numbered SOURCES list) actually used.",
    )


# (step_id, question, queries) describing one unit of research work.
_Target = tuple[int, str, list[str]]


class ResearcherAgent(BaseAgent[ResearcherInput, ResearchFindings]):
    name = "researcher"
    prompt_name = "researcher"

    def __init__(
        self,
        tools: ResearchTools,
        model=None,
        settings=None,
        *,
        max_results: int = 5,
        max_queries: int = 2,
    ) -> None:
        super().__init__(model, settings)
        self._tools = tools
        self._max_results = max_results
        self._max_queries = max_queries

    async def run(self, payload: ResearcherInput) -> ResearchFindings:
        new: list[ResearchFinding] = []
        for step_id, question, queries in self._targets(payload):
            sources = await self._gather(queries)
            finding = await self._synthesize(step_id, question, sources)
            await self._persist(finding)
            new.append(finding)
        return ResearchFindings(findings=[*payload.prior_findings, *new])

    def _targets(self, payload: ResearcherInput) -> list[_Target]:
        if payload.gaps:
            return [
                (
                    gap.related_step_id or 0,
                    gap.description,
                    [gap.suggested_query or gap.description],
                )
                for gap in payload.gaps
            ]
        return [
            (step.id, step.question, step.search_queries or [step.question])
            for step in payload.plan.steps
        ]

    async def _gather(self, queries: list[str]) -> list[SearchResult]:
        """Run searches for a step and de-duplicate hits by URL (order-preserving)."""
        seen: dict[str, SearchResult] = {}
        for query in queries[: self._max_queries]:
            for result in await self._tools.search(query, self._max_results):
                seen.setdefault(result.url, result)
        return list(seen.values())

    async def _synthesize(
        self, step_id: int, question: str, sources: list[SearchResult]
    ) -> ResearchFinding:
        numbered = "\n".join(
            f"[{i}] {s.title or s.url}\n    url: {s.url}\n    {s.content[:500]}"
            for i, s in enumerate(sources)
        )
        user = (
            f"RESEARCH QUESTION:\n{question}\n\n"
            f"SOURCES (cite by index):\n{numbered or '(no sources found)'}\n\n"
            "Write a summary grounded in these sources and list the indices you used."
        )
        draft = await self._complete(user=user, schema=FindingDraft)

        citations: list[Citation] = []
        for i in draft.used_source_indices:
            if 0 <= i < len(sources):
                citation = _to_citation(sources[i])
                if citation is not None:
                    citations.append(citation)

        return ResearchFinding(
            step_id=step_id,
            summary=draft.summary or f"No findings for: {question}",
            citations=citations,
            confidence=draft.confidence,
        )

    async def _persist(self, finding: ResearchFinding) -> None:
        body = [f"# Finding for step {finding.step_id}", "", finding.summary, ""]
        if finding.citations:
            body.append("## Sources")
            body += [f"- [{c.title}]({c.url})" for c in finding.citations]
        await self._tools.write_file(
            f"findings/step_{finding.step_id}.md", "\n".join(body)
        )


def _to_citation(source: SearchResult) -> Citation | None:
    """Build a Citation, or None if the source URL is unusable (skip it)."""
    try:
        return Citation(
            url=source.url,
            title=source.title or source.url,
            snippet=(source.content[:300] or None),
        )
    except ValidationError:
        return None
