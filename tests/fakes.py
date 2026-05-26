"""Test doubles: a fake chat model + canned A2A messages.

``FakeChatModel`` stands in for a LangChain chat model. It maps the requested
structured-output *schema* to a canned instance (or a per-schema queue for
sequenced responses) and records calls so tests can assert on the prompts sent.
This keeps every agent/graph test fully offline.
"""

from __future__ import annotations

from typing import Any

from research_assistant.agents.researcher import FindingDraft
from research_assistant.messages import (
    A2AMessage,
    CritiqueResult,
    Gap,
    Report,
    ResearchFinding,
    ResearchFindings,
    ResearchPlan,
    ResearchStep,
    SearchResult,
    Severity,
)


class _FakeRaw:
    """Stand-in for a raw AIMessage carrying token usage."""

    usage_metadata = {"input_tokens": 12, "output_tokens": 8, "total_tokens": 20}


class _FakeStructuredRunnable:
    def __init__(
        self, schema: type[A2AMessage], parent: "FakeChatModel", include_raw: bool
    ) -> None:
        self._schema = schema
        self._parent = parent
        self._include_raw = include_raw

    async def ainvoke(self, messages: list[Any], config: Any = None) -> Any:
        self._parent.calls.append((self._schema, messages))
        resp = self._parent.responses.get(self._schema)
        if isinstance(resp, list):
            if not resp:
                raise AssertionError(f"FakeChatModel queue exhausted for {self._schema}")
            resp = resp.pop(0)
        if resp is None:
            raise AssertionError(f"FakeChatModel has no response for {self._schema}")
        if self._include_raw:
            return {"raw": _FakeRaw(), "parsed": resp, "parsing_error": None}
        return resp


class FakeChatModel:
    """Minimal structured-output chat model stand-in.

    ``responses`` maps a schema type to either a single canned instance or a list
    used as a FIFO queue (for agents called more than once, e.g. the critic in a
    loop test).
    """

    def __init__(self, responses: dict[type[A2AMessage], Any] | None = None) -> None:
        self.responses: dict[type[A2AMessage], Any] = responses or {}
        self.calls: list[tuple[type[A2AMessage], list[Any]]] = []

    def with_structured_output(
        self,
        schema: type[A2AMessage],
        method: str | None = None,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> _FakeStructuredRunnable:
        return _FakeStructuredRunnable(schema, self, include_raw)

    def last_user_text(self) -> str:
        """The human-message content of the most recent call."""
        _, messages = self.calls[-1]
        return messages[-1].content


# --- Canned message builders ------------------------------------------------ #
def make_plan(n_steps: int = 2, question: str = "Q") -> ResearchPlan:
    return ResearchPlan(
        question=question,
        objective="cover the question",
        steps=[
            ResearchStep(id=i + 1, question=f"sub-q {i + 1}", rationale="because")
            for i in range(n_steps)
        ],
    )


def make_findings(*step_ids: int) -> ResearchFindings:
    return ResearchFindings(
        findings=[
            ResearchFinding(step_id=sid, summary=f"finding for step {sid}")
            for sid in step_ids
        ]
    )


def make_critique(with_gaps: bool) -> CritiqueResult:
    gaps = (
        [Gap(description="missing detail", related_step_id=1, severity=Severity.MEDIUM)]
        if with_gaps
        else []
    )
    return CritiqueResult(
        coverage_score=0.5 if with_gaps else 0.95,
        citation_quality=0.1,
        assessment="assessment text",
        gaps=gaps,
    )


def make_report(question: str = "Q") -> Report:
    return Report(
        title="A Report",
        question=question,
        summary="executive summary",
        body_markdown="# A Report\n\nBody.",
    )


def make_search_results(n: int = 2) -> list[SearchResult]:
    return [
        SearchResult(
            title=f"Source {i}",
            url=f"https://example.com/{i}",
            content=f"content snippet {i}",
            score=0.9 - i * 0.1,
        )
        for i in range(n)
    ]


def make_draft(
    used: tuple[int, ...] = (0, 1), summary: str = "synthesized summary"
) -> FindingDraft:
    return FindingDraft(
        summary=summary, confidence=0.8, used_source_indices=list(used)
    )


class FakeResearchTools:
    """In-memory ``ResearchTools`` stand-in -- a 'fake MCP server' for tests."""

    def __init__(self, results: list[SearchResult] | None = None) -> None:
        self._results = results if results is not None else make_search_results(2)
        self.searched: list[str] = []
        self.written: dict[str, str] = {}

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        self.searched.append(query)
        return list(self._results)

    async def write_file(self, path: str, content: str) -> None:
        self.written[path] = content
