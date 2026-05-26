"""Eval harness: run the golden set through the pipeline and score with an
LLM-as-judge running on a *different* model than the one that produced the answer.

Usage:
    uv run python evals/run_eval.py            # run all questions
    uv run python evals/run_eval.py --limit 3  # smoke-test on the first 3

Writes a timestamped JSON report and a markdown summary table to evals/results/.
Requires OPENAI_API_KEY and TAVILY_API_KEY (it runs the real pipeline).

Scores (all 0-1; higher is better except ``hallucination``):
  relevance        - does the report answer the question asked?
  coverage         - are the expected key points covered?
  citation_quality - are claims backed by relevant, real citations?
  hallucination    - unsupported/fabricated claims (0 = fully grounded, 1 = severe)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from research_assistant.agents.base import build_chat_model
from research_assistant.config import get_settings
from research_assistant.graph import run_research
from research_assistant.observability import (
    configure_logging,
    configure_tracing,
    get_logger,
    metrics_session,
)

_log = get_logger("research_assistant.eval")

GOLDEN_SET = Path(__file__).parent / "golden_set.jsonl"
RESULTS_DIR = Path(__file__).parent / "results"

JUDGE_SYSTEM = """You are a strict evaluator of research reports. Given a question,
a list of expected key points, and a generated report, score the report on four
0-1 dimensions:
- relevance: does it answer the question asked?
- coverage: how many expected key points are addressed?
- citation_quality: are claims supported by relevant, real-looking citations?
- hallucination: unsupported or fabricated claims (0 = fully grounded, 1 = severe).
Be calibrated and critical. Briefly justify in `rationale`."""


class JudgeScores(BaseModel):
    relevance: float = Field(ge=0.0, le=1.0)
    coverage: float = Field(ge=0.0, le=1.0)
    citation_quality: float = Field(ge=0.0, le=1.0)
    hallucination: float = Field(ge=0.0, le=1.0)
    rationale: str


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested offline)
# --------------------------------------------------------------------------- #
def load_golden_set(path: Path = GOLDEN_SET) -> list[dict[str, Any]]:
    """Parse the JSONL golden set into a list of question records."""
    items: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def composite_score(scores: dict[str, float]) -> float:
    """Single 0-1 summary: mean of the three 'higher-better' dims and groundedness."""
    return statistics.mean(
        [
            scores["relevance"],
            scores["coverage"],
            scores["citation_quality"],
            1.0 - scores["hallucination"],
        ]
    )


def aggregate(records: list[dict[str, Any]]) -> dict[str, float]:
    """Mean of each scored dimension across successfully-judged records."""
    scored = [r["scores"] for r in records if r.get("scores")]
    if not scored:
        return {}
    dims = ["relevance", "coverage", "citation_quality", "hallucination"]
    agg = {d: round(statistics.mean(s[d] for s in scored), 3) for d in dims}
    agg["composite"] = round(statistics.mean(composite_score(s) for s in scored), 3)
    return agg


def render_markdown(records: list[dict[str, Any]], agg: dict[str, float]) -> str:
    """Markdown summary table over per-question scores + an aggregate row."""
    rows = [
        "| Question | Rel | Cov | Cite | Halluc | Composite | Cost |",
        "|----------|----:|----:|-----:|-------:|----------:|-----:|",
    ]
    for r in records:
        s = r.get("scores")
        cost = r.get("cost_usd", 0.0)
        if not s:
            rows.append(f"| {r['id']} | — | — | — | — | _error_ | ${cost:.4f} |")
            continue
        rows.append(
            f"| {r['id']} | {s['relevance']:.2f} | {s['coverage']:.2f} | "
            f"{s['citation_quality']:.2f} | {s['hallucination']:.2f} | "
            f"{composite_score(s):.2f} | ${cost:.4f} |"
        )
    if agg:
        rows.append(
            f"| **mean** | **{agg['relevance']:.2f}** | **{agg['coverage']:.2f}** | "
            f"**{agg['citation_quality']:.2f}** | **{agg['hallucination']:.2f}** | "
            f"**{agg['composite']:.2f}** | |"
        )
    return "\n".join(rows)


# --------------------------------------------------------------------------- #
# Live evaluation (needs API keys)
# --------------------------------------------------------------------------- #
async def _judge(item: dict[str, Any], report_md: str, citations: int) -> JudgeScores:
    settings = get_settings()
    judge = build_chat_model(settings, model=settings.judge_model)
    structured = judge.with_structured_output(JudgeScores)
    user = (
        f"QUESTION:\n{item['question']}\n\n"
        f"EXPECTED KEY POINTS:\n- " + "\n- ".join(item["expected_points"]) + "\n\n"
        f"CITATION COUNT: {citations}\n\n"
        f"REPORT:\n{report_md}"
    )
    return await structured.ainvoke(
        [SystemMessage(content=JUDGE_SYSTEM), HumanMessage(content=user)]
    )


async def evaluate_one(item: dict[str, Any]) -> dict[str, Any]:
    record: dict[str, Any] = {"id": item["id"], "question": item["question"]}
    try:
        with metrics_session() as metrics:
            report = await run_research(item["question"])
        citations = len(report.citations)
        scores = await _judge(item, report.body_markdown, citations)
        record.update(
            scores=scores.model_dump(),
            n_citations=citations,
            cost_usd=round(sum(m.cost_usd for m in metrics), 5),
            latency_s=round(sum(m.latency_s for m in metrics), 2),
        )
        _log.info("scored %s: composite=%.2f", item["id"], composite_score(record["scores"]))
    except Exception as e:  # keep going; one bad question shouldn't sink the run
        _log.exception("eval failed for %s", item["id"])
        record["error"] = f"{type(e).__name__}: {e}"
    return record


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the research eval suite.")
    parser.add_argument("--limit", type=int, default=None, help="Only the first N.")
    args = parser.parse_args(argv)

    configure_logging()
    configure_tracing()

    items = load_golden_set()
    if args.limit:
        items = items[: args.limit]

    records = [await evaluate_one(item) for item in items]
    agg = aggregate(records)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    report = {"timestamp": stamp, "aggregate": agg, "records": records}
    (RESULTS_DIR / f"eval-{stamp}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    md = f"# Eval results — {stamp}\n\n{render_markdown(records, agg)}\n"
    (RESULTS_DIR / f"eval-{stamp}.md").write_text(md, encoding="utf-8")
    (RESULTS_DIR / "latest.md").write_text(md, encoding="utf-8")

    _log.info("Aggregate: %s", agg)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
