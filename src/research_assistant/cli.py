"""CLI entry point: ``research "your question"``.

Thin sync wrapper that parses args, runs the async graph to completion, and
prints the final markdown report to stdout.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from research_assistant.graph import DEFAULT_MAX_ITERATIONS, run_research
from research_assistant.observability import configure_logging, configure_tracing


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="research",
        description="Multi-agent research assistant.",
    )
    parser.add_argument("question", help="The research question to investigate.")
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        help="Max critic<->researcher loop cycles (default: %(default)s).",
    )
    return parser.parse_args(argv)


async def _amain(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    configure_logging()
    configure_tracing()  # enables LangSmith auto-tracing if configured in .env
    report = await run_research(args.question, max_iterations=args.max_iterations)
    sys.stdout.write(report.body_markdown + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Console-script entry point (see ``[project.scripts]`` in pyproject)."""
    return asyncio.run(_amain(argv))


if __name__ == "__main__":
    raise SystemExit(main())
