"""Streamlit demo UI for the multi-agent research assistant.

A thin *presentation layer* over ``run_research_state`` -- no agent logic lives
here. It surfaces the full pipeline output (plan, findings, critique, report) plus
per-agent cost/latency metrics, which makes for self-explanatory screenshots.

Run:  uv run --extra demo streamlit run app.py
Needs OPENAI_API_KEY + TAVILY_API_KEY in .env (LANGSMITH_* optional for traces).
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import streamlit as st

from research_assistant.config import get_settings
from research_assistant.graph import DEFAULT_MAX_ITERATIONS, run_research_state
from research_assistant.observability import (
    configure_tracing,
    metrics_session,
    summarize_metrics,
)

DEFAULT_QUESTION = "What are the tradeoffs of multi-agent vs single-agent LLM systems?"


def _run_pipeline(question: str, max_iterations: int) -> tuple[dict, list]:
    """Run the async graph in a dedicated thread/loop (keeps MCP subprocesses
    happy regardless of Streamlit's script-runner thread)."""
    box: dict[str, Any] = {}

    def worker() -> None:
        async def _go() -> tuple[dict, list]:
            with metrics_session() as metrics:
                state = await run_research_state(question, max_iterations=max_iterations)
            return state, list(metrics)

        try:
            box["result"] = asyncio.run(_go())
        except Exception as e:  # surface to the UI instead of a blank failure
            box["error"] = e

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    if "error" in box:
        raise box["error"]
    return box["result"]


def _render_metric_cards(state: dict, metrics: list) -> None:
    totals = summarize_metrics(metrics)
    total = next((r for r in totals if r["agent"] == "total"), None)
    report = state["report"]
    cols = st.columns(5)
    cols[0].metric("Est. cost", f"${(total['cost_usd'] if total else 0):.4f}")
    cols[1].metric("Latency", f"{(total['latency_s'] if total else 0):.1f}s")
    tokens = (total["input_tokens"] + total["output_tokens"]) if total else 0
    cols[2].metric("Tokens", f"{tokens:,}")
    cols[3].metric("Critic loops", state.get("iteration", 0))
    cols[4].metric("Citations", len(report.citations))


def _render_report(state: dict) -> None:
    report = state["report"]
    st.markdown(report.body_markdown)
    if report.citations:
        with st.expander(f"Sources ({len(report.citations)})"):
            for c in report.citations:
                st.markdown(f"- [{c.title}]({c.url})")


def _render_plan(state: dict) -> None:
    plan = state["plan"]
    st.caption(f"Objective: {plan.objective}")
    for step in plan.steps:
        st.markdown(f"**{step.id}. {step.question}**")
        st.caption(step.rationale)
        if step.search_queries:
            st.caption("Queries: " + " · ".join(f"`{q}`" for q in step.search_queries))


def _render_findings(state: dict) -> None:
    for f in state["findings"]:
        st.markdown(f"**Step {f.step_id}** — confidence {f.confidence:.0%}")
        st.write(f.summary)
        if f.citations:
            st.caption("Cites: " + ", ".join(f"[{c.title}]({c.url})" for c in f.citations))
        st.divider()


def _render_critique(state: dict) -> None:
    crit = state["critique"]
    c1, c2 = st.columns(2)
    c1.metric("Coverage", f"{crit.coverage_score:.0%}")
    c2.metric("Citation quality", f"{crit.citation_quality:.0%}")
    st.write(crit.assessment)
    if crit.gaps:
        st.markdown("**Outstanding gaps (last cycle):**")
        for g in crit.gaps:
            st.markdown(f"- _{g.severity.value}_: {g.description}")
    else:
        st.success("No outstanding gaps — the critic signed off.")


def _render_metrics(metrics: list) -> None:
    rows = summarize_metrics(metrics)
    st.table(
        [
            {
                "Agent": r["agent"],
                "Calls": int(r["calls"]),
                "Latency (s)": round(r["latency_s"], 2),
                "In tok": int(r["input_tokens"]),
                "Out tok": int(r["output_tokens"]),
                "Cost (USD)": round(r["cost_usd"], 5),
            }
            for r in rows
        ]
    )
    st.caption("Per-agent metrics, captured every run. Full nested trace in LangSmith.")


def main() -> None:
    st.set_page_config(
        page_title="Multi-Agent Research Assistant", page_icon="🔎", layout="wide"
    )
    configure_tracing()
    settings = get_settings()

    st.title("🔎 Multi-Agent Research Assistant")
    st.caption(
        "Planner → Researcher → Critic → Writer · typed A2A messages over LangGraph "
        "· MCP web search + filesystem · LangSmith-traced"
    )

    with st.sidebar:
        st.subheader("Run settings")
        max_iterations = st.slider("Max critic loops", 1, 5, DEFAULT_MAX_ITERATIONS)
        st.caption(f"Model: `{settings.research_model}` · provider: `{settings.provider}`")
        st.caption(
            "Needs `OPENAI_API_KEY` + `TAVILY_API_KEY` in `.env`. "
            "Set `LANGSMITH_TRACING=true` for traces."
        )

    question = st.text_input("Research question", value=DEFAULT_QUESTION)
    run = st.button("Run research", type="primary")

    if "result" not in st.session_state:
        st.session_state.result = None

    if run and question.strip():
        if not (settings.openai_api_key and settings.tavily_api_key):
            st.error(
                "Missing API keys. Add OPENAI_API_KEY and TAVILY_API_KEY to .env."
            )
            return
        with st.spinner("Agents researching… (planning, searching, critiquing, writing)"):
            try:
                state, metrics = _run_pipeline(question.strip(), max_iterations)
                st.session_state.result = {"state": state, "metrics": metrics}
            except Exception as e:
                st.session_state.result = None
                st.exception(e)

    if st.session_state.result:
        state = st.session_state.result["state"]
        metrics = st.session_state.result["metrics"]
        _render_metric_cards(state, metrics)
        tabs = st.tabs(["📄 Report", "🧭 Plan", "🔍 Findings", "🧪 Critique", "📊 Metrics"])
        with tabs[0]:
            _render_report(state)
        with tabs[1]:
            _render_plan(state)
        with tabs[2]:
            _render_findings(state)
        with tabs[3]:
            _render_critique(state)
        with tabs[4]:
            _render_metrics(metrics)


main()
