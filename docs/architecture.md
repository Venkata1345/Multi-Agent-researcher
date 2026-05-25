# Architecture

_Filled out in Phase 4 for an interviewer reading the repo. Phase 1 stub below._

## Overview

A LangGraph state machine coordinates four specialist agents. Each agent has one
responsibility and a single typed input/output (see `src/research_assistant/messages.py`).
State flows as Pydantic A2A messages; the critic can loop back to the researcher
to close gaps, capped at 3 cycles.

## Components

- **Planner** (`src/research_assistant/agents/planner.py`) — decomposes the question into a `ResearchPlan`.
- **Researcher** (`src/research_assistant/agents/researcher.py`) — executes steps, returns `ResearchFindings`.
- **Critic** (`src/research_assistant/agents/critic.py`) — scores coverage/citations, emits `Gap`s.
- **Writer** (`src/research_assistant/agents/writer.py`) — produces the final `Report`.
- **Graph** (`src/research_assistant/graph.py`) — wiring + conditional critic loop.
