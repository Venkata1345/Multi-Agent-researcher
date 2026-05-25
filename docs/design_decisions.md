# Design Decisions

_Tradeoffs, expanded in Phase 4. Captured here as they are made._

## Phase 1

- **`src/research_assistant/` src-layout package.** Imports are
  `from research_assistant.messages import ...` and the console script is
  `research = "research_assistant.cli:main"`. (Started as a flat `src/` package
  matching the brief's literal paths; refactored to a named package before
  Phase 2 to avoid polluting the import namespace as code volume grows.)
- **Typed agent inputs (`*Input` wrappers).** No untyped dict ever crosses an
  agent boundary; `BaseAgent[InputT, OutputT]` enforces a single typed
  in/out per agent.
- **Researcher owns accumulation.** It returns the *complete* `ResearchFindings`
  each call (prior + new), so graph nodes stay thin and stateless.
- **Critic loop driven by data, capped by config.** The stub critic decides gaps
  from the findings/steps relationship; the graph independently enforces a hard
  `max_iterations` cap so the loop can never run away.

## To be written (Phase 4)

- (a) Why LangGraph over LangChain agents
- (b) Why MCP over direct tool wrappers
- (c) What the critic loop trades off (quality vs. latency vs. cost)
