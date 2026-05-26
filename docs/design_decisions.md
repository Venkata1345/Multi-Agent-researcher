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

## Phase 3

- **MCP over direct API wrappers.** Tools are separate stdio servers spoken to via
  the MCP protocol, not Tavily/filesystem calls embedded in the agent. Cost: extra
  process + serialization. Benefit: a uniform, swappable tool interface; capability
  scoping enforced at the server boundary; the same servers work with any MCP
  client (Inspector, Claude Desktop). See [mcp.md](mcp.md).
- **Deterministic search-per-step, not LLM-driven tool-calling.** The researcher
  code orchestrates `search` per step and feeds results to the LLM, rather than
  letting the model decide when to call tools. More predictable, cheaper, far
  easier to test/trace; a linear research pipeline doesn't need agentic tool loops.
- **Index-based citations.** The LLM returns indices into the retrieved source
  list; code maps them to `Citation` objects. Guarantees no hallucinated URLs.
- **Tools injected into the researcher.** Lets the integration test run against a
  fake and keeps the MCP lifecycle owned by `run_research`.

## To be written (Phase 4)

- (a) Why LangGraph over LangChain agents
- (c) What the critic loop trades off (quality vs. latency vs. cost)
