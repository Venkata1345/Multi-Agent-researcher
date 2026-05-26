# Design Decisions

Why the project is built the way it is. Each entry is the **decision**, the
**reason**, and the **trade-off** accepted — recorded as choices are made.

> Looking for the *what* instead of the *why*? See
> [architecture.md](architecture.md) and [mcp.md](mcp.md).

---

## At a glance

| # | Area | Decision | One-line reason |
|---|------|----------|-----------------|
| 1 | Packaging | `src/research_assistant/` layout | Clean imports, no namespace pollution |
| 2 | A2A boundary | Typed `*Input`/output messages | No untyped dict ever crosses an agent |
| 3 | State | Researcher owns accumulation | Graph nodes stay thin and stateless |
| 4 | Loop safety | Critic decides, graph caps | Loop reacts to data but can't run away |
| 5 | LLM client | `ChatOpenAI.with_structured_output` | One-line provider swap + free tracing |
| 6 | Structured output | `function_calling` (not strict JSON) | Tolerates our full schema (`HttpUrl`) |
| 7 | Prompts | System prompts in `.md` files | Iterate on prose without touching code |
| 8 | Tools | MCP servers, not direct API calls | Uniform, scoped, swappable tool layer |
| 9 | Retrieval | Deterministic search-per-step | Predictable, cheap, easy to test/trace |
| 10 | Citations | LLM returns source *indices* | Impossible to hallucinate a URL |
| 11 | Testability | Tools injected into researcher | Integration tests run against a fake |

---

## Phase 1 — Skeleton & contracts

### 1. `src/research_assistant/` src-layout package
**Why:** imports read as `from research_assistant.messages import ...` and the
console script is `research = "research_assistant.cli:main"`.
**Trade-off:** started as a flat `src/` package (matching the brief's literal
paths), then refactored to a named package before Phase 2 to avoid polluting the
top-level import namespace as the code grew.

### 2. Typed agent inputs (`*Input` wrappers)
**Why:** `BaseAgent[InputT, OutputT]` enforces exactly one typed object in and
one out, so no untyped dict ever crosses an agent boundary.
**Trade-off:** a few extra small Pydantic models, in exchange for a fully typed
A2A contract.

### 3. Researcher owns accumulation
**Why:** the researcher returns the *complete* `ResearchFindings` each call
(prior + new), so graph nodes just replace state instead of merging it.
**Trade-off:** the researcher carries a little more responsibility; the graph
stays thin and stateless.

### 4. Critic decides gaps, the graph caps the loop
**Why:** the critic judges whether to loop (emit gaps); the graph independently
enforces a hard `max_iterations` cap.
**Trade-off:** two places touch loop control — but separating *intent* (agent)
from *budget* (graph) means even a buggy critic can never loop forever.

---

## Phase 2 — Real LLM calls

### 5. `ChatOpenAI.with_structured_output`, not the raw OpenAI SDK
**Why:** swapping OpenAI → Anthropic becomes a one-line change in
`build_chat_model`, and LangChain/LangGraph runnables **auto-trace to LangSmith**
with no extra instrumentation.
**Trade-off:** a dependency on `langchain-openai`. (This is still pure LangGraph
orchestration — *not* LangChain "agents".)

### 6. Structured output via `function_calling`, not strict `json_schema`
**Why:** `Citation.url` is an `HttpUrl` (`format: "uri"`), which OpenAI's strict
`json_schema` mode rejects. `function_calling` still returns a Pydantic-validated
object from the same schema and works with every model we have.
**Trade-off:** not literally the `response_format` API. It's a one-line config
flag (`structured_output_method`) — flip to `json_schema` once schemas are
constrained to the strict-mode subset.

### 7. System prompts live in `.md` files
**Why:** prose is iterable without code changes; user messages are built in code
from typed inputs (type-safe, no fragile string templating).
**Trade-off:** prompts load at runtime via `importlib.resources` instead of being
inline.

---

## Phase 3 — MCP tools & citations

### 8. MCP servers over direct API wrappers
**Why:** tools are separate stdio servers spoken to via the MCP protocol — a
uniform, swappable interface with capability scoping enforced at the server
boundary, usable by any MCP client (Inspector, Claude Desktop). See
[mcp.md](mcp.md).
**Trade-off:** an extra process and a serialization hop per tool call.

### 9. Deterministic search-per-step, not LLM-driven tool-calling
**Why:** the researcher code orchestrates `search` per step and feeds results to
the LLM, instead of letting the model decide when to call tools. More
predictable, cheaper, and far easier to test and trace.
**Trade-off:** less "agentic" flexibility — which a linear research pipeline
doesn't need.

### 10. Index-based citations
**Why:** the LLM returns *indices* into the retrieved source list; code maps them
to `Citation` objects. A cited URL is therefore always one search actually
returned — hallucinated sources are structurally impossible.
**Trade-off:** the model can't cite a source that wasn't retrieved (acceptable,
and arguably correct).

### 11. Tools injected into the researcher
**Why:** the integration test runs against an in-memory fake, and the MCP
subprocess lifecycle stays owned by `run_research`.
**Trade-off:** the researcher's constructor takes a required `tools` argument.

---

## Still to write (Phase 4)

- Why LangGraph over LangChain agents
- What the critic loop trades off: quality vs. latency vs. cost
