# Observability

Two complementary views into a run: **traces** (what happened, nested) and
**metrics** (how much it cost).

## LangSmith traces

Set `LANGSMITH_TRACING=true` + `LANGSMITH_API_KEY` in `.env`. LangChain/LangGraph
runnables then export automatically — no per-call instrumentation. A single run
appears as one nested tree:

```
research run
└─ LangGraph
   ├─ planner   → LLM call
   ├─ researcher → search (MCP)  ·  write_file (MCP)  ·  LLM call   ← repeats if the loop fires
   ├─ critic    → LLM call
   └─ writer    → LLM call
```

What to look at:
- **The tree shape** — confirms the critic loop fired (researcher appears twice).
- **MCP tool calls nested under the researcher** — proves tools go through MCP.
- **Per-run metadata** — `_complete` stamps each run with `agent`, `latency_s`,
  `input_tokens`, `output_tokens`, `cost_usd`.

## Local metrics table

Every agent LLM call is recorded (`observability.record_llm_call`) with latency
and token usage; cost is estimated from the `PRICING` table. At the end of a CLI
run the per-agent aggregate is logged:

```
| Agent     | Calls | Latency (s) | In tok | Out tok | Cost (USD) |
|-----------|------:|------------:|-------:|--------:|-----------:|
| planner   |     1 |        1.42 |    310 |     180 |  $0.00015  |
| researcher|     2 |        4.10 |   2950 |     520 |  $0.00075  |
| critic    |     2 |        1.88 |   1400 |     240 |  $0.00036  |
| writer    |     1 |        2.55 |   1600 |     600 |  $0.00060  |
| total     |     6 |        9.95 |   6260 |    1540 |  $0.00186  |
```

(Illustrative numbers.) The report itself goes to stdout and the table to stderr,
so `research "..." > report.md` stays clean.

## Evals

`evals/run_eval.py` runs the golden set and scores each report with an
LLM-as-judge on a different model, writing a JSON report and markdown table to
`evals/results/`. See the [README](../README.md#evaluation) for how to run it.
