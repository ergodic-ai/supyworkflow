# supyworkflow — Project Summary

**Last updated:** 2026-03-19
**Version:** 0.1.6
**GitHub:** https://github.com/ergodic-ai/supyworkflow

## What It Is

A "workflow-as-code" runtime that replaces JSON workflow execution with Python scripts. The LLM generates a Python script that IS the workflow — calling tools via supyagent's API and using Pydantic structured output for LLM intelligence. Fully integrated into cardamon with a builder UI, agent detail page, and chat-based refinement.

## Architecture

```
supyworkflow/
├── parser.py           — Cell splitting (# ---) + AST dependency graph
├── tool_proxy.py       — Flat tool callables → supyagent REST API + X-Account-Id
├── llm_builtin.py      — llm() with Pydantic structured output via LiteLLM
├── namespace.py        — Sandboxed exec() with blocklisted imports + injected tools
├── snapshot.py         — Capture/restore namespace for cell resumability
├── trace.py            — Structured logging + cost tracking + callback streaming
├── _trace_ctx.py       — Thread-local trace context for tool/LLM attribution
├── runtime.py          — Core engine: run(), dry_run(), resume, self-healing
├── healer.py           — LLM-powered cell repair (up to 3 retries)
├── generator.py        — Single-shot workflow generation
├── agent_generator.py  — Agentic generator with exploration + progress polling
├── validator.py        — Compile check + forward reference detection
└── cli.py              — run, generate, parse (all with --output-format json)
```

## Integration with Cardamon

- Cardamon calls supyworkflow via **subprocess** for generation and execution
- Tools are accessed via **supyagent API** with `X-Account-Id` for per-user scoping
- Generation uses **background jobs** with progress file polling (2s intervals)
- Results map to cardamon's **ExecutionLog format** (existing UI works)
- Builder has **Steps View** (card-based, default) + **Code View** (toggle)
- Agent detail has **chat-based refinement** for iterative script editing

## Current Status

Everything below is done and tested:

| Capability | Details |
|-----------|---------|
| Core runtime | Parser, executor, snapshots, resume, self-healing |
| Tool proxy | Flat callables, bodyDefaults, X-Account-Id on all requests |
| LLM builtin | Pydantic structured output, no max_tokens, Gemini default |
| Observability | Thread-local trace context, callback streaming, cost tracking |
| Agentic generator | Tool exploration, progress file, script validation, user_id scoping |
| Script validation | Compile check + forward reference detection |
| Cardamon integration | Executor routing, generation polling, trace mapping, retry logic |
| Builder UI | Steps View, Code View, exploration trace, test run, generation polling |
| Agent detail page | Workflow tab, inline editing, chat refinement, simplified menu |
| Stress tests | 10/10 through cardamon API (5 parallel, 9 services) |

## Open Items

### In supyworkflow:

1. **Generation speed** — Cap turns at 10, cache schemas, summarize tool results
2. **Script versioning** — Persist healed scripts as new versions
3. **Parallel cells** — Run independent cells concurrently via ThreadPoolExecutor
4. **Context window management** — Summarize tool results to reduce token usage
5. **Refine CLI command** — Currently refinement calls LLM directly from cardamon; could add a proper `supyworkflow refine` command

### In cardamon:

1. **Run detail page** — Show cell-by-cell trace with tool/LLM calls per cell
2. **Error handling** — Better messages in builder (timeout, rate limit, partial results)
3. **Refinement chat history** — Multi-turn conversation for complex changes
4. **Workflow diff view** — Visual diff when accepting refinement
5. **Python agent templates** — Pre-built scripts for common use cases
