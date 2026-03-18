# supyworkflow — Project Summary

## What It Is

A "workflow-as-code" runtime that replaces Cardamon's rigid JSON workflow engine. Instead of generating JSON step definitions, the LLM generates a Python script that IS the workflow. The script calls tools (via cardamon's API) and uses LLM intelligence (via Pydantic structured output) — all with implicit observability and self-healing.

## Architecture

```
supyworkflow/
├── parser.py           — Cell splitting (# ---) + AST dependency graph
├── tool_proxy.py       — Routes flat tool calls to cardamon REST API with metadata
├── llm_builtin.py      — llm() function with Pydantic structured output via LiteLLM
├── namespace.py        — Sandboxed exec() namespace (blocklisted imports, injected tools)
├── snapshot.py         — Capture/restore namespace state for cell resumability
├── trace.py            — Structured execution logging (tool calls, llm calls, timing)
├── runtime.py          — Core engine: run(), dry_run(), resume, self-healing
├── healer.py           — LLM-powered cell repair on failure
├── generator.py        — Single-shot workflow generation from prompts
├── agent_generator.py  — Agentic generator with tool exploration loop (branch: agentic-generator)
└── cli.py              — CLI entry point (supyworkflow run/parse)
```

## Integration Point

- Takes a cardamon `SUPYAGENT_API_KEY` (admin key) + `user_id`
- Tool calls route through cardamon's REST API (`/api/v1/{provider}/{action}`)
- LLM calls go directly through LiteLLM (default: gemini/gemini-3.1-pro-preview)
- No direct dependency on supyagent or cardamon code — HTTP only

## Current Status

| Capability | Status | Details |
|-----------|--------|---------|
| Cell parser + AST deps | Done | 11 unit tests |
| Tool proxy (cardamon API) | Done | Flat callable routing with metadata + bodyDefaults |
| LLM builtin (Pydantic) | Done | Structured output via schema-in-prompt, no max_tokens |
| Sandboxed namespace | Done | Blocklist imports (not allowlist), injected tools + builtins |
| Runtime execution | Done | Sequential cells, snapshots, resume from cell |
| Self-healing | Done | LLM patches failed cells, up to 3 retries |
| Execution trace | Done | Structured event log |
| Single-shot generation | Done | Natural language → Python script |
| Agentic generation | Done | Agent explores tools before writing script (branch) |
| Iterative refinement | Done | Refine script from dry-run analysis or errors |
| CLI | Done | run, parse commands |
| Unit tests | Done | 36 passing (parser, runtime, healer, generator) |
| E2E tests | Done | 6 tests with real Gemini API, all passing |
| Live examples | Done | 5 hand-written, 5 single-shot generated, all passing |
| Stress tests (single-shot) | Done | 10/10 passing across 8+ services, 60/60 cells |
| Stress tests (agentic) | Done | 9/10 passing, 1 partial (Pydantic nested model edge case) |

## Repository

- **GitHub**: https://github.com/ergodic-ai/supyworkflow
- **Main branch**: core runtime + single-shot generator + all tests
- **agentic-generator branch**: agentic generator with tool exploration

## Remaining Work

See individual plan files for details:

1. [Cardamon integration](01_cardamon_integration.md) — Wire into cardamon's execution pipeline
2. [Script versioning](02_script_versioning.md) — Persist healed scripts, track history
3. [Parallel cells](03_parallel_cells.md) — Concurrent execution of independent cells
4. [Tool discovery improvements](04_tool_discovery.md) — Better dry-run validation, action schemas
5. [Observability](05_observability.md) — Real-time progress, trace → UI integration
6. [Testing hardening](06_testing.md) — Edge cases, error scenarios, live tool tests
7. [Agentic generator](07_agentic_generator.md) — Findings, improvements, merge plan
