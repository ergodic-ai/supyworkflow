# supyworkflow — Project Summary

## What It Is

A "workflow-as-code" runtime that replaces Cardamon's rigid JSON workflow engine. The LLM generates a Python script that IS the workflow — calling tools via supyagent's API and using Pydantic structured output for LLM intelligence. Fully integrated into cardamon with a builder UI.

## Architecture

```
supyworkflow/
├── parser.py           — Cell splitting (# ---) + AST dependency graph
├── tool_proxy.py       — Routes flat tool calls to supyagent REST API + X-Account-Id scoping
├── llm_builtin.py      — llm() function with Pydantic structured output via LiteLLM
├── namespace.py        — Sandboxed exec() namespace (blocklisted imports, injected tools)
├── snapshot.py         — Capture/restore namespace state for cell resumability
├── trace.py            — Structured execution logging with cost tracking + callbacks
├── runtime.py          — Core engine: run(), dry_run(), resume, self-healing
├── healer.py           — LLM-powered cell repair on failure
├── generator.py        — Single-shot workflow generation from prompts
├── agent_generator.py  — Agentic generator with tool exploration + progress file polling
├── validator.py        — Script validation (compile check + forward reference detection)
└── cli.py              — CLI: run, generate, parse (all with --output-format json)
```

## Integration

- **supyagent API** for tool discovery and execution (X-Account-Id for per-user scoping)
- **LiteLLM** for LLM calls (default: gemini/gemini-3.1-pro-preview)
- **Cardamon** calls supyworkflow via subprocess for both generation and execution
- **Builder UI** in cardamon shows Steps View / Code View with live generation progress

## Current Status

| Capability | Status |
|-----------|--------|
| Core runtime (parser, executor, healer, trace) | Done |
| Tool proxy with X-Account-Id scoping | Done |
| LLM builtin with Pydantic structured output | Done |
| Self-healing (up to 3 retries) | Done |
| Observability (trace context, callbacks, cost tracking) | Done |
| Single-shot generator | Done |
| Agentic generator with exploration | Done |
| Script validation (compile + forward refs) | Done |
| Generation polling with progress file | Done |
| CLI with JSON output | Done |
| Cardamon integration (Phase 1 + Phase 2) | Done |
| Stress tests (10/10 through cardamon API) | Done |

## Repository

- **GitHub**: https://github.com/ergodic-ai/supyworkflow
- **Version**: 0.1.6
- **Install**: `uv tool install --from <path> supyworkflow`

## Remaining Work

See individual plan files. Key items:
- Script versioning (persist healed scripts)
- Parallel cell execution
- Generation speed optimization (fewer turns, cached schemas)
- Context window management for agentic generator
