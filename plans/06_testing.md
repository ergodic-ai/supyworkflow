# Testing Hardening

## Goal

Robust test coverage for edge cases, error scenarios, and live tool interactions.

## Current State

### Unit tests — 36 passing
- Parser: cell splitting, AST dependencies, dependency graphs (11 tests)
- Runtime: execution, failure propagation, resume, sandboxing (10 tests)
- Healer: heal success, skip unrecoverable, max retries (7 tests)
- Generator: basic generation, markdown stripping, prompts (8 tests)

### E2E tests — 6 passing
All use real Gemini API:
- Pure Python workflow (no tools)
- LLM structured output with Pydantic
- Multi-step LLM workflow with data flow
- Self-healing (intentional KeyError bug)
- Workflow generation from natural language
- Generate → dry-run → execute pipeline

### Live examples — 5 hand-written, all passing
- Email digest → Slack
- Calendar briefing
- Research report → Google Sheets
- Competitor monitor → Email
- Slack standup summary

### Single-shot generated — 5 LLM-generated, all passing
- Email triage with urgency alerts
- Meeting prep with per-meeting research
- Industry news digest
- Competitive landscape comparison
- Slack channel analytics

### Stress tests (single-shot) — 10/10 passing
Cross-platform digest, research pipeline, Linear+Slack, Drive audit, competitor deep dive, data pipeline (code+DB), Slack→Linear tickets, newsletter, financial research (SEC+patents), full workspace sync (8 services, 10 cells).

### Stress tests (agentic) — 9/10 passing
Same complexity but agent discovers everything autonomously. Only failure: nested Pydantic model edge case with Gemini (A03 partial — 5/6 cells worked).

## Lessons Learned from Testing

### Import sandbox
- Allowlist approach for imports was a mistake — transitive deps of allowed modules (e.g., `datetime` imports `time` internally) caused failures. Switched to blocklist.
- `__build_class__` must be in builtins for Pydantic model definitions to work.

### Tool proxy
- `bodyDefaults` in tool metadata must be merged into request bodies (Google Docs actions).
- PUT method was missing — Google Sheets uses it.
- Truncating API responses mid-JSON confuses the LLM — need to cut at object boundaries.

### LLM structured output
- Gemini needs schema-in-prompt approach (not `response_format` parameter).
- No `max_tokens` — Gemini counts reasoning tokens in the budget, causing truncation.
- Nested Pydantic models sometimes get flattened by Gemini. Prefer flat models.

### Self-healing
- Healed 12+ cells across stress tests, mostly: wrong dict keys, Pydantic model issues, wrong parameter names.
- Never needed more than 1-2 attempts.
- Unrecoverable errors (ImportError, SyntaxError) correctly skipped.

## Remaining Work

### Unit test gaps
- Snapshot serialization edge cases
- Healing retry exhaustion + healing of healed code
- Timeout handling mid-cell
- Complex resume scenarios
- Namespace escape attempts (security)
- Parser edge cases (no markers, unicode labels)

### Security tests
- Verify all blocklisted imports are blocked
- Verify `__builtins__` can't be tampered with
- Verify tool callables can't be replaced
- Test `exec()` containment

### Agentic generator tests
- Mock LLM responses to test the agent loop
- Test max_turns enforcement
- Test session serialization/deserialization
- Test `execute_tool` pagination (offset/max_chars)

### CI pipeline
- Unit tests: every PR (no API keys)
- E2E tests: nightly (GEMINI_API_KEY)
- Live integration: weekly (SUPYAGENT_API_KEY)
