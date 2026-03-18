# Testing Hardening

## Goal

Robust test coverage for edge cases, error scenarios, and live tool interactions.

## Current State

- 36 unit tests passing (parser, runtime, healer, generator)
- 6 E2E tests with real Gemini API (all passing)
- 5 live example workflows (3 fully working, 2 dry-run only due to needing real IDs)
- Unit tests mock LLM and API calls — no real network in CI

## Remaining Work

### 1. Unit test gaps

- **Snapshot serialization edge cases**: non-JSON-serializable objects, circular references, very large data
- **Healing retry exhaustion**: verify behavior when all 3 heal attempts fail
- **Healing of healed code**: what happens when the patched code also fails
- **Timeout handling**: cell exceeds time budget mid-execution
- **Resume from snapshot**: more complex scenarios with multiple dependency chains
- **Namespace isolation**: verify that cells can't escape the sandbox (eval, exec, __import__ tricks)
- **Parser edge cases**: scripts with no cell markers, cells with only comments, unicode in cell labels

### 2. Security tests

- Verify blocked imports actually block (os, subprocess, sys, socket, etc.)
- Verify that `__builtins__` can't be tampered with from within a cell
- Verify that tool callables can't be replaced or monkey-patched
- Test that `exec()` within a cell is contained (can it access the outer namespace?)

### 3. Live integration tests

A test suite that runs against the real supyagent API (gated by env var):
- Gmail: list → read → (don't send in tests)
- Calendar: list events → verify structure
- Search: web search → verify results
- Sheets: create → write → read → delete
- Full pipeline: generate → dry-run → execute → verify outputs

### 4. Regression tests for self-healing

Curated set of scripts with known bugs and their expected fixes:
- KeyError (wrong dict key)
- TypeError (wrong argument type to tool)
- AttributeError (wrong Pydantic field name)
- Tool call with wrong parameter name (e.g., `q` vs `query`)
- Pydantic model missing required field

Each test runs the buggy script, verifies healing succeeded, and checks the patched code.

### 5. CI pipeline

- Unit tests run on every PR (no API keys needed)
- E2E tests run nightly or on-demand (requires GEMINI_API_KEY)
- Live integration tests run weekly (requires SUPYAGENT_API_KEY)
