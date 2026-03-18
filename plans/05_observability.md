# Observability

## Goal

Real-time execution progress visible in the cardamon UI, with detailed traces for debugging.

## Current State

- `ExecutionTrace` captures events (cell_start, cell_end, tool_call, llm_call, error) with timestamps
- `trace.summary()` provides aggregate stats (cell count, tool calls, llm calls, total duration, token usage)
- Tool calls and LLM calls emit structured logs via `logging.getLogger("supyworkflow")`
- The trace is only available after execution completes — no real-time streaming

## Remaining Work

### 1. Log-derived workflow visualization

The trace already records every tool call and LLM call with timing. Build a post-execution visualizer that:
- Renders cells as a timeline (Gantt chart)
- Shows tool calls and LLM calls within each cell
- Highlights healed cells with before/after diff
- Shows data flow between cells (which variables passed from one to the next)

This can be built entirely from the trace data — no runtime changes needed.

### 2. Real-time progress streaming

For long-running workflows, the UI should show progress as it happens:

**Option A: Callback-based**
```python
def on_event(event: TraceEvent):
    send_sse(event)

rt.run(source, on_event=on_event)
```

**Option B: Async generator**
```python
async for event in rt.run_streaming(source):
    yield event
```

Recommendation: Option A (callback) — simpler, works with sync execution, callback can do anything (SSE, websocket, log, store).

### 3. Cost tracking

The trace captures prompt/completion tokens per LLM call. Add:
- Cost calculation per cell and per run (using model-specific pricing)
- Cumulative cost tracking across runs for an agent
- Cost estimate in dry-run (count LLM calls, estimate token usage)

### 4. Tool call logging

The tool proxy logs calls via Python `logging`, but the trace doesn't capture them because the logging happens inside the tool callable, not in the runtime loop. Options:
- Thread-local trace context that tool callables write to
- Wrap tool callables in the runtime to intercept and log
- Parse structured log output post-execution

Recommendation: Wrap tool callables in the runtime — cleanest, no global state.
