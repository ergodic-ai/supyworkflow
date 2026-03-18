# Observability

## Goal

Full visibility into what happens during workflow execution — every tool call, LLM call, and data flow — both for post-execution analysis and real-time streaming to the cardamon UI.

## Current State

### What we have
- `ExecutionTrace` with events: `cell_start`, `cell_end`, `error`
- `trace.summary()` for aggregate stats
- Python `logging` in tool proxy and `llm()` builtin
- Cell-level timing and status tracking
- Healer tracking (which cells were patched and how)

### The gap
**Tool calls and LLM calls inside cells are invisible to the trace.** The trace has `tool_call()` and `llm_call()` methods but they're never called — because tools and `llm()` execute inside `exec()` and only emit Python logging. The trace records "cell 2 took 15 seconds" but not "cell 2 made 3 tool calls and 1 LLM call totaling 14.8 seconds."

This means:
- `trace.summary()` always shows 0 tool calls and 0 llm calls
- No way to attribute cost or latency to specific operations
- No way to show real-time progress within a cell

## Plan

### Phase 1: Wire tool/LLM calls into the trace (do now)

**Approach: Thread-local trace context**

The runtime sets a thread-local `_current_trace` before executing each cell. Tool callables and `llm()` check for it and write events.

```python
# In runtime.py, before exec():
import threading
_trace_ctx = threading.local()
_trace_ctx.trace = trace
_trace_ctx.cell_index = cell.index

# In tool_proxy.py, inside the callable:
from supyworkflow._trace_ctx import get_current_trace
trace = get_current_trace()
if trace:
    trace.tool_call(provider=name, action=name, duration_ms=..., ok=True)

# In llm_builtin.py, after completion:
trace = get_current_trace()
if trace:
    trace.llm_call(model=model, duration_ms=..., prompt_tokens=..., completion_tokens=...)
```

This is lightweight (no wrapping, no global state beyond thread-local) and makes tool/LLM calls appear in the trace with correct cell attribution.

**Implementation:**
1. Create `supyworkflow/_trace_ctx.py` — thread-local trace holder with `set_trace()`, `get_trace()`, `get_cell_index()`
2. Update `runtime.py` — set trace context before each cell's `exec()`, clear after
3. Update `tool_proxy._make_tool_callable()` — write `tool_call` event to trace
4. Update `llm_builtin.llm()` — write `llm_call` event to trace
5. Update `trace.tool_call()` and `trace.llm_call()` to accept `cell_index` so events are attributed to the right cell

**Result:** `trace.summary()` will correctly report tool_calls and llm_calls. Each event is attributed to a cell.

### Phase 2: Callback-based streaming (do now)

Add an `on_event` callback to `runtime.run()`:

```python
result = rt.run(
    source,
    on_event=lambda event: print(f"[{event.event_type}] {event.cell_label}")
)
```

**Implementation:**
1. Add `on_event: Callable[[TraceEvent], None] | None = None` param to `run()`
2. After each `trace.append()`, call `on_event(event)` if set
3. The callback can do anything: SSE push, websocket, log, store to DB

**Wire it through the trace object:**
```python
class ExecutionTrace:
    def __init__(self, run_id, on_event=None):
        self.on_event = on_event

    def _emit(self, event):
        self.events.append(event)
        if self.on_event:
            self.on_event(event)
```

Then every `cell_start`, `cell_end`, `tool_call`, `llm_call`, `error` event streams in real time.

### Phase 3: Cost tracking (do now)

Add cost calculation to the trace summary using model-specific pricing.

```python
# Pricing table (per 1M tokens)
MODEL_PRICING = {
    "gemini/gemini-3.1-pro-preview": {"prompt": 1.25, "completion": 5.00},
    "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
    "gpt-4o": {"prompt": 2.50, "completion": 10.00},
    # ...
}
```

**Implementation:**
1. Store model name in each `llm_call` trace event (already done)
2. `trace.summary()` calculates cost per LLM call and total
3. Add `estimated_cost` to `RunResult`
4. For dry-run: count `llm()` calls in AST, estimate tokens from data size

### Phase 4: Rich trace serialization (do now)

Make the trace JSON-serializable for storage and API responses.

```python
trace.to_dict() → {
    "run_id": "abc123",
    "total_duration_ms": 15234,
    "events": [
        {"t": 0.0, "type": "cell_start", "cell": 0, "label": "Fetch emails"},
        {"t": 0.5, "type": "tool_call", "cell": 0, "data": {"action": "gmail_list_messages", "duration_ms": 480, "ok": true}},
        {"t": 1.2, "type": "cell_end", "cell": 0, "label": "Fetch emails", "data": {"status": "completed", "duration_ms": 1200}},
        {"t": 1.2, "type": "cell_start", "cell": 1, "label": "Analyze"},
        {"t": 1.3, "type": "llm_call", "cell": 1, "data": {"model": "gemini/...", "duration_ms": 8500, "prompt_tokens": 3200, "completion_tokens": 800, "cost": 0.008}},
        ...
    ],
    "summary": {
        "cells": 3, "tool_calls": 2, "llm_calls": 1,
        "total_cost": 0.012, "prompt_tokens": 3200, "completion_tokens": 800
    }
}
```

This is what cardamon's UI consumes to render the execution timeline.

### Phase 5: Agentic generation trace (do now)

The agentic generator also needs observability. `GenerateSession` already tracks tool calls, but in a different format from `ExecutionTrace`.

Unify: make the agent loop emit `TraceEvent`s too, so both generation and execution use the same trace format. Event types for the agent phase:
- `agent_tool_call` — schema lookup or tool execution during exploration
- `agent_write_script` — script committed
- `agent_turn` — each LLM turn in the loop

This lets the cardamon UI show a continuous timeline: exploration → generation → execution.

## What NOT to build now

- **Gantt chart UI** — that's a cardamon frontend feature, not supyworkflow. We just need to emit the right data.
- **Log aggregation** — Python logging is fine for debugging. The trace is the structured output.
- **Distributed tracing (OpenTelemetry)** — overkill until supyworkflow runs as a service. The trace object is sufficient.
- **Alerting** — belongs in cardamon's monitoring layer.

## Priority Order

1. **Thread-local trace context** (Phase 1) — fixes the "0 tool calls" bug, most impactful
2. **Callback streaming** (Phase 2) — enables real-time UI, small change
3. **Cost tracking** (Phase 3) — users need this, simple math
4. **Trace serialization** (Phase 4) — needed for cardamon integration
5. **Agent trace unification** (Phase 5) — nice to have, can wait

Phases 1-4 are all small, self-contained changes. Could be done in a single PR.
