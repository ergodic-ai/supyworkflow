# Agentic Generator

## Goal

Replace the single-shot generator with an agentic loop where the LLM can explore tools, inspect schemas, execute tools to gather context, and only write the script when it has enough information.

## What We Built

### Architecture

The agent has 3 tools (down from 4 — `list_tools()` was removed):

1. **`get_tool_schemas(tool_names)`** — Batch lookup of parameter schemas for multiple tools in one call. Returns parameters, types, required fields.
2. **`execute_tool(tool_name, arguments, max_chars, offset)`** — Run any tool and see the result. Supports pagination via `offset`/`max_chars` for large responses.
3. **`write_script(script)`** — Commit the final workflow. Terminates the loop.

### Key Design Decisions

**Tool catalog seeded in system prompt.** All 248 tool names + descriptions are injected directly into the system prompt (~3-4k tokens). This eliminates the `list_tools()` call that every session used to start with. The agent can immediately identify which tools it needs and call `get_tool_schemas()`.

**Batch schema lookups.** Changed from `get_tool_schema(name)` (singular) to `get_tool_schemas(names)` (plural). The agent can fetch all schemas it needs in a single call instead of 3-6 sequential calls.

**Paginated execute_tool.** Large API responses (e.g., Slack channels, Drive files) are truncated with smart JSON boundary detection. The agent can use `offset` and `max_chars` to paginate if needed, or use a smaller `max_chars` for a quick peek.

**Full session history.** `GenerateSession` stores every message, tool call record, and the final script. Ready for persistence and replay.

**Efficiency guidance in prompt.** System prompt instructs the agent to avoid redundant calls and aim for <10 tool calls.

### What the Agent Does

Typical exploration pattern (observed from stress tests):
1. `get_tool_schemas(["gmail_list_messages", "calendar_list_events", "slack_send_message", ...])` — batch schema lookup
2. `execute_tool("slack_list_channels", {})` — discover real channel IDs
3. `execute_tool("calendar_list_events", {...})` — peek at today's events
4. `write_script(...)` — write script with real IDs and correct parameter names

More complex tasks (e.g., test A09 Financial Intel):
- 12 turns, 14 tool calls
- Searched SEC filings, patents, news
- Created test spreadsheet to verify API works
- Wrote a 6-cell workflow

## Test Results

### Simple Tests (run_agentic.py) — 3/3 passed

| Test | Turns | Tool Calls | Tokens | Result |
|------|-------|-----------|--------|--------|
| Slack message (needs channel discovery) | 5 | 5 | 32k | completed |
| Cross-service report (Gmail+Cal+Slack) | 7 | 12 | 62k | completed |
| Research + Spreadsheet | 10 | 17 | 120k | completed |

### Stress Tests (run_stress_agentic.py) — 9/10 passed

| Test | Services | Turns | Tool Calls | Tokens | Cells | Result |
|------|----------|-------|-----------|--------|-------|--------|
| A01 Morning Brief | Gmail+Cal+Slack+Linear+Docs | 9 | 11 | 127k | 8/8 | completed |
| A02 Meeting Intel | Cal+Search+News+Gmail | 5 | 5 | 54k | 4/4 | completed |
| A03 Competitive Radar | Search+News+Sheets+Slack+Gmail | 4 | 4 | 40k | 5/6 | **partial** |
| A04 Multi-Channel Slack | Slack+Docs+Gmail | 16 | 16 | 232k | 7/7 | completed |
| A05 Drive Cleanup | Drive+Sheets+Slack | 8 | 8 | 85k | 5/5 | completed |
| A06 Data Pipeline | Code+DB+Sheets+Slack | 11 | 11 | 139k | 6/6 | completed |
| A07 Sprint Review | Linear+Slack+Gmail | 7 | 7 | 71k | 4/4 | completed |
| A08 Research Paper | Search+Scholar+Docs+Sheets+Gmail | 7 | 7 | 73k | 7/7 | completed |
| A09 Financial Intel | SEC+Patents+News+Sheets+Gmail | 12 | 14 | 190k | 6/6 | completed |
| A10 Workspace Health | All services | 3 | 7 | 36k | 7/7 | completed |

## Issues Found

### 1. Redundant tool calls (A04 — 16 calls)

The agent called `slack_list_channels` 4 times because the response was truncated mid-JSON and it couldn't parse the result. **Fixed** by:
- Bumping default `max_chars` from 4000 → 8000
- Smart truncation at JSON object boundaries instead of mid-object
- Adding `offset`/`max_chars` params so the agent can paginate
- System prompt guidance to not re-call with same args

### 2. Nested Pydantic model flattening (A03 — partial)

The agent defined a `TopFindings` model nested inside `CompetitiveIntelligence`. Gemini sometimes flattens nested models in structured output, returning a string instead of the nested object. The healer tried to fix it but couldn't resolve the structural mismatch.

**Root cause**: Gemini's structured output (via schema-in-prompt) doesn't enforce nesting as reliably as constrained decoding. This is a model quality issue, not a supyworkflow bug.

**Mitigations**:
- Prefer flat Pydantic models over deeply nested ones in the system prompt guidance
- Could add a retry at the `llm()` level when Pydantic validation fails
- Could use `response_format` parameter for providers that support it (OpenAI) and fall back to schema-in-prompt for others (Gemini)

### 3. Token usage is high

A04 used 232k tokens for a 7-cell workflow. Most tokens come from large tool responses being included in the conversation history. **Mitigations**:
- Summarize tool results before adding to context (e.g., "Got 26 Slack channels, here are the names and IDs: ...")
- Implement context window management — drop old tool results once the agent has moved on
- Could offer a `compact` mode that strips verbose tool results after each turn

## Tool Call Record Fields

Each entry in `session.tool_calls_made` now contains:

| Field | Type | Description |
|-------|------|-------------|
| `tool` | string | Tool name (`get_tool_schemas`, `execute_tool`, `write_script`) |
| `args` | dict | Arguments passed to the tool |
| `timestamp` | float | Unix timestamp |
| `status` | string | `"ok"` or `"error"` |
| `result_length` | int | Length of the tool result string |
| `reasoning` | string? | The LLM's reasoning text from the assistant message for this turn (max 500 chars). Only present on the first tool call of a turn, and only when the model produces `content` alongside tool calls. |
| `result_preview` | string? | First 500 chars of the tool execution result. Not included for `write_script` calls. |
| `error` | string? | Error message if `status` is `"error"` (max 200 chars) |

The progress file (`--progress-file`) includes `reasoning` (truncated to 300 chars) in each `tool_calls` entry during the `"exploring"` phase, enabling real-time display of the agent's thought process in UIs.

## Remaining Work

### Merge to main

The agentic generator is on the `agentic-generator` branch. Before merging:
- Add unit tests for the agent loop (mock LLM responses)
- Add a session persistence layer (save/load GenerateSession to disk/DB)
- Consider making the agentic generator the default in the CLI

### Context window management

For complex workflows, the agent's message history grows large. Implement:
- Tool result summarization (replace full JSON with a compact summary after the agent has processed it)
- Max context budget with oldest-first eviction
- Token counting before each LLM call

### Session persistence

`GenerateSession` stores everything but only in memory. Add:
- Save to disk (JSON) after each turn for crash recovery
- Load a previous session to continue exploration
- Expose via CLI: `supyworkflow generate --resume <session_id>`

### Smarter tool result handling

Instead of raw JSON, post-process common tool results:
- Slack channels → table of name/ID/memberCount
- Calendar events → formatted list with times
- Search results → title + snippet (strip HTML/boilerplate)

This would dramatically reduce token usage.
