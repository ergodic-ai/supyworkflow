# Cardamon Integration

## Goal

Replace (or sit alongside) cardamon's JSON workflow executor with supyworkflow so agents generate Python scripts instead of JSON step arrays.

## Current State

- supyworkflow is a standalone Python package
- Communicates with cardamon only via HTTP (tool calls through `/api/v1/` routes)
- No code-level integration yet

## Remaining Work

### 1. Execution endpoint in cardamon

Add a new API route or mode flag to cardamon's agent execution pipeline:

```
POST /api/v1/agents/[id]/run
Body: { mode: "workflow-as-code", inputs: {...} }
```

When `mode === "workflow-as-code"`:
- Load the agent's Python script (instead of JSON workflow)
- Call supyworkflow via subprocess or HTTP
- Stream back cell execution progress
- Store result in AgentRun table

Options:
- **Option A: subprocess** — cardamon shells out to `supyworkflow run <script> --api-key ... --user-id ...`. Simple, isolated, but no streaming.
- **Option B: HTTP service** — supyworkflow runs as a sidecar with a simple API. Cardamon POSTs the script, gets back results. Supports streaming via SSE.
- **Option C: Python SDK** — Cardamon's TypeScript calls supyworkflow via a thin Python bridge. Most complex.

Recommendation: **Option A to start** (subprocess), migrate to Option B for production streaming.

### 2. Agent spec → Python script generation

Currently cardamon generates agent specs → JSON workflows via `step-based-planner.ts`. Need to:
- Add a path where spec → Python script via `supyworkflow.generator.generate_workflow()`
- This can be called from cardamon's TypeScript via subprocess: `supyworkflow generate --prompt "..." --tools gmail,slack`
- Or via an HTTP call to a supyworkflow generation endpoint

### 3. Database schema changes

- Add `workflow_type` field to Agent model: `"json" | "python"`
- Store Python scripts in blob storage (same as JSON workflows)
- AgentRun logs should handle cell-based trace format

### 4. Builder UI updates

- The agent builder wizard needs a path for workflow-as-code agents
- Show generated Python script with syntax highlighting
- Allow editing individual cells
- Show cell-by-cell execution progress

### 5. Scheduling

- Scheduler needs to handle Python workflow agents
- Same cron/schedule model, different executor path
