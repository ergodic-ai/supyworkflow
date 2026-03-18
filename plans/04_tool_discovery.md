# Tool Discovery Improvements

## Goal

Better validation at dry-run time, richer tool information for the generator, and elimination of false-positive warnings.

## Current State

- `build_tool_callables()` fetches tool metadata from `/api/v1/tools` and builds callable functions
- `dry_run()` does AST analysis but flags builtins (len, min, max) and Pydantic types (BaseModel) as unresolved references
- The generator knows tool names but not their full parameter schemas
- No validation that a tool action exists before execution

## Remaining Work

### 1. Fix dry-run false positives

The dry-run warns about variables like `len`, `min`, `str`, `list`, `BaseModel` because they're not in any cell's `writes` set. Fix by maintaining a set of "known globals" that includes:
- All SAFE_BUILTINS keys
- All BLOCKED_IMPORTS (to distinguish "blocked" from "unknown")
- All injected tool function names
- `llm`, `BaseModel`, `Field`

Then `unresolved = cell.depends_on - all_writes - known_globals`.

### 2. Parameter schema validation in dry-run

At dry-run time, we have the tool metadata (including parameter schemas). We can AST-parse the tool calls and check:
- Does the tool function exist?
- Are required parameters provided?
- Are parameter names spelled correctly?

This catches errors before execution without making any API calls.

### 3. Richer tool info for generator

Currently the generator gets:
```python
tools=["gmail", "slack"]
tool_actions={"gmail": ["list_messages", "send_email"]}
```

Could be enhanced to include parameter schemas:
```python
tool_actions={
    "gmail_list_messages": {"q": "string (Gmail search query)", "maxResults": "integer"},
    "gmail_send_message": {"to": "string (required)", "subject": "string", "body": "string"},
}
```

This helps the LLM generate correct tool calls on the first try, reducing healing.

### 4. Tool action caching

Currently every `SupyWorkflow()` instantiation fetches tool metadata from the API. Add a TTL cache (e.g., 5 minutes) to avoid redundant HTTP calls across multiple runs.
