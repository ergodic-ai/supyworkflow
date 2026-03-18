# Tool Discovery Improvements

## Goal

Better validation at dry-run time, richer tool information for the generator, and elimination of false-positive warnings.

## What Changed Since Initial Plan

### bodyDefaults fix (done)

The tool metadata includes a `bodyDefaults` field (e.g., `{"action": "insert_text"}` for Google Docs). The tool proxy now merges these into the request body automatically. This fixed all `docs_insert_text` failures across stress tests.

### Import blocklist instead of allowlist (done)

Switched from an allowlist of safe imports to a blocklist of dangerous ones. This fixed the `datetime` → `time` transitive dependency issue and similar problems where allowed modules import internal modules that weren't in the allowlist.

### Flat tool naming (done)

Tool proxy was rewritten from the `ToolProxy.__getattr__` pattern (e.g., `gmail.send_email()`) to flat callables (e.g., `gmail_send_message()`), matching the actual supyagent API naming. This eliminated a whole class of naming mismatches.

### HTTP method support (done)

Added PUT method support to the tool proxy. Google Sheets `update_values` uses PUT, which was missing.

## Remaining Work

### Fix dry-run false positives

The dry-run still warns about builtins (`len`, `min`, `str`, `list`) and injected globals (`BaseModel`, `Field`) as unresolved references. Fix by maintaining a `known_globals` set in the dry-run that includes all SAFE_BUILTINS keys + tool names + `llm`, `BaseModel`, `Field`.

### Parameter schema validation in dry-run

At dry-run time, we have the tool metadata. AST-parse tool calls and check:
- Does the tool function exist?
- Are required parameters provided?
- Are parameter names correct?

### Tool action caching

Currently every `SupyWorkflow()` instantiation fetches tool metadata from the API. Add a TTL cache (5 minutes) to avoid redundant HTTP calls.

### Smarter tool result formatting

For the agentic generator, raw JSON tool results waste tokens. Post-process common responses:
- Slack channels → compact table of name/ID/memberCount
- Calendar events → time + title list
- Search results → title + snippet only
- Large arrays → first N items + count
