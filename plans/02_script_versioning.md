# Script Versioning

## Goal

When self-healing patches a script, persist the fix so future runs use the corrected version. Track the history of changes over time.

## Current State

- `RunResult.patched_source` returns the full script with healed cells substituted
- `RunResult.healed_cells` tracks which cells were patched and the diff
- Nothing is persisted — the patched source is returned but not saved anywhere

## Remaining Work

### 1. Version storage

Each workflow script gets a version history:

```
storage/workflows/{agent_id}/
├── v1.py          — Original generated script
├── v2.py          — After self-healing fix
├── v3.py          — After user edit
└── versions.json  — Version metadata
```

`versions.json`:
```json
[
  {"version": 1, "created_at": "...", "source": "generated", "prompt": "..."},
  {"version": 2, "created_at": "...", "source": "healed", "cell": 2, "error": "KeyError: 'totals'"},
  {"version": 3, "created_at": "...", "source": "user_edit", "diff": "..."}
]
```

### 2. Auto-save on heal

After a successful run where cells were healed:
1. Build patched source from `result.patched_source`
2. Save as new version
3. Future runs use the latest version

This needs a decision: save immediately on heal success, or wait for N successful runs to confirm the fix is stable?

Recommendation: save immediately. If the heal introduced a regression, it'll fail on next run and get healed again (or reverted).

### 3. Diff viewer

For the cardamon UI:
- Show version history timeline
- Side-by-side diff between versions
- Annotate healed cells with the error that triggered the fix
- Allow reverting to a previous version

### 4. CLI support

```bash
supyworkflow versions <script>          # List versions
supyworkflow diff <script> v1 v2        # Show diff
supyworkflow revert <script> <version>  # Revert to version
```
