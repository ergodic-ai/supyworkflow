# Parallel Cell Execution

## Goal

Run independent cells concurrently to speed up workflows with parallel branches.

## Current State

- `build_dependency_graph()` already detects which cells depend on which
- Execution is strictly sequential (cell 0 → cell 1 → cell 2 → ...)
- Diamond dependencies are correctly identified (e.g., cells 1 and 2 both depend on cell 0, cell 3 depends on both)

## Remaining Work

### 1. Topological execution scheduler

Replace the linear `for cell in cells` loop with a scheduler that:
1. Builds the dependency graph
2. Identifies cells with no unmet dependencies (ready to run)
3. Executes ready cells concurrently (via `concurrent.futures.ThreadPoolExecutor`)
4. When a cell completes, checks if new cells become ready
5. Continues until all cells are done or failed

```python
# Pseudocode
ready = {c for c in cells if not graph[c.index]}
running = set()
completed = set()

while ready or running:
    for cell in ready:
        submit(cell)
        running.add(cell)
    ready.clear()

    done_cell = wait_for_any(running)
    running.remove(done_cell)
    completed.add(done_cell.index)

    # Check what's now unblocked
    for cell in cells:
        if cell.index not in completed and cell.index not in running:
            if graph[cell.index] <= completed:
                ready.add(cell)
```

### 2. Namespace isolation per parallel branch

Challenge: parallel cells share the same namespace dict. Concurrent writes are unsafe.

Options:
- **Option A: Copy namespace per branch, merge after** — Each parallel cell gets a copy. After all complete, merge writes back. Conflict if two cells write the same variable.
- **Option B: Lock-free with separate output dicts** — Each cell writes to its own output dict. Main namespace is read-only during parallel execution. Merge outputs after.
- **Option C: Don't parallelize cells that write overlapping variables** — Only run cells in parallel if their `writes` sets don't overlap.

Recommendation: **Option C** — simplest, safe, and covers the common case (parallel API calls writing to different variables).

### 3. Opt-in flag

Add `parallel=True` to `runtime.run()`. Default False for backwards compatibility.

### 4. Trace updates

Trace events need to record which cells ran in parallel so the UI can show a Gantt-style view.
