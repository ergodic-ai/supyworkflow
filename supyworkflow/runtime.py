"""SupyWorkflow runtime — the core execution engine."""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from supyworkflow.healer import HealResult, heal_cell
from supyworkflow.namespace import build_namespace, discover_tools
from supyworkflow.parser import Cell, build_dependency_graph, parse_cells
from supyworkflow.snapshot import capture_snapshot, restore_snapshot
from supyworkflow.trace import ExecutionTrace

logger = logging.getLogger("supyworkflow")


@dataclass
class RunResult:
    """Result of executing a workflow."""

    run_id: str
    status: str  # completed | failed | partial
    namespace: dict[str, Any] = field(default_factory=dict)
    cells: list[Cell] = field(default_factory=list)
    trace: ExecutionTrace | None = None
    error: Exception | None = None
    healed_cells: dict[int, HealResult] = field(default_factory=dict)

    @property
    def outputs(self) -> dict[str, Any]:
        """User-defined variables in the final namespace (excludes tools/builtins)."""
        skip = {"__builtins__", "llm", "BaseModel", "Field"}
        return {
            k: v
            for k, v in self.namespace.items()
            if not k.startswith("_") and k not in skip and not callable(v)
        }

    @property
    def patched_source(self) -> str | None:
        """If any cells were healed, return the full patched script."""
        if not self.healed_cells:
            return None
        # Rebuild source from cells, substituting healed versions
        parts = []
        for cell in self.cells:
            if cell.label:
                parts.append(f"# --- {cell.label}")
            heal = self.healed_cells.get(cell.index)
            if heal and heal.healed:
                parts.append(heal.patched_source)
            else:
                parts.append(cell.source)
        return "\n\n".join(parts) + "\n"


class SupyWorkflow:
    """Workflow-as-code runtime.

    Usage:
        runtime = SupyWorkflow(api_key="sk_live_...", user_id="user_123")
        result = runtime.run(script_source)
        print(result.outputs)
    """

    def __init__(
        self,
        api_key: str,
        user_id: str,
        base_url: str = "https://app.supyagent.com",
        timeout_ms: int = 240_000,
        heal: bool = True,
        heal_model: str = "gemini/gemini-3.1-pro-preview",
    ):
        self.api_key = api_key
        self.user_id = user_id
        self.base_url = base_url
        self.timeout_ms = timeout_ms
        self.heal = heal
        self.heal_model = heal_model
        self._tools: list[str] | None = None

    @property
    def tools(self) -> list[str]:
        if self._tools is None:
            self._tools = discover_tools(self.api_key, self.base_url)
        return self._tools

    def run(
        self,
        source: str,
        inputs: dict[str, Any] | None = None,
        from_cell: int = 0,
        snapshots: dict[int, dict] | None = None,
    ) -> RunResult:
        """Execute a workflow script.

        Args:
            source: The Python workflow script.
            inputs: Optional input variables to seed into the namespace.
            from_cell: Cell index to start from (for re-runs). Requires snapshots.
            snapshots: Previously captured cell snapshots (for resuming).

        Returns:
            RunResult with final namespace, cell statuses, and execution trace.
        """
        run_id = uuid.uuid4().hex[:12]
        trace = ExecutionTrace(run_id=run_id)
        trace.start()

        # Parse
        cells = parse_cells(source)
        if not cells:
            return RunResult(run_id=run_id, status="completed", cells=[], trace=trace)

        dep_graph = build_dependency_graph(cells)

        # Build namespace
        namespace = build_namespace(
            api_key=self.api_key,
            user_id=self.user_id,
            base_url=self.base_url,
            tools=self.tools,
            extra_globals=inputs,
        )
        tool_names = set(self.tools)

        # Restore snapshot if resuming from a specific cell
        if from_cell > 0 and snapshots:
            for i in range(from_cell - 1, -1, -1):
                if i in snapshots:
                    restore_snapshot(namespace, snapshots[i])
                    break
            for cell in cells[:from_cell]:
                cell.status = "skipped"

        # Track state
        failed_keys: set[str] = set()
        cell_snapshots: dict[int, dict] = {}
        healed_cells: dict[int, HealResult] = {}

        # Execute
        deadline = time.monotonic() + (self.timeout_ms / 1000)
        last_error: Exception | None = None

        for cell in cells[from_cell:]:
            if time.monotonic() > deadline:
                cell.status = "failed"
                cell.error = TimeoutError("Workflow exceeded time budget")
                last_error = cell.error
                trace.error(cell.index, "timeout", "Workflow exceeded time budget")
                break

            # Check if any dependency failed
            deps = dep_graph.get(cell.index, set())
            upstream_writes: set[str] = set()
            for dep_idx in deps:
                upstream_writes |= cells[dep_idx].writes
            if upstream_writes & failed_keys:
                cell.status = "skipped"
                failed_keys |= cell.writes
                trace.cell_start(cell.index, cell.label)
                trace.cell_end(cell.index, cell.label, "skipped", 0)
                continue

            # Execute cell (with healing on failure)
            trace.cell_start(cell.index, cell.label)
            cell.status = "running"
            start = time.monotonic()

            cell_source = cell.source
            success = False

            try:
                compiled = compile(cell_source, f"<cell-{cell.index}-{cell.label}>", "exec")
                exec(compiled, namespace)  # noqa: S102
                success = True
            except Exception as e:
                # Attempt self-healing
                if self.heal and time.monotonic() < deadline:
                    snapshot_for_heal = capture_snapshot(namespace, tool_names)
                    heal_result = heal_cell(
                        cell_source=cell_source,
                        cell_label=cell.label,
                        error=e,
                        namespace_snapshot=snapshot_for_heal,
                        tool_names=list(tool_names),
                        model=self.heal_model,
                    )
                    healed_cells[cell.index] = heal_result

                    if heal_result.healed:
                        trace.error(
                            cell.index,
                            type(e).__name__,
                            f"{e} (healed after {heal_result.attempts} attempts)",
                        )
                        # Try the patched version
                        try:
                            compiled = compile(
                                heal_result.patched_source,
                                f"<cell-{cell.index}-{cell.label}-healed>",
                                "exec",
                            )
                            exec(compiled, namespace)  # noqa: S102
                            success = True
                            logger.info(
                                "heal_success",
                                extra={
                                    "cell": cell.index,
                                    "label": cell.label,
                                    "attempts": heal_result.attempts,
                                },
                            )
                        except Exception as heal_error:
                            # Healed code also failed
                            cell.error = heal_error
                            last_error = heal_error
                            trace.error(cell.index, type(heal_error).__name__, str(heal_error))
                    else:
                        cell.error = e
                        last_error = e
                        trace.error(cell.index, type(e).__name__, str(e))
                else:
                    cell.error = e
                    last_error = e
                    trace.error(cell.index, type(e).__name__, str(e))

            if success:
                cell.status = "completed"
            else:
                cell.status = "failed"
                failed_keys |= cell.writes

            cell.duration_ms = (time.monotonic() - start) * 1000
            trace.cell_end(cell.index, cell.label, cell.status, cell.duration_ms)

            # Snapshot after each successful cell
            if cell.status == "completed":
                cell_snapshots[cell.index] = capture_snapshot(namespace, tool_names)
                cell.snapshot = cell_snapshots[cell.index]

        # Determine overall status
        statuses = {c.status for c in cells}
        if "failed" in statuses:
            status = (
                "failed"
                if all(s in ("failed", "skipped", "pending") for s in statuses)
                else "partial"
            )
        else:
            status = "completed"

        return RunResult(
            run_id=run_id,
            status=status,
            namespace=namespace,
            cells=cells,
            trace=trace,
            error=last_error,
            healed_cells=healed_cells,
        )

    def dry_run(self, source: str) -> dict:
        """Analyze a workflow without executing it.

        Returns:
            Dict with cells, dependency graph, required tools, and warnings.
        """
        cells = parse_cells(source)
        graph = build_dependency_graph(cells)

        # Collect all tool references
        tool_refs: set[str] = set()
        missing_tools: set[str] = set()
        all_writes: set[str] = set()
        for c in cells:
            all_writes |= c.writes

        connected = set(self.tools)
        for cell in cells:
            for name in cell.reads:
                if name in connected:
                    tool_refs.add(name)

        warnings = []
        for cell in cells:
            unresolved = cell.depends_on - all_writes - connected - {"llm", "BaseModel", "Field"}
            if unresolved:
                warnings.append(
                    f"Cell {cell.index} ({cell.label}): unresolved references: {unresolved}"
                )

        return {
            "cells": [
                {
                    "index": c.index,
                    "label": c.label,
                    "reads": sorted(c.reads),
                    "writes": sorted(c.writes),
                    "depends_on": sorted(graph.get(c.index, set())),
                }
                for c in cells
            ],
            "tools_used": sorted(tool_refs),
            "missing_tools": sorted(missing_tools),
            "warnings": warnings,
        }
