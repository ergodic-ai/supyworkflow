"""Cell parser — splits workflow scripts into executable cells with AST-based dependency tracking."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field


@dataclass
class Cell:
    """A single executable cell in a workflow script."""

    index: int
    source: str
    label: str = ""
    reads: set[str] = field(default_factory=set)
    writes: set[str] = field(default_factory=set)
    status: str = "pending"  # pending | running | completed | failed | skipped
    result: object = None
    error: Exception | None = None
    snapshot: dict | None = None
    duration_ms: float = 0

    @property
    def depends_on(self) -> set[str]:
        """Variables this cell reads that it doesn't write itself (external deps)."""
        return self.reads - self.writes


def parse_cells(source: str) -> list[Cell]:
    """Parse a workflow script into cells.

    Splits on `# ---` markers. Each marker can have an optional label:
        # --- Fetch orders
        # --- Analyze trends

    Extracts read/write variable dependencies via AST analysis.
    """
    # Split on cell markers, capturing the label
    parts = re.split(r'\n?# ---[ ]*([^\n]*)\n', source)

    cells: list[Cell] = []

    # First part (before any marker) may have content
    if parts[0].strip():
        cell = _build_cell(0, parts[0].strip(), label="setup")
        cells.append(cell)

    # Remaining parts come in pairs: (label, source)
    for i in range(1, len(parts), 2):
        label = parts[i].strip() if i < len(parts) else ""
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if body:
            cell = _build_cell(len(cells), body, label=label)
            cells.append(cell)

    return cells


def _build_cell(index: int, source: str, label: str = "") -> Cell:
    """Build a Cell with AST-derived read/write sets."""
    reads, writes = _analyze_dependencies(source)
    return Cell(index=index, source=source, label=label, reads=reads, writes=writes)


def _analyze_dependencies(source: str) -> tuple[set[str], set[str]]:
    """Extract variable reads and writes from Python source via AST.

    Returns (reads, writes) where:
        - writes: names assigned to (=, for, with, augmented assign)
        - reads: names loaded that aren't builtins or injected tools
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set(), set()

    writes: set[str] = set()
    reads: set[str] = set()

    for node in ast.walk(tree):
        # Assignments: x = ...
        if isinstance(node, ast.Assign):
            for target in node.targets:
                _collect_names(target, writes)
        # Augmented assignments: x += ...
        elif isinstance(node, ast.AugAssign):
            _collect_names(node.target, writes)
        # For loops: for x in ...
        elif isinstance(node, ast.For):
            _collect_names(node.target, writes)
        # With statements: with ... as x
        elif isinstance(node, ast.With):
            for item in node.items:
                if item.optional_vars:
                    _collect_names(item.optional_vars, writes)
        # Name loads (reads)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            reads.add(node.id)

    return reads, writes


def _collect_names(node: ast.AST, names: set[str]) -> None:
    """Recursively collect Name nodes from assignment targets (handles tuples, lists)."""
    if isinstance(node, ast.Name):
        names.add(node.id)
    elif isinstance(node, (ast.Tuple, ast.List)):
        for elt in node.elts:
            _collect_names(elt, names)


def build_dependency_graph(cells: list[Cell]) -> dict[int, set[int]]:
    """Build a cell dependency graph.

    Returns a dict mapping cell index → set of cell indices it depends on.
    A cell depends on another if it reads a variable that the other writes.
    """
    # Map variable name → cell index that writes it (last writer wins)
    writers: dict[str, int] = {}
    for cell in cells:
        for var in cell.writes:
            writers[var] = cell.index

    graph: dict[int, set[int]] = {cell.index: set() for cell in cells}
    for cell in cells:
        for var in cell.depends_on:
            if var in writers and writers[var] != cell.index:
                graph[cell.index].add(writers[var])

    return graph
