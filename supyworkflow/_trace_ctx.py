"""Thread-local trace context — allows tool callables and llm() to write to the current trace."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from supyworkflow.trace import ExecutionTrace

_ctx = threading.local()


def set_trace(trace: ExecutionTrace | None, cell_index: int | None = None) -> None:
    """Set the current trace context for this thread."""
    _ctx.trace = trace
    _ctx.cell_index = cell_index


def get_trace() -> ExecutionTrace | None:
    """Get the current trace, or None if not in a traced execution."""
    return getattr(_ctx, "trace", None)


def get_cell_index() -> int | None:
    """Get the current cell index, or None."""
    return getattr(_ctx, "cell_index", None)
