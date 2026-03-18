"""Namespace snapshots — serialize/restore cell state for replay and re-runnability."""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger("supyworkflow")

# Keys we never snapshot (injected by runtime, not user data)
SKIP_KEYS = {"__builtins__", "llm", "BaseModel", "Field"}


def capture_snapshot(namespace: dict[str, Any], tool_names: set[str]) -> dict[str, Any]:
    """Capture user-defined variables from the namespace.

    Skips builtins, callables (tool functions), and private names.
    Stores each value as JSON-serializable data where possible,
    falls back to repr() for non-serializable objects.
    """
    snapshot: dict[str, Any] = {}

    for key, value in namespace.items():
        if key in SKIP_KEYS or key in tool_names or key.startswith("_"):
            continue
        if callable(value) and not isinstance(value, type):
            continue

        try:
            # Test that it's JSON-serializable
            json.dumps(value, default=str)
            snapshot[key] = {"type": "json", "value": value}
        except (TypeError, ValueError, OverflowError):
            snapshot[key] = {"type": "repr", "value": repr(value)}

    return snapshot


def restore_snapshot(
    namespace: dict[str, Any],
    snapshot: dict[str, Any],
) -> None:
    """Restore snapshotted variables into a namespace.

    Only restores JSON-typed values. Repr-typed values are logged as warnings
    since they can't be faithfully restored.
    """
    for key, entry in snapshot.items():
        if entry["type"] == "json":
            namespace[key] = entry["value"]
        else:
            logger.warning(
                "snapshot_restore_skip",
                extra={"key": key, "reason": "non-serializable, stored as repr"},
            )
