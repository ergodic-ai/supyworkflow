"""ToolProvider protocol — defines the interface for pluggable tool sources."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass
class ToolSpec:
    """Metadata for a single tool action."""

    name: str
    description: str
    parameters: dict[str, Any]
    method: str = "POST"
    path: str = ""
    metadata: dict[str, Any] | None = None


@runtime_checkable
class ToolProvider(Protocol):
    """Protocol for pluggable tool sources.

    Implementations provide tool discovery, execution, and schema access.
    supyworkflow doesn't care where tools come from — it just calls
    discover() to learn what's available and execute() to run them.
    """

    def discover(self) -> list[dict]:
        """Return tool metadata in OpenAI function-calling format.

        Each item should have the shape:
            {
                "function": {
                    "name": "tool_name",
                    "description": "...",
                    "parameters": { JSON Schema }
                },
                "metadata": {
                    "method": "POST",
                    "path": "/api/v1/...",
                    ...provider-specific
                }
            }
        """
        ...

    def execute(self, tool_name: str, **kwargs: Any) -> Any:
        """Execute a tool by name with the given keyword arguments.

        Returns the tool result (usually a dict). Raises on failure.
        """
        ...

    def get_tool_names(self) -> list[str]:
        """Return just the tool names (lightweight discovery)."""
        ...
