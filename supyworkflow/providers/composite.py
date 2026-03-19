"""CompositeToolProvider — merges multiple providers into a single unified source."""

from __future__ import annotations

import logging
from typing import Any

from supyworkflow.tool_provider import ToolProvider

logger = logging.getLogger("supyworkflow")


class CompositeToolProvider:
    """Merges multiple ToolProviders into one.

    Discovery returns the union of all tools. On name collision,
    later providers override earlier ones (last-writer-wins).

    Execution dispatches to the provider that owns the tool name.
    """

    def __init__(self, providers: list[ToolProvider]):
        self._providers = providers
        self._tool_to_provider: dict[str, ToolProvider] | None = None
        self._all_metadata: list[dict] | None = None

    def _ensure_index(self) -> None:
        """Build the tool → provider mapping from all providers."""
        if self._tool_to_provider is not None:
            return

        self._tool_to_provider = {}
        merged: dict[str, dict] = {}

        for provider in self._providers:
            tools = provider.discover()
            for tool in tools:
                name = tool.get("function", {}).get("name", "")
                if not name:
                    continue
                merged[name] = tool
                self._tool_to_provider[name] = provider

        self._all_metadata = list(merged.values())

        logger.info(
            "composite_provider_built",
            extra={
                "total_tools": len(merged),
                "providers": len(self._providers),
            },
        )

    def discover(self) -> list[dict]:
        """Return merged tool metadata from all providers."""
        self._ensure_index()
        return self._all_metadata or []

    def execute(self, tool_name: str, **kwargs: Any) -> Any:
        """Execute a tool, dispatching to the correct provider."""
        self._ensure_index()
        assert self._tool_to_provider is not None

        provider = self._tool_to_provider.get(tool_name)
        if provider is None:
            raise KeyError(
                f"Tool '{tool_name}' not found in any provider. "
                f"Available: {sorted(self._tool_to_provider.keys())[:20]}"
            )
        return provider.execute(tool_name, **kwargs)

    def get_tool_names(self) -> list[str]:
        """Return all tool names across all providers."""
        self._ensure_index()
        assert self._tool_to_provider is not None
        return list(self._tool_to_provider.keys())
