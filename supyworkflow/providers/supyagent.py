"""SupyagentToolProvider — wraps the existing tool_proxy logic as a ToolProvider."""

from __future__ import annotations

import logging
from typing import Any

from supyworkflow.tool_proxy import (
    DEFAULT_BASE_URL,
    DEFAULT_TIMEOUT,
    _fetch_tools_metadata,
    build_tool_callables,
)

logger = logging.getLogger("supyworkflow")


class SupyagentToolProvider:
    """Tool provider that talks directly to the supyagent REST API.

    This is the default provider — wraps the existing tool_proxy.py logic
    so that standalone supyworkflow usage is unchanged.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        user_id: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.user_id = user_id
        self.timeout = timeout
        self._metadata: list[dict] | None = None
        self._callables: dict[str, Any] | None = None

    def _ensure_metadata(self) -> list[dict]:
        if self._metadata is None:
            self._metadata = _fetch_tools_metadata(
                self.api_key, self.base_url, self.timeout, user_id=self.user_id
            )
        return self._metadata

    def _ensure_callables(self) -> dict[str, Any]:
        if self._callables is None:
            metadata = self._ensure_metadata()
            self._callables = build_tool_callables(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
                tools_metadata=metadata,
                user_id=self.user_id,
            )
        return self._callables

    def discover(self) -> list[dict]:
        """Return tool metadata from supyagent in OpenAI function-calling format."""
        return self._ensure_metadata()

    def execute(self, tool_name: str, **kwargs: Any) -> Any:
        """Execute a tool via the supyagent REST API."""
        callables = self._ensure_callables()
        if tool_name not in callables:
            raise KeyError(f"Tool '{tool_name}' not found in supyagent provider")
        return callables[tool_name](**kwargs)

    def get_tool_names(self) -> list[str]:
        """Return tool names from supyagent."""
        metadata = self._ensure_metadata()
        return [t["function"]["name"] for t in metadata if "function" in t]
