"""HttpGatewayToolProvider — routes tool calls through an HTTP gateway.

This provider is used when supyworkflow runs as a subprocess and needs
to call back to the orchestrator (e.g., cardamon) for tool execution.
The gateway handles multi-provider dispatch (supyagent, composio, etc.).
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from supyworkflow.tool_proxy import ToolCallError

logger = logging.getLogger("supyworkflow")


class HttpGatewayToolProvider:
    """Tool provider that routes all calls through an HTTP gateway endpoint.

    The gateway exposes two endpoints:
        GET  {gateway_url}/discover  — returns tool metadata
        POST {gateway_url}/execute   — executes a tool by name

    This lets supyworkflow (running as subprocess) call back to the
    orchestrator, which dispatches to the correct backend.
    """

    def __init__(
        self,
        gateway_url: str,
        api_key: str,
        user_id: str | None = None,
        timeout: float = 120.0,
    ):
        self.gateway_url = gateway_url.rstrip("/")
        self.api_key = api_key
        self.user_id = user_id
        self.timeout = timeout
        self._metadata: list[dict] | None = None

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if user_id:
            headers["X-Account-Id"] = user_id

        self._client = httpx.Client(timeout=timeout, headers=headers)

    def discover(self) -> list[dict]:
        """Fetch tool metadata from the gateway's /discover endpoint."""
        if self._metadata is not None:
            return self._metadata

        resp = self._client.get(f"{self.gateway_url}/discover")
        resp.raise_for_status()
        data = resp.json()

        # Unwrap response envelope
        if isinstance(data, dict):
            inner = data.get("data", data)
            if isinstance(inner, dict):
                tools = inner.get("tools", [])
            else:
                tools = inner
        else:
            tools = data

        self._metadata = tools
        return tools

    def execute(self, tool_name: str, **kwargs: Any) -> Any:
        """Execute a tool through the gateway's /execute endpoint."""
        from supyworkflow._trace_ctx import get_cell_index, get_trace

        start = time.monotonic()
        input_keys = list(kwargs.keys())
        logger.info("gateway_tool_call_start", extra={"action": tool_name, "input_keys": input_keys})

        try:
            resp = self._client.post(
                f"{self.gateway_url}/execute",
                json={
                    "tool_name": tool_name,
                    "arguments": kwargs,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        except httpx.HTTPStatusError as e:
            duration_ms = (time.monotonic() - start) * 1000
            logger.error(
                "gateway_tool_call_error",
                extra={"action": tool_name, "status_code": e.response.status_code, "duration_ms": duration_ms},
            )
            trace = get_trace()
            if trace:
                trace.tool_call(
                    action=tool_name, duration_ms=duration_ms, ok=False,
                    cell_index=get_cell_index(),
                    error=str(e.response.status_code),
                    input_keys=input_keys,
                )
            detail = e.response.text[:500] if e.response.text else str(e)
            raise ToolCallError(action=tool_name, status_code=e.response.status_code, detail=detail) from e

        duration_ms = (time.monotonic() - start) * 1000
        logger.info("gateway_tool_call_end", extra={"action": tool_name, "duration_ms": duration_ms})

        trace = get_trace()
        if trace:
            trace.tool_call(
                action=tool_name, duration_ms=duration_ms, ok=True,
                cell_index=get_cell_index(), input_keys=input_keys,
            )

        # Unwrap response envelope
        if isinstance(data, dict):
            if data.get("ok") is False:
                raise ToolCallError(action=tool_name, detail=data.get("error", "Unknown error"))
            if "data" in data:
                return data["data"]
        return data

    def get_tool_names(self) -> list[str]:
        """Return tool names from the gateway."""
        metadata = self.discover()
        return [t["function"]["name"] for t in metadata if "function" in t]
