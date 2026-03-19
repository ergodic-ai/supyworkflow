"""ToolProxy — routes flat tool calls to cardamon's REST API using tool metadata."""

from __future__ import annotations

import logging
import re
import time
from typing import Any

import httpx

logger = logging.getLogger("supyworkflow")

DEFAULT_BASE_URL = "https://app.supyagent.com"
DEFAULT_TIMEOUT = 120.0


class ToolCallError(Exception):
    """Raised when a tool call fails."""

    def __init__(self, action: str, status_code: int | None = None, detail: str = ""):
        self.action = action
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"{action} failed: {detail}")


def build_tool_callables(
    api_key: str,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = DEFAULT_TIMEOUT,
    tools_metadata: list[dict] | None = None,
) -> dict[str, callable]:
    """Build a dict of callable functions from tool metadata.

    Each tool like gmail_list_messages becomes a callable that routes
    to the correct REST endpoint with the right HTTP method.

    Returns:
        Dict mapping tool_name -> callable(**kwargs) -> result
    """
    if tools_metadata is None:
        tools_metadata = _fetch_tools_metadata(api_key, base_url, timeout)

    client = httpx.Client(
        timeout=timeout,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )

    callables: dict[str, callable] = {}

    for tool in tools_metadata:
        func = tool.get("function", {})
        meta = tool.get("metadata", {})
        name = func.get("name", "")
        method = meta.get("method", "POST").upper()
        path = meta.get("path", "")
        params = func.get("parameters", {})

        if not name or not path:
            continue

        body_defaults = meta.get("bodyDefaults", {})

        callables[name] = _make_tool_callable(
            name=name,
            method=method,
            path_template=path,
            parameters=params,
            client=client,
            base_url=base_url,
            body_defaults=body_defaults,
        )

    return callables


def _make_tool_callable(
    name: str,
    method: str,
    path_template: str,
    parameters: dict,
    client: httpx.Client,
    base_url: str,
    body_defaults: dict | None = None,
) -> callable:
    """Create a callable function for a single tool action."""

    # Extract path parameter names from template like /api/v1/gmail/messages/{messageId}
    path_params = re.findall(r'\{(\w+)\}', path_template)

    def call(**kwargs: Any) -> Any:
        from supyworkflow._trace_ctx import get_cell_index, get_trace

        start = time.monotonic()
        input_keys = list(kwargs.keys())
        logger.info("tool_call_start", extra={"action": name, "input_keys": input_keys})

        # Separate path params from body/query params
        path_values = {}
        remaining = dict(kwargs)
        for p in path_params:
            if p in remaining:
                path_values[p] = remaining.pop(p)

        # Merge body defaults (e.g. {"action": "insert_text"} for docs)
        if body_defaults and method in ("POST", "PUT", "PATCH"):
            remaining = {**body_defaults, **remaining}

        # Build the URL
        path = path_template
        for p, v in path_values.items():
            path = path.replace(f"{{{p}}}", str(v))
        url = f"{base_url.rstrip('/')}{path}"

        # Make the request
        try:
            if method == "GET":
                resp = client.get(url, params=remaining if remaining else None)
            elif method == "DELETE":
                resp = client.delete(url, params=remaining if remaining else None)
            elif method == "PATCH":
                resp = client.patch(url, json=remaining if remaining else None)
            elif method == "PUT":
                resp = client.put(url, json=remaining if remaining else None)
            else:  # POST
                resp = client.post(url, json=remaining if remaining else None)

            resp.raise_for_status()
            data = resp.json()

        except httpx.HTTPStatusError as e:
            duration_ms = (time.monotonic() - start) * 1000
            logger.error(
                "tool_call_error",
                extra={"action": name, "status_code": e.response.status_code, "duration_ms": duration_ms},
            )
            trace = get_trace()
            if trace:
                trace.tool_call(
                    action=name, duration_ms=duration_ms, ok=False,
                    cell_index=get_cell_index(),
                    error=str(e.response.status_code),
                    input_keys=input_keys,
                )
            detail = e.response.text[:500] if e.response.text else str(e)
            raise ToolCallError(action=name, status_code=e.response.status_code, detail=detail) from e

        duration_ms = (time.monotonic() - start) * 1000
        logger.info("tool_call_end", extra={"action": name, "duration_ms": duration_ms})

        trace = get_trace()
        if trace:
            trace.tool_call(
                action=name, duration_ms=duration_ms, ok=True,
                cell_index=get_cell_index(), input_keys=input_keys,
            )

        # Unwrap cardamon's response envelope
        if isinstance(data, dict):
            if data.get("ok") is False:
                raise ToolCallError(action=name, detail=data.get("error", "Unknown error"))
            if "data" in data:
                return data["data"]
        return data

    call.__name__ = name
    call.__doc__ = f"Execute {name} via cardamon API"
    return call


def _fetch_tools_metadata(api_key: str, base_url: str, timeout: float) -> list[dict]:
    """Fetch tool metadata from the API."""
    resp = httpx.get(
        f"{base_url.rstrip('/')}/api/v1/tools",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()

    # Handle {"ok": true, "data": {"tools": [...]}} envelope
    if isinstance(data, dict):
        inner = data.get("data", data)
        if isinstance(inner, dict):
            return inner.get("tools", [])
        return inner
    return data
