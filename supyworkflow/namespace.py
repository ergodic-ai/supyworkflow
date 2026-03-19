"""Namespace builder — constructs the execution environment for workflow scripts."""

from __future__ import annotations

import builtins as _builtins
import json
import logging
import time
from typing import TYPE_CHECKING, Any

import httpx
from pydantic import BaseModel, Field

from supyworkflow.llm_builtin import llm
from supyworkflow.tool_proxy import build_tool_callables

if TYPE_CHECKING:
    from supyworkflow.tool_provider import ToolProvider

logger = logging.getLogger("supyworkflow")

# Modules the workflow can import
ALLOWED_IMPORTS = {
    "json",
    "re",
    "math",
    "time",
    "datetime",
    "calendar",
    "collections",
    "itertools",
    "functools",
    "textwrap",
    "copy",
    "dataclasses",
    "typing",
    "decimal",
    "fractions",
    "statistics",
    "string",
    "urllib.parse",
    "base64",
    "hashlib",
    "uuid",
    "csv",
    "io",
}

# Modules we block explicitly (dangerous). Everything else is allowed
# if it's a transitive dep of an allowed module.
BLOCKED_IMPORTS = {
    "os",
    "sys",
    "subprocess",
    "shutil",
    "signal",
    "socket",
    "http",
    "urllib.request",
    "ftplib",
    "smtplib",
    "ctypes",
    "multiprocessing",
    "importlib",
    "code",
    "codeop",
    "compile",
    "compileall",
    "py_compile",
    "runpy",
    "builtins",
    "gc",
    "inspect",
    "pathlib",
    "tempfile",
    "glob",
    "fnmatch",
    "pickle",
    "shelve",
    "sqlite3",
    "webbrowser",
    "antigravity",
    "turtle",
    "tkinter",
}


def _restricted_import(name: str, *args: Any, **kwargs: Any) -> Any:
    """Block dangerous imports, allow everything else.

    We use a blocklist rather than an allowlist because allowed modules
    (like datetime) have internal transitive deps that are impossible
    to enumerate exhaustively.
    """
    top_level = name.split(".")[0]
    if top_level in BLOCKED_IMPORTS:
        raise ImportError(
            f"Import '{name}' is not allowed in workflows. "
            f"Blocked modules: {', '.join(sorted(BLOCKED_IMPORTS))}"
        )
    return _builtins.__import__(name, *args, **kwargs)


# Builtins we allow in the workflow sandbox
SAFE_BUILTINS = {
    # Types
    "True": True,
    "False": False,
    "None": None,
    "bool": bool,
    "int": int,
    "float": float,
    "str": str,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "set": set,
    "frozenset": frozenset,
    "bytes": bytes,
    "bytearray": bytearray,
    "type": type,
    "object": object,
    "property": property,
    "staticmethod": staticmethod,
    "classmethod": classmethod,
    "super": super,
    # Class machinery (needed for `class Foo(BaseModel):`)
    "__build_class__": __build_class__,
    "__name__": "__main__",
    # Functions
    "len": len,
    "range": range,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "sorted": sorted,
    "reversed": reversed,
    "min": min,
    "max": max,
    "sum": sum,
    "abs": abs,
    "round": round,
    "any": any,
    "all": all,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "hasattr": hasattr,
    "getattr": getattr,
    "setattr": setattr,
    "print": print,
    "repr": repr,
    "format": format,
    "id": id,
    "hash": hash,
    "callable": callable,
    "iter": iter,
    "next": next,
    # String / data
    "chr": chr,
    "ord": ord,
    "hex": hex,
    "oct": oct,
    "bin": bin,
    # Exceptions
    "Exception": Exception,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "IndexError": IndexError,
    "AttributeError": AttributeError,
    "RuntimeError": RuntimeError,
    "StopIteration": StopIteration,
    # Imports
    "json": json,
    "__import__": _restricted_import,
    # Pydantic (for structured LLM output)
    "BaseModel": BaseModel,
    "Field": Field,
}


def discover_tools(
    api_key: str,
    base_url: str = "https://app.supyagent.com",
) -> list[str]:
    """Discover available tool names for this API key."""
    resp = httpx.get(
        f"{base_url.rstrip('/')}/api/v1/tools",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    resp.raise_for_status()
    data = resp.json()

    # Extract tool function names from metadata
    inner = data.get("data", data) if isinstance(data, dict) else data
    tools_list = inner.get("tools", inner) if isinstance(inner, dict) else inner
    return [t["function"]["name"] for t in tools_list if "function" in t]


def build_namespace(
    api_key: str,
    user_id: str,
    base_url: str = "https://app.supyagent.com",
    tools: list[str] | None = None,
    extra_globals: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the execution namespace for a workflow script.

    Injects:
        - Callable functions for each connected tool (gmail_list_messages, etc.)
        - llm() builtin function
        - Safe builtins (no os, sys, subprocess, etc.)
        - Any extra globals provided
    """
    namespace: dict[str, Any] = {"__builtins__": SAFE_BUILTINS.copy()}

    # Inject tool callables (each tool_name becomes a function in the namespace)
    # When tools is explicitly set to [] (e.g. in tests), skip API discovery
    if tools is None:
        tool_callables = build_tool_callables(api_key=api_key, base_url=base_url, user_id=user_id)
        namespace.update(tool_callables)
    elif tools:
        # Caller provided specific tool names — still fetch metadata for routing
        tool_callables = build_tool_callables(api_key=api_key, base_url=base_url, user_id=user_id)
        namespace.update({k: v for k, v in tool_callables.items() if k in set(tools)})
    # else: tools == [] means no tools, skip entirely

    # Inject llm builtin
    namespace["llm"] = llm

    # Inject extras
    if extra_globals:
        namespace.update(extra_globals)

    logger.info(
        "namespace_built",
        extra={"extra_keys": list(extra_globals.keys()) if extra_globals else []},
    )

    return namespace


def build_namespace_from_providers(
    providers: list[ToolProvider],
    extra_tools: dict[str, Any] | None = None,
    extra_globals: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the execution namespace using pluggable tool providers.

    Instead of calling build_tool_callables() directly against supyagent,
    this builds callables from each provider's discover()/execute() methods.

    Args:
        providers: List of ToolProvider instances to source tools from.
        extra_tools: Plain callables to merge into the namespace (no schema needed).
        extra_globals: Additional global variables (e.g., input data).

    Returns:
        Execution namespace dict with __builtins__, tools, llm, and extras.
    """
    namespace: dict[str, Any] = {"__builtins__": SAFE_BUILTINS.copy()}

    # Build callables from each provider
    for provider in providers:
        tools = provider.discover()
        for tool in tools:
            func = tool.get("function", {})
            name = func.get("name", "")
            if not name:
                continue

            # Create a closure that calls provider.execute() with tracing
            namespace[name] = _make_provider_callable(name, provider)

    # Merge extra tools (plain callables — override provider tools on collision)
    if extra_tools:
        namespace.update(extra_tools)

    # Inject llm builtin
    namespace["llm"] = llm

    # Inject extras
    if extra_globals:
        namespace.update(extra_globals)

    logger.info(
        "namespace_built_from_providers",
        extra={
            "provider_count": len(providers),
            "extra_tools": list(extra_tools.keys()) if extra_tools else [],
            "extra_keys": list(extra_globals.keys()) if extra_globals else [],
        },
    )

    return namespace


def _make_provider_callable(name: str, provider: ToolProvider) -> Any:
    """Create a callable that routes through provider.execute() with tracing."""

    def call(**kwargs: Any) -> Any:
        from supyworkflow._trace_ctx import get_cell_index, get_trace

        start = time.monotonic()
        input_keys = list(kwargs.keys())
        logger.info("provider_tool_call_start", extra={"action": name, "input_keys": input_keys})

        try:
            result = provider.execute(name, **kwargs)
        except Exception:
            duration_ms = (time.monotonic() - start) * 1000
            trace = get_trace()
            if trace:
                trace.tool_call(
                    action=name, duration_ms=duration_ms, ok=False,
                    cell_index=get_cell_index(),
                    error="provider_error",
                    input_keys=input_keys,
                )
            raise

        duration_ms = (time.monotonic() - start) * 1000
        logger.info("provider_tool_call_end", extra={"action": name, "duration_ms": duration_ms})

        trace = get_trace()
        if trace:
            trace.tool_call(
                action=name, duration_ms=duration_ms, ok=True,
                cell_index=get_cell_index(), input_keys=input_keys,
            )

        return result

    call.__name__ = name
    call.__doc__ = f"Execute {name} via tool provider"
    return call
