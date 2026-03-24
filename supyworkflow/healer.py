"""Self-healing — LLM-powered cell repair when execution fails."""

from __future__ import annotations

import json
import logging
import traceback
from dataclasses import dataclass
from typing import Any

import litellm

logger = logging.getLogger("supyworkflow")

MAX_HEAL_ATTEMPTS = 3

# Errors that are never recoverable — don't waste LLM calls
UNRECOVERABLE_ERRORS = (
    ImportError,
    SystemExit,
    KeyboardInterrupt,
    MemoryError,
    RecursionError,
    # Note: SyntaxError is NOT here — the healer can attempt to fix syntax issues
)


@dataclass
class HealResult:
    """Result of a healing attempt."""

    healed: bool
    patched_source: str
    attempts: int
    patches: list[str]  # each attempted patch


def heal_cell(
    cell_source: str,
    cell_label: str,
    error: Exception,
    namespace_snapshot: dict[str, Any],
    tool_names: list[str],
    model: str = "gemini/gemini-3.1-pro-preview",
) -> HealResult:
    """Attempt to fix a failed cell by asking an LLM to patch the code.

    Args:
        cell_source: The Python source that failed.
        cell_label: Human-readable label for the cell.
        error: The exception that was raised.
        namespace_snapshot: Serialized state of variables available to the cell.
        tool_names: List of tool names available in the namespace.
        model: LLM model to use for healing.

    Returns:
        HealResult with the patched source if successful.
    """
    # Don't try to heal unrecoverable errors
    if isinstance(error, UNRECOVERABLE_ERRORS):
        logger.info("heal_skip", extra={"reason": type(error).__name__})
        return HealResult(
            healed=False, patched_source=cell_source, attempts=0, patches=[]
        )

    # Build context about available variables
    available_vars = _summarize_namespace(namespace_snapshot)
    error_tb = traceback.format_exception(type(error), error, error.__traceback__)
    error_info = "".join(error_tb[-3:])  # last 3 frames, not the full stack

    patches: list[str] = []
    current_source = cell_source

    for attempt in range(MAX_HEAL_ATTEMPTS):
        logger.info(
            "heal_attempt",
            extra={"cell": cell_label, "attempt": attempt + 1, "error": str(error)},
        )

        prompt = _build_heal_prompt(
            source=current_source,
            label=cell_label,
            error_type=type(error).__name__,
            error_message=str(error),
            error_traceback=error_info,
            available_vars=available_vars,
            tool_names=tool_names,
            previous_patches=patches,
        )

        response = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": HEALER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )

        raw = response.choices[0].message.content.strip()
        patched = _extract_code(raw)

        if patched and patched != current_source:
            patches.append(patched)
            current_source = patched
            return HealResult(
                healed=True,
                patched_source=patched,
                attempts=attempt + 1,
                patches=patches,
            )

        # LLM returned the same code or couldn't extract — try again with more context
        patches.append(raw)

    return HealResult(
        healed=False,
        patched_source=cell_source,
        attempts=MAX_HEAL_ATTEMPTS,
        patches=patches,
    )


HEALER_SYSTEM_PROMPT = """\
You are a code repair agent embedded in a workflow runtime. Your job is to fix \
a Python code cell that failed during execution.

Rules:
- Return ONLY the fixed Python code inside a ```python fenced block
- Do NOT add imports (the runtime sandbox controls imports)
- Do NOT change the overall logic — fix the specific error
- Available builtins: BaseModel, Field, llm(), json, and standard Python builtins
- Tool calls look like: gmail.send_email(to=..., subject=..., body=...)
- LLM calls look like: llm("prompt", data=..., format=SomeModel)
- Variables from previous cells are available in the namespace
- Keep the fix minimal — change as little as possible
"""


def _build_heal_prompt(
    source: str,
    label: str,
    error_type: str,
    error_message: str,
    error_traceback: str,
    available_vars: str,
    tool_names: list[str],
    previous_patches: list[str],
) -> str:
    parts = [
        f"## Failed Cell: {label}\n",
        f"```python\n{source}\n```\n",
        f"## Error\n",
        f"**{error_type}**: {error_message}\n",
        f"```\n{error_traceback}```\n",
        f"## Available Variables\n{available_vars}\n",
        f"## Available Tools\n{', '.join(tool_names) if tool_names else 'None'}\n",
    ]

    if previous_patches:
        parts.append(f"\n## Previous fix attempts (all failed)\n")
        for i, patch in enumerate(previous_patches):
            parts.append(f"Attempt {i + 1}:\n```\n{patch}\n```\n")
        parts.append("Try a different approach.\n")

    parts.append("\nReturn the fixed code in a ```python block.")
    return "".join(parts)


def _summarize_namespace(snapshot: dict[str, Any]) -> str:
    """Create a concise summary of available variables for the LLM."""
    if not snapshot:
        return "No variables from previous cells."

    lines = []
    for key, entry in snapshot.items():
        if entry.get("type") == "json":
            value = entry["value"]
            # Truncate large values
            preview = json.dumps(value, default=str)
            if len(preview) > 200:
                preview = preview[:200] + "..."
            lines.append(f"- `{key}`: {preview}")
        else:
            lines.append(f"- `{key}`: {entry.get('value', '(non-serializable)')}")

    return "\n".join(lines)


def _extract_code(llm_response: str) -> str | None:
    """Extract Python code from a fenced code block in the LLM response."""
    # Look for ```python ... ``` blocks
    import re

    pattern = r"```(?:python)?\s*\n(.*?)```"
    matches = re.findall(pattern, llm_response, re.DOTALL)

    if matches:
        # Return the longest match (most likely the full fix)
        return max(matches, key=len).strip()

    # If no fenced block, and the response looks like pure code, use it directly
    lines = llm_response.strip().split("\n")
    if lines and not lines[0].startswith("#") and "def " not in lines[0]:
        # Probably not code
        return None

    return llm_response.strip() if llm_response.strip() else None
