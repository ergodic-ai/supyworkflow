"""llm() builtin — natural LLM calls inside workflow scripts."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger("supyworkflow")

# Supported models for workflow generation.
# The first entry is the default. Add new models here.
SUPPORTED_MODELS = [
    "gemini/gemini-3-flash-preview",
    "gemini/gemini-3.1-pro-preview",
    "openrouter/minimax/minimax-m2.5",
    "openrouter/x-ai/grok-4.1-fast",
]

DEFAULT_MODEL = SUPPORTED_MODELS[0]


def _is_pydantic_model(obj: Any) -> bool:
    """Check if obj is a Pydantic BaseModel class (not instance)."""
    try:
        return isinstance(obj, type) and issubclass(obj, BaseModel)
    except TypeError:
        return False


def llm(
    prompt: str,
    *,
    data: Any = None,
    format: type[BaseModel] | None = None,
    model: str | None = None,
    temperature: float = 0.0,
) -> BaseModel | str:
    """Call an LLM from a workflow script.

    Args:
        prompt: The instruction / question.
        data: Optional data to include in the prompt (will be JSON-serialized).
        format: Optional Pydantic BaseModel class for structured output.
            When provided, the schema is injected into the system prompt and
            the response is parsed and validated by Pydantic.
        model: LiteLLM model identifier. Defaults to DEFAULT_MODEL.
        temperature: Sampling temperature.

    Returns:
        If format is None: plain text string.
        If format is a BaseModel subclass: validated Pydantic model instance.
    """
    import litellm

    litellm.suppress_debug_info = True

    if model is None:
        model = DEFAULT_MODEL

    # Build the user message
    parts = [prompt]
    if data is not None:
        serialized = json.dumps(data, default=str, indent=2)
        parts.append(f"\n\nData:\n```json\n{serialized}\n```")
    user_message = "".join(parts)

    # System message
    system_message = "You are a helpful assistant embedded in an automated workflow."
    use_structured = False

    if format is not None and _is_pydantic_model(format):
        schema = format.model_json_schema()
        system_message += (
            "\n\nYou MUST respond with ONLY valid JSON matching this schema. "
            "No text before or after the JSON.\n"
            f"Schema: {json.dumps(schema)}"
        )
        use_structured = True
    else:
        system_message += "\n\nRespond concisely and directly."

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    start = time.monotonic()
    logger.info(
        "llm_call_start",
        extra={
            "model": model,
            "prompt_length": len(user_message),
            "format": format.__name__ if format is not None else None,
        },
    )

    response = litellm.completion(**kwargs)

    duration_ms = (time.monotonic() - start) * 1000
    usage = response.usage
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0

    logger.info(
        "llm_call_end",
        extra={
            "model": model,
            "duration_ms": duration_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    )

    # Write to trace if in a traced execution
    from supyworkflow._trace_ctx import get_cell_index, get_trace

    trace = get_trace()
    if trace:
        trace.llm_call(
            model=model,
            duration_ms=duration_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cell_index=get_cell_index(),
            has_format=use_structured,
        )

    raw = response.choices[0].message.content

    # Parse based on format
    if not use_structured or format is None:
        return raw.strip()

    # Strip markdown fences if present
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        import re

        match = re.search(r"```(?:json)?\s*\n(.*?)```", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(1).strip()

    return format.model_validate_json(cleaned)
