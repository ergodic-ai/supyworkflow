"""Workflow generator — LLM generates a Python workflow script from a natural language prompt."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import litellm
from pydantic import BaseModel

logger = logging.getLogger("supyworkflow")


@dataclass
class GenerateResult:
    """Result of workflow generation."""

    source: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


SYSTEM_PROMPT = """\
You are a workflow code generator. You write Python scripts that automate tasks \
by calling external tools and using LLM intelligence.

## Runtime Environment

The script runs in a sandboxed Python environment with these pre-injected globals:

### Tools
Tools are pre-authenticated HTTP clients. Call them like regular functions:
```python
emails = gmail.list_messages(query="from:boss", max_results=10)
slack.send_message(channel="general", text="Hello!")
orders = shopify.get_orders(status="completed", days=7)
```
Each tool call returns the response data directly (already unwrapped from the API envelope).

### LLM Calls
Use `llm()` for any intelligence — analysis, summarization, decision-making, data extraction:
```python
# Plain text
summary = llm("Summarize these emails concisely", data=emails)

# Structured output with Pydantic (preferred for anything beyond plain text)
class Analysis(BaseModel):
    trends: list[str]
    recommendation: str
    urgency: bool

result = llm("Analyze this data", data=sales_data, format=Analysis)
print(result.trends)  # typed, validated
```

`BaseModel` and `Field` from Pydantic are available as globals — no import needed.

### Allowed Imports
You may import: json, re, math, datetime, collections, itertools, functools, \
textwrap, copy, dataclasses, typing, decimal, statistics, string, base64, hashlib, uuid, csv, io.

No other imports are allowed (no os, sys, subprocess, requests, etc.).

## Cell Structure

Divide your script into logical cells using `# ---` markers:
```python
# --- Fetch data
orders = shopify.get_orders(status="completed", days=7)
high_value = [o for o in orders if o["total"] > 100]

# --- Analyze trends
class TrendAnalysis(BaseModel):
    trends: list[str]
    top_product: str

analysis = llm("What are the sales trends?", data=high_value, format=TrendAnalysis)

# --- Send report
slack.send_message(
    channel="sales",
    text=f"Weekly trends:\\n" + "\\n".join(f"- {t}" for t in analysis.trends)
)
```

Cell rules:
- Each cell should do ONE logical thing
- Give cells descriptive labels after `# ---`
- Variables from earlier cells are available in later cells
- If a cell fails, cells that depend on its variables are automatically skipped

## Input Variables

The caller may inject input variables into the namespace. Reference them directly:
```python
# --- Send report
# 'recipient_email' and 'date_range' are provided as inputs
gmail.send_email(to=recipient_email, subject="Report", body=summary)
```

## Best Practices

1. **Define Pydantic models before llm() calls** — always use structured output for data you'll process downstream
2. **Keep cells small** — one API call + one transform per cell is ideal
3. **Use native Python** for data manipulation — list comprehensions, dicts, f-strings
4. **Don't add try/except** — the runtime has self-healing that handles errors better than inline catches
5. **Don't add comments explaining obvious code** — the cell labels are documentation enough
6. **Don't print()** — results are captured from the namespace, not stdout

## Output

Return ONLY the Python script. No markdown fences. No explanation before or after.
"""


def generate_workflow(
    prompt: str,
    tools: list[str],
    tool_actions: dict[str, list[str]] | None = None,
    inputs: dict[str, str] | None = None,
    model: str = "gemini/gemini-3.1-pro-preview",
) -> GenerateResult:
    """Generate a workflow script from a natural language description.

    Args:
        prompt: What the workflow should do.
        tools: List of available tool names (e.g., ["gmail", "slack", "shopify"]).
        tool_actions: Optional map of tool → available actions for more precise generation.
        inputs: Optional map of input variable names → descriptions.
        model: LLM model to use for generation.

    Returns:
        GenerateResult with the generated Python source.
    """
    user_parts = [f"## Task\n{prompt}\n"]

    # Tool info
    user_parts.append("## Available Tools\n")
    if tool_actions:
        for tool, actions in tool_actions.items():
            user_parts.append(f"- **{tool}**: {', '.join(actions)}")
    else:
        user_parts.append(f"Tools: {', '.join(tools)}")
    user_parts.append("")

    # Input info
    if inputs:
        user_parts.append("## Input Variables (pre-injected into namespace)\n")
        for name, desc in inputs.items():
            user_parts.append(f"- `{name}`: {desc}")
        user_parts.append("")

    user_message = "\n".join(user_parts)

    response = litellm.completion(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.0,
    )

    raw = response.choices[0].message.content.strip()
    usage = response.usage

    # Strip markdown fences if the LLM wrapped it anyway
    source = _strip_fences(raw)

    return GenerateResult(
        source=source,
        model=model,
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
    )


def refine_workflow(
    source: str,
    dry_run_result: dict,
    error_info: str | None = None,
    feedback: str | None = None,
    model: str = "gemini/gemini-3.1-pro-preview",
) -> GenerateResult:
    """Refine a workflow script based on dry-run analysis or execution errors.

    This is the iterative refinement loop:
    1. generate_workflow() produces v1
    2. dry_run() analyzes it
    3. refine_workflow() fixes issues → v2
    4. Repeat until clean

    Args:
        source: The current workflow script.
        dry_run_result: Output from runtime.dry_run().
        error_info: Optional error from a failed execution.
        feedback: Optional human feedback on what to change.
        model: LLM model to use.

    Returns:
        GenerateResult with the refined Python source.
    """
    user_parts = [
        "## Current Workflow\n",
        f"```python\n{source}\n```\n",
        "## Analysis\n",
        f"```json\n{json.dumps(dry_run_result, indent=2)}\n```\n",
    ]

    if error_info:
        user_parts.append(f"## Execution Error\n{error_info}\n")

    if feedback:
        user_parts.append(f"## Feedback\n{feedback}\n")

    issues = []
    if dry_run_result.get("warnings"):
        issues.extend(dry_run_result["warnings"])
    if dry_run_result.get("missing_tools"):
        issues.append(f"Missing tools: {dry_run_result['missing_tools']}")

    if issues:
        user_parts.append("## Issues to Fix\n")
        for issue in issues:
            user_parts.append(f"- {issue}")
        user_parts.append("")

    user_parts.append("Return the fixed workflow script. Preserve the cell structure.")

    response = litellm.completion(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(user_parts)},
        ],
        temperature=0.0,
    )

    raw = response.choices[0].message.content.strip()
    usage = response.usage
    source = _strip_fences(raw)

    return GenerateResult(
        source=source,
        model=model,
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
    )


def _strip_fences(text: str) -> str:
    """Remove markdown code fences if present."""
    import re

    # Match ```python\n...\n``` or ```\n...\n```
    match = re.match(r'^```(?:python)?\s*\n(.*?)```\s*$', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text
