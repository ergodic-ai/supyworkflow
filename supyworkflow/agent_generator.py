"""Agentic workflow generator — explores tools and data before writing the script.

Instead of a single-shot LLM call, the generator runs an agent loop where the LLM
can discover tools, inspect schemas, execute tools to explore data, and only commits
to a final script when it has enough context.

Tools available to the agent:
- list_tools() → tool names + descriptions (lightweight, no schemas)
- get_tool_schema(tool_name) → full parameter schema for one tool
- execute_tool(tool_name, **args) → run a tool and see the result
- write_script(source) → commit the final workflow (terminates the loop)
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import litellm
from pydantic import BaseModel

from supyworkflow.llm_builtin import DEFAULT_MODEL
from supyworkflow.tool_proxy import _fetch_tools_metadata, build_tool_callables

logger = logging.getLogger("supyworkflow")

MAX_TURNS = 20


@dataclass
class AgentTurn:
    """A single turn in the agent conversation."""

    role: str  # assistant | tool
    content: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    tool_call_id: str = ""
    name: str = ""


@dataclass
class GenerateSession:
    """Full history of an agentic generation session."""

    session_id: str
    prompt: str
    messages: list[dict] = field(default_factory=list)
    tool_calls_made: list[dict] = field(default_factory=list)
    script: str | None = None
    turns: int = 0
    total_tokens: int = 0
    duration_ms: float = 0

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "prompt": self.prompt,
            "messages": self.messages,
            "tool_calls_made": self.tool_calls_made,
            "script": self.script,
            "turns": self.turns,
            "total_tokens": self.total_tokens,
            "duration_ms": self.duration_ms,
        }


SYSTEM_PROMPT_TEMPLATE = """\
You are a workflow script generator. Your job is to write a Python workflow script \
that automates a task described by the user.

Before writing the script, you should EXPLORE to gather the information you need. \
You have tools to discover what's available and test things out.

## Available Tools

Here are all the tools available in this workspace:

{tool_listing}

## Your Agent Tools

1. **get_tool_schema(tool_name)** — Get the full parameter schema for a specific tool. \
Call this to learn exact parameter names, types, and which are required.

2. **execute_tool(tool_name, arguments)** — Run a tool and see the result. Use this to \
explore: list Slack channels to find the right ID, check calendar events, search the web, etc. \
The `arguments` parameter is a JSON object with the tool's parameters.

3. **write_script(script)** — Submit the final workflow script. This terminates the session. \
Only call this when you have all the information needed.

## Exploration Strategy

- You already know what tools exist (listed above). Skip straight to get_tool_schema() \
for the tools you plan to use.
- Use execute_tool() to discover IDs, test APIs, and gather context.
- Only write_script() when you're confident the script will work.

## Script Format

The script runs in a sandboxed Python environment with:
- Tool functions as globals (e.g., gmail_list_messages, slack_send_message)
- `llm(prompt, data=..., format=PydanticModel)` for LLM intelligence
- `BaseModel` and `Field` from Pydantic for structured output
- Standard library imports: json, re, math, datetime, collections, etc.
- Cell markers: `# ---` to divide logical steps

Example:
```python
# --- Fetch data
emails = gmail_list_messages(maxResults=5)

# --- Analyze
class Summary(BaseModel):
    highlights: list[str]
    action_items: list[str]

summary = llm("Summarize these emails", data=emails, format=Summary)

# --- Send
slack_send_message(channel="C069536PYEQ", text="\\n".join(summary.highlights))
```

IMPORTANT: Use exact tool names and parameter names from get_tool_schema(). \
Use real IDs discovered via execute_tool(), not placeholders.
"""


# Tool definitions for the LLM (OpenAI function calling format)
AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_tool_schema",
            "description": "Get the full parameter schema for a specific tool. Use this to understand exact parameter names, types, and which are required.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tool_name": {
                        "type": "string",
                        "description": "The tool name, e.g. 'gmail_list_messages'",
                    },
                },
                "required": ["tool_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_tool",
            "description": "Execute a tool and return the result. Use this to explore: list Slack channels, check calendar events, search the web, etc. Results are truncated to 4000 chars.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tool_name": {
                        "type": "string",
                        "description": "The tool name to execute",
                    },
                    "arguments": {
                        "type": "object",
                        "description": "Arguments to pass to the tool as key-value pairs",
                    },
                },
                "required": ["tool_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_script",
            "description": "Submit the final workflow script. This terminates the generation session. Only call when you have gathered enough information and are confident the script will work.",
            "parameters": {
                "type": "object",
                "properties": {
                    "script": {
                        "type": "string",
                        "description": "The complete Python workflow script",
                    },
                },
                "required": ["script"],
            },
        },
    },
]


def generate_workflow_agentic(
    prompt: str,
    api_key: str,
    base_url: str = "https://app.supyagent.com",
    model: str | None = None,
    max_turns: int = MAX_TURNS,
    context: str | None = None,
) -> GenerateSession:
    """Generate a workflow script using an agentic exploration loop.

    The LLM can discover tools, inspect schemas, execute tools to explore,
    and only commits to a script when ready.

    Args:
        prompt: What the workflow should do.
        api_key: Cardamon API key for tool execution.
        base_url: Cardamon API base URL.
        model: LLM model identifier.
        max_turns: Maximum agent turns before forcing termination.
        context: Optional additional context (e.g., user preferences, prior session info).

    Returns:
        GenerateSession with the script, full message history, and metadata.
    """
    if model is None:
        model = DEFAULT_MODEL

    session = GenerateSession(
        session_id=uuid.uuid4().hex[:12],
        prompt=prompt,
    )
    start = time.monotonic()

    # Fetch tool metadata
    tools_metadata = _fetch_tools_metadata(api_key, base_url, 30)
    tool_index = {t["function"]["name"]: t for t in tools_metadata}

    # Build callables (for execute_tool)
    tool_callables = build_tool_callables(api_key=api_key, base_url=base_url, tools_metadata=tools_metadata)

    # Build tool listing for the system prompt
    tool_listing = _list_tools(tool_index)

    # Build initial messages with tools seeded in
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(tool_listing=tool_listing)
    messages = [{"role": "system", "content": system_prompt}]

    user_content = f"## Task\n{prompt}"
    if context:
        user_content += f"\n\n## Additional Context\n{context}"

    messages.append({"role": "user", "content": user_content})

    # Agent loop
    for turn in range(max_turns):
        session.turns = turn + 1

        logger.info("agent_turn", extra={"turn": turn + 1, "session": session.session_id})

        response = litellm.completion(
            model=model,
            messages=messages,
            tools=AGENT_TOOLS,
            temperature=0.0,
        )

        usage = response.usage
        if usage:
            session.total_tokens += usage.total_tokens or 0

        choice = response.choices[0]
        assistant_msg = choice.message

        # Add assistant message to history
        msg_dict: dict[str, Any] = {"role": "assistant"}
        if assistant_msg.content:
            msg_dict["content"] = assistant_msg.content
        if assistant_msg.tool_calls:
            msg_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in assistant_msg.tool_calls
            ]
        messages.append(msg_dict)

        # If no tool calls, the agent is done talking without writing a script
        if not assistant_msg.tool_calls:
            if choice.finish_reason == "stop" and session.script is None:
                # Agent stopped without writing — nudge it
                messages.append({
                    "role": "user",
                    "content": "You haven't written the script yet. Please call write_script() with the final workflow.",
                })
                continue
            break

        # Process tool calls
        for tc in assistant_msg.tool_calls:
            fn_name = tc.function.name
            fn_args = json.loads(tc.function.arguments) if tc.function.arguments else {}

            logger.info(
                "agent_tool_call",
                extra={"tool": fn_name, "args_keys": list(fn_args.keys()), "session": session.session_id},
            )

            result = _handle_tool_call(
                fn_name, fn_args, tool_index, tool_callables, session
            )

            # Add tool result to messages
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

            # If write_script was called, we're done
            if session.script is not None:
                break

        if session.script is not None:
            break

    # If we exhausted turns without a script, force one final attempt
    if session.script is None and session.turns >= max_turns:
        messages.append({
            "role": "user",
            "content": "Maximum exploration turns reached. Please call write_script() NOW with the best script you can produce from what you've learned.",
        })

        response = litellm.completion(
            model=model,
            messages=messages,
            tools=AGENT_TOOLS,
            temperature=0.0,
        )

        choice = response.choices[0]
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                if tc.function.name == "write_script":
                    args = json.loads(tc.function.arguments)
                    session.script = args.get("script", "")

    session.messages = messages
    session.duration_ms = (time.monotonic() - start) * 1000

    logger.info(
        "agent_session_complete",
        extra={
            "session": session.session_id,
            "turns": session.turns,
            "total_tokens": session.total_tokens,
            "duration_ms": session.duration_ms,
            "has_script": session.script is not None,
            "tool_calls": len(session.tool_calls_made),
        },
    )

    return session


def _handle_tool_call(
    fn_name: str,
    fn_args: dict,
    tool_index: dict[str, dict],
    tool_callables: dict[str, callable],
    session: GenerateSession,
) -> str:
    """Execute an agent tool call and return the result as a string."""

    call_record = {"tool": fn_name, "args": fn_args, "timestamp": time.time()}

    try:
        if fn_name == "get_tool_schema":
            tool_name = fn_args.get("tool_name", "")
            result = _get_tool_schema(tool_name, tool_index)

        elif fn_name == "execute_tool":
            tool_name = fn_args.get("tool_name", "")
            arguments = fn_args.get("arguments", {})
            result = _execute_tool(tool_name, arguments, tool_callables)

        elif fn_name == "write_script":
            script = fn_args.get("script", "")
            session.script = _strip_fences(script)
            result = "Script saved. Generation complete."

        else:
            result = f"Unknown tool: {fn_name}"

        call_record["status"] = "ok"
        call_record["result_length"] = len(result)

    except Exception as e:
        result = f"Error: {type(e).__name__}: {str(e)[:500]}"
        call_record["status"] = "error"
        call_record["error"] = str(e)[:200]

    session.tool_calls_made.append(call_record)
    return result


def _list_tools(tool_index: dict[str, dict]) -> str:
    """Return lightweight tool listing — names + descriptions only."""
    lines = []
    # Group by service
    services: dict[str, list[tuple[str, str]]] = {}
    for name, meta in tool_index.items():
        service = name.split("_")[0]
        desc = meta["function"].get("description", "")[:80]
        if service not in services:
            services[service] = []
        services[service].append((name, desc))

    for service in sorted(services.keys()):
        lines.append(f"\n## {service}")
        for name, desc in services[service]:
            lines.append(f"- {name}: {desc}")

    return "\n".join(lines)


def _get_tool_schema(tool_name: str, tool_index: dict[str, dict]) -> str:
    """Return full parameter schema for one tool."""
    if tool_name not in tool_index:
        return f"Tool '{tool_name}' not found. Use list_tools() to see available tools."

    tool = tool_index[tool_name]
    func = tool["function"]
    meta = tool.get("metadata", {})

    info = {
        "name": func["name"],
        "description": func.get("description", ""),
        "parameters": func.get("parameters", {}),
        "method": meta.get("method", ""),
        "path": meta.get("path", ""),
    }
    return json.dumps(info, indent=2)


def _execute_tool(
    tool_name: str,
    arguments: dict,
    tool_callables: dict[str, callable],
) -> str:
    """Execute a tool and return truncated result."""
    if tool_name not in tool_callables:
        return f"Tool '{tool_name}' not found or not callable."

    result = tool_callables[tool_name](**arguments)
    serialized = json.dumps(result, default=str, indent=2)

    # Truncate large results
    if len(serialized) > 4000:
        serialized = serialized[:4000] + "\n... (truncated)"

    return serialized


def _strip_fences(text: str) -> str:
    """Remove markdown code fences if present."""
    import re
    match = re.match(r'^```(?:python)?\s*\n(.*?)```\s*$', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text
