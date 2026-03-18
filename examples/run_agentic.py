"""Test the agentic workflow generator.

The agent explores tools and data before writing the script.

Usage: uv run python examples/run_agentic.py
"""

import os
import sys
import json
from pathlib import Path

for line in Path(__file__).parent.parent.joinpath(".env.testing").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        os.environ[k.strip()] = v.strip().strip('"').strip("'")
os.environ["GEMINI_API_KEY"] = os.environ.get("GCP_API_KEY", "")

sys.path.insert(0, str(Path(__file__).parent.parent))

from supyworkflow.agent_generator import generate_workflow_agentic
from supyworkflow.runtime import SupyWorkflow

API_KEY = os.environ["SUPYAGENT_API_KEY"]


def divider(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def test_agentic(name, prompt, context=None):
    divider(f"AGENTIC: {name}")

    session = generate_workflow_agentic(
        prompt=prompt,
        api_key=API_KEY,
        context=context,
    )

    print(f"Session: {session.session_id}")
    print(f"Turns: {session.turns}")
    print(f"Tool calls: {len(session.tool_calls_made)}")
    print(f"Tokens: {session.total_tokens}")
    print(f"Duration: {session.duration_ms:.0f}ms")

    print(f"\nExploration trace:")
    for i, call in enumerate(session.tool_calls_made):
        tool = call["tool"]
        status = call.get("status", "?")
        args = call.get("args", {})
        # Summarize args
        if tool == "execute_tool":
            desc = f"{args.get('tool_name', '?')}({json.dumps(args.get('arguments', {}))[:80]})"
        elif tool == "get_tool_schema":
            desc = args.get("tool_name", "?")
        elif tool == "write_script":
            desc = f"({len(args.get('script', ''))} chars)"
        else:
            desc = ""
        print(f"  {i+1:2d}. [{status:2s}] {tool}  {desc}")

    if session.script:
        print(f"\n--- Generated Script ---")
        print(session.script)
        print(f"--- End ({len(session.script)} chars) ---\n")

        # Execute it
        rt = SupyWorkflow(api_key=API_KEY, user_id="test", timeout_ms=180_000)
        result = rt.run(session.script)

        print(f"Execution: {result.status}")
        for cell in result.cells:
            heal = ""
            if cell.index in (result.healed_cells or {}):
                h = result.healed_cells[cell.index]
                if h.healed:
                    heal = f" [HEALED x{h.attempts}]"
            print(f"  [{cell.status:9s}] {cell.label} ({cell.duration_ms:.0f}ms){heal}")
            if cell.error:
                print(f"              {str(cell.error)[:120]}")
        return result
    else:
        print("\nNo script generated!")
        return None


# ── Test 1: Simple but requires discovery ───────────────────────
# The agent needs to find the right Slack channel ID

def test01():
    return test_agentic(
        "Slack Message (needs channel discovery)",
        "Send a message to the #random Slack channel saying "
        "'Hello from supyworkflow agentic generator! This message was crafted by an AI agent that explored the workspace first.'",
    )


# ── Test 2: Complex — needs to explore multiple services ────────

def test02():
    return test_agentic(
        "Cross-Service Report (needs exploration)",
        "Create a brief status report: check my latest 3 emails, "
        "today's calendar, and recent Slack activity in any active channel. "
        "Post a summary to a Slack channel that seems appropriate for team updates. "
        "Email the full report to andre@ergodic.ai.",
    )


# ── Test 3: Research with specific data gathering ───────────────

def test03():
    return test_agentic(
        "Research + Spreadsheet (needs web exploration)",
        "Research the current state of 'AI coding assistants' — search the web, "
        "find the top 5 products, and create a Google Spreadsheet comparing them "
        "on features, pricing, and user reviews. Email me a link at andre@ergodic.ai.",
    )


if __name__ == "__main__":
    r1 = test01()
    r2 = test02()
    r3 = test03()

    divider("RESULTS")
    for name, r in [("test01", r1), ("test02", r2), ("test03", r3)]:
        if r is None:
            print(f"  {name}: NO SCRIPT")
        else:
            print(f"  {name}: {r.status}")
