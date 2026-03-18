"""Generate workflows from natural language prompts, then execute them.

Tests the full pipeline: prompt → generate → dry_run → execute
No hand-written workflow code — everything is LLM-generated.

Usage: uv run python examples/run_generated.py
"""

import os
import sys
import json
from pathlib import Path

# Load env
for line in Path(__file__).parent.parent.joinpath(".env.testing").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        os.environ[k.strip()] = v.strip().strip('"').strip("'")
os.environ["GEMINI_API_KEY"] = os.environ.get("GCP_API_KEY", "")

sys.path.insert(0, str(Path(__file__).parent.parent))

from supyworkflow.generator import generate_workflow
from supyworkflow.runtime import SupyWorkflow
from supyworkflow.tool_proxy import _fetch_tools_metadata

API_KEY = os.environ["SUPYAGENT_API_KEY"]
SLACK_GENERAL = "C069536PYEQ"
RECIPIENT = "andre@ergodic.ai"

# Fetch real tool metadata so the generator knows exact parameter names
print("Discovering tools...")
meta = _fetch_tools_metadata(API_KEY, "https://app.supyagent.com", 30)

# Build tool_actions map with parameter info
tool_actions = {}
for t in meta:
    name = t["function"]["name"]
    params = t["function"].get("parameters", {}).get("properties", {})
    param_desc = []
    required = t["function"].get("parameters", {}).get("required", [])
    for pname, pinfo in params.items():
        req = " (required)" if pname in required else ""
        param_desc.append(f"{pname}: {pinfo.get('type', '?')} - {pinfo.get('description', '')}{req}")
    tool_actions[name] = param_desc

# Only include tools we'll actually use
RELEVANT_TOOLS = [
    "gmail_list_messages", "gmail_send_message",
    "calendar_list_events",
    "slack_list_channels", "slack_get_channel_messages", "slack_send_message",
    "search_web", "search_news",
    "sheets_create_spreadsheet", "sheets_update_values",
]
filtered_actions = {k: v for k, v in tool_actions.items() if k in RELEVANT_TOOLS}

print(f"Tools available: {len(filtered_actions)}")


def divider(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def generate_and_run(name, prompt, inputs=None):
    divider(f"GENERATING: {name}")

    gen = generate_workflow(
        prompt=prompt,
        tools=list(filtered_actions.keys()),
        tool_actions=filtered_actions,
        inputs=inputs,
    )

    print(f"Generated {len(gen.source)} chars ({gen.prompt_tokens} prompt, {gen.completion_tokens} completion tokens)")
    print(f"\n--- Generated Script ---")
    print(gen.source)
    print(f"--- End ---\n")

    rt = SupyWorkflow(api_key=API_KEY, user_id="test", timeout_ms=180_000)

    # Dry run
    analysis = rt.dry_run(gen.source)
    print(f"Dry run: {len(analysis['cells'])} cells, tools={analysis['tools_used']}")

    # Execute
    result = rt.run(gen.source, inputs=inputs)

    print(f"\nStatus: {result.status}")
    for cell in result.cells:
        print(f"  [{cell.status}] {cell.label} ({cell.duration_ms:.1f}ms)")
        if cell.error:
            print(f"       ERROR: {cell.error}")
        if cell.index in (result.healed_cells or {}):
            h = result.healed_cells[cell.index]
            if h.healed:
                print(f"       HEALED ({h.attempts}x)")

    if result.status in ("completed", "partial"):
        print(f"\nOutputs:")
        for k, v in result.outputs.items():
            preview = str(v)
            if len(preview) > 150:
                preview = preview[:150] + "..."
            print(f"  {k} = {preview}")

    return result


# ── Prompt 1: Email triage ──────────────────────────────────────

def gen01_email_triage():
    return generate_and_run(
        "Email Triage",
        f"Check my latest 5 emails from Gmail. Categorize each email as 'urgent', "
        f"'important', or 'low priority'. For any urgent emails, send a Slack message "
        f"to channel {SLACK_GENERAL} with the subject and why it's urgent.",
    )


# ── Prompt 2: Meeting prep ─────────────────────────────────────

def gen02_meeting_prep():
    return generate_and_run(
        "Meeting Prep",
        "Get my calendar events for today. For each meeting, search the web for "
        "recent news about the companies or people involved. Create a prep briefing "
        "and email it to me.",
        inputs={"my_email": RECIPIENT},
    )


# ── Prompt 3: Industry news digest ─────────────────────────────

def gen03_industry_news():
    return generate_and_run(
        "Industry News Digest",
        f"Search for the latest news about 'AI agents' and 'enterprise automation'. "
        f"Combine the results, identify the top 5 most important stories, and write "
        f"a brief intelligence brief. Post it to Slack channel {SLACK_GENERAL}.",
    )


# ── Prompt 4: Competitive landscape ────────────────────────────

def gen04_competitive_landscape():
    return generate_and_run(
        "Competitive Landscape",
        f"Search the web for recent news about these companies: Anthropic, OpenAI, Google DeepMind. "
        f"Compare what each is doing in the AI agents space. Write a competitive analysis "
        f"and email it to {RECIPIENT} with the subject 'Weekly Competitive Landscape'.",
    )


# ── Prompt 5: Slack channel analytics ──────────────────────────

def gen05_slack_analytics():
    return generate_and_run(
        "Slack Analytics",
        f"Read the last 30 messages from Slack channel {SLACK_GENERAL}. "
        f"Analyze the conversation: who's most active, what topics are trending, "
        f"what's the overall mood. Create a structured report and post a summary "
        f"back to the same channel.",
    )


if __name__ == "__main__":
    results = {}
    results["01_email_triage"] = gen01_email_triage()
    results["02_meeting_prep"] = gen02_meeting_prep()
    results["03_industry_news"] = gen03_industry_news()
    results["04_competitive"] = gen04_competitive_landscape()
    results["05_slack_analytics"] = gen05_slack_analytics()

    divider("FINAL RESULTS")
    for name, r in results.items():
        healed = len([h for h in r.healed_cells.values() if h.healed]) if r.healed_cells else 0
        cells = len(r.cells)
        ok = sum(1 for c in r.cells if c.status == "completed")
        print(f"  {name:25s} {r.status:12s} ({ok}/{cells} cells, {healed} healed)")
