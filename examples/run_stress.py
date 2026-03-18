"""Stress test: LLM-generated workflows using every available service.

Tests complex multi-service workflows, edge cases, deep pipelines.
All workflows generated from prompts — zero hand-written code.

Usage: uv run python examples/run_stress.py
"""

import os
import sys
import json
import time
from pathlib import Path

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
SLACK_RANDOM = "C0692G2MMT5"
RECIPIENT = "andre@ergodic.ai"

# Build full tool action metadata
print("Discovering tools...")
meta = _fetch_tools_metadata(API_KEY, "https://app.supyagent.com", 30)

tool_actions = {}
for t in meta:
    name = t["function"]["name"]
    params = t["function"].get("parameters", {}).get("properties", {})
    required = t["function"].get("parameters", {}).get("required", [])
    param_desc = []
    for pname, pinfo in params.items():
        req = " (required)" if pname in required else ""
        param_desc.append(f"{pname}: {pinfo.get('type', '?')} - {pinfo.get('description', '')}{req}")
    tool_actions[name] = param_desc

print(f"Total tools: {len(tool_actions)}")

rt = SupyWorkflow(api_key=API_KEY, user_id="test", timeout_ms=300_000)


def divider(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def generate_and_run(name, prompt, inputs=None, tools=None):
    divider(f"STRESS {name}")

    filtered = {k: v for k, v in tool_actions.items() if k in tools} if tools else tool_actions

    t0 = time.monotonic()
    gen = generate_workflow(
        prompt=prompt,
        tools=list(filtered.keys()),
        tool_actions=filtered,
        inputs=inputs,
    )
    gen_time = (time.monotonic() - t0) * 1000

    print(f"Generated {len(gen.source)} chars in {gen_time:.0f}ms")
    print(f"\n--- Script ---\n{gen.source}\n--- End ---\n")

    result = rt.run(gen.source, inputs=inputs)

    print(f"Status: {result.status}")
    for cell in result.cells:
        status_icon = {"completed": "OK", "failed": "FAIL", "skipped": "SKIP"}.get(cell.status, "?")
        heal_note = ""
        if cell.index in (result.healed_cells or {}):
            h = result.healed_cells[cell.index]
            if h.healed:
                heal_note = f" [HEALED x{h.attempts}]"
        print(f"  [{status_icon:4s}] {cell.label} ({cell.duration_ms:.0f}ms){heal_note}")
        if cell.error:
            err_str = str(cell.error)[:120]
            print(f"         {err_str}")

    if result.status in ("completed", "partial"):
        out_keys = [k for k in result.outputs.keys() if not k.startswith("_")]
        print(f"\nOutputs: {out_keys}")

    return result


# ══════════════════════════════════════════════════════════════════
# STRESS TESTS
# ══════════════════════════════════════════════════════════════════


def s01_cross_platform_digest():
    """Gmail + Calendar + Slack + Google Docs — 4 services in one workflow."""
    return generate_and_run(
        "01: Cross-Platform Daily Digest (Gmail + Calendar + Slack → Google Docs)",
        f"Create a comprehensive daily digest by doing ALL of the following:\n"
        f"1. Fetch my last 10 emails from Gmail\n"
        f"2. Get today's calendar events\n"
        f"3. Read the last 20 messages from Slack channel {SLACK_RANDOM}\n"
        f"4. Analyze everything together: categorize emails by urgency, summarize meetings, "
        f"identify key Slack discussions\n"
        f"5. Create a new Google Doc titled 'Daily Digest - [today's date]' with a formatted "
        f"report containing all the analysis\n"
        f"6. Send a brief summary to Slack channel {SLACK_RANDOM} with a link to the doc",
        tools=[
            "gmail_list_messages", "calendar_list_events",
            "slack_get_channel_messages", "slack_send_message",
            "docs_create_document", "docs_insert_text",
        ],
    )


def s02_research_to_spreadsheet():
    """Web search + News + Sheets + Gmail — multi-source research pipeline."""
    return generate_and_run(
        "02: Deep Research Pipeline (Web + News + Scholar → Sheets → Email)",
        f"Conduct comprehensive research on 'Large Language Model agents in production':\n"
        f"1. Search the web for general information (10 results)\n"
        f"2. Search news for recent developments (10 results)\n"
        f"3. Search academic papers on the topic (5 results)\n"
        f"4. Synthesize all findings into a structured analysis with: key themes, "
        f"top findings, market trends, academic insights, and gaps in knowledge\n"
        f"5. Create a new Google Spreadsheet titled 'LLM Agents Research'\n"
        f"6. Populate the spreadsheet with: a 'Findings' sheet with source/insight/category columns, "
        f"and summary data\n"
        f"7. Email the research summary to {RECIPIENT} with the spreadsheet link",
        tools=[
            "search_web", "search_news", "search_scholar",
            "sheets_create_spreadsheet", "sheets_update_values",
            "gmail_send_message",
        ],
    )


def s03_linear_slack_sync():
    """Linear + Slack — project management integration."""
    return generate_and_run(
        "03: Linear → Slack Project Status (Linear + Slack)",
        f"Create a project status update:\n"
        f"1. List all teams in Linear\n"
        f"2. For each team, list their recent issues (last 10)\n"
        f"3. Analyze the issues: how many are open vs closed, what's the velocity, "
        f"are there any blockers or stale issues\n"
        f"4. Generate a project health report with team-by-team breakdown\n"
        f"5. Post the report to Slack channel {SLACK_RANDOM}",
        tools=[
            "linear_list_teams", "linear_list_issues", "linear_list_members",
            "slack_send_message",
        ],
    )


def s04_drive_audit():
    """Google Drive + Sheets — file audit and reporting."""
    return generate_and_run(
        "04: Google Drive Audit → Spreadsheet (Drive + Sheets)",
        f"Perform an audit of my Google Drive:\n"
        f"1. List files from Google Drive (get up to 20 files)\n"
        f"2. Analyze the files: categorize by type (docs, sheets, slides, etc.), "
        f"identify which are recent vs old, find any naming pattern issues\n"
        f"3. Create a new Google Spreadsheet titled 'Drive Audit Report'\n"
        f"4. Populate it with: file name, type, last modified date, size, "
        f"and an AI-generated 'health' score (well-named, recent, properly organized)\n"
        f"5. Add a summary row at the top with total files, breakdown by type, "
        f"and overall organization score",
        tools=[
            "drive_list_files",
            "sheets_create_spreadsheet", "sheets_update_values",
        ],
    )


def s05_competitor_deep_dive():
    """News + Web + Docs + Email — massive research with doc creation."""
    return generate_and_run(
        "05: Multi-Competitor Deep Dive (Search + News + Docs + Email)",
        f"Do a deep competitive analysis of AI agent platforms:\n"
        f"1. Search news about each: 'LangChain agents', 'CrewAI', 'AutoGen Microsoft', 'Anthropic Claude agents'\n"
        f"2. Search the web for each company's agent capabilities\n"
        f"3. Compare all competitors on: features, pricing model, enterprise readiness, "
        f"developer experience, and ecosystem/integrations\n"
        f"4. Create a comparison matrix (structured data)\n"
        f"5. Write an executive brief with winner for each category\n"
        f"6. Create a Google Doc titled 'AI Agent Platforms - Competitive Analysis' with the full report\n"
        f"7. Email a short executive summary to {RECIPIENT}",
        tools=[
            "search_news", "search_web",
            "docs_create_document", "docs_insert_text",
            "gmail_send_message",
        ],
    )


def s06_data_pipeline():
    """Code execution + Sheets + Slack — compute and report."""
    return generate_and_run(
        "06: Data Pipeline (Code Execute + Sheets + Slack)",
        f"Run a data analysis pipeline:\n"
        f"1. Use code_execute to run Python code that generates sample sales data: "
        f"100 rows with columns [date, product, region, revenue, units_sold]. "
        f"Use random but realistic values. Return the data as JSON.\n"
        f"2. Analyze the data using LLM: find top products, regional trends, "
        f"revenue patterns, and anomalies\n"
        f"3. Create a Google Spreadsheet titled 'Sales Analysis Pipeline'\n"
        f"4. Write the raw data to Sheet1 and a summary to Sheet1 below the data\n"
        f"5. Post the key insights to Slack channel {SLACK_RANDOM}",
        tools=[
            "code_execute",
            "sheets_create_spreadsheet", "sheets_update_values",
            "slack_send_message",
        ],
    )


def s07_slack_to_linear():
    """Slack + Linear + Email — extract action items and create tickets."""
    return generate_and_run(
        "07: Slack → Linear Tickets (Slack + Linear + Email)",
        f"Extract action items from Slack and create Linear issues:\n"
        f"1. Read the last 30 messages from Slack channel {SLACK_RANDOM}\n"
        f"2. Use LLM to identify any action items, bugs, feature requests, "
        f"or tasks mentioned in the messages\n"
        f"3. List existing Linear teams to find the right team\n"
        f"4. For each action item found (max 3), create a Linear issue "
        f"with appropriate title, description, and priority\n"
        f"5. Post a confirmation to Slack channel {SLACK_RANDOM} listing the created tickets\n"
        f"6. Email a summary to {RECIPIENT}",
        tools=[
            "slack_get_channel_messages", "slack_send_message",
            "linear_list_teams", "linear_create_issue",
            "gmail_send_message",
        ],
    )


def s08_newsletter_generator():
    """Search + Slides + Gmail — create a visual newsletter."""
    return generate_and_run(
        "08: Auto Newsletter (Search + Docs + Email)",
        f"Generate a weekly AI newsletter:\n"
        f"1. Search news for 'artificial intelligence breakthroughs' (10 results)\n"
        f"2. Search news for 'AI startups funding' (10 results)\n"
        f"3. Search web for 'AI tools launched this week' (10 results)\n"
        f"4. Curate the top 8 stories across all sources\n"
        f"5. Write a newsletter with sections: 'Breakthroughs', 'Funding & Startups', "
        f"'New Tools', and an 'Editor's Pick' for the most important story\n"
        f"6. Create a Google Doc titled 'AI Weekly Newsletter' with the formatted content\n"
        f"7. Email the newsletter to {RECIPIENT} with subject 'AI Weekly Newsletter'",
        tools=[
            "search_news", "search_web",
            "docs_create_document", "docs_insert_text",
            "gmail_send_message",
        ],
    )


def s09_financial_research():
    """Specialized search APIs — SEC, patents, economic data."""
    return generate_and_run(
        "09: Financial Research (SEC + Patents + Economic Data → Sheets)",
        f"Conduct financial and IP research on NVIDIA:\n"
        f"1. Search SEC filings for NVIDIA\n"
        f"2. Search patents related to 'NVIDIA GPU AI training'\n"
        f"3. Search for NVIDIA stock news\n"
        f"4. Analyze: recent SEC filings, patent activity trends, market sentiment\n"
        f"5. Create a Google Spreadsheet titled 'NVIDIA Research Report'\n"
        f"6. Populate with: SEC filings summary, patent count and themes, "
        f"news sentiment analysis\n"
        f"7. Email the findings to {RECIPIENT}",
        tools=[
            "search_sec_filings", "search_patents", "search_news",
            "sheets_create_spreadsheet", "sheets_update_values",
            "gmail_send_message",
        ],
    )


def s10_full_workspace_sync():
    """Everything at once — the ultimate stress test."""
    return generate_and_run(
        "10: FULL WORKSPACE SYNC (Gmail + Calendar + Slack + Linear + Drive + Sheets + Docs + Email)",
        f"Create a comprehensive weekly workspace report:\n"
        f"1. Fetch my last 15 emails from Gmail\n"
        f"2. Get this week's calendar events\n"
        f"3. Read last 20 messages from Slack channel {SLACK_RANDOM}\n"
        f"4. List all Linear issues across all teams\n"
        f"5. List recent files from Google Drive\n"
        f"6. Analyze EVERYTHING together:\n"
        f"   - Email: volume, categories, response-needed count\n"
        f"   - Calendar: meeting load, free time available\n"
        f"   - Slack: team activity level, key discussions\n"
        f"   - Linear: sprint progress, blockers, velocity\n"
        f"   - Drive: recent document activity\n"
        f"7. Generate a 'Weekly Workspace Health Score' (0-100) with breakdown\n"
        f"8. Create a Google Spreadsheet titled 'Weekly Workspace Report' with all the data\n"
        f"9. Create a Google Doc titled 'Weekly Workspace Summary' with the narrative report\n"
        f"10. Post a brief summary to Slack channel {SLACK_RANDOM}\n"
        f"11. Email the full report to {RECIPIENT} with links to the doc and spreadsheet",
        tools=[
            "gmail_list_messages", "calendar_list_events",
            "slack_get_channel_messages", "slack_send_message",
            "linear_list_teams", "linear_list_issues",
            "drive_list_files",
            "sheets_create_spreadsheet", "sheets_update_values",
            "docs_create_document", "docs_insert_text",
            "gmail_send_message",
        ],
    )


if __name__ == "__main__":
    results = {}

    tests = [
        ("s01_cross_platform", s01_cross_platform_digest),
        ("s02_research_pipeline", s02_research_to_spreadsheet),
        ("s03_linear_slack", s03_linear_slack_sync),
        ("s04_drive_audit", s04_drive_audit),
        ("s05_competitor_deep", s05_competitor_deep_dive),
        ("s06_data_pipeline", s06_data_pipeline),
        ("s07_slack_to_linear", s07_slack_to_linear),
        ("s08_newsletter", s08_newsletter_generator),
        ("s09_financial", s09_financial_research),
        ("s10_full_workspace", s10_full_workspace_sync),
    ]

    for name, fn in tests:
        try:
            results[name] = fn()
        except Exception as e:
            print(f"\n  CRASHED: {e}")
            results[name] = None

    divider("FINAL STRESS TEST RESULTS")
    total = len(tests)
    passed = 0
    partial = 0
    failed = 0

    for name, r in results.items():
        if r is None:
            status = "CRASHED"
            cells_info = ""
            failed += 1
        else:
            status = r.status
            healed = len([h for h in r.healed_cells.values() if h.healed]) if r.healed_cells else 0
            cells = len(r.cells)
            ok = sum(1 for c in r.cells if c.status == "completed")
            cells_info = f"({ok}/{cells} cells, {healed} healed)"
            if status == "completed":
                passed += 1
            elif status == "partial":
                partial += 1
            else:
                failed += 1

        print(f"  {name:25s} {status:12s} {cells_info}")

    print(f"\n  TOTAL: {passed} passed, {partial} partial, {failed} failed out of {total}")
