"""Agentic stress tests — complex multi-service workflows where the agent explores autonomously.

No tool lists provided. The agent discovers everything from the seeded tool catalog,
explores schemas, executes tools to gather context, then writes the script.

Usage: uv run python examples/run_stress_agentic.py
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

from supyworkflow.agent_generator import generate_workflow_agentic
from supyworkflow.runtime import SupyWorkflow

API_KEY = os.environ["SUPYAGENT_API_KEY"]
RECIPIENT = "andre@ergodic.ai"

rt = SupyWorkflow(api_key=API_KEY, user_id="test", timeout_ms=300_000)


def divider(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def run_agentic_stress(name, prompt, context=None):
    divider(f"AGENTIC STRESS: {name}")

    t0 = time.monotonic()
    session = generate_workflow_agentic(
        prompt=prompt,
        api_key=API_KEY,
        context=context,
        max_turns=25,
    )
    gen_time = (time.monotonic() - t0) * 1000

    print(f"Generation: {session.turns} turns, {len(session.tool_calls_made)} tool calls, "
          f"{session.total_tokens} tokens, {gen_time:.0f}ms")

    print(f"\nExploration trace:")
    for i, call in enumerate(session.tool_calls_made):
        tool = call["tool"]
        args = call.get("args", {})
        status = call.get("status", "?")
        if tool == "execute_tool":
            desc = f"{args.get('tool_name', '?')}({json.dumps(args.get('arguments', {}))[:60]})"
        elif tool == "get_tool_schemas":
            names = args.get("tool_names", [])
            desc = f"[{', '.join(names)}]"
        elif tool == "write_script":
            desc = f"({len(args.get('script', ''))} chars)"
        else:
            desc = json.dumps(args)[:80]
        print(f"  {i+1:2d}. [{status:2s}] {tool:20s} {desc}")

    if not session.script:
        print("\n  NO SCRIPT GENERATED")
        return None, session

    print(f"\n--- Script ({len(session.script)} chars) ---")
    print(session.script)
    print(f"--- End ---\n")

    # Execute
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

    return result, session


# ══════════════════════════════════════════════════════════════════
# AGENTIC STRESS TESTS
# ══════════════════════════════════════════════════════════════════


def a01_cross_platform_morning_brief():
    """Gmail + Calendar + Slack + Linear + Google Docs — the agent must discover all IDs."""
    return run_agentic_stress(
        "01: Morning Brief (5 services, all IDs discovered)",
        "Build me a morning briefing. Check my emails, today's calendar, recent activity "
        "in our most active Slack channel (you'll need to figure out which one), "
        "and any open Linear issues assigned to the team. "
        "Compile everything into a Google Doc and send me the link at "
        f"{RECIPIENT}.",
    )


def a02_meeting_intel():
    """Calendar + Web search + News + Gmail — research people/companies from calendar."""
    return run_agentic_stress(
        "02: Meeting Intelligence (Calendar → Research → Email)",
        "Look at my calendar for today. For each meeting, research the people or companies "
        "involved by searching the web and news. Compile a meeting prep dossier with "
        f"key talking points and recent developments. Email the full dossier to {RECIPIENT}.",
    )


def a03_competitive_radar():
    """Search + News + Sheets + Slack — multi-competitor monitoring with spreadsheet output."""
    return run_agentic_stress(
        "03: Competitive Radar (Search + News → Sheets + Slack)",
        "Build a competitive intelligence report. Research the latest news and web presence "
        "for these AI agent platforms: LangChain, CrewAI, AutoGen, Semantic Kernel, and Haystack. "
        "Create a Google Spreadsheet with a comparison matrix (features, funding, latest release, "
        "community size). Post the top 3 most important findings to a Slack channel that seems "
        f"relevant for competitive intelligence. Email a summary to {RECIPIENT}.",
    )


def a04_slack_digest_multi_channel():
    """Slack (multiple channels) + LLM analysis + Google Docs + Email."""
    return run_agentic_stress(
        "04: Multi-Channel Slack Digest (discover channels, read many, synthesize)",
        "Create a comprehensive Slack digest. First, find all the Slack channels in our workspace. "
        "Then read recent messages from the 3 most interesting non-bot channels. "
        "Analyze the conversations: who's most active, what topics are trending, "
        "any decisions made or blockers mentioned. Create a Google Doc with the full analysis "
        f"and email it to {RECIPIENT}.",
    )


def a05_drive_cleanup_report():
    """Drive + Sheets + Slack — audit files, create structured report."""
    return run_agentic_stress(
        "05: Drive Cleanup Report (Drive → Analysis → Sheets + Slack)",
        "Do a Google Drive audit. List my recent files, analyze them for: "
        "naming consistency, duplicate-looking files, old files that haven't been touched, "
        "and overall organization quality. Create a Google Spreadsheet with each file scored "
        "on organization quality (1-5). Add a summary sheet with recommendations. "
        "Post the top 3 recommendations to Slack.",
    )


def a06_data_pipeline_with_db():
    """Code execute + DB + Sheets + Slack — generate data, store in DB, query, report."""
    return run_agentic_stress(
        "06: Data Pipeline (Code → DB → Sheets → Slack)",
        "Run a data analysis pipeline: "
        "1) Use code execution to generate realistic sample data — 50 sales records "
        "with date, product, region, revenue, and units_sold. "
        "2) Create a SQLite database and store the data in it. "
        "3) Query the database for: total revenue by product, top region, and monthly trend. "
        "4) Create a Google Spreadsheet with the query results. "
        "5) Post the key insights to Slack.",
    )


def a07_linear_sprint_review():
    """Linear + Slack + Gmail — deep project management analysis."""
    return run_agentic_stress(
        "07: Sprint Review (Linear → Analysis → Slack + Email)",
        "Create a sprint review report. Explore Linear to find all teams and their issues. "
        "Analyze: how many issues are open/closed, which team has the most blockers, "
        "what's the overall velocity, any overdue items. "
        "Generate a report with team-by-team breakdown and recommendations. "
        f"Post a summary to Slack and email the full report to {RECIPIENT}.",
    )


def a08_research_and_document():
    """Search + Scholar + Docs (with formatting) + Sheets + Email — academic-grade research."""
    return run_agentic_stress(
        "08: Deep Research Paper (Search + Scholar → Docs + Sheets + Email)",
        "Write a research brief on 'The impact of AI agents on software development productivity'. "
        "Search the web for industry reports and search academic papers for peer-reviewed studies. "
        "Create a Google Doc with the research brief structured as: "
        "Executive Summary, Key Findings (from web), Academic Evidence (from papers), "
        "Methodology Notes, and Conclusions. "
        "Create a Google Spreadsheet cataloging every source with title, URL, type, and relevance score. "
        f"Email me at {RECIPIENT} with links to both.",
    )


def a09_financial_intelligence():
    """SEC + Patents + News + Sheets + Email — financial research combining multiple data sources."""
    return run_agentic_stress(
        "09: Financial Intelligence (SEC + Patents + News → Sheets + Email)",
        "Create a financial intelligence report on Microsoft. "
        "Search for recent SEC filings, patents related to AI and cloud computing, "
        "and latest financial news. Analyze: regulatory filings summary, patent activity trends, "
        "market sentiment from news. Create a Google Spreadsheet with separate tabs worth of data "
        "(use a single sheet with clear sections if needed): filings, patents, news sentiment. "
        f"Email an executive summary to {RECIPIENT}.",
    )


def a10_full_workspace_health():
    """Everything — the ultimate agentic stress test with zero hints."""
    return run_agentic_stress(
        "10: FULL WORKSPACE HEALTH CHECK (all services, zero hints)",
        "Run a complete workspace health check. You have full autonomy to explore all available "
        "tools and services. Check everything you can: email, calendar, Slack, Linear, Google Drive. "
        "For each service, assess: activity level, organization quality, and any red flags. "
        "Compute an overall 'Workspace Health Score' from 0-100 with per-service breakdown. "
        "Create a comprehensive Google Spreadsheet with all data. "
        "Create a Google Doc with the narrative health report. "
        "Post a brief scoreboard to Slack. "
        f"Email the full report with all links to {RECIPIENT}. "
        "Be thorough — this is the weekly executive workspace review.",
    )


if __name__ == "__main__":
    results = {}

    tests = [
        ("a01_morning_brief", a01_cross_platform_morning_brief),
        ("a02_meeting_intel", a02_meeting_intel),
        ("a03_competitive_radar", a03_competitive_radar),
        ("a04_slack_digest", a04_slack_digest_multi_channel),
        ("a05_drive_cleanup", a05_drive_cleanup_report),
        ("a06_data_pipeline", a06_data_pipeline_with_db),
        ("a07_sprint_review", a07_linear_sprint_review),
        ("a08_research_paper", a08_research_and_document),
        ("a09_financial_intel", a09_financial_intelligence),
        ("a10_workspace_health", a10_full_workspace_health),
    ]

    for name, fn in tests:
        try:
            result, session = fn()
            results[name] = {
                "result": result,
                "session": session,
            }
        except Exception as e:
            import traceback
            print(f"\n  CRASHED: {type(e).__name__}: {str(e)[:200]}")
            traceback.print_exc()
            results[name] = {"result": None, "session": None}

    divider("AGENTIC STRESS TEST RESULTS")

    total = len(tests)
    passed = partial = failed = 0

    for name, data in results.items():
        r = data["result"]
        s = data["session"]

        if r is None:
            status = "CRASHED" if s is None else "NO SCRIPT"
            cells_info = ""
            failed += 1
        else:
            status = r.status
            healed = len([h for h in r.healed_cells.values() if h.healed]) if r.healed_cells else 0
            cells = len(r.cells)
            ok = sum(1 for c in r.cells if c.status == "completed")
            explore = len(s.tool_calls_made) if s else 0
            cells_info = f"({ok}/{cells} cells, {healed} healed, {explore} explored)"
            if status == "completed":
                passed += 1
            elif status == "partial":
                partial += 1
            else:
                failed += 1

        print(f"  {name:25s} {status:12s} {cells_info}")

    print(f"\n  TOTAL: {passed} passed, {partial} partial, {failed} failed out of {total}")
