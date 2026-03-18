"""Run all example workflows with real data.

Usage: uv run python examples/run_live.py
"""

import os
import sys
import json
from pathlib import Path

# Load env
env_file = Path(__file__).parent.parent / ".env.testing"
for line in env_file.read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        os.environ[k.strip()] = v.strip().strip('"').strip("'")
os.environ["GEMINI_API_KEY"] = os.environ.get("GCP_API_KEY", "")

sys.path.insert(0, str(Path(__file__).parent.parent))

from supyworkflow.runtime import SupyWorkflow

API_KEY = os.environ["SUPYAGENT_API_KEY"]
SLACK_GENERAL = "C069536PYEQ"
SPREADSHEET_ID = "1qKQ0FdHoaB5MGqOhKXKS4IYoUzygS7ExYo7HyVxZGPU"
RECIPIENT = "andre@ergodic.ai"


def divider(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def run(name, source, inputs=None):
    divider(name)
    rt = SupyWorkflow(api_key=API_KEY, user_id="test", timeout_ms=180_000)
    result = rt.run(source, inputs=inputs)

    print(f"Status: {result.status}")
    for cell in result.cells:
        print(f"  [{cell.status}] {cell.label} ({cell.duration_ms:.1f}ms)")
        if cell.error:
            print(f"       ERROR: {cell.error}")
        if cell.index in (result.healed_cells or {}):
            h = result.healed_cells[cell.index]
            if h.healed:
                print(f"       HEALED ({h.attempts} attempts)")

    if result.status in ("completed", "partial"):
        print(f"\nKey outputs:")
        for k, v in result.outputs.items():
            preview = str(v)
            if len(preview) > 150:
                preview = preview[:150] + "..."
            print(f"  {k} = {preview}")

    if result.trace:
        s = result.trace.summary()
        print(f"\nTrace: {s['tool_calls']} tool, {s['llm_calls']} llm, {s['total_duration_ms']:.0f}ms")

    return result


# ── 01: Email Digest → Slack #general ───────────────────────────

def ex01_email_digest():
    return run("01 Email Digest → Slack", f"""
# --- Fetch recent emails
emails = gmail_list_messages(maxResults=5)

# --- Analyze emails
class EmailDigest(BaseModel):
    total_count: int
    unread_count: int
    highlights: list[str]
    action_items: list[str]
    summary: str

digest = llm(
    "Analyze these emails. Identify highlights, action items, and write a brief summary.",
    data=emails,
    format=EmailDigest,
)

# --- Format and send to Slack
lines = ["*Email Digest* (" + str(digest.total_count) + " emails, " + str(digest.unread_count) + " unread)\\n"]
lines.append("_" + digest.summary + "_\\n")

if digest.highlights:
    lines.append("*Highlights:*")
    for h in digest.highlights:
        lines.append("  - " + h)

if digest.action_items:
    lines.append("\\n*Action Items:*")
    for a in digest.action_items:
        lines.append("  - " + a)

message = "\\n".join(lines)
slack_send_message(channel="{SLACK_GENERAL}", text=message)
""")


# ── 02: Calendar Briefing ───────────────────────────────────────

def ex02_calendar_briefing():
    return run("02 Calendar Briefing", """
import datetime

# --- Get today's events
today = datetime.date.today().isoformat()
events = calendar_list_events(timeMin=today + "T00:00:00Z", timeMax=today + "T23:59:59Z", maxResults=20)

# --- Generate briefing
class MeetingPrep(BaseModel):
    title: str
    time: str
    attendees: list[str]
    prep_notes: str

class DayBriefing(BaseModel):
    meeting_count: int
    free_hours: float
    meetings: list[MeetingPrep]
    day_summary: str

briefing = llm(
    "Create a daily briefing from these calendar events. "
    "For each meeting, suggest brief prep notes based on the title and attendees.",
    data=events,
    format=DayBriefing,
)

# --- Format output
lines = ["*Daily Briefing for " + today + "*"]
lines.append(str(briefing.meeting_count) + " meetings, ~" + str(briefing.free_hours) + "h free\\n")
lines.append("_" + briefing.day_summary + "_\\n")

for m in briefing.meetings:
    lines.append("*" + m.time + " - " + m.title + "*")
    lines.append("  Attendees: " + ", ".join(m.attendees))
    lines.append("  Prep: " + m.prep_notes + "\\n")

output = "\\n".join(lines)
""")


# ── 03: Research Report → Google Sheets ─────────────────────────

def ex03_research_report():
    return run("03 Research Report → Sheets", f"""
# --- Search for information
results = search_web(query="AI agents enterprise adoption 2025", num=8)

# --- Analyze and synthesize
class ResearchFinding(BaseModel):
    source: str
    key_insight: str
    relevance: str

class ResearchReport(BaseModel):
    findings: list[ResearchFinding]
    executive_summary: str
    key_themes: list[str]

report = llm(
    "Analyze these search results about AI agents in enterprise. "
    "Extract key findings from each source, identify themes.",
    data=results,
    format=ResearchReport,
)

# --- Save to Google Sheets
rows = [["Source", "Key Insight", "Relevance"]]
for f in report.findings:
    rows.append([f.source, f.key_insight, f.relevance])

rows.append([])
rows.append(["Executive Summary"])
rows.append([report.executive_summary])
rows.append([])
rows.append(["Key Themes"])
for theme in report.key_themes:
    rows.append([theme])

sheets_update_values(
    spreadsheetId="{SPREADSHEET_ID}",
    range="Sheet1!A1",
    values=rows,
)
""")


# ── 04: Competitor Monitor → Email ──────────────────────────────

def ex04_competitor_monitor():
    return run("04 Competitor Monitor → Email", f"""
# --- Search for competitor news
news = search_news(query="OpenAI latest news", num=8)

# --- Analyze coverage
class NewsItem(BaseModel):
    headline: str
    source: str
    sentiment: str
    key_takeaway: str

class CompetitorReport(BaseModel):
    items: list[NewsItem]
    overall_sentiment: str
    strategic_implications: list[str]
    recommended_actions: list[str]

analysis = llm(
    "Analyze recent news coverage about OpenAI. "
    "Assess sentiment for each item and identify strategic implications.",
    data=news,
    format=CompetitorReport,
)

# --- Build and send email report
lines = ["Competitor Intelligence: OpenAI\\n"]
lines.append("Overall Sentiment: " + analysis.overall_sentiment + "\\n")

lines.append("Recent Coverage:")
for item in analysis.items:
    lines.append("  [" + item.sentiment.upper() + "] " + item.headline + " (" + item.source + ")")
    lines.append("    -> " + item.key_takeaway)

lines.append("\\nStrategic Implications:")
for imp in analysis.strategic_implications:
    lines.append("  - " + imp)

lines.append("\\nRecommended Actions:")
for action in analysis.recommended_actions:
    lines.append("  - " + action)

report_text = "\\n".join(lines)

gmail_send_message(
    to="{RECIPIENT}",
    subject="Competitor Report: OpenAI",
    body=report_text,
)
""")


# ── 05: Slack Standup Summary ───────────────────────────────────

def ex05_slack_standup():
    return run("05 Slack Standup Summary", f"""
# --- Fetch recent messages from #general
messages = slack_get_channel_messages(channelId="{SLACK_GENERAL}", limit=20)

# --- Analyze messages
class ChannelSummary(BaseModel):
    active_people: list[str]
    topics_discussed: list[str]
    summary: str

analysis = llm(
    "Analyze these Slack messages from #general. "
    "Identify who was active and what topics were discussed. Be concise.",
    data=messages,
    format=ChannelSummary,
)

# --- Post summary back to Slack
parts = []
parts.append("*Channel Summary* (#general)")
parts.append(analysis.summary)

if analysis.active_people:
    parts.append("*Active:* " + ", ".join(analysis.active_people))

if analysis.topics_discussed:
    parts.append("*Topics:*")
    for t in analysis.topics_discussed:
        parts.append("  - " + str(t))

summary_text = chr(10).join(parts)
slack_send_message(channel="{SLACK_GENERAL}", text=summary_text)
""")


if __name__ == "__main__":
    results = {}
    results["01_email_digest"] = ex01_email_digest()
    results["02_calendar"] = ex02_calendar_briefing()
    results["03_research"] = ex03_research_report()
    results["04_competitor"] = ex04_competitor_monitor()
    results["05_standup"] = ex05_slack_standup()

    divider("FINAL RESULTS")
    for name, r in results.items():
        healed = len([h for h in r.healed_cells.values() if h.healed]) if r.healed_cells else 0
        cells = len(r.cells)
        ok = sum(1 for c in r.cells if c.status == "completed")
        print(f"  {name:25s} {r.status:12s} ({ok}/{cells} cells, {healed} healed)")
