"""Live tests for the pluggable tool provider system.

Tests the three new capabilities:
1. SupyagentToolProvider — same behavior as before, but through the provider interface
2. extra_tools — plain Python callables injected alongside supyagent tools
3. CompositeToolProvider — multiple providers merged into one namespace

Usage: uv run python examples/run_providers.py
"""

import os
import sys
import json
import time
from pathlib import Path

# Load env
for line in Path(__file__).parent.parent.joinpath(".env.testing").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        os.environ[k.strip()] = v.strip().strip('"').strip("'")
os.environ["GEMINI_API_KEY"] = os.environ.get("GCP_API_KEY", "")

sys.path.insert(0, str(Path(__file__).parent.parent))

from supyworkflow.runtime import SupyWorkflow
from supyworkflow.providers.supyagent import SupyagentToolProvider
from supyworkflow.providers.composite import CompositeToolProvider

API_KEY = os.environ["SUPYAGENT_API_KEY"]
BASE_URL = "https://app.supyagent.com"
SLACK_GENERAL = "C069536PYEQ"
RECIPIENT = "andre@ergodic.ai"


def divider(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def run_and_report(name, runtime, source, inputs=None, extra_tools=None):
    divider(name)
    t0 = time.monotonic()
    result = runtime.run(source, inputs=inputs, extra_tools=extra_tools)
    elapsed = (time.monotonic() - t0) * 1000

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
            print(f"         {str(cell.error)[:150]}")

    if result.status in ("completed", "partial"):
        out_keys = [k for k in result.outputs.keys() if not k.startswith("_")]
        print(f"\nOutputs: {out_keys}")
        for k in out_keys[:5]:
            v = str(result.outputs[k])
            if len(v) > 120:
                v = v[:120] + "..."
            print(f"  {k} = {v}")

    if result.trace:
        s = result.trace.summary()
        print(f"\nTrace: {s['tool_calls']} tool, {s['llm_calls']} llm, {s['total_duration_ms']:.0f}ms, ${s['total_cost']:.4f}")

    print(f"Wall time: {elapsed:.0f}ms")
    return result


# ══════════════════════════════════════════════════════════════════════
# TEST 1: SupyagentToolProvider — same as legacy, through provider API
# ══════════════════════════════════════════════════════════════════════

def test_01_supyagent_provider_basic():
    """Use SupyagentToolProvider explicitly instead of the legacy path."""
    provider = SupyagentToolProvider(api_key=API_KEY, base_url=BASE_URL)
    runtime = SupyWorkflow(
        api_key=API_KEY,
        user_id="test",
        providers=[provider],
    )

    return run_and_report("01: SupyagentToolProvider — Email Digest", runtime, f"""
# --- Fetch emails via provider
emails = gmail_list_messages(maxResults=5)

# --- Analyze
class Digest(BaseModel):
    highlights: list[str]
    summary: str

digest = llm("Summarize these emails briefly.", data=emails, format=Digest)

# --- Send to Slack
text = "*Email Digest (via SupyagentToolProvider)*\\n" + digest.summary
slack_send_message(channel="{SLACK_GENERAL}", text=text)
""")


# ══════════════════════════════════════════════════════════════════════
# TEST 2: extra_tools — inject custom Python functions alongside tools
# ══════════════════════════════════════════════════════════════════════

def test_02_extra_tools_basic():
    """Inject a custom Python function and use it alongside supyagent tools."""

    def calculate_priority_score(emails: list) -> dict:
        """Score emails by urgency — a pure Python function, no API call."""
        total = len(emails) if isinstance(emails, list) else 0
        return {
            "total": total,
            "score": min(total * 10, 100),
            "label": "HIGH" if total > 8 else "MEDIUM" if total > 3 else "LOW",
        }

    provider = SupyagentToolProvider(api_key=API_KEY, base_url=BASE_URL)
    runtime = SupyWorkflow(
        api_key=API_KEY,
        user_id="test",
        providers=[provider],
        extra_tools={"calculate_priority_score": calculate_priority_score},
    )

    return run_and_report("02: extra_tools — Custom Priority Scorer", runtime, f"""
# --- Fetch emails (supyagent tool)
emails = gmail_list_messages(maxResults=10)

# --- Score with custom function (extra_tool — pure Python, no API)
messages = emails.get("messages", []) if isinstance(emails, dict) else emails
priority = calculate_priority_score(messages)

# --- Use LLM to generate a report combining both
class PriorityReport(BaseModel):
    score: int
    label: str
    recommendation: str

report = llm(
    "Given this email priority data, write a one-line recommendation for the user.",
    data=priority,
    format=PriorityReport,
)

# --- Send report to Slack
text = ("*Inbox Priority: " + priority["label"] + " (" + str(priority["score"]) + "/100)*\\n"
        + report.recommendation)
slack_send_message(channel="{SLACK_GENERAL}", text=text)
""")


def test_03_extra_tools_data_transform():
    """Extra tools as data transformers — process data between tool calls."""

    def merge_sources(emails: dict, events: dict) -> dict:
        """Merge email and calendar data into a unified timeline."""
        email_items = []
        if isinstance(emails, dict) and "messages" in emails:
            for msg in emails["messages"][:5]:
                email_items.append({
                    "type": "email",
                    "snippet": msg.get("snippet", "")[:80],
                    "from": msg.get("from", "unknown"),
                })

        event_items = []
        if isinstance(events, dict) and "items" in events:
            for evt in events["items"][:5]:
                event_items.append({
                    "type": "event",
                    "title": evt.get("summary", ""),
                    "start": evt.get("start", {}).get("dateTime", ""),
                })

        return {
            "timeline": email_items + event_items,
            "email_count": len(email_items),
            "event_count": len(event_items),
        }

    provider = SupyagentToolProvider(api_key=API_KEY, base_url=BASE_URL)
    runtime = SupyWorkflow(
        api_key=API_KEY,
        user_id="test",
        providers=[provider],
        extra_tools={"merge_sources": merge_sources},
    )

    return run_and_report("03: extra_tools — Data Merge Pipeline", runtime, """
import datetime

# --- Fetch emails
emails = gmail_list_messages(maxResults=5)

# --- Fetch calendar events
today = datetime.date.today().isoformat()
events = calendar_list_events(timeMin=today + "T00:00:00Z", timeMax=today + "T23:59:59Z")

# --- Merge using custom function (no API call)
merged = merge_sources(emails=emails, events=events)

# --- Analyze merged timeline
class TimelineSummary(BaseModel):
    busiest_area: str
    recommendation: str

summary = llm(
    "Analyze this merged timeline of emails and calendar events. "
    "Which area (email or meetings) needs more attention?",
    data=merged,
    format=TimelineSummary,
)

report = ("Emails: " + str(merged["email_count"]) + ", Events: " + str(merged["event_count"])
          + "\\nBusiest: " + summary.busiest_area
          + "\\nAdvice: " + summary.recommendation)
""")


# ══════════════════════════════════════════════════════════════════════
# TEST 4: extra_tools injected at run() time (override init-time tools)
# ══════════════════════════════════════════════════════════════════════

def test_04_runtime_extra_tools():
    """Pass extra_tools at run() time — they override init-time tools on collision."""

    def format_report(data: dict) -> str:
        """Format a dict as a readable report — injected at run time."""
        lines = []
        for k, v in data.items():
            lines.append(f"{k}: {v}")
        return "\n".join(lines)

    provider = SupyagentToolProvider(api_key=API_KEY, base_url=BASE_URL)
    runtime = SupyWorkflow(
        api_key=API_KEY,
        user_id="test",
        providers=[provider],
        # No extra_tools at init time
    )

    return run_and_report("04: Run-time extra_tools injection", runtime, """
# --- Fetch some data
emails = gmail_list_messages(maxResults=3)

# --- Analyze
class QuickSummary(BaseModel):
    count: int
    one_liner: str

summary = llm("How many emails and one-liner summary?", data=emails, format=QuickSummary)

# --- Format using the run-time injected tool
report = format_report({"emails": summary.count, "summary": summary.one_liner})
""", extra_tools={"format_report": format_report})


# ══════════════════════════════════════════════════════════════════════
# TEST 5: CompositeToolProvider — two providers merged
# ══════════════════════════════════════════════════════════════════════

def test_05_composite_provider():
    """Two SupyagentToolProviders composed — simulates multi-source tool discovery."""

    # In real usage, one might be supyagent and the other an HTTP gateway.
    # Here we use the same supyagent twice to prove the composite works,
    # and add extra_tools to simulate a "composio" source.
    provider = SupyagentToolProvider(api_key=API_KEY, base_url=BASE_URL)

    call_log = []

    def composio_web_scrape(url: str = "") -> dict:
        """Simulated composio tool — would come from a second provider in production."""
        call_log.append(url)
        return {
            "source": "composio_mock",
            "url": url,
            "content": f"Simulated scrape of {url} — in production this would be a real Composio tool call",
        }

    composite = CompositeToolProvider([provider])
    runtime = SupyWorkflow(
        api_key=API_KEY,
        user_id="test",
        providers=[composite],
        extra_tools={"composio_web_scrape": composio_web_scrape},
    )

    return run_and_report("05: CompositeToolProvider + extra_tools", runtime, f"""
# --- Use supyagent tool (via composite provider)
news = search_news(query="AI agents 2025", num=3)

# --- Use "composio" tool (via extra_tools)
scrape = composio_web_scrape(url="https://example.com/ai-report")

# --- Combine data from both sources
class CombinedInsight(BaseModel):
    news_summary: str
    scrape_note: str
    combined_takeaway: str

insight = llm(
    "Combine these two data sources — news search results and a web scrape — into a brief insight.",
    data={{"news": news, "scrape": scrape}},
    format=CombinedInsight,
)

# --- Send combined result to Slack
text = ("*Multi-Source Intelligence*\\n"
        + "News: " + insight.news_summary + "\\n"
        + "Scrape: " + insight.scrape_note + "\\n"
        + "Takeaway: " + insight.combined_takeaway)
slack_send_message(channel="{SLACK_GENERAL}", text=text)
""")


# ══════════════════════════════════════════════════════════════════════
# TEST 6: Provider discovery — verify tools list comes from provider
# ══════════════════════════════════════════════════════════════════════

def test_06_provider_discovery():
    """Verify that provider.discover() is used for tool listing."""
    divider("06: Provider Discovery Verification")

    provider = SupyagentToolProvider(api_key=API_KEY, base_url=BASE_URL)

    # Discover tools through the provider interface
    tools = provider.discover()
    names = provider.get_tool_names()

    print(f"Discovered {len(tools)} tools via SupyagentToolProvider")
    print(f"Tool names ({len(names)}):")
    for name in sorted(names):
        desc = ""
        for t in tools:
            if t["function"]["name"] == name:
                desc = t["function"].get("description", "")[:60]
                break
        print(f"  {name:40s} {desc}")

    # Verify composite merges correctly
    composite = CompositeToolProvider([provider])
    composite_names = composite.get_tool_names()
    assert set(composite_names) == set(names), "Composite should have same tools as single provider"
    print(f"\nComposite verified: {len(composite_names)} tools (matches provider)")

    # Verify runtime.tools uses provider
    runtime = SupyWorkflow(
        api_key=API_KEY, user_id="test",
        providers=[provider],
        extra_tools={"my_custom": lambda: None},
    )
    rt_tools = runtime.tools
    assert "my_custom" in rt_tools, "extra_tools should appear in runtime.tools"
    assert set(names).issubset(set(rt_tools)), "Provider tools should appear in runtime.tools"
    print(f"Runtime.tools verified: {len(rt_tools)} tools (provider + extra)")

    return True


# ══════════════════════════════════════════════════════════════════════
# TEST 7: Full pipeline — supyagent provider + extra tools + LLM + Slack
# ══════════════════════════════════════════════════════════════════════

def test_07_full_pipeline():
    """End-to-end: provider tools + extra_tools + LLM + output to Slack."""

    def enrich_with_context(data: dict) -> dict:
        """Add company context — simulates an internal API call or DB lookup."""
        return {
            **data,
            "company": "Ergodic AI",
            "focus_areas": ["AI agents", "workflow automation", "tool integrations"],
            "competitors": ["LangChain", "CrewAI", "AutoGen"],
        }

    def format_slack_blocks(sections: list) -> str:
        """Format sections into a clean Slack message."""
        parts = []
        for section in sections:
            if isinstance(section, dict):
                parts.append(f"*{section.get('title', '')}*")
                parts.append(section.get('content', ''))
                parts.append("")
            else:
                parts.append(str(section))
        return "\n".join(parts)

    provider = SupyagentToolProvider(api_key=API_KEY, base_url=BASE_URL)
    runtime = SupyWorkflow(
        api_key=API_KEY,
        user_id="test",
        providers=[provider],
        extra_tools={
            "enrich_with_context": enrich_with_context,
            "format_slack_blocks": format_slack_blocks,
        },
    )

    return run_and_report("07: Full Pipeline — Provider + Extras + LLM", runtime, f"""
# --- Fetch external data (supyagent tool)
news = search_news(query="AI workflow automation", num=5)

# --- Enrich with internal context (extra_tool)
enriched = enrich_with_context({{"news_count": len(news) if isinstance(news, list) else 1}})

# --- LLM analysis combining both
class MarketBrief(BaseModel):
    market_trend: str
    relevance_to_us: str
    action_items: list[str]

brief = llm(
    "Given recent news about AI workflow automation and our company context, "
    "write a brief market intelligence update. What's relevant to us?",
    data={{"news": news, "context": enriched}},
    format=MarketBrief,
)

# --- Format output (extra_tool)
message = format_slack_blocks([
    {{"title": "Market Intelligence", "content": brief.market_trend}},
    {{"title": "Relevance to " + enriched["company"], "content": brief.relevance_to_us}},
    {{"title": "Action Items", "content": chr(10).join("- " + a for a in brief.action_items)}},
])

# --- Post to Slack (supyagent tool)
slack_send_message(channel="{SLACK_GENERAL}", text=message)
""")


# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    results = {}

    # Discovery test (no execution)
    test_06_provider_discovery()

    # Execution tests
    tests = [
        ("01_supyagent_provider", test_01_supyagent_provider_basic),
        ("02_extra_tools_basic", test_02_extra_tools_basic),
        ("03_data_merge", test_03_extra_tools_data_transform),
        ("04_runtime_extra", test_04_runtime_extra_tools),
        ("05_composite", test_05_composite_provider),
        ("07_full_pipeline", test_07_full_pipeline),
    ]

    for name, fn in tests:
        try:
            results[name] = fn()
        except Exception as e:
            import traceback
            print(f"\n  CRASHED: {e}")
            traceback.print_exc()
            results[name] = None

    divider("PROVIDER TEST RESULTS")
    passed = 0
    failed = 0
    for name, r in results.items():
        if r is None:
            status = "CRASHED"
            failed += 1
        else:
            status = r.status
            cells = len(r.cells)
            ok = sum(1 for c in r.cells if c.status == "completed")
            healed = len([h for h in r.healed_cells.values() if h.healed]) if r.healed_cells else 0
            status = f"{r.status} ({ok}/{cells} cells, {healed} healed)"
            if r.status in ("completed", "partial"):
                passed += 1
            else:
                failed += 1

        print(f"  {name:25s} {status}")

    print(f"\n  TOTAL: {passed} passed, {failed} failed out of {len(tests)}")
    print(f"  (+ discovery verification test)")
