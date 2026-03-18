"""Run example workflows against the real supyagent API.

Usage: uv run python examples/run_examples.py
"""

import os
import sys
from pathlib import Path

# Load env
env_file = Path(__file__).parent.parent / ".env.testing"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ[key.strip()] = val.strip().strip('"').strip("'")

os.environ["GEMINI_API_KEY"] = os.environ.get("GCP_API_KEY", "")

sys.path.insert(0, str(Path(__file__).parent.parent))

from supyworkflow.runtime import SupyWorkflow


def divider(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def run_workflow(name: str, source: str, inputs: dict | None = None):
    divider(name)

    rt = SupyWorkflow(
        api_key=os.environ["SUPYAGENT_API_KEY"],
        user_id="test",  # not needed, admin key
    )

    # Dry run first
    analysis = rt.dry_run(source)
    print(f"Dry run: {len(analysis['cells'])} cells, tools={analysis['tools_used']}")
    if analysis["warnings"]:
        for w in analysis["warnings"]:
            print(f"  warn: {w}")

    # Execute
    result = rt.run(source, inputs=inputs)

    print(f"\nStatus: {result.status}")
    for cell in result.cells:
        print(f"  [{cell.status}] {cell.label} ({cell.duration_ms:.1f}ms)")
        if cell.error:
            print(f"       ERROR: {cell.error}")

    if result.healed_cells:
        for idx, heal in result.healed_cells.items():
            if heal.healed:
                print(f"  Cell {idx} healed in {heal.attempts} attempts")
                print(f"  Patch: {heal.patched_source[:200]}")

    if result.status in ("completed", "partial"):
        print(f"\nOutputs:")
        for k, v in result.outputs.items():
            preview = str(v)
            if len(preview) > 200:
                preview = preview[:200] + "..."
            print(f"  {k} = {preview}")

    if result.trace:
        s = result.trace.summary()
        print(f"\nTrace: {s['tool_calls']} tool calls, {s['llm_calls']} llm calls, {s['total_duration_ms']:.0f}ms")

    return result


# ── Example 1: Email Digest ─────────────────────────────────────

def example_email_digest():
    source = Path(__file__).parent / "01_email_digest.py"
    return run_workflow("Email Digest", source.read_text())


# ── Example 2: Calendar Briefing ────────────────────────────────

def example_calendar_briefing():
    source = Path(__file__).parent / "02_calendar_briefing.py"
    return run_workflow("Calendar Briefing", source.read_text())


# ── Example 3: Research Report (web search only, no sheets) ────

def example_research():
    source = """
# --- Search the web
results = search_web(q="latest developments in AI agents 2025", num=5)

# --- Analyze findings
class ResearchFinding(BaseModel):
    source: str
    key_insight: str

class ResearchReport(BaseModel):
    findings: list[ResearchFinding]
    executive_summary: str
    key_themes: list[str]

report = llm(
    "Analyze these search results about AI agents. "
    "Extract key findings and identify major themes.",
    data=results,
    format=ResearchReport,
)

# --- Format output
output_lines = [f"Research: AI Agents 2025\\n"]
output_lines.append(report.executive_summary + "\\n")
output_lines.append("Key Themes:")
for theme in report.key_themes:
    output_lines.append(f"  - {theme}")
output_lines.append("\\nFindings:")
for f in report.findings:
    output_lines.append(f"  [{f.source}] {f.key_insight}")

report_text = "\\n".join(output_lines)
"""
    return run_workflow("Web Research", source)


# ── Example 4: Competitor News ──────────────────────────────────

def example_competitor_news():
    source = """
# --- Search for news
news = search_news(q="OpenAI", num=5)

# --- Analyze
class NewsAnalysis(BaseModel):
    headlines: list[str]
    overall_sentiment: str
    key_developments: list[str]

analysis = llm(
    "Analyze this news coverage about OpenAI. "
    "List headlines, assess overall sentiment, identify key developments.",
    data=news,
    format=NewsAnalysis,
)

# --- Summary
summary = {
    "company": "OpenAI",
    "headlines": analysis.headlines,
    "sentiment": analysis.overall_sentiment,
    "developments": analysis.key_developments,
}
"""
    return run_workflow("Competitor News", source)


# ── Example 5: Generate workflow from prompt then run it ────────

def example_generated_workflow():
    divider("Generated Workflow (full pipeline)")

    from supyworkflow.generator import generate_workflow

    gen = generate_workflow(
        prompt="Search the web for recent news about renewable energy, "
               "analyze the top findings, and create a brief intelligence report "
               "with key trends and implications.",
        tools=[
            "search_web", "search_news",
        ],
    )

    print(f"Generated {len(gen.source)} chars")
    print(f"\n--- Source ---\n{gen.source}\n--- End ---\n")

    return run_workflow("Generated: Renewable Energy Report", gen.source)


if __name__ == "__main__":
    results = {}

    results["email_digest"] = example_email_digest()
    results["calendar"] = example_calendar_briefing()
    results["research"] = example_research()
    results["competitor"] = example_competitor_news()
    results["generated"] = example_generated_workflow()

    divider("RESULTS SUMMARY")
    for name, r in results.items():
        print(f"  {name:25s} {r.status}")
