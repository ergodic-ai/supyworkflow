"""End-to-end tests using real LLM (Gemini) and real Supyagent API.

Run manually: uv run python tests/e2e_test.py
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
            val = val.strip().strip('"').strip("'")
            os.environ[key.strip()] = val

os.environ["GEMINI_API_KEY"] = os.environ.get("GCP_API_KEY", "")

API_KEY = os.environ.get("SUPYAGENT_API_KEY", "")

sys.path.insert(0, str(Path(__file__).parent.parent))


def divider(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


# ── Test 1: Pure Python workflow (no tools, no API) ─────────────

def test_pure_python_workflow():
    divider("Test 1: Pure Python workflow (no external calls)")

    from supyworkflow.runtime import SupyWorkflow

    rt = SupyWorkflow(api_key="fake", user_id="fake", heal=False)
    rt._tools = []

    source = """
# --- Generate data
import json
import math

data = [{"product": "Widget A", "sales": 150, "price": 29.99},
        {"product": "Widget B", "sales": 320, "price": 14.99},
        {"product": "Gadget X", "sales": 85, "price": 99.99},
        {"product": "Gadget Y", "sales": 200, "price": 49.99}]

# --- Calculate metrics
total_revenue = sum(d["sales"] * d["price"] for d in data)
avg_price = sum(d["price"] for d in data) / len(data)
top_product = max(data, key=lambda d: d["sales"] * d["price"])

# --- Build report
report = {
    "total_revenue": round(total_revenue, 2),
    "average_price": round(avg_price, 2),
    "top_product": top_product["product"],
    "top_revenue": round(top_product["sales"] * top_product["price"], 2),
    "product_count": len(data),
}
"""

    result = rt.run(source)

    print(f"Status: {result.status}")
    print(f"Cells: {len(result.cells)}")
    for cell in result.cells:
        print(f"  [{cell.status}] {cell.label} ({cell.duration_ms:.1f}ms)")

    outputs = result.outputs
    print(f"\nOutputs:")
    print(f"  total_revenue: {outputs['report']['total_revenue']}")
    print(f"  top_product: {outputs['report']['top_product']}")
    print(f"  product_count: {outputs['report']['product_count']}")

    assert result.status == "completed"
    assert outputs["report"]["product_count"] == 4
    assert outputs["report"]["total_revenue"] > 0
    print("\n  PASSED")


# ── Test 2: LLM calls with structured output ────────────────────

def test_llm_structured_output():
    divider("Test 2: LLM calls with Pydantic structured output")

    from supyworkflow.runtime import SupyWorkflow

    rt = SupyWorkflow(api_key="fake", user_id="fake", heal=False)
    rt._tools = []

    source = """
# --- Define schema and get recommendation
class MovieRecommendation(BaseModel):
    title: str
    year: int
    genre: str
    reason: str

rec = llm(
    "Recommend a classic sci-fi movie from the 1980s",
    format=MovieRecommendation,
)

# --- Extract fields
title = rec.title
year = rec.year
genre = rec.genre
summary = f"{rec.title} ({rec.year}) - {rec.genre}: {rec.reason}"
"""

    result = rt.run(source)

    print(f"Status: {result.status}")
    for cell in result.cells:
        print(f"  [{cell.status}] {cell.label} ({cell.duration_ms:.1f}ms)")
        if cell.error:
            print(f"       ERROR: {cell.error}")

    if result.status == "completed":
        outputs = result.outputs
        print(f"\nMovie: {outputs['title']} ({outputs['year']})")
        print(f"Genre: {outputs['genre']}")
        print(f"Summary: {outputs['summary']}")
        assert isinstance(outputs["year"], int)
        print("\n  PASSED")
    else:
        print(f"\n  FAILED: {result.error}")


# ── Test 3: Multi-step LLM workflow with data flow ──────────────

def test_multi_step_llm_workflow():
    divider("Test 3: Multi-step LLM workflow with data flow between cells")

    from supyworkflow.runtime import SupyWorkflow

    rt = SupyWorkflow(api_key="fake", user_id="fake", heal=False)
    rt._tools = []

    source = """
# --- Raw data
raw_feedback = [
    "The product is great but shipping was slow",
    "Terrible quality, broke after one day",
    "Love it! Best purchase I've made this year",
    "Decent for the price, nothing special",
    "Customer service was unhelpful when I had issues",
]

# --- Analyze sentiment
class FeedbackAnalysis(BaseModel):
    positive: list[str]
    negative: list[str]
    neutral: list[str]
    overall_sentiment: str

analysis = llm(
    "Categorize each piece of feedback as positive, negative, or neutral. "
    "Put the full text in the appropriate list.",
    data=raw_feedback,
    format=FeedbackAnalysis,
)

# --- Generate response strategy
class ResponseStrategy(BaseModel):
    priority_issues: list[str]
    suggested_actions: list[str]
    response_tone: str

strategy = llm(
    "Based on this customer feedback analysis, suggest a response strategy. "
    "Identify priority issues and suggest concrete actions.",
    data={
        "positive": analysis.positive,
        "negative": analysis.negative,
        "neutral": analysis.neutral,
        "overall": analysis.overall_sentiment,
    },
    format=ResponseStrategy,
)

# --- Build summary
summary = {
    "feedback_count": len(raw_feedback),
    "positive_count": len(analysis.positive),
    "negative_count": len(analysis.negative),
    "neutral_count": len(analysis.neutral),
    "overall_sentiment": analysis.overall_sentiment,
    "priority_issues": strategy.priority_issues,
    "suggested_actions": strategy.suggested_actions,
    "response_tone": strategy.response_tone,
}
"""

    result = rt.run(source)

    print(f"Status: {result.status}")
    for cell in result.cells:
        print(f"  [{cell.status}] {cell.label} ({cell.duration_ms:.1f}ms)")
        if cell.error:
            print(f"       ERROR: {cell.error}")

    if result.status == "completed":
        summary = result.outputs["summary"]
        print(f"\nFeedback count: {summary['feedback_count']}")
        print(f"Positive: {summary['positive_count']}, Negative: {summary['negative_count']}, Neutral: {summary['neutral_count']}")
        print(f"Overall: {summary['overall_sentiment']}")
        print(f"Priority issues: {summary['priority_issues']}")
        print(f"Suggested actions: {summary['suggested_actions']}")
        print(f"Tone: {summary['response_tone']}")

        assert summary["feedback_count"] == 5
        assert summary["positive_count"] + summary["negative_count"] + summary["neutral_count"] == 5
        print("\n  PASSED")
    else:
        print(f"\n  FAILED: {result.error}")

    if result.trace:
        s = result.trace.summary()
        print(f"\nTrace: {s['cells']} cells, {s['total_duration_ms']:.0f}ms total")


# ── Test 4: Self-healing ────────────────────────────────────────

def test_self_healing():
    divider("Test 4: Self-healing (cell has a bug, healer fixes it)")

    from supyworkflow.runtime import SupyWorkflow

    rt = SupyWorkflow(api_key="fake", user_id="fake", heal=True)
    rt._tools = []

    # Intentional bug: 'totals' key doesn't exist, should be 'total'
    source = """
# --- Setup data
data = {"items": [10, 20, 30], "total": 60}

# --- Process (has a bug!)
result = data["totals"] * 2
"""

    result = rt.run(source)

    print(f"Status: {result.status}")
    for cell in result.cells:
        print(f"  [{cell.status}] {cell.label} ({cell.duration_ms:.1f}ms)")
        if cell.error:
            print(f"       ERROR: {cell.error}")

    if result.healed_cells:
        for idx, heal in result.healed_cells.items():
            print(f"\n  Cell {idx} healing: healed={heal.healed}, attempts={heal.attempts}")
            if heal.healed:
                print(f"  Patched code: {heal.patched_source}")

    if result.status == "completed":
        print(f"\n  result = {result.outputs.get('result')}")
        assert result.outputs.get("result") == 120
        print("\n  PASSED")
    elif result.healed_cells and any(h.healed for h in result.healed_cells.values()):
        print("\n  PARTIALLY PASSED (healing worked but execution may have other issues)")
    else:
        print(f"\n  FAILED: {result.error}")


# ── Test 5: Workflow generation ─────────────────────────────────

def test_workflow_generation():
    divider("Test 5: Workflow generation from natural language")

    from supyworkflow.generator import generate_workflow

    result = generate_workflow(
        prompt="Analyze a list of customer support tickets. "
               "Categorize them by urgency (high/medium/low), "
               "identify common themes, and produce a summary report.",
        tools=["gmail", "slack"],
        tool_actions={
            "gmail": ["list_messages", "send_email", "get_message"],
            "slack": ["send_message", "list_channels"],
        },
        inputs={
            "tickets": "List of support ticket dicts with 'subject', 'body', 'customer' fields",
        },
    )

    print(f"Generated {len(result.source)} chars, {result.prompt_tokens} prompt / {result.completion_tokens} completion tokens")
    print(f"\n--- Generated Workflow ---")
    print(result.source)
    print(f"--- End ---\n")

    assert "# ---" in result.source
    assert "BaseModel" in result.source or "llm(" in result.source

    from supyworkflow.parser import parse_cells
    cells = parse_cells(result.source)
    print(f"Cells: {len(cells)}")
    for cell in cells:
        print(f"  [{cell.index}] {cell.label}: writes={sorted(cell.writes)}")

    assert len(cells) >= 2
    print("\n  PASSED")


# ── Test 6: Generate + Dry Run + Execute ────────────────────────

def test_generate_and_execute():
    divider("Test 6: Generate -> Dry Run -> Execute (full pipeline)")

    from supyworkflow.generator import generate_workflow
    from supyworkflow.runtime import SupyWorkflow

    gen = generate_workflow(
        prompt="Given a list of numbers, compute basic statistics (mean, median, min, max, std dev) "
               "and classify the distribution as 'normal', 'skewed', or 'uniform' using LLM analysis.",
        tools=[],
        inputs={"numbers": "List of numeric values"},
    )

    print("Generated workflow:")
    print(gen.source)
    print()

    rt = SupyWorkflow(api_key="fake", user_id="fake", heal=True)
    rt._tools = []

    analysis = rt.dry_run(gen.source)
    print(f"Dry run: {len(analysis['cells'])} cells, warnings={analysis['warnings']}")

    test_numbers = [12, 15, 14, 10, 18, 13, 16, 11, 14, 15, 13, 17, 12, 14, 16]

    result = rt.run(gen.source, inputs={"numbers": test_numbers})

    print(f"\nExecution status: {result.status}")
    for cell in result.cells:
        print(f"  [{cell.status}] {cell.label} ({cell.duration_ms:.1f}ms)")
        if cell.error:
            print(f"       ERROR: {cell.error}")

    if result.healed_cells:
        for idx, heal in result.healed_cells.items():
            if heal.healed:
                print(f"  Cell {idx} was healed ({heal.attempts} attempts)")

    if result.status == "completed":
        print(f"\nOutputs: {list(result.outputs.keys())}")
        for k, v in result.outputs.items():
            preview = str(v)
            if len(preview) > 100:
                preview = preview[:100] + "..."
            print(f"  {k} = {preview}")
        print("\n  PASSED")
    else:
        print(f"\n  FAILED: {result.error}")


# ── Run all tests ───────────────────────────────────────────────

if __name__ == "__main__":
    from supyworkflow.llm_builtin import DEFAULT_MODEL
    print(f"Default model: {DEFAULT_MODEL}")
    print(f"API Key: {API_KEY[:15]}..." if API_KEY else "NO API KEY")

    test_pure_python_workflow()
    test_llm_structured_output()
    test_multi_step_llm_workflow()
    test_self_healing()
    test_workflow_generation()
    test_generate_and_execute()

    divider("ALL TESTS COMPLETE")
