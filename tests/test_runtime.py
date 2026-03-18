"""Tests for the workflow runtime."""

from unittest.mock import patch

import pytest

from supyworkflow.runtime import SupyWorkflow


@pytest.fixture
def runtime():
    """Create a runtime with no tools (unit tests don't hit the API)."""
    rt = SupyWorkflow(api_key="sk_test", user_id="user_1", heal=False)
    rt._tools = []
    yield rt


class TestRuntime:
    def test_simple_script(self, runtime):
        source = """
# --- Compute
x = 1 + 2
y = x * 3
"""
        result = runtime.run(source)
        assert result.status == "completed"
        assert result.outputs["x"] == 3
        assert result.outputs["y"] == 9

    def test_multi_cell(self, runtime):
        source = """
# --- Step 1
data = [1, 2, 3, 4, 5]

# --- Step 2
total = sum(data)

# --- Step 3
average = total / len(data)
"""
        result = runtime.run(source)
        assert result.status == "completed"
        assert result.outputs["total"] == 15
        assert result.outputs["average"] == 3.0
        assert len(result.cells) == 3
        assert all(c.status == "completed" for c in result.cells)

    def test_cell_failure_skips_dependents(self, runtime):
        source = """
# --- Step 1
x = 1

# --- Step 2 (will fail)
y = 1 / 0

# --- Step 3 (depends on y)
z = y + 1
"""
        result = runtime.run(source)
        assert result.status in ("failed", "partial")
        assert result.cells[0].status == "completed"
        assert result.cells[1].status == "failed"
        assert result.cells[2].status == "skipped"

    def test_inputs_injected(self, runtime):
        source = """
# --- Use inputs
result = name.upper()
"""
        result = runtime.run(source, inputs={"name": "hello"})
        assert result.status == "completed"
        assert result.outputs["result"] == "HELLO"

    def test_empty_script(self, runtime):
        result = runtime.run("")
        assert result.status == "completed"
        assert result.cells == []

    def test_from_cell_resume(self, runtime):
        source = """
# --- Step 1
x = 10

# --- Step 2
y = x * 2
"""
        # First run to get snapshots
        result1 = runtime.run(source)
        snapshots = {c.index: c.snapshot for c in result1.cells if c.snapshot}

        # Resume from cell 1
        result2 = runtime.run(source, from_cell=1, snapshots=snapshots)
        assert result2.outputs["y"] == 20
        assert result2.cells[0].status == "skipped"
        assert result2.cells[1].status == "completed"

    def test_trace_generated(self, runtime):
        source = """
# --- A
x = 1

# --- B
y = 2
"""
        result = runtime.run(source)
        assert result.trace is not None
        summary = result.trace.summary()
        assert summary["cells"] == 2
        assert summary["errors"] == 0

    def test_restricted_imports(self, runtime):
        source = """
# --- Bad import
import os
"""
        result = runtime.run(source)
        assert result.status == "failed"
        assert result.cells[0].status == "failed"

    def test_allowed_imports(self, runtime):
        source = """
# --- Safe import
import json
import re
data = json.dumps({"x": 1})
match = re.search(r"\\d+", data)
"""
        result = runtime.run(source)
        assert result.status == "completed"


class TestDryRun:
    def test_basic_analysis(self):
        rt = SupyWorkflow(api_key="sk_test", user_id="user_1", heal=False)
        rt._tools = ["gmail_list_messages", "gmail_send_message", "slack_send_message"]

        source = """
# --- Fetch
emails = gmail_list_messages(q="from:boss")

# --- Analyze
summary = llm("Summarize", data=emails)

# --- Send
slack_send_message(channel="general", text=summary)
"""
        analysis = rt.dry_run(source)
        assert len(analysis["cells"]) == 3
        assert "gmail_list_messages" in analysis["tools_used"]
        assert analysis["cells"][1]["depends_on"] == [0]  # Analyze depends on Fetch
        assert analysis["cells"][2]["depends_on"] == [1]  # Send depends on Analyze
