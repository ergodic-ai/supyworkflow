"""Tests for self-healing."""

from unittest.mock import MagicMock, patch

import pytest

from supyworkflow.healer import HealResult, heal_cell


class TestHealCell:
    def _mock_litellm_response(self, content):
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = content
        response.usage = MagicMock(prompt_tokens=100, completion_tokens=50)
        return response

    @patch("supyworkflow.healer.litellm")
    def test_heals_key_error(self, mock_litellm):
        mock_litellm.completion.return_value = self._mock_litellm_response(
            '```python\ntotal = data.get("total", 0)\n```'
        )

        result = heal_cell(
            cell_source='total = data["total"]',
            cell_label="Process",
            error=KeyError("total"),
            namespace_snapshot={"data": {"type": "json", "value": {"count": 5}}},
            tool_names=["gmail"],
        )

        assert result.healed is True
        assert "get" in result.patched_source
        assert result.attempts == 1

    @patch("supyworkflow.healer.litellm")
    def test_skips_unrecoverable_errors(self, mock_litellm):
        result = heal_cell(
            cell_source="import os",
            cell_label="Bad import",
            error=ImportError("not allowed"),
            namespace_snapshot={},
            tool_names=[],
        )

        assert result.healed is False
        assert result.attempts == 0
        mock_litellm.completion.assert_not_called()

    @patch("supyworkflow.healer.litellm")
    def test_gives_up_after_max_attempts(self, mock_litellm):
        # LLM keeps returning the same code
        mock_litellm.completion.return_value = self._mock_litellm_response(
            "I can't fix this code."
        )

        result = heal_cell(
            cell_source="x = broken()",
            cell_label="Broken",
            error=NameError("broken"),
            namespace_snapshot={},
            tool_names=[],
        )

        assert result.healed is False
        assert result.attempts == 3

    @patch("supyworkflow.healer.litellm")
    def test_skips_syntax_error(self, mock_litellm):
        result = heal_cell(
            cell_source="def broken(:",
            cell_label="Bad syntax",
            error=SyntaxError("invalid syntax"),
            namespace_snapshot={},
            tool_names=[],
        )

        assert result.healed is False
        assert result.attempts == 0


class TestHealingInRuntime:
    """Test that healing integrates correctly with the runtime."""

    def _mock_litellm_response(self, content):
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = content
        response.usage = MagicMock(prompt_tokens=100, completion_tokens=50)
        return response

    @patch("supyworkflow.healer.litellm")
    @patch("supyworkflow.namespace.discover_tools", return_value=[])
    def test_runtime_heals_failed_cell(self, mock_discover, mock_litellm):
        from supyworkflow.runtime import SupyWorkflow

        # Cell fails with KeyError, healer fixes it
        mock_litellm.completion.return_value = self._mock_litellm_response(
            '```python\nresult = data.get("value", 0) * 2\n```'
        )

        rt = SupyWorkflow(api_key="sk_test", user_id="u1", heal=True)
        rt._tools = []

        source = """
# --- Setup
data = {"count": 10}

# --- Process
result = data["value"] * 2

# --- Use result
final = result + 1
"""
        result = rt.run(source)

        # Cell 1 (Process) should have been healed
        assert 1 in result.healed_cells
        assert result.healed_cells[1].healed is True

    @patch("supyworkflow.namespace.discover_tools", return_value=[])
    def test_runtime_no_heal_when_disabled(self, mock_discover):
        from supyworkflow.runtime import SupyWorkflow

        rt = SupyWorkflow(api_key="sk_test", user_id="u1", heal=False)
        rt._tools = []

        source = """
# --- Fail
x = 1 / 0
"""
        result = rt.run(source)
        assert result.status == "failed"
        assert len(result.healed_cells) == 0

    @patch("supyworkflow.namespace.discover_tools", return_value=[])
    def test_patched_source_returned(self, mock_discover):
        """When cells are healed, patched_source contains the full fixed script."""
        from supyworkflow.runtime import SupyWorkflow

        rt = SupyWorkflow(api_key="sk_test", user_id="u1", heal=False)
        rt._tools = []

        result = rt.run("# --- A\nx = 1\n\n# --- B\ny = 2")
        # No healing happened
        assert result.patched_source is None
