"""Tests for workflow generation."""

from unittest.mock import MagicMock, patch

import pytest

from supyworkflow.generator import GenerateResult, generate_workflow, refine_workflow


class TestGenerateWorkflow:
    def _mock_litellm_response(self, content):
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = content
        response.usage = MagicMock(prompt_tokens=200, completion_tokens=150)
        return response

    @patch("supyworkflow.generator.litellm")
    def test_basic_generation(self, mock_litellm):
        script = """# --- Fetch emails
emails = gmail.list_messages(query="is:unread", max_results=5)

# --- Summarize
class Summary(BaseModel):
    count: int
    highlights: list[str]

summary = llm("Summarize these emails", data=emails, format=Summary)

# --- Notify
slack.send_message(channel="general", text=f"You have {summary.count} unread emails")
"""
        mock_litellm.completion.return_value = self._mock_litellm_response(script)

        result = generate_workflow(
            prompt="Check my unread emails and send a summary to Slack",
            tools=["gmail", "slack"],
        )

        assert isinstance(result, GenerateResult)
        assert "gmail.list_messages" in result.source
        assert "slack.send_message" in result.source
        assert "BaseModel" in result.source
        assert result.prompt_tokens == 200

    @patch("supyworkflow.generator.litellm")
    def test_strips_markdown_fences(self, mock_litellm):
        mock_litellm.completion.return_value = self._mock_litellm_response(
            '```python\nx = 1\n```'
        )

        result = generate_workflow(prompt="simple", tools=[])
        assert result.source == "x = 1"

    @patch("supyworkflow.generator.litellm")
    def test_passes_tool_actions(self, mock_litellm):
        mock_litellm.completion.return_value = self._mock_litellm_response("x = 1")

        generate_workflow(
            prompt="do stuff",
            tools=["gmail"],
            tool_actions={"gmail": ["list_messages", "send_email", "get_message"]},
        )

        # Verify the tool actions were included in the prompt
        call_args = mock_litellm.completion.call_args
        user_msg = call_args.kwargs["messages"][1]["content"]
        assert "list_messages" in user_msg
        assert "send_email" in user_msg

    @patch("supyworkflow.generator.litellm")
    def test_passes_inputs(self, mock_litellm):
        mock_litellm.completion.return_value = self._mock_litellm_response("x = 1")

        generate_workflow(
            prompt="send report",
            tools=["gmail"],
            inputs={"recipient": "Email address to send to", "date_range": "How many days back"},
        )

        call_args = mock_litellm.completion.call_args
        user_msg = call_args.kwargs["messages"][1]["content"]
        assert "recipient" in user_msg
        assert "date_range" in user_msg

    @patch("supyworkflow.generator.litellm")
    def test_system_prompt_content(self, mock_litellm):
        mock_litellm.completion.return_value = self._mock_litellm_response("x = 1")

        generate_workflow(prompt="test", tools=[])

        call_args = mock_litellm.completion.call_args
        system_msg = call_args.kwargs["messages"][0]["content"]
        # Key concepts should be in the system prompt
        assert "# ---" in system_msg
        assert "BaseModel" in system_msg
        assert "llm(" in system_msg
        assert "self-healing" in system_msg.lower() or "try/except" in system_msg.lower()


class TestRefineWorkflow:
    def _mock_litellm_response(self, content):
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = content
        response.usage = MagicMock(prompt_tokens=300, completion_tokens=200)
        return response

    @patch("supyworkflow.generator.litellm")
    def test_refine_with_warnings(self, mock_litellm):
        mock_litellm.completion.return_value = self._mock_litellm_response(
            "# --- Fixed\nx = gmail.list_messages()"
        )

        result = refine_workflow(
            source="# --- Broken\nx = unknown_tool.do_stuff()",
            dry_run_result={
                "cells": [{"index": 0, "label": "Broken", "reads": ["unknown_tool"], "writes": ["x"], "depends_on": []}],
                "tools_used": [],
                "missing_tools": [],
                "warnings": ["Cell 0 (Broken): unresolved references: {'unknown_tool'}"],
            },
        )

        assert isinstance(result, GenerateResult)
        assert result.source

    @patch("supyworkflow.generator.litellm")
    def test_refine_with_error(self, mock_litellm):
        mock_litellm.completion.return_value = self._mock_litellm_response("x = 1")

        result = refine_workflow(
            source="x = 1 / 0",
            dry_run_result={"cells": [], "tools_used": [], "missing_tools": [], "warnings": []},
            error_info="ZeroDivisionError: division by zero",
        )

        assert result.source

    @patch("supyworkflow.generator.litellm")
    def test_refine_with_feedback(self, mock_litellm):
        mock_litellm.completion.return_value = self._mock_litellm_response("x = 1")

        refine_workflow(
            source="x = 1",
            dry_run_result={"cells": [], "tools_used": [], "missing_tools": [], "warnings": []},
            feedback="Also CC my manager on the email",
        )

        call_args = mock_litellm.completion.call_args
        user_msg = call_args.kwargs["messages"][1]["content"]
        assert "CC my manager" in user_msg
