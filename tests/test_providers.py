"""Tests for the pluggable tool provider system."""

import unittest
from unittest.mock import MagicMock, patch

from supyworkflow.providers.composite import CompositeToolProvider
from supyworkflow.providers.http_gateway import HttpGatewayToolProvider
from supyworkflow.providers.supyagent import SupyagentToolProvider
from supyworkflow.tool_provider import ToolProvider


def _make_tool_metadata(name: str, description: str = "") -> dict:
    return {
        "function": {
            "name": name,
            "description": description or f"Does {name}",
            "parameters": {"type": "object", "properties": {}},
        },
        "metadata": {"method": "POST", "path": f"/api/v1/{name}"},
    }


class MockProvider:
    """A simple mock provider for testing."""

    def __init__(self, tools: dict[str, dict]):
        self._tools = tools
        self._results: dict[str, object] = {}

    def set_result(self, tool_name: str, result: object) -> None:
        self._results[tool_name] = result

    def discover(self) -> list[dict]:
        return [_make_tool_metadata(name, desc) for name, desc in self._tools.items()]

    def execute(self, tool_name: str, **kwargs) -> object:
        if tool_name not in self._tools:
            raise KeyError(f"Tool '{tool_name}' not found")
        return self._results.get(tool_name, {"ok": True})

    def get_tool_names(self) -> list[str]:
        return list(self._tools.keys())


class TestToolProviderProtocol(unittest.TestCase):
    def test_mock_provider_is_tool_provider(self):
        p = MockProvider({})
        self.assertIsInstance(p, ToolProvider)

    def test_supyagent_provider_is_tool_provider(self):
        # SupyagentToolProvider should satisfy the protocol
        p = SupyagentToolProvider(api_key="test", base_url="http://localhost")
        self.assertIsInstance(p, ToolProvider)

    def test_gateway_provider_is_tool_provider(self):
        p = HttpGatewayToolProvider(
            gateway_url="http://localhost/gateway",
            api_key="test",
        )
        self.assertIsInstance(p, ToolProvider)

    def test_composite_provider_is_tool_provider(self):
        p = CompositeToolProvider(providers=[])
        self.assertIsInstance(p, ToolProvider)


class TestCompositeToolProvider(unittest.TestCase):
    def test_merges_tools_from_multiple_providers(self):
        p1 = MockProvider({"gmail_send": "Send email", "gmail_list": "List emails"})
        p2 = MockProvider({"slack_send": "Send slack", "slack_list": "List channels"})

        composite = CompositeToolProvider([p1, p2])
        names = composite.get_tool_names()

        self.assertEqual(sorted(names), ["gmail_list", "gmail_send", "slack_list", "slack_send"])

    def test_later_provider_overrides_on_collision(self):
        p1 = MockProvider({"gmail_send": "Old impl"})
        p1.set_result("gmail_send", {"from": "p1"})

        p2 = MockProvider({"gmail_send": "New impl"})
        p2.set_result("gmail_send", {"from": "p2"})

        composite = CompositeToolProvider([p1, p2])
        result = composite.execute("gmail_send")
        self.assertEqual(result, {"from": "p2"})

    def test_dispatches_to_correct_provider(self):
        p1 = MockProvider({"gmail_send": "Send email"})
        p1.set_result("gmail_send", {"provider": "supyagent"})

        p2 = MockProvider({"slack_send": "Send slack"})
        p2.set_result("slack_send", {"provider": "composio"})

        composite = CompositeToolProvider([p1, p2])

        self.assertEqual(composite.execute("gmail_send"), {"provider": "supyagent"})
        self.assertEqual(composite.execute("slack_send"), {"provider": "composio"})

    def test_execute_unknown_tool_raises(self):
        composite = CompositeToolProvider([MockProvider({"a": "a tool"})])
        with self.assertRaises(KeyError):
            composite.execute("nonexistent")

    def test_empty_providers(self):
        composite = CompositeToolProvider([])
        self.assertEqual(composite.discover(), [])
        self.assertEqual(composite.get_tool_names(), [])

    def test_discover_returns_full_metadata(self):
        p = MockProvider({"gmail_send": "Send email"})
        composite = CompositeToolProvider([p])
        tools = composite.discover()
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0]["function"]["name"], "gmail_send")
        self.assertIn("parameters", tools[0]["function"])


class TestSupyagentToolProvider(unittest.TestCase):
    @patch("supyworkflow.providers.supyagent._fetch_tools_metadata")
    def test_discover_calls_fetch(self, mock_fetch):
        mock_fetch.return_value = [_make_tool_metadata("gmail_send")]

        p = SupyagentToolProvider(api_key="sk_test", base_url="http://localhost", user_id="user1")
        tools = p.discover()

        self.assertEqual(len(tools), 1)
        mock_fetch.assert_called_once_with("sk_test", "http://localhost", 120.0, user_id="user1")

    @patch("supyworkflow.providers.supyagent._fetch_tools_metadata")
    def test_get_tool_names(self, mock_fetch):
        mock_fetch.return_value = [
            _make_tool_metadata("gmail_send"),
            _make_tool_metadata("slack_list"),
        ]

        p = SupyagentToolProvider(api_key="sk_test")
        names = p.get_tool_names()

        self.assertEqual(sorted(names), ["gmail_send", "slack_list"])

    @patch("supyworkflow.providers.supyagent.build_tool_callables")
    @patch("supyworkflow.providers.supyagent._fetch_tools_metadata")
    def test_execute_routes_to_callable(self, mock_fetch, mock_build):
        mock_fetch.return_value = [_make_tool_metadata("gmail_send")]
        mock_callable = MagicMock(return_value={"sent": True})
        mock_build.return_value = {"gmail_send": mock_callable}

        p = SupyagentToolProvider(api_key="sk_test")
        result = p.execute("gmail_send", to="test@example.com")

        mock_callable.assert_called_once_with(to="test@example.com")
        self.assertEqual(result, {"sent": True})

    @patch("supyworkflow.providers.supyagent._fetch_tools_metadata")
    def test_caches_metadata(self, mock_fetch):
        mock_fetch.return_value = [_make_tool_metadata("gmail_send")]

        p = SupyagentToolProvider(api_key="sk_test")
        p.discover()
        p.discover()

        mock_fetch.assert_called_once()


class TestHttpGatewayToolProvider(unittest.TestCase):
    @patch("httpx.Client")
    def test_discover_calls_gateway(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "ok": True,
            "data": {"tools": [_make_tool_metadata("gmail_send")]},
        }
        mock_client.get.return_value = mock_resp

        p = HttpGatewayToolProvider(
            gateway_url="http://localhost:3000/api/v1/tools/gateway",
            api_key="sk_test",
            user_id="user1",
        )
        tools = p.discover()

        self.assertEqual(len(tools), 1)
        mock_client.get.assert_called_once_with(
            "http://localhost:3000/api/v1/tools/gateway/discover"
        )

    @patch("httpx.Client")
    def test_execute_posts_to_gateway(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"ok": True, "data": {"sent": True}}
        mock_client.post.return_value = mock_resp

        # Also mock discover for the provider init
        mock_get_resp = MagicMock()
        mock_get_resp.json.return_value = {"data": {"tools": [_make_tool_metadata("gmail_send")]}}
        mock_client.get.return_value = mock_get_resp

        p = HttpGatewayToolProvider(
            gateway_url="http://localhost:3000/api/v1/tools/gateway",
            api_key="sk_test",
        )
        result = p.execute("gmail_send", to="test@example.com")

        self.assertEqual(result, {"sent": True})
        mock_client.post.assert_called_once_with(
            "http://localhost:3000/api/v1/tools/gateway/execute",
            json={"tool_name": "gmail_send", "arguments": {"to": "test@example.com"}},
        )

    @patch("httpx.Client")
    def test_headers_include_account_id(self, mock_client_cls):
        HttpGatewayToolProvider(
            gateway_url="http://localhost/gw",
            api_key="sk_test",
            user_id="user_42",
        )

        # Check the headers passed to httpx.Client
        call_kwargs = mock_client_cls.call_args[1]
        headers = call_kwargs["headers"]
        self.assertEqual(headers["X-Account-Id"], "user_42")
        self.assertEqual(headers["Authorization"], "Bearer sk_test")


class TestRuntimeWithProviders(unittest.TestCase):
    def test_extra_tools_available_in_namespace(self):
        """extra_tools should be callable from the workflow script."""
        from supyworkflow.runtime import SupyWorkflow

        call_log = []

        def my_tool(x: int = 0) -> dict:
            call_log.append(x)
            return {"result": x * 2}

        runtime = SupyWorkflow(
            api_key="test",
            user_id="test",
            providers=[],  # empty providers = no supyagent tools
            extra_tools={"my_tool": my_tool},
        )

        script = "result = my_tool(x=5)"
        result = runtime.run(script)

        self.assertEqual(result.status, "completed")
        self.assertEqual(call_log, [5])
        self.assertEqual(result.outputs["result"], {"result": 10})

    def test_run_extra_tools_override_init(self):
        """run-time extra_tools should override init-time on collision."""
        from supyworkflow.runtime import SupyWorkflow

        runtime = SupyWorkflow(
            api_key="test",
            user_id="test",
            providers=[],
            extra_tools={"my_tool": lambda: "init"},
        )

        result = runtime.run(
            "val = my_tool()",
            extra_tools={"my_tool": lambda: "runtime"},
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.outputs["val"], "runtime")

    def test_tools_property_with_providers(self):
        """tools property should aggregate names from providers + extra_tools."""
        from supyworkflow.runtime import SupyWorkflow

        p = MockProvider({"gmail_send": "Send email", "slack_list": "List channels"})
        runtime = SupyWorkflow(
            api_key="test",
            user_id="test",
            providers=[p],
            extra_tools={"custom_fn": lambda: None},
        )

        names = runtime.tools
        self.assertIn("gmail_send", names)
        self.assertIn("slack_list", names)
        self.assertIn("custom_fn", names)

    def test_provider_execute_called_during_run(self):
        """When running with providers, provider.execute() should be called for tool usage."""
        from supyworkflow.runtime import SupyWorkflow

        p = MockProvider({"get_data": "Fetch data"})
        p.set_result("get_data", {"items": [1, 2, 3]})

        runtime = SupyWorkflow(
            api_key="test",
            user_id="test",
            providers=[p],
        )

        script = "data = get_data()"
        result = runtime.run(script)

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.outputs["data"], {"items": [1, 2, 3]})


class TestRuntimeBackwardCompatibility(unittest.TestCase):
    @patch("supyworkflow.namespace.build_tool_callables")
    @patch("supyworkflow.namespace.httpx")
    def test_no_providers_uses_legacy_path(self, mock_httpx, mock_build):
        """When no providers are set, the legacy build_namespace path should be used."""
        from supyworkflow.runtime import SupyWorkflow

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": {"tools": []}}
        mock_httpx.get.return_value = mock_resp
        mock_build.return_value = {}

        runtime = SupyWorkflow(api_key="test", user_id="test")
        # providers should be None
        self.assertFalse(runtime._use_providers)


if __name__ == "__main__":
    unittest.main()
