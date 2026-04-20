from __future__ import annotations

import io
import os
import unittest
from pathlib import Path
from urllib.error import HTTPError
from unittest.mock import patch

from healthcare_support_agents.config import AppConfig
from healthcare_support_agents.orchestrate_deployment import (
    DeploymentPaths,
    _TOKEN_CACHE,
    _candidate_run_urls,
    _load_mcsp_token_from_cli_cache,
    _normalized_bearer_token,
    _request_mcsp_token,
    _resolve_bearer_token,
    invoke_via_api,
    agent_import_command,
    chat_command,
    tools_import_command,
)


class OrchestrateDeploymentTests(unittest.TestCase):
    def setUp(self) -> None:
        _TOKEN_CACHE.clear()
        os.environ.pop("ORCHESTRATE_BEARER_TOKEN", None)

    def test_command_builders_include_expected_files(self) -> None:
        repo_root = Path("C:/tmp/healthcare")
        paths = DeploymentPaths(
            repo_root=repo_root,
            tool_file=repo_root / "src" / "healthcare_support_agents" / "orchestrate_adk_tools.py",
            requirements_file=repo_root / "requirements-orchestrate.txt",
            agent_spec_file=repo_root / "deploy" / "orchestrate" / "healthcare_care_coordinator.agent.yaml",
            package_root=repo_root,
        )

        tools_command = tools_import_command(paths)
        agent_command = agent_import_command(paths)

        self.assertEqual(tools_command[:4], ["orchestrate", "tools", "import", "--kind"])
        self.assertIn("python", tools_command)
        self.assertIn(str(paths.tool_file), tools_command)
        self.assertIn(str(paths.requirements_file), tools_command)
        self.assertIn(str(paths.package_root), tools_command)

        self.assertEqual(agent_command[:3], ["orchestrate", "agents", "import"])
        self.assertIn(str(paths.agent_spec_file), agent_command)

    def test_chat_command_reasoning_flag(self) -> None:
        base = chat_command("Healthcare_Care_Coordinator", "Help me", include_reasoning=False)
        with_reasoning = chat_command("Healthcare_Care_Coordinator", "Help me", include_reasoning=True)
        self.assertNotIn("--include-reasoning", base)
        self.assertIn("--include-reasoning", with_reasoning)

    def test_orchestrate_api_ready_when_endpoint_and_key_exist(self) -> None:
        config = AppConfig(
            watsonx_apikey=None,
            watsonx_project_id=None,
            watsonx_url="https://us-south.ml.cloud.ibm.com",
            watsonx_model="watsonx/ibm/granite-3-8b-instruct",
            serper_api_key=None,
            orchestrate_api_endpoint="https://example.orchestrate.ibm.com",
            orchestrate_api_key="abc",
        )
        self.assertTrue(config.orchestrate_api_ready)

    def test_bearer_token_from_config_is_reused_and_exported(self) -> None:
        config = AppConfig(
            watsonx_apikey=None,
            watsonx_project_id=None,
            watsonx_url="https://us-south.ml.cloud.ibm.com",
            watsonx_model="watsonx/ibm/granite-3-8b-instruct",
            serper_api_key=None,
            orchestrate_api_endpoint="https://example.orchestrate.ibm.com",
            orchestrate_bearer_token="token-123",
            orchestrate_auth_type="mcsp",
        )
        token = _resolve_bearer_token(config)
        self.assertEqual(token, "token-123")
        self.assertEqual(os.environ.get("ORCHESTRATE_BEARER_TOKEN"), "token-123")

    def test_explicit_token_overrides_cached_token(self) -> None:
        endpoint = "https://example.orchestrate.ibm.com"
        _TOKEN_CACHE[endpoint] = ("stale-token", None)
        config = AppConfig(
            watsonx_apikey=None,
            watsonx_project_id=None,
            watsonx_url="https://us-south.ml.cloud.ibm.com",
            watsonx_model="watsonx/ibm/granite-3-8b-instruct",
            serper_api_key=None,
            orchestrate_api_endpoint=endpoint,
            orchestrate_bearer_token="fresh-token",
            orchestrate_auth_type="mcsp",
        )
        token = _resolve_bearer_token(config)
        self.assertEqual(token, "fresh-token")

    def test_placeholder_token_is_rejected(self) -> None:
        config = AppConfig(
            watsonx_apikey=None,
            watsonx_project_id=None,
            watsonx_url="https://us-south.ml.cloud.ibm.com",
            watsonx_model="watsonx/ibm/granite-3-8b-instruct",
            serper_api_key=None,
            orchestrate_api_endpoint="https://example.orchestrate.ibm.com",
            orchestrate_bearer_token="<valid bearer token>",
            orchestrate_auth_type="mcsp",
        )
        self.assertIsNone(_normalized_bearer_token(config.orchestrate_bearer_token))

    def test_candidate_urls_prioritize_mcsp_pattern_for_mcsp(self) -> None:
        urls = _candidate_run_urls(
            "https://api.dl.watson-orchestrate.ibm.com/instances/example",
            "mcsp",
        )
        self.assertEqual(urls[0], "https://api.dl.watson-orchestrate.ibm.com/instances/example/v1/orchestrate/runs")
        self.assertEqual(urls[1], "https://api.dl.watson-orchestrate.ibm.com/instances/example/api/v1/orchestrate/runs")

    def test_load_mcsp_token_from_cli_cache_uses_named_env(self) -> None:
        fake_yaml = (
            "auth:\n"
            "  healthcare-aws:\n"
            "    wxo_mcsp_token: token-healthcare\n"
            "  local:\n"
            "    wxo_mcsp_token: token-local\n"
        )
        with (
            patch("healthcare_support_agents.orchestrate_deployment.Path.home", return_value=Path("/tmp")),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.read_text", return_value=fake_yaml),
        ):
            token = _load_mcsp_token_from_cli_cache("healthcare-aws")
            self.assertEqual(token, "token-healthcare")

    def test_request_mcsp_token_uses_api_key_endpoint(self) -> None:
        config = AppConfig(
            watsonx_apikey=None,
            watsonx_project_id=None,
            watsonx_url="https://us-south.ml.cloud.ibm.com",
            watsonx_model="watsonx/ibm/granite-3-8b-instruct",
            serper_api_key=None,
            orchestrate_api_endpoint="https://api.dl.watson-orchestrate.ibm.com/instances/example",
            orchestrate_api_key="api-key-123",
            orchestrate_auth_type="mcsp",
        )

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return b'{"token":"jwt-token-abc","expires_in":1200}'

        with patch("healthcare_support_agents.orchestrate_deployment.request.urlopen", return_value=FakeResponse()):
            token, expiry = _request_mcsp_token(config)
            self.assertEqual(token, "jwt-token-abc")
            self.assertIsNotNone(expiry)

    def test_resolve_bearer_token_prefers_mcsp_api_key_refresh(self) -> None:
        config = AppConfig(
            watsonx_apikey=None,
            watsonx_project_id=None,
            watsonx_url="https://us-south.ml.cloud.ibm.com",
            watsonx_model="watsonx/ibm/granite-3-8b-instruct",
            serper_api_key=None,
            orchestrate_api_endpoint="https://api.dl.watson-orchestrate.ibm.com/instances/example",
            orchestrate_api_key="api-key-123",
            orchestrate_auth_type="mcsp",
        )

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return b'{"access_token":"refreshed-token","expires_in":1200}'

        with patch("healthcare_support_agents.orchestrate_deployment.request.urlopen", return_value=FakeResponse()):
            token = _resolve_bearer_token(config)
            self.assertEqual(token, "refreshed-token")

    def test_invoke_via_api_retries_with_refreshed_token_after_401(self) -> None:
        config = AppConfig(
            watsonx_apikey=None,
            watsonx_project_id=None,
            watsonx_url="https://us-south.ml.cloud.ibm.com",
            watsonx_model="watsonx/ibm/granite-3-8b-instruct",
            serper_api_key=None,
            orchestrate_api_endpoint="https://api.dl.watson-orchestrate.ibm.com/instances/example",
            orchestrate_api_key="api-key-123",
            orchestrate_bearer_token="stale-token",
            orchestrate_auth_type="mcsp",
        )

        class FakeResponse:
            def __init__(self, body: str):
                self._body = body.encode("utf-8")

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return self._body

        def fake_urlopen(req, timeout=30):
            url = req.full_url
            auth = req.headers.get("Authorization", "")

            if url == config.orchestrate_mcsp_token_url:
                return FakeResponse('{"token":"fresh-token","expires_in":1200}')

            if url.endswith("/v1/orchestrate/runs") or url.endswith("/api/v1/orchestrate/runs"):
                if auth == "Bearer stale-token":
                    raise HTTPError(
                        url=url,
                        code=401,
                        msg="Unauthorized",
                        hdrs=None,
                        fp=io.BytesIO(b'{"error":"Unauthorized"}'),
                    )
                if auth == "Bearer fresh-token":
                    return FakeResponse('{"id":"run-1"}')

            if url.endswith("/events"):
                return FakeResponse('[{"event":"run.completed"}]')

            raise AssertionError(f"Unexpected request url/auth: {url} {auth}")

        with patch("healthcare_support_agents.orchestrate_deployment.request.urlopen", side_effect=fake_urlopen):
            payload = invoke_via_api(config, "test prompt", poll_timeout_sec=2)
            self.assertEqual(payload["run"]["id"], "run-1")


if __name__ == "__main__":
    unittest.main()
