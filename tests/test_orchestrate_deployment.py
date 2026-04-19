from __future__ import annotations

import os
import unittest
from pathlib import Path

from healthcare_support_agents.config import AppConfig
from healthcare_support_agents.orchestrate_deployment import (
    DeploymentPaths,
    _TOKEN_CACHE,
    _candidate_run_urls,
    _resolve_bearer_token,
    agent_import_command,
    chat_command,
    tools_import_command,
)


class OrchestrateDeploymentTests(unittest.TestCase):
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
        with self.assertRaises(RuntimeError):
            _resolve_bearer_token(config)

    def test_candidate_urls_prioritize_mcsp_pattern_for_mcsp(self) -> None:
        urls = _candidate_run_urls(
            "https://api.dl.watson-orchestrate.ibm.com/instances/example",
            "mcsp",
        )
        self.assertEqual(urls[0], "https://api.dl.watson-orchestrate.ibm.com/instances/example/v1/orchestrate/runs")
        self.assertEqual(urls[1], "https://api.dl.watson-orchestrate.ibm.com/instances/example/api/v1/orchestrate/runs")


if __name__ == "__main__":
    unittest.main()
