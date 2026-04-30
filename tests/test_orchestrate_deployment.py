from __future__ import annotations

import io
import json
import os
import unittest
from pathlib import Path
from urllib.error import HTTPError
from unittest.mock import patch

from healthcare_support_agents.config import AppConfig
from healthcare_support_agents.orchestrate_deployment import (
    DeploymentPaths,
    _AGENT_ID_CACHE,
    _TOKEN_CACHE,
    _candidate_run_urls,
    _extract_run_identifiers,
    _load_mcsp_token_from_cli_cache,
    _normalized_bearer_token,
    _request_mcsp_token,
    _resolve_orchestrate_agent_id,
    _resolve_bearer_token,
    _summarize_http_error_response,
    invoke_via_api,
    agent_import_command,
    chat_command,
    tools_import_command,
)


class OrchestrateDeploymentTests(unittest.TestCase):
    def setUp(self) -> None:
        _TOKEN_CACHE.clear()
        _AGENT_ID_CACHE.clear()
        os.environ.pop("ORCHESTRATE_BEARER_TOKEN", None)

    def test_command_builders_include_expected_files(self) -> None:
        repo_root = Path("C:/tmp/healthcare")
        paths = DeploymentPaths(
            repo_root=repo_root,
            tool_file=repo_root / "src" / "healthcare_support_agents" / "orchestrate_adk_tools.py",
            requirements_file=repo_root / "requirements-orchestrate.txt",
            agent_spec_file=repo_root / "deploy" / "orchestrate" / "ai_healthcare_coordinator.agent.yaml",
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
        base = chat_command("AI_Healthcare_Coordinator", "Help me", include_reasoning=False)
        with_reasoning = chat_command("AI_Healthcare_Coordinator", "Help me", include_reasoning=True)
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

    def test_extract_run_identifiers_supports_nested_run_shape(self) -> None:
        payload = {
            "run": {
                "thread_id": "thread-1",
                "run_id": "run-123",
                "task_id": "task-abc",
            }
        }
        self.assertEqual(_extract_run_identifiers(payload), ["run-123", "task-abc"])

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

    def test_summarize_http_error_response_for_html_access_denied(self) -> None:
        raw = "<html><head><title>Access denied | iam.platform.saas.ibm.com used Cloudflare</title></head></html>"
        summary = _summarize_http_error_response(raw, "https://iam.platform.saas.ibm.com/siusermgr/api/1.0/apikeys/token")
        self.assertIn("HTML access denied page", summary)
        self.assertIn("iam.platform.saas.ibm.com", summary)

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

    def test_resolve_orchestrate_agent_id_by_name(self) -> None:
        config = AppConfig(
            watsonx_apikey=None,
            watsonx_project_id=None,
            watsonx_url="https://us-south.ml.cloud.ibm.com",
            watsonx_model="watsonx/ibm/granite-3-8b-instruct",
            serper_api_key=None,
            orchestrate_api_endpoint="https://api.dl.watson-orchestrate.ibm.com/instances/example",
            orchestrate_auth_type="mcsp",
            orchestrate_agent_name="AI_Healthcare_Coordinator",
        )

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return (
                    b'{"native":[{"name":"AI_Healthcare_Coordinator","id":"agent-123"}],'
                    b'"assistant":[],"external":[]}'
                )

        with patch("healthcare_support_agents.orchestrate_deployment.request.urlopen", return_value=FakeResponse()):
            self.assertEqual(_resolve_orchestrate_agent_id(config, "token"), "agent-123")

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

            if url.endswith("/v1/orchestrate/agents") or url.endswith("/api/v1/orchestrate/agents"):
                return FakeResponse('{"native":[{"name":"AI_Healthcare_Coordinator","id":"agent-123"}]}')

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

    def test_invoke_via_api_polls_with_fallback_identifier_when_run_id_events_missing(self) -> None:
        config = AppConfig(
            watsonx_apikey=None,
            watsonx_project_id=None,
            watsonx_url="https://us-south.ml.cloud.ibm.com",
            watsonx_model="watsonx/ibm/granite-3-8b-instruct",
            serper_api_key=None,
            orchestrate_api_endpoint="https://api.dl.watson-orchestrate.ibm.com/instances/example",
            orchestrate_bearer_token="fresh-token",
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

            if url.endswith("/v1/orchestrate/agents") or url.endswith("/api/v1/orchestrate/agents"):
                return FakeResponse('{"native":[{"name":"AI_Healthcare_Coordinator","id":"agent-123"}]}')

            if url.endswith("/v1/orchestrate/runs") or url.endswith("/api/v1/orchestrate/runs"):
                return FakeResponse('{"run":{"run_id":"run-1","task_id":"task-1"}}')

            if url.endswith("/run-1/events"):
                raise HTTPError(
                    url=url,
                    code=404,
                    msg="Not Found",
                    hdrs=None,
                    fp=io.BytesIO(b'{"error":"Not Found"}'),
                )

            if url.endswith("/task-1/events"):
                return FakeResponse('[{"event":"run.completed"}]')

            raise AssertionError(f"Unexpected request url: {url}")

        with patch("healthcare_support_agents.orchestrate_deployment.request.urlopen", side_effect=fake_urlopen):
            payload = invoke_via_api(config, "test prompt", poll_timeout_sec=2)
            self.assertEqual(payload["events"][0]["event"], "run.completed")

    def test_invoke_via_api_raises_when_agent_cannot_be_resolved(self) -> None:
        config = AppConfig(
            watsonx_apikey=None,
            watsonx_project_id=None,
            watsonx_url="https://us-south.ml.cloud.ibm.com",
            watsonx_model="watsonx/ibm/granite-3-8b-instruct",
            serper_api_key=None,
            orchestrate_api_endpoint="https://api.dl.watson-orchestrate.ibm.com/instances/example",
            orchestrate_bearer_token="fresh-token",
            orchestrate_auth_type="mcsp",
            orchestrate_agent_name="AI_Healthcare_Coordinator",
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
            if url.endswith("/v1/orchestrate/agents") or url.endswith("/api/v1/orchestrate/agents"):
                return FakeResponse('{"native": []}')
            raise AssertionError(f"Unexpected request url: {url}")

        with patch("healthcare_support_agents.orchestrate_deployment.request.urlopen", side_effect=fake_urlopen):
            with self.assertRaises(RuntimeError) as ctx:
                invoke_via_api(config, "test prompt", poll_timeout_sec=2)
            self.assertIn("Could not resolve target Orchestrate agent ID", str(ctx.exception))

    def test_invoke_via_api_retries_when_configured_agent_id_is_stale(self) -> None:
        config = AppConfig(
            watsonx_apikey=None,
            watsonx_project_id=None,
            watsonx_url="https://us-south.ml.cloud.ibm.com",
            watsonx_model="watsonx/ibm/granite-3-8b-instruct",
            serper_api_key=None,
            orchestrate_api_endpoint="https://api.dl.watson-orchestrate.ibm.com/instances/example",
            orchestrate_bearer_token="fresh-token",
            orchestrate_auth_type="mcsp",
            orchestrate_agent_name="AI_Healthcare_Coordinator",
            orchestrate_agent_id="stale-agent-id",
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

        state = {"phase": "stale"}

        def fake_urlopen(req, timeout=30):
            url = req.full_url
            if url.endswith("/v1/orchestrate/runs") or url.endswith("/api/v1/orchestrate/runs"):
                return FakeResponse('{"run":{"run_id":"run-1","task_id":"task-1"}}')

            if url.endswith("/task-1/events"):
                if state["phase"] == "stale":
                    state["phase"] = "fresh"
                    return FakeResponse(
                        '[{"event":"run.failed","data":{"last_error":"User message mentions an invalid assistant for routing"}}]'
                    )
                return FakeResponse('[{"event":"run.completed"}]')

            if url.endswith("/run-1/events"):
                return FakeResponse("[]")

            if url.endswith("/v1/orchestrate/agents") or url.endswith("/api/v1/orchestrate/agents"):
                return FakeResponse('{"native":[{"name":"AI_Healthcare_Coordinator","id":"fresh-agent-id"}]}')

            raise AssertionError(f"Unexpected request url: {url}")

        with patch("healthcare_support_agents.orchestrate_deployment.request.urlopen", side_effect=fake_urlopen):
            payload = invoke_via_api(config, "test prompt", poll_timeout_sec=2)
            self.assertEqual(payload["events"][0]["event"], "run.completed")

    def test_invoke_via_api_retries_payload_when_uuid_not_json_serializable(self) -> None:
        config = AppConfig(
            watsonx_apikey=None,
            watsonx_project_id=None,
            watsonx_url="https://us-south.ml.cloud.ibm.com",
            watsonx_model="watsonx/ibm/granite-3-8b-instruct",
            serper_api_key=None,
            orchestrate_api_endpoint="https://api.dl.watson-orchestrate.ibm.com/instances/example",
            orchestrate_bearer_token="fresh-token",
            orchestrate_auth_type="mcsp",
            orchestrate_agent_name="AI_Healthcare_Coordinator",
            orchestrate_agent_id="agent-123",
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

        state = {"run_calls": 0}

        def fake_urlopen(req, timeout=30):
            url = req.full_url
            if url.endswith("/v1/orchestrate/runs") or url.endswith("/api/v1/orchestrate/runs"):
                state["run_calls"] += 1
                payload = json.loads(req.data.decode("utf-8"))
                if payload.get("message", {}).get("assistant_id"):
                    raise HTTPError(
                        url=url,
                        code=500,
                        msg="Internal Server Error",
                        hdrs=None,
                        fp=io.BytesIO(
                            b'{"detail":"Failed to execute run: builtins.TypeError UUID is not JSON serializable"}'
                        ),
                    )
                return FakeResponse('{"run":{"run_id":"run-1"}}')

            if url.endswith("/run-1/events"):
                return FakeResponse('[{"event":"run.completed"}]')

            raise AssertionError(f"Unexpected request url: {url}")

        with patch("healthcare_support_agents.orchestrate_deployment.request.urlopen", side_effect=fake_urlopen):
            payload = invoke_via_api(config, "test prompt", poll_timeout_sec=2)
            self.assertEqual(payload["events"][0]["event"], "run.completed")
            self.assertGreaterEqual(state["run_calls"], 2)


if __name__ == "__main__":
    unittest.main()
