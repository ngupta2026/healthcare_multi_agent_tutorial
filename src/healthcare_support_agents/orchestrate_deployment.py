from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib import parse, request

from .config import AppConfig

_TOKEN_CACHE: dict[str, tuple[str, float | None]] = {}
_AGENT_ID_CACHE: dict[str, str] = {}
MCSP_DEFAULT_TOKEN_TTL_SEC = 2 * 60 * 60


def _summarize_http_error_response(raw: str, url: str | None = None) -> str:
    text = (raw or "").strip()
    if not text:
        return "No response body."

    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        payload = None

    if isinstance(payload, dict):
        for key in ("message", "error_description", "error", "details", "last_error"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        compact = json.dumps(payload, ensure_ascii=False)
        return compact[:220] + "..." if len(compact) > 220 else compact

    lowered = text.lower()
    host = parse.urlparse(url or "").hostname or ""
    host_label = f" from {host}" if host else ""
    if "<html" in lowered or "<!doctype html" in lowered:
        title_match = re.search(r"<title>(.*?)</title>", text, flags=re.IGNORECASE | re.DOTALL)
        title = title_match.group(1).strip() if title_match else ""
        title_label = f" ({title})" if title else ""
        if "cloudflare" in lowered or "access denied" in lowered or "error 1010" in lowered:
            return f"HTML access denied page{host_label}{title_label}. Check VPN/firewall/proxy rules."
        return f"HTML error page{host_label}{title_label}."

    compact_text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)).strip()
    return compact_text[:220] + "..." if len(compact_text) > 220 else compact_text


@dataclass(slots=True)
class DeploymentPaths:
    repo_root: Path
    tool_file: Path
    requirements_file: Path
    agent_spec_file: Path
    package_root: Path

    @classmethod
    def defaults(cls) -> "DeploymentPaths":
        repo_root = Path(__file__).resolve().parents[2]
        return cls(
            repo_root=repo_root,
            tool_file=repo_root / "src" / "healthcare_support_agents" / "orchestrate_adk_tools.py",
            requirements_file=repo_root / "requirements-orchestrate.txt",
            agent_spec_file=repo_root / "deploy" / "orchestrate" / "healthcare_care_coordinator.agent.yaml",
            package_root=repo_root / "src",
        )


def _run(command: list[str], cwd: Path) -> str:
    child_env = os.environ.copy()
    # Avoid Windows cp1252/charmap crashes when the Orchestrate CLI prints emoji.
    child_env.setdefault("PYTHONUTF8", "1")
    child_env.setdefault("PYTHONIOENCODING", "utf-8")

    with tempfile.TemporaryFile(mode="w+b") as stdout_file, tempfile.TemporaryFile(mode="w+b") as stderr_file:
        completed = subprocess.run(
            command,
            cwd=cwd,
            stdout=stdout_file,
            stderr=stderr_file,
            env=child_env,
        )
        stdout_file.seek(0)
        stderr_file.seek(0)
        stdout = stdout_file.read().decode("utf-8", errors="replace").strip()
        stderr = stderr_file.read().decode("utf-8", errors="replace").strip()
    if stdout:
        print(stdout)
    if completed.returncode != 0:
        message = [
            f"Command failed with exit code {completed.returncode}: {' '.join(command)}",
        ]
        if stdout:
            message.append(f"stdout:\n{stdout}")
        if stderr:
            message.append(f"stderr:\n{stderr}")
        raise RuntimeError("\n\n".join(message))
    return stdout


def _run_streaming(command: list[str], cwd: Path) -> None:
    child_env = os.environ.copy()
    child_env.setdefault("PYTHONUTF8", "1")
    child_env.setdefault("PYTHONIOENCODING", "utf-8")

    completed = subprocess.run(
        command,
        cwd=cwd,
        env=child_env,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {' '.join(command)}")


def ensure_cli_available() -> None:
    try:
        _run(["orchestrate", "--help"], cwd=DeploymentPaths.defaults().repo_root)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "The orchestrate CLI is not installed. Install ADK first: pip install --upgrade ibm-watsonx-orchestrate"
        ) from exc


def _refresh_mcsp_token_via_cli(env_name: str | None) -> None:
    ensure_cli_available()
    env = env_name or "default"
    try:
        _run(["orchestrate", "env", "activate", env], cwd=DeploymentPaths.defaults().repo_root)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Failed to refresh MCSP token for environment '{env}'. "
            "Ensure the environment name is correct and the Orchestrate CLI is authenticated."
        ) from exc


def _request_mcsp_token(config: AppConfig) -> tuple[str, float | None]:
    if not config.orchestrate_api_key:
        raise RuntimeError("ORCHESTRATE_API_KEY is required for MCSP token refresh.")

    req = request.Request(
        config.orchestrate_mcsp_token_url,
        data=json.dumps({"apikey": config.orchestrate_api_key}).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=30) as response:
            body = json.loads(response.read().decode("utf-8", errors="replace"))
    except HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Failed to refresh MCSP token at {config.orchestrate_mcsp_token_url}. "
            f"HTTP {exc.code}. Response: {_summarize_http_error_response(details, config.orchestrate_mcsp_token_url)}"
        ) from exc

    token = _extract_mcsp_token(body)
    if not token:
        raise RuntimeError("MCSP token refresh response did not include a token.")

    now = time.time()
    expiry = _extract_token_expiry(body, now)
    if expiry is None:
        expiry = _decode_jwt_exp(token)
    if expiry is None:
        expiry = now + MCSP_DEFAULT_TOKEN_TTL_SEC
    return token, expiry


def _refresh_mcsp_token(config: AppConfig) -> tuple[str, float | None]:
    errors: list[str] = []
    if config.orchestrate_api_key:
        try:
            return _request_mcsp_token(config)
        except RuntimeError as exc:
            errors.append(f"api_key_refresh_failed: {exc}")

    try:
        _refresh_mcsp_token_via_cli(config.orchestrate_env_name)
        cli_token = _load_mcsp_token_from_cli_cache(config.orchestrate_env_name)
        if cli_token:
            return cli_token, None
        errors.append("cli_refresh_failed: no token found in CLI credentials cache after activate")
    except RuntimeError as exc:
        errors.append(f"cli_refresh_failed: {exc}")

    raise RuntimeError("; ".join(errors) if errors else "unknown MCSP refresh failure")


def tools_import_command(
    paths: DeploymentPaths,
    include_package_root: bool = True,
    include_requirements: bool = True,
) -> list[str]:
    command = [
        "orchestrate",
        "tools",
        "import",
        "--kind",
        "python",
        "--file",
        str(paths.tool_file),
    ]
    if include_package_root:
        command.extend(["--package-root", str(paths.package_root)])
    if include_requirements:
        command.extend(["--requirements-file", str(paths.requirements_file)])
    return command


def agent_import_command(paths: DeploymentPaths) -> list[str]:
    return [
        "orchestrate",
        "agents",
        "import",
        "-f",
        str(paths.agent_spec_file),
    ]


def agent_list_command() -> list[str]:
    return ["orchestrate", "agents", "list", "-v"]


def chat_command(agent_name: str, prompt: str, include_reasoning: bool = False) -> list[str]:
    command = ["orchestrate", "chat", "ask", "--agent-name", agent_name, prompt]
    if include_reasoning:
        command.append("--include-reasoning")
    return command


def register_tools(paths: DeploymentPaths) -> None:
    ensure_cli_available()
    print("Registering Python tools into watsonx Orchestrate...")
    try:
        _run(tools_import_command(paths, include_package_root=True, include_requirements=True), cwd=paths.repo_root)
    except RuntimeError as first_error:
        print("Primary import attempt failed. Retrying without requirements file...")
        try:
            _run(tools_import_command(paths, include_package_root=True, include_requirements=False), cwd=paths.repo_root)
        except RuntimeError:
            raise first_error
    _run(["orchestrate", "tools", "list", "-v"], cwd=paths.repo_root)


def wire_agent(paths: DeploymentPaths) -> None:
    ensure_cli_available()
    print("Importing/updating the native orchestrate agent spec...")
    _run(agent_import_command(paths), cwd=paths.repo_root)
    _run(agent_list_command(), cwd=paths.repo_root)


def invoke_via_cli(config: AppConfig, prompt: str, include_reasoning: bool = False) -> None:
    ensure_cli_available()
    print(f"Invoking agent '{config.orchestrate_agent_name}' via Orchestrate chat CLI...")
    # Run in streaming mode to avoid Windows cp1252 decode crashes from captured output.
    _run_streaming(
        chat_command(config.orchestrate_agent_name, prompt, include_reasoning=include_reasoning),
        cwd=DeploymentPaths.defaults().repo_root,
    )


def _resolve_bearer_token(config: AppConfig) -> str:
    cache_key = (config.orchestrate_endpoint or "default").rstrip("/")
    now = time.time()

    direct_token = _normalized_bearer_token(config.orchestrate_bearer_token)
    if direct_token:
        token = direct_token
        _TOKEN_CACHE[cache_key] = (token, None)
        os.environ["ORCHESTRATE_BEARER_TOKEN"] = token
        return token

    cached = _TOKEN_CACHE.get(cache_key)
    if cached:
        token, expiry = cached
        if not expiry or now < expiry - 30:
            return token

    if (config.orchestrate_auth_type or "").lower() == "mcsp":
        try:
            refreshed_token, refreshed_expiry = _refresh_mcsp_token(config)
        except RuntimeError as exc:
            raise RuntimeError(
                "ORCHESTRATE_AUTH_TYPE is set to 'mcsp', but token refresh failed. "
                "Set ORCHESTRATE_API_KEY for server-side refresh or ensure `orchestrate env activate <env-name>` works."
                f" Details: {exc}"
            ) from exc
        _TOKEN_CACHE[cache_key] = (refreshed_token, refreshed_expiry)
        os.environ["ORCHESTRATE_BEARER_TOKEN"] = refreshed_token
        return refreshed_token

    if not config.orchestrate_api_key:
        missing = ", ".join(config.missing_orchestrate_fields())
        raise RuntimeError(f"Cannot call Orchestrate API. Missing: {missing}")

    body = parse.urlencode(
        {
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
            "apikey": config.orchestrate_api_key,
        }
    ).encode("utf-8")
    req = request.Request(
        config.orchestrate_iam_url,
        data=body,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Failed to exchange ORCHESTRATE_API_KEY at IAM endpoint ({config.orchestrate_iam_url}). "
            f"HTTP {exc.code}. Response: {_summarize_http_error_response(details, config.orchestrate_iam_url)}"
        ) from exc
    token = payload["access_token"]
    expiry = None
    if "expiration" in payload:
        try:
            expiry = float(payload["expiration"])
        except (TypeError, ValueError):
            expiry = None
    elif "expires_in" in payload:
        try:
            expiry = now + float(payload["expires_in"])
        except (TypeError, ValueError):
            expiry = None

    _TOKEN_CACHE[cache_key] = (token, expiry)
    os.environ["ORCHESTRATE_BEARER_TOKEN"] = token
    return token


def _normalized_bearer_token(raw: str | None) -> str | None:
    if not raw:
        return None
    token = raw.strip().strip('"').strip("'")
    if not token:
        return None
    if token.startswith("<") and token.endswith(">"):
        return None
    return token


def _extract_mcsp_token(payload: Any) -> str | None:
    if isinstance(payload, str):
        return _normalized_bearer_token(payload)
    if not isinstance(payload, dict):
        return None

    for key in ("token", "access_token", "jwt", "id_token", "wxo_mcsp_token"):
        value = payload.get(key)
        if isinstance(value, str):
            token = _normalized_bearer_token(value)
            if token:
                return token

    for nested_key in ("data", "result"):
        nested = payload.get(nested_key)
        token = _extract_mcsp_token(nested)
        if token:
            return token

    return None


def _extract_token_expiry(payload: Any, now: float) -> float | None:
    if not isinstance(payload, dict):
        return None

    for key in ("expiration", "expires_at", "exp"):
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                continue

    expires_in = payload.get("expires_in")
    if isinstance(expires_in, (int, float)):
        return now + float(expires_in)
    if isinstance(expires_in, str):
        try:
            return now + float(expires_in.strip())
        except ValueError:
            return None

    return None


def _decode_jwt_exp(token: str) -> float | None:
    parts = token.split(".")
    if len(parts) < 2:
        return None
    payload = parts[1]
    payload += "=" * (-len(payload) % 4)
    try:
        decoded = base64.urlsafe_b64decode(payload.encode("utf-8")).decode("utf-8", errors="replace")
        parsed = json.loads(decoded)
    except Exception:
        return None

    exp = parsed.get("exp")
    if isinstance(exp, (int, float)):
        return float(exp)
    if isinstance(exp, str):
        try:
            return float(exp.strip())
        except ValueError:
            return None
    return None


def _load_mcsp_token_from_cli_cache(env_name: str | None) -> str | None:
    tokens = _load_mcsp_tokens_from_cli_cache()
    if not tokens:
        return None

    preferred = [env_name, _load_active_orchestrate_env_name()]
    for name in preferred:
        if name and name in tokens:
            return tokens[name]

    return next(iter(tokens.values()), None)


def _load_active_orchestrate_env_name() -> str | None:
    config_path = Path.home() / ".config" / "orchestrate" / "config.yaml"
    if not config_path.exists():
        return None

    in_context = False
    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        if not in_context:
            if stripped == "context:":
                in_context = True
            continue

        if re.match(r"^\S", line):
            break

        match = re.match(r"^\s{2}active_environment:\s*(.+)\s*$", line)
        if match:
            value = match.group(1).strip().strip('"').strip("'")
            if value and value.lower() not in {"null", "none"}:
                return value
            return None

    return None


def _load_mcsp_tokens_from_cli_cache() -> dict[str, str]:
    credentials_path = Path.home() / ".cache" / "orchestrate" / "credentials.yaml"
    if not credentials_path.exists():
        return {}

    current_env: str | None = None
    in_auth_section = False
    tokens: dict[str, str] = {}
    for raw_line in credentials_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            continue

        if not in_auth_section:
            if stripped == "auth:":
                in_auth_section = True
            continue

        env_match = re.match(r"^\s{2}([A-Za-z0-9_.-]+):\s*(.*)$", line)
        if env_match:
            current_env = env_match.group(1)
            continue

        if re.match(r"^\S", line):
            break

        token_match = re.match(r"^\s{4}wxo_mcsp_token:\s*(.+)\s*$", line)
        if token_match:
            token = _normalized_bearer_token(token_match.group(1))
            if token and current_env:
                tokens[current_env] = token

    return tokens


def _candidate_run_urls(endpoint: str, auth_type: str | None) -> list[str]:
    base = endpoint.rstrip("/")
    is_mcsp = (auth_type or "").lower().startswith("mcsp")
    if is_mcsp:
        return [
            f"{base}/v1/orchestrate/runs",
            f"{base}/api/v1/orchestrate/runs",
        ]
    return [
        f"{base}/api/v1/orchestrate/runs",
        f"{base}/v1/orchestrate/runs",
    ]


def _candidate_agents_urls(endpoint: str, auth_type: str | None) -> list[str]:
    base = endpoint.rstrip("/")
    is_mcsp = (auth_type or "").lower().startswith("mcsp")
    if is_mcsp:
        return [
            f"{base}/v1/orchestrate/agents",
            f"{base}/api/v1/orchestrate/agents",
        ]
    return [
        f"{base}/api/v1/orchestrate/agents",
        f"{base}/v1/orchestrate/agents",
    ]


def _extract_agents(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]

    if not isinstance(payload, dict):
        return []

    candidates: list[dict[str, Any]] = []
    for key in ("agents", "data", "native", "assistant", "external"):
        value = payload.get(key)
        if isinstance(value, list):
            candidates.extend(item for item in value if isinstance(item, dict))
    return candidates


def _resolve_orchestrate_agent_id(config: AppConfig, token: str) -> str | None:
    if config.orchestrate_agent_id:
        return config.orchestrate_agent_id

    endpoint = config.orchestrate_endpoint
    if not endpoint:
        return None
    target_name = (config.orchestrate_agent_name or "").strip().lower()
    if not target_name:
        return None
    cache_key = f"{endpoint.rstrip('/')}::{target_name}"
    if cache_key in _AGENT_ID_CACHE:
        return _AGENT_ID_CACHE[cache_key]

    for agents_url in _candidate_agents_urls(endpoint, config.orchestrate_auth_type):
        agents_request = request.Request(
            agents_url,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
            },
            method="GET",
        )
        try:
            with request.urlopen(agents_request, timeout=10) as response:
                agents_payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            if exc.code == 404:
                continue
            continue
        except Exception:
            continue

        for agent in _extract_agents(agents_payload):
            name = str(agent.get("name") or "").strip().lower()
            display_name = str(agent.get("display_name") or agent.get("title") or "").strip().lower()
            if target_name in {name, display_name}:
                agent_id = agent.get("id")
                if isinstance(agent_id, str) and agent_id:
                    _AGENT_ID_CACHE[cache_key] = agent_id
                    return agent_id
    return None


def _extract_last_error(events: list[dict[str, Any]]) -> str | None:
    for event in events:
        if not isinstance(event, dict):
            continue
        event_name = event.get("event")
        if event_name not in {"run.failed", "error"}:
            continue
        data = event.get("data")
        if not isinstance(data, dict):
            continue
        detail = data.get("last_error") or data.get("error") or data.get("message") or data.get("reason")
        if isinstance(detail, str) and detail.strip():
            return detail.strip()
    return None


def _build_run_request_payloads(
    prompt: str,
    target_agent_id: str,
    orchestrate_agent_name: str,
    environment_id: str | None,
) -> list[dict[str, Any]]:
    base_message: dict[str, Any] = {
        "role": "user",
        "content": [
            {
                "response_type": "text",
                "text": prompt,
            }
        ],
    }

    payloads: list[dict[str, Any]] = []

    message_with_assistant = dict(base_message)
    message_with_assistant["assistant_id"] = target_agent_id
    payload_with_agent: dict[str, Any] = {
        "message": message_with_assistant,
        "agent_id": target_agent_id,
    }
    if environment_id:
        payload_with_agent["environment_id"] = environment_id
    payloads.append(payload_with_agent)

    if environment_id:
        payloads.append(
            {
                "message": message_with_assistant,
                "agent_id": target_agent_id,
            }
        )

    payloads.append(
        {
            "message": dict(base_message),
            "agent_id": target_agent_id,
        }
    )

    message_with_mentions = dict(base_message)
    message_with_mentions["mentions"] = [
        {
            "type": "agent",
            "id": target_agent_id,
            "name": orchestrate_agent_name,
        }
    ]
    payloads.append({"message": message_with_mentions})

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for payload in payloads:
        marker = json.dumps(payload, sort_keys=True)
        if marker in seen:
            continue
        seen.add(marker)
        deduped.append(payload)
    return deduped


def _extract_run_identifiers(run_payload: dict[str, Any]) -> list[str]:
    direct_keys = ("id", "run_id", "task_id", "a2a_task_id")
    candidates: list[str] = []
    sources: list[dict[str, Any]] = [run_payload]

    nested_run = run_payload.get("run")
    if isinstance(nested_run, dict):
        sources.append(nested_run)

    for source in sources:
        for key in direct_keys:
            value = source.get(key)
            if isinstance(value, str) and value:
                candidates.append(value)

    deduped: list[str] = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)
    return deduped


def _extract_run_identifier(run_payload: dict[str, Any]) -> str | None:
    identifiers = _extract_run_identifiers(run_payload)
    return identifiers[0] if identifiers else None


def invoke_via_api(
    config: AppConfig,
    prompt: str,
    poll_timeout_sec: int = 40,
    retried_after_401: bool = False,
    retried_after_invalid_assistant: bool = False,
) -> dict[str, Any]:
    endpoint = config.orchestrate_endpoint
    if not endpoint:
        missing = ", ".join(config.missing_orchestrate_fields())
        raise RuntimeError(f"Cannot call Orchestrate API. Missing: {missing}")

    token = _resolve_bearer_token(config)
    target_agent_id = _resolve_orchestrate_agent_id(config, token)
    if not target_agent_id:
        raise RuntimeError(
            "Could not resolve target Orchestrate agent ID for "
            f"ORCHESTRATE_AGENT_NAME='{config.orchestrate_agent_name}'. "
            "Set ORCHESTRATE_AGENT_ID in .env/.streamlit secrets (or ensure agent listing is accessible) to avoid routing to default assistant."
        )

    payload_variants = _build_run_request_payloads(
        prompt=prompt,
        target_agent_id=target_agent_id,
        orchestrate_agent_name=config.orchestrate_agent_name,
        environment_id=config.orchestrate_agent_environment_id,
    )
    run_payload: dict[str, Any] | None = None
    run_base_url: str | None = None
    errors: list[str] = []

    for runs_url in _candidate_run_urls(endpoint, config.orchestrate_auth_type):
        for payload_index, body_payload in enumerate(payload_variants):
            body = json.dumps(body_payload).encode("utf-8")
            run_request = request.Request(
                runs_url,
                data=body,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                method="POST",
            )
            try:
                with request.urlopen(run_request, timeout=45) as response:
                    run_payload = json.loads(response.read().decode("utf-8"))
                    run_base_url = runs_url
                    break
            except HTTPError as exc:
                details = exc.read().decode("utf-8", errors="replace")
                summarized = _summarize_http_error_response(details, runs_url)
                if exc.code == 404:
                    errors.append(f"{runs_url} -> 404 Not Found")
                    continue
                if exc.code == 500 and "uuid is not json serializable" in summarized.lower():
                    errors.append(f"{runs_url} payload[{payload_index}] -> 500 UUID serialization")
                    continue
                if exc.code == 401:
                    _TOKEN_CACHE.pop(endpoint.rstrip("/"), None)
                    if (
                        (config.orchestrate_auth_type or "").lower() == "mcsp"
                        and not retried_after_401
                    ):
                        try:
                            refreshed_token, refreshed_expiry = _refresh_mcsp_token(config)
                            _TOKEN_CACHE[endpoint.rstrip("/")] = (refreshed_token, refreshed_expiry)
                            os.environ["ORCHESTRATE_BEARER_TOKEN"] = refreshed_token
                            # Clear any stale static token from config so retry uses refreshed cache/auth path.
                            retry_config = replace(config, orchestrate_bearer_token=None)
                            return invoke_via_api(
                                retry_config,
                                prompt,
                                poll_timeout_sec=poll_timeout_sec,
                                retried_after_401=True,
                                retried_after_invalid_assistant=retried_after_invalid_assistant,
                            )
                        except RuntimeError as refresh_exc:
                            raise RuntimeError(
                                "Orchestrate API returned HTTP 401 Unauthorized and automatic MCSP token refresh failed. "
                                "Verify ORCHESTRATE_API_KEY (server-side refresh) or `orchestrate env activate <env-name>` (CLI refresh), then retry. "
                                f"Details: {refresh_exc}"
                            ) from exc
                    raise RuntimeError(
                        "Orchestrate API returned HTTP 401 Unauthorized. "
                        "Ensure ORCHESTRATE_API_ENDPOINT is correct and provide valid auth: "
                        "ORCHESTRATE_API_KEY (recommended for MCSP) or fresh ORCHESTRATE_BEARER_TOKEN."
                    ) from exc
                raise RuntimeError(
                    f"Orchestrate API run creation failed at {runs_url} with HTTP {exc.code}. "
                    f"Response: {summarized}"
                ) from exc
        if run_payload is not None and run_base_url is not None:
            break

    if run_payload is None or run_base_url is None:
        attempted = "; ".join(errors) if errors else "no candidate URL succeeded"
        raise RuntimeError(
            "Orchestrate API run creation failed with HTTP 404 on all known endpoint patterns. "
            f"Attempted: {attempted}"
        )

    run_ids = _extract_run_identifiers(run_payload)
    if not run_ids:
        return {"run": run_payload, "events": []}

    deadline = time.time() + poll_timeout_sec
    events: list[dict[str, Any]] = []
    events_urls = [f"{run_base_url.rstrip('/')}/{run_id}/events" for run_id in run_ids]
    terminal_reached = False
    while time.time() < deadline:
        for events_url in events_urls:
            events_request = request.Request(
                events_url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/json",
                },
                method="GET",
            )
            try:
                with request.urlopen(events_request, timeout=45) as response:
                    candidate_events = json.loads(response.read().decode("utf-8"))
            except HTTPError as exc:
                if exc.code == 404:
                    continue
                raise

            if isinstance(candidate_events, list):
                events = candidate_events
            else:
                events = []

            if any(
                isinstance(event, dict) and event.get("event") in {"run.completed", "run.failed", "run.cancelled", "done"}
                for event in events
            ):
                terminal_reached = True
                break
        if terminal_reached:
            break
        time.sleep(2)

    last_error = _extract_last_error(events)
    if (
        last_error
        and "invalid assistant for routing" in last_error.lower()
        and config.orchestrate_agent_id
        and not retried_after_invalid_assistant
    ):
        retry_config = replace(config, orchestrate_agent_id=None)
        return invoke_via_api(
            retry_config,
            prompt,
            poll_timeout_sec=poll_timeout_sec,
            retried_after_401=retried_after_401,
            retried_after_invalid_assistant=True,
        )

    return {"run": run_payload, "events": events}


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="True watsonx Orchestrate deployment flow helper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("register-tools", help="Import Python tools into Orchestrate")
    subparsers.add_parser("wire-agent", help="Import/update the agent spec with tool wiring")

    invoke = subparsers.add_parser("invoke", help="Invoke the agent using CLI or REST API")
    invoke.add_argument("--mode", choices=["cli", "api"], default="cli")
    invoke.add_argument("--prompt", required=True)
    invoke.add_argument("--include-reasoning", action="store_true")

    all_cmd = subparsers.add_parser("all", help="Run tool registration + agent wiring + optional invoke")
    all_cmd.add_argument("--invoke", action="store_true")
    all_cmd.add_argument("--mode", choices=["cli", "api"], default="cli")
    all_cmd.add_argument("--prompt", default="Summarize PT-1001 recovery status and call out any escalation.")
    all_cmd.add_argument("--include-reasoning", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    config = AppConfig.from_env()
    paths = DeploymentPaths.defaults()

    if args.command == "register-tools":
        register_tools(paths)
        return 0

    if args.command == "wire-agent":
        wire_agent(paths)
        return 0

    if args.command == "invoke":
        if args.mode == "cli":
            invoke_via_cli(config, args.prompt, include_reasoning=args.include_reasoning)
        else:
            payload = invoke_via_api(config, args.prompt)
            print(json.dumps(payload, indent=2))
        return 0

    if args.command == "all":
        register_tools(paths)
        wire_agent(paths)
        if args.invoke:
            if args.mode == "cli":
                invoke_via_cli(config, args.prompt, include_reasoning=args.include_reasoning)
            else:
                payload = invoke_via_api(config, args.prompt)
                print(json.dumps(payload, indent=2))
        return 0

    raise RuntimeError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
