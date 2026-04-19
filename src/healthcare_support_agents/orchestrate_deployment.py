from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib import parse, request

from .config import AppConfig


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
    return completed.stdout


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
    if config.orchestrate_bearer_token:
        return config.orchestrate_bearer_token

    if (config.orchestrate_auth_type or "").lower() == "mcsp":
        raise RuntimeError(
            "ORCHESTRATE_AUTH_TYPE is set to 'mcsp'. "
            "API mode requires ORCHESTRATE_BEARER_TOKEN for mcsp environments. "
            "The WXO API key cannot be exchanged at IBM IAM token endpoint."
        )

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
            f"HTTP {exc.code}. Response: {details}"
        ) from exc
    return payload["access_token"]


def invoke_via_api(config: AppConfig, prompt: str, poll_timeout_sec: int = 40) -> dict[str, Any]:
    endpoint = config.orchestrate_endpoint
    if not endpoint:
        missing = ", ".join(config.missing_orchestrate_fields())
        raise RuntimeError(f"Cannot call Orchestrate API. Missing: {missing}")

    token = _resolve_bearer_token(config)
    message: dict[str, Any] = {
        "role": "user",
        "content": [
            {
                "response_type": "text",
                "text": prompt,
            }
        ],
    }

    if config.orchestrate_agent_id:
        message["mentions"] = [
            {
                "type": "agent",
                "id": config.orchestrate_agent_id,
                "name": config.orchestrate_agent_name,
            }
        ]

    body = json.dumps({"message": message}).encode("utf-8")
    run_request = request.Request(
        f"{endpoint.rstrip('/')}/api/v1/orchestrate/runs",
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    with request.urlopen(run_request, timeout=45) as response:
        run_payload = json.loads(response.read().decode("utf-8"))

    run_id = run_payload.get("id")
    if not run_id:
        return {"run": run_payload, "events": []}

    deadline = time.time() + poll_timeout_sec
    events: list[dict[str, Any]] = []
    while time.time() < deadline:
        events_request = request.Request(
            f"{endpoint.rstrip('/')}/api/v1/orchestrate/runs/{run_id}/events",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
            },
            method="GET",
        )
        with request.urlopen(events_request, timeout=45) as response:
            events = json.loads(response.read().decode("utf-8"))
        if any(event.get("event") in {"run.completed", "run.failed", "run.cancelled", "done"} for event in events):
            break
        time.sleep(2)

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
