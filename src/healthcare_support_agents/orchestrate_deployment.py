from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import parse, request

from .config import AppConfig


@dataclass(slots=True)
class DeploymentPaths:
    repo_root: Path
    tool_file: Path
    requirements_file: Path
    agent_spec_file: Path

    @classmethod
    def defaults(cls) -> "DeploymentPaths":
        repo_root = Path(__file__).resolve().parents[2]
        return cls(
            repo_root=repo_root,
            tool_file=repo_root / "src" / "healthcare_support_agents" / "orchestrate_adk_tools.py",
            requirements_file=repo_root / "requirements-orchestrate.txt",
            agent_spec_file=repo_root / "deploy" / "orchestrate" / "healthcare_care_coordinator.agent.yaml",
        )


def _run(command: list[str], cwd: Path) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    )
    if completed.stdout.strip():
        print(completed.stdout.strip())
    return completed.stdout


def ensure_cli_available() -> None:
    try:
        _run(["orchestrate", "--help"], cwd=DeploymentPaths.defaults().repo_root)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "The orchestrate CLI is not installed. Install ADK first: pip install --upgrade ibm-watsonx-orchestrate"
        ) from exc


def tools_import_command(paths: DeploymentPaths) -> list[str]:
    return [
        "orchestrate",
        "tools",
        "import",
        "-k",
        "python",
        "-f",
        str(paths.tool_file),
        "-r",
        str(paths.requirements_file),
    ]


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
    _run(tools_import_command(paths), cwd=paths.repo_root)
    _run(["orchestrate", "tools", "list", "-v"], cwd=paths.repo_root)


def wire_agent(paths: DeploymentPaths) -> None:
    ensure_cli_available()
    print("Importing/updating the native orchestrate agent spec...")
    _run(agent_import_command(paths), cwd=paths.repo_root)
    _run(agent_list_command(), cwd=paths.repo_root)


def invoke_via_cli(config: AppConfig, prompt: str, include_reasoning: bool = False) -> None:
    ensure_cli_available()
    print(f"Invoking agent '{config.orchestrate_agent_name}' via Orchestrate chat CLI...")
    _run(chat_command(config.orchestrate_agent_name, prompt, include_reasoning=include_reasoning), cwd=DeploymentPaths.defaults().repo_root)


def _resolve_bearer_token(config: AppConfig) -> str:
    if config.orchestrate_bearer_token:
        return config.orchestrate_bearer_token

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
    with request.urlopen(req, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
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
