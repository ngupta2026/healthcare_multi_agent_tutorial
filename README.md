# healthcare_multi_agent_tutorial

Healthcare multi-agent recovery app with a **true watsonx Orchestrate deployment flow**:

- Tool registration into Orchestrate (ADK CLI)
- Agent wiring/import (native agent spec)
- Invocation path through both Orchestrate CLI chat and Orchestrate REST runs API

The repo also includes a local deterministic orchestrator and a Streamlit demo UI.

## What is included

- Local care orchestration stack:
  - [orchestrator.py](src/healthcare_support_agents/orchestrator.py)
  - [agents.py](src/healthcare_support_agents/agents.py)
  - [connectors.py](src/healthcare_support_agents/connectors.py)
- Orchestrate ADK tools:
  - [orchestrate_adk_tools.py](src/healthcare_support_agents/orchestrate_adk_tools.py)
- Orchestrate deployment automation:
  - [orchestrate_deployment.py](src/healthcare_support_agents/orchestrate_deployment.py)
- Agent spec for Orchestrate import:
  - [healthcare_care_coordinator.agent.yaml](deploy/orchestrate/healthcare_care_coordinator.agent.yaml)
- Optional watsonx tool-calling demo path:
  - [watsonx_tool_calling_example.py](watsonx_tool_calling_example.py)

## Environment setup

1. Install local package:

```powershell
pip install -e .
```

2. Copy env template:

```powershell
Copy-Item .env.example .env
```

3. Fill required values in `.env`:

```text
WATSONX_APIKEY=your_watsonx_api_key_here
WATSONX_PROJECT_ID=your_watsonx_project_id_here
WATSONX_URL=https://us-south.ml.cloud.ibm.com
WATSONX_MODEL=watsonx/ibm/granite-3-8b-instruct
SERPER_API_KEY=your_serper_api_key_here

ORCHESTRATE_INSTANCE_URL=https://your-orchestrate-instance-url
ORCHESTRATE_API_ENDPOINT=https://your-orchestrate-instance-url
ORCHESTRATE_API_KEY=your_orchestrate_api_key_here
ORCHESTRATE_BEARER_TOKEN=
ORCHESTRATE_ENV_NAME=healthcare-dev
ORCHESTRATE_AGENT_NAME=Healthcare_Care_Coordinator
ORCHESTRATE_AGENT_ID=
ORCHESTRATE_AUTH_TYPE=ibm_iam
ORCHESTRATE_IAM_URL=https://iam.cloud.ibm.com/identity/token
```

## True Orchestrate deployment flow

### 1. Configure and activate ADK environment

Install ADK CLI, then add/activate your Orchestrate environment:

```powershell
pip install --upgrade ibm-watsonx-orchestrate
orchestrate env add -n healthcare-dev -u https://your-orchestrate-instance-url --type ibm_iam --activate
```

### 2. Register tools in Orchestrate

```powershell
$env:PYTHONPATH="src"
python -m healthcare_support_agents.orchestrate_deployment register-tools
```

This imports [orchestrate_adk_tools.py](src/healthcare_support_agents/orchestrate_adk_tools.py) as Python tools.

### 3. Wire agent (import/update)

```powershell
$env:PYTHONPATH="src"
python -m healthcare_support_agents.orchestrate_deployment wire-agent
```

This imports [healthcare_care_coordinator.agent.yaml](deploy/orchestrate/healthcare_care_coordinator.agent.yaml), wiring the registered tools to the native Orchestrate agent.

### 4A. Invoke through Orchestrate CLI chat

```powershell
$env:PYTHONPATH="src"
python -m healthcare_support_agents.orchestrate_deployment invoke --mode cli --prompt "Review PT-1001 and summarize escalation risk."
```

### 4B. Invoke through Orchestrate REST API

```powershell
$env:PYTHONPATH="src"
python -m healthcare_support_agents.orchestrate_deployment invoke --mode api --prompt "Review PT-1001 and summarize escalation risk."
```

The API mode posts to `/api/v1/orchestrate/runs` and polls `/api/v1/orchestrate/runs/{run_id}/events`.

### One-command setup (register + wire)

```powershell
$env:PYTHONPATH="src"
python -m healthcare_support_agents.orchestrate_deployment all
```

Or include invocation:

```powershell
$env:PYTHONPATH="src"
python -m healthcare_support_agents.orchestrate_deployment all --invoke --mode cli --prompt "Review PT-2002 logistics blockers."
```

## Streamlit UI

```powershell
$env:PYTHONPATH="src"
streamlit run src/healthcare_support_agents/streamlit_app.py
```

The Streamlit app now supports three execution modes from the sidebar:

- `Local deterministic` (always available, local data-driven flow)
- `Live watsonx tools` (requires `WATSONX_APIKEY`, `WATSONX_PROJECT_ID`, `WATSONX_URL`, `WATSONX_MODEL`)
- `Orchestrate REST API` (requires `ORCHESTRATE_API_ENDPOINT` or `ORCHESTRATE_INSTANCE_URL`, plus auth via `ORCHESTRATE_BEARER_TOKEN` or `ORCHESTRATE_API_KEY` for IAM flows)

For `ORCHESTRATE_AUTH_TYPE=mcsp`, set a valid `ORCHESTRATE_BEARER_TOKEN` before using API mode.

## Tests

```powershell
$env:PYTHONPATH="src"
python -m unittest tests.test_app tests.test_watsonx_integration tests.test_orchestrate_deployment
```

## Sources

- [Getting started with ADK](https://developer.watson-orchestrate.ibm.com/_releases/1.15.0/getting_started/installing)
- [Managing agents (CLI import/list/chat)](https://developer.watson-orchestrate.ibm.com/agents/manage_agent)
- [Tool import with Python tools](https://developer.watson-orchestrate.ibm.com/connections/using_connections)
- [Chat with Orchestrate assistant API](https://developer.watson-orchestrate.ibm.com/apis/orchestrate-agent/chat-with-orchestrate-assistant)
- [Run events API](https://developer.watson-orchestrate.ibm.com/apis/orchestrate-agent/get-orchestrate-assistant-run-events)
