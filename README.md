# healthcare_multi_agent_tutorial

Standalone GitHub-ready multi-agent healthcare recovery support app inspired by IBM watsonx Orchestrate tutorial patterns.

This project simulates a healthcare care-coordination workflow by coordinating four specialized agents:

- `orchestrator_agent`
- `translator_agent`
- `monitoring_agent`
- `logistics_agent`

## Features

- Converts dense discharge instructions into plain-language recovery checklists.
- Monitors symptom reports and biometrics for escalating health risks.
- Detects medication pickup delays and suggests alternate pharmacies.
- Confirms follow-up appointments and transportation readiness.
- Produces a transparent reasoning log for every care-coordination decision.
- Escalates high-priority health shifts to a human nurse.

## Repo layout

```text
healthcare_multi_agent_tutorial/
|-- agents.yaml
|-- tasks.yaml
|-- pyproject.toml
|-- README.md
|-- data/
|   |-- appointments.json
|   |-- discharge_plans.json
|   |-- patients.json
|   |-- pharmacy_status.json
|   `-- vitals.json
|-- src/
|   `-- healthcare_support_agents/
|       |-- __init__.py
|       |-- agents.py
|       |-- app.py
|       |-- connectors.py
|       |-- models.py
|       |-- orchestrator.py
|       `-- repository.py
`-- tests/
    `-- test_app.py
```

## Run locally

```powershell
$env:PYTHONPATH="src"
python -m healthcare_support_agents.app
```

## Example scenarios

- `PT-1001` with symptom report `A little tired after walking, but no fever and breathing is normal.`
- `PT-1001` with symptom report `I feel short of breath and dizzy this morning.`
- `PT-2002` with symptom report `My leg is more swollen and I missed my antibiotic pickup.`

## Behavior rules

- High-risk symptom and vital combinations escalate to a human nurse.
- Mild recovery symptoms remain in routine monitoring with clear self-care guidance.
- Medication delays are flagged with alternate fill options when available.
- Appointment coordination includes transportation status and next steps.
- Final output always includes checklist, logistics status, and reasoning log.

## IBM watsonx alignment

This repo is designed so the functions in [src/healthcare_support_agents/agents.py](src/healthcare_support_agents/agents.py)
can be adapted into IBM watsonx Orchestrate tools.

The included `agents.yaml` and `tasks.yaml` mirror the healthcare care workflow you described so you can reuse them in a watsonx agent configuration.
