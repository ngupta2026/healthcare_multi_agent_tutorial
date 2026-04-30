# orchestrate_commands.ps1
# ---------------------------------------------------------------------------
# Reference sheet of all IBM watsonx Orchestrate CLI commands used in this project.
# Run individual sections as needed. Assumes the venv is activated:
#   & .\src\.venv\Scripts\Activate.ps1
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# ENVIRONMENT MANAGEMENT
# ---------------------------------------------------------------------------

# List all configured environments
orchestrate env list

# Add a new environment (replace <instance_url> with your ORCHESTRATE_INSTANCE_URL)
orchestrate env add -n healthcare-iam -u <instance_url> -t ibm_iam
orchestrate env add -n healthcare-mcsp -u <instance_url> -t mcsp

# Activate an environment (required before deploying or chatting)
orchestrate env activate healthcare-iam

# ---------------------------------------------------------------------------
# TOOL REGISTRATION
# ---------------------------------------------------------------------------

# Import Python tools into Orchestrate (with package root and requirements)
orchestrate tools import `
    --kind python `
    --file src/healthcare_support_agents/orchestrate_adk_tools.py `
    --package-root src `
    --requirements-file requirements-orchestrate.txt

# List registered tools (verbose)
orchestrate tools list -v

# ---------------------------------------------------------------------------
# AGENT DEPLOYMENT
# ---------------------------------------------------------------------------
# Deploy specialist agents FIRST, then the coordinator last.

# 1. Patient Context Agent
orchestrate agents import -f deploy/orchestrate/patient_context_agent.agent.yaml

# 2. Discharge Translator Agent
orchestrate agents import -f deploy/orchestrate/discharge_translator_agent.agent.yaml

# 3. Recovery Monitoring Agent
orchestrate agents import -f deploy/orchestrate/recovery_monitoring_agent.agent.yaml

# 4. Care Logistics Agent
orchestrate agents import -f deploy/orchestrate/care_logistics_agent.agent.yaml

# 5. AI Healthcare Coordinator (supervisor — deploy LAST)
orchestrate agents import -f deploy/orchestrate/ai_healthcare_coordinator.agent.yaml

# List deployed agents (verbose)
orchestrate agents list -v

# ---------------------------------------------------------------------------
# FULL DEPLOYMENT (shortcut — runs all of the above via the deploy script)
# ---------------------------------------------------------------------------
powershell -ExecutionPolicy Bypass -File .\deploy_all_agents.ps1
powershell -ExecutionPolicy Bypass -File .\deploy_all_agents.ps1 -SkipToolImport

# ---------------------------------------------------------------------------
# CHAT WITH AGENT
# ---------------------------------------------------------------------------

# Basic chat
orchestrate chat ask --agent-name AI_Healthcare_Coordinator "Summarize PT-1001 recovery status and call out any escalation."

# Chat with reasoning trace
orchestrate chat ask --agent-name AI_Healthcare_Coordinator --include-reasoning "Summarize PT-1001 recovery status and call out any escalation."

# Chat about architecture
orchestrate chat ask --agent-name AI_Healthcare_Coordinator "summarize multi-agent architecture and its tools"

# Custom patient query
orchestrate chat ask --agent-name AI_Healthcare_Coordinator "What is the discharge checklist for patient PT-1002?"

# ---------------------------------------------------------------------------
# INVOKE VIA PYTHON HELPER (uses orchestrate_deployment.py)
# ---------------------------------------------------------------------------

# Register tools via Python helper
python -m healthcare_support_agents.orchestrate_deployment register-tools

# Wire agent spec via Python helper
python -m healthcare_support_agents.orchestrate_deployment wire-agent

# Invoke agent via CLI through Python helper
python -m healthcare_support_agents.orchestrate_deployment invoke --mode cli --prompt "Summarize PT-1001 recovery status"

# Invoke agent via REST API through Python helper
python -m healthcare_support_agents.orchestrate_deployment invoke --mode api --prompt "Summarize PT-1001 recovery status"
