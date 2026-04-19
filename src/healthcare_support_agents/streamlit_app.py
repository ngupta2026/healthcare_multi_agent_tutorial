from __future__ import annotations

import os
from dataclasses import asdict, replace
from typing import Any

import streamlit as st

try:
    from .config import AppConfig
    from .orchestrate_deployment import invoke_via_api
    from .orchestrator import Orchestrator
    from .repository import DataRepository
    from .serper_client import SerperSearchClient
    from .tool_wrappers import HealthcareToolRuntime, build_repository
    from .watsonx_client import WatsonxClient
    from .watsonx_orchestrator import WatsonxCareOrchestrator
except ImportError:
    from healthcare_support_agents.config import AppConfig
    from healthcare_support_agents.orchestrate_deployment import invoke_via_api
    from healthcare_support_agents.orchestrator import Orchestrator
    from healthcare_support_agents.repository import DataRepository
    from healthcare_support_agents.serper_client import SerperSearchClient
    from healthcare_support_agents.tool_wrappers import HealthcareToolRuntime, build_repository
    from healthcare_support_agents.watsonx_client import WatsonxClient
    from healthcare_support_agents.watsonx_orchestrator import WatsonxCareOrchestrator


def load_orchestrator() -> tuple[Orchestrator, DataRepository]:
    repository = build_repository()
    return Orchestrator(repository), repository


def load_watsonx_orchestrator(repository: DataRepository) -> WatsonxCareOrchestrator | None:
    config = AppConfig.from_env()
    if not config.watsonx_ready:
        return None
    search_client = SerperSearchClient(config.serper_api_key) if config.serper_ready else None
    runtime = HealthcareToolRuntime(repository, search_client=search_client)
    return WatsonxCareOrchestrator(WatsonxClient(config), runtime)


def severity_label(risk_status: str) -> tuple[str, str]:
    if risk_status == "high_priority_health_shift":
        return "High Priority", "error"
    if risk_status == "watch_closely":
        return "Watch Closely", "warning"
    return "Routine", "success"


def build_orchestrate_prompt(patient_id: str, symptom_report: str) -> str:
    return (
        f'Use patient_id "{patient_id}" and symptom_report "{symptom_report}". '
        "Call resolve_recovery_case first. "
        "Return only tool-grounded output with risk_status, escalation_required, key_findings, and immediate_actions. "
        "Do not ask follow-up questions."
    )


def extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if "text" in item and isinstance(item["text"], str):
                    parts.append(item["text"].strip())
                elif "content" in item:
                    nested = extract_text(item["content"])
                    if nested:
                        parts.append(nested)
        return "\n".join(part for part in parts if part).strip()
    if isinstance(content, dict):
        if "text" in content and isinstance(content["text"], str):
            return content["text"].strip()
        if "content" in content:
            return extract_text(content["content"])
    return ""


def parse_orchestrate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    run = payload.get("run", {})
    events = payload.get("events", [])
    assistant_messages: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for event in events:
        if not isinstance(event, dict):
            continue
        data = event.get("data", event)
        if not isinstance(data, dict):
            continue
        message = data.get("message")
        if not isinstance(message, dict):
            continue

        if message.get("role") == "assistant":
            text = extract_text(message.get("content"))
            if text:
                assistant_messages.append(text)

        event_tool_calls = message.get("tool_calls", [])
        if isinstance(event_tool_calls, list):
            tool_calls.extend(call for call in event_tool_calls if isinstance(call, dict))

    return {
        "run_id": run.get("id", "n/a"),
        "run_status": run.get("status") or run.get("state") or "unknown",
        "assistant_messages": assistant_messages,
        "tool_calls": tool_calls,
        "final_message": assistant_messages[-1] if assistant_messages else "",
        "events": events,
    }


def main() -> None:
    st.set_page_config(
        page_title="Healthcare Multi-Agent Coordinator",
        page_icon=":hospital:",
        layout="wide",
    )

    orchestrator, repository = load_orchestrator()
    watsonx_orchestrator = load_watsonx_orchestrator(repository)
    config = AppConfig.from_env()
    patients = repository.patients
    patient_options = {
        f"{record['patient_id']} - {record['name']}": record["patient_id"] for record in patients.values()
    }

    st.title("Healthcare Multi-Agent Care Coordinator")
    st.caption("Interactive demo for discharge translation, symptom monitoring, logistics coordination, and nurse escalation.")

    with st.sidebar:
        st.subheader("Case Setup")
        selected_label = st.selectbox("Patient", list(patient_options.keys()))
        patient_id = patient_options[selected_label]
        patient = patients[patient_id]
        discharge_plan = repository.discharge_plans[patient_id]
        default_report = {
            "PT-1001": "A little tired after walking, but no fever and breathing is normal.",
            "PT-2002": "My leg is more swollen and I missed my antibiotic pickup.",
        }.get(patient_id, "")

        symptom_report = st.text_area(
            "Symptom report",
            value=default_report,
            height=140,
            help="Describe what the patient or caregiver is reporting today.",
        )
        execution_mode = st.radio(
            "Execution mode",
            options=["Local deterministic", "Live watsonx tools", "Orchestrate REST API"],
            index=0,
            help="Switch between local deterministic orchestration, direct watsonx tool-calling, or hosted Orchestrate API runs.",
        )
        token_key = "orchestrate_bearer_token"
        default_token = st.session_state.get(token_key, config.orchestrate_bearer_token or "")
        bearer_token = st.text_input(
            "Orchestrate bearer token (optional override)",
            value=default_token,
            type="password",
            help="Used for Orchestrate REST API mode. Stored in this Streamlit session only.",
        )
        if bearer_token:
            st.session_state[token_key] = bearer_token
            os.environ["ORCHESTRATE_BEARER_TOKEN"] = bearer_token
        submitted = st.button("Run care coordination", type="primary", use_container_width=True)

        st.markdown("### Engine Readiness")
        st.write(f"watsonx tools: {'Ready' if watsonx_orchestrator else 'Missing config'}")
        st.write(f"Orchestrate API: {'Ready' if config.orchestrate_endpoint else 'Missing endpoint'}")

        st.markdown("### Patient Snapshot")
        st.write(f"Literacy level: `{patient['literacy_level']}`")
        st.write(discharge_plan["summary"])
        if watsonx_orchestrator is None:
            missing = ", ".join(config.missing_watsonx_fields())
            st.caption(f"watsonx disabled until these values exist in .env: {missing}")
        if not config.orchestrate_endpoint:
            st.caption("Orchestrate API disabled until ORCHESTRATE_API_ENDPOINT (or ORCHESTRATE_INSTANCE_URL) is set.")
        st.caption("Quick PowerShell launcher for Orchestrate mode:")
        st.code(r".\run_streamlit_orchestrate.local.ps1", language="powershell")

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@500;700;800&display=swap');
        html, body, [class*="css"] {
            font-family: 'Manrope', sans-serif;
        }
        .result-card {
            border: 1px solid rgba(49, 51, 63, 0.2);
            border-radius: 18px;
            padding: 1.1rem 1.2rem;
            background: linear-gradient(140deg, rgba(233,245,255,1) 0%, rgba(255,252,244,1) 100%);
        }
        .small-label {
            font-size: 0.8rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: #5b6472;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if not submitted:
        st.info("Choose a patient, adjust the symptom report if needed, and run the workflow to see the agents coordinate.")
        return

    result = orchestrator.resolve(patient_id, symptom_report)
    selected_summary = result.recovery_summary
    selected_mode_badge = execution_mode
    watsonx_result = None
    watsonx_error = None
    orchestrate_payload = None
    orchestrate_error = None

    if execution_mode == "Live watsonx tools":
        if watsonx_orchestrator is None:
            watsonx_error = "watsonx credentials are missing. Add WATSONX_APIKEY and WATSONX_PROJECT_ID to .env."
        else:
            try:
                watsonx_result = watsonx_orchestrator.resolve(patient_id, symptom_report)
                selected_summary = watsonx_result.final_response
            except Exception as exc:  # pragma: no cover
                watsonx_error = str(exc)

    if execution_mode == "Orchestrate REST API":
        prompt = build_orchestrate_prompt(patient_id, symptom_report)
        try:
            run_config = replace(
                config,
                orchestrate_bearer_token=st.session_state.get("orchestrate_bearer_token", config.orchestrate_bearer_token),
            )
            orchestrate_payload = invoke_via_api(run_config, prompt, poll_timeout_sec=90)
            parsed = parse_orchestrate_payload(orchestrate_payload)
            if parsed["final_message"]:
                selected_summary = parsed["final_message"]
        except Exception as exc:  # pragma: no cover
            orchestrate_error = str(exc)

    monitoring_output = result.details["monitoring_output"]
    logistics_output = result.details["logistics_output"]
    translator_output = result.details["translator_output"]
    risk_label, risk_style = severity_label(result.risk_status)

    overview_col, status_col, actions_col = st.columns((1.5, 1, 1))

    with overview_col:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown('<div class="small-label">Recovery Summary</div>', unsafe_allow_html=True)
        st.subheader(selected_summary)
        st.caption(f"Execution mode: {selected_mode_badge}")
        st.write(f"Patient ID: `{result.patient_id}`")
        st.write(f"Symptom report: {symptom_report}")
        st.markdown("</div>", unsafe_allow_html=True)

    with status_col:
        st.metric("Risk Status", risk_label)
        if risk_style == "error":
            st.error("Human nurse escalation required.")
        elif risk_style == "warning":
            st.warning("Close observation recommended today.")
        else:
            st.success("Recovery appears stable.")

    with actions_col:
        st.metric("Checklist Items", len(result.checklist))
        st.metric("Coordination Actions", len(result.coordination_status))
        st.metric("Escalated", "Yes" if result.escalated else "No")

    checklist_col, logistics_col = st.columns(2)

    with checklist_col:
        st.subheader("Daily Checklist")
        for item in translator_output["checklist"]:
            st.checkbox(item, value=False, disabled=True)

        st.subheader("Medication Schedule")
        for item in translator_output["medication_schedule"]:
            st.write(f"- {item}")

        st.subheader("Safety Tips")
        for item in translator_output["safety_tips"]:
            st.write(f"- {item}")

    with logistics_col:
        st.subheader("Risk Review")
        st.write(f"Triage level: `{monitoring_output['triage_level']}`")
        st.write(monitoring_output["recommended_action"])
        if monitoring_output["concerning_signals"]:
            for signal in monitoring_output["concerning_signals"]:
                st.write(f"- {signal}")
        else:
            st.write("- No concerning signals were detected.")

        st.subheader("Logistics Coordination")
        for line in logistics_output["medication_status"]:
            st.write(f"- {line}")
        for line in logistics_output["appointment_status"]:
            st.write(f"- {line}")
        for line in logistics_output["resolved_actions"]:
            st.write(f"- {line}")

    with st.expander("Transparent reasoning log", expanded=True):
        for entry in result.reasoning_log:
            st.write(f"- {entry}")

    if watsonx_result:
        with st.expander("watsonx Tool Trace", expanded=True):
            for call in watsonx_result.tool_trace:
                st.write(f"Tool: `{call['tool_name']}`")
                st.json(call)
    elif watsonx_error:
        st.error(f"watsonx invocation failed: {watsonx_error}")

    if orchestrate_payload:
        parsed = parse_orchestrate_payload(orchestrate_payload)
        with st.expander("Orchestrate Run Trace", expanded=True):
            st.write(f"Run ID: `{parsed['run_id']}`")
            st.write(f"Run status: `{parsed['run_status']}`")
            if parsed["assistant_messages"]:
                st.write("Assistant responses:")
                for message in parsed["assistant_messages"]:
                    st.write(f"- {message}")
            if parsed["tool_calls"]:
                st.write("Tool calls:")
                for call in parsed["tool_calls"]:
                    st.json(call)
            st.write("Raw payload:")
            st.json(orchestrate_payload)
    elif orchestrate_error:
        st.error(f"Orchestrate API invocation failed: {orchestrate_error}")

    with st.expander("Structured payload"):
        st.json(asdict(result))


if __name__ == "__main__":
    main()
