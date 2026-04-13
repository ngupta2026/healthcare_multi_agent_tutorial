from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import streamlit as st

try:
    from .config import AppConfig
    from .orchestrator import Orchestrator
    from .repository import DataRepository
    from .serper_client import SerperSearchClient
    from .tool_wrappers import HealthcareToolRuntime
    from .watsonx_client import WatsonxClient
    from .watsonx_orchestrator import WatsonxCareOrchestrator
except ImportError:
    from healthcare_support_agents.config import AppConfig
    from healthcare_support_agents.orchestrator import Orchestrator
    from healthcare_support_agents.repository import DataRepository
    from healthcare_support_agents.serper_client import SerperSearchClient
    from healthcare_support_agents.tool_wrappers import HealthcareToolRuntime
    from healthcare_support_agents.watsonx_client import WatsonxClient
    from healthcare_support_agents.watsonx_orchestrator import WatsonxCareOrchestrator


def load_orchestrator() -> tuple[Orchestrator, DataRepository]:
    repo_root = Path(__file__).resolve().parents[2]
    repository = DataRepository(repo_root / "data")
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
        use_watsonx = st.toggle(
            "Use live watsonx tool calling",
            value=bool(watsonx_orchestrator),
            disabled=watsonx_orchestrator is None,
            help="Requires a configured .env file with watsonx credentials.",
        )
        submitted = st.button("Run care coordination", type="primary", use_container_width=True)

        st.markdown("### Patient Snapshot")
        st.write(f"Literacy level: `{patient['literacy_level']}`")
        st.write(discharge_plan["summary"])
        if watsonx_orchestrator is None:
            missing = ", ".join(config.missing_watsonx_fields())
            st.caption(f"watsonx disabled until these values exist in .env: {missing}")

    st.markdown(
        """
        <style>
        .result-card {
            border: 1px solid rgba(49, 51, 63, 0.2);
            border-radius: 18px;
            padding: 1.1rem 1.2rem;
            background: linear-gradient(180deg, rgba(245,248,252,1) 0%, rgba(255,255,255,1) 100%);
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
    watsonx_result = None
    watsonx_error = None
    if use_watsonx and watsonx_orchestrator is not None:
        try:
            watsonx_result = watsonx_orchestrator.resolve(patient_id, symptom_report)
        except Exception as exc:  # pragma: no cover
            watsonx_error = str(exc)

    monitoring_output = result.details["monitoring_output"]
    logistics_output = result.details["logistics_output"]
    translator_output = result.details["translator_output"]
    risk_label, risk_style = severity_label(result.risk_status)

    overview_col, status_col, actions_col = st.columns((1.5, 1, 1))

    with overview_col:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown('<div class="small-label">Recovery Summary</div>', unsafe_allow_html=True)
        st.subheader(watsonx_result.final_response if watsonx_result else result.recovery_summary)
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
        st.warning(f"watsonx live orchestration failed, so the app fell back to the local workflow: {watsonx_error}")

    with st.expander("Structured payload"):
        st.json(asdict(result))


if __name__ == "__main__":
    main()
