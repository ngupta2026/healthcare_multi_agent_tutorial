from __future__ import annotations

import os
import re
import time
from html import escape
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import streamlit as st

try:
    import yaml
except ImportError:  # pragma: no cover - dependency managed via requirements
    yaml = None

try:
    from .config import AppConfig
    from .orchestrate_deployment import invoke_via_api
    from .orchestrator import Orchestrator
    from .repository import DataRepository
    from .serper_client import SerperSearchClient
    from .tool_wrappers import HealthcareToolRuntime, build_repository, tool_message
    from .watsonx_client import WatsonxClient
    from .watsonx_orchestrator import WatsonxCareOrchestrator
except ImportError:
    from healthcare_support_agents.config import AppConfig
    from healthcare_support_agents.orchestrate_deployment import invoke_via_api
    from healthcare_support_agents.orchestrator import Orchestrator
    from healthcare_support_agents.repository import DataRepository
    from healthcare_support_agents.serper_client import SerperSearchClient
    from healthcare_support_agents.tool_wrappers import HealthcareToolRuntime, build_repository, tool_message
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


_SEVERITY_PHRASE_PATTERN = re.compile(
    r"\\b(immediate attention required|high severity|escalation|escalate|urgent|critical|high priority)\\b",
    re.IGNORECASE,
)


def _highlight_severity_phrases(text: str) -> str:
    raw = str(text or "")
    if not raw:
        return ""

    fragments: list[str] = []
    cursor = 0
    for match in _SEVERITY_PHRASE_PATTERN.finditer(raw):
        fragments.append(escape(raw[cursor:match.start()]))
        fragments.append(f'<span class="severity-highlight">{escape(match.group(0))}</span>')
        cursor = match.end()
    fragments.append(escape(raw[cursor:]))
    return "".join(fragments)


def _write_highlighted_line(text: str, bullet: bool = False) -> None:
    rendered = _highlight_severity_phrases(text)
    if bullet:
        st.markdown(f'<div class="severity-line">&bull; {rendered}</div>', unsafe_allow_html=True)
        return
    st.markdown(rendered, unsafe_allow_html=True)


def build_orchestrate_prompt(patient_id: str, symptom_report: str) -> str:
    return (
        f'Use patient_id "{patient_id}" and symptom_report "{symptom_report}". '
        "Call resolve_recovery_case first. "
        "Return ONLY a concise tool-grounded summary with four lines exactly: "
        "risk_status, escalation_required, key_findings, immediate_actions. "
        "Do not include code blocks, pseudocode, or implementation examples. "
        "Do not ask follow-up questions."
    )


def clean_orchestrate_summary(text: str) -> str:
    cleaned = re.sub(r"```[\s\S]*?```", "", text or "")
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if not lines:
        return ""

    ignored_prefixes = (
        "### recovery case resolution",
        "#### patient id:",
        "#### symptom report:",
        "output",
        "print(",
        "def ",
        "patient_id =",
        "symptom_report =",
    )
    filtered = [
        line for line in lines
        if not line.lower().startswith(ignored_prefixes)
    ]
    summary = "\n".join(filtered).strip()
    if not summary:
        summary = "\n".join(lines[:6]).strip()
    return summary


def _default_ui_content_payload() -> dict[str, Any]:
    return {
        "sample_prompts": {
            "Nurse": [
                "Which patients need nurse escalation today?",
                "Summarize PT-1002's recovery status and flag urgent concerns.",
                "What symptoms is PT-1001 reporting and what action is recommended?",
                "Are there any patients with high-priority health shifts?",
            ],
            "Physician": [
                "Review the discharge plan for PT-1001 and highlight medication risks.",
                "What vitals are concerning for PT-2002?",
                "Give me a clinical summary of PT-1003's post-op recovery.",
                "Are there any patients with oxygen saturation below 92%?",
            ],
            "Patient / Caregiver": [
                "What should I do today after my discharge from hospital?",
                "When do I take my medications and what are the doses?",
                "What warning signs should I watch for, and when should I call for help?",
                "How do I use my walker safely?",
            ],
            "Care Coordinator": [
                "Are all prescriptions filled for PT-2002?",
                "Check appointment and transport status for PT-1001.",
                "Which patients have unresolved logistics barriers?",
                "Find alternative pharmacies near PT-2002 for prescription pickup.",
            ],
        },
        "symptom_reports": {
            "PT-1001": "A little tired after walking, but no fever and breathing is normal.",
            "PT-2002": "My leg is more swollen and I missed my antibiotic pickup.",
            "PT-1002": "I feel short of breath and dizzy this morning.",
            "PT-1003": "My knee wound looks clean and the pain is manageable, but I feel a little stiff when walking.",
        },
        "discharge_plan_suggestions": {
            "Heart failure follow-up": {
                "summary": "Discharged after heart failure stabilization with medication adjustment and home monitoring.",
                "clinical_note": "Take diuretic in the morning. Track daily weight before breakfast. Call care team if weight rises by >2 lb in 24h, breathing worsens, or chest pain occurs.",
                "daily_routine": "Wake at 7:00 AM, morning meds after breakfast, light walk in afternoon, rest by 9:30 PM.",
            },
            "Post-infection recovery": {
                "summary": "Discharged after infection treatment with oral antibiotic continuation and symptom watch.",
                "clinical_note": "Take antibiotic as prescribed until complete. Monitor fever, swelling, or spreading redness. Seek same-day review for worsening symptoms.",
                "daily_routine": "Hydration reminders every 3-4 hours, medications with meals, gentle movement, and evening symptom check.",
            },
            "Post-op mobility": {
                "summary": "Discharged after procedure with pain control plan and mobility safety precautions.",
                "clinical_note": "Use assistive device for transfers, keep wound clean and dry, and report severe pain, drainage, fever, or dizziness immediately.",
                "daily_routine": "Morning wound check, timed pain medications, short supervised walks, and early bedtime for recovery.",
            },
        },
    }


_INITIAL_UI_CONTENT = _default_ui_content_payload()
SAMPLE_PROMPTS: dict[str, list[str]] = {
    role: list(prompts) for role, prompts in _INITIAL_UI_CONTENT["sample_prompts"].items()
}
_DEFAULT_SYMPTOM_REPORTS: dict[str, str] = dict(_INITIAL_UI_CONTENT["symptom_reports"])
_DISCHARGE_PLAN_SUGGESTIONS: dict[str, dict[str, str]] = {
    name: dict(content) for name, content in _INITIAL_UI_CONTENT["discharge_plan_suggestions"].items()
}


_SESSION_ADDED_PATIENTS_KEY = "added_patients_records"
_ADDED_PATIENTS_YAML = "added_patients.yaml"
_UI_CONTENT_YAML = "ui_content.yaml"
_SHOW_MANAGE_PATIENTS_DIALOG_KEY = "show_manage_patients_dialog"
_MANAGE_PATIENT_EDIT_ID_KEY = "manage_patient_edit_id"


def _apply_ui_content_payload(payload: dict[str, Any] | None = None) -> None:
    global SAMPLE_PROMPTS, _DEFAULT_SYMPTOM_REPORTS, _DISCHARGE_PLAN_SUGGESTIONS

    defaults = _default_ui_content_payload()
    source = payload if isinstance(payload, dict) else {}

    raw_prompts = source.get("sample_prompts", defaults["sample_prompts"])
    sample_prompts: dict[str, list[str]] = {}
    if isinstance(raw_prompts, dict):
        for role, prompts in raw_prompts.items():
            if isinstance(prompts, list):
                cleaned_prompts = [str(prompt) for prompt in prompts if str(prompt).strip()]
                if cleaned_prompts:
                    sample_prompts[str(role)] = cleaned_prompts
    if not sample_prompts:
        sample_prompts = {
            role: list(prompts) for role, prompts in defaults["sample_prompts"].items()
        }

    raw_symptoms = source.get("symptom_reports", defaults["symptom_reports"])
    symptom_reports = dict(defaults["symptom_reports"])
    if isinstance(raw_symptoms, dict):
        symptom_reports.update({
            str(patient_id): str(report)
            for patient_id, report in raw_symptoms.items()
            if str(patient_id).strip()
        })

    raw_suggestions = source.get("discharge_plan_suggestions", defaults["discharge_plan_suggestions"])
    discharge_plan_suggestions: dict[str, dict[str, str]] = {}
    if isinstance(raw_suggestions, dict):
        for suggestion_name, suggestion_values in raw_suggestions.items():
            if not isinstance(suggestion_values, dict):
                continue
            discharge_plan_suggestions[str(suggestion_name)] = {
                "summary": str(suggestion_values.get("summary", "")),
                "clinical_note": str(suggestion_values.get("clinical_note", "")),
                "daily_routine": str(suggestion_values.get("daily_routine", "")),
            }
    if not discharge_plan_suggestions:
        discharge_plan_suggestions = {
            name: dict(content) for name, content in defaults["discharge_plan_suggestions"].items()
        }

    SAMPLE_PROMPTS = sample_prompts
    _DEFAULT_SYMPTOM_REPORTS = symptom_reports
    _DISCHARGE_PLAN_SUGGESTIONS = discharge_plan_suggestions


def _build_ui_content_payload() -> dict[str, Any]:
    return {
        "sample_prompts": SAMPLE_PROMPTS,
        "symptom_reports": _DEFAULT_SYMPTOM_REPORTS,
        "discharge_plan_suggestions": _DISCHARGE_PLAN_SUGGESTIONS,
    }


def _load_ui_content_from_yaml(repository: "DataRepository") -> None:
    _apply_ui_content_payload()
    if yaml is None:
        return
    yaml_path = Path(repository.data_dir) / _UI_CONTENT_YAML
    if not yaml_path.exists():
        _persist_ui_content_to_yaml(repository)
        return
    try:
        with yaml_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
    except Exception:
        return
    if isinstance(payload, dict):
        _apply_ui_content_payload(payload)


def _persist_ui_content_to_yaml(repository: "DataRepository") -> None:
    if yaml is None:
        raise RuntimeError("PyYAML is not installed. Please install pyyaml to persist UI content.")
    yaml_path = Path(repository.data_dir) / _UI_CONTENT_YAML
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with yaml_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(_build_ui_content_payload(), handle, sort_keys=False, allow_unicode=True)


def _persist_patient_state(repository: "DataRepository") -> None:
    _persist_added_patients_to_yaml(repository)
    _persist_ui_content_to_yaml(repository)


def _empty_added_patient_payload() -> dict[str, dict[str, dict[str, Any]]]:
    return {
        "patients": {},
        "discharge_plans": {},
        "vitals": {},
        "pharmacy_status": {},
        "appointments": {},
        "symptom_reports": {},
        "deleted_patients": [],
    }


def _get_added_patient_store() -> dict[str, dict[str, dict[str, Any]]]:
    if _SESSION_ADDED_PATIENTS_KEY not in st.session_state:
        st.session_state[_SESSION_ADDED_PATIENTS_KEY] = _empty_added_patient_payload()
    return st.session_state[_SESSION_ADDED_PATIENTS_KEY]


def _merge_added_patient_payload(payload: dict[str, Any]) -> None:
    store = _get_added_patient_store()
    for section in ("patients", "discharge_plans", "vitals", "pharmacy_status", "appointments", "symptom_reports"):
        section_data = payload.get(section, {}) if isinstance(payload, dict) else {}
        if isinstance(section_data, dict):
            store[section].update(section_data)
    deleted_patients = payload.get("deleted_patients", []) if isinstance(payload, dict) else []
    if isinstance(deleted_patients, list):
        merged_deleted = [str(pid) for pid in store.get("deleted_patients", [])]
        for pid in deleted_patients:
            pid_str = str(pid)
            if pid_str not in merged_deleted:
                merged_deleted.append(pid_str)
        store["deleted_patients"] = merged_deleted


def _load_added_patients_from_yaml(repository: "DataRepository") -> None:
    if yaml is None:
        return
    yaml_path = Path(repository.data_dir) / _ADDED_PATIENTS_YAML
    if not yaml_path.exists():
        return
    try:
        with yaml_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
    except Exception:
        return
    if isinstance(payload, dict):
        _merge_added_patient_payload(payload)


def _persist_added_patients_to_yaml(repository: "DataRepository") -> None:
    if yaml is None:
        raise RuntimeError("PyYAML is not installed. Please install pyyaml to persist added patients.")
    yaml_path = Path(repository.data_dir) / _ADDED_PATIENTS_YAML
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _get_added_patient_store()
    with yaml_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _apply_added_patient_store(repository: "DataRepository") -> None:
    store = _get_added_patient_store()
    deleted_patients = {str(pid) for pid in store.get("deleted_patients", [])}
    for patient_id in deleted_patients:
        repository.patients.pop(patient_id, None)
        repository.discharge_plans.pop(patient_id, None)
        repository.vitals.pop(patient_id, None)
        repository.pharmacy_status.pop(patient_id, None)
        repository.appointments.pop(patient_id, None)
        _DEFAULT_SYMPTOM_REPORTS.pop(patient_id, None)
    repository.patients.update(store["patients"])
    repository.discharge_plans.update(store["discharge_plans"])
    repository.vitals.update(store["vitals"])
    repository.pharmacy_status.update(store["pharmacy_status"])
    repository.appointments.update(store["appointments"])
    for pid, symptom in store["symptom_reports"].items():
        _DEFAULT_SYMPTOM_REPORTS[str(pid)] = str(symptom)


def _save_new_patient_record(
    patient_id: str,
    patient_record: dict[str, Any],
    discharge_record: dict[str, Any],
    vitals_record: dict[str, Any],
    pharmacy_record: dict[str, Any],
    appointment_record: dict[str, Any],
    symptom_report: str,
) -> None:
    store = _get_added_patient_store()
    store["deleted_patients"] = [pid for pid in store.get("deleted_patients", []) if pid != patient_id]
    store["patients"][patient_id] = patient_record
    store["discharge_plans"][patient_id] = discharge_record
    store["vitals"][patient_id] = vitals_record
    store["pharmacy_status"][patient_id] = pharmacy_record
    store["appointments"][patient_id] = appointment_record
    store["symptom_reports"][patient_id] = symptom_report
    _DEFAULT_SYMPTOM_REPORTS[patient_id] = symptom_report


def _delete_patient_record(patient_id: str) -> None:
    store = _get_added_patient_store()
    for section in ("patients", "discharge_plans", "vitals", "pharmacy_status", "appointments", "symptom_reports"):
        store[section].pop(patient_id, None)
    deleted_patients = store.setdefault("deleted_patients", [])
    if patient_id not in deleted_patients:
        deleted_patients.append(patient_id)
    _DEFAULT_SYMPTOM_REPORTS.pop(patient_id, None)


def _patient_form_defaults(repository: "DataRepository", patient_id: str | None = None) -> dict[str, Any]:
    if patient_id is None:
        return {
            "patient_id": "",
            "patient_name": "",
            "literacy_level": "plain_language",
            "care_plan_id": "",
            "discharge_plan_id": "",
            "pharmacy_plan_id": "",
            "appointment_plan_id": "",
            "discharge_summary": "",
            "clinical_note": "",
            "daily_routine": "",
            "heart_rate": 90,
            "oxygen_saturation": 96,
            "temperature_f": 98.6,
            "systolic_bp": 120,
            "weight_delta_lb": 0.0,
            "medication_name": "",
            "medication_status": "ready_for_pickup",
            "delay_reason": "",
            "alternate_source": "",
            "specialty": "",
            "scheduled_at": "",
            "transportation_status": "confirmed",
            "transportation_note": "",
            "symptom_report": "No specific symptoms reported today.",
            "suggestion_name": next(iter(_DISCHARGE_PLAN_SUGGESTIONS)),
        }

    patient = repository.patients[patient_id]
    discharge_plan = repository.discharge_plans.get(patient_id, {})
    vitals = repository.vitals.get(patient_id, {})
    pharmacy = repository.pharmacy_status.get(patient_id, {})
    medications = pharmacy.get("medications", []) if isinstance(pharmacy, dict) else []
    medication = medications[0] if medications else {}
    appointments = repository.appointments.get(patient_id, {}).get("appointments", [])
    appointment = appointments[0] if appointments else {}
    return {
        "patient_id": patient_id,
        "patient_name": patient.get("name", ""),
        "literacy_level": patient.get("literacy_level", "plain_language"),
        "care_plan_id": patient.get("care_plan_id", ""),
        "discharge_plan_id": patient.get("discharge_plan_id", ""),
        "pharmacy_plan_id": patient.get("pharmacy_plan_id", ""),
        "appointment_plan_id": patient.get("appointment_plan_id", ""),
        "discharge_summary": discharge_plan.get("summary", ""),
        "clinical_note": discharge_plan.get("clinical_note", ""),
        "daily_routine": discharge_plan.get("daily_routine", ""),
        "heart_rate": int(vitals.get("heart_rate", 90)),
        "oxygen_saturation": int(vitals.get("oxygen_saturation", 96)),
        "temperature_f": float(vitals.get("temperature_f", 98.6)),
        "systolic_bp": int(vitals.get("systolic_bp", 120)),
        "weight_delta_lb": float(vitals.get("weight_delta_lb", 0.0)),
        "medication_name": medication.get("name", ""),
        "medication_status": medication.get("status", "ready_for_pickup"),
        "delay_reason": medication.get("delay_reason", ""),
        "alternate_source": medication.get("alternate_source", ""),
        "specialty": appointment.get("specialty", ""),
        "scheduled_at": appointment.get("scheduled_at", ""),
        "transportation_status": appointment.get("transportation_status", "confirmed"),
        "transportation_note": appointment.get("transportation_note", ""),
        "symptom_report": _DEFAULT_SYMPTOM_REPORTS.get(patient_id, "No specific symptoms reported today."),
        "suggestion_name": next(iter(_DISCHARGE_PLAN_SUGGESTIONS)),
    }


def _render_patient_form(
    repository: "DataRepository",
    *,
    mode: str,
    patient_id: str | None = None,
) -> None:
    defaults = _patient_form_defaults(repository, patient_id)
    key_prefix = f"{mode}_{patient_id or 'new'}"
    literacy_options = ["plain_language", "clinical_language"]
    medication_status_options = ["ready_for_pickup", "delayed"]
    transportation_options = ["confirmed", "needs_booking", "not_required"]
    suggestion_options = list(_DISCHARGE_PLAN_SUGGESTIONS.keys())
    suggestion_name_default = defaults.get("suggestion_name", suggestion_options[0])
    suggestion_index = suggestion_options.index(suggestion_name_default) if suggestion_name_default in suggestion_options else 0

    with st.form(f"{key_prefix}_patient_form"):
        st.markdown("### Patient Profile")
        pcol1, pcol2 = st.columns(2)
        with pcol1:
            patient_id_value = st.text_input(
                "Patient ID",
                value=defaults["patient_id"],
                placeholder="PT-3003",
                disabled=mode == "edit",
                key=f"{key_prefix}_patient_id",
            ).strip().upper()
            literacy_level = st.selectbox(
                "Literacy Level",
                options=literacy_options,
                index=literacy_options.index(defaults["literacy_level"]),
                key=f"{key_prefix}_literacy_level",
            )
        with pcol2:
            patient_name = st.text_input(
                "Patient Name",
                value=defaults["patient_name"],
                placeholder="Jane Doe",
                key=f"{key_prefix}_patient_name",
            ).strip()
            care_plan_id = st.text_input(
                "Care Plan ID",
                value=defaults["care_plan_id"],
                placeholder="CP-3003",
                key=f"{key_prefix}_care_plan_id",
            ).strip().upper()

        st.markdown("### Plan IDs")
        idcol1, idcol2 = st.columns(2)
        with idcol1:
            discharge_plan_id = st.text_input(
                "Discharge Plan ID",
                value=defaults["discharge_plan_id"],
                placeholder="DP-3003",
                key=f"{key_prefix}_discharge_plan_id",
            ).strip().upper()
            appointment_plan_id = st.text_input(
                "Appointment Plan ID",
                value=defaults["appointment_plan_id"],
                placeholder="AP-3003",
                key=f"{key_prefix}_appointment_plan_id",
            ).strip().upper()
        with idcol2:
            pharmacy_plan_id = st.text_input(
                "Pharmacy Plan ID",
                value=defaults["pharmacy_plan_id"],
                placeholder="RX-3003",
                key=f"{key_prefix}_pharmacy_plan_id",
            ).strip().upper()

        st.markdown("### Discharge Plan")
        suggestion_name = st.selectbox(
            "Discharge Plan Suggestion",
            options=suggestion_options,
            index=suggestion_index,
            help="Pick a template to guide the discharge summary, clinical note, and daily routine.",
            key=f"{key_prefix}_suggestion_name",
        )
        selected_suggestion = _DISCHARGE_PLAN_SUGGESTIONS[suggestion_name]
        with st.expander("Preview suggestion", expanded=False):
            st.markdown(f"Summary suggestion: {selected_suggestion['summary']}")
            st.markdown(f"Clinical note suggestion: {selected_suggestion['clinical_note']}")
            st.markdown(f"Daily routine suggestion: {selected_suggestion['daily_routine']}")
        dcol1, dcol2 = st.columns(2)
        with dcol1:
            discharge_summary = st.text_area(
                "Discharge Summary",
                value=defaults["discharge_summary"],
                height=70,
                placeholder=selected_suggestion["summary"],
                key=f"{key_prefix}_discharge_summary",
            )
            daily_routine = st.text_area(
                "Daily Routine",
                value=defaults["daily_routine"],
                height=70,
                placeholder=selected_suggestion["daily_routine"],
                key=f"{key_prefix}_daily_routine",
            )
        with dcol2:
            clinical_note = st.text_area(
                "Clinical Note",
                value=defaults["clinical_note"],
                height=100,
                placeholder=selected_suggestion["clinical_note"],
                key=f"{key_prefix}_clinical_note",
            )

        st.markdown("### Vitals")
        vcol1, vcol2 = st.columns(2)
        with vcol1:
            heart_rate = st.number_input(
                "Heart Rate",
                min_value=20,
                max_value=250,
                value=defaults["heart_rate"],
                step=1,
                key=f"{key_prefix}_heart_rate",
            )
            oxygen_saturation = st.number_input(
                "Oxygen Saturation",
                min_value=50,
                max_value=100,
                value=defaults["oxygen_saturation"],
                step=1,
                key=f"{key_prefix}_oxygen_saturation",
            )
            temperature_f = st.number_input(
                "Temperature (F)",
                min_value=90.0,
                max_value=110.0,
                value=defaults["temperature_f"],
                step=0.1,
                key=f"{key_prefix}_temperature_f",
            )
        with vcol2:
            systolic_bp = st.number_input(
                "Systolic BP",
                min_value=70,
                max_value=260,
                value=defaults["systolic_bp"],
                step=1,
                key=f"{key_prefix}_systolic_bp",
            )
            weight_delta_lb = st.number_input(
                "Weight Delta (lb)",
                min_value=-20.0,
                max_value=20.0,
                value=defaults["weight_delta_lb"],
                step=0.1,
                key=f"{key_prefix}_weight_delta_lb",
            )

        st.markdown("### Pharmacy")
        phcol1, phcol2 = st.columns(2)
        with phcol1:
            medication_name = st.text_input(
                "Medication Name",
                value=defaults["medication_name"],
                placeholder="cephalexin 500 mg",
                key=f"{key_prefix}_medication_name",
            ).strip()
            medication_status = st.selectbox(
                "Medication Status",
                options=medication_status_options,
                index=medication_status_options.index(defaults["medication_status"]),
                key=f"{key_prefix}_medication_status",
            )
        with phcol2:
            delay_reason = st.text_area(
                "Delay Reason",
                value=defaults["delay_reason"],
                height=68,
                key=f"{key_prefix}_delay_reason",
            )
            alternate_source = st.text_area(
                "Alternate Source",
                value=defaults["alternate_source"],
                height=68,
                key=f"{key_prefix}_alternate_source",
            )

        st.markdown("### Appointment")
        acol1, acol2 = st.columns(2)
        with acol1:
            specialty = st.text_input(
                "Specialty",
                value=defaults["specialty"],
                placeholder="Cardiology",
                key=f"{key_prefix}_specialty",
            ).strip()
            scheduled_at = st.text_input(
                "Scheduled At (ISO)",
                value=defaults["scheduled_at"],
                placeholder="2026-04-20T10:30:00",
                key=f"{key_prefix}_scheduled_at",
            ).strip()
        with acol2:
            transportation_status = st.selectbox(
                "Transportation Status",
                options=transportation_options,
                index=transportation_options.index(defaults["transportation_status"]),
                key=f"{key_prefix}_transportation_status",
            )
            transportation_note = st.text_area(
                "Transportation Note",
                value=defaults["transportation_note"],
                height=68,
                key=f"{key_prefix}_transportation_note",
            )

        st.markdown("### Symptom Report")
        symptom_report = st.text_area(
            "Default Symptom Report",
            value=defaults["symptom_report"],
            height=68,
            help="Used as default symptom context in chat and workflow helper prompts.",
            key=f"{key_prefix}_symptom_report",
        )

        submitted = st.form_submit_button(
            "Save Patient" if mode == "add" else "Update Patient",
            type="primary",
            use_container_width=True,
        )

    if not submitted:
        return

    resolved_patient_id = defaults["patient_id"] if mode == "edit" else patient_id_value

    missing = [
        field
        for field, value in {
            "Patient ID": resolved_patient_id,
            "Patient Name": patient_name,
            "Care Plan ID": care_plan_id,
            "Discharge Plan ID": discharge_plan_id,
            "Pharmacy Plan ID": pharmacy_plan_id,
            "Appointment Plan ID": appointment_plan_id,
            "Discharge Summary": discharge_summary,
            "Clinical Note": clinical_note,
            "Daily Routine": daily_routine,
            "Medication Name": medication_name,
            "Specialty": specialty,
            "Scheduled At": scheduled_at,
        }.items()
        if not str(value).strip()
    ]
    if missing:
        st.error(f"Please fill required fields: {', '.join(missing)}")
        return

    if mode == "add" and resolved_patient_id in repository.patients:
        st.error(f"Patient ID {resolved_patient_id} already exists.")
        return

    patient_record = {
        "patient_id": resolved_patient_id,
        "name": patient_name,
        "literacy_level": literacy_level,
        "discharge_plan_id": discharge_plan_id,
        "care_plan_id": care_plan_id,
        "pharmacy_plan_id": pharmacy_plan_id,
        "appointment_plan_id": appointment_plan_id,
    }
    discharge_record = {
        "discharge_plan_id": discharge_plan_id,
        "patient_id": resolved_patient_id,
        "summary": discharge_summary,
        "clinical_note": clinical_note,
        "daily_routine": daily_routine,
    }
    vitals_record = {
        "patient_id": resolved_patient_id,
        "heart_rate": int(heart_rate),
        "oxygen_saturation": int(oxygen_saturation),
        "temperature_f": float(temperature_f),
        "systolic_bp": int(systolic_bp),
        "weight_delta_lb": float(weight_delta_lb),
    }
    pharmacy_record = {
        "pharmacy_plan_id": pharmacy_plan_id,
        "patient_id": resolved_patient_id,
        "medications": [
            {
                "name": medication_name,
                "status": medication_status,
                "delay_reason": delay_reason,
                "alternate_source": alternate_source,
            }
        ],
    }
    appointment_record = {
        "appointment_plan_id": appointment_plan_id,
        "patient_id": resolved_patient_id,
        "appointments": [
            {
                "specialty": specialty,
                "scheduled_at": scheduled_at,
                "transportation_status": transportation_status,
                "transportation_note": transportation_note,
            }
        ],
    }

    _save_new_patient_record(
        resolved_patient_id,
        patient_record,
        discharge_record,
        vitals_record,
        pharmacy_record,
        appointment_record,
        symptom_report.strip() or "No specific symptoms reported today.",
    )
    try:
        _persist_patient_state(repository)
    except Exception as exc:
        st.warning(f"Patient saved in current session, but YAML persistence failed: {exc}")
    if mode == "edit":
        st.session_state.pop(_MANAGE_PATIENT_EDIT_ID_KEY, None)
        st.session_state[_SHOW_MANAGE_PATIENTS_DIALOG_KEY] = False
    st.success(
        f"{'Added' if mode == 'add' else 'Updated'} patient {resolved_patient_id} - {patient_name}"
    )
    st.rerun()


def _render_add_patient_form(repository: "DataRepository") -> None:
    _render_patient_form(repository, mode="add")


def _open_add_patient_dialog(repository: "DataRepository") -> None:
    @st.dialog("Add New Patient")
    def _dialog() -> None:
        st.markdown(
            """
            <style>
            div[data-testid="stDialog"] div[role="dialog"] {
                width: min(1240px, 95vw) !important;
                max-width: min(1240px, 95vw) !important;
            }
            div[data-testid="stDialog"] div[role="dialog"] [data-testid="stForm"] {
                max-height: 78vh;
                overflow-y: auto;
                padding-right: 0.4rem;
            }
            div[data-testid="stDialog"] div[role="dialog"] [data-testid="stVerticalBlock"] {
                gap: 0.45rem;
            }
            div[data-testid="stDialog"] div[role="dialog"] h3 {
                margin-top: 0.25rem;
                margin-bottom: 0.15rem;
                font-size: 1rem;
            }
            div[data-testid="stDialog"] div[role="dialog"] [data-testid="stFormSubmitButton"] {
                position: sticky;
                bottom: 0;
                z-index: 3;
                padding-top: 0.35rem;
                background: linear-gradient(to top, #009db0 72%, rgba(0, 157, 176, 0));
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        _render_add_patient_form(repository)

    _dialog()
def _open_manage_patients_dialog(repository: "DataRepository") -> None:
    @st.dialog("Edit/Delete Patient")
    def _dialog() -> None:
        editing_patient_id = st.session_state.get(_MANAGE_PATIENT_EDIT_ID_KEY)
        st.markdown(
            """
            <style>
            div[data-testid="stDialog"] div[role="dialog"] {
                width: min(1240px, 95vw) !important;
                max-width: min(1240px, 95vw) !important;
            }
            div[data-testid="stDialog"] div[role="dialog"] [data-testid="stForm"] {
                max-height: 78vh;
                overflow-y: auto;
                padding-right: 0.4rem;
            }
            div[data-testid="stDialog"] div[role="dialog"] [data-testid="stVerticalBlock"] {
                gap: 0.45rem;
            }
            div[data-testid="stDialog"] div[role="dialog"] h3 {
                margin-top: 0.25rem;
                margin-bottom: 0.15rem;
                font-size: 1rem;
            }
            div[data-testid="stDialog"] div[role="dialog"] [data-testid="stFormSubmitButton"] {
                position: sticky;
                bottom: 0;
                z-index: 3;
                padding-top: 0.35rem;
                background: linear-gradient(to top, #009db0 72%, rgba(0, 157, 176, 0));
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        if editing_patient_id:
            top_col, _ = st.columns([1, 4])
            with top_col:
                if st.button("Back", key="manage_patients_back", use_container_width=True):
                    st.session_state.pop(_MANAGE_PATIENT_EDIT_ID_KEY, None)
                    st.session_state[_SHOW_MANAGE_PATIENTS_DIALOG_KEY] = True
                    st.rerun()
            _render_patient_form(repository, mode="edit", patient_id=editing_patient_id)
            return

        st.caption("Manage patient records across the case setup list.")

        if not repository.patients:
            st.info("No patients available.")
            return

        for pid, record in sorted(repository.patients.items()):
            info_col, edit_col, delete_col = st.columns([4, 1, 1])
            with info_col:
                st.markdown(f"**{pid}**  \\n+{record['name']}")
            with edit_col:
                if st.button("Edit", key=f"edit_patient_{pid}", use_container_width=True):
                    st.session_state[_MANAGE_PATIENT_EDIT_ID_KEY] = pid
                    st.session_state[_SHOW_MANAGE_PATIENTS_DIALOG_KEY] = True
                    st.rerun()
            with delete_col:
                if st.button("Delete", key=f"delete_patient_{pid}", use_container_width=True):
                    _delete_patient_record(pid)
                    try:
                        _persist_patient_state(repository)
                    except Exception as exc:
                        st.warning(f"Patient deleted in current session, but YAML persistence failed: {exc}")
                    st.session_state[_SHOW_MANAGE_PATIENTS_DIALOG_KEY] = True
                    st.success(f"Deleted patient {pid}")
                    st.rerun()

    _dialog()


def _detect_relevant_patients(prompt: str, all_patient_ids: list[str]) -> list[str]:
    """Return patient IDs explicitly mentioned in the prompt, or all if it's a broad question."""
    mentioned = [pid for pid in all_patient_ids if pid.upper() in prompt.upper()]
    return mentioned if mentioned else list(all_patient_ids)


def _build_rag_knowledge(
    prompt: str,
    orchestrator: "Orchestrator",
    repository: "DataRepository",
) -> str:
    """Run the deterministic agentic pipeline for relevant patients and return a structured knowledge block."""
    patient_ids = _detect_relevant_patients(prompt, list(repository.patients.keys()))
    sections: list[str] = []
    for pid in patient_ids:
        symptom = _DEFAULT_SYMPTOM_REPORTS.get(pid, "No specific symptoms reported today.")
        try:
            result = orchestrator.resolve(pid, symptom)
        except Exception as exc:
            sections.append(f"=== PATIENT {pid} ===\nError retrieving data: {exc}\n")
            continue
        patient = repository.patients[pid]
        vitals = repository.vitals.get(pid, {})
        escalation = "YES — escalate to nurse immediately" if result.escalated else "No"
        checklist_preview = "; ".join(result.checklist[:4]) if result.checklist else "None"
        coordination_preview = "; ".join(result.coordination_status[:3]) if result.coordination_status else "None"
        reasoning_preview = "; ".join(result.reasoning_log) if result.reasoning_log else "None"
        sections.append(
            f"=== PATIENT {pid}: {patient['name']} ===\n"
            f"Risk Status: {result.risk_status}\n"
            f"Escalation Required: {escalation}\n"
            f"Recovery Summary: {result.recovery_summary}\n"
            f"Latest Symptom Report: {symptom}\n"
            f"Checklist (top items): {checklist_preview}\n"
            f"Coordination Status: {coordination_preview}\n"
            f"Agent Reasoning: {reasoning_preview}\n"
            f"Vitals — O2: {vitals.get('oxygen_saturation')}%, "
            f"Temp: {vitals.get('temperature_f')}°F, "
            f"HR: {vitals.get('heart_rate')} bpm, "
            f"Weight delta: {vitals.get('weight_delta_lb')} lb\n"
        )
    n = len(patient_ids)
    header = (
        f"REAL-TIME CARE KNOWLEDGE BASE — {n} patient(s) retrieved by agentic pipeline\n"
        "This data was generated fresh by running the multi-agent orchestration pipeline.\n"
        "Use ONLY this data to answer the user. Do not guess or fabricate.\n\n"
    )
    return header + "\n".join(sections)


def _build_watsonx_chat_system_prompt(repository: "DataRepository") -> str:
    patient_lines = "\n".join(
        f"  - {pid}: {record['name']} (condition: {record.get('primary_condition', 'unknown')})"
        for pid, record in repository.patients.items()
    )
    return (
        "You are an AI Healthcare Multi-Agent Coordinator with access to real patient data.\n\n"
        "AVAILABLE PATIENTS IN THIS SYSTEM:\n"
        f"{patient_lines}\n\n"
        "IMPORTANT RULES:\n"
        "1. When asked broad questions (e.g. 'which patients need escalation?', 'check all patients'), "
        "call the relevant tool for EVERY patient in the list above — do not ask the user which patients to check.\n"
        "2. Use get_patient_snapshot, monitor_recovery_status, coordinate_care_logistics, and "
        "translate_discharge_plan tools to retrieve real data. Never guess or make up values.\n"
        "3. For escalation questions, call monitor_recovery_status for each patient using their "
        "latest symptom context, then summarise who needs escalation and why.\n"
        "4. Answer concisely. Always prioritise patient safety and flag escalation needs clearly."
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
    run_meta = run
    if isinstance(run, dict):
        nested_run = run.get("run")
        if isinstance(nested_run, dict):
            run_meta = nested_run

    events = payload.get("events")
    if not isinstance(events, list) and isinstance(run, dict) and isinstance(run.get("events"), list):
        events = run.get("events")
    if not isinstance(events, list):
        events = []

    assistant_messages: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    terminal_event_status = "unknown"
    failure_reason = ""
    terminal_map = {
        "run.completed": "completed",
        "done": "completed",
        "run.failed": "failed",
        "run.cancelled": "cancelled",
    }

    for event in events:
        if not isinstance(event, dict):
            continue
        event_name = event.get("event")
        if isinstance(event_name, str) and event_name in terminal_map:
            terminal_event_status = terminal_map[event_name]
            if event_name in {"run.failed", "run.cancelled"}:
                data = event.get("data", {})
                if isinstance(data, dict):
                    detail = (
                        data.get("last_error")
                        or data.get("error")
                        or data.get("reason")
                        or data.get("message")
                    )
                    if isinstance(detail, str) and detail.strip():
                        failure_reason = detail.strip()

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
        "run_id": (
            run_meta.get("id")
            or run_meta.get("run_id")
            or run_meta.get("task_id")
            or run_meta.get("thread_id")
            or "n/a"
        ),
        "run_status": (
            run_meta.get("status")
            or run_meta.get("state")
            or run_meta.get("run_status")
            or run_meta.get("a2a_task_status")
            or terminal_event_status
            or "unknown"
        ),
        "assistant_messages": assistant_messages,
        "tool_calls": tool_calls,
        "final_message": assistant_messages[-1] if assistant_messages else "",
        "events": events,
        "failure_reason": failure_reason,
    }


def _run_watsonx_chat(
    watsonx_orchestrator: "WatsonxCareOrchestrator",
    chat_history: list[dict[str, Any]],
    repository: "DataRepository",
    orchestrator: "Orchestrator",
) -> str:
    """RAG chat: retrieve fresh care summaries via agentic pipeline, inject as knowledge, call LLM once."""
    client = watsonx_orchestrator.client

    # Retrieve the current user prompt (last message)
    user_prompt = chat_history[-1]["content"] if chat_history else ""

    # Step 1 — Run agentic pipeline for relevant patients (RAG retrieval)
    knowledge = _build_rag_knowledge(user_prompt, orchestrator, repository)

    # Step 2 — Build augmented system prompt (RAG augmentation)
    base_system = _build_watsonx_chat_system_prompt(repository)
    augmented_system = (
        f"{base_system}\n\n"
        f"--- RETRIEVED KNOWLEDGE (run-time agentic pipeline output) ---\n"
        f"{knowledge}\n"
        f"--- END OF RETRIEVED KNOWLEDGE ---\n\n"
        "Answer the user's question using ONLY the retrieved knowledge above. "
        "Do not call tools — all data has already been retrieved. "
        "Be concise and clinically precise."
    )

    # Step 3 — Build message list with augmented context (RAG generation)
    wx_messages: list[dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": augmented_system}]}
    ]
    # Include conversation history (excluding the last user message — added below cleanly)
    for msg in chat_history[:-1]:
        wx_messages.append({
            "role": msg["role"],
            "content": [{"type": "text", "text": msg["content"]}],
        })
    # Add current user message
    wx_messages.append({
        "role": "user",
        "content": [{"type": "text", "text": user_prompt}],
    })

    # Single LLM call — no tool loop needed, knowledge already injected
    raw = client.chat(messages=wx_messages, tools=None, tool_choice_option=None, max_tokens=1400, temperature=0.2)
    choices = raw.get("choices", [])
    if not choices:
        return "No response from watsonx."
    return extract_text(choices[0].get("message", {}).get("content", "")) or "No response from watsonx."


def render_chat_interface(
    execution_mode: str,
    watsonx_orchestrator: Any,
    config: "AppConfig",
    repository: "DataRepository",
    orchestrator: Any,
) -> None:
    st.subheader("Chat with the Healthcare AI Coordinator")
    st.caption(
        f"Mode: **{execution_mode}** — "
        "RAG: agentic pipeline runs on every prompt to retrieve fresh care summaries."
    )

    if execution_mode == "Live watsonx tools" and watsonx_orchestrator is None:
        st.warning("watsonx is not configured. Add WATSONX_APIKEY and WATSONX_PROJECT_ID to .env.")
        return
    if execution_mode == "Orchestrate REST API" and not config.orchestrate_endpoint:
        st.warning("Orchestrate endpoint is not configured. Add ORCHESTRATE_INSTANCE_URL to .env.")
        return

    # --- Sample prompts by role ---
    st.markdown("#### Sample prompts — click to send")
    role_cols = st.columns(len(SAMPLE_PROMPTS))
    for col, (role, prompts) in zip(role_cols, SAMPLE_PROMPTS.items()):
        with col:
            st.markdown(f"**{role}**")
            for prompt_text in prompts:
                btn_key = f"sp_{role}_{hash(prompt_text)}"
                if st.button(prompt_text, key=btn_key, use_container_width=True):
                    st.session_state["_chat_pending"] = prompt_text

    st.divider()

    chat_key = f"chat_msgs_{execution_mode}"
    if chat_key not in st.session_state:
        st.session_state[chat_key] = []

    messages: list[dict[str, Any]] = st.session_state[chat_key]

    # Render existing history
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
        if msg.get("knowledge"):
            with st.expander("Knowledge retrieved by agentic pipeline", expanded=False):
                st.code(msg["knowledge"], language="text")

    # Resolve input (sample button click or typed)
    pending = st.session_state.pop("_chat_pending", None)
    user_input = st.chat_input("Ask about patients, discharge plans, medications, vitals…")
    prompt = pending or user_input

    if prompt:
        messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        knowledge_block: str = ""
        reply: str = ""

        # Step 1 — RAG retrieval (agentic pipeline)
        relevant = _detect_relevant_patients(prompt, list(repository.patients.keys()))
        with st.spinner(f"Running agentic pipeline for {len(relevant)} patient(s)…"):
            knowledge_block = _build_rag_knowledge(prompt, orchestrator, repository)

        # Step 2 — LLM generation grounded on retrieved knowledge
        with st.chat_message("assistant"):
            with st.spinner("Generating response…"):
                if execution_mode == "Live watsonx tools":
                    try:
                        # Pass full history so far (including latest user message)
                        reply = _run_watsonx_chat(
                            watsonx_orchestrator, messages, repository, orchestrator
                        )
                    except Exception as exc:
                        reply = f"watsonx error: {exc}"
                else:  # Orchestrate REST API — prepend knowledge as context
                    try:
                        augmented_prompt = (
                            f"REAL-TIME CARE KNOWLEDGE (retrieved by agentic pipeline):\n"
                            f"{knowledge_block}\n\n"
                            f"---\nUser question: {prompt}"
                        )
                        run_config = replace(
                            config,
                            orchestrate_bearer_token=st.session_state.get(
                                "orchestrate_bearer_token", config.orchestrate_bearer_token
                            ),
                        )
                        payload = invoke_via_api(run_config, augmented_prompt, poll_timeout_sec=45)
                        refreshed = os.environ.get("ORCHESTRATE_BEARER_TOKEN")
                        if refreshed and not st.session_state.get("orchestrate_bearer_token"):
                            st.session_state["orchestrate_bearer_token"] = refreshed
                        parsed = parse_orchestrate_payload(payload)
                        run_status = str(parsed.get("run_status", "unknown")).lower()
                        if run_status in {"failed", "cancelled"}:
                            reply = f"Run {run_status}: {parsed.get('failure_reason', 'Unknown error.')}"
                        else:
                            reply = (
                                clean_orchestrate_summary(parsed.get("final_message", ""))
                                or "No response from Orchestrate."
                            )
                    except Exception as exc:
                        reply = f"Orchestrate error: {exc}"

            st.markdown(reply)
            if knowledge_block:
                with st.expander("Knowledge retrieved by agentic pipeline", expanded=False):
                    st.code(knowledge_block, language="text")

        messages.append({"role": "assistant", "content": reply, "knowledge": knowledge_block})
        st.session_state[chat_key] = messages
        st.rerun()

    if messages:
        if st.button("Clear chat history", key="clear_chat"):
            st.session_state[chat_key] = []
            st.rerun()


def main() -> None:
    st.set_page_config(
        page_title="Healthcare Multi-Agent Coordinator",
        page_icon=":hospital:",
        layout="wide",
    )

    # ---------------------------------------------------------------------------
    # Google Authentication gate — handles both null and expired sessions.
    #
    # NULL SESSION  : user.is_logged_in is False  → show login page, st.stop()
    # EXPIRED SESSION: logged in but idle > SESSION_TIMEOUT_HOURS → st.logout()
    #
    # The gate is a no-op when [auth] is absent or still has REPLACE placeholders
    # so the app works locally without OAuth configured.
    # ---------------------------------------------------------------------------
    _SESSION_TIMEOUT_HOURS = 8  # idle timeout; set to 0 to disable

    try:
        auth_configured = (
            "auth" in st.secrets
            and "google" in st.secrets["auth"]
            and not str(st.secrets["auth"]["cookie_secret"]).startswith("REPLACE")
            and not str(st.secrets["auth"]["google"]["client_id"]).startswith("REPLACE")
        )
    except Exception:
        auth_configured = False

    current_user = None
    if auth_configured:
        user = st.user  # st.experimental_user was removed; st.user is the current API
        current_user = user

        # --- NULL SESSION: not logged in at all ---
        if not user.is_logged_in:
            st.set_page_config(
                page_title="Sign in — AI Healthcare Coordinator",
                page_icon=":hospital:",
                layout="centered",
            ) if False else None  # page_config already set above; skip duplicate
            st.markdown(
                """
                <div style="display:flex;flex-direction:column;align-items:center;padding-top:4rem;">
                  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/2f/"
                           "Google_2015_logo.svg/640px-Google_2015_logo.svg.png"
                       width="110" style="margin-bottom:1.5rem;" />
                  <h2 style="margin-bottom:0.4rem;">Healthcare Multi-Agent Coordinator</h2>
                  <p style="color:#555;margin-bottom:2rem;text-align:center;max-width:420px;">
                    Sign in with your Google account to access patient care workflows,
                    discharge planning, and risk monitoring.
                  </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.login("google")
            st.stop()

        # --- EXPIRED SESSION: logged in but idle too long ---
        now = time.time()
        if _SESSION_TIMEOUT_HOURS > 0:
            last_active = st.session_state.get("_auth_last_active", now)
            idle_hours = (now - last_active) / 3600
            if idle_hours > _SESSION_TIMEOUT_HOURS:
                st.session_state.clear()
                st.warning("Your session has expired. Please sign in again.")
                st.login("google")
                st.logout()
                st.stop()
        st.session_state["_auth_last_active"] = now

        # Authenticated user controls are rendered in the page header (top-right).

    orchestrator, repository = load_orchestrator()
    _load_ui_content_from_yaml(repository)
    _load_added_patients_from_yaml(repository)
    _apply_added_patient_store(repository)
    watsonx_orchestrator = load_watsonx_orchestrator(repository)
    config = AppConfig.from_env()
    patients = repository.patients
    patient_options = {
        f"{record['patient_id']} - {record['name']}": record["patient_id"] for record in patients.values()
    }
    has_patients = bool(patient_options)

    st.title("AI Healthcare Multi-Agent Care Coordinator")
    st.caption("Interactive demo for discharge translation, symptom monitoring, logistics coordination, and nurse escalation.")

    # ── Healthcare AI themed decorative banner (always visible) ───────────────
    st.markdown(
        """
        <div class="hc-theme-banner">
          <svg class="hc-banner-svg" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 148" preserveAspectRatio="none">
            <g stroke="rgba(0,198,215,0.10)" stroke-width="1">
              <line x1="0" y1="37" x2="1200" y2="37"/>
              <line x1="0" y1="74" x2="1200" y2="74"/>
              <line x1="0" y1="111" x2="1200" y2="111"/>
              <line x1="150" y1="0" x2="150" y2="148"/>
              <line x1="300" y1="0" x2="300" y2="148"/>
              <line x1="450" y1="0" x2="450" y2="148"/>
              <line x1="600" y1="0" x2="600" y2="148"/>
              <line x1="750" y1="0" x2="750" y2="148"/>
              <line x1="900" y1="0" x2="900" y2="148"/>
              <line x1="1050" y1="0" x2="1050" y2="148"/>
            </g>
            <polyline
              points="0,74 80,74 100,74 115,20 130,128 145,40 160,108 175,74 260,74 290,74 305,30 318,118 330,55 343,95 356,74 440,74 480,74 495,18 510,130 525,38 540,110 555,74 640,74 680,74 695,22 710,126 725,42 738,106 751,74 840,74 880,74 895,26 910,122 925,46 938,102 951,74 1040,74 1080,74 1095,24 1110,124 1125,44 1138,104 1151,74 1200,74"
              fill="none"
              stroke="rgba(0,220,230,0.72)"
              stroke-width="2.2"
              stroke-linecap="round"
              stroke-linejoin="round"
            />
            <g fill="rgba(0,198,215,0.15)">
              <rect x="56" y="50" width="6" height="22" rx="2"/>
              <rect x="48" y="58" width="22" height="6" rx="2"/>
              <rect x="256" y="30" width="5" height="18" rx="2"/>
              <rect x="249.5" y="36.5" width="18" height="5" rx="2"/>
              <rect x="880" y="100" width="5" height="18" rx="2"/>
              <rect x="873.5" y="106.5" width="18" height="5" rx="2"/>
              <rect x="1100" y="50" width="6" height="22" rx="2"/>
              <rect x="1092" y="58" width="22" height="6" rx="2"/>
            </g>
            <g fill="rgba(0,220,230,0.85)">
              <circle cx="115" cy="20" r="3.5"/>
              <circle cx="130" cy="128" r="3.5"/>
              <circle cx="145" cy="40" r="3.5"/>
              <circle cx="160" cy="108" r="3.5"/>
              <circle cx="495" cy="18" r="3.5"/>
              <circle cx="510" cy="130" r="3.5"/>
              <circle cx="525" cy="38" r="3.5"/>
              <circle cx="540" cy="110" r="3.5"/>
              <circle cx="895" cy="26" r="3.5"/>
              <circle cx="910" cy="122" r="3.5"/>
              <circle cx="925" cy="46" r="3.5"/>
              <circle cx="938" cy="102" r="3.5"/>
            </g>
            <g stroke="rgba(30,144,255,0.18)" stroke-width="1" fill="none">
              <polygon points="1110,8 1128,18 1128,38 1110,48 1092,38 1092,18"/>
              <polygon points="1146,8 1164,18 1164,38 1146,48 1128,38 1128,18"/>
              <polygon points="1128,38 1146,48 1146,68 1128,78 1110,68 1110,48"/>
              <polygon points="1164,38 1182,48 1182,68 1164,78 1146,68 1146,48"/>
            </g>
            <g stroke="rgba(0,198,215,0.22)" stroke-width="1.4" fill="none">
              <path d="M28,10 Q42,37 28,64 Q14,91 28,118 Q42,145 28,148"/>
              <path d="M42,10 Q28,37 42,64 Q56,91 42,118 Q28,145 42,148"/>
              <line x1="28" y1="28" x2="42" y2="28" stroke="rgba(0,198,215,0.30)"/>
              <line x1="28" y1="46" x2="42" y2="46" stroke="rgba(0,198,215,0.30)"/>
              <line x1="28" y1="64" x2="42" y2="64" stroke="rgba(0,198,215,0.30)"/>
              <line x1="28" y1="82" x2="42" y2="82" stroke="rgba(0,198,215,0.30)"/>
              <line x1="28" y1="100" x2="42" y2="100" stroke="rgba(0,198,215,0.30)"/>
              <line x1="28" y1="118" x2="42" y2="118" stroke="rgba(0,198,215,0.30)"/>
            </g>
            <g stroke="rgba(30,144,255,0.20)" stroke-width="1.2" fill="none">
              <path d="M1190,20 L1185,20 L1185,60 L1175,60 L1175,100 L1190,100"/>
              <circle cx="1185" cy="20" r="2.5" fill="rgba(30,144,255,0.40)"/>
              <circle cx="1175" cy="60" r="2.5" fill="rgba(30,144,255,0.40)"/>
              <circle cx="1185" cy="100" r="2.5" fill="rgba(30,144,255,0.40)"/>
            </g>
          </svg>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # ─────────────────────────────────────────────────────────────────────────

    if auth_configured and current_user is not None and current_user.is_logged_in:
        # Handle sign-out via query param (triggered from the HTML tooltip link)
        if st.query_params.get("signout") == "1":
            st.query_params.clear()
            st.session_state.clear()
            st.logout()

        pic = getattr(current_user, "picture", None) or getattr(current_user, "avatar_url", None)
        name = getattr(current_user, "name", None) or getattr(current_user, "email", "User")
        email = getattr(current_user, "email", "")
        initials = "".join(w[0].upper() for w in (name or "U").split()[:2])

        # Avatar button: photo if available, else colored initials circle
        if pic:
            avatar_btn_html = f'<img src="{pic}" class="hc-avatar-img" alt="{initials}" />'
            avatar_large_html = f'<img src="{pic}" class="hc-drop-photo" alt="{initials}" />'
        else:
            avatar_btn_html = f'<span class="hc-avatar-init">{initials}</span>'
            avatar_large_html = (
                f'<div class="hc-drop-initials">{initials}</div>'
            )

        st.markdown(
            f"""
            <style>
            /* ── Profile widget: fixed in top toolbar, left of Deploy button ─── */
            .hc-profile-wrap {{
                position: fixed;
                top: 0.42rem;
                right: 9.2rem;
                z-index: 99999999;
            }}
            .hc-avatar-btn {{
                width: 36px;
                height: 36px;
                border-radius: 50%;
                overflow: hidden;
                cursor: pointer;
                border: 2px solid #d0d5dd;
                background: #4285F4;
                display: flex;
                align-items: center;
                justify-content: center;
            }}
            .hc-avatar-img {{
                width: 36px;
                height: 36px;
                border-radius: 50%;
                object-fit: cover;
                display: block;
            }}
            .hc-avatar-init {{
                color: white;
                font-weight: 700;
                font-size: 0.85rem;
                font-family: sans-serif;
                line-height: 1;
            }}
            /* ── Hover dropdown card ────────────────────────────────────────── */
            .hc-profile-dropdown {{
                display: none;
                position: absolute;
                top: 44px;
                right: 0;
                background: #ffffff;
                border: 1px solid #e0e4ea;
                border-radius: 14px;
                box-shadow: 0 10px 32px rgba(0,0,0,0.14);
                padding: 1.2rem 1.4rem 1rem;
                min-width: 230px;
                text-align: center;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            }}
            .hc-profile-wrap:hover .hc-profile-dropdown {{
                display: block;
            }}
            .hc-drop-photo {{
                width: 60px;
                height: 60px;
                border-radius: 50%;
                object-fit: cover;
                margin-bottom: 0.6rem;
                border: 2px solid #e8eaed;
            }}
            .hc-drop-initials {{
                width: 60px;
                height: 60px;
                border-radius: 50%;
                background: #4285F4;
                color: white;
                font-weight: 700;
                font-size: 1.3rem;
                display: flex;
                align-items: center;
                justify-content: center;
                margin: 0 auto 0.6rem;
            }}
            .hc-drop-name {{
                font-weight: 700;
                font-size: 0.95rem;
                color: #1a1a2e;
                margin-bottom: 0.25rem;
            }}
            .hc-drop-email {{
                font-size: 0.8rem;
                color: #5b6472;
                margin-bottom: 0.8rem;
                word-break: break-all;
            }}
            .hc-drop-divider {{
                border: none;
                border-top: 1px solid #eeeff2;
                margin: 0.5rem 0 0.7rem;
            }}
            .hc-signout-link {{
                display: inline-block;
                padding: 0.38rem 1.4rem;
                border: 1px solid #d0d3d9;
                border-radius: 7px;
                font-size: 0.84rem;
                color: #333;
                text-decoration: none;
                background: #fafafa;
            }}
            .hc-signout-link:hover {{
                background: #f0f0f0;
                color: #111;
                text-decoration: none;
            }}
            </style>

            <div class="hc-profile-wrap">
              <div class="hc-avatar-btn" title="Account">
                {avatar_btn_html}
              </div>
              <div class="hc-profile-dropdown">
                {avatar_large_html}
                <div class="hc-drop-name">{name}</div>
                <div class="hc-drop-email">{email}</div>
                <hr class="hc-drop-divider" />
                <a href="?signout=1" class="hc-signout-link">Sign out</a>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.sidebar:
        app_mode = st.radio(
            "App mode",
            options=["Care Workflow", "Chat"],
            index=0,
            horizontal=True,
            help="Care Workflow: structured patient form. Chat: freeform conversation with the AI coordinator.",
        )
        if app_mode == "Care Workflow":
            add_col, manage_col = st.columns(2)
            with add_col:
                if st.button("Add New Patient", use_container_width=True):
                    _open_add_patient_dialog(repository)
            with manage_col:
                if st.button("Edit/Delete Patient", use_container_width=True):
                    st.session_state[_SHOW_MANAGE_PATIENTS_DIALOG_KEY] = True
        st.divider()

        st.subheader("Case Setup" if app_mode == "Care Workflow" else "Chat Setup")
        if has_patients:
            selected_label = st.selectbox("Patient", list(patient_options.keys()))
            patient_id = patient_options[selected_label]
            patient = patients[patient_id]
            discharge_plan = repository.discharge_plans[patient_id]
            default_report = _DEFAULT_SYMPTOM_REPORTS.get(patient_id, "")
        else:
            st.info("No patients available. Add a new patient to continue.")
            selected_label = None
            patient_id = ""
            patient = None
            discharge_plan = None
            default_report = ""

        if app_mode == "Care Workflow" and has_patients:
            symptom_report = st.text_area(
                "Symptom report",
                value=default_report,
                height=140,
                help="Describe what the patient or caregiver is reporting today.",
            )
        else:
            symptom_report = default_report  # not used in chat mode

        if app_mode == "Care Workflow":
            execution_mode = st.radio(
                "Execution mode",
                options=["Local deterministic", "Live watsonx tools", "Orchestrate REST API"],
                index=0,
                help="Switch between local deterministic orchestration, direct watsonx tool-calling, or hosted Orchestrate API runs.",
            )
        else:
            execution_mode = st.radio(
                "Execution mode",
                options=["Live watsonx tools", "Orchestrate REST API"],
                index=0,
                help="Chat is supported for Live watsonx tools and Orchestrate REST API modes.",
            )
        token_key = "orchestrate_bearer_token"
        env_token = os.environ.get("ORCHESTRATE_BEARER_TOKEN", "")
        if token_key not in st.session_state and env_token:
            st.session_state[token_key] = env_token
        default_token = st.session_state.get(token_key) or env_token or config.orchestrate_bearer_token or ""
        bearer_token = st.text_input(
            "Orchestrate bearer token (optional override)",
            value=default_token,
            type="password",
            help="Used for Orchestrate REST API mode. Stored in this Streamlit session only.",
        )
        if bearer_token:
            st.session_state[token_key] = bearer_token
            os.environ["ORCHESTRATE_BEARER_TOKEN"] = bearer_token
        elif token_key in st.session_state:
            # Allow clearing a previously entered token from session.
            st.session_state.pop(token_key, None)
            os.environ.pop("ORCHESTRATE_BEARER_TOKEN", None)

        active_token = st.session_state.get(token_key) or os.environ.get("ORCHESTRATE_BEARER_TOKEN") or config.orchestrate_bearer_token
        token_source = "none"
        if st.session_state.get(token_key):
            token_source = "session"
        elif os.environ.get("ORCHESTRATE_BEARER_TOKEN"):
            token_source = "env"
        elif config.orchestrate_bearer_token:
            token_source = "config"

        st.caption(f"Token source: `{token_source}`")
        st.caption(f"API key present: `{bool(config.orchestrate_api_key)}`")
        if active_token:
            if len(active_token) > 24:
                preview = f"{active_token[:12]}...{active_token[-8:]}"
            else:
                preview = active_token
            st.caption(f"Token preview: `{preview}`")
            if st.checkbox("Show active token (local debug only)", value=False):
                st.code(active_token, language="text")
        else:
            st.caption("Token preview: `none`")
        if app_mode == "Care Workflow":
            submitted = st.button("Run care coordination", type="primary", use_container_width=True)
        else:
            submitted = False

        st.markdown("### Engine Readiness")
        st.write(f"watsonx tools: {'Ready' if watsonx_orchestrator else 'Missing config'}")
        st.write(f"Orchestrate API: {'Ready' if config.orchestrate_endpoint else 'Missing endpoint'}")

        if app_mode == "Care Workflow" and patient is not None and discharge_plan is not None:
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

    if st.session_state.get(_SHOW_MANAGE_PATIENTS_DIALOG_KEY):
        # One-shot open guard: avoid reopening the dialog on unrelated reruns/clicks.
        st.session_state[_SHOW_MANAGE_PATIENTS_DIALOG_KEY] = False
        _open_manage_patients_dialog(repository)

    # Load local background image as base64 data URL
    import base64 as _b64, pathlib as _pl
    _bg_path = _pl.Path(__file__).parent / "data" / "bg_healthcare.jpg"
    _bg_data_url = ""
    if _bg_path.exists():
        _bg_data_url = "data:image/jpeg;base64," + _b64.b64encode(_bg_path.read_bytes()).decode()

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@500;700;800&display=swap');
        html, body, [class*="css"] {{
            font-family: 'Manrope', sans-serif;
        }}

        /* ── Full-page image background theme ───────────────────────────────── */
        .stApp,
        [data-testid="stAppViewContainer"] {{
            background: linear-gradient(160deg, #5b9db5 0%, #4d8fa6 40%, #3d7f96 70%, #2e6f86 100%) !important;
        }}
        [data-testid="stMain"] {{
            background-image:
                linear-gradient(rgba(10, 28, 50, 0.38), rgba(10, 28, 50, 0.38)),
                url("{_bg_data_url}") !important;
            background-size: cover !important;
            background-position: center !important;
            background-repeat: no-repeat !important;
            background-attachment: fixed !important;
        }}
        [data-testid="stHeader"] {{
            background: rgba(75, 130, 158, 0.92) !important;
            border-bottom: 1px solid rgba(100, 180, 210, 0.25);
        }}
        [data-testid="stHeader"] * {{
            color: #002a33 !important;
        }}
        [data-testid="stHeader"] button,
        [data-testid="stHeader"] a {{
            color: #002a33 !important;
        }}
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #6a9eb8 0%, #5a8eaa 100%) !important;
            border-right: 1px solid rgba(50, 90, 120, 0.25);
        }}
        [data-testid="stSidebar"] * {{
            color: #0d2a3a !important;
        }}
        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stTextArea label,
        [data-testid="stSidebar"] .stTextInput label,
        [data-testid="stSidebar"] .stRadio label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span {{
            color: #0d2a3a !important;
        }}
        /* Main content text — light for dark image overlay */
        [data-testid="stMain"] h1,
        [data-testid="stMain"] h2,
        [data-testid="stMain"] h3,
        [data-testid="stMain"] h4,
        [data-testid="stMain"] p,
        [data-testid="stMain"] span,
        [data-testid="stMain"] li,
        [data-testid="stMain"] label {{
            color: #e8f4ff !important;
        }}
        [data-testid="stMain"] .stCaption,
        [data-testid="stMain"] [data-testid="stCaptionContainer"] {{
            color: #a8d8f0 !important;
        }}
        /* Input / select widgets */
        [data-testid="stMain"] .stTextInput input,
        [data-testid="stMain"] .stTextArea textarea,
        [data-testid="stMain"] .stSelectbox > div > div,
        [data-testid="stMain"] .stNumberInput input {{
            background: rgba(0, 20, 40, 0.55) !important;
            border: 1px solid rgba(0, 198, 215, 0.40) !important;
            color: #e8f4ff !important;
            border-radius: 10px;
        }}
        /* Tabs */
        [data-testid="stMain"] button[data-baseweb="tab"] {{
            color: #a8d8f0 !important;
        }}
        [data-testid="stMain"] button[data-baseweb="tab"][aria-selected="true"] {{
            color: #00e0f0 !important;
            border-bottom-color: #00e0f0 !important;
        }}
        /* Buttons */
        [data-testid="stMain"] .stButton > button {{
            background: linear-gradient(90deg, #006f80, #008899) !important;
            color: #ffffff !important;
            border: 1px solid rgba(0, 198, 215, 0.50) !important;
            border-radius: 10px !important;
        }}
        [data-testid="stMain"] .stButton > button:hover {{
            background: linear-gradient(90deg, #00a0b8, #00c6d7) !important;
            border-color: #00c6d7 !important;
        }}
        /* Info / warning / error boxes */
        [data-testid="stMain"] [data-testid="stAlert"] {{
            background: rgba(0, 20, 40, 0.60) !important;
            border: 1px solid rgba(0, 198, 215, 0.30) !important;
            color: #e8f4ff !important;
            border-radius: 12px !important;
        }}
        /* Code blocks */
        [data-testid="stMain"] pre, [data-testid="stMain"] code {{
            background: rgba(0, 10, 25, 0.70) !important;
            border: 1px solid rgba(0, 198, 215, 0.25) !important;
            color: #00e0f0 !important;
        }}
        /* Expanders — header + content */
        [data-testid="stMain"] details,
        [data-testid="stMain"] [data-testid="stExpander"] {{
            background: rgba(0, 20, 45, 0.75) !important;
            border: 1px solid rgba(0, 198, 215, 0.30) !important;
            border-radius: 12px !important;
        }}
        [data-testid="stMain"] details summary,
        [data-testid="stMain"] .streamlit-expanderHeader,
        [data-testid="stMain"] [data-testid="stExpanderToggleIcon"],
        [data-testid="stMain"] [data-testid="stExpander"] summary {{
            background: rgba(0, 30, 60, 0.85) !important;
            color: #e8f4ff !important;
            border-radius: 12px;
        }}
        [data-testid="stMain"] details summary p,
        [data-testid="stMain"] details summary span,
        [data-testid="stMain"] .streamlit-expanderHeader p,
        [data-testid="stMain"] .streamlit-expanderHeader span {{
            color: #e8f4ff !important;
        }}
        [data-testid="stMain"] details[open] summary {{
            border-radius: 12px 12px 0 0;
        }}
        /* JSON viewer */
        [data-testid="stMain"] [data-testid="stJson"],
        [data-testid="stMain"] [data-testid="stJson"] > div,
        [data-testid="stMain"] .stJson {{
            background: rgba(0, 10, 28, 0.88) !important;
            border-radius: 8px !important;
            color: #a8d8f0 !important;
        }}
        [data-testid="stMain"] [data-testid="stJson"] * {{
            color: #a8d8f0 !important;
            background: transparent !important;
        }}
        /* Override any white backgrounds inside expander content */
        [data-testid="stMain"] [data-testid="stExpanderDetails"],
        [data-testid="stMain"] [data-testid="stExpanderDetails"] > div {{
            background: rgba(0, 15, 38, 0.80) !important;
            color: #e8f4ff !important;
        }}
        /* Metric tiles */
        [data-testid="stMain"] [data-testid="stMetric"] {{
            background: rgba(0, 20, 45, 0.55) !important;
            border: 1px solid rgba(0, 198, 215, 0.25) !important;
            border-radius: 12px !important;
            padding: 0.6rem 1rem;
        }}
        /* Dividers */
        hr {{
            border-color: rgba(0, 198, 215, 0.20) !important;
        }}
        /* ─────────────────────────────────────────────────────────────────── */

        .result-card {{
            border: 1px solid rgba(0, 198, 215, 0.28);
            border-radius: 18px;
            padding: 1.1rem 1.2rem;
            background: rgba(0, 20, 45, 0.58);
            backdrop-filter: blur(10px);
        }}
        .result-card p, .result-card span, .result-card li {{
            color: #e8f4ff !important;
        }}
        .small-label {{
            font-size: 0.8rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: #a8d8f0 !important;
        }}
        .hc-theme-banner {{
            position: relative;
            width: 100%;
            height: 148px;
            border-radius: 20px;
            overflow: hidden;
            margin-bottom: 1.4rem;
            background: linear-gradient(120deg, #050f2c 0%, #0a1f4e 30%, #0d3b7a 60%, #0a6ea6 85%, #00c6d7 100%);
            box-shadow: 0 8px 32px rgba(0, 100, 200, 0.22);
        }}
        .hc-theme-banner::before {{
            content: "";
            position: absolute;
            width: 320px; height: 320px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(0,198,215,0.18) 0%, transparent 70%);
            top: -110px; right: -60px;
        }}
        .hc-theme-banner::after {{
            content: "";
            position: absolute;
            width: 200px; height: 200px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(30,144,255,0.22) 0%, transparent 70%);
            bottom: -80px; left: 80px;
        }}
        .hc-banner-svg {{
            position: absolute;
            inset: 0;
            width: 100%;
            height: 100%;
        }}
        .severity-highlight {{
            color: #cc1d1d !important;
            background: rgba(255, 87, 87, 0.22);
            border: 1px solid rgba(255, 87, 87, 0.35);
            border-radius: 6px;
            padding: 0 0.24rem;
            font-weight: 700;
        }}
        .severity-line {{
            color: #e8f4ff;
            margin: 0.2rem 0;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    if app_mode == "Chat":
        render_chat_interface(execution_mode, watsonx_orchestrator, config, repository, orchestrator)
        return

    if not has_patients:
        st.info("No patients available in Case Setup. Use Add New Patient or restore a patient via YAML data.")
        return

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
    parsed_orchestrate: dict[str, Any] | None = None

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
            with st.spinner("Running Orchestrate workflow..."):
                orchestrate_payload = invoke_via_api(run_config, prompt, poll_timeout_sec=45)
            refreshed_token = os.environ.get("ORCHESTRATE_BEARER_TOKEN")
            if refreshed_token and not st.session_state.get("orchestrate_bearer_token"):
                st.session_state["orchestrate_bearer_token"] = refreshed_token
            parsed_orchestrate = parse_orchestrate_payload(orchestrate_payload)
            run_status = str(parsed_orchestrate.get("run_status", "unknown")).lower()
            if run_status in {"failed", "cancelled"}:
                reason = parsed_orchestrate.get("failure_reason") or f"Orchestrate run status is {run_status}."
                orchestrate_error = str(reason)
                selected_mode_badge = "Orchestrate REST API (fallback: local deterministic)"
            elif parsed_orchestrate.get("final_message"):
                selected_summary = clean_orchestrate_summary(str(parsed_orchestrate["final_message"])) or selected_summary
            else:
                selected_mode_badge = "Orchestrate REST API (no assistant message, fallback: local deterministic)"
        except Exception as exc:  # pragma: no cover
            orchestrate_error = str(exc)
            selected_mode_badge = "Orchestrate REST API (fallback: local deterministic)"

    monitoring_output = result.details["monitoring_output"]
    logistics_output = result.details["logistics_output"]
    translator_output = result.details["translator_output"]
    risk_label, risk_style = severity_label(result.risk_status)

    overview_col, status_col, actions_col = st.columns((1.5, 1, 1))

    with overview_col:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown('<div class="small-label">Recovery Summary</div>', unsafe_allow_html=True)
        st.markdown(f"### {_highlight_severity_phrases(selected_summary)}", unsafe_allow_html=True)
        st.caption(f"Execution mode: {selected_mode_badge}")
        st.write(f"Patient ID: `{result.patient_id}`")
        _write_highlighted_line(f"Symptom report: {symptom_report}")
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
            _write_highlighted_line(item, bullet=True)

        st.subheader("Safety Tips")
        for item in translator_output["safety_tips"]:
            _write_highlighted_line(item, bullet=True)

    with logistics_col:
        st.subheader("Risk Review")
        st.write(f"Triage level: `{monitoring_output['triage_level']}`")
        _write_highlighted_line(monitoring_output["recommended_action"])
        if monitoring_output["concerning_signals"]:
            for signal in monitoring_output["concerning_signals"]:
                _write_highlighted_line(signal, bullet=True)
        else:
            st.write("- No concerning signals were detected.")

        st.subheader("Logistics Coordination")
        for line in logistics_output["medication_status"]:
            _write_highlighted_line(line, bullet=True)
        for line in logistics_output["appointment_status"]:
            _write_highlighted_line(line, bullet=True)
        for line in logistics_output["resolved_actions"]:
            _write_highlighted_line(line, bullet=True)

    with st.expander("Transparent reasoning log", expanded=True):
        for entry in result.reasoning_log:
            _write_highlighted_line(entry, bullet=True)

    if watsonx_result:
        with st.expander("watsonx Tool Trace", expanded=True):
            for call in watsonx_result.tool_trace:
                label = call.get("tool") or call.get("tool_name") or call.get("name", "unknown")
                st.write(f"Tool: `{label}`")
                st.json(call)
    elif watsonx_error:
        st.error(f"watsonx invocation failed: {watsonx_error}")

    if orchestrate_payload:
        parsed = parsed_orchestrate or parse_orchestrate_payload(orchestrate_payload)
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
    if orchestrate_error:
        st.error(f"Orchestrate API invocation failed: {orchestrate_error}")

    with st.expander("Structured payload"):
        st.json(asdict(result))


if __name__ == "__main__":
    main()
