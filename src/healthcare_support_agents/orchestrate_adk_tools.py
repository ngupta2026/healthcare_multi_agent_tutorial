from __future__ import annotations

from typing import Any

from .config import AppConfig
from .serper_client import SerperSearchClient
from .tool_wrappers import HealthcareToolRuntime, build_repository

try:
    from ibm_watsonx_orchestrate.agent_builder.tools import tool
except ImportError:  # pragma: no cover - local fallback when the ADK is not installed
    def tool(*_args: Any, **_kwargs: Any):
        def decorator(func):
            return func
        return decorator


def _runtime() -> HealthcareToolRuntime:
    config = AppConfig.from_env()
    search_client = SerperSearchClient(config.serper_api_key) if config.serper_ready else None
    return HealthcareToolRuntime(build_repository(), search_client=search_client)


@tool
def get_patient_snapshot(patient_id: str) -> dict[str, Any]:
    """Return the full local patient snapshot for one patient identifier."""
    return _runtime().get_patient_snapshot(patient_id)


@tool
def translate_discharge_plan(patient_id: str) -> dict[str, Any]:
    """Translate the patient's discharge plan into a daily checklist, medication schedule, and safety tips."""
    return _runtime().translate_discharge_plan(patient_id)


@tool
def monitor_recovery_status(patient_id: str, symptom_report: str) -> dict[str, Any]:
    """Evaluate recovery risk using the current symptom report and the stored vitals for the patient."""
    return _runtime().monitor_recovery_status(patient_id, symptom_report)


@tool
def coordinate_care_logistics(patient_id: str) -> dict[str, Any]:
    """Review medication fill status, follow-up appointments, and transportation barriers for the patient."""
    return _runtime().coordinate_care_logistics(patient_id)


@tool
def resolve_recovery_case(patient_id: str, symptom_report: str) -> dict[str, Any]:
    """Run the complete healthcare recovery workflow and return the consolidated result payload."""
    return _runtime().resolve_recovery_case(patient_id, symptom_report)


@tool
def search_support_services(query: str) -> dict[str, Any]:
    """Search public web results for support-service information such as transportation or pharmacy options."""
    return _runtime().search_support_services(query)
