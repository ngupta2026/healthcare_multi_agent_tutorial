from __future__ import annotations

# Bug 1 Fix — Removed heavy top-level imports (WatsonxClient, WatsonxCareOrchestrator,
# AppConfig, SerperSearchClient).
#
# Root cause: the original file imported these at module load time:
#
#   from .config import AppConfig
#   from .serper_client import SerperSearchClient
#   from .watsonx_client import WatsonxClient
#   from .watsonx_orchestrator import WatsonxCareOrchestrator
#
# WatsonxClient and WatsonxCareOrchestrator transitively pull in `ibm-watsonx-ai`
# (a large SDK). That package is NOT listed in requirements-orchestrate.txt and is
# NOT available in IBM Orchestrate's cloud tool execution sandbox.
# Result: every tool call failed at import time before any user code ran.
#
# Fix: only import the pure-data stack (tool_wrappers → repository → connectors → agents).
# These modules have zero external dependencies — they read from JSON files and apply
# rule-based logic only. IBM Orchestrate's native agent layer (Granite model) handles
# all LLM reasoning; the tools just supply structured data.
import os
from typing import Any

from .tool_wrappers import HealthcareToolRuntime, build_repository

try:
    from ibm_watsonx_orchestrate.agent_builder.tools import tool
except ImportError:  # pragma: no cover - local fallback when the ADK is not installed
    def tool(*_args: Any, **_kwargs: Any):
        def decorator(func):
            return func
        return decorator


def _runtime() -> HealthcareToolRuntime:
    # Uses the pure-data orchestrator (no LLM calls) — IBM Orchestrate's native
    # agents handle all LLM reasoning via their own granite model.
    return HealthcareToolRuntime(build_repository())


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
    """Run the full healthcare recovery pipeline (PatientContext → DischargeTranslator
    → RecoveryMonitoring → CareLogistics) and return the consolidated result."""
    # Bug 2 Fix — Double-orchestration: the original code called WatsonxCareOrchestrator,
    # which internally ran a full 5-step LLM pipeline (PatientContext → DischargeTranslator
    # → RecoveryMonitoring → CareLogistics → Supervisor LLM synthesis).
    #
    # Why that was wrong:
    #   - IBM Orchestrate's AI_Healthcare_Coordinator native agent IS the supervisor
    #     orchestrator. It already manages the 4 sub-agents and synthesises the final response.
    #   - Having a tool spawn its own LLM pipeline creates double-orchestration: one LLM
    #     chain inside the tool + another driven by the native agent — conflicting outputs.
    #   - WatsonxCareOrchestrator requires WATSONX_APIKEY and WATSONX_PROJECT_ID at runtime,
    #     which are not available (and should not be needed) in IBM Orchestrate's tool sandbox.
    #
    # Fix: delegate to the local rule-based Orchestrator (orchestrator.py) which chains
    # repository → connectors → agents using pure data logic only — no LLM calls,
    # no external credentials. IBM Orchestrate's granite model does all the reasoning.
    return _runtime().resolve_recovery_case(patient_id, symptom_report)


@tool
def search_support_services(query: str) -> dict[str, Any]:
    """Search public web results for support-service information such as transportation or pharmacy options."""
    # Bug 1 Fix — SerperSearchClient is imported lazily (inside the function) instead of
    # at module top-level. This avoids loading it during module import in the cloud sandbox
    # where SERPER_API_KEY may not be set. The tool gracefully returns an empty result
    # rather than crashing if the key is absent, keeping the other 5 tools unaffected.
    serper_key = os.getenv("SERPER_API_KEY")
    if not serper_key:
        return {
            "query": query,
            "results": [],
            "note": "SERPER_API_KEY not configured. Set this environment variable to enable web search.",
        }
    from .serper_client import SerperSearchClient
    return SerperSearchClient(serper_key).search(query=query)
