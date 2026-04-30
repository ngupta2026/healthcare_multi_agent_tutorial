from __future__ import annotations

import json
import time
from typing import Any

from .agent_logger import get_logger, log_separator
from .llm_agents import (
    CareLogisticsAgent,
    DischargeTranslatorAgent,
    PatientContextAgent,
    RecoveryMonitoringAgent,
)
from .models import WatsonxResponse
from .tool_wrappers import HealthcareToolRuntime, tool_message
from .watsonx_client import WatsonxClient

_log = get_logger(__name__)


SUPERVISOR_PROMPT = """
You are the AI Healthcare Coordinator — a supervisor agent that synthesises
outputs from four specialist sub-agents into one final clinical summary.

You will receive structured outputs from:
- Patient Context Agent: patient profile and data snapshot
- Discharge Translator Agent: plain-language checklist and medication schedule
- Recovery Monitoring Agent: triage level, concerning signals, escalation flag
- Care Logistics Agent: medication status, appointment status, and resolved barriers

Your job: produce a concise, clinically useful recovery summary that:
1. States the overall risk status (routine / watch_closely / high_priority_health_shift)
2. Calls out nurse escalation clearly if required
3. Lists the top immediate actions from the logistics and monitoring outputs
4. Keeps the tone plain and actionable for clinical staff
""".strip()


class WatsonxCareOrchestrator:
    """Multi-agent orchestrator.

    Workflow
    --------
    1. PatientContextAgent    – loads and summarises the patient record
    2. DischargeTranslatorAgent – produces the plain-language checklist
    3. RecoveryMonitoringAgent  – triages the symptom report + vitals
    4. CareLogisticsAgent       – resolves pharmacy / transport barriers
    5. Supervisor LLM           – synthesises all outputs into a final summary
    """

    def __init__(self, client: WatsonxClient, runtime: HealthcareToolRuntime) -> None:
        self.client = client
        self.runtime = runtime
        self._context_agent = PatientContextAgent(client, runtime)
        self._translator_agent = DischargeTranslatorAgent(client, runtime)
        self._monitoring_agent = RecoveryMonitoringAgent(client, runtime)
        self._logistics_agent = CareLogisticsAgent(client, runtime)

    def resolve(self, patient_id: str, symptom_report: str) -> WatsonxResponse:
        log_separator(f"NEW CASE — patient_id={patient_id}")
        _log.info(
            "[WatsonxCareOrchestrator] CASE STARTED for patient_id=%s\n"
            "  Overview of the 5-step multi-agent pipeline that will now run:\n"
            "    Step 1 — PatientContextAgent   : load and summarise the patient's full record\n"
            "    Step 2 — DischargeTranslatorAgent: convert discharge notes to plain language\n"
            "    Step 3 — RecoveryMonitoringAgent : triage the patient's symptoms and vitals\n"
            "    Step 4 — CareLogisticsAgent      : check medications, appointments, transport\n"
            "    Step 5 — Supervisor LLM          : synthesise all 4 outputs into a final summary\n"
            "  Symptom report submitted by patient/caregiver: %s",
            patient_id, symptom_report,
        )
        t_start = time.monotonic()
        full_trace: list[dict[str, Any]] = []

        # --- Step 1: Patient Context Agent ---
        _log.info(
            "[WatsonxCareOrchestrator] STEP 1 of 5 — Launching PatientContextAgent\n"
            "  This agent calls the 'get_patient_snapshot' tool to load all stored data\n"
            "  for patient %s from the healthcare data repository.\n"
            "  It will return a structured summary of demographics, discharge plan,\n"
            "  vitals, pharmacy status, and upcoming appointments.",
            patient_id,
        )
        context_summary, context_trace = self._context_agent.run(patient_id)
        full_trace.extend(context_trace)
        _log.info(
            "[WatsonxCareOrchestrator] STEP 1 COMPLETE — PatientContextAgent finished.\n"
            "  The patient record has been retrieved and summarised.\n"
            "  Summary (first 300 chars): %.300s",
            context_summary,
        )

        # --- Step 2: Discharge Translator Agent ---
        _log.info(
            "[WatsonxCareOrchestrator] STEP 2 of 5 — Launching DischargeTranslatorAgent\n"
            "  This agent reads the discharge instructions for patient %s and rewrites\n"
            "  them into plain, patient-friendly language using the\n"
            "  'translate_discharge_plan' tool.\n"
            "  Output will include: daily checklist, medication schedule, safety tips.",
            patient_id,
        )
        checklist_summary, translator_trace = self._translator_agent.run(patient_id)
        full_trace.extend(translator_trace)
        _log.info(
            "[WatsonxCareOrchestrator] STEP 2 COMPLETE — DischargeTranslatorAgent finished.\n"
            "  Discharge plan translated to plain language.\n"
            "  Summary (first 300 chars): %.300s",
            checklist_summary,
        )

        # --- Step 3: Recovery Monitoring Agent ---
        _log.info(
            "[WatsonxCareOrchestrator] STEP 3 of 5 — Launching RecoveryMonitoringAgent\n"
            "  This agent analyses the patient's symptom report alongside their recorded\n"
            "  vitals using the 'monitor_recovery_status' tool.\n"
            "  It will assign a triage level and flag if a nurse must be notified.\n"
            "  Symptom report being evaluated: %s",
            symptom_report,
        )
        risk_summary, monitoring_trace = self._monitoring_agent.run(patient_id, symptom_report)
        full_trace.extend(monitoring_trace)
        _log.info(
            "[WatsonxCareOrchestrator] STEP 3 COMPLETE — RecoveryMonitoringAgent finished.\n"
            "  Triage assessment complete. Risk level determined.\n"
            "  Summary (first 300 chars): %.300s",
            risk_summary,
        )

        # --- Step 4: Care Logistics Agent ---
        _log.info(
            "[WatsonxCareOrchestrator] STEP 4 of 5 — Launching CareLogisticsAgent\n"
            "  This agent uses the 'coordinate_care_logistics' tool to check whether\n"
            "  the patient's prescriptions are filled, appointments are booked, and\n"
            "  transportation is arranged.\n"
            "  If any barriers are found it will call 'search_support_services'\n"
            "  to find alternative pharmacies, transport, or community support options.",
        )
        logistics_summary, logistics_trace = self._logistics_agent.run(patient_id)
        full_trace.extend(logistics_trace)
        _log.info(
            "[WatsonxCareOrchestrator] STEP 4 COMPLETE — CareLogisticsAgent finished.\n"
            "  Logistics barriers identified and action plan prepared.\n"
            "  Summary (first 300 chars): %.300s",
            logistics_summary,
        )

        # --- Step 5: Supervisor LLM synthesis ---
        _log.info(
            "[WatsonxCareOrchestrator] STEP 5 of 5 — Calling Supervisor LLM\n"
            "  All four specialist agent outputs are now combined and sent to the\n"
            "  Supervisor LLM (IBM WatsonX Granite) in a single request.\n"
            "  The Supervisor will:\n"
            "    - Determine the overall risk status\n"
            "    - Highlight any nurse escalation requirement\n"
            "    - List the top immediate actions for clinical staff\n"
            "  No tools are called at this step — this is pure synthesis.",
        )
        synthesis_input = (
            f"Patient ID: {patient_id}\n"
            f"Symptom report: {symptom_report}\n\n"
            f"--- Patient Context Agent output ---\n{context_summary or '(no output)'}\n\n"
            f"--- Discharge Translator Agent output ---\n{checklist_summary or '(no output)'}\n\n"
            f"--- Recovery Monitoring Agent output ---\n{risk_summary or '(no output)'}\n\n"
            f"--- Care Logistics Agent output ---\n{logistics_summary or '(no output)'}\n\n"
            "Synthesise the above into a final clinical recovery summary."
        )

        transcript: list[dict[str, Any]] = [
            {"role": "system", "content": [{"type": "text", "text": SUPERVISOR_PROMPT}]},
            {"role": "user", "content": [{"type": "text", "text": synthesis_input}]},
        ]

        raw_response = self.client.chat(messages=transcript, tools=None, tool_choice_option=None)
        choices = raw_response.get("choices", [])
        if choices:
            msg = choices[0].get("message", {})
            content = msg.get("content", "")
            if isinstance(content, list):
                parts = [c.get("text", "") for c in content if isinstance(c, dict)]
                final_response = " ".join(parts).strip()
            else:
                final_response = (content or "").strip()
            _log.info(
                "[WatsonxCareOrchestrator] STEP 5 COMPLETE — Supervisor LLM produced final summary.\n"
                "  This is the response that will be shown to clinical staff in the UI.\n"
                "  Final summary (first 400 chars): %.400s",
                final_response,
            )
        else:
            # Fallback: LLM returned no choices (e.g. service unavailable)
            _log.warning(
                "[WatsonxCareOrchestrator] STEP 5 — Supervisor LLM returned no response.\n"
                "  This usually means the WatsonX service was unavailable or the token\n"
                "  quota was exceeded. Falling back to the local deterministic orchestrator\n"
                "  which produces a rule-based summary without calling the LLM.",
            )
            fallback = self.runtime.resolve_recovery_case(patient_id, symptom_report)
            final_response = fallback["recovery_summary"]
            _log.info(
                "[WatsonxCareOrchestrator] FALLBACK summary produced locally.\n"
                "  Fallback output (first 300 chars): %.300s",
                final_response,
            )

        elapsed = time.monotonic() - t_start
        _log.info(
            "[WatsonxCareOrchestrator] CASE COMPLETE — patient_id=%s\n"
            "  Total time taken      : %.2f seconds\n"
            "  Total tool calls made : %d (across all 4 specialist agents)\n"
            "  The final summary has been returned and is ready to display.",
            patient_id, elapsed, len(full_trace),
        )
        log_separator()

        return WatsonxResponse(
            patient_id=patient_id,
            final_response=final_response,
            tool_trace=full_trace,
            transcript=transcript,
            raw_response=raw_response,
        )
