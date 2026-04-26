from __future__ import annotations

import json
from typing import Any

from .agent_logger import get_logger
from .tool_wrappers import HealthcareToolRuntime, tool_message
from .watsonx_client import WatsonxClient

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseAgent:
    """Single-responsibility LLM-backed agent with a scoped system prompt and
    a narrowly chosen set of tool definitions."""

    SYSTEM_PROMPT: str = ""
    MAX_ROUNDS: int = 4

    def __init__(self, client: WatsonxClient, runtime: HealthcareToolRuntime) -> None:
        self.client = client
        self.runtime = runtime

    # Sub-classes override this to expose only the tools they need.
    def _tool_definitions(self) -> list[dict[str, Any]]:
        return []

    def _run_loop(
        self,
        user_text: str,
        extra_tools: list[dict[str, Any]] | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Run the tool-calling loop and return (final_text, trace)."""
        agent_name = self.__class__.__name__
        tools = self._tool_definitions() + (extra_tools or [])
        tool_names = [t["function"]["name"] for t in tools]

        _log.debug(
            "[%s] Loop initialised.\n"
            "  What this agent does: %s\n"
            "  Tools available to this agent: %s\n"
            "  User prompt being sent to the LLM: %.200s",
            agent_name,
            self.__class__.__doc__,
            tool_names,
            user_text,
        )

        transcript: list[dict[str, Any]] = [
            {"role": "system", "content": [{"type": "text", "text": self.SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "text", "text": user_text}]},
        ]
        trace: list[dict[str, Any]] = []

        for round_num in range(self.MAX_ROUNDS):
            _log.debug(
                "[%s] Round %d/%d — Sending conversation to IBM WatsonX LLM.\n"
                "  The LLM will decide: either call one of the tools to fetch data,\n"
                "  or return a final text response if it has enough information.",
                agent_name, round_num + 1, self.MAX_ROUNDS,
            )
            raw = self.client.chat(messages=transcript, tools=tools, tool_choice_option="auto")
            choices = raw.get("choices", [])
            if not choices:
                _log.warning(
                    "[%s] Round %d — IBM WatsonX returned an empty response (no choices).\n"
                    "  This can happen if the model is unavailable or the request was malformed.\n"
                    "  Stopping the loop early and returning an empty result.",
                    agent_name, round_num + 1,
                )
                break
            message = choices[0].get("message", {})
            tool_calls = message.get("tool_calls") or []

            if not tool_calls:
                # LLM produced a final text response — no more tool calls needed
                content = message.get("content", "")
                if isinstance(content, list):
                    parts = [c.get("text", "") for c in content if isinstance(c, dict)]
                    final_text = " ".join(parts).strip()
                else:
                    final_text = (content or "").strip()
                _log.info(
                    "[%s] FINISHED in %d round(s).\n"
                    "  The LLM decided it had enough information and returned a final answer.\n"
                    "  Agent output (first 300 chars): %.300s",
                    agent_name, round_num + 1, final_text,
                )
                return final_text, trace

            # LLM requested one or more tool calls
            _log.debug(
                "[%s] Round %d — The LLM requested %d tool call(s).\n"
                "  The LLM needs real data before it can give a final answer,\n"
                "  so it is calling healthcare data tools to retrieve that information.",
                agent_name, round_num + 1, len(tool_calls),
            )
            assistant_msg: dict[str, Any] = {"role": "assistant", "tool_calls": tool_calls}
            if message.get("content"):
                assistant_msg["content"] = message["content"]
            transcript.append(assistant_msg)

            for call in tool_calls:
                fn_name = call["function"]["name"]
                raw_args = call.get("function", {}).get("arguments", "{}")
                try:
                    args = json.loads(raw_args) if raw_args else {}
                    _log.info(
                        "[%s] TOOL CALLED: %s\n"
                        "  Why: The LLM needs specific healthcare data to complete its task.\n"
                        "  Input arguments passed to the tool: %s",
                        agent_name, fn_name,
                        json.dumps(args, indent=4, ensure_ascii=False),
                    )
                    result = self.runtime.execute_tool(fn_name, args)
                    _log.debug(
                        "[%s] TOOL RESULT received from: %s\n"
                        "  The tool returned the following data (first 300 chars):\n"
                        "  %s",
                        agent_name, fn_name,
                        json.dumps(result, indent=4, ensure_ascii=False)[:300],
                    )
                except Exception as exc:
                    args = {}
                    result = {"error": str(exc)}
                    _log.error(
                        "[%s] TOOL ERROR from: %s\n"
                        "  The tool execution failed. The error will be fed back to the LLM\n"
                        "  so it can handle the failure gracefully.\n"
                        "  Error details: %s",
                        agent_name, fn_name, exc, exc_info=True,
                    )
                trace.append({"agent": agent_name, "tool": fn_name, "args": args, "result": result})
                transcript.append(tool_message(call["id"], result))

        _log.warning(
            "[%s] MAX ROUNDS REACHED (%d rounds used).\n"
            "  The agent used all allowed LLM rounds without producing a final response.\n"
            "  This may mean the LLM kept requesting tools without converging.\n"
            "  Returning an empty string — the orchestrator will use a fallback.",
            agent_name, self.MAX_ROUNDS,
        )
        return "", trace


# ---------------------------------------------------------------------------
# Agent 1 – Patient Context Agent
# ---------------------------------------------------------------------------

class PatientContextAgent(BaseAgent):
    """Retrieves and summarises the full patient snapshot: profile, discharge
    plan, current vitals, pharmacy status, and appointments."""

    SYSTEM_PROMPT = (
        "You are the Patient Context Agent. Your only job is to retrieve the "
        "complete patient record for the requested patient ID and return a "
        "concise structured summary covering: patient profile, discharge plan "
        "summary, latest vitals, pharmacy fill status, and upcoming appointments. "
        "Do not assess risk or recommend actions. Report facts only."
    )

    def _tool_definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_patient_snapshot",
                    "description": "Retrieve the full patient profile, discharge plan, vitals, pharmacy status, and appointments.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "patient_id": {"type": "string", "description": "Patient identifier such as PT-1001."}
                        },
                        "required": ["patient_id"],
                    },
                },
            }
        ]

    def run(self, patient_id: str) -> tuple[str, list[dict[str, Any]]]:
        _log.info(
            "[PatientContextAgent] STARTED for patient_id=%s\n"
            "  Purpose: Retrieve the complete patient record from the data store.\n"
            "  This includes: demographics, discharge plan, current vitals,\n"
            "  pharmacy fill status, and upcoming appointments.\n"
            "  This is always the FIRST step — other agents depend on this context.",
            patient_id,
        )
        prompt = (
            f"Retrieve and summarise the full patient record for patient_id '{patient_id}'. "
            "Use the get_patient_snapshot tool."
        )
        return self._run_loop(prompt)


# ---------------------------------------------------------------------------
# Agent 2 – Discharge Translator Agent
# ---------------------------------------------------------------------------

class DischargeTranslatorAgent(BaseAgent):
    """Converts dense discharge notes into a plain-language daily checklist,
    medication schedule, and safety tips."""

    SYSTEM_PROMPT = (
        "You are the Discharge Translator Agent. Your only job is to convert "
        "the patient's clinical discharge instructions into plain, easy-to-follow "
        "language. Produce three sections: daily_checklist, medication_schedule, "
        "and safety_tips. Use simple words a non-medical person can understand. "
        "Do not diagnose or assess risk."
    )

    def _tool_definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "translate_discharge_plan",
                    "description": "Convert the patient's discharge instructions into a plain-language checklist, medication schedule, and safety tips.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "patient_id": {"type": "string", "description": "Patient identifier for the discharge plan."}
                        },
                        "required": ["patient_id"],
                    },
                },
            }
        ]

    def run(self, patient_id: str) -> tuple[str, list[dict[str, Any]]]:
        _log.info(
            "[DischargeTranslatorAgent] STARTED for patient_id=%s\n"
            "  Purpose: Convert the hospital discharge instructions into plain, everyday language.\n"
            "  Discharge documents are often written in clinical shorthand that patients\n"
            "  struggle to understand. This agent rewrites them as:\n"
            "    - A daily checklist of tasks\n"
            "    - A clear medication schedule\n"
            "    - Simple safety warning signs to watch for",
            patient_id,
        )
        prompt = (
            f"Translate the discharge plan for patient_id '{patient_id}' into plain language. "
            "Use the translate_discharge_plan tool. Return the result as daily_checklist, "
            "medication_schedule, and safety_tips sections."
        )
        return self._run_loop(prompt)


# ---------------------------------------------------------------------------
# Agent 3 – Recovery Monitoring Agent
# ---------------------------------------------------------------------------

class RecoveryMonitoringAgent(BaseAgent):
    """Evaluates symptom reports and biometrics to triage the patient as
    routine, watch_closely, or high_priority_health_shift."""

    SYSTEM_PROMPT = (
        "You are the Recovery Monitoring Agent. Your job is to assess the "
        "patient's current health status using their symptom report and stored "
        "vitals. Classify the risk as one of: routine, watch_closely, or "
        "high_priority_health_shift. List all concerning signals and state "
        "clearly whether the patient must be escalated to a nurse immediately. "
        "Do not suggest logistics or discharge instructions."
    )

    def _tool_definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "monitor_recovery_status",
                    "description": "Assess symptom reports and biometrics to decide whether recovery is routine, needs close watching, or requires nurse escalation.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "patient_id": {"type": "string", "description": "Patient identifier to monitor."},
                            "symptom_report": {"type": "string", "description": "Latest patient or caregiver symptom report."},
                        },
                        "required": ["patient_id", "symptom_report"],
                    },
                },
            }
        ]

    def run(self, patient_id: str, symptom_report: str) -> tuple[str, list[dict[str, Any]]]:
        _log.info(
            "[RecoveryMonitoringAgent] STARTED for patient_id=%s\n"
            "  Purpose: Triage the patient's current health status by cross-referencing\n"
            "  the symptom report they provided with their recorded vitals.\n"
            "  The agent will assign one of three risk levels:\n"
            "    - routine            : patient is recovering as expected\n"
            "    - watch_closely      : minor concerns, keep monitoring\n"
            "    - high_priority_health_shift : escalate to a nurse immediately\n"
            "  Symptom report received: %.200s",
            patient_id, symptom_report,
        )
        prompt = (
            f"Assess the recovery status for patient_id '{patient_id}'. "
            f"Symptom report: \"{symptom_report}\". "
            "Use the monitor_recovery_status tool. State the triage_level, "
            "list concerning_signals, and declare whether nurse escalation is required."
        )
        return self._run_loop(prompt)


# ---------------------------------------------------------------------------
# Agent 4 – Care Logistics Agent
# ---------------------------------------------------------------------------

class CareLogisticsAgent(BaseAgent):
    """Reviews medication fill status, transportation readiness, and follow-up
    scheduling barriers, then produces a coordination plan."""

    SYSTEM_PROMPT = (
        "You are the Care Logistics Agent. Your job is to check medication "
        "fill status, upcoming appointments, and transportation readiness for "
        "the patient. Identify any barriers or blockers and propose specific "
        "next actions to resolve them. Do not assess clinical risk or provide "
        "discharge instructions."
    )

    def _tool_definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "coordinate_care_logistics",
                    "description": "Review medication fill status, transport readiness, and follow-up scheduling barriers for a patient.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "patient_id": {"type": "string", "description": "Patient identifier for logistics coordination."}
                        },
                        "required": ["patient_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_support_services",
                    "description": "Search for nearby pharmacies, transportation, or community support services when a barrier is detected.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query for support services."}
                        },
                        "required": ["query"],
                    },
                },
            },
        ]

    def run(self, patient_id: str) -> tuple[str, list[dict[str, Any]]]:
        _log.info(
            "[CareLogisticsAgent] STARTED for patient_id=%s\n"
            "  Purpose: Identify and resolve any practical barriers preventing the patient\n"
            "  from following their recovery plan. This covers:\n"
            "    - Medication: Are prescriptions filled? Any delays or insurance issues?\n"
            "    - Appointments: Are follow-ups scheduled? Any missed visits?\n"
            "    - Transport: Can the patient get to appointments?\n"
            "  If barriers are found, this agent will search for alternative services\n"
            "  (nearby pharmacies, transport options, community support).",
            patient_id,
        )
        prompt = (
            f"Review care logistics for patient_id '{patient_id}'. "
            "Use the coordinate_care_logistics tool. If medication or transport "
            "barriers are found, use search_support_services to find alternatives. "
            "Return medication_status, appointment_status, barriers, and resolved_actions."
        )
        return self._run_loop(prompt)
