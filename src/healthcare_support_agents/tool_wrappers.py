from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from .agents import build_agent_stack
from .orchestrator import Orchestrator
from .repository import DataRepository
from .serper_client import SerperSearchClient


def _serialize(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    return value


def build_repository() -> DataRepository:
    repo_root = Path(__file__).resolve().parents[2]
    return DataRepository(repo_root / "data")


class HealthcareToolRuntime:
    def __init__(self, repository: DataRepository, search_client: SerperSearchClient | None = None) -> None:
        self.repository = repository
        self.search_client = search_client
        self.translator_agent, self.monitoring_agent, self.logistics_agent = build_agent_stack(repository)
        self.orchestrator = Orchestrator(repository)

    def get_patient_snapshot(self, patient_id: str) -> dict[str, Any]:
        if patient_id not in self.repository.patients:
            raise ValueError(f"Unknown patient_id: {patient_id}")

        return {
            "patient": self.repository.patients[patient_id],
            "discharge_plan": self.repository.discharge_plans[patient_id],
            "latest_vitals": self.repository.vitals[patient_id],
            "pharmacy_status": self.repository.pharmacy_status[patient_id],
            "appointments": self.repository.appointments[patient_id],
        }

    def translate_discharge_plan(self, patient_id: str) -> dict[str, Any]:
        return _serialize(self.translator_agent.translate_discharge(patient_id))

    def monitor_recovery_status(self, patient_id: str, symptom_report: str) -> dict[str, Any]:
        return _serialize(self.monitoring_agent.monitor(patient_id, symptom_report))

    def coordinate_care_logistics(self, patient_id: str) -> dict[str, Any]:
        return _serialize(self.logistics_agent.coordinate(patient_id))

    def resolve_recovery_case(self, patient_id: str, symptom_report: str) -> dict[str, Any]:
        return _serialize(self.orchestrator.resolve(patient_id, symptom_report))

    def search_support_services(self, query: str) -> dict[str, Any]:
        if not self.search_client:
            raise RuntimeError("SERPER_API_KEY is not configured for public web search.")
        return self.search_client.search(query=query)

    def build_tool_definitions(self) -> list[dict[str, Any]]:
        definitions = [
            {
                "type": "function",
                "function": {
                    "name": "get_patient_snapshot",
                    "description": "Retrieve the patient profile, discharge plan, vitals, pharmacy status, and appointments for one patient ID.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "patient_id": {
                                "type": "string",
                                "description": "Patient identifier such as PT-1001.",
                            }
                        },
                        "required": ["patient_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "translate_discharge_plan",
                    "description": "Convert dense discharge instructions into a plain-language checklist, medication schedule, and safety tips.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "patient_id": {
                                "type": "string",
                                "description": "Patient identifier for the discharge plan under review.",
                            }
                        },
                        "required": ["patient_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "monitor_recovery_status",
                    "description": "Assess symptom reports and biometrics to decide whether recovery is routine, needs close watching, or requires nurse escalation.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "patient_id": {
                                "type": "string",
                                "description": "Patient identifier to monitor.",
                            },
                            "symptom_report": {
                                "type": "string",
                                "description": "Latest patient or caregiver symptom report.",
                            },
                        },
                        "required": ["patient_id", "symptom_report"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "coordinate_care_logistics",
                    "description": "Review medication fill status, transport readiness, and follow-up scheduling barriers for a patient.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "patient_id": {
                                "type": "string",
                                "description": "Patient identifier for logistics coordination.",
                            }
                        },
                        "required": ["patient_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "resolve_recovery_case",
                    "description": "Run the full local healthcare recovery workflow and return the consolidated summary payload.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "patient_id": {
                                "type": "string",
                                "description": "Patient identifier for the case.",
                            },
                            "symptom_report": {
                                "type": "string",
                                "description": "Latest patient or caregiver symptom report.",
                            },
                        },
                        "required": ["patient_id", "symptom_report"],
                    },
                },
            },
        ]

        if self.search_client:
            definitions.append(
                {
                    "type": "function",
                    "function": {
                        "name": "search_support_services",
                        "description": "Search current public web results for general support services or logistics resources such as transportation or pharmacy options.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "Public web search query for support resources.",
                                }
                            },
                            "required": ["query"],
                        },
                    },
                }
            )

        return definitions

    def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        handlers = {
            "get_patient_snapshot": lambda args: self.get_patient_snapshot(args["patient_id"]),
            "translate_discharge_plan": lambda args: self.translate_discharge_plan(args["patient_id"]),
            "monitor_recovery_status": lambda args: self.monitor_recovery_status(
                args["patient_id"],
                args["symptom_report"],
            ),
            "coordinate_care_logistics": lambda args: self.coordinate_care_logistics(args["patient_id"]),
            "resolve_recovery_case": lambda args: self.resolve_recovery_case(
                args["patient_id"],
                args["symptom_report"],
            ),
            "search_support_services": lambda args: self.search_support_services(args["query"]),
        }
        if tool_name not in handlers:
            raise ValueError(f"Unsupported tool: {tool_name}")
        return handlers[tool_name](arguments)


def build_watsonx_payload_example(patient_id: str, symptom_report: str, runtime: HealthcareToolRuntime) -> dict[str, Any]:
    return {
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a healthcare care orchestrator. Use tools before giving a patient-specific answer.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Review patient {patient_id}. Symptom report: {symptom_report}. "
                            "Use the tools to gather the case details and produce a concise clinician-ready summary."
                        ),
                    }
                ],
            },
        ],
        "tools": runtime.build_tool_definitions(),
        "tool_choice_option": "auto",
    }


def tool_message(tool_call_id: str, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": [
            {
                "type": "text",
                "text": json.dumps(result),
            }
        ],
    }
