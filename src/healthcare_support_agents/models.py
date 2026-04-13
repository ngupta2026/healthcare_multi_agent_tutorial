from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class DischargeChecklist:
    patient_id: str
    checklist: list[str]
    medication_schedule: list[str]
    safety_tips: list[str]
    source_summary: str


@dataclass(slots=True)
class RiskAssessment:
    patient_id: str
    triage_level: str
    concerning_signals: list[str]
    recommended_action: str
    escalate_to_nurse: bool


@dataclass(slots=True)
class CoordinationPlan:
    patient_id: str
    medication_status: list[str]
    appointment_status: list[str]
    barriers: list[str]
    resolved_actions: list[str]


@dataclass(slots=True)
class RecoverySummary:
    patient_id: str
    recovery_summary: str
    checklist: list[str] = field(default_factory=list)
    risk_status: str = ""
    coordination_status: list[str] = field(default_factory=list)
    reasoning_log: list[str] = field(default_factory=list)
    escalated: bool = False
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class WatsonxResponse:
    patient_id: str
    final_response: str
    tool_trace: list[dict[str, Any]] = field(default_factory=list)
    transcript: list[dict[str, Any]] = field(default_factory=list)
    raw_response: dict[str, Any] = field(default_factory=dict)
