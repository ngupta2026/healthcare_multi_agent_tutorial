from __future__ import annotations

from dataclasses import asdict

from .agents import build_agent_stack
from .models import RecoverySummary
from .repository import DataRepository


class Orchestrator:
    def __init__(self, repository: DataRepository) -> None:
        self.repository = repository
        self.translator_agent, self.monitoring_agent, self.logistics_agent = build_agent_stack(repository)

    def resolve(self, patient_id: str, symptom_report: str) -> RecoverySummary:
        reasoning_log: list[str] = []

        if patient_id not in self.repository.patients:
            reasoning_log.append("Patient ID was not found in the recovery registry.")
            return RecoverySummary(
                patient_id=patient_id,
                recovery_summary="Please provide a valid patient ID before the care workflow can continue.",
                reasoning_log=reasoning_log,
                escalated=False,
                risk_status="unknown",
            )

        patient = self.repository.patients[patient_id]
        checklist = self.translator_agent.translate_discharge(patient_id)
        reasoning_log.append("Translated discharge instructions into a plain-language recovery checklist.")

        risk = self.monitoring_agent.monitor(patient_id, symptom_report)
        reasoning_log.append(f"Risk monitoring classified the current status as {risk.triage_level}.")

        logistics = self.logistics_agent.coordinate(patient_id)
        reasoning_log.append("Checked pharmacy and appointment coordination status.")

        if logistics.barriers:
            reasoning_log.append("Logistical barriers were identified and mitigation actions were generated.")

        if risk.escalate_to_nurse:
            reasoning_log.append("Safety checkpoint triggered a human nurse escalation.")
            summary = (
                f"{patient['name']} shows a high-priority health shift. Escalate to a nurse now. "
                f"Continue the checklist only while arranging immediate clinical follow-up."
            )
        elif risk.triage_level == "watch_closely":
            summary = (
                f"{patient['name']} is not in immediate crisis, but today's recovery needs closer observation. "
                f"Continue the care plan and repeat symptom monitoring later today."
            )
        else:
            summary = (
                f"{patient['name']} appears to be on a routine recovery path. Continue the daily checklist, "
                f"medication plan, and scheduled follow-up."
            )

        coordination_status = logistics.medication_status + logistics.appointment_status + logistics.resolved_actions

        return RecoverySummary(
            patient_id=patient_id,
            recovery_summary=summary,
            checklist=checklist.checklist + checklist.medication_schedule + checklist.safety_tips,
            risk_status=risk.triage_level,
            coordination_status=coordination_status,
            reasoning_log=reasoning_log,
            escalated=risk.escalate_to_nurse,
            details={
                "patient": patient,
                "translator_output": asdict(checklist),
                "monitoring_output": asdict(risk),
                "logistics_output": asdict(logistics),
                "symptom_report": symptom_report,
            },
        )
