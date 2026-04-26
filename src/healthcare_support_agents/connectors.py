from __future__ import annotations

from .models import CoordinationPlan, DischargeChecklist, RiskAssessment
from .repository import DataRepository


class DiscoveryConnector:
    def __init__(self, repository: DataRepository) -> None:
        self.repository = repository

    def translate_discharge_plan(self, patient_id: str) -> DischargeChecklist:
        patient = self.repository.patients[patient_id]
        plan = self.repository.discharge_plans[patient_id]
        clinical_note = plan["clinical_note"].lower()

        medication_schedule: list[str] = []
        if "furosemide" in clinical_note:
            medication_schedule.append("7:00 AM: Take furosemide 20 mg with water before breakfast.")
        if "cephalexin" in clinical_note:
            medication_schedule.extend(
                [
                    "7:30 AM: Take cephalexin 500 mg with breakfast.",
                    "12:30 PM: Take cephalexin 500 mg with lunch.",
                    "6:00 PM: Take cephalexin 500 mg with dinner.",
                    "10:00 PM: Take cephalexin 500 mg before bed.",
                ]
            )

        checklist = [
            f"Follow this routine: {plan['daily_routine']}",
            "Check in once in the morning and once in the evening on symptoms.",
        ]
        if "weight" in clinical_note:
            checklist.append("Weigh yourself after waking up and before breakfast.")
        if "elevated" in clinical_note:
            checklist.append("Keep your leg elevated whenever you are sitting down.")
        if "walker" in clinical_note:
            checklist.append("Use the walker for every transfer, even for short bathroom trips.")

        safety_tips = []
        if patient["literacy_level"] == "plain_language":
            if "sodium" in clinical_note:
                safety_tips.append("Choose low-salt foods today and avoid canned or very salty meals.")
            if "trouble breathing" in clinical_note:
                safety_tips.append("Get urgent help if breathing becomes hard while resting.")
            if "fainting" in clinical_note or "confusion" in clinical_note:
                safety_tips.append("Call for help right away if you faint or feel newly confused.")
            if "fever above 100.4" in clinical_note:
                safety_tips.append("Contact the care team the same day if your temperature goes above 100.4 F.")

        return DischargeChecklist(
            patient_id=patient_id,
            checklist=checklist,
            medication_schedule=medication_schedule,
            safety_tips=safety_tips,
            source_summary=plan["summary"],
        )


class MonitoringConnector:
    HIGH_RISK_KEYWORDS = ("short of breath", "chest pain", "faint", "confusion", "passed out")
    MODERATE_RISK_KEYWORDS = ("dizzy", "swelling", "worse", "fever", "redness")
    REASSURING_PHRASES = ("no fever", "breathing is normal", "no trouble breathing", "not dizzy")

    def __init__(self, repository: DataRepository) -> None:
        self.repository = repository

    def assess_health_shift(self, patient_id: str, symptom_report: str) -> RiskAssessment:
        vitals = self.repository.vitals[patient_id]
        normalized = symptom_report.lower()
        concerning_signals: list[str] = []
        triage_level = "routine"
        escalate_to_nurse = False
        filtered_report = normalized

        for phrase in self.REASSURING_PHRASES:
            filtered_report = filtered_report.replace(phrase, "")

        if any(keyword in filtered_report for keyword in self.HIGH_RISK_KEYWORDS):
            triage_level = "high_priority_health_shift"
            escalate_to_nurse = True
            concerning_signals.append("Patient reported a red-flag symptom in the symptom check-in.")

        if vitals["oxygen_saturation"] is not None and vitals["oxygen_saturation"] < 92:
            triage_level = "high_priority_health_shift"
            escalate_to_nurse = True
            concerning_signals.append(f"Oxygen saturation is low at {vitals['oxygen_saturation']}%.")

        temp_high = vitals["temperature_f"] is not None and vitals["temperature_f"] >= 100.4
        hr_high = vitals["heart_rate"] is not None and vitals["heart_rate"] >= 110
        weight_up = vitals["weight_delta_lb"] is not None and vitals["weight_delta_lb"] > 2
        if temp_high or hr_high or weight_up:
            if triage_level != "high_priority_health_shift":
                triage_level = "watch_closely"
            if temp_high:
                concerning_signals.append(f"Temperature is elevated at {vitals['temperature_f']} F.")
            if hr_high:
                concerning_signals.append(f"Heart rate is elevated at {vitals['heart_rate']} bpm.")
            if weight_up:
                concerning_signals.append(
                    f"Weight increased by {vitals['weight_delta_lb']} pounds since the prior check-in."
                )

        if any(keyword in filtered_report for keyword in self.MODERATE_RISK_KEYWORDS) and not concerning_signals:
            triage_level = "watch_closely"
            concerning_signals.append("Reported symptoms suggest a change that should be monitored today.")

        if triage_level == "high_priority_health_shift":
            recommended_action = "Escalate to the on-call nurse now and advise immediate clinical review."
        elif triage_level == "watch_closely":
            recommended_action = "Monitor closely today, repeat the symptom check-in, and notify the care team if symptoms worsen."
        else:
            recommended_action = "Continue the routine recovery plan and document the check-in as stable."

        return RiskAssessment(
            patient_id=patient_id,
            triage_level=triage_level,
            concerning_signals=concerning_signals,
            recommended_action=recommended_action,
            escalate_to_nurse=escalate_to_nurse,
        )


class LogisticsConnector:
    def __init__(self, repository: DataRepository) -> None:
        self.repository = repository

    def coordinate_care(self, patient_id: str) -> CoordinationPlan:
        pharmacy = self.repository.pharmacy_status[patient_id]
        appointments = self.repository.appointments[patient_id]

        medication_status: list[str] = []
        appointment_status: list[str] = []
        barriers: list[str] = []
        resolved_actions: list[str] = []

        for medication in pharmacy["medications"]:
            if medication["status"] == "delayed":
                medication_status.append(
                    f"{medication['name']} is delayed. Reason: {medication['delay_reason']}"
                )
                barriers.append(f"Medication delay for {medication['name']}.")
                if medication["alternate_source"]:
                    resolved_actions.append(
                        f"Suggested alternate fill option: {medication['alternate_source']}"
                    )
            else:
                medication_status.append(f"{medication['name']} is {medication['status'].replace('_', ' ')}.")

        for appointment in appointments["appointments"]:
            appointment_status.append(
                f"{appointment['specialty']} follow-up on {appointment['scheduled_at']} with transport status "
                f"{appointment['transportation_status'].replace('_', ' ')}."
            )
            if appointment["transportation_status"] == "needs_booking":
                barriers.append(f"Transportation is not set for {appointment['specialty']}.")
                resolved_actions.append(
                    f"Flagged transport booking need: {appointment['transportation_note']}"
                )

        return CoordinationPlan(
            patient_id=patient_id,
            medication_status=medication_status,
            appointment_status=appointment_status,
            barriers=barriers,
            resolved_actions=resolved_actions,
        )
