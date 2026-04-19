from __future__ import annotations

import json
from pathlib import Path
from typing import Any


FALLBACK_DATA: dict[str, list[dict[str, Any]]] = {
    "patients.json": [
        {
            "patient_id": "PT-1001",
            "name": "Margaret Ellis",
            "literacy_level": "plain_language",
            "discharge_plan_id": "DP-1001",
            "care_plan_id": "CP-1001",
            "pharmacy_plan_id": "RX-1001",
            "appointment_plan_id": "AP-1001",
        },
        {
            "patient_id": "PT-2002",
            "name": "David Ramirez",
            "literacy_level": "plain_language",
            "discharge_plan_id": "DP-2002",
            "care_plan_id": "CP-2002",
            "pharmacy_plan_id": "RX-2002",
            "appointment_plan_id": "AP-2002",
        },
    ],
    "discharge_plans.json": [
        {
            "discharge_plan_id": "DP-1001",
            "patient_id": "PT-1001",
            "summary": "Discharged after congestive heart failure stabilization with medication adjustment.",
            "clinical_note": "Take furosemide 20 mg by mouth every morning. Check weight after waking and before breakfast. If weight increases by more than 2 pounds in 24 hours, call cardiology. Limit sodium to less than 2 grams daily. Mild fatigue can occur for one week. Seek urgent care for chest pain, trouble breathing at rest, fainting, or new confusion.",
            "daily_routine": "Wakes at 7:00 AM, eats breakfast at 8:00 AM, afternoon walk at 2:00 PM, bedtime at 9:30 PM.",
        },
        {
            "discharge_plan_id": "DP-2002",
            "patient_id": "PT-2002",
            "summary": "Discharged after cellulitis treatment with oral antibiotics and mobility precautions.",
            "clinical_note": "Take cephalexin 500 mg four times daily for 7 days. Keep the leg elevated while seated. Mild redness may improve slowly, but spreading redness, fever above 100.4 F, or worsening swelling needs same-day review. Use the walker for all transfers to reduce fall risk.",
            "daily_routine": "Wakes at 6:30 AM, breakfast at 7:30 AM, lunch at 12:30 PM, dinner at 6:00 PM, bedtime at 10:00 PM.",
        },
    ],
    "vitals.json": [
        {
            "patient_id": "PT-1001",
            "heart_rate": 96,
            "oxygen_saturation": 95,
            "temperature_f": 98.6,
            "systolic_bp": 126,
            "weight_delta_lb": 0.5,
        },
        {
            "patient_id": "PT-2002",
            "heart_rate": 108,
            "oxygen_saturation": 97,
            "temperature_f": 100.6,
            "systolic_bp": 118,
            "weight_delta_lb": 0.0,
        },
    ],
    "pharmacy_status.json": [
        {
            "pharmacy_plan_id": "RX-1001",
            "patient_id": "PT-1001",
            "medications": [
                {
                    "name": "furosemide 20 mg",
                    "status": "ready_for_pickup",
                    "delay_reason": "",
                    "alternate_source": "",
                }
            ],
        },
        {
            "pharmacy_plan_id": "RX-2002",
            "patient_id": "PT-2002",
            "medications": [
                {
                    "name": "cephalexin 500 mg",
                    "status": "delayed",
                    "delay_reason": "Primary pharmacy is out of stock until tomorrow afternoon.",
                    "alternate_source": "Greenview Pharmacy, 1.2 miles away, has a same-day fill available.",
                }
            ],
        },
    ],
    "appointments.json": [
        {
            "appointment_plan_id": "AP-1001",
            "patient_id": "PT-1001",
            "appointments": [
                {
                    "specialty": "Cardiology",
                    "scheduled_at": "2026-04-15T10:30:00",
                    "transportation_status": "confirmed",
                    "transportation_note": "Daughter will drive to the clinic.",
                }
            ],
        },
        {
            "appointment_plan_id": "AP-2002",
            "patient_id": "PT-2002",
            "appointments": [
                {
                    "specialty": "Primary Care",
                    "scheduled_at": "2026-04-13T14:00:00",
                    "transportation_status": "needs_booking",
                    "transportation_note": "Paratransit request was not submitted.",
                }
            ],
        },
    ],
}


class DataRepository:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.patients = self._load_index("patients.json", "patient_id")
        self.discharge_plans = self._load_index("discharge_plans.json", "patient_id")
        self.vitals = self._load_index("vitals.json", "patient_id")
        self.pharmacy_status = self._load_index("pharmacy_status.json", "patient_id")
        self.appointments = self._load_index("appointments.json", "patient_id")

    def _load_index(self, filename: str, key_field: str) -> dict[str, dict[str, Any]]:
        file_path = self.data_dir / filename
        if file_path.exists():
            with file_path.open("r", encoding="utf-8") as handle:
                records = json.load(handle)
        else:
            records = FALLBACK_DATA.get(filename)
            if records is None:
                raise FileNotFoundError(f"Unable to load {filename} from {self.data_dir} and no fallback exists.")
        return {record[key_field]: record for record in records}
