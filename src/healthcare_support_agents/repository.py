from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class DataRepository:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.patients = self._load_index("patients.json", "patient_id")
        self.discharge_plans = self._load_index("discharge_plans.json", "patient_id")
        self.vitals = self._load_index("vitals.json", "patient_id")
        self.pharmacy_status = self._load_index("pharmacy_status.json", "patient_id")
        self.appointments = self._load_index("appointments.json", "patient_id")

    def _load_index(self, filename: str, key_field: str) -> dict[str, dict[str, Any]]:
        with (self.data_dir / filename).open("r", encoding="utf-8") as handle:
            records = json.load(handle)
        return {record[key_field]: record for record in records}
