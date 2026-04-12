from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .orchestrator import Orchestrator
from .repository import DataRepository


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "data"
    repository = DataRepository(data_dir)
    orchestrator = Orchestrator(repository)

    sample_cases = [
        ("PT-1001", "A little tired after walking, but no fever and breathing is normal."),
        ("PT-1001", "I feel short of breath and dizzy this morning."),
        ("PT-2002", "My leg is more swollen and I missed my antibiotic pickup."),
    ]

    for patient_id, symptom_report in sample_cases:
        print(f"PATIENT: {patient_id}")
        print(f"SYMPTOMS: {symptom_report}")
        result = orchestrator.resolve(patient_id, symptom_report)
        print(json.dumps(asdict(result), indent=2))
        print("-" * 72)


if __name__ == "__main__":
    main()
