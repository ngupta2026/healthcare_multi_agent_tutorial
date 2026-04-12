from __future__ import annotations

import unittest
from pathlib import Path

from healthcare_support_agents.orchestrator import Orchestrator
from healthcare_support_agents.repository import DataRepository


def build_orchestrator() -> Orchestrator:
    repo_root = Path(__file__).resolve().parents[1]
    return Orchestrator(DataRepository(repo_root / "data"))


class AppTests(unittest.TestCase):
    def test_routine_recovery_case_stays_unescalated(self) -> None:
        orchestrator = build_orchestrator()
        result = orchestrator.resolve("PT-1001", "A little tired after walking, but no fever and breathing is normal.")
        self.assertFalse(result.escalated)
        self.assertEqual(result.risk_status, "routine")

    def test_red_flag_symptom_escalates_to_nurse(self) -> None:
        orchestrator = build_orchestrator()
        result = orchestrator.resolve("PT-1001", "I feel short of breath and dizzy this morning.")
        self.assertTrue(result.escalated)
        self.assertEqual(result.risk_status, "high_priority_health_shift")

    def test_logistics_plan_surfaces_medication_delay(self) -> None:
        orchestrator = build_orchestrator()
        result = orchestrator.resolve("PT-2002", "My leg is more swollen and I missed my antibiotic pickup.")
        combined = " ".join(result.coordination_status).lower()
        self.assertIn("alternate fill option", combined)
        self.assertIn("transport", combined)


if __name__ == "__main__":
    unittest.main()
