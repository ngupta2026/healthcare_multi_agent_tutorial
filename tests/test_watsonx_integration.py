from __future__ import annotations

import unittest
from pathlib import Path

from healthcare_support_agents.config import AppConfig
from healthcare_support_agents.repository import DataRepository
from healthcare_support_agents.tool_wrappers import HealthcareToolRuntime
from healthcare_support_agents.watsonx_orchestrator import WatsonxCareOrchestrator


class FakeWatsonxClient:
    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages, tools, tool_choice_option="auto"):
        self.calls += 1
        if self.calls == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "tool-1",
                                    "type": "function",
                                    "function": {
                                        "name": "translate_discharge_plan",
                                        "arguments": '{"patient_id": "PT-1001"}',
                                    },
                                },
                                {
                                    "id": "tool-2",
                                    "type": "function",
                                    "function": {
                                        "name": "monitor_recovery_status",
                                        "arguments": (
                                            '{"patient_id": "PT-1001", '
                                            '"symptom_report": "I feel short of breath and dizzy this morning."}'
                                        ),
                                    },
                                },
                            ]
                        }
                    }
                ]
            }
        return {
            "choices": [
                {
                    "message": {
                        "content": [
                            {
                                "type": "text",
                                "text": "Escalate to the on-call nurse and continue the safety checklist while arranging follow-up.",
                            }
                        ]
                    }
                }
            ]
        }


def build_runtime() -> HealthcareToolRuntime:
    repo_root = Path(__file__).resolve().parents[1]
    return HealthcareToolRuntime(DataRepository(repo_root / "data"))


class WatsonxIntegrationTests(unittest.TestCase):
    def test_config_normalizes_model_prefix(self) -> None:
        config = AppConfig(
            watsonx_apikey="abc",
            watsonx_project_id="project",
            watsonx_url="https://us-south.ml.cloud.ibm.com",
            watsonx_model="watsonx/ibm/granite-3-8b-instruct",
            serper_api_key=None,
        )
        self.assertEqual(config.normalized_model, "ibm/granite-3-8b-instruct")

    def test_tool_definitions_include_recovery_tools(self) -> None:
        runtime = build_runtime()
        names = {item["function"]["name"] for item in runtime.build_tool_definitions()}
        self.assertIn("translate_discharge_plan", names)
        self.assertIn("monitor_recovery_status", names)
        self.assertIn("coordinate_care_logistics", names)

    def test_watsonx_orchestrator_executes_tool_calls(self) -> None:
        runtime = build_runtime()
        orchestrator = WatsonxCareOrchestrator(FakeWatsonxClient(), runtime)
        result = orchestrator.resolve("PT-1001", "I feel short of breath and dizzy this morning.")
        self.assertIn("nurse", result.final_response.lower())
        self.assertEqual(len(result.tool_trace), 2)
        self.assertEqual(result.tool_trace[0]["tool_name"], "translate_discharge_plan")


if __name__ == "__main__":
    unittest.main()
