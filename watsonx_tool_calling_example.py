from __future__ import annotations

import json
from dataclasses import asdict

from healthcare_support_agents.config import AppConfig
from healthcare_support_agents.serper_client import SerperSearchClient
from healthcare_support_agents.tool_wrappers import HealthcareToolRuntime, build_repository, build_watsonx_payload_example
from healthcare_support_agents.watsonx_client import WatsonxClient
from healthcare_support_agents.watsonx_orchestrator import WatsonxCareOrchestrator


def main() -> None:
    config = AppConfig.from_env()
    repository = build_repository()
    search_client = SerperSearchClient(config.serper_api_key) if config.serper_ready else None
    runtime = HealthcareToolRuntime(repository, search_client=search_client)

    patient_id = "PT-1001"
    symptom_report = "I feel short of breath and dizzy this morning."

    print("IBM watsonx tool-calling payload example:")
    print(json.dumps(build_watsonx_payload_example(patient_id, symptom_report, runtime), indent=2))

    if not config.watsonx_ready:
        print("\nwatsonx credentials are not configured. Add a .env file from .env.example to run the live API flow.")
        return

    coordinator = WatsonxCareOrchestrator(WatsonxClient(config), runtime)
    result = coordinator.resolve(patient_id, symptom_report)
    print("\nLive watsonx result:")
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
