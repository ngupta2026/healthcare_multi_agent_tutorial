from __future__ import annotations

from .connectors import DiscoveryConnector, LogisticsConnector, MonitoringConnector
from .models import CoordinationPlan, DischargeChecklist, RiskAssessment
from .repository import DataRepository


class TranslatorAgent:
    def __init__(self, connector: DiscoveryConnector) -> None:
        self.connector = connector

    def translate_discharge(self, patient_id: str) -> DischargeChecklist:
        return self.connector.translate_discharge_plan(patient_id)


class MonitoringAgent:
    def __init__(self, connector: MonitoringConnector) -> None:
        self.connector = connector

    def monitor(self, patient_id: str, symptom_report: str) -> RiskAssessment:
        return self.connector.assess_health_shift(patient_id, symptom_report)


class LogisticsAgent:
    def __init__(self, connector: LogisticsConnector) -> None:
        self.connector = connector

    def coordinate(self, patient_id: str) -> CoordinationPlan:
        return self.connector.coordinate_care(patient_id)


def build_agent_stack(repository: DataRepository) -> tuple[TranslatorAgent, MonitoringAgent, LogisticsAgent]:
    translator = TranslatorAgent(DiscoveryConnector(repository))
    monitoring = MonitoringAgent(MonitoringConnector(repository))
    logistics = LogisticsAgent(LogisticsConnector(repository))
    return translator, monitoring, logistics
