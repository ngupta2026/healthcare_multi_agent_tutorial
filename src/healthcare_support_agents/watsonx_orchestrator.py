from __future__ import annotations

import json
from typing import Any

from .models import WatsonxResponse
from .tool_wrappers import HealthcareToolRuntime, tool_message
from .watsonx_client import WatsonxClient


SYSTEM_PROMPT = """
You are a Healthcare Care Orchestrator working in an IBM watsonx-style tool-calling workflow.
Always use tools before making patient-specific claims.
Prioritize patient safety, call out nurse escalation when risk is high, and keep the final answer concise but clinically useful.
""".strip()


class WatsonxCareOrchestrator:
    def __init__(self, client: WatsonxClient, runtime: HealthcareToolRuntime, max_round_trips: int = 6) -> None:
        self.client = client
        self.runtime = runtime
        self.max_round_trips = max_round_trips

    def resolve(self, patient_id: str, symptom_report: str) -> WatsonxResponse:
        transcript: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Patient ID: {patient_id}\n"
                            f"Symptom report: {symptom_report}\n"
                            "Use the available tools to review discharge instructions, risk, and logistics. "
                            "Then provide a final recovery summary for clinical staff."
                        ),
                    }
                ],
            },
        ]
        trace: list[dict[str, Any]] = []
        raw_response: dict[str, Any] = {}

        for _ in range(self.max_round_trips):
            raw_response = self.client.chat(
                messages=transcript,
                tools=self.runtime.build_tool_definitions(),
                tool_choice_option="auto",
            )
            choices = raw_response.get("choices", [])
            if not choices:
                raise RuntimeError("watsonx.ai returned no choices in the chat response.")

            message = choices[0].get("message", {})
            tool_calls = message.get("tool_calls") or []

            if tool_calls:
                assistant_message: dict[str, Any] = {
                    "role": "assistant",
                    "tool_calls": tool_calls,
                }
                if message.get("content"):
                    assistant_message["content"] = message["content"]
                transcript.append(assistant_message)

                for call in tool_calls:
                    arguments_raw = call.get("function", {}).get("arguments", "{}")
                    try:
                        arguments = json.loads(arguments_raw) if arguments_raw else {}
                    except json.JSONDecodeError as exc:
                        arguments = {}
                        result = {"error": f"Invalid tool arguments: {exc}"}
                    else:
                        try:
                            result = self.runtime.execute_tool(call["function"]["name"], arguments)
                        except Exception as exc:
                            result = {"error": str(exc)}

                    trace.append(
                        {
                            "tool_name": call["function"]["name"],
                            "arguments": arguments,
                            "result": result,
                        }
                    )
                    transcript.append(tool_message(call["id"], result))
                continue

            final_response = self._extract_text(message.get("content"))
            if not final_response:
                fallback = self.runtime.resolve_recovery_case(patient_id, symptom_report)
                final_response = fallback["recovery_summary"]

            return WatsonxResponse(
                patient_id=patient_id,
                final_response=final_response,
                tool_trace=trace,
                transcript=transcript,
                raw_response=raw_response,
            )

        fallback = self.runtime.resolve_recovery_case(patient_id, symptom_report)
        return WatsonxResponse(
            patient_id=patient_id,
            final_response=fallback["recovery_summary"],
            tool_trace=trace,
            transcript=transcript,
            raw_response=raw_response,
        )

    @staticmethod
    def _extract_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"]
            return "\n".join(part for part in parts if part).strip()
        return ""
