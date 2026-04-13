from __future__ import annotations

import json
import time
from typing import Any
from urllib import parse, request

from .config import AppConfig


class WatsonxClient:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._access_token: str | None = None
        self._expires_at: float = 0.0

    def _get_access_token(self) -> str:
        if self._access_token and time.time() < self._expires_at:
            return self._access_token

        if not self.config.watsonx_apikey:
            raise RuntimeError("WATSONX_APIKEY is required before calling watsonx.ai.")

        token_body = parse.urlencode(
            {
                "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
                "apikey": self.config.watsonx_apikey,
            }
        ).encode("utf-8")
        req = request.Request(
            self.config.iam_url,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
            },
            data=token_body,
            method="POST",
        )

        with request.urlopen(req, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))

        self._access_token = payload["access_token"]
        self._expires_at = time.time() + max(int(payload.get("expires_in", 3600)) - 60, 60)
        return self._access_token

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice_option: str | None = "auto",
        max_tokens: int = 900,
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        if not self.config.watsonx_ready:
            missing = ", ".join(self.config.missing_watsonx_fields())
            raise RuntimeError(f"watsonx is not configured. Missing: {missing}")

        payload: dict[str, Any] = {
            "model_id": self.config.normalized_model,
            "project_id": self.config.watsonx_project_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools
        if tool_choice_option:
            payload["tool_choice_option"] = tool_choice_option

        endpoint = (
            f"{self.config.watsonx_url.rstrip('/')}/ml/v1/text/chat"
            f"?version={parse.quote(self.config.watsonx_version)}"
        )
        req = request.Request(
            endpoint,
            headers={
                "Authorization": f"Bearer {self._get_access_token()}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
        )
        with request.urlopen(req, timeout=90) as response:
            return json.loads(response.read().decode("utf-8"))
