from __future__ import annotations

import json
from typing import Any
from urllib import request


class SerperSearchClient:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def search(self, query: str, limit: int = 5) -> dict[str, Any]:
        payload = json.dumps({"q": query, "num": limit}).encode("utf-8")
        req = request.Request(
            "https://google.serper.dev/search",
            headers={
                "X-API-KEY": self.api_key,
                "Content-Type": "application/json",
            },
            data=payload,
            method="POST",
        )
        with request.urlopen(req, timeout=30) as response:
            payload_data = json.loads(response.read().decode("utf-8"))

        organic = payload_data.get("organic", [])
        return {
            "query": query,
            "results": [
                {
                    "title": item.get("title"),
                    "link": item.get("link"),
                    "snippet": item.get("snippet"),
                }
                for item in organic[:limit]
            ],
        }
