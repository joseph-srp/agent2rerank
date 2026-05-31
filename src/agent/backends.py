from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol


class AgentBackend(Protocol):
    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        ...


@dataclass
class OpenAICompatibleBackend:
    model: str
    base_url: str
    api_key: str
    temperature: float = 0.2
    max_tokens: int = 2048
    timeout: int = 120

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "OpenAICompatibleBackend":
        api_key_env = config.get("api_key_env")
        api_key = config.get("api_key") or (os.environ.get(api_key_env) if api_key_env else None)
        if not api_key:
            raise ValueError(f"missing API key; set {api_key_env or 'api_key'}")
        return cls(
            model=config["model"],
            base_url=config["base_url"],
            api_key=api_key,
            temperature=float(config.get("temperature", 0.2)),
            max_tokens=int(config.get("max_tokens", 2048)),
            timeout=int(config.get("timeout", 120)),
        )

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        url = self.base_url.rstrip("/")
        if not url.endswith("/chat/completions"):
            url += "/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM HTTP {exc.code}: {detail}") from exc
        return json.loads(body)["choices"][0]["message"]
