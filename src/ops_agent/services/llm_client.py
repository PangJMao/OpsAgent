from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

from ops_agent.config import settings


@dataclass(frozen=True)
class LlmMessage:
    role: str
    content: str


class DeepSeekChatClient:
    """Minimal OpenAI-compatible client for DeepSeek chat completions."""

    def __init__(
        self,
        api_key: str = settings.deepseek_api_key,
        base_url: str = settings.deepseek_base_url,
        model: str = settings.deepseek_model,
        timeout_seconds: float = settings.llm_timeout_seconds,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def complete(self, messages: list[LlmMessage], temperature: float = 0.2) -> str:
        if not self.enabled:
            raise RuntimeError("DEEPSEEK_API_KEY is not configured.")

        payload = {
            "model": self.model,
            "messages": [{"role": message.role, "content": message.content} for message in messages],
            "temperature": temperature,
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise RuntimeError(f"DeepSeek request failed: {exc}") from exc

        data = json.loads(body)
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("DeepSeek returned no choices.")
        return str(choices[0]["message"]["content"]).strip()
