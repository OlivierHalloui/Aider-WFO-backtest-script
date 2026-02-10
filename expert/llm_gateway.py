from __future__ import annotations

import time
from typing import Any, Dict

import requests

from .models import LLMConfig


class OpenAICompatibleGateway:
    """Minimal HTTP gateway for OpenAI-compatible chat completion endpoints."""

    _DEFAULT_BASE_URLS = {
        "openai": "https://api.openai.com/v1",
        "grok": "https://api.x.ai/v1",
    }

    def _resolve_base_url(self, provider: str, base_url: str | None) -> str:
        if base_url and str(base_url).strip():
            return str(base_url).rstrip("/")
        return self._DEFAULT_BASE_URLS.get(provider, self._DEFAULT_BASE_URLS["openai"])

    def _build_payload(self, system_prompt: str, user_prompt: str, config: LLMConfig) -> Dict[str, Any]:
        return {
            "model": config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": float(config.temperature),
            "max_tokens": int(config.max_tokens),
        }

    def generate(self, system_prompt: str, user_prompt: str, config: LLMConfig) -> str:
        api_key = str(config.api_key or "").strip()
        if not api_key:
            raise RuntimeError(
                "API key is empty after trimming. "
                "Provide a non-empty key (e.g. starts with 'sk-' for OpenAI)."
            )
        base_url = self._resolve_base_url(config.provider, config.base_url)
        url = f"{base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = self._build_payload(system_prompt, user_prompt, config)

        last_error: Exception | None = None
        retries = max(0, int(config.retries))
        for attempt in range(retries + 1):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=float(config.timeout_s))
                response.raise_for_status()
                data = response.json()
                return self._extract_text(data)
            except Exception as e:  # noqa: BLE001
                last_error = e
                if attempt >= retries:
                    break
                time.sleep(0.7 * (attempt + 1))

        if last_error is None:
            raise RuntimeError("Unknown LLM gateway error")
        raise RuntimeError(f"LLM request failed: {last_error}")

    @staticmethod
    def _extract_text(data: Dict[str, Any]) -> str:
        choices = data.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content

        # Some providers may return structured content blocks.
        if isinstance(content, list):
            texts = []
            for block in content:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    texts.append(block.get("text"))
            return "\n".join(texts).strip()

        return str(content or "")
