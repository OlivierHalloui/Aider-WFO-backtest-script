from __future__ import annotations

import json
import time
from typing import Any, Dict

import requests

from .models import LLMConfig


class OpenAICompatibleGateway:
    """Minimal HTTP gateway for OpenAI-compatible chat completion endpoints."""

    _DEFAULT_BASE_URLS = {
        "openai": "https://api.openai.com/v1",
        "grok": "https://api.x.ai/v1",
        "gemini": "https://generativelanguage.googleapis.com/v1beta",
    }

    def _resolve_base_url(self, provider: str, base_url: str | None) -> str:
        """Return explicit base URL or provider default endpoint."""
        if base_url and str(base_url).strip():
            return str(base_url).rstrip("/")
        return self._DEFAULT_BASE_URLS.get(provider, self._DEFAULT_BASE_URLS["openai"])

    def _build_payload(
        self,
        system_prompt: str,
        user_prompt: str,
        config: LLMConfig,
        token_budget: int | None = None,
    ) -> Dict[str, Any]:
        """Build chat/completions payload with provider/model-specific compatibility knobs."""
        provider = str(config.provider or "").lower()
        model = str(config.model or "").lower()
        is_openai_gpt5 = provider == "openai" and model.startswith("gpt-5")
        effective_tokens = int(token_budget if token_budget is not None else config.max_tokens)

        payload = {
            "model": config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        # GPT-5 on OpenAI only supports default temperature value.
        if not is_openai_gpt5:
            payload["temperature"] = float(config.temperature)
        elif float(config.temperature) == 1.0:
            payload["temperature"] = 1.0

        # OpenAI GPT-5 chat/completions rejects `max_tokens`; use `max_completion_tokens`.
        if is_openai_gpt5:
            payload["max_completion_tokens"] = effective_tokens
        else:
            payload["max_tokens"] = effective_tokens

        if provider == "openai":
            # Ask for JSON-only output for parser stability.
            payload["response_format"] = {"type": "json_object"}
            if is_openai_gpt5:
                # Latest-model guide: GPT-5 migration knobs for chat/completions.
                payload["reasoning_effort"] = "none"
                payload["verbosity"] = "low"
        return payload

    def generate(self, system_prompt: str, user_prompt: str, config: LLMConfig) -> str:
        """Dispatch generation to provider-specific implementation with shared key checks."""
        api_key = str(config.api_key or "").strip()
        if api_key.lower().startswith("bearer "):
            api_key = api_key.split(" ", 1)[1].strip()
        if not api_key:
            raise RuntimeError(
                "API key is empty after trimming. "
                "Provide a non-empty key (e.g. starts with 'sk-' for OpenAI)."
            )
        provider = str(config.provider or "").lower()
        base_url = self._resolve_base_url(provider, config.base_url)

        if provider == "gemini":
            return self._generate_gemini(base_url, api_key, system_prompt, user_prompt, config)
        if provider == "openai":
            return self._generate_openai(base_url, api_key, system_prompt, user_prompt, config)

        # Generic OpenAI-compatible providers (ex: xAI/Grok): chat/completions.
        return self._generate_chat_completions(base_url, api_key, system_prompt, user_prompt, config)

    def _generate_chat_completions(
        self,
        base_url: str,
        api_key: str,
        system_prompt: str,
        user_prompt: str,
        config: LLMConfig,
    ) -> str:
        """Call OpenAI-compatible chat/completions endpoint with retry/backoff."""
        url = f"{base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        token_budget = int(config.max_tokens)

        last_error: Exception | None = None
        retries = max(0, int(config.retries))
        for attempt in range(retries + 1):
            try:
                payload = self._build_payload(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    config=config,
                    token_budget=token_budget,
                )
                response = requests.post(url, headers=headers, json=payload, timeout=float(config.timeout_s))
                if response.status_code >= 400:
                    raise RuntimeError(self._format_http_error(response))
                data = response.json()
                try:
                    finish_reason = str(((data.get("choices") or [{}])[0]).get("finish_reason", ""))
                except Exception:  # noqa: BLE001
                    finish_reason = ""
                if finish_reason == "length":
                    if token_budget < 8000 and attempt < retries:
                        token_budget = min(8000, max(token_budget + 400, int(token_budget * 1.6)))
                        continue
                    raise RuntimeError(
                        "OpenAI chat completion was truncated (finish_reason=length). "
                        "Increase Max tokens."
                    )
                return self._extract_text(data)
            except Exception as e:  # noqa: BLE001
                last_error = e
                if attempt >= retries:
                    break
                time.sleep(0.7 * (attempt + 1))

        if last_error is None:
            raise RuntimeError("Unknown LLM gateway error")
        raise RuntimeError(f"LLM request failed: {last_error}")

    def _generate_openai(
        self,
        base_url: str,
        api_key: str,
        system_prompt: str,
        user_prompt: str,
        config: LLMConfig,
    ) -> str:
        """Call OpenAI Responses API (preferred) with fallback to chat/completions."""
        # Prefer OpenAI Responses API for modern models; fallback to chat/completions when needed.
        model = str(config.model or "").strip()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        responses_url = f"{base_url}/responses"
        token_budget = int(config.max_tokens)
        is_gpt5_family = model.lower().startswith("gpt-5")

        last_error: Exception | None = None
        retries = max(0, int(config.retries))
        for attempt in range(retries + 1):
            try:
                responses_payload = {
                    "model": model,
                    "input": [
                        {
                            "role": "system",
                            "content": [{"type": "input_text", "text": system_prompt}],
                        },
                        {
                            "role": "user",
                            "content": [{"type": "input_text", "text": user_prompt}],
                        },
                    ],
                    "max_output_tokens": token_budget,
                    "text": {"format": {"type": "json_object"}},
                }
                if is_gpt5_family:
                    # Latest-model guide: use reasoning/verbosity controls instead of sampling knobs.
                    responses_payload["reasoning"] = {"effort": "none"}
                    responses_payload["text"]["verbosity"] = "low"
                # GPT-5 on OpenAI only supports default temperature value.
                if not is_gpt5_family:
                    responses_payload["temperature"] = float(config.temperature)
                elif float(config.temperature) == 1.0:
                    responses_payload["temperature"] = 1.0

                response = requests.post(
                    responses_url,
                    headers=headers,
                    json=responses_payload,
                    timeout=float(config.timeout_s),
                )
                if response.status_code >= 400:
                    # Try chat/completions fallback for compatibility if endpoint/model mismatch.
                    if response.status_code in (400, 404):
                        return self._generate_chat_completions(
                            base_url=base_url,
                            api_key=api_key,
                            system_prompt=system_prompt,
                            user_prompt=user_prompt,
                            config=config,
                        )
                    raise RuntimeError(self._format_http_error(response))

                data = response.json()
                if str(data.get("status", "")).lower() == "incomplete":
                    reason = str((data.get("incomplete_details") or {}).get("reason", "unknown"))
                    if reason == "max_output_tokens" and token_budget < 8000 and attempt < retries:
                        token_budget = min(8000, max(token_budget + 400, int(token_budget * 1.6)))
                        continue
                    raise RuntimeError(
                        f"OpenAI response incomplete (reason={reason}). "
                        "Increase Max tokens or reduce response verbosity."
                    )
                text = self._extract_openai_response_text(data)
                if text:
                    return text
                # If parsing failed, fallback to generic string output rather than hard fail.
                return str(data)
            except Exception as e:  # noqa: BLE001
                last_error = e
                if attempt >= retries:
                    break
                time.sleep(0.7 * (attempt + 1))

        if last_error is None:
            raise RuntimeError("Unknown OpenAI gateway error")
        raise RuntimeError(f"LLM request failed: {last_error}")

    def _generate_gemini(
        self,
        base_url: str,
        api_key: str,
        system_prompt: str,
        user_prompt: str,
        config: LLMConfig,
    ) -> str:
        """Call Gemini generateContent endpoint and extract text safely."""
        model = str(config.model or "").strip()
        if not model:
            raise RuntimeError("Gemini model is empty.")
        if not model.startswith("models/"):
            model = f"models/{model}"
        url = f"{base_url.rstrip('/')}/{model}:generateContent"
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": f"{(system_prompt or '').strip()}\n\n{(user_prompt or '').strip()}"}
                    ],
                }
            ],
            "generationConfig": {
                "temperature": float(config.temperature),
                "maxOutputTokens": int(config.max_tokens),
            },
        }

        last_error: Exception | None = None
        retries = max(0, int(config.retries))
        for attempt in range(retries + 1):
            try:
                response = requests.post(
                    url,
                    params={"key": api_key},
                    json=payload,
                    timeout=float(config.timeout_s),
                )
                response.raise_for_status()
                data = response.json()
                return self._extract_gemini_text(data)
            except Exception as e:  # noqa: BLE001
                last_error = e
                if attempt >= retries:
                    break
                time.sleep(0.7 * (attempt + 1))

        if last_error is None:
            raise RuntimeError("Unknown Gemini gateway error")
        raise RuntimeError(f"Gemini request failed: {last_error}")

    @staticmethod
    def _extract_text(data: Dict[str, Any]) -> str:
        """Extract assistant text from chat/completions response payload."""
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

    @staticmethod
    def _extract_openai_response_text(data: Dict[str, Any]) -> str:
        """Extract text from OpenAI Responses API payload variants."""
        output = data.get("output") if isinstance(data, dict) else None
        if isinstance(output, list):
            chunks: list[str] = []
            for item in output:
                if not isinstance(item, dict):
                    continue
                for c in item.get("content", []) or []:
                    if isinstance(c, dict):
                        text_val = c.get("text")
                        if isinstance(text_val, str) and text_val.strip():
                            chunks.append(text_val)
            if chunks:
                return "\n".join(chunks).strip()

        output_text = data.get("output_text") if isinstance(data, dict) else None
        if isinstance(output_text, str):
            return output_text.strip()
        return ""

    @staticmethod
    def _extract_gemini_text(data: Dict[str, Any]) -> str:
        """Extract text from Gemini candidate parts."""
        candidates = data.get("candidates") or []
        if not candidates:
            return ""
        first = candidates[0] if isinstance(candidates[0], dict) else {}
        content = first.get("content", {}) if isinstance(first, dict) else {}
        parts = content.get("parts") if isinstance(content, dict) else []
        if isinstance(parts, list):
            texts = []
            for p in parts:
                if isinstance(p, dict) and isinstance(p.get("text"), str):
                    texts.append(p.get("text"))
            return "\n".join(texts).strip()
        return str(data)

    @staticmethod
    def _format_http_error(response: requests.Response) -> str:
        """Format HTTP error with truncated response body for diagnostics."""
        status_line = f"{response.status_code} {response.reason} for url: {response.url}"
        body_text = ""
        try:
            parsed = response.json()
            body_text = json.dumps(parsed, ensure_ascii=False)
        except Exception:  # noqa: BLE001
            body_text = (response.text or "").strip()
        if body_text:
            body_text = body_text[:1500]
            return f"{status_line} | body={body_text}"
        return status_line
