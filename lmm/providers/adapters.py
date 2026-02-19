"""Concrete LLM provider adapters — Claude, Gemini, OpenAI, Ollama.

Each adapter translates the unified LLMRequest/LLMResponse into the
provider-specific API format, handling authentication, streaming, and
error mapping.
"""

from __future__ import annotations

import json
import os
from typing import Any, AsyncIterator, Dict

from lmm.providers.base import LLMProvider, LLMRequest, LLMResponse


# ---------------------------------------------------------------------------
# Claude (Anthropic)
# ---------------------------------------------------------------------------

class ClaudeProvider(LLMProvider):
    """Anthropic Claude API adapter."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
    ) -> None:
        self._api_key = api_key or os.environ.get("CLAUDE_API_KEY", "")
        self._model = model

    @property
    def name(self) -> str:
        return "claude"

    @property
    def default_model(self) -> str:
        return self._model

    def is_available(self) -> bool:
        return bool(self._api_key)

    async def complete(self, request: LLMRequest) -> LLMResponse:
        import httpx

        body: Dict[str, Any] = {
            "model": self._model,
            "max_tokens": request.max_tokens,
            "system": request.system,
            "messages": request.messages,
            "temperature": request.temperature,
        }
        if request.top_k > 0:
            body["top_k"] = request.top_k
        if request.top_p < 1.0:
            body["top_p"] = request.top_p
        if request.stop_sequences:
            body["stop_sequences"] = request.stop_sequences

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": self._api_key,
                    "anthropic-version": "2023-06-01",
                },
                json=body,
            )
            resp.raise_for_status()
            data = resp.json()

        text = "".join(
            b["text"] for b in data.get("content", []) if b.get("type") == "text"
        )
        return LLMResponse(
            text=text,
            provider="claude",
            model=self._model,
            usage=data.get("usage", {}),
            finish_reason=data.get("stop_reason", ""),
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        import httpx

        body: Dict[str, Any] = {
            "model": self._model,
            "max_tokens": request.max_tokens,
            "system": request.system,
            "messages": request.messages,
            "temperature": request.temperature,
            "stream": True,
        }
        if request.top_k > 0:
            body["top_k"] = request.top_k
        if request.top_p < 1.0:
            body["top_p"] = request.top_p

        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream(
                "POST",
                "https://api.anthropic.com/v1/messages",
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": self._api_key,
                    "anthropic-version": "2023-06-01",
                },
                json=body,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    raw = line[6:]
                    if raw == "[DONE]":
                        break
                    try:
                        event = json.loads(raw)
                        if event.get("type") == "content_block_delta":
                            delta = event.get("delta", {})
                            if delta.get("type") == "text_delta":
                                yield delta["text"]
                    except (json.JSONDecodeError, KeyError):
                        continue


# ---------------------------------------------------------------------------
# Gemini (Google)
# ---------------------------------------------------------------------------

class GeminiProvider(LLMProvider):
    """Google Gemini API adapter."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.0-flash",
    ) -> None:
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self._model = model

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def default_model(self) -> str:
        return self._model

    def is_available(self) -> bool:
        return bool(self._api_key)

    async def complete(self, request: LLMRequest) -> LLMResponse:
        import httpx

        contents = []
        for msg in request.messages:
            role = "user" if msg["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": msg["content"]}]})

        body: Dict[str, Any] = {
            "contents": contents,
            "systemInstruction": {"parts": [{"text": request.system}]},
            "generationConfig": {
                "temperature": request.temperature,
                "maxOutputTokens": request.max_tokens,
            },
        }
        if request.top_p < 1.0:
            body["generationConfig"]["topP"] = request.top_p
        if request.top_k > 0:
            body["generationConfig"]["topK"] = request.top_k

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/"
            f"models/{self._model}:generateContent?key={self._api_key}"
        )

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, json=body)
            resp.raise_for_status()
            data = resp.json()

        candidates = data.get("candidates", [])
        text = ""
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts)

        return LLMResponse(
            text=text,
            provider="gemini",
            model=self._model,
            usage=data.get("usageMetadata", {}),
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        import httpx

        contents = []
        for msg in request.messages:
            role = "user" if msg["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": msg["content"]}]})

        body: Dict[str, Any] = {
            "contents": contents,
            "systemInstruction": {"parts": [{"text": request.system}]},
            "generationConfig": {
                "temperature": request.temperature,
                "maxOutputTokens": request.max_tokens,
            },
        }
        if request.top_p < 1.0:
            body["generationConfig"]["topP"] = request.top_p
        if request.top_k > 0:
            body["generationConfig"]["topK"] = request.top_k

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/"
            f"models/{self._model}:streamGenerateContent"
            f"?key={self._api_key}&alt=sse"
        )

        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream("POST", url, json=body) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    raw = line[6:]
                    try:
                        event = json.loads(raw)
                        for cand in event.get("candidates", []):
                            for part in cand.get("content", {}).get("parts", []):
                                if "text" in part:
                                    yield part["text"]
                    except (json.JSONDecodeError, KeyError):
                        continue


# ---------------------------------------------------------------------------
# OpenAI (GPT-4o, etc.)
# ---------------------------------------------------------------------------

class OpenAIProvider(LLMProvider):
    """OpenAI ChatCompletion API adapter (GPT-4o, GPT-4, etc.)."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o",
        base_url: str = "https://api.openai.com/v1",
    ) -> None:
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._model = model
        self._base_url = base_url

    @property
    def name(self) -> str:
        return "openai"

    @property
    def default_model(self) -> str:
        return self._model

    def is_available(self) -> bool:
        return bool(self._api_key)

    async def complete(self, request: LLMRequest) -> LLMResponse:
        import httpx

        messages = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.extend(request.messages)

        body: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
        if request.top_p < 1.0:
            body["top_p"] = request.top_p
        if request.stop_sequences:
            body["stop"] = request.stop_sequences

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{self._base_url}/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._api_key}",
                },
                json=body,
            )
            resp.raise_for_status()
            data = resp.json()

        choice = data.get("choices", [{}])[0]
        text = choice.get("message", {}).get("content", "")
        return LLMResponse(
            text=text,
            provider="openai",
            model=self._model,
            usage=data.get("usage", {}),
            finish_reason=choice.get("finish_reason", ""),
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        import httpx

        messages = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.extend(request.messages)

        body: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": True,
        }
        if request.top_p < 1.0:
            body["top_p"] = request.top_p

        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream(
                "POST",
                f"{self._base_url}/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._api_key}",
                },
                json=body,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    raw = line[6:]
                    if raw == "[DONE]":
                        break
                    try:
                        event = json.loads(raw)
                        delta = event["choices"][0].get("delta", {})
                        if "content" in delta:
                            yield delta["content"]
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue


# ---------------------------------------------------------------------------
# Ollama (local LLM)
# ---------------------------------------------------------------------------

class OllamaProvider(LLMProvider):
    """Ollama local LLM adapter — privacy-first, zero-cloud inference.

    Connects to a locally running Ollama server for complete data sovereignty.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3",
    ) -> None:
        self._base_url = base_url
        self._model = model

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def default_model(self) -> str:
        return self._model

    def is_available(self) -> bool:
        try:
            import httpx
            resp = httpx.get(f"{self._base_url}/api/tags", timeout=2)
            return resp.status_code == 200
        except Exception:
            return False

    async def complete(self, request: LLMRequest) -> LLMResponse:
        import httpx

        messages = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.extend(request.messages)

        body = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
            },
        }
        if request.top_p < 1.0:
            body["options"]["top_p"] = request.top_p
        if request.top_k > 0:
            body["options"]["top_k"] = request.top_k

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{self._base_url}/api/chat",
                json=body,
            )
            resp.raise_for_status()
            data = resp.json()

        text = data.get("message", {}).get("content", "")
        return LLMResponse(
            text=text,
            provider="ollama",
            model=self._model,
            usage={
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
            },
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        import httpx

        messages = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.extend(request.messages)

        body = {
            "model": self._model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
            },
        }

        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST",
                f"{self._base_url}/api/chat",
                json=body,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        content = data.get("message", {}).get("content", "")
                        if content:
                            yield content
                        if data.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue
