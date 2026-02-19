"""Tests for lmm.providers — LLM Provider Abstraction (言葉・Vach)."""

from __future__ import annotations

import os
import pytest

from lmm.providers.base import (
    LLMProvider,
    LLMRequest,
    LLMResponse,
    ProviderRegistry,
)
from lmm.providers.adapters import (
    ClaudeProvider,
    GeminiProvider,
    OpenAIProvider,
    OllamaProvider,
)


# ---------------------------------------------------------------------------
# Tests: ProviderRegistry
# ---------------------------------------------------------------------------

class TestProviderRegistry:
    def test_register_and_get(self):
        reg = ProviderRegistry()
        p = ClaudeProvider(api_key="test-key")
        reg.register(p)
        assert reg.count == 1
        assert reg.get("claude") is p

    def test_auto_select_with_available(self):
        reg = ProviderRegistry()
        claude = ClaudeProvider(api_key="test-key")
        gemini = GeminiProvider(api_key="")
        reg.register(claude, priority=0)
        reg.register(gemini, priority=1)
        selected = reg.auto_select()
        assert selected is claude  # Claude has a key, so it's available

    def test_auto_select_preferred(self):
        reg = ProviderRegistry()
        claude = ClaudeProvider(api_key="key1")
        gemini = GeminiProvider(api_key="key2")
        reg.register(claude, priority=0)
        reg.register(gemini, priority=1)
        selected = reg.auto_select(preferred="gemini")
        assert selected is gemini

    def test_auto_select_fallback(self):
        reg = ProviderRegistry()
        claude = ClaudeProvider(api_key="")  # Not available
        gemini = GeminiProvider(api_key="key")
        reg.register(claude, priority=0)
        reg.register(gemini, priority=1)
        selected = reg.auto_select()
        assert selected is gemini

    def test_list_available(self):
        reg = ProviderRegistry()
        reg.register(ClaudeProvider(api_key="key"))
        reg.register(GeminiProvider(api_key=""))
        available = reg.list_available()
        assert "claude" in available
        assert "gemini" not in available


# ---------------------------------------------------------------------------
# Tests: Provider adapters (unit-level, no API calls)
# ---------------------------------------------------------------------------

class TestClaudeProvider:
    def test_name_and_model(self):
        p = ClaudeProvider(api_key="test")
        assert p.name == "claude"
        assert "claude" in p.default_model

    def test_is_available(self):
        p = ClaudeProvider(api_key="test")
        assert p.is_available() is True
        p2 = ClaudeProvider(api_key="")
        assert p2.is_available() is False


class TestGeminiProvider:
    def test_name_and_model(self):
        p = GeminiProvider(api_key="test")
        assert p.name == "gemini"
        assert "gemini" in p.default_model

    def test_is_available(self):
        p = GeminiProvider(api_key="test")
        assert p.is_available() is True


class TestOpenAIProvider:
    def test_name_and_model(self):
        p = OpenAIProvider(api_key="test")
        assert p.name == "openai"
        assert "gpt" in p.default_model

    def test_is_available(self):
        p = OpenAIProvider(api_key="test")
        assert p.is_available() is True


class TestOllamaProvider:
    def test_name_and_model(self):
        p = OllamaProvider()
        assert p.name == "ollama"
        assert p.default_model == "llama3"

    def test_is_available_no_server(self):
        # Ollama won't be running in CI, so this should return False
        p = OllamaProvider(base_url="http://localhost:99999")
        assert p.is_available() is False


# ---------------------------------------------------------------------------
# Tests: LLMRequest
# ---------------------------------------------------------------------------

class TestLLMRequest:
    def test_defaults(self):
        req = LLMRequest(
            messages=[{"role": "user", "content": "hello"}],
            system="You are helpful.",
        )
        assert req.temperature == 0.7
        assert req.max_tokens == 2048
        assert req.top_p == 1.0
