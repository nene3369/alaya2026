"""LLM Providers (言葉・Vach) — Model-agnostic abstraction layer.

Provides a unified interface for calling any LLM (Claude, Gemini, OpenAI,
local Ollama/vLLM) so the system is not hardcoded to specific providers.
"""

from lmm.providers.base import LLMProvider, LLMResponse, ProviderRegistry
from lmm.providers.adapters import (
    ClaudeProvider,
    GeminiProvider,
    OpenAIProvider,
    OllamaProvider,
)

__all__ = [
    "LLMProvider",
    "LLMResponse",
    "ProviderRegistry",
    "ClaudeProvider",
    "GeminiProvider",
    "OpenAIProvider",
    "OllamaProvider",
]
