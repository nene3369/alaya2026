"""LLM Provider base — model-agnostic interface for language model calls.

All providers implement the same async interface so the LLMRouter can
transparently switch between Claude, Gemini, OpenAI, or local models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional


@dataclass
class LLMResponse:
    """Unified response from any LLM provider."""

    text: str
    provider: str
    model: str
    usage: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    finish_reason: str = ""


@dataclass
class LLMRequest:
    """Unified request to any LLM provider."""

    messages: List[Dict[str, str]]
    system: str = ""
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 1.0
    top_k: int = 0
    stop_sequences: List[str] = field(default_factory=list)


class LLMProvider(ABC):
    """Abstract base for all LLM providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier (e.g. 'claude', 'gemini', 'openai', 'ollama')."""

    @property
    @abstractmethod
    def default_model(self) -> str:
        """Default model to use when none is specified."""

    @abstractmethod
    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Send a completion request and return the full response."""

    @abstractmethod
    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        """Stream completion tokens."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is usable (API key set, server reachable, etc.)."""


class ProviderRegistry:
    """Registry of available LLM providers with auto-selection.

    The registry maintains a priority-ordered list of providers and can
    automatically select one based on availability and mode requirements.
    """

    def __init__(self) -> None:
        self._providers: Dict[str, LLMProvider] = {}
        self._priority: List[str] = []

    def register(
        self,
        provider: LLMProvider,
        priority: int | None = None,
    ) -> None:
        """Register a provider. Lower priority number = preferred."""
        self._providers[provider.name] = provider
        if provider.name not in self._priority:
            if priority is not None and 0 <= priority <= len(self._priority):
                self._priority.insert(priority, provider.name)
            else:
                self._priority.append(provider.name)

    def get(self, name: str) -> LLMProvider | None:
        return self._providers.get(name)

    def auto_select(
        self,
        preferred: str | None = None,
    ) -> LLMProvider | None:
        """Select the best available provider.

        If *preferred* is specified and available, use it.  Otherwise
        iterate by priority order and return the first available.
        """
        if preferred and preferred in self._providers:
            p = self._providers[preferred]
            if p.is_available():
                return p

        for name in self._priority:
            p = self._providers[name]
            if p.is_available():
                return p
        return None

    def list_available(self) -> List[str]:
        return [n for n in self._priority if self._providers[n].is_available()]

    @property
    def count(self) -> int:
        return len(self._providers)
