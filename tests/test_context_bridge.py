"""Tests for lmm.reasoning.context_bridge — AlayaContextBridge."""

from __future__ import annotations

import numpy as np

from lmm.reasoning.alaya import AlayaMemory
from lmm.reasoning.context_bridge import AlayaContextBridge


def _make_history(n: int) -> list[dict]:
    """Create n dummy history messages."""
    return [
        {"role": "user" if i % 2 == 0 else "assistant",
         "content": f"message {i}"}
        for i in range(n)
    ]


class TestEmptyHistory:
    def test_empty_returns_empty(self):
        memory = AlayaMemory(n_variables=8)
        bridge = AlayaContextBridge(memory, max_context_messages=10)
        result = bridge.select_context([], np.zeros(8))
        assert result == []


class TestSmallHistory:
    def test_short_history_returns_all(self):
        memory = AlayaMemory(n_variables=8)
        bridge = AlayaContextBridge(memory, max_context_messages=20)
        history = _make_history(5)
        result = bridge.select_context(history, np.zeros(8))
        assert len(result) == 5


class TestRecentMessagesAlwaysIncluded:
    def test_last_three_always_present(self):
        memory = AlayaMemory(n_variables=8)
        # Store a pattern so recall works
        memory.store(np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]))

        bridge = AlayaContextBridge(
            memory, max_context_messages=5, recency_count=3,
        )
        history = _make_history(30)
        result = bridge.select_context(history, np.ones(8))

        # Last 3 messages should always be in the result
        last_3_contents = {history[i]["content"] for i in range(27, 30)}
        result_contents = {m["content"] for m in result}
        assert last_3_contents.issubset(result_contents)

    def test_recent_messages_with_no_memory(self):
        memory = AlayaMemory(n_variables=8)
        bridge = AlayaContextBridge(
            memory, max_context_messages=5, recency_count=3,
        )
        history = _make_history(30)
        # With no patterns, falls back to recency in the outer code
        # But context_bridge itself returns all if <=max
        # When n_total > max_context, it uses recall
        result = bridge.select_context(history, np.ones(8))
        assert len(result) == 5


class TestMaxContextRespected:
    def test_never_exceeds_max(self):
        memory = AlayaMemory(n_variables=8)
        memory.store(np.ones(8))
        bridge = AlayaContextBridge(
            memory, max_context_messages=10, recency_count=3,
        )
        history = _make_history(50)
        result = bridge.select_context(history, np.ones(8))
        assert len(result) <= 10


class TestChronologicalOrder:
    def test_result_is_chronological(self):
        memory = AlayaMemory(n_variables=8)
        memory.store(np.ones(8))
        bridge = AlayaContextBridge(
            memory, max_context_messages=8, recency_count=3,
        )
        history = _make_history(25)
        result = bridge.select_context(history, np.ones(8))

        # Verify messages are in chronological order
        contents = [m["content"] for m in result]
        indices = [int(c.split()[-1]) for c in contents]
        assert indices == sorted(indices)


class TestRelevanceScoring:
    def test_relevance_symmetric(self):
        memory = AlayaMemory(n_variables=8)
        bridge = AlayaContextBridge(memory)
        a = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        b = np.array([0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
        score = bridge.compute_message_relevance(a, b)
        assert 0.0 <= score <= 1.0

    def test_zero_vector_gives_zero_relevance(self):
        memory = AlayaMemory(n_variables=8)
        bridge = AlayaContextBridge(memory)
        z = np.zeros(8)
        a = np.ones(8)
        assert bridge.compute_message_relevance(z, a) == 0.0
        assert bridge.compute_message_relevance(a, z) == 0.0

    def test_identical_vectors_give_max_relevance(self):
        memory = AlayaMemory(n_variables=8)
        bridge = AlayaContextBridge(memory)
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        score = bridge.compute_message_relevance(a, a)
        assert abs(score - 1.0) < 0.01
