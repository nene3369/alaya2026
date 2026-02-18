"""AlayaContextBridge — memory-driven LLM context selection.

Uses Modern Hopfield recall patterns from AlayaMemory to intelligently
select which conversation history messages to include in LLM context,
replacing naive truncation (history[-20:]) with relevance-based selection.
"""

from __future__ import annotations

import numpy as np

from lmm.reasoning.alaya import AlayaMemory


class AlayaContextBridge:
    """Bridge between AlayaMemory patterns and LLM context management.

    Uses Modern Hopfield recall to score conversation history messages
    by relevance to the current emotional state, then selects the most
    relevant subset within the message budget.

    Parameters
    ----------
    memory : AlayaMemory
        The Alaya memory instance for pattern recall.
    max_context_messages : int
        Maximum number of history messages to return.
    recency_count : int
        Number of most recent messages always included (recency bias).
    """

    def __init__(
        self,
        memory: AlayaMemory,
        max_context_messages: int = 20,
        recency_count: int = 3,
    ):
        self.memory = memory
        self.max_context_messages = max_context_messages
        self.recency_count = recency_count

    def _text_to_vec(self, text: str, dim: int) -> np.ndarray:
        """Convert text to a fixed-size character-frequency vector.

        Mirrors the _text_to_vec pattern used in server.py.
        """
        vec = np.zeros(dim, dtype=float)
        for ch in text:
            vec[ord(ch) % dim] = vec[ord(ch) % dim] + 1.0
        norm = float(np.linalg.norm(vec))
        if norm > 1e-9:
            vec = vec / norm
        return vec

    def compute_message_relevance(
        self,
        message_vec: np.ndarray,
        recalled_pattern: np.ndarray,
    ) -> float:
        """Score a message's relevance via cosine similarity with recalled pattern.

        Parameters
        ----------
        message_vec : (n,) array
            Feature vector of the message.
        recalled_pattern : (n,) array
            Pattern recalled from AlayaMemory.

        Returns
        -------
        float
            Cosine similarity in [0, 1].
        """
        norm_m = float(np.linalg.norm(message_vec))
        norm_r = float(np.linalg.norm(recalled_pattern))
        if norm_m < 1e-9 or norm_r < 1e-9:
            return 0.0
        dot = float(np.sum(message_vec * recalled_pattern))
        cos = dot / (norm_m * norm_r)
        # Clamp to [0, 1] (negative similarity treated as irrelevant)
        return max(0.0, cos)

    def select_context(
        self,
        history: list[dict],
        current_wave: np.ndarray,
    ) -> list[dict]:
        """Select the most relevant history messages for LLM context.

        Always includes the most recent `recency_count` messages.
        Remaining slots filled by highest-relevance messages scored
        via Hopfield recall similarity.

        Parameters
        ----------
        history : list of dict
            Conversation history with 'role' and 'content' keys.
        current_wave : (n,) array
            Current emotional/state vector used as recall cue.

        Returns
        -------
        list of dict
            Selected messages, ordered chronologically.
        """
        if not history:
            return []

        n_total = len(history)
        if n_total <= self.max_context_messages:
            return list(history)

        # Always include last recency_count messages
        recency_start = max(0, n_total - self.recency_count)
        recent_indices = set(range(recency_start, n_total))

        # Remaining budget for relevance-based selection
        budget = self.max_context_messages - len(recent_indices)
        if budget <= 0:
            return [history[i] for i in sorted(recent_indices)]

        # Recall pattern from Alaya memory using current wave as cue
        cue = np.asarray(current_wave).flatten()[: self.memory.n]
        if len(cue) < self.memory.n:
            padded = np.zeros(self.memory.n)
            padded[: len(cue)] = cue
            cue = padded

        recalled = self.memory.recall(cue)

        # Score each non-recent message by relevance
        candidate_indices = [i for i in range(n_total) if i not in recent_indices]
        scores: list[tuple[int, float]] = []
        for idx in candidate_indices:
            msg = history[idx]
            content = msg.get("content", "")
            msg_vec = self._text_to_vec(content, self.memory.n)
            score = self.compute_message_relevance(msg_vec, recalled)
            scores.append((idx, score))

        # Sort by relevance (descending) and take top-budget
        scores.sort(key=lambda x: x[1], reverse=True)
        selected_indices = recent_indices | {idx for idx, _ in scores[:budget]}

        # Return in chronological order
        return [history[i] for i in sorted(selected_indices)]
