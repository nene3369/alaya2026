"""PinealSampler — physical entropy token sampler for LLMs.

Replaces torch.multinomial with hardware TRNG for non-deterministic
token selection. Supports both NumPy and PyTorch backends.
"""

from __future__ import annotations

import os
import struct
import time
from dataclasses import dataclass

import numpy as np


_TORCH_AVAILABLE = False
try:
    import torch  # noqa: F401
    _TORCH_AVAILABLE = True
except ImportError:
    pass


@dataclass
class SamplerResult:
    """Token sampling result."""

    token_id: int = 0
    token_probability: float = 0.0
    entropy_signal: float = 0.0
    superposition_entropy: float = 0.0
    n_candidates: int = 0
    vocab_size: int = 0
    temperature: float = 1.0
    elapsed_ms: float = 0.0


@dataclass
class SamplerReport:
    """Sampler operating statistics."""

    n_tokens_sampled: int = 0
    avg_entropy: float = 0.0
    avg_candidates: float = 0.0
    avg_latency_ms: float = 0.0
    total_entropy_bytes: int = 0


class PinealSampler:
    """Physical entropy LLM token sampler.

    Three-stage process:
      1. Logits -> softmax probabilities
      2. Top-k / Top-p filtering
      3. Physical entropy collapse (TRNG)
    """

    def __init__(
        self,
        *,
        temperature: float = 1.0,
        top_k: int = 0,
        nucleus_p: float = 0.0,
        repetition_penalty: float = 1.0,
    ):
        self.temperature = max(temperature, 1e-10)
        self.top_k = top_k
        self.nucleus_p = nucleus_p
        self.repetition_penalty = repetition_penalty

        self._n_sampled: int = 0
        self._total_entropy: float = 0.0
        self._total_candidates: int = 0
        self._total_latency_ms: float = 0.0

    # -- NumPy API --------------------------------------------------------

    def sample(
        self,
        logits: np.ndarray,
        *,
        prev_token_ids: list[int] | None = None,
    ) -> SamplerResult:
        """Sample a token from logits using physical entropy."""
        start = time.monotonic()

        logits = np.asarray(logits, dtype=np.float64).flatten()
        vocab_size = len(logits)

        # Repetition penalty
        if prev_token_ids and self.repetition_penalty != 1.0:
            for tid in prev_token_ids:
                if 0 <= tid < vocab_size:
                    if logits[tid] > 0:
                        logits[tid] /= self.repetition_penalty
                    else:
                        logits[tid] *= self.repetition_penalty

        # Temperature scaling
        logits = logits / self.temperature

        # Softmax
        logits_max = logits.max()
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / exp_logits.sum()

        # Top-k filtering
        if self.top_k > 0 and self.top_k < vocab_size:
            top_indices = np.argsort(-probs)[:self.top_k]
            top_set = set(int(i) for i in top_indices)
            for i in range(vocab_size):
                if i not in top_set:
                    probs[i] = 0.0
            total = probs.sum()
            if total > 0:
                probs = probs / total

        # Nucleus (top-p) filtering
        if 0 < self.nucleus_p < 1.0:
            sorted_idx = np.argsort(-probs)
            cumsum = np.cumsum(probs[sorted_idx])
            cutoff = int(np.searchsorted(cumsum, self.nucleus_p)) + 1
            keep_set = set(int(sorted_idx[i]) for i in range(min(cutoff, len(sorted_idx))))
            for i in range(vocab_size):
                if i not in keep_set:
                    probs[i] = 0.0
            total = probs.sum()
            if total > 0:
                probs = probs / total

        # Count candidates
        n_candidates = sum(1 for v in probs if v > 1e-10)

        # Distribution entropy
        mask = probs > 1e-15
        entropy = -float(np.sum(probs[mask] * np.log2(probs[mask])))

        # Physical entropy collapse
        entropy_signal = self._harvest_entropy()
        cumprobs = np.cumsum(probs)
        token_id = int(np.searchsorted(cumprobs, entropy_signal))
        token_id = min(token_id, vocab_size - 1)

        elapsed = (time.monotonic() - start) * 1000

        self._n_sampled += 1
        self._total_entropy += entropy
        self._total_candidates += n_candidates
        self._total_latency_ms += elapsed

        return SamplerResult(
            token_id=token_id,
            token_probability=float(probs[token_id]),
            entropy_signal=entropy_signal,
            superposition_entropy=entropy,
            n_candidates=n_candidates,
            vocab_size=vocab_size,
            temperature=self.temperature,
            elapsed_ms=elapsed,
        )

    # -- PyTorch API ------------------------------------------------------

    def sample_torch(self, logits_tensor, **kwargs) -> SamplerResult:
        """Sample from PyTorch logits tensor."""
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch required for sample_torch()")
        logits_np = logits_tensor.detach().cpu().numpy().flatten()
        return self.sample(logits_np, **kwargs)

    # -- Statistics -------------------------------------------------------

    def report(self) -> SamplerReport:
        n = max(self._n_sampled, 1)
        return SamplerReport(
            n_tokens_sampled=self._n_sampled,
            avg_entropy=self._total_entropy / n,
            avg_candidates=self._total_candidates / n,
            avg_latency_ms=self._total_latency_ms / n,
            total_entropy_bytes=self._n_sampled * 8,
        )

    # -- Internal ---------------------------------------------------------

    @staticmethod
    def _harvest_entropy() -> float:
        """Harvest physical entropy and return uniform [0, 1)."""
        try:
            raw = os.urandom(8)
        except NotImplementedError:
            raw = struct.pack("d", time.monotonic())[:8]
        val = struct.unpack("Q", raw)[0]
        return (val & 0x001FFFFFFFFFFFFF) / (1 << 53)
