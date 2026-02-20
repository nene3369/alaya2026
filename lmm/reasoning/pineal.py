"""PinealGland — quantum consciousness with hardware TRNG."""

from __future__ import annotations

import hashlib
import os
import struct
import time
from dataclasses import dataclass

import numpy as np
from scipy import sparse

from lmm.dharma.fep import solve_fep_kcl_analog
from lmm.reasoning.base import BaseReasoner, ReasonerResult


@dataclass
class EntropySource:
    """Physical entropy source info."""

    source: str
    bits_available: int
    quality: float


class PinealGland(BaseReasoner):
    """Quantum consciousness — physical entropy for non-deterministic reasoning.

    Uses hardware TRNG (True Random Number Generator) or OS entropy pool
    to inject genuine randomness into the FEP ODE, enabling exploration
    beyond deterministic local minima.
    """

    def __init__(
        self,
        n_variables: int,
        k: int,
        *,
        entropy_scale: float = 0.1,
        G_prec: float = 8.0,
        tau_leak: float = 2.0,
        max_steps: int = 300,
        nirvana_threshold: float = 1e-4,
    ):
        super().__init__(n_variables, k, nirvana_threshold=nirvana_threshold)
        self.entropy_scale = entropy_scale
        self.G_prec = G_prec
        self.tau_leak = tau_leak
        self.max_steps = max_steps

    @property
    def mode(self) -> str:
        return "pineal"

    def harvest_entropy(self, n_bytes: int = 32) -> np.ndarray:
        """Harvest physical entropy from OS entropy pool."""
        try:
            raw = os.urandom(n_bytes)
        except NotImplementedError:
            # Fallback: hash current time with high precision
            raw = hashlib.sha256(
                struct.pack("d", time.monotonic())
            ).digest()[:n_bytes]

        # Convert bytes to floats in [-1, 1]
        values = []
        for b in raw:
            values.append((b / 127.5) - 1.0)

        return np.array(values)

    def entropy_source_info(self) -> EntropySource:
        """Check available entropy source."""
        try:
            os.urandom(1)
            return EntropySource(
                source="os.urandom", bits_available=256, quality=0.99,
            )
        except NotImplementedError:
            return EntropySource(
                source="time_hash_fallback", bits_available=256, quality=0.5,
            )

    def reason(
        self,
        h: np.ndarray,
        J: sparse.csr_matrix,
        *,
        sila_gamma: float = 0.0,
    ) -> ReasonerResult:
        # Inject physical entropy into sensory input
        entropy_raw = self.harvest_entropy(self.n * 4)
        entropy_vec = np.zeros(self.n)
        for i in range(min(len(entropy_raw), self.n)):
            entropy_vec[i] = float(entropy_raw[i])
        entropy_vec *= self.entropy_scale

        V_s = -h.copy() + entropy_vec
        if sila_gamma > 0:
            V_s = V_s - sila_gamma * (1.0 - 2.0 * self.k)

        J_dynamic = J * (-1.0) if J.nnz > 0 else J

        V_mu, x_final, steps_used, power = solve_fep_kcl_analog(
            V_s=V_s, J_dynamic=J_dynamic, n=self.n,
            G_prec_base=self.G_prec, tau_leak=self.tau_leak,
            max_steps=self.max_steps,
            nirvana_threshold=self.nirvana_threshold,
        )

        selected = self._topk_from_activations(x_final, self.k)
        energy = self._evaluate_energy(h, J, sila_gamma, selected)

        source_info = self.entropy_source_info()
        return ReasonerResult(
            selected_indices=selected, energy=energy,
            solver_used="pineal_fep_analog", reasoning_mode="pineal",
            steps_used=steps_used, power_history=power,
            diagnostics={
                "entropy_source": source_info.source,
                "entropy_quality": source_info.quality,
                "entropy_scale": self.entropy_scale,
            },
        )
