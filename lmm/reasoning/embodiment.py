"""EmbodiedAgent — 6-sense multimodal fusion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from lmm.dharma.fep import solve_fep_kcl_analog
from lmm.reasoning.base import BaseReasoner, ReasonerResult


@dataclass
class SenseInput:
    """A single sensory input channel."""

    name: str
    values: np.ndarray
    weight: float = 1.0


class EmbodiedAgent(BaseReasoner):
    """Embodied cognition — 6-sense multimodal fusion.

    Fuses multiple sensory channels into a unified belief state
    via weighted combination and FEP ODE resolution.

    The 6 senses (Buddhist Ayatana):
      1. Visual (sight)
      2. Auditory (hearing)
      3. Olfactory (smell)
      4. Gustatory (taste)
      5. Tactile (touch)
      6. Mental (manas — thought/cognition)
    """

    def __init__(
        self,
        n_variables: int,
        k: int,
        *,
        G_prec: float = 8.0,
        tau_leak: float = 1.5,
        max_steps: int = 300,
        nirvana_threshold: float = 1e-4,
    ):
        super().__init__(n_variables, k, nirvana_threshold=nirvana_threshold)
        self.G_prec = G_prec
        self.tau_leak = tau_leak
        self.max_steps = max_steps
        self._senses: list[SenseInput] = []

    @property
    def mode(self) -> str:
        return "embodied"

    def add_sense(self, name: str, values: np.ndarray, weight: float = 1.0) -> None:
        """Register a sensory channel."""
        self._senses.append(SenseInput(name=name, values=values, weight=weight))

    def clear_senses(self) -> None:
        """Clear all registered sensory channels."""
        self._senses = []

    def fuse_senses(self) -> np.ndarray:
        """Fuse all sensory channels into unified input."""
        if not self._senses:
            return np.zeros(self.n)

        total_weight = sum(s.weight for s in self._senses)
        if total_weight < 1e-10:
            return np.zeros(self.n)

        fused = np.zeros(self.n)
        for sense in self._senses:
            v = np.asarray(sense.values).flatten()
            n = min(len(v), self.n)
            fused[:n] += sense.weight * v[:n]

        return fused / total_weight

    def reason(
        self,
        h: np.ndarray,
        J: sparse.csr_matrix,
        *,
        sila_gamma: float = 0.0,
    ) -> ReasonerResult:
        # Fuse sensory input with energy landscape
        sensory = self.fuse_senses()
        V_s = -h.copy() + sensory
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

        return ReasonerResult(
            selected_indices=selected, energy=energy,
            solver_used="embodied_fep_analog", reasoning_mode="embodied",
            steps_used=steps_used, power_history=power,
            diagnostics={
                "n_senses": len(self._senses),
                "sense_names": [s.name for s in self._senses],
            },
        )
