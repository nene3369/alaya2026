"""AdaptiveReasoner — dynamic FEP parameter tuning based on complexity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from lmm.dharma.fep import solve_fep_kcl_analog
from lmm.reasoning.base import (
    BaseReasoner,
    ComplexityProfile,
    ReasonerResult,
    compute_complexity,
)


@dataclass
class AdaptiveParams:
    """Auto-determined FEP parameters."""

    tau_leak: float
    max_steps: int
    dt: float
    G_prec: float


class AdaptiveReasoner(BaseReasoner):
    """Adaptive reasoning — auto-tune FEP parameters from complexity."""

    def __init__(
        self,
        n_variables: int,
        k: int,
        *,
        tau_range: tuple[float, float] = (0.5, 3.0),
        steps_range: tuple[int, int] = (200, 1000),
        dt_range: tuple[float, float] = (0.005, 0.05),
        G_prec_range: tuple[float, float] = (3.0, 15.0),
        nirvana_threshold: float = 1e-4,
    ):
        super().__init__(n_variables, k, nirvana_threshold=nirvana_threshold)
        self.tau_range = tau_range
        self.steps_range = steps_range
        self.dt_range = dt_range
        self.G_prec_range = G_prec_range

    @property
    def mode(self) -> str:
        return "adaptive"

    def adapt_parameters(self, complexity: ComplexityProfile) -> AdaptiveParams:
        """Adapt FEP parameters from complexity profile."""
        c = complexity.complexity_score
        tau = self.tau_range[0] + (self.tau_range[1] - self.tau_range[0]) * c
        steps = int(self.steps_range[0] + (self.steps_range[1] - self.steps_range[0]) * c)
        dt = self.dt_range[1] - (self.dt_range[1] - self.dt_range[0]) * c
        G = self.G_prec_range[1] - (self.G_prec_range[1] - self.G_prec_range[0]) * c
        return AdaptiveParams(tau_leak=tau, max_steps=steps, dt=dt, G_prec=G)

    def reason(
        self,
        h: np.ndarray,
        J: sparse.csr_matrix,
        *,
        relevance_scores: np.ndarray | None = None,
        sila_gamma: float = 0.0,
    ) -> ReasonerResult:
        values = relevance_scores if relevance_scores is not None else np.abs(h)
        complexity = compute_complexity(values)
        params = self.adapt_parameters(complexity)

        V_s = -h.copy()
        if sila_gamma > 0:
            V_s = V_s - sila_gamma * (1.0 - 2.0 * self.k)

        J_dynamic = J * (-1.0) if J.nnz > 0 else J

        V_mu, x_final, steps_used, power_history = solve_fep_kcl_analog(
            V_s=V_s, J_dynamic=J_dynamic, n=self.n,
            G_prec_base=params.G_prec, tau_leak=params.tau_leak,
            dt=params.dt, max_steps=params.max_steps,
            nirvana_threshold=self.nirvana_threshold,
        )

        selected = self._topk_from_activations(x_final, self.k)
        energy = self._evaluate_energy(h, J, sila_gamma, selected)

        return ReasonerResult(
            selected_indices=selected, energy=energy,
            solver_used="adaptive_fep_analog", reasoning_mode="adaptive",
            steps_used=steps_used, power_history=power_history,
            complexity=complexity,
            diagnostics={"params": params},
        )
