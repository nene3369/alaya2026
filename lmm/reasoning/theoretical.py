"""TheoreticalReasoner — logical graph reasoning (implication/contradiction/support)."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from lmm.dharma.fep import solve_fep_kcl_analog
from lmm.reasoning.base import BaseReasoner, ReasonerResult


class TheoreticalReasoner(BaseReasoner):
    """Logical graph reasoning — build implication/contradiction/support edges.

    Constructs a logical interaction matrix from proposition relationships
    and uses FEP ODE to find consistent, high-value selections.
    """

    def __init__(
        self,
        n_variables: int,
        k: int,
        *,
        support_weight: float = 0.5,
        contradiction_weight: float = -1.0,
        implication_weight: float = 0.3,
        nirvana_threshold: float = 1e-4,
        max_steps: int = 500,
        G_prec: float = 8.0,
        tau_leak: float = 1.5,
    ):
        super().__init__(n_variables, k, nirvana_threshold=nirvana_threshold)
        self.support_weight = support_weight
        self.contradiction_weight = contradiction_weight
        self.implication_weight = implication_weight
        self.max_steps = max_steps
        self.G_prec = G_prec
        self.tau_leak = tau_leak

    @property
    def mode(self) -> str:
        return "theoretical"

    def build_logical_graph(
        self,
        relations: list[tuple[int, int, str]],
    ) -> sparse.csr_matrix:
        """Build logical interaction matrix from relations.

        Parameters
        ----------
        relations : list of (i, j, type)
            type is one of "support", "contradiction", "implication"
        """
        rows, cols, vals = [], [], []
        for i, j, rtype in relations:
            if rtype == "support":
                w = self.support_weight
            elif rtype == "contradiction":
                w = self.contradiction_weight
            elif rtype == "implication":
                w = self.implication_weight
            else:
                continue
            rows.extend([i, j])
            cols.extend([j, i])
            vals.extend([w, w])

        if not vals:
            return self._empty_csr(self.n)

        return sparse.csr_matrix(
            (vals, (rows, cols)), shape=(self.n, self.n),
        )

    def reason(
        self,
        h: np.ndarray,
        J: sparse.csr_matrix,
        *,
        relations: list[tuple[int, int, str]] | None = None,
        sila_gamma: float = 0.0,
    ) -> ReasonerResult:
        if relations is not None:
            J_logical = self.build_logical_graph(relations)
            if J.nnz > 0:
                J_combined = J + J_logical
            else:
                J_combined = J_logical
        else:
            J_combined = J

        V_s = -h.copy()
        if sila_gamma > 0:
            V_s = V_s - sila_gamma * (1.0 - 2.0 * self.k)

        J_dynamic = J_combined * (-1.0) if J_combined.nnz > 0 else J_combined

        V_mu, x_final, steps_used, power_history = solve_fep_kcl_analog(
            V_s=V_s, J_dynamic=J_dynamic, n=self.n,
            G_prec_base=self.G_prec, tau_leak=self.tau_leak,
            max_steps=self.max_steps,
            nirvana_threshold=self.nirvana_threshold,
        )

        selected = self._topk_from_activations(x_final, self.k)
        energy = self._evaluate_energy(h, J, sila_gamma, selected)

        n_relations = len(relations) if relations else 0
        return ReasonerResult(
            selected_indices=selected, energy=energy,
            solver_used="theoretical_fep_analog", reasoning_mode="theoretical",
            steps_used=steps_used, power_history=power_history,
            diagnostics={"n_relations": n_relations},
        )
