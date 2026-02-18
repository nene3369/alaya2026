"""HyperReasoner — abductive reasoning with latent node injection."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from lmm.dharma.fep import solve_fep_kcl_analog
from lmm.reasoning.base import BaseReasoner, ReasonerResult


class HyperReasoner(BaseReasoner):
    """Abductive reasoning — inject latent hypothetical nodes.

    Adds virtual nodes representing latent hypotheses that can explain
    observed patterns. The FEP ODE then finds selections that are
    consistent with both observed and latent evidence.
    """

    def __init__(
        self,
        n_variables: int,
        k: int,
        *,
        n_latent: int = 5,
        latent_coupling: float = 0.3,
        nirvana_threshold: float = 1e-4,
        max_steps: int = 500,
        G_prec: float = 8.0,
        tau_leak: float = 2.0,
    ):
        super().__init__(n_variables, k, nirvana_threshold=nirvana_threshold)
        self.n_latent = n_latent
        self.latent_coupling = latent_coupling
        self.max_steps = max_steps
        self.G_prec = G_prec
        self.tau_leak = tau_leak

    @property
    def mode(self) -> str:
        return "hyper"

    def _inject_latent_nodes(
        self,
        h: np.ndarray,
        J: sparse.csr_matrix,
    ) -> tuple[np.ndarray, sparse.csr_matrix, int]:
        """Inject latent nodes into the problem space.

        Number of latent nodes adapts to problem complexity:
        n_latent = base + int(4 * complexity), capped at 2*base.
        """
        n_orig = len(h)
        # Adaptive latent count based on h complexity
        abs_h = np.array([abs(float(x)) for x in h])
        _mean_h = float(np.sum(abs_h)) / max(len(abs_h), 1)
        _var_h = float(np.sum((abs_h - _mean_h) ** 2)) / max(len(abs_h), 1)
        _std_h = float(np.sqrt(_var_h)) if _var_h > 0 else 0.0
        complexity = _std_h / max(_mean_h, 1e-8)
        n_latent = max(2, min(self.n_latent * 2, self.n_latent + int(4 * complexity)))
        n_total = n_orig + n_latent

        # Extend h with small priors for latent nodes
        h_ext = np.zeros(n_total)
        h_ext[:n_orig] = h

        # Build extended J with latent-to-observed couplings
        rows, cols, vals = [], [], []

        # Copy original J
        if J.nnz > 0:
            coo = J.tocoo()
            rows.extend(int(r) for r in coo.row)
            cols.extend(int(c) for c in coo.col)
            vals.extend(float(v) for v in coo.data)

        # Connect latent nodes to top-scoring observed nodes
        top_indices = np.argsort(-abs_h)[:min(n_orig, n_latent * 3)]

        rng = np.random.default_rng(42)
        for lat_idx in range(n_latent):
            node_id = n_orig + lat_idx
            # Each latent node connects to a subset of top observed nodes
            n_connect = min(len(top_indices), 3)
            # Energy-weighted connection probability
            top_h = abs_h[top_indices]
            probs = top_h / max(float(top_h.sum()), 1e-8)
            chosen = rng.choice(top_indices, size=n_connect, replace=False, p=probs)
            for obs_idx in chosen:
                w = self.latent_coupling * rng.uniform(0.5, 1.0)
                rows.extend([node_id, int(obs_idx)])
                cols.extend([int(obs_idx), node_id])
                vals.extend([w, w])

        J_ext = sparse.csr_matrix(
            (vals, (rows, cols)), shape=(n_total, n_total),
        ) if vals else self._empty_csr(n_total)

        return h_ext, J_ext, n_total

    def reason(
        self,
        h: np.ndarray,
        J: sparse.csr_matrix,
        *,
        sila_gamma: float = 0.0,
    ) -> ReasonerResult:
        n_orig = len(h)
        h_ext, J_ext, n_total = self._inject_latent_nodes(h, J)

        V_s = -h_ext.copy()
        if sila_gamma > 0:
            V_s = V_s - sila_gamma * (n_orig - 2 * self.k)

        J_dynamic = J_ext * (-1.0) if J_ext.nnz > 0 else J_ext

        V_mu, x_final, steps_used, power_history = solve_fep_kcl_analog(
            V_s=V_s, J_dynamic=J_dynamic, n=n_total,
            G_prec_base=self.G_prec, tau_leak=self.tau_leak,
            max_steps=self.max_steps,
            nirvana_threshold=self.nirvana_threshold,
        )

        # Select only from original nodes
        x_orig = x_final[:n_orig]
        selected = self._topk_from_activations(x_orig, self.k)

        # Latent node activations for diagnostics
        latent_activations = x_final[n_orig:]
        energy = self._evaluate_energy(h, J, sila_gamma, selected)

        return ReasonerResult(
            selected_indices=selected, energy=energy,
            solver_used="hyper_fep_analog", reasoning_mode="hyper",
            steps_used=steps_used, power_history=power_history,
            diagnostics={
                "n_latent": n_total - n_orig,
                "latent_activations": latent_activations,
            },
        )
