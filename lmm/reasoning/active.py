"""ActiveInferenceEngine — action-perception loop via FEP."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

from lmm.dharma.fep import solve_fep_kcl_analog
from lmm.reasoning.base import BaseReasoner, ReasonerResult


@dataclass
class ActiveState:
    """State of the active inference loop."""

    belief: np.ndarray
    action_history: list[np.ndarray] = field(default_factory=list)
    prediction_errors: list[float] = field(default_factory=list)


@dataclass
class ActionResult:
    """External action result to inject into active inference.

    Represents knowledge obtained from external actions (e.g., tool calls,
    database lookups) that should influence the belief update.
    """

    node_indices: np.ndarray
    """Indices of nodes affected by the action."""

    relevance_score: float = 0.5
    """How relevant the action result is, in [0, 1]."""


class ActiveInferenceEngine(BaseReasoner):
    """Active inference — iterative action-perception loop.

    Each iteration:
      1. Perceive: compute prediction error from current belief
      2. Infer: update belief via FEP ODE
      3. Act: select actions that minimize expected free energy
    """

    def __init__(
        self,
        n_variables: int,
        k: int,
        *,
        n_iterations: int = 3,
        G_prec: float = 8.0,
        tau_leak: float = 1.5,
        max_steps: int = 300,
        nirvana_threshold: float = 1e-4,
        action_learning_rate: float = 0.1,
    ):
        super().__init__(n_variables, k, nirvana_threshold=nirvana_threshold)
        self.n_iterations = n_iterations
        self.G_prec = G_prec
        self.tau_leak = tau_leak
        self.max_steps = max_steps
        self.action_learning_rate = action_learning_rate

    @property
    def mode(self) -> str:
        return "active"

    def _inject_action_result(
        self,
        V_s: np.ndarray,
        action_result: ActionResult,
        sila_gamma: float = 0.0,
    ) -> np.ndarray:
        """Inject external action result into sensory input with Sila bias.

        Modifies the sensory potential *V_s* to incorporate external knowledge
        about specific nodes.  The *relevance_score* is applied as the new
        V_s value for affected nodes, with the same sila_gamma cardinality
        bias that existing nodes receive.

        Parameters
        ----------
        V_s : (n,) array
            Current sensory potential.
        action_result : ActionResult
            External knowledge to inject.
        sila_gamma : float
            Cardinality constraint weight for Sila bias.

        Returns
        -------
        V_s_new : (n,) array
            Updated sensory potential.
        """
        V_s_new = V_s.copy()
        score = (
            action_result.relevance_score
            if action_result.relevance_score > 0
            else 0.5
        )

        # Apply sila_gamma bias consistently with existing V_s computation
        if sila_gamma > 0:
            score = score - sila_gamma * (self.n - 2 * self.k)

        for idx in action_result.node_indices:
            idx_int = int(idx)
            if 0 <= idx_int < len(V_s_new):
                V_s_new[idx_int] = score

        return V_s_new

    def reason(
        self,
        h: np.ndarray,
        J: sparse.csr_matrix,
        *,
        observations: np.ndarray | None = None,
        sila_gamma: float = 0.0,
        action_results: list[ActionResult] | None = None,
    ) -> ReasonerResult:
        state = ActiveState(belief=np.zeros(self.n))
        total_steps = 0
        all_power: list[float] = []

        V_s_base = -h.copy()
        if sila_gamma > 0:
            V_s_base = V_s_base - sila_gamma * (self.n - 2 * self.k)

        J_dynamic = J * (-1.0) if J.nnz > 0 else J

        for iteration in range(self.n_iterations):
            # Modulate sensory input with current belief
            V_s = V_s_base.copy()
            if observations is not None:
                pred_error = observations - state.belief
                V_s = V_s + self.action_learning_rate * pred_error
                state.prediction_errors.append(float(np.dot(pred_error, pred_error)))

            # Inject external action results if available
            if action_results:
                for ar in action_results:
                    V_s = self._inject_action_result(V_s, ar, sila_gamma=sila_gamma)

            V_mu, x_final, steps_used, power = solve_fep_kcl_analog(
                V_s=V_s, J_dynamic=J_dynamic, n=self.n,
                G_prec_base=self.G_prec, tau_leak=self.tau_leak,
                max_steps=self.max_steps,
                nirvana_threshold=self.nirvana_threshold,
                initial_state=state.belief if iteration > 0 else None,
            )

            state.belief = V_mu.copy()
            state.action_history.append(x_final.copy())
            total_steps += steps_used
            all_power.extend(power)

        selected = self._topk_from_activations(x_final, self.k)
        energy = self._evaluate_energy(h, J, sila_gamma, selected)

        return ReasonerResult(
            selected_indices=selected, energy=energy,
            solver_used="active_inference", reasoning_mode="active",
            steps_used=total_steps, power_history=all_power,
            diagnostics={
                "n_iterations": self.n_iterations,
                "prediction_errors": state.prediction_errors,
            },
        )
