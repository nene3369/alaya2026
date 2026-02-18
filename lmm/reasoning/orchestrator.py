"""DharmaReasonerOrchestrator — mode selection and ensemble reasoning."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

from lmm.reasoning.alaya import AlayaMemory
from lmm.reasoning.base import BaseReasoner, ReasonerResult, compute_complexity


@dataclass
class OrchestratorResult:
    """Result from orchestrated reasoning."""

    best_result: ReasonerResult
    all_results: dict[str, ReasonerResult] = field(default_factory=dict)
    mode_selected: str = ""
    ensemble_indices: np.ndarray | None = None


class DharmaReasonerOrchestrator:
    """Orchestrator — mode selection, run_all, ensemble voting.

    Auto-selects the best reasoning mode based on input characteristics,
    or runs all modes and uses ensemble voting.
    """

    def __init__(self):
        self._reasoners: dict[str, BaseReasoner] = {}

    def register(self, reasoner: BaseReasoner) -> None:
        """Register a reasoning mode."""
        self._reasoners[reasoner.mode] = reasoner

    @property
    def available_modes(self) -> list[str]:
        return list(self._reasoners.keys())

    def select_mode(
        self,
        h: np.ndarray,
        **kwargs,
    ) -> str:
        """Auto-select reasoning mode based on input characteristics."""
        if not self._reasoners:
            raise RuntimeError("No reasoners registered")

        complexity = compute_complexity(np.abs(h))
        c = complexity.complexity_score

        # Low complexity → adaptive (fast convergence)
        if c < 0.3 and "adaptive" in self._reasoners:
            return "adaptive"
        # Medium complexity → theoretical (logical consistency)
        if c < 0.6 and "theoretical" in self._reasoners:
            return "theoretical"
        # High complexity → hyper (abductive exploration)
        if "hyper" in self._reasoners:
            return "hyper"

        # Fallback to first available
        return next(iter(self._reasoners))

    def reason(
        self,
        h: np.ndarray,
        J: sparse.csr_matrix,
        *,
        mode: str | None = None,
        **kwargs,
    ) -> OrchestratorResult:
        """Run reasoning with selected or auto-detected mode."""
        if mode is None:
            mode = self.select_mode(h, **kwargs)

        if mode not in self._reasoners:
            raise ValueError(
                f"Unknown mode {mode!r}. Available: {self.available_modes}"
            )

        result = self._reasoners[mode].reason(h, J, **kwargs)

        return OrchestratorResult(
            best_result=result,
            all_results={mode: result},
            mode_selected=mode,
        )

    def run_all(
        self,
        h: np.ndarray,
        J: sparse.csr_matrix,
        **kwargs,
    ) -> OrchestratorResult:
        """Run all registered modes and select the best."""
        if not self._reasoners:
            raise RuntimeError("No reasoners registered")

        results: dict[str, ReasonerResult] = {}
        for mode_name, reasoner in self._reasoners.items():
            try:
                results[mode_name] = reasoner.reason(h, J, **kwargs)
            except (ValueError, RuntimeError, TypeError, np.linalg.LinAlgError):
                continue

        if not results:
            raise RuntimeError("All reasoning modes failed")

        # Select best by energy (lowest)
        best_mode = min(results, key=lambda m: results[m].energy)
        best = results[best_mode]

        return OrchestratorResult(
            best_result=best,
            all_results=results,
            mode_selected=best_mode,
        )

    def ensemble_vote(
        self,
        h: np.ndarray,
        J: sparse.csr_matrix,
        k: int,
        **kwargs,
    ) -> OrchestratorResult:
        """Ensemble voting across all modes."""
        all_result = self.run_all(h, J, **kwargs)

        # Count votes for each index
        n = len(h)
        votes = np.zeros(n)
        for result in all_result.all_results.values():
            for idx in result.selected_indices:
                votes[int(idx)] += 1.0

        # Select top-k by vote count (tie-break by |h|)
        abs_h = np.array([abs(float(x)) for x in h])
        # Combine votes with h as secondary sort
        combined = votes * n + abs_h
        ensemble_indices = np.argsort(-combined)[:k]
        ensemble_indices = np.array(sorted(int(i) for i in ensemble_indices))

        all_result.ensemble_indices = ensemble_indices
        return all_result

    def think(
        self,
        h: np.ndarray,
        J: sparse.csr_matrix,
        *,
        alaya: AlayaMemory | None = None,
        hyper_convergence_threshold: float = 0.01,
        sila_gamma: float = 0.0,
        **kwargs,
    ) -> OrchestratorResult:
        """Multi-phase reasoning with escalation and AlayaMemory recording.

        Phase 1: Standard mode (auto-selected based on complexity)
        Phase 2-3: Escalate to hyper if P_err >= threshold
        Phase 4: Escalate to active inference if still not converged
        Phase 5: Re-compute final_P_err after each escalation
        Phase 6: Record to AlayaMemory if converged

        Parameters
        ----------
        h : (n,) array
            Linear bias vector.
        J : (n, n) sparse matrix
            Coupling matrix.
        alaya : AlayaMemory, optional
            Memory store for recording successful reasoning patterns.
        hyper_convergence_threshold : float
            P_err threshold below which reasoning is considered converged.
        sila_gamma : float
            Cardinality constraint weight.
        **kwargs
            Mode-specific parameters forwarded to the initial reasoner.
        """
        if not self._reasoners:
            raise RuntimeError("No reasoners registered")

        # Phase 1: Standard reasoning
        mode = self.select_mode(h, **kwargs)
        result = self._reasoners[mode].reason(
            h, J, sila_gamma=sila_gamma, **kwargs,
        )
        power_history = list(result.power_history) if result.power_history else []
        final_P_err = power_history[-1] if power_history else float("inf")
        status = mode

        # Phase 2-3: Escalate to hyper reasoner
        if (
            final_P_err >= hyper_convergence_threshold
            and "hyper" in self._reasoners
            and status != "hyper"
        ):
            hyper_result = self._reasoners["hyper"].reason(
                h, J, sila_gamma=sila_gamma,
            )
            power_history.extend(hyper_result.power_history or [])
            final_P_err = power_history[-1] if power_history else float("inf")
            if hyper_result.energy < result.energy:
                result = hyper_result
                status = "hyper"

        # Phase 4: Escalate to active inference
        if (
            final_P_err >= hyper_convergence_threshold
            and "active" in self._reasoners
            and status != "active"
        ):
            active_result = self._reasoners["active"].reason(
                h, J, sila_gamma=sila_gamma,
            )
            power_history.extend(active_result.power_history or [])
            final_P_err = power_history[-1] if power_history else float("inf")
            if active_result.energy < result.energy:
                result = active_result
                status = "active"

        # Phase 5: De-escalation — if converged at higher level, try simpler
        if (
            final_P_err < hyper_convergence_threshold * 0.1
            and status != mode
            and "adaptive" in self._reasoners
        ):
            # Problem was easier than expected; de-escalate to save compute
            adaptive_result = self._reasoners["adaptive"].reason(
                h, J, sila_gamma=sila_gamma, **kwargs,
            )
            if adaptive_result.energy <= result.energy * 1.1:
                result = adaptive_result
                status = "adaptive"

        # Phase 6: Record to AlayaMemory
        converged = (status == mode) or (
            final_P_err < hyper_convergence_threshold
        )
        if alaya is not None:
            n = len(h)
            binary = np.zeros(n)
            for idx in result.selected_indices:
                binary[int(idx)] = 1.0
            alaya.record_and_learn(pattern=binary, converged=converged)

        return OrchestratorResult(
            best_result=result,
            all_results={status: result},
            mode_selected=status,
        )
