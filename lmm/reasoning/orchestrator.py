"""DharmaReasonerOrchestrator — mode selection and ensemble reasoning."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

from lmm.reasoning.alaya import AlayaMemory
from lmm.reasoning.base import BaseReasoner, ReasonerResult, compute_complexity
from lmm.reasoning.recovery import CircuitBreaker

logger = logging.getLogger(__name__)


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
        self._breakers: dict[str, CircuitBreaker] = {}

    def register(self, reasoner: BaseReasoner) -> None:
        """Register a reasoning mode."""
        self._reasoners[reasoner.mode] = reasoner
        self._breakers[reasoner.mode] = CircuitBreaker(
            name=f"reasoner:{reasoner.mode}",
            failure_threshold=3,
            cooldown=30.0,
        )

    @property
    def available_modes(self) -> list[str]:
        return list(self._reasoners.keys())

    def _is_available(self, mode: str) -> bool:
        """Check if a mode is registered and its circuit breaker allows requests."""
        if mode not in self._reasoners:
            return False
        breaker = self._breakers.get(mode)
        return breaker is None or breaker.allow_request()

    def select_mode(
        self,
        h: np.ndarray,
        **kwargs,
    ) -> str:
        """Auto-select reasoning mode based on input characteristics.

        Skips modes whose circuit breakers are open (too many recent failures).
        """
        if not self._reasoners:
            raise RuntimeError("No reasoners registered")

        complexity = compute_complexity(np.abs(h))
        c = complexity.complexity_score

        # Low complexity → adaptive (fast convergence)
        if c < 0.3 and self._is_available("adaptive"):
            return "adaptive"
        # Medium complexity → theoretical (logical consistency)
        if c < 0.6 and self._is_available("theoretical"):
            return "theoretical"
        # High complexity → hyper (abductive exploration)
        if self._is_available("hyper"):
            return "hyper"

        # Fallback to first available mode with open circuit
        for mode in self._reasoners:
            if self._is_available(mode):
                return mode

        # All circuits open — force first reasoner (will reset on success)
        fallback = next(iter(self._reasoners))
        logger.debug("All circuit breakers open, forcing %s", fallback)
        return fallback

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
        # abs_h を [0,1) に正規化してタイブレーク専用にする。
        # votes は整数なので abs_h_norm < 1 との衝突がなく、票数が常に支配する。
        abs_h = np.array([abs(float(x)) for x in h])
        abs_h_norm = 0.99 * (abs_h - abs_h.min()) / (abs_h.max() - abs_h.min() + 1e-12)
        combined = votes + abs_h_norm
        ensemble_indices = np.argsort(-combined)[:k]
        ensemble_indices = np.array(sorted(int(i) for i in ensemble_indices))

        all_result.ensemble_indices = ensemble_indices
        return all_result

    PHASE_TIMEOUT: float = 5.0  # max seconds per reasoning phase
    # Pythonはスレッドを強制終了できないため、タイムアウト後もスレッドが生き続ける。
    # セマフォで同時実行スレッド数の上限を設けることで「ゾンビスレッドの無限増殖」を防ぐ。
    # タイムアウトしたスレッドは finally ブロックでセマフォを解放するため、
    # 時間が経てば枠が回復し、枯渇は一時的なものにとどまる。
    _inflight_sem: threading.Semaphore = threading.Semaphore(8)

    def _reason_with_timeout(
        self,
        reasoner_name: str,
        h: np.ndarray,
        J: sparse.csr_matrix,
        **kwargs,
    ) -> ReasonerResult | None:
        """Run a single reasoner with circuit breaker + wall-clock timeout.

        Returns None if the circuit is open, the reasoner times out,
        or raises an exception.  Records success/failure to the breaker.
        """
        breaker = self._breakers.get(reasoner_name)
        if breaker is not None and not breaker.allow_request():
            return None  # Circuit is open — skip this reasoner

        # 上限（8スレッド）に達していたら即 None を返してスレッドを作らない
        if not DharmaReasonerOrchestrator._inflight_sem.acquire(blocking=False):
            if breaker is not None:
                breaker.record_failure()
            return None

        reasoner = self._reasoners[reasoner_name]
        result_box: list[ReasonerResult | None] = [None]
        error_box: list[Exception | None] = [None]

        def _run():
            try:
                result_box[0] = reasoner.reason(h, J, **kwargs)
            except Exception as exc:
                error_box[0] = exc
            finally:
                # タイムアウト後に完了した場合も必ずセマフォを解放する
                DharmaReasonerOrchestrator._inflight_sem.release()

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        thread.join(timeout=self.PHASE_TIMEOUT)

        if thread.is_alive() or error_box[0] is not None:
            if breaker is not None:
                breaker.record_failure()
            return None

        if breaker is not None:
            breaker.record_success()
        return result_box[0]

    @property
    def circuit_breaker_stats(self) -> dict[str, dict]:
        """Return circuit breaker stats for all registered reasoners."""
        return {
            name: {
                "state": b.state.value,
                "failures": b._failure_count,
                "consecutive_failures": b._consecutive_failures,
            }
            for name, b in self._breakers.items()
        }

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

        Each phase has a wall-clock timeout (PHASE_TIMEOUT) to prevent
        freezing on slow or stuck reasoners.

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

        # Phase 1: Standard reasoning (with timeout)
        mode = self.select_mode(h, **kwargs)
        result = self._reason_with_timeout(
            mode, h, J, sila_gamma=sila_gamma, **kwargs,
        )
        if result is None:
            # Phase 1 timed out — direct call as last resort
            fallback = next(iter(self._reasoners))
            logger.debug("Phase 1 timeout, falling back to %s", fallback)
            try:
                result = self._reasoners[fallback].reason(
                    h, J, sila_gamma=sila_gamma,
                )
            except Exception:
                logger.debug("Fallback reasoner also failed", exc_info=True)
                raise RuntimeError("All reasoning phases failed")
            mode = fallback
        power_history = list(result.power_history) if result.power_history else []
        final_P_err = power_history[-1] if power_history else float("inf")
        status = mode

        # Phase 2-3: Escalate to hyper reasoner
        if (
            final_P_err >= hyper_convergence_threshold
            and "hyper" in self._reasoners
            and status != "hyper"
        ):
            hyper_result = self._reason_with_timeout(
                "hyper", h, J, sila_gamma=sila_gamma,
            )
            if hyper_result is not None:
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
            active_result = self._reason_with_timeout(
                "active", h, J, sila_gamma=sila_gamma,
            )
            if active_result is not None:
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
            adaptive_result = self._reason_with_timeout(
                "adaptive", h, J, sila_gamma=sila_gamma, **kwargs,
            )
            if adaptive_result is not None and adaptive_result.energy <= result.energy + abs(result.energy) * 0.1:
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
