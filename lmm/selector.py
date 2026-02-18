"""SmartSelector — adaptive selection with cascade filtering and history learning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lmm.qubo import QUBOBuilder
from lmm.solvers import ClassicalQUBOSolver
from lmm.surprise import SurpriseCalculator


@dataclass
class SelectionResult:
    indices: np.ndarray
    scores: np.ndarray
    method_used: str
    confidence: float
    budget_used: float


@dataclass
class _SolverRecord:
    method: str
    confidence: float


class SmartSelector:
    """Adaptive selection: cascade filter → solver selection → QUBO solve."""

    def __init__(
        self,
        k: int = 10,
        alpha: float = 1.0,
        gamma: float = 10.0,
        surprise_method: str = "entropy",
    ):
        self.k = k
        self.alpha = alpha
        self.gamma = gamma
        self._calculator = SurpriseCalculator(method=surprise_method)
        self._history: list[_SolverRecord] = []

    def fit(self, reference_data: np.ndarray) -> SmartSelector:
        self._calculator.fit(reference_data)
        return self

    def select(self, candidates: np.ndarray, budget: float = 1.0) -> SelectionResult:
        n = len(candidates)
        k = min(self.k, n)
        surprises = self._calculator.compute(candidates)

        if n > k * 5 and budget < 0.8:
            survivors, surprises_f = self._cascade_filter(surprises, k)
        else:
            survivors, surprises_f = np.arange(n), surprises

        method = self._pick_method(len(survivors), budget)
        selected_local = self._solve(surprises_f, k, method)
        selected_global = survivors[selected_local]
        confidence = self._compute_confidence(surprises, selected_global)
        self._history.append(_SolverRecord(method=method, confidence=confidence))

        return SelectionResult(
            indices=selected_global, scores=surprises[selected_global],
            method_used=method, confidence=confidence, budget_used=budget,
        )

    def select_from_surprises(
        self, surprises: np.ndarray, budget: float = 1.0,
    ) -> SelectionResult:
        n = len(surprises)
        k = min(self.k, n)

        if n > k * 5 and budget < 0.8:
            survivors, surprises_f = self._cascade_filter(surprises, k)
        else:
            survivors, surprises_f = np.arange(n), surprises

        method = self._pick_method(len(survivors), budget)
        selected_local = self._solve(surprises_f, k, method)
        selected_global = survivors[selected_local]
        confidence = self._compute_confidence(surprises, selected_global)
        self._history.append(_SolverRecord(method=method, confidence=confidence))

        return SelectionResult(
            indices=selected_global, scores=surprises[selected_global],
            method_used=method, confidence=confidence, budget_used=budget,
        )

    # -- internals -------------------------------------------------------

    def _cascade_filter(
        self, surprises: np.ndarray, k: int, ratio: float = 3.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        keep = min(int(k * ratio), len(surprises))
        top = np.argsort(surprises)[-keep:]
        return top, surprises[top]

    def _pick_method(self, n_candidates: int, budget: float) -> str:
        if n_candidates <= 50 or budget < 0.3:
            return "greedy"
        if self._history:
            best = self._best_historical_method()
            if best is not None:
                return best
        return "sa" if budget >= 0.7 else "relaxation"

    def _best_historical_method(self) -> str | None:
        if len(self._history) < 3:
            return None
        scores: dict[str, list[float]] = {}
        for rec in self._history[-20:]:
            scores.setdefault(rec.method, []).append(rec.confidence)
        best_method, best_avg = None, -1.0
        for m, sc in scores.items():
            avg = sum(sc) / len(sc)
            if avg > best_avg:
                best_avg, best_method = avg, m
        return best_method

    def _solve(self, surprises: np.ndarray, k: int, method: str) -> np.ndarray:
        n = len(surprises)
        k = min(k, n)
        builder = QUBOBuilder(n)
        builder.add_surprise_objective(surprises, alpha=self.alpha)
        builder.add_cardinality_constraint(k, gamma=self.gamma)
        solver = ClassicalQUBOSolver(builder)
        x = solver.solve(method=method, k=k)
        return np.where(x > 0.5)[0][:k]

    def _compute_confidence(
        self, all_surprises: np.ndarray, selected: np.ndarray,
    ) -> float:
        if len(selected) == 0:
            return 0.0
        sel_mean = all_surprises[selected].mean()
        total_mean = all_surprises.mean()
        total_std = all_surprises.std()
        if total_std < 1e-10:
            return 0.5
        z = (sel_mean - total_mean) / total_std
        return float(np.clip(1.0 / (1.0 + np.exp(-z)), 0.0, 1.0))
