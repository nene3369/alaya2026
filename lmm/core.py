"""LMM main module — surprise-based optimal selection pipeline.

Pipeline: data → surprise → QUBO → classical solver → top-K indices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from lmm._validation import (
    validate_array_finite,
    validate_k,
    validate_nonneg,
    warn_k_clamped,
)
from lmm.qubo import QUBOBuilder
from lmm.solvers import ClassicalQUBOSolver
from lmm.surprise import SurpriseCalculator

SurpriseMethod = Literal["kl", "entropy", "bayesian"]
SolverMethod = Literal["sa", "ising_sa", "greedy", "relaxation"]


@dataclass
class LMMResult:
    """Optimisation result."""

    selected_indices: np.ndarray
    surprise_values: np.ndarray
    energy: float
    method: str


class LMM:
    """Surprise-based optimal selection — no quantum computer required.

    Usage::

        model = LMM(k=10, solver_method="sa")
        model.fit(reference_data)
        result = model.select(candidates)
    """

    def __init__(
        self,
        k: int = 10,
        alpha: float = 1.0,
        gamma: float = 10.0,
        beta: float = 0.0,
        surprise_method: SurpriseMethod = "entropy",
        solver_method: SolverMethod = "sa",
    ):
        validate_k(k)
        validate_nonneg(alpha, "alpha")
        validate_nonneg(gamma, "gamma")
        validate_nonneg(beta, "beta")
        self.k = k
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.surprise_method = surprise_method
        self.solver_method = solver_method
        self._calculator = SurpriseCalculator(method=surprise_method)

    def fit(self, reference_data: np.ndarray) -> LMM:
        """Learn prior distribution from reference data."""
        self._calculator.fit(reference_data)
        return self

    def select(
        self,
        candidates: np.ndarray,
        similarity_matrix: np.ndarray | None = None,
    ) -> LMMResult:
        """Select K optimal items from candidates."""
        if len(candidates) == 0:
            raise ValueError("candidates must not be empty")
        validate_array_finite(candidates, "candidates")
        if similarity_matrix is not None:
            validate_array_finite(similarity_matrix, "similarity_matrix")
        n = len(candidates)
        k = warn_k_clamped(self.k, n)

        surprises = self._calculator.compute(candidates)

        builder = QUBOBuilder(n)
        builder.add_surprise_objective(surprises, alpha=self.alpha)
        builder.add_cardinality_constraint(k, gamma=self.gamma)
        if similarity_matrix is not None and self.beta > 0:
            builder.add_diversity_penalty(similarity_matrix, beta=self.beta)

        solver = ClassicalQUBOSolver(builder)
        x = solver.solve(method=self.solver_method, k=k)
        selected = np.where(x > 0.5)[0][:k]
        energy = builder.evaluate(x)

        return LMMResult(
            selected_indices=selected,
            surprise_values=surprises[selected],
            energy=energy,
            method=self.solver_method,
        )

    def select_from_surprises(self, surprises: np.ndarray) -> LMMResult:
        """Select from pre-computed surprise values."""
        if len(surprises) == 0:
            raise ValueError("surprises must not be empty")
        validate_array_finite(surprises, "surprises")
        n = len(surprises)
        k = warn_k_clamped(self.k, n)

        builder = QUBOBuilder(n)
        builder.add_surprise_objective(surprises, alpha=self.alpha)
        builder.add_cardinality_constraint(k, gamma=self.gamma)

        solver = ClassicalQUBOSolver(builder)
        x = solver.solve(method=self.solver_method, k=k)
        selected = np.where(x > 0.5)[0][:k]
        energy = builder.evaluate(x)

        return LMMResult(
            selected_indices=selected,
            surprise_values=surprises[selected],
            energy=energy,
            method=self.solver_method,
        )
