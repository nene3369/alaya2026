"""Buddhist philosophical concepts as mathematical structures."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DharmaWeights:
    """Optimisation weights with Buddhist interpretation.

    alpha (Prajna): surprise weight = depth of wisdom
    beta  (Karuna): propagation weight = strength of compassion
    gamma (Sila):   constraint weight = strictness of discipline
    """

    alpha: float = 1.0
    beta: float = 0.5
    gamma: float = 10.0


@dataclass
class BodhisattvaObjective:
    """Bodhisattva objective function.

    min: -alpha * sum(s_i * x_i) + beta * sum(W_ij * x_i * (1-x_j)) + gamma * (sum(x_i) - k)^2
    """

    surprises: np.ndarray
    impact_graph: np.ndarray
    k: int
    weights: DharmaWeights

    @property
    def n(self) -> int:
        return len(self.surprises)


@dataclass
class MadhyamakaCriterion:
    """Middle Way criterion: target CV(surprise) = 0.5 (edge of chaos)."""

    target_cv: float = 0.5
    learning_rate: float = 0.1

    def evaluate(self, surprises: np.ndarray) -> float:
        mean = surprises.mean()
        if mean < 1e-10:
            return 0.0
        cv = surprises.std() / mean
        return abs(cv - self.target_cv)

    def is_balanced(self, surprises: np.ndarray, tolerance: float = 0.1) -> bool:
        return self.evaluate(surprises) < tolerance
