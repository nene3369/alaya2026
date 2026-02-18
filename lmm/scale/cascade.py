"""MultiLevelCascade — 3-level filter (Stream -> Percentile -> QUBO)."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from lmm._compat import HAS_ARGPARTITION
from lmm.qubo import QUBOBuilder
from lmm.scale.stream import StreamingSurprise
from lmm.solvers import ClassicalQUBOSolver


@dataclass
class CascadeStats:
    """Per-level cascade statistics."""

    level_sizes: list[int] = field(default_factory=list)
    level_names: list[str] = field(default_factory=list)

    def reduction_ratio(self) -> float:
        if len(self.level_sizes) < 2:
            return 1.0
        return self.level_sizes[0] / max(self.level_sizes[-1], 1)


@dataclass
class CascadeResult:
    """Cascade final result."""

    indices: np.ndarray
    scores: np.ndarray
    stats: CascadeStats


class MultiLevelCascade:
    """Multi-level filter: trillions -> thousands -> K."""

    def __init__(
        self,
        k: int = 10,
        stream_buffer: int = 10_000,
        percentile_keep: float = 0.3,
        alpha: float = 1.0,
        gamma: float = 10.0,
        solver_method: str = "sa",
        chunk_size: int = 100_000,
    ):
        self.k = k
        self.stream_buffer = max(stream_buffer, k * 10)
        self.percentile_keep = percentile_keep
        self.alpha = alpha
        self.gamma = gamma
        self.solver_method = solver_method
        self.chunk_size = chunk_size
        self._streamer = StreamingSurprise(
            k=self.stream_buffer, chunk_size=chunk_size,
        )

    def fit_stream(self, data_iter: Iterable[np.ndarray]) -> MultiLevelCascade:
        self._streamer.fit_stream(data_iter)
        return self

    def fit_array(self, data: np.ndarray) -> MultiLevelCascade:
        self._streamer.fit_array(data)
        return self

    def select_stream(self, data_iter: Iterable[np.ndarray]) -> CascadeResult:
        """Select K items from trillion-scale stream."""
        stats = CascadeStats()
        total_processed = 0
        for chunk_result in self._streamer.compute_stream(data_iter):
            total_processed += chunk_result.n_processed
        stats.level_sizes.append(total_processed)
        stats.level_names.append("stream_input")

        stream_indices, stream_scores = self._streamer.get_top_k()
        stats.level_sizes.append(len(stream_indices))
        stats.level_names.append("stream_top_k")

        if len(stream_indices) == 0:
            return CascadeResult(
                indices=np.array([], dtype=int),
                scores=np.array([], dtype=float), stats=stats,
            )

        # Level 1: Percentile filter
        keep_n = max(self.k, int(len(stream_scores) * self.percentile_keep))
        keep_n = min(keep_n, len(stream_scores))

        if keep_n < len(stream_scores):
            if HAS_ARGPARTITION:
                threshold_idx = np.argpartition(stream_scores, -keep_n)[-keep_n:]
            else:
                threshold_idx = np.argsort(stream_scores)[-keep_n:]
            pct_indices = stream_indices[threshold_idx]
            pct_scores = stream_scores[threshold_idx]
        else:
            pct_indices, pct_scores = stream_indices, stream_scores

        stats.level_sizes.append(len(pct_indices))
        stats.level_names.append("percentile_filter")

        # Level 2: QUBO
        k = min(self.k, len(pct_scores))
        selected_local = self._solve_qubo(pct_scores, k)
        stats.level_sizes.append(len(selected_local))
        stats.level_names.append("qubo_selected")

        return CascadeResult(
            indices=pct_indices[selected_local],
            scores=pct_scores[selected_local], stats=stats,
        )

    def select_array(self, data: np.ndarray) -> CascadeResult:
        def chunks():
            for start in range(0, len(data), self.chunk_size):
                yield data[start:start + self.chunk_size]
        return self.select_stream(chunks())

    def _solve_qubo(self, surprises: np.ndarray, k: int) -> np.ndarray:
        n = len(surprises)
        k = min(k, n)
        builder = QUBOBuilder(n)
        builder.add_surprise_objective(surprises, alpha=self.alpha)
        builder.add_cardinality_constraint(k, gamma=self.gamma)
        solver = ClassicalQUBOSolver(builder)
        if self.solver_method == "sa":
            x = solver.solve_sa(k=k)
        elif self.solver_method == "relaxation":
            x = solver.solve_relaxation()
        else:
            x = solver.solve_greedy(k=k)
        return np.where(x > 0.5)[0][:k]
