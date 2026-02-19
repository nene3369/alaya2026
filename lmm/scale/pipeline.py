"""ScalablePipeline — trillion-token orchestration.

Combines MultiLevelCascade (Stream → Percentile → QUBO) with SmartProcessor
(priority-tiered processing) and optional SanghaOrchestrator (council
deliberation with AlayaMemory RAG) for cost-efficient end-to-end selection.
"""

from __future__ import annotations

from collections.abc import Callable, Generator, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from lmm.dharma.sangha import CouncilResult, SanghaOrchestrator
from lmm.processor import ProcessingReport, SmartProcessor
from lmm.scale.cascade import CascadeResult, CascadeStats, MultiLevelCascade


@dataclass
class ScalableResult:
    """Scalable pipeline result."""

    indices: np.ndarray
    scores: np.ndarray
    cascade_stats: CascadeStats
    processing: ProcessingReport
    council: CouncilResult | None = field(default=None, repr=False)

    @property
    def summary(self) -> dict:
        d = {
            "selected": len(self.indices),
            "total_input": self.cascade_stats.level_sizes[0] if self.cascade_stats.level_sizes else 0,
            "reduction": f"{self.cascade_stats.reduction_ratio():.0f}x",
            "cascade_levels": dict(zip(
                self.cascade_stats.level_names,
                self.cascade_stats.level_sizes,
            )),
            "critical": self.processing.n_critical,
            "normal": self.processing.n_normal,
            "background": self.processing.n_background,
        }
        if self.council is not None:
            d["council_verdict"] = self.council.final
        return d


class ScalablePipeline:
    """Trillion-token pipeline with automatic mode selection.

    When *use_sangha* is True, a Sangha council deliberates over the
    cascade-filtered selection.  An optional *alaya_memory* (duck-typed
    vector DB with ``search(query, limit)`` method) is forwarded to
    Sariputra and Mahakasyapa for near-zero-cost RAG retrieval.
    """

    def __init__(
        self,
        k: int = 10,
        stream_buffer: int = 10_000,
        percentile_keep: float = 0.3,
        alpha: float = 1.0,
        gamma: float = 10.0,
        solver_method: str = "sa",
        chunk_size: int = 100_000,
        process_fn: Callable[[int, float], object] | None = None,
        critical_threshold: float = 0.8,
        use_sangha: bool = False,
        alaya_memory: Any = None,
        sangha_timeout: float = 5.0,
    ):
        self._cascade = MultiLevelCascade(
            k=k, stream_buffer=stream_buffer,
            percentile_keep=percentile_keep, alpha=alpha,
            gamma=gamma, solver_method=solver_method,
            chunk_size=chunk_size,
        )
        self._processor = SmartProcessor(
            process_fn=process_fn, critical_threshold=critical_threshold,
        )
        self._chunk_size = chunk_size
        self._sangha = SanghaOrchestrator(
            alaya_memory=alaya_memory, timeout=sangha_timeout,
        ) if use_sangha else None

    def fit_stream(self, data_iter: Iterable[np.ndarray]) -> ScalablePipeline:
        self._cascade.fit_stream(data_iter)
        return self

    def fit_array(self, data: np.ndarray) -> ScalablePipeline:
        self._cascade.fit_array(data)
        return self

    def fit_files(self, paths: list[Path]) -> ScalablePipeline:
        self._cascade.fit_stream(_file_chunk_iter(paths, self._chunk_size))
        return self

    def run_stream(self, data_iter: Iterable[np.ndarray]) -> ScalableResult:
        return self._post_process(self._cascade.select_stream(data_iter))

    def run_array(self, data: np.ndarray) -> ScalableResult:
        return self._post_process(self._cascade.select_array(data))

    def run_files(self, paths: list[Path]) -> ScalableResult:
        return self._post_process(
            self._cascade.select_stream(_file_chunk_iter(paths, self._chunk_size))
        )

    def _post_process(self, cascade_result: CascadeResult) -> ScalableResult:
        if len(cascade_result.indices) == 0:
            processing = self._processor.process(
                np.array([], dtype=int), np.array([], dtype=float),
            )
        else:
            processing = self._processor.process(
                cascade_result.indices, cascade_result.scores,
            )

        council: CouncilResult | None = None
        if self._sangha is not None and len(cascade_result.indices) > 0:
            council = self._sangha.hold_council_sync({
                "query": f"cascade_select: k={len(cascade_result.indices)}",
                "issue_id": "scalable-pipeline",
                "fep_state": 0.5,
            })

        return ScalableResult(
            indices=cascade_result.indices,
            scores=cascade_result.scores,
            cascade_stats=cascade_result.stats,
            processing=processing,
            council=council,
        )


def _file_chunk_iter(
    paths: list[Path], chunk_size: int,
) -> Generator[np.ndarray, None, None]:
    """Read .npy files in chunks via memory mapping."""
    for path in paths:
        data = np.load(path, mmap_mode="r")
        for start in range(0, len(data), chunk_size):
            yield np.array(data[start:start + chunk_size])
