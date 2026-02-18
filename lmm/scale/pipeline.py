"""ScalablePipeline — trillion-token orchestration."""

from __future__ import annotations

from collections.abc import Callable, Generator, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lmm.processor import ProcessingReport, SmartProcessor
from lmm.scale.cascade import CascadeResult, CascadeStats, MultiLevelCascade


@dataclass
class ScalableResult:
    """Scalable pipeline result."""

    indices: np.ndarray
    scores: np.ndarray
    cascade_stats: CascadeStats
    processing: ProcessingReport

    @property
    def summary(self) -> dict:
        return {
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


class ScalablePipeline:
    """Trillion-token pipeline with automatic mode selection."""

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
        return ScalableResult(
            indices=cascade_result.indices,
            scores=cascade_result.scores,
            cascade_stats=cascade_result.stats,
            processing=processing,
        )


def _file_chunk_iter(
    paths: list[Path], chunk_size: int,
) -> Generator[np.ndarray, None, None]:
    """Read .npy files in chunks via memory mapping."""
    for path in paths:
        data = np.load(path, mmap_mode="r")
        for start in range(0, len(data), chunk_size):
            yield np.array(data[start:start + chunk_size])
