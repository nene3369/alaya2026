"""Pipeline — SmartSelector + SmartProcessor orchestration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from lmm.processor import ProcessingReport, SmartProcessor
from lmm.selector import SelectionResult, SmartSelector

SurpriseMethod = Literal["kl", "entropy", "bayesian"]


@dataclass
class PipelineResult:
    selection: SelectionResult
    processing: ProcessingReport

    @property
    def summary(self) -> dict:
        return {
            "selected": len(self.selection.indices),
            "method": self.selection.method_used,
            "confidence": round(self.selection.confidence, 3),
            "critical": self.processing.n_critical,
            "normal": self.processing.n_normal,
            "background": self.processing.n_background,
            "cache_hits": self.processing.cache_hits,
        }


class Pipeline:
    """Select-then-process pipeline.

    Usage::

        pipe = Pipeline(k=10)
        pipe.fit(reference_data)
        result = pipe.run(candidates, budget=0.8)
    """

    def __init__(
        self,
        k: int = 10,
        alpha: float = 1.0,
        gamma: float = 10.0,
        surprise_method: SurpriseMethod = "entropy",
        process_fn: Callable[[int, float], object] | None = None,
        critical_threshold: float = 0.8,
    ):
        self._selector = SmartSelector(
            k=k, alpha=alpha, gamma=gamma, surprise_method=surprise_method,
        )
        self._processor = SmartProcessor(
            process_fn=process_fn, critical_threshold=critical_threshold,
        )

    def fit(self, reference_data: np.ndarray) -> Pipeline:
        self._selector.fit(reference_data)
        return self

    def run(self, candidates: np.ndarray, budget: float = 1.0) -> PipelineResult:
        selection = self._selector.select(candidates, budget=budget)
        processing = self._processor.process(selection.indices, selection.scores)
        return PipelineResult(selection=selection, processing=processing)

    def run_from_surprises(
        self, surprises: np.ndarray, budget: float = 1.0,
    ) -> PipelineResult:
        selection = self._selector.select_from_surprises(surprises, budget=budget)
        processing = self._processor.process(selection.indices, selection.scores)
        return PipelineResult(selection=selection, processing=processing)

    def run_loop(
        self, candidates_stream: list[np.ndarray], budget: float = 0.5,
    ) -> list[PipelineResult]:
        return [self.run(c, budget=budget) for c in candidates_stream]
