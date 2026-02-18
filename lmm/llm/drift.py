"""DriftDetector — LLM output distribution drift detection."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from lmm.llm.embeddings import ngram_vectors as _ngram_vectors
from lmm.surprise import SurpriseCalculator


@dataclass
class DriftReport:
    """Drift detection report."""

    is_drifting: bool
    drift_score: float
    window_mean: float
    baseline_mean: float
    baseline_std: float
    n_observations: int
    window_size: int


class DriftDetector:
    """LLM output distribution drift detector.

    z-score based: |window_mean - baseline_mean| / baseline_std > threshold
    """

    def __init__(
        self,
        window_size: int = 100,
        threshold: float = 2.0,
        surprise_method: str = "entropy",
    ):
        self.window_size = window_size
        self.threshold = threshold
        self._calculator = SurpriseCalculator(method=surprise_method)
        self._window: deque[float] = deque(maxlen=window_size)
        self._baseline_mean: float = 0.0
        self._baseline_std: float = 1.0
        self._n_observations: int = 0
        self._fitted = False

    def fit_baseline(self, texts: list[str] | np.ndarray) -> DriftDetector:
        """Learn baseline distribution from normal outputs."""
        if isinstance(texts, list):
            vecs = _ngram_vectors(texts)
        else:
            vecs = np.asarray(texts)
        self._calculator.fit(vecs)
        surprises = self._calculator.compute(vecs)
        self._baseline_mean = float(surprises.mean())
        self._baseline_std = max(float(surprises.std()), 1e-10)
        self._fitted = True
        return self

    def update(self, text: str | np.ndarray) -> DriftReport:
        """Add observation and check for drift."""
        if not self._fitted:
            raise RuntimeError("Call fit_baseline() first")
        if isinstance(text, str):
            vec = _ngram_vectors([text])
        else:
            vec = np.asarray(text).reshape(1, -1) if np.asarray(text).ndim == 1 else np.asarray(text)
        self._window.append(float(self._calculator.compute(vec)[0]))
        self._n_observations += 1
        return self._make_report()

    def update_batch(self, texts: list[str] | np.ndarray) -> DriftReport:
        if not self._fitted:
            raise RuntimeError("Call fit_baseline() first")
        if isinstance(texts, list):
            vecs = _ngram_vectors(texts)
        else:
            vecs = np.asarray(texts)
        for s in self._calculator.compute(vecs):
            self._window.append(float(s))
            self._n_observations += 1
        return self._make_report()

    def check(self) -> DriftReport:
        if not self._fitted:
            raise RuntimeError("Call fit_baseline() first")
        return self._make_report()

    def reset_window(self) -> None:
        self._window.clear()
        self._n_observations = 0

    def _make_report(self) -> DriftReport:
        if not self._window:
            return DriftReport(
                is_drifting=False, drift_score=0.0, window_mean=0.0,
                baseline_mean=self._baseline_mean,
                baseline_std=self._baseline_std,
                n_observations=self._n_observations, window_size=0,
            )
        window_mean = sum(self._window) / len(self._window)
        drift_score = abs(window_mean - self._baseline_mean) / self._baseline_std
        return DriftReport(
            is_drifting=drift_score > self.threshold,
            drift_score=drift_score, window_mean=window_mean,
            baseline_mean=self._baseline_mean,
            baseline_std=self._baseline_std,
            n_observations=self._n_observations,
            window_size=len(self._window),
        )
