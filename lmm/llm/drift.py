"""DriftDetector — LLM output distribution drift detection.

Also provides :class:`SemanticDriftDetector` for embedding-space concept
drift (Anicca / 諸行無常) and :class:`MemoryDecayAdvisor` for bridging
drift detection to memory strength decay.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from lmm.llm.embeddings import ngram_vectors as _ngram_vectors
from lmm.surprise import SurpriseCalculator


def _as_array(obj) -> np.ndarray:
    """Shim-safe np.asarray — handles vendored ndarray identity."""
    if hasattr(obj, "ndim") and hasattr(obj, "shape") and hasattr(obj, "mean"):
        return obj  # already array-like (works across module boundaries)
    return np.asarray(obj)


def _variance(arr: np.ndarray, axis: int = 0) -> np.ndarray:
    """Shim-compatible variance (mean of squared deviations)."""
    m = arr.mean(axis=axis)
    if axis == 0:
        # Broadcasting: (n, d) - (d,) works row-wise
        diff = arr - m
    else:
        # axis=1: need to reshape m from (n,) to (n, 1)
        diff = arr - m.reshape(-1, 1)
    return (diff * diff).mean(axis=axis)


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


# ===================================================================
# Semantic Drift Detection (Anicca / 諸行無常)
# ===================================================================


@dataclass
class SemanticDriftReport:
    """Report from semantic embedding drift detection.

    Attributes
    ----------
    is_drifting : bool
        True when ``drift_score`` exceeds the detector's threshold.
    drift_score : float
        Combined score (0 = stable, higher = more drift).
    centroid_shift : float
        L2 distance between baseline and window centroids.
    variance_ratio : float
        Ratio of current window variance to baseline variance.
    cosine_drift : float
        1 - cosine_similarity(baseline_centroid, window_centroid).
    n_observations : int
        Total embeddings observed since fitting.
    window_size : int
        Number of embeddings currently in the window.
    stale_pattern_indices : list[int]
        Indices of patterns that have drifted most from the current
        distribution (populated by :meth:`identify_stale_patterns`).
    """

    is_drifting: bool
    drift_score: float
    centroid_shift: float
    variance_ratio: float
    cosine_drift: float
    n_observations: int
    window_size: int
    stale_pattern_indices: list[int] = field(default_factory=list)


class SemanticDriftDetector:
    """Detect semantic drift in vector memory distributions (Anicca / 諸行無常).

    Compares embedding distributions between a baseline epoch and a sliding
    recent window.  Reports when the meaning space has shifted enough to
    warrant memory decay.

    The combined ``drift_score`` is a weighted sum of:

    * **centroid shift** (L2 distance, normalized by baseline norm)
    * **variance ratio deviation** (|1 - window_var/baseline_var|)
    * **cosine drift** (1 - cosine similarity between centroids)

    Parameters
    ----------
    window_size : int
        Number of recent embeddings to retain.
    threshold : float
        Combined drift score above which ``is_drifting=True``.
    centroid_weight : float
        Weight for centroid shift in the combined score.
    variance_weight : float
        Weight for variance ratio deviation.
    cosine_weight : float
        Weight for cosine drift.
    """

    def __init__(
        self,
        window_size: int = 200,
        threshold: float = 0.3,
        centroid_weight: float = 0.4,
        variance_weight: float = 0.2,
        cosine_weight: float = 0.4,
    ):
        self.window_size = window_size
        self.threshold = threshold
        self._w_centroid = centroid_weight
        self._w_variance = variance_weight
        self._w_cosine = cosine_weight

        self._baseline_centroid: np.ndarray | None = None
        self._baseline_var: np.ndarray | None = None
        self._baseline_norm: float = 1.0
        self._baseline_mean_var: float = 1.0

        self._window: deque[np.ndarray] = deque(maxlen=window_size)
        self._n_observations: int = 0
        self._fitted = False

    def fit_baseline(
        self, embeddings: np.ndarray,
    ) -> SemanticDriftDetector:
        """Set baseline distribution from reference embeddings.

        Computes and stores: centroid, per-dimension variance, mean norm.

        Parameters
        ----------
        embeddings : (m, d) array of baseline embedding vectors.
        """
        emb = _as_array(embeddings)
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)

        self._baseline_centroid = emb.mean(axis=0)
        self._baseline_var = _variance(emb, axis=0)
        self._baseline_norm = max(
            float(np.linalg.norm(self._baseline_centroid)), 1e-10,
        )
        self._baseline_mean_var = max(float(self._baseline_var.mean()), 1e-10)
        self._fitted = True
        return self

    def update(self, embedding: np.ndarray) -> SemanticDriftReport:
        """Add a single embedding observation and check for drift."""
        if not self._fitted:
            raise RuntimeError("Call fit_baseline() first")
        emb = _as_array(embedding).flatten()
        self._window.append(emb)
        self._n_observations += 1
        return self._make_report()

    def update_batch(self, embeddings: np.ndarray) -> SemanticDriftReport:
        """Add a batch of embeddings and check for drift."""
        if not self._fitted:
            raise RuntimeError("Call fit_baseline() first")
        emb = _as_array(embeddings)
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        for row in emb:
            self._window.append(row)
            self._n_observations += 1
        return self._make_report()

    def check(self) -> SemanticDriftReport:
        """Check current drift state without adding observations."""
        if not self._fitted:
            raise RuntimeError("Call fit_baseline() first")
        return self._make_report()

    def identify_stale_patterns(
        self,
        patterns: np.ndarray,
        top_k: int = 10,
    ) -> list[int]:
        """Identify which stored patterns have drifted most from the current
        distribution.

        Compares each pattern against the current window centroid and returns
        indices sorted by staleness (most stale first).

        Parameters
        ----------
        patterns : (m, d) array of stored pattern vectors.
        top_k : number of most-stale indices to return.

        Returns
        -------
        list of pattern indices sorted by staleness descending.
        """
        if not self._window:
            return []

        pats = _as_array(patterns)
        if pats.ndim == 1:
            pats = pats.reshape(1, -1)

        window_arr = np.array(list(self._window))
        window_centroid = window_arr.mean(axis=0)

        # Cosine distance from each pattern to window centroid
        wc_norm = max(float(np.linalg.norm(window_centroid)), 1e-10)
        pat_norms = np.linalg.norm(pats, axis=1)
        pat_norms = np.maximum(pat_norms, 1e-10)
        cosine_sims = pats @ window_centroid / (pat_norms * wc_norm)
        staleness = 1.0 - cosine_sims

        top_k = min(top_k, len(staleness))
        indices = np.argsort(-staleness)[:top_k]
        return [int(i) for i in indices]

    def reset_baseline(self) -> None:
        """Promote the current window to the new baseline.

        Resets the window after promoting, so the detector starts fresh.
        """
        if self._window:
            window_arr = np.array(list(self._window))
            self.fit_baseline(window_arr)
        self._window.clear()
        self._n_observations = 0

    def _make_report(self) -> SemanticDriftReport:
        if not self._window:
            return SemanticDriftReport(
                is_drifting=False,
                drift_score=0.0,
                centroid_shift=0.0,
                variance_ratio=1.0,
                cosine_drift=0.0,
                n_observations=self._n_observations,
                window_size=0,
            )

        window_arr = np.array(list(self._window))
        window_centroid = window_arr.mean(axis=0)
        window_var = _variance(window_arr, axis=0)

        # 1. Centroid shift (L2, normalized by baseline norm)
        diff = window_centroid - self._baseline_centroid
        centroid_shift = float(np.linalg.norm(diff)) / self._baseline_norm

        # 2. Variance ratio
        window_mean_var = max(float(window_var.mean()), 1e-10)
        variance_ratio = window_mean_var / self._baseline_mean_var

        # 3. Cosine drift
        wc_norm = max(float(np.linalg.norm(window_centroid)), 1e-10)
        cos_sim = float(
            np.dot(self._baseline_centroid, window_centroid)
            / (self._baseline_norm * wc_norm)
        )
        cos_sim = max(-1.0, min(1.0, cos_sim))
        cosine_drift = 1.0 - cos_sim

        # Combined score
        drift_score = (
            self._w_centroid * min(centroid_shift, 2.0) / 2.0
            + self._w_variance * min(abs(1.0 - variance_ratio), 2.0) / 2.0
            + self._w_cosine * min(cosine_drift, 2.0) / 2.0
        )

        return SemanticDriftReport(
            is_drifting=drift_score > self.threshold,
            drift_score=drift_score,
            centroid_shift=centroid_shift,
            variance_ratio=variance_ratio,
            cosine_drift=cosine_drift,
            n_observations=self._n_observations,
            window_size=len(self._window),
        )


# ===================================================================
# Memory Decay Advisor
# ===================================================================


class MemoryDecayAdvisor:
    """Advise on memory decay rates based on semantic drift (Anicca).

    Bridges :class:`SemanticDriftDetector` to a memory backend.  When drift
    is detected, the advisor computes elevated decay rates and per-pattern
    weight multipliers to reduce the strength of stale memories.

    This class is advisory only — it does not directly mutate any
    :class:`~lmm.memory.backend.MemoryBackend`.  The caller uses the
    returned rates/weights to perform the actual decay.

    Parameters
    ----------
    detector : SemanticDriftDetector
        Source of drift reports.
    base_decay_rate : float
        Default decay rate when no drift is detected.
    drift_decay_multiplier : float
        Factor to multiply the decay rate by when drift is detected.
    """

    def __init__(
        self,
        detector: SemanticDriftDetector,
        base_decay_rate: float = 0.001,
        drift_decay_multiplier: float = 5.0,
    ):
        self.detector = detector
        self.base_decay_rate = base_decay_rate
        self.drift_decay_multiplier = drift_decay_multiplier

    def compute_decay_rate(
        self, report: SemanticDriftReport | None = None,
    ) -> float:
        """Return the effective decay rate given current drift state.

        When drifting: ``base_decay_rate * drift_decay_multiplier * drift_score``
        When stable: ``base_decay_rate``

        Parameters
        ----------
        report : optional pre-computed drift report; if None, calls
            ``detector.check()`` internally.

        Returns
        -------
        float : effective decay rate.
        """
        if report is None:
            report = self.detector.check()
        if report.is_drifting:
            return self.base_decay_rate * self.drift_decay_multiplier * max(
                report.drift_score, 1.0,
            )
        return self.base_decay_rate

    def compute_pattern_weights(
        self,
        patterns: np.ndarray,
        report: SemanticDriftReport | None = None,
    ) -> np.ndarray:
        """Return per-pattern weight multipliers in [0, 1].

        Patterns close to the current window centroid get weights near 1.0.
        Stale patterns that have drifted from the current distribution get
        weights approaching 0.0.

        Parameters
        ----------
        patterns : (m, d) array of stored pattern vectors.
        report : optional pre-computed drift report.

        Returns
        -------
        (m,) array of weight multipliers in [0, 1].
        """
        if report is None:
            report = self.detector.check()

        pats = _as_array(patterns)
        if pats.ndim == 1:
            pats = pats.reshape(1, -1)

        if not self.detector._window:
            return np.ones(pats.shape[0])

        window_arr = np.array(list(self.detector._window))
        window_centroid = window_arr.mean(axis=0)

        wc_norm = max(float(np.linalg.norm(window_centroid)), 1e-10)
        pat_norms = np.linalg.norm(pats, axis=1)
        pat_norms = np.maximum(pat_norms, 1e-10)
        cosine_sims = pats @ window_centroid / (pat_norms * wc_norm)
        # Map similarity [-1, 1] to weight [0, 1]
        weights = np.clip((cosine_sims + 1.0) / 2.0, 0.0, 1.0)

        return weights
