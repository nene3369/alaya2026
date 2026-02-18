"""StreamingSurprise — streaming surprise computation for trillions.

2-pass approach:
  Pass 1 (fit_stream): learn distribution from data stream (fixed memory)
  Pass 2 (compute_stream): compute surprise per chunk + accumulate top-k
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Generator, Iterable
from dataclasses import dataclass

import numpy as np

from lmm._compat import HAS_ARGPARTITION, HAS_SEARCHSORTED
from lmm.scale.sketch import CountMinSketch, StreamingHistogram


@dataclass
class ChunkResult:
    """Per-chunk processing result."""

    surprises: np.ndarray
    indices_offset: int
    n_processed: int
    top_k_local: np.ndarray
    top_k_scores: np.ndarray


class StreamingSurprise:
    """Streaming surprise computation with global top-K."""

    def __init__(
        self,
        k: int = 10,
        chunk_size: int = 100_000,
        sketch_width: int = 10_000,
        sketch_depth: int = 8,
        hist_bins: int = 1024,
        use_sketch: bool = False,
    ):
        self.k = k
        self.chunk_size = chunk_size
        self._histogram = StreamingHistogram(max_bins=hist_bins)
        self._use_sketch = use_sketch
        self._sketch: CountMinSketch | None = None
        if use_sketch:
            self._sketch = CountMinSketch(width=sketch_width, depth=sketch_depth)
        self._fitted = False
        self._top_buffer = _TopKBuffer(k)

    # --- Pass 1: distribution learning ---

    def fit_stream(self, data_iter: Iterable[np.ndarray]) -> StreamingSurprise:
        """Learn distribution from data stream."""
        if self._sketch is not None:
            for chunk in data_iter:
                values = self._to_scalar(chunk)
                self._histogram.add_batch(values)
                self._sketch.add_batch(values)
        else:
            for chunk in data_iter:
                values = self._to_scalar(chunk)
                self._histogram.add_batch(values)
        self._fitted = True
        return self

    def fit_array(self, data: np.ndarray) -> StreamingSurprise:
        return self.fit_stream(self._chunk_array(data))

    # --- Pass 2: surprise computation ---

    def compute_stream(
        self, data_iter: Iterable[np.ndarray],
    ) -> Generator[ChunkResult, None, None]:
        """Compute surprises per chunk, accumulating global top-K."""
        if not self._fitted:
            raise RuntimeError("Call fit_stream() first")

        self._top_buffer.clear()
        offset = 0

        for chunk in data_iter:
            values = self._to_scalar(chunk)
            surprises = self._compute_chunk(values)

            local_k = min(self.k, len(surprises))
            if HAS_ARGPARTITION and len(surprises) > local_k:
                top_local = np.argpartition(surprises, -local_k)[-local_k:]
            else:
                top_local = np.argsort(surprises)[-local_k:]
            top_scores = surprises[top_local]

            for local_idx, score in zip(top_local, top_scores):
                self._top_buffer.push(offset + int(local_idx), float(score))

            yield ChunkResult(
                surprises=surprises, indices_offset=offset,
                n_processed=len(values),
                top_k_local=top_local, top_k_scores=top_scores,
            )
            offset += len(values)

    def compute_array(self, data: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit_stream() first")
        return self._compute_chunk(self._to_scalar(data))

    def compute_and_select(
        self, data_iter: Iterable[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        for _ in self.compute_stream(data_iter):
            pass
        return self.get_top_k()

    def get_top_k(self) -> tuple[np.ndarray, np.ndarray]:
        return self._top_buffer.get()

    # --- Internal ---

    def _compute_chunk(self, values: np.ndarray) -> np.ndarray:
        n = len(values)
        hist = self._histogram
        bins = hist._bins
        n_bins = len(bins)
        total = hist._total

        if total < 2 or n_bins < 2:
            return np.full(n, -math.log(1e-10))

        bin_width = (hist._max - hist._min) / n_bins

        if HAS_SEARCHSORTED:
            return self._compute_chunk_vectorized(values, bins, n_bins, total, bin_width)

        surprises = np.empty(n)
        for i in range(n):
            v = float(values[i])
            lo, hi = 0, n_bins
            while lo < hi:
                mid = (lo + hi) >> 1
                if bins[mid].center < v:
                    lo = mid + 1
                else:
                    hi = mid
            left_idx = lo - 1 if lo > 0 else 0
            right_idx = lo if lo < n_bins else n_bins - 1
            if left_idx == right_idx:
                density = bins[left_idx].count / total
            else:
                b_l, b_r = bins[left_idx], bins[right_idx]
                span = b_r.center - b_l.center
                if span < 1e-15:
                    density = (b_l.count + b_r.count) / (2 * total)
                else:
                    t = max(0.0, min(1.0, (v - b_l.center) / span))
                    density = (b_l.count * (1.0 - t) + b_r.count * t) / total
            p = max(density * bin_width, 1e-10)
            surprises[i] = -math.log(p)
        return surprises

    def _compute_chunk_vectorized(self, values, bins, n_bins, total, bin_width):
        vals = np.array([float(values[i]) for i in range(len(values))])
        cache = getattr(self, "_bin_cache", None)
        if cache is None or cache[0] != total or cache[1] != n_bins:
            centers = np.array([b.center for b in bins])
            counts = np.array([float(b.count) for b in bins])
            self._bin_cache = (total, n_bins, centers, counts)
        else:
            centers, counts = cache[2], cache[3]

        pos = np.searchsorted(centers, vals, side="left")
        left = np.clip(pos - 1, 0, n_bins - 1)
        right = np.clip(pos, 0, n_bins - 1)
        c_left, c_right = counts[left], counts[right]
        center_left, center_right = centers[left], centers[right]
        same = left == right
        span = center_right - center_left
        small_span = span < 1e-15
        safe_span = np.where(small_span, 1.0, span)
        t = np.clip((vals - center_left) / safe_span, 0.0, 1.0)
        interpolated = c_left * (1.0 - t) + c_right * t
        density = np.where(same, c_left, np.where(small_span, (c_left + c_right) * 0.5, interpolated)) / total
        p = np.maximum(density * bin_width, 1e-10)
        return -np.log(p)

    def _to_scalar(self, data: np.ndarray) -> np.ndarray:
        if data.ndim > 1:
            return np.linalg.norm(data, axis=1)
        return data.astype(float)

    def _chunk_array(self, data: np.ndarray) -> Generator[np.ndarray, None, None]:
        for start in range(0, len(data), self.chunk_size):
            yield data[start:start + self.chunk_size]

    @property
    def total_seen(self) -> int:
        return self._histogram.total


class _TopKBuffer:
    """Fixed-size top-k buffer using min-heap. O(N log k)."""

    def __init__(self, k: int):
        self.k = k
        self._heap: list[tuple[float, int]] = []

    def push(self, index: int, score: float) -> None:
        if len(self._heap) < self.k:
            heapq.heappush(self._heap, (score, index))
        elif score > self._heap[0][0]:
            heapq.heapreplace(self._heap, (score, index))

    def get(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._heap:
            return np.array([], dtype=int), np.array([], dtype=float)
        sorted_items = sorted(self._heap, key=lambda x: -x[0])
        return (
            np.array([item[1] for item in sorted_items]),
            np.array([item[0] for item in sorted_items]),
        )

    def clear(self) -> None:
        self._heap.clear()
