"""Probabilistic data structures — fixed-memory summaries for trillions.

CountMinSketch: frequency estimation (Carter-Wegman hashing)
StreamingHistogram: online distribution estimation (Ben-Haim & Tom-Tov)
"""

from __future__ import annotations

import bisect as _bisect_mod
import struct
from dataclasses import dataclass

import numpy as np

from lmm._compat import HAS_VIEW, HAS_INT64_ARITH, HAS_SEARCHSORTED, IS_SHIM


_CMS_PRIME = (1 << 61) - 1  # Mersenne prime 2^61-1


def _make_hash_params(depth: int, seed: int = 42) -> list[tuple[int, int]]:
    """Generate (a, b) pairs for Carter-Wegman hashing."""
    rng = seed
    params = []
    for _ in range(depth):
        rng = (rng * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        a = (rng % (_CMS_PRIME - 1)) + 1
        rng = (rng * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        b = rng % _CMS_PRIME
        params.append((a, b))
    return params


def _float_to_int(v: float) -> int:
    """Map float to reproducible int (IEEE 754 bit pattern)."""
    return struct.unpack("q", struct.pack("d", v))[0]


class CountMinSketch:
    """Count-Min Sketch: approximate frequency counting.

    Memory: width * depth * 8 bytes. No underestimation.
    """

    def __init__(self, width: int = 10_000, depth: int = 8):
        self.width = width
        self.depth = depth
        self._table = [np.zeros(width, dtype=np.int64) for _ in range(depth)]
        self._total = 0
        self._params = _make_hash_params(depth)

    def _hash(self, key: int | float, depth: int) -> int:
        x = _float_to_int(float(key))
        a, b = self._params[depth]
        return ((a * x + b) % _CMS_PRIME) % self.width

    def add(self, key: int | float, count: int = 1) -> None:
        for d in range(self.depth):
            idx = self._hash(key, d)
            self._table[d][idx] += count
        self._total += count

    def add_batch(self, keys: np.ndarray, counts: np.ndarray | None = None) -> None:
        """Vectorized batch add."""
        if counts is None:
            counts = np.ones(len(keys), dtype=np.int64)

        n = len(keys)
        p = _CMS_PRIME
        w = self.width

        if HAS_VIEW and HAS_INT64_ARITH:
            keys_f64 = np.asarray(keys, dtype=np.float64).ravel()
            x_obj = keys_f64.view(np.int64).astype(object)
            for d in range(self.depth):
                a, b = self._params[d]
                indices = (((a * x_obj + b) % p) % w).astype(np.int64)
                np.add.at(self._table[d], indices, counts)
        else:
            x_ints = [_float_to_int(float(keys[i])) for i in range(n)]
            for d in range(self.depth):
                a, b = self._params[d]
                indices = np.empty(n, dtype=np.int64)
                for i in range(n):
                    indices[i] = ((a * x_ints[i] + b) % p) % w
                np.add.at(self._table[d], indices, counts)

        self._total += int(counts.sum())

    def query(self, key: int | float) -> int:
        estimates = np.empty(self.depth, dtype=np.int64)
        for d in range(self.depth):
            idx = self._hash(key, d)
            estimates[d] = self._table[d][idx]
        return int(estimates.min())

    def query_batch(self, keys: np.ndarray) -> np.ndarray:
        """Vectorized batch query."""
        p = _CMS_PRIME
        w = self.width

        if HAS_VIEW and HAS_INT64_ARITH:
            keys_f64 = np.asarray(keys, dtype=np.float64).ravel()
            x_obj = keys_f64.view(np.int64).astype(object)
            a0, b0 = self._params[0]
            min_est = self._table[0][(((a0 * x_obj + b0) % p) % w).astype(np.int64)]
            for d in range(1, self.depth):
                a, b = self._params[d]
                est = self._table[d][(((a * x_obj + b) % p) % w).astype(np.int64)]
                min_est = np.minimum(min_est, est)
        else:
            n = len(keys)
            x_ints = [_float_to_int(float(keys[i])) for i in range(n)]
            a0, b0 = self._params[0]
            indices0 = np.empty(n, dtype=np.int64)
            for i in range(n):
                indices0[i] = ((a0 * x_ints[i] + b0) % p) % w
            min_est = self._table[0][indices0]
            for d in range(1, self.depth):
                a, b = self._params[d]
                indices = np.empty(n, dtype=np.int64)
                for i in range(n):
                    indices[i] = ((a * x_ints[i] + b) % p) % w
                est = self._table[d][indices]
                min_est = np.minimum(min_est, est)
        return min_est

    def frequency(self, key: int | float) -> float:
        if self._total == 0:
            return 0.0
        return self.query(key) / self._total

    @property
    def total(self) -> int:
        return self._total

    def merge(self, other: CountMinSketch) -> CountMinSketch:
        if self.width != other.width or self.depth != other.depth:
            raise ValueError("Sketch dimensions must match")
        merged = CountMinSketch(self.width, self.depth)
        merged._table = [self._table[d] + other._table[d] for d in range(self.depth)]
        merged._total = self._total + other._total
        merged._params = self._params[:]
        return merged


# ---------------------------------------------------------------------------
# StreamingHistogram
# ---------------------------------------------------------------------------

@dataclass
class _Bin:
    center: float
    count: int


class StreamingHistogram:
    """Streaming Histogram (Ben-Haim & Tom-Tov).

    Fixed-memory online distribution estimation.
    Memory: max_bins * 16 bytes.
    """

    def __init__(self, max_bins: int = 1024):
        self.max_bins = max_bins
        self._bins: list[_Bin] = []
        self._total: int = 0
        self._min: float = float("inf")
        self._max: float = float("-inf")
        self._frozen: bool = False
        self._min_gap_idx: int = -1
        self._centers_cache: tuple | None = None

    def add(self, value: float) -> None:
        self._total += 1
        self._min = min(self._min, value)
        self._max = max(self._max, value)

        pos = self._bisect(value)
        self._bins.insert(pos, _Bin(center=value, count=1))
        self._min_gap_idx = -1
        self._centers_cache = None

        if len(self._bins) > self.max_bins:
            self._compress()

    def add_batch(self, values: np.ndarray) -> None:
        """Vectorized batch add with histogram pre-binning."""
        n_new = len(values)
        if n_new == 0:
            return

        if IS_SHIM:
            if values.ndim == 1:
                arr = values
            else:
                flat = []
                for i in range(len(values)):
                    row = values[i]
                    if hasattr(row, "__len__"):
                        for j in range(len(row)):
                            flat.append(float(row[j]))
                    else:
                        flat.append(float(row))
                arr = np.array(flat)
                n_new = len(flat)
        else:
            arr = np.asarray(values, dtype=np.float64).ravel()
            n_new = len(arr)

        self._total += n_new
        arr_min = float(arr.min())
        arr_max = float(arr.max())
        self._min = min(self._min, arr_min)
        self._max = max(self._max, arr_max)

        if n_new > self.max_bins * 2:
            n_hist_bins = min(self.max_bins, n_new)
            try:
                hist_counts, edges = np.histogram(
                    arr, bins=n_hist_bins, range=(arr_min, arr_max),
                )
            except TypeError:
                hist_counts, edges = np.histogram(arr, bins=n_hist_bins)
            centers = (edges[:-1] + edges[1:]) * 0.5
            new_bins = [
                _Bin(center=float(centers[i]), count=int(hist_counts[i]))
                for i in range(len(centers)) if hist_counts[i] > 0
            ]
        else:
            sorted_vals = sorted(float(arr[i]) for i in range(n_new))
            new_bins = [_Bin(center=v, count=1) for v in sorted_vals]

        n_old, n_nb = len(self._bins), len(new_bins)
        merged = [None] * (n_old + n_nb)
        i, j, ptr = 0, 0, 0
        while i < n_old and j < n_nb:
            if self._bins[i].center <= new_bins[j].center:
                merged[ptr] = self._bins[i]
                i += 1
            else:
                merged[ptr] = new_bins[j]
                j += 1
            ptr += 1
        while i < n_old:
            merged[ptr] = self._bins[i]
            i += 1
            ptr += 1
        while j < n_nb:
            merged[ptr] = new_bins[j]
            j += 1
            ptr += 1

        self._bins = merged[:ptr]
        self._min_gap_idx = -1
        self._centers_cache = None

        n_over = len(self._bins) - self.max_bins
        if n_over > 0:
            self._compress_loop(n_over)

    def defer_compress(self, defer: bool = True) -> None:
        pass

    def ensure_compressed(self) -> None:
        pass

    def density(self, value: float) -> float:
        if self._total == 0 or len(self._bins) < 2:
            return 0.0
        pos = self._bisect(value)
        left = max(0, pos - 1)
        right = min(len(self._bins) - 1, pos)
        if left == right:
            return self._bins[left].count / self._total
        b_left, b_right = self._bins[left], self._bins[right]
        span = b_right.center - b_left.center
        if span < 1e-15:
            return (b_left.count + b_right.count) / (2 * self._total)
        t = max(0.0, min(1.0, (value - b_left.center) / span))
        interpolated = b_left.count * (1 - t) + b_right.count * t
        return interpolated / self._total

    def probability(self, value: float) -> float:
        d = self.density(value)
        if self._total < 2:
            return 1e-10
        bin_width = (self._max - self._min) / max(len(self._bins), 1)
        return max(d * bin_width, 1e-10)

    def density_batch(self, values: np.ndarray) -> np.ndarray:
        """Vectorized density estimation."""
        vals = np.array([float(values[i]) for i in range(len(values))])
        n_vals = len(vals)
        if self._total == 0 or len(self._bins) < 2:
            return np.zeros(n_vals)

        centers_list = [b.center for b in self._bins]
        centers = np.array(centers_list)
        counts = np.array([float(b.count) for b in self._bins])
        n_bins = len(centers)

        if HAS_SEARCHSORTED:
            pos = np.searchsorted(centers, vals, side="left")
        else:
            pos = np.empty(n_vals, dtype=int)
            for i in range(n_vals):
                pos[i] = _bisect_mod.bisect_left(centers_list, float(vals[i]))

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
        result = np.where(same, c_left, np.where(small_span, (c_left + c_right) * 0.5, interpolated))
        return result / self._total

    def probability_batch(self, values: np.ndarray) -> np.ndarray:
        d = self.density_batch(values)
        if self._total < 2:
            return np.full(len(values), 1e-10)
        bin_width = (self._max - self._min) / max(len(self._bins), 1)
        return np.maximum(d * bin_width, 1e-10)

    def quantile(self, q: float) -> float:
        if not self._bins:
            return 0.0
        target = q * self._total
        cumulative = 0
        for b in self._bins:
            cumulative += b.count
            if cumulative >= target:
                return b.center
        return self._bins[-1].center

    def merge(self, other: StreamingHistogram) -> StreamingHistogram:
        merged = StreamingHistogram(max_bins=self.max_bins)
        merged._bins = sorted(self._bins + other._bins, key=lambda b: b.center)
        merged._total = self._total + other._total
        merged._min = min(self._min, other._min)
        merged._max = max(self._max, other._max)
        while len(merged._bins) > self.max_bins:
            merged._compress()
        return merged

    @property
    def total(self) -> int:
        return self._total

    def _bisect(self, value: float) -> int:
        cache = getattr(self, "_centers_cache", None)
        if cache is None or cache[0] != len(self._bins):
            centers = [b.center for b in self._bins]
            self._centers_cache = (len(self._bins), centers)
        else:
            centers = cache[1]
        return _bisect_mod.bisect_left(centers, value)

    def _compress(self) -> None:
        n_bins = len(self._bins)
        if n_bins < 2:
            return
        if self._min_gap_idx < 0 or self._min_gap_idx >= n_bins - 1:
            if not IS_SHIM and n_bins > 32:
                centers = np.array([b.center for b in self._bins])
                gaps = np.diff(centers)
                self._min_gap_idx = int(np.argmin(gaps))
            else:
                min_gap, min_idx = float("inf"), 0
                for i in range(n_bins - 1):
                    gap = self._bins[i + 1].center - self._bins[i].center
                    if gap < min_gap:
                        min_gap, min_idx = gap, i
                self._min_gap_idx = min_idx
        idx = self._min_gap_idx
        b1, b2 = self._bins[idx], self._bins[idx + 1]
        total = b1.count + b2.count
        self._bins[idx] = _Bin(center=(b1.center * b1.count + b2.center * b2.count) / total, count=total)
        del self._bins[idx + 1]
        self._min_gap_idx = -1
        self._centers_cache = None

    def _compress_loop(self, n_remove: int) -> None:
        bins = self._bins
        if len(bins) < 2 or n_remove <= 0:
            return
        if n_remove <= 2:
            for _ in range(n_remove):
                self._compress()
            return
        gaps = [bins[i + 1].center - bins[i].center for i in range(len(bins) - 1)]
        for _ in range(n_remove):
            if not gaps:
                break
            min_idx = min(range(len(gaps)), key=lambda i: gaps[i])
            b1, b2 = bins[min_idx], bins[min_idx + 1]
            total_count = b1.count + b2.count
            bins[min_idx] = _Bin(center=(b1.center * b1.count + b2.center * b2.count) / total_count, count=total_count)
            del bins[min_idx + 1]
            del gaps[min_idx]
            if min_idx < len(gaps):
                gaps[min_idx] = bins[min_idx + 1].center - bins[min_idx].center
            if min_idx > 0:
                gaps[min_idx - 1] = bins[min_idx].center - bins[min_idx - 1].center
        self._bins = bins
        self._min_gap_idx = -1
        self._centers_cache = None
