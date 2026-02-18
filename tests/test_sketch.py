"""Tests for lmm.scale.sketch — CountMinSketch + StreamingHistogram."""

from __future__ import annotations

import numpy as np

from lmm.scale.sketch import CountMinSketch, StreamingHistogram


class TestCountMinSketch:
    def test_add_and_query(self):
        cms = CountMinSketch(width=100, depth=5)
        cms.add(42)
        cms.add(42)
        cms.add(42)
        cms.add(7)
        assert cms.query(42) >= 3
        assert cms.query(7) >= 1

    def test_add_batch(self):
        cms = CountMinSketch(width=200, depth=5)
        items = np.array([1, 2, 3, 1, 1, 2])
        cms.add_batch(items)
        assert cms.query(1) >= 3
        assert cms.query(2) >= 2
        assert cms.query(3) >= 1

    def test_merge(self):
        cms1 = CountMinSketch(width=100, depth=5)
        cms2 = CountMinSketch(width=100, depth=5)
        cms1.add(1)
        cms2.add(1)
        merged = cms1.merge(cms2)
        assert merged.query(1) >= 2

    def test_never_underestimates(self):
        rng = np.random.RandomState(42)
        cms = CountMinSketch(width=500, depth=7)
        counts = {}
        for _ in range(1000):
            item = rng.randint(0, 50)
            cms.add(item)
            counts[item] = counts.get(item, 0) + 1
        for item, true_count in counts.items():
            assert cms.query(item) >= true_count


class TestStreamingHistogram:
    def test_add_and_quantile(self):
        hist = StreamingHistogram(max_bins=50)
        rng = np.random.RandomState(42)
        for x in rng.randn(1000):
            hist.add(float(x))
        # Median should be near 0
        median = hist.quantile(0.5)
        assert abs(median) < 0.5

    def test_add_batch(self):
        hist = StreamingHistogram(max_bins=50)
        data = np.random.RandomState(42).randn(500)
        hist.add_batch(data)
        assert hist.quantile(0.0) <= hist.quantile(0.5) <= hist.quantile(1.0)

    def test_density(self):
        hist = StreamingHistogram(max_bins=50)
        data = np.random.RandomState(42).randn(500)
        hist.add_batch(data)
        d = hist.density(0.0)
        assert d >= 0

    def test_probability(self):
        hist = StreamingHistogram(max_bins=50)
        data = np.random.RandomState(42).randn(500)
        hist.add_batch(data)
        p = hist.probability(0.0)
        assert p > 0
