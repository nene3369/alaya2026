"""Tests for lmm.scale.cascade — MultiLevelCascade."""

from __future__ import annotations

import numpy as np

from lmm.scale.cascade import CascadeResult, MultiLevelCascade


class TestMultiLevelCascade:
    def test_fit_and_select_array(self):
        rng = np.random.RandomState(42)
        data = rng.randn(200)
        cascade = MultiLevelCascade(k=5)
        cascade.fit_array(data)
        result = cascade.select_array(data)
        assert isinstance(result, CascadeResult)
        assert len(result.indices) == 5
        assert hasattr(result, "stats")

    def test_fit_and_select_stream(self):
        rng = np.random.RandomState(42)

        def chunk_iter():
            for _ in range(3):
                yield rng.randn(100)

        cascade = MultiLevelCascade(k=5)
        cascade.fit_stream(chunk_iter())
        result = cascade.select_stream(chunk_iter())
        assert isinstance(result, CascadeResult)
        assert len(result.indices) == 5

    def test_cascade_stats(self):
        rng = np.random.RandomState(42)
        data = rng.randn(200)
        cascade = MultiLevelCascade(k=5)
        cascade.fit_array(data)
        result = cascade.select_array(data)
        stats = result.stats
        assert hasattr(stats, "level_sizes")
        assert hasattr(stats, "level_names")
        assert len(stats.level_sizes) > 0

    def test_small_data(self):
        data = np.array([1.0, 2.0, 3.0])
        cascade = MultiLevelCascade(k=10)
        cascade.fit_array(data)
        result = cascade.select_array(data)
        assert len(result.indices) <= 3
