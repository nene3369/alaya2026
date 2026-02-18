"""Tests for lmm.scale.pipeline — ScalablePipeline."""

from __future__ import annotations

import numpy as np

from lmm.scale.pipeline import ScalablePipeline, ScalableResult


class TestScalablePipeline:
    def test_fit_and_run_array(self):
        rng = np.random.RandomState(42)
        data = rng.randn(200)
        pipe = ScalablePipeline(k=5)
        pipe.fit_array(data)
        result = pipe.run_array(data)
        assert isinstance(result, ScalableResult)
        assert len(result.indices) == 5

    def test_run_stream(self):
        rng = np.random.RandomState(42)

        def chunk_iter():
            for _ in range(3):
                yield rng.randn(100)

        pipe = ScalablePipeline(k=5)
        pipe.fit_stream(chunk_iter())
        result = pipe.run_stream(chunk_iter())
        assert isinstance(result, ScalableResult)
        assert len(result.indices) == 5

    def test_summary(self):
        rng = np.random.RandomState(42)
        data = rng.randn(200)
        pipe = ScalablePipeline(k=5)
        pipe.fit_array(data)
        result = pipe.run_array(data)
        summary = result.summary
        assert "selected" in summary
        assert "reduction" in summary
        assert summary["selected"] == 5
