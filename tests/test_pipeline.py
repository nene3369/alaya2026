"""Tests for lmm.pipeline — Pipeline."""

from __future__ import annotations

import numpy as np

from lmm.pipeline import Pipeline, PipelineResult


class TestPipeline:
    def test_run(self):
        rng = np.random.RandomState(42)
        data = rng.randn(100)
        pipe = Pipeline(k=5)
        pipe.fit(data)
        result = pipe.run(data)
        assert isinstance(result, PipelineResult)
        assert len(result.selection.indices) == 5

    def test_run_from_surprises(self):
        rng = np.random.RandomState(42)
        surprises = rng.rand(50)
        pipe = Pipeline(k=3)
        result = pipe.run_from_surprises(surprises)
        assert isinstance(result, PipelineResult)
        assert len(result.selection.indices) == 3

    def test_pipeline_small_data(self):
        data = np.array([1.0, 2.0, 3.0])
        pipe = Pipeline(k=10)
        pipe.fit(data)
        result = pipe.run(data)
        assert len(result.selection.indices) <= 3

    def test_run_loop(self):
        rng = np.random.RandomState(42)
        pipe = Pipeline(k=3)
        ref = rng.randn(50)
        pipe.fit(ref)
        results = []
        for _ in range(3):
            data = rng.randn(50)
            result = pipe.run(data)
            results.append(result)
        assert len(results) == 3
        for r in results:
            assert len(r.selection.indices) == 3
