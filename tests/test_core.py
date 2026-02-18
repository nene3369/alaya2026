"""Tests for lmm.core — LMM + LMMResult."""

from __future__ import annotations

import numpy as np

from lmm.core import LMM, LMMResult


class TestLMM:
    def test_fit_and_select(self):
        rng = np.random.RandomState(42)
        data = rng.randn(100)
        model = LMM(k=5)
        model.fit(data)
        result = model.select(data)
        assert isinstance(result, LMMResult)
        assert len(result.selected_indices) == 5
        assert len(result.surprise_values) == 5

    def test_select_from_surprises(self):
        rng = np.random.RandomState(42)
        surprises = rng.rand(50)
        model = LMM(k=3)
        result = model.select_from_surprises(surprises)
        assert isinstance(result, LMMResult)
        assert len(result.selected_indices) == 3

    def test_k_larger_than_data(self):
        data = np.array([1.0, 2.0, 3.0])
        model = LMM(k=10)
        model.fit(data)
        result = model.select(data)
        assert len(result.selected_indices) <= 3

    def test_result_attributes(self):
        rng = np.random.RandomState(42)
        data = rng.randn(50)
        model = LMM(k=5)
        model.fit(data)
        result = model.select(data)
        assert hasattr(result, "selected_indices")
        assert hasattr(result, "surprise_values")
        assert hasattr(result, "energy")
        assert hasattr(result, "method")
        assert result.selected_indices.dtype in (np.int64, np.int32, np.intp)

    def test_different_methods(self):
        rng = np.random.RandomState(42)
        data = rng.randn(80)
        for method in ["sa", "greedy"]:
            model = LMM(k=5, solver_method=method)
            model.fit(data)
            result = model.select(data)
            assert len(result.selected_indices) == 5
