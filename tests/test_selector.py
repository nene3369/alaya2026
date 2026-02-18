"""Tests for lmm.selector — SmartSelector."""

from __future__ import annotations

import numpy as np

from lmm.selector import SmartSelector, SelectionResult


class TestSmartSelector:
    def test_basic_select(self):
        rng = np.random.RandomState(42)
        data = rng.randn(50)
        selector = SmartSelector(k=5)
        selector.fit(data)
        result = selector.select(data)
        assert isinstance(result, SelectionResult)
        assert len(result.indices) == 5
        assert hasattr(result, "scores")

    def test_select_from_surprises(self):
        rng = np.random.RandomState(42)
        surprises = rng.rand(30)
        selector = SmartSelector(k=5)
        # select_from_surprises doesn't need fit
        result = selector.select_from_surprises(surprises)
        assert len(result.indices) == 5

    def test_k_larger_than_n(self):
        data = np.array([0.5, 0.3, 0.8])
        selector = SmartSelector(k=10)
        selector.fit(data)
        result = selector.select(data)
        assert len(result.indices) <= 3

    def test_confidence_scores(self):
        rng = np.random.RandomState(42)
        data = rng.randn(20)
        selector = SmartSelector(k=5)
        selector.fit(data)
        result = selector.select(data)
        assert hasattr(result, "confidence")
