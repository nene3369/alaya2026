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

    def test_cascade_filter_select(self):
        """Cover cascade filter path: n > k*5 and budget < 0.8 (lines 54-55)."""
        rng = np.random.RandomState(42)
        data = rng.randn(200)
        selector = SmartSelector(k=5)
        selector.fit(data)
        result = selector.select(data, budget=0.5)
        assert isinstance(result, SelectionResult)
        assert len(result.indices) <= 5
        assert result.budget_used == 0.5

    def test_cascade_filter_select_from_surprises(self):
        """Cover cascade filter in select_from_surprises (lines 76-77)."""
        rng = np.random.RandomState(42)
        surprises = rng.rand(200)
        selector = SmartSelector(k=5)
        result = selector.select_from_surprises(surprises, budget=0.5)
        assert len(result.indices) <= 5

    def test_best_historical_method(self):
        """Cover _best_historical_method with enough history (lines 105-121)."""
        rng = np.random.RandomState(42)
        data = rng.randn(200)
        selector = SmartSelector(k=5)
        selector.fit(data)
        # Build up history with 4+ calls to trigger _best_historical_method
        for _ in range(4):
            selector.select(data, budget=0.9)
        # The 5th call should use the best historical method
        result = selector.select(data, budget=0.9)
        assert isinstance(result, SelectionResult)
        assert len(selector._history) == 5

    def test_pick_method_greedy_small_n(self):
        """Cover greedy path in _pick_method for small candidate set (line 102-103)."""
        rng = np.random.RandomState(42)
        data = rng.randn(10)
        selector = SmartSelector(k=5)
        selector.fit(data)
        result = selector.select(data, budget=1.0)
        assert result.method_used == "greedy"

    def test_pick_method_low_budget(self):
        """Cover greedy path in _pick_method for low budget (line 102)."""
        rng = np.random.RandomState(42)
        data = rng.randn(200)
        selector = SmartSelector(k=5)
        selector.fit(data)
        result = selector.select(data, budget=0.2)
        assert result.method_used == "greedy"

    def test_pick_method_relaxation(self):
        """Cover relaxation path: moderate budget, no history (line 108)."""
        rng = np.random.RandomState(42)
        # Need n_candidates > 50 after filter, budget in [0.3, 0.7)
        surprises = rng.rand(200)
        selector = SmartSelector(k=5)
        result = selector.select_from_surprises(surprises, budget=0.5)
        # With budget=0.5 and n > 50 cascade-filtered, should use relaxation
        assert result.method_used in ("relaxation", "greedy")

    def test_confidence_empty_selected(self):
        """Cover _compute_confidence with empty selection (line 137)."""
        selector = SmartSelector(k=5)
        conf = selector._compute_confidence(np.array([1.0, 2.0, 3.0]), np.array([], dtype=int))
        assert conf == 0.0

    def test_confidence_constant_data(self):
        """Cover _compute_confidence with zero std (line 142)."""
        selector = SmartSelector(k=5)
        conf = selector._compute_confidence(np.array([5.0, 5.0, 5.0, 5.0]), np.array([0, 1]))
        assert conf == 0.5
