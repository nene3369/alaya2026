"""Tests for lmm.surprise — SurpriseCalculator."""

from __future__ import annotations

import pytest
import numpy as np

from lmm.surprise import SurpriseCalculator


class TestSurpriseCalculator:
    def test_fit_and_compute_kl(self):
        rng = np.random.RandomState(42)
        data = rng.randn(200)
        calc = SurpriseCalculator(method="kl")
        calc.fit(data)
        surprises = calc.compute(data)
        assert surprises.shape == (200,)
        assert np.all(np.isfinite(surprises))
        assert np.all(surprises >= 0)

    def test_fit_and_compute_entropy(self):
        rng = np.random.RandomState(42)
        data = rng.randn(100)
        calc = SurpriseCalculator(method="entropy")
        calc.fit(data)
        surprises = calc.compute(data)
        assert surprises.shape == (100,)
        assert np.all(np.isfinite(surprises))

    def test_fit_and_compute_bayesian(self):
        rng = np.random.RandomState(42)
        data = rng.randn(50)
        calc = SurpriseCalculator(method="bayesian")
        calc.fit(data)
        surprises = calc.compute(data)
        assert surprises.shape == (50,)
        assert np.all(np.isfinite(surprises))

    def test_outliers_higher_surprise(self):
        rng = np.random.RandomState(42)
        normal = rng.randn(500)
        calc = SurpriseCalculator(method="kl")
        calc.fit(normal)

        inliers = np.array([0.0, 0.1, -0.1])
        outliers = np.array([10.0, -10.0, 15.0])
        s_in = calc.compute(inliers)
        s_out = calc.compute(outliers)
        assert s_out.mean() > s_in.mean()

    def test_empty_data(self):
        calc = SurpriseCalculator(method="kl")
        calc.fit(np.array([1.0, 2.0, 3.0]))
        result = calc.compute(np.array([]))
        assert len(result) == 0

    def test_constant_data(self):
        data = np.ones(50)
        calc = SurpriseCalculator(method="kl")
        calc.fit(data)
        surprises = calc.compute(data)
        assert surprises.shape == (50,)
        assert np.all(np.isfinite(surprises))

    def test_single_element(self):
        data = np.array([1.0])
        calc = SurpriseCalculator(method="entropy")
        calc.fit(data)
        surprises = calc.compute(data)
        assert surprises.shape == (1,)

    def test_unknown_method_raises(self):
        """Cover ValueError for unknown method (line 35)."""
        calc = SurpriseCalculator(method="unknown")
        calc.fit(np.array([1.0, 2.0, 3.0]))
        with pytest.raises(ValueError, match="Unknown method"):
            calc.compute(np.array([1.0]))

    def test_surprise_from_prior_not_fitted(self):
        """Cover RuntimeError when prior is None in _surprise_from_prior (line 56)."""
        calc = SurpriseCalculator(method="kl")
        with pytest.raises(RuntimeError, match="fit"):
            calc.compute(np.array([1.0, 2.0]))

    def test_bayesian_surprise_not_fitted(self):
        """Cover RuntimeError when prior is None in _bayesian_surprise (line 65)."""
        calc = SurpriseCalculator(method="bayesian")
        with pytest.raises(RuntimeError, match="fit"):
            calc.compute(np.array([1.0, 2.0]))

    def test_get_bin_method(self):
        """Cover _get_bin method (lines 94-100)."""
        calc = SurpriseCalculator(method="kl")
        # Without fit (no bin_edges)
        idx = calc._get_bin(0.5)
        assert 0 <= idx < 50

        # With fit (has bin_edges)
        calc.fit(np.random.RandomState(42).randn(100))
        idx_fitted = calc._get_bin(0.0)
        assert 0 <= idx_fitted < 50

    def test_get_bin_with_ndarray_value(self):
        """Cover _get_bin with ndarray value (line 95-96)."""
        calc = SurpriseCalculator(method="kl")
        calc.fit(np.random.RandomState(42).randn(100))
        idx = calc._get_bin(np.array([1.0, 2.0, 3.0]))
        assert 0 <= idx < 50

    def test_vectorized_bin_indices_without_edges(self):
        """Cover _vectorized_bin_indices fallback without bin_edges (line 52)."""
        calc = SurpriseCalculator(method="kl")
        # Don't fit, so _bin_edges is None
        indices = calc._vectorized_bin_indices(np.array([0.1, 0.5, 0.9]))
        assert indices.shape == (3,)
        assert np.all(indices >= 0)
        assert np.all(indices < 50)

    def test_multidimensional_input(self):
        """Cover _to_scalar and _estimate_distribution with multi-dim data (lines 40-41, 89-90)."""
        rng = np.random.RandomState(42)
        data = rng.randn(100, 5)
        calc = SurpriseCalculator(method="kl")
        calc.fit(data)
        surprises = calc.compute(data)
        assert surprises.shape == (100,)
        assert np.all(np.isfinite(surprises))

    def test_multidimensional_bayesian(self):
        """Cover _to_scalar in bayesian path with multi-dim data."""
        rng = np.random.RandomState(42)
        data = rng.randn(30, 3)
        calc = SurpriseCalculator(method="bayesian")
        calc.fit(data)
        surprises = calc.compute(data)
        assert surprises.shape == (30,)
        assert np.all(np.isfinite(surprises))
