"""Tests for lmm.surprise — SurpriseCalculator."""

from __future__ import annotations

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
