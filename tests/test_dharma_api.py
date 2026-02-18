"""Tests for lmm.dharma.api — DharmaLMM."""

from __future__ import annotations

import numpy as np

from lmm.dharma.api import DharmaLMM, DharmaResult


class TestDharmaLMM:
    def test_fit_and_select(self):
        rng = np.random.RandomState(42)
        data = rng.randn(50)
        dlmm = DharmaLMM(k=5)
        dlmm.fit(data)
        result = dlmm.select_dharma(data)
        assert isinstance(result, DharmaResult)
        assert hasattr(result, "selected_indices")

    def test_select_from_scores(self):
        rng = np.random.RandomState(42)
        scores = rng.rand(30)

        dlmm = DharmaLMM(k=5)
        result = dlmm.select_from_scores(scores)
        assert isinstance(result, DharmaResult)
        assert hasattr(result, "selected_indices")

    def test_small_data(self):
        rng = np.random.RandomState(42)
        data = rng.randn(3)
        dlmm = DharmaLMM(k=10)
        dlmm.fit(data)
        result = dlmm.select_dharma(data)
        assert result is not None
        assert len(result.selected_indices) <= 3
