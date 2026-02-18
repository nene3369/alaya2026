"""Tests for lmm.dharma.engine — UniversalDharmaEngine."""

from __future__ import annotations

import numpy as np

from lmm.dharma.energy import KarunaTerm, PrajnaTerm, SilaTerm
from lmm.dharma.engine import EngineResult, UniversalDharmaEngine


class TestUniversalDharmaEngine:
    def _make_inputs(self, n: int = 20) -> dict:
        rng = np.random.RandomState(42)
        return {
            "surprises": rng.rand(n),
            "impact_graph": rng.rand(n, n) * 0.3,
        }

    def test_auto_mode(self):
        n = 20
        inputs = self._make_inputs(n)
        engine = UniversalDharmaEngine(n_variables=n)
        engine.add(PrajnaTerm(inputs["surprises"]))
        engine.add(SilaTerm(k=5, weight=10.0))
        result = engine.synthesize_and_solve(k=5)
        assert isinstance(result, EngineResult)
        assert len(result.selected_indices) == 5

    def test_submodular_mode(self):
        n = 20
        inputs = self._make_inputs(n)
        engine = UniversalDharmaEngine(n_variables=n)
        engine.add(PrajnaTerm(inputs["surprises"]))
        engine.add(SilaTerm(k=5, weight=10.0))
        result = engine.synthesize_and_solve(k=5)
        assert len(result.selected_indices) == 5

    def test_top_k_mode(self):
        n = 20
        inputs = self._make_inputs(n)
        # top_k is auto-routed when there's no quadratic term
        engine = UniversalDharmaEngine(n_variables=n)
        engine.add(PrajnaTerm(inputs["surprises"]))
        result = engine.synthesize_and_solve(k=5)
        assert len(result.selected_indices) == 5

    def test_ising_sa_mode(self):
        n = 20
        inputs = self._make_inputs(n)
        engine = UniversalDharmaEngine(n_variables=n, solver="ising_sa")
        engine.add(PrajnaTerm(inputs["surprises"]))
        engine.add(KarunaTerm(inputs["impact_graph"]))
        engine.add(SilaTerm(k=5, weight=10.0))
        result = engine.synthesize_and_solve(k=5)
        assert len(result.selected_indices) == 5

    def test_fep_mode(self):
        n = 15
        inputs = self._make_inputs(n)
        engine = UniversalDharmaEngine(n_variables=n, solver="fep")
        engine.add(PrajnaTerm(inputs["surprises"]))
        engine.add(KarunaTerm(inputs["impact_graph"]))
        engine.add(SilaTerm(k=4, weight=10.0))
        result = engine.synthesize_and_solve(k=4)
        assert len(result.selected_indices) == 4

    def test_fep_analog_mode(self):
        n = 15
        inputs = self._make_inputs(n)
        engine = UniversalDharmaEngine(n_variables=n, solver="fep_analog")
        engine.add(PrajnaTerm(inputs["surprises"]))
        engine.add(KarunaTerm(inputs["impact_graph"]))
        engine.add(SilaTerm(k=4, weight=10.0))
        result = engine.synthesize_and_solve(k=4)
        assert len(result.selected_indices) == 4

    def test_k_larger_than_n(self):
        n = 3
        inputs = self._make_inputs(n)
        engine = UniversalDharmaEngine(n_variables=n)
        engine.add(PrajnaTerm(inputs["surprises"]))
        engine.add(SilaTerm(k=10, weight=10.0))
        result = engine.synthesize_and_solve(k=10)
        assert len(result.selected_indices) <= 3
