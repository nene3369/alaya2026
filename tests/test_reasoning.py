"""Tests for lmm.reasoning — all reasoning modes + orchestrator."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from lmm.reasoning.base import ReasonerResult, compute_complexity
from lmm.reasoning.adaptive import AdaptiveReasoner
from lmm.reasoning.theoretical import TheoreticalReasoner
from lmm.reasoning.hyper import HyperReasoner
from lmm.reasoning.active import ActiveInferenceEngine
from lmm.reasoning.alaya import AlayaMemory
from lmm.reasoning.sleep import SleepConsolidation
from lmm.reasoning.embodiment import EmbodiedAgent
from lmm.reasoning.pineal import PinealGland
from lmm.reasoning.orchestrator import DharmaReasonerOrchestrator


def _make_problem(n: int = 20, density: float = 0.3):
    rng = np.random.RandomState(42)
    h = rng.randn(n)
    dense = rng.rand(n, n) * density
    dense = (dense + dense.T) / 2
    np.fill_diagonal(dense, 0)
    J = sparse.csr_matrix(dense)
    return h, J


class TestComplexityProfile:
    def test_compute_complexity(self):
        rng = np.random.RandomState(42)
        data = rng.rand(50)
        profile = compute_complexity(data)
        assert hasattr(profile, "gini")
        assert hasattr(profile, "cv")
        assert hasattr(profile, "entropy")
        assert 0 <= profile.gini <= 1


class TestAdaptiveReasoner:
    def test_reason(self):
        h, J = _make_problem(30)
        reasoner = AdaptiveReasoner(n_variables=30, k=5)
        result = reasoner.reason(h, J)
        assert isinstance(result, ReasonerResult)
        assert len(result.selected_indices) == 5


class TestTheoreticalReasoner:
    def test_reason(self):
        h, J = _make_problem(20)
        reasoner = TheoreticalReasoner(n_variables=20, k=5)
        result = reasoner.reason(h, J)
        assert isinstance(result, ReasonerResult)
        assert len(result.selected_indices) == 5


class TestHyperReasoner:
    def test_reason(self):
        h, J = _make_problem(20)
        reasoner = HyperReasoner(n_variables=20, k=5)
        result = reasoner.reason(h, J)
        assert isinstance(result, ReasonerResult)
        assert len(result.selected_indices) == 5


class TestActiveInferenceEngine:
    def test_reason(self):
        h, J = _make_problem(20)
        engine = ActiveInferenceEngine(n_variables=20, k=5, n_iterations=2)
        result = engine.reason(h, J)
        assert isinstance(result, ReasonerResult)
        assert len(result.selected_indices) == 5


class TestAlayaMemory:
    def test_store_and_recall(self):
        rng = np.random.RandomState(42)
        memory = AlayaMemory(n_variables=10, learning_rate=0.1)
        pattern = rng.randn(10)
        memory.store(pattern)
        recalled = memory.recall(pattern + rng.randn(10) * 0.1)
        assert recalled is not None

    def test_decay(self):
        rng = np.random.RandomState(42)
        memory = AlayaMemory(n_variables=10)
        for _ in range(5):
            memory.store(rng.randn(10))
        memory.decay()


class TestSleepConsolidation:
    def test_consolidate(self):
        rng = np.random.RandomState(42)
        memory = AlayaMemory(n_variables=10, learning_rate=0.1)
        for _ in range(5):
            memory.store(rng.rand(10))
        sleep = SleepConsolidation(memory, nrem_replay_cycles=2)
        result = sleep.consolidate()
        assert result is not None


class TestEmbodiedAgent:
    def test_add_sense_and_fuse(self):
        rng = np.random.RandomState(42)
        agent = EmbodiedAgent(n_variables=10, k=3)
        agent.add_sense("vision", rng.randn(10))
        agent.add_sense("audio", rng.randn(10))
        fused = agent.fuse_senses()
        assert fused is not None
        assert fused.shape == (10,)


class TestPinealGland:
    def test_harvest_entropy(self):
        pg = PinealGland(n_variables=20, k=5)
        entropy = pg.harvest_entropy(n_bytes=16)
        assert len(entropy) == 16

    def test_reason(self):
        h, J = _make_problem(20)
        pg = PinealGland(n_variables=20, k=5)
        result = pg.reason(h, J)
        assert isinstance(result, ReasonerResult)
        assert len(result.selected_indices) == 5


class TestDharmaReasonerOrchestrator:
    def test_select_mode(self):
        h, J = _make_problem(20)
        orch = DharmaReasonerOrchestrator()
        orch.register(AdaptiveReasoner(n_variables=20, k=5))
        mode = orch.select_mode(h)
        assert isinstance(mode, str)

    def test_reason(self):
        h, J = _make_problem(30)
        orch = DharmaReasonerOrchestrator()
        orch.register(AdaptiveReasoner(n_variables=30, k=5))
        result = orch.reason(h, J)
        assert result.best_result is not None
        assert len(result.best_result.selected_indices) == 5

    def test_run_all(self):
        h, J = _make_problem(20)
        orch = DharmaReasonerOrchestrator()
        orch.register(AdaptiveReasoner(n_variables=20, k=5))
        result = orch.run_all(h, J)
        assert isinstance(result.all_results, dict)
        assert len(result.all_results) > 0

    def test_ensemble_vote(self):
        h, J = _make_problem(20)
        orch = DharmaReasonerOrchestrator()
        orch.register(AdaptiveReasoner(n_variables=20, k=5))
        result = orch.ensemble_vote(h, J, k=5)
        assert result.ensemble_indices is not None
        assert len(result.ensemble_indices) == 5
