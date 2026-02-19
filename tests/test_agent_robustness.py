"""Tests for agent robustness — orchestrator think(), heartbeat safety, recovery."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
from scipy import sparse

from lmm.reasoning.adaptive import AdaptiveReasoner
from lmm.reasoning.alaya import AlayaMemory
from lmm.reasoning.heartbeat import HeartbeatDaemon
from lmm.reasoning.hyper import HyperReasoner
from lmm.reasoning.orchestrator import DharmaReasonerOrchestrator
from lmm.reasoning.recovery import AgentWatchdog


def _make_problem(n: int = 20, density: float = 0.3):
    rng = np.random.RandomState(42)
    h = rng.randn(n)
    dense = rng.rand(n, n) * density
    dense = (dense + dense.T) / 2
    np.fill_diagonal(dense, 0)
    J = sparse.csr_matrix(dense)
    return h, J


# ===================================================================
# Orchestrator.think() tests
# ===================================================================


class TestOrchestratorThink:
    def test_think_basic(self):
        """think() should return a valid result with a single reasoner."""
        h, J = _make_problem(20)
        orch = DharmaReasonerOrchestrator()
        orch.register(AdaptiveReasoner(n_variables=20, k=5))
        result = orch.think(h, J)
        assert result.best_result is not None
        assert len(result.best_result.selected_indices) == 5
        assert result.mode_selected in orch.available_modes

    def test_think_with_multiple_modes(self):
        """think() should work with multiple reasoners registered."""
        h, J = _make_problem(20)
        orch = DharmaReasonerOrchestrator()
        orch.register(AdaptiveReasoner(n_variables=20, k=5))
        orch.register(HyperReasoner(n_variables=20, k=5))
        result = orch.think(h, J)
        assert result.best_result is not None
        assert len(result.best_result.selected_indices) == 5

    def test_think_records_to_alaya(self):
        """think() should record pattern to AlayaMemory when provided."""
        h, J = _make_problem(20)
        orch = DharmaReasonerOrchestrator()
        orch.register(AdaptiveReasoner(n_variables=20, k=5))
        alaya = AlayaMemory(n_variables=20)
        result = orch.think(h, J, alaya=alaya)
        assert result.best_result is not None
        # AlayaMemory should have at least one pattern recorded
        assert alaya.n_patterns > 0

    def test_think_no_reasoners_raises(self):
        """think() without reasoners should raise RuntimeError."""
        h, J = _make_problem(20)
        orch = DharmaReasonerOrchestrator()
        with pytest.raises(RuntimeError, match="No reasoners registered"):
            orch.think(h, J)


# ===================================================================
# Circuit breaker fallback tests
# ===================================================================


class TestAllBreakersOpen:
    def test_select_mode_all_open(self):
        """select_mode should return first reasoner even when all breakers open."""
        h, _ = _make_problem(20)
        orch = DharmaReasonerOrchestrator()
        orch.register(AdaptiveReasoner(n_variables=20, k=5))
        # Force the circuit breaker to OPEN
        breaker = orch._breakers["adaptive"]
        for _ in range(breaker.failure_threshold):
            breaker.record_failure()
        # Should still return a mode (forced fallback)
        mode = orch.select_mode(h)
        assert mode == "adaptive"

    def test_reason_with_open_breaker(self):
        """reason() should still work even if breaker is open (forced)."""
        h, J = _make_problem(20)
        orch = DharmaReasonerOrchestrator()
        orch.register(AdaptiveReasoner(n_variables=20, k=5))
        # Force OPEN
        breaker = orch._breakers["adaptive"]
        for _ in range(breaker.failure_threshold):
            breaker.record_failure()
        # Direct reason() bypasses circuit breaker check
        result = orch.reason(h, J, mode="adaptive")
        assert result.best_result is not None


# ===================================================================
# HeartbeatDaemon asyncio.Event stop tests
# ===================================================================


class TestHeartbeatStopEvent:
    def test_stop_prevents_run(self):
        """Calling stop() before run() should prevent ticks."""
        daemon = HeartbeatDaemon(n_dims=4, dt=0.01)
        daemon.stop()
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(daemon.run())
        finally:
            loop.close()
        assert daemon._tick_count == 0

    def test_deterministic_run_still_works(self):
        """Deterministic mode should still run one tick and stop."""
        daemon = HeartbeatDaemon(n_dims=4, deterministic=True)
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(daemon.run())
        finally:
            loop.close()
        assert daemon._tick_count == 1


# ===================================================================
# HeartbeatDaemon parameter validation tests
# ===================================================================


class TestHeartbeatParameterValidation:
    def test_n_dims_zero_raises(self):
        with pytest.raises(ValueError, match="n_dims must be >= 1"):
            HeartbeatDaemon(n_dims=0)

    def test_n_dims_negative_raises(self):
        with pytest.raises(ValueError, match="n_dims must be >= 1"):
            HeartbeatDaemon(n_dims=-1)

    def test_dt_zero_raises(self):
        with pytest.raises(ValueError, match="dt must be positive"):
            HeartbeatDaemon(dt=0.0)

    def test_dt_negative_raises(self):
        with pytest.raises(ValueError, match="dt must be positive"):
            HeartbeatDaemon(dt=-0.1)

    def test_n_dims_exceeds_max_raises(self):
        with pytest.raises(ValueError, match="exceeds maximum"):
            HeartbeatDaemon(n_dims=2000)


# ===================================================================
# AgentWatchdog reset_recovery_count tests
# ===================================================================


class TestWatchdogRecoveryReset:
    def test_reset_recovery_count(self):
        wd = AgentWatchdog("test")
        state = np.array([float("nan"), 1.0])
        report = wd.check_health(state)
        wd.execute_recovery(state, report)
        assert wd.recovery_count == 1
        wd.reset_recovery_count()
        assert wd.recovery_count == 0

    def test_recovery_count_accumulates(self):
        wd = AgentWatchdog("test")
        state = np.array([float("nan"), 1.0])
        report = wd.check_health(state)
        wd.execute_recovery(state, report)
        wd.execute_recovery(state, report)
        wd.execute_recovery(state, report)
        assert wd.recovery_count == 3
        wd.reset_recovery_count()
        assert wd.recovery_count == 0


# ===================================================================
# Circuit breaker stats snapshot test
# ===================================================================


class TestCircuitBreakerStats:
    def test_stats_after_failures(self):
        orch = DharmaReasonerOrchestrator()
        orch.register(AdaptiveReasoner(n_variables=10, k=3))
        breaker = orch._breakers["adaptive"]
        breaker.record_failure()
        breaker.record_failure()
        stats = orch.circuit_breaker_stats
        assert "adaptive" in stats
        assert stats["adaptive"]["failures"] == 2
        assert stats["adaptive"]["consecutive_failures"] == 2
