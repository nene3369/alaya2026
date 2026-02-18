"""Tests for lmm.reasoning.recovery — CircuitBreaker + AgentWatchdog."""

from __future__ import annotations

import time

import numpy as np

from lmm.reasoning.recovery import (
    AgentWatchdog,
    CBState,
    CircuitBreaker,
    HealthReport,
)


class TestCircuitBreaker:
    def test_initial_state_closed(self):
        cb = CircuitBreaker("test", failure_threshold=3)
        assert cb.state == CBState.CLOSED
        assert cb.allow_request()

    def test_stays_closed_under_threshold(self):
        cb = CircuitBreaker("test", failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CBState.CLOSED
        assert cb.allow_request()

    def test_opens_at_threshold(self):
        cb = CircuitBreaker("test", failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CBState.OPEN
        assert not cb.allow_request()

    def test_success_resets_consecutive_failures(self):
        cb = CircuitBreaker("test", failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        cb.record_failure()
        # Only 1 consecutive failure after reset
        assert cb.state == CBState.CLOSED

    def test_open_to_half_open_after_cooldown(self):
        cb = CircuitBreaker("test", failure_threshold=2, cooldown=0.01)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CBState.OPEN
        time.sleep(0.02)
        assert cb.state == CBState.HALF_OPEN
        assert cb.allow_request()

    def test_half_open_to_closed_on_success(self):
        cb = CircuitBreaker("test", failure_threshold=2, cooldown=0.01)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.02)
        assert cb.state == CBState.HALF_OPEN
        cb.record_success()
        assert cb.state == CBState.CLOSED

    def test_half_open_to_open_on_failure(self):
        cb = CircuitBreaker("test", failure_threshold=2, cooldown=0.01)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.02)
        assert cb.state == CBState.HALF_OPEN
        cb.record_failure()
        assert cb.state == CBState.OPEN

    def test_manual_reset(self):
        cb = CircuitBreaker("test", failure_threshold=2)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CBState.OPEN
        cb.reset()
        assert cb.state == CBState.CLOSED
        assert cb.allow_request()

    def test_stats(self):
        cb = CircuitBreaker("test_comp", failure_threshold=3)
        cb.record_success()
        cb.record_failure()
        stats = cb.stats()
        assert stats.name == "test_comp"
        assert stats.state == "closed"
        assert stats.failure_count == 1
        assert stats.success_count == 1
        assert stats.total_calls == 2

    def test_success_threshold_in_half_open(self):
        cb = CircuitBreaker(
            "test", failure_threshold=2, cooldown=0.01, success_threshold=2,
        )
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.02)
        assert cb.state == CBState.HALF_OPEN
        cb.record_success()
        # Need 2 successes; still HALF_OPEN after 1
        assert cb.state == CBState.HALF_OPEN
        cb.record_success()
        assert cb.state == CBState.CLOSED


class TestAgentWatchdog:
    def test_healthy_state(self):
        wd = AgentWatchdog("test")
        wd.record_tick()
        state = np.array([1.0, 2.0, -3.0])
        report = wd.check_health(state)
        assert report.healthy
        assert report.checks["liveness"]
        assert report.checks["finite"]
        assert report.checks["in_bounds"]

    def test_detects_nan(self):
        wd = AgentWatchdog("test")
        wd.record_tick()
        state = np.array([1.0, float("nan"), 3.0])
        report = wd.check_health(state)
        assert not report.healthy
        assert not report.checks["finite"]
        assert report.recovery_action == "reset_state"

    def test_detects_inf(self):
        wd = AgentWatchdog("test")
        wd.record_tick()
        state = np.array([1.0, float("inf"), 3.0])
        report = wd.check_health(state)
        assert not report.healthy
        assert not report.checks["finite"]

    def test_detects_out_of_bounds(self):
        wd = AgentWatchdog("test", value_bounds=(-10.0, 10.0))
        wd.record_tick()
        state = np.array([1.0, 200.0, 3.0])
        report = wd.check_health(state)
        assert not report.healthy
        assert not report.checks["in_bounds"]
        assert report.recovery_action == "reset_state"

    def test_detects_liveness_failure(self):
        wd = AgentWatchdog("test", max_silent_seconds=0.01)
        wd.record_tick()
        time.sleep(0.02)
        report = wd.check_health(np.array([1.0]))
        assert not report.healthy
        assert not report.checks["liveness"]
        assert report.recovery_action == "restart"

    def test_detects_stagnation(self):
        wd = AgentWatchdog("test", stagnation_window=5, stagnation_threshold=1e-6)
        for _ in range(10):
            wd.record_tick(energy=1.0)  # Same energy every tick
        report = wd.check_health(np.array([1.0]))
        assert not report.healthy
        assert not report.checks["progressing"]
        assert report.recovery_action == "perturb_state"

    def test_no_stagnation_with_variation(self):
        wd = AgentWatchdog("test", stagnation_window=5, stagnation_threshold=1e-6)
        for i in range(10):
            wd.record_tick(energy=float(i))
        report = wd.check_health(np.array([1.0]))
        assert report.healthy
        assert report.checks["progressing"]

    def test_execute_recovery_reset(self):
        wd = AgentWatchdog("test")
        state = np.array([float("nan"), 100.0, -50.0])
        report = HealthReport(
            healthy=False,
            checks={"finite": False},
            recovery_action="reset_state",
        )
        recovered = wd.execute_recovery(state, report)
        assert np.all(recovered == 0.0)
        assert wd.recovery_count == 1

    def test_execute_recovery_perturb(self):
        wd = AgentWatchdog("test", value_bounds=(-10.0, 10.0))
        state = np.array([5.0, 5.0, 5.0])
        report = HealthReport(
            healthy=False,
            checks={"progressing": False},
            recovery_action="perturb_state",
        )
        recovered = wd.execute_recovery(state, report)
        # Should be slightly different from original
        assert recovered.shape == state.shape
        assert np.all(recovered >= -10.0)
        assert np.all(recovered <= 10.0)

    def test_health_check_without_state(self):
        wd = AgentWatchdog("test")
        wd.record_tick()
        report = wd.check_health(state=None)
        assert report.healthy
        assert report.checks["finite"]
        assert report.checks["in_bounds"]

    def test_not_enough_data_for_stagnation(self):
        wd = AgentWatchdog("test", stagnation_window=10)
        for i in range(3):
            wd.record_tick(energy=1.0)
        report = wd.check_health(np.array([1.0]))
        assert report.checks["progressing"]  # Not enough data, skip check
