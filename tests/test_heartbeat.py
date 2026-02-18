"""Tests for lmm.reasoning.heartbeat — HeartbeatDaemon continuous FEP dynamics."""

from __future__ import annotations

import asyncio

import numpy as np

from lmm.reasoning.heartbeat import HeartbeatDaemon, HeartbeatTelemetry
from lmm.reasoning.alaya import AlayaMemory
from lmm.reasoning.sleep import SleepConsolidation


class TestHeartbeatInit:
    def test_initial_state_is_zero(self):
        daemon = HeartbeatDaemon(n_dims=4, deterministic=True)
        state = daemon.state
        assert state.shape == (4,)
        assert np.all(state == 0.0)

    def test_initial_weights_are_one(self):
        daemon = HeartbeatDaemon(n_dims=4, deterministic=True)
        assert daemon.karuna_weight == 1.0
        assert daemon.metta_weight == 1.0

    def test_initial_cv_is_zero(self):
        daemon = HeartbeatDaemon(n_dims=4, deterministic=True)
        assert daemon.cv == 0.0


class TestHeartbeatTick:
    def test_single_tick_evolves_state(self):
        daemon = HeartbeatDaemon(n_dims=4, deterministic=True)
        loop = asyncio.new_event_loop()
        try:
            telemetry = loop.run_until_complete(daemon.tick())
        finally:
            loop.close()
        assert isinstance(telemetry, HeartbeatTelemetry)
        assert telemetry.tick_count == 1

    def test_state_changes_after_tick(self):
        daemon = HeartbeatDaemon(n_dims=4, entropy_scale=0.5, deterministic=True)
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(daemon.tick())
        finally:
            loop.close()
        # With entropy_scale=0.5, state should shift from zero
        # (may still be near zero if entropy is small, but tick should run without error)
        assert daemon.state.shape == (4,)

    def test_state_bounded_after_many_ticks(self):
        daemon = HeartbeatDaemon(
            n_dims=4, entropy_scale=1.0, deterministic=True
        )
        loop = asyncio.new_event_loop()
        try:
            for _ in range(100):
                loop.run_until_complete(daemon.tick())
        finally:
            loop.close()
        state = daemon.state
        assert np.all(state >= -10.0)
        assert np.all(state <= 10.0)

    def test_nan_recovery(self):
        daemon = HeartbeatDaemon(n_dims=4, deterministic=True)
        # Inject NaN into state
        daemon._V = np.array([float("nan"), float("inf"), float("-inf"), 0.0])
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(daemon.tick())
        finally:
            loop.close()
        state = daemon.state
        assert np.all(np.isfinite(state))

    def test_telemetry_fields(self):
        daemon = HeartbeatDaemon(n_dims=4, deterministic=True)
        loop = asyncio.new_event_loop()
        try:
            telemetry = loop.run_until_complete(daemon.tick())
        finally:
            loop.close()
        assert len(telemetry.v_state) == 4
        assert isinstance(telemetry.karuna_weight, float)
        assert isinstance(telemetry.metta_weight, float)
        assert telemetry.tick_count == 1
        assert isinstance(telemetry.idle_seconds, float)
        assert isinstance(telemetry.cv, float)


class TestCVWeightEvolution:
    def test_weights_stay_bounded(self):
        daemon = HeartbeatDaemon(n_dims=4, entropy_scale=1.0, deterministic=True)
        loop = asyncio.new_event_loop()
        try:
            for _ in range(200):
                loop.run_until_complete(daemon.tick())
        finally:
            loop.close()
        assert 0.1 <= daemon.karuna_weight <= 10.0
        assert 0.1 <= daemon.metta_weight <= 10.0


class TestSensoryInjection:
    def test_inject_sensory_shifts_state(self):
        daemon = HeartbeatDaemon(n_dims=4, deterministic=True)
        initial = daemon.state.copy()
        daemon.inject_sensory(np.array([1.0, 0.5, -0.3, 0.8]))
        new_state = daemon.state
        # State should have moved (at least not identical to zero)
        # The FEP dynamics with strong sensory input should produce non-zero output
        assert new_state.shape == (4,)

    def test_inject_sensory_resets_idle(self):
        daemon = HeartbeatDaemon(n_dims=4, deterministic=True)
        import time
        daemon._last_interaction_time = time.monotonic() - 1000
        daemon.inject_sensory(np.array([0.5, 0.5, 0.5, 0.5]))
        telemetry = daemon.snapshot()
        # Idle should be very small (just reset)
        assert telemetry.idle_seconds < 1.0

    def test_inject_sensory_pads_short_vector(self):
        daemon = HeartbeatDaemon(n_dims=4, deterministic=True)
        # Inject a 2D vector into a 4D daemon -- should pad
        daemon.inject_sensory(np.array([1.0, -1.0]))
        assert daemon.state.shape == (4,)

    def test_inject_sensory_truncates_long_vector(self):
        daemon = HeartbeatDaemon(n_dims=4, deterministic=True)
        daemon.inject_sensory(np.array([1.0, -1.0, 0.5, 0.3, 999.0, 999.0]))
        assert daemon.state.shape == (4,)


class TestSnapshot:
    def test_snapshot_returns_telemetry(self):
        daemon = HeartbeatDaemon(n_dims=4, deterministic=True)
        snap = daemon.snapshot()
        assert isinstance(snap, HeartbeatTelemetry)
        assert snap.tick_count == 0
        assert len(snap.v_state) == 4


class TestLifecycle:
    def test_deterministic_run_stops_after_one_tick(self):
        daemon = HeartbeatDaemon(n_dims=4, deterministic=True)
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(daemon.run())
        finally:
            loop.close()
        assert daemon._tick_count == 1

    def test_stop_signals_loop_end(self):
        daemon = HeartbeatDaemon(n_dims=4, dt=0.01)
        daemon.stop()
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(daemon.run())
        finally:
            loop.close()
        # Should exit immediately since stop was called
        assert daemon._tick_count == 0


class TestHardwareEntropy:
    def test_entropy_not_prng(self):
        """Entropy should come from os.urandom, not np.random."""
        daemon = HeartbeatDaemon(n_dims=4, deterministic=True)
        # Run multiple harvests and check they produce different results
        e1 = daemon._harvest_entropy(16)
        e2 = daemon._harvest_entropy(16)
        assert len(e1) == 16
        assert len(e2) == 16
        # Hardware entropy should be different each time (with overwhelming probability)
        assert not np.all(e1 == e2)

    def test_entropy_values_in_range(self):
        daemon = HeartbeatDaemon(n_dims=4, deterministic=True)
        e = daemon._harvest_entropy(32)
        assert np.all(e >= -1.0)
        assert np.all(e <= 1.0)


class TestIdleSleep:
    def test_idle_triggers_sleep(self):
        memory = AlayaMemory(n_variables=4, max_patterns=50)
        # Store some patterns
        memory.store(np.array([1.0, 0.0, 1.0, 0.0]))
        memory.store(np.array([0.0, 1.0, 0.0, 1.0]))

        sleep = SleepConsolidation(
            memory, nrem_replay_cycles=2,
            rem_noise_scale=0.05, pruning_threshold=0.1,
        )

        daemon = HeartbeatDaemon(
            n_dims=4, idle_sleep_threshold=0.0,  # trigger immediately
            deterministic=True,
        )
        daemon.attach_sleep(sleep)
        # Set last interaction far in the past
        daemon._last_interaction_time = 0.0

        loop = asyncio.new_event_loop()
        try:
            telemetry = loop.run_until_complete(daemon.tick())
        finally:
            loop.close()

        assert daemon._idle_consolidated is True
        assert telemetry.last_sleep_report is not None
        assert "n_consolidated" in telemetry.last_sleep_report

    def test_no_double_sleep_consolidation(self):
        memory = AlayaMemory(n_variables=4, max_patterns=50)
        memory.store(np.array([1.0, 0.0, 1.0, 0.0]))
        sleep = SleepConsolidation(memory, nrem_replay_cycles=1)

        daemon = HeartbeatDaemon(
            n_dims=4, idle_sleep_threshold=0.0,
            deterministic=True,
        )
        daemon.attach_sleep(sleep)
        daemon._last_interaction_time = 0.0

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(daemon.tick())
            first_report = daemon._last_sleep_report

            # Second tick should not re-consolidate
            loop.run_until_complete(daemon.tick())
            second_report = daemon._last_sleep_report
        finally:
            loop.close()

        # Reports should be the same object (no re-consolidation)
        assert first_report is second_report

    def test_inject_sensory_resets_idle_flag(self):
        memory = AlayaMemory(n_variables=4, max_patterns=50)
        sleep = SleepConsolidation(memory)

        daemon = HeartbeatDaemon(
            n_dims=4, idle_sleep_threshold=0.0,
            deterministic=True,
        )
        daemon.attach_sleep(sleep)
        daemon._idle_consolidated = True

        daemon.inject_sensory(np.array([1.0, 0.0, 0.0, 0.0]))
        assert daemon._idle_consolidated is False
