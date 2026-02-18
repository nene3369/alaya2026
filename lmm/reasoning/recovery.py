"""Auto-recovery primitives — CircuitBreaker + AgentWatchdog.

CircuitBreaker: tracks per-component failure rates and automatically
bypasses broken components until they recover.

AgentWatchdog: monitors agent state health (NaN, stuck, silent crash)
and triggers automatic resets when anomalies are detected.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


# ===================================================================
# Circuit Breaker
# ===================================================================

class CBState(Enum):
    CLOSED = "closed"        # Normal operation
    OPEN = "open"            # Failures exceeded threshold — bypassed
    HALF_OPEN = "half_open"  # Cooldown expired — probing recovery


@dataclass
class CircuitStats:
    """Snapshot of circuit breaker metrics."""

    name: str
    state: str
    failure_count: int
    success_count: int
    consecutive_failures: int
    last_failure_time: float | None
    total_calls: int


class CircuitBreaker:
    """Per-component circuit breaker with automatic recovery.

    State machine::

        CLOSED --(failures >= threshold)--> OPEN
        OPEN   --(cooldown expires)-------> HALF_OPEN
        HALF_OPEN --(success)-------------> CLOSED
        HALF_OPEN --(failure)-------------> OPEN

    Parameters
    ----------
    name : str
        Identifier for the protected component.
    failure_threshold : int
        Consecutive failures before opening the circuit.
    cooldown : float
        Seconds to wait before probing recovery (OPEN → HALF_OPEN).
    success_threshold : int
        Consecutive successes in HALF_OPEN before closing.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        cooldown: float = 30.0,
        success_threshold: int = 1,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.cooldown = cooldown
        self.success_threshold = success_threshold

        self._state = CBState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._consecutive_failures = 0
        self._consecutive_successes = 0
        self._last_failure_time: float | None = None
        self._last_state_change = time.monotonic()
        self._total_calls = 0

    @property
    def state(self) -> CBState:
        # Auto-transition OPEN → HALF_OPEN after cooldown
        if self._state == CBState.OPEN and self._last_failure_time is not None:
            elapsed = time.monotonic() - self._last_failure_time
            if elapsed >= self.cooldown:
                self._state = CBState.HALF_OPEN
                self._consecutive_successes = 0
                self._last_state_change = time.monotonic()
        return self._state

    def allow_request(self) -> bool:
        """Check whether a request should be allowed through."""
        s = self.state  # triggers auto-transition
        if s == CBState.CLOSED:
            return True
        if s == CBState.HALF_OPEN:
            return True  # Allow probe request
        return False  # OPEN — skip this component

    def record_success(self) -> None:
        """Record a successful call."""
        self._total_calls += 1
        self._success_count += 1
        self._consecutive_failures = 0
        self._consecutive_successes += 1

        if self._state == CBState.HALF_OPEN:
            if self._consecutive_successes >= self.success_threshold:
                self._state = CBState.CLOSED
                self._last_state_change = time.monotonic()

    def record_failure(self) -> None:
        """Record a failed call (exception, timeout, or invalid result)."""
        self._total_calls += 1
        self._failure_count += 1
        self._consecutive_failures += 1
        self._consecutive_successes = 0
        self._last_failure_time = time.monotonic()

        if self._state == CBState.HALF_OPEN:
            self._state = CBState.OPEN
            self._last_state_change = time.monotonic()
        elif (
            self._state == CBState.CLOSED
            and self._consecutive_failures >= self.failure_threshold
        ):
            self._state = CBState.OPEN
            self._last_state_change = time.monotonic()

    def reset(self) -> None:
        """Manually reset to CLOSED state."""
        self._state = CBState.CLOSED
        self._consecutive_failures = 0
        self._consecutive_successes = 0
        self._last_state_change = time.monotonic()

    def stats(self) -> CircuitStats:
        return CircuitStats(
            name=self.name,
            state=self.state.value,
            failure_count=self._failure_count,
            success_count=self._success_count,
            consecutive_failures=self._consecutive_failures,
            last_failure_time=self._last_failure_time,
            total_calls=self._total_calls,
        )


# ===================================================================
# Agent Watchdog
# ===================================================================

@dataclass
class HealthReport:
    """Result of a health check."""

    healthy: bool
    checks: dict[str, bool] = field(default_factory=dict)
    recovery_action: str = ""


class AgentWatchdog:
    """Monitors agent state and triggers auto-recovery.

    Checks:
      1. State finiteness — detects NaN/Inf in state vectors
      2. Liveness — detects stuck agents (no tick for too long)
      3. Value range — detects state explosion (values outside bounds)
      4. Progress — detects oscillation (no convergence over window)

    Parameters
    ----------
    name : str
        Identifier for the monitored agent.
    max_silent_seconds : float
        Maximum time without a tick before declaring the agent stuck.
    value_bounds : tuple[float, float]
        Acceptable range for state vector values.
    stagnation_window : int
        Number of recent observations to check for progress.
    stagnation_threshold : float
        Minimum variance required across the stagnation window.
    """

    def __init__(
        self,
        name: str,
        max_silent_seconds: float = 30.0,
        value_bounds: tuple[float, float] = (-100.0, 100.0),
        stagnation_window: int = 10,
        stagnation_threshold: float = 1e-12,
    ) -> None:
        self.name = name
        self.max_silent_seconds = max_silent_seconds
        self.value_bounds = value_bounds
        self.stagnation_window = stagnation_window
        self.stagnation_threshold = stagnation_threshold

        self._last_tick_time = time.monotonic()
        self._recent_energies: list[float] = []
        self._recovery_count = 0

    def record_tick(self, energy: float | None = None) -> None:
        """Record that the agent has ticked (is alive)."""
        self._last_tick_time = time.monotonic()
        if energy is not None:
            self._recent_energies.append(energy)
            if len(self._recent_energies) > self.stagnation_window:
                self._recent_energies = self._recent_energies[-self.stagnation_window:]

    def check_health(self, state: np.ndarray | None = None) -> HealthReport:
        """Run all health checks and return a report.

        Parameters
        ----------
        state : ndarray, optional
            Current state vector to validate.
        """
        checks: dict[str, bool] = {}

        # 1. Liveness check
        elapsed = time.monotonic() - self._last_tick_time
        checks["liveness"] = elapsed < self.max_silent_seconds

        # 2. Finiteness check
        if state is not None:
            checks["finite"] = bool(np.all(np.isfinite(state)))
        else:
            checks["finite"] = True  # Can't check without state

        # 3. Value range check
        if state is not None:
            lo, hi = self.value_bounds
            checks["in_bounds"] = bool(np.all(state >= lo) and np.all(state <= hi))
        else:
            checks["in_bounds"] = True

        # 4. Progress check (no stagnation)
        if len(self._recent_energies) >= self.stagnation_window:
            window = self._recent_energies[-self.stagnation_window:]
            mean = sum(window) / len(window)
            variance = sum((x - mean) ** 2 for x in window) / len(window)
            checks["progressing"] = variance > self.stagnation_threshold
        else:
            checks["progressing"] = True  # Not enough data yet

        healthy = all(checks.values())
        recovery_action = ""
        if not healthy:
            failed = [k for k, v in checks.items() if not v]
            recovery_action = self._suggest_recovery(failed)

        return HealthReport(
            healthy=healthy,
            checks=checks,
            recovery_action=recovery_action,
        )

    def _suggest_recovery(self, failed_checks: list[str]) -> str:
        """Determine recovery action based on which checks failed."""
        if "finite" in failed_checks or "in_bounds" in failed_checks:
            return "reset_state"
        if "liveness" in failed_checks:
            return "restart"
        if "progressing" in failed_checks:
            return "perturb_state"
        return "reset_state"

    def execute_recovery(
        self,
        state: np.ndarray,
        report: HealthReport,
    ) -> np.ndarray:
        """Apply the suggested recovery action to the state vector.

        Returns the recovered state vector.
        """
        self._recovery_count += 1
        action = report.recovery_action

        if action == "reset_state":
            return np.zeros(state.shape)

        if action == "perturb_state":
            # Add small noise to escape stagnation
            noise = np.random.default_rng().normal(0, 0.01, size=state.shape)
            return np.clip(
                state + noise,
                self.value_bounds[0],
                self.value_bounds[1],
            )

        # Default: reset
        return np.zeros(state.shape)

    @property
    def recovery_count(self) -> int:
        return self._recovery_count
