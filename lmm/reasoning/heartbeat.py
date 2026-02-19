"""HeartbeatDaemon — continuous-time FEP state evolution between interactions.

Runs an async loop that evolves a 4D state vector [Love, Logic, Fear, Creation]
using solve_fep_kcl_analog at a configurable tick rate, injecting hardware entropy
from PinealGland.harvest_entropy().

The evolved state provides a warm initial_state for per-request reasoning,
bridging the gap between discrete LLM calls and continuous dynamical systems.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import struct
import time
from dataclasses import dataclass

import numpy as np
from scipy import sparse

from lmm.dharma.fep import solve_fep_kcl_analog
from lmm.reasoning.recovery import AgentWatchdog

logger = logging.getLogger(__name__)


@dataclass
class HeartbeatTelemetry:
    """Snapshot of heartbeat state for telemetry."""

    v_state: list[float]
    karuna_weight: float
    metta_weight: float
    tick_count: int
    idle_seconds: float
    cv: float
    last_sleep_report: dict | None = None


class HeartbeatDaemon:
    """Continuous-time FEP state evolution between interactions.

    Each tick:
      1. Harvest hardware entropy (os.urandom via PinealGland pattern)
      2. Run one step of FEP KCL ODE via solve_fep_kcl_analog(max_steps=1)
      3. Update karuna/metta weights via Madhyamaka CV tracking
      4. Detect idle periods and trigger SleepConsolidation

    Parameters
    ----------
    n_dims : int
        Dimensionality of state vector (default 4: Love, Logic, Fear, Creation).
    dt : float
        Base tick interval in seconds.
    r_leak : float
        FEP leak time constant (tau_leak).
    c_mem : float
        Membrane capacitance (not directly used by solve_fep_kcl_analog,
        but scales the entropy injection).
    entropy_scale : float
        Scale factor for hardware entropy injection.
    target_cv : float
        Madhyamaka target coefficient of variation.
    idle_sleep_threshold : float
        Seconds of inactivity before triggering sleep consolidation.
    deterministic : bool
        If True, run() returns after one tick (for testing).
    """

    def __init__(
        self,
        n_dims: int = 4,
        dt: float = 0.1,
        r_leak: float = 10.0,
        c_mem: float = 1.0,
        entropy_scale: float = 0.01,
        target_cv: float = 0.5,
        idle_sleep_threshold: float = 60.0,
        deterministic: bool = False,
    ):
        if n_dims < 1:
            raise ValueError(f"n_dims must be >= 1, got {n_dims}")
        if n_dims > 1024:
            raise ValueError(
                f"n_dims={n_dims} exceeds maximum 1024 for heartbeat daemon"
            )
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        self.n_dims = n_dims
        self.dt_base = dt
        self._dt = dt
        self.r_leak = r_leak
        self.c_mem = c_mem
        self.entropy_scale = entropy_scale
        self.target_cv = target_cv
        self.idle_sleep_threshold = idle_sleep_threshold
        self.deterministic = deterministic

        # State
        self._V = np.zeros(n_dims)
        self._karuna_weight = 1.0
        self._metta_weight = 1.0
        self._tick_count = 0
        self._last_interaction_time = time.monotonic()
        self._idle_consolidated = False
        self._last_sleep_report: dict | None = None

        # Control
        self._stop_event = asyncio.Event()
        self._lock = asyncio.Lock()

        # Sleep integration (set externally via attach_sleep)
        self._sleep_consolidation = None

        # Build a minimal J matrix for solve_fep_kcl_analog
        # Diagonal-free sparse matrix for n_dims nodes
        self._J = sparse.lil_matrix((n_dims, n_dims)).tocsr()

        # Watchdog for auto-recovery
        self._watchdog = AgentWatchdog(
            name="heartbeat",
            max_silent_seconds=30.0,
            value_bounds=(-10.0, 10.0),
        )

    def attach_sleep(self, sleep_consolidation) -> None:
        """Attach a SleepConsolidation instance for idle-triggered consolidation."""
        self._sleep_consolidation = sleep_consolidation

    def _harvest_entropy(self, n_bytes: int = 32) -> np.ndarray:
        """Harvest physical entropy from OS entropy pool.

        Mirrors PinealGland.harvest_entropy() pattern without requiring
        BaseReasoner inheritance.
        """
        try:
            raw = os.urandom(n_bytes)
        except NotImplementedError:
            raw = hashlib.sha256(
                struct.pack("d", time.monotonic())
            ).digest()[:n_bytes]

        values = []
        for b in raw:
            values.append((b / 127.5) - 1.0)

        return np.array(values)

    TICK_TIMEOUT: float = 2.0  # max seconds per tick

    async def tick(self) -> HeartbeatTelemetry:
        """Execute a single tick of the heartbeat and return telemetry.

        This is the core method. The run() loop calls this repeatedly.
        Exposed publicly for deterministic testing.
        """
        try:
            return await asyncio.wait_for(
                self._tick_inner(), timeout=self.TICK_TIMEOUT,
            )
        except (asyncio.TimeoutError, TimeoutError):
            # Tick timed out — return stale snapshot rather than blocking
            return self.snapshot()

    async def _tick_inner(self) -> HeartbeatTelemetry:
        """Inner tick logic, separated for timeout wrapping."""
        async with self._lock:
            # 1. Harvest hardware entropy
            entropy_raw = self._harvest_entropy(self.n_dims * 4)
            entropy_vec = np.zeros(self.n_dims)
            for i in range(min(len(entropy_raw), self.n_dims)):
                entropy_vec[i] = float(entropy_raw[i])
            entropy_vec *= self.entropy_scale

            # 2. Construct sensory input: current state perturbation + entropy
            # When no user input, V_s = entropy only (self-driven evolution)
            V_s = entropy_vec

            # 3. Run one FEP KCL ODE step
            V_mu, x_final, steps_used, power = solve_fep_kcl_analog(
                V_s=V_s,
                J_dynamic=self._J,
                n=self.n_dims,
                G_prec_base=5.0,
                tau_leak=self.r_leak,
                dt=0.01,
                max_steps=1,
                nirvana_threshold=1e-4,
                initial_state=self._V.copy(),
            )

            # 4. Update state with numerical stability
            self._V = V_mu
            self._V = np.clip(self._V, -10.0, 10.0)
            if not np.all(np.isfinite(self._V)):
                self._V = np.zeros(self.n_dims)

            # 5. Madhyamaka CV tracking: evolve karuna/metta weights
            abs_mean = float(np.abs(self._V).mean())
            if abs_mean > 1e-8:
                cv = float(self._V.std() / abs_mean)
            else:
                cv = 0.0

            cv_error = cv - self.target_cv
            self._karuna_weight *= np.exp(-0.01 * cv_error)
            self._metta_weight *= np.exp(0.01 * cv_error)
            self._karuna_weight = float(np.clip(self._karuna_weight, 0.1, 10.0))
            self._metta_weight = float(np.clip(self._metta_weight, 0.1, 10.0))

            # 6. Adaptive dt: slow down during idle to save CPU
            idle_seconds = time.monotonic() - self._last_interaction_time
            if idle_seconds > 10.0:
                self._dt = min(self._dt * 1.05, 5.0)
            else:
                self._dt = self.dt_base

            self._tick_count += 1

            # 7. Check idle sleep trigger
            if (
                idle_seconds > self.idle_sleep_threshold
                and not self._idle_consolidated
                and self._sleep_consolidation is not None
            ):
                try:
                    report = await asyncio.to_thread(
                        self._sleep_consolidation.consolidate
                    )
                    self._idle_consolidated = True
                    self._last_sleep_report = {
                        "n_consolidated": report.n_consolidated,
                        "n_pruned": report.n_pruned,
                        "replay_cycles": report.replay_cycles,
                        "pre_strength": report.pre_strength,
                        "post_strength": report.post_strength,
                    }
                except (ValueError, RuntimeError, TypeError):
                    pass

            return HeartbeatTelemetry(
                v_state=[float(v) for v in self._V],
                karuna_weight=self._karuna_weight,
                metta_weight=self._metta_weight,
                tick_count=self._tick_count,
                idle_seconds=idle_seconds,
                cv=cv,
                last_sleep_report=self._last_sleep_report,
            )

    async def run(self) -> None:
        """Main heartbeat loop. Launch via asyncio.create_task(daemon.run())."""
        while not self._stop_event.is_set():
            try:
                telemetry = await self.tick()
                self._watchdog.record_tick(energy=telemetry.cv)

                # Periodic health check (every 10 ticks)
                if self._tick_count % 10 == 0:
                    report = self._watchdog.check_health(self._V)
                    if not report.healthy:
                        self._V = self._watchdog.execute_recovery(
                            self._V, report,
                        )
            except asyncio.CancelledError:
                break
            except Exception:
                logger.debug("heartbeat tick error", exc_info=True)
            if self.deterministic:
                break
            await asyncio.sleep(self._dt)

    def inject_sensory(self, v_sensory: np.ndarray) -> None:
        """Inject a sensory event from a user interaction.

        This forces a large state update via FEP dynamics and resets
        the idle timer.
        """
        v_s = np.asarray(v_sensory).flatten()[: self.n_dims]
        if len(v_s) < self.n_dims:
            padded = np.zeros(self.n_dims)
            padded[: len(v_s)] = v_s
            v_s = padded

        V_mu, x_final, steps_used, power = solve_fep_kcl_analog(
            V_s=v_s,
            J_dynamic=self._J,
            n=self.n_dims,
            G_prec_base=5.0,
            tau_leak=self.r_leak,
            dt=0.01,
            max_steps=10,
            nirvana_threshold=1e-4,
            initial_state=self._V.copy(),
        )

        self._V = np.clip(V_mu, -10.0, 10.0)
        if not np.all(np.isfinite(self._V)):
            self._V = np.zeros(self.n_dims)

        self._last_interaction_time = time.monotonic()
        self._idle_consolidated = False
        self._dt = self.dt_base

    def stop(self) -> None:
        """Signal the heartbeat loop to stop."""
        self._stop_event.set()

    @property
    def state(self) -> np.ndarray:
        """Current state vector (copy for thread safety)."""
        return self._V.copy()

    @property
    def karuna_weight(self) -> float:
        return self._karuna_weight

    @property
    def metta_weight(self) -> float:
        return self._metta_weight

    @property
    def cv(self) -> float:
        abs_mean = float(np.abs(self._V).mean())
        if abs_mean > 1e-8:
            return float(self._V.std() / abs_mean)
        return 0.0

    @property
    def watchdog(self) -> AgentWatchdog:
        """Access the health watchdog for external monitoring."""
        return self._watchdog

    def snapshot(self) -> HeartbeatTelemetry:
        """Return a telemetry snapshot without ticking."""
        return HeartbeatTelemetry(
            v_state=[float(v) for v in self._V],
            karuna_weight=self._karuna_weight,
            metta_weight=self._metta_weight,
            tick_count=self._tick_count,
            idle_seconds=time.monotonic() - self._last_interaction_time,
            cv=self.cv,
            last_sleep_report=self._last_sleep_report,
        )
