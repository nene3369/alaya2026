"""Neuromorphic ASIC simulator — memristor crossbar arrays for FEP ODE.

Simulates analog hardware that natively executes FEP ODE:
  - MemristorCrossbar: O(1) analog matrix-vector product via KCL
  - NeuromorphicChip: tiled crossbar with leaky integrator
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from scipy import sparse

from lmm._compat import HAS_VECTORIZED_TANH


@dataclass
class MemristorState:
    """Memristor device state."""

    conductance: float = 0.0
    min_conductance: float = 1e-6
    max_conductance: float = 1e-3
    write_count: int = 0


@dataclass
class CrossbarStats:
    """Crossbar array statistics."""

    n_rows: int = 0
    n_cols: int = 0
    total_devices: int = 0
    active_devices: int = 0
    total_conductance: float = 0.0
    avg_conductance: float = 0.0
    max_conductance: float = 0.0
    total_write_count: int = 0
    energy_per_mac_pJ: float = 0.01


@dataclass
class ChipReport:
    """Chip-level execution report."""

    n_tiles: int = 0
    tile_size: tuple[int, int] = (0, 0)
    total_devices: int = 0
    active_devices: int = 0
    fep_steps_simulated: int = 0
    convergence_time_ns: float = 0.0
    energy_consumption_nJ: float = 0.0
    throughput_gops: float = 0.0
    duration_ms: float = 0.0


class MemristorCrossbar:
    """Memristor crossbar array — O(1) analog matrix-vector product.

    M x N crossbar with memristors at intersections.
    I_out[j] = sum_i G[i,j] * V_in[i]  (KCL)
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        *,
        G_min: float = 1e-6,
        G_max: float = 1e-3,
        noise_sigma: float = 0.0,
        quantization_bits: int = 0,
    ):
        self.rows = rows
        self.cols = cols
        self.G_min = G_min
        self.G_max = G_max
        self.noise_sigma = noise_sigma
        self.quantization_bits = quantization_bits

        self._G: dict[int, float] = {}
        self._write_counts: dict[int, int] = {}
        self._nonzero_idx: set[int] = set()
        self._rng = np.random.default_rng(42)

    # -- Programming API --------------------------------------------------

    def program_from_sparse(self, J: sparse.csr_matrix) -> None:
        """Program conductance matrix from sparse weight matrix."""
        self._G = {}
        self._nonzero_idx = set()

        coo = J.tocoo()
        for i in range(len(coo.data)):
            r, c, v = int(coo.row[i]), int(coo.col[i]), float(coo.data[i])
            if r < self.rows and c < self.cols:
                g = self._weight_to_conductance(v)
                idx = r * self.cols + c
                self._G[idx] = g
                self._write_counts[idx] = self._write_counts.get(idx, 0) + 1
                if abs(g) > 1e-15:
                    self._nonzero_idx.add(idx)

    def program_cell(self, row: int, col: int, weight: float) -> None:
        """Program a single cell."""
        if 0 <= row < self.rows and 0 <= col < self.cols:
            idx = row * self.cols + col
            g = self._weight_to_conductance(weight)
            self._G[idx] = g
            self._write_counts[idx] = self._write_counts.get(idx, 0) + 1
            if abs(g) > 1e-15:
                self._nonzero_idx.add(idx)
            else:
                self._nonzero_idx.discard(idx)
                self._G.pop(idx, None)

    # -- Analog compute API -----------------------------------------------

    def mac(self, V_in: np.ndarray) -> np.ndarray:
        """Analog multiply-accumulate: I_out[j] = sum_i G[i,j] * V_in[i]."""
        n_in = min(len(V_in), self.rows)
        I_out = [0.0] * self.cols
        cols = self.cols

        for flat_idx in self._nonzero_idx:
            i = flat_idx // cols
            if i >= n_in:
                continue
            j = flat_idx % cols
            I_out[j] += self._G[flat_idx] * float(V_in[i])

        if self.noise_sigma > 0:
            for j in range(self.cols):
                I_out[j] += self._rng.normal(0, self.noise_sigma)

        return np.array(I_out)

    def hebbian_update(
        self,
        pre: np.ndarray,
        post: np.ndarray,
        learning_rate: float,
    ) -> None:
        """Hebbian conductance update: dG[i,j] = eta * pre[i] * post[j]."""
        n_pre = min(len(pre), self.rows)
        n_post = min(len(post), self.cols)

        for i in range(n_pre):
            vi = float(pre[i])
            if abs(vi) < 1e-8:
                continue
            for j in range(n_post):
                vj = float(post[j])
                if abs(vj) < 1e-8:
                    continue
                delta_g = learning_rate * vi * vj
                idx = i * self.cols + j
                new_g = self._G.get(idx, 0.0) + delta_g
                if new_g > 0:
                    new_g = min(max(new_g, self.G_min), self.G_max)
                elif new_g < 0:
                    new_g = max(min(new_g, -self.G_min), -self.G_max)
                self._G[idx] = new_g
                self._write_counts[idx] = self._write_counts.get(idx, 0) + 1
                if abs(new_g) > 1e-15:
                    self._nonzero_idx.add(idx)
                else:
                    self._nonzero_idx.discard(idx)
                    self._G.pop(idx, None)

    # -- Sparse conversion ------------------------------------------------

    def to_sparse(self) -> sparse.csr_matrix:
        """Return conductance matrix as sparse weight matrix."""
        rows, cols, vals = [], [], []
        c = self.cols
        for flat_idx in self._nonzero_idx:
            g = self._G[flat_idx]
            w = self._conductance_to_weight(g)
            if abs(w) > 1e-10:
                rows.append(flat_idx // c)
                cols.append(flat_idx % c)
                vals.append(w)
        return sparse.csr_matrix(
            (vals, (rows, cols)), shape=(self.rows, self.cols),
        )

    # -- Statistics -------------------------------------------------------

    def stats(self) -> CrossbarStats:
        """Crossbar statistics."""
        active = 0
        total_g = 0.0
        max_g = 0.0

        for flat_idx in self._nonzero_idx:
            g = abs(self._G[flat_idx])
            if g > 1e-10:
                active += 1
                total_g += g
                if g > max_g:
                    max_g = g

        return CrossbarStats(
            n_rows=self.rows,
            n_cols=self.cols,
            total_devices=self.rows * self.cols,
            active_devices=active,
            total_conductance=total_g,
            avg_conductance=total_g / max(active, 1),
            max_conductance=max_g,
            total_write_count=sum(self._write_counts.values()),
        )

    # -- Internal ---------------------------------------------------------

    def _weight_to_conductance(self, weight: float) -> float:
        if abs(weight) < 1e-10:
            return 0.0
        g = weight * self.G_max
        if self.quantization_bits > 0:
            n_levels = 2 ** self.quantization_bits
            step = (self.G_max - self.G_min) / n_levels
            if g > 0:
                g = round((g - self.G_min) / step) * step + self.G_min
            elif g < 0:
                g = -(round((-g - self.G_min) / step) * step + self.G_min)
        return g

    def _conductance_to_weight(self, g: float) -> float:
        if abs(g) < 1e-10:
            return 0.0
        return g / self.G_max


class NeuromorphicChip:
    """Neuromorphic chip — tiled crossbar with FEP ODE execution."""

    def __init__(
        self,
        n_variables: int,
        *,
        tile_size: int = 64,
        G_min: float = 1e-6,
        G_max: float = 1e-3,
        noise_sigma: float = 0.0,
        quantization_bits: int = 0,
        rc_time_constant_ns: float = 10.0,
    ):
        self.n = n_variables
        self.tile_size = tile_size
        self.rc_time_constant_ns = rc_time_constant_ns
        self.n_tiles = math.ceil(n_variables / tile_size)

        self._crossbar = MemristorCrossbar(
            rows=n_variables, cols=n_variables,
            G_min=G_min, G_max=G_max,
            noise_sigma=noise_sigma,
            quantization_bits=quantization_bits,
        )
        self._V_mu = np.zeros(n_variables)
        self._x = np.zeros(n_variables)

    def program(self, J: sparse.csr_matrix) -> None:
        """Program J matrix onto chip."""
        self._crossbar.program_from_sparse(J)

    def run_fep(
        self,
        V_s: np.ndarray,
        *,
        G_prec: float = 8.0,
        tau_leak: float = 1.5,
        dt: float = 0.02,
        n_steps: int = 300,
        nirvana_threshold: float = 1e-4,
    ) -> ChipReport:
        """Execute FEP ODE on hardware simulation."""
        start = time.monotonic()
        n = min(len(V_s), self.n)
        self._V_mu = np.zeros(self.n)
        steps_used = 0

        for step in range(n_steps):
            steps_used = step + 1

            if HAS_VECTORIZED_TANH:
                g_V = np.tanh(self._V_mu[:n])
                g_prime = 1.0 - g_V * g_V
            else:
                g_list = [math.tanh(float(self._V_mu[i])) for i in range(n)]
                g_V = np.array(g_list)
                g_prime = 1.0 - g_V * g_V

            I_J = self._crossbar.mac(g_V)

            P_err = 0.0
            for i in range(n):
                eps = float(V_s[i]) + float(I_J[i]) - float(g_V[i])
                P_err += eps * eps
                leak = -float(self._V_mu[i]) / tau_leak
                drive = float(g_prime[i]) * G_prec * eps
                self._V_mu[i] = float(self._V_mu[i]) + (leak + drive) * dt
            P_err *= G_prec

            if P_err < nirvana_threshold:
                break

        if HAS_VECTORIZED_TANH:
            self._x = (np.tanh(self._V_mu[:n]) + 1.0) * 0.5
        else:
            self._x = np.array([
                (math.tanh(float(self._V_mu[i])) + 1.0) * 0.5
                for i in range(n)
            ])

        elapsed_ms = (time.monotonic() - start) * 1000
        convergence_ns = steps_used * self.rc_time_constant_ns
        energy_nJ = steps_used * n * n * 1e-14 * 1e9
        throughput_gops = (steps_used * n * n) / max(convergence_ns, 1e-9) * 1e-9

        cb_stats = self._crossbar.stats()
        return ChipReport(
            n_tiles=self.n_tiles,
            tile_size=(self.tile_size, self.tile_size),
            total_devices=cb_stats.total_devices,
            active_devices=cb_stats.active_devices,
            fep_steps_simulated=steps_used,
            convergence_time_ns=convergence_ns,
            energy_consumption_nJ=energy_nJ,
            throughput_gops=throughput_gops,
            duration_ms=elapsed_ms,
        )

    def read_membrane_potentials(self) -> np.ndarray:
        return self._V_mu.copy()

    def read_activations(self) -> np.ndarray:
        return self._x.copy()

    def hebbian_update(
        self, pre: np.ndarray, post: np.ndarray, learning_rate: float,
    ) -> None:
        self._crossbar.hebbian_update(pre, post, learning_rate)

    def stats(self) -> CrossbarStats:
        return self._crossbar.stats()
