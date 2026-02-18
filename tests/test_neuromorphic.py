"""Tests for lmm.dharma.neuromorphic — MemristorCrossbar + NeuromorphicChip."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from lmm.dharma.neuromorphic import (
    ChipReport,
    CrossbarStats,
    MemristorCrossbar,
    NeuromorphicChip,
)


class TestMemristorCrossbar:
    def test_create(self):
        xbar = MemristorCrossbar(rows=8, cols=8)
        assert xbar.rows == 8
        assert xbar.cols == 8

    def test_program_cell(self):
        xbar = MemristorCrossbar(rows=4, cols=4)
        xbar.program_cell(0, 1, 0.5)
        xbar.program_cell(2, 3, -0.3)
        stats = xbar.stats()
        assert isinstance(stats, CrossbarStats)
        assert stats.active_devices >= 2

    def test_program_from_sparse(self):
        xbar = MemristorCrossbar(rows=4, cols=4)
        dense = np.array([[0, 0.5, 0, 0], [0.5, 0, 0.3, 0], [0, 0.3, 0, 0], [0, 0, 0, 0]])
        J = sparse.csr_matrix(dense)
        xbar.program_from_sparse(J)
        stats = xbar.stats()
        assert stats.active_devices > 0

    def test_mac(self):
        xbar = MemristorCrossbar(rows=4, cols=4)
        for i in range(4):
            for j in range(4):
                xbar.program_cell(i, j, 0.1 * (i + j))
        v_in = np.array([1.0, 0.5, 0.5, 1.0])
        result = xbar.mac(v_in)
        assert result.shape == (4,)

    def test_hebbian_update(self):
        xbar = MemristorCrossbar(rows=4, cols=4)
        pre = np.array([1.0, 0.0, 1.0, 0.0])
        post = np.array([0.0, 1.0, 1.0, 0.0])
        xbar.hebbian_update(pre, post, learning_rate=0.1)
        stats = xbar.stats()
        assert stats.active_devices > 0


class TestNeuromorphicChip:
    def test_create(self):
        chip = NeuromorphicChip(n_variables=16, tile_size=8)
        assert chip is not None

    def test_run_fep(self):
        rng = np.random.RandomState(42)
        n = 8
        dense = rng.rand(n, n) * 0.3
        dense = (dense + dense.T) / 2
        np.fill_diagonal(dense, 0)
        J = sparse.csr_matrix(dense)

        chip = NeuromorphicChip(n_variables=n, tile_size=8)
        chip.program(J)
        V_s = rng.rand(n)
        report = chip.run_fep(V_s, n_steps=50)
        assert isinstance(report, ChipReport)
        assert report.fep_steps_simulated > 0

    def test_read_activations(self):
        n = 8
        chip = NeuromorphicChip(n_variables=n, tile_size=8)
        V_s = np.ones(n) * 0.5
        chip.run_fep(V_s, n_steps=10)
        x = chip.read_activations()
        assert x.shape == (n,)
