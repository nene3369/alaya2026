"""Tests for lmm.dharma.fep — FEP KCL ODE solver."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from lmm.dharma.fep import solve_fep_kcl, solve_fep_kcl_analog


class TestSolveFepKcl:
    def test_basic(self):
        rng = np.random.RandomState(42)
        n = 10
        h = rng.randn(n)
        dense = rng.rand(n, n) * 0.3
        dense = (dense + dense.T) / 2
        np.fill_diagonal(dense, 0)
        J = sparse.csr_matrix(dense)
        V_mu, x, steps, power = solve_fep_kcl(h=h, J=J, k=3, n=n)
        assert V_mu.shape == (n,)
        assert x.shape == (n,)
        assert steps > 0

    def test_small(self):
        n = 2
        h = np.array([-1.0, -2.0])
        J = sparse.csr_matrix(np.array([[0.0, 0.5], [0.5, 0.0]]))
        V_mu, x, steps, power = solve_fep_kcl(h=h, J=J, k=1, n=n)
        assert V_mu.shape == (n,)

    def test_identity(self):
        n = 5
        h = -np.ones(n)
        J = sparse.csr_matrix(np.zeros((n, n)))
        V_mu, x, steps, power = solve_fep_kcl(h=h, J=J, k=3, n=n)
        assert x.shape == (n,)


class TestSolveFepKclAnalog:
    def test_basic(self):
        rng = np.random.RandomState(42)
        n = 10
        V_s = rng.rand(n)
        dense = rng.rand(n, n) * 0.3
        dense = (dense + dense.T) / 2
        np.fill_diagonal(dense, 0)
        J_dynamic = sparse.csr_matrix(dense)
        V_mu, x, steps, power = solve_fep_kcl_analog(V_s=V_s, J_dynamic=J_dynamic, n=n)
        assert V_mu.shape == (n,)
        assert x.shape == (n,)

    def test_small(self):
        n = 2
        V_s = np.array([1.0, 2.0])
        J_dynamic = sparse.csr_matrix(np.array([[0.0, 0.5], [0.5, 0.0]]))
        V_mu, x, steps, power = solve_fep_kcl_analog(V_s=V_s, J_dynamic=J_dynamic, n=n)
        assert V_mu.shape == (n,)

    def test_convergence(self):
        rng = np.random.RandomState(42)
        n = 8
        V_s = rng.rand(n)
        J_dynamic = sparse.csr_matrix(np.zeros((n, n)))
        V_mu, x, steps, power = solve_fep_kcl_analog(V_s=V_s, J_dynamic=J_dynamic, n=n, max_steps=500)
        assert len(power) > 0
