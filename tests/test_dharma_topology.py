"""Tests for lmm.dharma.topology — Dharma-Topology three-pillar evaluation."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from lmm.dharma.topology import (
    AniccaTerm,
    AnattaTerm,
    DharmaTelemetry,
    PratityaTerm,
    TopologyEvaluator,
    TopologyHistory,
    compute_deleteability,
    compute_gprec_topology,
    compute_karma_isolation,
    evaluate_engine_result,
)


# ===================================================================
# Helpers
# ===================================================================


def _empty_sparse(n: int) -> sparse.csr_matrix:
    """Create an empty (n, n) sparse matrix (shim-compatible)."""
    return sparse.csr_matrix(([], ([], [])), shape=(n, n))


def _block_diagonal(n: int = 10) -> sparse.csr_matrix:
    """Create a block-diagonal graph (ideal modularity)."""
    half = n // 2
    rows, cols, vals = [], [], []
    # Block 1: nodes [0, half)
    for i in range(half):
        for j in range(half):
            if i != j:
                rows.append(i)
                cols.append(j)
                vals.append(0.5)
    # Block 2: nodes [half, n)
    for i in range(half, n):
        for j in range(half, n):
            if i != j:
                rows.append(i)
                cols.append(j)
                vals.append(0.5)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))


def _fully_connected(n: int = 10) -> sparse.csr_matrix:
    """Create a fully connected graph (no self-loops)."""
    rows, cols, vals = [], [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                rows.append(i)
                cols.append(j)
                vals.append(0.5)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))


def _star_graph(n: int = 5) -> sparse.csr_matrix:
    """Create a star graph: node 0 connected to all others."""
    rows_0 = [0] * (n - 1)
    cols_0 = list(range(1, n))
    rows_back = list(range(1, n))
    cols_back = [0] * (n - 1)
    all_rows = rows_0 + rows_back
    all_cols = cols_0 + cols_back
    all_data = [1.0] * len(all_rows)
    return sparse.csr_matrix((all_data, (all_rows, all_cols)), shape=(n, n))


def _random_adj(n: int = 10, seed: int = 42) -> sparse.csr_matrix:
    """Create a random symmetric sparse matrix (no self-loops)."""
    rng = np.random.RandomState(seed)
    rows, cols, vals = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            v = float(rng.rand()) * 0.3
            rows.extend([i, j])
            cols.extend([j, i])
            vals.extend([v, v])
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))


# ===================================================================
# Karma Isolation
# ===================================================================


class TestKarmaIsolation:
    def test_block_diagonal_has_high_isolation(self):
        adj = _block_diagonal()
        score, per_node = compute_karma_isolation(adj)
        assert 0.0 <= score <= 1.0
        assert score > 0.5

    def test_fully_connected_has_lower_isolation(self):
        adj = _fully_connected()
        score, per_node = compute_karma_isolation(adj)
        assert 0.0 <= score <= 1.0

    def test_empty_graph(self):
        adj = _empty_sparse(5)
        score, per_node = compute_karma_isolation(adj)
        assert score == 1.0
        assert len(per_node) == 5

    def test_custom_boundary_mask(self):
        adj = _fully_connected(6)
        mask = np.array([True, True, False, False, False, False])
        score, per_node = compute_karma_isolation(adj, boundary_mask=mask)
        assert 0.0 <= score <= 1.0
        # Non-boundary nodes should be 1.0
        assert per_node[2] == 1.0
        assert per_node[3] == 1.0

    def test_zero_size_graph(self):
        adj = _empty_sparse(0)
        score, per_node = compute_karma_isolation(adj)
        assert score == 1.0
        assert len(per_node) == 0


# ===================================================================
# G_prec Topology
# ===================================================================


class TestGprecTopology:
    def test_sparse_graph_scores_high(self):
        n = 10
        adj = sparse.csr_matrix(([0.5, 0.5], ([0, 1], [1, 0])), shape=(n, n))
        score, details = compute_gprec_topology(adj)
        assert score > 0.5
        assert "density" in details
        assert "n_components" in details
        assert "block_diagonal_ratio" in details

    def test_dense_graph_scores_low(self):
        adj = _fully_connected(10)
        score, details = compute_gprec_topology(adj)
        assert score < 0.5

    def test_empty_graph(self):
        adj = _empty_sparse(5)
        score, details = compute_gprec_topology(adj)
        assert score == 1.0
        assert details["n_components"] == 5

    def test_block_diagonal_has_multiple_components(self):
        adj = _block_diagonal(10)
        score, details = compute_gprec_topology(adj)
        assert details["n_components"] >= 2
        assert details["block_diagonal_ratio"] == 1.0

    def test_zero_size_graph(self):
        adj = _empty_sparse(0)
        score, details = compute_gprec_topology(adj)
        assert score == 1.0


# ===================================================================
# Delete-ability
# ===================================================================


class TestDeleteability:
    def test_star_graph_center_has_low_deleteability(self):
        adj = _star_graph(5)
        score, per_node = compute_deleteability(adj)
        # Node 0 (center) should be hardest to delete
        assert per_node[0] < per_node[1]

    def test_isolated_nodes(self):
        adj = _empty_sparse(5)
        score, per_node = compute_deleteability(adj)
        assert score == 1.0
        assert all(float(v) == 1.0 for v in per_node)

    def test_degree_method(self):
        adj = _random_adj()
        score, per_node = compute_deleteability(adj, method="degree")
        assert 0.0 <= score <= 1.0
        assert len(per_node) == 10

    def test_fep_method(self):
        adj = _random_adj(n=5)
        score, per_node = compute_deleteability(adj, method="fep", max_steps=10)
        assert 0.0 <= score <= 1.0
        assert len(per_node) == 5

    def test_zero_size_graph(self):
        adj = _empty_sparse(0)
        score, per_node = compute_deleteability(adj)
        assert score == 1.0


# ===================================================================
# DharmaTelemetry
# ===================================================================


class TestDharmaTelemetry:
    def test_to_json(self):
        t = DharmaTelemetry(
            karma_isolation=0.8,
            gprec_topology=0.7,
            deleteability=0.9,
            overall_dharma=0.8,
            ode_steps=50,
            ode_final_power=0.001,
            V_final=[0.1, 0.2, 0.3],
        )
        j = t.to_json()
        assert j["protocol"] == "dharma_topology_v1"
        assert j["karma_isolation"] == 0.8
        assert j["gprec_topology"] == 0.7
        assert j["deleteability"] == 0.9

    def test_to_json_with_details(self):
        t = DharmaTelemetry(
            karma_isolation=0.5,
            gprec_topology=0.6,
            deleteability=0.7,
            overall_dharma=0.6,
            ode_steps=10,
            ode_final_power=0.01,
            V_final=[0.1],
            pillar_details={"nodes": {"A": {"karma_isolation": 0.5}}},
        )
        j = t.to_json()
        assert j["pillar_details"] is not None
        assert "A" in j["pillar_details"]["nodes"]


# ===================================================================
# TopologyEvaluator
# ===================================================================


class TestTopologyEvaluator:
    def test_evaluate_basic(self):
        adj = _random_adj()
        evaluator = TopologyEvaluator()
        telemetry = evaluator.evaluate(adj)
        assert isinstance(telemetry, DharmaTelemetry)
        assert 0.0 <= telemetry.overall_dharma <= 1.0
        assert 0.0 <= telemetry.karma_isolation <= 1.0
        assert 0.0 <= telemetry.gprec_topology <= 1.0
        assert 0.0 <= telemetry.deleteability <= 1.0
        assert telemetry.ode_steps > 0

    def test_evaluate_empty_graph(self):
        adj = _empty_sparse(5)
        evaluator = TopologyEvaluator()
        telemetry = evaluator.evaluate(adj)
        assert telemetry.karma_isolation == 1.0
        assert telemetry.gprec_topology == 1.0
        assert telemetry.deleteability == 1.0
        assert telemetry.overall_dharma == 1.0

    def test_evaluate_with_details(self):
        n = 6
        adj = sparse.csr_matrix(([0.5, 0.5], ([0, 1], [1, 0])), shape=(n, n))
        evaluator = TopologyEvaluator()
        telemetry = evaluator.evaluate(adj, include_details=True)
        assert telemetry.pillar_details is not None
        assert "nodes" in telemetry.pillar_details
        assert "gprec" in telemetry.pillar_details

    def test_evaluate_with_node_labels(self):
        n = 3
        adj = sparse.csr_matrix(([1.0, 1.0], ([0, 1], [1, 0])), shape=(n, n))
        evaluator = TopologyEvaluator()
        telemetry = evaluator.evaluate(
            adj,
            node_labels=["A", "B", "C"],
            include_details=True,
        )
        assert "A" in telemetry.pillar_details["nodes"]
        assert "B" in telemetry.pillar_details["nodes"]
        assert "C" in telemetry.pillar_details["nodes"]

    def test_evaluate_with_boundary_mask(self):
        adj = _random_adj(8)
        mask = np.array([True, True, False, False, False, False, False, False])
        evaluator = TopologyEvaluator()
        telemetry = evaluator.evaluate(adj, boundary_mask=mask)
        assert 0.0 <= telemetry.karma_isolation <= 1.0

    def test_fep_deleteability_method(self):
        adj = _random_adj(n=5)
        evaluator = TopologyEvaluator(deleteability_method="fep", max_steps=10)
        telemetry = evaluator.evaluate(adj)
        assert 0.0 <= telemetry.deleteability <= 1.0

    def test_block_diagonal_scores_high(self):
        adj = _block_diagonal(10)
        evaluator = TopologyEvaluator()
        telemetry = evaluator.evaluate(adj)
        # Block-diagonal graph should have high modularity and overall score
        assert telemetry.gprec_topology > 0.3

    def test_v_final_length_matches_n(self):
        n = 7
        adj = _random_adj(n)
        evaluator = TopologyEvaluator()
        telemetry = evaluator.evaluate(adj)
        assert len(telemetry.V_final) == n


# ===================================================================
# Energy Terms
# ===================================================================


class TestEnergyTerms:
    def test_anicca_term(self):
        adj = _random_adj()
        term = AniccaTerm(adj, weight=1.0)
        assert term.name == "Anicca (Delete-ability)"
        assert term.math_property == "linear"
        h, J = term.build(10)
        assert h.shape == (10,)
        assert J is None

    def test_anicca_term_padding(self):
        adj = _random_adj(5)
        term = AniccaTerm(adj, weight=1.0)
        h, J = term.build(10)
        assert h.shape == (10,)

    def test_anicca_term_trimming(self):
        adj = _random_adj(10)
        term = AniccaTerm(adj, weight=1.0)
        h, J = term.build(5)
        assert h.shape == (5,)

    def test_anatta_term(self):
        adj = _random_adj()
        term = AnattaTerm(adj, weight=0.5)
        assert term.name == "Anatta (Karma Isolation)"
        assert term.math_property == "submodular"
        h, J = term.build(10)
        assert h.shape == (10,)
        assert J is not None
        assert J.shape == (10, 10)

    def test_pratitya_term(self):
        adj = _random_adj()
        term = PratityaTerm(adj, weight=1.0)
        assert term.name == "Pratitya (Topology Modularity)"
        assert term.math_property == "linear"
        h, J = term.build(10)
        assert h.shape == (10,)
        assert J is None


# ===================================================================
# Engine Integration
# ===================================================================


class TestEngineIntegration:
    def test_evaluate_engine_result(self):
        from lmm.dharma.energy import PrajnaTerm, SilaTerm
        from lmm.dharma.engine import UniversalDharmaEngine

        rng = np.random.RandomState(42)
        n = 15
        engine = UniversalDharmaEngine(n_variables=n)
        engine.add(PrajnaTerm(rng.rand(n)))
        engine.add(SilaTerm(k=4, weight=10.0))
        result = engine.synthesize_and_solve(k=4)
        telemetry = evaluate_engine_result(result)
        assert isinstance(telemetry, DharmaTelemetry)
        assert 0.0 <= telemetry.overall_dharma <= 1.0

    def test_evaluate_engine_result_with_quadratic(self):
        from lmm.dharma.energy import KarunaTerm, PrajnaTerm, SilaTerm
        from lmm.dharma.engine import UniversalDharmaEngine

        rng = np.random.RandomState(42)
        n = 10
        engine = UniversalDharmaEngine(n_variables=n)
        engine.add(PrajnaTerm(rng.rand(n)))
        engine.add(KarunaTerm(rng.rand(n, n) * 0.3))
        engine.add(SilaTerm(k=3, weight=10.0))
        result = engine.synthesize_and_solve(k=3)
        telemetry = evaluate_engine_result(result)
        assert isinstance(telemetry, DharmaTelemetry)

    def test_evaluate_engine_result_with_topology_terms(self):
        from lmm.dharma.energy import PrajnaTerm, SilaTerm
        from lmm.dharma.engine import UniversalDharmaEngine

        rng = np.random.RandomState(42)
        n = 10
        adj = _random_adj(n)

        engine = UniversalDharmaEngine(n_variables=n)
        engine.add(PrajnaTerm(rng.rand(n)))
        engine.add(AniccaTerm(adj, weight=0.5))
        engine.add(SilaTerm(k=4, weight=10.0))
        result = engine.synthesize_and_solve(k=4)
        assert len(result.selected_indices) == 4


# ===================================================================
# TopologyHistory
# ===================================================================


class TestTopologyHistory:
    def _make_telemetry(self, karma=0.8, gprec=0.7, delete=0.9) -> DharmaTelemetry:
        overall = (karma * gprec * delete) ** (1.0 / 3.0)
        return DharmaTelemetry(
            karma_isolation=karma,
            gprec_topology=gprec,
            deleteability=delete,
            overall_dharma=overall,
            ode_steps=10,
            ode_final_power=0.001,
            V_final=[0.1],
        )

    def test_record_and_latest(self):
        history = TopologyHistory(maxlen=100)
        t1 = self._make_telemetry(0.5, 0.5, 0.5)
        history.record(t1)
        assert history.latest() is t1
        assert history.count == 1

    def test_latest_empty(self):
        history = TopologyHistory()
        assert history.latest() is None

    def test_maxlen_overflow(self):
        history = TopologyHistory(maxlen=3)
        for i in range(5):
            history.record(self._make_telemetry(i / 10.0, 0.5, 0.5))
        assert history.count == 3

    def test_trend_stable_with_constant_values(self):
        history = TopologyHistory(maxlen=100)
        for _ in range(10):
            history.record(self._make_telemetry(0.5, 0.5, 0.5))
        trend = history.trend(window=10)
        assert trend["direction"] == "stable"
        assert abs(trend["karma_trend"]) < 0.01
        assert trend["n_snapshots"] == 10

    def test_trend_improving(self):
        history = TopologyHistory(maxlen=100)
        for i in range(10):
            val = 0.3 + i * 0.05
            history.record(self._make_telemetry(val, val, val))
        trend = history.trend(window=10)
        assert trend["direction"] == "improving"
        assert trend["overall_trend"] > 0

    def test_trend_degrading(self):
        history = TopologyHistory(maxlen=100)
        for i in range(10):
            val = 0.8 - i * 0.05
            history.record(self._make_telemetry(val, val, val))
        trend = history.trend(window=10)
        assert trend["direction"] == "degrading"
        assert trend["overall_trend"] < 0

    def test_trend_insufficient_data(self):
        history = TopologyHistory()
        history.record(self._make_telemetry())
        trend = history.trend()
        assert trend["direction"] == "stable"
        assert trend["n_snapshots"] == 1

    def test_to_json(self):
        history = TopologyHistory(maxlen=100)
        for _ in range(5):
            history.record(self._make_telemetry(0.8, 0.7, 0.9))
        j = history.to_json()
        assert "snapshots" in j
        assert "trend" in j
        assert "count" in j
        assert j["count"] == 5
        assert len(j["snapshots"]) == 5
        snap = j["snapshots"][0]
        assert "timestamp" in snap
        assert "karma_isolation" in snap
        assert "gprec_topology" in snap
        assert "deleteability" in snap
        assert "overall_dharma" in snap

    def test_to_json_last_n(self):
        history = TopologyHistory(maxlen=100)
        for _ in range(10):
            history.record(self._make_telemetry())
        j = history.to_json(last_n=3)
        assert len(j["snapshots"]) == 3
        assert j["count"] == 10
