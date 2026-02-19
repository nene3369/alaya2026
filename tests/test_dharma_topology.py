"""Tests for lmm.dharma.topology — Dharma-Topology three-pillar evaluation."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from lmm.dharma.topology import (
    AniccaTerm,
    AnattaTerm,
    DharmaTelemetry,
    PratityaTerm,
    TopologyDriftDetector,
    TopologyEvaluator,
    TopologyHistory,
    compute_deleteability,
    compute_gprec_topology,
    compute_karma_isolation,
    evaluate_engine_result,
)
from lmm.dharma.topology_ast import (
    ChangeImpactAnalyzer,
    ChangeImpactReport,
    ModuleHealthReport,
    compute_module_health,
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

    def test_save_and_load(self):
        import tempfile
        from pathlib import Path

        history = TopologyHistory(maxlen=100)
        for i in range(5):
            history.record(self._make_telemetry(0.5 + i * 0.05, 0.6, 0.7))

        tmpfile = Path(tempfile.mkdtemp()) / "history.json"
        history.save(tmpfile)
        assert tmpfile.exists()

        history2 = TopologyHistory(maxlen=100)
        loaded = history2.load(tmpfile)
        assert loaded == 5
        assert history2.count == 5
        latest = history2.latest()
        assert latest is not None
        assert abs(latest.karma_isolation - 0.7) < 0.01

    def test_load_nonexistent(self):
        history = TopologyHistory()
        loaded = history.load("/tmp/nonexistent_file_xyz.json")
        assert loaded == 0

    def test_load_corrupt_file(self):
        import tempfile
        from pathlib import Path

        tmpfile = Path(tempfile.mkdtemp()) / "bad.json"
        tmpfile.write_text("not valid json{{{")
        history = TopologyHistory()
        loaded = history.load(tmpfile)
        assert loaded == 0


# ===================================================================
# TopologyDriftDetector
# ===================================================================


class TestTopologyDriftDetector:
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

    def test_no_alert_on_first_check(self):
        detector = TopologyDriftDetector()
        alerts = detector.check(self._make_telemetry())
        assert alerts == []

    def test_no_alert_on_small_change(self):
        detector = TopologyDriftDetector(warning_threshold=0.05)
        detector.check(self._make_telemetry(0.8, 0.7, 0.9))
        alerts = detector.check(self._make_telemetry(0.81, 0.71, 0.91))
        assert len(alerts) == 0

    def test_warning_alert_on_moderate_change(self):
        detector = TopologyDriftDetector(warning_threshold=0.05, cooldown=0)
        detector.check(self._make_telemetry(0.8, 0.7, 0.9))
        alerts = detector.check(self._make_telemetry(0.7, 0.7, 0.9))
        # karma changed by 0.1, which > 0.05
        assert any(a.pillar == "karma_isolation" for a in alerts)
        assert any(a.severity == "warning" for a in alerts)

    def test_critical_alert_on_large_change(self):
        detector = TopologyDriftDetector(
            warning_threshold=0.05,
            critical_threshold=0.15,
            cooldown=0,
        )
        detector.check(self._make_telemetry(0.8, 0.7, 0.9))
        alerts = detector.check(self._make_telemetry(0.5, 0.7, 0.9))
        critical = [a for a in alerts if a.severity == "critical"]
        assert len(critical) > 0

    def test_cooldown_suppresses_repeated_alerts(self):
        detector = TopologyDriftDetector(
            warning_threshold=0.05,
            cooldown=9999,
        )
        detector.check(self._make_telemetry(0.8, 0.7, 0.9))
        detector.check(self._make_telemetry(0.5, 0.7, 0.9))
        # Second large change should be suppressed by cooldown
        alerts = detector.check(self._make_telemetry(0.2, 0.7, 0.9))
        karma_alerts = [a for a in alerts if a.pillar == "karma_isolation"]
        assert len(karma_alerts) == 0

    def test_recent_alerts_accumulate(self):
        detector = TopologyDriftDetector(warning_threshold=0.05, cooldown=0)
        detector.check(self._make_telemetry(0.8, 0.7, 0.9))
        detector.check(self._make_telemetry(0.5, 0.7, 0.9))
        assert len(detector.recent_alerts) > 0

    def test_to_json(self):
        detector = TopologyDriftDetector(warning_threshold=0.05, cooldown=0)
        detector.check(self._make_telemetry(0.8, 0.7, 0.9))
        detector.check(self._make_telemetry(0.5, 0.7, 0.9))
        j = detector.to_json()
        assert isinstance(j, list)
        if j:
            assert "pillar" in j[0]
            assert "severity" in j[0]
            assert "message" in j[0]


# ===================================================================
# ChangeImpactAnalyzer
# ===================================================================


def _chain_graph(n: int = 5) -> tuple[sparse.csr_matrix, list[str]]:
    """Build a linear import chain: A->B->C->D->E.

    adj[i, i+1] = 1 means module i imports module i+1.
    Changing E affects D, C, B, A transitively.
    """
    rows = list(range(n - 1))
    cols = list(range(1, n))
    vals = [1.0] * (n - 1)
    adj = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    labels = [f"mod_{i}" for i in range(n)]
    return adj, labels


def _hub_graph() -> tuple[sparse.csr_matrix, list[str]]:
    """Build a hub graph: modules 1-4 all import module 0 (the hub).

    adj[i, 0] = 1 for i in 1..4.
    """
    n = 5
    rows = [1, 2, 3, 4]
    cols = [0, 0, 0, 0]
    vals = [1.0, 1.0, 1.0, 1.0]
    adj = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    labels = ["hub", "leaf_1", "leaf_2", "leaf_3", "leaf_4"]
    return adj, labels


def _isolated_graph(n: int = 4) -> tuple[sparse.csr_matrix, list[str]]:
    """Build a graph with no edges (all isolated modules)."""
    adj = sparse.csr_matrix(([], ([], [])), shape=(n, n))
    labels = [f"isolated_{i}" for i in range(n)]
    return adj, labels


class TestChangeImpactAnalyzer:
    def test_direct_dependents_chain(self):
        """In A->B->C, changing C directly affects B (B imports C)."""
        adj, labels = _chain_graph(3)
        analyzer = ChangeImpactAnalyzer(adj, labels)
        report = analyzer.analyze_change("mod_2")
        assert "mod_1" in report.directly_affected
        assert isinstance(report, ChangeImpactReport)

    def test_transitive_dependents_chain(self):
        """In A->B->C->D->E, changing E affects D, C, B, A transitively."""
        adj, labels = _chain_graph(5)
        analyzer = ChangeImpactAnalyzer(adj, labels)
        report = analyzer.analyze_change("mod_4")
        assert len(report.transitively_affected) == 4
        assert report.critical_path_length == 4

    def test_hub_bottleneck(self):
        """Hub module (imported by all) should be a bottleneck."""
        adj, labels = _hub_graph()
        analyzer = ChangeImpactAnalyzer(adj, labels)
        report = analyzer.analyze_change("hub")
        assert len(report.directly_affected) == 4
        assert report.is_bottleneck

    def test_leaf_no_dependents(self):
        """Leaf module (imports hub but nobody imports it) has no dependents."""
        adj, labels = _hub_graph()
        analyzer = ChangeImpactAnalyzer(adj, labels)
        report = analyzer.analyze_change("leaf_1")
        assert len(report.directly_affected) == 0
        assert len(report.transitively_affected) == 0
        assert report.impact_score == 0.0
        assert not report.is_bottleneck

    def test_isolated_modules(self):
        """Isolated modules have zero impact."""
        adj, labels = _isolated_graph()
        analyzer = ChangeImpactAnalyzer(adj, labels)
        report = analyzer.analyze_change("isolated_0")
        assert report.impact_score == 0.0
        assert report.critical_path_length == 0

    def test_unknown_module_raises(self):
        adj, labels = _chain_graph(3)
        analyzer = ChangeImpactAnalyzer(adj, labels)
        try:
            analyzer.analyze_change("nonexistent")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_analyze_changes_multi(self):
        """Changing multiple modules combines their blast radius."""
        adj, labels = _chain_graph(5)
        analyzer = ChangeImpactAnalyzer(adj, labels)
        report = analyzer.analyze_changes(["mod_3", "mod_4"])
        assert len(report.transitively_affected) >= 3

    def test_analyze_changes_empty(self):
        adj, labels = _chain_graph(3)
        analyzer = ChangeImpactAnalyzer(adj, labels)
        report = analyzer.analyze_changes([])
        assert report.impact_score == 0.0

    def test_find_bottlenecks(self):
        """Hub should be found as a bottleneck."""
        adj, labels = _hub_graph()
        analyzer = ChangeImpactAnalyzer(adj, labels)
        bottlenecks = analyzer.find_bottlenecks(threshold=0.3)
        assert "hub" in bottlenecks

    def test_find_safe_to_modify(self):
        """Leaf modules should be safe to modify."""
        adj, labels = _hub_graph()
        analyzer = ChangeImpactAnalyzer(adj, labels)
        safe = analyzer.find_safe_to_modify(threshold=0.1)
        assert "leaf_1" in safe

    def test_critical_path_chain(self):
        """Critical path in A->B->C->D->E from E should be [E,D,C,B,A]."""
        adj, labels = _chain_graph(5)
        analyzer = ChangeImpactAnalyzer(adj, labels)
        path = analyzer.critical_path("mod_4")
        assert path[0] == "mod_4"
        assert len(path) == 5

    def test_critical_path_leaf(self):
        """Critical path from a leaf is just the leaf itself."""
        adj, labels = _hub_graph()
        analyzer = ChangeImpactAnalyzer(adj, labels)
        path = analyzer.critical_path("leaf_1")
        assert path == ["leaf_1"]


# ===================================================================
# Module Health
# ===================================================================


class TestModuleHealth:
    def test_hub_has_high_afferent(self):
        """Hub module (many importers) should have high afferent coupling."""
        adj, labels = _hub_graph()
        health = compute_module_health(adj, labels)
        hub_health = [h for h in health if h.module_name == "hub"][0]
        assert hub_health.in_degree == 4
        assert hub_health.afferent_coupling > 0.5
        assert hub_health.is_hub

    def test_leaf_has_high_efferent(self):
        """Leaf that imports hub has efferent coupling."""
        adj, labels = _hub_graph()
        health = compute_module_health(adj, labels)
        leaf = [h for h in health if h.module_name == "leaf_1"][0]
        assert leaf.out_degree == 1
        assert leaf.efferent_coupling > 0.0

    def test_instability_hub_stable(self):
        """Hub with only afferent coupling should have low instability."""
        adj, labels = _hub_graph()
        health = compute_module_health(adj, labels)
        hub = [h for h in health if h.module_name == "hub"][0]
        assert hub.instability < 0.5

    def test_instability_leaf_unstable(self):
        """Leaf with only efferent coupling should have high instability."""
        adj, labels = _hub_graph()
        health = compute_module_health(adj, labels)
        leaf = [h for h in health if h.module_name == "leaf_1"][0]
        assert leaf.instability > 0.5

    def test_empty_graph(self):
        adj = sparse.csr_matrix(([], ([], [])), shape=(0, 0))
        health = compute_module_health(adj, [])
        assert health == []

    def test_isolated_modules(self):
        """Isolated modules have zero coupling."""
        adj, labels = _isolated_graph()
        health = compute_module_health(adj, labels)
        for h in health:
            assert h.in_degree == 0
            assert h.out_degree == 0

    def test_returns_module_health_report(self):
        adj, labels = _chain_graph(3)
        health = compute_module_health(adj, labels)
        assert len(health) == 3
        assert all(isinstance(h, ModuleHealthReport) for h in health)
