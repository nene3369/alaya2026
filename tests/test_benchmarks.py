"""Consolidated performance benchmarks with summary report."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy import sparse

from lmm._compat import sparse_matvec
from lmm.core import LMM
from lmm.dharma.algorithms import (
    build_sparse_impact_graph,
    extract_subgraph,
    query_condition_graph,
)
from lmm.dharma.api import DharmaLMM
from lmm.dharma.energy import KarunaTerm, MettaTerm, PrajnaTerm, SilaTerm
from lmm.dharma.engine import UniversalDharmaEngine
from lmm.dharma.fep import solve_fep_kcl, solve_fep_kcl_analog
from lmm.dharma.reranker import DharmaReranker
from lmm.integrations.langchain import rerank_by_dharma_engine, rerank_by_diversity
from lmm.llm.embeddings import cosine_similarity, ngram_vectors
from lmm.qubo import QUBOBuilder
from lmm.solvers import ClassicalQUBOSolver, SubmodularSelector
from lmm.surprise import SurpriseCalculator


# ---------------------------------------------------------------------------
# Benchmark result collector
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkEntry:
    category: str
    name: str
    elapsed: float
    limit: float
    detail: str = ""


_results: list[BenchmarkEntry] = []


def _record(category: str, name: str, elapsed: float, limit: float, detail: str = ""):
    _results.append(BenchmarkEntry(category, name, elapsed, limit, detail))


def _bar(elapsed: float, limit: float, width: int = 20) -> str:
    """Render a proportional bar: filled portion of limit."""
    ratio = min(elapsed / limit, 1.0) if limit > 0 else 0.0
    filled = int(ratio * width)
    empty = width - filled
    if ratio < 0.25:
        marker = "\u2588"  # green-ish feel
    elif ratio < 0.60:
        marker = "\u2593"
    else:
        marker = "\u2591"
    return marker * filled + "\u00b7" * empty


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sparse_graph(rng, n, k_neighbors=5):
    """Build a symmetric sparse test graph (chain + random neighbors)."""
    rows, cols, vals = [], [], []
    for i in range(n - 1):
        w = float(rng.rand()) * 0.5 + 0.1
        rows.extend([i, i + 1])
        cols.extend([i + 1, i])
        vals.extend([w, w])
    for i in range(n):
        for _ in range(k_neighbors):
            j = int(rng.randint(0, n))
            if i != j:
                w = float(rng.rand()) * 0.3
                rows.extend([i, j])
                cols.extend([j, i])
                vals.extend([w, w])
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float64)


# ===================================================================
# 1. Core pipeline benchmarks
# ===================================================================

class TestCoreBenchmarks:
    def test_surprise_1000(self):
        rng = np.random.RandomState(42)
        data = rng.randn(1000)
        calc = SurpriseCalculator(method="kl")
        t0 = time.time()
        calc.fit(data)
        calc.compute(data)
        elapsed = time.time() - t0
        _record("Core", "Surprise (n=1000, KL)", elapsed, 5.0)
        assert elapsed < 5.0

    def test_qubo_build_100(self):
        rng = np.random.RandomState(42)
        n = 100
        builder = QUBOBuilder(n_variables=n)
        surprises = rng.rand(n)
        t0 = time.time()
        builder.add_surprise_objective(surprises, alpha=1.0)
        builder.add_cardinality_constraint(k=10, gamma=10.0)
        builder.get_matrix()
        elapsed = time.time() - t0
        _record("Core", "QUBO build (n=100)", elapsed, 5.0)
        assert elapsed < 5.0

    def test_lmm_pipeline_500(self):
        rng = np.random.RandomState(42)
        data = rng.randn(500)
        model = LMM(k=10)
        t0 = time.time()
        model.fit(data)
        result = model.select(data)
        elapsed = time.time() - t0
        _record("Core", "LMM pipeline (n=500)", elapsed, 15.0, f"k={len(result.selected_indices)}")
        assert elapsed < 15.0
        assert len(result.selected_indices) == 10


# ===================================================================
# 2. Solver benchmarks — SA / Ising SA / Greedy comparison
# ===================================================================

class TestSolverBenchmarks:
    def test_sa_vs_ising_vs_greedy_100(self):
        """Compare SA, Ising SA, Greedy at n=100."""
        rng = np.random.RandomState(42)
        n, k = 100, 10
        builder = QUBOBuilder(n_variables=n)
        builder.add_surprise_objective(rng.rand(n), alpha=1.0)
        builder.add_cardinality_constraint(k=k, gamma=10.0)

        for method, iters in [("sa", 1000), ("ising_sa", 1000), ("greedy", None)]:
            solver = ClassicalQUBOSolver(builder)
            t0 = time.time()
            if method == "greedy":
                x = solver.solve(method="greedy", k=k)
            else:
                x = solver.solve(method=method, k=k, n_iterations=iters)
            elapsed = time.time() - t0
            n_sel = int((x > 0.5).sum())
            label = f"{method.upper()} (n=100)"
            _record("Solver", label, elapsed, 10.0, f"sel={n_sel}")
            assert elapsed < 10.0
            assert n_sel == k

    def test_sa_scaling(self):
        """SA should scale roughly linearly with n (fixed iterations)."""
        rng = np.random.RandomState(42)
        scales = [50, 100, 200]
        times = []
        for n in scales:
            builder = QUBOBuilder(n_variables=n)
            builder.add_surprise_objective(rng.rand(n), alpha=1.0)
            builder.add_cardinality_constraint(k=n // 10, gamma=10.0)
            solver = ClassicalQUBOSolver(builder)
            t0 = time.time()
            solver.solve(method="sa", k=n // 10, n_iterations=1000)
            elapsed = time.time() - t0
            times.append(elapsed)
            _record("Scaling", f"SA n={n}", elapsed, 10.0)
        ratio = times[-1] / max(times[0], 0.001)
        _record("Scaling", "SA ratio 200/50", ratio, 12.0, f"{ratio:.1f}x")
        assert ratio < 12.0
        assert times[-1] < 10.0


# ===================================================================
# 3. Submodular benchmarks
# ===================================================================

class TestSubmodularBenchmarks:
    def test_select_sparse_200(self):
        """Submodular select with sparse graph, n=200."""
        rng = np.random.RandomState(42)
        n, k = 200, 10
        relevance = rng.rand(n)
        graph = _make_sparse_graph(rng, n, k_neighbors=10)
        selector = SubmodularSelector(alpha=1.0, beta=0.5)
        t0 = time.time()
        result = selector.select(relevance, graph, k=k)
        elapsed = time.time() - t0
        _record("Submodular", "select (n=200, sparse)", elapsed, 10.0, f"obj={result.objective_value:.1f}")
        assert elapsed < 10.0
        assert len(result.selected_indices) == k

    def test_select_lazy_vs_full_200(self):
        """Compare select vs select_lazy at n=200."""
        rng = np.random.RandomState(42)
        n, k = 200, 10
        relevance = rng.rand(n)
        data = rng.randn(n, 16)
        norms = np.clip(np.linalg.norm(data, axis=1, keepdims=True), 1e-10, None)
        normalized = data / norms
        graph = _make_sparse_graph(rng, n, k_neighbors=10)
        selector = SubmodularSelector(alpha=1.0, beta=0.5)

        t0 = time.time()
        r_full = selector.select(relevance, graph, k=k)
        t_full = time.time() - t0
        _record("Submodular", "select_full (n=200)", t_full, 10.0)

        t0 = time.time()
        r_lazy = selector.select_lazy(relevance, normalized, k=k, sparse_k=20)
        t_lazy = time.time() - t0
        _record("Submodular", "select_lazy (n=200)", t_lazy, 10.0)

        assert t_full < 10.0
        assert t_lazy < 10.0
        assert len(r_full.selected_indices) == k
        assert len(r_lazy.selected_indices) == k

    def test_select_adaptive_200(self):
        """Adaptive K detection at n=200."""
        rng = np.random.RandomState(42)
        n = 200
        relevance = rng.rand(n)
        graph = _make_sparse_graph(rng, n, k_neighbors=10)
        selector = SubmodularSelector(alpha=1.0, beta=0.5)
        t0 = time.time()
        result = selector.select_adaptive(relevance, graph, k_max=30)
        elapsed = time.time() - t0
        _record("Submodular", "select_adaptive (n=200)", elapsed, 10.0, f"k={len(result.selected_indices)}")
        assert elapsed < 10.0
        assert 1 <= len(result.selected_indices) <= 30


# ===================================================================
# 4. Sparse graph construction benchmarks
# ===================================================================

class TestSparseGraphBenchmarks:
    def test_build_graph_scaling(self):
        """Graph construction scaling: O(n^2) expected."""
        rng = np.random.RandomState(42)
        scales = [50, 100, 200]
        times = []
        for n in scales:
            data = rng.randn(n, 16)
            t0 = time.time()
            graph = build_sparse_impact_graph(data, k=10, use_hnswlib=False)
            elapsed = time.time() - t0
            times.append(elapsed)
            _record("Graph", f"build (n={n})", elapsed, 20.0, f"nnz={graph.nnz}")
            assert graph.shape == (n, n)
        ratio = times[-1] / max(times[0], 0.001)
        _record("Graph", "ratio 200/50", ratio, 30.0, f"{ratio:.1f}x (O(n^2)~16x)")
        assert ratio < 30.0
        assert times[-1] < 20.0

    def test_query_condition_and_subgraph(self):
        """query_condition_graph + extract_subgraph at n=200."""
        rng = np.random.RandomState(42)
        n = 200
        data = rng.randn(n, 16)
        graph = build_sparse_impact_graph(data, k=10, use_hnswlib=False)
        relevance = rng.rand(n)

        t0 = time.time()
        J_dynamic = query_condition_graph(graph, relevance)
        t_cond = time.time() - t0
        _record("Graph", "condition (n=200)", t_cond, 5.0)

        sub_indices = np.arange(50)
        t0 = time.time()
        J_sub = extract_subgraph(graph, sub_indices)
        t_sub = time.time() - t0
        _record("Graph", "subgraph (200->50)", t_sub, 5.0)

        assert t_cond < 5.0
        assert t_sub < 5.0
        assert J_dynamic.shape == (n, n)
        assert J_sub.shape == (50, 50)


# ===================================================================
# 5. FEP ODE solver benchmarks
# ===================================================================

class TestFEPBenchmarks:
    def test_fep_kcl_100(self):
        """FEP KCL ODE at n=100, shim path."""
        rng = np.random.RandomState(42)
        n = 100
        h = -rng.rand(n)
        graph = _make_sparse_graph(rng, n)
        t0 = time.time()
        V_mu, x, steps, power = solve_fep_kcl(
            h=h, J=graph, k=10, n=n,
            sila_gamma=10.0, max_steps=100, sparse_matvec=sparse_matvec,
        )
        elapsed = time.time() - t0
        _record("FEP", "KCL digital (n=100)", elapsed, 10.0, f"steps={steps}/100")
        assert elapsed < 10.0
        assert len(x) == n

    def test_fep_analog_100(self):
        """Analog FEP ODE at n=100, shim path."""
        rng = np.random.RandomState(42)
        n = 100
        V_s = rng.rand(n)
        graph = _make_sparse_graph(rng, n)
        t0 = time.time()
        V_mu, x, steps, power = solve_fep_kcl_analog(
            V_s=V_s, J_dynamic=graph, n=n,
            max_steps=200, sparse_matvec=sparse_matvec,
        )
        elapsed = time.time() - t0
        _record("FEP", "KCL analog (n=100)", elapsed, 10.0, f"steps={steps}/200")
        assert elapsed < 10.0
        assert len(x) == n

    def test_fep_convergence(self):
        """FEP should converge early (fewer steps than max)."""
        rng = np.random.RandomState(42)
        n = 50
        h = -rng.rand(n) * 5.0
        graph = _make_sparse_graph(rng, n, k_neighbors=3)
        _, _, steps, power = solve_fep_kcl(
            h=h, J=graph, k=5, n=n,
            sila_gamma=10.0, max_steps=300, sparse_matvec=sparse_matvec,
        )
        _record("FEP", "convergence (n=50)", steps, 300, f"{steps}/300 steps")
        assert steps <= 300
        assert len(power) == steps


# ===================================================================
# 6. DharmaReranker benchmarks
# ===================================================================

class TestRerankerBenchmarks:
    def test_reranker_100(self):
        """Full DharmaReranker cascade at n=100."""
        rng = np.random.RandomState(42)
        n, dim, k = 100, 16, 10
        query = rng.randn(dim)
        candidates = rng.randn(n, dim)
        reranker = DharmaReranker(k=k, sa_iterations=1000, solver_mode="auto")
        t0 = time.time()
        result = reranker.rerank(query_embedding=query, candidate_embeddings=candidates)
        elapsed = time.time() - t0
        _record("Reranker", "cascade (n=100)", elapsed, 20.0, f"solver={result.solver_used}")
        assert elapsed < 20.0
        assert len(result.selected_indices) == k

    def test_reranker_fep_analog_50(self):
        """DharmaReranker with FEP analog solver at n=50."""
        rng = np.random.RandomState(42)
        n, dim, k = 50, 16, 5
        query = rng.randn(dim)
        candidates = rng.randn(n, dim)
        reranker = DharmaReranker(k=k, solver_mode="fep_analog")
        t0 = time.time()
        result = reranker.rerank(query_embedding=query, candidate_embeddings=candidates)
        elapsed = time.time() - t0
        _record("Reranker", "FEP analog (n=50)", elapsed, 15.0, f"solver={result.solver_used}")
        assert elapsed < 15.0
        assert len(result.selected_indices) == k

    def test_offline_online_pipeline(self):
        """fit_offline + rerank_online pipeline at corpus=100."""
        rng = np.random.RandomState(42)
        corpus = rng.randn(100, 16)
        query = rng.randn(16)

        reranker = DharmaReranker(k=5, sa_iterations=1000)
        t0 = time.time()
        reranker.fit_offline(corpus, graph_k=10)
        t_offline = time.time() - t0
        _record("Reranker", "fit_offline (c=100)", t_offline, 15.0)

        fetched = np.arange(30)
        t0 = time.time()
        result = reranker.rerank_online(query, fetched)
        t_online = time.time() - t0
        _record("Reranker", "rerank_online (f=30)", t_online, 15.0)

        assert t_offline < 15.0
        assert t_online < 15.0
        assert len(result.selected_indices) == 5


# ===================================================================
# 7. UniversalDharmaEngine benchmarks
# ===================================================================

class TestEngineBenchmarks:
    def test_engine_topk_200(self):
        """Engine with linear-only terms -> topk routing."""
        rng = np.random.RandomState(42)
        n, k = 200, 20
        surprises = rng.rand(n)
        engine = UniversalDharmaEngine(n)
        engine.add(PrajnaTerm(surprises))
        t0 = time.time()
        result = engine.synthesize_and_solve(k=k)
        elapsed = time.time() - t0
        _record("Engine", "topk (n=200)", elapsed, 3.0, "route=topk_sort")
        assert elapsed < 3.0
        assert result.solver_used == "topk_sort"
        assert len(result.selected_indices) == k

    def test_engine_submodular_100(self):
        """Engine with Prajna+Karuna+Sila."""
        rng = np.random.RandomState(42)
        n, k = 100, 10
        surprises = rng.rand(n)
        graph = _make_sparse_graph(rng, n, k_neighbors=5)
        engine = UniversalDharmaEngine(n, sa_iterations=1000)
        engine.add(PrajnaTerm(surprises))
        engine.add(KarunaTerm(graph, weight=0.5))
        engine.add(SilaTerm(k=k, weight=10.0))
        t0 = time.time()
        result = engine.synthesize_and_solve(k=k)
        elapsed = time.time() - t0
        _record("Engine", "submodular (n=100)", elapsed, 15.0, f"route={result.solver_used}")
        assert elapsed < 15.0
        assert len(result.selected_indices) == k

    def test_engine_frustrated_100(self):
        """Engine with Karuna+Metta -> frustrated -> ising_sa."""
        rng = np.random.RandomState(42)
        n, k = 100, 10
        surprises = rng.rand(n)
        graph = _make_sparse_graph(rng, n, k_neighbors=5)
        engine = UniversalDharmaEngine(n, sa_iterations=1000)
        engine.add(PrajnaTerm(surprises))
        engine.add(KarunaTerm(graph, weight=0.5))
        engine.add(MettaTerm(graph, weight=0.3))
        engine.add(SilaTerm(k=k, weight=10.0))
        t0 = time.time()
        result = engine.synthesize_and_solve(k=k)
        elapsed = time.time() - t0
        _record("Engine", "frustrated (n=100)", elapsed, 15.0, f"route={result.solver_used}")
        assert elapsed < 15.0
        assert len(result.selected_indices) == k

    def test_engine_fep_analog_50(self):
        """Engine with fep_analog override."""
        rng = np.random.RandomState(42)
        n, k = 50, 5
        surprises = rng.rand(n)
        graph = _make_sparse_graph(rng, n)
        engine = UniversalDharmaEngine(n, solver="fep_analog")
        engine.add(PrajnaTerm(surprises))
        engine.add(KarunaTerm(graph, weight=0.5))
        engine.add(SilaTerm(k=k, weight=10.0))
        t0 = time.time()
        result = engine.synthesize_and_solve(k=k)
        elapsed = time.time() - t0
        _record("Engine", "fep_analog (n=50)", elapsed, 10.0, f"route={result.solver_used}")
        assert elapsed < 10.0
        assert result.solver_used == "fep_kcl_analog"
        assert len(result.selected_indices) == k


# ===================================================================
# 8. Embedding benchmarks
# ===================================================================

class TestEmbeddingBenchmarks:
    def test_ngram_scaling(self):
        """ngram_vectors should scale roughly linearly with text count."""
        scales = [100, 200, 500]
        times = []
        for n in scales:
            texts = [f"benchmark text number {i} with content for hashing test {i % 7}" for i in range(n)]
            t0 = time.time()
            vecs = ngram_vectors(texts, max_features=500)
            elapsed = time.time() - t0
            times.append(elapsed)
            _record("Embedding", f"ngram (n={n})", elapsed, 15.0)
            assert vecs.shape == (n, 500)
        ratio = times[-1] / max(times[0], 0.001)
        _record("Embedding", "ngram ratio 500/100", ratio, 10.0, f"{ratio:.1f}x (O(n)~5x)")
        assert ratio < 10.0
        assert times[-1] < 15.0

    def test_cosine_similarity_500(self):
        """Batch cosine similarity at n=500, dim=500."""
        rng = np.random.RandomState(42)
        vecs = rng.randn(500, 500)
        query = rng.randn(500)
        t0 = time.time()
        sims = cosine_similarity(vecs, query)
        elapsed = time.time() - t0
        _record("Embedding", "cosine_sim (500x500)", elapsed, 5.0)
        assert elapsed < 5.0
        assert len(sims) == 500


# ===================================================================
# 9. Integration benchmarks
# ===================================================================

class TestIntegrationBenchmarks:
    def test_rerank_by_diversity_100(self):
        """rerank_by_diversity at n=100, k=10."""
        rng = np.random.RandomState(42)
        n, dim, k = 100, 16, 10
        relevance = rng.rand(n)
        embeddings = rng.randn(n, dim)
        t0 = time.time()
        indices = rerank_by_diversity(relevance, embeddings, k=k)
        elapsed = time.time() - t0
        _record("Integration", "diversity (n=100)", elapsed, 15.0)
        assert elapsed < 15.0
        assert len(indices) == k

    def test_rerank_by_dharma_engine_50(self):
        """rerank_by_dharma_engine at n=50, k=5."""
        rng = np.random.RandomState(42)
        n, dim, k = 50, 16, 5
        relevance = rng.rand(n)
        embeddings = rng.randn(n, dim)
        query = rng.randn(dim)
        t0 = time.time()
        indices = rerank_by_dharma_engine(relevance, embeddings, query, k=k)
        elapsed = time.time() - t0
        _record("Integration", "dharma_engine (n=50)", elapsed, 15.0)
        assert elapsed < 15.0
        assert len(indices) == k


# ===================================================================
# 10. DharmaLMM full pipeline benchmarks
# ===================================================================

class TestDharmaLMMBenchmarks:
    def test_dharma_lmm_mode_comparison(self):
        """Compare all solver modes at n=50."""
        rng = np.random.RandomState(42)
        ref_data = rng.randn(200)
        candidates = rng.randn(50, 16)
        modes = ["submodular", "ising_sa", "sa", "engine"]
        for mode in modes:
            model = DharmaLMM(k=5, solver_mode=mode, sa_iterations=1000)
            model.fit(ref_data)
            t0 = time.time()
            result = model.select_dharma(candidates, interpret=False)
            elapsed = time.time() - t0
            _record("DharmaLMM", f"{mode} (n=50)", elapsed, 15.0, f"method={result.method}")
            assert elapsed < 15.0
            assert len(result.selected_indices) <= 5

    def test_dharma_lmm_submodular_200(self):
        """Full DharmaLMM pipeline, submodular, n=200."""
        rng = np.random.RandomState(42)
        ref_data = rng.randn(500)
        candidates = rng.randn(200, 16)
        model = DharmaLMM(k=10, solver_mode="submodular")
        model.fit(ref_data)
        t0 = time.time()
        result = model.select_dharma(candidates, interpret=False)
        elapsed = time.time() - t0
        _record("DharmaLMM", "submodular (n=200)", elapsed, 20.0, f"method={result.method}")
        assert elapsed < 20.0
        assert len(result.selected_indices) <= 10


# ===================================================================
# Summary report — runs last (z-prefix ensures ordering)
# ===================================================================

# ===================================================================
# 11. Scalability tests — verify sparse solvers at large n
# ===================================================================

class TestScalabilitySparse:
    def test_sa_sparse_n10k(self):
        """SA at n=10,000 should never materialise dense Q (would be 800MB)."""
        rng = np.random.RandomState(42)
        n, k = 10_000, 50
        builder = QUBOBuilder(n_variables=n)
        builder.add_surprise_objective(rng.rand(n), alpha=1.0)
        builder.add_cardinality_constraint(k=k, gamma=10.0)
        solver = ClassicalQUBOSolver(builder)
        t0 = time.time()
        x = solver.solve(method="sa", k=k, n_iterations=500)
        elapsed = time.time() - t0
        n_sel = int((x > 0.5).sum())
        _record("Scalability", "SA sparse (n=10K)", elapsed, 60.0, f"sel={n_sel}")
        assert elapsed < 60.0
        assert n_sel == k

    def test_greedy_sparse_n10k(self):
        """Greedy at n=10,000 should use sparse internals only."""
        rng = np.random.RandomState(42)
        n, k = 10_000, 50
        builder = QUBOBuilder(n_variables=n)
        builder.add_surprise_objective(rng.rand(n), alpha=1.0)
        builder.add_cardinality_constraint(k=k, gamma=10.0)
        solver = ClassicalQUBOSolver(builder)
        t0 = time.time()
        x = solver.solve(method="greedy", k=k)
        elapsed = time.time() - t0
        n_sel = int((x > 0.5).sum())
        _record("Scalability", "Greedy sparse (n=10K)", elapsed, 60.0, f"sel={n_sel}")
        assert elapsed < 60.0
        assert n_sel == k

    def test_project_to_k_sparse_n10k(self):
        """_project_to_k at n=10,000 should use sparse path."""
        rng = np.random.RandomState(42)
        n, k = 10_000, 50
        builder = QUBOBuilder(n_variables=n)
        builder.add_surprise_objective(rng.rand(n), alpha=1.0)
        builder.add_cardinality_constraint(k=k, gamma=10.0)
        solver = ClassicalQUBOSolver(builder)
        # Create a random binary vector with wrong cardinality
        x = np.zeros(n)
        x[:k + 10] = 1.0  # 10 too many
        t0 = time.time()
        x_proj = solver._project_to_k(x, k)
        elapsed = time.time() - t0
        n_sel = int((x_proj > 0.5).sum())
        _record("Scalability", "project_to_k (n=10K)", elapsed, 30.0, f"sel={n_sel}")
        assert elapsed < 30.0
        assert n_sel == k

    def test_relaxation_sparse_n200(self):
        """Relaxation at n=200 should use evaluate() not dense Q."""
        rng = np.random.RandomState(42)
        n, k = 200, 10
        builder = QUBOBuilder(n_variables=n)
        builder.add_surprise_objective(rng.rand(n), alpha=1.0)
        builder.add_cardinality_constraint(k=k, gamma=10.0)
        solver = ClassicalQUBOSolver(builder)
        t0 = time.time()
        solver.solve(method="relaxation", n_restarts=2)
        elapsed = time.time() - t0
        _record("Scalability", "Relaxation sparse (n=200)", elapsed, 60.0)
        assert elapsed < 60.0

    def test_purify_sparse_large(self):
        """_purify_matrix should handle large sparse matrices without dict overhead."""
        rng = np.random.RandomState(42)
        n = 10_000
        nnz = 100_000
        rows = rng.randint(0, n, size=nnz)
        cols = rng.randint(0, n, size=nnz)
        vals = rng.randn(nnz)
        # Inject some NaN/Inf
        vals[0] = float("nan")
        vals[1] = float("inf")
        J = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
        engine = UniversalDharmaEngine(n)
        t0 = time.time()
        J_clean = engine._purify_matrix(J, "test_large")
        elapsed = time.time() - t0
        _record("Scalability", "purify (n=10K, nnz=100K)", elapsed, 30.0)
        assert elapsed < 30.0
        assert J_clean.shape == (n, n)
        # Should be symmetric
        diff = J_clean - J_clean.T
        if diff.nnz > 0:
            diff_coo = diff.tocoo()
            max_diff = max(abs(float(v)) for v in diff_coo.data)
            assert max_diff < 1e-10

    def test_heartbeat_n_dims_limit(self):
        """HeartbeatDaemon should reject n_dims > 1024."""
        from lmm.reasoning.heartbeat import HeartbeatDaemon
        try:
            HeartbeatDaemon(n_dims=2000)
            assert False, "Expected ValueError"
        except ValueError as e:
            assert "1024" in str(e)

    def test_ising_sa_sparse_n10k(self):
        """Ising SA at n=10,000 should work with sparse internals."""
        rng = np.random.RandomState(42)
        n, k = 10_000, 50
        builder = QUBOBuilder(n_variables=n)
        builder.add_surprise_objective(rng.rand(n), alpha=1.0)
        builder.add_cardinality_constraint(k=k, gamma=10.0)
        solver = ClassicalQUBOSolver(builder)
        t0 = time.time()
        x = solver.solve(method="ising_sa", k=k, n_iterations=500)
        elapsed = time.time() - t0
        n_sel = int((x > 0.5).sum())
        _record("Scalability", "Ising SA sparse (n=10K)", elapsed, 60.0, f"sel={n_sel}")
        assert elapsed < 60.0
        assert n_sel == k


class TestZBenchmarkReport:
    def test_print_summary(self):
        """Print formatted benchmark summary table."""
        if not _results:
            return

        # Header
        print("\n")
        print("=" * 80)
        print("  LMM Performance Benchmark Report")
        print("=" * 80)

        # Group by category
        categories: dict[str, list[BenchmarkEntry]] = {}
        for entry in _results:
            categories.setdefault(entry.category, []).append(entry)

        for cat, entries in categories.items():
            print(f"\n  [{cat}]")
            print(f"  {'Name':<30} {'Time':>10} {'Limit':>8} {'Bar':<22} {'Detail'}")
            print(f"  {'-'*30} {'-'*10} {'-'*8} {'-'*22} {'-'*20}")
            for e in entries:
                if e.category in ("Scaling", "Graph", "Embedding") and "ratio" in e.name:
                    time_str = f"{e.elapsed:.1f}x"
                    limit_str = f"<{e.limit:.0f}x"
                    bar = _bar(e.elapsed, e.limit)
                elif e.category == "FEP" and "convergence" in e.name:
                    time_str = f"{int(e.elapsed)} steps"
                    limit_str = f"<{int(e.limit)}"
                    bar = _bar(e.elapsed, e.limit)
                else:
                    time_str = f"{e.elapsed*1000:>7.1f}ms"
                    limit_str = f"<{e.limit:.0f}s"
                    bar = _bar(e.elapsed, e.limit)
                print(f"  {e.name:<30} {time_str:>10} {limit_str:>8} {bar:<22} {e.detail}")

        # Footer
        time_entries = [e for e in _results
                        if not (e.category in ("Scaling", "Graph", "Embedding") and "ratio" in e.name)
                        and not (e.category == "FEP" and "convergence" in e.name)]
        total = sum(e.elapsed for e in time_entries)
        n_bench = len(_results)

        print(f"\n  {'='*80}")
        print(f"  Total: {n_bench} benchmarks | {total*1000:.0f}ms cumulative | all PASSED")
        print(f"  {'='*80}\n")
