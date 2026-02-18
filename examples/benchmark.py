"""LMM Performance Benchmark - Complete Runtime Test

numpy/scipy をインストールして以下を実行:
  PYTHONPATH=/home/user/LMM python run_benchmark.py
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np

print("=" * 70)
print("LMM Performance Benchmark")
print("=" * 70)
print(f"numpy={np.__version__}, python={sys.version.split()[0]}")
print()

rng = np.random.default_rng(42)

# ============================================================
# Benchmark 1: QUBO Solver Comparison
# ============================================================
print("[1] QUBO Solver Comparison (n=200, k=15)")
print("-" * 50)

from lmm.qubo import QUBOBuilder
from lmm.solvers import ClassicalQUBOSolver

n, k = 200, 15
surprises = rng.exponential(2.0, size=n)

builder = QUBOBuilder(n)
builder.add_surprise_objective(surprises, alpha=1.0)
builder.add_cardinality_constraint(k, gamma=10.0)
solver = ClassicalQUBOSolver(builder)

# SA (射影なし)
t0 = time.perf_counter()
for _ in range(5):
    x_sa = solver.solve_sa(n_iterations=5000)
t_sa = (time.perf_counter() - t0) / 5
e_sa = builder.evaluate(x_sa)

# SA (射影あり)
t0 = time.perf_counter()
for _ in range(5):
    x_sa_proj = solver.solve_sa(n_iterations=5000, k=k)
t_sa_proj = (time.perf_counter() - t0) / 5
e_sa_proj = builder.evaluate(x_sa_proj)

# Ising SA (射影なし)
t0 = time.perf_counter()
for _ in range(5):
    x_ising = solver.solve_sa_ising(n_iterations=5000)
t_ising = (time.perf_counter() - t0) / 5
e_ising = builder.evaluate(x_ising)

# Ising SA (射影あり)
t0 = time.perf_counter()
for _ in range(5):
    x_ising_proj = solver.solve_sa_ising(n_iterations=5000, k=k)
t_ising_proj = (time.perf_counter() - t0) / 5
e_ising_proj = builder.evaluate(x_ising_proj)

# Greedy
t0 = time.perf_counter()
for _ in range(5):
    x_greedy = solver.solve_greedy(k=k)
t_greedy = (time.perf_counter() - t0) / 5
e_greedy = builder.evaluate(x_greedy)

# Warm Start SA (射影あり)
initial = np.zeros(n)
initial[np.argsort(surprises)[-k:]] = 1.0
t0 = time.perf_counter()
for _ in range(5):
    x_warm = solver.solve_sa(initial_state=initial, n_iterations=5000, k=k)
t_warm = (time.perf_counter() - t0) / 5
e_warm = builder.evaluate(x_warm)

# Warm Start Ising (射影あり)
t0 = time.perf_counter()
for _ in range(5):
    x_warm_ising = solver.solve_sa_ising(initial_state=initial, n_iterations=5000, k=k)
t_warm_ising = (time.perf_counter() - t0) / 5
e_warm_ising = builder.evaluate(x_warm_ising)

print(f"  {'Method':<30} {'Time(ms)':>10} {'Energy':>12} {'Selected':>10}")
print(f"  {'─'*30} {'─'*10} {'─'*12} {'─'*10}")
for name, t, e, x in [
    ("SA (no proj)", t_sa, e_sa, x_sa),
    ("SA (proj k=15)", t_sa_proj, e_sa_proj, x_sa_proj),
    ("Ising SA (no proj)", t_ising, e_ising, x_ising),
    ("Ising SA (proj k=15)", t_ising_proj, e_ising_proj, x_ising_proj),
    ("Warm SA (proj)", t_warm, e_warm, x_warm),
    ("Warm Ising (proj)", t_warm_ising, e_warm_ising, x_warm_ising),
    ("Greedy", t_greedy, e_greedy, x_greedy),
]:
    sel = int((x > 0.5).sum())
    print(f"  {name:<30} {t*1000:>10.1f} {e:>12.2f} {sel:>10}")

speedup = t_sa / t_ising if t_ising > 0 else float('inf')
print(f"\n  Ising SA speedup vs SA: {speedup:.2f}x")
print(f"  Projection fix: Ising {int(x_ising.sum())} → {int(x_ising_proj.sum())} items (target: {k})")

# ============================================================
# Benchmark 2: Sparse Graph Construction
# ============================================================
print(f"\n[2] Sparse Graph Construction")
print("-" * 50)

from lmm.dharma.algorithms import build_sparse_impact_graph

for n_test in [100, 500, 1000]:
    data = rng.normal(0, 1, size=(n_test, 50))
    t0 = time.perf_counter()
    graph = build_sparse_impact_graph(data, k=20, use_hnswlib=False)
    t_sparse = time.perf_counter() - t0
    density = graph.nnz / (n_test * n_test) * 100
    print(f"  n={n_test:>5}: {t_sparse*1000:>8.1f}ms  nnz={graph.nnz:>8}  density={density:.1f}%")

# ============================================================
# Benchmark 3: Supermodular Greedy vs Plain Greedy
# ============================================================
print(f"\n[3] Supermodular Greedy vs Plain Greedy (n=200, k=15)")
print("-" * 50)

from lmm.dharma.algorithms import vectorized_greedy_initialize

data = rng.normal(0, 1, size=(200, 50))
surprises = rng.exponential(2.0, size=200)
graph = build_sparse_impact_graph(data, k=20, use_hnswlib=False)

t0 = time.perf_counter()
for _ in range(10):
    x_sm = vectorized_greedy_initialize(surprises, graph, k=15, alpha=1.0, beta=0.5)
t_sm = (time.perf_counter() - t0) / 10

# Compare: plain greedy (no impact)
t0 = time.perf_counter()
for _ in range(10):
    x_plain = vectorized_greedy_initialize(surprises, graph, k=15, alpha=1.0, beta=0.0)
t_plain = (time.perf_counter() - t0) / 10

# Evaluate with QUBO including karuna
from lmm.dharma.algorithms import BodhisattvaQUBO
qubo = BodhisattvaQUBO(200)
qubo.add_prajna_term(surprises, alpha=1.0)
qubo.add_karuna_term_sparse(graph, beta=0.5)
qubo.add_sila_term(15, gamma=10.0)

e_sm = qubo.get_builder().evaluate(x_sm)

qubo2 = BodhisattvaQUBO(200)
qubo2.add_prajna_term(surprises, alpha=1.0)
qubo2.add_karuna_term_sparse(graph, beta=0.5)
qubo2.add_sila_term(15, gamma=10.0)
e_plain = qubo2.get_builder().evaluate(x_plain)

print(f"  Supermodular greedy: {t_sm*1000:.1f}ms  energy={e_sm:.2f}  selected={int(x_sm.sum())}")
print(f"  Plain greedy (β=0):  {t_plain*1000:.1f}ms  energy={e_plain:.2f}  selected={int(x_plain.sum())}")
improvement = (e_plain - e_sm) / abs(e_plain) * 100 if e_plain != 0 else 0
print(f"  Energy improvement: {improvement:+.1f}%")

# ============================================================
# Benchmark 4: Madhyamaka Balancer
# ============================================================
print(f"\n[4] Exponential vs Linear Balance (10 iterations)")
print("-" * 50)

from lmm.dharma.algorithms import MadhyamakaBalancer

balancer = MadhyamakaBalancer(target_cv=0.5, learning_rate=0.1)
skewed_data = np.concatenate([rng.normal(0, 0.1, 90), rng.normal(10, 1, 10)])

alpha_lin, beta_lin = 1.0, 0.5
alpha_exp, beta_exp = 1.0, 0.5

for i in range(10):
    alpha_lin, beta_lin = balancer.balance(skewed_data, alpha_lin, beta_lin)
    alpha_exp, beta_exp = balancer.balance_exponential(skewed_data, alpha_exp, beta_exp)

cv = skewed_data.std() / skewed_data.mean()
print(f"  Input CV: {cv:.3f}  (target: 0.500)")
print(f"  Linear:      α={alpha_lin:.4f}  β={beta_lin:.4f}")
print(f"  Exponential: α={alpha_exp:.4f}  β={beta_exp:.4f}")

# ============================================================
# Benchmark 5: Submodular vs QUBO vs Random (n=200, k=15)
# ============================================================
print(f"\n[5] Submodular vs QUBO+SA vs Random (n=200, k=15)")
print("-" * 50)

from lmm.core import LMM
from lmm.dharma.api import DharmaLMM
from lmm.solvers import SubmodularSelector

ref = rng.normal(0, 1, size=(500, 50))
candidates = rng.exponential(1.0, size=(200, 50))

# --- Submodular greedy (推奨: 保証付き) ---
dharma_sub = DharmaLMM(k=15, solver_mode="submodular", use_sparse_graph=True)
dharma_sub.fit(ref)
t0 = time.perf_counter()
for _ in range(10):
    r_sub = dharma_sub.select_dharma(candidates, interpret=False)
t_sub = (time.perf_counter() - t0) / 10

# --- QUBO+Ising SA (従来) ---
dharma_ising = DharmaLMM(
    k=15, solver_mode="ising_sa", use_sparse_graph=True,
    use_greedy_warmstart=True, use_exponential_balance=True,
)
dharma_ising.fit(ref)
t0 = time.perf_counter()
for _ in range(3):
    r_ising = dharma_ising.select_dharma(candidates, interpret=False)
t_ising = (time.perf_counter() - t0) / 3

# --- LMM classic (SA) ---
lmm = LMM(k=15, solver_method="sa")
lmm.fit(ref)
t0 = time.perf_counter()
for _ in range(3):
    r_lmm = lmm.select(candidates)
t_lmm = (time.perf_counter() - t0) / 3

# --- Random baseline ---
t0 = time.perf_counter()
for _ in range(100):
    random_idx = rng.choice(200, size=15, replace=False)
t_rand = (time.perf_counter() - t0) / 100

# Evaluate random with submodular objective for fair comparison
sub_eval = SubmodularSelector(alpha=1.0, beta=0.5)
surprises_eval = dharma_sub._calculator.compute(candidates)
graph_eval = build_sparse_impact_graph(candidates, k=20, use_hnswlib=False)
random_val = sub_eval.evaluate(random_idx, surprises_eval, graph_eval)

print(f"  {'Method':<25} {'Time(ms)':>10} {'Obj/Energy':>12} {'Selected':>8} {'Guarantee':>10}")
print(f"  {'─'*25} {'─'*10} {'─'*12} {'─'*8} {'─'*10}")
print(f"  {'Submodular greedy':<25} {t_sub*1000:>10.1f} {r_sub.energy:>12.2f} {len(r_sub.selected_indices):>8} {'(1-1/e)':>10}")
print(f"  {'QUBO+Ising SA':<25} {t_ising*1000:>10.1f} {r_ising.energy:>12.2f} {len(r_ising.selected_indices):>8} {'none':>10}")
print(f"  {'LMM classic (SA)':<25} {t_lmm*1000:>10.1f} {r_lmm.energy:>12.2f} {len(r_lmm.selected_indices):>8} {'none':>10}")
print(f"  {'Random':<25} {t_rand*1000:>10.4f} {-random_val:>12.2f} {15:>8} {'none':>10}")

speedup = t_lmm / t_sub if t_sub > 0 else 0
print(f"\n  Submodular vs LMM classic: {speedup:.1f}x faster")
print(f"  Submodular vs Random obj:  {abs(r_sub.energy)/random_val:.1f}x better objective")

# ============================================================  
# Benchmark 6: Scale - Streaming Performance
# ============================================================
print(f"\n[6] Streaming Surprise (simulated 10M tokens)")
print("-" * 50)

from lmm.scale.stream import StreamingSurprise

ss = StreamingSurprise(k=100, chunk_size=100_000)

# Fit
t0 = time.perf_counter()
chunks_fit = [rng.normal(0, 1, size=100_000) for _ in range(10)]
ss.fit_stream(iter(chunks_fit))
t_fit = time.perf_counter() - t0

# Compute
chunks_compute = [rng.exponential(2.0, size=100_000) for _ in range(100)]
t0 = time.perf_counter()
for chunk_result in ss.compute_stream(iter(chunks_compute)):
    pass
indices, scores = ss.get_top_k()
t_compute = time.perf_counter() - t0

total_tokens = 100 * 100_000
print(f"  Fit:     {t_fit*1000:.0f}ms ({10*100_000:,} tokens)")
print(f"  Compute: {t_compute*1000:.0f}ms ({total_tokens:,} tokens)")
print(f"  Throughput: {total_tokens / t_compute:,.0f} tokens/sec")
print(f"  Top-100 selected, peak score: {scores[0]:.2f}")
sketch_mem = 0
if ss._sketch is not None:
    sketch_mem = ss._sketch.width * ss._sketch.depth * 8
print(f"  Memory: ~{ss._histogram.max_bins * 16 / 1024:.0f}KB histogram + ~{sketch_mem / 1024:.0f}KB sketch")

# ============================================================
# Benchmark 7: Sketch Data Structures
# ============================================================
print(f"\n[7] Count-Min Sketch & Streaming Histogram")
print("-" * 50)

from lmm.scale.sketch import CountMinSketch, StreamingHistogram

# CMS
cms = CountMinSketch(width=10000, depth=8)
data_batch = rng.normal(0, 1, size=1_000_000)
t0 = time.perf_counter()
cms.add_batch(data_batch)
t_cms = time.perf_counter() - t0
print(f"  CMS add_batch(1M): {t_cms*1000:.0f}ms  ({1_000_000/t_cms:,.0f} items/sec)")

# Histogram
hist = StreamingHistogram(max_bins=1024)
t0 = time.perf_counter()
hist.add_batch(data_batch[:100_000])
t_hist = time.perf_counter() - t0
print(f"  Histogram add_batch(100K): {t_hist*1000:.0f}ms  ({100_000/t_hist:,.0f} items/sec)")
print(f"  Histogram bins used: {len(hist._bins)}")

# ============================================================
# Summary
# ============================================================
print(f"\n{'=' * 70}")
print("PERFORMANCE SUMMARY")
print(f"{'=' * 70}")
print(f"  Greedy QUBO:             {t_greedy*1000:.1f}ms (n={n})")
print(f"  Ising SA proj fix:       {int(x_ising.sum())} → {int(x_ising_proj.sum())} items (target: {k})")
sub_speedup = t_lmm / t_sub if t_sub > 0 else 0
print(f"  Submodular vs LMM:       {sub_speedup:.1f}x faster, (1-1/e) guarantee")
print(f"  Submodular selected:     {len(r_sub.selected_indices)} items (target: 15)")
print(f"  Streaming throughput:    {total_tokens / t_compute:,.0f} tokens/sec")
print(f"  Memory (streaming):      ~{(ss._histogram.max_bins * 16 + sketch_mem) / 1024:.0f}KB fixed")
print(f"{'=' * 70}")
