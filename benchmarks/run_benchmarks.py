#!/usr/bin/env python3
"""LMM Performance Benchmarks

Measures performance of key components:
1. sparse_matvec centralized vs inline
2. ClassicalQUBOSolver lazy Q effect
3. QUBOBuilder.evaluate() sparse vs dense path
4. Engine end-to-end (synthesize_and_solve)
5. Input validation overhead (add_linear / add_quadratic)
"""

import time
import numpy as np
from scipy import sparse
from lmm._compat import sparse_matvec


def to_list(arr):
    """Convert ndarray to list (shim-compatible)."""
    return [int(x) for x in arr]


def to_float_list(arr):
    """Convert ndarray to float list (shim-compatible)."""
    return [float(x) for x in arr]


def bench(fn, *args, n_iter=100):
    # warmup
    for _ in range(3):
        fn(*args)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn(*args)
    return (time.perf_counter() - t0) / n_iter * 1000  # ms


# Old inline version (what was in engine.py before centralization)
def old_sparse_matvec(J, v):
    n = J.shape[0]
    result = np.zeros(n)
    coo = J.tocoo()
    for r, c, val in zip(coo.row, coo.col, coo.data):
        result[int(r)] += val * float(v[int(c)])
    return result


# ============================================================
# Benchmark 1: sparse_matvec centralized vs inline
# ============================================================
print("=" * 60)
print("Benchmark 1: sparse_matvec centralized vs inline")
print("=" * 60)
rng = np.random.default_rng(42)
for n in [50, 200, 500, 1000]:
    density = min(0.1, 20.0 / n)  # ~20 nnz per row
    nnz = int(n * n * density)
    rows = to_list(rng.integers(0, n, nnz))
    cols = to_list(rng.integers(0, n, nnz))
    vals = to_float_list(rng.random(nnz))
    J = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    v = np.array(to_float_list(rng.random(n)))

    t_new = bench(sparse_matvec, J, v, n_iter=50)
    t_old = bench(old_sparse_matvec, J, v, n_iter=50)
    print(f"  n={n:4d} nnz={J.nnz:5d}: new={t_new:.3f}ms old={t_old:.3f}ms ratio={t_old / max(t_new, 0.001):.2f}x")


# ============================================================
# Benchmark 2: Lazy Q -- solve_sa() without dense materialization
# ============================================================
from lmm.qubo import QUBOBuilder
from lmm.solvers import ClassicalQUBOSolver

print()
print("=" * 60)
print("Benchmark 2: Lazy Q -- solve_sa() without dense materialization")
print("=" * 60)

for n in [50, 200, 500, 1000]:
    builder = QUBOBuilder(n)
    s = np.array(to_float_list(rng.exponential(1.0, n)))
    builder.add_surprise_objective(s)
    k = max(1, n // 5)
    builder.add_cardinality_constraint(k, gamma=10.0)

    # Time solve_sa (should NOT materialize Q)
    t0 = time.perf_counter()
    solver = ClassicalQUBOSolver(builder)
    x = solver.solve_sa(k=k, n_iterations=200)
    t_sa = time.perf_counter() - t0
    q_materialized = solver._Q_cache is not None

    # Time solve_greedy (DOES materialize Q)
    solver2 = ClassicalQUBOSolver(builder)
    t0 = time.perf_counter()
    x2 = solver2.solve_greedy(k=k)
    t_greedy = time.perf_counter() - t0
    q_materialized2 = solver2._Q_cache is not None

    print(f"  n={n:4d}: SA={t_sa * 1000:.1f}ms (Q={q_materialized}) | Greedy={t_greedy * 1000:.1f}ms (Q={q_materialized2})")


# ============================================================
# Benchmark 3: evaluate() sparse vs dense x^T Q x
# ============================================================
print()
print("=" * 60)
print("Benchmark 3: evaluate() sparse vs dense x^T Q x")
print("=" * 60)

for n in [50, 200, 500, 1000]:
    builder = QUBOBuilder(n)
    s = np.array(to_float_list(rng.exponential(1.0, n)))
    builder.add_surprise_objective(s)
    k = max(1, n // 5)
    builder.add_cardinality_constraint(k, gamma=10.0)

    # Random solution
    x = np.zeros(n)
    chosen = rng.choice(n, size=min(k, n), replace=False)
    for c in chosen:
        x[int(c)] = 1.0

    # Sparse evaluate
    t_sparse = bench(builder.evaluate, x, n_iter=50)

    # Dense x^T Q x
    Q = builder.Q

    def dense_eval(x, Q=Q, n=n):
        e = 0.0
        for i in range(n):
            for j in range(n):
                e += float(Q[i, j]) * float(x[i]) * float(x[j])
        return e

    t_dense = bench(dense_eval, x, n_iter=5 if n <= 200 else 1)

    print(f"  n={n:4d}: sparse={t_sparse:.3f}ms dense={t_dense:.1f}ms speedup={t_dense / max(t_sparse, 0.001):.1f}x")


# ============================================================
# Benchmark 4: Engine end-to-end (synthesize_and_solve)
# ============================================================
from lmm.dharma.engine import UniversalDharmaEngine
from lmm.dharma.energy import PrajnaTerm, SilaTerm

print()
print("=" * 60)
print("Benchmark 4: Engine end-to-end (synthesize_and_solve)")
print("=" * 60)

for n in [30, 100, 300]:
    surprises = np.array(to_float_list(rng.exponential(1.0, n)))
    k = max(1, n // 5)

    engine = UniversalDharmaEngine(n, sa_iterations=500)
    engine.add(PrajnaTerm(surprises, weight=1.0))
    # SilaTerm uses 'weight' parameter (not 'gamma'), which becomes sila_gamma
    engine.add(SilaTerm(k=k, weight=10.0))

    # Warmup
    engine.synthesize_and_solve(k=k)

    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        result = engine.synthesize_and_solve(k=k)
        times.append(time.perf_counter() - t0)

    avg_ms = sum(times) / len(times) * 1000
    print(f"  n={n:4d} k={k:3d}: {avg_ms:.1f}ms solver={result.solver_used}")


# ============================================================
# Benchmark 5: Validation overhead (add_linear/add_quadratic)
# ============================================================
print()
print("=" * 60)
print("Benchmark 5: Validation overhead (add_linear/add_quadratic)")
print("=" * 60)

n = 1000
builder = QUBOBuilder(n)

t0 = time.perf_counter()
for i in range(n):
    builder.add_linear(i, float(rng.random()))
t_linear = time.perf_counter() - t0

t0 = time.perf_counter()
for _ in range(n):
    i, j = int(rng.integers(0, n)), int(rng.integers(0, n))
    builder.add_quadratic(i, j, float(rng.random()))
t_quad = time.perf_counter() - t0

print(f"  add_linear x {n}: {t_linear * 1000:.2f}ms ({t_linear / n * 1e6:.1f}us/call)")
print(f"  add_quadratic x {n}: {t_quad * 1000:.2f}ms ({t_quad / n * 1e6:.1f}us/call)")

print()
print("=" * 60)
print("All benchmarks complete.")
print("=" * 60)
