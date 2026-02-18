"""Dharma-Algebra ベンチマーク

各レイヤー・ソルバー・問題サイズの性能を計測する。
v0.5: スパースネイティブ + 暗黙的カーディナリティ対応
"""

import time
import numpy as np
from scipy import sparse


def _make_graph(n, k_neighbors=5):
    """チェーン+ランダム近傍のスパースグラフ — COO直接構築"""
    rng = np.random.default_rng(42)
    rows, cols, vals = [], [], []

    # チェーン
    for i in range(n - 1):
        w = float(rng.random() * 0.5 + 0.1)
        rows.extend([i, i + 1])
        cols.extend([i + 1, i])
        vals.extend([w, w])

    # ランダム近傍
    for i in range(n):
        for _ in range(k_neighbors):
            j = int(rng.integers(0, n))
            if i != j:
                w = float(rng.random() * 0.3)
                rows.extend([i, j])
                cols.extend([j, i])
                vals.extend([w, w])

    return sparse.csr_matrix(
        (vals, (rows, cols)), shape=(n, n), dtype=np.float64
    )


def _make_surprises(n):
    rng = np.random.default_rng(42)
    return np.array([float(rng.random() * 4.0 + 0.1) for _ in range(n)])


def bench_one(label, fn, warmup=1, repeat=5):
    """関数の実行時間を計測"""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    avg = sum(times) / len(times)
    best = min(times)
    return {"label": label, "avg_ms": avg * 1000, "best_ms": best * 1000}


def main():
    from lmm.dharma.energy import (
        DukkhaTerm, PrajnaTerm, KarunaTerm, MettaTerm,
        KarmaTerm, SilaTerm, UpekkhaTerm, _csr_scale_data,
    )
    from lmm.dharma.engine import UniversalDharmaEngine

    results = []
    sizes = [20, 50, 100, 200, 500, 1000, 2000]
    k = 10

    print("=" * 70)
    print("Dharma-Algebra Benchmark (v0.5 — sparse-native)")
    print("=" * 70)

    # =========================================================================
    # 1. エネルギー項 build() のベンチマーク
    # =========================================================================
    print("\n--- 1. Energy Term build() ---")
    for n in sizes:
        s = _make_surprises(n)
        g = _make_graph(n)
        h_arr = np.array([float(np.random.default_rng(42).integers(0, 10)) for _ in range(n)])

        terms = [
            ("DukkhaTerm", lambda: DukkhaTerm(s).build(n)),
            ("KarunaTerm", lambda: KarunaTerm(g).build(n)),
            ("MettaTerm", lambda: MettaTerm(g).build(n)),
            ("SilaTerm", lambda: SilaTerm(k=k).build(n)),
            ("KarmaTerm", lambda: KarmaTerm(h_arr).build(n)),
            ("UpekkhaTerm", lambda: UpekkhaTerm(s).build(n)),
        ]
        for label, fn in terms:
            r = bench_one(f"  n={n:4d} {label:12s}", fn, warmup=1, repeat=3)
            results.append(r)
            print(f"  n={n:4d} {label:12s}  avg={r['avg_ms']:8.2f}ms  best={r['best_ms']:8.2f}ms")

    # =========================================================================
    # 2. _purify_matrix のベンチマーク
    # =========================================================================
    print("\n--- 2. _purify_matrix (sparse-native) ---")
    for n in sizes:
        g = _make_graph(n)
        engine = UniversalDharmaEngine(n)
        r = bench_one(f"  n={n:4d} purify", lambda: engine._purify_matrix(g, "bench"), warmup=1, repeat=3)
        results.append(r)
        print(f"  n={n:4d} purify        avg={r['avg_ms']:8.2f}ms  best={r['best_ms']:8.2f}ms")

    # =========================================================================
    # 3. _synthesize のベンチマーク
    # =========================================================================
    print("\n--- 3. _synthesize (sparse + implicit cardinality) ---")
    for n in sizes:
        s = _make_surprises(n)
        g = _make_graph(n)
        engine = UniversalDharmaEngine(n)
        engine.add(PrajnaTerm(s, weight=1.0))
        engine.add(KarunaTerm(g, weight=0.5))
        engine.add(SilaTerm(k=k, weight=10.0))
        r = bench_one(f"  n={n:4d} synthesize", lambda: engine._synthesize(k), warmup=1, repeat=3)
        results.append(r)
        print(f"  n={n:4d} synthesize    avg={r['avg_ms']:8.2f}ms  best={r['best_ms']:8.2f}ms")

    # =========================================================================
    # 4. 各ソルバーの E2E ベンチマーク
    # =========================================================================
    print("\n--- 4. End-to-end solve ---")
    for n in sizes:
        s = _make_surprises(n)
        g = _make_graph(n)
        h_arr = np.array([float(np.random.default_rng(42).integers(0, 10)) for _ in range(n)])

        # topk (linear only)
        def run_topk():
            e = UniversalDharmaEngine(n)
            e.add(DukkhaTerm(s))
            e.add(KarmaTerm(h_arr))
            return e.synthesize_and_solve(k=k)

        r = bench_one(f"  n={n:4d} topk", run_topk, warmup=1, repeat=3)
        results.append(r)
        print(f"  n={n:4d} topk          avg={r['avg_ms']:8.2f}ms  best={r['best_ms']:8.2f}ms")

        # submodular_greedy
        def run_submod():
            e = UniversalDharmaEngine(n)
            e.add(DukkhaTerm(s))
            e.add(MettaTerm(g, weight=0.5))
            return e.synthesize_and_solve(k=k)

        r = bench_one(f"  n={n:4d} submod_greedy", run_submod, warmup=1, repeat=3)
        results.append(r)
        print(f"  n={n:4d} submod_greedy avg={r['avg_ms']:8.2f}ms  best={r['best_ms']:8.2f}ms")

        # supermodular_warmstart
        def run_super():
            e = UniversalDharmaEngine(n, sa_iterations=500)
            e.add(PrajnaTerm(s))
            e.add(KarunaTerm(g, weight=0.5))
            return e.synthesize_and_solve(k=k)

        r = bench_one(f"  n={n:4d} super_warm", run_super, warmup=1, repeat=3)
        results.append(r)
        print(f"  n={n:4d} super_warm    avg={r['avg_ms']:8.2f}ms  best={r['best_ms']:8.2f}ms")

        # ising_sa (frustrated)
        def run_ising():
            e = UniversalDharmaEngine(n, sa_iterations=500)
            e.add(PrajnaTerm(s))
            e.add(KarunaTerm(g, weight=0.5))
            e.add(SilaTerm(k=k, weight=10.0))
            return e.synthesize_and_solve(k=k)

        r = bench_one(f"  n={n:4d} ising_sa", run_ising, warmup=1, repeat=3)
        results.append(r)
        print(f"  n={n:4d} ising_sa      avg={r['avg_ms']:8.2f}ms  best={r['best_ms']:8.2f}ms")

    # =========================================================================
    # 5. _csr_scale_data vs toarray ベンチマーク
    # =========================================================================
    print("\n--- 5. _csr_scale_data micro ---")
    for n in sizes:
        g = _make_graph(n)
        r = bench_one(f"  n={n:4d} csr_scale", lambda: _csr_scale_data(g, -2.0), warmup=1, repeat=5)
        results.append(r)
        print(f"  n={n:4d} csr_scale     avg={r['avg_ms']:8.2f}ms  best={r['best_ms']:8.2f}ms")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("Summary: hotspots for n=500")
    print("=" * 70)
    for r in results:
        if "n= 500" in r["label"] or "n=500" in r["label"]:
            print(f"  {r['label']:30s}  {r['avg_ms']:8.2f}ms")

    print("\n" + "=" * 70)
    print("Summary: scaling test n=2000")
    print("=" * 70)
    for r in results:
        if "n=2000" in r["label"]:
            print(f"  {r['label']:30s}  {r['avg_ms']:8.2f}ms")


if __name__ == "__main__":
    main()
