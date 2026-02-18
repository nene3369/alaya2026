"""ベンチマーク — FEP ソルバーのハードウェア適性分析

FEP ODE と Ising SA の並列性を演算レベルで比較し、
GPU/CPU SIMD 上での理論スループットを推定する。
"""

import math
import time

import numpy as np


class Timer:
    def __init__(self):
        self.elapsed_ms = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000


def run_hardware_benchmark():
    from lmm._compat import (
        IS_SHIM, HAS_FEP_FAST_PATH, HAS_CUPY, HAS_JAX,
        HAS_VECTORIZED_TANH, HAS_SPARSE_MATMUL,
    )

    print("=" * 78)
    print("  FEP ソルバー — ハードウェア適性分析")
    print("=" * 78)

    # =================================================================
    # 1. ハードウェア検出
    # =================================================================
    print("\n  1. 検出されたバックエンド")
    print("  " + "─" * 60)
    print(f"  NumPy shim:          {'YES (Python fallback)' if IS_SHIM else 'NO (native NumPy)'}")
    print(f"  np.tanh vectorized:  {'YES' if HAS_VECTORIZED_TANH else 'NO'}")
    print(f"  scipy.sparse @:      {'YES' if HAS_SPARSE_MATMUL else 'NO'}")
    print(f"  CuPy (NVIDIA GPU):   {'YES' if HAS_CUPY else 'NO'}")
    print(f"  JAX:                 {'YES' if HAS_JAX else 'NO'}")
    print(f"  FEP fast path:       {'YES' if HAS_FEP_FAST_PATH else 'NO'}")

    selected = "shim (Python loop)"
    if HAS_CUPY:
        selected = "GPU (CuPy/CUDA)"
    elif HAS_FEP_FAST_PATH:
        selected = "CPU BLAS (vectorized)"
    print(f"\n  → 自動選択: {selected}")

    # =================================================================
    # 2. 演算レベルの並列性比較
    # =================================================================
    print("\n  2. 演算レベルの並列性比較")
    print("  " + "─" * 60)
    print()
    print("  ┌─────────────────────────┬──────────────────┬──────────────────┐")
    print("  │ 演算                    │ FEP ODE          │ Ising SA         │")
    print("  ├─────────────────────────┼──────────────────┼──────────────────┤")
    print("  │ 基本操作                │ tanh, +, *, SpMV │ RNG, exp, cmp    │")
    print("  │ 1ステップの並列度       │ O(n) 全ノード    │ O(1) 1スピン     │")
    print("  │ ステップ間依存          │ V_{t+1} = f(V_t) │ s_{t+1} = f(s_t) │")
    print("  │ ステップ数 (典型)       │ 300              │ 5000             │")
    print("  │ 分岐 (warp divergence)  │ なし             │ accept/reject    │")
    print("  │ 乱数生成               │ 不要             │ 毎ステップ       │")
    print("  │ 総演算量 (n=1000)       │ 300 * 1000       │ 5000 * 1        │")
    print("  │ GPU 効率               │ 極めて高い       │ 低い (逐次的)    │")
    print("  │ CPU SIMD 効率          │ 極めて高い       │ 中程度           │")
    print("  └─────────────────────────┴──────────────────┴──────────────────┘")

    # =================================================================
    # 3. 演算プロファイル (現環境)
    # =================================================================
    print("\n  3. 演算プロファイル — 各操作のレイテンシ (現環境)")
    print("  " + "─" * 60)

    from lmm.dharma.algorithms import build_sparse_impact_graph
    from lmm._compat import sparse_matvec as _sparse_matvec

    for n in [50, 200, 1000]:
        rng = np.random.default_rng(42)
        V = np.array([float(rng.normal()) for _ in range(n)])
        embeddings = rng.normal(0, 1, size=(n, 32))
        J = build_sparse_impact_graph(embeddings, k=min(10, n - 1), use_hnswlib=True)
        x = np.array([0.5] * n)

        # tanh (element-wise)
        with Timer() as t_tanh:
            for _ in range(100):
                g = np.array([float(math.tanh(float(v))) for v in V])
        tanh_us = t_tanh.elapsed_ms * 10  # per-call in microseconds

        # sparse matvec
        with Timer() as t_spmv:
            for _ in range(100):
                _sparse_matvec(J, x)
        spmv_us = t_spmv.elapsed_ms * 10

        # element-wise ops
        with Timer() as t_ew:
            for _ in range(100):
                _ = 1.0 - g * g
                _ = V + g * 5.0 * x
        ew_us = t_ew.elapsed_ms * 10

        # dot product
        with Timer() as t_dot:
            for _ in range(100):
                float(np.dot(x, x))
        dot_us = t_dot.elapsed_ms * 10

        total_us = tanh_us + spmv_us + ew_us + dot_us
        print(f"\n  n={n:4d} (nnz={J.nnz})")
        print(f"    tanh(V):    {tanh_us:8.1f} us/call ({100*tanh_us/total_us:4.1f}%)")
        print(f"    J @ x:      {spmv_us:8.1f} us/call ({100*spmv_us/total_us:4.1f}%)")
        print(f"    elem-wise:  {ew_us:8.1f} us/call ({100*ew_us/total_us:4.1f}%)")
        print(f"    dot(e,e):   {dot_us:8.1f} us/call ({100*dot_us/total_us:4.1f}%)")
        print(f"    ──────────────────────")
        print(f"    合計:       {total_us:8.1f} us/step")
        print(f"    300 step:   {total_us*300/1000:8.1f} ms")

    # =================================================================
    # 4. 理論加速率の推定
    # =================================================================
    print("\n  4. 理論加速率 — バックエンド別推定")
    print("  " + "─" * 60)
    print()
    print("  ┌──────────────┬──────────────────────┬────────────────┬──────────┐")
    print("  │ バックエンド │ 主要高速化           │ 予想加速率     │ 備考     │")
    print("  ├──────────────┼──────────────────────┼────────────────┼──────────┤")
    print("  │ shim (現在)  │ ─                    │ 1x (基準)      │ 動作確認 │")
    print("  │ NumPy BLAS   │ np.tanh, SIMD ufunc  │ 10-50x         │ CPU 最適 │")
    print("  │ CuPy (GPU)  │ cuSPARSE, CUDA kern  │ 100-500x       │ n>1000   │")
    print("  │ JAX          │ XLA JIT, GPU/TPU     │ 200-1000x      │ jit化    │")
    print("  │ ニューロ IC  │ 物理アナログ回路     │ 10^6x          │ 将来     │")
    print("  └──────────────┴──────────────────────┴────────────────┴──────────┘")
    print()
    print("  注: Ising SA は本質的に逐次 (1 spin/step) のため、")
    print("      GPU 加速は並列 SA (複数初期値) でしか効かない。")
    print("      FEP は 1 step 内で O(n) 並列 → GPU の本領発揮。")

    # =================================================================
    # 5. FEP vs SA: n スケーリング
    # =================================================================
    print("\n  5. n スケーリング — 問題サイズが大きいほど FEP が有利")
    print("  " + "─" * 60)

    from lmm.dharma.engine import UniversalDharmaEngine
    from lmm.dharma.energy import PrajnaTerm, KarunaTerm, SilaTerm

    print(f"\n  {'n':>5s}  │ {'FEP ms':>8s} {'SA ms':>8s}  │ {'FEP/SA':>7s}  │ {'GPU推定':>10s}")
    print(f"  {'─'*5}  │ {'─'*8} {'─'*8}  │ {'─'*7}  │ {'─'*10}")

    for n in [20, 50, 100, 200, 500]:
        k = max(3, n // 5)
        rng = np.random.default_rng(42)
        surprises = rng.exponential(1.0, size=n)
        embeddings = rng.normal(0, 1, size=(n, 32))
        graph = build_sparse_impact_graph(embeddings, k=min(10, n - 1), use_hnswlib=True)

        with Timer() as t_fep:
            engine = UniversalDharmaEngine(n, solver="fep")
            engine.add(PrajnaTerm(surprises, weight=1.0))
            engine.add(KarunaTerm(graph, weight=0.5))
            engine.add(SilaTerm(k=k, weight=10.0))
            engine.synthesize_and_solve(k=k)

        with Timer() as t_sa:
            engine2 = UniversalDharmaEngine(n, sa_iterations=5000)
            engine2.add(PrajnaTerm(surprises, weight=1.0))
            engine2.add(KarunaTerm(graph, weight=0.5))
            engine2.add(SilaTerm(k=k, weight=10.0))
            engine2.synthesize_and_solve(k=k)

        ratio = t_fep.elapsed_ms / t_sa.elapsed_ms
        # GPU推定: shim→BLAS で 10-50x, BLAS→GPU で 10-100x
        gpu_est_ms = t_fep.elapsed_ms / 50.0  # 保守的推定
        print(f"  {n:5d}  │ {t_fep.elapsed_ms:7.1f}ms {t_sa.elapsed_ms:7.1f}ms  │ {ratio:7.1f}x  │ {gpu_est_ms:8.1f}ms")

    # =================================================================
    # 6. Ising SA の並列化限界
    # =================================================================
    print("\n  6. なぜ Ising SA は GPU に向かないか")
    print("  " + "─" * 60)
    print("""
  Ising SA の 1 ステップ:
    1. idx = random_int(0, n)     ← RNG (GPU では高コスト)
    2. delta_e = -2*s[idx]*f[idx] ← ランダムアクセス
    3. if rand() < exp(-dE/T):    ← 分岐 (warp divergence!)
         s[idx] *= -1             ← 条件付き書込み
         update local_field       ← O(nnz_col) 逐次更新

  → 本質的に「1ステップ = 1スピン」の逐次アルゴリズム
  → GPU の数千コアのうち 1 コアしか使えない
  → 並列化は「複数の独立 SA を同時実行」でしか効かない

  FEP ODE の 1 ステップ:
    1. g = tanh(V)                ← n 並列 (1 GPU kernel)
    2. x = (g + 1) / 2           ← n 並列 (fused)
    3. Jx = J @ x                ← cuSPARSE SpMV (最適化済み)
    4. grad = h + 2*Jx + homeo   ← n 並列 (fused)
    5. dV = -V/τ + jac*G*(-grad) ← n 並列 (fused)
    6. V += dV * dt              ← n 並列 (fused)
    7. P = G * dot(err, err)     ← parallel reduction

  → 全演算が O(n) 並列 → GPU の数千コアをフル活用
  → 分岐なし、RNG なし → warp divergence ゼロ
  → ステップ 2-6 は kernel fusion で 1 launch に圧縮可能""")

    # =================================================================
    # 総括
    # =================================================================
    print("=" * 78)
    print("  総括: FEP ≡ KCL は GPU/CPU ハードウェアに最適化された設計")
    print("=" * 78)
    print("""
  FEP ODE がハードウェアに適する根本的理由:

  1. 物理回路のソフトウェア・エミュレーション
     → 回路は本質的に全ノード同時更新 → GPU の SIMT と同型

  2. 演算が全て element-wise + SpMV + reduction
     → GPU カーネルの基本3パターンに完全一致

  3. 分岐なし、乱数なし
     → warp divergence ゼロ、RNG オーバーヘッドゼロ

  4. ステップ数が少ない (300 vs SA の 5000)
     → カーネル起動オーバーヘッドが小さい

  5. 将来のニューロモルフィック IC で物理実行可能
     → ソフトウェア・エミュレータが不要になる究極形態""")
    print("=" * 78)


if __name__ == "__main__":
    run_hardware_benchmark()
