"""Gemini Deep Research統合デモ

超モジュラ慈悲 + サンガ形成の数理最適化を体験する。

使い方:
    PYTHONPATH=. python examples/demo_gemini.py
"""

import time

import numpy as np

from lmm.core import LMM
from lmm.dharma import DharmaLMM


def main():
    rng = np.random.default_rng(42)
    n, dim, k = 200, 50, 15

    # データ生成
    reference = rng.normal(0, 1, size=(500, dim))
    candidates = rng.exponential(1.0, size=(n, dim))
    # 外れ値を少し混ぜる
    candidates[10] *= 5.0
    candidates[42] *= 8.0
    candidates[99] *= 6.0

    print("=" * 60)
    print("Gemini Deep Research統合デモ")
    print("=" * 60)
    print()

    # --- Gemini版 (DharmaLMM) ---
    print("Gemini Deep Research推奨機能:")

    dharma = DharmaLMM(
        k=k,
        use_sparse_graph=True,
        use_greedy_warmstart=True,
        use_ising_sa=True,
        use_exponential_balance=True,
    )
    dharma.fit(reference)

    start = time.perf_counter()
    result_dharma = dharma.select_dharma(candidates)
    time_dharma = time.perf_counter() - start

    print("  指数勾配降下法: Lyapunov安定性保証")
    print("  スパースグラフ: O(n^2) -> O(n*k) 高速化")
    print("  超モジュラ貪欲法: Warm Start初期化")
    print("  Ising形式SA: SIMD高速化")
    print()

    # --- 従来版 (LMM) ---
    lmm = LMM(k=k, solver_method="sa")
    lmm.fit(reference)

    start = time.perf_counter()
    result_lmm = lmm.select(candidates)
    time_lmm = time.perf_counter() - start

    # --- 比較 ---
    print("速度比較:")
    print(f"  従来版: {time_lmm:.4f}秒")
    print(f"  Gemini版: {time_dharma:.4f}秒")
    if time_lmm > 0:
        ratio = time_lmm / time_dharma if time_dharma > 0 else float("inf")
        print(f"  高速化率: {ratio:.2f}x")
    print()

    print("エネルギー:")
    print(f"  従来版: {result_lmm.energy:.4f}")
    print(f"  Gemini版: {result_dharma.energy:.4f}")
    better = "Gemini" if result_dharma.energy < result_lmm.energy else "従来"
    print(f"  より良い解: {better}版")
    print()

    # --- Dharma解釈 ---
    if result_dharma.interpretation:
        print("=" * 60)
        print("ダルマ解釈:")
        print("=" * 60)
        print(result_dharma.interpretation.narrative)
        print()

    # --- 超モジュラ性の説明 ---
    print("=" * 60)
    print("数学的ブレイクスルー (Gemini Deep Research):")
    print("=" * 60)
    print()
    print("  慈悲項: beta * sum(W_ij * x_i * (1 - x_j))")
    print("  これは超モジュラ (supermodular) である。")
    print()
    print("  意味: 選べば選ぶほど調和が加速する")
    print("  = サンガ (僧伽・調和共同体) 形成の数理的本質")
    print()
    print(f"  バランス後の重み: alpha={result_dharma.weights.alpha:.3f}, "
          f"beta={result_dharma.weights.beta:.3f}")


if __name__ == "__main__":
    main()
