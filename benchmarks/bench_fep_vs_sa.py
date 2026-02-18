"""ベンチマーク — FEP ≡ KCL vs Ising SA 完全対決

問題スケール、グラフ密度、精度コンダクタンスを網羅的に走査し、
両ソルバーの最適化品質・レイテンシ・収束特性を定量比較する。
"""

import math
import time

import numpy as np
from scipy import sparse


class Timer:
    def __init__(self):
        self.elapsed_ms = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000


def run_benchmark():
    from lmm.dharma.engine import UniversalDharmaEngine
    from lmm.dharma.energy import PrajnaTerm, KarunaTerm, MettaTerm, SilaTerm
    from lmm.dharma.algorithms import build_sparse_impact_graph
    from lmm.dharma.fep import solve_fep_kcl

    print("=" * 78)
    print("  FEP ≡ KCL vs Ising SA  — 完全ベンチマーク")
    print("  ソフトウェア定義ニューロモルフィック回路 vs 確率的スピングラス")
    print("=" * 78)

    # =================================================================
    # 1. スケール比較: n=20,50,100,200  k=n/5
    # =================================================================
    print("\n" + "─" * 78)
    print("  1. スケール比較 — 問題サイズ vs 品質・速度")
    print("─" * 78)
    print(f"  {'n':>5s}  {'k':>3s}  {'nnz':>6s}  │ {'FEP energy':>11s} {'FEP ms':>8s}  │ {'SA energy':>10s} {'SA ms':>7s}  │ {'ratio':>6s}")
    print(f"  {'─'*5}  {'─'*3}  {'─'*6}  │ {'─'*11} {'─'*8}  │ {'─'*10} {'─'*7}  │ {'─'*6}")

    scale_results = []
    for n in [20, 50, 100, 200]:
        k = max(3, n // 5)
        rng = np.random.default_rng(42)
        surprises = rng.exponential(1.0, size=n)
        embeddings = rng.normal(0, 1, size=(n, 32))
        graph = build_sparse_impact_graph(embeddings, k=min(10, n - 1), use_hnswlib=True)

        # FEP
        with Timer() as t_fep:
            engine = UniversalDharmaEngine(n, solver="fep")
            engine.add(PrajnaTerm(surprises, weight=1.0))
            engine.add(KarunaTerm(graph, weight=0.5))
            engine.add(SilaTerm(k=k, weight=10.0))
            r_fep = engine.synthesize_and_solve(k=k)

        # Ising SA
        with Timer() as t_sa:
            engine2 = UniversalDharmaEngine(n, sa_iterations=5000)
            engine2.add(PrajnaTerm(surprises, weight=1.0))
            engine2.add(KarunaTerm(graph, weight=0.5))
            engine2.add(SilaTerm(k=k, weight=10.0))
            r_sa = engine2.synthesize_and_solve(k=k)

        ratio = r_fep.energy / r_sa.energy if r_sa.energy != 0 else 1.0
        nnz = graph.nnz
        scale_results.append((n, k, nnz, r_fep.energy, t_fep.elapsed_ms,
                              r_sa.energy, t_sa.elapsed_ms, ratio))

        print(f"  {n:5d}  {k:3d}  {nnz:6d}  │ {r_fep.energy:11.2f} {t_fep.elapsed_ms:7.1f}ms  │ {r_sa.energy:10.2f} {t_sa.elapsed_ms:6.1f}ms  │ {ratio:6.3f}")

    # =================================================================
    # 2. 混合問題: Metta(多様性) + Karuna(相乗) + Sila(基数)
    # =================================================================
    print("\n" + "─" * 78)
    print("  2. 混合問題 — submodular + supermodular + frustrated")
    print("─" * 78)
    print(f"  {'config':>20s}  │ {'FEP energy':>11s} {'FEP ms':>8s}  │ {'SA energy':>10s} {'SA ms':>7s}  │ {'ratio':>6s}")
    print(f"  {'─'*20}  │ {'─'*11} {'─'*8}  │ {'─'*10} {'─'*7}  │ {'─'*6}")

    n, k = 50, 8
    rng = np.random.default_rng(42)
    surprises = rng.exponential(1.0, size=n)
    embeddings = rng.normal(0, 1, size=(n, 32))
    impact_graph = build_sparse_impact_graph(embeddings, k=10, use_hnswlib=True)

    # 多様性グラフ (類似度が正 → ペナルティ)
    sim_matrix = embeddings @ embeddings.T
    # shim互換: np.min がないので手動で最小値を求める
    sim_flat = sim_matrix.flatten()
    sim_min = float(sim_flat[0])
    for v in sim_flat:
        if float(v) < sim_min:
            sim_min = float(v)
    sim_matrix = sim_matrix - sim_min  # 非負化
    for i in range(n):
        sim_matrix[i, i] = 0.0
    # スパースに変換 (上位k近傍のみ保持して計算量を抑える)
    rows, cols, vals = [], [], []
    for i in range(n):
        for j in range(n):
            v = float(sim_matrix[i, j])
            if i != j and v > 0.01:
                rows.append(i)
                cols.append(j)
                vals.append(v)
    diversity_graph = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))

    configs = [
        ("Prajna+Sila", [
            PrajnaTerm(surprises, weight=1.0),
            SilaTerm(k=k, weight=10.0),
        ]),
        ("Prajna+Karuna+Sila", [
            PrajnaTerm(surprises, weight=1.0),
            KarunaTerm(impact_graph, weight=0.5),
            SilaTerm(k=k, weight=10.0),
        ]),
        ("Prajna+Metta+Sila", [
            PrajnaTerm(surprises, weight=1.0),
            MettaTerm(diversity_graph, weight=0.3),
            SilaTerm(k=k, weight=10.0),
        ]),
        ("Full (all terms)", [
            PrajnaTerm(surprises, weight=1.0),
            KarunaTerm(impact_graph, weight=0.5),
            MettaTerm(diversity_graph, weight=0.3),
            SilaTerm(k=k, weight=10.0),
        ]),
    ]

    for name, terms in configs:
        with Timer() as t_fep:
            engine = UniversalDharmaEngine(n, solver="fep")
            for t in terms:
                engine.add(t)
            r_fep = engine.synthesize_and_solve(k=k)

        with Timer() as t_sa:
            engine2 = UniversalDharmaEngine(n, sa_iterations=5000)
            for t in terms:
                engine2.add(t)
            r_sa = engine2.synthesize_and_solve(k=k)

        ratio = r_fep.energy / r_sa.energy if r_sa.energy != 0 else 1.0
        print(f"  {name:>20s}  │ {r_fep.energy:11.2f} {t_fep.elapsed_ms:7.1f}ms  │ {r_sa.energy:10.2f} {t_sa.elapsed_ms:6.1f}ms  │ {ratio:6.3f}")

    # =================================================================
    # 3. G_prec 感度分析
    # =================================================================
    print("\n" + "─" * 78)
    print("  3. G_prec 感度分析 — 精度コンダクタンスの影響 (n=50, k=10)")
    print("─" * 78)
    print(f"  {'G_prec':>7s}  │ {'energy':>10s}  {'steps':>6s}  {'ms':>7s}  │ {'P_init':>12s}  {'P_final':>12s}")
    print(f"  {'─'*7}  │ {'─'*10}  {'─'*6}  {'─'*7}  │ {'─'*12}  {'─'*12}")

    n, k = 50, 10
    rng = np.random.default_rng(42)
    surprises = rng.exponential(1.0, size=n)
    embeddings = rng.normal(0, 1, size=(n, 32))
    graph = build_sparse_impact_graph(embeddings, k=10, use_hnswlib=True)

    # 合成して h, J を取得
    engine_tmp = UniversalDharmaEngine(n, solver="fep")
    engine_tmp.add(PrajnaTerm(surprises, weight=1.0))
    engine_tmp.add(KarunaTerm(graph, weight=0.5))
    engine_tmp.add(SilaTerm(k=k, weight=10.0))
    total_h, total_J, properties, sila_gamma = engine_tmp._synthesize(k)

    from lmm._compat import sparse_matvec as _sparse_matvec, sparse_dot as _sparse_dot

    for G in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
        with Timer() as t:
            V_mu, x_final, steps, power_hist = solve_fep_kcl(
                h=total_h, J=total_J, k=k, n=n,
                sila_gamma=sila_gamma, G_prec=G,
                max_steps=500, nirvana_threshold=1e-4,
                sparse_matvec=_sparse_matvec,
            )
            best_x = engine_tmp._project_to_k_sparse(x_final, total_h, total_J, sila_gamma, k)
            sel = np.where(best_x > 0.5)[0][:k]
            energy = engine_tmp._evaluate_energy_sparse(total_h, total_J, sila_gamma, sel)

        p_init = power_hist[0] if power_hist else 0
        p_final = power_hist[-1] if power_hist else 0
        print(f"  {G:7.1f}  │ {energy:10.2f}  {steps:6d}  {t.elapsed_ms:6.1f}ms  │ {p_init:12.1f}  {p_final:12.4f}")

    # =================================================================
    # 4. 収束曲線の可視化 (ASCII)
    # =================================================================
    print("\n" + "─" * 78)
    print("  4. 収束曲線 — 散逸電力 P_err の時間発展 (G_prec=5.0)")
    print("─" * 78)

    _, _, steps, power_hist = solve_fep_kcl(
        h=total_h, J=total_J, k=k, n=n,
        sila_gamma=sila_gamma, G_prec=5.0,
        max_steps=500, nirvana_threshold=1e-6,
        sparse_matvec=_sparse_matvec,
    )

    # 対数スケールの ASCII プロット
    if power_hist:
        log_powers = [math.log10(max(p, 1e-10)) for p in power_hist]
        p_max = max(log_powers)
        p_min = min(log_powers)
        width = 60

        # 20ステップごとにサンプリング
        sample_interval = max(1, len(power_hist) // 25)
        print(f"  step  │ log10(P_err)")
        print(f"  {'─'*5} │ {'─'*width}")
        for i in range(0, len(power_hist), sample_interval):
            lp = log_powers[i]
            if p_max > p_min:
                bar_len = int((lp - p_min) / (p_max - p_min) * width)
            else:
                bar_len = width // 2
            bar_len = max(1, bar_len)
            print(f"  {i:5d} │ {'█' * bar_len} {lp:.1f}")

        print(f"\n  収束: {steps} steps, P_init={power_hist[0]:.1f} → P_final={power_hist[-1]:.4f}")

    # =================================================================
    # 5. 決定性 vs 確率性
    # =================================================================
    print("\n" + "─" * 78)
    print("  5. 再現性テスト — 5回実行の分散")
    print("─" * 78)

    n, k = 50, 10

    fep_energies = []
    sa_energies = []

    for trial in range(5):
        engine = UniversalDharmaEngine(n, solver="fep")
        engine.add(PrajnaTerm(surprises, weight=1.0))
        engine.add(KarunaTerm(graph, weight=0.5))
        engine.add(SilaTerm(k=k, weight=10.0))
        r = engine.synthesize_and_solve(k=k)
        fep_energies.append(r.energy)

        engine2 = UniversalDharmaEngine(n, sa_iterations=5000)
        engine2.add(PrajnaTerm(surprises, weight=1.0))
        engine2.add(KarunaTerm(graph, weight=0.5))
        engine2.add(SilaTerm(k=k, weight=10.0))
        r2 = engine2.synthesize_and_solve(k=k)
        sa_energies.append(r2.energy)

    fep_mean = sum(fep_energies) / len(fep_energies)
    sa_mean = sum(sa_energies) / len(sa_energies)

    fep_var = sum((e - fep_mean) ** 2 for e in fep_energies) / len(fep_energies)
    sa_var = sum((e - sa_mean) ** 2 for e in sa_energies) / len(sa_energies)

    fep_std = math.sqrt(fep_var)
    sa_std = math.sqrt(sa_var)

    print(f"  FEP: energies = {['%.2f' % e for e in fep_energies]}")
    print(f"        mean = {fep_mean:.2f}, std = {fep_std:.4f}")
    print(f"  SA:  energies = {['%.2f' % e for e in sa_energies]}")
    print(f"        mean = {sa_mean:.2f}, std = {sa_std:.4f}")
    print(f"\n  FEP 分散 = {fep_var:.6f}  (決定的 → 理論上 0)")
    print(f"  SA  分散 = {sa_var:.6f}  (確率的 → > 0)")

    # =================================================================
    # 6. 選択の重複率 — 両ソルバーの合意度
    # =================================================================
    print("\n" + "─" * 78)
    print("  6. 選択合意度 — FEP と SA が同じアイテムを選ぶ割合")
    print("─" * 78)

    n, k = 50, 10
    overlap_total = 0
    n_trials = 5

    for trial in range(n_trials):
        rng_t = np.random.default_rng(100 + trial)
        surp_t = rng_t.exponential(1.0, size=n)
        emb_t = rng_t.normal(0, 1, size=(n, 32))
        graph_t = build_sparse_impact_graph(emb_t, k=10, use_hnswlib=True)

        engine = UniversalDharmaEngine(n, solver="fep")
        engine.add(PrajnaTerm(surp_t, weight=1.0))
        engine.add(KarunaTerm(graph_t, weight=0.5))
        engine.add(SilaTerm(k=k, weight=10.0))
        r_fep = engine.synthesize_and_solve(k=k)

        engine2 = UniversalDharmaEngine(n, sa_iterations=5000)
        engine2.add(PrajnaTerm(surp_t, weight=1.0))
        engine2.add(KarunaTerm(graph_t, weight=0.5))
        engine2.add(SilaTerm(k=k, weight=10.0))
        r_sa = engine2.synthesize_and_solve(k=k)

        fep_set = set(int(i) for i in r_fep.selected_indices)
        sa_set = set(int(i) for i in r_sa.selected_indices)
        overlap = len(fep_set & sa_set)
        overlap_total += overlap
        print(f"  Trial {trial+1}: FEP={sorted(fep_set)}")
        print(f"          SA ={sorted(sa_set)}")
        print(f"          overlap = {overlap}/{k} ({100*overlap/k:.0f}%)")

    avg_overlap = overlap_total / n_trials
    print(f"\n  平均合意度: {avg_overlap:.1f}/{k} ({100*avg_overlap/k:.0f}%)")

    # =================================================================
    # 総括
    # =================================================================
    print("\n" + "=" * 78)
    print("  総括")
    print("=" * 78)

    print(f"\n  スケーリング:")
    for n, k, nnz, e_fep, t_fep, e_sa, t_sa, ratio in scale_results:
        winner = "FEP" if ratio >= 0.99 else "SA"
        speed = "FEP" if t_fep < t_sa else "SA"
        print(f"    n={n:3d}: 品質={winner} (ratio={ratio:.3f}), 速度={speed} (FEP {t_fep:.0f}ms vs SA {t_sa:.0f}ms)")

    print(f"\n  FEP の特性:")
    print(f"    - 決定的: std = {fep_std:.6f} (SA: {sa_std:.4f})")
    print(f"    - 温度スケジュール不要")
    print(f"    - 乱数不要")
    print(f"    - 涅槃 (P_err < ε) による自動停止")
    print("=" * 78)


if __name__ == "__main__":
    run_benchmark()
