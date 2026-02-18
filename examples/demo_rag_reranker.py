"""DharmaReranker デモ — 仏教哲学駆動型 RAG リランカー

3つの Bridge を統合し、Vector DB の素朴な Top-N 検索結果を
「互いに補完し合う最高の Top-K サンガ」へ昇華させるデモ。

カスケード・アーキテクチャ:
  声聞 (Vector DB)  →  縁覚 (動的グラフ)  →  菩薩 (Dharma Engine)
    数十ms                数ms                 ~140ms

使い方:
    PYTHONPATH=. python examples/demo_rag_reranker.py
"""

import time

import numpy as np

from lmm.dharma.reranker import DharmaReranker, IntentAwareRouter
from lmm.dharma.algorithms import build_sparse_impact_graph


def cosine_top_n(query, corpus, n):
    """声聞フェーズ: コサイン類似度による粗選別 (FAISS/Qdrant の代替)"""
    q = query.flatten()
    q_norm = float(np.linalg.norm(q))
    if q_norm < 1e-10:
        return np.arange(min(n, len(corpus)))
    c_norms = np.linalg.norm(corpus, axis=1)
    c_norms = np.clip(c_norms, 1e-10, None)
    scores = (corpus @ q) / (c_norms * q_norm)
    # argsort の上位 N 件を返す
    ranked = np.argsort(scores)
    return np.array([int(ranked[i]) for i in range(len(ranked) - 1, max(len(ranked) - n - 1, -1), -1)])


def generate_clustered_corpus(n_clusters, chunks_per_cluster, dim, rng):
    """テスト用: クラスタ構造を持つ疑似コーパスを生成

    各クラスタは特定の「トピック方向」を持ち、その周辺にチャンクが分布する。
    現実の RAG コーパスで起きる「同一トピックの冗長チャンク」問題を再現。
    """
    embeddings = []
    labels = []
    for c in range(n_clusters):
        # クラスタ中心: ランダムな方向ベクトル
        center = rng.normal(0, 1, size=(dim,))
        center_norm = float(np.linalg.norm(center))
        if center_norm > 1e-10:
            center = center / center_norm
        for _ in range(chunks_per_cluster):
            noise = rng.normal(0, 0.15, size=(dim,))
            vec = center + noise
            embeddings.append([float(x) for x in vec])
            labels.append(c)
    return np.array(embeddings), labels


def cluster_diversity(selected_indices, labels, n_clusters):
    """選択されたチャンクが何クラスタをカバーしているか"""
    covered = set()
    for idx in selected_indices:
        covered.add(labels[int(idx)])
    return len(covered), n_clusters


def main():
    rng = np.random.default_rng(42)

    # === パラメータ ===
    n_clusters = 20         # トピック数
    chunks_per_cluster = 50  # 各トピックのチャンク数
    dim = 64                 # 埋め込み次元
    n_fetch = 200            # Vector DB からフェッチする件数
    k_final = 10             # 最終選択数

    n_total = n_clusters * chunks_per_cluster

    print("=" * 65)
    print("  DharmaReranker — 仏教哲学駆動型 RAG リランカー デモ")
    print("=" * 65)
    print()
    print(f"  コーパス: {n_total} チャンク ({n_clusters} トピック × {chunks_per_cluster} 件)")
    print(f"  埋め込み次元: {dim}")
    print(f"  Vector DB フェッチ: Top-{n_fetch}")
    print(f"  最終選択: Top-{k_final}")
    print()

    # === Step 0: コーパス生成 ===
    print("--- Step 0: コーパス生成 ---")
    corpus_emb, labels = generate_clustered_corpus(
        n_clusters, chunks_per_cluster, dim, rng
    )
    print(f"  {n_total} チャンクの埋め込みを生成")
    print()

    # === Step 1: オフライン修行 (静的グラフ構築) ===
    print("--- Step 1: オフライン修行 (fit_offline) ---")

    # 2つのモードを比較:
    #   サンガモード: Karuna (超モジュラ相乗効果) 主導
    #   サンガ+慈愛モード: Karuna + Metta (劣モジュラ多様性) の二諦
    reranker_sangha = DharmaReranker(
        k=k_final,
        alpha=1.0,              # QueryDukkha
        beta=0.5,               # Karuna (相乗効果)
        gamma=10.0,
        diversity_weight=0.0,   # Metta 無効
        query_weight=0.7,
        corpus_weight=0.3,
        sa_iterations=1000,
        sparse_k=15,
    )
    reranker_metta = DharmaReranker(
        k=k_final,
        alpha=1.0,
        beta=0.3,               # Karuna (やや弱め)
        gamma=10.0,
        diversity_weight=0.8,   # Metta 有効 — 劣モジュラ多様性
        query_weight=0.7,
        corpus_weight=0.3,
        sa_iterations=1000,
        sparse_k=15,
    )

    t0 = time.perf_counter()
    reranker_sangha.fit_offline(corpus_emb)
    t_offline = time.perf_counter() - t0

    # 静的グラフを共有 (同一コーパス)
    reranker_metta._corpus_embeddings = reranker_sangha._corpus_embeddings
    reranker_metta.J_static = reranker_sangha.J_static

    print(f"  静的グラフ構築: {t_offline:.3f}秒")
    print(f"  グラフ非ゼロ要素: {reranker_sangha.J_static.nnz}")
    print(f"  スパース率: {1.0 - reranker_sangha.J_static.nnz / (n_total * n_total):.6f}")
    print()

    # === Step 2: クエリ到着 (シミュレーション) ===
    query_cluster = 5
    query = corpus_emb[query_cluster * chunks_per_cluster]
    query_noise = rng.normal(0, 0.3, size=(dim,))
    query = query + query_noise

    print("--- Step 2: クエリ到着 ---")
    print(f"  クエリ方向: クラスタ {query_cluster} 近傍 (ノイズ付き)")
    print()

    # === Step 3: 声聞フェーズ ===
    print("--- Step 3: 声聞フェーズ (粗選別) ---")
    t0 = time.perf_counter()
    fetched_idx = cosine_top_n(query, corpus_emb, n_fetch)
    t_fetch = time.perf_counter() - t0

    fetched_clusters = [labels[int(i)] for i in fetched_idx]
    unique_fetched = len(set(fetched_clusters))
    print(f"  Top-{n_fetch} フェッチ: {t_fetch * 1000:.1f}ms")
    print(f"  フェッチされたクラスタ数: {unique_fetched}/{n_clusters}")
    print()

    # === Step 4: 縁覚+菩薩フェーズ ===
    print("--- Step 4: 縁覚+菩薩フェーズ (rerank_online) ---")
    print()

    # (A) サンガモード (Karuna 主導)
    t0 = time.perf_counter()
    result_sangha = reranker_sangha.rerank_online(
        query_embedding=query, fetched_indices=fetched_idx,
    )
    t_sangha = time.perf_counter() - t0

    # (B) 慈愛モード (Metta 主導)
    t0 = time.perf_counter()
    result_metta = reranker_metta.rerank_online(
        query_embedding=query, fetched_indices=fetched_idx,
    )
    t_metta = time.perf_counter() - t0

    print(f"  (A) サンガモード: {t_sangha * 1000:.1f}ms, ソルバー={result_sangha.solver_used}")
    print(f"  (B) 慈愛モード:   {t_metta * 1000:.1f}ms, ソルバー={result_metta.solver_used}")
    print()

    # === Step 5: 結果分析 ===
    def show_result(label, indices):
        covered, total = cluster_diversity(indices, labels, n_clusters)
        clusters = [labels[int(i)] for i in indices]
        counts = {}
        for c in clusters:
            counts[c] = counts.get(c, 0) + 1
        print(f"  [{label}]")
        print(f"    カバークラスタ数: {covered}/{total}")
        print(f"    クラスタ分布: {dict(sorted(counts.items()))}")
        return covered

    print("=" * 65)
    print("  結果分析: 三者比較")
    print("=" * 65)
    print()

    naive_topk = fetched_idx[:k_final]
    c_naive = show_result("素朴な Top-K (コサイン類似度のみ)", naive_topk)
    print()
    c_sangha = show_result("DharmaReranker サンガモード (Karuna=0.5, Metta=0)", result_sangha.original_indices)
    print(f"    関連度: [{', '.join(f'{float(s):.3f}' for s in result_sangha.relevance_scores)}]")
    print()
    c_metta = show_result("DharmaReranker 慈愛モード (Karuna=0.3, Metta=0.8)", result_metta.original_indices)
    print(f"    関連度: [{', '.join(f'{float(s):.3f}' for s in result_metta.relevance_scores)}]")
    print()

    print("  --- 多様性比較 ---")
    print(f"    素朴 Top-K:      {c_naive} クラスタ")
    print(f"    サンガモード:    {c_sangha} クラスタ (相乗効果最大化)")
    print(f"    慈愛モード:      {c_metta} クラスタ (多様性最大化)")
    print()
    if c_metta > c_naive:
        print(f"  慈愛モード = +{c_metta - c_naive} クラスタの多様性改善!")
        print(f"  「Lost in the Middle」問題を Metta (慈愛) で克服。")
    print()

    print("  --- レイテンシ ---")
    print(f"    声聞 (粗選別):      {t_fetch * 1000:>8.1f} ms")
    print(f"    サンガモード:       {t_sangha * 1000:>8.1f} ms")
    print(f"    慈愛モード:         {t_metta * 1000:>8.1f} ms")
    print()

    # =====================================================================
    # Step 6: IntentAwareRouter — 中道自動調心
    # =====================================================================
    print("=" * 65)
    print("  IntentAwareRouter — 中道 (Madhyamaka) 自動調心")
    print("=" * 65)
    print()
    print("  手動で Karuna/Metta を指定する代わりに、")
    print("  スコア分布からクエリの意図を自動判定し、")
    print("  最適なブレンド比を算出する。")
    print()

    router = IntentAwareRouter(
        k=k_final,
        beta_range=(0.1, 0.8),
        metta_range=(0.0, 1.0),
        sa_iterations=1000,
        sparse_k=15,
    )
    # 静的グラフを共有 (再構築不要)
    router._corpus_embeddings = reranker_sangha._corpus_embeddings
    router._J_static = reranker_sangha.J_static

    # --- (A) 集中型クエリ: 特定クラスタに近い ---
    print("  (A) 集中型クエリ: クラスタ5 の中心に近い方向")
    t0 = time.perf_counter()
    result_a, diag_a = router.route(
        query_embedding=query, fetched_indices=fetched_idx,
    )
    t_a = time.perf_counter() - t0
    c_a = show_result(f"IntentAwareRouter [{diag_a.intent}]", result_a.original_indices)
    print(f"    診断: 集中度={diag_a.concentration:.3f}, "
          f"エントロピー={diag_a.entropy:.3f}")
    print(f"    自動決定: Karuna={diag_a.beta_used:.3f}, Metta={diag_a.metta_used:.3f}")
    print(f"    時間: {t_a * 1000:.1f}ms")
    print()

    # --- (B) 探索型クエリ: 全方向に等しく向く ---
    query_explore = rng.normal(0, 0.5, size=(dim,))
    fetched_explore = cosine_top_n(query_explore, corpus_emb, n_fetch)

    print("  (B) 探索型クエリ: ランダム方向 (全トピックに等距離)")
    t0 = time.perf_counter()
    result_b, diag_b = router.route(
        query_embedding=query_explore, fetched_indices=fetched_explore,
    )
    t_b = time.perf_counter() - t0
    c_b = show_result(f"IntentAwareRouter [{diag_b.intent}]", result_b.original_indices)
    print(f"    診断: 集中度={diag_b.concentration:.3f}, "
          f"エントロピー={diag_b.entropy:.3f}")
    print(f"    自動決定: Karuna={diag_b.beta_used:.3f}, Metta={diag_b.metta_used:.3f}")
    print(f"    時間: {t_b * 1000:.1f}ms")
    print()

    print("  --- 自動調心の効果 ---")
    print(f"    集中型 → 意図={diag_a.intent}: "
          f"Karuna={diag_a.beta_used:.2f}, Metta={diag_a.metta_used:.2f} "
          f"→ {c_a} クラスタ")
    print(f"    探索型 → 意図={diag_b.intent}: "
          f"Karuna={diag_b.beta_used:.2f}, Metta={diag_b.metta_used:.2f} "
          f"→ {c_b} クラスタ")
    print()
    print("  人間がパラメータを指定する必要は、もうありません。")
    print("  中道 (Madhyamaka) がクエリの性質を観て、自ら調心します。")
    print()

    # =====================================================================
    print("=" * 65)
    print("  仏教哲学の設計原理")
    print("=" * 65)
    print()
    print("  Karuna (慈悲/超モジュラ): 選べば選ぶほど調和が加速")
    print("    → 高い関連度のクラスタ内で「最強チーム」を形成")
    print()
    print("  Metta (慈愛/劣モジュラ): 類似ペアの同時選択にペナルティ")
    print("    → 異なるクラスタから代表を1つずつ選び、全体を網羅")
    print()
    print("  中道 (Madhyamaka): スコア分布の集中度 (CV + Gini) を観て")
    print("    Karuna と Metta のブレンド比を自動で決定する")
    print("    → 人間の介入なしに、クエリの意図に最適な戦略を選択")
    print()

    print("=" * 65)
    print("  デモ完了")
    print("=" * 65)


if __name__ == "__main__":
    main()
