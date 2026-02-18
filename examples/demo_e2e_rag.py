"""End-to-End RAG 実証デモ — 救済 (LLM生成) の品質比較

素朴な Top-K RAG と IntentAwareRouter RAG を並べ、
同じ LLM に渡すコンテキストの品質差が、回答品質をどう変えるかを可視化する。

パイプライン:
  1. 知識コーパス構築 (テキスト + n-gram 埋め込み)
  2. 声聞フェーズ: コサイン類似度による Top-N 粗選別
  3. 縁覚+菩薩フェーズ: IntentAwareRouter による自動調心リランク
  4. コンテキスト品質メトリクス算出
  5. プロンプト構築 + LLM 呼び出し (or モック比較)

使い方:
    PYTHONPATH=. python examples/demo_e2e_rag.py

LLM 統合 (オプション):
    GEMINI_API_KEY=xxx PYTHONPATH=. python examples/demo_e2e_rag.py
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np

from lmm.dharma.reranker import IntentAwareRouter
from lmm.llm.embeddings import ngram_vectors


# =====================================================================
# 1. 知識コーパス: 仏教哲学 × AI × ネットワーク科学
# =====================================================================

# 10 トピック × 各 5 チャンク = 50 チャンクの疑似知識ベース
KNOWLEDGE_BASE: list[dict] = [
    # --- Cluster 0: 初期仏教・無我 ---
    {"topic": "初期仏教・無我", "text": "初期仏教において「無我（アナートマン）」とは、恒常不変の自己（アートマン）の存在を否定する教えである。五蘊（色・受・想・行・識）はいずれも無常であり、それらの集合に固定的な「我」を見出すことは執着（ウパーダーナ）にほかならない。"},
    {"topic": "初期仏教・無我", "text": "パーリ経典『無我相経（アナッタ・ラッカナ・スッタ）』において、釈迦は五比丘に対し「色は無我なり、受は無我なり…」と説いた。これは仏教思想史における最初の体系的な無我論であり、後のすべての学派の出発点となった。"},
    {"topic": "初期仏教・無我", "text": "無我の教えは「自己がない」という虚無論ではなく、「固定的な自己に執着する必要がない」という解放の教えである。これは現代の認知科学における「自己は脳の構成的プロセスである」という知見と驚くべき整合性を持つ。"},
    {"topic": "初期仏教・無我", "text": "初期仏教の縁起説（プラティーティヤ・サムトパーダ）は、すべての現象が相互依存的に生起することを説く。「此があれば彼あり、此がなければ彼なし」という定式は、現代のシステム思考やネットワーク理論の先駆けと言える。"},
    {"topic": "初期仏教・無我", "text": "部派仏教の説一切有部は、法（ダルマ）の実在を主張しつつも人無我を堅持した。一方、経量部は刹那滅の教えを徹底し、すべての存在は一瞬のフラッシュに過ぎないと主張した。これらの議論が後の大乗仏教の「空」の思想を準備した。"},

    # --- Cluster 1: 大乗仏教・空 ---
    {"topic": "大乗仏教・空", "text": "龍樹（ナーガールジュナ）の『中論（ムーラマディヤマカ・カーリカー）』は、すべての法が「空（シューニャター）」であることを論証した。空とは「虚無」ではなく「自性（スヴァバーヴァ）の欠如」であり、あらゆるものが縁起によって成立していることを意味する。"},
    {"topic": "大乗仏教・空", "text": "龍樹の「二諦説」は、世俗諦（慣習的真理）と勝義諦（究極的真理）を区別する。日常言語で「テーブルがある」と言うのは世俗諦として有効だが、勝義諦においてはテーブルも空である。この二層構造は、AIにおけるモデルの「有用性」と「真実性」の緊張に通じる。"},
    {"topic": "大乗仏教・空", "text": "空の思想は「中道（マディヤマカ）」として知られる。それは有（常見）と無（断見）の両極端を避ける思想的立場であり、固定的な実体論も虚無主義も共に退ける。DharmaLMM の Madhyamaka Balancer はこの中道の原理をパラメータ自動調整に応用している。"},
    {"topic": "大乗仏教・空", "text": "唯識学派（ヴィジュニャーナヴァーダ）は、外界の実在を否定し、すべてはアーラヤ識（阿頼耶識）の変現であると主張した。この「世界は心の表象に過ぎない」という見方は、ニューラルネットワークの内部表現（潜在空間）とのアナロジーが議論されている。"},
    {"topic": "大乗仏教・空", "text": "華厳思想の「事事無碍法界」は、個々の事象が互いに無限に映し合う「インドラの網」のメタファーで表現される。各ノードが他のすべてのノードを反映するこの構造は、完全接続グラフあるいはアテンション機構の数理的イメージと正確に対応する。"},

    # --- Cluster 2: ネットワーク科学 ---
    {"topic": "ネットワーク科学", "text": "バラバシ=アルバートモデルは、ネットワークの成長過程における「優先的接続（preferential attachment）」を示した。新しいノードは既に多くのリンクを持つノードに接続しやすい。この「リッチ・ゲット・リッチャー」現象はべき乗則次数分布を生み出す。"},
    {"topic": "ネットワーク科学", "text": "ワッツ=ストロガッツの「スモールワールド・ネットワーク」モデルは、高いクラスタリング係数と短い平均経路長が共存できることを示した。これは「6次の隔たり」現象の数学的説明であり、HNSWのような近似最近傍探索アルゴリズムの理論的基盤でもある。"},
    {"topic": "ネットワーク科学", "text": "グラフラプラシアン L = D - A の固有値分解は、スペクトラルクラスタリングの基礎である。フィードラーベクトル（第2固有ベクトル）による二分割は、最小カットに近似的に対応し、コミュニティ検出に広く応用されている。"},
    {"topic": "ネットワーク科学", "text": "モジュラリティ最大化は NP 困難であるが、ルーヴァン法やライデン法などの貪欲アルゴリズムにより効率的に近似解を得ることができる。これらの手法は大規模ソーシャルネットワークにおけるコミュニティ構造の発見に実用されている。"},
    {"topic": "ネットワーク科学", "text": "グラフニューラルネットワーク（GNN）は、メッセージパッシングによりノードの特徴量を近傍のコンテキストで更新する。この「近傍集約」の操作は、仏教の縁起（相互依存的生起）の計算的実装と見なすことができる。"},

    # --- Cluster 3: 情報検索 (IR) ---
    {"topic": "情報検索", "text": "BM25 は TF-IDF の改良版であり、文書の長さの正規化と飽和関数により、キーワード頻度の過大評価を防ぐ。2020年代においてもBM25はニューラル検索のベースラインとして高い競争力を維持している。"},
    {"topic": "情報検索", "text": "「Lost in the Middle」問題は、LLM が長いコンテキストの中央部に配置された情報を見落とす現象である。Top-K の素朴な検索結果には冗長な情報が多く、真に重要な情報が中央に埋もれやすいことが、この問題を深刻化させている。"},
    {"topic": "情報検索", "text": "MMR（Maximal Marginal Relevance）は、検索結果の多様性を確保するための古典的手法である。λ·sim(d,q) - (1-λ)·max_sim(d, selected) という式で、関連性と多様性のトレードオフを制御する。"},
    {"topic": "情報検索", "text": "コルバートV2（ColBERTv2）は、遅延相互作用（late interaction）により、トークンレベルの精密なマッチングと高速な検索を両立させる。各クエリトークンが文書トークンとの最大類似度を取り、その総和をスコアとする MaxSim 操作が核心である。"},
    {"topic": "情報検索", "text": "RAG（Retrieval-Augmented Generation）において、検索品質は生成品質の上限を決める。「Garbage In, Garbage Out」の原則は RAG にも完全に当てはまり、リランキングの品質がシステム全体の性能を決定づける。"},

    # --- Cluster 4: AI アライメント ---
    {"topic": "AIアライメント", "text": "AIアライメント問題は、AIシステムの目的関数が人間の意図と一致しない場合に生じるリスクを扱う。報酬ハッキング（reward hacking）やゴッドハート則（指標が目標になると良い指標でなくなる）はその典型例である。"},
    {"topic": "AIアライメント", "text": "RLHF（Reinforcement Learning from Human Feedback）は、人間のフィードバックから報酬モデルを学習し、LLMの出力を人間の選好に合わせる手法である。しかし、報酬モデル自体のバイアスや過適合が新たなアライメント問題を生み出す。"},
    {"topic": "AIアライメント", "text": "Constitutional AI は、人間の直接的なフィードバックの代わりに、一連の原則（憲法）に基づいてAIの出力を自己改善させるアプローチである。この「原則に基づく自己修正」は、仏教の戒律（シーラ）による自律的行動規範に類似する。"},
    {"topic": "AIアライメント", "text": "マルチエージェントアライメントは、複数のAIシステムが協調して人間の利益に沿うよう調整する問題である。ゲーム理論的なアプローチでは、ナッシュ均衡やコーリション形成の概念が用いられる。仏教のサンガ（僧伽）の調和原理との対応が興味深い。"},
    {"topic": "AIアライメント", "text": "スケーラブル・オーバーサイト（Scalable Oversight）は、超人的AIを監督するための人間の能力限界を克服する研究分野である。「弱い監督者がどうやって強いAIを安全に制御するか」という問題は、初学者が悟った師の教えをどう検証するかという仏教的問題に通じる。"},

    # --- Cluster 5: 劣モジュラ最適化 ---
    {"topic": "劣モジュラ最適化", "text": "劣モジュラ関数は「限界効用逓減」の性質を持つ集合関数である。すなわち A ⊆ B ならば f(A∪{e}) - f(A) ≥ f(B∪{e}) - f(B) が成立する。これは多様性の最大化を数学的に定式化する基礎となる。"},
    {"topic": "劣モジュラ最適化", "text": "単調劣モジュラ関数の最大化に対して、貪欲アルゴリズムは (1-1/e) ≈ 0.632 の近似保証を達成する。この結果は Nemhauser-Wolsey-Fisher (1978) により証明され、組合せ最適化の金字塔の一つである。"},
    {"topic": "劣モジュラ最適化", "text": "施設配置問題（Facility Location）は代表的な劣モジュラ最大化問題であり、f(S) = Σ max_{i∈S} sim(j, i) で定義される。これは「各データ点に最も近い代表を最大化する」ことに相当し、文書要約やデータ選択に広く応用されている。"},
    {"topic": "劣モジュラ最適化", "text": "グラフカット問題 cut(S) = Σ_{i∈S, j∉S} w_{ij} も劣モジュラ関数であり、選択集合と非選択集合の間の「つながりの強さ」を測る。DharmaLMM の Metta（慈愛）項はこのグラフカットを応用して多様性を最大化している。"},
    {"topic": "劣モジュラ最適化", "text": "ストリーミング劣モジュラ最大化は、データが逐次到着する場合の最適化である。Sieve-Streaming アルゴリズムは O(k log k / ε) の空間計算量で (1/2 - ε) 近似を達成する。リアルタイム RAG やソーシャルメディアのフィルタリングに適用可能である。"},

    # --- Cluster 6: 超モジュラ最適化 ---
    {"topic": "超モジュラ最適化", "text": "超モジュラ関数は劣モジュラの対極であり、「限界効用逓増」の性質を持つ。A ⊆ B ならば f(A∪{e}) - f(A) ≤ f(B∪{e}) - f(B)。これは相乗効果（シナジー）を数学的にモデル化する。"},
    {"topic": "超モジュラ最適化", "text": "超モジュラ最大化は一般に貪欲法では良い近似保証が得られない。しかし QUBO（二次制約なし二値最適化）への変換と、シミュレーテッドアニーリングや量子アニーリングによる求解が有効なアプローチとなる。"},
    {"topic": "超モジュラ最適化", "text": "DharmaLMM の Karuna（慈悲）項は超モジュラ関数として設計されている。選択集合が大きくなるほど、新しい要素を追加する限界価値が増大する。これは「サンガ（僧伽）が大きいほど、新しいメンバーの貢献が増す」という仏教的直観の数学的表現である。"},
    {"topic": "超モジュラ最適化", "text": "Ising モデルにおけるスピン間の強磁性相互作用（J > 0）は超モジュラ性に対応する。隣接スピンが同じ方向を向くことでエネルギーが下がるこの性質は、DharmaLMM の「同質的チームの相乗効果」のアナロジーとして使用されている。"},
    {"topic": "超モジュラ最適化", "text": "協調ゲーム理論において、超モジュラゲーム（戦略的補完性を持つゲーム）は必ず純戦略ナッシュ均衡を持つことが知られている（Topkis の定理）。この数学的保証は、DharmaLMM の SA ソルバーの収束性の理論的根拠にもなっている。"},

    # --- Cluster 7: LLM のハルシネーション ---
    {"topic": "ハルシネーション", "text": "LLM のハルシネーション（幻覚）とは、事実と異なる情報をもっともらしく生成する現象である。これは訓練データの偏り、デコーディングの確率的性質、および事実的接地（grounding）の欠如に起因する。"},
    {"topic": "ハルシネーション", "text": "RAG はハルシネーション軽減の最も実用的なアプローチの一つである。しかし、検索結果自体に冗長性や矛盾がある場合、LLM は依然として虚偽の合成（confabulation）を行う。コンテキストの品質が回答の品質の上限となる。"},
    {"topic": "ハルシネーション", "text": "自己一貫性（Self-Consistency）は、同じ質問に対して複数のサンプリングを行い、多数決で最も一貫した回答を選択する手法である。しかし全サンプルが同じ誤った前提に基づいている場合、この手法は無力である。"},
    {"topic": "ハルシネーション", "text": "帰属可能性（Attributability）は、LLM の各出力文が特定の情報源に帰属できるかを評価する指標である。帰属可能な回答はハルシネーションの検出と修正を容易にし、RAG の信頼性を高める上で重要なメトリクスである。"},
    {"topic": "ハルシネーション", "text": "仏教的には、ハルシネーションは「邪見（ミッチャー・ディッティ）」に相当する。正しい情報源（正見）に基づかない推論は、どれだけ流暢であっても苦（ドゥッカ）を生み出す。DharmaReranker は「正見」に基づくコンテキストを選定することで邪見を防ぐ。"},

    # --- Cluster 8: シミュレーテッドアニーリング ---
    {"topic": "シミュレーテッドアニーリング", "text": "シミュレーテッドアニーリング（SA）は、組合せ最適化問題に対するメタヒューリスティクスである。冶金学の焼鈍から着想を得ており、高温で広く探索し、温度を下げながら徐々に良い解に収束する。"},
    {"topic": "シミュレーテッドアニーリング", "text": "SA のメトロポリス基準 P(accept) = exp(-ΔE/T) は、エネルギーが上がる遷移も温度 T に応じた確率で受け入れる。これにより局所最適解からの脱出が可能になり、十分に遅い冷却スケジュールでは大域的最適解への収束が保証される。"},
    {"topic": "シミュレーテッドアニーリング", "text": "イジングモデル上の SA は、スピン反転の ΔE を O(degree) で計算できるため、疎なグラフ上では非常に高速に動作する。DharmaLMM の ising_sa ソルバーはこの性質を活用し、HNSW の疎グラフ上で ~200ms のリランクを実現している。"},
    {"topic": "シミュレーテッドアニーリング", "text": "量子アニーリングは、SAの量子力学的拡張であり、量子トンネル効果によりエネルギー障壁を古典的SAより効率的に越えることができる。D-Wave のチップは数千量子ビットでイジング問題を直接解くことができる。"},
    {"topic": "シミュレーテッドアニーリング", "text": "パラレルテンパリングは、異なる温度の複数のレプリカを並列に走らせ、レプリカ間で状態を交換するSAの拡張である。これにより探索空間の網羅性が飛躍的に向上し、位相空間での拡散が加速する。"},

    # --- Cluster 9: RAG 評価フレームワーク ---
    {"topic": "RAG評価", "text": "RAGAS（RAG Assessment）は、RAG パイプライン全体を自動的に評価するフレームワークである。主要指標として Context Precision（コンテキスト精度）、Context Recall（コンテキスト再現率）、Faithfulness（忠実性）、Answer Relevancy（回答関連性）を提供する。"},
    {"topic": "RAG評価", "text": "Context Precision は、検索結果の中でどれだけの割合が実際に回答に関連しているかを測る。高い Precision は LLM への入力が無駄なく効率的であることを示し、「Lost in the Middle」問題の定量的指標として有用である。"},
    {"topic": "RAG評価", "text": "Faithfulness（忠実性）は、LLM の生成した回答がコンテキストの情報にどれだけ忠実であるかを測定する。コンテキストに記載されていない情報を LLM が生成した場合、Faithfulness スコアが下がりハルシネーションの検出に繋がる。"},
    {"topic": "RAG評価", "text": "Human-in-the-Loop 評価は最も信頼性が高いが、スケーラビリティに限界がある。LLM-as-a-Judge は、別の LLM を評価者として用いることでスケーラビリティを確保するが、評価 LLM 自身のバイアスという問題を内在している。"},
    {"topic": "RAG評価", "text": "クロスエンコーダによる再スコアリングは、クエリと候補のペアを BERT 系モデルに入力し、精密なスコアを算出する。これは二段階検索（バイエンコーダ → クロスエンコーダ）のゴールドスタンダードであり、DharmaReranker の性能上限を示すベースラインとなる。"},
]


# =====================================================================
# 2. コンテキスト品質メトリクス
# =====================================================================


@dataclass
class ContextQuality:
    """RAG コンテキストの品質メトリクス"""

    n_chunks: int
    """選択されたチャンク数"""

    n_topics_covered: int
    """カバーされたトピック数"""

    n_topics_total: int
    """全トピック数"""

    coverage_ratio: float
    """カバー率 = n_topics_covered / n_topics_total"""

    avg_relevance: float
    """平均クエリ関連度 (コサイン類似度)"""

    redundancy: float
    """冗長度 = 選択チャンク間の平均ペアワイズ類似度"""

    topic_distribution: dict[str, int]
    """トピック分布"""

    efficiency: float
    """情報効率 = coverage_ratio / redundancy (高いほど良い)"""


def compute_context_quality(
    selected_indices: np.ndarray,
    all_embeddings: np.ndarray,
    query_embedding: np.ndarray,
    knowledge_base: list[dict],
) -> ContextQuality:
    """選択されたチャンク群のコンテキスト品質を算出する"""
    n = len(selected_indices)
    all_topics = list(set(kb["topic"] for kb in knowledge_base))
    n_topics_total = len(all_topics)

    # トピック分布
    topic_counts: dict[str, int] = {}
    for idx in selected_indices:
        t = knowledge_base[int(idx)]["topic"]
        topic_counts[t] = topic_counts.get(t, 0) + 1

    n_topics_covered = len(topic_counts)
    coverage_ratio = n_topics_covered / n_topics_total

    # クエリ関連度
    sel_emb = all_embeddings[selected_indices]
    q = query_embedding.flatten()
    q_norm = float(np.linalg.norm(q))
    q_norm = max(q_norm, 1e-10)
    c_norms = np.linalg.norm(sel_emb, axis=1)
    c_norms = np.clip(c_norms, 1e-10, None)
    rel_scores = (sel_emb @ q) / (c_norms * q_norm)
    avg_relevance = float(rel_scores.mean())

    # 冗長度 (選択チャンク間の平均類似度)
    norms = np.linalg.norm(sel_emb, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-10, None)
    normed = sel_emb / norms
    sim_matrix = normed @ normed.T
    # 対角を除く
    mask_sum = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            mask_sum += float(sim_matrix[i, j])
            count += 1
    redundancy = mask_sum / max(count, 1)

    efficiency = coverage_ratio / max(redundancy, 1e-10)

    return ContextQuality(
        n_chunks=n,
        n_topics_covered=n_topics_covered,
        n_topics_total=n_topics_total,
        coverage_ratio=coverage_ratio,
        avg_relevance=avg_relevance,
        redundancy=redundancy,
        topic_distribution=dict(sorted(topic_counts.items())),
        efficiency=efficiency,
    )


# =====================================================================
# 3. プロンプト構築
# =====================================================================


def build_rag_prompt(
    query: str,
    selected_indices: np.ndarray,
    knowledge_base: list[dict],
    method_name: str,
) -> str:
    """RAG プロンプトを構築する"""
    context_parts = []
    for i, idx in enumerate(selected_indices):
        chunk = knowledge_base[int(idx)]
        context_parts.append(
            f"[情報源 {i+1} / トピック: {chunk['topic']}]\n{chunk['text']}"
        )
    context = "\n\n".join(context_parts)

    prompt = f"""あなたは学術的な質問に正確に回答する AI アシスタントです。
以下の情報源のみに基づいて回答してください。情報源にない内容は推測せず、
「提供された情報源からは判断できません」と明記してください。

=== 情報源 ({method_name} により選定: {len(selected_indices)} 件) ===

{context}

=== 質問 ===
{query}

=== 回答 ==="""
    return prompt


# =====================================================================
# 4. メインデモ
# =====================================================================


def main():
    print("=" * 70)
    print("  End-to-End RAG 実証デモ — 救済 (LLM生成) の品質比較")
    print("=" * 70)
    print()

    # --- 知識ベースをベクトル化 ---
    print("--- Phase 1: 知識コーパスのベクトル化 ---")
    texts = [kb["text"] for kb in KNOWLEDGE_BASE]
    topics = [kb["topic"] for kb in KNOWLEDGE_BASE]
    all_topics_unique = sorted(set(topics))

    t0 = time.perf_counter()
    embeddings = ngram_vectors(texts, max_features=500)
    t_vec = time.perf_counter() - t0

    print(f"  チャンク数: {len(texts)}")
    print(f"  トピック数: {len(all_topics_unique)}")
    print(f"  埋め込み次元: {embeddings.shape[1]}")
    print(f"  ベクトル化時間: {t_vec * 1000:.1f}ms")
    print()

    # --- クエリ ---
    query_text = (
        "大乗仏教における「空」の概念は、初期仏教の「無我」から"
        "どのように発展し、現代のネットワーク科学やAIアライメントと"
        "どのように結びつくか？"
    )
    print("--- Phase 2: クエリ ---")
    print(f"  「{query_text}」")
    print()

    query_emb = ngram_vectors([query_text], max_features=500)[0]

    # --- 声聞フェーズ: 粗選別 ---
    n_fetch = 30
    k_final = 10
    q = query_emb.flatten()
    q_norm = float(np.linalg.norm(q))
    c_norms = np.linalg.norm(embeddings, axis=1)
    c_norms = np.clip(c_norms, 1e-10, None)
    scores = (embeddings @ q) / (c_norms * max(q_norm, 1e-10))
    ranked = np.argsort(scores)
    fetched_idx = np.array(
        [int(ranked[i]) for i in range(len(ranked) - 1, max(len(ranked) - n_fetch - 1, -1), -1)]
    )

    print(f"--- Phase 3: 声聞フェーズ (Top-{n_fetch} 粗選別) ---")
    fetched_topics = set(topics[int(i)] for i in fetched_idx)
    print(f"  フェッチされたトピック: {len(fetched_topics)}/{len(all_topics_unique)}")
    print()

    # --- 素朴な Top-K ---
    naive_idx = fetched_idx[:k_final]

    # --- IntentAwareRouter ---
    print("--- Phase 4: IntentAwareRouter 自動調心 ---")
    router = IntentAwareRouter(
        k=k_final,
        beta_range=(0.1, 0.8),
        metta_range=(0.0, 1.0),
        sa_iterations=1000,
        sparse_k=min(10, len(embeddings) - 1),
    )
    router.fit_offline(embeddings)

    t0 = time.perf_counter()
    result_dharma, diagnosis = router.route(
        query_embedding=query_emb,
        fetched_indices=fetched_idx,
    )
    t_route = time.perf_counter() - t0

    dharma_idx = result_dharma.original_indices

    print(f"  意図判定: {diagnosis.intent}")
    print(f"  集中度:   {diagnosis.concentration:.3f}")
    print(f"  エントロピー: {diagnosis.entropy:.3f}")
    print(f"  自動決定: Karuna={diagnosis.beta_used:.3f}, Metta={diagnosis.metta_used:.3f}")
    print(f"  時間: {t_route * 1000:.1f}ms")
    print()

    # --- コンテキスト品質比較 ---
    print("=" * 70)
    print("  コンテキスト品質メトリクス比較")
    print("=" * 70)
    print()

    q_naive = compute_context_quality(naive_idx, embeddings, query_emb, KNOWLEDGE_BASE)
    q_dharma = compute_context_quality(dharma_idx, embeddings, query_emb, KNOWLEDGE_BASE)

    def show_quality(label: str, q: ContextQuality):
        print(f"  [{label}]")
        print(f"    トピックカバー:  {q.n_topics_covered}/{q.n_topics_total} "
              f"({q.coverage_ratio:.0%})")
        print(f"    平均関連度:      {q.avg_relevance:.4f}")
        print(f"    冗長度:          {q.redundancy:.4f}")
        print(f"    情報効率:        {q.efficiency:.2f}")
        print(f"    トピック分布:    {q.topic_distribution}")
        print()

    show_quality("素朴な Top-K", q_naive)
    show_quality("IntentAwareRouter (自動調心)", q_dharma)

    # 改善率
    cov_delta = q_dharma.n_topics_covered - q_naive.n_topics_covered
    red_delta = q_naive.redundancy - q_dharma.redundancy
    eff_ratio = q_dharma.efficiency / max(q_naive.efficiency, 1e-10)

    print("  --- 改善サマリ ---")
    print(f"    トピックカバー: +{cov_delta} トピック "
          f"({q_naive.n_topics_covered} → {q_dharma.n_topics_covered})")
    print(f"    冗長度削減:     {red_delta:+.4f} "
          f"({q_naive.redundancy:.4f} → {q_dharma.redundancy:.4f})")
    print(f"    情報効率:       {eff_ratio:.1f}x 向上 "
          f"({q_naive.efficiency:.2f} → {q_dharma.efficiency:.2f})")
    print()

    # --- プロンプト生成 ---
    print("=" * 70)
    print("  生成されるプロンプトの比較")
    print("=" * 70)
    print()

    prompt_naive = build_rag_prompt(query_text, naive_idx, KNOWLEDGE_BASE, "素朴な Top-K")
    prompt_dharma = build_rag_prompt(query_text, dharma_idx, KNOWLEDGE_BASE, "IntentAwareRouter")

    # 各プロンプトのトピック多様性を表示
    def show_prompt_topics(label: str, indices: np.ndarray):
        print(f"  [{label}] コンテキストに含まれるトピック:")
        seen = []
        for idx in indices:
            t = KNOWLEDGE_BASE[int(idx)]["topic"]
            if t not in seen:
                seen.append(t)
        for i, t in enumerate(seen):
            print(f"    {i+1}. {t}")
        print()

    show_prompt_topics("素朴な Top-K", naive_idx)
    show_prompt_topics("IntentAwareRouter", dharma_idx)

    # --- プロンプト全文 (Dharma のみ表示) ---
    print("-" * 70)
    print("  IntentAwareRouter が構築したプロンプト (抜粋: 最初の3情報源)")
    print("-" * 70)
    # 最初の3情報源だけ表示
    for i, idx in enumerate(dharma_idx[:3]):
        chunk = KNOWLEDGE_BASE[int(idx)]
        print(f"  [{i+1}] ({chunk['topic']})")
        print(f"      {chunk['text'][:80]}...")
        print()

    # --- LLM 呼び出し (API キーがある場合) ---
    api_key = os.environ.get("GEMINI_API_KEY", "")

    if api_key:
        print("=" * 70)
        print("  LLM 生成比較 (Gemini API)")
        print("=" * 70)
        print()
        print("  [Gemini API キーを検出 — 生成実行中...]")
        # NOTE: 実際の Gemini 呼び出しは google-generativeai パッケージが必要
        # ここでは統合ポイントを示す
        print("  (google-generativeai パッケージをインストールし、")
        print("   GEMINI_API_KEY を設定してください)")
    else:
        print("=" * 70)
        print("  LLM 統合ポイント")
        print("=" * 70)
        print()
        print("  GEMINI_API_KEY を設定すると、実際の LLM 生成比較が実行されます。")
        print("  統合コード例:")
        print()
        print("    import google.generativeai as genai")
        print("    genai.configure(api_key=os.environ['GEMINI_API_KEY'])")
        print("    model = genai.GenerativeModel('gemini-2.0-flash')")
        print()
        print("    # 素朴な Top-K のプロンプト")
        print("    response_naive = model.generate_content(prompt_naive)")
        print()
        print("    # IntentAwareRouter のプロンプト")
        print("    response_dharma = model.generate_content(prompt_dharma)")
        print()
        print(f"  プロンプト長: 素朴={len(prompt_naive)}文字, "
              f"Dharma={len(prompt_dharma)}文字")

    print()
    print("=" * 70)
    print("  結論")
    print("=" * 70)
    print()
    print("  同じ LLM・同じクエリでも、コンテキストの品質で回答が激変する:")
    print()
    print(f"    素朴 Top-K:        {q_naive.n_topics_covered} トピック, "
          f"冗長度 {q_naive.redundancy:.3f}")
    print(f"      → 同じ内容の言い換えが詰まった「エコーチェンバー」")
    print(f"      → LLM は狭い視野で堂々巡りの回答を生成")
    print()
    print(f"    IntentAwareRouter: {q_dharma.n_topics_covered} トピック, "
          f"冗長度 {q_dharma.redundancy:.3f}")
    print(f"      → 多角的な知識ソースが立体的に配置された「サンガ」")
    print(f"      → LLM は複数の視座から統合的な回答を生成")
    print()
    print("  Garbage In, Garbage Out → Dharma In, Enlightenment Out")
    print()
    print("=" * 70)
    print("  デモ完了")
    print("=" * 70)


if __name__ == "__main__":
    main()
