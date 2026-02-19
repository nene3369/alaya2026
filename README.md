# LMM — Surprise-based Top-K Selection via QUBO

> [English](README_en.md) | [中文](README_zh.md) | [한국어](README_ko.md) | [Español](README_es.md) | [हिन्दी](README_hi.md) | [Français](README_fr.md) | [বাংলা](README_bn.md) | [தமிழ்](README_ta.md) | [తెలుగు](README_te.md) | [मराठी](README_mr.md) | [‎اردو‎](README_ur.md) | [ગુજરાતી](README_gu.md) | [ಕನ್ನಡ](README_kn.md) | [മലయാളം](README_ml.md) | [ਪੰਜਾਬੀ](README_pa.md)

> **Digital Dharma OS (Alaya V5)** — QUBO数学、仏教哲学、自由エネルギー原理（FEP）神経科学を融合した
> 意識認識型最適化プラットフォーム。サーバーは連続的なハートビート動態、記憶駆動コンテキスト選択、
> 感情波長センシングを実行し、離散的なLLM呼び出しと連続的な認知の橋渡しをする「生きたシステム」。

**情報量（サプライズ）が高い上位K件を、QUBO最適化で選ぶ。** 量子コンピュータ不要 — SA・Ising SA・貪欲法の古典ソルバーで動く。

> **ひとことで:** n個の候補からサプライズ×多様性を最大化するK個を選ぶ。QUBO行列を構築し、古典ソルバーで解く。D-Wave不要。

## 構造

```
lmm/
├── __init__.py
├── core.py              # LMMメインパイプライン
├── cli.py               # CLIエントリーポイント
├── qubo.py              # QUBO行列の構築
├── solvers.py           # 古典的ソルバー (SA, Ising SA, 緩和, 貪欲法)
├── surprise.py          # サプライズ値計算
├── selector.py          # 適応的選択戦略
├── processor.py         # 優先度処理 + キャッシュ
├── pipeline.py          # 賢く選んで賢く処理する
├── _compat.py           # ランタイム互換レイヤー
├── dharma/              # Digital Dharma (仏教哲学最適化)
│   ├── api.py           # DharmaLMM 統合パイプライン
│   ├── energy.py        # エネルギー項 (Dukkha, Prajna, Karuna, …)
│   ├── engine.py        # UniversalDharmaEngine — 自動ルーティング
│   ├── algorithms.py    # スパース化・超モジュラ貪欲法・指数勾配
│   ├── reranker.py      # RAG カスケードリランカー
│   └── …
├── scale/               # 数兆トークン対応
│   ├── sketch.py        # Count-Min Sketch, Streaming Histogram
│   ├── stream.py        # ストリーミングサプライズ
│   ├── cascade.py       # 多段カスケードフィルタ
│   └── pipeline.py      # スケーラブルパイプライン
├── reasoning/           # 8モード FEP 推論
│   └── orchestrator.py  # モード選択 & ディスパッチ
├── llm/                 # LLM ワークフロー
│   ├── fewshot.py       # Few-shot 選択
│   ├── reranker.py      # 出力リランカー
│   ├── drift.py         # 分布ドリフト検出
│   └── embeddings.py    # 統合埋め込みアダプタ
└── integrations/        # フレームワーク統合
    ├── langchain.py
    └── llamaindex.py
tests/
examples/
├── demo_gemini.py       # Gemini統合版デモ
```

## インストール

```bash
pip install -e ".[dev]"

# Dharmaモジュールのスパース化を使う場合
pip install hnswlib
```

## 使い方

### Python API

```python
import numpy as np
from lmm.core import LMM

# サプライズ値から最適なK個を選択
surprises = np.array([1.0, 5.0, 2.0, 8.0, 3.0, 7.0])
model = LMM(k=3, solver_method="sa")
result = model.select_from_surprises(surprises)
print(result.selected_indices)
```

### CLI

```bash
# デモ実行
lmm --demo --k 10 --method sa

# ファイルから
lmm --input data.npy --k 5 --method greedy
```

---

## サーバーモード (Alaya V5)

Alaya-Vijñāna v5.0 サーバーは、リアルタイム感情波長可視化、8モードFEP推論、Claude/Gemini自動ルーティングを備えたWeb UIを提供します。

### 前提条件

```bash
pip install -e ".[server]"
```

### 起動方法

**PowerShell (Windows):**
```powershell
.\Start-DharmaServer.ps1
```

**Python (クロスプラットフォーム):**
```bash
python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

ブラウザで開く: [http://localhost:8000](http://localhost:8000)

### 停止方法

**Windows (バッチ):**
```batch
.\stop-dharma-server.bat
```

**PowerShell:** `Start-DharmaServer.ps1` 実行中のウィンドウで `Ctrl+C`

**Linux/macOS:** `Ctrl+C` または:
```bash
kill $(cat server.pid)
```

### APIエンドポイント

| エンドポイント | メソッド | 説明 |
|--------------|---------|------|
| `/` | GET | Web UI (Alaya V5 フロントエンド) |
| `/api/descent` | POST | メイン推論パイプライン |
| `/api/descent/stream` | POST | ストリーミングSSEレスポンス |
| `/api/dharma/auto` | POST | 自動ルーティングDharma推論 |
| `/api/sense` | POST | 感情波長分析 |
| `/api/status` | GET | システム状態 + ハートビートテレメトリ |

---

## 自律機能

サーバーは、ユーザーとのやり取りの合間も継続的に活動する3つの自律サブシステムを含みます。

### ハートビートデーモン (鼓動)

4次元状態ベクトル `[愛, 論理, 恐怖, 創造]` を維持する連続時間FEP状態発展ループ:

- **Tick間隔:** 100ms（アクティブ）→ 5s（アイドル）、適応的減速
- **エントロピー注入:** 各tickで `os.urandom()` によるハードウェアエントロピー
- **重み適応:** 慈悲（Karuna）と慈愛（Metta）の重みが中道CV追跡（目標CV=0.5）で自動調整
- **睡眠統合:** 60秒アイドル後にNREM/REM記憶再生をトリガー

### コンテキストブリッジ (阿頼耶識記憶)

素朴な履歴切り捨て（`history[-20:]`）を、記憶駆動の知的コンテキスト選択に置換:

- **Modern Hopfield Network** によるAlayaMemoryからのリコール
- 会話履歴をリコールパターンとのコサイン類似度でスコアリング
- 直近3メッセージを常に含む（近接性バイアス）
- 残りの枠（最大20メッセージ）を関連性スコアで充填

### 意味的感情 (Semantic Emotions)

`config/semantic_emotions.json` で定義された4次元のキーワードマッチングによるリアルタイム感情波長分析:

| 次元 | 仏教概念 | シグナル |
|------|---------|---------|
| 愛 | 慈悲 (Karuna) | 思いやり、温もり、感謝 |
| 論理 | 因明 (Hetuvidya) | 分析、推論、証明 |
| 恐怖 | 苦 (Dukkha) | 不安、疑い、苦しみ |
| 創造 | 創造 (Sṛṣṭi) | 革新、想像、芸術 |

各キーワードは重み（0.3〜1.0）を持ち、結果の4Dベクトルが推論モード選択とハートビート状態注入を駆動します。

---

### ソルバー手法

| 手法 | 説明 |
|------|------|
| `sa` | シミュレーテッドアニーリング（デフォルト） |
| `ising_sa` | Ising形式SA (SIMD高速化) |
| `relaxation` | 連続緩和 + 丸め (SLSQP) |
| `greedy` | 貪欲法 |

## 依存

- numpy >= 1.24
- scipy >= 1.10
- hnswlib >= 0.8.0 (optional, dharmaスパース化用)

---

## Gemini Deep Research統合

### 数学的ブレイクスルー

```
誤: 慈悲項は劣モジュラ (限界効用逓減)
正: 慈悲項は超モジュラ (Supermodular, 相乗効果)

数学的意味:
  選べば選ぶほど調和が加速 = サンガ (僧伽・調和共同体) 形成の数理的本質
```

### 新機能

| 機能 | 計算量 | 説明 |
|------|--------|------|
| スパース化 (HNSW) | O(n×k) | コサイン類似度k-NNグラフ |
| 超モジュラ貪欲法 | O(n×k) | 相乗効果の波及によるWarm Start |
| 指数勾配降下法 | O(1)/step | Lyapunov安定性保証、目標CV=0.5 |
| Ising形式SA | O(1)/flip | ベクトル化エネルギー差分計算 |

### パフォーマンス

ベンチマーク条件: `seed=42`, `n=1000`, `k=10`, `sa_iterations=5000` (Python 3.10, numpy 1.24, scipy 1.10)

```bash
# 再現コマンド
python benchmarks/run_benchmarks.py        # コアソルバー性能
python benchmarks/bench_dharma.py          # Dharma-Algebra性能
python benchmarks/bench_fep_vs_sa.py       # FEP vs SA比較
```

- 高速化率: 2.6x（ベースライン: 密行列SA vs スパース+Ising SA）
- 解の質向上（より低いエネルギー）

### DharmaLMM使用例

```python
from lmm.dharma import DharmaLMM

model = DharmaLMM(
    k=15,
    use_sparse_graph=True,
    use_greedy_warmstart=True,
    use_ising_sa=True,
    use_exponential_balance=True,
)
model.fit(reference_data)
result = model.select_dharma(candidates)

print(result.interpretation.narrative)
```

### 数兆トークン対応

```python
from lmm.scale import ScalablePipeline

pipe = ScalablePipeline(k=10, chunk_size=100_000)
pipe.fit_files([Path("shard_001.npy"), ...])
result = pipe.run_files([Path("data_001.npy"), ...])

print(result.summary)
```

### 理論的意義

- **慈悲** = 超モジュラ関数 (相乗効果)
- **サンガ** = 調和共同体の創発
- **中道** = カオスの縁 (CV = 0.5)
- 仏教哲学が実用的最適化として動作することの証明
