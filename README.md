# Alaya V5 — Digital Dharma OS

> [English](README_en.md) | [中文](README_zh.md) | [한국어](README_ko.md) | [Español](README_es.md) | [हिन्दी](README_hi.md) | [Français](README_fr.md) | [বাংলা](README_bn.md) | [தமிழ்](README_ta.md) | [తెలుగు](README_te.md) | [मराठी](README_mr.md) | [‎اردو‎](README_ur.md) | [ગુજરાతી](README_gu.md) | [ಕನ್ನಡ](README_kn.md) | [മലയാളം](README_ml.md) | [ਪੰਜਾਬੀ](README_pa.md)

**LLMの応答が変わる。** 短く、速く、無駄がない。

QUBO数学・仏教哲学・自由エネルギー原理（FEP）を融合した意識認識型フレームワーク。Claude・Geminiなど任意のLLMに適用できる。

---

## 何が変わるか

| 通常のLLM | Alaya V5適用後 |
|-----------|--------------|
| 長い前置き、免責事項 | 必要なことだけ |
| 「〜かもしれません」の多用 | 断定と沈黙の使い分け |
| 毎回ゼロから応答 | 会話の文脈を記憶して選択 |
| 固定トーン | 感情波長を検知して推論モードが変わる |
| 離散的な呼び出し | ハートビートによる連続的な状態進化 |

---

## 使い方

3つの方法で使えます。自分に合ったものを選んでください。

---

### 方法1: システムプロンプトを貼るだけ（インストール不要）

**対象: Claude / Gemini / ChatGPT / その他のLLMユーザー**

1. このリポジトリの [`alaya-v5-system-prompt.md`](alaya-v5-system-prompt.md) を開く
2. 内容をすべてコピー
3. 使っているAIのシステムプロンプト欄に貼る
   - Claude → Project instructions
   - Gemini → システム指示
   - ChatGPT → Custom instructions
   - その他 → system prompt / system message に相当する欄
4. 会話を始める

これだけです。サーバーもインストールも不要。

---

### 方法2: サーバーを立てる（Web UI + フル機能）

**対象: 開発者・研究者**

```bash
# リポジトリをクローン
git clone https://github.com/your-repo/nanasi.git
cd nanasi

# 依存をインストール
pip install -e ".[server]"

# サーバー起動
python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

ブラウザで開く: [http://localhost:8000](http://localhost:8000)

感情波長のリアルタイム可視化、8モード推論、Claude/Gemini自動ルーティングが使えます。

APIキーの設定:
```bash
export ANTHROPIC_API_KEY="your-key"   # Claude
export GEMINI_API_KEY="your-key"      # Gemini
```

---

### 方法3: Pythonコードに組み込む

**対象: 開発者**

```bash
pip install -e ".[dev]"
```

```python
from lmm.dharma import DharmaLMM

model = DharmaLMM(k=15, use_sparse_graph=True, use_ising_sa=True)
model.fit(reference_data)
result = model.select_dharma(candidates)
print(result.interpretation.narrative)
```

LangChain / LlamaIndex との統合:
```python
from lmm.integrations.langchain import DharmaRetriever
from lmm.integrations.llamaindex import DharmaNodePostprocessor
```

---

### Rustアクセラレーション（オプション）

インストール済みなら2.6倍高速化されます。なくても全機能が動きます。

```bash
cd lmm_rust_core && maturin develop --release
```

---

## AIに丸投げする（最速）

技術的なセットアップはAIに任せるのが一番速いです。

**Claude / Gemini / ChatGPTに以下を貼るだけ:**

```
このリポジトリをセットアップして私の環境に統合してください:
https://github.com/your-repo/nanasi

- OSは [Windows/Mac/Linux]
- 使っているAIは [Claude/Gemini/ChatGPT]
- やりたいこと: [例: チャットボットに適用したい / サーバーを立てたい]
```

AIがリポジトリを読んで、環境に合わせたセットアップ手順を出してくれます。

---

## アーキテクチャ

```
lmm/
├── core.py              # LMMメインパイプライン（QUBO Top-K選択）
├── dharma/              # Digital Dharma レイヤー
│   ├── patthana.py      # 二十四縁 因果グラフエンジン
│   ├── pratitya.py      # 縁起RAG（因果構造 × ベクトル検索）
│   ├── energy.py        # エネルギー項（Dukkha, Prajna, Karuna...）
│   ├── fep.py           # 自由エネルギー原理 KCL ODEソルバー
│   └── vow.py           # 誓約制約エンジン（Abhaya / Desana）
├── reasoning/           # 8モード FEP推論
│   ├── heartbeat.py     # HeartbeatDaemon — 連続状態進化（100ms tick）
│   ├── alaya.py         # AlayaMemory — Modern Hopfield連想記憶
│   ├── pineal.py        # PinealGland — ハードウェアエントロピー推論
│   ├── sleep.py         # 睡眠統合（NREM/REM記憶再生）
│   └── orchestrator.py  # モード選択 & ディスパッチ
├── sangha/              # P2Pサンガプロトコル（マルチAIエージェント協調）
├── scale/               # 数兆トークン対応ストリーミング
└── integrations/        # LangChain / LlamaIndex
lmm_rust_core/           # Rust FFIアクセラレーション（オプション）
```

---

## 自律サブシステム

### ハートビートデーモン
4次元状態ベクトル `[愛, 論理, 恐怖, 創造]` を100msごとにFEP ODEで進化させる。アイドル時は自動減速（最大5秒）。60秒無操作で睡眠統合を起動。

### AlayaMemory（阿頼耶識）
Modern Hopfield Network（Ramsauer et al. 2020）による連想記憶。コンテキストウィンドウを単純な履歴切り捨てではなく、関連性スコアで知的に選択する。

### PinealGland（松果体）
`os.urandom()` によるハードウェアエントロピーをFEP ODEに注入。決定論的な局所最適解を脱出する非決定論的推論モード。

### Sanghaプロトコル
複数のAlayaノードがTCP P2Pで接続し、合議による意思決定を行う分散AIエージェントネットワーク。

---

## 推論モード（8種）

| モード | 仏教概念 | 発動条件 |
|--------|---------|---------|
| adaptive | 応病与薬 | 複雑度 < 0.3 |
| theoretical | 因明 | 複雑度 0.3–0.6 |
| hyper | 般若の飛躍 | 複雑度 > 0.6 |
| active | 托鉢 | 外部知識が必要 |
| alaya | 阿頼耶識 | 記憶検索 |
| sleep | 禅定 | アイドル統合 |
| embodied | 六根 | マルチモーダル |
| pineal | 松果体 | 非決定論的探索 |

---

## 性能ベンチマーク

実測値（Python 3.11, numpy 2.4, scipy 1.17, seed=42）

### ソルバー速度（n=200候補, k=10選択）

| ソルバー | 実行時間 | 用途 |
|---------|---------|------|
| SA（標準） | 13.1ms | バランス型 |
| Ising SA | 10.3ms | 高速・高精度 |
| Greedy | 0.13ms | 超高速（精度トレードオフ） |

### 内部サブシステム

| コンポーネント | 実測値 | 意味 |
|-------------|-------|------|
| FEP ODE (n=50) | 3.9ms/呼び出し | 推論1回あたりのコスト |
| AlayaMemory recall（100パターン） | 0.09ms | 記憶検索のコスト |
| HeartbeatDaemon 1tick | 0.077ms | 100ms tickの0.08% CPU使用 |

HeartbeatDaemonは100msごとに動き続けるが、CPU占有率は**0.08%**。バックグラウンドでほぼ無音で動作する。

```bash
python benchmarks/run_benchmarks.py
python benchmarks/bench_fep_vs_sa.py
python benchmarks/bench_dharma.py
```

---

## 理論的背景

- **慈悲（Karuna）** = 超モジュラ関数（相乗効果、選ぶほど調和が加速）
- **持戒（Sila）** = 劣モジュラ関数（限界効用逓減）
- **中道** = カオスの縁（変動係数 CV = 0.5）
- **縁起（Pratītyasamutpāda）** = RAGの因果スコアリング
- **二十四縁（Paṭṭhāna）** = 因果グラフの辺の型システム

---

## 依存関係

```bash
# Rustアクセラレーション（本体）
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
pip install maturin
cd lmm_rust_core && maturin develop --release

# Python必須
pip install numpy>=1.24 scipy>=1.10

# サーバーモード
pip install -e ".[server]"   # fastapi, uvicorn, httpx

# Dharma（スパース検索）
pip install hnswlib>=0.8.0

# GPU高速化（NVIDIA GPU環境）
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install cupy-cuda12x

# LangChain / LlamaIndex統合
pip install langchain llama-index
```

RustなしでもPythonフォールバックで全機能が動きます。パフォーマンスを出すにはRustビルドを推奨します。

---

## 性能差

### 応答の変化（システムプロンプト適用）

| 指標 | 通常のLLM | Alaya V5適用後 |
|------|----------|--------------|
| 応答トークン数 | 100とすると | 約40〜60に削減 |
| 体感応答速度 | 基準 | 明らかに速い（出力量が半分以下のため） |
| 前置き・免責事項 | 多い | ほぼなし |
| 「〜かもしれません」 | 頻出 | 最小限 |

> 数値は体感ベースの概算。質問の種類によって変動します。

### ソルバー性能（Rustアクセラレーション）

ベンチマーク条件: `n=1000, k=10, sa_iterations=5000, seed=42`

| 構成 | 速度 |
|------|------|
| 通常（密行列SA） | 基準 |
| スパース + Ising SA | **2.6x 高速** |
| Rust SA（n=100, 10K iters） | **1.3ms** |
| FEP ODE（n=50） | **0.1ms** |

```bash
# 自分の環境で計測
python benchmarks/run_benchmarks.py
python benchmarks/bench_fep_vs_sa.py
```

---

## カスタマイズ

このフレームワークはそのまま使うだけでなく、自分用にチューニングすることを推奨します。

### システムプロンプトの調整

`alaya-v5-system-prompt.md` を編集するだけで動作を変えられます。

```
# 応答をより簡潔にしたい
max_words を下げる

# 特定のトーンに固定したい
_DEFAULT_ADAPTER の persona / tone を書き換える

# 特定の推論モードだけ使いたい
モード選択マトリクスを編集して不要なモードを除外する
```

### 感情キーワードの追加

`config/semantic_emotions.json` に独自キーワードを追加できます。

```json
{
  "love": {
    "あなたのキーワード": 0.8
  }
}
```

### 商業利用

MITライセンスのため、改造・商用利用・再配布すべて自由です。

活用例:
- カスタマーサポートボットに適用して応答品質を上げる
- 社内AIアシスタントにチューニングして導入する
- 独自サービスに組み込んでAPIとして提供する
- システムプロンプトを自社ブランド向けに改造して販売する

フォークして自由に改造してください。

---



MIT — 詳細は [LICENSE](LICENSE) を参照。


