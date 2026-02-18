# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [1.0.0] - 2025-02-15

### Added
- Sub-package re-exports: `from lmm.dharma import DharmaLMM` etc. now work
- `AlayaMemory.record_and_learn()`: QUBO-aware Hebbian learning with convergence gating
- `DharmaReasonerOrchestrator.think()`: Multi-phase reasoning with hyper/active escalation
- `ActiveInferenceEngine._inject_action_result()`: External knowledge injection with Sila bias
- CLI `--method ising_sa` support

### Fixed
- `HyperReasoner` sila_gamma using `n_total` instead of `n_orig` for cardinality constraint
- Broken import paths in examples/benchmarks
- Version badge and documentation version references

## [0.9.0] - 2025-02-12

### Added
- **Pineal Quantum OS** — 人工松果体・量子意識受肉モデル
- `PinealReasoner`: ハードウェア TRNG ベースの量子意識サンプラー
- `PinealSampler`: 物理エントロピー源によるトークンサンプリング
- n=3M ベンチマーク (26 tests)

### Performance
- Python ループを NumPy ベクトル化で高速化
- `collapse()` チャンク化 cumsum + batch entropy 事前計算

## [0.8.0] - 2025-02-10

### Added
- **三壁突破** — Sleep / Embodiment / Neuromorphic
- `SleepReasoner`: NREM/REM 記憶統合エンジン
- `EmbodimentReasoner`: 六根マルチモーダル融合
- `MemristorCrossbar`: メムリスタクロスバー ASIC シミュレータ
- テラベンチマーク (n=30M, 12 tests)

### Performance
- MemristorCrossbar をスパース辞書化
- 三壁モジュールの 3 大ボトルネックを解消

## [0.7.0] - 2025-02-08

### Added
- **Active Inference + Alaya Memory** — 能動的推論 + 阿頼耶識
- `ActiveInferenceReasoner`: 外部知識獲得エンジン
- `AlayaReasoner`: ヘブ則シナプス記憶

## [0.6.0] - 2025-02-06

### Added
- **DharmaReasonerOrchestrator** — 三種推論エンジン統合
- 8 モード推論ディスパッチャ (Adaptive / Theoretical / Abductive / Active / Alaya / Sleep / Embodiment / Pineal)
- FEP ベース推論モード 3 種
- Orchestrator Phase 別プロファイリング

### Performance
- Orchestrator 2 つのボトルネック解消 — raw FEP より高速化

## [0.5.0] - 2025-02-04

### Added
- **FEP Analog Solver** — 自由エネルギー原理 ≡ KCL ODE 等価証明
- `AnalogFepSolver`: キルヒホッフ電流則による ODE ソルバー
- LangChain 統合 (`DharmaDocumentCompressor`, `DharmaExampleSelector`, `DharmaRetriever`)
- LlamaIndex 統合 (`DharmaNodePostprocessor`)
- `FewShotSelector`, `OutputReranker`, `DriftDetector`

## [0.4.0] - 2025-01-30

### Added
- `AnalogFepSolver` (KCL ODE)
- RAG カスケードリランカー + インテントルーター
- E2E RAG パイプラインデモ

## [0.3.0] - 2025-01-25

### Added
- Digital Dharma Engine — 仏教哲学エネルギー関数
- 5 つのエネルギー項: Dukkha / Prajna / Karuna / Sila / Metta
- `UniversalDharmaEngine`: 自動ルーティングソルバー
- Submodular 最適化 (lazy greedy, (1-1/e) 保証)
- Count-Min Sketch + Streaming Histogram
- 多段カスケードフィルタ (1T → 100M → 10K → K)

## [0.2.0] - 2025-01-20

### Added
- Ising SA ソルバー (SIMD 最適化)
- スパース CSR/COO QUBO 行列
- ストリーミングサプライズ計算

## [0.1.0] - 2025-01-15

### Added
- 初期リリース
- QUBO ビルダー + 古典ソルバー (SA, 緩和, 貪欲法)
- サプライズ計算エンジン
- SmartSelector / SmartProcessor パイプライン
- CLI エントリーポイント
