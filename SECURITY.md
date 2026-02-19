# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

セキュリティ上の問題を発見した場合は、**公開 Issue を作成せず**、
以下の方法で報告してください:

1. GitHub の [Security Advisories](https://github.com/nene3369/LMM/security/advisories) から非公開レポートを作成
2. または、リポジトリオーナーに直接連絡

報告には以下を含めてください:

- 問題の説明
- 再現手順
- 影響範囲の評価

通常 **72 時間以内** に対応します。

---

## サンドボックス・実行安全性

### サーバーバインド

Alaya V5 サーバーはデフォルトで `0.0.0.0:8000` にバインドし、CORS を
`allow_origins=["*"]` に設定しています。これは **ローカル開発専用** です。

**本番デプロイ時の推奨:**
- `127.0.0.1` にバインドするか、リバースプロキシを使用
- CORS オリジンを自ドメインに限定
- HTTPS/TLS を有効化

### バックグラウンドプロセス

HeartbeatDaemon は連続的な非同期ループ（デフォルト tick 間隔: 100ms、
アイドル時は最大 5s まで減速）を実行します。CPU 消費は軽微ですが、
リソース制約のある環境では監視を推奨します。

### 外部ネットワーク通信

サーバーはユーザーリクエストに応じて外部 LLM プロバイダ（Anthropic Claude、
Google Gemini）に API 呼び出しを行います。ユーザー操作なしにデータが
送信されることはありません。

---

## 実験的機能の注意

以下の機能は **研究段階** であり、現状のまま提供されます:

| 機能 | モジュール | 注意事項 |
|------|----------|---------|
| 松果体 (Pineal) | `lmm/reasoning/pineal.py` | `os.urandom()` 使用 — 暗号用途ではない |
| ニューロモーフィック | `lmm/dharma/neuromorphic.py` | ソフトウェアシミュレーションのみ |
| 睡眠統合 (Sleep) | `lmm/reasoning/sleep.py` | メモリ刈り込みによりパターンが失われる場合あり |
| FEP KCL ODE | `lmm/dharma/fep.py` | 数値積分 — 安定性の検証を推奨 |

---

## API キーの取り扱い

API キーは以下の方法で受け渡します:
1. HTTP ヘッダー: `X-Claude-Api-Key`, `X-Gemini-Api-Key`
2. 環境変数: `CLAUDE_API_KEY`, `GEMINI_API_KEY`

**キーは一切:**
- ディスクやログファイルに書き込まれません
- サーバーレスポンスに含まれません
- リクエストのライフサイクルを超えて保持されません

ローカル開発では `.env` ファイルを使用してください（`.gitignore` に登録済み）。
