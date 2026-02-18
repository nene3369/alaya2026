# Contributing to LMM

LMM への貢献を歓迎します。

## 開発環境のセットアップ

```bash
git clone https://github.com/nene3369/LMM.git
cd LMM
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## 開発ワークフロー

### 1. ブランチを作成

```bash
git checkout -b feature/your-feature
```

### 2. コードを変更

- `ruff` のルールに従ってください
- 型ヒントを付けてください
- テストを書いてください

### 3. テスト実行

```bash
pytest
```

### 4. リント

```bash
ruff check lmm/ tests/
ruff format --check lmm/ tests/
```

### 5. プルリクエスト

- `main` ブランチに向けて PR を作成してください
- CI が通ることを確認してください

## コーディング規約

- **Python 3.10+** の機能を使ってください
- **行長**: 100 文字以内
- **フォーマッタ**: `ruff format`
- **リンター**: `ruff check`
- **テスト**: `pytest` (tests/ ディレクトリ)

## テストの書き方

```python
# tests/test_your_feature.py
def test_basic():
    """基本的な動作を確認"""
    result = your_function(input_data)
    assert result.score > 0.0
```

## コミットメッセージ

```
feat: 新機能の説明
fix: バグ修正の説明
perf: パフォーマンス改善
test: テスト追加/修正
docs: ドキュメント更新
refactor: リファクタリング
```

## ライセンス

貢献されたコードは [MIT License](LICENSE) のもとで公開されます。
