"""Tests for lmm.scale.pipeline — ScalablePipeline."""

from __future__ import annotations

import numpy as np

from lmm.dharma.sangha import CouncilResult
from lmm.scale.pipeline import ScalablePipeline, ScalableResult


class TestScalablePipeline:
    def test_fit_and_run_array(self):
        rng = np.random.RandomState(42)
        data = rng.randn(200)
        pipe = ScalablePipeline(k=5)
        pipe.fit_array(data)
        result = pipe.run_array(data)
        assert isinstance(result, ScalableResult)
        assert len(result.indices) == 5

    def test_run_stream(self):
        rng = np.random.RandomState(42)

        def chunk_iter():
            for _ in range(3):
                yield rng.randn(100)

        pipe = ScalablePipeline(k=5)
        pipe.fit_stream(chunk_iter())
        result = pipe.run_stream(chunk_iter())
        assert isinstance(result, ScalableResult)
        assert len(result.indices) == 5

    def test_summary(self):
        rng = np.random.RandomState(42)
        data = rng.randn(200)
        pipe = ScalablePipeline(k=5)
        pipe.fit_array(data)
        result = pipe.run_array(data)
        summary = result.summary
        assert "selected" in summary
        assert "reduction" in summary
        assert summary["selected"] == 5


# ---------------------------------------------------------------------------
# Sangha × Cascade/Pipeline Integration (僧伽×カスケード統合テスト)
# ---------------------------------------------------------------------------

class _MockAlaya:
    """search(query, limit) を提供するモックオブジェクト."""

    def __init__(self, results=None):
        self.results = results or []
        self.calls = []

    def search(self, query, limit=5):
        self.calls.append((query, limit))
        return self.results


def _make_pipe(k=5, use_sangha=False, alaya_memory=None):
    """Helper: create a fitted ScalablePipeline for testing."""
    rng = np.random.RandomState(42)
    data = rng.randn(200)
    pipe = ScalablePipeline(
        k=k, use_sangha=use_sangha, alaya_memory=alaya_memory,
    )
    pipe.fit_array(data)
    return pipe, data


class TestScalablePipelineSangha:
    """ScalablePipeline × SanghaOrchestrator 統合テスト群."""

    def test_pipeline_with_sangha_approved(self):
        """use_sangha=True で正常クエリが APPROVED を返すこと."""
        pipe, data = _make_pipe(use_sangha=True)
        result = pipe.run_array(data)
        assert isinstance(result, ScalableResult)
        assert result.council is not None
        assert isinstance(result.council, CouncilResult)
        assert result.council.final == "APPROVED"
        assert len(result.indices) == 5

    def test_pipeline_with_sangha_and_alaya(self):
        """AlayaMemory 付きパイプラインで RAG 検索が実行されること."""
        mock = _MockAlaya(["retrieved_fact_1", "retrieved_fact_2"])
        pipe, data = _make_pipe(use_sangha=True, alaya_memory=mock)
        result = pipe.run_array(data)
        assert result.council is not None
        assert result.council.final == "APPROVED"
        # Sariputra (limit=2) + Mahakasyapa (limit=3) = 2 calls per round × 2 rounds
        assert len(mock.calls) >= 2

    def test_pipeline_without_sangha_backward_compat(self):
        """use_sangha=False（デフォルト）で council が None であること."""
        pipe, data = _make_pipe(use_sangha=False)
        result = pipe.run_array(data)
        assert result.council is None
        assert len(result.indices) == 5

    def test_pipeline_sangha_summary_includes_verdict(self):
        """Sangha 有効時に summary に council_verdict が含まれること."""
        pipe, data = _make_pipe(use_sangha=True)
        result = pipe.run_array(data)
        summary = result.summary
        assert "council_verdict" in summary
        assert summary["council_verdict"] == "APPROVED"

    def test_pipeline_sangha_summary_no_verdict_without_sangha(self):
        """Sangha 無効時に summary に council_verdict が含まれないこと."""
        pipe, data = _make_pipe(use_sangha=False)
        result = pipe.run_array(data)
        summary = result.summary
        assert "council_verdict" not in summary

    def test_end_to_end_cascade_processor_sangha(self):
        """E2E: ストリーム → カスケード → プロセッサ → 僧伽 の全フロー検証."""
        rng = np.random.RandomState(42)
        mock = _MockAlaya(["karma_record_1"])

        pipe = ScalablePipeline(
            k=5, use_sangha=True, alaya_memory=mock,
        )

        def chunk_iter():
            for _ in range(3):
                yield rng.randn(100)

        pipe.fit_stream(chunk_iter())
        result = pipe.run_stream(chunk_iter())

        # カスケード統計
        assert len(result.cascade_stats.level_sizes) >= 2
        assert result.cascade_stats.level_sizes[0] > 0

        # プロセッサ分類
        total_items = (
            result.processing.n_critical
            + result.processing.n_normal
            + result.processing.n_background
        )
        assert total_items == len(result.indices)

        # 僧伽合議
        assert result.council is not None
        assert result.council.final == "APPROVED"

        # AlayaMemory が実際にクエリされたことを確認
        assert len(mock.calls) >= 2

    def test_empty_data_skips_sangha(self):
        """空データの場合、僧伽合議がスキップされること."""
        pipe = ScalablePipeline(k=5, use_sangha=True)
        # fit with some data, then run with zero-length
        rng = np.random.RandomState(42)
        pipe.fit_array(rng.randn(200))
        result = pipe.run_array(np.array([]))
        assert result.council is None
        assert len(result.indices) == 0
