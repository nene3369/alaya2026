"""Tests for lmm.llm — embeddings, drift, fewshot, reranker, sampler."""

from __future__ import annotations

import numpy as np
import pytest

from lmm.llm.embeddings import EmbeddingAdapter, auto_embed, cosine_similarity, ngram_vectors
from lmm.llm.drift import (
    DriftDetector,
    MemoryDecayAdvisor,
    SemanticDriftDetector,
    SemanticDriftReport,
)
from lmm.llm.fewshot import FewShotSelector
from lmm.llm.reranker import OutputReranker
from lmm.llm.sampler import PinealSampler


class TestNgramVectors:
    def test_basic(self):
        texts = ["hello world", "foo bar baz", "hello foo"]
        vecs = ngram_vectors(texts)
        assert vecs.shape[0] == 3
        assert vecs.shape[1] > 0

    def test_empty_list(self):
        vecs = ngram_vectors([])
        assert vecs.shape[0] == 0

    def test_single(self):
        vecs = ngram_vectors(["hello"])
        assert vecs.shape[0] == 1


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = np.array([1.0, 0.0, 0.0])
        mat = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        sims = cosine_similarity(mat, v)
        assert sims[0] == pytest.approx(1.0)
        assert sims[1] == pytest.approx(0.0)


class TestEmbeddingAdapter:
    def test_ngram_backend(self):
        adapter = EmbeddingAdapter(backend="ngram")
        vecs = adapter.embed(["hello world", "foo bar"])
        assert vecs.shape[0] == 2


class TestAutoEmbed:
    def test_basic(self):
        vecs = auto_embed(["hello world", "test text"])
        assert vecs.shape[0] == 2


class TestDriftDetector:
    def test_no_drift(self):
        det = DriftDetector()
        texts_baseline = ["hello world", "foo bar"] * 10
        det.fit_baseline(texts_baseline)
        report = det.update("hello world")
        assert hasattr(report, "is_drifting")

    def test_drift(self):
        det = DriftDetector()
        texts_baseline = ["cat dog animal"] * 20
        det.fit_baseline(texts_baseline)
        for t in ["quantum physics molecule"] * 20:
            report = det.update(t)
        assert report is not None

    def test_check(self):
        det = DriftDetector()
        texts_baseline = ["hello world"] * 10
        det.fit_baseline(texts_baseline)
        report = det.check()
        assert hasattr(report, "drift_score")


class TestFewShotSelector:
    def test_select(self):
        examples = [
            {"input": "hello", "output": "world"},
            {"input": "foo", "output": "bar"},
            {"input": "test", "output": "case"},
            {"input": "quantum", "output": "physics"},
            {"input": "deep", "output": "learning"},
        ]
        selector = FewShotSelector(k=3)
        selector.fit(examples)
        result = selector.select("hello world")
        assert len(result.indices) <= 3
        assert len(result.examples) == len(result.indices)


class TestOutputReranker:
    def test_rerank(self):
        outputs = ["output A", "output B", "output C", "output D"]
        reranker = OutputReranker(k=2)
        result = reranker.rerank(outputs)
        assert len(result.indices) == 2
        assert len(result.texts) == 2


class TestPinealSampler:
    def test_sample(self):
        sampler = PinealSampler()
        logits = np.array([1.0, 2.0, 3.0, 0.5])
        result = sampler.sample(logits)
        assert 0 <= result.token_id < 4

    def test_sample_with_top_k(self):
        sampler = PinealSampler(top_k=2)
        logits = np.array([1.0, 2.0, 3.0, 0.5])
        result = sampler.sample(logits)
        assert 0 <= result.token_id < 4

    def test_report(self):
        sampler = PinealSampler()
        logits = np.array([1.0, 2.0, 3.0])
        sampler.sample(logits)
        report = sampler.report()
        assert report.n_tokens_sampled == 1


# ===================================================================
# SemanticDriftDetector (Anicca / 諸行無常)
# ===================================================================


class TestSemanticDriftDetector:
    def test_fit_baseline(self):
        rng = np.random.RandomState(42)
        det = SemanticDriftDetector()
        baseline = rng.randn(50, 8)
        det.fit_baseline(baseline)
        assert det._fitted
        assert det._baseline_centroid.shape == (8,)

    def test_no_drift_on_similar_data(self):
        rng = np.random.RandomState(42)
        baseline = rng.randn(100, 8)
        det = SemanticDriftDetector(threshold=0.8)
        det.fit_baseline(baseline)

        # Feed data from the same distribution (larger sample for stability)
        for _ in range(100):
            det.update(rng.randn(8))

        report = det.check()
        assert isinstance(report, SemanticDriftReport)
        assert not report.is_drifting

    def test_detects_centroid_shift(self):
        rng = np.random.RandomState(42)
        baseline = rng.randn(100, 8)
        det = SemanticDriftDetector(threshold=0.1)
        det.fit_baseline(baseline)

        # Feed data from a shifted distribution (large offset)
        shifted = rng.randn(100, 8) + 10.0
        det.update_batch(shifted)
        report = det.check()
        assert report.centroid_shift > 0.1
        assert report.is_drifting

    def test_detects_variance_change(self):
        rng = np.random.RandomState(42)
        baseline = rng.randn(100, 8)
        det = SemanticDriftDetector(threshold=0.05, variance_weight=0.8,
                                    centroid_weight=0.1, cosine_weight=0.1)
        det.fit_baseline(baseline)

        # Feed data with much higher variance
        high_var = rng.randn(100, 8) * 10.0
        det.update_batch(high_var)
        report = det.check()
        assert report.variance_ratio > 5.0

    def test_update_single(self):
        rng = np.random.RandomState(42)
        det = SemanticDriftDetector()
        det.fit_baseline(rng.randn(20, 4))
        report = det.update(rng.randn(4))
        assert report.n_observations == 1
        assert report.window_size == 1

    def test_update_batch(self):
        rng = np.random.RandomState(42)
        det = SemanticDriftDetector()
        det.fit_baseline(rng.randn(20, 4))
        report = det.update_batch(rng.randn(10, 4))
        assert report.n_observations == 10
        assert report.window_size == 10

    def test_identify_stale_patterns(self):
        rng = np.random.RandomState(42)
        det = SemanticDriftDetector()
        det.fit_baseline(rng.randn(20, 8))

        # Window is centered near origin, stale patterns far away
        for _ in range(20):
            det.update(rng.randn(8) * 0.1)

        # Build patterns array row-by-row (vendored numpy doesn't support int setitem)
        row0 = list(rng.randn(8).flatten() * 0.1)   # close to window centroid
        row1 = [100.0] * 8                            # far away
        row2 = [-100.0] * 8                           # far away
        row3 = list(rng.randn(8).flatten() * 0.1)   # close
        row4 = [50.0] * 8                             # far
        patterns = np.array([row0, row1, row2, row3, row4])

        stale = det.identify_stale_patterns(patterns, top_k=3)
        assert len(stale) == 3
        # The patterns far from centroid should rank as most stale
        assert 1 in stale or 2 in stale

    def test_reset_baseline_promotes_window(self):
        rng = np.random.RandomState(42)
        det = SemanticDriftDetector()
        det.fit_baseline(rng.randn(20, 4))
        for _ in range(10):
            det.update(rng.randn(4) + 5.0)

        det.reset_baseline()
        assert det._n_observations == 0
        assert len(det._window) == 0
        # Baseline centroid should now be near the shifted data
        assert det._baseline_centroid is not None

    def test_unfitted_raises(self):
        det = SemanticDriftDetector()
        try:
            det.update(np.zeros(4))
            assert False, "Should have raised RuntimeError"
        except RuntimeError:
            pass

    def test_check_empty_window(self):
        rng = np.random.RandomState(42)
        det = SemanticDriftDetector()
        det.fit_baseline(rng.randn(10, 4))
        report = det.check()
        assert report.drift_score == 0.0
        assert report.window_size == 0


# ===================================================================
# MemoryDecayAdvisor
# ===================================================================


class TestMemoryDecayAdvisor:
    def test_stable_returns_base_rate(self):
        rng = np.random.RandomState(42)
        det = SemanticDriftDetector(threshold=0.5)
        det.fit_baseline(rng.randn(50, 8))
        for _ in range(20):
            det.update(rng.randn(8))

        advisor = MemoryDecayAdvisor(det, base_decay_rate=0.01)
        rate = advisor.compute_decay_rate()
        assert rate == pytest.approx(0.01)

    def test_drifting_returns_elevated_rate(self):
        rng = np.random.RandomState(42)
        det = SemanticDriftDetector(threshold=0.05)
        det.fit_baseline(rng.randn(50, 8))
        # Shift distribution far away
        for _ in range(50):
            det.update(rng.randn(8) + 20.0)

        advisor = MemoryDecayAdvisor(
            det, base_decay_rate=0.01, drift_decay_multiplier=5.0,
        )
        rate = advisor.compute_decay_rate()
        assert rate > 0.01

    def test_pattern_weights_stale_low(self):
        rng = np.random.RandomState(42)
        det = SemanticDriftDetector()
        det.fit_baseline(rng.randn(20, 4))
        # Window near origin
        for _ in range(20):
            det.update(rng.randn(4) * 0.1)

        advisor = MemoryDecayAdvisor(det)
        # Use a pattern aligned with centroid vs one opposite
        patterns = np.array([
            [100.0, 100.0, 100.0, 100.0],   # one direction
            [-100.0, -100.0, -100.0, -100.0],  # opposite direction
        ])
        weights = advisor.compute_pattern_weights(patterns)
        assert weights.shape == (2,)
        # Weights should differ for opposite-direction patterns
        assert float(weights[0]) != float(weights[1])

    def test_pattern_weights_empty_window(self):
        rng = np.random.RandomState(42)
        det = SemanticDriftDetector()
        det.fit_baseline(rng.randn(10, 4))

        advisor = MemoryDecayAdvisor(det)
        patterns = rng.randn(5, 4)
        weights = advisor.compute_pattern_weights(patterns)
        # Empty window returns all ones
        assert np.allclose(weights, 1.0)
