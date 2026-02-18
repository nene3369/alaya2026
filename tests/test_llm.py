"""Tests for lmm.llm — embeddings, drift, fewshot, reranker, sampler."""

from __future__ import annotations

import numpy as np
import pytest

from lmm.llm.embeddings import EmbeddingAdapter, auto_embed, cosine_similarity, ngram_vectors
from lmm.llm.drift import DriftDetector
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
