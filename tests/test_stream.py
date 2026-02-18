"""Tests for lmm.scale.stream — StreamingSurprise."""

from __future__ import annotations

import numpy as np

from lmm.scale.stream import StreamingSurprise


class TestStreamingSurprise:
    def _make_chunks(self, n_chunks: int = 5, chunk_size: int = 100) -> list[np.ndarray]:
        rng = np.random.RandomState(42)
        return [rng.randn(chunk_size) for _ in range(n_chunks)]

    def test_fit_and_compute_stream(self):
        ss = StreamingSurprise(k=10)
        chunks = self._make_chunks()
        ss.fit_stream(iter(chunks))
        # compute_stream returns a generator
        results = list(ss.compute_stream(iter(self._make_chunks())))
        assert len(results) > 0
        # Get global top-k
        indices, scores = ss.get_top_k()
        assert len(indices) <= 10

    def test_compute_and_select(self):
        ss = StreamingSurprise(k=5)
        chunks = self._make_chunks()
        ss.fit_stream(iter(chunks))
        indices, scores = ss.compute_and_select(iter(self._make_chunks()))
        assert len(indices) <= 5
        assert len(scores) == len(indices)

    def test_single_chunk(self):
        ss = StreamingSurprise(k=5)
        chunks = [np.random.RandomState(42).randn(50)]
        ss.fit_stream(iter(chunks))
        results = list(ss.compute_stream(iter([np.random.RandomState(42).randn(50)])))
        assert len(results) == 1
        assert results[0].n_processed == 50

    def test_compute_array(self):
        ss = StreamingSurprise(k=5)
        data = np.random.RandomState(42).randn(100)
        ss.fit_stream(iter([data]))
        surprises = ss.compute_array(data)
        assert surprises.shape == (100,)
