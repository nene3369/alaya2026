"""Tests for lmm.scale.stream — StreamingSurprise."""

from __future__ import annotations

import pytest
import numpy as np

from lmm.scale.stream import StreamingSurprise, _TopKBuffer


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

    def test_use_sketch_mode(self):
        """Cover use_sketch=True init and sketch path in fit_stream (lines 49-50, 59-62)."""
        ss = StreamingSurprise(k=5, use_sketch=True, sketch_width=100, sketch_depth=4)
        assert ss._sketch is not None
        chunks = self._make_chunks(n_chunks=3, chunk_size=50)
        ss.fit_stream(iter(chunks))
        assert ss._fitted
        results = list(ss.compute_stream(iter(self._make_chunks(n_chunks=2, chunk_size=50))))
        assert len(results) == 2

    def test_compute_stream_not_fitted_raises(self):
        """Cover RuntimeError when compute_stream called before fit (line 80)."""
        ss = StreamingSurprise(k=5)
        with pytest.raises(RuntimeError, match="fit_stream"):
            list(ss.compute_stream(iter([np.array([1.0, 2.0])])))

    def test_compute_array_not_fitted_raises(self):
        """Cover RuntimeError when compute_array called before fit (line 108)."""
        ss = StreamingSurprise(k=5)
        with pytest.raises(RuntimeError, match="fit_stream"):
            ss.compute_array(np.array([1.0, 2.0]))

    def test_compute_chunk_sparse_histogram(self):
        """Cover fallback when total < 2 or n_bins < 2 (line 131)."""
        ss = StreamingSurprise(k=5)
        # Fit with a single value so histogram has very few bins
        ss.fit_stream(iter([np.array([1.0])]))
        surprises = ss.compute_array(np.array([1.0, 2.0, 3.0]))
        assert surprises.shape == (3,)
        assert np.all(np.isfinite(surprises))

    def test_multidimensional_input(self):
        """Cover _to_scalar with multi-dim data (line 191)."""
        ss = StreamingSurprise(k=5)
        rng = np.random.RandomState(42)
        chunks_2d = [rng.randn(50, 3) for _ in range(3)]
        ss.fit_stream(iter(chunks_2d))
        results = list(ss.compute_stream(iter([rng.randn(30, 3)])))
        assert len(results) == 1
        assert results[0].n_processed == 30

    def test_total_seen_property(self):
        """Cover total_seen property (line 200)."""
        ss = StreamingSurprise(k=5)
        chunks = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0])]
        ss.fit_stream(iter(chunks))
        assert ss.total_seen == 5

    def test_fit_array_convenience(self):
        """Cover fit_array method (line 71)."""
        ss = StreamingSurprise(k=5, chunk_size=30)
        data = np.random.RandomState(42).randn(100)
        ss.fit_array(data)
        assert ss._fitted
        surprises = ss.compute_array(data)
        assert surprises.shape == (100,)

    def test_vectorized_cache_reuse(self):
        """Cover _bin_cache reuse in _compute_chunk_vectorized (line 171-172)."""
        ss = StreamingSurprise(k=5)
        data = np.random.RandomState(42).randn(200)
        ss.fit_stream(iter([data]))
        # First call creates cache
        s1 = ss.compute_array(data[:50])
        # Second call reuses cache (same total and n_bins)
        s2 = ss.compute_array(data[50:100])
        assert s1.shape == (50,)
        assert s2.shape == (50,)


class TestTopKBuffer:
    def test_empty_buffer(self):
        """Cover empty heap in get() (line 218)."""
        buf = _TopKBuffer(k=5)
        indices, scores = buf.get()
        assert len(indices) == 0
        assert len(scores) == 0

    def test_heapreplace(self):
        """Cover heapreplace when buffer is full (line 214)."""
        buf = _TopKBuffer(k=3)
        buf.push(0, 1.0)
        buf.push(1, 2.0)
        buf.push(2, 3.0)
        # Buffer is now full; pushing higher score triggers heapreplace
        buf.push(3, 5.0)
        indices, scores = buf.get()
        assert len(indices) == 3
        assert 5.0 in scores
        # The lowest (1.0) should have been replaced
        assert 1.0 not in scores

    def test_push_lower_score_ignored(self):
        """Cover case where score <= heap[0] is ignored (line 213-214)."""
        buf = _TopKBuffer(k=2)
        buf.push(0, 5.0)
        buf.push(1, 10.0)
        # Score lower than min in heap, should be ignored
        buf.push(2, 3.0)
        indices, scores = buf.get()
        assert set(indices) == {0, 1}

    def test_clear(self):
        """Cover clear method."""
        buf = _TopKBuffer(k=3)
        buf.push(0, 1.0)
        buf.push(1, 2.0)
        buf.clear()
        indices, scores = buf.get()
        assert len(indices) == 0
