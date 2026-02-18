"""Tests for lmm.processor — SmartProcessor."""

from __future__ import annotations

import numpy as np

from lmm.processor import SmartProcessor, ProcessingReport


class TestSmartProcessor:
    def test_basic_process(self):
        indices = np.array([0, 3, 7, 12, 15])
        scores = np.array([0.9, 0.7, 0.5, 0.3, 0.1])
        processor = SmartProcessor()
        report = processor.process(indices, scores)
        assert isinstance(report, ProcessingReport)
        assert report.n_critical + report.n_normal + report.n_background == len(indices)

    def test_empty_input(self):
        processor = SmartProcessor()
        report = processor.process(np.array([], dtype=int), np.array([], dtype=float))
        assert isinstance(report, ProcessingReport)
        assert report.n_critical == 0

    def test_custom_threshold(self):
        indices = np.array([0, 1, 2, 3, 4])
        scores = np.array([0.95, 0.85, 0.75, 0.65, 0.55])
        processor = SmartProcessor(critical_threshold=0.9)
        report = processor.process(indices, scores)
        assert report.n_critical >= 1

    def test_all_same_score(self):
        indices = np.arange(5)
        scores = np.ones(5) * 0.99
        processor = SmartProcessor(critical_threshold=0.5)
        report = processor.process(indices, scores)
        # All same score -> range=0 -> all "normal"
        assert report.n_normal == 5

    def test_high_scores_critical(self):
        indices = np.arange(5)
        # Need spread: some high, some low
        scores = np.array([0.95, 0.90, 0.5, 0.1, 0.05])
        processor = SmartProcessor(critical_threshold=0.7)
        report = processor.process(indices, scores)
        assert report.n_critical >= 1

    def test_with_process_fn(self):
        processed = []

        def my_fn(idx: int, score: float) -> str:
            processed.append(idx)
            return f"item_{idx}"

        processor = SmartProcessor(process_fn=my_fn)
        indices = np.array([0, 1, 2])
        scores = np.array([0.9, 0.5, 0.1])
        processor.process(indices, scores)
        assert len(processed) == 3
