"""Tests for lmm.dharma.energy — DharmaEnergyTerm and concrete terms."""

from __future__ import annotations

import numpy as np

from lmm.dharma.energy import (
    DukkhaTerm,
    KarunaTerm,
    KarmaTerm,
    MettaTerm,
    PrajnaTerm,
    QueryDukkhaTerm,
    SilaTerm,
    UpekkhaTerm,
)


class TestEnergyTerms:
    def _make_data(self, n: int = 10) -> dict:
        rng = np.random.RandomState(42)
        return {
            "surprises": rng.rand(n),
            "embeddings": rng.randn(n, 5),
            "query_embedding": rng.randn(5),
            "impact_graph": rng.rand(n, n) * 0.5,
            "history_counts": rng.rand(n),
            "k": 3,
        }

    def test_dukkha_builds(self):
        data = self._make_data()
        term = DukkhaTerm(data["surprises"])
        h, J = term.build(10)
        assert h.shape == (10,)

    def test_prajna_builds(self):
        data = self._make_data()
        term = PrajnaTerm(data["surprises"])
        h, J = term.build(10)
        assert h.shape == (10,)

    def test_karuna_builds(self):
        data = self._make_data()
        term = KarunaTerm(data["impact_graph"])
        h, J = term.build(10)
        assert h.shape == (10,)

    def test_metta_builds(self):
        data = self._make_data()
        term = MettaTerm(data["impact_graph"])
        h, J = term.build(10)
        assert h.shape == (10,)

    def test_karma_builds(self):
        data = self._make_data()
        term = KarmaTerm(data["history_counts"])
        h, J = term.build(10)
        assert h.shape == (10,)

    def test_sila_builds(self):
        term = SilaTerm(k=3)
        h, J = term.build(10)
        assert h.shape == (10,)

    def test_query_dukkha_builds(self):
        data = self._make_data()
        term = QueryDukkhaTerm(
            data["query_embedding"],
            data["embeddings"],
        )
        h, J = term.build(10)
        assert h.shape == (10,)

    def test_upekkha_builds(self):
        data = self._make_data()
        term = UpekkhaTerm(data["surprises"])
        h, J = term.build(10)
        assert h.shape == (10,)

    def test_all_terms_have_name(self):
        data = self._make_data()
        terms = [
            DukkhaTerm(data["surprises"]),
            PrajnaTerm(data["surprises"]),
            KarunaTerm(data["impact_graph"]),
            MettaTerm(data["impact_graph"]),
            KarmaTerm(data["history_counts"]),
            SilaTerm(k=3),
            QueryDukkhaTerm(data["query_embedding"], data["embeddings"]),
            UpekkhaTerm(data["surprises"]),
        ]
        for t in terms:
            assert hasattr(t, "name")
            assert isinstance(t.name, str)
            assert len(t.name) > 0
