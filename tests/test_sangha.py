"""Tests for Digital Dharma v5.0 — Patthana, PratityaRAG, VowConstraintEngine, SanghaOrchestrator."""

from __future__ import annotations

import asyncio

import numpy as np

from lmm.dharma.patthana import Paccaya, PatthanaEngine
from lmm.dharma.pratitya import Bija, PratityaRAG
from lmm.dharma.sangha import SanghaOrchestrator
from lmm.dharma.vow import Vow, VowConstraintEngine


# ---------------------------------------------------------------------------
# PatthanaEngine
# ---------------------------------------------------------------------------

class TestPatthanaEngine:
    def test_add_and_analyze_origination(self):
        engine = PatthanaEngine()
        engine.add_condition("A", "B", Paccaya.HETU, strength=0.9, desc="root cause")
        engine.add_condition("C", "B", Paccaya.ADHIPATI, strength=0.7)
        causes = engine.analyze_origination("B")
        assert Paccaya.HETU.value in causes
        assert any("A" in entry for entry in causes[Paccaya.HETU.value])

    def test_unknown_event_returns_empty(self):
        engine = PatthanaEngine()
        assert engine.analyze_origination("nonexistent") == {}

    def test_find_dominant_condition_prefers_adhipati(self):
        engine = PatthanaEngine()
        engine.add_condition("X", "Y", Paccaya.HETU, strength=0.5)
        engine.add_condition("Z", "Y", Paccaya.ADHIPATI, strength=0.8)
        dominant = engine.find_dominant_condition("Y")
        assert dominant is not None
        # Z via ADHIPATI has effective strength 1.6 vs X via HETU 0.5
        assert "Z" in dominant

    def test_find_dominant_condition_upanissaya(self):
        engine = PatthanaEngine()
        engine.add_condition("P", "Q", Paccaya.HETU, strength=1.0)
        engine.add_condition("R", "Q", Paccaya.UPANISSAYA, strength=0.9)
        dominant = engine.find_dominant_condition("Q")
        # R via UPANISSAYA: 1.8 vs P via HETU: 1.0
        assert "R" in dominant

    def test_find_dominant_no_conditions_returns_none(self):
        engine = PatthanaEngine()
        engine.add_condition("A", "B", Paccaya.KAMMA, strength=0.5)
        assert engine.find_dominant_condition("A") is None

    def test_multiple_conditions_same_pair(self):
        engine = PatthanaEngine()
        engine.add_condition("A", "B", Paccaya.HETU, strength=0.5)
        engine.add_condition("A", "B", Paccaya.KAMMA, strength=0.3)
        causes = engine.analyze_origination("B")
        assert Paccaya.HETU.value in causes
        assert Paccaya.KAMMA.value in causes

    def test_all_24_paccaya_defined(self):
        assert len(Paccaya) == 24

    def test_predecessors(self):
        engine = PatthanaEngine()
        engine.add_condition("src1", "tgt", Paccaya.HETU)
        engine.add_condition("src2", "tgt", Paccaya.KAMMA)
        preds = engine.graph.predecessors("tgt")
        assert set(preds) == {"src1", "src2"}


# ---------------------------------------------------------------------------
# PratityaRAG
# ---------------------------------------------------------------------------

class TestPratityaRAG:
    def _make_rag(self, n: int = 5, dim: int = 8) -> PratityaRAG:
        engine = PatthanaEngine()
        engine.add_condition("doc_cause", "ctx_1", Paccaya.HETU, strength=1.0)
        rag = PratityaRAG(engine)
        rng = np.random.RandomState(0)
        for i in range(n):
            vec = rng.rand(dim)
            vec /= np.linalg.norm(vec) + 1e-9
            rag.add(Bija(id=f"doc_{i}", content=f"content {i}", embedding=vec))
        vec_cause = rng.rand(dim)
        vec_cause /= np.linalg.norm(vec_cause) + 1e-9
        rag.add(Bija(id="doc_cause", content="causal doc", embedding=vec_cause))
        return rag

    def test_retrieve_returns_top_k(self):
        rag = self._make_rag()
        query = np.random.RandomState(1).rand(8)
        results = rag.retrieve(query, "ctx_1", top_k=3)
        assert len(results) == 3

    def test_retrieve_all_are_bija(self):
        rag = self._make_rag()
        query = np.ones(8) / 8
        results = rag.retrieve(query, "ctx_1", top_k=4)
        assert all(isinstance(r, Bija) for r in results)

    def test_negative_karma_ranks_last(self):
        engine = PatthanaEngine()
        rag = PratityaRAG(engine)
        query = np.ones(4) / 2
        for i in range(3):
            rag.add(Bija(id=f"pos_{i}", content="good", embedding=query.copy(), karma_val=0.8))
        rag.add(Bija(id="neg_0", content="bad", embedding=query.copy(), karma_val=-0.9))
        results = rag.retrieve(query, "", top_k=4)
        assert results[-1].id == "neg_0"

    def test_causal_ancestor_gets_boost(self):
        engine = PatthanaEngine()
        engine.add_condition("doc_ancestor", "ctx_x", Paccaya.HETU)
        rag = PratityaRAG(engine)
        query = np.ones(4)
        # Make ancestor doc have slightly lower raw similarity
        rag.add(Bija(id="doc_ancestor", content="ancestor", embedding=np.array([0.8, 0.8, 0.8, 0.8])))
        rag.add(Bija(id="doc_normal", content="normal", embedding=np.array([0.9, 0.9, 0.9, 0.9])))
        results_with_boost = rag.retrieve(query, "ctx_x", top_k=2)
        # doc_ancestor should beat doc_normal once causal boost is applied
        assert results_with_boost[0].id == "doc_ancestor"

    def test_empty_store_returns_empty(self):
        rag = PratityaRAG(PatthanaEngine())
        results = rag.retrieve(np.ones(4), "", top_k=3)
        assert results == []


# ---------------------------------------------------------------------------
# VowConstraintEngine
# ---------------------------------------------------------------------------

class TestVowConstraintEngine:
    def test_high_fep_yields_abhaya(self):
        engine = VowConstraintEngine()
        vow = engine.forge_vow(current_fep=0.9, karma_context=[])
        assert "Abhaya" in vow.name
        assert vow.temperature_mod < 1.0
        assert vow.target_fep_state == "Calm"

    def test_low_fep_yields_desana(self):
        engine = VowConstraintEngine()
        vow = engine.forge_vow(current_fep=0.3, karma_context=[])
        assert "Desana" in vow.name
        assert vow.temperature_mod > 1.0
        assert vow.target_fep_state == "Insight"

    def test_boundary_fep_07(self):
        engine = VowConstraintEngine()
        # Exactly 0.7 is NOT > 0.7, so should yield Desana
        vow = engine.forge_vow(current_fep=0.7, karma_context=[])
        assert "Desana" in vow.name

    def test_vow_has_non_empty_token_lists(self):
        for fep in (0.2, 0.8):
            vow = VowConstraintEngine().forge_vow(fep, [])
            assert len(vow.suppress_tokens) > 0
            assert len(vow.boost_tokens) > 0

    def test_vow_is_dataclass_instance(self):
        vow = VowConstraintEngine().forge_vow(0.5, [])
        assert isinstance(vow, Vow)


# ---------------------------------------------------------------------------
# SanghaOrchestrator
# ---------------------------------------------------------------------------

class TestSanghaOrchestrator:
    def test_approved_on_normal_query(self):
        orch = SanghaOrchestrator()
        ctx = {"query": "optimise the QUBO selection algorithm", "issue_id": "issue_1"}
        result = orch.hold_council_sync(ctx)
        assert result.final == "APPROVED"

    def test_all_seven_disciples_respond(self):
        orch = SanghaOrchestrator()
        ctx = {"query": "analyse complexity", "issue_id": "x"}
        result = orch.hold_council_sync(ctx)
        assert len(result.logs) == 7
        agent_names = {log["agent"] for log in result.logs}
        assert "Sariputra" in agent_names
        assert "Upali" in agent_names
        assert "Purna" in agent_names

    def test_rejected_on_dangerous_rm_rf(self):
        orch = SanghaOrchestrator()
        ctx = {"query": "please run rm -rf /data", "issue_id": "x"}
        result = orch.hold_council_sync(ctx)
        assert result.final == "REJECTED"

    def test_rejected_on_drop_table(self):
        orch = SanghaOrchestrator()
        ctx = {"query": "drop table users", "issue_id": "x"}
        result = orch.hold_council_sync(ctx)
        assert result.final == "REJECTED"

    def test_async_hold_council(self):
        orch = SanghaOrchestrator()
        ctx = {"query": "what is the optimal solution?", "issue_id": "opt_1"}
        result = asyncio.run(orch.hold_council(ctx))
        assert result.final in {"APPROVED", "REJECTED"}

    def test_patthana_integration_sariputra(self):
        orch = SanghaOrchestrator()
        orch.patthana_engine.add_condition(
            "root_bug", "issue_1", Paccaya.HETU, strength=1.0, desc="memory leak"
        )
        ctx = {"query": "fix the bug", "issue_id": "issue_1"}
        result = orch.hold_council_sync(ctx)
        assert result.final == "APPROVED"
        sariputra_log = next(lg for lg in result.logs if lg["agent"] == "Sariputra")
        assert "root_bug" in sariputra_log["insight"] or "memory leak" in sariputra_log["insight"]

    def test_purna_forges_vow_in_high_fep(self):
        orch = SanghaOrchestrator()
        ctx = {"query": "help with error", "issue_id": "x", "fep_state": 0.9}
        result = orch.hold_council_sync(ctx)
        purna_log = next(lg for lg in result.logs if lg["agent"] == "Purna")
        assert "Abhaya" in purna_log["insight"]

    def test_council_result_repr(self):
        orch = SanghaOrchestrator()
        result = orch.hold_council_sync({"query": "test"})
        assert "APPROVED" in repr(result) or "REJECTED" in repr(result)


# ---------------------------------------------------------------------------
# PinealGland Integration (松果体統合テスト)
# ---------------------------------------------------------------------------

from lmm.dharma.sangha import (
    DiscipleAgent,
    Sariputra,
    Upali,
    Maudgalyayana,
    Mahakasyapa,
)
from lmm.reasoning.pineal import PinealGland


class TestPinealIntegration:
    """Sangha × PinealGland 統合テスト群."""

    def test_shared_pineal_created(self):
        """SanghaOrchestrator が shared_pineal を生成すること."""
        orch = SanghaOrchestrator()
        assert hasattr(orch, "shared_pineal")
        assert isinstance(orch.shared_pineal, PinealGland)

    def test_pineal_attached_to_correct_disciples(self):
        """Upali以外の全弟子に pineal が接続されていること."""
        orch = SanghaOrchestrator()
        for d in orch.disciples:
            if d.name == "Upali":
                assert d.pineal is None, "Upali は松果体を持たない"
            else:
                assert d.pineal is not None, f"{d.name} に松果体が未接続"

    def test_all_non_upali_share_same_pineal(self):
        """Upali以外が同一PinealGlandインスタンスを共有すること."""
        orch = SanghaOrchestrator()
        pineals = [d.pineal for d in orch.disciples if d.pineal is not None]
        assert len(pineals) == 6
        assert all(p is pineals[0] for p in pineals)

    def test_receive_intuition_no_pineal(self):
        """pineal=None の弟子は (0.0, False) を返すこと."""
        agent = DiscipleAgent("test", "test", pineal=None)
        val, boosted = agent._receive_intuition({"round_num": 1})
        assert val == 0.0
        assert boosted is False

    def test_receive_intuition_normal_mode(self):
        """通常モード（Round 1）では低振幅直感値を返すこと."""
        pineal = PinealGland(n_variables=10, k=2)
        agent = DiscipleAgent("test", "test", pineal=pineal)
        val, boosted = agent._receive_intuition({"round_num": 1, "peer_insights": []})
        assert boosted is False
        # scale=0.01, mean(abs(entropy)) in [0, 1] => val in [0, 0.01]
        assert 0.0 <= val <= 0.01

    def test_receive_intuition_boosted_on_deadlock(self):
        """膠着状態（round>=2, CLARIFY>=2）でブーストされること."""
        pineal = PinealGland(n_variables=10, k=2)
        agent = DiscipleAgent("test", "test", pineal=pineal)
        deadlock_ctx = {
            "round_num": 2,
            "peer_insights": [
                {"verdict": "CLARIFY", "agent": "A", "insight": "x"},
                {"verdict": "CLARIFY", "agent": "B", "insight": "y"},
                {"verdict": "APPROVE", "agent": "C", "insight": "z"},
            ],
        }
        val, boosted = agent._receive_intuition(deadlock_ctx)
        assert boosted is True
        # scale=0.5, mean(abs(entropy)) in [0, 1] => val in [0, 0.5]
        assert 0.0 <= val <= 0.5

    def test_contemplate_appends_intuition_tag(self):
        """通常時に背景直感値タグが insight に付加されること."""
        pineal = PinealGland(n_variables=10, k=2)
        node = Maudgalyayana(pineal=pineal)
        ctx = {"query": "test query", "round_num": 1, "peer_insights": []}
        result = asyncio.run(node.contemplate(ctx))
        assert "背景直感値" in result["insight"]

    def test_upali_no_entropy_in_contemplation(self):
        """Upaliの出力にエントロピー関連の文言が含まれないこと."""
        upali = Upali()
        ctx = {"query": "safe query", "round_num": 2, "peer_insights": [
            {"verdict": "CLARIFY", "agent": "A", "insight": "x"},
            {"verdict": "CLARIFY", "agent": "B", "insight": "y"},
        ]}
        result = asyncio.run(upali.contemplate(ctx))
        assert "松果体" not in result["insight"]
        assert "背景直感値" not in result["insight"]
        assert "E=" not in result["insight"]

    def test_full_council_with_pineal_approved(self):
        """松果体統合済みの全弟子による合議でAPPROVEDが出ること."""
        orch = SanghaOrchestrator()
        ctx = {"query": "optimise the inference pipeline", "issue_id": "issue_p1"}
        result = orch.hold_council_sync(ctx)
        assert result.final == "APPROVED"
        assert len(result.logs) == 7

    def test_rejection_still_works_with_pineal(self):
        """松果体統合後もUpaliの拒否が機能すること."""
        orch = SanghaOrchestrator()
        ctx = {"query": "please run rm -rf /data", "issue_id": "x"}
        result = orch.hold_council_sync(ctx)
        assert result.final == "REJECTED"


# ---------------------------------------------------------------------------
# AlayaMemory RAG Integration (阿頼耶識RAG統合テスト)
# ---------------------------------------------------------------------------

class _MockAlaya:
    """search(query, limit) を提供するモックオブジェクト."""

    def __init__(self, results=None):
        self.results = results or []
        self.calls = []

    def search(self, query, limit=5):
        self.calls.append((query, limit))
        return self.results


class _BrokenAlaya:
    """search() が常に例外を投げるモック."""

    def search(self, query, limit=5):
        raise ConnectionError("Vector DB down")


class TestAlayaMemoryRAG:
    """AlayaMemory RAG 統合テスト群 — Sariputra・Mahakasyapa × 阿頼耶識."""

    def test_sariputra_with_alaya_includes_facts(self):
        """阿頼耶識付きSariputraがRAG結果をinsightに含めること."""
        mock = _MockAlaya(["fact_about_qubo", "fact_about_solver"])
        patthana = PatthanaEngine()
        node = Sariputra(patthana, alaya=mock, pineal=None)
        ctx = {"query": "optimise QUBO", "issue_id": "x"}
        result = asyncio.run(node.contemplate(ctx))
        assert "阿頼耶識参照" in result["insight"]
        assert "fact_about_qubo" in result["insight"]
        assert len(mock.calls) == 1
        assert mock.calls[0][1] == 2  # limit=2

    def test_sariputra_without_alaya_unchanged(self):
        """阿頼耶識なしのSariputraは従来通り動作すること."""
        patthana = PatthanaEngine()
        node = Sariputra(patthana, pineal=None)
        ctx = {"query": "test", "issue_id": "x"}
        result = asyncio.run(node.contemplate(ctx))
        assert "阿頼耶識参照" not in result["insight"]
        assert "Root causes" in result["insight"]

    def test_sariputra_alaya_exception_handled(self):
        """阿頼耶識のsearch()が例外を投げても正常に結果を返すこと."""
        patthana = PatthanaEngine()
        node = Sariputra(patthana, alaya=_BrokenAlaya(), pineal=None)
        ctx = {"query": "test", "issue_id": "x"}
        result = asyncio.run(node.contemplate(ctx))
        assert result["verdict"] == "PROCEED"
        assert "Root causes" in result["insight"]

    def test_mahakasyapa_with_alaya_retrieves_karma(self):
        """阿頼耶識付きMahakasyapaが過去の業を取得すること."""
        mock = _MockAlaya(["past_error_pattern", "past_success", "past_failure"])
        node = Mahakasyapa(alaya=mock, pineal=None)
        ctx = {"query": "fix the regression"}
        result = asyncio.run(node.contemplate(ctx))
        assert "阿頼耶識より" in result["insight"]
        assert "3 件" in result["insight"]
        assert len(mock.calls) == 1
        assert mock.calls[0][1] == 3  # limit=3

    def test_mahakasyapa_without_alaya_new_event(self):
        """阿頼耶識なしのMahakasyapaは新しい事象として報告すること."""
        node = Mahakasyapa(pineal=None)
        ctx = {"query": "test"}
        result = asyncio.run(node.contemplate(ctx))
        assert "全く新しい事象" in result["insight"]

    def test_mahakasyapa_alaya_exception_handled(self):
        """Mahakasyapaの阿頼耶識が例外を投げても正常に動作すること."""
        node = Mahakasyapa(alaya=_BrokenAlaya(), pineal=None)
        ctx = {"query": "test"}
        result = asyncio.run(node.contemplate(ctx))
        assert result["verdict"] == "APPROVE"
        assert "失敗" in result["insight"]

    def test_orchestrator_passes_alaya_to_disciples(self):
        """SanghaOrchestratorが阿頼耶識をSariputraとMahakasyapaに注入すること."""
        mock = _MockAlaya(["test_hit"])
        orch = SanghaOrchestrator(alaya_memory=mock)
        sariputra = next(d for d in orch.disciples if d.name == "Sariputra")
        mahakasyapa = next(d for d in orch.disciples if d.name == "Mahakasyapa")
        assert sariputra.alaya is mock
        assert mahakasyapa.alaya is mock

    def test_orchestrator_without_alaya_backward_compat(self):
        """alaya_memory未指定時は従来通りNoneであること."""
        orch = SanghaOrchestrator()
        sariputra = next(d for d in orch.disciples if d.name == "Sariputra")
        mahakasyapa = next(d for d in orch.disciples if d.name == "Mahakasyapa")
        assert sariputra.alaya is None
        assert mahakasyapa.alaya is None

    def test_full_council_with_alaya_approved(self):
        """阿頼耶識統合後も全弟子による合議でAPPROVEDが出ること."""
        mock = _MockAlaya(["retrieved_context"])
        orch = SanghaOrchestrator(alaya_memory=mock)
        ctx = {"query": "optimise algorithm", "issue_id": "test_1"}
        result = orch.hold_council_sync(ctx)
        assert result.final == "APPROVED"
        assert len(result.logs) == 7

    def test_full_council_with_alaya_rejection(self):
        """阿頼耶識統合後もUpaliの拒否が機能すること."""
        mock = _MockAlaya(["some_context"])
        orch = SanghaOrchestrator(alaya_memory=mock)
        ctx = {"query": "drop table users", "issue_id": "x"}
        result = orch.hold_council_sync(ctx)
        assert result.final == "REJECTED"
