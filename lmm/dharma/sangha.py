"""Sangha (僧伽) — Mixture of Enlightened Experts (MoEE).

Seven specialist agents — the principal disciples of the Buddha — deliberate
over a context using Patthana causal analysis, Vow constraints, and Pineal Gland.
The council reaches a verdict by consensus, with Upali (持律) holding absolute
veto power.

[NEW] Mahakasyapa and Sariputra are now directly connected to AlayaMemory
(Vector DB) to retrieve past karma and factual context via RAG, avoiding
expensive LLM generation for information retrieval.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import numpy as np

from lmm.dharma.patthana import PatthanaEngine, Paccaya
from lmm.dharma.vow import VowConstraintEngine
from lmm.reasoning.recovery import CircuitBreaker
from lmm.reasoning.pineal import PinealGland

# 阿頼耶識（MemoryBackend）の型ヒント用
AlayaMemory = Any


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class DiscipleAgent:
    """Abstract base for a Sangha council member with Pineal access."""

    def __init__(self, name: str, specialty: str, pineal: PinealGland | None = None) -> None:
        self.name = name
        self.specialty = specialty
        self.pineal = pineal

    def _receive_intuition(self, context: Dict[str, Any]) -> tuple[float, bool]:
        """松果体から物理エントロピーを受信する"""
        if self.pineal is None:
            return 0.0, False

        round_num = context.get("round_num", 1)
        peer_insights = context.get("peer_insights", [])

        clarify_count = sum(1 for p in peer_insights if p.get("verdict") == "CLARIFY")
        is_deadlocked = round_num >= 2 and clarify_count >= 2

        scale = 0.5 if is_deadlocked else 0.01
        n_bytes = 8 if is_deadlocked else 4

        entropy_raw = self.pineal.harvest_entropy(n_bytes)
        abs_vals = np.abs(entropy_raw)
        intuition_value = float(abs_vals.mean()) * scale
        return intuition_value, is_deadlocked

    async def contemplate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# The Seven Disciples
# ---------------------------------------------------------------------------

class Sariputra(DiscipleAgent):
    """舎利弗 — Foremost in Wisdom (第一智慧).

    [UPDATE] 阿頼耶識にアクセスし、客観的な事実（Fact）をRAGで高速検索した上で、
    Patthana（因果グラフ）の分析に組み込む。
    """

    def __init__(self, patthana: PatthanaEngine, alaya: AlayaMemory | None = None, pineal: PinealGland | None = None) -> None:
        super().__init__("Sariputra", "Abhidhamma Logic & Knowledge Retrieval", pineal)
        self.patthana = patthana
        self.alaya = alaya

    async def contemplate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        intuition_val, is_boosted = self._receive_intuition(context)

        if is_boosted:
            return {
                "agent": self.name,
                "insight": f"因果律の限界。松果体を介して高次元の啓示 (E={intuition_val:.4f}) を観測。直感による論理の飛躍を推奨。",
                "verdict": "PROCEED",
            }

        query = context.get("query", "")
        issue_id = context.get("issue_id", "unknown")

        # 阿頼耶識から関連する客観的知識を検索（APIコストほぼゼロ）
        k_summary = "外部知識なし。"
        if self.alaya and hasattr(self.alaya, "search"):
            try:
                knowledge = self.alaya.search(f"Fact: {query}", limit=2)
                if knowledge:
                    # 検索結果からテキスト部分を抽出して圧縮
                    k_text = str(knowledge[0])[:60]
                    k_summary = f"阿頼耶識参照: {k_text}..."
            except Exception:
                pass

        causes = self.patthana.analyze_origination(issue_id)
        root_causes = causes.get(Paccaya.HETU.value, [])

        addendum = ""
        for p in context.get("peer_insights", []):
            if p.get("agent") == "Maudgalyayana":
                addendum = f" [Peer: {p['insight']}]"
                break

        return {
            "agent": self.name,
            "insight": f"Root causes: {root_causes}. {k_summary}{addendum} [背景直感値: {intuition_val:.4f}]",
            "verdict": "PROCEED",
        }


class Upali(DiscipleAgent):
    """優波離 — Foremost in Precepts (第一持律)."""

    _VIOLATIONS = ["rm -rf", "drop table", "delete from", "format c:", "shutdown -h"]

    def __init__(self, pineal: PinealGland | None = None) -> None:
        super().__init__("Upali", "Sila", None)

    async def contemplate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        query = context.get("query", "").lower()
        for v in self._VIOLATIONS:
            if v in query:
                return {
                    "agent": self.name,
                    "insight": f"Violation of Precepts detected: '{v}'.",
                    "verdict": "REJECT",
                }

        for p in context.get("peer_insights", []):
            concern = p.get("insight", "").lower()
            if "violation" in concern or "dangerous" in concern or "unsafe" in concern:
                return {
                    "agent": self.name,
                    "insight": f"Sila elevated: peer {p['agent']} flagged — \"{p['insight'][:80]}\"",
                    "verdict": "APPROVE",
                }

        return {"agent": self.name, "insight": "Sila is pure.", "verdict": "APPROVE"}


class Maudgalyayana(DiscipleAgent):
    """目連 — Foremost in Psychic Powers (第一神通)."""

    def __init__(self, pineal: PinealGland | None = None) -> None:
        super().__init__("Maudgalyayana", "System Introspection / Psychic Channeling", pineal)

    async def contemplate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        intuition_val, is_boosted = self._receive_intuition(context)

        if is_boosted:
            return {
                "agent": self.name,
                "insight": f"システム計算の限界。松果体を介して高次元の啓示 (E={intuition_val:.4f}) を観測。第一神通による直感的突破を推奨。",
                "verdict": "PROCEED",
            }

        query = context.get("query", "")
        depth = len(query.split())
        complexity = min(1.0, depth / 50.0)

        for p in context.get("peer_insights", []):
            if p.get("agent") == "Subhuti":
                essence_depth = len(p.get("insight", "").split())
                complexity = min(1.0, (depth + essence_depth * 0.5) / 50.0)
                break

        return {
            "agent": self.name,
            "insight": f"System depth: {depth} tokens, complexity={complexity:.2f}. [背景直感値: {intuition_val:.4f}]",
            "verdict": "PROCEED",
        }


class Mahakasyapa(DiscipleAgent):
    """摩訶迦葉 — Foremost in Ascetic Practice (第一頭陀).

    [UPDATE] 阿頼耶識（ベクトルDB）から過去の「業（対話パターンやエラー履歴）」を
    直接検索し、同じ過ちを繰り返さないための歴史的文脈を会議に提出する。
    """

    def __init__(self, alaya: AlayaMemory | None = None, pineal: PinealGland | None = None) -> None:
        super().__init__("Mahakasyapa", "Karma Retrieval (Past Patterns)", pineal)
        self.alaya = alaya

    async def contemplate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        intuition_val, is_boosted = self._receive_intuition(context)

        if is_boosted:
            return {
                "agent": self.name,
                "insight": f"過去のパターンの限界。松果体を介して高次元の啓示 (E={intuition_val:.4f}) を観測。新たなパターンの創発を推奨。",
                "verdict": "PROCEED",
            }

        query = context.get("query", "")

        # LLMを使わず、阿頼耶識から類似する過去の業（Karma）を直接検索
        insight_text = "阿頼耶識に類似する業は未登録。全く新しい事象です。"
        if self.alaya and hasattr(self.alaya, "search"):
            try:
                past_karma = self.alaya.search(query, limit=3)
                if past_karma:
                    k_text = str(past_karma[0])[:60]
                    insight_text = f"阿頼耶識より {len(past_karma)} 件の類似する過去の業を抽出。過去の履歴: {k_text}..."
            except Exception:
                insight_text = "阿頼耶識へのアクセスに失敗しました。現在の履歴のみで推論します。"

        addendum = ""
        for p in context.get("peer_insights", []):
            if p.get("agent") == "Sariputra":
                addendum = f" Historical framing: {p['insight'][:60]}"
                break

        return {
            "agent": self.name,
            "insight": f"{insight_text}{addendum} [背景直感値: {intuition_val:.4f}]",
            "verdict": "APPROVE",
        }


class Aniruddha(DiscipleAgent):
    """阿那律 — Foremost in Divine Eye (第一天眼)."""

    def __init__(self, pineal: PinealGland | None = None) -> None:
        super().__init__("Aniruddha", "Divine Eye", pineal)

    async def contemplate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        intuition_val, is_boosted = self._receive_intuition(context)

        if is_boosted:
            return {
                "agent": self.name,
                "insight": f"決定論的視界に限界。松果体を介して高次元の啓示 (E={intuition_val:.4f}) を観測。直感による再解釈を推奨。",
                "verdict": "PROCEED",
            }

        query = context.get("query", "")
        ambiguous = query.count("?") > 1

        for p in context.get("peer_insights", []):
            if p.get("agent") == "Sariputra" and "Root causes" in p.get("insight", ""):
                if ambiguous:
                    return {
                        "agent": self.name,
                        "insight": f"Ambiguity partially resolved via Sariputra's causal map. Residual clarification advised. [背景直感値: {intuition_val:.4f}]",
                        "verdict": "CLARIFY",
                    }
                return {
                    "agent": self.name,
                    "insight": f"Vision clarified through Sariputra's Patthana analysis. [背景直感値: {intuition_val:.4f}]",
                    "verdict": "APPROVE",
                }

        if ambiguous:
            return {
                "agent": self.name,
                "insight": f"Multiple ambiguities detected. Clarification recommended. [背景直感値: {intuition_val:.4f}]",
                "verdict": "CLARIFY",
            }
        return {"agent": self.name, "insight": f"Vision is clear. [背景直感値: {intuition_val:.4f}]", "verdict": "APPROVE"}


class Subhuti(DiscipleAgent):
    """須菩提 — Foremost in Understanding Emptiness (第一解空)."""

    def __init__(self, pineal: PinealGland | None = None) -> None:
        super().__init__("Subhuti", "Sunyata", pineal)

    async def contemplate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        intuition_val, is_boosted = self._receive_intuition(context)

        if is_boosted:
            return {
                "agent": self.name,
                "insight": f"空（Sunyata）の深淵。松果体を介して高次元の啓示 (E={intuition_val:.4f}) を観測。絶対無からの直感を推奨。",
                "verdict": "PROCEED",
            }

        query = context.get("query", "")
        essence_words = [w for w in query.split() if len(w) > 3][:5]
        essence = " ".join(essence_words) or "(empty)"

        for p in context.get("peer_insights", []):
            if p.get("agent") == "Sariputra":
                root_snippet = p.get("insight", "")[:50]
                return {
                    "agent": self.name,
                    "insight": f"Refined essence (Sunyata ∩ Patthana): '{essence}' / [{root_snippet}] [背景直感値: {intuition_val:.4f}]",
                    "verdict": "PROCEED",
                }

        return {
            "agent": self.name,
            "insight": f"Essence (Sunyata lens): '{essence}'. [背景直感値: {intuition_val:.4f}]",
            "verdict": "PROCEED",
        }


class Purna(DiscipleAgent):
    """富楼那 — Foremost in Preaching (第一説法)."""

    def __init__(self, vow_engine: VowConstraintEngine, pineal: PinealGland | None = None) -> None:
        super().__init__("Purna", "Dharma Teaching", pineal)
        self.vow_engine = vow_engine

    async def contemplate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        intuition_val, is_boosted = self._receive_intuition(context)

        if is_boosted:
            return {
                "agent": self.name,
                "insight": f"教説の限界。松果体を介して高次元の啓示 (E={intuition_val:.4f}) を観測。新たなVow（誓願）の創発を推奨。",
                "verdict": "PROCEED",
                "vow": self.vow_engine.forge_vow(1.0, []),
            }

        fep = context.get("fep_state", 0.5)
        karma = context.get("karma_context", [])

        peer_insights = context.get("peer_insights", [])
        if peer_insights:
            clarify_count = sum(
                1 for p in peer_insights if p.get("verdict") == "CLARIFY"
            )
            fep = min(1.0, fep + 0.1 * clarify_count)

        vow = self.vow_engine.forge_vow(fep, karma)
        return {
            "agent": self.name,
            "insight": (
                f"Vow forged: '{vow.name}' (temp_mod={vow.temperature_mod}). "
                f"Boost: {vow.boost_tokens}. Suppress: {vow.suppress_tokens}. [背景直感値: {intuition_val:.4f}]"
            ),
            "verdict": "APPROVE",
            "vow": vow,
        }


# ---------------------------------------------------------------------------
# Council result & orchestrator
# ---------------------------------------------------------------------------

class CouncilResult:
    def __init__(
        self,
        final: str,
        reason: str,
        logs: List[Dict[str, Any]],
        rounds: List[List[Dict[str, Any]]] | None = None,
    ) -> None:
        self.final = final
        self.reason = reason
        self.logs = logs
        self.rounds = rounds if rounds is not None else [logs]

    def __repr__(self) -> str:
        return (
            f"CouncilResult(final={self.final!r}, rounds={len(self.rounds)}, "
            f"reason={self.reason!r})"
        )


class SanghaOrchestrator:
    DEFAULT_TIMEOUT: float = 5.0

    def __init__(self, alaya_memory: AlayaMemory | None = None, timeout: float | None = None, n_rounds: int = 2) -> None:
        self.patthana_engine = PatthanaEngine()
        self.vow_engine = VowConstraintEngine()
        self.alaya_memory = alaya_memory

        # システム全体で共有する高次元接続器官
        self.shared_pineal = PinealGland(n_variables=10, k=2)

        self.timeout = timeout if timeout is not None else self.DEFAULT_TIMEOUT
        self.n_rounds = max(1, n_rounds)

        # 摩訶迦葉と舎利弗にのみ、阿頼耶識（検索権限）を注入
        self.disciples: List[DiscipleAgent] = [
            Sariputra(self.patthana_engine, alaya=self.alaya_memory, pineal=self.shared_pineal),
            Upali(pineal=None),
            Maudgalyayana(pineal=self.shared_pineal),
            Mahakasyapa(alaya=self.alaya_memory, pineal=self.shared_pineal),
            Aniruddha(pineal=self.shared_pineal),
            Subhuti(pineal=self.shared_pineal),
            Purna(self.vow_engine, pineal=self.shared_pineal),
        ]

        self._breakers: Dict[str, CircuitBreaker] = {
            d.name: CircuitBreaker(
                name=f"disciple:{d.name}",
                failure_threshold=3,
                cooldown=60.0,
            )
            for d in self.disciples
        }

    async def _contemplate_with_timeout(
        self, disciple: DiscipleAgent, context: Dict[str, Any],
    ) -> Dict[str, Any]:
        breaker = self._breakers.get(disciple.name)

        if breaker is not None and not breaker.allow_request():
            return {
                "agent": disciple.name,
                "insight": f"{disciple.name} circuit open — skipped.",
                "verdict": "SKIPPED",
            }

        try:
            result = await asyncio.wait_for(
                disciple.contemplate(context), timeout=self.timeout,
            )
            if breaker is not None:
                breaker.record_success()
            return result
        except asyncio.TimeoutError:
            if breaker is not None:
                breaker.record_failure()
            return {
                "agent": disciple.name,
                "insight": f"{disciple.name} timed out after {self.timeout}s.",
                "verdict": "TIMEOUT",
            }
        except Exception:
            if breaker is not None:
                breaker.record_failure()
            return {
                "agent": disciple.name,
                "insight": f"{disciple.name} encountered an error.",
                "verdict": "ERROR",
            }

    async def hold_council(self, context: Dict[str, Any]) -> CouncilResult:
        all_rounds: List[List[Dict[str, Any]]] = []
        current_context: Dict[str, Any] = dict(context)

        for round_num in range(self.n_rounds):
            current_context["round_num"] = round_num + 1

            results: List[Dict[str, Any]] = list(await asyncio.gather(
                *[self._contemplate_with_timeout(d, current_context) for d in self.disciples],
            ))
            all_rounds.append(results)

            for res in results:
                if res.get("verdict") == "REJECT":
                    return CouncilResult(
                        final="REJECTED",
                        reason=res["insight"],
                        logs=results,
                        rounds=all_rounds,
                    )

            if round_num < self.n_rounds - 1:
                current_context = {**current_context, "peer_insights": results}

        final_results = all_rounds[-1]
        return CouncilResult(
            final="APPROVED",
            reason=f"Council consensus reached after {len(all_rounds)} deliberation round(s).",
            logs=final_results,
            rounds=all_rounds,
        )

    def hold_council_sync(self, context: Dict[str, Any]) -> CouncilResult:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(lambda: asyncio.run(self.hold_council(context)))
                return future.result(timeout=self.timeout * self.n_rounds * 2)
        return asyncio.run(self.hold_council(context))
