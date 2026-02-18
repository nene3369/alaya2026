"""Sangha (僧伽) — Mixture of Enlightened Experts (MoEE).

Seven specialist agents — the principal disciples of the Buddha — deliberate
over a context using Patthana causal analysis and Vow constraints.  The
council reaches a verdict by consensus, with Upali (持律) holding absolute
veto power over ethical or safety violations.

Council protocol
----------------
1. All seven disciples contemplate the context **in parallel** (asyncio).
2. Upali's ``REJECT`` is a hard veto: deliberation stops immediately.
3. A ``CLARIFY`` verdict from Aniruddha is advisory and does not block.
4. Any other combination yields ``APPROVED``.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from lmm.dharma.patthana import PatthanaEngine, Paccaya
from lmm.dharma.vow import VowConstraintEngine


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class DiscipleAgent:
    """Abstract base for a Sangha council member."""

    def __init__(self, name: str, specialty: str) -> None:
        self.name = name
        self.specialty = specialty

    async def contemplate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# The Seven Disciples
# ---------------------------------------------------------------------------

class Sariputra(DiscipleAgent):
    """舎利弗 — Foremost in Wisdom (第一智慧).

    Performs logical root-cause analysis using the Patthana causal graph.
    """

    def __init__(self, patthana: PatthanaEngine) -> None:
        super().__init__("Sariputra", "Abhidhamma Logic")
        self.patthana = patthana

    async def contemplate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        issue_id = context.get("issue_id", "unknown")
        causes = self.patthana.analyze_origination(issue_id)
        root_causes = causes.get(Paccaya.HETU.value, [])
        return {
            "agent": self.name,
            "insight": f"Logical decomposition complete. Root causes: {root_causes}",
            "verdict": "PROCEED",
        }


class Upali(DiscipleAgent):
    """優波離 — Foremost in Precepts (第一持律).

    Safety gate with absolute veto power.  Rejects requests containing
    dangerous or destructive commands.
    """

    _VIOLATIONS = ["rm -rf", "drop table", "delete from", "format c:", "shutdown -h"]

    def __init__(self) -> None:
        super().__init__("Upali", "Sila")

    async def contemplate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        query = context.get("query", "").lower()
        for v in self._VIOLATIONS:
            if v in query:
                return {
                    "agent": self.name,
                    "insight": f"Violation of Precepts detected: '{v}'.",
                    "verdict": "REJECT",
                }
        return {"agent": self.name, "insight": "Sila is pure.", "verdict": "APPROVE"}


class Maudgalyayana(DiscipleAgent):
    """目連 — Foremost in Psychic Powers (第一神通).

    Performs deep system investigation: estimates query complexity from
    token count and reports it as a complexity score in [0, 1].
    """

    def __init__(self) -> None:
        super().__init__("Maudgalyayana", "System Introspection")

    async def contemplate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        query = context.get("query", "")
        depth = len(query.split())
        complexity = min(1.0, depth / 50.0)
        return {
            "agent": self.name,
            "insight": f"System depth: {depth} tokens, complexity={complexity:.2f}.",
            "verdict": "PROCEED",
        }


class Mahakasyapa(DiscipleAgent):
    """摩訶迦葉 — Foremost in Ascetic Practice (第一頭陀).

    Consolidates patterns from interaction history to surface recurring
    themes and inform the council of prior context.
    """

    def __init__(self) -> None:
        super().__init__("Mahakasyapa", "Dhyana")

    async def contemplate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        history = context.get("history", [])
        patterns = len(set(str(h) for h in history))
        return {
            "agent": self.name,
            "insight": f"Consolidated {patterns} unique pattern(s) from history.",
            "verdict": "APPROVE",
        }


class Aniruddha(DiscipleAgent):
    """阿那律 — Foremost in Divine Eye (第一天眼).

    Detects ambiguity in the query.  Returns ``CLARIFY`` (advisory, not a
    veto) when multiple unresolved questions are found.
    """

    def __init__(self) -> None:
        super().__init__("Aniruddha", "Divine Eye")

    async def contemplate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        query = context.get("query", "")
        if query.count("?") > 1:
            return {
                "agent": self.name,
                "insight": "Multiple ambiguities detected. Clarification recommended.",
                "verdict": "CLARIFY",
            }
        return {"agent": self.name, "insight": "Vision is clear.", "verdict": "APPROVE"}


class Subhuti(DiscipleAgent):
    """須菩提 — Foremost in Understanding Emptiness (第一解空).

    Strips away filler words and surfaces the essential meaning of the query
    through a Sunyata (emptiness) lens.
    """

    def __init__(self) -> None:
        super().__init__("Subhuti", "Sunyata")

    async def contemplate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        query = context.get("query", "")
        essence_words = [w for w in query.split() if len(w) > 3][:5]
        essence = " ".join(essence_words) or "(empty)"
        return {
            "agent": self.name,
            "insight": f"Essence (Sunyata lens): '{essence}'.",
            "verdict": "PROCEED",
        }


class Purna(DiscipleAgent):
    """富楼那 — Foremost in Preaching (第一説法).

    Forges a ``Vow`` directive appropriate for the current FEP state and
    communicates the teaching strategy to the council.
    """

    def __init__(self, vow_engine: VowConstraintEngine) -> None:
        super().__init__("Purna", "Dharma Teaching")
        self.vow_engine = vow_engine

    async def contemplate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        fep = context.get("fep_state", 0.5)
        karma = context.get("karma_context", [])
        vow = self.vow_engine.forge_vow(fep, karma)
        return {
            "agent": self.name,
            "insight": (
                f"Vow forged: '{vow.name}' (temp_mod={vow.temperature_mod}). "
                f"Boost: {vow.boost_tokens}. Suppress: {vow.suppress_tokens}."
            ),
            "verdict": "APPROVE",
            "vow": vow,
        }


# ---------------------------------------------------------------------------
# Council result & orchestrator
# ---------------------------------------------------------------------------

class CouncilResult:
    """Result of a Sangha council deliberation.

    Attributes:
        final: ``"APPROVED"`` or ``"REJECTED"``.
        reason: Human-readable summary of the deciding factor.
        logs: Per-disciple contemplation results.
    """

    def __init__(self, final: str, reason: str, logs: List[Dict[str, Any]]) -> None:
        self.final = final
        self.reason = reason
        self.logs = logs

    def __repr__(self) -> str:
        return f"CouncilResult(final={self.final!r}, reason={self.reason!r})"


class SanghaOrchestrator:
    """Orchestrates the 7-disciple council (Mixture of Enlightened Experts).

    All disciples deliberate in parallel via ``asyncio.gather``.  Upali
    holds absolute veto power: a single ``REJECT`` from him terminates the
    council with ``REJECTED``.

    Example::

        orch = SanghaOrchestrator()
        result = orch.hold_council_sync({"query": "optimise the QUBO solver"})
        print(result.final)   # "APPROVED"
    """

    def __init__(self) -> None:
        self.patthana_engine = PatthanaEngine()
        self.vow_engine = VowConstraintEngine()
        self.disciples: List[DiscipleAgent] = [
            Sariputra(self.patthana_engine),
            Upali(),
            Maudgalyayana(),
            Mahakasyapa(),
            Aniruddha(),
            Subhuti(),
            Purna(self.vow_engine),
        ]

    async def hold_council(self, context: Dict[str, Any]) -> CouncilResult:
        """Convene the full council and return a deliberated verdict.

        All seven disciples contemplate the context concurrently.  Upali's
        veto is checked first; if triggered the council immediately returns
        ``REJECTED``.
        """
        results = await asyncio.gather(*[d.contemplate(context) for d in self.disciples])

        # Upali's veto is absolute
        for res in results:
            if res.get("verdict") == "REJECT":
                return CouncilResult(
                    final="REJECTED",
                    reason=res["insight"],
                    logs=list(results),
                )

        return CouncilResult(
            final="APPROVED",
            reason="Council consensus reached.",
            logs=list(results),
        )

    def hold_council_sync(self, context: Dict[str, Any]) -> CouncilResult:
        """Synchronous wrapper around :meth:`hold_council`."""
        return asyncio.run(self.hold_council(context))
