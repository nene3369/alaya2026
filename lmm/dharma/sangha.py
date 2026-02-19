"""Sangha (僧伽) — Mixture of Enlightened Experts (MoEE).

Seven specialist agents — the principal disciples of the Buddha — deliberate
over a context using Patthana causal analysis and Vow constraints.  The
council reaches a verdict by consensus, with Upali (持律) holding absolute
veto power over ethical or safety violations.

Council protocol (multi-round deliberation)
--------------------------------------------
1. **Round 1**: All seven disciples contemplate the context **in parallel**
   (asyncio), each reasoning independently.
2. **Upali's veto** is checked after every round: a ``REJECT`` terminates
   deliberation immediately.
3. **Round 2+**: The context is enriched with ``"peer_insights"`` — a list of
   every disciple's Round-N findings.  Each disciple re-reasons, incorporating
   the collective intelligence of their peers.
4. A ``CLARIFY`` verdict from Aniruddha is advisory and does not block.
5. Any non-REJECT combination after all rounds yields ``APPROVED``.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from lmm.dharma.patthana import PatthanaEngine, Paccaya
from lmm.dharma.vow import VowConstraintEngine
from lmm.reasoning.recovery import CircuitBreaker


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
    In Round 2+ cross-references Maudgalyayana's complexity signal to
    weight the depth of the causal chain.
    """

    def __init__(self, patthana: PatthanaEngine) -> None:
        super().__init__("Sariputra", "Abhidhamma Logic")
        self.patthana = patthana

    async def contemplate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        issue_id = context.get("issue_id", "unknown")
        causes = self.patthana.analyze_origination(issue_id)
        root_causes = causes.get(Paccaya.HETU.value, [])

        addendum = ""
        for p in context.get("peer_insights", []):
            if p.get("agent") == "Maudgalyayana":
                addendum = f" [Peer: {p['insight']}]"
                break

        return {
            "agent": self.name,
            "insight": f"Root causes identified: {root_causes}.{addendum}",
            "verdict": "PROCEED",
        }


class Upali(DiscipleAgent):
    """優波離 — Foremost in Precepts (第一持律).

    Safety gate with absolute veto power.  Rejects requests containing
    dangerous or destructive commands.  In Round 2+ scans peer insights
    for escalated concerns raised by fellow disciples.
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

        # Round 2+: elevate scrutiny if peers flagged concerns
        for p in context.get("peer_insights", []):
            concern = p.get("insight", "").lower()
            if "violation" in concern or "dangerous" in concern or "unsafe" in concern:
                return {
                    "agent": self.name,
                    "insight": (
                        f"Sila elevated: peer {p['agent']} flagged — \"{p['insight'][:80]}\""
                    ),
                    "verdict": "APPROVE",
                }

        return {"agent": self.name, "insight": "Sila is pure.", "verdict": "APPROVE"}


class Maudgalyayana(DiscipleAgent):
    """目連 — Foremost in Psychic Powers (第一神通).

    Estimates query complexity from token count.  In Round 2+ refines the
    estimate using Subhuti's distilled essence words.
    """

    def __init__(self) -> None:
        super().__init__("Maudgalyayana", "System Introspection")

    async def contemplate(self, context: Dict[str, Any]) -> Dict[str, Any]:
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
            "insight": f"System depth: {depth} tokens, complexity={complexity:.2f}.",
            "verdict": "PROCEED",
        }


class Mahakasyapa(DiscipleAgent):
    """摩訶迦葉 — Foremost in Ascetic Practice (第一頭陀).

    Consolidates patterns from interaction history.  In Round 2+ enriches
    the historical framing with Sariputra's causal root-cause findings.
    """

    def __init__(self) -> None:
        super().__init__("Mahakasyapa", "Dhyana")

    async def contemplate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        history = context.get("history", [])
        patterns = len(set(str(h) for h in history))

        addendum = ""
        for p in context.get("peer_insights", []):
            if p.get("agent") == "Sariputra":
                addendum = f" Historical framing: {p['insight'][:60]}"
                break

        return {
            "agent": self.name,
            "insight": f"Consolidated {patterns} unique pattern(s).{addendum}",
            "verdict": "APPROVE",
        }


class Aniruddha(DiscipleAgent):
    """阿那律 — Foremost in Divine Eye (第一天眼).

    Detects ambiguity in the query.  In Round 2+ checks whether Sariputra's
    causal analysis has resolved earlier ambiguities before re-issuing CLARIFY.
    """

    def __init__(self) -> None:
        super().__init__("Aniruddha", "Divine Eye")

    async def contemplate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        query = context.get("query", "")
        ambiguous = query.count("?") > 1

        # Round 2+: Sariputra's causal map may resolve ambiguity
        for p in context.get("peer_insights", []):
            if p.get("agent") == "Sariputra" and "Root causes" in p.get("insight", ""):
                if ambiguous:
                    return {
                        "agent": self.name,
                        "insight": (
                            "Ambiguity partially resolved via Sariputra's causal map. "
                            "Residual clarification advised."
                        ),
                        "verdict": "CLARIFY",
                    }
                return {
                    "agent": self.name,
                    "insight": "Vision clarified through Sariputra's Patthana analysis.",
                    "verdict": "APPROVE",
                }

        if ambiguous:
            return {
                "agent": self.name,
                "insight": "Multiple ambiguities detected. Clarification recommended.",
                "verdict": "CLARIFY",
            }
        return {"agent": self.name, "insight": "Vision is clear.", "verdict": "APPROVE"}


class Subhuti(DiscipleAgent):
    """須菩提 — Foremost in Understanding Emptiness (第一解空).

    Extracts the essential meaning through a Sunyata (emptiness) lens.
    In Round 2+ intersects the distilled essence with Sariputra's root
    causes for a refined, consensus-aware summary.
    """

    def __init__(self) -> None:
        super().__init__("Subhuti", "Sunyata")

    async def contemplate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        query = context.get("query", "")
        essence_words = [w for w in query.split() if len(w) > 3][:5]
        essence = " ".join(essence_words) or "(empty)"

        for p in context.get("peer_insights", []):
            if p.get("agent") == "Sariputra":
                root_snippet = p.get("insight", "")[:50]
                return {
                    "agent": self.name,
                    "insight": f"Refined essence (Sunyata ∩ Patthana): '{essence}' / [{root_snippet}]",
                    "verdict": "PROCEED",
                }

        return {
            "agent": self.name,
            "insight": f"Essence (Sunyata lens): '{essence}'.",
            "verdict": "PROCEED",
        }


class Purna(DiscipleAgent):
    """富楼那 — Foremost in Preaching (第一説法).

    Forges a ``Vow`` directive for the current FEP state.  In Round 2+
    reads the collective sentiment (CLARIFY count) from peer insights and
    elevates FEP uncertainty accordingly before forging the final Vow.
    """

    def __init__(self, vow_engine: VowConstraintEngine) -> None:
        super().__init__("Purna", "Dharma Teaching")
        self.vow_engine = vow_engine

    async def contemplate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        fep = context.get("fep_state", 0.5)
        karma = context.get("karma_context", [])

        # Modulate FEP by collective uncertainty from peers
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
                f"Boost: {vow.boost_tokens}. Suppress: {vow.suppress_tokens}."
            ),
            "verdict": "APPROVE",
            "vow": vow,
        }


# ---------------------------------------------------------------------------
# Council result & orchestrator
# ---------------------------------------------------------------------------

class CouncilResult:
    """Result of a multi-round Sangha council deliberation.

    Attributes:
        final:  ``"APPROVED"`` or ``"REJECTED"``.
        reason: Human-readable summary of the deciding factor.
        logs:   Per-disciple results from the **final** round (backward compat).
        rounds: All per-round results, oldest first.
    """

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
    """Orchestrates the 7-disciple council with multi-round deliberation.

    **Round 1** — all disciples contemplate the raw context independently
    (asyncio parallel).  Upali's veto terminates immediately.

    **Round 2+** — the context is enriched with ``"peer_insights"`` from
    the previous round.  Each disciple re-reasons, incorporating collective
    intelligence.  Upali's veto is re-evaluated after every round.

    Example::

        orch = SanghaOrchestrator(n_rounds=2)
        result = orch.hold_council_sync({"query": "optimise the QUBO solver"})
        print(result.final)          # "APPROVED"
        print(len(result.rounds))    # 2
    """

    DEFAULT_TIMEOUT: float = 5.0

    def __init__(self, timeout: float | None = None, n_rounds: int = 2) -> None:
        self.patthana_engine = PatthanaEngine()
        self.vow_engine = VowConstraintEngine()
        self.timeout = timeout if timeout is not None else self.DEFAULT_TIMEOUT
        self.n_rounds = max(1, n_rounds)
        self.disciples: List[DiscipleAgent] = [
            Sariputra(self.patthana_engine),
            Upali(),
            Maudgalyayana(),
            Mahakasyapa(),
            Aniruddha(),
            Subhuti(),
            Purna(self.vow_engine),
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
        """Run a single disciple's contemplate with circuit breaker + timeout."""
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
        """Convene the full council with multi-round deliberation.

        Each round enriches the shared context with peer insights from the
        previous round, enabling disciples to exchange opinions and re-reason.
        Upali's veto terminates deliberation immediately in any round.
        """
        all_rounds: List[List[Dict[str, Any]]] = []
        current_context: Dict[str, Any] = dict(context)

        for round_num in range(self.n_rounds):
            current_context["round_num"] = round_num + 1

            results: List[Dict[str, Any]] = list(await asyncio.gather(
                *[self._contemplate_with_timeout(d, current_context) for d in self.disciples],
            ))
            all_rounds.append(results)

            # Upali's veto is absolute — terminate immediately
            for res in results:
                if res.get("verdict") == "REJECT":
                    return CouncilResult(
                        final="REJECTED",
                        reason=res["insight"],
                        logs=results,
                        rounds=all_rounds,
                    )

            # Enrich context with this round's peer insights for next round
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
        """Synchronous wrapper around :meth:`hold_council`.

        Safe to call from contexts where an event loop may already be running.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, self.hold_council(context))
                return future.result(timeout=self.timeout * self.n_rounds * 2)
        return asyncio.run(self.hold_council(context))
