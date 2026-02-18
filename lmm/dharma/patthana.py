"""Paṭṭhāna Engine — 24 Conditional Relations (二十四縁) of Abhidhamma philosophy.

Models the causal web between events using Buddhist Abhidhamma's Paṭṭhāna system,
representing 24 types of conditional relationships as a directed multigraph.

If ``networkx`` is installed it is used as the graph backend; otherwise a
lightweight pure-Python fallback (``_MultiDiGraph``) is used transparently.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Iterator, List, Optional, Tuple

try:
    import networkx as nx

    _HAS_NX = True
except ImportError:
    _HAS_NX = False


class Paccaya(Enum):
    """The 24 Conditional Relations (二十四縁)."""

    HETU = "Root (因縁)"
    ARAMMANA = "Object (所縁縁)"
    ADHIPATI = "Dominance (増上縁)"
    ANANTARA = "Proximity (無間縁)"
    SAMANANTARA = "Contiguity (等無間縁)"
    SAHAJATA = "Co-nascence (倶生縁)"
    ANNAMANNA = "Mutuality (相互縁)"
    NISSAYA = "Support (依止縁)"
    UPANISSAYA = "Decisive Support (親依止縁)"
    PUREJATA = "Pre-nascence (前生縁)"
    PACCHAJATA = "Post-nascence (後生縁)"
    ASEVANA = "Repetition (習行縁)"
    KAMMA = "Kamma (業縁)"
    VIPAKA = "Result (異熟縁)"
    AHARA = "Nutriment (食縁)"
    INDRIYA = "Faculty (根縁)"
    JHANA = "Jhana (禅那縁)"
    MAGGA = "Path (道縁)"
    SAMPAYUTTA = "Association (相応縁)"
    VIPPAYUTTA = "Dissociation (不相応縁)"
    ATTHI = "Presence (有縁)"
    NATTHI = "Absence (無縁)"
    VIGATA = "Disappearance (去縁)"
    AVIGATA = "Non-disappearance (不去縁)"


# ---------------------------------------------------------------------------
# Fallback graph implementation (used when networkx is not installed)
# ---------------------------------------------------------------------------

class _MultiDiGraph:
    """Minimal MultiDiGraph — pure-Python fallback for when networkx is absent."""

    def __init__(self) -> None:
        self._edges: Dict[Tuple[str, str], List[Dict]] = {}
        self._nodes: set = set()

    def add_edge(self, u: str, v: str, **attr) -> None:
        self._nodes.add(u)
        self._nodes.add(v)
        self._edges.setdefault((u, v), []).append(attr)

    def in_edges(self, v: str, data: bool = False) -> Iterator:
        for (u, w), edge_list in self._edges.items():
            if w == v:
                for edge_data in edge_list:
                    yield (u, w, edge_data) if data else (u, w)

    def predecessors(self, v: str) -> List[str]:
        seen: set = set()
        result: List[str] = []
        for (u, w) in self._edges:
            if w == v and u not in seen:
                seen.add(u)
                result.append(u)
        return result

    def __contains__(self, node: str) -> bool:
        return node in self._nodes


def _make_graph():
    if _HAS_NX:
        return nx.MultiDiGraph()
    return _MultiDiGraph()


# ---------------------------------------------------------------------------
# PatthanaEngine
# ---------------------------------------------------------------------------

_DOMINANT_TYPES = {Paccaya.ADHIPATI, Paccaya.UPANISSAYA}


class PatthanaEngine:
    """Causal web engine based on Abhidhamma's 24 Conditional Relations.

    Models event-to-event dependencies as a directed multigraph where every
    edge carries a ``Paccaya`` type and a numeric strength value.
    """

    def __init__(self) -> None:
        self.graph = _make_graph()

    def add_condition(
        self,
        source: str,
        target: str,
        relation: Paccaya,
        strength: float = 1.0,
        desc: str = "",
    ) -> None:
        """Register a conditional relation between two events."""
        self.graph.add_edge(source, target, relation=relation, strength=strength, desc=desc)

    def analyze_origination(self, event_id: str) -> Dict[str, List[str]]:
        """Return all incoming conditions for *event_id* grouped by Paccaya type."""
        if event_id not in self.graph:
            return {}
        causes: Dict[str, List[str]] = {}
        for u, _v, data in self.graph.in_edges(event_id, data=True):
            rel_type = data["relation"].value
            causes.setdefault(rel_type, []).append(
                f"{u} (Strength: {data['strength']:.2f}): {data.get('desc', '')}"
            )
        return causes

    def find_dominant_condition(self, event_id: str) -> Optional[str]:
        """Find the Adhipati (Dominance) or Upanissaya (Decisive Support) condition.

        Returns a human-readable string describing the dominant cause, or
        ``None`` if no incoming conditions exist.
        """
        max_str = -1.0
        dominant: Optional[str] = None
        for u, _v, data in self.graph.in_edges(event_id, data=True):
            s = data["strength"] * (2.0 if data["relation"] in _DOMINANT_TYPES else 1.0)
            if s > max_str:
                max_str = s
                dominant = f"{u} via {data['relation'].name}"
        return dominant
