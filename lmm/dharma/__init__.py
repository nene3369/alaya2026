"""Digital Dharma — Buddhist philosophy-inspired optimization engine."""

from lmm.dharma.api import DharmaLMM
from lmm.dharma.energy import DharmaEnergyTerm, DukkhaTerm, KarunaTerm
from lmm.dharma.engine import EngineResult, UniversalDharmaEngine
from lmm.dharma.patthana import Paccaya, PatthanaEngine
from lmm.dharma.pratitya import Bija, PratityaRAG
from lmm.dharma.sangha import CouncilResult, SanghaOrchestrator
from lmm.dharma.vow import Vow, VowConstraintEngine
from lmm.dharma.topology import (
    AniccaTerm,
    AnattaTerm,
    DharmaTelemetry,
    PratityaTerm,
    TopologyAlert,
    TopologyDriftDetector,
    TopologyEvaluator,
    TopologyHistory,
    evaluate_engine_result,
)
from lmm.dharma.topology_ast import (
    build_gprec_from_codebase,
    compare_codebases,
    get_cached_gprec,
)

__all__ = [
    "DharmaLMM",
    "UniversalDharmaEngine",
    "EngineResult",
    "DharmaEnergyTerm",
    "DukkhaTerm",
    "KarunaTerm",
    # v5.0 — Digital Dharma Expansion Pack
    "Paccaya",
    "PatthanaEngine",
    "Bija",
    "PratityaRAG",
    "Vow",
    "VowConstraintEngine",
    "SanghaOrchestrator",
    "CouncilResult",
    # v6.0 — Dharma-Topology Protocol
    "DharmaTelemetry",
    "TopologyEvaluator",
    "TopologyHistory",
    "TopologyDriftDetector",
    "TopologyAlert",
    "AniccaTerm",
    "AnattaTerm",
    "PratityaTerm",
    "evaluate_engine_result",
    "build_gprec_from_codebase",
    "get_cached_gprec",
    "compare_codebases",
]
