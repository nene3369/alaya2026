"""Digital Dharma — Buddhist philosophy-inspired optimization engine."""

from lmm.dharma.api import DharmaLMM
from lmm.dharma.energy import DharmaEnergyTerm, DukkhaTerm, KarunaTerm
from lmm.dharma.engine import EngineResult, UniversalDharmaEngine
from lmm.dharma.patthana import Paccaya, PatthanaEngine
from lmm.dharma.pratitya import Bija, PratityaRAG
from lmm.dharma.sangha import CouncilResult, SanghaOrchestrator
from lmm.dharma.vow import Vow, VowConstraintEngine

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
]
