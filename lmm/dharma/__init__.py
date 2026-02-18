"""Digital Dharma — Buddhist philosophy-inspired optimization engine."""

from lmm.dharma.api import DharmaLMM
from lmm.dharma.energy import DharmaEnergyTerm, DukkhaTerm, KarunaTerm
from lmm.dharma.engine import EngineResult, UniversalDharmaEngine

__all__ = [
    "DharmaLMM",
    "UniversalDharmaEngine",
    "EngineResult",
    "DharmaEnergyTerm",
    "DukkhaTerm",
    "KarunaTerm",
]
