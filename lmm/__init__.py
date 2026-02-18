"""LMM - Classical QUBO Optimizer with Digital Dharma (no D-Wave required).

Install extras for framework integrations::

    pip install lmm[langchain]     # DharmaDocumentCompressor, DharmaExampleSelector
    pip install lmm[llamaindex]    # DharmaNodePostprocessor
    pip install lmm[all]           # all integrations
"""

from __future__ import annotations

# Activate pure-Python shims when real numpy/scipy are unavailable.
from lmm._vendor import inject as _inject_vendor

_inject_vendor()
del _inject_vendor

__version__ = "1.0.0"

from lmm.core import LMM, LMMResult
from lmm.dharma.api import DharmaLMM
from lmm.llm.drift import DriftDetector
from lmm.llm.fewshot import FewShotSelector
from lmm.llm.reranker import OutputReranker
from lmm.pipeline import Pipeline
from lmm.processor import SmartProcessor
from lmm.qubo import QUBOBuilder
from lmm.selector import SmartSelector
from lmm.solvers import ClassicalQUBOSolver, SubmodularSelector, solve_classical
from lmm.surprise import SurpriseCalculator

__all__ = [
    "LMM",
    "LMMResult",
    "QUBOBuilder",
    "ClassicalQUBOSolver",
    "solve_classical",
    "SubmodularSelector",
    "SurpriseCalculator",
    "SmartSelector",
    "SmartProcessor",
    "Pipeline",
    "DharmaLMM",
    "FewShotSelector",
    "OutputReranker",
    "DriftDetector",
]
