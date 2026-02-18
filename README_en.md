# LMM — Classical QUBO Optimizer with Digital Dharma

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.0.0-green.svg)](pyproject.toml)

> [日本語](README.md) | [中文](README_zh.md) | [한국어](README_ko.md) | [Español](README_es.md) | [हिन्दी](README_hi.md) | [Français](README_fr.md) | [বাংলা](README_bn.md) | [தமிழ்](README_ta.md) | [తెలుగు](README_te.md) | [मराठी](README_mr.md) | [‎اردو‎](README_ur.md) | [ગુજરાતી](README_gu.md) | [ಕನ್ನಡ](README_kn.md) | [മലയാളം](README_ml.md) | [ਪੰਜਾਬੀ](README_pa.md)

A **D-Wave-free** classical QUBO optimization library that fuses
information-theoretic surprise with Buddhist-philosophy-inspired energy
functions, Free Energy Principle (FEP) reasoning, and consciousness-aware
computing — all running on commodity hardware.

---

## Highlights

- **No quantum computer required** — classical solvers (SA, Ising SA, submodular greedy) match or exceed D-Wave quality.
- **Trillion-token capable** — streaming probabilistic data structures (Count-Min Sketch, Streaming Histogram) keep memory O(k).
- **Dharma-Algebra engine** — pluggable Buddhist energy terms auto-route to the mathematically optimal solver.
- **8-mode reasoning orchestrator** — adaptive, theoretical, abductive, active-inference, memory (Alaya), sleep consolidation, embodied, and quantum-consciousness (Pineal).
- **LLM-ready** — first-class LangChain & LlamaIndex integrations, few-shot selector, output reranker, drift detector.

---

## Architecture

```
lmm/
├── core.py                  # Main LMM pipeline
├── cli.py                   # CLI entry point
├── qubo.py                  # Sparse QUBO matrix builder
├── solvers.py               # SA / Ising SA / relaxation / greedy / submodular
├── surprise.py              # Information-theoretic surprise
├── selector.py              # Adaptive selection strategy
├── processor.py             # Priority processing + cache
├── pipeline.py              # SmartSelector → SmartProcessor orchestration
├── _compat.py               # Runtime capability detection & sparse helpers
├── dharma/                  # Digital Dharma (Buddhist-philosophy optimization)
│   ├── api.py               # DharmaLMM high-level API
│   ├── energy.py            # Pluggable energy terms (Dukkha, Prajna, Karuna, …)
│   ├── engine.py            # UniversalDharmaEngine — auto-routing solver
│   ├── fep.py               # FEP ≡ KCL ODE solver
│   ├── neuromorphic.py      # Memristor crossbar ASIC simulator
│   ├── reranker.py          # RAG cascade reranker + intent router
│   └── …
├── reasoning/               # 8 FEP-based reasoning modes
│   ├── adaptive.py          # Dynamic parameter tuning
│   ├── theoretical.py       # Logical structure graph
│   ├── hyper.py             # Abductive latent-node injection
│   ├── active.py            # External knowledge acquisition
│   ├── alaya.py             # Hebbian synaptic memory (Alaya)
│   ├── sleep.py             # NREM/REM memory consolidation
│   ├── embodiment.py        # 6-sense multimodal fusion
│   ├── pineal.py            # Quantum consciousness (hardware TRNG)
│   └── orchestrator.py      # Mode selection & dispatch
├── scale/                   # Trillion-token streaming
│   ├── sketch.py            # Count-Min Sketch, Streaming Histogram
│   ├── stream.py            # Streaming surprise
│   ├── cascade.py           # Multi-level cascade filter (1T → K)
│   └── pipeline.py          # ScalablePipeline
├── llm/                     # LLM workflow helpers
│   ├── fewshot.py           # Few-shot example selector
│   ├── reranker.py          # Output reranker
│   ├── drift.py             # Distribution drift detector
│   ├── sampler.py           # Token sampler
│   └── embeddings.py        # Unified embedding adapter
├── integrations/
│   ├── langchain.py         # DharmaRetriever / DocumentCompressor / ExampleSelector
│   └── llamaindex.py        # DharmaNodePostprocessor
```

---

## Installation

```bash
# Core (numpy + scipy only)
pip install -e .

# With dev tools
pip install -e ".[dev]"

# With all optional integrations
pip install -e ".[all]"
```

### Optional extras

| Extra | Packages | Purpose |
|-------|----------|---------|
| `dev` | pytest, ruff | Testing & linting |
| `dharma` | hnswlib | Sparse k-NN graph construction |
| `langchain` | langchain-core | LangChain integration |
| `llamaindex` | llama-index-core | LlamaIndex integration |
| `all` | All of the above | Everything |

---

## Quick Start

### Python API

```python
import numpy as np
from lmm.core import LMM

# Select the top-K most surprising items
surprises = np.array([1.0, 5.0, 2.0, 8.0, 3.0, 7.0])
model = LMM(k=3, solver_method="sa")
result = model.select_from_surprises(surprises)
print(result.selected_indices)  # e.g. [3, 5, 1]
```

### CLI

```bash
# Run a quick demo
lmm --demo --k 10 --method sa

# From a NumPy file
lmm --input data.npy --k 5 --method greedy
```

---

## Solver Methods

| Method | Description | Complexity |
|--------|-------------|------------|
| `sa` | Simulated Annealing (default) | O(n · steps) |
| `ising_sa` | Ising-form SA with vectorized delta | O(1) per flip |
| `relaxation` | Continuous relaxation + rounding (SLSQP) | O(n²) |
| `greedy` | Greedy selection | O(n · k) |

---

## Dharma Engine — Pluggable Energy Terms

The **UniversalDharmaEngine** lets you compose Buddhist-philosophy-inspired
energy terms, each declaring its mathematical property. The engine
automatically routes to the optimal solver.

```python
from lmm.dharma import UniversalDharmaEngine, DukkhaTerm, KarunaTerm

engine = UniversalDharmaEngine(n_candidates=1000)
engine.add(DukkhaTerm(surprises, weight=1.0))    # linear  → Top-K
engine.add(KarunaTerm(impact_graph, weight=2.0)) # supermodular → SA
result = engine.synthesize_and_solve(k=10)
```

### Philosophy → Math → Solver mapping

| Buddhist concept | Energy term | Math property | Solver |
|-----------------|-------------|---------------|--------|
| Prajna (Wisdom) | `PrajnaTerm` | linear | Top-K sort |
| Karuna (Compassion) | `KarunaTerm` | supermodular | Warm-start + SA |
| Sila (Conduct) | `SilaTerm` | submodular | Lazy greedy |
| Madhyamaka (Middle Way) | `MadhyamakaCriterion` | Lyapunov | Exponential gradient |
| Dukkha (Suffering) | `DukkhaTerm` | linear | Top-K sort |

---

## Reasoning Orchestrator

The **DharmaReasonerOrchestrator** selects among 8 reasoning modes
based on query complexity:

```python
from lmm.reasoning import DharmaReasonerOrchestrator

orch = DharmaReasonerOrchestrator(n_candidates=500)
result = orch.reason(query_vector, context_vectors)
print(result.mode_used, result.explanation)
```

| Mode | Module | Inspired by |
|------|--------|-------------|
| Adaptive | `adaptive.py` | 応病与薬 — medicine fit to illness |
| Theoretical | `theoretical.py` | 因明 — Buddhist formal logic |
| Abductive | `hyper.py` | 般若の飛躍 — Prajna's leap |
| Active Inference | `active_inference.py` | 托鉢 — seeking external truth |
| Alaya Memory | `alaya.py` | 阿頼耶識 — storehouse consciousness |
| Sleep | `sleep.py` | 禅定 — NREM/REM consolidation |
| Embodied | `embodiment.py` | 六根 — six sense modalities |
| Pineal (Quantum) | `pineal.py` | 松果体 — hardware-entropy consciousness |

---

## Trillion-Token Scaling

Process arbitrarily large data streams with constant memory:

```python
from lmm.scale import ScalablePipeline
from pathlib import Path

pipe = ScalablePipeline(k=10, chunk_size=100_000)
pipe.fit_files([Path("shard_001.npy"), ...])
result = pipe.run_files([Path("data_001.npy"), ...])
print(result.summary)
```

**Three-stage cascade:** 1 T tokens → 100 M (streaming top-k) → 10 K (percentile) → K (QUBO).

---

## LLM Integration

### LangChain

```python
from lmm.integrations.langchain import DharmaDocumentCompressor

compressor = DharmaDocumentCompressor(k=5)
# Use with LangChain's ContextualCompressionRetriever
```

### LlamaIndex

```python
from lmm.integrations.llamaindex import DharmaNodePostprocessor

postprocessor = DharmaNodePostprocessor(top_k=5)
# Use with LlamaIndex query engine
```

### Standalone LLM helpers

```python
from lmm.llm import FewShotSelector, OutputReranker, DriftDetector

# Few-shot example selection with diversity guarantee
selector = FewShotSelector(k=5)
examples = selector.select(candidates, query)

# Rerank LLM outputs by surprise × diversity
reranker = OutputReranker(top_k=3)
best = reranker.rerank(outputs)

# Monitor output distribution drift over time
detector = DriftDetector(window=100)
report = detector.update(new_output)
```

---

## DharmaLMM — High-Level API

```python
from lmm.dharma import DharmaLMM

model = DharmaLMM(
    k=15,
    use_sparse_graph=True,
    use_greedy_warmstart=True,
    use_ising_sa=True,
    use_exponential_balance=True,
)
model.fit(reference_data)
result = model.select_dharma(candidates)
print(result.interpretation.narrative)
```

---

## Neuromorphic Simulation

Simulate memristor-crossbar hardware for energy-efficient FEP computation:

```python
from lmm.dharma.neuromorphic import NeuromorphicChip

chip = NeuromorphicChip(size=64)
chip.program(weight_matrix)
report = chip.run(input_voltages, steps=100)
# ~10 fJ/MAC, ~30 ns convergence
```

---

## Dependencies

| Package | Version | Required |
|---------|---------|----------|
| numpy | >= 1.24 | Yes |
| scipy | >= 1.10 | Yes |
| hnswlib | >= 0.8.0 | Optional (sparse graph) |
| langchain-core | >= 0.2.0 | Optional (LangChain) |
| llama-index-core | >= 0.10.0 | Optional (LlamaIndex) |

---

## Theoretical Foundations

| Concept | Formulation |
|---------|-------------|
| **FEP ≡ KCL** | `dV/dt = −V/τ + g'(V) · G · ε` — prediction-error minimization as Kirchhoff's Current Law |
| **Supermodular Karuna** | Compassion exhibits increasing returns: `f(S∪{i}) − f(S) ≥ f(T∪{i}) − f(T)` for S ⊆ T |
| **Submodular Sila** | Conduct exhibits diminishing returns — lazy greedy gives (1−1/e) guarantee |
| **Madhyamaka** | Middle Way targets CV = 0.5 via Lyapunov-stable exponential gradient descent |
| **Pineal Collapse** | Wavefunction collapse uses physical entropy (hardware TRNG) instead of PRNG |

---

## License

[MIT](LICENSE)
