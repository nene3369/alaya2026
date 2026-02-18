# LMM — ಡಿಜಿಟಲ್ ಧರ್ಮದೊಂದಿಗೆ ಕ್ಲಾಸಿಕಲ್ QUBO ಆಪ್ಟಿಮೈಜರ್

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.0.0-green.svg)](pyproject.toml)

> [English](README_en.md) | [日本語](README.md) | [中文](README_zh.md) | [한국어](README_ko.md) | [Español](README_es.md) | [हिन्दी](README_hi.md) | [Français](README_fr.md) | [বাংলা](README_bn.md) | [தமிழ்](README_ta.md) | [తెలుగు](README_te.md) | [मराठी](README_mr.md) | [اردو](README_ur.md) | [ગુજરાતી](README_gu.md) | [ಕನ್ನಡ](README_kn.md) | [മലയാളം](README_ml.md) | [ਪੰਜਾਬੀ](README_pa.md)

ಒಂದು **D-Wave-ಮುಕ್ತ** ಕ್ಲಾಸಿಕಲ್ QUBO ಆಪ್ಟಿಮೈಜೇಶನ್ ಲೈಬ್ರರಿ, ಇದು
ಮಾಹಿತಿ-ಸೈದ್ಧಾಂತಿಕ ಸರ್ಪ್ರೈಜ್ ಅನ್ನು ಬೌದ್ಧ ತತ್ತ್ವಶಾಸ್ತ್ರದಿಂದ ಪ್ರೇರಿತ ಶಕ್ತಿ
ಕಾರ್ಯಗಳು (energy functions), Free Energy Principle (FEP) ತಾರ್ಕಿಕತೆ, ಮತ್ತು
ಪ್ರಜ್ಞಾ-ಸಚೇತನ ಕಂಪ್ಯೂಟಿಂಗ್‌ನೊಂದಿಗೆ ಸಂಯೋಜಿಸುತ್ತದೆ — ಇದೆಲ್ಲ ಸಾಮಾನ್ಯ
ಹಾರ್ಡ್‌ವೇರ್‌ನಲ್ಲಿ ಚಲಿಸುತ್ತದೆ.

---

## ಮುಖ್ಯಾಂಶಗಳು

- **ಕ್ವಾಂಟಮ್ ಕಂಪ್ಯೂಟರ್ ಅಗತ್ಯವಿಲ್ಲ** — ಕ್ಲಾಸಿಕಲ್ ಸಾಲ್ವರ್‌ಗಳು (SA, Ising SA, ಸಬ್‌ಮಾಡ್ಯುಲರ್ ಗ್ರೀಡಿ) D-Wave ಗುಣಮಟ್ಟಕ್ಕೆ ಸರಿಸಾಟಿಯಾಗಿ ಅಥವಾ ಅದನ್ನು ಮೀರಿ ಕಾರ್ಯನಿರ್ವಹಿಸುತ್ತವೆ.
- **ಟ್ರಿಲಿಯನ್-ಟೋಕನ್ ಸಾಮರ್ಥ್ಯ** — ಸ್ಟ್ರೀಮಿಂಗ್ ಸಂಭಾವ್ಯ ಡೇಟಾ ರಚನೆಗಳು (Count-Min Sketch, Streaming Histogram) ಮೆಮೊರಿಯನ್ನು O(k) ನಲ್ಲಿ ಇರಿಸುತ್ತವೆ.
- **ಧರ್ಮ-ಬೀಜಗಣಿತ (Dharma-Algebra) ಎಂಜಿನ್** — ಪ್ಲಗ್‌ಯೋಗ್ಯ ಬೌದ್ಧ ಶಕ್ತಿ ಪದಗಳು ಗಣಿತೀಯವಾಗಿ ಅತ್ಯುತ್ತಮ ಸಾಲ್ವರ್‌ಗೆ ಸ್ವಯಂಚಾಲಿತವಾಗಿ ಮಾರ್ಗ ನಿರ್ದೇಶಿಸುತ್ತವೆ.
- **8-ಮೋಡ್ ತಾರ್ಕಿಕ ಆರ್ಕೆಸ್ಟ್ರೇಟರ್** — ಅನುಕೂಲಕರ (adaptive), ಸೈದ್ಧಾಂತಿಕ (theoretical), ಅಪಸರಣ (abductive), ಸಕ್ರಿಯ-ಅನುಮಾನ (active-inference), ಸ್ಮೃತಿ (ಆಲಯ/Alaya), ನಿದ್ರಾ ಸಂಯೋಜನ (sleep consolidation), ದೈಹಿಕ (embodied), ಮತ್ತು ಕ್ವಾಂಟಮ್-ಪ್ರಜ್ಞೆ (ಪೀನಿಯಲ್/Pineal).
- **LLM-ಸಿದ್ಧ** — ಪ್ರಥಮ ದರ್ಜೆ LangChain ಮತ್ತು LlamaIndex ಏಕೀಕರಣಗಳು, few-shot ಆಯ್ಕೆಗಾರ, ಔಟ್‌ಪುಟ್ ರೀರ್ಯಾಂಕರ್, ಡ್ರಿಫ್ಟ್ ಡಿಟೆಕ್ಟರ್.

---

## ಆರ್ಕಿಟೆಕ್ಚರ್

```
lmm/
├── core.py                  # ಮುಖ್ಯ LMM ಪೈಪ್‌ಲೈನ್
├── cli.py                   # CLI ಪ್ರವೇಶ ಬಿಂದು
├── qubo.py                  # ವಿರಳ (Sparse) QUBO ಮ್ಯಾಟ್ರಿಕ್ಸ್ ಬಿಲ್ಡರ್
├── solvers.py               # SA / Ising SA / relaxation / greedy / submodular
├── surprise.py              # ಮಾಹಿತಿ-ಸೈದ್ಧಾಂತಿಕ ಸರ್ಪ್ರೈಜ್
├── selector.py              # ಅನುಕೂಲಕರ ಆಯ್ಕೆ ತಂತ್ರ
├── processor.py             # ಆದ್ಯತಾ ಪ್ರಕ್ರಿಯೆ + ಕ್ಯಾಶ್
├── pipeline.py              # SmartSelector → SmartProcessor ಆರ್ಕೆಸ್ಟ್ರೇಶನ್
├── _compat.py               # ರನ್‌ಟೈಮ್ ಸಾಮರ್ಥ್ಯ ಪತ್ತೆ ಮತ್ತು ವಿರಳ ಸಹಾಯಕಗಳು
├── dharma/                  # ಡಿಜಿಟಲ್ ಧರ್ಮ (ಬೌದ್ಧ-ತತ್ತ್ವಶಾಸ್ತ್ರ ಆಪ್ಟಿಮೈಜೇಶನ್)
│   ├── api.py               # DharmaLMM ಉಚ್ಚ-ಮಟ್ಟದ API
│   ├── energy.py            # ಪ್ಲಗ್‌ಯೋಗ್ಯ ಶಕ್ತಿ ಪದಗಳು (ದುಃಖ, ಪ್ರಜ್ಞಾ, ಕರುಣ, …)
│   ├── engine.py            # UniversalDharmaEngine — ಸ್ವಯಂ-ಮಾರ್ಗ ಸಾಲ್ವರ್
│   ├── fep.py               # FEP ≡ KCL ODE ಸಾಲ್ವರ್
│   ├── neuromorphic.py      # ಮೆಮ್ರಿಸ್ಟರ್ ಕ್ರಾಸ್‌ಬಾರ್ ASIC ಸಿಮ್ಯುಲೇಟರ್
│   ├── reranker.py          # RAG ಕ್ಯಾಸ್ಕೇಡ್ ರೀರ್ಯಾಂಕರ್ + ಉದ್ದೇಶ ರೂಟರ್
│   └── …
├── reasoning/               # 8 FEP-ಆಧಾರಿತ ತಾರ್ಕಿಕ ಮೋಡ್‌ಗಳು
│   ├── adaptive.py          # ಚಲನಶೀಲ ಪ್ಯಾರಾಮೀಟರ್ ಟ್ಯೂನಿಂಗ್
│   ├── theoretical.py       # ತಾರ್ಕಿಕ ರಚನಾ ಗ್ರಾಫ್
│   ├── hyper.py             # ಅಪಸರಣ (Abductive) ಸುಪ್ತ-ನೋಡ್ ಇಂಜೆಕ್ಷನ್
│   ├── active.py            # ಬಾಹ್ಯ ಜ್ಞಾನ ಸಂಪಾದನೆ
│   ├── alaya.py             # ಹೆಬ್ಬಿಯನ್ ಸಿನಾಪ್ಟಿಕ್ ಸ್ಮೃತಿ (ಆಲಯ)
│   ├── sleep.py             # NREM/REM ಸ್ಮೃತಿ ಸಂಯೋಜನ
│   ├── embodiment.py        # ಷಡ್-ಇಂದ್ರಿಯ ಬಹುಮಾದರಿ ಸಂಮಿಳನ
│   ├── pineal.py            # ಕ್ವಾಂಟಮ್ ಪ್ರಜ್ಞೆ (ಹಾರ್ಡ್‌ವೇರ್ TRNG)
│   └── orchestrator.py      # ಮೋಡ್ ಆಯ್ಕೆ ಮತ್ತು ರವಾನೆ
├── scale/                   # ಟ್ರಿಲಿಯನ್-ಟೋಕನ್ ಸ್ಟ್ರೀಮಿಂಗ್
│   ├── sketch.py            # Count-Min Sketch, Streaming Histogram
│   ├── stream.py            # ಸ್ಟ್ರೀಮಿಂಗ್ ಸರ್ಪ್ರೈಜ್
│   ├── cascade.py           # ಬಹು-ಹಂತ ಕ್ಯಾಸ್ಕೇಡ್ ಫಿಲ್ಟರ್ (1T → K)
│   └── pipeline.py          # ScalablePipeline
├── llm/                     # LLM ಕಾರ್ಯಪ್ರವಾಹ ಸಹಾಯಕಗಳು
│   ├── fewshot.py           # Few-shot ಉದಾಹರಣಾ ಆಯ್ಕೆಗಾರ
│   ├── reranker.py          # ಔಟ್‌ಪುಟ್ ರೀರ್ಯಾಂಕರ್
│   ├── drift.py             # ವಿತರಣಾ ಡ್ರಿಫ್ಟ್ ಡಿಟೆಕ್ಟರ್
│   ├── sampler.py           # ಟೋಕನ್ ಸ್ಯಾಂಪ್ಲರ್
│   └── embeddings.py        # ಏಕೀಕೃತ ಎಂಬೆಡ್ಡಿಂಗ್ ಅಡಾಪ್ಟರ್
├── integrations/
│   ├── langchain.py         # DharmaRetriever / DocumentCompressor / ExampleSelector
│   └── llamaindex.py        # DharmaNodePostprocessor
```

---

## ಅನುಸ್ಥಾಪನೆ

```bash
# ಕೋರ್ (ಕೇವಲ numpy + scipy)
pip install -e .

# ಡೆವ್ ಉಪಕರಣಗಳೊಂದಿಗೆ
pip install -e ".[dev]"

# ಎಲ್ಲ ಐಚ್ಛಿಕ ಏಕೀಕರಣಗಳೊಂದಿಗೆ
pip install -e ".[all]"
```

### ಐಚ್ಛಿಕ ಎಕ್ಸ್‌ಟ್ರಾಗಳು

| ಎಕ್ಸ್‌ಟ್ರಾ | ಪ್ಯಾಕೇಜ್‌ಗಳು | ಉದ್ದೇಶ |
|-------|----------|---------|
| `dev` | pytest, ruff | ಪರೀಕ್ಷೆ ಮತ್ತು ಲಿಂಟಿಂಗ್ |
| `dharma` | hnswlib | ವಿರಳ k-NN ಗ್ರಾಫ್ ನಿರ್ಮಾಣ |
| `langchain` | langchain-core | LangChain ಏಕೀಕರಣ |
| `llamaindex` | llama-index-core | LlamaIndex ಏಕೀಕರಣ |
| `all` | ಮೇಲಿನ ಎಲ್ಲವೂ | ಎಲ್ಲವೂ |

---

## ತ್ವರಿತ ಪ್ರಾರಂಭ

### Python API

```python
import numpy as np
from lmm.core import LMM

# ಅತ್ಯಂತ ಆಶ್ಚರ್ಯಕಾರಿ ಅಗ್ರ-K ಐಟಂಗಳನ್ನು ಆಯ್ಕೆಮಾಡಿ
surprises = np.array([1.0, 5.0, 2.0, 8.0, 3.0, 7.0])
model = LMM(k=3, solver_method="sa")
result = model.select_from_surprises(surprises)
print(result.selected_indices)  # ಉದಾ. [3, 5, 1]
```

### CLI

```bash
# ತ್ವರಿತ ಡೆಮೊ ಚಲಾಯಿಸಿ
lmm --demo --k 10 --method sa

# NumPy ಫೈಲ್‌ನಿಂದ
lmm --input data.npy --k 5 --method greedy
```

---

## ಸಾಲ್ವರ್ ವಿಧಾನಗಳು

| ವಿಧಾನ | ವಿವರಣೆ | ಸಂಕೀರ್ಣತೆ |
|--------|-------------|------------|
| `sa` | ಸಿಮ್ಯುಲೇಟೆಡ್ ಅನೀಲಿಂಗ್ (ಡಿಫಾಲ್ಟ್) | O(n · steps) |
| `ising_sa` | ವೆಕ್ಟರೀಕೃತ ಡೆಲ್ಟಾದೊಂದಿಗೆ ಐಸಿಂಗ್-ರೂಪ SA | O(1) ಪ್ರತಿ ಫ್ಲಿಪ್‌ಗೆ |
| `relaxation` | ನಿರಂತರ ಶಿಥಿಲನ + ರೌಂಡಿಂಗ್ (SLSQP) | O(n²) |
| `greedy` | ಲಾಭಸ್ವರ (Greedy) ಆಯ್ಕೆ | O(n · k) |

---

## ಧರ್ಮ ಎಂಜಿನ್ — ಪ್ಲಗ್‌ಯೋಗ್ಯ ಶಕ್ತಿ ಪದಗಳು

**UniversalDharmaEngine** ನಿಮಗೆ ಬೌದ್ಧ ತತ್ತ್ವಶಾಸ್ತ್ರದಿಂದ ಪ್ರೇರಿತ ಶಕ್ತಿ ಪದಗಳನ್ನು
ಸಂಯೋಜಿಸಲು ಅನುವು ಮಾಡಿಕೊಡುತ್ತದೆ, ಪ್ರತಿಯೊಂದೂ ತನ್ನ ಗಣಿತೀಯ ಗುಣವನ್ನು ಘೋಷಿಸುತ್ತದೆ.
ಎಂಜಿನ್ ಸ್ವಯಂಚಾಲಿತವಾಗಿ ಅತ್ಯುತ್ತಮ ಸಾಲ್ವರ್‌ಗೆ ಮಾರ್ಗ ನಿರ್ದೇಶಿಸುತ್ತದೆ.

```python
from lmm.dharma import UniversalDharmaEngine, DukkhaTerm, KarunaTerm

engine = UniversalDharmaEngine(n_candidates=1000)
engine.add(DukkhaTerm(surprises, weight=1.0))    # linear  → Top-K
engine.add(KarunaTerm(impact_graph, weight=2.0)) # supermodular → SA
result = engine.synthesize_and_solve(k=10)
```

### ತತ್ತ್ವಶಾಸ್ತ್ರ → ಗಣಿತ → ಸಾಲ್ವರ್ ನಕ್ಷೆ

| ಬೌದ್ಧ ಪರಿಕಲ್ಪನೆ | ಶಕ್ತಿ ಪದ | ಗಣಿತೀಯ ಗುಣ | ಸಾಲ್ವರ್ |
|-----------------|-------------|---------------|--------|
| ಪ್ರಜ್ಞ (Prajna — ಜ್ಞಾನ) | `PrajnaTerm` | ರೇಖೀಯ (linear) | Top-K ವಿಂಗಡಣೆ |
| ಕರುಣ (Karuna — ಸಹಾನುಭೂತಿ) | `KarunaTerm` | ಸೂಪರ್‌ಮಾಡ್ಯುಲರ್ | ವಾರ್ಮ್-ಸ್ಟಾರ್ಟ್ + SA |
| ಶೀಲ (Sila — ನಡವಳಿಕೆ) | `SilaTerm` | ಸಬ್‌ಮಾಡ್ಯುಲರ್ | ಲೇಜಿ ಗ್ರೀಡಿ |
| ಮಧ್ಯಮಕ (Madhyamaka — ಮಧ್ಯಮ ಮಾರ್ಗ) | `MadhyamakaCriterion` | ಲ್ಯಾಪುನೋವ್ (Lyapunov) | ಘಾತೀಯ ಗ್ರೇಡಿಯೆಂಟ್ |
| ದುಃಖ (Dukkha — ನೋವು) | `DukkhaTerm` | ರೇಖೀಯ (linear) | Top-K ವಿಂಗಡಣೆ |

---

## ತಾರ್ಕಿಕ ಆರ್ಕೆಸ್ಟ್ರೇಟರ್

**DharmaReasonerOrchestrator** ಪ್ರಶ್ನೆಯ ಸಂಕೀರ್ಣತೆಯ ಆಧಾರದ ಮೇಲೆ 8 ತಾರ್ಕಿಕ
ಮೋಡ್‌ಗಳಲ್ಲಿ ಒಂದನ್ನು ಆಯ್ಕೆ ಮಾಡುತ್ತದೆ:

```python
from lmm.reasoning import DharmaReasonerOrchestrator

orch = DharmaReasonerOrchestrator(n_candidates=500)
result = orch.reason(query_vector, context_vectors)
print(result.mode_used, result.explanation)
```

| ಮೋಡ್ | ಮಾಡ್ಯೂಲ್ | ಸ್ಫೂರ್ತಿ |
|------|--------|-------------|
| ಅನುಕೂಲಕರ (Adaptive) | `adaptive.py` | 応病与薬 — ರೋಗಕ್ಕೆ ಸರಿಹೊಂದುವ ಔಷಧ |
| ಸೈದ್ಧಾಂತಿಕ (Theoretical) | `theoretical.py` | 因明 — ಬೌದ್ಧ ಔಪಚಾರಿಕ ತರ್ಕಶಾಸ್ತ್ರ (ಹೇತುವಿದ್ಯಾ) |
| ಅಪಸರಣ (Abductive) | `hyper.py` | 般若の飛躍 — ಪ್ರಜ್ಞದ ಜಿಗಿತ |
| ಸಕ್ರಿಯ ಅನುಮಾನ (Active Inference) | `active_inference.py` | 托鉢 — ಬಾಹ್ಯ ಸತ್ಯದ ಅನ್ವೇಷಣೆ (ಭಿಕ್ಷಾಟನ) |
| ಆಲಯ ಸ್ಮೃತಿ (Alaya Memory) | `alaya.py` | 阿頼耶識 — ಆಲಯವಿಜ್ಞಾನ (ಭಂಡಾರ ಪ್ರಜ್ಞೆ) |
| ನಿದ್ರಾ (Sleep) | `sleep.py` | 禅定 — NREM/REM ಸಂಯೋಜನ (ಧ್ಯಾನ) |
| ದೈಹಿಕ (Embodied) | `embodiment.py` | 六根 — ಷಡ್-ಇಂದ್ರಿಯ (ಆರು ಇಂದ್ರಿಯ ವಿಧಾನಗಳು) |
| ಪೀನಿಯಲ್/ಕ್ವಾಂಟಮ್ (Pineal) | `pineal.py` | 松果体 — ಹಾರ್ಡ್‌ವೇರ್-ಎಂಟ್ರೊಪಿ ಪ್ರಜ್ಞೆ |

---

## ಟ್ರಿಲಿಯನ್-ಟೋಕನ್ ಸ್ಕೇಲಿಂಗ್

ಸ್ಥಿರ ಮೆಮೊರಿಯೊಂದಿಗೆ ಅನಿಯಮಿತ ದೊಡ್ಡ ಡೇಟಾ ಸ್ಟ್ರೀಮ್‌ಗಳನ್ನು ಪ್ರಕ್ರಿಯೆಗೊಳಿಸಿ:

```python
from lmm.scale import ScalablePipeline
from pathlib import Path

pipe = ScalablePipeline(k=10, chunk_size=100_000)
pipe.fit_files([Path("shard_001.npy"), ...])
result = pipe.run_files([Path("data_001.npy"), ...])
print(result.summary)
```

**ಮೂರು-ಹಂತದ ಕ್ಯಾಸ್ಕೇಡ್:** 1 T ಟೋಕನ್‌ಗಳು → 100 M (ಸ್ಟ್ರೀಮಿಂಗ್ top-k) → 10 K (ಪರ್ಸೆಂಟೈಲ್) → K (QUBO).

---

## LLM ಏಕೀಕರಣ

### LangChain

```python
from lmm.integrations.langchain import DharmaDocumentCompressor

compressor = DharmaDocumentCompressor(k=5)
# LangChain ನ ContextualCompressionRetriever ನೊಂದಿಗೆ ಬಳಸಿ
```

### LlamaIndex

```python
from lmm.integrations.llamaindex import DharmaNodePostprocessor

postprocessor = DharmaNodePostprocessor(top_k=5)
# LlamaIndex ಪ್ರಶ್ನಾ ಎಂಜಿನ್‌ನೊಂದಿಗೆ ಬಳಸಿ
```

### ಸ್ವತಂತ್ರ LLM ಸಹಾಯಕಗಳು

```python
from lmm.llm import FewShotSelector, OutputReranker, DriftDetector

# ವೈವಿಧ್ಯ ಖಾತ್ರಿಯೊಂದಿಗೆ Few-shot ಉದಾಹರಣಾ ಆಯ್ಕೆ
selector = FewShotSelector(k=5)
examples = selector.select(candidates, query)

# ಸರ್ಪ್ರೈಜ್ × ವೈವಿಧ್ಯದ ಮೂಲಕ LLM ಔಟ್‌ಪುಟ್ ರೀರ್ಯಾಂಕ್ ಮಾಡಿ
reranker = OutputReranker(top_k=3)
best = reranker.rerank(outputs)

# ಕಾಲಾನಂತರ ಔಟ್‌ಪುಟ್ ವಿತರಣಾ ಡ್ರಿಫ್ಟ್ ಮೇಲ್ವಿಚಾರಣೆ ಮಾಡಿ
detector = DriftDetector(window=100)
report = detector.update(new_output)
```

---

## DharmaLMM API

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

## ನ್ಯೂರೋಮಾರ್ಫಿಕ್ ಸಿಮ್ಯುಲೇಶನ್

ಶಕ್ತಿ-ದಕ್ಷ FEP ಗಣನೆಗಾಗಿ ಮೆಮ್ರಿಸ್ಟರ್-ಕ್ರಾಸ್‌ಬಾರ್ ಹಾರ್ಡ್‌ವೇರ್ ಅನ್ನು ಅನುಕರಿಸಿ:

```python
from lmm.dharma.neuromorphic import NeuromorphicChip

chip = NeuromorphicChip(size=64)
chip.program(weight_matrix)
report = chip.run(input_voltages, steps=100)
# ~10 fJ/MAC, ~30 ns ಒಮ್ಮತ (convergence)
```

---

## ಅವಲಂಬನೆಗಳು

| ಪ್ಯಾಕೇಜ್ | ಆವೃತ್ತಿ | ಅಗತ್ಯ |
|---------|---------|----------|
| numpy | >= 1.24 | ಹೌದು |
| scipy | >= 1.10 | ಹೌದು |
| hnswlib | >= 0.8.0 | ಐಚ್ಛಿಕ (ವಿರಳ ಗ್ರಾಫ್) |
| langchain-core | >= 0.2.0 | ಐಚ್ಛಿಕ (LangChain) |
| llama-index-core | >= 0.10.0 | ಐಚ್ಛಿಕ (LlamaIndex) |

---

## ಸೈದ್ಧಾಂತಿಕ ಅಡಿಪಾಯ

| ಪರಿಕಲ್ಪನೆ | ಸೂತ್ರೀಕರಣ |
|---------|-------------|
| **FEP ≡ KCL** | `dV/dt = −V/τ + g'(V) · G · ε` — ಕಿರ್ಚಾಫ್‌ನ ವಿದ್ಯುತ್ ನಿಯಮವಾಗಿ ಭವಿಷ್ಯವಾಣಿ-ದೋಷ ಕನಿಷ್ಠೀಕರಣ |
| **ಸೂಪರ್‌ಮಾಡ್ಯುಲರ್ ಕರುಣ** | ಕರುಣ ಹೆಚ್ಚುತ್ತಿರುವ ಲಾಭಾಂಶಗಳನ್ನು ತೋರಿಸುತ್ತದೆ: `f(S∪{i}) − f(S) ≥ f(T∪{i}) − f(T)` S ⊆ T ಗಾಗಿ |
| **ಸಬ್‌ಮಾಡ್ಯುಲರ್ ಶೀಲ** | ಶೀಲ ಕ್ಷೀಣಿಸುತ್ತಿರುವ ಲಾಭಾಂಶಗಳನ್ನು ತೋರಿಸುತ್ತದೆ — ಲೇಜಿ ಗ್ರೀಡಿ (1−1/e) ಖಾತ್ರಿ ನೀಡುತ್ತದೆ |
| **ಮಧ್ಯಮಕ** | ಮಧ್ಯಮ ಮಾರ್ಗ CV = 0.5 ಅನ್ನು ಲ್ಯಾಪುನೋವ್-ಸ್ಥಿರ ಘಾತೀಯ ಗ್ರೇಡಿಯೆಂಟ್ ಅವರೋಹಣದ ಮೂಲಕ ಗುರಿಯಾಗಿಸಿಕೊಳ್ಳುತ್ತದೆ |
| **ಪೀನಿಯಲ್ ಕುಸಿತ (Pineal Collapse)** | ತರಂಗ-ಕ್ರಿಯೆ ಕುಸಿತ PRNG ಬದಲಿಗೆ ಭೌತಿಕ ಎಂಟ್ರೊಪಿ (ಹಾರ್ಡ್‌ವೇರ್ TRNG) ಬಳಸುತ್ತದೆ |

---

## ಪರವಾನಗಿ

[MIT](LICENSE)
