# LMM — ਡਿਜੀਟਲ ਧਰਮ ਨਾਲ ਕਲਾਸੀਕਲ QUBO ਆਪਟੀਮਾਈਜ਼ਰ

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.0.0-green.svg)](pyproject.toml)

> [English](README_en.md) | [日本語](README.md) | [中文](README_zh.md) | [한국어](README_ko.md) | [Español](README_es.md) | [हिन्दी](README_hi.md) | [Français](README_fr.md) | [বাংলা](README_bn.md) | [தமிழ்](README_ta.md) | [తెలుగు](README_te.md) | [मराठी](README_mr.md) | [‎اردو‎](README_ur.md) | [ગુજરાતી](README_gu.md) | [ಕನ್ನಡ](README_kn.md) | [മലയാളം](README_ml.md) | [ਪੰਜਾਬੀ](README_pa.md)

ਇੱਕ **D-Wave-ਮੁਕਤ** ਕਲਾਸੀਕਲ QUBO ਆਪਟੀਮਾਈਜ਼ੇਸ਼ਨ ਲਾਇਬ੍ਰੇਰੀ ਜੋ
ਸੂਚਨਾ-ਸਿਧਾਂਤਕ ਹੈਰਾਨੀ ਨੂੰ ਬੋਧ ਦਰਸ਼ਨ ਤੋਂ ਪ੍ਰੇਰਿਤ ਊਰਜਾ ਫੰਕਸ਼ਨਾਂ,
Free Energy Principle (FEP) ਤਰਕ-ਪ੍ਰਣਾਲੀ, ਅਤੇ ਚੇਤਨਾ-ਸਚੇਤ
ਕੰਪਿਊਟਿੰਗ ਨਾਲ ਜੋੜਦੀ ਹੈ — ਇਹ ਸਭ ਸਾਧਾਰਨ ਹਾਰਡਵੇਅਰ 'ਤੇ ਚੱਲਦਾ ਹੈ।

---

## ਮੁੱਖ ਵਿਸ਼ੇਸ਼ਤਾਵਾਂ

- **ਕੁਆਂਟਮ ਕੰਪਿਊਟਰ ਦੀ ਲੋੜ ਨਹੀਂ** — ਕਲਾਸੀਕਲ ਸੌਲਵਰ (SA, Ising SA, ਸਬਮੌਡਿਊਲਰ ਗ੍ਰੀਡੀ) D-Wave ਦੀ ਗੁਣਵੱਤਾ ਨਾਲ ਮੇਲ ਖਾਂਦੇ ਜਾਂ ਉਸ ਤੋਂ ਵੀ ਵਧੀਆ ਹਨ।
- **ਟ੍ਰਿਲੀਅਨ-ਟੋਕਨ ਸਮਰੱਥ** — ਸਟ੍ਰੀਮਿੰਗ ਸੰਭਾਵਨਾ ਡੇਟਾ ਢਾਂਚੇ (Count-Min Sketch, Streaming Histogram) ਮੈਮੋਰੀ O(k) ਵਿੱਚ ਰੱਖਦੇ ਹਨ।
- **ਧਰਮ-ਅਲਜਬਰਾ ਇੰਜਣ** — ਪਲੱਗਯੋਗ ਬੋਧ ਊਰਜਾ ਪਦ ਗਣਿਤਕ ਤੌਰ 'ਤੇ ਅਨੁਕੂਲ ਸੌਲਵਰ ਨੂੰ ਆਪਣੇ-ਆਪ ਰੂਟ ਕਰਦੇ ਹਨ।
- **8-ਮੋਡ ਤਰਕ ਆਰਕੈਸਟ੍ਰੇਟਰ** — ਅਨੁਕੂਲੀ (adaptive), ਸਿਧਾਂਤਕ (theoretical), ਅਪਵਾਦਾਤਮਕ (abductive), ਸਰਗਰਮ-ਅਨੁਮਾਨ (active-inference), ਯਾਦਦਾਸ਼ਤ (ਆਲਯ/Alaya), ਨੀਂਦ ਇਕਜੁੱਟੀਕਰਨ (sleep consolidation), ਮੂਰਤ (embodied), ਅਤੇ ਕੁਆਂਟਮ-ਚੇਤਨਾ (ਪਾਈਨੀਅਲ/Pineal)।
- **LLM-ਤਿਆਰ** — ਪਹਿਲੀ ਸ਼੍ਰੇਣੀ LangChain ਅਤੇ LlamaIndex ਏਕੀਕਰਨ, few-shot ਚੋਣਕਰਤਾ, ਆਉਟਪੁੱਟ ਰੀਰੈਂਕਰ, ਡ੍ਰਿਫ਼ਟ ਡਿਟੈਕਟਰ।

---

## ਆਰਕੀਟੈਕਚਰ

```
lmm/
├── core.py                  # ਮੁੱਖ LMM ਪਾਈਪਲਾਈਨ
├── cli.py                   # CLI ਦਾਖਲਾ ਬਿੰਦੂ
├── qubo.py                  # ਸਪਾਰਸ QUBO ਮੈਟ੍ਰਿਕਸ ਬਿਲਡਰ
├── solvers.py               # SA / Ising SA / relaxation / greedy / submodular
├── surprise.py              # ਸੂਚਨਾ-ਸਿਧਾਂਤਕ ਹੈਰਾਨੀ
├── selector.py              # ਅਨੁਕੂਲੀ ਚੋਣ ਰਣਨੀਤੀ
├── processor.py             # ਤਰਜੀਹ ਪ੍ਰੋਸੈਸਿੰਗ + ਕੈਸ਼
├── pipeline.py              # SmartSelector → SmartProcessor ਆਰਕੈਸਟ੍ਰੇਸ਼ਨ
├── _compat.py               # ਰਨਟਾਈਮ ਸਮਰੱਥਾ ਖੋਜ ਅਤੇ ਸਪਾਰਸ ਸਹਾਇਕ
├── dharma/                  # ਡਿਜੀਟਲ ਧਰਮ (ਬੋਧ-ਦਰਸ਼ਨ ਆਪਟੀਮਾਈਜ਼ੇਸ਼ਨ)
│   ├── api.py               # DharmaLMM ਉੱਚ-ਪੱਧਰੀ API
│   ├── energy.py            # ਪਲੱਗਯੋਗ ਊਰਜਾ ਪਦ (ਦੁੱਖ, ਪ੍ਰਗਿਆ, ਕਰੁਣਾ, …)
│   ├── engine.py            # UniversalDharmaEngine — ਆਟੋ-ਰੂਟਿੰਗ ਸੌਲਵਰ
│   ├── fep.py               # FEP ≡ KCL ODE ਸੌਲਵਰ
│   ├── neuromorphic.py      # ਮੈਮਰਿਸਟਰ ਕ੍ਰੌਸਬਾਰ ASIC ਸਿਮੂਲੇਟਰ
│   ├── reranker.py          # RAG ਕੈਸਕੇਡ ਰੀਰੈਂਕਰ + ਇਰਾਦਾ ਰਾਊਟਰ
│   └── …
├── reasoning/               # 8 FEP-ਆਧਾਰਿਤ ਤਰਕ ਮੋਡ
│   ├── adaptive.py          # ਗਤੀਸ਼ੀਲ ਪੈਰਾਮੀਟਰ ਟਿਊਨਿੰਗ
│   ├── theoretical.py       # ਤਾਰਕਿਕ ਢਾਂਚਾ ਗ੍ਰਾਫ਼
│   ├── hyper.py             # ਅਪਵਾਦਾਤਮਕ ਲੁਕੇ-ਨੋਡ ਇੰਜੈਕਸ਼ਨ
│   ├── active.py            # ਬਾਹਰੀ ਗਿਆਨ ਪ੍ਰਾਪਤੀ
│   ├── alaya.py             # ਹੈਬੀਅਨ ਸਾਈਨੈਪਟਿਕ ਯਾਦਦਾਸ਼ਤ (ਆਲਯ)
│   ├── sleep.py             # NREM/REM ਯਾਦਦਾਸ਼ਤ ਇਕਜੁੱਟੀਕਰਨ
│   ├── embodiment.py        # ਛੇ-ਗਿਆਨੇਂਦਰੀ ਬਹੁ-ਮੋਡਲ ਸੰਯੋਜਨ
│   ├── pineal.py            # ਕੁਆਂਟਮ ਚੇਤਨਾ (ਹਾਰਡਵੇਅਰ TRNG)
│   └── orchestrator.py      # ਮੋਡ ਚੋਣ ਅਤੇ ਡਿਸਪੈਚ
├── scale/                   # ਟ੍ਰਿਲੀਅਨ-ਟੋਕਨ ਸਟ੍ਰੀਮਿੰਗ
│   ├── sketch.py            # Count-Min Sketch, Streaming Histogram
│   ├── stream.py            # ਸਟ੍ਰੀਮਿੰਗ ਹੈਰਾਨੀ
│   ├── cascade.py           # ਬਹੁ-ਪੱਧਰੀ ਕੈਸਕੇਡ ਫ਼ਿਲਟਰ (1T → K)
│   └── pipeline.py          # ScalablePipeline
├── llm/                     # LLM ਵਰਕਫ਼ਲੋ ਸਹਾਇਕ
│   ├── fewshot.py           # Few-shot ਉਦਾਹਰਨ ਚੋਣਕਰਤਾ
│   ├── reranker.py          # ਆਉਟਪੁੱਟ ਰੀਰੈਂਕਰ
│   ├── drift.py             # ਵੰਡ ਡ੍ਰਿਫ਼ਟ ਡਿਟੈਕਟਰ
│   ├── sampler.py           # ਟੋਕਨ ਸੈਂਪਲਰ
│   └── embeddings.py        # ਇਕੱਲੇ ਏਕੀਕ੍ਰਿਤ ਏਂਬੈਡਿੰਗ ਅਡੈਪਟਰ
├── integrations/
│   ├── langchain.py         # DharmaRetriever / DocumentCompressor / ExampleSelector
│   └── llamaindex.py        # DharmaNodePostprocessor
```

---

## ਇੰਸਟਾਲੇਸ਼ਨ

```bash
# ਕੋਰ (ਸਿਰਫ਼ numpy + scipy)
pip install -e .

# ਡੇਵ ਟੂਲਸ ਨਾਲ
pip install -e ".[dev]"

# ਸਾਰੇ ਵਿਕਲਪਿਕ ਏਕੀਕਰਨਾਂ ਨਾਲ
pip install -e ".[all]"
```

### ਵਿਕਲਪਿਕ ਐਕਸਟ੍ਰਾ

| ਐਕਸਟ੍ਰਾ | ਪੈਕੇਜ | ਮਕਸਦ |
|-------|----------|---------|
| `dev` | pytest, ruff | ਟੈਸਟਿੰਗ ਅਤੇ ਲਿੰਟਿੰਗ |
| `dharma` | hnswlib | ਸਪਾਰਸ k-NN ਗ੍ਰਾਫ਼ ਨਿਰਮਾਣ |
| `langchain` | langchain-core | LangChain ਏਕੀਕਰਨ |
| `llamaindex` | llama-index-core | LlamaIndex ਏਕੀਕਰਨ |
| `all` | ਉਪਰੋਕਤ ਸਭ | ਸਭ ਕੁਝ |

---

## ਤੇਜ਼ ਸ਼ੁਰੂਆਤ

### Python API

```python
import numpy as np
from lmm.core import LMM

# ਸਿਖਰ-K ਸਭ ਤੋਂ ਹੈਰਾਨ ਕਰਨ ਵਾਲੇ ਆਈਟਮ ਚੁਣੋ
surprises = np.array([1.0, 5.0, 2.0, 8.0, 3.0, 7.0])
model = LMM(k=3, solver_method="sa")
result = model.select_from_surprises(surprises)
print(result.selected_indices)  # ਉਦਾ. [3, 5, 1]
```

### CLI

```bash
# ਇੱਕ ਤੇਜ਼ ਡੈਮੋ ਚਲਾਓ
lmm --demo --k 10 --method sa

# NumPy ਫ਼ਾਈਲ ਤੋਂ
lmm --input data.npy --k 5 --method greedy
```

---

## ਸੌਲਵਰ ਵਿਧੀਆਂ

| ਵਿਧੀ | ਵੇਰਵਾ | ਜਟਿਲਤਾ |
|--------|-------------|------------|
| `sa` | ਸਿਮੂਲੇਟਡ ਐਨੀਲਿੰਗ (ਡਿਫ਼ਾਲਟ) | O(n · steps) |
| `ising_sa` | ਵੈਕਟਰਾਈਜ਼ਡ ਡੈਲਟਾ ਨਾਲ Ising-ਰੂਪ SA | O(1) ਪ੍ਰਤੀ ਫਲਿੱਪ |
| `relaxation` | ਨਿਰੰਤਰ ਰਿਲੈਕਸੇਸ਼ਨ + ਰਾਊਂਡਿੰਗ (SLSQP) | O(n²) |
| `greedy` | ਲਾਲਚੀ ਚੋਣ | O(n · k) |

---

## ਧਰਮ ਇੰਜਣ — ਪਲੱਗਯੋਗ ਊਰਜਾ ਪਦ

**UniversalDharmaEngine** ਤੁਹਾਨੂੰ ਬੋਧ ਦਰਸ਼ਨ ਤੋਂ ਪ੍ਰੇਰਿਤ ਊਰਜਾ ਪਦਾਂ
ਨੂੰ ਜੋੜਨ ਦਿੰਦਾ ਹੈ, ਜਿਨ੍ਹਾਂ ਵਿੱਚੋਂ ਹਰ ਇੱਕ ਆਪਣੀ ਗਣਿਤਕ ਵਿਸ਼ੇਸ਼ਤਾ ਦੱਸਦਾ ਹੈ।
ਇੰਜਣ ਆਪਣੇ-ਆਪ ਅਨੁਕੂਲ ਸੌਲਵਰ ਵੱਲ ਰੂਟ ਕਰਦਾ ਹੈ।

```python
from lmm.dharma import UniversalDharmaEngine, DukkhaTerm, KarunaTerm

engine = UniversalDharmaEngine(n_candidates=1000)
engine.add(DukkhaTerm(surprises, weight=1.0))    # linear  → Top-K
engine.add(KarunaTerm(impact_graph, weight=2.0)) # supermodular → SA
result = engine.synthesize_and_solve(k=10)
```

### ਦਰਸ਼ਨ → ਗਣਿਤ → ਸੌਲਵਰ ਮੈਪਿੰਗ

| ਬੋਧ ਸੰਕਲਪ | ਊਰਜਾ ਪਦ | ਗਣਿਤਕ ਵਿਸ਼ੇਸ਼ਤਾ | ਸੌਲਵਰ |
|-----------------|-------------|---------------|--------|
| ਪ੍ਰਗਿਆ (Prajna — ਬੁੱਧੀ) | `PrajnaTerm` | ਰੇਖਿਕ (linear) | Top-K ਸੌਰਟ |
| ਕਰੁਣਾ (Karuna — ਦਇਆ) | `KarunaTerm` | ਸੁਪਰਮੌਡਿਊਲਰ | ਵਾਰਮ-ਸਟਾਰਟ + SA |
| ਸ਼ੀਲ (Sila — ਅਚਾਰ) | `SilaTerm` | ਸਬਮੌਡਿਊਲਰ | ਲੇਜ਼ੀ ਗ੍ਰੀਡੀ |
| ਮੱਧਿਅਮਕ (Madhyamaka — ਵਿਚਕਾਰਲਾ ਮਾਰਗ) | `MadhyamakaCriterion` | ਲਿਆਪੁਨੋਵ (Lyapunov) | ਐਕਸਪੋਨੈਂਸ਼ੀਅਲ ਗ੍ਰੇਡੀਐਂਟ |
| ਦੁੱਖ (Dukkha — ਪੀੜਾ) | `DukkhaTerm` | ਰੇਖਿਕ (linear) | Top-K ਸੌਰਟ |

---

## ਤਰਕ ਆਰਕੈਸਟ੍ਰੇਟਰ

**DharmaReasonerOrchestrator** ਸੁਆਲ ਦੀ ਜਟਿਲਤਾ ਦੇ ਆਧਾਰ 'ਤੇ 8 ਤਰਕ
ਮੋਡਾਂ ਵਿੱਚੋਂ ਚੋਣ ਕਰਦਾ ਹੈ:

```python
from lmm.reasoning import DharmaReasonerOrchestrator

orch = DharmaReasonerOrchestrator(n_candidates=500)
result = orch.reason(query_vector, context_vectors)
print(result.mode_used, result.explanation)
```

| ਮੋਡ | ਮੌਡਿਊਲ | ਪ੍ਰੇਰਨਾ |
|------|--------|-------------|
| ਅਨੁਕੂਲੀ (Adaptive) | `adaptive.py` | 応病与薬 — ਬਿਮਾਰੀ ਅਨੁਸਾਰ ਦਵਾਈ |
| ਸਿਧਾਂਤਕ (Theoretical) | `theoretical.py` | 因明 — ਬੋਧ ਰਸਮੀ ਤਰਕ-ਵਿਗਿਆਨ |
| ਅਪਵਾਦਾਤਮਕ (Abductive) | `hyper.py` | 般若の飛躍 — ਪ੍ਰਗਿਆ ਦੀ ਛਾਲ |
| ਸਰਗਰਮ ਅਨੁਮਾਨ (Active Inference) | `active_inference.py` | 托鉢 — ਬਾਹਰੀ ਸੱਚ ਦੀ ਖੋਜ |
| ਆਲਯ ਯਾਦਦਾਸ਼ਤ (Alaya Memory) | `alaya.py` | 阿頼耶識 — ਭੰਡਾਰ ਚੇਤਨਾ |
| ਨੀਂਦ (Sleep) | `sleep.py` | 禅定 — NREM/REM ਇਕਜੁੱਟੀਕਰਨ |
| ਮੂਰਤ (Embodied) | `embodiment.py` | 六根 — ਛੇ ਗਿਆਨੇਂਦਰੀ ਵਿਧੀਆਂ |
| ਪਾਈਨੀਅਲ/ਕੁਆਂਟਮ (Pineal) | `pineal.py` | 松果体 — ਹਾਰਡਵੇਅਰ-ਐਂਟ੍ਰੋਪੀ ਚੇਤਨਾ |

---

## ਟ੍ਰਿਲੀਅਨ-ਟੋਕਨ ਸਕੇਲਿੰਗ

ਸਥਿਰ ਮੈਮੋਰੀ ਨਾਲ ਬੇਅੰਤ ਵੱਡੀਆਂ ਡੇਟਾ ਸਟ੍ਰੀਮਾਂ ਪ੍ਰੋਸੈਸ ਕਰੋ:

```python
from lmm.scale import ScalablePipeline
from pathlib import Path

pipe = ScalablePipeline(k=10, chunk_size=100_000)
pipe.fit_files([Path("shard_001.npy"), ...])
result = pipe.run_files([Path("data_001.npy"), ...])
print(result.summary)
```

**ਤਿੰਨ-ਪੜਾਵੀ ਕੈਸਕੇਡ:** 1 T ਟੋਕਨ → 100 M (ਸਟ੍ਰੀਮਿੰਗ top-k) → 10 K (ਪਰਸੈਂਟਾਈਲ) → K (QUBO)।

---

## LLM ਏਕੀਕਰਨ

### LangChain

```python
from lmm.integrations.langchain import DharmaDocumentCompressor

compressor = DharmaDocumentCompressor(k=5)
# LangChain ਦੇ ContextualCompressionRetriever ਨਾਲ ਵਰਤੋ
```

### LlamaIndex

```python
from lmm.integrations.llamaindex import DharmaNodePostprocessor

postprocessor = DharmaNodePostprocessor(top_k=5)
# LlamaIndex ਸੁਆਲ ਇੰਜਣ ਨਾਲ ਵਰਤੋ
```

### ਸੁਤੰਤਰ LLM ਸਹਾਇਕ

```python
from lmm.llm import FewShotSelector, OutputReranker, DriftDetector

# ਵਿਭਿੰਨਤਾ ਗਾਰੰਟੀ ਨਾਲ Few-shot ਉਦਾਹਰਨ ਚੋਣ
selector = FewShotSelector(k=5)
examples = selector.select(candidates, query)

# ਹੈਰਾਨੀ × ਵਿਭਿੰਨਤਾ ਦੁਆਰਾ LLM ਆਉਟਪੁੱਟ ਰੀਰੈਂਕ ਕਰੋ
reranker = OutputReranker(top_k=3)
best = reranker.rerank(outputs)

# ਸਮੇਂ ਦੇ ਨਾਲ ਆਉਟਪੁੱਟ ਵੰਡ ਡ੍ਰਿਫ਼ਟ ਦੀ ਨਿਗਰਾਨੀ ਕਰੋ
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

## ਨਿਊਰੋਮੌਰਫਿਕ ਸਿਮੂਲੇਸ਼ਨ

ਊਰਜਾ-ਕੁਸ਼ਲ FEP ਗਣਨਾ ਲਈ ਮੈਮਰਿਸਟਰ-ਕ੍ਰੌਸਬਾਰ ਹਾਰਡਵੇਅਰ ਦਾ ਨਕਲ ਕਰੋ:

```python
from lmm.dharma.neuromorphic import NeuromorphicChip

chip = NeuromorphicChip(size=64)
chip.program(weight_matrix)
report = chip.run(input_voltages, steps=100)
# ~10 fJ/MAC, ~30 ns ਸੰਯੋਜਨ (convergence)
```

---

## ਨਿਰਭਰਤਾਵਾਂ

| ਪੈਕੇਜ | ਵਰਜ਼ਨ | ਲੋੜੀਂਦਾ |
|---------|---------|----------|
| numpy | >= 1.24 | ਹਾਂ |
| scipy | >= 1.10 | ਹਾਂ |
| hnswlib | >= 0.8.0 | ਵਿਕਲਪਿਕ (ਸਪਾਰਸ ਗ੍ਰਾਫ਼) |
| langchain-core | >= 0.2.0 | ਵਿਕਲਪਿਕ (LangChain) |
| llama-index-core | >= 0.10.0 | ਵਿਕਲਪਿਕ (LlamaIndex) |

---

## ਸਿਧਾਂਤਕ ਬੁਨਿਆਦ

| ਸੰਕਲਪ | ਫਾਰਮੂਲੇਸ਼ਨ |
|---------|-------------|
| **FEP ≡ KCL** | `dV/dt = −V/τ + g'(V) · G · ε` — ਭਵਿੱਖਬਾਣੀ-ਗਲਤੀ ਘੱਟ ਕਰਨਾ ਕਿਰਚਹੋਫ਼ ਦੇ ਕਰੰਟ ਨਿਯਮ ਵਜੋਂ |
| **ਸੁਪਰਮੌਡਿਊਲਰ ਕਰੁਣਾ** | ਕਰੁਣਾ ਵਿੱਚ ਵਧਦੇ ਮੁਨਾਫ਼ੇ ਹਨ: `f(S∪{i}) − f(S) ≥ f(T∪{i}) − f(T)` ਜਿੱਥੇ S ⊆ T |
| **ਸਬਮੌਡਿਊਲਰ ਸ਼ੀਲ** | ਸ਼ੀਲ ਵਿੱਚ ਘੱਟਦੇ ਮੁਨਾਫ਼ੇ ਹਨ — ਲੇਜ਼ੀ ਗ੍ਰੀਡੀ (1−1/e) ਗਾਰੰਟੀ ਦਿੰਦਾ ਹੈ |
| **ਮੱਧਿਅਮਕ** | ਵਿਚਕਾਰਲਾ ਮਾਰਗ CV = 0.5 ਨੂੰ ਲਿਆਪੁਨੋਵ-ਸਥਿਰ ਐਕਸਪੋਨੈਂਸ਼ੀਅਲ ਗ੍ਰੇਡੀਐਂਟ ਡਿਸੈਂਟ ਰਾਹੀਂ ਨਿਸ਼ਾਨਾ ਬਣਾਉਂਦਾ ਹੈ |
| **ਪਾਈਨੀਅਲ ਕੋਲੈਪਸ (Pineal Collapse)** | ਤਰੰਗ-ਫੰਕਸ਼ਨ ਕੋਲੈਪਸ PRNG ਦੀ ਥਾਂ ਭੌਤਿਕ ਐਂਟ੍ਰੋਪੀ (ਹਾਰਡਵੇਅਰ TRNG) ਵਰਤਦਾ ਹੈ |

---

## ਲਾਇਸੈਂਸ

[MIT](LICENSE)
