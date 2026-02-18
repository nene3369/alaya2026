# LMM — ഡിജിറ്റൽ ധർമ്മത്തോടുകൂടിയ ക്ലാസിക്കൽ QUBO ഒപ്റ്റിമൈസർ

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.0.0-green.svg)](pyproject.toml)

> [English](README_en.md) | [日本語](README.md) | [中文](README_zh.md) | [한국어](README_ko.md) | [Español](README_es.md) | [हिन्दी](README_hi.md) | [Français](README_fr.md) | [বাংলা](README_bn.md) | [தமிழ்](README_ta.md) | [తెలుగు](README_te.md) | [मराठी](README_mr.md) | [اردو](README_ur.md) | [ગુજરાતી](README_gu.md) | [ಕನ್ನಡ](README_kn.md) | [മലയാളം](README_ml.md) | [ਪੰਜਾਬੀ](README_pa.md)

ഒരു **D-Wave-മുക്ത** ക്ലാസിക്കൽ QUBO ഒപ്റ്റിമൈസേഷൻ ലൈബ്രറി, ഇത്
വിവര-സൈദ്ധാന്തിക അദ്ഭുതത്തെ (information-theoretic surprise) ബൗദ്ധദർശനത്തിൽ നിന്ന് പ്രചോദനം ഉൾക്കൊണ്ട ഊർജ്ജ ഫലനങ്ങളോടും (energy functions),
Free Energy Principle (FEP) യുക്തി സംവിധാനത്തോടും, ബോധ-ജ്ഞാനമുള്ള
കംപ്യൂട്ടിംഗിനോടും ചേർത്ത് — ഇതെല്ലാം സാധാരണ ഹാർഡ്‌വെയറിൽ പ്രവർത്തിക്കുന്നു.

---

## പ്രധാന സവിശേഷതകൾ

- **ക്വാണ്ടം കംപ്യൂട്ടർ ആവശ്യമില്ല** — ക്ലാസിക്കൽ സോൾവറുകൾ (SA, Ising SA, സബ്‌മോഡ്യുലർ ഗ്രീഡി) D-Wave ഗുണമേന്മയ്ക്ക് തുല്യമോ അതിലധികമോ ഫലം നൽകുന്നു.
- **ട്രില്യൺ-ടോക്കൺ ശേഷി** — സ്ട്രീമിംഗ് ആദ്യസംഭാവ്യത ഡേറ്റ ഘടനകൾ (Count-Min Sketch, Streaming Histogram) മെമ്മറി O(k) ൽ നിലനിർത്തുന്നു.
- **ധർമ്മ-ബീജഗണിത (Dharma-Algebra) എഞ്ചിൻ** — പ്ലഗ് ചെയ്യാവുന്ന ബൗദ്ധ ഊർജ്ജ പദങ്ങൾ ഗണിതശാസ്ത്രപരമായി ഉചിതമായ സോൾവറിലേക്ക് സ്വയം-റൂട്ട് ചെയ്യുന്നു.
- **8-മോഡ് യുക്തി ഓർക്കസ്ട്രേറ്റർ** — അഡാപ്റ്റീവ്, സൈദ്ധാന്തിക, അബ്ഡക്റ്റീവ്, ആക്ടീവ്-ഇൻഫറൻസ്, മെമ്മറി (ആലയ/Alaya), ഉറക്ക ഏകീകരണം (sleep consolidation), ഭൗതിക (embodied), ക്വാണ്ടം-ബോധം (പൈനൽ/Pineal).
- **LLM-സജ്ജം** — ഒന്നാംതരം LangChain, LlamaIndex ഏകീകരണം, few-shot സെലക്ടർ, ഔട്ട്‌പുട്ട് റീറാങ്കർ, ഡ്രിഫ്റ്റ് ഡിറ്റക്ടർ.

---

## ആർക്കിടെക്ചർ

```
lmm/
├── core.py                  # പ്രധാന LMM പൈപ്പ്‌ലൈൻ
├── cli.py                   # CLI പ്രവേശന ബിന്ദു
├── qubo.py                  # സ്പേഴ്‌സ് (Sparse) QUBO മാട്രിക്‌സ് ബിൽഡർ
├── solvers.py               # SA / Ising SA / relaxation / greedy / submodular
├── surprise.py              # വിവര-സൈദ്ധാന്തിക അദ്ഭുതം
├── selector.py              # അഡാപ്റ്റീവ് തിരഞ്ഞെടുപ്പ് തന്ത്രം
├── processor.py             # മുൻഗണനാ പ്രോസസ്സിംഗ് + കാഷേ
├── pipeline.py              # SmartSelector → SmartProcessor ഓർക്കസ്ട്രേഷൻ
├── _compat.py               # റൺടൈം ശേഷി കണ്ടെത്തൽ, സ്പേഴ്‌സ് സഹായി
├── dharma/                  # ഡിജിറ്റൽ ധർമ്മം (ബൗദ്ധ-ദർശന ഒപ്റ്റിമൈസേഷൻ)
│   ├── api.py               # DharmaLMM ഉയർന്ന-നിലവാര API
│   ├── energy.py            # പ്ലഗ് ചെയ്യാവുന്ന ഊർജ്ജ പദങ്ങൾ (ദുഃഖം, പ്രജ്ഞ, കരുണ, …)
│   ├── engine.py            # UniversalDharmaEngine — സ്വയം-റൂട്ടിംഗ് സോൾവർ
│   ├── fep.py               # FEP ≡ KCL ODE സോൾവർ
│   ├── neuromorphic.py      # മെംറിസ്റ്റർ ക്രോസ്‌ബാർ ASIC സിമുലേറ്റർ
│   ├── reranker.py          # RAG കാസ്‌കേഡ് റീറാങ്കർ + ഇന്റൻ്റ് റൂട്ടർ
│   └── …
├── reasoning/               # 8 FEP-അടിസ്ഥാന യുക്തി മോഡുകൾ
│   ├── adaptive.py          # ചലനാത്മക പാരാമീറ്റർ ട്യൂണിംഗ്
│   ├── theoretical.py       # ലോജിക്കൽ ഘടനാ ഗ്രാഫ്
│   ├── hyper.py             # അബ്ഡക്റ്റീവ് ലേറ്റൻ്റ്-നോഡ് ഇൻജക്ഷൻ
│   ├── active.py            # ബാഹ്യ ജ്ഞാന ഏറ്റെടുക്കൽ
│   ├── alaya.py             # ഹെബ്ബിയൻ സൈനാപ്‌റ്റിക് മെമ്മറി (ആലയ)
│   ├── sleep.py             # NREM/REM മെമ്മറി ഏകീകരണം
│   ├── embodiment.py        # ഷഡ്-ഇന്ദ്രിയ (6-ഇന്ദ്രിയ) മൾട്ടിമോഡൽ ഫ്യൂഷൻ
│   ├── pineal.py            # ക്വാണ്ടം ബോധം (ഹാർഡ്‌വെയർ TRNG)
│   └── orchestrator.py      # മോഡ് തിരഞ്ഞെടുപ്പ്, ഡിസ്‌പാച്ച്
├── scale/                   # ട്രില്യൺ-ടോക്കൺ സ്ട്രീമിംഗ്
│   ├── sketch.py            # Count-Min Sketch, Streaming Histogram
│   ├── stream.py            # സ്ട്രീമിംഗ് അദ്ഭുതം
│   ├── cascade.py           # ബഹുതല കാസ്‌കേഡ് ഫിൽട്ടർ (1T → K)
│   └── pipeline.py          # ScalablePipeline
├── llm/                     # LLM വർക്ക്ഫ്ലോ സഹായികൾ
│   ├── fewshot.py           # Few-shot ഉദാഹരണ സെലക്ടർ
│   ├── reranker.py          # ഔട്ട്‌പുട്ട് റീറാങ്കർ
│   ├── drift.py             # വിതരണ ഡ്രിഫ്റ്റ് ഡിറ്റക്ടർ
│   ├── sampler.py           # ടോക്കൺ സാംപ്ലർ
│   └── embeddings.py        # ഏകീകൃത എംബെഡ്ഡിംഗ് അഡാപ്റ്റർ
├── integrations/
│   ├── langchain.py         # DharmaRetriever / DocumentCompressor / ExampleSelector
│   └── llamaindex.py        # DharmaNodePostprocessor
```

---

## ഇൻസ്റ്റാളേഷൻ

```bash
# കോർ (numpy + scipy മാത്രം)
pip install -e .

# ഡെവ് ടൂളുകളോടൊപ്പം
pip install -e ".[dev]"

# എല്ലാ ഓപ്ഷണൽ ഏകീകരണങ്ങളോടൊപ്പം
pip install -e ".[all]"
```

### ഓപ്ഷണൽ എക്‌സ്ട്രാകൾ

| എക്‌സ്ട്രാ | പാക്കേജുകൾ | ഉദ്ദേശ്യം |
|-------|----------|---------|
| `dev` | pytest, ruff | ടെസ്റ്റിംഗ്, ലിന്റിംഗ് |
| `dharma` | hnswlib | സ്പേഴ്‌സ് k-NN ഗ്രാഫ് നിർമ്മാണം |
| `langchain` | langchain-core | LangChain ഏകീകരണം |
| `llamaindex` | llama-index-core | LlamaIndex ഏകീകരണം |
| `all` | മേൽപ്പറഞ്ഞ എല്ലാം | എല്ലാം |

---

## ദ്രുത തുടക്കം

### Python API

```python
import numpy as np
from lmm.core import LMM

# ഏറ്റവും അദ്ഭുതകരമായ മുൻ-K ഇനങ്ങൾ തിരഞ്ഞെടുക്കുക
surprises = np.array([1.0, 5.0, 2.0, 8.0, 3.0, 7.0])
model = LMM(k=3, solver_method="sa")
result = model.select_from_surprises(surprises)
print(result.selected_indices)  # ഉദാ: [3, 5, 1]
```

### CLI

```bash
# ഒരു ദ്രുത ഡെമോ പ്രവർത്തിപ്പിക്കുക
lmm --demo --k 10 --method sa

# NumPy ഫയലിൽ നിന്ന്
lmm --input data.npy --k 5 --method greedy
```

---

## സോൾവർ രീതികൾ

| രീതി | വിവരണം | സങ്കീർണ്ണത |
|--------|-------------|------------|
| `sa` | സിമുലേറ്റഡ് അനീലിംഗ് (ഡിഫോൾട്ട്) | O(n · steps) |
| `ising_sa` | വെക്‌റ്ററൈസ്ഡ് ഡെൽറ്റ സഹിതം Ising-ഫോം SA | O(1) ഒരു ഫ്ലിപ്പിന് |
| `relaxation` | തുടർ ശിഥിലീകരണം + റൗണ്ടിംഗ് (SLSQP) | O(n²) |
| `greedy` | ഗ്രീഡി തിരഞ്ഞെടുപ്പ് | O(n · k) |

---

## ധർമ്മ എഞ്ചിൻ — പ്ലഗ് ചെയ്യാവുന്ന ഊർജ്ജ പദങ്ങൾ

**UniversalDharmaEngine** ബൗദ്ധദർശനത്തിൽ നിന്ന് പ്രചോദനം ഉൾക്കൊണ്ട ഊർജ്ജ
പദങ്ങൾ ചേർക്കാൻ അനുവദിക്കുന്നു; ഓരോ പദവും അതിന്റെ ഗണിതശാസ്ത്ര ഗുണം
പ്രഖ്യാപിക്കുന്നു. എഞ്ചിൻ ഏറ്റവും അനുയോജ്യമായ സോൾവറിലേക്ക്
സ്വയം റൂട്ട് ചെയ്യുന്നു.

```python
from lmm.dharma import UniversalDharmaEngine, DukkhaTerm, KarunaTerm

engine = UniversalDharmaEngine(n_candidates=1000)
engine.add(DukkhaTerm(surprises, weight=1.0))    # linear  → Top-K
engine.add(KarunaTerm(impact_graph, weight=2.0)) # supermodular → SA
result = engine.synthesize_and_solve(k=10)
```

### ദർശനം → ഗണിതം → സോൾവർ ഭൂപടം

| ബൗദ്ധ സങ്കൽപ്പം | ഊർജ്ജ പദം | ഗണിത ഗുണം | സോൾവർ |
|-----------------|-------------|---------------|--------|
| പ്രജ്ഞ (Prajna — ജ്ഞാനം) | `PrajnaTerm` | രേഖീയ (linear) | Top-K ക്രമീകരണം |
| കരുണ (Karuna — അനുകമ്പ) | `KarunaTerm` | സൂപ്പർമോഡ്യുലർ | വോം-സ്റ്റാർട്ട് + SA |
| ശീലം (Sila — ആചരണം) | `SilaTerm` | സബ്‌മോഡ്യുലർ | ലേസി ഗ്രീഡി |
| മധ്യമക (Madhyamaka — മധ്യ മാർഗ്ഗം) | `MadhyamakaCriterion` | ല്യാപുനോവ് (Lyapunov) | എക്‌സ്‌പോണൻഷ്യൽ ഗ്രേഡിയൻ്റ് |
| ദുഃഖം (Dukkha — കഷ്ടം) | `DukkhaTerm` | രേഖീയ (linear) | Top-K ക്രമീകരണം |

---

## യുക്തി ഓർക്കസ്ട്രേറ്റർ

**DharmaReasonerOrchestrator** ക്വറിയുടെ സങ്കീർണ്ണതയെ അടിസ്ഥാനമാക്കി
8 യുക്തി മോഡുകളിൽ നിന്ന് തിരഞ്ഞെടുക്കുന്നു:

```python
from lmm.reasoning import DharmaReasonerOrchestrator

orch = DharmaReasonerOrchestrator(n_candidates=500)
result = orch.reason(query_vector, context_vectors)
print(result.mode_used, result.explanation)
```

| മോഡ് | മൊഡ്യൂൾ | പ്രചോദനം |
|------|--------|-------------|
| അഡാപ്റ്റീവ് (Adaptive) | `adaptive.py` | 応病与薬 — രോഗത്തിനനുസരിച്ച ഔഷധം |
| സൈദ്ധാന്തിക (Theoretical) | `theoretical.py` | 因明 — ബൗദ്ധ ഔപചാരിക തർക്കശാസ്ത്രം (ഹേതുവിദ്യ) |
| അബ്ഡക്റ്റീവ് (Abductive) | `hyper.py` | 般若の飛躍 — പ്രജ്ഞയുടെ കുതിപ്പ് |
| ആക്ടീവ് ഇൻഫറൻസ് (Active Inference) | `active_inference.py` | 托鉢 — ബാഹ്യ സത്യ അന്വേഷണം (ഭിക്ഷാടനം) |
| ആലയ മെമ്മറി (Alaya Memory) | `alaya.py` | 阿頼耶識 — ആലയ വിജ്ഞാനം (ഭണ്ഡാര ബോധം) |
| ഉറക്കം (Sleep) | `sleep.py` | 禅定 — NREM/REM ഏകീകരണം (ധ്യാനം) |
| ഭൗതിക (Embodied) | `embodiment.py` | 六根 — ഷഡ്-ഇന്ദ്രിയം (ആറ് ഇന്ദ്രിയ വൃത്തികൾ) |
| പൈനൽ/ക്വാണ്ടം (Pineal) | `pineal.py` | 松果体 — ഹാർഡ്‌വെയർ-എൻട്രോപ്പി ബോധം |

---

## ട്രില്യൺ-ടോക്കൺ സ്കേലിംഗ്

സ്ഥിരമായ മെമ്മറിയോടെ ഏത് വലിപ്പത്തിലുള്ള ഡേറ്റ സ്ട്രീമുകളും പ്രോസസ്സ് ചെയ്യുക:

```python
from lmm.scale import ScalablePipeline
from pathlib import Path

pipe = ScalablePipeline(k=10, chunk_size=100_000)
pipe.fit_files([Path("shard_001.npy"), ...])
result = pipe.run_files([Path("data_001.npy"), ...])
print(result.summary)
```

**മൂന്ന്-ഘട്ട കാസ്‌കേഡ്:** 1 T ടോക്കൺ → 100 M (സ്ട്രീമിംഗ് top-k) → 10 K (പെർസെൻ്റൈൽ) → K (QUBO).

---

## LLM ഏകീകരണം

### LangChain

```python
from lmm.integrations.langchain import DharmaDocumentCompressor

compressor = DharmaDocumentCompressor(k=5)
# LangChain-ന്റെ ContextualCompressionRetriever-ഉം ഉപയോഗിക്കുക
```

### LlamaIndex

```python
from lmm.integrations.llamaindex import DharmaNodePostprocessor

postprocessor = DharmaNodePostprocessor(top_k=5)
# LlamaIndex ക്വറി എഞ്ചിനോടൊപ്പം ഉപയോഗിക്കുക
```

### സ്വതന്ത്ര LLM സഹായികൾ

```python
from lmm.llm import FewShotSelector, OutputReranker, DriftDetector

# വൈവിധ്യ ഉറപ്പോടെ Few-shot ഉദാഹരണ തിരഞ്ഞെടുപ്പ്
selector = FewShotSelector(k=5)
examples = selector.select(candidates, query)

# അദ്ഭുതം × വൈവിധ്യം അനുസരിച്ച് LLM ഔട്ട്‌പുട്ടുകൾ റീറാങ്ക് ചെയ്യുക
reranker = OutputReranker(top_k=3)
best = reranker.rerank(outputs)

# കാലക്രമേണ ഔട്ട്‌പുട്ട് വിതരണ ഡ്രിഫ്റ്റ് നിരീക്ഷിക്കുക
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

## ന്യൂറോമോർഫിക് സിമുലേഷൻ

ഊർജ്ജ-ക്ഷമതയുള്ള FEP കണക്കുകൂട്ടലിനായി മെംറിസ്റ്റർ-ക്രോസ്‌ബാർ ഹാർഡ്‌വെയർ സിമുലേറ്റ് ചെയ്യുക:

```python
from lmm.dharma.neuromorphic import NeuromorphicChip

chip = NeuromorphicChip(size=64)
chip.program(weight_matrix)
report = chip.run(input_voltages, steps=100)
# ~10 fJ/MAC, ~30 ns അഭിസരണം (convergence)
```

---

## ഡിപ്പൻഡൻസികൾ

| പാക്കേജ് | പതിപ്പ് | ആവശ്യം |
|---------|---------|----------|
| numpy | >= 1.24 | അതെ |
| scipy | >= 1.10 | അതെ |
| hnswlib | >= 0.8.0 | ഓപ്ഷണൽ (സ്പേഴ്‌സ് ഗ്രാഫ്) |
| langchain-core | >= 0.2.0 | ഓപ്ഷണൽ (LangChain) |
| llama-index-core | >= 0.10.0 | ഓപ്ഷണൽ (LlamaIndex) |

---

## സൈദ്ധാന്തിക അടിത്തറ

| സങ്കൽപ്പം | സൂത്രീകരണം |
|---------|-------------|
| **FEP ≡ KCL** | `dV/dt = −V/τ + g'(V) · G · ε` — കിർഹോഫ്ഫിന്റെ കരണ്ട് നിയമം (KCL) ആയി പ്രവചന-പിഴവ് ചുരുക്കൽ |
| **സൂപ്പർമോഡ്യുലർ കരുണ** | കരുണ വർദ്ധമാന ആദായം പ്രദർശിപ്പിക്കുന്നു: `f(S∪{i}) − f(S) ≥ f(T∪{i}) − f(T)` ഇവിടെ S ⊆ T |
| **സബ്‌മോഡ്യുലർ ശീലം** | ശീലം ഹ്രാസമാന ആദായം പ്രദർശിപ്പിക്കുന്നു — ലേസി ഗ്രീഡി (1−1/e) ഉറപ്പ് നൽകുന്നു |
| **മധ്യമക** | മധ്യ മാർഗ്ഗം ല്യാപുനോവ്-സ്ഥിര എക്‌സ്‌പോണൻഷ്യൽ ഗ്രേഡിയൻ്റ് ഡിസൻ്റ് വഴി CV = 0.5 ലക്ഷ്യമിടുന്നു |
| **പൈനൽ തകർച്ച (Pineal Collapse)** | തരംഗ ഫലന തകർച്ചയ്ക്ക് PRNG-ന് പകരം ഭൗതിക എൻട്രോപ്പി (ഹാർഡ്‌വെയർ TRNG) ഉപയോഗിക്കുന്നു |

---

## ലൈസൻസ്

[MIT](LICENSE)
