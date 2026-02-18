# LMM — ડિજિટલ ધર્મ સાથે ક્લાસિકલ QUBO ઓપ્ટિમાઇઝર

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.0.0-green.svg)](pyproject.toml)

> [English](README_en.md) | [日本語](README.md) | [中文](README_zh.md) | [한국어](README_ko.md) | [Español](README_es.md) | [हिन्दी](README_hi.md) | [Français](README_fr.md) | [বাংলা](README_bn.md) | [தமிழ்](README_ta.md) | [తెలుగు](README_te.md) | [मराठी](README_mr.md) | [‎اردو‎](README_ur.md) | [ગુજરાતી](README_gu.md) | [ಕನ್ನಡ](README_kn.md) | [മലയാളം](README_ml.md) | [ਪੰਜਾਬੀ](README_pa.md)

**D-Wave-મુક્ત** ક્લાસિકલ QUBO ઓપ્ટિમાઇઝેશન લાઇબ્રેરી, જે
ઇન્ફર્મેશન-થિઓરેટિક સરપ્રાઇઝ, બૌદ્ધ ફિલોસોફીથી પ્રેરિત એનર્જી
ફંક્શન, ફ્રી એનર્જી પ્રિન્સિપલ (FEP) તર્ક, અને ચેતના-સભ્ય
કમ્પ્યુટિંગ — આ બધું સામાન્ય હાર્ડવેર પર — ભેગું કરે છે.

---

## મુખ્ય વિશેષતાઓ

- **ક્વોન્ટમ કમ્પ્યુટર જરૂરી નથી** — ક્લાસિકલ સોલ્વર (SA, Ising SA, submodular greedy) D-Wave જેટલી કે તેનાથી વધુ સારી ગુણવત્તા આપે છે.
- **ટ્રિલિયન-ટોકન સક્ષમ** — સ્ટ્રીમિંગ પ્રોબેબિલિસ્ટિક ડેટા સ્ટ્રક્ચર (Count-Min Sketch, Streaming Histogram) મેમરી O(k) રાખે છે.
- **ધર્મ-એલ્જેબ્રા એન્જિન** — પ્લગ-ઇન કરી શકાય તેવા બૌદ્ધ એનર્જી ટર્મ ગણિત-શ્રેષ્ઠ સોલ્વર પ્રત્યે આપોઆપ રૂટ થાય છે.
- **8-મોડ તર્ક ઓર્કેસ્ટ્રેટર** — અનુકૂલ, સૈદ્ધાંતિક, abductive, active-inference, સ્મૃતિ (Alaya), ઊંઘ-એકત્રીકરણ, embodied, અને ક્વોન્ટમ-ચેતના (Pineal).
- **LLM-તૈયાર** — LangChain અને LlamaIndex સાથે ફર્સ્ટ-ક્લાસ ઇન્ટિગ્રેશન, few-shot સિલેક્ટર, આઉટપુટ રીરેન્કર, ડ્રિફ્ટ ડિટેક્ટર.

---

## આર્કિટેક્ચર

```
lmm/
├── core.py                  # મુખ્ય LMM પાઇપલાઇન
├── cli.py                   # CLI એન્ટ્રી પૉઇન્ટ
├── qubo.py                  # Sparse QUBO મેટ્રિક્સ બિલ્ડર
├── solvers.py               # SA / Ising SA / relaxation / greedy / submodular
├── surprise.py              # ઇન્ફર્મેશન-થિઓરેટિક સરપ્રાઇઝ
├── selector.py              # અનુકૂલ સિલેક્શન સ્ટ્રેટેજી
├── processor.py             # પ્રાધાન્ય પ્રોસેસિંગ + કૅશ
├── pipeline.py              # SmartSelector → SmartProcessor ઓર્કેસ્ટ્રેશન
├── _compat.py               # રનટાઇમ ક્ષમતા ડિટેક્શન અને sparse હેલ્પર
├── dharma/                  # ડિજિટલ ધર્મ (બૌદ્ધ-ફિલોસોફી ઓપ્ટિમાઇઝેશન)
│   ├── api.py               # DharmaLMM ઉચ્ચ-સ્તરીય API
│   ├── energy.py            # પ્લગ-ઇન એનર્જી ટર્મ (Dukkha, Prajna, Karuna, …)
│   ├── engine.py            # UniversalDharmaEngine — ઓટો-રૂટિંગ સોલ્વર
│   ├── fep.py               # FEP ≡ KCL ODE સોલ્વર
│   ├── neuromorphic.py      # Memristor crossbar ASIC સિમ્યુલેટર
│   ├── reranker.py          # RAG cascade રીરેન્કર + ઇન્ટેન્ટ રૂટર
│   └── …
├── reasoning/               # 8 FEP-આધારિત તર્ક મોડ
│   ├── adaptive.py          # ડાયનેમિક પેરામીટર ટ્યૂનિંગ
│   ├── theoretical.py       # તાર્કિક માળખું ગ્રાફ
│   ├── hyper.py             # Abductive latent-node ઇન્જેક્શન
│   ├── active.py            # બાહ્ય જ્ઞાન સંપાદન
│   ├── alaya.py             # Hebbian synaptic સ્મૃતિ (Alaya)
│   ├── sleep.py             # NREM/REM સ્મૃતિ એકત્રીકરણ
│   ├── embodiment.py        # 6-ઇન્દ્રિય મલ્ટિમોડલ ફ્યૂઝન
│   ├── pineal.py            # ક્વોન્ટમ ચેતના (hardware TRNG)
│   └── orchestrator.py      # મોડ સિલેક્શન અને ડિસ્પૅચ
├── scale/                   # ટ્રિલિયન-ટોકન સ્ટ્રીમિંગ
│   ├── sketch.py            # Count-Min Sketch, Streaming Histogram
│   ├── stream.py            # સ્ટ્રીમિંગ સરપ્રાઇઝ
│   ├── cascade.py           # મલ્ટિ-લેવલ cascade ફિલ્ટર (1T → K)
│   └── pipeline.py          # ScalablePipeline
├── llm/                     # LLM વર્કફ્લો હેલ્પર
│   ├── fewshot.py           # Few-shot ઉદાહરણ સિલેક્ટર
│   ├── reranker.py          # આઉટપુટ રીરેન્કર
│   ├── drift.py             # ડિસ્ટ્રિબ્યૂશન ડ્રિફ્ટ ડિટેક્ટર
│   ├── sampler.py           # ટોકન સૅમ્પ્લર
│   └── embeddings.py        # એકીકૃત એમ્બેડિંગ અડૅપ્ટર
├── integrations/
│   ├── langchain.py         # DharmaRetriever / DocumentCompressor / ExampleSelector
│   └── llamaindex.py        # DharmaNodePostprocessor
```

---

## ઇન્સ્ટોલેશન

```bash
# કોર (numpy + scipy માત્ર)
pip install -e .

# dev ટૂલ સાથે
pip install -e ".[dev]"

# બધા વૈકલ્પિક ઇન્ટિગ્રેશન સાથે
pip install -e ".[all]"
```

### વૈકલ્પિક extras

| Extra | પૅકેજ | હેતુ |
|-------|--------|------|
| `dev` | pytest, ruff | ટેસ્ટિંગ અને linting |
| `dharma` | hnswlib | Sparse k-NN ગ્રાફ નિર્માણ |
| `langchain` | langchain-core | LangChain ઇન્ટિગ્રેશન |
| `llamaindex` | llama-index-core | LlamaIndex ઇન્ટિગ્રેશન |
| `all` | ઉપર્યુક્ત તમામ | બધું |

---

## ઝડપી શરૂઆત

### Python API

```python
import numpy as np
from lmm.core import LMM

# Top-K સૌથી વધુ સરપ્રાઇઝ કરતી આઇટેમ પસંદ કરો
surprises = np.array([1.0, 5.0, 2.0, 8.0, 3.0, 7.0])
model = LMM(k=3, solver_method="sa")
result = model.select_from_surprises(surprises)
print(result.selected_indices)  # દા.ત. [3, 5, 1]
```

### CLI

```bash
# ઝડપી ડેમો ચલાવો
lmm --demo --k 10 --method sa

# NumPy ફાઇલ માંથી
lmm --input data.npy --k 5 --method greedy
```

---

## સોલ્વર પદ્ધતિઓ

| પદ્ધતિ | વર્ણન | જટિલતા |
|--------|--------|---------|
| `sa` | Simulated Annealing (ડિફૉલ્ટ) | O(n · steps) |
| `ising_sa` | vectorized delta સાથે Ising-ફૉર્મ SA | O(1) પ્રતિ flip |
| `relaxation` | સતત relaxation + rounding (SLSQP) | O(n²) |
| `greedy` | Greedy સિલેક્શન | O(n · k) |

---

## ધર્મ એન્જિન — પ્લગ-ઇન એનર્જી ટર્મ

**UniversalDharmaEngine** તમને બૌદ્ધ ફિલોસોફીથી પ્રેરિત
એનર્જી ટર્મ જોડવા દે છે, જ્યાં દરેક ટર્મ પોતાનો ગાણિતિક ગુણ
જાહેર કરે છે. એન્જિન આપોઆપ શ્રેષ્ઠ સોલ્વર પ્રત્યે રૂટ કરે છે.

```python
from lmm.dharma import UniversalDharmaEngine, DukkhaTerm, KarunaTerm

engine = UniversalDharmaEngine(n_candidates=1000)
engine.add(DukkhaTerm(surprises, weight=1.0))    # linear  → Top-K
engine.add(KarunaTerm(impact_graph, weight=2.0)) # supermodular → SA
result = engine.synthesize_and_solve(k=10)
```

### ફિલોસોફી → ગણિત → સોલ્વર મૅપિંગ

| બૌદ્ધ સંકલ્પ | એનર્જી ટર્મ | ગાણિતિક ગુણ | સોલ્વર |
|--------------|-------------|-------------|--------|
| પ્રજ્ઞા (Wisdom) | `PrajnaTerm` | linear | Top-K sort |
| કરુણા (Compassion) | `KarunaTerm` | supermodular | Warm-start + SA |
| શીલ (Conduct) | `SilaTerm` | submodular | Lazy greedy |
| મધ્યમક (Middle Way) | `MadhyamakaCriterion` | Lyapunov | Exponential gradient |
| દુઃખ (Suffering) | `DukkhaTerm` | linear | Top-K sort |

---

## તર્ક ઓર્કેસ્ટ્રેટર

**DharmaReasonerOrchestrator** ક્વૅરી જટિલતાના આધારે 8 તર્ક મોડ
માંથી એક પસંદ કરે છે:

```python
from lmm.reasoning import DharmaReasonerOrchestrator

orch = DharmaReasonerOrchestrator(n_candidates=500)
result = orch.reason(query_vector, context_vectors)
print(result.mode_used, result.explanation)
```

| મોડ | મૉડ્યૂલ | પ્રેરણા |
|-----|--------|---------|
| Adaptive | `adaptive.py` | 応病与薬 — બીમારી અનુસાર દવા |
| Theoretical | `theoretical.py` | 因明 — બૌદ્ધ ઔપચારિક તર્ક |
| Abductive | `hyper.py` | 般若の飛躍 — પ્રજ્ઞાની છલાંગ |
| Active Inference | `active_inference.py` | 托鉢 — બાહ્ય સત્ય ખોજ |
| Alaya Memory | `alaya.py` | 阿頼耶識 — ભંડારગૃહ ચેતના |
| Sleep | `sleep.py` | 禅定 — NREM/REM એકત્રીકરણ |
| Embodied | `embodiment.py` | 六根 — છ ઇન્દ્રિય |
| Pineal (Quantum) | `pineal.py` | 松果体 — hardware-entropy ચેતના |

---

## ટ્રિલિયન-ટોકન સ્કેલિંગ

સ્થિર મેમરી સાથે અત્યંત મોટા ડેટા સ્ટ્રીમ પ્રક્રિયા કરો:

```python
from lmm.scale import ScalablePipeline
from pathlib import Path

pipe = ScalablePipeline(k=10, chunk_size=100_000)
pipe.fit_files([Path("shard_001.npy"), ...])
result = pipe.run_files([Path("data_001.npy"), ...])
print(result.summary)
```

**ત્રણ-તબક્કાની cascade:** 1 T ટોકન → 100 M (streaming top-k) → 10 K (percentile) → K (QUBO).

---

## LLM એકીકરણ

### LangChain

```python
from lmm.integrations.langchain import DharmaDocumentCompressor

compressor = DharmaDocumentCompressor(k=5)
# LangChain ના ContextualCompressionRetriever સાથે ઉપયોગ કરો
```

### LlamaIndex

```python
from lmm.integrations.llamaindex import DharmaNodePostprocessor

postprocessor = DharmaNodePostprocessor(top_k=5)
# LlamaIndex query engine સાથે ઉપયોગ કરો
```

### સ્વતંત્ર LLM હેલ્પર

```python
from lmm.llm import FewShotSelector, OutputReranker, DriftDetector

# વૈવિધ્ય ગૅરૅન્ટી સાથે few-shot ઉદાહરણ સિલેક્શન
selector = FewShotSelector(k=5)
examples = selector.select(candidates, query)

# સરપ્રાઇઝ × વૈવિધ્ય દ્વારા LLM આઉટપુટ રીરૅન્ક કરો
reranker = OutputReranker(top_k=3)
best = reranker.rerank(outputs)

# સમય સાથે આઉટપુટ ડિસ્ટ્રિબ્યૂશન ડ્રિફ્ટ મૉનિટર કરો
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

## ન્યુરોમોર્ફિક સિમ્યુલેશન

ઊર્જા-કાર્યક્ષમ FEP ગણના માટે memristor-crossbar હાર્ડવેર સિમ્યુલેટ કરો:

```python
from lmm.dharma.neuromorphic import NeuromorphicChip

chip = NeuromorphicChip(size=64)
chip.program(weight_matrix)
report = chip.run(input_voltages, steps=100)
# ~10 fJ/MAC, ~30 ns convergence
```

---

## ડિપેન્ડન્સીઝ

| પૅકેજ | વર્ઝન | જરૂરી |
|--------|--------|--------|
| numpy | >= 1.24 | હા |
| scipy | >= 1.10 | હા |
| hnswlib | >= 0.8.0 | વૈકલ્પિક (sparse graph) |
| langchain-core | >= 0.2.0 | વૈકલ્પિક (LangChain) |
| llama-index-core | >= 0.10.0 | વૈકલ્પિક (LlamaIndex) |

---

## સૈદ્ધાંતિક પાયા

| સંકલ્પ | સૂત્ર |
|---------|--------|
| **FEP ≡ KCL** | `dV/dt = −V/τ + g'(V) · G · ε` — Kirchhoff ના વર્તમાન નિયમ તરીકે prediction-error ન્યૂનીકરણ |
| **Supermodular Karuna** | કરુણા વધતા વળતર દર્શાવે છે: `f(S∪{i}) − f(S) ≥ f(T∪{i}) − f(T)` S ⊆ T માટે |
| **Submodular Sila** | શીલ ઘટતા વળતર દર્શાવે છે — lazy greedy (1−1/e) ગૅરૅન્ટી આપે છે |
| **Madhyamaka** | મધ્ય માર્ગ Lyapunov-સ્થિર exponential gradient descent દ્વારા CV = 0.5 નિશ્ચિત કરે છે |
| **Pineal Collapse** | Wavefunction collapse PRNG ને બદલે ભૌતિક entropy (hardware TRNG) ઉપયોગ કરે છે |

---

## લાઇસન્સ

[MIT](LICENSE)
