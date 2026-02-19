# LMM — డిజిటల్ ధర్మంతో కూడిన క్లాసికల్ QUBO ఆప్టిమైజర్

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.0.0-green.svg)](pyproject.toml)

> [English](README_en.md) | [日本語](README.md) | [中文](README_zh.md) | [한국어](README_ko.md) | [Español](README_es.md) | [हिन्दी](README_hi.md) | [Français](README_fr.md) | [বাংলা](README_bn.md) | [தமிழ்](README_ta.md) | [తెలుగు](README_te.md) | [मराठी](README_mr.md) | [‎اردو‎](README_ur.md) | [ગુજરાતી](README_gu.md) | [ಕನ್ನಡ](README_kn.md) | [മലయാളം](README_ml.md) | [ਪੰਜਾਬੀ](README_pa.md)

> **Digital Dharma OS (Alaya V5)** — QUBO గణితం, బౌద్ధ తత్వశాస్త్రం మరియు స్వేచ్ఛా శక్తి
> సూత్రం (FEP) నరాల శాస్త్రాన్ని ఒక సజీవ, స్వయం-పరిణామ వ్యవస్థలో కలిపే చైతన్య-అవగాహన
> ఆప్టిమైజేషన్ వేదిక. సర్వర్ నిరంతర హృదయస్పందన డైనమిక్స్, జ్ఞాపకం-ఆధారిత సందర్భ ఎంపిక
> మరియు భావోద్వేగ తరంగదైర్ఘ్య సెన్సింగ్‌ను అమలు చేస్తుంది — విడివిడి LLM కాల్‌లు మరియు
> నిరంతర జ్ఞానం మధ్య వారధిగా పనిచేస్తుంది.

**D-Wave అవసరం లేని** క్లాసికల్ QUBO ఆప్టిమైజేషన్ లైబ్రరీ, ఇది సమాచార-సైద్ధాంతిక ఆశ్చర్యాన్ని బౌద్ధ-తత్వం ఆధారిత శక్తి ఫంక్షన్లతో, ఫ్రీ ఎనర్జీ ప్రిన్సిపల్ (FEP) తార్కికంతో మరియు చేతన-అవగాహన కంప్యూటింగ్‌తో కలుపుతుంది — అన్నీ సాధారణ హార్డ్‌వేర్‌పై నడుస్తాయి.

---

## ముఖ్యాంశాలు

- **క్వాంటం కంప్యూటర్ అవసరం లేదు** — క్లాసికల్ సాల్వర్లు (SA, Ising SA, సబ్‌మాడ్యులర్ గ్రీడీ) D-Wave నాణ్యతను సరిపోల్చాయి లేదా మించాయి.
- **ట్రిలియన్-టోకెన్ సామర్థ్యం** — స్ట్రీమింగ్ ప్రాబబిలిస్టిక్ డేటా స్ట్రక్చర్లు (Count-Min Sketch, Streaming Histogram) మెమరీని O(k)గా ఉంచుతాయి.
- **ధర్మ-బీజగణిత ఇంజన్** — ప్లగ్గబుల్ బౌద్ధ శక్తి పదాలు గణితాత్మకంగా అత్యుత్తమ సాల్వర్‌కు స్వయంచాలకంగా మళ్ళించబడతాయి.
- **8-మోడ్ తార్కిక ఆర్కెస్ట్రేటర్** — అనుకూల, సైద్ధాంతిక, అపగమన, చురుకు-అనుమితి, జ్ఞాపకం (ఆలయ), నిద్ర సంఘటన, స్వీయీకరణ మరియు క్వాంటం-చేతన (పైనియల్) మోడ్లు.
- **LLM-సిద్ధం** — ప్రధాన LangChain & LlamaIndex ఇంటిగ్రేషన్లు, ఫ్యూ-షాట్ సెలెక్టర్, అవుట్‌పుట్ రీర్యాంకర్, డ్రిఫ్ట్ డిటెక్టర్.

---

## ఆర్కిటెక్చర్

```
lmm/
├── core.py                  # ప్రధాన LMM పైప్‌లైన్
├── cli.py                   # CLI ప్రవేశ బిందువు
├── qubo.py                  # స్పార్స్ QUBO మాతృక నిర్మాత
├── solvers.py               # SA / Ising SA / రిలాక్సేషన్ / గ్రీడీ / సబ్‌మాడ్యులర్
├── surprise.py              # సమాచార-సైద్ధాంతిక ఆశ్చర్యం
├── selector.py              # అనుకూల ఎంపిక వ్యూహం
├── processor.py             # ప్రాధాన్యత ప్రాసెసింగ్ + కాష్
├── pipeline.py              # SmartSelector → SmartProcessor ఆర్కెస్ట్రేషన్
├── _compat.py               # రన్‌టైమ్ సామర్థ్య శోధన & స్పార్స్ సహాయకులు
├── dharma/                  # డిజిటల్ ధర్మం (బౌద్ధ-తత్వం ఆప్టిమైజేషన్)
│   ├── api.py               # DharmaLMM అధిక-స్థాయి API
│   ├── energy.py            # ప్లగ్గబుల్ శక్తి పదాలు (దుఃఖ, ప్రజ్ఞ, కరుణ, …)
│   ├── engine.py            # UniversalDharmaEngine — స్వయంచాలక-మళ్ళింపు సాల్వర్
│   ├── fep.py               # FEP ≡ KCL ODE సాల్వర్
│   ├── neuromorphic.py      # మెమ్రిస్టర్ క్రాస్‌బార్ ASIC సిమ్యులేటర్
│   ├── reranker.py          # RAG క్యాస్కేడ్ రీర్యాంకర్ + ఉద్దేశ్య రూటర్
│   └── …
├── reasoning/               # 8 FEP-ఆధారిత తార్కిక మోడ్లు
│   ├── adaptive.py          # డైనమిక్ పారామీటర్ ట్యూనింగ్
│   ├── theoretical.py       # తార్కిక నిర్మాణ గ్రాఫ్
│   ├── hyper.py             # అపగమన సుప్త-నోడ్ ఇంజెక్షన్
│   ├── active.py            # బాహ్య జ్ఞాన సేకరణ
│   ├── alaya.py             # హెబ్బియన్ సైనాప్టిక్ జ్ఞాపకం (ఆలయ)
│   ├── sleep.py             # NREM/REM జ్ఞాపక సంఘటన
│   ├── embodiment.py        # 6-ఇంద్రియ మల్టీమోడల్ ఫ్యూజన్
│   ├── pineal.py            # క్వాంటం చేతన (హార్డ్‌వేర్ TRNG)
│   └── orchestrator.py      # మోడ్ ఎంపిక & పంపిణీ
├── scale/                   # ట్రిలియన్-టోకెన్ స్ట్రీమింగ్
│   ├── sketch.py            # Count-Min Sketch, Streaming Histogram
│   ├── stream.py            # స్ట్రీమింగ్ ఆశ్చర్యం
│   ├── cascade.py           # బహుళ-స్థాయి క్యాస్కేడ్ ఫిల్టర్ (1T → K)
│   └── pipeline.py          # ScalablePipeline
├── llm/                     # LLM వర్క్‌ఫ్లో సహాయకులు
│   ├── fewshot.py           # ఫ్యూ-షాట్ ఉదాహరణ సెలెక్టర్
│   ├── reranker.py          # అవుట్‌పుట్ రీర్యాంకర్
│   ├── drift.py             # పంపిణీ డ్రిఫ్ట్ డిటెక్టర్
│   ├── sampler.py           # టోకెన్ శాంపుల్లర్
│   └── embeddings.py        # ఏకీకృత ఎంబెడింగ్ అడాప్టర్
├── integrations/
│   ├── langchain.py         # DharmaRetriever / DocumentCompressor / ExampleSelector
│   └── llamaindex.py        # DharmaNodePostprocessor
```

---

## ఇన్‌స్టాలేషన్

```bash
# కోర్ (numpy + scipy మాత్రమే)
pip install -e .

# డెవ్ సాధనాలతో
pip install -e ".[dev]"

# అన్ని ఐచ్ఛిక ఇంటిగ్రేషన్లతో
pip install -e ".[all]"
```

### ఐచ్ఛిక అదనపు ప్యాకేజీలు

| అదనపు | ప్యాకేజీలు | ఉద్దేశ్యం |
|-------|-----------|-----------|
| `dev` | pytest, ruff | పరీక్ష & లింటింగ్ |
| `dharma` | hnswlib | స్పార్స్ k-NN గ్రాఫ్ నిర్మాణం |
| `langchain` | langchain-core | LangChain ఇంటిగ్రేషన్ |
| `llamaindex` | llama-index-core | LlamaIndex ఇంటిగ్రేషన్ |
| `all` | పై అన్నీ | సమస్తం |

---

## త్వరిత ప్రారంభం

### Python API

```python
import numpy as np
from lmm.core import LMM

# అత్యంత ఆశ్చర్యకరమైన top-K అంశాలను ఎంచుకోండి
surprises = np.array([1.0, 5.0, 2.0, 8.0, 3.0, 7.0])
model = LMM(k=3, solver_method="sa")
result = model.select_from_surprises(surprises)
print(result.selected_indices)  # ఉదా. [3, 5, 1]
```

### CLI

```bash
# త్వరిత డెమో నడుపు
lmm --demo --k 10 --method sa

# NumPy ఫైల్ నుండి
lmm --input data.npy --k 5 --method greedy
```

---

## సర్వర్ మోడ్ (Alaya V5)

Alaya-Vijñāna v5.0 సర్వర్ రియల్-టైమ్ భావోద్వేగ తరంగదైర్ఘ్య విజువలైజేషన్, 8-మోడ్ FEP తర్కం మరియు Claude/Gemini ఆటో-రూటింగ్‌తో వెబ్ UI అందిస్తుంది.

### ముందస్తు అవసరాలు

```bash
pip install -e ".[server]"
```

### సర్వర్‌ను ప్రారంభించండి

**PowerShell (Windows):**
```powershell
.\Start-DharmaServer.ps1
```

**Python (క్రాస్-ప్లాట్‌ఫారమ్):**
```bash
python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

బ్రౌజర్‌లో తెరవండి: [http://localhost:8000](http://localhost:8000)

### సర్వర్‌ను ఆపండి

**Windows (బ్యాచ్):**
```batch
.\stop-dharma-server.bat
```

**PowerShell:** `Start-DharmaServer.ps1` నడుస్తున్న విండోలో `Ctrl+C` నొక్కండి.

**Linux/macOS:** `Ctrl+C` నొక్కండి లేదా అమలు చేయండి:
```bash
kill $(cat server.pid)
```

### API ఎండ్‌పాయింట్లు

| ఎండ్‌పాయింట్ | మెథడ్ | వివరణ |
|-------------|--------|-------|
| `/` | GET | వెబ్ UI (Alaya V5 ఫ్రంటెండ్) |
| `/api/descent` | POST | ప్రధాన తర్క పైప్‌లైన్ |
| `/api/descent/stream` | POST | స్ట్రీమింగ్ SSE ప్రతిస్పందన |
| `/api/dharma/auto` | POST | ఆటో-రూటింగ్ Dharma తర్కం |
| `/api/sense` | POST | భావోద్వేగ తరంగదైర్ఘ్య విశ్లేషణ |
| `/api/status` | GET | సిస్టమ్ స్థితి + హృదయస్పందన టెలిమెట్రీ |

---

## స్వయంప్రతిపత్తి లక్షణాలు

సర్వర్‌లో వినియోగదారు పరస్పర చర్యల మధ్య నిరంతరం పనిచేసే మూడు స్వయంప్రతిపత్తి ఉప-వ్యవస్థలు ఉన్నాయి:

### హృదయస్పందన డెమన్ (Heartbeat)

4-కొలత స్థితి వెక్టర్ `[ప్రేమ, తర్కం, భయం, సృష్టి]`ను నిర్వహించే నిరంతర-సమయ FEP స్థితి పరిణామ లూప్:

- **Tick వ్యవధి:** 100ms (చురుకు) → 5s (నిష్క్రియ), అనుకూల నెమ్మదింపు
- **ఎంట్రోపీ ఇంజెక్షన్:** ప్రతి tick-లో `os.urandom()` ద్వారా హార్డ్‌వేర్ ఎంట్రోపీ
- **బరువు అనుసరణ:** కరుణ (Karuna) మరియు మైత్రి (Metta) బరువులు మధ్యమ మార్గం CV ట్రాకింగ్ (లక్ష్యం CV=0.5) ద్వారా స్వయంచాలక సర్దుబాటు
- **నిద్ర సంఘటన:** 60s నిష్క్రియత తర్వాత NREM/REM జ్ఞాపకం రీప్లే ట్రిగ్గర్

### సందర్భ వారధి (ఆలయ జ్ఞాపకం)

సాధారణ చరిత్ర కత్తిరింపు (`history[-20:]`)ను జ్ఞాపకం-ఆధారిత తెలివైన సందర్భ ఎంపికతో భర్తీ చేస్తుంది:

- **Modern Hopfield Network** ద్వారా AlayaMemory నుండి రీకాల్
- రీకాల్ నమూనాలతో కోసైన్ సారూప్యత ద్వారా సంభాషణ చరిత్ర స్కోరింగ్
- ఎల్లప్పుడూ అత్యంత ఇటీవలి 3 సందేశాలను చేర్చడం (రీసెన్సీ బయాస్)
- మిగిలిన బడ్జెట్ (గరిష్టంగా 20 సందేశాలు) ఔచిత్యం స్కోర్ ద్వారా నింపడం

### అర్థ భావోద్వేగాలు (Semantic Emotions)

`config/semantic_emotions.json`లో నిర్వచించిన నాలుగు కొలతలలో కీవర్డ్ మ్యాచింగ్ ద్వారా రియల్-టైమ్ భావోద్వేగ తరంగదైర్ఘ్య విశ్లేషణ:

| కొలత | బౌద్ధ భావన | సంకేతం |
|------|-----------|--------|
| ప్రేమ | కరుణ (Karuna) | జాలి, వెచ్చదనం, కృతజ్ఞత |
| తర్కం | హేతువిద్య (Hetuvidya) | విశ్లేషణ, తర్కం, రుజువు |
| భయం | దుఃఖ (Dukkha) | ఆందోళన, సందేహం, బాధ |
| సృష్టి | సృష్టి (Sṛṣṭi) | ఆవిష్కరణ, ఊహ, కళ |

ప్రతి కీవర్డ్‌కు బరువు (0.3–1.0) ఉంటుంది. ఫలిత 4D వెక్టర్ తర్క మోడ్ ఎంపిక మరియు హృదయస్పందన స్థితి ఇంజెక్షన్‌ను నడిపిస్తుంది.

---

## సాల్వర్ పద్ధతులు

| పద్ధతి | వివరణ | సంక్లిష్టత |
|--------|--------|------------|
| `sa` | సిమ్యులేటెడ్ అనీలింగ్ (డిఫాల్ట్) | O(n · steps) |
| `ising_sa` | వెక్టరైజ్డ్ డెల్టాతో Ising-రూపం SA | O(1) ప్రతి ఫ్లిప్‌కు |
| `relaxation` | నిరంతర రిలాక్సేషన్ + రౌండింగ్ (SLSQP) | O(n²) |
| `greedy` | గ్రీడీ ఎంపిక | O(n · k) |

---

## ధర్మ ఇంజన్ — ప్లగ్గబుల్ శక్తి పదాలు

**UniversalDharmaEngine** మీకు బౌద్ధ-తత్వం ఆధారిత శక్తి పదాలను కూర్చడానికి అనుమతిస్తుంది, ప్రతి ఒక్కటి దాని గణిత లక్షణాన్ని ప్రకటిస్తుంది. ఇంజన్ స్వయంచాలకంగా అత్యుత్తమ సాల్వర్‌కు మళ్ళిస్తుంది.

```python
from lmm.dharma import UniversalDharmaEngine, DukkhaTerm, KarunaTerm

engine = UniversalDharmaEngine(n_candidates=1000)
engine.add(DukkhaTerm(surprises, weight=1.0))    # రేఖీయ  → Top-K
engine.add(KarunaTerm(impact_graph, weight=2.0)) # సూపర్‌మాడ్యులర్ → SA
result = engine.synthesize_and_solve(k=10)
```

### తత్వం → గణితం → సాల్వర్ మ్యాపింగ్

| బౌద్ధ భావన | శక్తి పదం | గణిత లక్షణం | సాల్వర్ |
|------------|-----------|-------------|---------|
| ప్రజ్ఞ (జ్ఞానం) | `PrajnaTerm` | రేఖీయ | Top-K వర్గీకరణ |
| కరుణ (దయ) | `KarunaTerm` | సూపర్‌మాడ్యులర్ | వార్మ్-స్టార్ట్ + SA |
| శీల (నడవడిక) | `SilaTerm` | సబ్‌మాడ్యులర్ | లేజీ గ్రీడీ |
| మధ్యమక (మధ్యమ మార్గం) | `MadhyamakaCriterion` | లియాపునోవ్ | ఎక్స్‌పోనెన్షియల్ గ్రేడియంట్ |
| దుఃఖం (బాధ) | `DukkhaTerm` | రేఖీయ | Top-K వర్గీకరణ |

---

## తార్కిక ఆర్కెస్ట్రేటర్

**DharmaReasonerOrchestrator** ప్రశ్న సంక్లిష్టత ఆధారంగా 8 తార్కిక మోడ్లలో ఒకదాన్ని ఎంచుకుంటుంది:

```python
from lmm.reasoning import DharmaReasonerOrchestrator

orch = DharmaReasonerOrchestrator(n_candidates=500)
result = orch.reason(query_vector, context_vectors)
print(result.mode_used, result.explanation)
```

| మోడ్ | మాడ్యూల్ | ప్రేరణ |
|------|---------|--------|
| అనుకూల | `adaptive.py` | 応病与薬 — రోగికి తగిన ఔషధం |
| సైద్ధాంతిక | `theoretical.py` | 因明 — బౌద్ధ ఔపచారిక తర్కం |
| అపగమన | `hyper.py` | 般若の飛躍 — ప్రజ్ఞ యొక్క జ్ఞానోదయ దూకుడు |
| చురుకు అనుమితి | `active_inference.py` | 托鉢 — బాహ్య సత్యాన్వేషణ |
| ఆలయ జ్ఞాపకం | `alaya.py` | 阿頼耶識 — గిడ్డంగి చేతన |
| నిద్ర | `sleep.py` | 禅定 — NREM/REM సంఘటన |
| స్వీయీకరణ | `embodiment.py` | 六根 — ఆరు ఇంద్రియ పద్ధతులు |
| పైనియల్ (క్వాంటం) | `pineal.py` | 松果体 — హార్డ్‌వేర్-ఎంట్రోపీ చేతన |

---

## ట్రిలియన్-టోకెన్ స్కేలింగ్

స్థిర మెమరీతో అనంత పెద్ద డేటా స్ట్రీమ్‌లను ప్రాసెస్ చేయండి:

```python
from lmm.scale import ScalablePipeline
from pathlib import Path

pipe = ScalablePipeline(k=10, chunk_size=100_000)
pipe.fit_files([Path("shard_001.npy"), ...])
result = pipe.run_files([Path("data_001.npy"), ...])
print(result.summary)
```

**మూడు-దశల క్యాస్కేడ్:** 1 T టోకెన్లు → 100 M (స్ట్రీమింగ్ top-k) → 10 K (శాతాంక) → K (QUBO).

---

## LLM ఏకీకరణ

### LangChain

```python
from lmm.integrations.langchain import DharmaDocumentCompressor

compressor = DharmaDocumentCompressor(k=5)
# LangChain యొక్క ContextualCompressionRetriever తో ఉపయోగించండి
```

### LlamaIndex

```python
from lmm.integrations.llamaindex import DharmaNodePostprocessor

postprocessor = DharmaNodePostprocessor(top_k=5)
# LlamaIndex క్వెరీ ఇంజన్‌తో ఉపయోగించండి
```

### స్వతంత్ర LLM సహాయకులు

```python
from lmm.llm import FewShotSelector, OutputReranker, DriftDetector

# వైవిధ్య హామీతో ఫ్యూ-షాట్ ఉదాహరణ ఎంపిక
selector = FewShotSelector(k=5)
examples = selector.select(candidates, query)

# ఆశ్చర్యం × వైవిధ్యం ఆధారంగా LLM అవుట్‌పుట్లను రీర్యాంక్ చేయండి
reranker = OutputReranker(top_k=3)
best = reranker.rerank(outputs)

# కాలక్రమేణా అవుట్‌పుట్ పంపిణీ డ్రిఫ్ట్‌ను పర్యవేక్షించండి
detector = DriftDetector(window=100)
report = detector.update(new_output)
```

---

## DharmaLMM — అధిక-స్థాయి API

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

## న్యూరోమార్ఫిక్ సిమ్యులేషన్

శక్తి-సమర్థమైన FEP గణన కోసం మెమ్రిస్టర్-క్రాస్‌బార్ హార్డ్‌వేర్‌ను సిమ్యులేట్ చేయండి:

```python
from lmm.dharma.neuromorphic import NeuromorphicChip

chip = NeuromorphicChip(size=64)
chip.program(weight_matrix)
report = chip.run(input_voltages, steps=100)
# ~10 fJ/MAC, ~30 ns కన్వర్జెన్స్
```

---

## డిపెండెన్సీలు

| ప్యాకేజీ | వెర్షన్ | అవసరమా |
|---------|---------|---------|
| numpy | >= 1.24 | అవును |
| scipy | >= 1.10 | అవును |
| hnswlib | >= 0.8.0 | ఐచ్ఛికం (స్పార్స్ గ్రాఫ్) |
| langchain-core | >= 0.2.0 | ఐచ్ఛికం (LangChain) |
| llama-index-core | >= 0.10.0 | ఐచ్ఛికం (LlamaIndex) |

---

## సైద్ధాంతిక పునాదులు

| భావన | సూత్రీకరణ |
|------|----------|
| **FEP ≡ KCL** | `dV/dt = −V/τ + g'(V) · G · ε` — కిర్చాఫ్ యొక్క కరెంట్ చట్టంగా అంచనా-తప్పిదం కనిష్టీకరణ |
| **సూపర్‌మాడ్యులర్ కరుణ** | కరుణ పెరుగుతున్న రాబడులను ప్రదర్శిస్తుంది: `f(S∪{i}) − f(S) ≥ f(T∪{i}) − f(T)` S ⊆ T కోసం |
| **సబ్‌మాడ్యులర్ శీల** | నడవడిక తగ్గుతున్న రాబడులను ప్రదర్శిస్తుంది — లేజీ గ్రీడీ (1−1/e) హామీ ఇస్తుంది |
| **మధ్యమక** | మధ్యమ మార్గం లియాపునోవ్-స్థిర ఎక్స్‌పోనెన్షియల్ గ్రేడియంట్ డిసెంట్ ద్వారా CV = 0.5ని లక్ష్యంగా చేసుకుంటుంది |
| **పైనియల్ కోలాప్స్** | తరంగ ఫంక్షన్ కోలాప్స్ PRNG కాకుండా భౌతిక ఎంట్రోపీ (హార్డ్‌వేర్ TRNG) ఉపయోగిస్తుంది |

---

## లైసెన్స్

[MIT](LICENSE)
