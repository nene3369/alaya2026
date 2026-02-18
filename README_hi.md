# LMM — डिजिटल धर्म के साथ क्लासिकल QUBO ऑप्टिमाइज़र

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.0.0-green.svg)](pyproject.toml)

> [English](README_en.md) | [日本語](README.md) | [中文](README_zh.md) | [한국어](README_ko.md) | [Español](README_es.md) | [Français](README_fr.md) | [বাংলা](README_bn.md) | [தமிழ்](README_ta.md) | [తెలుగు](README_te.md) | [मराठी](README_mr.md) | [‎اردو‎](README_ur.md) | [ગુજરાતી](README_gu.md) | [ಕನ್ನಡ](README_kn.md) | [മലయാളം](README_ml.md) | [ਪੰਜਾਬੀ](README_pa.md)

> **Digital Dharma OS (Alaya V5)** — QUBO गणित, बौद्ध दर्शन और मुक्त ऊर्जा सिद्धांत (FEP)
> तंत्रिका विज्ञान को एक जीवित, स्व-विकासशील प्रणाली में विलय करने वाला चेतना-जागरूक
> अनुकूलन मंच। सर्वर निरंतर हृदय गति गतिकी, स्मृति-चालित संदर्भ चयन और भावनात्मक
> तरंगदैर्ध्य संवेदन चलाता है — असतत LLM कॉल और निरंतर संज्ञान के बीच सेतु का काम करता है।

एक **D-Wave-मुक्त** क्लासिकल QUBO ऑप्टिमाइज़ेशन लाइब्रेरी जो
सूचना-सैद्धांतिक सरप्राइज़ को बौद्ध दर्शन से प्रेरित ऊर्जा फलनों (energy functions),
Free Energy Principle (FEP) तर्कप्रणाली, और चेतना-सचेत कम्प्यूटिंग के साथ
मिलाती है — यह सब सामान्य हार्डवेयर पर चलता है।

---

## मुख्य विशेषताएँ

- **क्वांटम कंप्यूटर की आवश्यकता नहीं** — क्लासिकल सॉल्वर (SA, Ising SA, सबमॉड्यूलर ग्रीडी) D-Wave की गुणवत्ता से मेल खाते या उससे बेहतर हैं।
- **ट्रिलियन-टोकन सक्षम** — स्ट्रीमिंग प्रायिकता डेटा संरचनाएँ (Count-Min Sketch, Streaming Histogram) मेमोरी O(k) में रखती हैं।
- **धर्म-बीजगणित (Dharma-Algebra) इंजन** — प्लगेबल बौद्ध ऊर्जा पद गणितीय रूप से इष्टतम सॉल्वर को स्वचालित रूप से रूट करते हैं।
- **8-मोड तर्क ऑर्केस्ट्रेटर** — अनुकूली (adaptive), सैद्धांतिक (theoretical), अपहरणात्मक (abductive), सक्रिय अनुमान (active-inference), स्मृति (आलय/Alaya), निद्रा समेकन (sleep consolidation), देहधारी (embodied), और क्वांटम-चेतना (पीनियल/Pineal)।
- **LLM-तैयार** — प्रथम श्रेणी LangChain और LlamaIndex एकीकरण, few-shot चयनकर्ता, आउटपुट रीरैंकर, ड्रिफ्ट डिटेक्टर।

---

## आर्किटेक्चर

```
lmm/
├── core.py                  # मुख्य LMM पाइपलाइन
├── cli.py                   # CLI प्रवेश बिंदु
├── qubo.py                  # विरल (Sparse) QUBO मैट्रिक्स बिल्डर
├── solvers.py               # SA / Ising SA / relaxation / greedy / submodular
├── surprise.py              # सूचना-सैद्धांतिक सरप्राइज़
├── selector.py              # अनुकूली चयन रणनीति
├── processor.py             # प्राथमिकता प्रसंस्करण + कैश
├── pipeline.py              # SmartSelector → SmartProcessor ऑर्केस्ट्रेशन
├── _compat.py               # रनटाइम क्षमता पहचान और विरल सहायक
├── dharma/                  # डिजिटल धर्म (बौद्ध-दर्शन ऑप्टिमाइज़ेशन)
│   ├── api.py               # DharmaLMM उच्च-स्तरीय API
│   ├── energy.py            # प्लगेबल ऊर्जा पद (दुःख, प्रज्ञा, करुणा, …)
│   ├── engine.py            # UniversalDharmaEngine — स्वचालित-रूटिंग सॉल्वर
│   ├── fep.py               # FEP ≡ KCL ODE सॉल्वर
│   ├── neuromorphic.py      # मेमरिस्टर क्रॉसबार ASIC सिम्युलेटर
│   ├── reranker.py          # RAG कैस्केड रीरैंकर + इंटेंट राउटर
│   └── …
├── reasoning/               # 8 FEP-आधारित तर्क मोड
│   ├── adaptive.py          # गतिशील पैरामीटर ट्यूनिंग
│   ├── theoretical.py       # तार्किक संरचना ग्राफ
│   ├── hyper.py             # अपहरणात्मक (Abductive) अव्यक्त-नोड इंजेक्शन
│   ├── active.py            # बाह्य ज्ञान अधिग्रहण
│   ├── alaya.py             # हेबियन सिनैप्टिक स्मृति (आलय)
│   ├── sleep.py             # NREM/REM स्मृति समेकन
│   ├── embodiment.py        # षड्-इन्द्रिय बहुमोडल संलयन
│   ├── pineal.py            # क्वांटम चेतना (हार्डवेयर TRNG)
│   └── orchestrator.py      # मोड चयन और प्रेषण
├── scale/                   # ट्रिलियन-टोकन स्ट्रीमिंग
│   ├── sketch.py            # Count-Min Sketch, Streaming Histogram
│   ├── stream.py            # स्ट्रीमिंग सरप्राइज़
│   ├── cascade.py           # बहु-स्तरीय कैस्केड फ़िल्टर (1T → K)
│   └── pipeline.py          # ScalablePipeline
├── llm/                     # LLM कार्यप्रवाह सहायक
│   ├── fewshot.py           # Few-shot उदाहरण चयनकर्ता
│   ├── reranker.py          # आउटपुट रीरैंकर
│   ├── drift.py             # वितरण ड्रिफ्ट डिटेक्टर
│   ├── sampler.py           # टोकन सैंपलर
│   └── embeddings.py        # एकीकृत एम्बेडिंग एडेप्टर
├── integrations/
│   ├── langchain.py         # DharmaRetriever / DocumentCompressor / ExampleSelector
│   └── llamaindex.py        # DharmaNodePostprocessor
```

---

## स्थापना (Installation)

```bash
# कोर (केवल numpy + scipy)
pip install -e .

# डेव टूल्स के साथ
pip install -e ".[dev]"

# सभी वैकल्पिक एकीकरणों के साथ
pip install -e ".[all]"
```

### वैकल्पिक एक्स्ट्रा

| एक्स्ट्रा | पैकेज | उद्देश्य |
|-------|----------|---------|
| `dev` | pytest, ruff | परीक्षण और लिंटिंग |
| `dharma` | hnswlib | विरल k-NN ग्राफ निर्माण |
| `langchain` | langchain-core | LangChain एकीकरण |
| `llamaindex` | llama-index-core | LlamaIndex एकीकरण |
| `all` | उपरोक्त सभी | सब कुछ |

---

## त्वरित प्रारंभ (Quick Start)

### Python API

```python
import numpy as np
from lmm.core import LMM

# शीर्ष-K सबसे आश्चर्यजनक आइटम चुनें
surprises = np.array([1.0, 5.0, 2.0, 8.0, 3.0, 7.0])
model = LMM(k=3, solver_method="sa")
result = model.select_from_surprises(surprises)
print(result.selected_indices)  # उदा. [3, 5, 1]
```

### CLI

```bash
# एक त्वरित डेमो चलाएँ
lmm --demo --k 10 --method sa

# NumPy फ़ाइल से
lmm --input data.npy --k 5 --method greedy
```

---

## सर्वर मोड (Alaya V5)

Alaya-Vijñāna v5.0 सर्वर वास्तविक समय भावनात्मक तरंगदैर्ध्य दृश्यीकरण, 8-मोड FEP तर्क और Claude/Gemini स्वचालित रूटिंग के साथ एक वेब UI प्रदान करता है।

### पूर्वापेक्षाएँ

```bash
pip install -e ".[server]"
```

### सर्वर प्रारंभ करें

**PowerShell (Windows):**
```powershell
.\Start-DharmaServer.ps1
```

**Python (क्रॉस-प्लेटफ़ॉर्म):**
```bash
python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

ब्राउज़र में खोलें: [http://localhost:8000](http://localhost:8000)

### सर्वर बंद करें

**Windows (बैच):**
```batch
.\stop-dharma-server.bat
```

**PowerShell:** `Start-DharmaServer.ps1` चलाने वाली विंडो में `Ctrl+C` दबाएँ।

**Linux/macOS:** `Ctrl+C` दबाएँ या चलाएँ:
```bash
kill $(cat server.pid)
```

### API अंतिम बिंदु

| अंतिम बिंदु | विधि | विवरण |
|------------|-------|-------|
| `/` | GET | वेब UI (Alaya V5 फ्रंटएंड) |
| `/api/descent` | POST | मुख्य तर्क पाइपलाइन |
| `/api/descent/stream` | POST | स्ट्रीमिंग SSE प्रतिक्रिया |
| `/api/dharma/auto` | POST | स्वचालित रूटिंग Dharma तर्क |
| `/api/sense` | POST | भावनात्मक तरंगदैर्ध्य विश्लेषण |
| `/api/status` | GET | सिस्टम स्थिति + हृदय गति टेलीमेट्री |

---

## स्वायत्त सुविधाएँ

सर्वर में तीन स्वायत्त उपप्रणालियाँ शामिल हैं जो उपयोगकर्ता इंटरैक्शन के बीच निरंतर संचालित होती हैं:

### हृदय गति डेमन (Heartbeat)

4-आयामी स्थिति सदिश `[प्रेम, तर्क, भय, सृजन]` को बनाए रखने वाला निरंतर-समय FEP स्थिति विकास लूप:

- **Tick अंतराल:** 100ms (सक्रिय) → 5s (निष्क्रिय), अनुकूली मंदी
- **एन्ट्रॉपी इंजेक्शन:** प्रत्येक tick पर `os.urandom()` के माध्यम से हार्डवेयर एन्ट्रॉपी
- **भार अनुकूलन:** करुणा (Karuna) और मैत्री (Metta) भार मध्यम मार्ग CV ट्रैकिंग (लक्ष्य CV=0.5) द्वारा स्वचालित समायोजन
- **नींद समेकन:** 60s निष्क्रियता के बाद NREM/REM स्मृति पुनर्चालन ट्रिगर

### संदर्भ सेतु (आलय स्मृति)

सरल इतिहास कटौती (`history[-20:]`) को स्मृति-चालित बुद्धिमान संदर्भ चयन से प्रतिस्थापित करता है:

- **Modern Hopfield Network** के माध्यम से AlayaMemory से स्मरण
- स्मरण पैटर्न के साथ कोसाइन समानता द्वारा वार्तालाप इतिहास स्कोरिंग
- हमेशा 3 सबसे हाल के संदेश शामिल (समीपता पूर्वाग्रह)
- शेष बजट (अधिकतम 20 संदेश) प्रासंगिकता स्कोर द्वारा भरना

### अर्थपूर्ण भावनाएँ (Semantic Emotions)

`config/semantic_emotions.json` में परिभाषित चार आयामों में कीवर्ड मिलान द्वारा वास्तविक समय भावनात्मक तरंगदैर्ध्य विश्लेषण:

| आयाम | बौद्ध अवधारणा | संकेत |
|-------|--------------|-------|
| प्रेम | करुणा (Karuna) | दया, उष्मा, कृतज्ञता |
| तर्क | हेतुविद्या (Hetuvidya) | विश्लेषण, तर्क, प्रमाण |
| भय | दुःख (Dukkha) | चिंता, संदेह, पीड़ा |
| सृजन | सृष्टि (Sṛṣṭi) | नवाचार, कल्पना, कला |

प्रत्येक कीवर्ड का भार (0.3–1.0) होता है। परिणामी 4D सदिश तर्क मोड चयन और हृदय गति स्थिति इंजेक्शन को चलाता है।

---

## सॉल्वर विधियाँ

| विधि | विवरण | जटिलता |
|--------|-------------|------------|
| `sa` | सिम्युलेटेड एनीलिंग (डिफ़ॉल्ट) | O(n · steps) |
| `ising_sa` | इज़िंग-रूप SA वेक्टरीकृत डेल्टा के साथ | O(1) प्रति फ्लिप |
| `relaxation` | सतत शिथिलन + राउंडिंग (SLSQP) | O(n²) |
| `greedy` | लालची (Greedy) चयन | O(n · k) |

---

## धर्म इंजन — प्लगेबल ऊर्जा पद

**UniversalDharmaEngine** आपको बौद्ध दर्शन से प्रेरित ऊर्जा पदों
को संयोजित करने देता है, जिनमें प्रत्येक अपने गणितीय गुण की घोषणा करता है।
इंजन स्वचालित रूप से इष्टतम सॉल्वर को रूट करता है।

```python
from lmm.dharma import UniversalDharmaEngine, DukkhaTerm, KarunaTerm

engine = UniversalDharmaEngine(n_candidates=1000)
engine.add(DukkhaTerm(surprises, weight=1.0))    # linear  → Top-K
engine.add(KarunaTerm(impact_graph, weight=2.0)) # supermodular → SA
result = engine.synthesize_and_solve(k=10)
```

### दर्शन → गणित → सॉल्वर मैपिंग

| बौद्ध अवधारणा | ऊर्जा पद | गणितीय गुण | सॉल्वर |
|-----------------|-------------|---------------|--------|
| प्रज्ञा (Prajna — ज्ञान) | `PrajnaTerm` | रैखिक (linear) | Top-K सॉर्ट |
| करुणा (Karuna — दया) | `KarunaTerm` | सुपरमॉड्यूलर | वार्म-स्टार्ट + SA |
| शील (Sila — आचरण) | `SilaTerm` | सबमॉड्यूलर | लेज़ी ग्रीडी |
| मध्यमक (Madhyamaka — मध्यम मार्ग) | `MadhyamakaCriterion` | ल्यापुनोव (Lyapunov) | एक्सपोनेंशियल ग्रेडिएंट |
| दुःख (Dukkha — पीड़ा) | `DukkhaTerm` | रैखिक (linear) | Top-K सॉर्ट |

---

## तर्क ऑर्केस्ट्रेटर (Reasoning Orchestrator)

**DharmaReasonerOrchestrator** क्वेरी की जटिलता के आधार पर 8 तर्क
मोड में से चुनता है:

```python
from lmm.reasoning import DharmaReasonerOrchestrator

orch = DharmaReasonerOrchestrator(n_candidates=500)
result = orch.reason(query_vector, context_vectors)
print(result.mode_used, result.explanation)
```

| मोड | मॉड्यूल | प्रेरणा |
|------|--------|-------------|
| अनुकूली (Adaptive) | `adaptive.py` | 応病与薬 — रोग के अनुसार औषधि |
| सैद्धांतिक (Theoretical) | `theoretical.py` | 因明 — बौद्ध औपचारिक तर्कशास्त्र (हेतुविद्या) |
| अपहरणात्मक (Abductive) | `hyper.py` | 般若の飛躍 — प्रज्ञा की छलाँग |
| सक्रिय अनुमान (Active Inference) | `active_inference.py` | 托鉢 — बाह्य सत्य की खोज (भिक्षाटन) |
| आलय स्मृति (Alaya Memory) | `alaya.py` | 阿頼耶識 — आलयविज्ञान (भण्डार चेतना) |
| निद्रा (Sleep) | `sleep.py` | 禅定 — NREM/REM समेकन (ध्यान) |
| देहधारी (Embodied) | `embodiment.py` | 六根 — षड्-इन्द्रिय (छह ज्ञानेन्द्रियाँ) |
| पीनियल/क्वांटम (Pineal) | `pineal.py` | 松果体 — हार्डवेयर-एन्ट्रॉपी चेतना |

---

## ट्रिलियन-टोकन स्केलिंग

स्थिर मेमोरी के साथ मनमाने बड़े डेटा स्ट्रीम को प्रोसेस करें:

```python
from lmm.scale import ScalablePipeline
from pathlib import Path

pipe = ScalablePipeline(k=10, chunk_size=100_000)
pipe.fit_files([Path("shard_001.npy"), ...])
result = pipe.run_files([Path("data_001.npy"), ...])
print(result.summary)
```

**तीन-चरणीय कैस्केड:** 1 T टोकन → 100 M (स्ट्रीमिंग top-k) → 10 K (परसेंटाइल) → K (QUBO)।

---

## LLM एकीकरण

### LangChain

```python
from lmm.integrations.langchain import DharmaDocumentCompressor

compressor = DharmaDocumentCompressor(k=5)
# LangChain के ContextualCompressionRetriever के साथ उपयोग करें
```

### LlamaIndex

```python
from lmm.integrations.llamaindex import DharmaNodePostprocessor

postprocessor = DharmaNodePostprocessor(top_k=5)
# LlamaIndex क्वेरी इंजन के साथ उपयोग करें
```

### स्वतंत्र LLM सहायक

```python
from lmm.llm import FewShotSelector, OutputReranker, DriftDetector

# विविधता गारंटी के साथ Few-shot उदाहरण चयन
selector = FewShotSelector(k=5)
examples = selector.select(candidates, query)

# सरप्राइज़ × विविधता द्वारा LLM आउटपुट रीरैंक करें
reranker = OutputReranker(top_k=3)
best = reranker.rerank(outputs)

# समय के साथ आउटपुट वितरण ड्रिफ्ट की निगरानी करें
detector = DriftDetector(window=100)
report = detector.update(new_output)
```

---

## DharmaLMM — उच्च-स्तरीय API

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

## न्यूरोमॉर्फिक सिमुलेशन

ऊर्जा-कुशल FEP गणना के लिए मेमरिस्टर-क्रॉसबार हार्डवेयर का अनुकरण करें:

```python
from lmm.dharma.neuromorphic import NeuromorphicChip

chip = NeuromorphicChip(size=64)
chip.program(weight_matrix)
report = chip.run(input_voltages, steps=100)
# ~10 fJ/MAC, ~30 ns अभिसरण (convergence)
```

---

## निर्भरताएँ (Dependencies)

| पैकेज | संस्करण | आवश्यक |
|---------|---------|----------|
| numpy | >= 1.24 | हाँ |
| scipy | >= 1.10 | हाँ |
| hnswlib | >= 0.8.0 | वैकल्पिक (विरल ग्राफ) |
| langchain-core | >= 0.2.0 | वैकल्पिक (LangChain) |
| llama-index-core | >= 0.10.0 | वैकल्पिक (LlamaIndex) |

---

## सैद्धांतिक आधार

| अवधारणा | सूत्रीकरण |
|---------|-------------|
| **FEP ≡ KCL** | `dV/dt = −V/τ + g'(V) · G · ε` — भविष्यवाणी-त्रुटि न्यूनीकरण किर्चहॉफ़ के करंट नियम के रूप में |
| **सुपरमॉड्यूलर करुणा** | करुणा में वर्धमान प्रतिफल होता है: `f(S∪{i}) − f(S) ≥ f(T∪{i}) − f(T)` जहाँ S ⊆ T |
| **सबमॉड्यूलर शील** | शील में ह्रासमान प्रतिफल होता है — लेज़ी ग्रीडी (1−1/e) गारंटी देता है |
| **मध्यमक** | मध्यम मार्ग CV = 0.5 को ल्यापुनोव-स्थिर एक्सपोनेंशियल ग्रेडिएंट डिसेंट द्वारा लक्षित करता है |
| **पीनियल पतन (Pineal Collapse)** | तरंग-फलन पतन PRNG के बजाय भौतिक एन्ट्रॉपी (हार्डवेयर TRNG) का उपयोग करता है |

---

## लाइसेंस

[MIT](LICENSE)
