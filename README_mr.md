# LMM — डिजिटल धर्मासह क्लासिकल QUBO ऑप्टिमायझर

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.0.0-green.svg)](pyproject.toml)

> [English](README_en.md) | [日本語](README.md) | [中文](README_zh.md) | [한국어](README_ko.md) | [Español](README_es.md) | [हिन्दी](README_hi.md) | [Français](README_fr.md) | [বাংলা](README_bn.md) | [தமிழ்](README_ta.md) | [తెలుగు](README_te.md) | [मराठी](README_mr.md) | [‎اردو‎](README_ur.md) | [ગુજરાતી](README_gu.md) | [ಕನ್ನಡ](README_kn.md) | [മലയാളം](README_ml.md) | [ਪੰਜਾਬੀ](README_pa.md)

एक **D-Wave-मुक्त** क्लासिकल QUBO ऑप्टिमायझेशन लायब्ररी जी माहिती-सैद्धांतिक आश्चर्य (information-theoretic surprise), बौद्ध तत्त्वज्ञानाने प्रेरित ऊर्जा फंक्शन्स, मुक्त ऊर्जा तत्त्व (FEP) तर्क, आणि चेतना-जागरूक संगणन यांना एकत्र जोडते — हे सर्व सामान्य हार्डवेअरवर चालते.

---

## मुख्य वैशिष्ट्ये

- **क्वांटम संगणक आवश्यक नाही** — क्लासिकल सोल्व्हर्स (SA, Ising SA, submodular greedy) D-Wave च्या गुणवत्तेशी बरोबरी करतात किंवा त्यापेक्षा श्रेष्ठ आहेत.
- **ट्रिलियन-टोकन सक्षम** — स्ट्रीमिंग प्रोबेबिलिस्टिक डेटा संरचना (Count-Min Sketch, Streaming Histogram) मेमरी O(k) ठेवतात.
- **धर्म-बीजगणित इंजिन** — प्लगेबल बौद्ध ऊर्जा पदे गणितीयदृष्ट्या सर्वोत्तम सोल्व्हरकडे स्वयंचलितपणे मार्गदर्शित होतात.
- **८-मोड तर्क ऑर्केस्ट्रेटर** — अनुकूलनशील, सैद्धांतिक, अपवर्तक, सक्रिय-अनुमान, स्मृती (अलय), निद्रा एकत्रीकरण, मूर्त, आणि क्वांटम-चेतना (पिनियल).
- **LLM-तयार** — प्रथम श्रेणीचे LangChain आणि LlamaIndex एकत्रीकरण, few-shot निवडक, आउटपुट पुन:क्रमांकक, ड्रिफ्ट डिटेक्टर.

---

## आर्किटेक्चर

```
lmm/
├── core.py                  # मुख्य LMM पाइपलाइन
├── cli.py                   # CLI प्रवेश बिंदू
├── qubo.py                  # स्पार्स QUBO मॅट्रिक्स बिल्डर
├── solvers.py               # SA / Ising SA / relaxation / greedy / submodular
├── surprise.py              # माहिती-सैद्धांतिक आश्चर्य
├── selector.py              # अनुकूलनशील निवड धोरण
├── processor.py             # प्राधान्य प्रक्रिया + कॅशे
├── pipeline.py              # SmartSelector → SmartProcessor ऑर्केस्ट्रेशन
├── _compat.py               # रनटाइम क्षमता शोध आणि स्पार्स सहाय्यक
├── dharma/                  # डिजिटल धर्म (बौद्ध तत्त्वज्ञान ऑप्टिमायझेशन)
│   ├── api.py               # DharmaLMM उच्च-स्तरीय API
│   ├── energy.py            # प्लगेबल ऊर्जा पदे (दुःख, प्रज्ञा, करुणा, …)
│   ├── engine.py            # UniversalDharmaEngine — स्वयं-मार्गदर्शन सोल्व्हर
│   ├── fep.py               # FEP ≡ KCL ODE सोल्व्हर
│   ├── neuromorphic.py      # मेमरिस्टर क्रॉसबार ASIC सिम्युलेटर
│   ├── reranker.py          # RAG कॅस्केड पुन:क्रमांकक + इंटेंट राउटर
│   └── …
├── reasoning/               # ८ FEP-आधारित तर्क मोड
│   ├── adaptive.py          # गतिशील पॅरामीटर ट्यूनिंग
│   ├── theoretical.py       # तार्किक संरचना आलेख
│   ├── hyper.py             # अपवर्तक अव्यक्त-नोड इंजेक्शन
│   ├── active.py            # बाह्य ज्ञान संपादन
│   ├── alaya.py             # हेबियन सिनॅप्टिक स्मृती (अलय)
│   ├── sleep.py             # NREM/REM स्मृती एकत्रीकरण
│   ├── embodiment.py        # ६-इंद्रिय मल्टीमोडल संलयन
│   ├── pineal.py            # क्वांटम चेतना (हार्डवेअर TRNG)
│   └── orchestrator.py      # मोड निवड आणि प्रेषण
├── scale/                   # ट्रिलियन-टोकन स्ट्रीमिंग
│   ├── sketch.py            # Count-Min Sketch, Streaming Histogram
│   ├── stream.py            # स्ट्रीमिंग आश्चर्य
│   ├── cascade.py           # बहु-स्तरीय कॅस्केड फिल्टर (1T → K)
│   └── pipeline.py          # ScalablePipeline
├── llm/                     # LLM वर्कफ्लो सहाय्यक
│   ├── fewshot.py           # Few-shot उदाहरण निवडक
│   ├── reranker.py          # आउटपुट पुन:क्रमांकक
│   ├── drift.py             # वितरण ड्रिफ्ट डिटेक्टर
│   ├── sampler.py           # टोकन सॅम्पलर
│   └── embeddings.py        # एकीकृत एम्बेडिंग अडॅप्टर
├── integrations/
│   ├── langchain.py         # DharmaRetriever / DocumentCompressor / ExampleSelector
│   └── llamaindex.py        # DharmaNodePostprocessor
```

---

## स्थापना

```bash
# मूलभूत (numpy + scipy फक्त)
pip install -e .

# dev टूल्ससह
pip install -e ".[dev]"

# सर्व ऐच्छिक एकत्रीकरणांसह
pip install -e ".[all]"
```

### ऐच्छिक एक्स्ट्राज

| एक्स्ट्रा | पॅकेजेस | उद्देश |
|-----------|---------|--------|
| `dev` | pytest, ruff | चाचणी आणि लिंटिंग |
| `dharma` | hnswlib | स्पार्स k-NN आलेख बांधणी |
| `langchain` | langchain-core | LangChain एकत्रीकरण |
| `llamaindex` | llama-index-core | LlamaIndex एकत्रीकरण |
| `all` | वरील सर्व | सर्वकाही |

---

## जलद प्रारंभ

### Python API

```python
import numpy as np
from lmm.core import LMM

# सर्वाधिक आश्चर्यकारक शीर्ष-K घटक निवडा
surprises = np.array([1.0, 5.0, 2.0, 8.0, 3.0, 7.0])
model = LMM(k=3, solver_method="sa")
result = model.select_from_surprises(surprises)
print(result.selected_indices)  # उदा. [3, 5, 1]
```

### CLI

```bash
# एक जलद डेमो चालवा
lmm --demo --k 10 --method sa

# NumPy फाइलमधून
lmm --input data.npy --k 5 --method greedy
```

---

## सोल्व्हर पद्धती

| पद्धत | वर्णन | जटिलता |
|-------|-------|--------|
| `sa` | सिम्युलेटेड अनीलिंग (डिफॉल्ट) | O(n · steps) |
| `ising_sa` | व्हेक्टराइज्ड डेल्टासह Ising-स्वरूप SA | O(1) प्रति फ्लिप |
| `relaxation` | सतत विश्रांती + पूर्णांकन (SLSQP) | O(n²) |
| `greedy` | लोभी निवड | O(n · k) |

---

## धर्म इंजिन — प्लगेबल ऊर्जा पदे

**UniversalDharmaEngine** तुम्हाला बौद्ध तत्त्वज्ञानाने प्रेरित ऊर्जा पदे रचण्यास देते, प्रत्येक त्याचे गणितीय गुणधर्म जाहीर करतो. इंजिन स्वयंचलितपणे सर्वोत्तम सोल्व्हरकडे मार्गदर्शन करतो.

```python
from lmm.dharma import UniversalDharmaEngine, DukkhaTerm, KarunaTerm

engine = UniversalDharmaEngine(n_candidates=1000)
engine.add(DukkhaTerm(surprises, weight=1.0))    # रेखीय  → Top-K
engine.add(KarunaTerm(impact_graph, weight=2.0)) # सुपरमॉड्युलर → SA
result = engine.synthesize_and_solve(k=10)
```

### तत्त्वज्ञान → गणित → सोल्व्हर मॅपिंग

| बौद्ध संकल्पना | ऊर्जा पद | गणित गुणधर्म | सोल्व्हर |
|----------------|----------|--------------|---------|
| प्रज्ञा (विवेक) | `PrajnaTerm` | रेखीय | Top-K क्रमवारी |
| करुणा (सहानुभूती) | `KarunaTerm` | सुपरमॉड्युलर | Warm-start + SA |
| शील (आचरण) | `SilaTerm` | सबमॉड्युलर | Lazy greedy |
| मध्यमक (मध्यम मार्ग) | `MadhyamakaCriterion` | Lyapunov | घातांकीय gradient |
| दुःख (यातना) | `DukkhaTerm` | रेखीय | Top-K क्रमवारी |

---

## तर्क ऑर्केस्ट्रेटर

**DharmaReasonerOrchestrator** क्वेरी जटिलतेनुसार ८ तर्क मोडांमधून निवड करतो:

```python
from lmm.reasoning import DharmaReasonerOrchestrator

orch = DharmaReasonerOrchestrator(n_candidates=500)
result = orch.reason(query_vector, context_vectors)
print(result.mode_used, result.explanation)
```

| मोड | मॉड्यूल | प्रेरणा |
|-----|---------|--------|
| अनुकूलनशील | `adaptive.py` | 応病与薬 — रोगाला अनुरूप औषध |
| सैद्धांतिक | `theoretical.py` | 因明 — बौद्ध औपचारिक तर्कशास्त्र |
| अपवर्तक | `hyper.py` | 般若の飛躍 — प्रज्ञेची झेप |
| सक्रिय अनुमान | `active_inference.py` | 托鉢 — बाह्य सत्याचा शोध |
| अलय स्मृती | `alaya.py` | 阿頼耶識 — भांडारगृह चेतना |
| निद्रा | `sleep.py` | 禅定 — NREM/REM एकत्रीकरण |
| मूर्त | `embodiment.py` | 六根 — सहा इंद्रिय प्रकार |
| पिनियल (क्वांटम) | `pineal.py` | 松果体 — हार्डवेअर-एन्ट्रॉपी चेतना |

---

## ट्रिलियन-टोकन स्केलिंग

स्थिर मेमरीसह अनियंत्रित मोठ्या डेटा स्ट्रीम्सवर प्रक्रिया करा:

```python
from lmm.scale import ScalablePipeline
from pathlib import Path

pipe = ScalablePipeline(k=10, chunk_size=100_000)
pipe.fit_files([Path("shard_001.npy"), ...])
result = pipe.run_files([Path("data_001.npy"), ...])
print(result.summary)
```

**तीन-टप्प्याचा कॅस्केड:** १ T टोकन → १०० M (स्ट्रीमिंग top-k) → १० K (percentile) → K (QUBO).

---

## LLM एकत्रीकरण

### LangChain

```python
from lmm.integrations.langchain import DharmaDocumentCompressor

compressor = DharmaDocumentCompressor(k=5)
# LangChain च्या ContextualCompressionRetriever सह वापरा
```

### LlamaIndex

```python
from lmm.integrations.llamaindex import DharmaNodePostprocessor

postprocessor = DharmaNodePostprocessor(top_k=5)
# LlamaIndex query engine सह वापरा
```

### स्वतंत्र LLM सहाय्यक

```python
from lmm.llm import FewShotSelector, OutputReranker, DriftDetector

# विविधता हमीसह few-shot उदाहरण निवड
selector = FewShotSelector(k=5)
examples = selector.select(candidates, query)

# आश्चर्य × विविधतेनुसार LLM आउटपुट पुन:क्रमांकन
reranker = OutputReranker(top_k=3)
best = reranker.rerank(outputs)

# कालांतराने आउटपुट वितरण ड्रिफ्ट निरीक्षण
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

## न्यूरोमॉर्फिक सिम्युलेशन

ऊर्जा-कार्यक्षम FEP संगणनासाठी मेमरिस्टर-क्रॉसबार हार्डवेअरचे सिम्युलेशन करा:

```python
from lmm.dharma.neuromorphic import NeuromorphicChip

chip = NeuromorphicChip(size=64)
chip.program(weight_matrix)
report = chip.run(input_voltages, steps=100)
# ~10 fJ/MAC, ~30 ns अभिसरण
```

---

## अवलंबित्व

| पॅकेज | आवृत्ती | आवश्यक |
|-------|--------|--------|
| numpy | >= 1.24 | होय |
| scipy | >= 1.10 | होय |
| hnswlib | >= 0.8.0 | ऐच्छिक (स्पार्स आलेख) |
| langchain-core | >= 0.2.0 | ऐच्छिक (LangChain) |
| llama-index-core | >= 0.10.0 | ऐच्छिक (LlamaIndex) |

---

## सैद्धांतिक पाया

| संकल्पना | सूत्रीकरण |
|----------|-----------|
| **FEP ≡ KCL** | `dV/dt = −V/τ + g'(V) · G · ε` — Kirchhoff च्या विद्युत प्रवाह नियमाप्रमाणे अंदाज-त्रुटी किमानीकरण |
| **सुपरमॉड्युलर करुणा** | करुणा वाढीव परतावे दर्शवते: `f(S∪{i}) − f(S) ≥ f(T∪{i}) − f(T)` जेव्हा S ⊆ T |
| **सबमॉड्युलर शील** | आचरण घटते परतावे दर्शवते — lazy greedy (1−1/e) हमी देते |
| **मध्यमक** | मध्यम मार्ग Lyapunov-स्थिर घातांकीय gradient descent द्वारे CV = 0.5 लक्ष्य करतो |
| **पिनियल कोलॅप्स** | तरंगफंक्शन कोलॅप्स PRNG ऐवजी भौतिक एन्ट्रॉपी (हार्डवेअर TRNG) वापरतो |

---

## परवाना

[MIT](LICENSE)
