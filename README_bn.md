# LMM — ডিজিটাল ধর্মসহ ক্লাসিক্যাল QUBO অপ্টিমাইজার

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.0.0-green.svg)](pyproject.toml)

> [English](README_en.md) | [日本語](README.md) | [中文](README_zh.md) | [한국어](README_ko.md) | [Español](README_es.md) | [हिन्दी](README_hi.md) | [Français](README_fr.md) | [বাংলা](README_bn.md) | [தமிழ்](README_ta.md) | [తెలుగు](README_te.md) | [मराठी](README_mr.md) | [‎اردو‎](README_ur.md) | [ગુજરાતી](README_gu.md) | [ಕನ್ನಡ](README_kn.md) | [മലയാളം](README_ml.md) | [ਪੰਜਾਬੀ](README_pa.md)

> **Digital Dharma OS (Alaya V5)** — QUBO গণিত, বৌদ্ধ দর্শন এবং মুক্ত শক্তি নীতি (FEP)
> স্নায়ুবিজ্ঞানকে একটি জীবন্ত, স্ব-বিবর্তনশীল ব্যবস্থায় মিশ্রিত করা চেতনা-সচেতন
> অপ্টিমাইজেশন প্ল্যাটফর্ম। সার্ভার ক্রমাগত হৃদস্পন্দন গতিবিদ্যা, স্মৃতি-চালিত
> প্রসঙ্গ নির্বাচন এবং আবেগীয় তরঙ্গদৈর্ঘ্য সংবেদন চালায় — বিচ্ছিন্ন LLM কল এবং
> ক্রমাগত জ্ঞানের মধ্যে সেতু তৈরি করে।

একটি **D-Wave-মুক্ত** ক্লাসিক্যাল QUBO অপ্টিমাইজেশন লাইব্রেরি যা তথ্য-তাত্ত্বিক বিস্ময়কে বৌদ্ধ দর্শন-অনুপ্রাণিত শক্তি ফাংশন, মুক্ত শক্তি নীতি (FEP) যুক্তি এবং চেতনা-সচেতন কম্পিউটিং-এর সাথে মিশিয়ে দেয় — সবকিছু সাধারণ হার্ডওয়্যারে চলে।

---

## বিশেষ বৈশিষ্ট্য

- **কোনো কোয়ান্টাম কম্পিউটার প্রয়োজন নেই** — ক্লাসিক্যাল সলভার (SA, Ising SA, submodular greedy) D-Wave-এর সমকক্ষ বা তার চেয়ে ভালো মান দেয়।
- **ট্রিলিয়ন-টোকেন সক্ষম** — স্ট্রিমিং প্রবাবিলিস্টিক ডেটা কাঠামো (Count-Min Sketch, Streaming Histogram) মেমোরি O(k)-তে রাখে।
- **ধর্ম-বীজগণিত ইঞ্জিন** — প্লাগযোগ্য বৌদ্ধ শক্তি পদ গণিতগতভাবে সর্বোত্তম সলভারে স্বয়ংক্রিয়ভাবে রুট হয়।
- **৮-মোড যুক্তি অর্কেস্ট্রেটর** — অভিযোজিত, তাত্ত্বিক, অপবাহী (abductive), সক্রিয়-অনুমান, স্মৃতি (আলয়), নিদ্রা একত্রীকরণ, মূর্ত, এবং কোয়ান্টাম-চেতনা (Pineal)।
- **LLM-প্রস্তুত** — প্রথম শ্রেণির LangChain ও LlamaIndex ইন্টিগ্রেশন, few-shot সিলেক্টর, আউটপুট রিরেংকার, ড্রিফট ডিটেক্টর।

---

## স্থাপত্য

```
lmm/
├── core.py                  # মূল LMM পাইপলাইন
├── cli.py                   # CLI প্রবেশবিন্দু
├── qubo.py                  # Sparse QUBO ম্যাট্রিক্স নির্মাতা
├── solvers.py               # SA / Ising SA / relaxation / greedy / submodular
├── surprise.py              # তথ্য-তাত্ত্বিক বিস্ময়
├── selector.py              # অভিযোজিত নির্বাচন কৌশল
├── processor.py             # অগ্রাধিকার প্রক্রিয়াকরণ + ক্যাশে
├── pipeline.py              # SmartSelector → SmartProcessor অর্কেস্ট্রেশন
├── _compat.py               # রানটাইম সক্ষমতা সনাক্তকরণ ও sparse সহায়ক
├── dharma/                  # ডিজিটাল ধর্ম (বৌদ্ধ-দর্শন অপ্টিমাইজেশন)
│   ├── api.py               # DharmaLMM উচ্চ-স্তরের API
│   ├── energy.py            # প্লাগযোগ্য শক্তি পদ (Dukkha, Prajna, Karuna, …)
│   ├── engine.py            # UniversalDharmaEngine — স্বয়ংক্রিয়-রুটিং সলভার
│   ├── fep.py               # FEP ≡ KCL ODE সলভার
│   ├── neuromorphic.py      # Memristor crossbar ASIC সিমুলেটর
│   ├── reranker.py          # RAG cascade রিরেংকার + ইন্টেন্ট রাউটার
│   └── …
├── reasoning/               # ৮টি FEP-ভিত্তিক যুক্তি মোড
│   ├── adaptive.py          # গতিশীল প্যারামিটার সুনিয়ন্ত্রণ
│   ├── theoretical.py       # যৌক্তিক কাঠামো গ্রাফ
│   ├── hyper.py             # অপবাহী লেটেন্ট-নোড ইনজেকশন
│   ├── active.py            # বাহ্যিক জ্ঞান অর্জন
│   ├── alaya.py             # হেব্বিয়ান সিনাপ্টিক স্মৃতি (আলয়)
│   ├── sleep.py             # NREM/REM স্মৃতি একত্রীকরণ
│   ├── embodiment.py        # ৬-ইন্দ্রিয় মাল্টিমোডাল ফিউশন
│   ├── pineal.py            # কোয়ান্টাম চেতনা (hardware TRNG)
│   └── orchestrator.py      # মোড নির্বাচন ও প্রেরণ
├── scale/                   # ট্রিলিয়ন-টোকেন স্ট্রিমিং
│   ├── sketch.py            # Count-Min Sketch, Streaming Histogram
│   ├── stream.py            # স্ট্রিমিং বিস্ময়
│   ├── cascade.py           # বহু-স্তর cascade ফিল্টার (1T → K)
│   └── pipeline.py          # ScalablePipeline
├── llm/                     # LLM ওয়ার্কফ্লো সহায়ক
│   ├── fewshot.py           # Few-shot উদাহরণ সিলেক্টর
│   ├── reranker.py          # আউটপুট রিরেংকার
│   ├── drift.py             # বিতরণ ড্রিফট ডিটেক্টর
│   ├── sampler.py           # টোকেন স্যাম্পলার
│   └── embeddings.py        # একীভূত এমবেডিং অ্যাডাপ্টার
├── integrations/
│   ├── langchain.py         # DharmaRetriever / DocumentCompressor / ExampleSelector
│   └── llamaindex.py        # DharmaNodePostprocessor
```

---

## ইনস্টলেশন

```bash
# মূল (শুধুমাত্র numpy + scipy)
pip install -e .

# ডেভ টুলসহ
pip install -e ".[dev]"

# সমস্ত ঐচ্ছিক ইন্টিগ্রেশনসহ
pip install -e ".[all]"
```

### ঐচ্ছিক অতিরিক্ত প্যাকেজ

| অতিরিক্ত | প্যাকেজ | উদ্দেশ্য |
|-----------|---------|----------|
| `dev` | pytest, ruff | পরীক্ষা ও লিন্টিং |
| `dharma` | hnswlib | Sparse k-NN গ্রাফ নির্মাণ |
| `langchain` | langchain-core | LangChain ইন্টিগ্রেশন |
| `llamaindex` | llama-index-core | LlamaIndex ইন্টিগ্রেশন |
| `all` | উপরের সবকিছু | সম্পূর্ণ |

---

## দ্রুত শুরু

### Python API

```python
import numpy as np
from lmm.core import LMM

# শীর্ষ-K সর্বাধিক বিস্ময়কর আইটেম নির্বাচন করুন
surprises = np.array([1.0, 5.0, 2.0, 8.0, 3.0, 7.0])
model = LMM(k=3, solver_method="sa")
result = model.select_from_surprises(surprises)
print(result.selected_indices)  # যেমন [3, 5, 1]
```

### CLI

```bash
# একটি দ্রুত ডেমো চালান
lmm --demo --k 10 --method sa

# NumPy ফাইল থেকে
lmm --input data.npy --k 5 --method greedy
```

---

## সার্ভার মোড (Alaya V5)

Alaya-Vijñāna v5.0 সার্ভার রিয়েল-টাইম আবেগীয় তরঙ্গদৈর্ঘ্য ভিজুয়ালাইজেশন, ৮-মোড FEP যুক্তি এবং Claude/Gemini স্বয়ংক্রিয় রাউটিং সহ একটি ওয়েব UI প্রদান করে।

### পূর্বশর্ত

```bash
pip install -e ".[server]"
```

### সার্ভার শুরু করুন

**PowerShell (Windows):**
```powershell
.\Start-DharmaServer.ps1
```

**Python (ক্রস-প্ল্যাটফর্ম):**
```bash
python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

ব্রাউজারে খুলুন: [http://localhost:8000](http://localhost:8000)

### সার্ভার বন্ধ করুন

**Windows (ব্যাচ):**
```batch
.\stop-dharma-server.bat
```

**PowerShell:** `Start-DharmaServer.ps1` চালানো উইন্ডোতে `Ctrl+C` চাপুন।

**Linux/macOS:** `Ctrl+C` চাপুন অথবা চালান:
```bash
kill $(cat server.pid)
```

### API এন্ডপয়েন্ট

| এন্ডপয়েন্ট | পদ্ধতি | বিবরণ |
|------------|--------|-------|
| `/` | GET | ওয়েব UI (Alaya V5 ফ্রন্টএন্ড) |
| `/api/descent` | POST | প্রধান যুক্তি পাইপলাইন |
| `/api/descent/stream` | POST | স্ট্রিমিং SSE প্রতিক্রিয়া |
| `/api/dharma/auto` | POST | স্বয়ংক্রিয় রাউটিং Dharma যুক্তি |
| `/api/sense` | POST | আবেগীয় তরঙ্গদৈর্ঘ্য বিশ্লেষণ |
| `/api/status` | GET | সিস্টেম স্থিতি + হৃদস্পন্দন টেলিমেট্রি |

---

## স্বায়ত্তশাসিত বৈশিষ্ট্য

সার্ভারে তিনটি স্বায়ত্তশাসিত উপব্যবস্থা রয়েছে যা ব্যবহারকারীর ইন্টারঅ্যাকশনের মধ্যে ক্রমাগত কাজ করে:

### হৃদস্পন্দন ডেমন (Heartbeat)

৪-মাত্রিক অবস্থা ভেক্টর `[প্রেম, যুক্তি, ভয়, সৃষ্টি]` বজায় রাখা একটি ক্রমাগত-সময় FEP অবস্থা বিবর্তন লুপ:

- **Tick ব্যবধান:** 100ms (সক্রিয়) → 5s (নিষ্ক্রিয়), অভিযোজিত মন্দন
- **এনট্রপি ইনজেকশন:** প্রতিটি tick-এ `os.urandom()` এর মাধ্যমে হার্ডওয়্যার এনট্রপি
- **ভার অভিযোজন:** করুণা (Karuna) এবং মৈত্রী (Metta) ভার মধ্যপথ CV ট্র্যাকিং (লক্ষ্য CV=0.5) দ্বারা স্বয়ংক্রিয় সমন্বয়
- **নিদ্রা সুসংহতকরণ:** 60s নিষ্ক্রিয়তার পরে NREM/REM স্মৃতি পুনঃচালন ট্রিগার

### প্রসঙ্গ সেতু (আলয় স্মৃতি)

সরল ইতিহাস কর্তন (`history[-20:]`) কে স্মৃতি-চালিত বুদ্ধিমান প্রসঙ্গ নির্বাচনে প্রতিস্থাপন করে:

- **Modern Hopfield Network** এর মাধ্যমে AlayaMemory থেকে স্মরণ
- স্মরণ প্যাটার্নের সাথে কোসাইন সাদৃশ্য দ্বারা কথোপকথন ইতিহাস স্কোরিং
- সর্বদা সাম্প্রতিকতম ৩টি বার্তা অন্তর্ভুক্ত (সাম্প্রতিকতা পক্ষপাত)
- অবশিষ্ট বাজেট (সর্বোচ্চ ২০টি বার্তা) প্রাসঙ্গিকতা স্কোর দ্বারা পূরণ

### অর্থবহ আবেগ (Semantic Emotions)

`config/semantic_emotions.json`-এ সংজ্ঞায়িত চারটি মাত্রায় কীওয়ার্ড মিলনের মাধ্যমে রিয়েল-টাইম আবেগীয় তরঙ্গদৈর্ঘ্য বিশ্লেষণ:

| মাত্রা | বৌদ্ধ ধারণা | সংকেত |
|--------|------------|-------|
| প্রেম | করুণা (Karuna) | সহানুভূতি, উষ্ণতা, কৃতজ্ঞতা |
| যুক্তি | হেতুবিদ্যা (Hetuvidya) | বিশ্লেষণ, যুক্তি, প্রমাণ |
| ভয় | দুঃখ (Dukkha) | উদ্বেগ, সন্দেহ, কষ্ট |
| সৃষ্টি | সৃষ্টি (Sṛṣṭi) | উদ্ভাবন, কল্পনা, শিল্প |

প্রতিটি কীওয়ার্ডের ভার (0.3–1.0) রয়েছে। ফলস্বরূপ 4D ভেক্টর যুক্তি মোড নির্বাচন এবং হৃদস্পন্দন অবস্থা ইনজেকশন চালায়।

---

## সলভার পদ্ধতি

| পদ্ধতি | বিবরণ | জটিলতা |
|--------|-------|---------|
| `sa` | Simulated Annealing (ডিফল্ট) | O(n · steps) |
| `ising_sa` | ভেক্টরাইজড ডেল্টাসহ Ising-ফর্ম SA | O(1) প্রতি ফ্লিপ |
| `relaxation` | ক্রমাগত রিল্যাক্সেশন + রাউন্ডিং (SLSQP) | O(n²) |
| `greedy` | লোভী নির্বাচন | O(n · k) |

---

## ধর্ম ইঞ্জিন — প্লাগযোগ্য শক্তি পদ

**UniversalDharmaEngine** আপনাকে বৌদ্ধ দর্শন-অনুপ্রাণিত শক্তি পদ রচনা করতে দেয়, প্রতিটি তার গাণিতিক বৈশিষ্ট্য ঘোষণা করে। ইঞ্জিন স্বয়ংক্রিয়ভাবে সর্বোত্তম সলভারে রুট করে।

```python
from lmm.dharma import UniversalDharmaEngine, DukkhaTerm, KarunaTerm

engine = UniversalDharmaEngine(n_candidates=1000)
engine.add(DukkhaTerm(surprises, weight=1.0))    # linear  → Top-K
engine.add(KarunaTerm(impact_graph, weight=2.0)) # supermodular → SA
result = engine.synthesize_and_solve(k=10)
```

### দর্শন → গণিত → সলভার ম্যাপিং

| বৌদ্ধ ধারণা | শক্তি পদ | গাণিতিক বৈশিষ্ট্য | সলভার |
|------------|---------|-----------------|-------|
| প্রজ্ঞা (জ্ঞান) | `PrajnaTerm` | linear | Top-K সর্ট |
| করুণা (সহানুভূতি) | `KarunaTerm` | supermodular | Warm-start + SA |
| শীল (আচরণ) | `SilaTerm` | submodular | Lazy greedy |
| মধ্যমক (মধ্যপথ) | `MadhyamakaCriterion` | Lyapunov | এক্সপোনেনশিয়াল গ্রেডিয়েন্ট |
| দুঃখ (কষ্ট) | `DukkhaTerm` | linear | Top-K সর্ট |

---

## যুক্তি অর্কেস্ট্রেটর

**DharmaReasonerOrchestrator** প্রশ্নের জটিলতার ভিত্তিতে ৮টি যুক্তি মোডের মধ্য থেকে নির্বাচন করে:

```python
from lmm.reasoning import DharmaReasonerOrchestrator

orch = DharmaReasonerOrchestrator(n_candidates=500)
result = orch.reason(query_vector, context_vectors)
print(result.mode_used, result.explanation)
```

| মোড | মডিউল | অনুপ্রেরণা |
|-----|-------|-----------|
| অভিযোজিত | `adaptive.py` | 応病与薬 — রোগ অনুযায়ী ওষুধ |
| তাত্ত্বিক | `theoretical.py` | 因明 — বৌদ্ধ আনুষ্ঠানিক যুক্তি |
| অপবাহী | `hyper.py` | 般若の飛躍 — প্রজ্ঞার উল্লম্ফন |
| সক্রিয় অনুমান | `active_inference.py` | 托鉢 — বাহ্যিক সত্য অনুসন্ধান |
| আলয় স্মৃতি | `alaya.py` | 阿頼耶識 — ভান্ডারঘর চেতনা |
| নিদ্রা | `sleep.py` | 禅定 — NREM/REM একত্রীকরণ |
| মূর্ত | `embodiment.py` | 六根 — ছয় ইন্দ্রিয় পদ্ধতি |
| Pineal (কোয়ান্টাম) | `pineal.py` | 松果体 — হার্ডওয়্যার-এনট্রপি চেতনা |

---

## ট্রিলিয়ন-টোকেন স্কেলিং

ধ্রুবক মেমোরিতে অসীম বড় ডেটা স্ট্রিম প্রক্রিয়া করুন:

```python
from lmm.scale import ScalablePipeline
from pathlib import Path

pipe = ScalablePipeline(k=10, chunk_size=100_000)
pipe.fit_files([Path("shard_001.npy"), ...])
result = pipe.run_files([Path("data_001.npy"), ...])
print(result.summary)
```

**তিন-স্তর cascade:** ১ ট্রিলিয়ন টোকেন → ১০০ মিলিয়ন (স্ট্রিমিং top-k) → ১০ হাজার (পার্সেন্টাইল) → K (QUBO)।

---

## LLM ইন্টিগ্রেশন

### LangChain

```python
from lmm.integrations.langchain import DharmaDocumentCompressor

compressor = DharmaDocumentCompressor(k=5)
# LangChain-এর ContextualCompressionRetriever-এর সাথে ব্যবহার করুন
```

### LlamaIndex

```python
from lmm.integrations.llamaindex import DharmaNodePostprocessor

postprocessor = DharmaNodePostprocessor(top_k=5)
# LlamaIndex কোয়েরি ইঞ্জিনের সাথে ব্যবহার করুন
```

### স্বতন্ত্র LLM সহায়ক

```python
from lmm.llm import FewShotSelector, OutputReranker, DriftDetector

# বৈচিত্র্য নিশ্চয়তাসহ few-shot উদাহরণ নির্বাচন
selector = FewShotSelector(k=5)
examples = selector.select(candidates, query)

# বিস্ময় × বৈচিত্র্য দ্বারা LLM আউটপুট রিরেংক করুন
reranker = OutputReranker(top_k=3)
best = reranker.rerank(outputs)

# সময়ের সাথে আউটপুট বিতরণ ড্রিফট পর্যবেক্ষণ করুন
detector = DriftDetector(window=100)
report = detector.update(new_output)
```

---

## DharmaLMM — উচ্চ-স্তরের API

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

## নিউরোমর্ফিক সিমুলেশন

শক্তি-দক্ষ FEP গণনার জন্য memristor-crossbar হার্ডওয়্যার সিমুলেট করুন:

```python
from lmm.dharma.neuromorphic import NeuromorphicChip

chip = NeuromorphicChip(size=64)
chip.program(weight_matrix)
report = chip.run(input_voltages, steps=100)
# ~10 fJ/MAC, ~30 ns কনভার্জেন্স
```

---

## নির্ভরতা

| প্যাকেজ | সংস্করণ | প্রয়োজনীয় |
|---------|---------|-----------|
| numpy | >= 1.24 | হ্যাঁ |
| scipy | >= 1.10 | হ্যাঁ |
| hnswlib | >= 0.8.0 | ঐচ্ছিক (sparse গ্রাফ) |
| langchain-core | >= 0.2.0 | ঐচ্ছিক (LangChain) |
| llama-index-core | >= 0.10.0 | ঐচ্ছিক (LlamaIndex) |

---

## তাত্ত্বিক ভিত্তি

| ধারণা | গণ্য সূত্র |
|-------|-----------|
| **FEP ≡ KCL** | `dV/dt = −V/τ + g'(V) · G · ε` — কির্শহফের বিদ্যুৎ সূত্র হিসেবে পূর্বানুমান-ত্রুটি হ্রাসকরণ |
| **Supermodular Karuna** | করুণা ক্রমবর্ধমান ফলন প্রদর্শন করে: `f(S∪{i}) − f(S) ≥ f(T∪{i}) − f(T)` যেখানে S ⊆ T |
| **Submodular Sila** | শীল ক্রমহ্রাসমান ফলন প্রদর্শন করে — lazy greedy (1−1/e) গ্যারান্টি দেয় |
| **Madhyamaka** | মধ্যপথ Lyapunov-স্থিতিশীল এক্সপোনেনশিয়াল গ্রেডিয়েন্ট ডিসেন্টের মাধ্যমে CV = 0.5 লক্ষ্য করে |
| **Pineal Collapse** | তরঙ্গফাংশন পতনে PRNG-এর পরিবর্তে ভৌত এনট্রপি (hardware TRNG) ব্যবহার করা হয় |

---

## লাইসেন্স

[MIT](LICENSE)
